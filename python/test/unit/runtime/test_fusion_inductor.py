"""Real Inductor compilation and migration tests for the graph-fusion entrypoint."""

from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from triton.experimental import fusion


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def graph(x, bias):
    # Broadcast, two iteration domains, nontrivial DAG and escaped intermediate.
    shifted = torch.sin(x) + bias
    row = shifted.square().mean(dim=-1, keepdim=True)
    return shifted * torch.rsqrt(row + 0.01), row, shifted


@pytest.mark.parametrize("input_kind", ["function", "module", "fx"])
def test_real_inductor_compilation(input_kind):
    class Model(torch.nn.Module):
        def forward(self, x, bias):
            return graph(x, bias)

    model = graph if input_kind == "function" else Model()
    if input_kind == "fx":
        model = torch.fx.symbolic_trace(model)
    # This instrumentation observes actual generated code, not a mock backend.
    from torch._inductor.utils import run_and_get_code
    x, bias = torch.randn(7, 129), torch.randn(129)
    compiled = fusion.compile(model)
    actual, sources = run_and_get_code(compiled, x, bias)
    torch.testing.assert_close(actual, model(x, bias), rtol=2e-5, atol=2e-5)
    assert sources, "test must exercise code generation, not eager fallback"


def test_dynamic_shapes_new_values_and_noncontiguous_inputs():
    compiled = fusion.compile(graph, dynamic=True)
    for rows, columns in [(3, 17), (9, 33), (4, 65)]:
        x = torch.randn(columns, rows).t()
        bias = torch.randn(columns)
        assert not x.is_contiguous()
        torch.testing.assert_close(compiled(x, bias), graph(x, bias), rtol=2e-5, atol=2e-5)
        x.add_(0.5)
        torch.testing.assert_close(compiled(x, bias), graph(x, bias), rtol=2e-5, atol=2e-5)


def test_autograd_through_reduction_and_branches():
    inputs = (torch.randn(5, 19), torch.randn(19))
    eager_inputs = tuple(x.clone().requires_grad_() for x in inputs)
    compiled_inputs = tuple(x.clone().requires_grad_() for x in inputs)
    eager = graph(*eager_inputs)
    actual = fusion.compile(graph)(*compiled_inputs)
    sum(x.square().sum() for x in eager).backward()
    sum(x.square().sum() for x in actual).backward()
    torch.testing.assert_close(actual, eager, rtol=2e-5, atol=2e-5)
    for a, b in zip(compiled_inputs, eager_inputs):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-4)


def test_decorator_and_live_module_state():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.randn(11))

        def forward(self, x):
            return (x + self.bias).relu()

    model = Model()
    compiled = fusion.compile(model)
    x = torch.randn(3, 11)
    torch.testing.assert_close(compiled(x), model(x))
    model.bias.add_(1)
    torch.testing.assert_close(compiled(x), model(x))

    @fusion.compile(dynamic=True)
    def decorated(x):
        return x.sin() * x

    torch.testing.assert_close(decorated(x), x.sin() * x)


def test_explicit_rounding_policy():
    from torch._inductor import config
    if not hasattr(config, "emulate_precision_casts"):
        pytest.skip("installed PyTorch has no intermediate-cast emulation option")

    def rounded(x):
        value = (x + 0.0003).to(torch.float16)
        return value.float() - x

    options = {"emulate_precision_casts": True}
    compiled = fusion.compile(rounded, options=options)
    x = torch.ones(257)
    torch.testing.assert_close(compiled(x), rounded(x), rtol=0, atol=0)
    assert options == {"emulate_precision_casts": True}


def test_fullgraph_failure_is_visible_and_does_not_run_body():
    calls = []

    @torch.compiler.disable
    def unsupported(x):
        calls.append("executed")
        return x + 1

    def model(x):
        return unsupported(x).sin()

    compiled = fusion.compile(model)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled(torch.ones(7))
    assert calls == []
    # Graph breaks are available only when the caller explicitly permits them.
    actual = fusion.compile(model, fullgraph=False)(torch.ones(7))
    torch.testing.assert_close(actual, torch.full((7,), 2.0).sin())
    assert calls == ["executed"]


def test_disable_and_lazy_compilation_do_not_run_inputs():
    calls = []

    def model(x):
        calls.append("executed")
        return x + 1

    compiled = fusion.compile(model, disable=True)
    assert calls == []
    torch.testing.assert_close(compiled(torch.ones(4)), torch.full((4,), 2.0))
    assert calls == ["executed"]


def test_removed_pass_and_capture_hooks():
    from triton._C.libtriton import passes
    from triton import jit as jit_decorator
    import importlib

    assert not hasattr(passes.ttir, "add_fuse_kernel_calls")
    assert not hasattr(importlib.import_module("triton.runtime.jit"), "_kernel_call_capture")
    assert callable(jit_decorator)
    with pytest.raises(RuntimeError, match="fusion.compile"):
        fusion.session()


@pytest.mark.parametrize("name", ["kernel_call_fusion", "kernel_call_fusion_private",
                                   "kernel_call_fusion_launch_grid", "kernel_call_fusion_readonly_reuse",
                                   "kernel_call_fusion_interleave", "kernel_call_fusion_prefetch"])
def test_legacy_compiler_options_fail_explicitly(name):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend

    with pytest.raises(ValueError, match="removed"):
        CUDABackend(GPUTarget("cuda", 89, 32)).parse_options({name: True})


def test_legacy_cached_kernel_is_rejected_before_launcher_compilation():
    from triton.backends.nvidia.driver import CudaLauncher

    for metadata in (SimpleNamespace(kernel_call_fusion='{"accepted":true}'),
                     SimpleNamespace(kernel_call_fusion_launch_grid=4)):
        # A missing source also ensures rejection happens before source handling.
        with pytest.raises(RuntimeError, match="removed kernel-call fusion"):
            CudaLauncher(None, metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a working CUDA device")
def test_cuda_inductor_uses_installed_flagtree():
    import triton
    from torch._inductor.utils import run_and_get_code

    x = torch.randn(11, 257, device="cuda")
    bias = torch.randn(257, device="cuda")
    actual, sources = run_and_get_code(fusion.compile(graph), x, bias)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, graph(x, bias), rtol=2e-5, atol=2e-5)
    assert any("@triton.jit" in source for source in sources)
    # Compiler provenance is recorded by the benchmark alongside these sources.
    assert hasattr(triton, "spec_call"), triton.__file__
