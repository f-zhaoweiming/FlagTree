"""Static matcher and end-to-end tests for the P800 Inductor template."""
import pytest
import torch


@pytest.fixture
def p800():
    from triton.runtime import driver
    if not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver':
        pytest.skip("requires the P800 Torch SDK")
    if driver.active.get_current_target().arch != 3:
        pytest.skip("requires P800")
    from triton.experimental.graph_fusion import _p800_inductor_gelu
    return _p800_inductor_gelu


def graph(a, b, *, alpha=1, share=False):
    fx = torch.fx.Graph()
    left, right = fx.placeholder("a"), fx.placeholder("b")
    left.meta["example_value"], right.meta["example_value"] = a, b
    add = fx.call_function(torch.add, (left, right), {"alpha": alpha})
    result = fx.call_function(torch.nn.functional.gelu, (add,), {"approximate": "tanh"})
    fx.output((result, add) if share else (result,))
    return fx


def test_rewrite_domain(p800):
    a = torch.empty((64, 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a[0])
    fx = graph(b, a)
    p800.rewrite(fx)
    node = next(n for n in fx.nodes if n.op == "call_function")
    assert node.target == torch.ops.flagtree_p800.bias_gelu.default
    assert node.args[0].meta["example_value"] is a
    assert node.args[1].meta["example_value"] is b
    fx.lint()
    for fx in (graph(a, b, alpha=2), graph(a, b, share=True), graph(a[:32], b),
               graph(a.t().contiguous().t(), b), graph(a.float(), b.float()), graph(a.detach().requires_grad_(), b)):
        before = str(fx)
        p800.rewrite(fx, small_only=True)
        assert str(fx) == before


@pytest.mark.parametrize("m,mode", [(64, "tanh"), (64, "none"), (512, "tanh"), (512, "none"), (2048, "none")])
def test_native_bias_gelu(p800, m, mode):
    from torch._inductor.utils import run_and_get_code

    class BiasGelu(torch.nn.Module):
        def forward(self, a, b):
            return torch.nn.functional.gelu(a + b, approximate=mode)

    model = BiasGelu().eval()
    fn = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
    a = torch.randn((m, 2048 if m == 2048 else 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a[0])
    with torch.no_grad():
        y, sources = run_and_get_code(fn, a, b)
        wrapper, = sources
        assert "float32" in wrapper and y.dtype == torch.bfloat16
        torch.testing.assert_close(y.cpu(), model(a.float(), b.float()).bfloat16().cpu(), rtol=.003,
                                   atol=.03, equal_nan=True)
        b[0] = float("inf")
        b[1] = float("-inf")
        b[2] = float("nan")
        z = fn(a, b)
        assert y.data_ptr() != z.data_ptr()
        torch.testing.assert_close(z.cpu(), model(a.float(), b.float()).bfloat16().cpu(), rtol=.003, atol=.03, equal_nan=True)


def test_intermediate_preserves_all_consumers(p800):
    from torch._inductor.utils import run_and_get_code

    # Use a matrix and a computed bias as inputs to the Add.
    def addition_block(a, b):
        activated = torch.nn.functional.gelu(a.relu() + (-b), approximate="tanh")
        return activated, -activated

    a = torch.randn((64, 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a[0])
    with torch.no_grad():
        compiled = torch.compile(addition_block, fullgraph=True, dynamic=False)
        result, sources = run_and_get_code(compiled, a, b)
        assert any("round_output" in src for src in sources)
        expected = tuple(x.bfloat16() for x in addition_block(a.float(), b.float()))
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=.003, atol=.03, equal_nan=True)


def test_do_not_move_add_across_mutation(p800):
    a = torch.empty((64, 1024), device="cuda", dtype=torch.bfloat16)
    fx = graph(a, a[0])
    matrix, _, add, gelu, _ = fx.nodes
    with fx.inserting_after(add):
        fx.call_method("zero_", (matrix,))
    assert p800._match(gelu, {(64, 1024)}) is None


def test_live_parameter_inference_and_training(p800):
    from torch._inductor.utils import run_and_get_code

    class LiveBias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.randn(1024, dtype=torch.bfloat16, device="cuda"))

        def forward(self, matrix):
            return torch.nn.functional.gelu(matrix + self.bias, approximate="tanh")

    model = LiveBias().eval()
    a = torch.randn((64, 1024), dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(model, fullgraph=True, dynamic=False)
    with torch.no_grad():
        y, inference = run_and_get_code(compiled, a)
        assert any("float32" in src for src in inference) and y.dtype == torch.bfloat16
        torch.testing.assert_close(y.cpu(), torch.nn.functional.gelu(a.float() + model.bias.float(), approximate="tanh").bfloat16().cpu(), rtol=.003, atol=.03, equal_nan=True)
        model.bias.copy_(-model.bias)
        torch.testing.assert_close(compiled(a).cpu(), torch.nn.functional.gelu(a.float() + model.bias.float(), approximate="tanh").bfloat16().cpu(), rtol=.003, atol=.03, equal_nan=True)
    with torch.enable_grad():
        y, training = run_and_get_code(compiled, a)
        assert not any("round_output" in src for src in training)
        y.float().sum().backward()
        assert model.bias.grad is not None


def test_vector_gelu_bf16_encodings(p800):
    from triton.experimental.graph_fusion import _p800_flat

    n = 1024
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    matrix = words.view(torch.bfloat16).reshape(64, n).cuda()
    bias = torch.zeros_like(matrix[0])
    spec = dict(kind='gelu_exact', m=64, n=n, total=65536, dtype='bfloat16')
    run, proof = _p800_flat.compile(spec, [matrix, bias])
    with torch.no_grad():
        for offset in (0., -0., .5):
            bias.fill_(offset)
            actual, = run((matrix, bias))
            reference = torch.nn.functional.gelu(matrix + bias).cpu()
            output = actual.cpu()
            assert torch.equal(output.isnan(), reference.isnan())
            valid = ~reference.isnan()
            assert torch.equal(output.view(torch.int16)[valid], reference.view(torch.int16)[valid])
    assert proof['resource_verified'] and proof['emitted'] == 'bf16_vector_gelu'
