"""Exercise the supported F06 row-softmax launch ABIs on a real P800."""
import pytest
import torch


@pytest.fixture
def p800():
    from triton.runtime import driver
    if (not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver'
            or driver.active.get_current_target().arch != 3):
        pytest.skip('requires P800')


@pytest.mark.parametrize('dtype', (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize('m,n,row_bias', [(3, 17, False), (768, 1024, True), (5, 16385, True)])
def test_softmax_warm_launch_reads_live_inputs(p800, dtype, m, n, row_bias, monkeypatch):
    from triton.experimental.graph_fusion import _p800_reduce
    from triton.runtime.jit import JITFunction

    with torch.no_grad():
        x = torch.randn(m, n, device='cuda').to(dtype)
        bias = torch.randn((1, n) if row_bias else (n,), device='cuda').to(dtype)
        gm = torch.fx.symbolic_trace(lambda a, b: torch.softmax(a + b, dim=-1))
        spec = _p800_reduce.match(gm, (x, bias))
        assert spec is not None
        kernel, evidence = _p800_reduce.compile(dict(spec, final_round_only=True), (x, bias))
        first, = kernel((x, bias))
        saved = first.clone()
        assert evidence['prepared_launch'] and evidence['resource_verified']

        def unexpected_jit(*args, **kwargs):
            pytest.fail('warm Softmax re-entered JIT dispatch')

        monkeypatch.setattr(JITFunction, 'run', unexpected_jit)
        new_x = torch.randn_like(x)
        new_x[0, 0] = -float('inf')
        new_x[1, -1] = float('inf')
        new_bias = torch.randn_like(bias)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result, = kernel((new_x, new_bias))
            expected = torch.softmax(new_x.float() + new_bias.float(), -1).to(dtype)
        stream.synchronize()
        rtol = {torch.float32: 3e-4, torch.float16: .002, torch.bfloat16: .009}[dtype]
        torch.testing.assert_close(result.float().cpu(), expected.float().cpu(), atol=1e-7, rtol=rtol, equal_nan=True)
        assert result.dtype == dtype and result.data_ptr() != first.data_ptr() and torch.equal(first, saved)


@pytest.mark.parametrize('shape', [(0, 513), (2, 0), (5, 67)])
def test_method_softmax_reversed_add_and_empty(p800, shape):
    from triton.experimental.graph_fusion import _p800_reduce

    gm = torch.fx.symbolic_trace(lambda bias, x: bias.add(x).softmax(-1))
    dtype = torch.bfloat16
    with torch.no_grad():
        x = torch.randn(shape).to(device='cuda', dtype=dtype)
        bias = torch.randn(shape[-1]).to(device='cuda', dtype=dtype)
        spec = _p800_reduce.match(gm, (bias, x))
        assert spec and spec['swapped_inputs']
        fn, proof = _p800_reduce.compile(dict(spec, final_round_only=True), (bias, x))
        y, = fn((bias, x))
        reference = torch.softmax(x.float().cpu() + bias.float().cpu(), -1).to(dtype)
        torch.testing.assert_close(y.cpu().float(), reference.float(), rtol=.009, atol=1e-7)
        assert y.shape == shape and y.dtype == dtype
        if x.numel():
            assert proof['prepared_launch']
        else:
            assert proof['kernel_count'] == 0


def test_softmax_match_preserves_requested_dtype_and_axis():
    from triton.experimental.graph_fusion import _p800_reduce

    x, bias = torch.zeros(3, 17), torch.zeros(17)
    functions = (lambda x,b: (x + b).softmax(-1, torch.float64),
                 lambda x,b: torch.softmax(x + b, dim=0))
    for fn in functions:
        assert _p800_reduce.match(torch.fx.symbolic_trace(fn), (x, bias)) is None


@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16))
def test_default_inductor_wide_row_method(p800, dtype):
    def op(x, bias):
        return bias.add(x).softmax(-1)
    with torch.no_grad():
        x = torch.randn(5, 16385).to(device='cuda', dtype=dtype)
        b = torch.randn(1, 16385).to(device='cuda', dtype=dtype)
        compiled = torch.compile(op, fullgraph=True, dynamic=False)
        for v in (x, x * .5):
            result = compiled(v, b)
            expected = torch.softmax(v.cpu().float() + b.cpu().float(), -1).to(dtype)
            assert result.dtype == dtype
            torch.testing.assert_close(result.float().cpu(), expected.float(), rtol=.009 if dtype == torch.bfloat16 else 3e-4, atol=1e-7)
