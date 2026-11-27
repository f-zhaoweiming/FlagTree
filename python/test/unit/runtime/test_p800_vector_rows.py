"""Machine-specific regression cases for the KL3 row kernels."""
import pytest
import torch


@pytest.fixture
def p800():
    from triton.runtime import driver
    if (not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver'
            or driver.active.get_current_target().arch != 3):
        pytest.skip("requires P800")
    from triton.experimental.graph_fusion import _p800_vector_rows
    return _p800_vector_rows


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_default_inductor_fp32_softmax_exp_links(p800, dtype):
    def operation(x, bias):
        return torch.softmax(x + bias, dim=-1)

    with torch.no_grad():
        x = torch.randn((64, 1024), device='cuda').to(dtype)
        bias = torch.randn(1024, device='cuda').to(dtype)
        result = torch.compile(operation, fullgraph=True, dynamic=False)(x, bias)
        expected = torch.softmax(x.float() + bias.float(), dim=-1).to(dtype)
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=.003, atol=.003)


@pytest.mark.parametrize('keep', [False, True])
def test_compensated_product_sum_keeps_vendor_row_order(p800, keep):
    from torch._inductor.utils import run_and_get_code
    from triton.experimental.graph_fusion import _p800_reduce

    m, n = 512, 1024
    a = torch.ones((m, n), device='cuda')
    b = torch.ones((n,), device='cuda')

    def operation(x, bias):
        return (x * bias).sum(-1, keepdim=keep)

    spec = dict(kind='mul_sum', m=m, n=n, dtype='float32', eps=0., keepdim=keep, nparam=0)
    with torch.no_grad():
        compiled = torch.compile(operation, fullgraph=True, dynamic=False)
        _, code = run_and_get_code(compiled, a, b)
        assert any('tle.raw.call(kl3_sum_cross' in s for s in code)
        direct, proof = _p800_reduce.compile(spec, [a, b])
        for row, col in ((0, 0), (21, 0), (0, 512), (344, 511), (511, 1023)):
            a[row, col] = float('inf')
            expected = operation(a, b)
            for result in (compiled(a, b), direct([a, b])[0]):
                assert result.shape == expected.shape
                torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)
            a[row, col] = 1
        assert proof['resource_verified'] and proof['prepared_launch']


@pytest.mark.parametrize('kind', ['gelu_exact', 'gelu_tanh', 'bias_add'])
def test_inductor_bfloat16_matmul_rounds_after_epilogue(p800, kind):
    from torch._inductor.utils import run_and_get_code

    a = torch.randn(512, 64, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(64, 1024, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn_like(b[0])
    def operation(x, w, h):
        value = x @ w + h
        if kind == 'bias_add':
            return value
        return torch.nn.functional.gelu(value, approximate='none' if kind == 'gelu_exact' else 'tanh')

    with torch.no_grad():
        compiled = torch.compile(operation, fullgraph=True, dynamic=False)
        value, code = run_and_get_code(compiled, a, b, bias)
        assert any('float32' in source for source in code) and value.dtype == torch.bfloat16
        torch.testing.assert_close(value.cpu(), operation(a.float(), b.float(), bias.float()).bfloat16().cpu(), atol=.03, rtol=.003, equal_nan=True)
        bias.fill_(0.)
        second = compiled(a, b, bias)
        assert value.data_ptr() != second.data_ptr()
        torch.testing.assert_close(second.cpu(), operation(a.float(), b.float(), bias.float()).bfloat16().cpu(), atol=.03, rtol=.003, equal_nan=True)


@pytest.mark.parametrize('m,n', [(512, 1024), (2048, 2048)])
def test_fp32_silu_uses_updated_bias_and_vendor_division(p800, m, n):
    # Include the FP32 overflow threshold and both sides of zero as well as
    # changing every bias slice after the first launch.
    from triton.experimental.graph_fusion import _p800_flat
    with torch.no_grad():
        x = torch.randn((m, n), device='cuda')
        bias = torch.randn(n, device='cuda')
        gm = torch.fx.symbolic_trace(lambda a, b: torch.nn.functional.silu(a + b))
        spec = _p800_flat.match(gm, (x, bias))
        kernel, proof = _p800_flat.compile(spec, (x, bias))
        first, = kernel((x, bias))
        saved = first.clone()
        special = torch.tensor([-float('inf'), -90., -88.72284, -88., -87., -0., 0., 89.,
                                float('inf'), float('nan')], device='cuda')
        x[:, :special.numel()] = special
        bias = -bias
        bias[:special.numel()] = 0.
        second, = kernel((x, bias))
        expected = torch.nn.functional.silu(x + bias)
        torch.testing.assert_close(second.cpu(), expected.cpu(), atol=0., rtol=0., equal_nan=True)
        assert second.data_ptr() != first.data_ptr() and torch.equal(saved, first)
        assert proof['prepared_launch']
        if m == 512:
            assert proof['tiles']['elements_per_core'] == (672, 704)
        else:
            assert proof['tiles']['width'] == 512
