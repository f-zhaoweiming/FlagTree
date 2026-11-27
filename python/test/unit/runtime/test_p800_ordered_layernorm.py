"""The KL3 rounding order matters when Add + LayerNorm sees a large common offset."""
import pytest
import torch


@pytest.fixture
def p800():
    from triton.runtime import driver
    if not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver':
        pytest.skip("requires the P800 Torch SDK")
    if driver.active.get_current_target().arch != 3:
        pytest.skip("requires P800")
    from triton.experimental.graph_fusion import _p800_layernorm as kernel
    assert kernel.toolchain_available()
    return kernel


@pytest.mark.parametrize("m,n,dtype", [(64, 1024, torch.float32), (512, 1024, torch.float16),
                                       (2048, 2048, torch.float32), (2048, 2048, torch.float16),
                                       (64, 1024, torch.bfloat16), (512, 1024, torch.bfloat16),
                                       (2048, 2048, torch.bfloat16)])
def test_conditioned_layernorm(p800, m, n, dtype):
    from torch._inductor.utils import run_and_get_code
    from triton.experimental.graph_fusion._p800_reduce import _decline_layernorm

    torch.manual_seed(137)
    # A contiguous view can start at a nonzero storage offset.
    a = torch.empty(m*n + 1, dtype=dtype, device='cuda')[1:].view(m, n)
    a.copy_((1000 + .1 * torch.randn(m, n, device='cuda')).to(dtype))
    bias, weight, shift = [torch.randn(n, device='cuda', dtype=dtype) for _ in range(3)]

    def operation(x, b, w, h):
        return torch.nn.functional.layer_norm(x + b, (n,), w, h, .001)

    with torch.no_grad():
        expected = operation(a, bias, weight, shift)
        compiled = torch.compile(operation, fullgraph=True, dynamic=False)
        actual, code = run_and_get_code(compiled, a, bias, weight, shift)
        assert any('tle.raw.call(kl3_add_layernorm' in unit for unit in code)
        dt = str(dtype).removeprefix('torch.')
        assert not _decline_layernorm(m, n, dt)
        width = p800.segment_width(m, n, dt)
        rows = p800.rows_per_program(m)
        cluster = torch.empty_like(a)
        p800._row_layernorm[((m + rows - 1) // rows,)](a, bias, weight, shift, cluster, n, width, .001, True, rows, m, num_warps=4)
        tol = .0003 if dtype == torch.float32 else .003
        wide = operation(a.float(), bias.float(), weight.float(), shift.float()).to(dtype)
        torch.testing.assert_close(actual.cpu(), wide.cpu(), atol=tol, rtol=tol, equal_nan=True)
        # Direct invocation of the dtype-specific vendor compatibility primitive.
        torch.testing.assert_close(cluster.cpu(), expected.cpu(), atol=tol, rtol=tol, equal_nan=True)


def test_cached_affine_launch_uses_live_inputs_and_stream(p800):
    from triton.experimental.graph_fusion import _p800_reduce

    m, n = 64, 1024
    spec = dict(kind='add_layernorm_affine', m=m, n=n, dtype='float32', eps=.001,
                keepdim=False, nparam=2)
    torch.manual_seed(191)
    with torch.no_grad():
        a = (1000 + .1 * torch.randn(m, n, device='cuda')).contiguous()
        b, w, h = [torch.randn(n, device='cuda') for _ in range(3)]
        run, proof = _p800_reduce.compile(spec, [a, b, w, h])
        first, = run([a, b, w, h])
        saved = first.cpu()
        expected = torch.nn.functional.layer_norm(a + b, (n,), w, h, .001)
        torch.testing.assert_close(saved, expected.cpu(), atol=.0003, rtol=.0003)
        assert proof['resource_verified']

        # Calibration passed detached weights. The public frame supplies live
        # Parameters, including replacements and valid nonzero-offset views.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for offset in (0, 1, 0):
                a2 = torch.empty(m * n + offset, device='cuda')[offset:].view(m, n)
                a2.copy_(a - 16)
                shape_bias = torch.randn(n + offset, device='cuda')[offset:]
                gamma = torch.nn.Parameter(torch.randn(n + offset, device='cuda')[offset:])
                beta = torch.nn.Parameter(torch.randn(n, device='cuda'))
                answer, = run([a2, shape_bias, gamma, beta])
                oracle = torch.nn.functional.layer_norm(a2 + shape_bias, (n,), gamma, beta, .001)
                torch.testing.assert_close(answer.cpu(), oracle.cpu(), atol=.0003, rtol=.0003)
                assert answer.data_ptr() != first.data_ptr()
        stream.synchronize()
        torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)


def test_full_tile_launcher_without_extent(p800):
    m = n = 2048
    with torch.no_grad():
        a = (1000 + .1 * torch.randn(m, n, device='cuda')).float()
        b = torch.randn(n, device='cuda')
        y = torch.full_like(a, float('nan'))
        rows = p800.rows_per_program(m)
        width = p800.segment_width(m, n, 'float32')
        p800._row_layernorm[(m // rows,)](a, b, b, b, y, n, width, .001, False, rows, num_warps=4)
        expected = torch.nn.functional.layer_norm(a + b, (n,), eps=.001)
        torch.testing.assert_close(y.cpu(), expected.cpu(), rtol=.0003, atol=.0003, equal_nan=True)


@pytest.mark.parametrize("m,n", [(64, 1024), (512, 1024), (2048, 2048)])
def test_bf16_reduction_order_across_inputs(p800, m, n):
    # Near a large common offset, changing the BF16 lane order changes the
    # rounded FP32 second moment and can affect a whole row after normalization.
    rows = p800.rows_per_program(m, allow_tail=True, dtype="bfloat16")
    width = p800.segment_width(m, n, "bfloat16")
    with torch.no_grad():
        for seed in (7, 23, 49, 61, 87, 99, 137):
            generator = torch.Generator().manual_seed(seed)
            matrix = torch.randn((m, n), generator=generator).bfloat16().cuda()
            bias, gamma, beta = [torch.randn(n, generator=generator).bfloat16().cuda() for _ in range(3)]
            for spread in (.1, 8.):
                a = (1000 + matrix.float() * spread).bfloat16()
                for affine in (False, True):
                    ref = torch.nn.functional.layer_norm(a + bias, (n,), gamma if affine else None,
                                                         beta if affine else None, .001).cpu()
                    out = torch.empty_like(a)
                    p800._row_layernorm[((m + rows - 1) // rows,)](a, bias, gamma, beta, out, n, width, .001,
                                                                  affine, rows, m, num_warps=4, num_stages=1)
                    torch.testing.assert_close(out.cpu(), ref, rtol=.003, atol=.03, equal_nan=True)
