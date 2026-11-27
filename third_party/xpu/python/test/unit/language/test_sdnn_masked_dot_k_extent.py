"""Masked DMA must preserve dot's logical reduction and SDNN K packing."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def masked_dot(a, b, out, valid_k, WIDTH: tl.constexpr, MASK_SIDE: tl.constexpr, OTHER: tl.constexpr):
    rows = tl.arange(0, 32)
    k = tl.arange(0, WIDTH)
    if MASK_SIDE == 1 or MASK_SIDE == 2:
        lhs = tl.load(a + rows[:, None] * WIDTH + k[None, :], k[None, :] < valid_k, other=OTHER)
    else:
        lhs = tl.load(a + rows[:, None] * WIDTH + k[None, :])
    if MASK_SIDE == 0 or MASK_SIDE == 2:
        rhs = tl.load(b + k[:, None] * 32 + rows[None, :], k[:, None] < valid_k, other=OTHER)
    else:
        rhs = tl.load(b + k[:, None] * 32 + rows[None, :])
    value = tl.dot(lhs.to(tl.float32), rhs.to(tl.float32), allow_tf32=False)
    tl.store(out + rows[:, None] * 32 + rows[None, :], value)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('width,valid_k', [(16, 0), (16, 1), (16, 15), (16, 16), (32, -3), (32, 0), (32, 33), (64, -3),
                                           (64, 0), (64, 65), (32, 1), (32, 16), (32, 17), (32, 31), (32, 32), (64, 33),
                                           (64, 48), (64, 49), (64, 63), (64, 64)])
@pytest.mark.parametrize('mask_side', [0, 1, 2])
@pytest.mark.parametrize('other', [0., 3.])
def test_masked_dot_k_extent(dtype, width, valid_k, mask_side, other):
    a = (torch.arange(32 * width).reshape(32, width) % 19 - 9).to(dtype)
    b = (torch.arange(width * 32).reshape(width, 32) % 13 - 6).to(dtype)
    lhs, rhs = a.double(), b.double()
    masked_from = max(0, min(width, valid_k))
    if mask_side in (1, 2): lhs[:, masked_from:] = other
    if mask_side in (0, 2): rhs[masked_from:, :] = other
    reference = lhs @ rhs
    result = torch.full((32 * 32 + 64, ), -999., device='cuda')
    masked_dot[(1, )](a.cuda(), b.cuda(), result, valid_k, width, mask_side, other, num_warps=8, num_stages=1,
                      is_sdnn=True)
    actual = result.cpu()
    torch.testing.assert_close(actual[:1024].reshape(32, 32).double(), reference, rtol=0., atol=0.)
    assert bool((actual[1024:] == -999.).all())


@pytest.mark.parametrize('valid_k', [1, 16, 17, 31, 32])
@pytest.mark.parametrize('row', [0, 15, 16, 31])
@pytest.mark.parametrize('col', [0, 31])
def test_unmasked_lhs_nan_row_is_preserved(valid_k, row, col):
    a = torch.ones(32, 32)
    a[row, col] = float('nan')
    b = torch.ones(32, 32)
    reference = a.double() @ torch.cat((b[:valid_k], torch.zeros_like(b[valid_k:]))).double()
    result = torch.empty((32, 32), device='cuda')
    masked_dot[(1, )](a.cuda(), b.cuda(), result, valid_k, 32, 0, 0., num_warps=8, num_stages=1, is_sdnn=True)
    torch.testing.assert_close(result.cpu().double(), reference, rtol=0., atol=0., equal_nan=True)


@triton.jit
def loop_masked_dot(a, b, out, left_k, right_k, OTHER: tl.constexpr, INIT: tl.constexpr):
    rows = tl.arange(0, 32)
    k = tl.arange(0, 32)
    acc = tl.full((32, 32), INIT, tl.float32)
    for tile in range(2):
        kk = tile * 32 + k
        lhs = tl.load(a + rows[:, None] * 64 + kk[None, :], kk[None, :] < left_k, other=OTHER)
        rhs = tl.load(b + kk[:, None] * 32 + rows[None, :], kk[:, None] < right_k, other=OTHER)
        acc = tl.dot(lhs.to(tl.float32), rhs.to(tl.float32), acc, allow_tf32=False)
    tl.store(out + rows[:, None] * 32 + rows[None, :], acc)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('left_k,right_k', [(0, 0), (0, 17), (17, 0), (16, 17), (33, 48), (64, 64), (65, -1)])
@pytest.mark.parametrize('other', [0., 3.])
@pytest.mark.parametrize('initial', [0., 7.])
def test_loop_dot_independent_masks(dtype, left_k, right_k, other, initial):
    a = (torch.arange(32 * 64).reshape(32, 64) % 19 - 9).to(dtype)
    b = (torch.arange(64 * 32).reshape(64, 32) % 13 - 6).to(dtype)
    lhs, rhs = a.double(), b.double()
    lhs[:, max(0, min(64, left_k)):] = other
    rhs[max(0, min(64, right_k)):] = other
    reference = lhs @ rhs + initial
    output = torch.full((1088, ), -999., device='cuda')
    loop_masked_dot[(1, )](a.cuda(), b.cuda(), output, left_k, right_k, other, initial, num_warps=8, num_stages=1,
                           is_sdnn=True)
    actual = output.cpu()
    torch.testing.assert_close(actual[:1024].reshape(32, 32).double(), reference, rtol=0., atol=0.)
    assert bool((actual[1024:] == -999.).all())
