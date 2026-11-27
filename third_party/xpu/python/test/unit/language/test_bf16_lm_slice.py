"""BF16 round trips at KL3 local-memory slice boundaries."""
import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra.xpu import libdevice


@triton.jit
def _broadcast_round(x, bias, rounded, narrow, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x + i).to(tl.float32)
    b = tl.load(bias + i % N).to(tl.float32)
    total = a + b
    tl.store(rounded + i, total.to(tl.bfloat16).to(tl.float32))
    tl.store(narrow + i, total)


@triton.jit
def _tanh(x, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, libdevice.tanh(tl.load(x + i)))


@pytest.fixture
def kl3():
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != 3 or target.warp_size != 1:
        pytest.skip('requires KL3')


@pytest.mark.parametrize('block', [1024, 2048, 4096, 8192])
def test_bf16_lm_slice_rounding(kl3, block):
    # Vary both operands across the 16-lane boundary. Include the midpoint
    # between adjacent BF16 numbers with either parity of the retained bit.
    rng = torch.Generator().manual_seed(61)
    a = torch.randn((16, 1024), generator=rng).bfloat16()
    b = torch.randn((1024,), generator=rng).bfloat16()
    a[:, :4] = torch.tensor([1., 1.0078125, -1., -1.0078125])
    b[:4] = torch.tensor([1/256, 1/256, -1/256, -1/256])
    b[4:7] = torch.tensor([float('nan'), float('inf'), -float('inf')])
    expected = (a.float() + b.float()).bfloat16()
    a, b = a.cuda(), b.cuda()
    rounded = torch.empty_like(a, dtype=torch.float32)
    narrow = torch.empty_like(a)
    _broadcast_round[(a.numel() // block,)](a, b, rounded, narrow, 1024, block, num_warps=4)
    torch.testing.assert_close(rounded.cpu(), expected.float(), atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(narrow.cpu(), expected, atol=0, rtol=0, equal_nan=True)


def test_cluster_tanh_saturation(kl3):
    x = torch.tensor([-float('inf'), -100., -16., -1., 1., 16., 100., float('inf'), float('nan')])
    x = x.repeat(16)[:128]
    y = torch.empty_like(x, device='cuda')
    _tanh[(1,)](x.cuda(), y, 128, num_warps=4)
    torch.testing.assert_close(y.cpu(), torch.tanh(x), rtol=2e-6, atol=1e-7, equal_nan=True)
