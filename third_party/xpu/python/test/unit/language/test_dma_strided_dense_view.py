"""A masked gather produces a dense LM value, including stride-zero other lanes."""
import pytest
import torch
import triton
import triton.language as tl

@triton.jit
def strided_read(a, out, N, STEP: tl.constexpr, B: tl.constexpr, MODE: tl.constexpr):
    r = tl.arange(0, B)
    offset = r * STEP + B * 3
    valid = r < N
    if MODE == 1:
        valid = valid & (r % 3 != 0)
    if MODE == 2:
        valid = valid & (r >= N)
    other = (r % 3 == 0) if a.dtype.element_ty == tl.int1 else (r % 17 - 8)
    value = tl.load(a + offset, valid, other=other)
    tl.store(out + r, value)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int64, torch.bool])
@pytest.mark.parametrize('step', [-3, -1, 0, 1, 2, 3])
@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('block', [512, 2048])
def test_strided_masked_read(dtype, step, mode, block):
    r = torch.arange(block, dtype=torch.int64)
    data = ((torch.arange(7 * block + 64) % 59 + 3).to(dtype) if dtype != torch.bool
            else torch.arange(7 * block + 64) % 5 == 0)
    valid = r < 129
    if mode == 1:
        valid &= r % 3 != 0
    if mode == 2:
        valid &= r >= 129
    other = (r % 3 == 0) if dtype == torch.bool else (r % 17 - 8).to(dtype)
    expected = torch.where(valid, data[r * step + block * 3], other)
    sentinel = False if dtype == torch.bool else -33
    out = torch.full((block + 64,), sentinel, dtype=dtype, device='cuda')
    strided_read[(1,)](data.cuda(), out, 129, step, block, mode, num_warps=4, num_stages=1, is_sdnn=False)
    actual = out.cpu()
    assert torch.equal(actual[:block], expected)
    assert bool((actual[block:] == sentinel).all())
    if dtype == torch.bool:
        assert bool((actual.view(torch.uint8) <= 1).all())
