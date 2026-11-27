"""KL3 ordered-maximum preserves the operands selected by its device comparison."""
import random
import pytest
import triton
import triton.language as tl


@triton.jit
def _compare_and_max(left, right, decision, output, BLOCK: tl.constexpr):
    if BLOCK == 1:
        i = tl.program_id(0)
    else:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(left + i).to(tl.float32, bitcast=True)
    b = tl.load(right + i).to(tl.float32, bitcast=True)
    tl.store(decision + i, (a > b).to(tl.int32))
    selected = tl.where(a > b, a, b)
    tl.store(output + i, selected.to(tl.int32, bitcast=True))


@pytest.mark.parametrize('block', [1, 512])
def test_ordered_max_preserves_kl3_comparison_and_operand_bits(block):
    import torch
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        pytest.skip('requires a KL3 device')
    if target.arch != 3 or target.warp_size != 1:
        pytest.skip('requires a KL3 device')
    special = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc54321, 0xffc12345,
               0x7f800001, 0xff800001, 0x7f7fffff, 0xff7fffff, 1, 2, 0x80000001,
               0x807fffff, 0x00800000, 0x3f800000, 0xc1200000]
    pairs = [(a, b) for a in special for b in special]
    rng = random.Random(53)
    pairs.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(512 - len(pairs)))
    tensors = [torch.tensor(col, dtype=torch.int64).to(torch.int32) for col in zip(*pairs)]
    a, b = [x.cuda() for x in tensors]
    comparison = torch.empty_like(a)
    output = torch.empty_like(a)
    _compare_and_max[(512 // block,)](a, b, comparison, output, BLOCK=block, num_warps=4, is_sdnn=False)
    expected = torch.where(comparison.cpu().bool(), *tensors)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
