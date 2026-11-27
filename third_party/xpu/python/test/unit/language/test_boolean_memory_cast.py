"""Boolean memory uses logical conversions before its byte representation."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def boolean_memory(a, values, stored, loaded, B: tl.constexpr, MASKED: tl.constexpr):
    r = tl.arange(0, B)
    value = tl.load(values + r)
    if MASKED:
        tl.store(stored + r, value, (r % 3) != 0)
    else:
        tl.store(stored + r, value)
    result = tl.load(a + r, (r % 3) == 0, other=value)
    tl.store(loaded + r, result)


@triton.jit
def boolean_scalar(a, values, stored, loaded, ENABLED: tl.constexpr):
    value = tl.load(values)
    tl.store(stored, value, ENABLED)
    result = tl.load(a, not ENABLED, other=value)
    tl.store(loaded, result)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize('masked', [False, True])
def test_boolean_memory_conversion(dtype, masked):
    block = 256
    if dtype in (torch.int32, torch.int64):
        samples = [-512, -256, -1, 0, 1, 2, 255, 256, 257, 512]
    else:
        samples = [-float('inf'), -256, -0.5, -0.0, 0.0, 0.5, 256, float('inf'), float('nan')]
    r = torch.arange(block)
    values = torch.tensor(samples, dtype=dtype)[r % len(samples)]
    truth = values != 0
    a = (r % 2) == 0
    stored = torch.zeros(block + 64, dtype=torch.bool, device='cuda')
    loaded = torch.zeros_like(stored)
    boolean_memory[(1,)](a.cuda(), values.cuda(), stored, loaded, block, masked,
                         num_warps=16, num_stages=1, is_sdnn=False)
    expected_store = truth & ((r % 3) != 0) if masked else truth
    expected_load = torch.where((r % 3) == 0, a, truth)
    for actual, expected in [(stored.cpu(), expected_store), (loaded.cpu(), expected_load)]:
        assert torch.equal(actual[:block], expected)
        assert bool((actual.view(torch.uint8) <= 1).all())
        assert not bool(actual[block:].any())


@pytest.mark.parametrize('value', [256, -256, 0, 2])
@pytest.mark.parametrize('enabled', [False, True])
def test_boolean_scalar_conversion(value, enabled):
    a = torch.tensor([False], device='cuda')
    values = torch.tensor([value], dtype=torch.int32, device='cuda')
    stored = torch.zeros(65, dtype=torch.bool, device='cuda')
    loaded = torch.zeros_like(stored)
    boolean_scalar[(1,)](a, values, stored, loaded, enabled,
                         num_warps=16, num_stages=1, is_sdnn=False)
    for result in (stored.cpu(), loaded.cpu()):
        assert bool(result[0]) == (enabled and value != 0)
        assert not bool(result[1:].any())
        assert bool((result.view(torch.uint8) <= 1).all())
