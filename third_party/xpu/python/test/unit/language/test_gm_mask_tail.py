"""P800 regressions for masked, non-contiguous GM-to-LM tail loads."""
import pytest
import triton
import triton.language as tl


@triton.jit
def masked_tail(src, dst, N: tl.constexpr, BLOCK: tl.constexpr,
                CLAMP: tl.constexpr, OTHER: tl.constexpr, VALUES: tl.constexpr):
    index = tl.arange(0, BLOCK)
    address = tl.where(index < N, index, 0) if CLAMP else index
    value = tl.load(src + address, index < N, other=OTHER).to(tl.float32)
    if VALUES:
        tl.store(dst + index, value)
    else:
        tl.store(dst, tl.sum(value, 0))


@pytest.fixture(scope="module")
def p800():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("P800 device required")
    from triton.backends import backends
    from triton.runtime import driver
    xpu = backends.get("xpu")
    if xpu is None or not isinstance(driver.active, xpu.driver) or driver.active.get_current_target().arch != 3:
        pytest.skip("P800 device required")
    return torch


# 33/64 has one element per core and missed the original defect. The larger
# tiles have partially valid cores and need individual DMA predicates.
@pytest.mark.parametrize("n,block,other,values", [
    (33, 64, 0, False), (129, 256, 0, False), (1000, 1024, 0, False),
    (1017, 1024, 7, False), (1017, 1024, 7, True),
])
@pytest.mark.parametrize("clamp", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int8"])
def test_masked_tail_preserves_other(p800, n, block, other, values, clamp, dtype):
    torch = p800
    host = (torch.arange(block) % 101 + 1).to(getattr(torch, dtype))
    host[0] = 113
    # Physical padding keeps an incorrect baseline load inside the allocation.
    # The reference depends only on the logical input and the specified fill.
    source = host.cuda()
    count = block if values else 1
    output = torch.full((count + 64,), -12345., device="cuda")
    masked_tail[(1,)](source[:n], output[:count], n, block, clamp, other, values,
                      num_warps=4, num_stages=1, is_sdnn=False)
    actual = output.cpu()
    expected = torch.cat((host[:n].float(), torch.full((block - n,), float(other))))
    if not values:
        expected = expected.sum().reshape(1)
    torch.testing.assert_close(actual[:count], expected, rtol=0, atol=0)
    assert bool((actual[count:] == -12345.).all())
    assert torch.equal(source.cpu(), host)
