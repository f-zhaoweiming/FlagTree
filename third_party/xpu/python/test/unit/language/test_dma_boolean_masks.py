"""Boolean DMA masks preserve holes, other values and suppressed writes."""
import pytest
import torch
import triton
import triton.language as tl

@triton.jit
def mask_io(a, out, stored, N: tl.constexpr, B: tl.constexpr, MODE: tl.constexpr, VARIABLE_OTHER: tl.constexpr=False):
    r = tl.arange(0, B)
    if MODE == 0:
        valid = (r < N) & (r < 129)
    elif MODE == 1:
        valid = (r < N) & (r > 0) & (r % 3 != 0)
    elif MODE == 2:
        valid = (r < N) & ((r < 7) | (r >= 125))
    elif MODE == 3:
        valid = (r * 3 < N) & (r > 0)
    elif MODE == 4:
        valid = (r < N) & (r >= N)
    else:
        valid = (r - 17 < N) & (r - 17 >= 0)
    # False reads stay inside the input allocation; false writes alias lane zero.
    # This exposes unwanted transfers without relying on a device fault.
    read_index = tl.where(valid, r, B)
    if VARIABLE_OTHER:
        other = r % 3 == 0 if a.dtype.element_ty == tl.int1 else r % 17 - 8
    else:
        other = -23
    value = tl.load(a + read_index, valid, other=other)
    tl.store(out + r, value)
    write_index = tl.where(valid, r, 0)
    tl.store(stored + write_index, r.to(tl.int32) + 1000, valid)

@triton.jit
def copy2d(a, out, stored, M: tl.constexpr, N: tl.constexpr, L: tl.constexpr, MODE: tl.constexpr, VARIABLE_OTHER: tl.constexpr=False):
    rows = tl.arange(0, 128)[:, None]
    cols = tl.arange(0, 128)[None, :]
    index = rows * N + cols
    if MODE == 2:
        index = rows * N + (N - 1 - cols)
    valid = (rows < M) & (cols < N) & (index < L)
    if MODE == 1:
        valid = valid & ((rows + cols) % 3 != 0)
    safe = tl.where(valid, index, L)
    other = (rows * 128 + cols) % 17 - 8 if VARIABLE_OTHER else -23
    values = tl.load(a + safe, valid, other=other)
    tl.store(out + rows * 128 + cols, values)
    destination = tl.where(valid, index, 0)
    tl.store(stored + destination, (rows * 128 + cols + 1000).to(tl.int32), valid)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int64, torch.bool])
@pytest.mark.parametrize('mode,n', [(0, 129), (1, 129), (2, 129), (3, 129), (4, 129), (5, 129), (0, 0)])
def test_boolean_dma_predicate(dtype, mode, n):
    _check_mask_io(dtype, mode, n, 256, False)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('no_unroll', [False, True])
def test_boolean_dma_large_tile(dtype, no_unroll):
    _check_mask_io(dtype, 1, 129, 2048, no_unroll)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int64, torch.bool])
@pytest.mark.parametrize('mode', [1, 4])
@pytest.mark.parametrize('no_unroll', [False, True])
def test_boolean_dma_position_dependent_other(dtype, mode, no_unroll):
    _check_mask_io(dtype, mode, 129, 2048, no_unroll, variable_other=True)

def _check_mask_io(dtype, mode, n, block, no_unroll, variable_other=False):
    r = torch.arange(block, dtype=torch.int32)
    if mode == 0:
        valid = (r < n) & (r < 129)
    elif mode == 1:
        valid = (r < n) & (r > 0) & (r % 3 != 0)
    elif mode == 2:
        valid = (r < n) & ((r < 7) | (r >= 125))
    elif mode == 3:
        valid = (r * 3 < n) & (r > 0)
    elif mode == 4:
        valid = (r < n) & (r >= n)
    else:
        valid = (r - 17 < n) & (r - 17 >= 0)
    is_bool = dtype == torch.bool
    data = torch.arange(block + 64) % 2 == 0 if is_bool else (torch.arange(block + 64) % 59 + 3).to(dtype)
    data[block:] = False if is_bool else -99
    fill = True if is_bool else -23
    sentinel = False if is_bool else -33
    other = (r % 3 == 0 if is_bool else (r % 17 - 8).to(dtype)) if variable_other else torch.full((block,), fill, dtype=dtype)
    expected = torch.where(valid, data[:block], other)
    expected_store = torch.where(valid, r + 1000, torch.full((block,), -777, dtype=torch.int32))
    a = data.cuda()
    out = torch.full((block + 64,), sentinel, dtype=dtype, device='cuda')
    stored = torch.full((block + 64,), -777, dtype=torch.int32, device='cuda')
    mask_io[1,](a, out, stored, n, block, mode, variable_other, num_warps=16, num_stages=1, is_sdnn=False, isCloseUnrollControl=no_unroll)
    actual, writes = (out.cpu(), stored.cpu())
    assert torch.equal(actual[:block], expected)
    assert torch.equal(writes[:block], expected_store)
    assert bool((actual[block:] == sentinel).all())
    assert bool((writes[block:] == -777).all())
    if is_bool:
        assert bool((actual[:block].view(torch.uint8) <= 1).all())

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('variable_other', [False, True])
def test_boolean_dma_multiple_rows_per_core(dtype, variable_other):
    m, n, length = (97, 17, 97 * 17 - 3)
    rows, cols = (torch.arange(128)[:, None], torch.arange(128)[None, :])
    index = rows * n + cols
    valid = (rows < m) & (cols < n) & (index < length) & ((rows + cols) % 3 != 0)
    data = (torch.arange(length + 128) % 59 + 3).to(dtype)
    data[length:] = -99
    expected = ((rows * 128 + cols) % 17 - 8).to(dtype) if variable_other else torch.full((128, 128), -23, dtype=dtype)
    expected[valid] = data[index[valid]]
    expected_store = torch.full((length + 128,), -777, dtype=torch.int32)
    expected_store[index[valid]] = (rows * 128 + cols + 1000)[valid].to(torch.int32)
    a = data.cuda()
    out = torch.full((16384 + 128,), -33, dtype=dtype, device='cuda')
    stored = torch.full((length + 128,), -777, dtype=torch.int32, device='cuda')
    copy2d[1,](a, out, stored, m, n, length, 1, variable_other, num_warps=16, num_stages=1, is_sdnn=False)
    actual, writes = (out.cpu(), stored.cpu())
    assert torch.equal(actual[:16384].reshape(128, 128), expected)
    assert torch.equal(writes, expected_store)
    assert bool((actual[16384:] == -33).all())


@triton.jit
def typed_masked_store(out, B: tl.constexpr, ALL_FALSE: tl.constexpr):
    r = tl.arange(0, B)
    valid = (r > 0) & (r < 129) & ((r % 3) != 0)
    if ALL_FALSE:
        valid = valid & (r >= B)
    destination = tl.where(valid, r, 0)
    tl.store(out + destination, r % 59 + 3, valid)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16,
                                  torch.int8, torch.int16, torch.int64, torch.bool])
@pytest.mark.parametrize('all_false', [False, True])
@pytest.mark.parametrize('block', [256, 2048])
def test_boolean_dma_typed_masked_store(dtype, all_false, block):
    r = torch.arange(block)
    valid = (r > 0) & (r < 129) & ((r % 3) != 0)
    if all_false:
        valid &= r >= block
    sentinel = False if dtype == torch.bool else -23
    expected = torch.full((block + 64,), sentinel, dtype=dtype)
    expected[:block][valid] = (r[valid] % 59 + 3).to(dtype)
    out = torch.full_like(expected, sentinel, device='cuda')
    typed_masked_store[(1,)](out, block, all_false, num_warps=16,
                             num_stages=1, is_sdnn=False)
    actual = out.cpu()
    assert torch.equal(actual, expected)
    if dtype == torch.bool:
        assert bool((actual.view(torch.uint8) <= 1).all())
