"""Independent tiled stores must each use the corresponding index slice."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def shared_index_stores(first, second, stamp, B: tl.constexpr,
                        DERIVED: tl.constexpr, REVERSE: tl.constexpr):
    i = tl.arange(0, B)
    index = i * 3 + 5 if DERIVED else i
    x = (index % 101 + 11).to(tl.float32)
    y = (index % 47 - 23).to(tl.float32)
    if REVERSE:
        tl.store(stamp + i, index + 1)
        tl.store(second + i, y)
        tl.store(first + i, x)
    else:
        tl.store(first + i, x)
        tl.store(second + i, y)
        tl.store(stamp + i, index + 1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('block', [64, 128, 256, 1024, 8192])
@pytest.mark.parametrize('derived', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_shared_index_stores(dtype, block, derived, reverse):
    first = torch.full((block + 64,), -99, dtype=dtype, device='cuda')
    second = torch.full_like(first, -99)
    stamp = torch.full((block + 64,), -999, dtype=torch.int32, device='cuda')
    binary = shared_index_stores[(1,)](first, second, stamp, block, derived, reverse,
                                      num_warps=4, num_stages=1, is_sdnn=False)
    i = torch.arange(block, dtype=torch.int32)
    index = i * 3 + 5 if derived else i
    expected_first = (index % 101 + 11).to(dtype)
    expected_second = (index % 47 - 23).to(dtype)
    actual_first, actual_second, actual_stamp = first.cpu(), second.cpu(), stamp.cpu()
    if os.environ.get('AUDIT_ROOT') and os.environ.get('AUDIT_RUN'):
        name = f'{str(dtype).split(".")[-1]}-b{block}-d{int(derived)}-r{int(reverse)}'
        root = Path(os.environ['AUDIT_ROOT']) / 'artifacts' / os.environ['AUDIT_RUN'] / name
        root.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for ext, value in binary.asm.items():
            data = value if isinstance(value, bytes) else value.encode()
            (root / ('kernel.' + ext)).write_bytes(data)
            hashes[ext] = hashlib.sha256(data).hexdigest()
        (root / 'metadata.json').write_text(json.dumps(dict(hashes=hashes,
            metadata=binary.metadata._asdict()), default=str, indent=2))
        torch.save(dict(first=actual_first, second=actual_second, stamp=actual_stamp,
                        expected_first=expected_first, expected_second=expected_second,
                        expected_stamp=index + 1), root / 'values.pt')
    assert torch.equal(actual_first[:block], expected_first)
    assert torch.equal(actual_second[:block], expected_second)
    assert torch.equal(actual_stamp[:block], index + 1)
    assert bool((actual_first[block:] == -99).all())
    assert bool((actual_second[block:] == -99).all())
    assert bool((actual_stamp[block:] == -999).all())
