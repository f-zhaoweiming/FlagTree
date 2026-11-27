"""Local scalar writes must finish before SIMD loads of the same buffer."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _local_row_sum(desc, out, seed, R: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    buffer = tle.gpu.alloc([R, B], desc.dtype, scope=tle.gpu.lmem)
    row = tl.broadcast_to(tl.arange(0, R)[:, None], (R, B))
    col = tl.broadcast_to(tl.arange(0, B)[None, :], (R, B))
    ptr = tle.gpu.local_ptr(buffer, (row, col))
    values = (((pid * R + row) * 31 + col * col + col * 17 + seed) % 251 - 125).to(desc.dtype)
    tl.store(ptr, values)
    reloaded = tl.load(ptr).to(tl.float32)
    tl.store(out + pid * R + tl.arange(0, R), tl.sum(reloaded, 1))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows,columns', [(32, 128), (8, 128), (16, 128),
                                         (32, 256), (8, 256), (16, 256),
                                         (32, 1024), (8, 1024), (16, 1024)])
def test_tle_lmem_ordering(dtype, rows, columns):
    if not torch.cuda.is_available():
        pytest.skip('requires P800')
    driver = triton.runtime.driver.active
    if type(driver).__module__ != 'triton.backends.xpu.driver' or driver.get_current_target().arch != 3:
        pytest.skip('requires P800')
    programs = 3
    r = torch.arange(programs * rows, dtype=torch.int32)[:, None]
    c = torch.arange(columns, dtype=torch.int32)[None, :]
    source = torch.zeros((programs * rows, columns), dtype=dtype)
    x = source.cuda()
    desc = TensorDescriptor.from_tensor(x, block_shape=[rows, columns])
    out = torch.full((programs * rows + 128,), -12345., device='cuda')
    # Vary a runtime input while reusing the same compiled kernel and allocation.
    # Nonuniform values expose stale elements that a constant fill can hide.
    for seed in [0, 47, 149]:
        out.fill_(-12345.)
        binary = _local_row_sum[(programs,)](desc, out, seed, rows, columns,
                                            num_warps=4, num_stages=1, is_sdnn=False)
        expected = ((r * 31 + c * c + c * 17 + seed) % 251 - 125).to(dtype).float().sum(1)
        actual = out.cpu()
        if os.environ.get('AUDIT_ROOT') and os.environ.get('AUDIT_RUN'):
            root = Path(os.environ['AUDIT_ROOT']) / 'artifacts' / os.environ['AUDIT_RUN']
            root = root / f'ordering-{dtype}-{rows}-{columns}-seed{seed}'
            root.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for ext, value in binary.asm.items():
                data = value if isinstance(value, bytes) else value.encode()
                (root / ('kernel.' + ext)).write_bytes(data)
                hashes[ext] = hashlib.sha256(data).hexdigest()
            (root / 'metadata.json').write_text(json.dumps(dict(hashes=hashes,
                metadata=binary.metadata._asdict()), default=str, indent=2))
            torch.save(dict(expected=expected, actual=actual, seed=seed), root / 'values.pt')
        torch.testing.assert_close(actual[:programs * rows], expected, rtol=0, atol=0)
        assert bool((actual[programs * rows:] == -12345.).all())
        assert torch.equal(x.cpu(), source)
