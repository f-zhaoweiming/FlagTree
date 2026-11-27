"""TLE DMA and vector accesses must address the same aligned local buffer."""
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
def _prefilled_row_sum(desc, out, R: tl.constexpr, B: tl.constexpr,
                       FILL: tl.constexpr):
    pid = tl.program_id(0)
    buffer = tle.gpu.alloc([R, B], desc.dtype, scope=tle.gpu.lmem)
    row = tl.broadcast_to(tl.arange(0, R)[:, None], (R, B))
    col = tl.broadcast_to(tl.arange(0, B)[None, :], (R, B))
    ptr = tle.gpu.local_ptr(buffer, (row, col))
    tl.store(ptr, tl.full((R, B), FILL, tl.float32).to(desc.dtype))
    tle.gpu.copy(desc, buffer, [R, B], [pid * R, 0])
    values = tl.load(ptr).to(tl.float32)
    tl.store(out + pid * R + tl.arange(0, R), tl.sum(values, 1))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows,columns', [(8, 128), (16, 128), (8, 256),
                                         (16, 256), (8, 1024), (16, 1024)])
@pytest.mark.parametrize('fill', [7, -19])
def test_tle_lmem_alignment(dtype, rows, columns, fill):
    if not torch.cuda.is_available():
        pytest.skip('requires P800')
    driver = triton.runtime.driver.active
    if type(driver).__module__ != 'triton.backends.xpu.driver' or driver.get_current_target().arch != 3:
        pytest.skip('requires P800')
    programs = 3
    r = torch.arange(programs * rows, dtype=torch.int32)[:, None]
    c = torch.arange(columns, dtype=torch.int32)[None, :]
    source = ((r * 31 + c * c + c * 17) % 251 - 125).to(dtype)
    expected = source.float().sum(1)
    x = source.cuda()
    y = torch.full((programs * rows + 64,), -12345., device='cuda')
    desc = TensorDescriptor.from_tensor(x, block_shape=[rows, columns])
    binary = _prefilled_row_sum[(programs,)](desc, y, rows, columns, fill,
        num_warps=4, num_stages=1, is_sdnn=False)
    actual = y.cpu()
    if os.environ.get('AUDIT_ROOT') and os.environ.get('AUDIT_RUN'):
        root = Path(os.environ['AUDIT_ROOT']) / 'artifacts' / os.environ['AUDIT_RUN']
        root = root / f'{str(dtype).split(".")[-1]}-{rows}-{columns}-fill{fill}'
        root.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for ext, value in binary.asm.items():
            data = value if isinstance(value, bytes) else value.encode()
            (root / ('kernel.' + ext)).write_bytes(data)
            hashes[ext] = hashlib.sha256(data).hexdigest()
        (root / 'metadata.json').write_text(json.dumps(dict(hashes=hashes,
            metadata=binary.metadata._asdict()), default=str, indent=2))
        torch.save(dict(input=source, expected=expected, actual=actual), root / 'values.pt')
    torch.testing.assert_close(actual[:programs * rows], expected, rtol=0, atol=0)
    assert bool((actual[programs * rows:] == -12345.).all())
    assert torch.equal(x.cpu(), source)
