"""Rotating LM/SM pipeline slots must retain their address space when lowered."""
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
def _pipelined_row_sum(src, dst, n, COLS: tl.constexpr,
                       STAGES: tl.constexpr):
    row = tl.program_id(0) * 64
    buf = tle.gpu.alloc([64, COLS], src.dtype, scope=tle.gpu.lmem)
    result = tle.gpu.alloc([64], tl.float32, scope=tle.gpu.lmem)
    rows = tl.broadcast_to(tl.arange(0, 64)[:, None], (64, COLS))
    cols = tl.broadcast_to(tl.arange(0, COLS)[None, :], (64, COLS))
    ptr = tle.gpu.local_ptr(buf, (rows, cols))
    out = tle.gpu.local_ptr(result, (tl.arange(0, 64),))
    total = tl.zeros([64], tl.float32)
    for k in tl.range(0, n, COLS, num_stages=STAGES):
        tle.gpu.copy(src, buf, [64, COLS], [row, k])
        total += tl.sum(tl.load(ptr).to(tl.float32), 1)
    tl.store(out, total)
    tle.gpu.copy(result, dst, [64], [row])


CASES = [(cols, stages, 17) for cols in (1, 2, 4, 8, 16, 32)
         for stages in (1, 2, 3)]
CASES += [(cols, 3, trips) for cols in (8, 32) for trips in (1, 2, 3)]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('cols,stages,trips', CASES)
def test_tle_pipeline_address_space(dtype, cols, stages, trips, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('requires P800')
    driver = triton.runtime.driver.active
    if type(driver).__module__ != 'triton.backends.xpu.driver' or driver.get_current_target().arch != 3:
        pytest.skip('requires P800')
    monkeypatch.setenv('TRITONXPU_TLE_PIPELINE', '1')
    m, n = 128, cols * trips
    row = torch.arange(m)[:, None]
    col = torch.arange(n)[None, :]
    source = ((row * 7 + col * 3) % 29 - 14).to(dtype)
    expected = source.float().sum(1)
    x = source.cuda()
    output = torch.full((m + 32,), -12345., device='cuda')
    src = TensorDescriptor.from_tensor(x, block_shape=[64, cols])
    dst = TensorDescriptor.from_tensor(output[:m], block_shape=[64])
    binary = _pipelined_row_sum[(2,)](src, dst, n, cols, stages,
        num_warps=4, num_stages=stages)
    actual = output.cpu()
    if os.environ.get('AUDIT_ROOT') and os.environ.get('AUDIT_RUN'):
        root = Path(os.environ['AUDIT_ROOT']) / 'artifacts' / os.environ['AUDIT_RUN']
        root /= f'{str(dtype).split(".")[-1]}-c{cols}-s{stages}-t{trips}'
        root.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for ext, value in binary.asm.items():
            data = value if isinstance(value, bytes) else value.encode()
            (root / ('kernel.' + ext)).write_bytes(data)
            hashes[ext] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (root / 'metadata.json').write_text(json.dumps(dict(hashes=hashes,
            metadata=binary.metadata._asdict()), default=str, indent=2))
        torch.save(dict(input=source, expected=expected, actual=actual), root / 'values.pt')
    torch.testing.assert_close(actual[:m], expected, rtol=0, atol=0)
    assert bool((actual[m:] == -12345.).all())
    assert torch.equal(x.cpu(), source)
