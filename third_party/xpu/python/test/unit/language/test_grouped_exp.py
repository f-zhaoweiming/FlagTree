"""KL3 natural-exp lowering through private LM, with independent live results."""
import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _two_exp(src_desc, pos_desc, neg_desc, ROWS: tl.constexpr, COLS: tl.constexpr):
    off = tl.program_id(0) * ROWS
    buf = tle.gpu.alloc([ROWS, COLS], dtype=tl.float32, scope=tle.gpu.lmem)
    rows = tl.broadcast_to(tl.arange(0, ROWS)[:, None], (ROWS, COLS))
    cols = tl.broadcast_to(tl.arange(0, COLS)[None, :], (ROWS, COLS))
    ptr = tle.gpu.local_ptr(buf, (rows, cols))
    tle.gpu.copy(src_desc, buf, [ROWS, COLS], [off, 0])
    x = tl.load(ptr)
    positive = tl.exp(x)
    negative = tl.exp(-x)
    tl.store(ptr, positive)
    tle.gpu.copy(buf, pos_desc, [ROWS, COLS], [off, 0])
    tl.store(ptr, negative)
    tle.gpu.copy(buf, neg_desc, [ROWS, COLS], [off, 0])


@pytest.mark.parametrize('rows,cols', [(8, 128), (16, 2048)])
def test_grouped_exp_private_lm(rows, cols, monkeypatch):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        pytest.skip('requires KL3')
    if target.arch != 3 or target.warp_size != 1:
        pytest.skip('requires KL3')
    monkeypatch.setenv('TRITON_STORE_BINARY_ONLY', '0')
    torch.manual_seed(37)
    x = (torch.randn(rows * 3, cols) * 12).cuda()
    x[0, :8] = torch.tensor([0., -0., 1., -1., 90., -90., float('inf'), float('nan')]).cuda()
    pos, neg = torch.empty_like(x), torch.empty_like(x)
    descriptors = [TensorDescriptor.from_tensor(t, [rows, cols]) for t in (x, pos, neg)]
    binary = _two_exp[(3,)](*descriptors, rows, cols, num_warps=8, num_stages=1)
    for output, value in [(pos, x), (neg, -x)]:
        torch.testing.assert_close(output.cpu(), torch.exp(value.cpu()), rtol=3e-6,
                                   atol=torch.finfo(torch.float32).tiny, equal_nan=True)
    ir = binary.asm['llir']
    assert 'asm sideeffect "mfence { lm }' in ir
    assert 'call <16 x float> @llvm.exp.' not in ir
    assert 'call <16 x float> @llvm.exp2.' not in ir
    # Both exponentials share one stack allocation across all programs.
    assert len(re.findall(r'alloca (?:\[4 x <16 x float>\]|<16 x float>, i32 4)', ir)) == 1
