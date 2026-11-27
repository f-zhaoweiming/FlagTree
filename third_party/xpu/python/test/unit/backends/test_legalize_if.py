"""Captured splats yielded from a tiled loop must match both scf.if result edges."""
import pytest

import triton
import triton.language as tl
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


@triton.jit
def _branch_accum(src, dst, steps, REVERSE: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    lanes = tl.arange(0, 1024)
    offsets = rows[:, None] * 2048 + lanes[None, :]
    zero = tl.full((16, 1024), 0., tl.float32)
    one = tl.full((16, 1024), 1., tl.float32)
    total = zero
    count = zero
    for i in range(steps):
        x = tl.load(src + offsets + i * 1024)
        if REVERSE:
            if i != 0:
                total, count = (total + x) * count, count + 1.
            else:
                total, count = zero, one
        else:
            if i == 0:
                total, count = zero, one
            else:
                total, count = (total + x) * count, count + 1.
    tl.store(dst + rows, tl.sum(total + count, 1))


@pytest.mark.parametrize("reverse", [False, True])
def test_legalize_tiled_if_captured_splats(reverse):
    if not hasattr(libtriton, "xpu"):
        pytest.skip("requires the XPU backend")
    src = ASTSource(_branch_accum, {"src": "*fp32", "dst": "*fp32", "steps": "i32"},
                    constexprs={"REVERSE": reverse})
    kernel = triton.compile(src, target=GPUTarget("xpu", 3, 1), options={"buffer_size_limit": 64})
    assert kernel.asm["xpubin"]
