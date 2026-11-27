"""Vendor-compatible FP32 moments for dense KL3 Add + LayerNorm."""
import os
from pathlib import Path

import triton
import triton.language as tl
import triton.experimental.tle as tle


def toolchain_available():
    # The optional XTDK distribution can omit the LLVM 19 linker for tle.raw.
    # Check at graph compilation, before offering a template to Inductor.
    from triton.backends.xpu import compiler
    return os.access(Path(compiler.__file__).parent / 'llvm19/bin/llvm-link', os.X_OK)


def segment_width(m, n, dtype):
    # KL3 stages 3072 input bytes per partial, except for the small FP32
    # row, whose four 256-element partials are reduced independently.
    # The one-core body has 4 KiB of FP32 scratch and consumes groups
    # of 64 lanes. The two-core body handles two exact 1024-column halves.
    if m <= 0 or n <= 0 or (n > 1024 and n != 2048) or m*n > 0x7fffffff:
        return None
    if dtype in ('bfloat16', 'float16'):
        return min(n, 1536)
    if dtype == 'float32':
        if n == 2048 and 128 < m <= 256:
            return 512
        return 256 if m <= 128 else 768
    return None


def rows_per_program(m, *, allow_tail=False, dtype=None):
    if m == 64:
        return 16
    if m == 512:
        return 48 if allow_tail and dtype == 'bfloat16' else 128
    # Callers using the original full-tile ABI can still use m // rows.
    # The public lowerings supply M and launch a twelfth, short program.
    return 176 if allow_tail and m == 2048 else 64


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_add_layernorm(a, bias, weight, shift, out, first, rows, n, segment, eps, affine, dtype_code):
    ...


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_quarter_float(a, bias, weight, shift, out, first, rows, eps, affine):
    ...


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_quarter_half(a, bias, weight, shift, out, first, rows, eps, affine):
    ...


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_quarter_bf(a, bias, weight, shift, out, first, rows, eps, affine):
    ...


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_bf_quarter_packed(a, bias, out, first, rows, eps):
    ...


@triton.jit
def _row_layernorm(A, B, W, SHIFT, OUT, N: tl.constexpr, SEGMENT: tl.constexpr,
                   EPS: tl.constexpr, AFFINE: tl.constexpr, ROWS: tl.constexpr, M: tl.constexpr = None, FINAL_ONLY: tl.constexpr = False):
    first = tl.program_id(0) * ROWS
    count = tl.full((), ROWS, tl.int32)
    if M is not None and M % ROWS:
        count = tl.minimum(ROWS, M - first)
    dtype_code = 2 if A.dtype.element_ty == tl.bfloat16 else A.dtype.element_ty == tl.float16
    if FINAL_ONLY:
        dtype_code = dtype_code + 2
    if N == 2048 and SEGMENT == 512 and A.dtype.element_ty == tl.bfloat16 and FINAL_ONLY and not AFFINE:
        tle.raw.call(kl3_bf_quarter_packed, (A, B, OUT, first, count, tl.full((), EPS, tl.float32)))
    elif N == 2048 and SEGMENT == 512:
        args = (A, B, W, SHIFT, OUT, first, count, tl.full((), EPS, tl.float32),
                tl.full((), AFFINE, tl.int32))
        if A.dtype.element_ty == tl.float32:
            tle.raw.call(kl3_quarter_float, args)
        elif A.dtype.element_ty == tl.float16:
            tle.raw.call(kl3_quarter_half, args)
        else:
            tle.raw.call(kl3_quarter_bf, args)
    else:
        tle.raw.call(kl3_add_layernorm, (A, B, W, SHIFT, OUT, first,
                     count, tl.full((), N, tl.int32), tl.full((), SEGMENT, tl.int32),
                     tl.full((), EPS, tl.float32), tl.full((), AFFINE, tl.int32),
                     tl.full((), dtype_code, tl.int32)))



@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_packed_bf16_norm(a, bias, weight, shift, out, first, rows, n, eps, affine):
    ...


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_packed_fp16_norm(a, bias, weight, shift, out, first, rows, n, eps, affine):
    ...


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_packed_bf16_norm_2048(a, bias, out, first, rows, eps):
    ...


@triton.jit
def _packed_layernorm(A, B, W, SHIFT, OUT, M: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
                      AFFINE: tl.constexpr, PROGRAMS: tl.constexpr):
    pid = tl.program_id(0)
    first = pid * (M // PROGRAMS) + tl.minimum(pid, M % PROGRAMS)
    count = M // PROGRAMS + (pid < M % PROGRAMS).to(tl.int32)
    args = (A, B, W, SHIFT, OUT, first, count, tl.full((), N, tl.int32),
            tl.full((), EPS, tl.float32), tl.full((), AFFINE, tl.int32))
    if A.dtype.element_ty == tl.bfloat16 and not AFFINE and N == 2048:
        tle.raw.call(kl3_packed_bf16_norm_2048, (A, B, OUT, first, count, tl.full((), EPS, tl.float32)))
    elif A.dtype.element_ty == tl.bfloat16:
        tle.raw.call(kl3_packed_bf16_norm, args)
    else:
        tle.raw.call(kl3_packed_fp16_norm, args)
