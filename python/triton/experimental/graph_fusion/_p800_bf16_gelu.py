"""KL3 SIMD exact GELU with the eager BF16 Add boundary."""
import triton
import triton.language as tl
import triton.experimental.tle as tle

PROGRAMS = 12


def tile_width(m, n):
    if m <= 0 or not 0 < n <= 2048 or m*n > 0x7fffffff:
        return None
    return 128 if m*n <= 128*1024 else 256 if m*n <= 1024*1024 else 512



@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_bf16_bias_gelu(a, bias, out, pid, programs, total, n, width):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_bf16_bias_gelu_final(a, bias, out, pid, programs, total, n, width):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_fp16_bias_gelu_final(a, bias, out, pid, programs, total, n, width):
    ...


@triton.jit(do_not_specialize=['A', 'B', 'OUT'], do_not_specialize_on_alignment=['A', 'B', 'OUT'])
def _vector_bias_gelu(A, B, OUT, N: tl.constexpr, M: tl.constexpr, WIDTH: tl.constexpr, PROGS: tl.constexpr, FINAL_ONLY: tl.constexpr = False):
    args = (A, B, OUT, tl.program_id(0), tl.full((), PROGS, tl.int32),
            tl.full((), M * N, tl.int32), tl.full((), N, tl.int32), tl.full((), WIDTH, tl.int32))
    if A.dtype.element_ty == tl.float16:
        tle.raw.call(kl3_fp16_bias_gelu_final, args)
    elif FINAL_ONLY:
        tle.raw.call(kl3_bf16_bias_gelu_final, args)
    else:
        tle.raw.call(kl3_bf16_bias_gelu, args)


def prepare(spec, width, proof):
    import torch
    from ._p800_bound_launch import bind
    from ._resource_codegen import bind_evidence

    m, n = spec['m'], spec['n']
    proof.update(emitted='fp16_vector_gelu' if spec['dtype'] == 'float16' else 'bf16_vector_gelu', tiles=dict(width=width, programs=PROGRAMS), autotune='fixed')
    bound = None

    def run(xs):
        nonlocal bound
        if bound is not None:
            out = bound(*xs)
            if out is not None:
                return (out,)
        a, b = xs
        out = torch.empty_like(a)
        binary = _vector_bias_gelu[(PROGRAMS,)](a, b, out, n, m, width, PROGRAMS, spec.get('final_round_only', False), num_warps=4, num_stages=1)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
            bound = bind(_vector_bias_gelu, binary, (PROGRAMS, 1, 1), xs, (m, n), (), 'cluster')
            proof['prepared_launch'] = bound is not None
        return (out,)

    return run, proof

@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_finish_bf16(src, out, pid, total, width, act):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_finish_fp16(src, out, pid, total, width, act):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipeline_cast_bf16(src, out, pid):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipeline_silu_fp16(src, out, pid):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipeline_relu_fp16(src, out, pid):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipeline_relu_bf16(src, out, pid):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipe_cast_bf(src, out, pid, total):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipe_relu_bf(src, out, pid, total):
    ...


@tle.raw.dialect('xpu3', file='_p800_bf16_gelu.xpu')
def kl3_pipe_relu_half(src, out, pid, total):
    ...


@triton.jit(do_not_specialize=['src', 'out'], do_not_specialize_on_alignment=['src', 'out'])
def vector_finish(src, out, TOTAL: tl.constexpr, WIDTH: tl.constexpr, ACT: tl.constexpr):
    args = (src, out, tl.program_id(0), tl.full((), TOTAL, tl.int32),
            tl.full((), WIDTH, tl.int32), tl.full((), ACT, tl.int32))
    if TOTAL == 2048 * 2048 and ACT == 0 and out.dtype.element_ty == tl.bfloat16:
        tle.raw.call(kl3_pipeline_cast_bf16, (src, out, tl.program_id(0)))
    elif TOTAL == 2048 * 2048 and ACT == 1 and out.dtype.element_ty == tl.float16:
        tle.raw.call(kl3_pipeline_silu_fp16, (src, out, tl.program_id(0)))
    elif TOTAL == 2048 * 2048 and ACT == 2:
        if out.dtype.element_ty == tl.bfloat16:
            tle.raw.call(kl3_pipeline_relu_bf16, (src, out, tl.program_id(0)))
        else:
            tle.raw.call(kl3_pipeline_relu_fp16, (src, out, tl.program_id(0)))
    elif TOTAL >= 2 * 12 * 64 * 512 and TOTAL % 512 == 0 and (ACT == 2 or (ACT == 0 and out.dtype.element_ty == tl.bfloat16)):
        args_pipe = (src, out, tl.program_id(0), tl.full((), TOTAL, tl.int32))
        if ACT == 0:
            tle.raw.call(kl3_pipe_cast_bf, args_pipe)
        elif out.dtype.element_ty == tl.bfloat16:
            tle.raw.call(kl3_pipe_relu_bf, args_pipe)
        else:
            tle.raw.call(kl3_pipe_relu_half, args_pipe)
    elif out.dtype.element_ty == tl.bfloat16:
        tle.raw.call(kl3_finish_bf16, args)
    else:
        tle.raw.call(kl3_finish_fp16, args)
