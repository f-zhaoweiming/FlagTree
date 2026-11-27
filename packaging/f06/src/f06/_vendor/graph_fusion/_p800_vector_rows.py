"""Dense KL3 broadcast activation, product reduction and softmax kernels."""
import triton
import triton.language as tl
import triton.experimental.tle as tle

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_bf_activation_rows(a, bias, out, first, rows, n, mode):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_point_bf(a, b, out, pid, total, n, width, mode):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_point_half(a, b, out, pid, total, n, width, mode):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_point_float(a, b, out, pid, total, n, width, mode):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_mixed_act_half(a, bias, out, pid, total, n, width, mode):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_mixed_act_bf(a, bias, out, pid, total, n, width, mode):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_silu_half_2048(a, bias, out, pid):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_silu_half_rows(a, bias, out, pid, total):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_silu_pipeline(a, bias, out, pid, n, total, width, grid):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_silu_balanced(a, bias, out, pid):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_sum_bf(a, b, out, first, rows, n):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_sum_half(a, b, out, first, rows, n):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_sum_float(a, b, out, first, rows, n):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_soft_bf(a, b, out, first, rows, n, lanes):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_soft_half(a, b, out, first, rows, n, lanes):
    ...

@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_sum_cross(a, bias, out, first, rows):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_sum_wide_bf(a, b, out, first, rows, n):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_sum_wide_half(a, b, out, first, rows, n):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_soft_wide_bf(a, b, out, first, rows, n, lanes):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_soft_wide_half(a, b, out, first, rows, n, lanes):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_soft_wide_float(a, b, out, first, rows, n, lanes):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_short_soft_float(a, b, out, pid, grid, rows, cols, final_only):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_short_soft_half(a, b, out, pid, grid, rows, cols, final_only):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_short_soft_bf(a, b, out, pid, grid, rows, cols, final_only):
    ...












@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_shard_sum_float(a, b, out, first, rows, n):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_shard_sum_bf(a, b, out, first, rows, n):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_shard_sum_half(a, b, out, first, rows, n):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_stream_soft_float(a, b, out, pid, grid, rows, cols, final_only):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_stream_soft_half(a, b, out, pid, grid, rows, cols, final_only):
    ...


@tle.raw.dialect('xpu3', file='_p800_vector_rows.xpu')
def kl3_stream_soft_bf(a, b, out, pid, grid, rows, cols, final_only):
    ...


@triton.jit(do_not_specialize=['A', 'B', 'OUT'], do_not_specialize_on_alignment=['A', 'B', 'OUT'])
def _stream_softmax(A, B, OUT, M: tl.constexpr, N: tl.constexpr, GRID: tl.constexpr, FINAL_ONLY: tl.constexpr):
    if N <= 1024 and N % 32 and M <= 0x7fffffff:
        short = (A, B, OUT, tl.program_id(0), tl.full((), GRID, tl.int32),
                 tl.full((), M, tl.int32), tl.full((), N, tl.int32), tl.full((), FINAL_ONLY, tl.int32))
        if A.dtype.element_ty == tl.float32:
            tle.raw.call(kl3_short_soft_float, short)
        elif A.dtype.element_ty == tl.float16:
            tle.raw.call(kl3_short_soft_half, short)
        else:
            tle.raw.call(kl3_short_soft_bf, short)
    else:
        args = (A, B, OUT, tl.program_id(0), tl.full((), GRID, tl.int32), tl.full((), M, tl.int64),
                tl.full((), N, tl.int64), tl.full((), FINAL_ONLY, tl.int32))
        if A.dtype.element_ty == tl.float32:
            tle.raw.call(kl3_stream_soft_float, args)
        elif A.dtype.element_ty == tl.float16:
            tle.raw.call(kl3_stream_soft_half, args)
        else:
            tle.raw.call(kl3_stream_soft_bf, args)



@triton.jit(do_not_specialize=['A', 'B', 'OUT'], do_not_specialize_on_alignment=['A', 'B', 'OUT'])
def _vector_rows(A, B, OUT, M: tl.constexpr, N: tl.constexpr, KIND: tl.constexpr, TILE: tl.constexpr,
                 ROWS: tl.constexpr, WIDE_SUM: tl.constexpr = False, FINAL_ACT: tl.constexpr = False):
    pid = tl.program_id(0)
    n = tl.full((), N, tl.int32)
    if KIND == 10 or KIND == 11:
        first = pid * (M // 12) + tl.minimum(pid, M % 12)
        count = M // 12 + (pid < M % 12).to(tl.int32)
        tle.raw.call(kl3_bf_activation_rows, (A, B, OUT, first, count, n, tl.full((), KIND - 8, tl.int32)))
    elif FINAL_ACT and KIND == 2 and M == 2048 and N == 2048 and TILE == 512 and A.dtype.element_ty == tl.float16:
        tle.raw.call(kl3_silu_half_2048, (A, B, OUT, pid))
    elif FINAL_ACT and KIND == 2 and N == 2048 and TILE == 512 and M*N >= 2*12*64*512 and A.dtype.element_ty == tl.float16:
        tle.raw.call(kl3_silu_half_rows, (A, B, OUT, pid, tl.full((), M*N, tl.int32)))
    elif FINAL_ACT:
        args = (A, B, OUT, pid, tl.full((), M*N, tl.int32), n, tl.full((), TILE, tl.int32),
                tl.full((), KIND, tl.int32))
        if A.dtype.element_ty == tl.bfloat16:
            tle.raw.call(kl3_mixed_act_bf, args)
        else:
            tle.raw.call(kl3_mixed_act_half, args)
    elif KIND == 8:
        tle.raw.call(kl3_silu_balanced, (A, B, OUT, pid))
    elif KIND == 7:
        tle.raw.call(kl3_silu_pipeline, (A, B, OUT, pid, n, tl.full((), M * N, tl.int32),
                                         tl.full((), TILE, tl.int32), tl.full((), 12, tl.int32)))
    elif KIND <= 3:
        args = (A, B, OUT, pid, tl.full((), M * N, tl.int32), n, tl.full((), TILE, tl.int32),
                tl.full((), KIND, tl.int32))
        if A.dtype.element_ty == tl.bfloat16:
            tle.raw.call(kl3_point_bf, args)
        elif A.dtype.element_ty == tl.float16:
            tle.raw.call(kl3_point_half, args)
        else:
            tle.raw.call(kl3_point_float, args)
    else:
        first = pid * ROWS
        count = tl.minimum(ROWS, M - first)
        if KIND == 9:
            first = pid * (M // 12) + tl.minimum(pid, M % 12)
            count = M // 12 + (pid < M % 12).to(tl.int32)
            args = (A, B, OUT, first, count, n)
            if A.dtype.element_ty == tl.float32:
                tle.raw.call(kl3_shard_sum_float, args)
            elif A.dtype.element_ty == tl.bfloat16:
                tle.raw.call(kl3_shard_sum_bf, args)
            else:
                tle.raw.call(kl3_shard_sum_half, args)
        elif KIND == 4:
            if WIDE_SUM:
                if N == 1024:
                    first = pid * (M // 12) + tl.minimum(pid, M % 12)
                    count = M // 12 + (pid < M % 12).to(tl.int32)
                if A.dtype.element_ty == tl.bfloat16:
                    tle.raw.call(kl3_sum_wide_bf, (A, B, OUT, first, count, n))
                else:
                    tle.raw.call(kl3_sum_wide_half, (A, B, OUT, first, count, n))
            elif A.dtype.element_ty == tl.float32 and N == 1024:
                first = pid * (M // 12) + tl.minimum(pid, M % 12)
                count = M // 12 + (pid < M % 12).to(tl.int32)
                tle.raw.call(kl3_sum_cross, (A, B, OUT, first, count))
            elif A.dtype.element_ty == tl.bfloat16:
                tle.raw.call(kl3_sum_bf, (A, B, OUT, first, count, n))
            elif A.dtype.element_ty == tl.float16:
                tle.raw.call(kl3_sum_half, (A, B, OUT, first, count, n))
            else:
                tle.raw.call(kl3_sum_float, (A, B, OUT, first, count, n))
        elif KIND == 6:
            args = (A, B, OUT, first, count, n, tl.full((), TILE, tl.int32))
            if A.dtype.element_ty == tl.bfloat16:
                tle.raw.call(kl3_soft_wide_bf, args)
            elif A.dtype.element_ty == tl.float16:
                tle.raw.call(kl3_soft_wide_half, args)
            else:
                tle.raw.call(kl3_soft_wide_float, args)
        else:
            args = (A, B, OUT, first, count, n, tl.full((), TILE, tl.int32))
            if A.dtype.element_ty == tl.bfloat16:
                tle.raw.call(kl3_soft_bf, args)
            else:
                tle.raw.call(kl3_soft_half, args)


_SHAPES = {(64, 1024): (128, 16), (512, 1024): (256, 48), (2048, 2048): (512, 176)}
_KINDS = {'bias_add': 0, 'relu': 1, 'silu': 2, 'gelu_tanh': 3, 'mul_sum': 4, 'add_softmax': 5}


def wide_softmax_configuration(m, n):
    # Each core holds at most 1024 FP32 logits, with 32 lanes per iteration.
    # Two cores can cooperate on equal, whole-vector halves of a wider row.
    parts = 1 if n <= 1024 else 2
    if m <= 0 or n <= 0 or n > 2048 or n % (32 * parts) or m * n > 0x7fffffff:
        return None
    from ._p800_layernorm import toolchain_available
    if not toolchain_available():
        return None
    # Spread short batches over more cores while retaining whole vectors.
    if m * 8 <= 12 * 64:
        while parts < 8 and n % (64 * parts) == 0:
            parts *= 2
    rows = (m + 11) // 12
    return 6, n // parts, rows, (m + rows - 1) // rows


def configuration(kind, dtype, m, n):
    if kind not in _KINDS:
        return None
    if (m, n) not in _SHAPES:
        # The double-buffered kernel needs a second wave to prefetch. Short
        # batches retain the ordinary flat candidate and its measured tile.
        wave = 12 * 64 * 512
        if (kind == 'silu' and dtype == 'float32' and n >= 512 and n % 512 == 0
                and wave % n == 0 and 2 * wave <= m * n <= 0x7fffffff):
            from ._p800_layernorm import toolchain_available
            return (7, 512, 1, 12) if toolchain_available() else None
        if kind == 'mul_sum' and dtype == 'float32' and 0 < m <= 12 * 64 and 512 < n <= 2048 and n != 1024:
            from ._p800_layernorm import toolchain_available
            return (9, 512, (m + 11) // 12, 12) if toolchain_available() else None
        if kind != 'mul_sum' or not 0 < n <= 2048 or n % 64 or m <= 0 or m*n > 0x7fffffff:
            return None
        # The 1024-column cross-core sum has 64 scratch slots per cluster.
        if n == 1024 and m > 12 * 64:
            return None
        from ._p800_layernorm import toolchain_available
        if not toolchain_available():
            return None
        rows = min(m, 64)
        grid = 12 if n == 1024 and dtype == 'float32' else (m + rows - 1) // rows
        return _KINDS[kind], 64, rows, grid
    # Default routing stays scoped to the measured layouts and vendor algorithms.
    if kind == 'mul_sum':
        supported = dtype in ('bfloat16', 'float16', 'float32')
    elif kind == 'add_softmax':
        supported = dtype == 'bfloat16' or (dtype == 'float16' and m == 2048)
    elif kind == 'gelu_tanh':
        supported = dtype in ('bfloat16', 'float16')
    elif kind == 'silu':
        supported = dtype == 'bfloat16' or (dtype == 'float32' and m in (512, 2048))
    else:
        supported = dtype == 'bfloat16'
    if not supported:
        return None
    from ._p800_layernorm import toolchain_available
    if not toolchain_available():
        return None
    if kind == 'silu' and dtype == 'float32' and (m, n) == (2048, 2048):
        return 7, 512, 1, 12
    if kind == 'silu' and dtype == 'float32' and (m, n) == (512, 1024):
        return 8, 704, 1, 12
    width, rows = _SHAPES[m, n]
    mode = _KINDS[kind]
    grid = 12 if mode < 4 or (mode == 4 and dtype == 'float32' and n == 1024) else (m + rows - 1) // rows
    return mode, width, rows, grid


def prepare(spec, config, proof):
    import torch
    from ._p800_bound_launch import bind
    from ._resource_codegen import bind_evidence

    mode, width, rows, grid = config
    m, n = spec['m'], spec['n']
    shape = ((m, 1) if spec.get('keepdim') else (m,)) if mode in (4, 9) else (m, n)
    tiles = dict(programs=grid)
    if mode == 8:
        tiles.update(elements_per_core=(672, 704))
    elif mode < 4 or mode == 7:
        tiles['width'] = width
    elif mode in (10, 11):
        tiles.update(rows=rows, columns=n)
    elif mode == 9:
        tiles.update(rows=(m // 12, rows), partial_elements=512)
    elif mode == 4:
        wide_sum = spec['dtype'] == 'float32' or spec.get('final_round_only', False)
        tiles.update(rows=rows, partial_elements=512 if wide_sum else 1024)
        if n == 1024 and wide_sum:
            tiles['rows'] = (m // 12, (m + 11) // 12)
    else:
        parts = n // width if mode == 6 else (1 if n <= 1024 else 2)
        tiles.update(rows=rows, columns_per_core=n // parts, cores_per_row=parts)
    proof.update(emitted='kl3_vector_rows', autotune='fixed', tiles=tiles)
    bound = None

    def run(xs):
        nonlocal bound
        if bound is not None:
            value = bound(*xs)
            if value is not None:
                return (value,)
        a, b = xs
        out = torch.empty(shape, device=a.device, dtype=a.dtype)
        binary = _vector_rows[(grid,)](a, b, out, m, n, mode, width, rows,
                                        WIDE_SUM=spec.get('final_round_only', False) and mode == 4,
                                        FINAL_ACT=spec.get('final_round_only', False) and mode in (2, 3),
                                        num_warps=4, num_stages=1)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
            bound = bind(_vector_rows, binary, (grid, 1, 1), xs, shape, (), 'cluster')
            proof['prepared_launch'] = bound is not None
        return (out,)

    return run, proof
