"""Mixed-input P800 SDNN MatMul + Bias, with an FP32 result."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['a', 'b', 'bias', 'out'],
            do_not_specialize_on_alignment=['a', 'b', 'bias', 'out'])
def mm_bias(a, b, bias, out, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            ACT: tl.constexpr = 0, FINAL_ONLY: tl.constexpr = False, K: tl.constexpr = 64):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, 64)
    if K == 64 and M % BM == 0 and N % BN == 0:
        lhs = tl.load(a + rm[:, None] * 64 + rk[None, :])
        rhs = tl.load(b + rk[:, None] * N + rn[None, :])
        product = tl.dot(lhs, rhs, input_precision='ieee', out_dtype=tl.float32)
        offset = tl.load(bias + rn)
    else:
        product = tl.zeros((BM, BN), tl.float32)
        for step in range(tl.cdiv(K, 64)):
            col = step * 64 + rk
            lhs = tl.load(a + rm[:, None] * K + col[None, :],
                          mask=(rm[:, None] < M) & (col[None, :] < K), other=0.)
            rhs = tl.load(b + col[:, None] * N + rn[None, :],
                          mask=(col[:, None] < K) & (rn[None, :] < N), other=0.)
            product = tl.dot(lhs, rhs, product, input_precision='ieee', out_dtype=tl.float32)
        offset = tl.load(bias + rn, mask=rn < N, other=0.)
    value = product + offset[None, :].to(tl.float32)
    if ACT == 1:
        x = value
        raw = value.to(tl.uint32, bitcast=True)
        e = tl.exp(-tl.abs(value))
        numerator = tl.where(value < 0., value * e, value)
        # SDK FP32 SiLU returns x exactly when exp(-x) can no longer
        # change 1.0. Avoid the SDNN approximate division at output ties.
        value = tl.where(x >= 32., x, numerator / (1.0 + e))
        value = tl.where(raw == 0x7f800000, float('inf'),
                         tl.where((raw == 0xff800000) | ((raw & 0x7fffffff) > 0x7f800000), float('nan'), value))
    elif ACT == 2:
        value = tl.where(value != value, value, tl.maximum(value, 0.0))
    elif ACT == 3:
        arg = .7978845608028654 * (value + .044715 * value * value * value)
        g = tl.sigmoid(tl.maximum(-20., tl.minimum(20., 2.0 * arg)))
        value = value * tl.where(arg < -10., 0., g)
    elif ACT == 4:
        value = .5 * value * (1.0 + tl.math.erf(.7071067811865476 * value))
    index = rm[:, None] * N + rn[None, :]
    # The SDNN precision pass handles the final FP32-to-BF16 truncation.
    # Keeping the original output pointer type also preserves descriptor layout.
    if M % BM == 0 and N % BN == 0:
        tl.store(out + index, value)
    else:
        tl.store(out + index, value, mask=(rm[:, None] < M) & (rn[None, :] < N))



def prepare(spec, *, final_only=False):
    import torch
    from ._p800_bound_launch import bind
    from ._resource_codegen import bind_evidence

    m, n, k = spec['M'], spec['N'], spec['K']
    bm, bn, warps = {(64, 1024): (64, 128, 4), (512, 1024): (128, 128, 8),
                     (2048, 2048): (64, 512, 8)}.get((m, n), (64, 128, 4))
    if final_only and spec['act'] in (1, 3, 4):
        bm, bn, warps = 64, 64, 4
    grouped = (m, n, k) == (2048, 2048, 64) and (not final_only or spec['act'] in (0, 2))
    # Reuse RHS panels on wide matrices with a partial final group. Mask
    # its rows on load and store while keeping the column and K tiles dense.
    partial_group = (m >= 1024 and m % 1024 != 0 and n >= 512 and n % 256 == 0 and k == 64
                     and (not final_only or spec['act'] in (0, 1, 2)))
    grouped = grouped or partial_group
    kernel = mm_bias
    if grouped:
        from ._p800_native_mm import _aligned_bias_kernel
        kernel = _aligned_bias_kernel()
        bm, bn, warps = 128, 256, 4
    grid = ((triton.cdiv(m, 8 * bm) * 8 * (n // bn), 1, 1) if grouped else
            (triton.cdiv(m, bm), triton.cdiv(n, bn), 1))
    proof = {'route': 'fp32_compute', 'resource': 'sdnn', 'resource_verified': False,
             'tiles': dict(bm=bm, bn=bn, k=k, bk=64, warps=warps)}
    bound = None

    def run(xs):
        nonlocal bound
        if bound is not None:
            out = bound(*xs)
            if out is not None:
                return (out,)
        a, b, bias = xs
        dtype = a.dtype if final_only else torch.float32
        out = torch.empty((m, n), device=a.device, dtype=dtype)
        from contextlib import nullcontext
        from ._p800_compiler import resource_scope
        scope = resource_scope('sdnn') if final_only else nullcontext()
        with scope:
            if grouped:
                binary = kernel[grid](a, b, bias, out, n, 64, bm, bn, RELU=final_only and spec['act'] == 2,
                                      M=m if partial_group else 0, SILU=partial_group and final_only and spec['act'] == 1,
                                      num_warps=warps, num_stages=3, is_sdnn=True)
            else:
                binary = kernel[grid](a, b, bias, out, m, n, bm, bn, ACT=spec['act'] if final_only else 0,
                                      FINAL_ONLY=final_only, K=k, num_warps=warps, num_stages=1, is_sdnn=True)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
            bound = bind(kernel, binary, grid, xs, (m, n), (), 'sdnn', output_dtype=dtype)
            proof['prepared_launch'] = bound is not None
        return (out,)
    return run, proof


@triton.jit(do_not_specialize=['src', 'out'], do_not_specialize_on_alignment=['src', 'out'])
def finish(src, out, ACT: tl.constexpr, BLOCK: tl.constexpr, TOTAL: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < TOTAL
    value = tl.load(src + idx, mask=valid, other=0.)
    if ACT == 1:
        bits = value.to(tl.uint32, bitcast=True)
        y = value / (1.0 + tl.exp(-value))
        value = tl.where(bits == 0x7f800000, float('inf'),
                         tl.where((bits == 0xff800000) | ((bits & 0x7fffffff) > 0x7f800000), float('nan'), y))
    elif ACT == 2:
        value = tl.where(value != value, value, tl.maximum(value, 0.0))
    elif ACT == 3:
        arg = .7978845608028654 * (value + .044715 * value * value * value)
        value = value * tl.sigmoid(2.0 * arg)
    elif ACT == 4:
        value = .5 * value * (1.0 + tl.math.erf(.7071067811865476 * value))
    if out.dtype.element_ty == tl.bfloat16:
        word = value.to(tl.uint32,bitcast=True)
        hi = (word + 0x7fff + ((word >> 16) & 1)) >> 16
        tl.store(out.to(tl.pointer_type(tl.uint16))+idx, hi.to(tl.uint16), mask=valid)
    else:
        tl.store(out+idx,value,mask=valid)


def prepare_sdk_store(spec):
    """Keep the grouped product in FP32 and let the SDK narrow its contiguous result."""
    product, first = prepare(spec)
    proof = {'route': 'grouped_mm_sdk_store', 'regions': [first], 'external_calls': 1,
             'external_op': 'aten._to_copy', 'resource_verified': False}

    def run(inputs):
        value, = product(inputs)
        return (value.to(inputs[0].dtype),)

    return run, proof


def prepare_split(spec):
    import torch
    from ._p800_bound_launch import bind
    from ._resource_codegen import bind_evidence
    product, first = prepare(spec)
    epilogue = {'resource':'cluster','resource_verified':False}
    proof = {'route':'mixed_mm_cluster_epilogue','resource_verified':False,'kernel_count':2}
    m,n = spec['M'],spec['N']
    block = 8192 if m * n >= 512 * 1024 else 2048
    grid = (triton.cdiv(m*n, block),1,1)
    bound = None

    def run(inputs):
        nonlocal bound
        z, = product(inputs)
        if bound is not None:
            y = bound(z)
            if y is not None:
                return (y,)
        y = torch.empty((m,n),device=z.device,dtype=inputs[0].dtype)
        from ._p800_layernorm import toolchain_available
        from ._p800_bf16_gelu import tile_width, vector_finish
        width = tile_width(m, n)
        if width is not None and (m*n) % width == 0 and spec['act'] in (0, 1, 2, 4) and toolchain_available():
            kernel, launch_grid = vector_finish, (12, 1, 1)
            binary = kernel[launch_grid](z, y, m*n, width, spec['act'], num_warps=4, num_stages=1)
        else:
            kernel, launch_grid = finish, grid
            binary = kernel[launch_grid](z, y, spec['act'], block, m*n, num_warps=4, num_stages=1)
        if not proof['resource_verified']:
            bind_evidence(epilogue,binary)
            proof.update(resource_verified=first['resource_verified'] and epilogue['resource_verified'],
                         kernel_evidence=first['kernel_evidence']+epilogue['kernel_evidence'])
            bound = bind(kernel,binary,launch_grid,(z,), (m,n),(), 'cluster',output_dtype=y.dtype)
        return (y,)
    return run, proof
