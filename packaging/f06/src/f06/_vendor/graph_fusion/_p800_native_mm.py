"""P800 native MatMul + bias, with fused or split activation epilogues."""

import os

_TILES = {
    # M upper bound -> (BM, BN, BK, num_warps); measured on P800 (idle card)
    512: (256, 256, 64, 8),
}
_DEFAULT_TILE = (64, 512, 64, 8)
_DTYPES = ('float32', 'float16')


def _tile(M, N):
    for limit in sorted(_TILES):
        if M <= limit and N >= _TILES[limit][1]:
            return _TILES[limit]
    return _DEFAULT_TILE


def match(gm, inputs, *, vendor_product=False):
    """Return a spec for (a @ b) + bias [-> relu] else None."""
    import torch
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    if len(inputs) != 3 or len(tensors) != 3:
        return None
    a, b, bias = tensors[0], tensors[1], tensors[2]
    if a.dim() != 2 or b.dim() != 2 or bias.dim() != 1:
        return None
    if int(b.shape[0]) != int(a.shape[1]) or int(b.shape[1]) != int(bias.shape[0]):
        return None
    dtypes = ('bfloat16', 'float16') if vendor_product else _DTYPES
    if str(a.dtype).replace('torch.', '') not in dtypes or b.dtype != a.dtype or bias.dtype != a.dtype:
        return None
    for t in (a, b, bias):
        if not t.is_contiguous() or int(t.stride(-1)) != 1:
            return None
    M, K, N = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])

    nodes = [n for n in gm.graph.nodes if n.op in ('call_function', 'call_method')]
    mm = add = act = None
    for node in nodes:
        name = getattr(getattr(node, 'target', None), '__name__', '')
        if mm is None and (name in ('mm', 'matmul') or node.target in (
                torch.mm, torch.matmul, torch.ops.aten.mm.default, torch.ops.aten.matmul.default)):
            mm = node
        elif mm is not None and add is None and (name in ('add', 'add_') or node.target in (
                torch.add, torch.ops.aten.add.Tensor)):
            add = node
        elif name in ('relu', 'relu_', 'clamp', 'clamp_min') or node.target in (
                torch.relu, torch.ops.aten.relu.default, torch.ops.aten.clamp_min.default):
            act = node
        elif name in ('silu', 'silu_') or node.target in (
                torch.nn.functional.silu, torch.ops.aten.silu.default):
            act = node
        elif name == 'gelu' or node.target is torch.ops.aten.gelu.default:
            act = node
    if mm is None or add is None:
        return None
    placeholders = tuple(node for node in gm.graph.nodes if node.op == 'placeholder')
    if len(placeholders) != 3 or tuple(mm.args) != placeholders[:2]:
        return None
    if (len(add.args) != 2 or set(add.args) != {mm, placeholders[2]} or
            add.kwargs.get('alpha', 1) != 1 or 'out' in add.kwargs):
        return None
    # Activation must directly consume the add.
    if act is not None:
        if act.args[0] is not add:
            return None
        if act.target in (torch.ops.aten.clamp_min.default,) or getattr(act.target, '__name__', '') in ('clamp', 'clamp_min'):
            min_v = act.kwargs.get('min', act.args[1] if len(act.args) > 1 else None)
            if min_v not in (0, 0.0) or len(act.args) > 2 or act.kwargs.get('max') is not None:
                return None
    act_name = getattr(getattr(act, 'target', None), '__name__', '') if act is not None else ''
    if act is None:
        act_code = 0
    elif act_name in ('silu', 'silu_') or act.target in (torch.nn.functional.silu,
                                                         torch.ops.aten.silu.default):
        act_code = 1
    elif act_name == 'gelu' or act.target is torch.ops.aten.gelu.default:
        # GELU is not admitted by the epilogue (`uni_sram`/dominate compile
        # failures on the wide matmul tiles); it runs as a second pointwise
        # kernel over the matmul+bias output. tanh uses the sigmoid identity
        # (vector vsigmoidf), exact uses scalar erf.
        act_code = 3 if str(act.kwargs.get('approximate', 'none')) == 'tanh' else 4
    else:
        act_code = 2
    if vendor_product:
        from ._p800_vector_rows import configuration
        from ._p800_layernorm import toolchain_available
        dtype = str(a.dtype).removeprefix('torch.')
        kind = ('bias_add', 'silu', 'relu', 'gelu_tanh', 'gelu_exact')[act_code]
        supported = (dtype == 'bfloat16' and act_code in (1, 2) and 0 < M * N <= 65536)
        if K == 64 and (M, N) in ((64, 1024), (512, 1024), (2048, 2048)):
            supported |= (act_code == 4 and dtype == 'bfloat16' and toolchain_available()
                          or (dtype == 'bfloat16' or act_code == 3) and configuration(kind, dtype, M, N) is not None)
        if not supported or os.environ.get('FLAGTREE_FLAT_ADD', '1') != '1':
            return None
    # GELU split is fp32-only: for fp16 the matmul kernel's accumulation order
    # differs from eager by up to 3.1e-2 (4 fp16 ulp, both within fp16 error vs
    # fp32 truth), which trips the harness's 3e-3 fp16 validity gate. fp16 GELU
    # keeps the SDNN/Inductor/eager routes.
    if not vendor_product and act_code in (3, 4) and str(a.dtype) != 'torch.float32':
        return None
    # Use the single-kernel epilogue by default; split routes require full EW tiles.
    silu_path = None
    if act_code == 1 and not vendor_product:
        fused_off = os.environ.get('FLAGTREE_NATIVE_SILU_FUSED', '1') == '0'
        forced = os.environ.get('FLAGTREE_NATIVE_SILU', '0') == '1'
        if forced:
            silu_path = 'sdnn'
        elif not fused_off:
            # Fused epilogue (vector sigmoid on the accumulator). Measured
            # 26.7/25.4/42.0 us at 64/512/2048 rows vs 39.3 us for the split
            # fp32 path and ~1.00x eager before.
            silu_path = 'epilogue'
        else:
            silu_path = 'cluster'
    if min(M, N, K) <= 0 or max(M*N, M*K, K*N) > 0x7fffffff:
        return None
    # Only admit graphs whose whole computation is mm -> add -> (relu).
    # Anything consuming the add (gelu/silu/cast/extra arithmetic) is declined:
    # the native kernel computes exactly matmul+bias(+relu).
    allowed = {mm, add} | ({act} if act is not None else set())
    if any(node not in allowed for node in nodes):
        return None
    for user in add.users:
        if user is act or user.op == 'output':
            continue
        return None
    outputs = [node for node in gm.graph.nodes if node.op == 'output']
    if len(outputs) != 1:
        return None
    output = outputs[0].args[0]
    if isinstance(output, (tuple, list)) and len(output) == 1:
        output = output[0]
    if output is not (act if act is not None else add):
        return None
    return dict(kind='native_mm', act=act_code, silu_path=silu_path,
                M=M, K=K, N=N, dtype=str(a.dtype).replace('torch.', ''),
                even_k=(K % 64 == 0))


def compile_vendor_linear(gm, inputs, candidate):
    import torch

    spec = match(gm, inputs, vendor_product=True)
    if spec is None or spec['act'] != 0 or candidate.single_kernel:
        raise ValueError('unsupported vendor linear graph')
    op = torch.ops.custom_ops.linear.default
    shape = (spec['M'], spec['N'])
    result = next(n for n in gm.graph.nodes if n.op == 'output').args[0]

    def run(a, b, bias):
        out = a.new_empty(shape)
        op(a, b, bias, False, False, 1., 0., 0, False, out=out)
        return (out,) if isinstance(result, tuple) else [out] if isinstance(result, list) else out

    # One SDK invocation can dispatch multiple hardware programs.
    return run, dict(route=candidate.name, vendor_op='custom_ops::linear', regions=[],
                     external_calls=1, resource_verified=False, kernel_count=None)


def compile_vendor_product(gm, inputs, candidate):
    """Keep the vendor BF16 MatMul node and its rounding boundary in the FX graph."""
    import copy
    import torch.fx
    from . import _p800_flat
    from ._resource_fx import build_region

    spec = match(gm, inputs, vendor_product=True)
    if spec is None or candidate.single_kernel:
        raise ValueError('unsupported vendor MatMul epilogue')
    output = inputs[0].new_empty((spec['M'], spec['N']))
    args = (output, inputs[2])
    region = build_region(gm, 1, 3 if spec['act'] else 2)
    if tuple(region.node_ids) != candidate.regions[1].node_ids:
        raise ValueError('MatMul epilogue nodes changed')
    if spec['act']:
        flat_spec = _p800_flat.match(region.gm, args)
    else:
        # The whole-graph matcher already checked both Add operands and alpha.
        flat_spec = dict(kind='bias_add', m=spec['M'], n=spec['N'], total=output.numel(), dtype=spec['dtype'])
    if flat_spec is None:
        raise ValueError('unsupported cluster bias epilogue')
    consumer, proof = _p800_flat.compile(flat_spec, args)
    proof['node_ids'] = list(region.node_ids)

    def apply_bias_activation(product, bias):
        return consumer((product, bias))[0]

    graph = copy.deepcopy(gm.graph)
    mm, add, *activation = [n for n in graph.nodes if n.op in ('call_method', 'call_function')]
    act = activation[0] if activation else add
    bias = next(n for n in add.args if n is not mm)
    act.op, act.target = 'call_function', apply_bias_activation
    act.args, act.kwargs = (mm, bias), {}
    if activation:
        graph.erase_node(add)
    graph.lint()
    fn = torch.fx.GraphModule(gm, graph).forward
    producer = dict(route='eager_mm', resource='eager', external_calls=1, resource_verified=False,
                    node_ids=[mm.name], kernel_evidence=[])
    return fn, dict(route=candidate.name, regions=[producer, proof], kernel_count=None)


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'bias_ptr', 'o_ptr'])
    def mm_act(a_ptr, b_ptr, bias_ptr, o_ptr, M, N, K, ACT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               EVEN_K: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, K, BK):
            if EVEN_K:
                av = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :], mask=rm[:, None] < M, other=0.0)
                bv = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.0)
            else:
                # K is not a multiple of BK: without the kk < K mask the tail
                # iteration reads past each row (into the next row / next tensor)
                # and produced 1e3-1e4 magnitude garbage for K=48.
                kk = k + rk
                av = tl.load(a_ptr + rm[:, None] * K + kk[None, :],
                             mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
                bv = tl.load(b_ptr + kk[:, None] * N + rn[None, :],
                             mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(av, bv, input_precision='ieee')
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
        z = (acc.to(a_ptr.dtype.element_ty) + bias[None, :]).to(a_ptr.dtype.element_ty).to(tl.float32)
        if ACT == 2:
            z = tl.where(z != z, z, tl.maximum(z, 0.0))
        elif ACT == 3:
            # sigmoid(40) rounds to 1 in float32. The P800 vector sigmoid
            # mishandles +inf, so cap only its input and keep the original factor.
            z = z * tl.sigmoid(tl.minimum(z, 40.0))
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], z,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    # Standalone 1D SDNN EW activation: the exp form canonicalises to tt.silu,
    # which the native LUT pass lowers to a vector table lookup (45 us/2048^2
    # vs 2290 us for the scalar extern epilogue).
    @triton.jit(do_not_specialize=['x_ptr', 'o_ptr'])
    def ew_silu(x_ptr, o_ptr, N, XB: tl.constexpr, TAIL: tl.constexpr = False):
        i = tl.program_id(0) * XB + tl.arange(0, XB)
        if TAIL:
            z = tl.load(x_ptr + i, mask=i < N, other=0.).to(tl.float32)
        else:
            z = tl.load(x_ptr + i).to(tl.float32)
        # Exact form: canonicalises to tt.silu so the native LUT pass lowers it
        # to a vector table lookup. Tail correction was measured: it does not
        # reduce the mid-range LUT error and slows the kernel down, so it is
        # not applied; f32 stays opt-in (FLAGTREE_NATIVE_SILU=1).
        value = z / (1.0 + tl.exp(-z))
        nan = (z.to(tl.int32, bitcast=True) & 0x7fffffff) > 0x7f800000
        y = tl.where(nan | (z == -float('inf')), float('nan'), tl.where(z == float('inf'), z, value))
        if TAIL:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty), mask=i < N)
        else:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty))

    # Cluster EW activation: FlagTree's own vectorised lowering (VExpFOp), exact
    # for f32. Used when the SDNN LUT window/tolerance is not acceptable.
    @triton.jit(do_not_specialize=['x_ptr', 'o_ptr'])
    def ew_silu_cluster(x_ptr, o_ptr, N, XB: tl.constexpr, TAIL: tl.constexpr = False):
        i = tl.program_id(0) * XB + tl.arange(0, XB)
        if TAIL:
            z = tl.load(x_ptr + i, mask=i < N, other=0.).to(tl.float32)
        else:
            z = tl.load(x_ptr + i).to(tl.float32)
        value = z / (1.0 + tl.exp(-z))
        nan = (z.to(tl.int32, bitcast=True) & 0x7fffffff) > 0x7f800000
        y = tl.where(nan | (z == -float('inf')), float('nan'), tl.where(z == float('inf'), z, value))
        if TAIL:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty), mask=i < N)
        else:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty))

    # Split GELU(tanh): x*sigmoid(2*sqrt(2/pi)*(x+0.044715 x^3)). The sigmoid
    # canonicalises to the vector vsigmoidf intrinsic, 2.5x faster than the
    # scalar tanh form and exact to ~5e-7 in f32.
    @triton.jit(do_not_specialize=['x_ptr', 'o_ptr'])
    def ew_gelu_tanh(x_ptr, o_ptr, N, XB: tl.constexpr, TAIL: tl.constexpr = False):
        i = tl.program_id(0) * XB + tl.arange(0, XB)
        if TAIL:
            z = tl.load(x_ptr + i, mask=i < N, other=0.).to(tl.float32)
        else:
            z = tl.load(x_ptr + i).to(tl.float32)
        a = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        y = z * tl.sigmoid(2.0 * a)
        if TAIL:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty), mask=i < N)
        else:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty))

    @triton.jit(do_not_specialize=['x_ptr', 'o_ptr'])
    def ew_gelu_exact(x_ptr, o_ptr, N, XB: tl.constexpr, TAIL: tl.constexpr = False):
        i = tl.program_id(0) * XB + tl.arange(0, XB)
        if TAIL:
            z = tl.load(x_ptr + i, mask=i < N, other=0.).to(tl.float32)
        else:
            z = tl.load(x_ptr + i).to(tl.float32)
        y = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
        if TAIL:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty), mask=i < N)
        else:
            tl.store(o_ptr + i, y.to(o_ptr.dtype.element_ty))

    return mm_act, ew_silu, ew_silu_cluster, ew_gelu_tanh, ew_gelu_exact


def _aligned_bias_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def grouped_mm_bias(A, B, Bias, Out, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        RELU: tl.constexpr = False, M: tl.constexpr = 0, SILU: tl.constexpr = False):
        # Complete tiles only. Group eight output row tiles before advancing
        # the column, as in the Kunlunxin AddMM and Inductor matrix templates.
        pid = tl.program_id(0)
        group_width = 8 * (N // BN)
        row = (pid // group_width * 8 + pid % 8) * BM + tl.arange(0, BM)
        col = (pid % group_width // 8) * BN + tl.arange(0, BN)
        k = tl.arange(0, K)
        if M:
            a = tl.load(A + row[:, None] * K + k[None, :], mask=row[:, None] < M, other=0.)
        else:
            a = tl.load(A + row[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] * N + col[None, :])
        result = tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision='ieee') + tl.load(Bias + col).to(tl.float32)[None, :]
        if RELU:
            if A.dtype.element_ty == tl.float32 and Out.dtype.element_ty == tl.float32:
                # The ordered comparison leaves NaNs and signed zero on the original-value branch.
                result = tl.where(result < 0., 0., result)
            else:
                result = tl.where(result != result, result, tl.maximum(result, 0.))
        if SILU:
            result = result * tl.sigmoid(tl.minimum(result, 40.))
        if M:
            tl.store(Out + row[:, None] * N + col[None, :], result, mask=row[:, None] < M)
        else:
            tl.store(Out + row[:, None] * N + col[None, :], result)

    return grouped_mm_bias


def supports_grouped_relu(spec):
    return (spec['dtype'] == 'float32' and spec['act'] == 2
            and (spec['M'], spec['N'], spec['K']) == (2048, 2048, 64))


def compile(spec, tensors, *, grouped_relu=False):
    """Return (callable(xs)->tuple, proof)."""
    import torch
    kernel, ew_kernel, ew_cluster, ew_gelu_tanh, ew_gelu_exact = _kernel()
    BM, BN, BK, warps = _tile(spec['M'], spec['N'])
    if not spec.get('even_k', True):
        # The K-tail masks require additional SRAM.
        BM, BN = min(BM, 64), min(BN, 256)
    grid = ((spec['M'] + BM - 1) // BM, (spec['N'] + BN - 1) // BN)
    from ._p800_bound_launch import bind
    from ._resource_codegen import _record_binary

    def record(*binaries):
        if not proof['resource_verified']:
            proof['kernel_evidence'] = [_record_binary(b, 'sdnn' if b.metadata.is_sdnn else 'cluster')
                                        for b in binaries]
            proof['resource_verified'] = True

    warm_launch = None
    if grouped_relu and not supports_grouped_relu(spec):
        raise ValueError('grouped ReLU requires a complete 2048 x 2048 FP32 product with K=64')
    grouped = grouped_relu or (spec['dtype'] == 'float32' and spec['act'] in (0, 3)
                               and (spec['M'], spec['N'], spec['K']) == (2048, 2048, 64))
    if grouped:
        kernel = _aligned_bias_kernel()
        BM, BN, BK, warps = 128, 256, 64, 4
        grid = (spec['M'] * spec['N'] // (BM * BN),)

    def run(a, b, bias):
        """Called as fn(*inputs) by the plan runner."""
        nonlocal warm_launch
        if warm_launch is not None:
            output = warm_launch(a, b, bias)
            if output is not None:
                return (output,)
        if a.dim() != 2 or a.shape[0] != spec['M'] or a.shape[1] != spec['K'] or b.shape[1] != spec['N']:
            raise RuntimeError('flagtree native matmul: shape changed at call time')
        out = torch.empty((spec['M'], spec['N']), device=a.device, dtype=a.dtype)
        act = spec['act']
        if grouped:
            binary = kernel[grid](a, b, bias, out, spec['N'], BK, BM, BN, RELU=grouped_relu,
                                  num_warps=warps, num_stages=3, is_sdnn=True)
            if act == 0 or grouped_relu:
                if not proof['resource_verified']:
                    record(binary)
                    warm_launch = bind(kernel, binary, (*grid, 1, 1), (a, b, bias), (spec['M'], spec['N']), (), 'sdnn')
                    proof.update(prepared_launch=warm_launch is not None, schedule='grouped_complete_tiles')
                return (out,)
        if act in (0, 2) or (act == 1 and spec.get('silu_path') == 'epilogue'):
            mode = 3 if act == 1 else act
            binary = kernel[grid](a, b, bias, out, spec['M'], spec['N'], spec['K'],
                                 ACT=mode, BM=BM, BN=BN, BK=BK, EVEN_K=spec.get('even_k', True),
                                 num_warps=warps, num_stages=1)
            if not proof['resource_verified']:
                record(binary)
                warm_launch = bind(kernel, binary, (*grid, 1), (a, b, bias), (spec['M'], spec['N']),
                                   (spec['M'], spec['N'], spec['K']), 'sdnn')
                proof['prepared_launch'] = warm_launch is not None
            return (out,)
        # Split activation: matmul+bias kernel, then a 1D pointwise EW kernel.
        if grouped:
            first, first_grid, scalars = binary, (*grid, 1, 1), ()
        else:
            first = kernel[grid](a, b, bias, out, spec['M'], spec['N'], spec['K'],
                                 ACT=0, BM=BM, BN=BN, BK=BK, EVEN_K=spec.get('even_k', True),
                                 num_warps=warps, num_stages=1)
            first_grid, scalars = (*grid, 1), (spec['M'], spec['N'], spec['K'])
        total = spec['M'] * spec['N']
        tail = {3: ew_gelu_tanh, 4: ew_gelu_exact}.get(act)
        if tail is None:
            tail = ew_kernel if spec['silu_path'] == 'sdnn' else ew_cluster
        tail_grid = ((total + 8191) // 8192, 1, 1)
        second = tail[tail_grid](out, out, total, XB=8192, TAIL=bool(total % 8192), num_warps=8, num_stages=1, is_sdnn=tail is ew_kernel)
        if not proof['resource_verified']:
            record(first, second)
            warm_launch = bind(kernel, first, first_grid, (a, b, bias), (spec['M'], spec['N']), scalars, 'sdnn',
                               inplace_tail=(tail, second, tail_grid, (total,)))
            proof['prepared_launch'] = warm_launch is not None
        return (out,)

    proof = {'route': 'native_mm_grouped_relu' if grouped_relu else 'native_mm',
             'resource': 'sdnn', 'resource_verified': False, 'external_calls': 0,
             'regions': [], 'tiles': {'BM': BM, 'BN': BN, 'BK': BK, 'warps': warps},
             'kernel_count': (1 if (spec['act'] in (0, 2)
                                    or (spec['act'] == 1 and spec.get('silu_path') == 'epilogue'))
                              else 2),
             'compiler_options': {}}
    return run, proof
