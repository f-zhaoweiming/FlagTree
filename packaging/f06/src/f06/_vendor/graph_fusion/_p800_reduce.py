"""Whole-kernel replacement for row-wise reduction regions (P800).

The native path lowers these regions to a persistent reduction that keeps the
whole row in registers (``[XBLOCK, 1024]``), which measures 5-17x slower than the
vendor eager kernels on this backend. These kernels instead read a *contiguous*
``[R, N]`` block per program (flat loads, like eager), reduce inside the block,
and write R results - the "block + single pass" shape that keeps memory access
coalesced.

Supported regions:
    (a * b).sum(-1), softmax(a + b, -1), layer_norm(a + b, eps)

Every kernel falls back to the native region on any shape/dtype mismatch.
"""
import hashlib
import json
import operator
import os
import tempfile

_VERSION = 'flat-red-v3'
_ROWS = (2, 4, 8, 16)
_CHUNKS = (128, 256, 512, 1024)
_WARPS = (4, 8)

# Measured fallback policy: these (kind, dtype, shape) classes are slower than
# eager with the current plain-path kernels (see the weekly report tables), so
# declining the region lets the graph run eager instead of regressing.
# Set FLAGTREE_FALLBACK_HEURISTIC=0 to disable and always attempt fusion.
def _fallback_enabled():
    return os.environ.get('FLAGTREE_FALLBACK_HEURISTIC', '1') != '0'


def _decline_mulsum(m, n, dt):
    if not _fallback_enabled():
        return False
    # fp16 reduce has no usable fast path: 0.20-0.72x eager at n >= 512.
    if dt == 'float16' and n >= 512:
        return True
    # fp32 reduce: only the 2048x2048 class wins (1.10x); smaller ones lose ~10%.
    if dt == 'float32' and m * n <= 512 * 1024:
        return True
    return False


def _decline_layernorm(m, n, dt):
    if not _fallback_enabled() or m * n < 1024 * 1024:
        return False
    # The old streaming and TLE kernels lose on these sizes. The ordered
    # kernel is a different implementation; let the public timing gate rank it.
    from ._p800_layernorm import segment_width, toolchain_available
    return segment_width(m, n, dt) is None or not toolchain_available()


_DTYPES = ('float32', 'float16')
_KIND = {'mul_sum': 1, 'add_softmax': 2, 'add_layernorm': 3, 'add_layernorm_affine': 4}


def _name(node):
    return getattr(getattr(node, 'target', None), '__name__', '')


def _is(node, targets, names):
    target = getattr(node, 'target', None)
    return target in targets or _name(node) in names


def _is_last_dim(dim, rank=2):
    if isinstance(dim, (list, tuple)):
        return len(dim) == 1 and int(dim[0]) in (-1, rank - 1)
    return isinstance(dim, int) and dim in (-1, rank - 1)


def _arg(node, name, index):
    value = node.kwargs.get(name)
    if value is None and len(node.args) > index:
        value = node.args[index]
    return value


def _output_nodes(gm):
    import torch
    out = next((n for n in gm.graph.nodes if n.op == 'output'), None)
    if out is None or not out.args:
        return frozenset()
    value = out.args[0]
    if isinstance(value, torch.fx.Node):
        return frozenset((value,))
    if isinstance(value, (tuple, list)):
        return frozenset(item for item in value if isinstance(item, torch.fx.Node))
    return frozenset()


def _placeholder_operands(node):
    import torch
    return (len(node.args) >= 2
            and all(isinstance(item, torch.fx.Node) and item.op == 'placeholder'
                    for item in node.args[:2]))


def _add_feeding(node, add_targets):
    import torch
    if node is None or not node.args:
        return None
    value = node.args[0]
    if not isinstance(value, torch.fx.Node) or not _is(value, add_targets, 'add'):
        return None
    if value.kwargs.get('alpha') not in (None, 1):
        return None
    return value if _placeholder_operands(value) else None


def match(gm, inputs):
    """Return a spec for a supported row reduction region, else None."""
    import torch
    import torch.nn.functional as F

    add_targets = (torch.add, operator.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.default, 'add')
    mul_targets = (torch.mul, operator.mul, torch.ops.aten.mul.Tensor)
    sum_targets = (torch.sum, torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default)
    softmax_targets = (torch.softmax, F.softmax, torch.ops.aten._softmax.default, torch.ops.aten.softmax.int, 'softmax')
    layernorm_targets = (F.layer_norm, torch.ops.aten.native_layer_norm.default,
                         torch.ops.aten.layer_norm.default)

    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    if len(tensors) < 2:
        return None
    a, b = tensors[0], tensors[1]
    swapped = (len(tensors) == 2 and b.dim() == 2
               and (a.dim() == 1 or (a.dim() == 2 and a.shape[0] == 1 and b.shape[0] != 1)))
    if swapped:
        a, b = b, a
    if a.dim() != 2:
        return None
    m, n = int(a.shape[0]), int(a.shape[1])
    dt = str(a.dtype).replace('torch.', '')
    if dt not in _DTYPES and dt != 'bfloat16':
        return None
    if b.device != a.device or b.dtype != a.dtype or tuple(b.shape) not in ((n,), (1, n)):
        return None
    # Official XPU guidance: layouts the kernel does not implement must be
    # rejected on the host, otherwise they fail silently (innermost stride != 1,
    # zero-stride broadcast sources). Our kernels index rows*N + col, so the row
    # pitch has to be exactly N and both tensors contiguous.
    if not a.is_contiguous() or int(a.stride(-1)) != 1:
        return None
    if not b.is_contiguous() or int(b.stride(-1)) != 1:
        return None

    nodes = [x for x in gm.graph.nodes if x.op != 'output']
    mul = sum_node = softmax = layernorm = None
    for node in nodes:
        if node.op not in ('call_function', 'call_method'):
            continue
        if mul is None and _is(node, mul_targets, ('mul',)):
            mul = node
        elif sum_node is None and _is(node, sum_targets, ('sum',)):
            sum_node = node
        elif softmax is None and _is(node, softmax_targets, ('softmax',)):
            softmax = node
        elif layernorm is None and _is(node, layernorm_targets, ('layer_norm',)):
            layernorm = node

    outputs = _output_nodes(gm)
    roots = [node for node in (sum_node, softmax, layernorm) if node is not None]
    # A region is exactly one fused reduction. In particular, the denominator
    # sum inside a softmax must never be matched as a plain row sum.
    if len(roots) != 1 or outputs != {roots[0]}:
        return None

    if softmax is None and (swapped or m == 0 or n == 0):
        return None
    if dt == 'bfloat16' and softmax is None:
        from ._p800_layernorm import segment_width, toolchain_available
        from ._p800_vector_rows import configuration
        ordered_norm = layernorm is not None and segment_width(m, n, dt) is not None and toolchain_available()
        vector_kind = 'add_softmax' if softmax is not None else 'mul_sum' if sum_node is not None else None
        if not ordered_norm and configuration(vector_kind, dt, m, n) is None:
            return None

    if softmax is not None:
        if _add_feeding(softmax, add_targets) is None:
            return None
        if not _is_last_dim(_arg(softmax, 'dim', 1)):
            return None
        if softmax.target == torch.ops.aten._softmax.default:
            if _arg(softmax, 'half_to_float', 2):
                return None
        elif _arg(softmax, 'dtype', 3 if softmax.target is F.softmax else 2) is not None:
            return None
        return dict(kind='add_softmax', m=m, n=n, dtype=dt, eps=0.0, swapped_inputs=swapped)

    if layernorm is not None:
        if _add_feeding(layernorm, add_targets) is None:
            return None
        shape = _arg(layernorm, 'normalized_shape', 1)
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)) or tuple(int(x) for x in shape) != (n,):
            return None
        args = layernorm.args
        kwargs = layernorm.kwargs
        weight = args[2] if len(args) > 2 else kwargs.get('weight')
        bias = args[3] if len(args) > 3 else kwargs.get('bias')
        affine = weight is not None or bias is not None
        eps = float(args[4]) if len(args) > 4 else float(kwargs.get('eps', 1e-5))
        nparam = len([t for t in tensors[2:] if isinstance(t, torch.Tensor)])
        if affine and (weight is None or bias is None or nparam != 2
                       or any(t.shape != (n,) or t.dtype != a.dtype or t.device != a.device
                              or not t.is_contiguous() for t in tensors[2:])):
            return None
        if _decline_layernorm(m, n, dt) and (a.device.type != 'cuda' or not (0 < n <= 2048 and m*n <= 0x7fffffff)):
            return None
        return dict(kind='add_layernorm_affine' if affine else 'add_layernorm',
                    m=m, n=n, dtype=dt, eps=eps, keepdim=False, nparam=nparam)

    # mul-sum: the remaining root is the sum, and it must fold a mul of the
    # two region inputs (not a product internal to a larger fused op).
    if mul is None or sum_node.args[0] is not mul:
        return None
    if not _is_last_dim(_arg(sum_node, 'dim', 1)):
        return None
    if (not _placeholder_operands(mul) or set(mul.args[:2]) != {n for n in nodes if n.op == 'placeholder'}
            or sum_node.kwargs.get('dtype') is not None):
        return None
    kd = sum_node.kwargs.get('keepdim')
    if kd is None and len(sum_node.args) > 2:
        kd = sum_node.args[2]
    if _decline_mulsum(m, n, dt):
        if a.device.type != 'cuda':
            return None
        if n > 2048 or m*n > 0x7fffffff:
            from ._p800_vector_rows import configuration
            if configuration('mul_sum', dt, m, n) is None:
                return None
    return dict(kind='mul_sum', m=m, n=n, dtype=dt, eps=0.0, keepdim=bool(kd),
                nparam=len([t for t in tensors[2:] if isinstance(t, torch.Tensor)]))


def _kernel():
    import triton
    import triton.language as tl

    # Pointers stay out of the specialization key (official XPU guidance), and
    # the raw pointers must not force a re-specialisation per call.
    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr', 'w_ptr', 'bias_ptr'])
    def _flat_row_reduce(a_ptr, b_ptr, o_ptr, w_ptr, bias_ptr, KIND: tl.constexpr,
                         N: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
                         ROUND: tl.constexpr, EPS: tl.constexpr, AFFINE: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)
        cols = tl.arange(0, C)
        acc = tl.zeros([R, C], tl.float32)
        if KIND == 1:
            for off in range(0, N, C):
                idx = rows[:, None] * N + off + cols[None, :]
                va = tl.load(a_ptr + idx).to(tl.float32)
                vb = tl.load(b_ptr + off + cols[None, :]).to(tl.float32)
                p = va * vb
                if ROUND == 1:
                    p = p.to(tl.float16).to(tl.float32)
                acc += p
            tl.store(o_ptr + rows, tl.sum(acc, 1))
        else:
            # Shift the observations before accumulating their moments.
            origin = (tl.load(a_ptr + rows * N) + tl.load(b_ptr)).to(tl.float32)
            acc2 = tl.zeros([R, C], tl.float32)
            for off in range(0, N, C):
                idx = rows[:, None] * N + off + cols[None, :]
                v = (tl.load(a_ptr + idx) + tl.load(b_ptr + off + cols[None, :])).to(tl.float32)
                centered = v - origin[:, None]
                acc += centered
                acc2 += centered * centered
            mean = tl.sum(acc, 1) / N
            var = tl.maximum(tl.sum(acc2, 1) / N - mean * mean, 0.0)
            inv = 1.0 / tl.sqrt(var + EPS)
            for off in range(0, N, C):
                idx = rows[:, None] * N + off + cols[None, :]
                v = (tl.load(a_ptr + idx) + tl.load(b_ptr + off + cols[None, :])).to(tl.float32)
                out = ((v - origin[:, None]) - mean[:, None]) * inv[:, None]
                if AFFINE:
                    wv = tl.load(w_ptr + off + cols[None, :]).to(tl.float32)
                    bv = tl.load(bias_ptr + off + cols[None, :]).to(tl.float32)
                    out = out * wv + bv
                tl.store(o_ptr + idx, out)

    return _flat_row_reduce


def _masked_row_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def row(a, b, w, shift, out, N: tl.constexpr, BLOCK: tl.constexpr, SUM: tl.constexpr,
            AFFINE: tl.constexpr, FINAL: tl.constexpr, EPS: tl.constexpr):
        r = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        pos = r * N + col
        valid = col < N
        x = tl.load(a + pos, mask=valid, other=0.).to(tl.float32)
        bias = tl.load(b + col, mask=valid, other=0.).to(tl.float32)
        if SUM:
            prod = x * bias
            if not FINAL:
                prod = prod.to(a.dtype.element_ty).to(tl.float32)
            value = tl.sum(prod, 0)
            tl.store(out + r, value)
        else:
            z = x + bias
            if not FINAL:
                z = z.to(a.dtype.element_ty).to(tl.float32)
            mean = tl.sum(z, 0) / N
            delta = tl.where(valid, z - mean, 0.)
            inv = 1. / tl.sqrt(tl.sum(delta * delta, 0) / N + EPS)
            value = delta * inv
            if AFFINE:
                value = value * tl.load(w + col, mask=valid, other=0.).to(tl.float32)
                value += tl.load(shift + col, mask=valid, other=0.).to(tl.float32)
            tl.store(out + pos, value, mask=valid)

    return row


def _masked_rows(spec, proof):
    import torch
    from ._p800_bound_launch import bind
    from ._resource_codegen import bind_evidence

    m, n = spec['m'], spec['n']
    is_sum = spec['kind'] == 'mul_sum'
    affine = spec['kind'] == 'add_layernorm_affine'
    shape = ((m, 1) if spec['keepdim'] else (m,)) if is_sum else (m, n)
    block = 1 << (n - 1).bit_length()
    args = n, block, is_sum, affine, spec.get('final_round_only', False), spec['eps']
    grid = (m, 1, 1)
    kernel = _masked_row_kernel()
    bound = None

    def run(xs):
        nonlocal bound
        a, b = xs[:2]
        operands = (a, b, xs[2], xs[3]) if affine else (a, b, b, b)
        if bound is not None:
            value = bound(*operands)
            if value is not None:
                return (value,)
        value = torch.empty(shape, device=a.device, dtype=a.dtype)
        binary = kernel[grid](*operands, value, *args, num_warps=4, num_stages=1)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
            bound = bind(kernel, binary, grid, operands, shape, (), 'cluster')
            proof.update(tiles={'columns': block}, prepared_launch=bound is not None)
        return (value,)

    return run, proof


def _softmax_pointer_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def logits(a, b, row, col, N: tl.constexpr, FINAL_ONLY: tl.constexpr, TAIL: tl.constexpr):
        if TAIL:
            x = tl.load(a + row * N + col, mask=col < N, other=0)
            bias = tl.load(b + col, mask=col < N, other=0)
        else:
            x = tl.load(a + row * N + col)
            bias = tl.load(b + col)
        value = x.to(tl.float32) + bias.to(tl.float32)
        if not FINAL_ONLY:
            value = value.to(a.dtype.element_ty).to(tl.float32)
        if TAIL:
            value = tl.where(col < N, value, -float('inf'))
        return value

    @triton.jit
    def store(out, offsets, mask, value, FINAL: tl.constexpr):
        if out.dtype.element_ty == tl.bfloat16:
            bits = value.to(tl.uint32, bitcast=True)
            high = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
            if not FINAL:
                high = tl.where((bits & 0x7fffffff) > 0x7f800000, 0x7fc0, high)
            tl.store(out.to(tl.pointer_type(tl.uint16)) + offsets, high.to(tl.uint16), mask=mask)
        else:
            tl.store(out + offsets, value, mask=mask)

    @triton.jit(do_not_specialize=['a', 'b', 'out'], do_not_specialize_on_alignment=['a', 'b', 'out'])
    def row_softmax(a, b, out, N: tl.constexpr, BLOCK: tl.constexpr, FINAL_ONLY: tl.constexpr,
                    LARGE_INDEX: tl.constexpr):
        row = tl.program_id(0)
        if LARGE_INDEX:
            row = row.to(tl.int64)
        lane = tl.arange(0, BLOCK)
        v = logits(a, b, row, lane, N, FINAL_ONLY, N != BLOCK)
        e = tl.exp(v - tl.max(v, 0))
        store(out, row * N + lane, lane < N, e * tl.div_rn(1., tl.sum(e, 0)), FINAL_ONLY)
    return row_softmax


def _tle_kernel():
    """TLE row-softmax: descriptor DMA + [XBLOCK] streaming online state.

    Mirrors the FlagGems `_tle_softmax_fused_kernel` structure: the row is
    streamed in YBLOCK columns through a per-programme LM buffer, the running
    max / sum-of-exp stay [XBLOCK] vectors, and the tail is handled by the
    descriptor clamp (no per-lane mask). Measured 63 us @2048x1024 vs 105 us
    for the pointer multirow kernel and 33 us for eager.
    """
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    @triton.jit
    def row_maximum(a, b):
        # Only the Softmax shift uses this ordered maximum. Any input NaN
        # still reaches exp and the row sum; either zero sign gives exp(0)=1.
        return tl.where(a > b, a, b)

    @triton.jit(do_not_specialize=['N'])
    def _tle_row_softmax(a_desc, b_desc, o_desc, N, XB: tl.constexpr, YB: tl.constexpr,
                         DT: tl.constexpr, SHARED_BIAS: tl.constexpr = False, FINAL_ONLY: tl.constexpr = False):
        # Single chunk (YB == N): the bias is copied into LM once per
        # programme, never inside a chunk loop (a per-chunk 1D descriptor copy
        # measured 1.2 ms). Plain GM loads cannot be mixed with descriptor
        # copies here: they miscompile (err ~4) and fall into the slow path.
        pid = tl.program_id(0)
        row_off = pid * XB
        buf = tle.gpu.alloc([XB, YB], dtype=DT, scope=tle.gpu.lmem)
        if SHARED_BIAS:
            bb = tle.gpu.alloc([YB], dtype=DT, scope=tle.gpu.smem)
        else:
            bb = tle.gpu.alloc([YB], dtype=DT, scope=tle.gpu.lmem)
        rows = tl.broadcast_to(tl.arange(0, XB)[:, None], (XB, YB))
        cols = tl.broadcast_to(tl.arange(0, YB)[None, :], (XB, YB))
        p = tle.gpu.local_ptr(buf, (rows, cols))
        bp = tle.gpu.local_ptr(bb, (tl.arange(0, YB),))
        tle.gpu.copy(b_desc, bb, [YB], [0])
        bv = tl.load(bp).to(tl.float32)
        tle.gpu.copy(a_desc, buf, [XB, YB], [row_off, 0])
        if FINAL_ONLY:
            v = tl.load(p).to(tl.float32) + bv[None, :]
        else:
            v = (tl.load(p) + bv[None, :].to(DT)).to(tl.float32)
        m = tl.reduce(v, 1, row_maximum)
        e = tl.exp(v - m[:, None])
        z = tl.sum(e, 1)
        inv = tl.div_rn(1.0, z)
        result = e * inv[:, None]
        if FINAL_ONLY and DT == tl.bfloat16:
            words = result.to(tl.uint32, bitcast=True)
            rounded = (words + 0x7fff + ((words >> 16) & 1)) & 0xffff0000
            result = rounded.to(tl.float32, bitcast=True)
        tl.store(p, result.to(DT))
        tle.gpu.copy(buf, o_desc, [XB, YB], [row_off, 0])

    return _tle_row_softmax


def _tle_ln_kernel():
    """TLE LayerNorm: single-chunk LM tile, statistics and normalisation in one
    pass over the row (one GM read + one GM write).  This is the norm-doc
    "statistics must not make a second global round trip" shape adapted to a
    single LM-resident chunk: 512x1024 = 21.7 us (1.33x eager) vs 24.5 us for
    the pointer kernel, 2048x2048 = 91.6 us (0.58x) vs 215.8 us."""
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    @triton.jit(do_not_specialize=['N', 'EPS'])
    def _tle_row_layernorm(a_desc, b_desc, w_desc, bias_desc, o_desc, N, EPS,
                           XB: tl.constexpr, YB: tl.constexpr, DT: tl.constexpr,
                           AFFINE: tl.constexpr):
        pid = tl.program_id(0)
        row_off = pid * XB
        buf = tle.gpu.alloc([XB, YB], dtype=DT, scope=tle.gpu.lmem)
        bb = tle.gpu.alloc([YB], dtype=DT, scope=tle.gpu.lmem)
        rows = tl.broadcast_to(tl.arange(0, XB)[:, None], (XB, YB))
        cols = tl.broadcast_to(tl.arange(0, YB)[None, :], (XB, YB))
        p = tle.gpu.local_ptr(buf, (rows, cols))
        bp = tle.gpu.local_ptr(bb, (tl.arange(0, YB),))
        tle.gpu.copy(b_desc, bb, [YB], [0])
        bv = tl.load(bp).to(tl.float32)
        if AFFINE:
            wl = tle.gpu.alloc([YB], dtype=DT, scope=tle.gpu.lmem)
            bl = tle.gpu.alloc([YB], dtype=DT, scope=tle.gpu.lmem)
            wp = tle.gpu.local_ptr(wl, (tl.arange(0, YB),))
            bp2 = tle.gpu.local_ptr(bl, (tl.arange(0, YB),))
            tle.gpu.copy(w_desc, wl, [YB], [0])
            tle.gpu.copy(bias_desc, bl, [YB], [0])
            wv = tl.load(wp).to(tl.float32)
            bvv = tl.load(bp2).to(tl.float32)
        tle.gpu.copy(a_desc, buf, [XB, YB], [row_off, 0])
        v = (tl.load(p) + bv[None, :].to(DT)).to(tl.float32)
        origin = tl.min(v, 1)
        shifted = v - origin[:, None]
        mean = tl.sum(shifted, 1) / N
        centered = shifted - mean[:, None]
        var = tl.sum(centered * centered, 1) / N
        rstd = 1.0 / tl.sqrt(var + EPS)
        o = centered * rstd[:, None]
        if AFFINE:
            o = o * wv[None, :] + bvv[None, :]
        tl.store(p, o.to(DT))
        tle.gpu.copy(buf, o_desc, [XB, YB], [row_off, 0])

    return _tle_row_layernorm


_TLE_DTYPE = {'float32': 'float32', 'float16': 'float16'}


def _cache_path(key):
    root = (os.environ.get('FLAGTREE_FLAT_CACHE') or os.environ.get('TRITON_CACHE_DIR')
            or os.path.join(tempfile.gettempdir(), 'flagtree-flat-autotune'))
    path = os.path.join(root, 'flagtree-flat-autotune')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, hashlib.sha256(key.encode()).hexdigest()[:32] + '.json')


def _load(key):
    try:
        with open(_cache_path(key)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _store(key, record):
    with open(_cache_path(key), 'w') as fh:
        json.dump(dict(record, version=_VERSION), fh)


def suggest_xblock(gm, inputs):
    spec = match(gm, inputs)
    return None if spec is None else 8



def _bind_row_softmax(binary, kernel, grid, ad, bd, od, bias_input):
    """Bind the validated cluster descriptor layout, keeping tensors and stream live."""
    import sys
    import torch
    from triton import knobs
    from triton.runtime import driver

    launcher = binary.run
    if (type(launcher).__module__ != 'triton.backends.xpu.driver'
            or binary.metadata.is_sdnn or not hasattr(launcher, 'launch')):
        return None
    # XPULauncher expands each cluster descriptor to base, shape and strides.
    # Only these integers are constant. Outputs and all three addresses are
    # supplied afresh, including during graph capture and concurrent calls.
    shape, strides = tuple(ad.shape), tuple(ad.strides)
    bias_shape, bias_strides = tuple(bd.shape), tuple(bd.strides)
    if (len(shape) != 2 or bias_shape != (shape[1],) or
            tuple(od.shape) != shape or tuple(od.strides) != strides):
        return None
    rows, cols = shape
    row_stride, col_stride = strides
    bias_stride, = bias_strides
    metadata = binary.packed_metadata
    device, dtype = ad.base.device, ad.base.dtype
    active = driver.active
    prism = getattr(sys.modules[type(kernel).__module__], '_flagprism', None)
    entry = launcher.launch
    log_launches = getattr(sys.modules[type(launcher).__module__], '_PRINT_TRITON_FUNC', False)
    runtime = knobs.runtime

    # The generated Inductor wrapper uses these Torch 2.9 primitives for a
    # static layout. Keep the full descriptor path for tensor subclasses,
    # dispatch modes and builds that do not expose the native allocator.
    input_bias_shape, input_bias_stride = tuple(bias_input.shape), tuple(bias_input.stride())
    native_layout = None
    native_empty = None
    if (device.type == 'cuda' and torch.__version__.startswith('2.9.')
            and type(ad.base) is torch.Tensor and type(bias_input) is torch.Tensor):
        guards = torch._C._dynamo.guards
        guard_type = getattr(guards, 'TensorGuards', None)
        native_empty = getattr(guards, '_empty_strided_cuda', None)
        if (guard_type is not None and native_empty is not None
                and not torch._C._len_torch_dispatch_stack() and not torch._C._is_torch_function_mode_enabled()):
            native_layout = guard_type(ad.base, bias_input, dynamic_dims_sizes=None, dynamic_dims_strides=None).check

    sized_pointers = native_layout is not None and getattr(launcher, 'supports_sized_pointers', False)
    elements = rows * cols
    from ._p800_bound_launch import _current_device, _pack_launch
    current_device = _current_device(active)
    packed_launch = None
    if sized_pointers:
        packed_launch = _pack_launch(
            entry, (grid, 1, 1, None, binary.function, metadata, None, None, None),
            (None, rows, cols, row_stride, col_stride, None, cols, bias_stride,
             None, rows, cols, row_stride, col_stride, cols),
            (9, 14, 17), (elements, cols, elements), native_empty, (shape, strides, dtype),
            active.get_current_stream, device.index)

    def launch(a, b):
        if native_layout is not None:
            if (not native_layout(a, b) or torch._C._len_torch_dispatch_stack()
                    or torch._C._is_torch_function_mode_enabled()):
                return None
        elif (a.device != device or b.device != device or a.dtype != dtype or b.dtype != dtype
              or a.shape != shape or b.shape != input_bias_shape or a.stride() != strides or b.stride() != input_bias_stride):
            return None
        enter, exit_hook = runtime.launch_enter_hook, runtime.launch_exit_hook
        if (current_device() != device.index
                or kernel.pre_run_hooks or runtime.debug
                or getattr(enter, 'calls', enter) or getattr(exit_hook, 'calls', exit_hook)
                or getattr(knobs.compilation, 'instrumentation_mode', '') or log_launches
                or os.environ.get('PRINT_TRITON_FUNC', '0') != '0'
                or (prism is not None and prism._registered_components())):
            return None
        if packed_launch is not None:
            return packed_launch(a, b)
        a_ptr, b_ptr = a.data_ptr(), b.data_ptr()
        if (a_ptr | b_ptr) & 15:
            return None
        out = native_empty(shape, strides, dtype) if native_layout is not None else torch.empty_like(a)
        stream = active.get_current_stream(device.index)
        if sized_pointers:
            a_arg, b_arg, out_arg = (a_ptr, elements), (b_ptr, cols), (out.data_ptr(), elements)
        else:
            a_arg, b_arg, out_arg = a, b, out
        entry(grid, 1, 1, stream, binary.function, metadata, None, None, None,
              a_arg, rows, cols, row_stride, col_stride, b_arg, cols, bias_stride,
              out_arg, rows, cols, row_stride, col_stride, cols)
        return out

    return launch


def _bind_layernorm_launch(binary, kernel, programs, inputs, affine):
    """Bind a fixed KL3 pointer ABI; every call supplies live tensors and stream."""
    import sys
    import torch
    from triton import knobs
    from triton.runtime import driver

    launcher = binary.run
    guards = torch._C._dynamo.guards
    if (type(launcher).__module__ != 'triton.backends.xpu.driver' or binary.metadata.is_sdnn
            or not getattr(launcher, 'supports_sized_pointers', False)
            or not hasattr(guards, '_empty_strided_cuda') or not hasattr(guards, 'TensorGuards')):
        return None
    if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in inputs):
        return None
    layout_ok = guards.TensorGuards(*inputs, dynamic_dims_sizes=None, dynamic_dims_strides=None).check
    expected_layouts = tuple((t.shape, t.stride()) for t in inputs)
    matrix = inputs[0]
    shape, stride, dtype = tuple(matrix.shape), tuple(matrix.stride()), matrix.dtype
    numel, cols, device = matrix.numel(), shape[1], matrix.device.index
    active = driver.active
    from ._p800_bound_launch import _current_device, _pack_launch
    current, stream = _current_device(active), active.get_current_stream
    empty = guards._empty_strided_cuda
    launch, function, metadata = launcher.launch, binary.function, binary.packed_metadata
    # KL3 launchers only consume element counts for metadata-listed arguments.
    counts = (numel, cols, cols, cols, numel)
    if getattr(binary.metadata, 'tensor_args', None) in ([], ()):
        counts = (None,) * 5
    packed_launch = _pack_launch(
        launch, (programs, 1, 1, None, function, metadata, None, None, None),
        (None,) * 5, (9, 10, 11, 12, 13), counts,
        empty, (shape, stride, dtype), stream, device)
    runtime = knobs.runtime
    prism = getattr(sys.modules[type(kernel).__module__], '_flagprism', None)
    launcher_module = sys.modules[type(launcher).__module__]

    def run(xs):
        nonlocal layout_ok
        enter, leave = runtime.launch_enter_hook, runtime.launch_exit_hook
        if (current() != device or torch.is_grad_enabled() or torch._C._len_torch_dispatch_stack()
                or torch._C._is_torch_function_mode_enabled() or kernel.pre_run_hooks or runtime.debug
                or runtime.disable_launch_fast_path or getattr(enter, 'calls', enter) or getattr(leave, 'calls', leave)
                or knobs.compilation.instrumentation_mode or getattr(launcher_module, '_PRINT_TRITON_FUNC', False)
                or (prism is not None and prism._registered_components())):
            return None
        if not layout_ok(*xs):
            # Calibration detaches module Parameters. The public frame supplies
            # the live Parameters. Rebind the guard only for the identical ABI.
            if (not affine or len(xs) != len(expected_layouts)
                    or any(type(t) not in (torch.Tensor, torch.nn.Parameter) or t.dtype != dtype
                           or t.device.type != 'cuda' or t.device.index != device
                           or t.shape != size or t.stride() != strides
                           for t, (size, strides) in zip(xs, expected_layouts))):
                return None
            layout_ok = guards.TensorGuards(*xs, dynamic_dims_sizes=None, dynamic_dims_strides=None).check
        if packed_launch is not None:
            if affine:
                return packed_launch(*xs)
            matrix, bias = xs
            return packed_launch(matrix, bias, bias, bias)
        a, b = xs[0].data_ptr(), xs[1].data_ptr()
        w, shift = (xs[2].data_ptr(), xs[3].data_ptr()) if affine else (b, b)
        # A new alignment class goes through Triton's normal specialization.
        if (a | b | w | shift) & 15:
            return None
        out = empty(shape, stride, dtype)
        launch(programs, 1, 1, stream(device), function, metadata, None, None, None,
               (a, numel), (b, cols), (w, cols), (shift, cols), (out.data_ptr(), numel))
        return out

    return run


def compile(spec, tensors):  # pragma: no cover - exercised on device
    import torch

    from ._resource_codegen import bind_evidence

    if spec.get('swapped_inputs'):
        canonical = dict(spec, swapped_inputs=False)
        fn, proof = compile(canonical, (tensors[1], tensors[0]))
        return lambda xs: fn((xs[1], xs[0])), proof
    if spec['kind'] == 'add_softmax' and (not spec['m'] or not spec['n']):
        return (lambda xs: (torch.empty_like(xs[0]),)), {
            'route': 'empty_softmax', 'resource': 'none', 'resource_verified': False,
            'kernel_count': 0, 'direct_launches': 0, 'kernel_evidence': [],
        }

    key = '%s|%s|%dx%d|%s' % (_VERSION, spec['kind'], spec['m'], spec['n'], spec['dtype'])
    proof = {'route': 'flat_reduce', 'resource': 'cluster', 'resource_verified': False,
             'direct_launches': 1, 'emitted': 'flat', 'region': dict(spec),
             'match_key': key, 'tiles': None,
             'autotune': 'cached' if _load(key) else 'pending'}
    state = dict(kernel=None, tile=None, launcher=None)
    kind = _KIND[spec['kind']]
    round_flag = 1 if (spec['dtype'] == 'float16' and kind == 1
                       and os.environ.get('FLAGTREE_REDUCE_ROUND', '1') != '0') else 0

    affine = spec['kind'] == 'add_layernorm_affine'

    final_only = spec.get('final_round_only', False)
    compute_dtype = 'float32' if final_only else spec['dtype']
    # A full descriptor tile requires at least 32 rows, in multiples of eight.
    # Use the guarded vector-row kernel for partial batches of any supported
    # precision. The narrow-input variant adds in FP32 only for a final-only graph.
    vector_rows = (spec['m'] < 32 or spec['m'] % 8 or spec['n'] not in (512, 1024, 2048)
                   or spec['n'] == 2048 and 2 * spec['m'] <= 12 * 64)
    if kind == 2 and (final_only and spec['dtype'] == 'bfloat16' or
                      vector_rows and (final_only or spec['dtype'] == 'float32')):
        from ._p800_vector_rows import wide_softmax_configuration, prepare as prepare_vector
        config = wide_softmax_configuration(spec['m'], spec['n'])
        if config is not None:
            return prepare_vector(spec, config, proof)
    if kind == 1 or kind == 2 and not final_only:
        from . import _p800_vector_rows
        config = _p800_vector_rows.configuration(spec['kind'], compute_dtype, spec['m'], spec['n'])
        if config is not None:
            return _p800_vector_rows.prepare(spec, config, proof)

    if kind in (3, 4):
        from ._p800_layernorm import _packed_layernorm, _row_layernorm, rows_per_program, segment_width, toolchain_available
        width = segment_width(spec['m'], spec['n'], compute_dtype)
        if width is not None and toolchain_available():
            rows = rows_per_program(spec['m'], allow_tail=True, dtype=compute_dtype)
            if final_only and not affine and spec['dtype'] == 'bfloat16' and width == 512 and spec['n'] == 2048:
                rows = (spec['m'] + 11) // 12
            if affine and spec['dtype'] == 'float32' and spec['n'] == 2048 and spec['m'] >= 1024:
                rows = ((spec['m'] + 12*16 - 1) // (12*16)) * 16
            grid = ((spec['m'] + rows - 1) // rows,)
            kernel = _row_layernorm
            constants = (spec['n'], width, spec['eps'], affine, rows, spec['m'], final_only)
            tiles = {'rows': rows, 'segment': width}
            if final_only and spec['dtype'] in ('float16', 'bfloat16') and spec['m'] >= 1024 and spec['n'] == 2048:
                grid = (12,)
                kernel = _packed_layernorm
                constants = (spec['m'], spec['n'], spec['eps'], affine, grid[0])
                tiles = {'programs': grid[0], 'segment': width, 'row_storage': spec['dtype']}
            proof.update(route='ordered_layernorm', autotune='fixed', tiles=tiles)
            warm_launch = None

            def run_ordered(xs):
                nonlocal warm_launch
                if warm_launch is not None:
                    out = warm_launch(xs)
                    if out is not None:
                        return (out,)
                a, b = xs[:2]
                weight, shift = xs[2:4] if affine else (b, b)
                out = torch.empty_like(a)
                binary = kernel[grid](a, b, weight, shift, out, *constants, num_warps=4, num_stages=1)
                if not proof['resource_verified']:
                    bind_evidence(proof, binary)
                    warm_launch = _bind_layernorm_launch(binary, kernel, grid[0], xs, affine)
                    proof['prepared_launch'] = warm_launch is not None
                return (out,)

            return run_ordered, proof

    if kind != 2 and 0 < spec['n'] <= 2048 and spec['m'] * spec['n'] <= 0x7fffffff:
        if final_only or spec['m'] % min(_ROWS) or spec['n'] % min(_CHUNKS):
            return _masked_rows(spec, proof)

    def launch(a, b, out, tile, w=None, bias=None):
        grid = ((spec['m'] + tile['r'] - 1) // tile['r'],)
        state['kernel'][grid](a, b, out, w if w is not None else b,
                              bias if bias is not None else b,
                              KIND=kind, N=spec['n'], R=tile['r'], C=tile['c'],
                              ROUND=round_flag, EPS=spec['eps'], AFFINE=1 if affine else 0,
                              num_warps=tile['warps'], num_stages=1)

    def reference(a, b, w=None, bias=None):
        if kind == 1:
            return torch.sum(a * b, dim=-1).float()
        z = (a + b).float()
        mean = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, keepdim=True, unbiased=False)
        out = (z - mean) / (var + spec['eps']).sqrt()
        if w is not None:
            out = out * w.float()[None, :]
        if bias is not None:
            out = out + bias.float()[None, :]
        return out

    def prepare(a, b, w=None, bias=None):
        if state['tile'] is not None:
            return state['tile']
        state['kernel'] = _kernel()
        tile = _load(key)
        if tile is None:
            out = (torch.empty((spec['m'],), device=a.device, dtype=a.dtype) if kind == 1
                   else torch.empty_like(a))
            ref = reference(a, b, w, bias)
            best = None
            for r in _ROWS:
                if spec['m'] % r != 0:
                    continue
                for c in _CHUNKS:
                    if spec['n'] % c != 0:
                        continue
                    if r * c > 4096:
                        continue
                    for warps in _WARPS:
                        cand = dict(r=r, c=c, warps=warps)
                        try:
                            launch(a, b, out, cand, w, bias)
                            torch.cuda.synchronize()
                            err = float((out.float() - ref).abs().max())
                            scale = float(ref.abs().max()) + 1e-3
                            if err / scale > 5e-2:
                                import os as _os
                                if _os.environ.get('FLAGTREE_REDUCE_DEBUG'):
                                    print('[reduce] tile', cand, 'BAD err', err, 'scale', scale, flush=True)
                                continue
                            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                            start.record()
                            for _ in range(20):
                                launch(a, b, out, cand, w, bias)
                            end.record()
                            torch.cuda.synchronize()
                            us = start.elapsed_time(end) * 1000 / 20
                        except Exception as exc:
                            import os as _os
                            if _os.environ.get('FLAGTREE_REDUCE_DEBUG'):
                                print('[reduce] tile', cand, 'EXC', type(exc).__name__, str(exc)[:200], flush=True)
                            continue
                        if best is None or us < best[0]:
                            best = (us, cand)
            if best is None:
                raise RuntimeError('flagtree flat reduce: no valid tile for %r' % (spec,))
            tile = dict(best[1], us=round(best[0], 2))
            _store(key, tile)
        state['tile'] = tile
        proof['tiles'] = dict(tile)
        proof['autotune'] = 'cached' if tile.get('version') == _VERSION else 'measured'
        return tile

    def _bind(tile):
        # Bind grid/constexprs once: the official XPU guidance measures ~9 us
        # per call recovered by a flat, pre-bound launcher.
        def launch_bound(a, b, out, w, bias):
            grid = ((spec['m'] + tile['r'] - 1) // tile['r'],)
            return state['kernel'][grid](a, b, out, w if w is not None else b,
                                  bias if bias is not None else b,
                                  KIND=kind, N=spec['n'], R=tile['r'], C=tile['c'],
                                  ROUND=round_flag, EPS=spec['eps'],
                                  AFFINE=1 if affine else 0,
                                  num_warps=tile['warps'], num_stages=1)
        return launch_bound

    tle_xb = 16 if spec['m'] >= 256 and spec['m'] % 16 == 0 else 8
    tle_ok = (os.environ.get('FLAGTREE_SOFTMAX_TLE', '1') == '1'
              and kind == 2 and spec['n'] in (512, 1024, 2048)
              and spec['m'] % tle_xb == 0 and spec['m'] >= 32
              and (spec['dtype'] in _TLE_DTYPE or final_only and spec['dtype'] == 'bfloat16'))

    # The 512-row FP32 kernel vector-loads bias from cluster-shared memory.
    # FP16 uses private LM and eight rows per tile to reduce its device time.
    medium_row = tle_ok and (spec['m'], spec['n']) == (512, 1024)
    shared_bias = medium_row and compute_dtype == 'float32'
    if medium_row and compute_dtype == 'float16':
        tle_xb = 8
    row_schedule = medium_row or (tle_ok and compute_dtype == 'float32' and
                                   (spec['m'], spec['n']) == (2048, 2048))

    # This compiled region owns one static signature. Only immutable launch
    # arguments survive a call; the tensors, output and stream are always live.

    def run_tle(a, b):
        bound = state.get('row_runner')
        if bound is not None:
            out = bound(a, b)
            if out is not None:
                return out
        import triton.language as tl
        from triton.tools.tensor_descriptor import TensorDescriptor
        dt = getattr(tl, spec['dtype'])
        if 'tle_kernel' not in state:
            state['tle_kernel'] = _tle_kernel()
        kernel = state['tle_kernel']
        xb, yb = tle_xb, spec['n']
        ad = TensorDescriptor.from_tensor(a, [xb, yb])
        bd = TensorDescriptor.from_tensor(b if b.ndim == 1 else b.view(-1), [yb])
        out = torch.empty_like(a)
        od = TensorDescriptor.from_tensor(out, [xb, yb])
        grid = spec['m'] // xb
        binary = kernel[(grid,)](ad, bd, od, spec['n'], XB=xb, YB=yb, DT=dt, SHARED_BIAS=shared_bias, FINAL_ONLY=final_only,
                                 kl3_softmax_shift=True, kl3_row_softmax=row_schedule,
                                 num_warps=8, num_stages=1)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
        if 'row_runner' not in state:
            state['row_runner'] = _bind_row_softmax(binary, kernel, grid, ad, bd, od, b)
            proof['prepared_launch'] = state['row_runner'] is not None
        state['tle_done'] = True
        proof['tiles'] = {'xb': xb, 'yb': yb, 'route': 'tle'}
        return out

    from ._p800_bound_launch import bind as bind_pointer

    def run_pointer_softmax(a, b):
        bound = state.get('pointer_runner')
        if bound is not None:
            out = bound(a, b)
            if out is not None:
                return out
        m, n = spec['m'], spec['n']
        out = torch.empty_like(a)
        from ._p800_layernorm import toolchain_available
        if n > 8192 or (n % 32 and toolchain_available()):
            from ._p800_vector_rows import _stream_softmax
            if not toolchain_available():
                raise RuntimeError('wide P800 row Softmax requires the XTDK kernel toolchain')
            kernel = _stream_softmax
            grid = (min(12, m), 1, 1)
            binary = kernel[grid](a, b, out, m, n, grid[0], final_only, num_warps=4, num_stages=1)
            tile = {'block': 512, 'route': 'stream'}
        else:
            if state['kernel'] is None:
                state['kernel'] = _softmax_pointer_kernel()
            kernel = state['kernel']
            grid = (m, 1, 1)
            block = 1 << (n - 1).bit_length()
            binary = kernel[grid](a, b, out, n, block, final_only, m * n > 0x7fffffff, num_warps=4, num_stages=1)
            tile = {'block': block, 'route': 'masked'}
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
            state['pointer_runner'] = bind_pointer(kernel, binary, grid, (a, b), (m, n), (), 'cluster')
            proof.update(tiles=tile, prepared_launch=state['pointer_runner'] is not None)
        return out

    tle_ln_elems = 16384
    tle_ln_xb = max(4, tle_ln_elems // spec['n']) if spec['n'] else 0
    tle_ln_ok = (os.environ.get('FLAGTREE_TLE_LN', '0') == '1'
                 and kind in (3, 4) and spec['n'] in (512, 1024, 2048)
                 and tle_ln_xb >= 4 and spec['m'] % tle_ln_xb == 0
                 and spec['dtype'] in _TLE_DTYPE)

    def run_tle_ln(a, b, w, bias):
        import torch as _t
        import triton.language as _tl
        from triton.tools.tensor_descriptor import TensorDescriptor
        from triton.experimental.tle import language as _tle  # noqa: F401
        dt = getattr(_tl, _TLE_DTYPE[spec['dtype']])
        if 'tle_ln_kernel' not in state:
            state['tle_ln_kernel'] = _tle_ln_kernel()
        kernel = state['tle_ln_kernel']
        xb, yb = tle_ln_xb, spec['n']
        ad = TensorDescriptor.from_tensor(a, [xb, yb])
        bd = TensorDescriptor.from_tensor(b, [yb])
        wd = TensorDescriptor.from_tensor(w, [yb]) if affine else ad
        bid = TensorDescriptor.from_tensor(bias, [yb]) if affine else ad
        out = _t.empty_like(a)
        od = TensorDescriptor.from_tensor(out, [xb, yb])
        binary = kernel[(spec['m'] // xb,)](ad, bd, wd, bid, od, spec['n'], spec['eps'],
                                            XB=xb, YB=yb, DT=dt, AFFINE=int(affine),
                                            num_warps=8, num_stages=1)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
        proof['tiles'] = {'xb': xb, 'yb': yb, 'route': 'tle_ln'}
        return out

    def run(xs):
        a, b = xs[0], xs[1]
        if tle_ln_ok and not state.get('tle_ln_failed'):
            try:
                w0 = xs[2] if affine and len(xs) > 2 else None
                bias0 = xs[3] if affine and len(xs) > 3 else None
                return (run_tle_ln(a, b, w0, bias0),)
            except Exception as exc:
                import os as _os
                if _os.environ.get('FLAGTREE_REDUCE_DEBUG'):
                    print('[reduce] TLE LN failed:', type(exc).__name__, str(exc)[:300], flush=True)
                proof['resource_verified'] = False
                state['tle_ln_failed'] = True
        if tle_ok and not state.get('tle_failed'):
            try:
                return (run_tle(a, b),)
            except Exception as exc:
                import os as _os
                if _os.environ.get('FLAGTREE_REDUCE_DEBUG'):
                    print('[reduce] TLE failed:', type(exc).__name__, str(exc)[:300], flush=True)
                proof['resource_verified'] = False
                state['tle_failed'] = True
        if kind == 2:
            return (run_pointer_softmax(a, b),)
        import torch as _t
        bound = state['launcher']
        if bound is None:
            if a.dim() != 2 or a.shape[0] != spec['m'] or a.shape[1] != spec['n']:
                raise RuntimeError('flagtree flat reduce: shape changed at call time')
            w0 = xs[2] if affine and len(xs) > 2 else None
            bias0 = xs[3] if affine and len(xs) > 3 else None
            bound = state['launcher'] = _bind(prepare(a, b, w0, bias0))
        w = xs[2] if affine and len(xs) > 2 else None
        bias = xs[3] if affine and len(xs) > 3 else None
        if kind == 1:
            shape = (_t.Size([spec['m']]) if not spec.get('keepdim') else _t.Size([spec['m'], 1]))
            out = _t.empty(shape, device=a.device, dtype=a.dtype)
        else:
            out = _t.empty_like(a)
        binary = bound(a, b, out, w, bias)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
        return (out,)

    return run, proof
