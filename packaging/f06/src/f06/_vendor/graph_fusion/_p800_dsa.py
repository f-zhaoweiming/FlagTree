"""Whole-kernel replacement for P800 SDNN Mul+Sum / row-sum regions.

Inductor emits a pointwise ``tl.load`` reduction for these graphs, which is the
slow (and, above one row per program, incorrect) fallback on the SDNN path.
This module emits the documented ``tle.dsa`` 2D-DMA form instead: one DMA in per
column block, a per-column accumulator looped over the reduced axis, one 2D
store. Tile shapes are searched once per (kind, dtype, M, N) and cached next to
the Triton cache.

dtypes: f32 stages through an f32 buffer; f16/bf16 go through an int16 buffer as
a bit pattern (the dsa buffer does not accept bf16 directly) and are bitcast
back to their float type after the DMA.

tails: a partial last row block is carried by ``sizes`` on the DMA and a masked
store; the column axis stays divisible because the dsa buffer is statically full
width and a narrow column tail cannot be cleared safely on this path.
"""
import hashlib
import json
import operator
import os
import tempfile
import time

_ACTIVE = {}
_CANDIDATES = ((64, 256, 8), (64, 512, 8), (32, 256, 8), (128, 256, 8), (128, 128, 8), (256, 256, 8))
_SHORT_ROW_TILES = ((8, 512, 8), (16, 512, 8), (32, 512, 8))
# f16 and bf16 are both exact standalone and under target_scope. The bf16 route
# currently feeds this region a product with ~53k non-finite values produced
# upstream (cluster mul), so bf16 stays on the native path until that is fixed.
# TL range pipelining depth for the dsa region kernel. The official XPU guide
# gates this on TRITONXPU_TLE_PIPELINE (set before the first compile); with the
# pass off, tl.range(num_stages=1) behaves exactly like a plain range.
_DSA_STAGES = max(1, int(os.environ.get('FLAGTREE_DSA_STAGES', '1')))
_DTYPES_REDUCE = ('float32', 'float16')
_DTYPES_EW = ('float32', 'float16', 'bfloat16')
_KIND = {'float32': 0, 'float16': 1, 'bfloat16': 2}
_VERSION = 'dsa-v13'
_DEBUG = bool(os.environ.get('FLAGTREE_DSA_DEBUG'))



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


def match(gm, inputs):
    """Return a spec when the region is ``(a*b).sum(-1)`` or ``a.sum(-1)`` over a 2D input."""
    import torch
    import torch.nn.functional as F
    mul_targets = (torch.mul, operator.mul, torch.ops.aten.mul.Tensor)
    sum_targets = (torch.sum, torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default)
    softmax_targets = (torch.softmax, torch.ops.aten._softmax.default, torch.ops.aten.softmax.int)
    layernorm_targets = (F.layer_norm, torch.ops.aten.native_layer_norm.default,
                         torch.ops.aten.layer_norm.default)
    nodes = [n for n in gm.graph.nodes if n.op != 'output']
    if not nodes or not isinstance(inputs, (tuple, list)):
        return None
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    if not tensors or not all(t.dim() == 2 for t in tensors[:1]):
        return None
    # Never steal a softmax/layer_norm region: their internal sum is not a row
    # sum of the input, and matching it would silently return the wrong tensor.
    if any(_is(n, softmax_targets, 'softmax') or _is(n, layernorm_targets, 'layer_norm')
           for n in nodes if n.op in ('call_function', 'call_method')):
        return None
    a = tensors[0]
    if len(tensors) > 1:
        b = tensors[1]
        if b.device != a.device or b.dtype != a.dtype or not b.is_contiguous():
            return None
    allowed = (set(_DTYPES_REDUCE) | set(_DTYPES_EW)
               | set(filter(None, os.environ.get('FLAGTREE_DSA_ALLOW', '').split(','))))
    # Reject layouts the dsa kernels do not implement: the copy uses stride n
    # and a contiguous row pitch, so non-contiguous / zero-stride inputs would
    # silently reduce the wrong columns (official copy guide). The second
    # (broadcast) operand is validated below once it is known.
    if not a.is_contiguous() or int(a.stride(-1)) != 1:
        return None
    if a.dtype is None or str(a.dtype).replace('torch.', '') not in allowed:
        return None
    # Elementwise region: mul(a, b) with no reduction.
    ew_node = None
    for node in nodes:
        if node.op in ('call_function', 'call_method') and _is(node, mul_targets, 'mul'):
            ew_node = node
            break
    mul_node = sum_node = None
    for node in nodes:
        if node.op in ('call_function', 'call_method'):
            if mul_node is None and _is(node, mul_targets, 'mul'):
                mul_node = node
            elif _is(node, sum_targets, 'sum'):
                sum_node = node
    keepdim = False
    if sum_node is not None:
        kd = sum_node.kwargs.get('keepdim')
        if kd is None and len(sum_node.args) > 2:
            kd = sum_node.args[2]
        keepdim = bool(kd)
    outputs = _output_nodes(gm)
    if sum_node is None:
        if ew_node is None or outputs != {ew_node}:
            return None
        if not _placeholder_operands(ew_node):
            return None
        m, n = int(a.shape[0]), int(a.shape[1])
        others = [t for t in tensors[1:] if isinstance(t, torch.Tensor)]
        if not others or m <= 0 or n <= 0:
            return None
        bshape = tuple(int(x) for x in others[0].shape)
        if bshape not in ((n,), (m, n), (1, n)):
            return None
        if str(a.dtype).replace('torch.', '') not in _DTYPES_EW:
            return None
        return dict(kind='ew', m=m, n=n, dtype=str(a.dtype).replace('torch.', ''), s_row=n,
                    b_row=1 if bshape == (m, n) else 0, tensor_count=2)
    # The sum must be the region output and reduce the last axis; an internal
    # sum (e.g. softmax/mean denominator) is not a row sum of the input.
    if sum_node.kwargs.get('dtype') is not None or len(sum_node.args) > 3 or len(outputs) != 1:
        return None
    if sum_node not in outputs or not _is_last_dim(_arg(sum_node, 'dim', 1)):
        return None
    kind = 'rowsum'
    if mul_node is not None:
        if sum_node.args[0] is not mul_node:
            return None
        if not _placeholder_operands(mul_node):
            return None
        if len(tensors) != 2 or tuple(b.shape) not in ((a.shape[1],), (1, a.shape[1])):
            return None
        kind = 'mulsum'
    elif not (sum_node.args and isinstance(sum_node.args[0], torch.fx.Node)
              and sum_node.args[0].op == 'placeholder'):
        return None
    m, n = int(a.shape[0]), int(a.shape[1])
    if m <= 0 or n <= 0:
        return None
    # Every candidate row tile divides m; when none does (e.g. m=16) the
    # autotune would exhaust all tiles and raise "no correct tile" at run time,
    # which kills the whole region instead of falling back. Skip here instead.
    if any(m % r == 0 for r, _, _ in _CANDIDATES) is False:
        return None
    if os.environ.get('FLAGTREE_DSA_DEBUG'):
        print('[dsa] matched', kind, m, n, a.dtype, flush=True)
    return dict(kind=kind, m=m, n=n, dtype=str(a.dtype).replace('torch.', ''), s_row=n,
                tensor_count=2 if kind == 'mulsum' else 1, keepdim=keepdim)


def _is(node, targets, name):
    target = getattr(node, 'target', None)
    if target is not None and target in targets:
        return True
    node_name = getattr(node, 'name', '')
    return node.op == 'call_method' and (node_name == name or node_name.rsplit('_', 1)[0] == name)


def _candidate_list(n, *, short_rows=False):
    """Only divisible column tiles: the staging buffer is statically full width."""
    candidates = _CANDIDATES + _SHORT_ROW_TILES if short_rows else _CANDIDATES
    return [c for c in candidates if n % c[1] == 0]


def _cache_path(key):
    root = (os.environ.get('FLAGTREE_DSA_CACHE') or os.environ.get('TRITON_CACHE_DIR')
            or os.path.join(tempfile.gettempdir(), 'flagtree-dsa-autotune'))
    path = os.path.join(root, 'flagtree-dsa-autotune')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, hashlib.sha256(key.encode()).hexdigest()[:32] + '.json')


def _load(key):
    try:
        with open(_cache_path(key)) as fh:
            data = json.load(fh)
        if data.get('version') == _VERSION:
            return data
    except (OSError, ValueError):
        pass
    return None


def _store(key, data):
    data = dict(data, version=_VERSION, measured_at=time.time())
    try:
        with open(_cache_path(key), 'w') as fh:
            json.dump(data, fh, indent=1)
    except OSError:
        pass


def _ew_kernel():
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr'])
    def _dsa_ew_kernel(a_ptr, b_ptr, o_ptr, N: tl.constexpr, ROWS: tl.constexpr,
                       COLS: tl.constexpr, BROW: tl.constexpr, KIND: tl.constexpr):
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        c = tl.program_id(1) * COLS + tl.arange(0, COLS)
        idx = r[:, None] * N + c[None, :]
        if KIND == 0:
            buf = tle.dsa.alloc([ROWS, COLS], tl.float32, tle.dsa.UNI_SRAM)
            tle.dsa.copy(a_ptr + idx, buf, sizes=[ROWS, COLS])
            v = tle.dsa.to_tensor(buf)
            if BROW:
                other = tl.load(b_ptr + idx)
            else:
                other = tl.broadcast_to(tl.load(b_ptr + c)[None, :], [ROWS, COLS])
            tle.dsa.copy(v * other, o_ptr + idx, sizes=[ROWS, COLS])
        else:
            buf = tle.dsa.alloc([ROWS, COLS], tl.int16, tle.dsa.UNI_SRAM)
            tle.dsa.copy(a_ptr + idx, buf, sizes=[ROWS, COLS])
            bits = tle.dsa.to_tensor(buf)
            if KIND == 1:
                v = bits.to(tl.float16, bitcast=True).to(tl.float32)
            else:
                v = (bits.to(tl.uint16).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
            if BROW:
                braw = tl.load(b_ptr + idx)
            else:
                braw = tl.broadcast_to(tl.load(b_ptr + c)[None, :], [ROWS, COLS])
            if KIND == 1:
                other = braw.to(tl.float16, bitcast=True).to(tl.float32)
                res = (v * other).to(tl.float16).to(tl.int16, bitcast=True)
            else:
                other = (braw.to(tl.uint16).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
                # f32 -> bf16 with round-to-nearest-even, written as bit math
                # because the backend rejects a 32->16 bitcast on this path.
                u = (v * other).to(tl.uint32, bitcast=True)
                u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
                res = u.to(tl.uint16).to(tl.int16, bitcast=True)
            tle.dsa.copy(res, o_ptr + idx, sizes=[ROWS, COLS])

    return _dsa_ew_kernel


def _kernel():
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr', 'rows', 'cols', 's_row'])
    def _dsa_region_kernel(a_ptr, b_ptr, o_ptr, rows, cols, s_row, MUL: tl.constexpr,
                           ROWS: tl.constexpr, COLS: tl.constexpr, KIND: tl.constexpr,
                           STAGES: tl.constexpr):
        pid = tl.program_id(0)
        row_blocks = tl.cdiv(rows, ROWS)
        row0 = (pid % row_blocks) * ROWS
        r = row0 + tl.arange(0, ROWS)
        row_tail = tl.minimum(rows - row0, ROWS)
        acc = tl.zeros([ROWS, COLS], tl.float32)
        if KIND == 0:
            buf = tle.dsa.alloc([ROWS, COLS], tl.float32, tle.dsa.UNI_SRAM)
            for i in tl.range(0, cols // COLS, num_stages=STAGES):
                c = i * COLS + tl.arange(0, COLS)
                tle.dsa.copy(a_ptr + r[:, None] * s_row + c[None, :], buf, sizes=[row_tail, COLS])
                v = tle.dsa.to_tensor(buf)
                if MUL:
                    v = v * tl.load(b_ptr + c)[None, :]
                acc += v
        else:
            # Low precision: move the 16-bit bit pattern through an int16 staging
            # buffer (the dsa buffer rejects bf16 outright) and widen in registers.
            buf = tle.dsa.alloc([ROWS, COLS], tl.int16, tle.dsa.UNI_SRAM)
            for i in tl.range(0, cols // COLS, num_stages=STAGES):
                c = i * COLS + tl.arange(0, COLS)
                tle.dsa.copy(a_ptr + r[:, None] * s_row + c[None, :], buf, sizes=[row_tail, COLS])
                bits = tle.dsa.to_tensor(buf)
                if KIND == 1:
                    v = bits.to(tl.float16, bitcast=True).to(tl.float32)
                else:
                    # bf16 is the high half of f32: shift the 16 bits into place.
                    v = (bits.to(tl.uint16).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
                if MUL:
                    bb = tl.load(b_ptr + c)
                    if KIND == 1:
                        bb = bb.to(tl.float16, bitcast=True).to(tl.float32)
                    else:
                        bb = (bb.to(tl.uint16).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
                    v = v * bb[None, :]
                acc += v
        part = tl.sum(acc, axis=1)[:, None]
        tl.store(o_ptr + r[:, None], part, mask=r[:, None] < rows)

    return _dsa_region_kernel


def _packed_kernel():
    """f16 as two 16-bit lanes in one int32 word: 4-byte DMA instead of 2-byte."""
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr', 'rows', 'cols', 's_row'])
    def _dsa_packed_kernel(a_ptr, b_ptr, o_ptr, rows, cols, s_row, MUL: tl.constexpr,
                           ROWS: tl.constexpr, COLS: tl.constexpr):
        W: tl.constexpr = COLS // 2
        pid = tl.program_id(0)
        row_blocks = tl.cdiv(rows, ROWS)
        row0 = (pid % row_blocks) * ROWS
        r = row0 + tl.arange(0, ROWS)
        row_tail = tl.minimum(rows - row0, ROWS)
        acc = tl.zeros([ROWS, W], tl.float32)
        # The f16 tensor is viewed as packed f32 words (two f16 lanes per word),
        # so the DSA buffer matches the source element type (float32) and the
        # integer bit operations bitcast the loaded words to i32 first.
        buf = tle.dsa.alloc([ROWS, W], tl.float32, tle.dsa.UNI_SRAM)
        for i in range(0, cols // COLS):
            cw = i * W + tl.arange(0, W)
            tle.dsa.copy(a_ptr + r[:, None] * s_row + cw[None, :], buf, sizes=[row_tail, W])
            w = tle.dsa.to_tensor(buf).to(tl.int32, bitcast=True)
            lo = (w & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            hi = (w >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            if MUL:
                bw = tl.load(b_ptr + cw).to(tl.int32, bitcast=True)
                blo = (bw & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                bhi = (bw >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                lo = lo * blo
                hi = hi * bhi
            acc += lo + hi
        part = tl.sum(acc, axis=1)[:, None]
        tl.store(o_ptr + r[:, None], part, mask=r[:, None] < rows)

    return _dsa_packed_kernel


def _plain_kernel():
    """Plain tl.load kernel: the low-precision path (dsa DMA is fast but its register
    level 16-bit conversion is scalarized by the backend, so dsa only pays off on f32)."""
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr'])
    def _dsa_plain_kernel(a_ptr, b_ptr, o_ptr, MUL: tl.constexpr, ROWS: tl.constexpr,
                          COLS: tl.constexpr, N: tl.constexpr, ROUND: tl.constexpr):
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        acc = tl.zeros([ROWS, COLS], tl.float32)
        for off in range(0, N, COLS):
            c = off + tl.arange(0, COLS)
            va = tl.load(a_ptr + r[:, None] * N + c[None, :]).to(tl.float32)
            if MUL:
                vb = tl.load(b_ptr + c).to(tl.float32)
                p = va * vb[None, :]
            else:
                p = va
            if ROUND == 1:
                p = p.to(tl.float16).to(tl.float32)
            if ROUND == 2:
                bits = p.to(tl.uint32, bitcast=True)
                rnd = bits + 0x7FFF + ((bits >> 16) & 1)
                p = ((rnd >> 16) << 16).to(tl.float32, bitcast=True)
            acc += p
        tl.store(o_ptr + r[:, None], tl.sum(acc, 1)[:, None])

    return _dsa_plain_kernel


def _view(tensor, dtype):
    import torch
    if dtype == 'float32':
        return tensor
    if dtype == 'float16':
        # packed: two f16 lanes per int32 word
        return tensor.view(torch.float32)
    return tensor.view(torch.int16)


def _word_stride(spec):
    return spec['n'] // 2 if spec['dtype'] == 'float16' else spec['n']


def _is_plain(spec):
    # NOTE: the f16 packed-DMA kernel (_packed_kernel) now compiles and is
    # correct, but its current tiling measures ~3 ms on 2048x2048 (DMA slices
    # too small), so f16 stays on the plain kernel until that is re-tiled.
    return spec['dtype'] != 'float32'


def _plain_round(spec):
    if spec['dtype'] == 'float16':
        return int(os.environ.get('FLAGTREE_DSA_PLAIN_ROUND', '1'))
    if spec['dtype'] == 'bfloat16':
        return 0 if os.environ.get('FLAGTREE_BF16_PLAIN_NO_RNE') else 2
    return 0


def _plain_candidates(n):
    return [c for c in ((64, 128, 4), (64, 128, 8), (64, 256, 8), (32, 256, 8),
                        (128, 128, 8), (64, 512, 8), (128, 256, 8)) if n % c[1] == 0]


def _kernel_for(spec):
    if _is_plain(spec):
        return _plain_kernel()
    return _packed_kernel() if spec['dtype'] == 'float16' else _kernel()


def _measure(kernel, spec, rows, cols, warps, tensors, packed=None, *, replay=False):
    import torch
    if packed is None:
        packed = spec['dtype'] == 'float16'
    if spec['kind'] == 'ew':
        return _measure_ew(kernel, spec, rows, cols, warps, tensors)
    if _is_plain(spec):
        a, b = tensors[0], (tensors[1] if len(tensors) > 1 else None)
        m, n = spec['m'], spec['n']
        out = torch.zeros((m,), device=a.device, dtype=a.dtype)
        grid = ((m + rows - 1) // rows,)
        mul = 1 if spec['kind'] == 'mulsum' else 0
        rnd = _plain_round(spec)

        if m % rows != 0:
            return None

        def prun():
            kernel[grid](a, b, out, MUL=mul, ROWS=rows, COLS=cols, N=n, ROUND=rnd,
                         num_warps=warps, num_stages=1)

        prun()
        torch.cuda.synchronize()
        ref = ((a * b) if mul else a).sum(-1)
        diff = (out - ref).abs()
        ratio = float((diff / ref.abs().clamp_min(1.0)).max())
        if os.environ.get('FLAGTREE_DSA_DEBUG'):
            print('[plain] tile', rows, cols, warps, 'max_abs', float(diff.max()), 'ratio', ratio,
                  'finite', bool(torch.isfinite(out).all()), flush=True)
        if not bool(torch.isfinite(out).all()) or ratio > 5e-3:
            return None
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        samples = []
        for _ in range(5):
            start.record()
            for _ in range(20):
                prun()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 20)
        samples.sort()
        return samples[len(samples) // 2]
    a, b = tensors[0], (tensors[1] if len(tensors) > 1 else None)
    m, n = spec['m'], spec['n']
    out = torch.zeros((m,), device=a.device, dtype=a.dtype)
    grid = ((m + rows - 1) // rows,)
    mul = 1 if spec['kind'] == 'mulsum' else 0
    if os.environ.get('FLAGTREE_DSA_DEBUG'):
        print('[dsa] input', a.shape, a.stride(), a.dtype, 'contig', a.is_contiguous(),
              'align16', a.data_ptr() % 16, 'type', type(a).__name__, 'device', a.device,
              'ptr', a.data_ptr(), flush=True)
    a_v = _view(a, spec['dtype'])
    b_v = _view(b, spec['dtype']) if b is not None else None

    def run():
        if packed:
            kernel[grid](a_v, b_v, out, m, n, _word_stride(spec), MUL=mul, ROWS=rows, COLS=cols,
                         num_warps=warps, num_stages=2, is_sdnn=True)
        else:
            kernel[grid](a_v, b_v, out, m, n, n, MUL=mul, ROWS=rows, COLS=cols, KIND=_KIND[spec['dtype']], STAGES=_DSA_STAGES,
                         num_warps=warps, num_stages=2, is_sdnn=True)

    run()
    torch.cuda.synchronize()
    ref = ((a.float() * b.float()) if mul else a.float()).sum(-1)
    diff = (out - ref).abs()
    scale = ref.abs().clamp_min(1.0)
    ratio = float((diff / scale).max())
    if os.environ.get('FLAGTREE_DSA_DEBUG'):
        print('[dsa] tile', rows, cols, warps, 'max_abs', float(diff.max()), 'ratio', ratio,
              'finite', bool(torch.isfinite(out).all()),
              'out', [round(float(x), 3) for x in out[:3]],
              'ref', [round(float(x), 3) for x in ref[:3]],
              'in', [round(float(x), 3) for x in a[0, :3].float()],
              'in_nonfinite', int((~torch.isfinite(a)).sum()), flush=True)
    if not bool(torch.isfinite(out).all()) or ratio > 5e-3:
        return None
    if replay:
        from ._resource_benchmark import _capture, replay_us
        import statistics
        # A Python launch takes longer than these kernels. Time the same
        # warmed launches in a device graph so tile ranking sees device work.
        calls = 32
        graph, _ = _capture(run, (), calls)
        return statistics.median(replay_us(graph, calls) for _ in range(5))
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(20):
            run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    samples.sort()
    return samples[len(samples) // 2]


def _measure_ew(kernel, spec, rows, cols, warps, tensors):
    import torch
    if len(tensors) < 2:
        return None
    a, b = tensors[0], tensors[1]
    m, n = spec['m'], spec['n']
    out = torch.empty_like(a)
    grid = (m // rows, n // cols)
    brow = 1 if spec.get('b_row') else 0
    av, bv, ov = _view(a, spec['dtype']), _view(b, spec['dtype']), _view(out, spec['dtype'])

    def run():
        kernel[grid](av, bv, ov, N=n, ROWS=rows, COLS=cols, BROW=brow,
                     KIND=_KIND[spec['dtype']], num_warps=warps, num_stages=2, is_sdnn=True)

    run()
    torch.cuda.synchronize()
    ref = a * b
    d = (out.float() - ref.float()).abs()
    if not bool(torch.isfinite(out).all()) or float(d.max()) > 1e-2:
        return None
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(20):
            run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    samples.sort()
    return samples[len(samples) // 2]


def _fp32_mulsum_launcher(kernel, grid, m, n, jit_launch, proof):
    """Reuse the SDNN executable for this plan, with live buffers and stream."""
    import sys
    import torch
    from triton import knobs
    from triton.runtime import driver

    active = driver.active
    device = active.get_current_device()
    jit = sys.modules[type(kernel).__module__]
    prism = getattr(jit, '_flagprism', None)
    distributed = getattr(jit, 'DistributedRtContext', None)
    runtime = knobs.runtime
    signature = ('*fp32',) * 3 + ('i32',) * 3 + ('constexpr',) * 5
    bound = None

    def launch(a, b, out):
        nonlocal bound
        enter, leave = runtime.launch_enter_hook, runtime.launch_exit_hook
        if (type(a) is not torch.Tensor or type(b) is not torch.Tensor
                or a.dtype != torch.float32 or b.dtype != torch.float32
                or a.device != out.device or b.device != a.device or a.device.index != device
                or active.get_current_device() != device
                or kernel.pre_run_hooks or kernel.used_global_vals or kernel.debug or runtime.debug
                or getattr(runtime, 'disable_launch_fast_path', False) or knobs.compilation.instrumentation_mode
                or getattr(enter, 'calls', enter)
                or getattr(leave, 'calls', leave)
                or (prism is not None and prism._registered_components())
                or (distributed is not None and distributed._init_count and distributed().is_lite_mode)):
            return jit_launch(a, b, out)
        a_ptr, b_ptr, out_ptr = a.data_ptr(), b.data_ptr(), out.data_ptr()
        if (a_ptr | b_ptr | out_ptr) & 15:
            return jit_launch(a, b, out)
        if bound is None:
            binary = jit_launch(a, b, out)
            runner = binary.run
            if (type(runner).__module__ == 'triton.backends.xpu.driver' and binary.metadata.is_sdnn
                    and getattr(binary.metadata, 'name', None) == '_dsa_region_kernel'
                    and tuple(getattr(runner, '_signature', {}).values()) == signature
                    and not getattr(sys.modules[type(runner).__module__], '_PRINT_TRITON_FUNC', False)
                    and hasattr(runner, 'launch')):
                # On KL3, only scaled arguments consume a launch-time numel.
                raw = binary.metadata.xpu_arch == 3 and getattr(binary.metadata, 'tensor_args', None) in ((), [])
                bound = binary, runner.launch, binary.function, binary.packed_metadata, raw
                proof['prepared_launch'] = 'sdnn_fp32_mulsum'
                proof['launch_pointer_args'] = 'address' if raw else 'tensor'
            return binary
        binary, entry, function, metadata, raw = bound
        if raw:
            a, b, out = a_ptr, b_ptr, out_ptr
        entry(grid[0], 1, 1, active.get_current_stream(device), function, metadata,
              None, None, None, a, b, out, m, n, n)
        return binary

    return launch


def compile(spec, tensors):
    """Return (callable(xs)->tuple, proof). Tuning happens on the first real call."""
    from ._resource_codegen import bind_evidence

    key = '%s|%s|%dx%d|%s|kd%d' % (_VERSION, spec['kind'], spec['m'], spec['n'], spec['dtype'],
                                    1 if spec.get('keepdim') else 0)
    resource = 'cluster' if spec['kind'] != 'ew' and _is_plain(spec) else 'sdnn'
    short_rows = (resource == 'sdnn' and spec['kind'] == 'mulsum' and spec['dtype'] == 'float32'
                  and spec['m'] <= 128 and spec['n'] % 512 == 0)
    if short_rows:
        # A persisted winner from the narrower search never saw 8-row tiles.
        key += '|short-rows-v1'
    proof = {
        'route': 'plain_reduce' if resource == 'cluster' else 'sdnn_dsa',
        'resource': resource, 'resource_verified': False,
        'direct_launches': 1, 'emitted': 'tle.dsa', 'region': dict(spec),
        'match_key': key, 'tiles': None, 'autotune': 'cached' if _load(key) else 'pending',
    }
    state = dict(kernel=None, tile=None)

    def prepare(a, b):
        tile = state['tile']
        if tile is not None:
            return state['kernel'], tile
        tile = _load(key)
        if tile is None:
            kernel = _ew_kernel() if spec['kind'] == 'ew' else _kernel_for(spec)
            best = None
            candidates = (_plain_candidates(spec['n']) if _is_plain(spec)
                          else _candidate_list(spec['n'], short_rows=short_rows))
            for rows, cols, warps in candidates:
                try:
                    us = _measure(kernel, spec, rows, cols, warps,
                                  (a, b) if b is not None else (a,),
                                  packed=(spec['kind'] != 'ew' and spec['dtype'] == 'float16'
                                          and os.environ.get('FLAGTREE_DSA_PACKED', '0') == '1'), replay=short_rows)
                except Exception as exc:
                    if os.environ.get('FLAGTREE_DSA_DEBUG'):
                        import traceback
                        print('[dsa] tile', rows, cols, warps, 'failed:', type(exc).__name__, str(exc)[:300], flush=True)
                        traceback.print_exc()
                    continue
                if us is not None and (best is None or us < best[0]):
                    best = (us, rows, cols, warps)
            if best is None:
                raise RuntimeError('flagtree dsa region: no correct tile for %r' % (spec,))
            tile = dict(rows=best[1], cols=best[2], warps=best[3], us=round(best[0], 2))
            _store(key, tile)
        state['tile'] = tile
        if state['kernel'] is None:
            state['kernel'] = _ew_kernel() if spec['kind'] == 'ew' else _kernel_for(spec)
        proof['tiles'] = dict(tile)
        proof['autotune'] = 'cached' if tile.get('version') == _VERSION else 'measured'
        return state['kernel'], tile

    def _make_launcher(a, b):
        kernel, tile = prepare(a, b)
        rows, cols, warps = tile['rows'], tile['cols'], tile['warps']
        m, n = spec['m'], spec['n']
        plain = _is_plain(spec)
        mul = 1 if spec['kind'] == 'mulsum' else 0
        if spec['kind'] == 'ew':
            grid = (m // rows, n // cols)
            brow = 1 if spec.get('b_row') else 0
            kind_id = _KIND[spec['dtype']]
            def launch(a, b, out):
                return kernel[grid](_view(a, spec['dtype']), _view(b, spec['dtype']),
                             _view(out, spec['dtype']), N=n, ROWS=rows, COLS=cols,
                             BROW=brow, KIND=kind_id, num_warps=warps, num_stages=2,
                             is_sdnn=True)
            return launch
        grid = ((m + rows - 1) // rows,)
        if plain:
            rnd = _plain_round(spec)
            def launch(a, b, out):
                return kernel[grid](a, b, out, MUL=mul, ROWS=rows, COLS=cols, N=n, ROUND=rnd,
                             num_warps=warps, num_stages=1)
            return launch
        if spec['dtype'] == 'float16':
            stride = _word_stride(spec)
            def launch(a, b, out):
                return kernel[grid](_view(a, spec['dtype']), _view(b, spec['dtype']), out, m, n, stride,
                             MUL=mul, ROWS=rows, COLS=cols, num_warps=warps, num_stages=2,
                             is_sdnn=True)
            return launch
        knd = _KIND[spec['dtype']]
        def launch(a, b, out):
            return kernel[grid](_view(a, spec['dtype']),
                         _view(b, spec['dtype']) if b is not None else None,
                         out, m, n, n, MUL=mul, ROWS=rows, COLS=cols, KIND=knd, STAGES=_DSA_STAGES,
                         num_warps=warps, num_stages=2, is_sdnn=True)
        if mul and spec['dtype'] == 'float32':
            return _fp32_mulsum_launcher(kernel, grid, m, n, launch, proof)
        return launch

    def run(xs):
        import torch
        a = xs[0]
        launcher = state.get('launcher')
        if launcher is None:
            if a.dim() != 2 or a.shape[0] != spec['m'] or a.shape[1] != spec['n']:
                raise RuntimeError('flagtree dsa region: shape changed at call time')
            b = xs[1] if spec['kind'] in ('mulsum', 'ew') else None
            state['out_shape'] = (spec['m'], 1) if spec.get('keepdim') else (spec['m'],)
            launcher = state['launcher'] = _make_launcher(a, b)
        if spec['kind'] == 'ew':
            b = xs[1]
            out = torch.empty_like(a)
            binary = launcher(a, b, out)
            if not proof['resource_verified']:
                bind_evidence(proof, binary)
            return (out,)
        out = torch.empty(state['out_shape'], device=a.device, dtype=a.dtype)
        binary = launcher(a, xs[1] if spec['kind'] == 'mulsum' else None, out)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
        if _DEBUG:
            print('[dsa] run', spec['kind'], spec['dtype'], a.shape, flush=True)
        return (out,)

    return run, proof
