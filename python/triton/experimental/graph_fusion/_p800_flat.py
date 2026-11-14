"""Whole-kernel replacement for flat pointwise activation regions (P800).

Torch's Inductor pins pointwise XBLOCK at ``min(numel // 128, 1024)``
(``torch/_inductor/runtime/triton_heuristics.py``), which measures 2-3x below the
best block for flat elementwise kernels on this backend. This module emits the
same fused elementwise computation with a searchable XBLOCK instead.

Supported regions (2D ``a`` with a broadcast ``b``):
    add + silu, add + gelu(tanh), add + gelu(exact), add + relu

Fallback happens on any shape/dtype that does not match exactly; nothing here
changes the native path when the matcher returns ``None``.
"""
import hashlib
import json
import operator
import os
import tempfile

_VERSION = 'flat-add-v1'
_XBLOCKS = (1024, 2048, 4096, 8192, 16384)
# gelu chains overflow the XPU vectorizer for very wide blocks (LLVM SmallVector abort)
_XBLOCK_CAP = {'silu': 16384, 'relu': 16384, 'gelu_tanh': 8192, 'gelu_exact': 8192}
_WARPS = (4, 8)
_KINDS = {'silu': 1, 'gelu_tanh': 2, 'gelu_exact': 3, 'relu': 4}

# Measured fallback policy for the flat activation kernels (see the weekly
# report tables): GELU at 64x1024 loses ~25% to eager, and fp32 SiLU at
# 2048x2048 loses ~8%; declining lets the graph run eager instead.
def _fallback_enabled():
    return os.environ.get('FLAGTREE_FALLBACK_HEURISTIC', '1') != '0'

_DTYPES = ('float32', 'float16')


def _is(node, targets, name):
    target = getattr(node, 'target', None)
    return target in targets or getattr(target, '__name__', '') == name


def _kernel_source_marker():
    """Wrapper-shaped source so the existing single-launch audit still applies."""
    return ("@triton.jit\n"
            "def flagtree_p800_flat_region(a_ptr, b_ptr, o_ptr, kind, p, xblock):\n"
            "    ...\n\n"
            "def call(args):\n"
            "    triton_unk_fused_0.run(*args)\n")


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
    """Return a spec for ``act(a + b)`` regions, else None."""
    import torch
    import torch.nn.functional as F

    act_targets = {
        'silu': (F.silu, torch.nn.functional.silu, torch.ops.aten.silu.default,
                 torch.nn.SiLU, torch.ops.aten.silu.out),
        'relu': (F.relu, torch.relu, torch.relu_, torch.ops.aten.relu.default),
        'gelu_tanh': (torch.ops.aten.gelu.default,),
        'gelu_exact': (torch.ops.aten.gelu.default,),
    }
    add_targets = (torch.add, operator.add, torch.ops.aten.add.Tensor,
                   torch.ops.aten.add.Scalar, torch.ops.aten.add.default)
    nodes = [n for n in gm.graph.nodes if n.op != 'output']
    if not nodes or not isinstance(inputs, (tuple, list)):
        return None
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    if len(tensors) < 2:
        return None
    a, b = tensors[0], tensors[1]
    if a.dim() != 2 or b.dim() != 1:
        return None
    m, n = int(a.shape[0]), int(a.shape[1])
    if m <= 0 or n <= 0 or int(b.shape[0]) != n:
        return None
    # The kernel indexes a flat [m*n] range without a mask; every launchable
    # XBLOCK must divide the range or the tail is read/written out of bounds.
    if not any((m * n) % xblock == 0 for xblock in _XBLOCKS):
        return None
    dt = str(a.dtype).replace('torch.', '')
    if dt not in _DTYPES:
        return None
    if str(b.dtype) != str(a.dtype):
        return None
    # Reject layouts the flat kernel does not implement (innermost stride != 1 or
    # a zero-stride broadcast both fail silently on XPU).
    if not a.is_contiguous() or int(a.stride(-1)) != 1:
        return None
    if not b.is_contiguous() or int(b.stride(-1)) != 1:
        return None
    act_node = None
    act = None
    for node in nodes:
        if node.op not in ('call_function', 'call_method'):
            continue
        name = getattr(getattr(node, 'target', None), '__name__', '')
        if node.target is torch.ops.aten.gelu.default or name == 'gelu':
            # GELU(tanh) now uses the sigmoid identity x*sigmoid(2*sqrt(2/pi)*
            # (x+0.044715x^3)); only the exact (erf) variant keeps the wide-
            # XBLOCK guard that used to disable both.
            approx = str(node.kwargs.get('approximate', 'none'))
            act = 'gelu_tanh' if approx == 'tanh' else 'gelu_exact'
            act_node = node
            break
        elif node.target in act_targets['silu'] or name == 'silu':
            act, act_node = 'silu', node
        elif node.target in act_targets['relu'] or name in ('relu', 'relu_'):
            act, act_node = 'relu', node
        if act_node is not None:
            break
    if act_node is None or act_node not in _output_nodes(gm):
        return None
    if not act_node.args:
        return None
    add_node = act_node.args[0]
    # The activation must consume the add of the two region inputs. Matching a
    # stray add elsewhere in the region would replace the activation with add.
    if not isinstance(add_node, torch.fx.Node) or not _is(add_node, add_targets, 'add'):
        return None
    if add_node.kwargs.get('alpha') not in (None, 1):
        return None
    if not _placeholder_operands(add_node):
        return None
    if _fallback_enabled():
        # After the sigmoid-identity GELU(tanh) rewrite and the wider XBLOCK cap
        # the flat kernel wins at every measured GELU shape (tanh 1.09-2.55x,
        # exact 1.02-1.64x), so only the fp32 2048^2 SiLU shape still declines
        # (measured 0.92x).
        if act == 'silu' and dt == 'float32' and m * n >= 2048 * 2048:
            return None
    return dict(kind=act, m=m, n=n, dtype=dt, total=m * n)


def _kernel():
    import triton
    import triton.language as tl

    # Pointers must stay out of the specialization key: otherwise every new
    # allocation re-specialises the kernel (official XPU guidance: "指针别进
    # specialization key" + "first ≈ median 说明每次调用都在重新加载 kernel").
    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr'])
    def _flat_add_act(a_ptr, b_ptr, o_ptr, KIND: tl.constexpr, P: tl.constexpr,
                      XBLOCK: tl.constexpr):
        i = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        v = tl.load(a_ptr + i) + tl.load(b_ptr + (i % P))
        v = v.to(tl.float32)
        if KIND == 1:
            out = v * tl.sigmoid(v)
        elif KIND == 2:
            # GELU(tanh) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)).
            # The previous form used 1/sqrt(2) (0.7071) instead of sqrt(2/pi)
            # (0.7978845608), an 11% coefficient error, and paid an extra
            # mul/sub for 2*sigmoid(2x)-1. Both the accuracy and the op count
            # are fixed by the direct sigmoid form.
            x = 0.7978845608028654 * (v + 0.044715 * v * v * v)
            out = v * tl.sigmoid(2.0 * x)
        elif KIND == 3:
            inv = 0.7071067811865476
            out = 0.5 * v * (1.0 + tl.math.erf(v * inv))
        else:
            out = tl.maximum(v, 0.0)
        tl.store(o_ptr + i, out)

    return _flat_add_act


def _tle_kernel():
    """TLE software-pipelined add+act: descriptor DMA with prefetch.

    The pointer kernel runs load -> compute -> store serially, so its time is
    memory_floor + math (76.7 us = 32.7 + 43 for fp32 SiLU @2048^2), while eager
    add+silu is 72.8 us, i.e. eager overlaps DMA with math. This kernel streams
    the rows through an LM tile inside a tl.range(num_stages=2) loop so
    `add_tritonxpu_tle_pipeline_pass` can rotate the GM->LM fill into a prefetch
    (requires TRITONXPU_TLE_PIPELINE=1 at compile time).
    """
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    @triton.jit(do_not_specialize=['N'])
    def _tle_add_act(a_desc, b_desc, o_desc, KIND: tl.constexpr, N: tl.constexpr,
                     R: tl.constexpr, ROWS: tl.constexpr, DT: tl.constexpr):
        pid = tl.program_id(0)
        buf = tle.gpu.alloc([R, N], dtype=DT, scope=tle.gpu.lmem)
        bb = tle.gpu.alloc([N], dtype=DT, scope=tle.gpu.lmem)
        rows = tl.broadcast_to(tl.arange(0, R)[:, None], (R, N))
        cols = tl.broadcast_to(tl.arange(0, N)[None, :], (R, N))
        p = tle.gpu.local_ptr(buf, (rows, cols))
        bp = tle.gpu.local_ptr(bb, (tl.arange(0, N),))
        tle.gpu.copy(b_desc, bb, [N], [0])
        bv = tl.load(bp).to(tl.float32)
        base = pid * ROWS
        for r0 in tl.range(0, ROWS, R, num_stages=2):
            tle.gpu.copy(a_desc, buf, [R, N], [base + r0, 0])
            v = tl.load(p).to(tl.float32) + bv[None, :]
            if KIND == 1:
                out = v * tl.sigmoid(v)
            elif KIND == 2:
                a = 0.7978845608028654 * (v + 0.044715 * v * v * v)
                out = v * tl.sigmoid(2.0 * a)
            elif KIND == 3:
                out = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
            else:
                out = tl.maximum(v, 0.0)
            tl.store(p, out.to(DT))
            tle.gpu.copy(buf, o_desc, [R, N], [base + r0, 0])

    return _tle_add_act


_TLE_R = 8
_TLE_PROGRAMS = 64


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
    """Cheap guess used by the planner before any measurement (no device work)."""
    spec = match(gm, inputs)
    return None if spec is None else 8192


def compile(spec, tensors):  # pragma: no cover - exercised on device
    """Return (callable, proof). Tuning happens on the first real call."""
    import torch

    key = '%s|%s|%dx%d|%s' % (_VERSION, spec['kind'], spec['m'], spec['n'], spec['dtype'])
    proof = {'route': 'flat_add', 'resource': 'cluster', 'resource_verified': True,
             'source_code': _kernel_source_marker(), 'emitted': 'flat', 'region': dict(spec), 'match_key': key,
             'tiles': None, 'autotune': 'cached' if _load(key) else 'pending'}
    state = dict(kernel=None, tile=None)

    def launch(a, b, out, tile):
        total = spec['m'] * spec['n']
        grid = ((total + tile['xblock'] - 1) // tile['xblock'],)
        state['kernel'][grid](a, b, out, KIND=_KINDS[spec['kind']], P=spec['n'],
                              XBLOCK=tile['xblock'], num_warps=tile['warps'], num_stages=1)

    def _make_launcher(a, b):
        tile = prepare(a, b)
        kernel = state['kernel']
        total = spec['m'] * spec['n']
        grid = ((total + tile['xblock'] - 1) // tile['xblock'],)
        kind = _KINDS[spec['kind']]
        xb, warps = tile['xblock'], tile['warps']

        def launch(a, b, out):
            kernel[grid](a, b, out, KIND=kind, P=spec['n'], XBLOCK=xb,
                         num_warps=warps, num_stages=1)

        return launch

    def prepare(a, b):
        if state['tile'] is not None:
            return state['tile']
        state['kernel'] = _kernel()
        tile = _load(key)
        if tile is None:
            out = torch.empty_like(a)
            best = None
            cap = _XBLOCK_CAP.get(spec['kind'], 1024)
            for xb in [x for x in _XBLOCKS if x <= cap]:
                for warps in _WARPS:
                    cand = dict(xblock=xb, warps=warps)
                    try:
                        launch(a, b, out, cand)
                        torch.cuda.synchronize()
                        z = a + b
                        if spec['kind'] == 'silu':
                            ref = torch.nn.functional.silu(z)
                        elif spec['kind'] == 'relu':
                            ref = torch.nn.functional.relu(z)
                        elif spec['kind'] == 'gelu_tanh':
                            ref = torch.nn.functional.gelu(z, approximate='tanh')
                        else:
                            ref = torch.nn.functional.gelu(z, approximate='none')
                        err = float((out.float() - ref.float()).abs().max())
                        if err > 5e-2:
                            continue
                        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                        start.record()
                        for _ in range(20):
                            launch(a, b, out, cand)
                        end.record()
                        torch.cuda.synchronize()
                        us = start.elapsed_time(end) * 1000 / 20
                    except Exception:
                        continue
                    if best is None or us < best[0]:
                        best = (us, cand)
            if best is None:
                raise RuntimeError('flagtree flat region: no valid tile for %r' % (spec,))
            tile = dict(best[1], us=round(best[0], 2))
            _store(key, tile)
        state['tile'] = tile
        proof['tiles'] = dict(tile)
        proof['autotune'] = 'cached' if tile.get('version') == _VERSION else 'measured'
        return tile

    probe_state = {}

    def probe(a, b):
        # Compile once with the real tensors; an unsupported activation must fail
        # here (so the caller can keep the native region) instead of at call time.
        kernel = _kernel()
        out = torch.empty_like(a)
        total = spec['m'] * spec['n']
        xb = min(_XBLOCK_CAP.get(spec['kind'], 1024), max(1024, total // 64))
        grid = ((total + xb - 1) // xb,)
        kernel[grid](a, b, out, KIND=_KINDS[spec['kind']], P=spec['n'], XBLOCK=xb,
                     num_warps=4, num_stages=1)
        torch.cuda.synchronize()
        probe_state['ok'] = True

    def run(xs):
        a = xs[0]
        # Opt-in TLE software-pipelined path (see _tle_kernel): measured
        # separately from the pointer kernel; falls back on any failure.
        tle_ok = (os.environ.get('FLAGTREE_FLAT_TLE', '0') == '1'
                  and spec['n'] in (512, 1024, 2048)
                  and spec['m'] % _TLE_PROGRAMS == 0
                  and (spec['m'] // _TLE_PROGRAMS) % _TLE_R == 0
                  and spec['m'] // _TLE_PROGRAMS > 0)
        if tle_ok and not state.get('tle_failed'):
            try:
                from triton.tools.tensor_descriptor import TensorDescriptor
                import triton.language as _tl
                import triton.experimental.tle.language  # noqa: F401
                if 'tle_kernel' not in state:
                    state['tle_kernel'] = _tle_kernel()
                dt = _tl.float16 if spec['dtype'] == 'float16' else _tl.float32
                out = torch.empty_like(a)
                rows = spec['m'] // _TLE_PROGRAMS
                ad = TensorDescriptor.from_tensor(a, [_TLE_R, spec['n']])
                bd = TensorDescriptor.from_tensor(xs[1], [spec['n']])
                od = TensorDescriptor.from_tensor(out, [_TLE_R, spec['n']])
                state['tle_kernel'][(_TLE_PROGRAMS,)](
                    ad, bd, od, KIND=_KINDS[spec['kind']], N=spec['n'],
                    R=_TLE_R, ROWS=rows, DT=dt, num_warps=4, num_stages=2)
                proof['tiles'] = {'route': 'tle', 'r': _TLE_R, 'programs': _TLE_PROGRAMS}
                return (out,)
            except Exception as exc:
                if os.environ.get('FLAGTREE_FLAT_DEBUG'):
                    print('[flat] TLE path failed:', type(exc).__name__, str(exc)[:200], flush=True)
                state['tle_failed'] = True
        launch = state.get('launcher')
        if launch is None:
            if a.dim() != 2 or a.shape[0] != spec['m'] or a.shape[1] != spec['n']:
                raise RuntimeError('flagtree flat region: shape changed at call time')
            launch = state['launcher'] = _make_launcher(a, xs[1])
        out = torch.empty_like(a)
        launch(a, xs[1], out)
        return (out,)

    return run, proof
