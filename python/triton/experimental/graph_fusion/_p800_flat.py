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

_VERSION = 'flat-add-v2'
_XBLOCKS = (1024, 2048, 4096, 8192, 16384)
# gelu chains overflow the XPU vectorizer for very wide blocks (LLVM SmallVector abort)
_XBLOCK_CAP = {'silu': 16384, 'relu': 16384, 'gelu_tanh': 8192, 'gelu_exact': 8192}
_WARPS = (4, 8)
_KINDS = {'silu': 1, 'gelu_tanh': 2, 'gelu_exact': 3, 'relu': 4}

# Size fallback for the measured large FP32 SiLU case.
def _fallback_enabled():
    return os.environ.get('FLAGTREE_FALLBACK_HEURISTIC', '1') != '0'

_TLE_DTYPES = ('float32', 'float16')


def _is(node, targets, name):
    target = getattr(node, 'target', None)
    return target in targets or getattr(target, '__name__', '') == name



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
    if m * n > 0x7fffffff:
        return None
    dt = str(a.dtype).replace('torch.', '')
    if dt not in ('float32', 'float16', 'bfloat16'):
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
    if act_node is None or _output_nodes(gm) != {act_node}:
        return None
    if dt == 'bfloat16' and act in ('relu', 'silu') and m * n > 65536:
        from ._p800_vector_rows import configuration
        if configuration(act, dt, m, n) is None:
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
    if not _placeholder_operands(add_node) or set(add_node.args[:2]) != {n for n in nodes if n.op == 'placeholder'}:
        return None
    if _fallback_enabled():
        # After the sigmoid-identity GELU(tanh) rewrite and the wider XBLOCK cap
        # the flat kernel wins at every measured GELU shape (tanh 1.09-2.55x,
        # exact 1.02-1.64x), so only the fp32 2048^2 SiLU shape still declines
        # (measured 0.92x).
        if act == 'silu' and dt == 'float32' and m * n >= 2048 * 2048:
            from ._p800_vector_rows import configuration
            if configuration(act, dt, m, n) is None:
                return None
    return dict(kind=act, m=m, n=n, dtype=dt, total=m * n)


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bf16_bits(value, FINAL: tl.constexpr = False):
        word = value.to(tl.uint32, bitcast=True)
        upper = (word + 0x7fff + ((word >> 16) & 1)) >> 16
        if FINAL:
            return upper
        return tl.where((word & 0x7fffffff) > 0x7f800000, 0x7fc0, upper)

    # Pointers must stay out of the specialization key: otherwise every new
    # allocation re-specialises the kernel (official XPU guidance: "指针别进
    # specialization key" + "first ≈ median 说明每次调用都在重新加载 kernel").
    @triton.jit(do_not_specialize=['a_ptr', 'b_ptr', 'o_ptr'],
                do_not_specialize_on_alignment=['a_ptr', 'b_ptr', 'o_ptr'])
    def _flat_add_act(a_ptr, b_ptr, o_ptr, KIND: tl.constexpr, P: tl.constexpr,
                      XBLOCK: tl.constexpr, TOTAL: tl.constexpr, FINAL_ONLY: tl.constexpr = False):
        i = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        if a_ptr.dtype.element_ty == tl.bfloat16:
            a_bits = tl.load(a_ptr.to(tl.pointer_type(tl.uint16)) + i, mask=i < TOTAL, other=0)
            b_bits = tl.load(b_ptr.to(tl.pointer_type(tl.uint16)) + (i % P))
            a = (a_bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
            b = (b_bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
            if FINAL_ONLY:
                v = a + b
            else:
                rounded = bf16_bits(a + b)
                if KIND == 4:
                    # The compatibility ReLU preserves negative zero (0x8000).
                    result = tl.where(rounded > 0x8000, 0, rounded).to(tl.uint16)
                    tl.store(o_ptr.to(tl.pointer_type(tl.uint16)) + i, result, mask=i < TOTAL)
                    return
                v = (rounded << 16).to(tl.float32, bitcast=True)
        else:
            if TOTAL % XBLOCK:
                a = tl.load(a_ptr + i, mask=i < TOTAL, other=0.0)
            else:
                a = tl.load(a_ptr + i)
            b = tl.load(b_ptr + (i % P))
            if FINAL_ONLY:
                v = a.to(tl.float32) + b.to(tl.float32)
            else:
                v = (a + b).to(tl.float32)
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
            out = tl.where(v != v, v, tl.maximum(v, 0.0))
        if a_ptr.dtype.element_ty == tl.bfloat16:
            tl.store(o_ptr.to(tl.pointer_type(tl.uint16)) + i, bf16_bits(out, FINAL_ONLY).to(tl.uint16), mask=i < TOTAL)
        elif TOTAL % XBLOCK:
            tl.store(o_ptr + i, out, mask=i < TOTAL)
        else:
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
                out = tl.where(v != v, v, tl.maximum(v, 0.0))
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


def _device_tile_us(launch, a, b, out, candidate):
    from statistics import median
    from ._resource_benchmark import _capture, replay_us

    # Short kernels can finish before Python submits the next launch. Replay
    # a pre-filled queue so tile selection compares the actual executables.
    graph, _ = _capture(lambda: launch(a, b, out, candidate), (), 32)
    return median(replay_us(graph, 32) for _ in range(5))


def suggest_xblock(gm, inputs):
    """Cheap guess used by the planner before any measurement (no device work)."""
    spec = match(gm, inputs)
    return None if spec is None else 8192


def compile(spec, tensors):  # pragma: no cover - exercised on device
    """Return (callable, proof). Tuning happens on the first real call."""
    import torch

    from ._p800_bound_launch import bind
    from ._resource_codegen import bind_evidence

    key = '%s|%s|%dx%d|%s' % (_VERSION, spec['kind'], spec['m'], spec['n'], spec['dtype'])
    final_only = spec.get('final_round_only', False)
    if final_only:
        key += '|fp32-final-only'
    replay_tiles = (spec['kind'] == 'gelu_tanh' and spec['dtype'] == 'float16' and spec['total'] <= 65536
                    or spec['dtype'] == 'float32' and spec['kind'] in ('silu', 'gelu_exact')
                    and spec['total'] <= 512 * 1024)
    if replay_tiles:
        key += '|device-tile-timing-v1'
    if spec['kind'] == 'relu' and spec['dtype'] == 'bfloat16':
        key += '|preserve-negative-zero'
    proof = {'route': 'flat_add', 'resource': 'cluster', 'resource_verified': False,
             'direct_launches': 1, 'emitted': 'flat', 'region': dict(spec), 'match_key': key,
             'tiles': None, 'autotune': 'cached' if _load(key) else 'pending'}
    from . import _p800_vector_rows
    config = _p800_vector_rows.configuration(spec['kind'], spec['dtype'], spec['m'], spec['n'])
    if (final_only and spec['dtype'] == 'bfloat16' and spec['kind'] in ('silu', 'gelu_tanh')
            and 0 < spec['n'] <= 1024 and spec['n'] % 32):
        from ._p800_layernorm import toolchain_available
        if toolchain_available():
            mode = 10 if spec['kind'] == 'silu' else 11
            return _p800_vector_rows.prepare(spec, (mode, 32, (spec['m'] + 11) // 12, 12), proof)
    if final_only and (spec['dtype'] == 'bfloat16' and spec['kind'] in ('silu', 'gelu_tanh')
                       or spec['dtype'] == 'float16' and spec['kind'] == 'silu' and spec['m'] >= 512):
        from ._p800_layernorm import toolchain_available
        width = 64 if spec['total'] <= 128*1024 else 128 if spec['total'] <= 1024*1024 else 512
        # Mixed activation reuses one aligned bias slice on successive tiles.
        while width >= 32 and (spec['n'] % width or (12 * 64 * width) % spec['n']):
            width //= 2
        if width >= 32 and toolchain_available():
            mode = 2 if spec['kind'] == 'silu' else 3
            return _p800_vector_rows.prepare(spec, (mode, width, 1, 12), proof)
    if config is not None and not final_only:
        return _p800_vector_rows.prepare(spec, config, proof)
    if (spec['kind'] == 'gelu_exact' and not spec.get('pointer_gelu')
            and (spec['dtype'] == 'bfloat16' or final_only and spec['dtype'] == 'float16')):
        from ._p800_layernorm import toolchain_available
        if toolchain_available():
            from . import _p800_bf16_gelu
            width = _p800_bf16_gelu.tile_width(spec['m'], spec['n'])
            if width is not None:
                return _p800_bf16_gelu.prepare(spec, width, proof)
    state = dict(kernel=None, tile=None)

    def launch(a, b, out, tile):
        total = spec['m'] * spec['n']
        grid = ((total + tile['xblock'] - 1) // tile['xblock'],)
        state['kernel'][grid](a, b, out, KIND=_KINDS[spec['kind']], P=spec['n'], TOTAL=total,
                              XBLOCK=tile['xblock'], FINAL_ONLY=final_only, num_warps=tile['warps'], num_stages=1)

    def _make_launcher(a, b):
        tile = prepare(a, b)
        kernel = state['kernel']
        total = spec['m'] * spec['n']
        grid = ((total + tile['xblock'] - 1) // tile['xblock'],)
        kind = _KINDS[spec['kind']]
        xb, warps = tile['xblock'], tile['warps']

        from triton import knobs
        from triton.runtime import driver
        import sys
        runtime_jit = sys.modules[type(kernel).__module__]
        prism = getattr(runtime_jit, '_flagprism', None)
        dtype = a.dtype
        device = driver.active.get_current_device()
        get_device = driver.active.get_current_device
        get_stream = driver.active.get_current_stream
        compiled = None
        runner = None

        def launch(a, b, out):
            nonlocal compiled, runner
            if (runner is not None and not kernel.pre_run_hooks and not knobs.runtime.debug
                    and not getattr(knobs.compilation, 'instrumentation_mode', '')
                    and (prism is None or not prism._registered_components())
                    and a.dtype == b.dtype == dtype and get_device() == device
                    and a.device.index == device and b.device == a.device):
                # CompiledKernel expects constexprs in signature order. Stream
                # selection remains per call, including during graph capture.
                runner(a, b, out, kind, spec['n'], xb, total, final_only, stream=get_stream(device))
                return compiled
            binary = kernel[grid](a, b, out, KIND=kind, P=spec['n'], XBLOCK=xb, TOTAL=total,
                                  FINAL_ONLY=final_only, num_warps=warps, num_stages=1)
            if runner is None and a.device.index == device and b.device == a.device and get_device() == device:
                compiled = binary
                runner = binary[(grid[0], 1, 1)]
            return binary

        return launch

    def prepare(a, b):
        if state['tile'] is not None:
            return state['tile']
        state['kernel'] = _kernel()
        tile = _load(key)
        cap = _XBLOCK_CAP.get(spec['kind'], 1024)
        blocks = [xb for xb in _XBLOCKS if xb <= cap]
        if tile is None or tile.get('xblock') not in blocks:
            out = torch.empty_like(a)
            best = None
            best_device = None
            for xb in blocks:
                for warps in _WARPS:
                    cand = dict(xblock=xb, warps=warps)
                    try:
                        launch(a, b, out, cand)
                        torch.cuda.synchronize()
                        z = a.float() + b.float() if final_only else a + b
                        if spec['kind'] == 'silu':
                            ref = torch.nn.functional.silu(z)
                        elif spec['kind'] == 'relu':
                            ref = torch.nn.functional.relu(z)
                        elif spec['kind'] == 'gelu_tanh':
                            ref = torch.nn.functional.gelu(z, approximate='tanh')
                        else:
                            ref = torch.nn.functional.gelu(z, approximate='none')
                        err = float((out.float() - ref.to(out.dtype).float()).abs().max())
                        if err > 5e-2:
                            continue
                        if replay_tiles:
                            try:
                                device_us = _device_tile_us(launch, a, b, out, cand)
                            except RuntimeError:
                                pass
                            else:
                                if best_device is None or device_us < best_device[0]:
                                    best_device = (device_us, cand)
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
            best = best_device or best
            if best is None:
                raise RuntimeError('flagtree flat region: no valid tile for %r' % (spec,))
            tile = dict(best[1], us=round(best[0], 2))
            if replay_tiles:
                tile['timing'] = 'device_graph' if best_device is not None else 'launch_loop_fallback'
            if not replay_tiles or best_device is not None:
                _store(key, tile)
        state['tile'] = tile
        proof['tiles'] = dict(tile)
        proof['autotune'] = 'cached' if tile.get('version') == _VERSION else 'measured'
        return tile

    tle_layout = (spec['dtype'] in _TLE_DTYPES and spec['n'] in (512, 1024, 2048)
                  and spec['m'] % _TLE_PROGRAMS == 0 and spec['m'] // _TLE_PROGRAMS > 0
                  and (spec['m'] // _TLE_PROGRAMS) % _TLE_R == 0)

    def run(xs):
        a = xs[0]
        # Opt-in TLE software-pipelined path (see _tle_kernel): measured
        # separately from the pointer kernel; falls back on any failure.
        if tle_layout and not state.get('tle_failed') and os.environ.get('FLAGTREE_FLAT_TLE', '0') == '1':
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
                binary = state['tle_kernel'][(_TLE_PROGRAMS,)](
                    ad, bd, od, KIND=_KINDS[spec['kind']], N=spec['n'],
                    R=_TLE_R, ROWS=rows, DT=dt, num_warps=4, num_stages=2)
                bind_evidence(proof, binary)
                proof['tiles'] = {'route': 'tle', 'r': _TLE_R, 'programs': _TLE_PROGRAMS}
                return (out,)
            except Exception as exc:
                if os.environ.get('FLAGTREE_FLAT_DEBUG'):
                    print('[flat] TLE path failed:', type(exc).__name__, str(exc)[:200], flush=True)
                proof['resource_verified'] = False
                state['tle_failed'] = True
        warm = state.get('warm')
        if warm is not None:
            out = warm(a, xs[1])
            if out is not None:
                return (out,)
        launch = state.get('launcher')
        if launch is None:
            if a.dim() != 2 or a.shape[0] != spec['m'] or a.shape[1] != spec['n']:
                raise RuntimeError('flagtree flat region: shape changed at call time')
            launch = state['launcher'] = _make_launcher(a, xs[1])
        out = torch.empty_like(a)
        binary = launch(a, xs[1], out)
        if not proof['resource_verified']:
            bind_evidence(proof, binary)
            xb = state['tile']['xblock']
            programs = (spec['total'] + xb - 1) // xb
            state['warm'] = bind(state['kernel'], binary, (programs, 1, 1), (a, xs[1]),
                                 (spec['m'], spec['n']), (), 'cluster')
            proof['prepared_launch'] = state['warm'] is not None
        return (out,)

    return run, proof
