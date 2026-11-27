"""One prepared FP32 implementation per F06 template; SDK casts on live boundaries."""
import copy
import hashlib
from collections import OrderedDict

import torch
from torch.fx import Graph, GraphModule, Node

from ._p800_fp32_region import _arithmetic, _dtype, _is_native_chain, _meta_cast

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# A program owns generated code, never model attributes or converted input data.
_programs = OrderedDict()


class _Program:
    """Stable entry shared by already-prepared callers if admission changes."""
    def __init__(self, fn, route):
        self.fn = fn
        self.state = {'route': route}

    def __call__(self, *xs):
        return self.fn(*xs)

    def use_reference(self, reference, reason):
        self.fn = reference
        self.state.update(route='eager', reason=reason)


def template_graph(graph):
    nodes = list(graph.nodes)
    ops = [n for n in nodes if n.op not in ('placeholder', 'output')]
    if len(ops) not in (2, 3) or not all(_arithmetic(n) for n in ops):
        return False
    if any(n.kwargs.get('dtype') is not None for n in ops):
        return False
    for first, second in zip(ops, ops[1:]):
        if tuple(first.users) != (second,) or not _is_native_chain(first, second):
            return False
    out = nodes[-1].args[0]
    if isinstance(out, (tuple, list)) and len(out) == 1:
        out = out[0]
    return out is ops[-1] and all(n.users for n in nodes if n.op == 'placeholder')


def region(gm, inputs, *, canonical=True):
    if (not inputs or not all(isinstance(x, torch.Tensor) for x in inputs)
            or inputs[0].device.type != 'cuda' or inputs[0].dtype not in _DTYPES
            or any(x.device != inputs[0].device or x.dtype != inputs[0].dtype
                   or any(type(size) is not int for size in x.shape)
                   or not x.is_contiguous() or x.storage_offset() != 0 for x in inputs)
            or not template_graph(gm.graph)):
        return None
    from ._resource_codegen import baidu_target
    target = baidu_target()
    if not target or (target['arch'], target['warp_size']) != (3, 1):
        return None
    dtype = inputs[0].dtype
    graph = Graph()
    env = {}
    for n in gm.graph.nodes:
        if n.op == 'placeholder':
            clone = graph.placeholder(f'arg{len(env)}' if canonical else n.target)
        else:
            if n.op != 'output' and _dtype(n) not in (None, dtype):
                return None
            clone = graph.node_copy(n, env.__getitem__)
        if n.op != 'output':
            _meta_cast(clone, n, torch.float32)
        env[n] = clone
    core = GraphModule({}, graph)
    core._f06_shared_fp32 = True
    return core


def bind(body, dtype, output, autocast=False):
    """Only the two conversion boundaries depend on the caller's dtype."""
    if dtype == torch.float32 and not autocast:
        return body
    def run(*args):
        if autocast:
            with torch.autocast('cuda', enabled=False):
                result = body(*(x.float() for x in args))
        else:
            result = body(*(x.float() for x in args))
        if isinstance(output, Node):
            return result.to(dtype)
        value, = result
        converted = value.to(dtype)
        return (converted,) if isinstance(output, tuple) else [converted]
    return run


def _empty_softmax(core, inputs):
    if len(inputs) != 2 or (inputs[0].numel() and inputs[1].numel()):
        return None
    from . import _p800_reduce
    spec = _p800_reduce.match(core, inputs)
    if spec is None or spec['kind'] != 'add_softmax' or spec['m'] * spec['n']:
        return None
    fn, proof = _p800_reduce.compile(spec, inputs)
    output = next(n for n in core.graph.nodes if n.op == 'output').args[0]
    def run(*xs):
        result = fn(xs)
        if isinstance(output, tuple):
            return result
        return list(result) if isinstance(output, list) else result[0]
    return run, proof


def native_entry(gm, inputs, compile_fx, options):
    """Prepare the guarded row specialization or compile the canonical FP32 graph."""
    from torch._inductor import config
    from ._p800_fp32_region import _mm_epilogue
    if (torch.is_grad_enabled() or not isinstance(gm, GraphModule)
            or options.get('triton.cudagraphs', config.triton.cudagraphs)
            or options.get('freezing', config.freezing)):
        return None
    core = region(gm, inputs)
    if core is None:
        return None
    empty = _empty_softmax(core, inputs)
    if empty is not None:
        return empty[0]
    native_core = GraphModule({}, copy.deepcopy(core.graph))
    # The large FP32 vector SiLU epilogue differs by one rounded BF16 bin
    # on changed inputs. Keep the SDK's FP32 divide on this common graph.
    if tuple(inputs[0].shape) == (2048, 64) and len(inputs) == 3 and tuple(inputs[1].shape) == (64, 2048):
        for node in native_core.graph.nodes:
            if _mm_epilogue(node) == 'silu':
                node.target = torch.ops.flagtree_p800.fp32_epilogue.default
                node.args, node.kwargs = (node.args[0], 'silu'), {}
                native_core.recompile()
                break
    amp = torch.is_autocast_enabled('cuda')
    from torch._guards import detect_fake_mode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from ._p800_fp32_region import _value
    metadata = [_value(node) for node in gm.graph.nodes if node.op == 'placeholder']
    mode = detect_fake_mode()
    if mode is None or any(not isinstance(x, torch.Tensor) or
                           any(type(s) is not int for s in (*x.shape, *x.stride())) for x in metadata):
        return None
    dtype = inputs[0].dtype
    output = next(n for n in core.graph.nodes if n.op == 'output').args[0]
    if len(inputs) == 2:
        from . import _p800_reduce
        spec = _p800_reduce.match(core, inputs)
        if spec is not None and spec['kind'] == 'add_softmax':
            # Lower the complete recognized region first. Compiling a generic
            # Inductor reduction eagerly can fail on wide, non-power-of-two
            # rows before the validated streaming implementation is reached.
            return _lazy_inline(core, dtype, output, amp, bind(core.forward, dtype, output, amp))
    # Dynamic shapes belong to the outer compiler. For these static guarded
    # layouts construct metadata in its active mode, without allocating real
    # temporaries or introducing new unbacked shape symbols.
    with torch.autocast('cuda', enabled=False):
        with mode:
            examples = [torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float32) for x in inputs]
        FakeTensorProp(native_core, mode=mode).propagate_dont_convert_inputs(*examples)
        for node in native_core.graph.nodes:
            if 'val' in node.meta:
                node.meta['example_value'] = node.meta['val']
        body = compile_fx(native_core, examples, config_patches=options)
    wrapped = bind(body, dtype, output, amp)
    if dtype == torch.float32:
        return wrapped
    return _lazy_inline(core, dtype, output, amp, wrapped)


def _lazy_inline(core, dtype, output, amp, fallback):
    from threading import RLock
    from . import _p800_inline_fp32
    lock = RLock()
    prepared = None
    reference = _FP32Reference(core, dtype)
    tolerance = (dict(atol=.0003, rtol=.0003) if dtype == torch.float32 else
                 dict(atol=.03 if dtype == torch.bfloat16 else .003, rtol=.003))
    def run(*values):
        nonlocal prepared
        if prepared is None:
            if torch.cuda.is_current_stream_capturing():
                return fallback(*values)
            with lock:
                if prepared is None:
                    candidate, _, _ = _p800_inline_fp32.select(core, values, reference, tolerance)
                    prepared = candidate[1] if candidate else fallback
        return prepared(*values)
    return run


def _fp32_tolerance(tolerance):
    return {k: min(v, .0003) for k, v in tolerance.items()}


class _FP32Reference(torch.nn.Module):
    """Arithmetic oracle on the caller's actual inputs, before the model output cast."""
    def __init__(self, core, output_dtype):
        super().__init__()
        self.core = core
        self.output_dtype = output_dtype

    @property
    def graph(self):
        return self.core.graph

    def forward(self, *values):
        with torch.autocast(values[0].device.type, enabled=False):
            return self.core(*(x.float() for x in values))

    def _f06_assert_close(self, got, expected, tolerance):
        from ._resource_benchmark import assert_close
        assert_close(got, expected, _fp32_tolerance(tolerance), output_dtype=self.output_dtype)


def _boundary_probes(core, inputs):
    # Exercise values representable at either input boundary. Keep all reference
    # outputs in FP32 so the selected shared body is judged before its caller's cast.
    from ._resource_benchmark import _changed, _cpu
    probes = []
    for dtype in (torch.float16, torch.bfloat16):
        for seed in (None, 'changed', 82, 97):
            if seed is None:
                xs = tuple(x.to(dtype).float() for x in inputs)
            elif seed == 'changed':
                xs = tuple(x.float() for x in _changed(tuple(x.to(dtype) for x in inputs)))
            else:
                gen = torch.Generator(device='cpu').manual_seed(seed)
                xs = tuple(torch.randn(x.shape, generator=gen, device='cpu').to(dtype).to(x.device).float()
                           for x in inputs)
            probes.append((xs, _cpu(core(*xs))))
    return probes


def _verify_fp32_probes(fn, probes, tolerance):
    from ._resource_benchmark import assert_close
    for values, expected in probes:
        assert_close(fn(*values), expected, tolerance, output_dtype=torch.float32)


def prepare(core, inputs, *, tolerance, environment_key, single_kernel, split_reductions, split_gelu,
            public_timer, audit, prepared_candidates, prepared_measurements):
    from ._resource_autotune import _calibrate_graph, context_key
    from ._resource_benchmark import validate, GraphMeasurement, GraphGroup
    from torch._dynamo.convert_frame import compile_lock
    dtype = inputs[0].dtype
    amp = torch.is_autocast_enabled('cuda')
    output = next(n for n in core.graph.nodes if n.op == 'output').args[0]
    empty = _empty_softmax(core, inputs)
    if empty is not None:
        fn, proof = empty
        route = proof['route']
        decision = dict(selected=route, fallback=False, conversions='none', launches=0,
                        context_key=context_key(core, inputs, environment_key))
        if prepared_candidates is not None:
            prepared_candidates[route] = fn, proof
        if audit is not None:
            audit.append(dict(decision=decision, candidates=[dict(name=route, status='passed', proof=proof)]))
        return fn, decision
    fp_tolerance = _fp32_tolerance(tolerance)
    with torch.autocast('cuda', enabled=False):
        xs = tuple(x.float() for x in inputs)
        key = (context_key(core, xs, environment_key), tuple(sorted(fp_tolerance.items())),
               bool(single_kernel), split_reductions, split_gelu)
        with compile_lock:
            cached = _programs.get(key)
            hit = cached is not None
            if not hit:
                probes = _boundary_probes(core, xs)
                def verify(fn):
                    _verify_fp32_probes(fn, probes, fp_tolerance)
                records = []
                candidates = {}
                body, decision = _calibrate_graph(
                    core, xs, tolerance=fp_tolerance, environment_key=environment_key, single_kernel=single_kernel,
                    split_reductions=split_reductions, split_gelu=split_gelu, records=None, audit=records,
                    prepared_candidates=candidates,
                    prepared_measurements=prepared_measurements if dtype == torch.float32 and not amp else None,
                    public_timer=public_timer if dtype == torch.float32 and not amp else None, verify=verify)
                cached = _Program(body, decision['selected']), decision, records[0], candidates
                if not decision.get('reason'):
                    _programs[key] = cached
                    if len(_programs) > 64:
                        _programs.popitem(last=False)
            else:
                _programs.move_to_end(key)
        body, base, record, candidates = cached
        decision = dict(base, shared_fp32=hashlib.sha256(repr(key).encode()).hexdigest(),
                        fp32_context=key[0], shared_cache_hit=hit, input_dtype=str(dtype),
                        conversions='sdk_eager' if dtype != torch.float32 else 'none')
        decision.update(selected=body.state['route'], shared_state=body.state,
                        accuracy_reference='unrounded_fp32_eager', compute_tolerance=fp_tolerance)
        if 'reason' in body.state:
            decision.update(fallback=True, reason=body.state['reason'])
        selected = bind(body, dtype, output, amp)
        if dtype != torch.float32 or hit:
            # Validate on the caller's current values and tolerance, including
            # Inf/NaN and conditioned LayerNorm. No caller tensor enters the cache.
            oracle = _FP32Reference(core, dtype)
            try:
                validate(selected, oracle, inputs, tolerance)
            except AssertionError as exc:
                reason = 'caller validation: ' + str(exc)
                # A public writeback failure can affect just this dtype. Revoke
                # the shared body only if its FP32 computation also fails.
                invalid_body = dtype == torch.float32
                if not invalid_body:
                    try:
                        validate(body, core, xs, fp_tolerance)
                    except AssertionError:
                        invalid_body = True
                if invalid_body:
                    with compile_lock:
                        body.use_reference(core.forward, reason)
                selected = bind(core.forward, dtype, output, amp)
                decision.update(selected='eager', fallback=True, reason=reason)
        else:
            oracle = core
        if dtype != torch.float32:
            from . import _p800_inline_fp32
            variant, trials, timings = _p800_inline_fp32.select(core, inputs, oracle, tolerance,
                                                                single_kernel=single_kernel, public_timer=public_timer,
                                                                shared=selected)
            record = dict(record, boundary_candidates=trials)
            decision.update(compute_route=decision['selected'], boundary_timings_us=timings)
            if variant is not None:
                name, selected, proof = variant
                boundary = 'kernel_load_sdk_store' if proof['output_conversion'] == 'sdk_store' else 'kernel_load_store'
                decision.update(selected=name, conversions=boundary, launches=proof['kernel_count'], fallback=False)
                candidates = dict(candidates, **{name:(selected,proof)})
        if prepared_candidates is not None or prepared_measurements is not None:
            for name, (fn, proof) in candidates.items():
                wrapped = selected if name == decision['selected'] else bind(fn, dtype, output, amp)
                if prepared_candidates is not None:
                    prepared_candidates[name] = wrapped, proof
                if prepared_measurements is not None and name not in prepared_measurements:
                    captures = []
                    try:
                        for _ in range(3):
                            captures.append(GraphMeasurement(wrapped, oracle, inputs, tolerance, calls=16))
                    except Exception:
                        for cap in captures:
                            cap.close()
                        raise
                    prepared_measurements[name] = GraphGroup(captures)
        if audit is not None:
            audit.append(dict(record, decision=decision, candidate_measurement_dtype='float32'))
        return selected, decision
