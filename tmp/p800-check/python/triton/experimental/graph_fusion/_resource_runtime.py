"""Opt-in native-Dynamo backend for already validated resource partitions.

Profiling and plan acceptance happen offline. No tuning, graph analysis or
resource decisions occur in the returned hot-path wrapper.
"""
import hashlib
import os
import json


def inference_entry(binary):
    """An instance-local Torch 2.9 entry; retain native math and cleanup.

    Only the unused record_function scope is omitted. Profiling/debugging and
    other Torch versions retain the complete native entry. No class, launcher,
    generated wrapper, allocator or baseline is patched.
    """
    import torch
    if torch.__version__.split('+')[0].split('.')[:2] != ['2', '9']:
        return binary, False
    from torch._inductor.output_code import CompiledFxGraph
    if type(binary) is not CompiledFxGraph:
        return binary, False
    from torch._inductor import debug
    from torch._dynamo.utils import get_runtime_metrics_context
    from torch._inductor.runtime.autotune_cache import AutotuneCacheBundler
    from torch.autograd import profiler
    original_call = CompiledFxGraph.__call__
    profiler_enabled = torch.autograd._profiler_enabled
    grad_enabled = torch.is_grad_enabled

    def run(inputs):
        if (grad_enabled() or profiler._is_profiler_enabled or profiler_enabled()
                or debug.RECORD_GRAPH_EXECUTION or CompiledFxGraph.__call__ is not original_call):
            return binary(inputs)
        assert binary.current_callable is not None
        try:
            # Includes native input alignment, shape/stride checks, allocation,
            # device/stream selection and CachingAutotuner.run unchanged.
            return binary.current_callable(inputs)
        finally:
            # Native finish/end_compile are no-ops in the empty states. Keep
            # their effects, including after exceptions and first-call tuning.
            metrics = get_runtime_metrics_context()
            if metrics._metrics:
                metrics.finish()
            if AutotuneCacheBundler._bundler is not None:
                AutotuneCacheBundler.end_compile()

    return run, True


def plan_identity(plan, matmul_path):
    payload = {
        'context': plan.context_key, 'matmul_path': matmul_path, 'regions':
        [(r.start, r.end, r.resource, r.evidence_id) for r in plan.regions]
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_plan(gm, example_inputs, plan, *, matmul_path='template'):
    """Compile a proposed plan for offline full-graph acceptance, not deployment."""
    fn, proof = prepare_regions(gm, example_inputs, plan.regions, matmul_path=matmul_path)
    proof['plan_id'] = plan_identity(plan, matmul_path)
    return fn, proof


def prepare_regions(gm, example_inputs, regions, *, matmul_path='template'):
    """Compile proposed regions using Inductor; does not assert acceptance."""
    import torch.fx
    from torch.fx.node import map_arg
    from ._resource_fx import build_region, collect_example_values
    from ._resource_codegen import compile_region
    import operator
    # Fake values propagate through the original operators, without running
    # kernels or choosing configs in a device capture.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._guards import detect_fake_mode
    fake_mode = detect_fake_mode(example_inputs) or FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        fake_inputs = tuple(fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in example_inputs)
        values = collect_example_values(gm, fake_inputs)
    graph = torch.fx.Graph()
    env = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            env[node.name] = graph.node_copy(node, lambda n: env[n.name])
        elif node.op == 'get_attr':

            def live_attr(target):

                def read():
                    value = gm
                    for atom in target.split('.'):
                        value = getattr(value, atom)
                    return value

                read.__name__ = 'flagtree_attr_' + target.replace('.', '_')
                read.__qualname__ = read.__name__
                return read

            env[node.name] = graph.call_function(live_attr(node.target), ())
    evidence = []
    cursor = 0
    compute = [n for n in gm.graph.nodes if n.op in ('call_function', 'call_method')]
    for item in regions:
        if item.start != cursor or item.end <= cursor or item.end > len(compute):
            raise ValueError('partition coverage is not contiguous/complete')
        region = build_region(gm, item.start, item.end)
        if tuple(region.node_ids) != tuple(item.node_ids):
            raise ValueError('partition compute nodes changed')
        # Match offline profiling: only matrix-containing regions enable native
        # template candidates; pointwise/reduction regions keep the native path.
        matrix_targets = {
            torch.matmul, torch.mm, torch.bmm, operator.matmul, torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
            torch.ops.aten.matmul.default
        }
        region_path = matmul_path if any(n.op == 'call_function' and n.target in matrix_targets
                                         for n in region.gm.graph.nodes) else 'external'
        region_inputs = tuple(values[n] for n in region.input_names)
        dsa_spec = None
        # Default route is the F06 native/fused path (the 11-03 implementation);
        # the tle.dsa whole-kernel replacement is opt-in because it was measured
        # slower than the native fused route for Mul+Sum on 2048x2048.
        # float16 also goes through dsa: the f16 dsa kernel measures 20.6 us
        # (512x1024) / 39.8 us (2048x2048), versus 158-186 us for the native
        # Triton reduction fallback, so gating on float32 alone was a 4x miss.
        _dsa_dtypes = ('float32', 'float16')
        if (os.environ.get('FLAGTREE_DSA_ENABLE', '1') == '1'
                and item.resource in ('sdnn', 'cluster')
                and region_inputs
                and str(getattr(region_inputs[0], 'dtype', '')).endswith(_dsa_dtypes)):
            from . import _p800_dsa
            dsa_spec = _p800_dsa.match(region.gm, region_inputs)
        if dsa_spec is not None:
            # Whole-kernel replacement: the documented tle.dsa 2D-DMA form beats
            # the native pointwise codegen for this region on the SDNN path.
            try:
                compiled, proof = _p800_dsa.compile(dsa_spec, region_inputs)
                fast_entry = True
            except RuntimeError:
                # No tile passed validation (or an unsupported dtype slipped in):
                # keep the native region rather than failing the whole plan.
                dsa_spec = None
        flat_spec = None
        if dsa_spec is None and os.environ.get('FLAGTREE_FLAT_ADD', '1') == '1':
            try:
                from . import _p800_flat
                flat_spec = _p800_flat.match(region.gm, region_inputs)
            except Exception:
                flat_spec = None
        if flat_spec is not None:
            try:
                compiled, proof = _p800_flat.compile(flat_spec, region_inputs)
                fast_entry = True
            except Exception:
                flat_spec = None
        reduce_spec = None
        if dsa_spec is None and flat_spec is None and os.environ.get('FLAGTREE_FLAT_REDUCE', '1') == '1':
            try:
                from . import _p800_reduce
                reduce_spec = _p800_reduce.match(region.gm, region_inputs)
            except Exception:
                reduce_spec = None
        if reduce_spec is not None:
            try:
                compiled, proof = _p800_reduce.compile(reduce_spec, region_inputs)
                fast_entry = True
            except Exception:
                reduce_spec = None
        if dsa_spec is None and flat_spec is None and reduce_spec is None:
            compiled, proof = compile_region(region.gm, region_inputs, item.resource,
                                             matmul_path=region_path)
            if not proof['resource_verified']:
                raise RuntimeError('region lacks verified resource evidence; external calls remain barriers')
            compiled, fast_entry = inference_entry(compiled)
        proof['inference_entry'] = dict(enabled=fast_entry, native_launchers=True,
                                       profiling_debug_fallback=True, native_cleanup=True)

        def bind(binary, key):

            def run(*xs):
                return tuple(binary(list(xs)))

            run.__name__ = 'flagtree_region_' + key
            run.__qualname__ = run.__name__
            return run

        call = graph.call_function(bind(compiled, str(item.start)), tuple(env[n] for n in region.input_names))
        for index, name in enumerate(region.output_names):
            env[name] = graph.call_function(operator.getitem, (call, index))
        evidence.append(proof)
        cursor = item.end
    if cursor != len(compute): raise ValueError('partition does not cover graph')
    output = next(n for n in gm.graph.nodes if n.op == 'output')
    graph.output(map_arg(output.args[0], lambda n: env[n.name]))
    prepared = torch.fx.GraphModule(gm, graph)
    return prepared.forward, {
        'regions': evidence, 'wrapper': prepared.code
    }


def resource_backend(observations, *, environment_key, accepted_plan_ids=(), matmul_path='template', audit=None):
    """Return a torch.compile backend retaining native Dynamo guards.

    `accepted_plan_ids` must come from full-plan ordinary/repeated/special-value
    and end-to-end performance acceptance in this exact environment. Missing or
    rejected profiles/acceptance retain native whole-graph Inductor behavior.
    The explicit external MatMul experiment is an unverified barrier route and
    must not be represented by fabricated SDNN observations.
    """
    from ._resource_planner import choose_partition, PartitionPlan
    from ._resource_fx import context_fingerprint, build_all_regions, UnsupportedGraph
    from ._resource_codegen import baidu_target
    if matmul_path not in ('external', 'template'): raise ValueError('unknown MatMul path')
    if not isinstance(environment_key, str) or not environment_key.strip(): raise ValueError('environment key required')
    profiles = tuple(observations)
    accepted = frozenset(accepted_plan_ids)

    def backend(gm, inputs):
        from torch._inductor.compile_fx import compile_fx

        def fallback(reason):
            if audit is not None: audit.append({'status': 'native_fallback', 'reason': reason})
            return compile_fx(gm, inputs)

        if baidu_target() is None: return fallback('non_baidu_backend')
        try:
            context = context_fingerprint(gm, inputs, environment_key)
            regions = build_all_regions(gm)
            nodes = next(r.node_ids for r in regions if r.start == 0 and r.end == max(x.end for x in regions))
        except (UnsupportedGraph, ValueError, StopIteration) as exc:
            return fallback('unsupported_graph: ' + str(exc))
        plan = choose_partition(nodes, context, profiles, is_baidu_backend=True)
        if not isinstance(plan, PartitionPlan): return fallback('no_valid_profiled_plan')
        if plan_identity(plan, matmul_path) not in accepted: return fallback('plan_not_accepted_end_to_end')
        fn, proof = prepare_plan(gm, inputs, plan, matmul_path=matmul_path)
        if audit is not None: audit.append({'status': 'resource_partition', 'proof': proof})
        return fn

    return backend
