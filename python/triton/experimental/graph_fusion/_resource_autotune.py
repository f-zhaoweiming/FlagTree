"""Whole-plan autotuning of native Inductor resource/partition candidates.

No arithmetic kernels or tiling configurations are supplied here. Calibration
is explicit (or lazy on the first ordinary torch.compile call), never in a GPU
capture. Native Dynamo guards delimit each prepared hot path. Offline workers
may supply independently validated records to exclude unsafe candidates.
"""
from dataclasses import asdict, dataclass
import ast
import hashlib
import json
import math
import operator
import statistics
import threading

from ._resource_fx import build_region, context_fingerprint, unsafe_sdnn_sum


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    resource: str
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    regions: tuple[Segment, ...] = ()
    single_kernel: bool = False


def _matrix(node):
    import torch
    return (node.op == 'call_method' and node.target in ('mm', 'matmul')) or (
        node.op == 'call_function' and node.target in (
            torch.mm, torch.matmul, operator.matmul, torch.ops.aten.mm.default,
            torch.ops.aten.matmul.default))


def _add_gelu_mode(nodes):
    import torch
    if len(nodes) != 2:
        return None
    add, gelu = nodes
    if not ((add.op == 'call_method' and add.target == 'add') or
            (add.op == 'call_function' and add.target in
             (operator.add, torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar))):
        return None
    if (gelu.op != 'call_function' or gelu.target not in
            (torch.nn.functional.gelu, torch.ops.aten.gelu.default)):
        return None
    operand = gelu.args[0] if gelu.args else gelu.kwargs.get('input', gelu.kwargs.get('self'))
    return gelu.kwargs.get('approximate', 'none') if operand is add else None


def candidate_plans(gm, inputs, *, single_kernel=None, split_reductions=False, split_gelu=False):
    """Enumerate bounded F06 plans; speed assumptions never select the winner."""
    import torch
    from . import _p800_shared_fp32
    if inputs and getattr(inputs[0], 'dtype', None) in (torch.float16, torch.bfloat16):
        core = _p800_shared_fp32.region(gm, inputs)
        if core is not None:
            choices = candidate_plans(core, tuple(x.float() for x in inputs), single_kernel=single_kernel,
                                      split_reductions=split_reductions, split_gelu=split_gelu)
            choices = [p for p in choices if not single_kernel or p.name in ('eager', 'inductor')]
            choices.append(Candidate('fp32_inline', single_kernel=bool(single_kernel)))
            if not single_kernel and any(_matrix(n) for n in core.graph.nodes):
                choices.append(Candidate('fp32_inline_split'))
                from ._p800_inline_fp32 import sdk_store_candidate
                if sdk_store_candidate(core, inputs):
                    choices.append(Candidate('fp32_product_sdk_store'))
            return tuple(choices)
    from ._p800_fp32_region import normalized, is_wide
    gm = normalized(gm, inputs)
    nodes = [n for n in gm.graph.nodes if n.op in ('call_function', 'call_method')]
    whole = build_region(gm, 0, len(nodes))  # purity/support gate, before execution
    dots = [i for i, n in enumerate(nodes) if _matrix(n)]
    if single_kernel is None:
        single_kernel = False
    if type(single_kernel) is not bool:
        raise TypeError('single_kernel must be bool or None')
    if type(split_reductions) is not bool:
        raise TypeError('split_reductions must be bool')
    if type(split_gelu) is not bool:
        raise TypeError('split_gelu must be bool')

    def segment(start, end, resource):
        return Segment(start, end, resource, tuple(n.name for n in nodes[start:end]))

    plans = [Candidate('eager'), Candidate('inductor')]
    if is_wide(gm):
        plans.append(Candidate('fp32_compute', single_kernel=single_kernel))
        return tuple(plans)
    if dots:
        # Measured on P800 (fixed rank-1 pass): the SDNN activation stage lowers
        # tanh/sigmoid through a scalar path, making fused MatMul+GELU(tanh) and
        # MatMul+SiLU 4-25x slower than eager. GELU(exact) is 1.3-3.1x faster.
        # Decline the SDNN plans only for the two slow activations.
        slow_act = False
        for node in nodes:
            name = getattr(getattr(node, 'target', None), '__name__', '')
            if name in ('silu', 'silu_', 'sigmoid'):
                slow_act = True
            if name == 'gelu' and str(node.kwargs.get('approximate', 'none')) == 'tanh':
                slow_act = True
        # Native matmul route: bias/relu in one kernel; silu as matmul kernel +
        # a separate SDNN EW kernel (the LUT path that is fast and exact).
        try:
            from ._p800_native_mm import match as _native_match, supports_grouped_relu
            spec = _native_match(gm, inputs)
            if spec is not None:
                fused = spec['act'] in (0, 2) or spec['silu_path'] == 'epilogue'
                if fused or not single_kernel:
                    plans.append(Candidate('native_mm', single_kernel=single_kernel))
                if supports_grouped_relu(spec):
                    plans.append(Candidate('native_mm_grouped_relu', single_kernel=single_kernel))
                if getattr(gm, '_f06_shared_fp32', False) and spec['act'] == 1 and not single_kernel:
                    plans.append(Candidate('native_mm_sdk_silu'))
            vendor = _native_match(gm, inputs, vendor_product=True) if not single_kernel else None
            if vendor is not None:
                if vendor['act'] == 0 and hasattr(torch.ops.custom_ops, 'linear'):
                    plans.append(Candidate('vendor_linear'))
                plans.append(Candidate('eager_mm_cluster_activation', (
                    segment(0, 1, 'eager'), segment(1, len(nodes), 'cluster'))))
        except Exception:
            pass
        if slow_act:
            # The dot-fused epilogue stays on the scalar extern path even with
            # LUT tables; tanh has no usable encoding, so keep eager/inductor.
            return plans
        plans.append(Candidate('sdnn', (segment(0, len(nodes), 'sdnn'),), single_kernel))
        # Native dot/MMA is SDNN-only in the tested SDK. Do not relabel it as
        # cluster simply by setting a launch flag. Whole-cluster is ineligible.
        if not single_kernel and len(dots) == 1:
            for cut in range(dots[0] + 1, len(nodes)):
                plans.append(Candidate(f'sdnn_cluster_at_{cut}', (
                    segment(0, cut, 'sdnn'), segment(cut, len(nodes), 'cluster'))))
    else:
        plans.append(Candidate('cluster', (segment(0, len(nodes), 'cluster'),), single_kernel))
        # Current SDK cannot compile these SDNN reduction graphs reliably.
        # Keep them on native Inductor/cluster; retain SDNN for Mul+Sum trials.
        targets = ' '.join(str(n.target) for n in nodes)
        sdnn_sum_unsafe = unsafe_sdnn_sum(nodes, inputs)
        gelu_mode = _add_gelu_mode(nodes)
        # P800's fused SDNN tanh-GELU and BF16 exact-GELU take much
        # longer than eager. Keep SDNN's fast FP16/FP32 exact epilogue.
        sdnn_gelu_slow = gelu_mode == 'tanh' or (
            gelu_mode == 'none' and len(inputs) > 0 and
            all(getattr(x, 'dtype', None) == torch.bfloat16 for x in inputs))
        if not sdnn_sum_unsafe and not sdnn_gelu_slow and not any(x in targets for x in ('softmax', 'layer_norm')):
            plans.append(Candidate('sdnn', (segment(0, len(nodes), 'sdnn'),), single_kernel))
        if gelu_mode in ('none', 'tanh') and not single_kernel:
            from ._p800_add_gelu import eligible
            if eligible(gm, inputs, nodes[0]):
                plans.append(Candidate('eager_add_cluster_gelu', (
                    segment(0, 1, 'eager'), segment(1, 2, 'cluster'))))
        if split_gelu and gelu_mode in ('none', 'tanh') and not single_kernel:
            for producer in ('cluster', 'sdnn'):
                for consumer in (('cluster', 'sdnn') if gelu_mode == 'none' else ('cluster',)):
                    plans.append(Candidate(f'{producer}_{consumer}_at_1', (
                        segment(0, 1, producer), segment(1, 2, consumer))))
        if split_reductions and not single_kernel and len(nodes) == 2:
            product, reduction = nodes
            is_mul = ((product.op == 'call_method' and product.target == 'mul') or
                      (product.op == 'call_function' and product.target in
                       (torch.mul, operator.mul, torch.ops.aten.mul.Tensor)))
            is_sum = ((reduction.op == 'call_method' and reduction.target == 'sum') or
                      (reduction.op == 'call_function' and reduction.target in
                       (torch.sum, torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default)))
            if is_mul and is_sum and reduction.args[0] is product:
                # Materialize the complete product with its original dtype.
                # Native Inductor chooses tiles independently for each region;
                # no column slicing, fixed XBLOCK, or launch count is supplied.
                # Prior microbenchmarks favor SDNN for some row reductions,
                # and cluster for others. Keep both consumers available: a
                # standalone-op speedup cannot certify the full split plan.
                for producer in ('cluster', 'sdnn'):
                    for consumer in (('cluster',) if sdnn_sum_unsafe else ('cluster', 'sdnn')):
                        plans.append(Candidate(producer + '_' + consumer + '_at_1', (
                            segment(0, 1, producer), segment(1, 2, consumer))))
    assert tuple(n.name for n in nodes) == whole.node_ids
    return tuple(plans)


def context_key(gm, inputs, environment_key):
    from ._p800_fp32_region import normalized
    gm = normalized(gm, inputs)
    from ._p800_compiler import policy_key
    # Include scheduler/validation/codegen source and the caller's device/SDK
    # identity. Profiles for another graph, shape, stride or dtype cannot match.
    from pathlib import Path
    source = b''.join((Path(__file__).parent / p).read_bytes() for p in (
        '_resource_autotune.py', '_p800_shared_fp32.py', '_p800_inline_fp32.py', '_p800_fp32_region.py', '_p800_fp32_mm.py', '_resource_benchmark.py', '_resource_runtime.py', '_resource_fx.py', '_resource_entry.py',
        '_p800_generated.py', '_resource_codegen.py', '_p800_matmul.py', '_p800_native_mm.py', '_p800_bound_launch.py', '_p800_reduce.py', '_p800_flat.py', '_p800_dsa.py',
        '_p800_inductor_softmax.py', '_p800_add_gelu.py', '_p800_bf16_gelu.py', '_p800_bf16_gelu.xpu', '_p800_vector_rows.py', '_p800_vector_rows.xpu', '_p800_inductor_vector_rows.py', '_p800_inductor_gelu.py', '_p800_layernorm.py', '_p800_layernorm.xpu', '_p800_inductor_layernorm.py'))
    cpu_entry = Path(__file__).with_name('_resource_entry.c')
    if cpu_entry.is_file():
        source += cpu_entry.read_bytes()
    import torch
    precision = json.dumps({
        'tf32': torch.backends.cuda.matmul.allow_tf32,
        'fp16_reduced': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'bf16_reduced': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }, sort_keys=True)
    version = hashlib.sha256(source).hexdigest() + policy_key() + precision
    return context_fingerprint(gm, inputs, environment_key + ':' + version)


def compiler_options():
    from torch._inductor import config, list_mode_options
    options = dict(list_mode_options('max-autotune-no-cudagraphs'))
    options.update({'triton.cudagraphs': False, 'compile_threads': 1,
                    'emulate_precision_casts': True, 'fx_graph_cache': False,
                    'fx_graph_remote_cache': False, 'autotune_in_subproc': False})
    for name in ('use_static_cuda_launcher', 'max_autotune_prune_choices_based_on_shared_mem'):
        if hasattr(config, name):
            options[name] = False
    return options


def launch_count(source):
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'call']
    if len(calls) != 1:
        raise RuntimeError('cannot identify native wrapper call')
    body = calls[0]
    if any(isinstance(n, (ast.For, ast.While, ast.If)) for n in ast.walk(body)):
        return None  # A conditional or loop has no fixed static launch count.
    return sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run'
               for n in ast.walk(body))


def _shared_mm_tile(inputs):
    if len(inputs) < 2:
        return None
    shapes = tuple(inputs[0].shape), tuple(inputs[1].shape)
    # The Inductor selector lists (BLOCK_M, BLOCK_K, BLOCK_N, stages, warps).
    if shapes == ((64, 64), (64, 1024)):
        return (64, 32, 128, 2, 8)

    return None


def compile_candidate(gm, inputs, candidate):
    """Compile only with Inductor; external calls stay explicitly unassigned."""
    from torch._inductor import config
    from ._resource_runtime import prepare_regions
    from ._p800_inductor_softmax import native_softmax_scope
    from ._p800_compiler import target_scope
    from ._p800_fp32_region import normalized, compile_wide, is_wide
    import torch
    from . import _p800_shared_fp32
    if inputs and getattr(inputs[0], 'dtype', None) in (torch.float16, torch.bfloat16):
        core = _p800_shared_fp32.region(gm, inputs)
        if core is not None:
            if candidate.name in ('fp32_inline', 'fp32_inline_split', 'fp32_product_sdk_store'):
                from ._p800_inline_fp32 import compile_body
                if candidate.single_kernel and candidate.name != 'fp32_inline':
                    raise ValueError('this matrix epilogue requires multiple launches')
                result = compile_body(core, inputs, split=candidate.name != 'fp32_inline',
                                      sdk_store=candidate.name == 'fp32_product_sdk_store')
                if result is None:
                    raise ValueError('unsupported boundary-fused hardware layout')
                return result
            if candidate.single_kernel and candidate.name not in ('eager', 'inductor'):
                raise ValueError('the SDK precision boundaries require additional launches')
            with torch.autocast('cuda', enabled=False):
                fn, proof = compile_candidate(core, tuple(x.float() for x in inputs), candidate)
            proof['boundary_launches'] = len(inputs) + 1
            output = next(n for n in core.graph.nodes if n.op == 'output').args[0]
            return _p800_shared_fp32.bind(fn, inputs[0].dtype, output, torch.is_autocast_enabled('cuda')), proof
    gm = normalized(gm, inputs)
    if candidate.name == 'fp32_compute':
        return compile_wide(gm, inputs, single_kernel=candidate.single_kernel)
    if candidate.name == 'eager':
        return gm.forward, {'route': 'eager', 'regions': [], 'compiler_options': {}}
    if candidate.name == 'eager_add_cluster_gelu':
        from . import _p800_add_gelu
        nodes = [n for n in gm.graph.nodes if n.op in ('call_function', 'call_method')]
        return _p800_add_gelu.compile(gm, inputs, candidate, _add_gelu_mode(nodes))
    if candidate.name == 'eager_mm_cluster_activation':
        from ._p800_native_mm import compile_vendor_product
        return compile_vendor_product(gm, inputs, candidate)
    if candidate.name == 'vendor_linear':
        from ._p800_native_mm import compile_vendor_linear
        return compile_vendor_linear(gm, inputs, candidate)
    if candidate.name in ('native_mm', 'native_mm_sdk_silu', 'native_mm_grouped_relu'):
        from ._p800_native_mm import match as _native_match, compile as _native_compile
        spec = _native_match(gm, inputs)
        if spec is None:
            raise RuntimeError('native matmul pattern no longer matches')
        sdk_silu = candidate.name == 'native_mm_sdk_silu'
        if sdk_silu:
            if spec['act'] != 1:
                raise RuntimeError('SDK SiLU requires a SiLU epilogue')
            spec = dict(spec, act=0, silu_path=None)
        fn, proof = _native_compile(spec, inputs, grouped_relu=candidate.name == 'native_mm_grouped_relu')
        if sdk_silu:
            product = fn
            def fn(*xs):
                value, = product(*xs)
                return (torch.nn.functional.silu(value),)
            proof.update(route=candidate.name, external_op='aten.silu', external_calls=1, kernel_count=2)
        output = next(n for n in gm.graph.nodes if n.op == 'output').args[0]
        if isinstance(output, tuple):
            return fn, proof
        if isinstance(output, list):
            return lambda *xs: list(fn(*xs)), proof
        return lambda *xs: fn(*xs)[0], proof
    options = compiler_options()
    shared = getattr(gm, '_f06_shared_fp32', False)
    shared_matrix = shared and any(_matrix(n) for n in gm.graph.nodes)
    if shared_matrix:
        # Speculative wide pointwise tiles can stall before the completed
        # plan is timed. GEMM-template selection stays enabled in compile_region.
        options.update(max_autotune=False, coordinate_descent_tuning=False)
    if shared_matrix and candidate.name == 'inductor':
        # Keep the SDK GEMM baseline. The emitted native_mm and SDNN candidates
        # still compete; speculative fused GEMM epilogue trials can hang KL3.
        options.update(max_autotune_gemm_backends='ATEN', benchmark_epilogue_fusion=False,
                       epilogue_fusion=False, coordinate_descent_tuning=False)
    if candidate.name == 'inductor' and is_wide(gm):
        # Coordinate descent can launch a faulting KL3 tile while timing
        # mixed-width conversion epilogues, before outputs can be checked.
        # Keep the SDK's initial layouts and the normal whole-plan timing.
        options['coordinate_descent_tuning'] = False
    with target_scope(), config.patch(options):
        if candidate.name == 'inductor':
            from torch._inductor.compile_fx import compile_fx
            from ._p800_matmul import native_template_eligibility
            import torch
            device = next(x.device for x in inputs if isinstance(x, torch.Tensor))
            with native_template_eligibility(device), native_softmax_scope(gm, inputs):
                import copy
                # compile_fx normalizes output/pre-grad passes in place. Keep
                # the semantic reference graph intact for every candidate.
                native_gm = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
                fn = compile_fx(native_gm, list(inputs))
                actual = fn(*inputs)  # finish native tuning while the ABI bridge is active
                expected = gm(*inputs)
                from torch.utils._pytree import tree_flatten, tree_unflatten
                leaves, actual_spec = tree_flatten(actual)
                expected_leaves, output_spec = tree_flatten(expected)
                if len(leaves) != len(expected_leaves):
                    raise RuntimeError('native Inductor changed the output arity')
                if actual_spec != output_spec:
                    native_fn = fn
                    def fn(*xs):
                        return tree_unflatten(tree_flatten(native_fn(*xs))[0], output_spec)
            return fn, {'route': 'inductor', 'mode': 'max-autotune-no-cudagraphs',
                        'resource_verified': False, 'compiler_options': options}
        # Prevent native addmm rewrites from erasing the original low-precision
        # MatMul -> cast -> Bias boundary. Native baseline retains its patterns.
        with config.patch({'pattern_matcher': False, 'benchmark_epilogue_fusion': not shared_matrix}):
            tile = _shared_mm_tile(inputs) if shared_matrix else None
            fn, proof = prepare_regions(gm, inputs, candidate.regions, verified_tile=tile)
            if tile is not None:
                proof['verified_mm_tile'] = dict(zip(('block_m', 'block_k', 'block_n', 'stages', 'warps'), tile))
    proof.update(route=candidate.name, compiler_options=options,
                 candidate=asdict(candidate), single_kernel_required=candidate.single_kernel)
    counts = [r['direct_launches'] if 'direct_launches' in r else launch_count(r['source_code'])
              for r in proof['regions']]
    proof['kernel_count'] = sum(counts) if None not in counts else None
    if candidate.single_kernel and proof['kernel_count'] != 1:
        raise RuntimeError(f'single kernel required, native wrapper launches {proof["kernel_count"]}')
    return fn, proof


CHECKS = ('ordinary', 'repeat', 'changed', 'special', 'immutable', 'graph_fresh_data')


def select_measurement(plans, records, key, tolerance, *, require_e2e=True, e2e_protocol='randomized_ab_ba_v1'):
    """Select measured whole plans, not sums of isolated operator estimates."""
    allowed = {p.name: p for p in plans}
    valid, rejected = [], []
    seen = set()
    for record in records:
        name = record.get('name')
        reason = None
        if name in seen:
            raise ValueError('duplicate candidate measurement: ' + str(name))
        seen.add(name)
        if name not in allowed or record.get('context_key') != key:
            reason = 'candidate/context mismatch'
        elif record.get('tolerance') != tolerance:
            reason = 'precision threshold mismatch'
        elif record.get('status') != 'passed' or any(record.get('checks', {}).get(k) is not True for k in CHECKS):
            reason = 'validation incomplete or failed'
        else:
            samples = record.get('gpu_samples_us', [])
            if not samples or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples):
                reason = 'invalid GPU timing samples'
            elif name == 'fp32_compute':
                proof = record.get('proof', {})
                if ((allowed[name].single_kernel and proof.get('kernel_count') != 1) or not proof.get('resource_verified')
                        or not proof.get('kernel_evidence') or not proof.get('final_round_only')):
                    reason = 'unverified FP32 computation'
            elif name in ('fp32_inline', 'fp32_inline_split'):
                proof = record.get('proof', {})
                count = 1 if name == 'fp32_inline' else 2
                if (not proof.get('resource_verified') or proof.get('compute_dtype') != 'float32'
                        or proof.get('kernel_count') != count or len(proof.get('kernel_evidence', ())) != count
                        or allowed[name].single_kernel and count != 1):
                    reason = 'unverified precision-boundary fusion'
            elif name == 'fp32_product_sdk_store':
                proof = record.get('proof', {})
                regions = proof.get('regions', ())
                binaries = regions[0].get('kernel_evidence', ()) if len(regions) == 1 else ()
                if (allowed[name].single_kernel or proof.get('external_op') != 'aten._to_copy'
                        or proof.get('external_calls') != 1 or proof.get('resource_verified') is not False
                        or not regions or not regions[0].get('resource_verified') or len(binaries) != 1
                        or binaries[0].get('effective') != 'sdnn' or proof.get('output_conversion') != 'sdk_store'):
                    reason = 'unverified FP32 product with external SDK writeback'
            elif name == 'vendor_linear':
                proof = record.get('proof', {})
                if (allowed[name].single_kernel or proof.get('vendor_op') != 'custom_ops::linear'
                        or proof.get('external_calls') != 1 or proof.get('resource_verified') is not False):
                    reason = 'unverified vendor linear call'
            elif name == 'native_mm_sdk_silu':
                proof = record.get('proof', {})
                binaries = proof.get('kernel_evidence', [])
                if (allowed[name].single_kernel or not proof.get('resource_verified') or len(binaries) != 1
                        or binaries[0].get('effective') != 'sdnn' or proof.get('external_op') != 'aten.silu'
                        or proof.get('external_calls') != 1 or proof.get('kernel_count') != 2):
                    reason = 'unverified FP32 MatMul + SDK SiLU'
            elif name in ('native_mm', 'native_mm_grouped_relu'):
                proof = record.get('proof', {})
                binaries = proof.get('kernel_evidence', [])
                if not proof.get('resource_verified') or len(binaries) != proof.get('kernel_count') or not binaries:
                    reason = 'unverified native matmul binaries'
                elif name == 'native_mm_grouped_relu' and (len(binaries) != 1 or binaries[0].get('effective') != 'sdnn'
                        or proof.get('schedule') != 'grouped_complete_tiles' or proof.get('external_calls') != 0):
                    reason = 'unverified grouped ReLU kernel'
                elif allowed[name].single_kernel and len(binaries) != 1:
                    reason = 'not a single kernel'
            elif allowed[name].regions:
                proofs = record.get('proof', {}).get('regions', [])
                if len(proofs) != len(allowed[name].regions):
                    reason = 'missing region evidence'
                else:
                    for region, proof in zip(allowed[name].regions, proofs):
                        if name in ('eager_add_cluster_gelu', 'eager_mm_cluster_activation') and region.resource == 'eager':
                            producer = 'eager_add' if name == 'eager_add_cluster_gelu' else 'eager_mm'
                            if (proof.get('route') != producer or proof.get('resource') != 'eager'
                                    or proof.get('external_calls') != 1 or proof.get('resource_verified') is not False
                                    or tuple(proof.get('node_ids', ())) != region.node_ids):
                                reason = 'unverified eager Add boundary' if producer == 'eager_add' else 'unverified eager MatMul boundary'
                            continue
                        if name == 'eager_add_cluster_gelu' and (
                                proof.get('route') != 'bf16_gelu' or proof.get('direct_launches') != 1):
                            reason = 'unverified GELU consumer'
                        if name == 'eager_mm_cluster_activation' and (
                                proof.get('route') != 'flat_add' or proof.get('direct_launches') != 1
                                or tuple(proof.get('node_ids', ())) != region.node_ids):
                            reason = 'unverified MatMul epilogue'
                        evidence = proof.get('kernel_evidence', [])
                        if (proof.get('resource_verified') is not True or proof.get('external_calls') != 0
                                or not evidence or ('direct_launches' in proof and len(evidence) != proof['direct_launches'])
                                or any(e.get('effective') != region.resource for e in evidence)):
                            reason = 'unverified actual resource'
                if allowed[name].single_kernel and record.get('proof', {}).get('kernel_count') != 1:
                    reason = 'not a single kernel'
        if reason:
            rejected.append({'name': name, 'reason': reason})
        else:
            valid.append((statistics.median(record['gpu_samples_us']), name, record))
    if not valid:
        raise RuntimeError('no candidate passed unchanged numerical and timing gates')
    acceptance = None
    if require_e2e:
        # Both independent, interleaved ordinary-call rounds must show a
        # material gain. Final reporting samples are never used to select.
        measured = {}
        def valid_rounds(rounds):
            return len(rounds) == 2 and all(len(row) >= 128 and all(
                type(x) in (int, float) and math.isfinite(x) and x > 0 for x in row) for row in rounds)

        direct = None
        direct_comparisons = {}
        for gpu, name, record in valid:
            rounds = record.get('e2e_round_samples_us', [])
            if record.get('e2e_protocol') != e2e_protocol or not valid_rounds(rounds):
                rejected.append(dict(name=name, reason='missing/invalid paired end-to-end acceptance'))
            else:
                measured[name] = [statistics.median(row) for row in rounds]
                if 'vs_direct_round_samples_us' in record:
                    comparison = record['vs_direct_round_samples_us']
                    if not valid_rounds(comparison):
                        raise ValueError('invalid paired direct eager measurement')
                    direct_comparisons[name] = [statistics.median(row) for row in comparison]
                if name == 'eager' and 'direct_eager_round_samples_us' in record:
                    control = record['direct_eager_round_samples_us']
                    if not valid_rounds(control):
                        raise ValueError('invalid direct eager control')
                    direct = [statistics.median(row) for row in control]
        if 'eager' not in measured:
            raise RuntimeError('validated paired eager end-to-end baseline is required')
        minimum_gain = 0.03
        def improves(name, baseline):
            return all(a <= b * (1 - minimum_gain)
                       for a, b in zip(measured[name], measured[baseline]))
        baseline = 'inductor' if 'inductor' in measured and improves('inductor', 'eager') else 'eager'
        accepted = {baseline}
        for _, name, _ in valid:
            if name not in measured or name == baseline:
                continue
            if name not in ('eager', 'inductor') and improves(name, baseline):
                accepted.add(name)
            else:
                rejected.append(dict(name=name, reason='no stable end-to-end gain over ' + baseline))
        direct_wins = []
        if direct is not None:
            direct_wins = sorted(name for name in accepted if all(
                a <= b * (1 - minimum_gain) for a, b in zip(direct_comparisons.get(name, measured[name]), direct)))
            # Direct eager is an external control: returning the FX eager plan
            # still incurs the module entry. Keep the fastest public route if
            # none reaches the direct control, and expose that outcome.
            if direct_wins:
                rejected.extend(dict(name=name, reason='no stable end-to-end gain over direct eager')
                                for name in sorted(accepted.difference(direct_wins)))
                accepted = set(direct_wins)
        acceptance = dict(protocol=e2e_protocol, baseline=baseline,
                          ranking='mean_of_two_round_medians_us',
                          minimum_gain=minimum_gain, round_medians_us=measured,
                          accepted=sorted(accepted))
        if direct is not None:
            acceptance['direct_eager'] = dict(round_medians_us=direct, faster_routes=direct_wins,
                                              candidate_medians_us=direct_comparisons)
        valid = [x for x in valid if x[1] in accepted]
    def rank(item):
        gpu_us, name, _ = item
        latency = statistics.mean(measured[name]) if require_e2e else gpu_us
        return latency, name not in ('eager', 'inductor'), name

    winner = min(valid, key=rank)
    return allowed[winner[1]], {'selected': winner[1], 'calibration_gpu_us': winner[0],
                              'context_key': key, 'rejections': rejected,
                              'e2e_acceptance': acceptance,
                              'fallback': winner[1] in ('eager', 'inductor')}


def calibrate_resources(gm, inputs, *, tolerance, environment_key, single_kernel=None, split_reductions=False, split_gelu=False, records=None, audit=None, prepared_candidates=None, prepared_measurements=None, public_timer=None):
    """Prepare a validated warm callable; perform calibration only outside capture.

    Passing records enables isolated offline calibration. These records must be
    from this exact context and threshold; selected code is revalidated after
    compilation. The ordinary online path creates equivalent records itself.
    """
    import torch
    from ._resource_codegen import baidu_target
    if not torch.is_grad_enabled():
        inputs = tuple(x.detach() if isinstance(x, torch.nn.Parameter) else x for x in inputs)
    if torch.is_grad_enabled() or any(getattr(x, 'requires_grad', False) for x in inputs):
        raise RuntimeError('resource autotuning supports no-grad inference only')
    if baidu_target() is None:
        raise RuntimeError('resource autotuning requires the actual P800 backend')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('warm/calibrate resource backend before CUDA Graph capture')
    from . import _p800_shared_fp32
    options = dict(tolerance=tolerance, environment_key=environment_key, single_kernel=single_kernel,
                   split_reductions=split_reductions, split_gelu=split_gelu, audit=audit,
                   prepared_candidates=prepared_candidates, prepared_measurements=prepared_measurements,
                   public_timer=public_timer)
    if records is None:
        core = _p800_shared_fp32.region(gm, inputs)
        if core is not None:
            return _p800_shared_fp32.prepare(core, inputs, **options)
    return _calibrate_graph(gm, inputs, records=records, **options)


def _calibrate_graph(gm, inputs, *, tolerance, environment_key, single_kernel, split_reductions,
                     split_gelu, records, audit, prepared_candidates, prepared_measurements, public_timer, verify=None):
    from ._resource_benchmark import measure_candidates
    from ._p800_fp32_region import normalized
    source = gm
    gm = normalized(source, inputs)
    plans = candidate_plans(source, inputs, single_kernel=single_kernel, split_reductions=split_reductions,
                            split_gelu=split_gelu)
    key = context_key(gm, inputs, environment_key)
    compiled = {}
    offline = records
    eligible = plans
    if records is not None:
        # Offline measurements admit safe candidates, but cannot certify the
        # speed of a newly retuned binary. Recalibrate the actual callables and
        # retain them for both selection and independent final measurement.
        _, admission = select_measurement(plans, records, key, tolerance, require_e2e=False)
        rejected = {r['name'] for r in admission['rejections']}
        measured = {r.get('name') for r in records}
        eligible = tuple(p for p in plans if p.name in measured and p.name not in rejected)
    records = []
    proofs = {}
    for plan in eligible:
        record = dict(name=plan.name, context_key=key, tolerance=dict(tolerance))
        try:
            fn, proof = compile_candidate(source, inputs, plan)
            if verify is not None and plan.name != 'eager':
                verify(fn)
            compiled[plan.name] = fn
            proofs[plan.name] = proof
            record['proof'] = proof
        except Exception as exc:
            record.update(status='rejected', error=type(exc).__name__ + ': ' + str(exc))
        records.append(record)
    oracle = gm
    if offline is not None:
        from ._p800_shared_fp32 import _FP32Reference, region
        core = region(source, inputs)
        if core is not None:
            oracle = _FP32Reference(core, inputs[0].dtype)
    measured = measure_candidates(compiled, oracle, inputs, tolerance, retain=prepared_measurements, public_timer=public_timer)
    for record in records:
        name = record['name']
        if name in measured:
            record.update(measured[name])
        if record.get('status') == 'passed' and prepared_candidates is not None:
            prepared_candidates[name] = (compiled[name], proofs[name])
    eager_record = next(record for record in records if record['name'] == 'eager') if offline is None else None
    if eager_record is not None and eager_record.get('status') != 'passed':
        decision = dict(selected='eager', fallback=True, context_key=key,
                        reason='eager calibration unavailable', rejections=[
                            dict(name='eager', reason=eager_record.get('error', 'timing unavailable'))])
        selected = next(plan for plan in plans if plan.name == 'eager')
    else:
        protocol = 'public_module_ab_ba_v1' if public_timer else 'randomized_ab_ba_v1'
        selected, decision = select_measurement(plans, records, key, tolerance, e2e_protocol=protocol)
    fn = compiled[selected.name]
    if audit is not None:
        audit.append(dict(decision=decision, candidates=records, offline_measurements=offline))
    return fn, decision


def autotune_resource_backend(*, tolerance, environment_key, single_kernel=None, split_reductions=False, split_gelu=False, audit=None):
    """torch.compile backend: native guards, first ordinary-call calibration."""
    from torch._inductor.compile_fx import compile_fx
    from ._resource_codegen import baidu_target

    # Only cold calls use this thread-local handshake. A warmed graph is a
    # single direct call to the committed plan. Concurrent callers never see
    # the calibration thread's candidate.
    calibration = threading.local()

    def backend(gm, example_inputs):
        if baidu_target() is None:
            return compile_fx(gm, example_inputs)
        from ._p800_fp32_region import normalized
        source = gm
        gm = normalized(gm, example_inputs)
        prepared = None
        lock = threading.RLock()

        def prepare(inputs, public=None, module_args=None):
            nonlocal prepared
            from ._resource_benchmark import paired_end_to_end

            def time_public(candidates, graph_inputs, *, seed):
                def select(name):
                    if not calibration.reached:
                        raise RuntimeError("public calibration missed its compiled graph")
                    if name == 'direct_eager':
                        calibration.trial = None
                        return
                    calibration.trial = (run, candidates[name])
                    calibration.reached = False

                calibration.reached = True
                try:
                    functions = dict.fromkeys(candidates, public)
                    # Keep the candidate ranking series in its original order.
                    # Interleaving a new arm changes predecessor/successor
                    # effects enough to flip marginal P800 routes.
                    samples = paired_end_to_end(functions, module_args, seed=seed, before_call=select)
                    functions['direct_eager'] = public._orig_mod
                    samples['vs_direct'] = paired_end_to_end(functions, module_args,
                                                             seed=seed + 1, before_call=select)
                    if not calibration.reached:
                        raise RuntimeError("public calibration missed its compiled graph")
                    return samples
                finally:
                    calibration.trial = None

            with lock:
                if prepared is None:
                    prepared, _ = calibrate_resources(
                        source, inputs, tolerance=tolerance, environment_key=environment_key,
                        single_kernel=single_kernel, split_reductions=split_reductions, split_gelu=split_gelu, audit=audit,
                        public_timer=time_public if public is not None else None)

        def run(*inputs):
            if prepared is None:
                trial = getattr(calibration, 'trial', None)
                if trial is not None:
                    if trial[0] is not run:
                        raise RuntimeError("public calibration reached a different compiled graph")
                    calibration.reached = True
                    return trial[1](*inputs)
                pending = getattr(calibration, 'pending', None)
                if pending is not None:
                    from ._resource_fx import _analyze_graph
                    import torch
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError('warm/calibrate resource backend before CUDA Graph capture')
                    # Discovery executes eager once. Reject side-effecting
                    # graphs before running them on the user's real inputs.
                    _analyze_graph(gm)
                    # Dynamo must finish installing its guards before we can
                    # time the real OptimizedModule with its original arguments.
                    pending.append((prepare, inputs, lock))
                    return gm.forward(*inputs)
                prepare(inputs)
            return prepared(*inputs)
        run._flagtree_backend = backend
        return run

    def wrap_module(optimized):
        from ._resource_entry import attach_module_entry
        return attach_module_entry(optimized, backend, calibration)

    backend._flagtree_wrap_module = wrap_module
    return backend
