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
import threading

from ._resource_fx import build_region, context_fingerprint


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


def candidate_plans(gm, inputs, *, single_kernel=None, split_reductions=False):
    """Enumerate bounded F06 plans; speed assumptions never select the winner."""
    import torch
    nodes = [n for n in gm.graph.nodes if n.op in ('call_function', 'call_method')]
    whole = build_region(gm, 0, len(nodes))  # purity/support gate, before execution
    dots = [i for i, n in enumerate(nodes) if _matrix(n)]
    if single_kernel is None:
        matrices = [x for x in inputs if isinstance(x, torch.Tensor) and x.ndim == 2]
        if dots and len(matrices) >= 2:
            a, b = matrices[:2]
            single_kernel = a.shape[0] <= 64 and a.shape[1] <= 128 and b.shape[1] <= 1024
        else:
            single_kernel = bool(matrices) and all(x.shape[0] <= 64 and x.shape[1] <= 1024 for x in matrices)
    if type(single_kernel) is not bool:
        raise TypeError('single_kernel must be bool or None')
    if type(split_reductions) is not bool:
        raise TypeError('split_reductions must be bool')

    def segment(start, end, resource):
        return Segment(start, end, resource, tuple(n.name for n in nodes[start:end]))

    plans = [Candidate('eager'), Candidate('inductor')]
    if dots:
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
        if not any(x in targets for x in ('softmax', 'layer_norm')):
            plans.append(Candidate('sdnn', (segment(0, len(nodes), 'sdnn'),), single_kernel))
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
                    for consumer in ('cluster', 'sdnn'):
                        plans.append(Candidate(producer + '_' + consumer + '_at_1', (
                            segment(0, 1, producer), segment(1, 2, consumer))))
    assert tuple(n.name for n in nodes) == whole.node_ids
    return tuple(plans)


def context_key(gm, inputs, environment_key):
    from ._p800_compiler import policy_key
    # Include scheduler/validation/codegen source and the caller's device/SDK
    # identity. Profiles for another graph, shape, stride or dtype cannot match.
    from pathlib import Path
    source = b''.join((Path(__file__).parent / p).read_bytes() for p in (
        '_resource_autotune.py', '_resource_benchmark.py', '_resource_runtime.py', '_resource_fx.py',
        '_p800_matmul.py'))
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
        raise RuntimeError('dynamic launch schedule cannot certify single-kernel execution')
    return sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run'
               for n in ast.walk(body))


def compile_candidate(gm, inputs, candidate):
    """Compile only with Inductor; external calls stay explicitly unassigned."""
    from torch._inductor import config
    from ._resource_runtime import prepare_regions
    from ._p800_compiler import target_scope
    if candidate.name == 'eager':
        return gm.forward, {'route': 'eager', 'regions': [], 'compiler_options': {}}
    options = compiler_options()
    with target_scope(), config.patch(options):
        if candidate.name == 'inductor':
            from torch._inductor.compile_fx import compile_fx
            from ._p800_matmul import native_template_eligibility
            import torch
            device = next(x.device for x in inputs if isinstance(x, torch.Tensor))
            with native_template_eligibility(device):
                import copy
                # compile_fx normalizes output/pre-grad passes in place. Keep
                # the semantic reference graph intact for every candidate.
                native_gm = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
                fn = compile_fx(native_gm, list(inputs))
                fn(*inputs)  # finish native tuning while the ABI bridge is active
            return fn, {'route': 'inductor', 'mode': 'max-autotune-no-cudagraphs',
                        'resource_verified': False, 'compiler_options': options}
        # Prevent native addmm rewrites from erasing the original low-precision
        # MatMul -> cast -> Bias boundary. Native baseline retains its patterns.
        with config.patch({'pattern_matcher': False, 'benchmark_epilogue_fusion': True}):
            fn, proof = prepare_regions(gm, inputs, candidate.regions)
    proof.update(route=candidate.name, compiler_options=options,
                 candidate=asdict(candidate), single_kernel_required=candidate.single_kernel)
    counts = [launch_count(r['source_code']) for r in proof['regions']]
    proof['kernel_count'] = sum(counts)
    if candidate.single_kernel and proof['kernel_count'] != 1:
        raise RuntimeError(f'single kernel required, native wrapper launches {proof["kernel_count"]}')
    return fn, proof


CHECKS = ('ordinary', 'repeat', 'changed', 'special', 'immutable', 'graph_fresh_data')


def select_measurement(plans, records, key, tolerance, *, require_e2e=True):
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
            elif allowed[name].regions:
                proofs = record.get('proof', {}).get('regions', [])
                if len(proofs) != len(allowed[name].regions):
                    reason = 'missing region evidence'
                else:
                    for region, proof in zip(allowed[name].regions, proofs):
                        evidence = proof.get('kernel_evidence', [])
                        if (proof.get('resource_verified') is not True or proof.get('external_calls') != 0
                                or not evidence or any(e.get('effective') != region.resource for e in evidence)):
                            reason = 'unverified actual resource'
                if allowed[name].single_kernel and record.get('proof', {}).get('kernel_count') != 1:
                    reason = 'not a single kernel'
        if reason:
            rejected.append({'name': name, 'reason': reason})
        else:
            import statistics
            valid.append((statistics.median(record['gpu_samples_us']), name, record))
    if not valid:
        raise RuntimeError('no candidate passed unchanged numerical and timing gates')
    acceptance = None
    if require_e2e:
        # Both independent, interleaved ordinary-call rounds must show a
        # material gain. Final reporting samples are never used to select.
        import statistics
        measured = {}
        for gpu, name, record in valid:
            rounds = record.get('e2e_round_samples_us', [])
            if (record.get('e2e_protocol') != 'randomized_ab_ba_v1' or len(rounds) != 2
                    or any(len(row) < 128 or any(type(x) not in (int, float)
                        or not math.isfinite(x) or x <= 0 for x in row) for row in rounds)):
                rejected.append(dict(name=name, reason='missing/invalid paired end-to-end acceptance'))
            else:
                measured[name] = [statistics.median(row) for row in rounds]
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
        acceptance = dict(protocol='randomized_ab_ba_v1', baseline=baseline,
                          minimum_gain=minimum_gain, round_medians_us=measured,
                          accepted=sorted(accepted))
        valid = [x for x in valid if x[1] in accepted]
    # A tie retains a baseline. No transfer penalty is added: full-plan replay
    # already includes launches, materialization, dependency and layout costs.
    winner = min(valid, key=lambda x: (x[0], x[1] not in ('eager', 'inductor'), x[1]))
    return allowed[winner[1]], {'selected': winner[1], 'calibration_gpu_us': winner[0],
                              'context_key': key, 'rejections': rejected,
                              'e2e_acceptance': acceptance,
                              'fallback': not bool(allowed[winner[1]].regions)}


def calibrate_resources(gm, inputs, *, tolerance, environment_key, single_kernel=None, split_reductions=False, records=None, audit=None, prepared_candidates=None, prepared_measurements=None):
    """Prepare a validated warm callable; perform calibration only outside capture.

    Passing records enables isolated offline calibration. These records must be
    from this exact context and threshold; selected code is revalidated after
    compilation. The ordinary online path creates equivalent records itself.
    """
    import torch
    from ._resource_codegen import baidu_target
    from ._resource_benchmark import measure_candidates
    if torch.is_grad_enabled() or any(getattr(x, 'requires_grad', False) for x in inputs):
        raise RuntimeError('resource autotuning supports no-grad inference only')
    if baidu_target() is None:
        raise RuntimeError('resource autotuning requires the actual P800 backend')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('warm/calibrate resource backend before CUDA Graph capture')
    plans = candidate_plans(gm, inputs, single_kernel=single_kernel, split_reductions=split_reductions)
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
            fn, proof = compile_candidate(gm, inputs, plan)
            compiled[plan.name] = fn
            proofs[plan.name] = proof
            record['proof'] = proof
        except Exception as exc:
            record.update(status='rejected', error=type(exc).__name__ + ': ' + str(exc))
        records.append(record)
    measured = measure_candidates(compiled, gm, inputs, tolerance, retain=prepared_measurements)
    for record in records:
        name = record['name']
        if name in measured:
            record.update(measured[name])
        if record.get('status') == 'passed' and prepared_candidates is not None:
            prepared_candidates[name] = (compiled[name], proofs[name])
    selected, decision = select_measurement(plans, records, key, tolerance)
    fn = compiled[selected.name]
    if audit is not None:
        audit.append(dict(decision=decision, candidates=records, offline_measurements=offline))
    return fn, decision


def autotune_resource_backend(*, tolerance, environment_key, single_kernel=None, split_reductions=False, audit=None):
    """torch.compile backend: native guards, first ordinary-call calibration."""
    from torch._inductor.compile_fx import compile_fx
    from ._resource_codegen import baidu_target

    def backend(gm, example_inputs):
        if baidu_target() is None:
            return compile_fx(gm, example_inputs)
        prepared = None
        lock = threading.RLock()

        def run(*inputs):
            nonlocal prepared
            if prepared is None:
                with lock:
                    if prepared is None:
                        prepared, _ = calibrate_resources(gm, inputs, tolerance=tolerance,
                                                          environment_key=environment_key,
                                                          single_kernel=single_kernel, split_reductions=split_reductions,
                                                          audit=audit)
            return prepared(*inputs)
        return run
    return backend
