"""Cold, calibrated pruning of pointwise implementations in complete graphs.

This is a heuristic over an explicit finite domain, not a lower bound or a
proof of global optimality. Precision and contraction schedules are never
pruned here. Unknown calibration domains keep all alternatives.
"""
import copy
import hashlib
import json
import math
import statistics
import time
import types

import torch
from torch import fx


def _spec(value):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError('Cost probes require strided tensor boundaries')
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                           torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('Unsupported calibration boundary dtype')
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                device=str(value.device), requires_grad=False)


def extract_regions(gm):
    """Extract actual connected choices and adjacent explicit dtype casts.

    Opaque public output casts stay in the expression as opaque operations;
    they must not be credited as eliminated launches. Reduction, matrix and
    other compute boundaries are not replaced by pointwise cost guesses.
    """
    from ._operator_precision import _value
    from ._pointwise_candidates import preserve_pointwise
    domain = preserve_pointwise(gm, 'native', dry_run=True)
    names = {r['node'] for r in domain}
    nodes = list(gm.graph.nodes)
    selected = {n for n in nodes if n.name in names}
    def cast(n):
        return (n.op == 'call_method' and n.target in ('to', 'half', 'float', 'bfloat16') or
                n.op == 'call_function' and (n.target == torch.ops.aten._to_copy.default or
                                             'final_cast' in str(n.target)))
    # Include neighboring conversions, both before and after arithmetic. Do
    # not silently turn a conversion into a preprepared input or omit output.
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n not in selected and cast(n) and any(
                    x in selected for x in (*n.all_input_nodes, *n.users)):
                selected.add(n); changed = True
    # Direct residual edges do not imply fusibility. A selected add can feed
    # both the next add and an opaque norm/Linear chain producing its other
    # operand. Contracting those adds would introduce a dependency cycle.
    # Track paths that leave the proposed domain and forbid re-entry inside
    # a component: every extracted region must be convex in the full DAG.
    ancestors, external = {}, {}
    for n in nodes:
        ancestors[n], external[n] = set(), set()
        for parent in n.all_input_nodes:
            ancestors[n].update(ancestors[parent])
            external[n].update(external[parent])
            if parent in selected:
                ancestors[n].add(parent)
            else:
                external[n].update(ancestors[parent])
    owner = {n: n for n in selected}
    groups = {n: {n} for n in selected}
    for n in nodes:
        if n not in selected: continue
        for parent in n.all_input_nodes:
            if parent not in selected or owner[parent] == owner[n]: continue
            left, right = owner[parent], owner[n]
            merged = groups[left] | groups[right]
            if any(external[v] & merged for v in merged):
                continue
            groups[left] = merged
            del groups[right]
            for v in merged: owner[v] = left
    components = [groups[n] for n in nodes if n in groups]
    result = []
    for group in components:
        graph = fx.Graph(); env = {}; inputs = []; specs = []
        ordered = [n for n in nodes if n in group]
        if len(ordered) > 64:
            raise ValueError('Pointwise component exceeds the bounded calibration domain')
        for n in ordered:
            for source in n.all_input_nodes:
                if source not in group and source not in env:
                    placeholder = graph.placeholder('arg' + str(len(inputs)))
                    placeholder.meta = dict(source.meta)
                    env[source] = placeholder; inputs.append(source)
                    specs.append(_spec(_value(source)))
            env[n] = graph.node_copy(n, lambda x: env[x])
        outputs = [n for n in ordered if any(user not in group for user in n.users)]
        graph.output(tuple(env[n] for n in outputs))
        region = fx.GraphModule({}, graph)
        # Canonical node names make repeated layers share measurements. The
        # graph text includes constants/casts, and specs include every stride.
        canonical = copy.deepcopy(region)
        for index, n in enumerate(canonical.graph.nodes):
            if n.op != 'output': n.name = 'v' + str(index)
        canonical.recompile()
        identity = dict(code=canonical.code, inputs=specs,
                        outputs=[_spec(_value(n)) for n in outputs])
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        result.append(dict(module=region, signature=digest, identity=identity,
                           nodes=[n.name for n in ordered]))
    return result, domain


def choose_policies(policies, scores, *, keep=2, margin=.03):
    """Keep diversity and uncertain near ties for every execution resource."""
    if len(policies) <= keep or not scores:
        return list(policies)
    retained = {'generated'} & set(policies)
    for resource_scores in scores.values():
        if set(resource_scores) != set(policies) or any(
                not math.isfinite(v) or v <= 0 for v in resource_scores.values()):
            return list(policies)
        ranked = sorted(policies, key=lambda p: (resource_scores[p], policies.index(p)))
        retained.update(ranked[:keep])
        # The uncertainty band protects contenders for the optimum. Applying
        # it around the runner-up instead would keep arbitrarily slow tied
        # alternatives even when both are clearly separated from the winner.
        cutoff = resource_scores[ranked[0]] * (1 + margin)
        retained.update(p for p in policies if resource_scores[p] <= cutoff)
    return [p for p in policies if p in retained]


def validate_survivors(candidates, *, validation_inputs, validate, record):
    """Find a fully valid survivor, without swallowing execution failures."""
    for name, call in candidates:
        row = dict(route=name, passed=False)
        record.append(row)
        for args, kwargs in validation_inputs:
            output = call(*args, **kwargs)
            try:
                validate(name, output, args, kwargs)
            except (AssertionError, ValueError) as error:
                row['error'] = str(error)
                break
            finally:
                del output
        else:
            row['passed'] = True
            return True
    return False


class ModelPruner:
    def __init__(self, record, *, policies, resource_policy='auto', progress=None):
        self.record = record
        self.policies = tuple(policies)
        self.resources = ('auto', 'cluster', 'sdnn') if resource_policy == 'auto' else (resource_policy,)
        self.progress = progress
        self.cache = {}
        record.update(method='compiled_actual_regions_v1', families=[], calibration=[],
                      calibration_seconds=0., capture_seconds=0.,
                      complete_candidate_latency_used_for_ranking=False,
                      scope='pointwise policies within each precision/Linear family')

    def _measure(self, region, policy, resource):
        from ._model_inductor import inductor_backend
        # Use the same guarded Dynamo/AOT entry for native and generated
        # labels. Neither raw eager costs nor candidate whole-model timings
        # enter this model. Calibration caches live only in this search.
        args = []
        for spec in region['identity']['inputs']:
            dtype = getattr(torch, spec['dtype'].split('.')[-1])
            x = torch.empty_strided(spec['shape'], spec['stride'], dtype=dtype, device=spec['device'])
            # fill_ is defined for expanded/overlapping views too, and avoids
            # negative/domain-sensitive synthetic values at these boundaries.
            # Only pure add/mul/sigmoid and explicit conversions are extracted.
            # Integer/bool inputs here are conversion/mask data, never gather
            # indices or control flow. Preserve their dtype and cast launches.
            x.fill_(.25 if x.is_floating_point() else 1); args.append(x)
        class Boundary(torch.nn.Module):
            def __init__(self, body):
                super().__init__(); self.body = body
            def forward(self, *values):
                return self.body(*values)
        f = Boundary.forward
        Boundary.forward = types.FunctionType(f.__code__.replace(), f.__globals__, f.__name__, f.__defaults__, f.__closure__)
        audit = []
        fn = torch.compile(Boundary(region['module']).eval(), fullgraph=True, dynamic=False,
            backend=inductor_backend(tolerance=dict(atol=.003, rtol=.003),
                environment_key='f06-cold-region-cost', audit=audit,
                precision_policy='preserve_operator',
                precision_options=dict(pointwise_policy=policy, require_compiled=True),
                resource_policy=resource))
        started = time.perf_counter()
        for _ in range(3):
            value = fn(*args); torch.cuda.synchronize(); del value
        compile_seconds = time.perf_counter() - started
        samples = []
        for _ in range(10):
            torch.cuda.synchronize(); t = time.perf_counter_ns()
            value = fn(*args); torch.cuda.synchronize(); del value
            samples.append((time.perf_counter_ns() - t) / 1000)
        if not all(math.isfinite(v) and v > 0 for v in samples):
            raise ValueError('Invalid calibration timing')
        return dict(median_us=statistics.median(samples), samples_us=samples,
                    compile_seconds=compile_seconds)

    def shortlist(self, name, graphs):
        family = dict(name=name, proposed=list(self.policies), kept=list(self.policies),
                      kept_by_resource={r:list(self.policies) for r in self.resources},
                      scores_us={}, regions=[], state='preparing')
        self.record['families'].append(family)
        if len(self.policies) <= 2:
            family['state'] = 'small_domain_kept'; return list(self.policies)
        started = time.perf_counter()
        try:
            regions = []
            for gm in graphs:
                rows, domain = extract_regions(gm)
                regions.extend(rows)
            if not regions:
                family['state'] = 'empty_domain_kept'; return list(self.policies)
        except ValueError as error:
            family.update(state='unsupported_domain_kept', reason=str(error))
            return list(self.policies)
        # Bound fresh calibration work. Unknown signatures retain all paths.
        fresh = {r['signature'] for r in regions} - set(self.cache)
        if len(self.cache) + len(fresh) > 48:
            family['state'] = 'calibration_budget_kept'; return list(self.policies)
        scores = {resource: {policy: 0. for policy in self.policies} for resource in self.resources}
        for region in regions:
            key = region['signature']
            if key not in self.cache:
                entry = dict(signature=key, identity=region['identity'], costs={})
                self.record['calibration'].append(entry)
                for resource in self.resources:
                    entry['costs'][resource] = {}
                    for policy in self.policies:
                        entry['costs'][resource][policy] = self._measure(region, policy, resource)
                    if self.progress: self.progress()
                self.cache[key] = entry
            for resource in self.resources:
                for policy in self.policies:
                    scores[resource][policy] += self.cache[key]['costs'][resource][policy]['median_us']
            family['regions'].append(dict(signature=key, nodes=region['nodes']))
        kept = choose_policies(self.policies, scores)
        by_resource = {r:choose_policies(self.policies, {r:costs}) for r,costs in scores.items()}
        family.update(state='ranked_before_candidate_measurement', scores_us=scores, kept=kept,
                      kept_by_resource=by_resource,
                      auto_parents_retained_for_resource_gate=kept)
        self.record['calibration_seconds'] += time.perf_counter() - started
        if self.progress: self.progress()
        return kept
