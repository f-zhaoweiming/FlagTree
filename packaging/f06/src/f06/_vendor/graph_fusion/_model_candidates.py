"""Bounded, cold-only precision proposals and complete-model implementations.

Native full-output screening proposes partitions; it never admits a compiled
route. Every implementation must pass the caller's original-FP32 validator
again, and select_model_routes compares complete-model end-to-end time.
"""
import copy
import weakref

import torch


class _CalibrationLinear(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        from ._model_inductor import _ParameterPack
        self.original = original
        self.narrow = False
        indices = (1,) if original.bias is None else (1, 2)
        self.pack = _ParameterPack(self.compute, indices, transform=lambda x: x.half())

    @staticmethod
    def compute(x, weight, bias):
        return torch.nn.functional.linear(x.half(), weight, bias).float()

    def forward(self, x):
        if not self.narrow:
            return self.original(x)
        return self.pack(x, self.original.weight, self.original.bias)


def _joint_projection_groups(probe, aliases, prepare, validation_inputs):
    """Couple Q/K/V precision, not their execution or arithmetic semantics.

    Names only identify possible sibling roles. Require compatible Linears
    and the same tensor/version at every observed call on every example.
    Actual graph fusion still applies its independent motion/layout guards.
    """
    candidates = []
    for parent, module in probe.named_modules(remove_duplicate=False):
        for names in (('q_proj', 'k_proj', 'v_proj'), ('query', 'key', 'value'), ('q', 'k', 'v')):
            children = [module._modules.get(name) for name in names]
            if not all(type(child) is torch.nn.Linear for child in children):
                continue
            ids = tuple(id(child) for child in children)
            if len(set(ids)) != 3 or not all(i in aliases for i in ids):
                continue
            signatures = {(tuple(child.weight.shape), child.weight.dtype, child.weight.device,
                           child.bias is not None) for child in children}
            if len(signatures) == 1:
                candidates.append((parent, names, ids))
    if not candidates:
        return [], []
    observed = {i: [] for _, _, ids in candidates for i in ids}
    tokens, handles = {}, []
    next_token = 0

    def observe(module, args, kwargs):
        nonlocal next_token
        x = args[0] if len(args) == 1 and not kwargs else kwargs.get('input') if not args else None
        if not isinstance(x, torch.Tensor):
            observed[id(module)][-1].append(None)
            return
        key = id(x)
        if key not in tokens or tokens[key][0]() is not x:
            tokens[key] = (weakref.ref(x), next_token)
            next_token += 1
        try:
            version = x._version
        except RuntimeError:
            # Inference tensors have no mutation counter. Do not infer shared
            # unmodified values from object identity alone in that context.
            observed[id(module)][-1].append(None)
            return
        observed[id(module)][-1].append((tokens[key][1], version))

    try:
        for i in observed:
            handles.append(aliases[i][0].register_forward_pre_hook(observe, with_kwargs=True))
        for args, kwargs in validation_inputs:
            for calls in observed.values():
                calls.append([])
            prepared, keywords = prepare(*args, **kwargs)
            output = probe(*prepared, **keywords)
            del output
            tokens.clear()
    finally:
        for handle in handles:
            handle.remove()
    groups, records = [], []
    for parent, names, ids in candidates:
        first = observed[ids[0]]
        valid = (all(calls and all(call is not None for call in calls) for calls in first)
                 and all(observed[i] == first for i in ids[1:]))
        records.append(dict(parent=parent, roles=list(names), joint=valid,
                            reason='same_input_version_all_examples' if valid else 'different_or_unverified_inputs'))
        if valid:
            groups.append(ids)
    return groups, records


def _profile_groups(paths, precision_groups):
    selected = set(paths)
    groups = []
    for group in precision_groups:
        overlap = selected.intersection(group)
        if overlap:
            if overlap != set(group):
                raise ValueError('A precision profile splits an indivisible Q/K/V or alias group')
            groups.append(group)
    if {path for group in groups for path in group} != selected:
        raise ValueError('A precision profile contains undiscovered modules')
    return groups


def discover_precision_partitions(model, *, prepare, output_template,
                                  validation_inputs, validate, record,
                                  max_trials=128, max_profiles=2, progress=None):
    """Greedily refine failed groups of native Linear precision boundaries.

    There are no model names, layer numbers or accuracy thresholds here.
    A shared/tied module and verified sibling Q/K/V are indivisible groups.
    FP32 is the reference, not a required execution dtype: proposed Linears
    execute the vendor FP16 operation directly. The original model and
    parameter dtype are never mutated. The finite search is not exhaustive;
    native-screen failures are explicitly not proofs about compiled accuracy.
    """
    from ._whole_model_selection import output_cast_like
    if (model.training or torch.is_grad_enabled() or not validation_inputs or
            type(max_trials) is not int or max_trials < 1 or
            type(max_profiles) is not int or max_profiles < 1):
        raise ValueError('Use frozen inference, validation examples and positive search limits')
    tensors = (*model.parameters(), *model.buffers())
    if any(x.is_floating_point() and x.dtype != torch.float32 for x in tensors):
        raise ValueError('Precision discovery requires original FP32 parameters and buffers')
    if any(x.requires_grad for x in model.parameters()):
        raise ValueError('Precision discovery requires frozen parameters')
    # Copy module structure while retaining the original FP32 masters. Only
    # this private probe gets wrapper modules; it is never returned as a route.
    probe = copy.deepcopy(model, {id(x): x for x in tensors})
    aliases = {}
    for path, module in probe.named_modules(remove_duplicate=False):
        if path and type(module) is torch.nn.Linear:
            aliases.setdefault(id(module), (module, []))[1].append(path)
    joints, joint_records = _joint_projection_groups(probe, aliases, prepare, validation_inputs)
    roots = {i: i for i in aliases}

    def root(i):
        while roots[i] != i:
            i = roots[i]
        return i

    for joint in joints:
        for i in joint[1:]:
            roots[root(i)] = root(joint[0])
    combined = {}
    for i, (original, paths) in aliases.items():
        wrapper = _CalibrationLinear(original)
        for path in paths:
            parent, _, name = path.rpartition('.')
            probe.get_submodule(parent)._modules[name] = wrapper
        group = combined.setdefault(root(i), dict(paths=[], weight_elements=0, wrappers=[]))
        group['paths'].extend(paths)
        group['weight_elements'] += original.weight.numel()
        group['wrappers'].append(wrapper)
    groups = list(combined.values())
    for group in groups:
        group['paths'] = tuple(sorted(group['paths']))
    groups.sort(key=lambda x: (-x['weight_elements'], x['paths']))
    record.update(state='native_precision_screening', trials=[],
                  linear_groups=len(groups), max_trials=max_trials,
                  linear_modules=len(aliases), joint_qkv=joint_records,
                  precision_groups=[list(group['paths']) for group in groups],
                  scope='bounded native full-output proposals; compiled routes must be revalidated',
                  original_fp32_sources_retained=True)
    accepted = set()
    passing = []
    pending = [tuple(range(len(groups)))] if groups else []
    while pending and len(record['trials']) < max_trials:
        chunk = pending.pop(0)
        proposed = accepted | set(chunk)
        for i, group in enumerate(groups):
            for wrapper in group['wrappers']:
                wrapper.narrow = i in proposed
        paths = sorted(p for i in proposed for p in groups[i]['paths'])
        name = 'precision_probe_' + str(len(record['trials']))
        error = None
        for args, kwargs in validation_inputs:
            prepared, keywords = prepare(*args, **kwargs)
            from torch.utils._pytree import tree_leaves
            if any(isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float32
                   for x in tree_leaves((prepared, keywords))):
                raise ValueError('Precision discovery requires original FP32 prepared inputs')
            output = output_cast_like(probe(*prepared, **keywords), output_template)
            try:
                validate(name, output, args, kwargs)
            except (AssertionError, ValueError) as exc:
                error = str(exc)
                break
            finally:
                del output
        passed = error is None
        row = dict(name=name, fp16_modules=paths, passed=passed,
                   weight_elements=sum(groups[i]['weight_elements'] for i in proposed))
        if error is not None:
            row['error'] = error
        record['trials'].append(row)
        if passed:
            accepted = proposed
            passing.append(row)
        elif len(chunk) > 1:
            middle = len(chunk) // 2
            pending[:0] = [chunk[:middle], chunk[middle:]]
        if progress is not None:
            progress()
    # Keep differing partitions, with coverage only a proposal priority, not
    # a performance score. The all-FP32 implementation remains a candidate.
    profiles = [dict(fp16_modules=[], fp16_ops=['linear'], origin='original_fp32')]
    for row in sorted(passing, key=lambda x: -x['weight_elements'])[:max_profiles]:
        profiles.append(dict(fp16_modules=row['fp16_modules'], fp16_ops=['linear'],
                             origin=row['name']))
    record.update(state='proposals_ready', profiles=profiles,
                  pending_groups=len(pending), budget_exhausted=bool(pending),
                  screening_failure_is_not_compiled_rejection=True)
    if progress is not None:
        progress()
    return profiles


def _refine_compiled_partition(groups, build, *, validation_inputs, validate,
                               record, max_trials, progress=None):
    """Feed compiled full-output failures back into a bounded partition search.

    Native screening cannot predict the error of generated normalization and
    pointwise code. Refine the proposed groups with the actual compiled graph;
    never treat passing native output as an admission decision.
    """
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError('Use a positive compiled refinement budget')
    accepted = ()
    accepted_call = None
    pending = [tuple(range(len(groups)))] if groups else []
    record.update(trials=[], max_trials=max_trials)
    while pending and len(record['trials']) < max_trials:
        chunk = pending.pop(0)
        proposed = tuple(sorted(set(accepted) | set(chunk)))
        paths = sorted(path for index in proposed for path in groups[index])
        name = 'compiled_precision_probe_' + str(len(record['trials']))
        call = build(paths, name)
        error = None
        for args, kwargs in validation_inputs:
            output = call(*args, **kwargs)
            try:
                validate(name, output, args, kwargs)
            except (AssertionError, ValueError) as exc:
                error = str(exc)
                break
            finally:
                del output
        row = dict(name=name, fp16_modules=paths, passed=error is None)
        if error is not None:
            row['error'] = error
        record['trials'].append(row)
        if error is None:
            accepted, accepted_call = proposed, call
        elif len(chunk) > 1:
            middle = len(chunk) // 2
            pending[:0] = [chunk[:middle], chunk[middle:]]
        del call
        if progress is not None:
            progress()
    paths = sorted(path for index in accepted for path in groups[index])
    record.update(fp16_modules=paths, pending_groups=len(pending),
                  budget_exhausted=bool(pending))
    return paths, accepted_call


def model_candidate_routes(model, *, prepare, output_template, validation_inputs,
                           validate, record, tolerance, max_trials=128,
                           max_profiles=2, max_compiled_trials=64, progress=None,
                           linear_schedules=None,
                           pointwise_policies=('generated', 'native_mul', 'native'),
                           resource_policy='auto', pruning='none',
                           capture_source='original_dtype', source_tensors=None,
                           source_prepare=None, precision_score=None):
    """Search the original execution graph; FP32 validation stays external.

    The historical FP32-rooted search requires explicit opt-in. Original-dtype
    search only uses explicitly supplied unrounded tensors for local precision
    upgrades. It never obtains a graph from a reference model.
    """
    if capture_source == 'original_dtype':
        from ._captured_candidates import captured_model_candidate_routes
        return captured_model_candidate_routes(model, prepare=prepare,
            output_template=output_template, validation_inputs=validation_inputs,
            validate=validate, record=record, tolerance=tolerance, progress=progress,
            linear_schedules=('native', 'inductor', 'grouped') if linear_schedules is None else linear_schedules,
            pointwise_policies=pointwise_policies, resource_policy=resource_policy, pruning=pruning,
            source_tensors=source_tensors, source_prepare=source_prepare,
            precision_score=precision_score, max_precision_trials=max_trials)
    if capture_source != 'legacy_fp32':
        raise ValueError('Unknown candidate capture source: '+str(capture_source))
    if linear_schedules is None:
        linear_schedules = ('native', 'grouped')
    from ._model_pipeline import compile_pipeline
    from ._whole_model_selection import ModelRoute
    import copy
    import time
    if pruning not in ('none', 'calibrated'):
        raise ValueError('Unknown model pruning method: ' + str(pruning))
    started = time.perf_counter()
    profiles = discover_precision_partitions(model, prepare=prepare,
        output_template=output_template, validation_inputs=validation_inputs,
        validate=validate, record=record, max_trials=max_trials,
        max_profiles=max_profiles, progress=progress)
    record['native_precision_search_seconds'] = time.perf_counter() - started
    if not linear_schedules or not pointwise_policies:
        raise ValueError('Provide Linear and pointwise implementations')
    representative = ('grouped' if 'grouped' in linear_schedules else linear_schedules[0],
                      'generated' if 'generated' in pointwise_policies else pointwise_policies[0])
    record['native_profiles'] = profiles
    record['compiled_refinement'] = []
    refine_started = time.perf_counter()
    refined = [profiles[0]]
    prepared = {}
    for profile in profiles[1:]:
        groups = _profile_groups(profile['fp16_modules'], record['precision_groups'])
        refinement = dict(representative_schedule=representative[0],
                          representative_pointwise=representative[1],
                          proposed_fp16_modules=profile['fp16_modules'])
        record['compiled_refinement'].append(refinement)
        record['state'] = 'compiled_precision_refinement'

        def build(paths, name):
            options = dict(fp16_modules=paths, fp16_ops=profile['fp16_ops'],
                           fp32_schedule=representative[0], pointwise_policy=representative[1])
            return compile_pipeline(model, prepare=prepare, output_template=output_template,
                source_precision='float32', precision_policy='source_fp32_mixed',
                precision_options=options, resource_policy=resource_policy,
                tolerance=tolerance, environment_key='f06-' + name, audit=[])

        paths, call = _refine_compiled_partition(groups, build,
            validation_inputs=validation_inputs, validate=validate,
            record=refinement, max_trials=max_compiled_trials, progress=progress)
        if paths and paths not in [item['fp16_modules'] for item in refined]:
            prepared[len(refined)] = call
            refined.append(dict(fp16_modules=paths, fp16_ops=profile['fp16_ops'],
                                origin='compiled_refinement_' + str(len(record['compiled_refinement']) - 1)))
    profiles = refined
    record['compiled_precision_search_seconds'] = time.perf_counter() - refine_started
    record.update(profiles=profiles, state='compiled_proposals_ready')
    routes = {}
    record['implementations'] = {}
    record['pruning'] = dict(mode=pruning)
    pruner = None
    if pruning == 'calibrated':
        from ._model_pruning import ModelPruner
        pruner = ModelPruner(record['pruning'], policies=pointwise_policies,
                            resource_policy=resource_policy, progress=progress)
    for index, profile in enumerate(profiles):
        for schedule in linear_schedules:
            policies = pointwise_policies
            resource_choices = None
            if pruner is not None and len(pointwise_policies) > 2:
                graphs = []
                options = {k: v for k, v in profile.items() if k != 'origin'}
                options.update(fp32_schedule=schedule, pointwise_policy='generated')
                capture_started = time.perf_counter()
                capture = compile_pipeline(model, prepare=prepare, output_template=output_template,
                    source_precision='float32', precision_policy='source_fp32_mixed',
                    precision_options=options, resource_policy=resource_policy,
                    tolerance=tolerance, environment_key='f06-cold-cost-capture', audit=[],
                    _cost_graph_observer=lambda gm, _: graphs.append(copy.deepcopy(gm)),
                    _capture_cost_graph=True)
                # This one-time graph inspection is never returned as a route.
                # Both FP32 parameter masters and public output casts remain
                # present in the graph used to construct local cost probes.
                args, kwargs = validation_inputs[0]
                output = capture(*args, **kwargs)
                import torch
                if torch.cuda.is_available(): torch.cuda.synchronize()
                del output, capture
                record['pruning']['capture_seconds'] += time.perf_counter() - capture_started
                if not graphs:
                    raise RuntimeError('Cost capture produced no graph')
                policies = pruner.shortlist(f'precision_{index}_{schedule}', graphs)
                family = record['pruning']['families'][-1]
                family.update(precision_index=index, linear_schedule=schedule)
                resource_choices = family['kept_by_resource']
                del graphs
            for pointwise in policies:
                name = f'auto_precision_{index}_{schedule}_{pointwise}'
                options = {k: v for k, v in profile.items() if k != 'origin'}
                options.update(fp32_schedule=schedule, pointwise_policy=pointwise)
                if index in prepared and (schedule, pointwise) == representative:
                    route = prepared[index]
                    audit = route._f06_audit
                else:
                    audit = []
                    route = compile_pipeline(model, prepare=prepare, output_template=output_template,
                        source_precision='float32', precision_policy='source_fp32_mixed',
                        precision_options=options, resource_policy=resource_policy,
                        tolerance=tolerance, environment_key='f06-model-candidates-' + name, audit=audit)
                if resource_choices is not None and resource_policy == 'auto':
                    # Keep the auto parent for the existing full-model resource
                    # gain gate. Expand only the resource/policy pairs retained
                    # in that resource's own shortlist: taking a union followed
                    # by a Cartesian expansion would undo the pruning.
                    route._f06_resource_allowlist = tuple(kind for kind in ('cluster','sdnn')
                        if pointwise in resource_choices[kind])
                routes[name] = ModelRoute(route)
                record['implementations'][name] = dict(options=options, audit=audit,
                    proposal_origin=profile['origin'], requires_complete_compiled_validation=True,
                    allowed_resource_variants=getattr(route, '_f06_resource_allowlist', None))
    if pruner is not None:
        from ._model_pruning import validate_survivors
        record['state'] = 'pruned_survivor_validation'
        recovery_started = time.perf_counter()
        recovery = record['pruning']['precision_recovery'] = []
        parent_checks = record['pruning']['resource_parent_preparations'] = []
        for family in record['pruning']['families']:
            index, schedule = family['precision_index'], family['linear_schedule']
            for kind, kept in family['kept_by_resource'].items():
                if kind == 'auto' and resource_policy == 'auto':
                    kept = [p for p in pointwise_policies
                            if f'auto_precision_{index}_{schedule}_{p}' in routes]
                if len(kept) == len(pointwise_policies):
                    continue
                row = dict(family=family['name'], resource=kind, checks=[], reopened=[])
                recovery.append(row)
                candidates = []
                for pointwise in kept:
                    name = f'auto_precision_{index}_{schedule}_{pointwise}'
                    parent = routes[name].call
                    is_parent = kind == 'auto' or resource_policy != 'auto'
                    if not is_parent and not parent._f06_audit:
                        # Resource variants derive their arithmetic graph from
                        # an executed parent. A cold parent has no such graph
                        # yet, including when all its auto policies were kept.
                        validate_survivors([(name, parent)], validation_inputs=validation_inputs,
                                           validate=validate, record=parent_checks)
                    call = parent if is_parent else parent._f06_resource_variants()[kind]
                    candidates.append((name if is_parent else name+'__'+kind, call))
                if validate_survivors(candidates, validation_inputs=validation_inputs,
                                      validate=validate, record=row['checks']):
                    row['state'] = 'valid_survivor'
                    continue
                # Local cost rankings do not establish full-model accuracy.
                # Restore the omitted domain if every retained implementation
                # of this precision/Linear/resource family fails its contract.
                row['state'] = 'reopened_after_all_survivors_rejected'
                family['kept_by_resource'][kind] = list(pointwise_policies)
                for pointwise in pointwise_policies:
                    if pointwise in kept: continue
                    name = f'auto_precision_{index}_{schedule}_{pointwise}'
                    if name not in routes:
                        profile = profiles[index]
                        options = {k:v for k,v in profile.items() if k!='origin'}
                        options.update(fp32_schedule=schedule, pointwise_policy=pointwise)
                        audit = []
                        parent = compile_pipeline(model, prepare=prepare, output_template=output_template,
                            source_precision='float32', precision_policy='source_fp32_mixed',
                            precision_options=options, resource_policy=resource_policy,
                            tolerance=tolerance, environment_key='f06-model-candidates-'+name, audit=audit)
                        if resource_policy == 'auto':
                            parent._f06_resource_allowlist = tuple(r for r in ('cluster','sdnn')
                                if pointwise in family['kept_by_resource'][r])
                        routes[name] = ModelRoute(parent)
                        record['implementations'][name] = dict(options=options, audit=audit,
                            proposal_origin=profile['origin'], requires_complete_compiled_validation=True,
                            recovered_after_precision_rejection=True)
                    parent = routes[name].call
                    if kind != 'auto' and resource_policy == 'auto':
                        parent._f06_resource_allowlist = tuple(dict.fromkeys(
                            (*parent._f06_resource_allowlist, kind)))
                    record['implementations'][name]['allowed_resource_variants'] = parent._f06_resource_allowlist
                    row['reopened'].append(name if kind == 'auto' or resource_policy != 'auto' else name+'__'+kind)
                if progress: progress()
        record['pruning']['survivor_validation_seconds'] = time.perf_counter() - recovery_started
        record['state'] = 'compiled_proposals_ready'
    record['candidate_construction_seconds'] = time.perf_counter() - started
    record['pruning'].update(proposed_implementations=len(profiles)*len(linear_schedules)*len(pointwise_policies),
                             kept_implementations=len(routes))
    return routes
