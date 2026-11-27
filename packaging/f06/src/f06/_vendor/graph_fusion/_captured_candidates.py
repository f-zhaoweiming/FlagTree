"""Complete candidates from captured dtypes, independent of the FP32 oracle."""
import copy
import time

import torch


def captured_model_candidate_routes(model, *, prepare, output_template, validation_inputs,
                                    validate, record, tolerance, progress=None,
                                    linear_schedules=('native', 'inductor', 'grouped'),
                                    pointwise_policies=('generated', 'native_mul', 'native'),
                                    resource_policy='auto', pruning='none', source_tensors=None,
                                    source_prepare=None, precision_score=None, max_precision_trials=64):
    from ._model_pipeline import compile_pipeline
    from ._model_pruning import ModelPruner, validate_survivors
    from ._whole_model_selection import ModelRoute
    if model.training or torch.is_grad_enabled() or not validation_inputs:
        raise ValueError('Use frozen inference with complete validation examples')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Candidate parameters must be frozen')
    if pruning not in ('none', 'calibrated'):
        raise ValueError('Unknown pruning method: '+str(pruning))
    if not linear_schedules or set(linear_schedules)-{'native','inductor','grouped'}:
        raise ValueError('Unknown captured Linear schedule')
    if not pointwise_policies or set(pointwise_policies)-{'generated','native_mul','native'}:
        raise ValueError('Unknown pointwise policy')
    started = time.perf_counter()
    record.update(state='captured_candidates', capture_source='original_dtype',
                  fp32_reference_role='external_complete_output_validator_only',
                  forced_fp32_inputs=False, forced_fp32_parameter_model=False,
                  low_precision_linear_output_forced_to_fp32=False,
                  parameter_dtypes=sorted({str(x.dtype) for x in model.parameters()}),
                  precision_policies=['preserve_operator','fuse_fp32_internals'],
                  implementations={}, pruning=dict(mode=pruning))
    pruner = (ModelPruner(record['pruning'], policies=pointwise_policies,
                         resource_policy=resource_policy, progress=progress)
              if pruning == 'calibrated' else None)
    routes, families = {}, []
    profile_models = {'': (model, prepare)}
    supplied = (source_tensors is not None, source_prepare is not None, precision_score is not None)
    if any(supplied) and not all(supplied):
        raise ValueError('Local search requires source tensors, original preprocessing and precision score together')
    if all(supplied):
        from ._local_precision import (LocalPrecisionSpace, discover_local_precision_profiles,
                                       refine_local_precision_profiles)
        local_record = record['local_precision_search'] = {}
        space = LocalPrecisionSpace(model, source_tensors=source_tensors, prepare=prepare,
            source_prepare=source_prepare, validation_inputs=validation_inputs, record=local_record)
        profiles = discover_local_precision_profiles(space, output_template=output_template,
            validation_inputs=validation_inputs, validate=validate, score=precision_score,
            record=local_record, max_trials=max_precision_trials, progress=progress)
        record['local_precision_profiles'] = profiles
        for index, profile in enumerate(profiles):
            profile_models['local_'+str(index)+'_'] = (space.build(profile['selected']), source_prepare)
        if profiles:
            fine_record = record['operator_precision_search'] = {}
            fine, seed = space.refined(profiles[0]['selected'], fine_record)
            fine_profiles = refine_local_precision_profiles(fine, seed,
                output_template=output_template, validation_inputs=validation_inputs,
                validate=validate, score=precision_score, record=fine_record,
                max_trials=max_precision_trials, progress=progress)
            record['operator_precision_profiles'] = fine_profiles
            for index, profile in enumerate(fine_profiles):
                profile_models['operator_'+str(index)+'_'] = (fine.build(profile['selected']), source_prepare)
        record['fp32_reference_role'] = 'external_validator; explicit_unrounded_values_for_local_regions_only'

    def create(precision, linear, policy, prefix=''):
        name = 'captured_'+prefix+precision+'_'+linear+'_'+policy
        if name not in routes:
            audit = []
            options = dict(linear_schedule=linear, pointwise_policy=policy, require_compiled=True)
            candidate_model, candidate_prepare = profile_models[prefix]
            call = compile_pipeline(candidate_model, prepare=candidate_prepare, output_template=output_template,
                precision_policy=precision, precision_options=options,
                resource_policy=resource_policy, tolerance=tolerance,
                environment_key='f06-original-dtype-'+name, audit=audit)
            routes[name] = ModelRoute(call)
            record['implementations'][name] = dict(precision_policy=precision, options=options,
                audit=audit, requires_complete_compiled_validation=True,
                capture_source='original_dtype', local_precision_profile=prefix or None)
        return name, routes[name].call

    for prefix in profile_models:
        candidate_model, candidate_prepare = profile_models[prefix]
        for precision in record['precision_policies']:
            for linear in linear_schedules:
                if linear == 'grouped' and not any(p.dtype == torch.float32 for p in candidate_model.parameters()):
                    # This schedule only packs frozen FP32 weights. It cannot
                    # introduce a new implementation for an all-half model.
                    continue
                policies = list(pointwise_policies)
                choices = None
                if pruner is not None and len(policies) > 2:
                    graphs = []
                    capture_started = time.perf_counter()
                    capture = compile_pipeline(candidate_model, prepare=candidate_prepare, output_template=output_template,
                        precision_policy=precision,
                        precision_options=dict(linear_schedule=linear, pointwise_policy='generated',
                                               require_compiled=True),
                        resource_policy=resource_policy, tolerance=tolerance, audit=[],
                        environment_key='f06-original-dtype-cost',
                        _cost_graph_observer=lambda gm, _: graphs.append(copy.deepcopy(gm)),
                        _capture_cost_graph=True)
                    args, kwargs = validation_inputs[0]
                    output = capture(*args, **kwargs)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    del output, capture
                    record['pruning']['capture_seconds'] += time.perf_counter()-capture_started
                    if not graphs: raise RuntimeError('Original-dtype cost capture produced no graph')
                    policies = pruner.shortlist(prefix+precision+'_'+linear, graphs)
                    family = record['pruning']['families'][-1]
                    family.update(precision_policy=precision, linear_schedule=linear, local_precision_profile=prefix or None)
                    choices = family['kept_by_resource']
                    del graphs
                families.append((prefix,precision,linear,choices))
                for policy in policies:
                    name, call = create(precision,linear,policy,prefix)
                    if choices is not None and resource_policy == 'auto':
                        call._f06_resource_allowlist = tuple(k for k in ('cluster','sdnn') if policy in choices[k])
                    record['implementations'][name]['allowed_resource_variants'] = call._f06_resource_allowlist
                if progress: progress()

    # Precision rejection must not turn a local cost shortlist into proof that
    # a whole family is unusable. Reopen omitted implementations, never widen
    # the entire input graph or substitute the FP32 reference as a candidate.
    recovery = record['pruning']['precision_recovery'] = []
    parent_checks = record['pruning']['resource_parent_preparations'] = []
    validation_started = time.perf_counter()
    for prefix,precision,linear,choices in families:
        if choices is None: continue
        for resource, policies in choices.items():
            if resource == 'auto' and resource_policy == 'auto':
                policies = [p for p in pointwise_policies if 'captured_'+prefix+precision+'_'+linear+'_'+p in routes]
            if len(policies) == len(pointwise_policies): continue
            row = dict(family=prefix+precision+'_'+linear,resource=resource,checks=[],reopened=[])
            recovery.append(row)
            candidates = []
            for policy in policies:
                name,parent = create(precision,linear,policy,prefix)
                is_parent = resource=='auto' or resource_policy!='auto'
                if not is_parent and not parent._f06_audit:
                    validate_survivors([(name,parent)],validation_inputs=validation_inputs,
                                       validate=validate,record=parent_checks)
                call = parent if is_parent else parent._f06_resource_variants()[resource]
                candidates.append((name if is_parent else name+'__'+resource,call))
            if validate_survivors(candidates,validation_inputs=validation_inputs,
                                  validate=validate,record=row['checks']):
                row['state']='valid_survivor'
            else:
                row['state']='reopened_after_all_survivors_rejected'
                choices[resource]=list(pointwise_policies)
                for policy in pointwise_policies:
                    if policy in policies: continue
                    name,parent = create(precision,linear,policy,prefix)
                    if resource_policy=='auto':
                        parent._f06_resource_allowlist=tuple(k for k in ('cluster','sdnn') if policy in choices[k])
                    record['implementations'][name]['allowed_resource_variants']=parent._f06_resource_allowlist
                    row['reopened'].append(name if resource=='auto' or resource_policy!='auto' else name+'__'+resource)
            if progress: progress()
    record['pruning'].update(survivor_validation_seconds=time.perf_counter()-validation_started,
        proposed_implementations=len(families)*len(pointwise_policies),
        kept_implementations=len(routes))
    # Whole-graph lowering can introduce costs outside the profitable region.
    # Enumerate local fusion as another complete-model implementation, keeping
    # vendor calls elsewhere. Do not prune it using whole-graph timings or
    # accept it on activation-only numerical checks.
    from ._local_fusion import local_fusion_candidate_routes
    local_record = record['local_fusion'] = {}
    local_routes = local_fusion_candidate_routes(model, prepare=prepare,
        validation_inputs=validation_inputs, record=local_record)
    for name, route in local_routes.items():
        assert name not in routes
        routes[name] = route
        record['implementations'][name] = dict(scope='complete_model_with_local_fusion',
            audit=route.call._f06_audit, capture_source='original_dtype',
            requires_complete_compiled_validation=True)
    from ._template_local_fusion import template_local_candidate_routes
    template_record = record['local_templates'] = {}
    template_routes = template_local_candidate_routes(model, prepare=prepare,
        validation_inputs=validation_inputs, record=template_record)
    for name, route in template_routes.items():
        assert name not in routes
        routes[name] = route
        record['implementations'][name] = dict(scope='complete_model_with_local_templates',
            audit=route.call._f06_audit, capture_source='original_dtype',
            requires_complete_compiled_validation=True)
    if progress:
        progress()
    record.update(state='compiled_proposals_ready',candidate_construction_seconds=time.perf_counter()-started)
    return routes
