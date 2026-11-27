"""Cold, bounded precision refinement starting with the original dtype model.

Source tensors supply unrounded values, never a second graph or reference
forward. Precision regions keep their original Python operations. Every
compiled implementation still needs the external complete-output validator.
"""
import copy
import math
import operator
import time

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def _state(model):
    return dict(list(model.named_parameters(remove_duplicate=False)) +
                list(model.named_buffers(remove_duplicate=False)))


class _InputBoundary(torch.nn.Module):
    def __init__(self, module, contract, promoted):
        super().__init__()
        self.region = module
        self.contract = contract
        self.promoted = promoted

    def forward(self, *args, **kwargs):
        values, spec = tree_flatten((args, kwargs))
        expected, dtypes = self.contract
        if spec != expected or len(values) != len(dtypes):
            raise ValueError('Precision-region input structure changed; recalibrate')
        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                value = values[i]
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    raise ValueError('Precision-region input kind changed')
                target = torch.float32 if self.promoted else dtype
                if value.dtype != target:
                    values[i] = value.to(dtype=target)
        prepared, keywords = tree_unflatten(values, spec)
        # No return cast: adjacent promoted regions share FP32 intermediates.
        # The next native region restores its own captured input contract.
        return self.region(*prepared, **keywords)


class LocalPrecisionSpace:
    def __init__(self, model, *, source_tensors, prepare, source_prepare,
                 validation_inputs, record, granularity='block', _captured_graphs=None):
        if model.training or torch.is_grad_enabled() or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Local precision search requires frozen inference')
        if not validation_inputs or not callable(source_prepare):
            raise ValueError('Original unrounded preprocessing and examples are required')
        if granularity not in ('block', 'operator'):
            raise ValueError('Unknown precision-region granularity')
        self.model, self.prepare = model, source_prepare
        self.native_prepare = prepare
        self.validation_inputs = validation_inputs
        self.sources = dict(source_tensors)
        self.original = _state(model)
        self.record = record
        self.groups, self.contracts, self.boundaries = {}, {}, set()
        # Tied state needs an indivisible precision group. Reject rather than
        # silently assigning two precisions to a shared mutable tensor.
        ids = [id(v) for v in self.original.values()]
        if len(ids) != len(set(ids)):
            raise ValueError('Tied parameter/buffer precision regions need explicit grouping')
        for name, value in self.original.items():
            if not value.is_floating_point():
                continue
            source = self.sources.get(name)
            if (not isinstance(source, torch.Tensor) or source.dtype != torch.float32 or
                    source.shape != value.shape or source.device != value.device or source.requires_grad):
                raise ValueError('Missing frozen original FP32 source: '+name)
            if not torch.equal(value, source.to(dtype=value.dtype)):
                raise ValueError('Source is not the original value of captured state: '+name)

        def visit(path, module):
            children = list(module.named_children())
            repeated = isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential))
            has_repeated = any(isinstance(v, (torch.nn.ModuleList, torch.nn.Sequential))
                               for v in module.modules() if v is not module)
            names = [n for n, v in self.original.items() if v.is_floating_point() and
                     (n.startswith(path+'.') if path else True)]
            if path and names and not repeated and not has_repeated:
                own_state = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
                state_children = any(any(v.is_floating_point() for v in
                    list(child.parameters()) + list(child.buffers())) for _, child in children)
                # Preserve modules mixing direct state and child operations:
                # their internal functional dtype joins need an explicit
                # boundary before they can safely be split further.
                refine = (granularity == 'operator' and state_children and
                          not any(v.is_floating_point() for _, v in own_state))
                if not refine:
                    self.groups[path] = dict(kind='module', path=path, tensors=names,
                                             coarse=path.split('.')[0])
                    return
                self.boundaries.add(path)
            for name, value in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
                if value.is_floating_point():
                    full = path+'.'+name if path else name
                    self.groups['state:'+full] = dict(kind='state', path=full,
                        tensors=[full], coarse='state:'+full)
            for name, child in children:
                visit(path+'.'+name if path else name, child)
        visit('', model)
        assigned = [n for group in self.groups.values() for n in group['tensors']]
        assert len(assigned) == len(set(assigned))
        assert set(assigned) == {n for n, v in self.original.items() if v.is_floating_point()}
        handles, active_inputs = [], {}
        def observe(path):
            def hook(module, args, kwargs):
                values, spec = tree_flatten((args, kwargs))
                contract = (spec, tuple(v.dtype if isinstance(v, torch.Tensor) and v.is_floating_point()
                                        else None for v in values))
                if path in self.contracts and self.contracts[path] != contract:
                    raise ValueError('Precision-region contract varies across calls: '+path)
                self.contracts[path] = contract
                try:
                    versions = [(v, v._version) for v in values if isinstance(v, torch.Tensor)]
                except RuntimeError as exc:
                    raise ValueError('Cannot verify precision-region input mutation: '+path) from exc
                active_inputs.setdefault(path, []).append(versions)
            return hook
        def check_effects(path):
            def hook(module, args, kwargs, result):
                versions = active_inputs[path].pop()
                if any(v._version != version for v, version in versions):
                    raise ValueError('Precision-region mutates its input: '+path)
                values, _ = tree_flatten(result)
                inputs = {v.untyped_storage().data_ptr() for v, _ in versions}
                if any(isinstance(v, torch.Tensor) and v.is_floating_point() and
                       v.untyped_storage().data_ptr() in inputs for v in values):
                    raise ValueError('Precision-region returns a floating input alias: '+path)
            return hook
        try:
            paths = {k:g['path'] for k,g in self.groups.items() if g['kind']=='module'}
            paths.update({path:path for path in self.boundaries})
            for key, path in paths.items():
                handles.append(model.get_submodule(path).register_forward_pre_hook(observe(key), with_kwargs=True))
                handles.append(model.get_submodule(path).register_forward_hook(check_effects(key), with_kwargs=True))
            for args, kwargs in validation_inputs:
                prepared, keywords = prepare(*args, **kwargs)
                result = model(*prepared, **keywords)
                del result
        finally:
            for handle in handles:
                handle.remove()
        missing = [k for k, v in self.groups.items() if v['kind'] == 'module' and k not in self.contracts]
        if missing:
            raise ValueError('Unobserved precision regions: '+str(missing))
        # Infer same-dtype joins from the original captured graph, before any
        # precision changes. For example, two projection outputs may meet at
        # an unwrapped final matmul. Their precision must be chosen together.
        from ._model import enable_vendor_capture
        from ._source_precision import module_path
        enable_vendor_capture()
        self.joints, joins = [], []
        same_dtype = {torch.matmul, torch.mm, torch.bmm, operator.matmul, torch.cat, torch.stack,
                      torch.nn.functional.scaled_dot_product_attention,
                      torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
                      torch.ops.aten.matmul.default, torch.ops.aten.cat.default, torch.ops.aten.stack.default}
        def inspect_graph(gm, inputs):
            origins = {}
            for node in gm.graph.nodes:
                path = module_path(node)
                owner = next((key for key, group in self.groups.items() if group['kind']=='module' and
                              (path == group['path'] or path.startswith(group['path']+'.'))), None)
                upstream = set().union(*(origins.get(n,set()) for n in node.all_input_nodes))
                origins[node] = {owner} if owner is not None else upstream
                if owner is None and node.op=='call_function' and node.target in same_dtype and len(upstream)>1:
                    self.joints.append(upstream)
                    joins.append(dict(node=node.name,operator=str(node.target),regions=sorted(upstream)))
            return gm.forward
        self.captured_graphs = []
        if _captured_graphs is not None:
            for graph in _captured_graphs:
                inspect_graph(graph, None)
            self.captured_graphs = list(_captured_graphs)
        else:
            def capture_graph(gm, inputs):
                self.captured_graphs.append(gm)
                return inspect_graph(gm, inputs)
            captured = torch.compile(model, backend=capture_graph, fullgraph=True)
            for args, kwargs in validation_inputs:
                prepared, keywords = prepare(*args, **kwargs)
                result = captured(*prepared, **keywords)
                del result
            del captured
        record.update(source='original_dtype_module_structure', source_forward_used=False,
            granularity=granularity,
            refined_input_boundaries=sorted(self.boundaries),
            unrounded_state_verified=True, input_mutation_and_output_alias_checked=True,
            groups=copy.deepcopy(self.groups),
            same_dtype_joins=joins,
            native_input_contracts={k:[str(d) for d in v[1]] for k, v in self.contracts.items()},
            selected_regions=[], trials=[])
        # Zero changes must reproduce the captured application, including its
        # input narrowing. This prevents preprocessing from biasing screening.
        baseline = self.build(())
        for args, kwargs in validation_inputs:
            low_args, low_kw = prepare(*args, **kwargs)
            raw_args, raw_kw = source_prepare(*args, **kwargs)
            torch.testing.assert_close(baseline(*raw_args, **raw_kw), model(*low_args, **low_kw),
                                       atol=0, rtol=0)
        record['zero_promotion_bitwise_equal'] = True

    def refined(self, selected, record):
        """Refine structure from the original dtype model, never from an FP32 graph."""
        fine = LocalPrecisionSpace(self.model, source_tensors=self.sources,
            prepare=self.native_prepare, source_prepare=self.prepare,
            validation_inputs=self.validation_inputs, record=record, granularity='operator',
            _captured_graphs=self.captured_graphs)
        promoted = {name for key in self.close_selected(selected) for name in self.groups[key]['tensors']}
        seed = {key for key, group in fine.groups.items() if set(group['tensors']) <= promoted}
        covered = {name for key in seed for name in fine.groups[key]['tensors']}
        if covered != promoted:
            raise ValueError('Refined precision groups must partition the original promoted state')
        return fine, fine.close_selected(seed)

    def close_selected(self, selected):
        selected = set(selected)
        while True:
            previous = set(selected)
            for joint in self.joints:
                if joint & selected:selected.update(joint)
            if selected == previous:return selected

    def remove_regions(self, selected, removed):
        return set(selected)-self.close_selected(removed)

    def build(self, selected):
        selected = self.close_selected(selected)
        if selected-set(self.groups):
            raise ValueError('Undiscovered precision region')
        # Copy structure, not full checkpoint storage. Only selected region
        # bindings change; the caller's original model remains untouched.
        model = copy.deepcopy(self.model, {id(v):v for v in self.original.values()})
        for key in selected:
            for name in self.groups[key]['tensors']:
                parent, _, field = name.rpartition('.')
                target = model.get_submodule(parent) if parent else model
                value = self.sources[name]
                if field in target._parameters:
                    if not isinstance(value, torch.nn.Parameter):
                        value = torch.nn.Parameter(value, requires_grad=False)
                    target._parameters[field] = value
                else:
                    target._buffers[field] = value
        for key, group in self.groups.items():
            if group['kind'] != 'module':
                continue
            path = group['path']; parent, _, field = path.rpartition('.')
            container = model.get_submodule(parent) if parent else model
            child = container._modules[field]
            container._modules[field] = _InputBoundary(child, self.contracts[key], key in selected).eval()
        # Wrapping leaves alone leaves raw source inputs on residual bypasses.
        # Preserve each split parent's original boundary when all descendants
        # stay native; promote its residual stream only with a selected child.
        for path in sorted(self.boundaries, key=lambda p:p.count('.'), reverse=True):
            parent, _, field = path.rpartition('.')
            container = model.get_submodule(parent) if parent else model
            promoted = any(self.groups[key]['path'].startswith(path+'.') for key in selected)
            container._modules[field] = _InputBoundary(container._modules[field],
                self.contracts[path], promoted).eval()
        return model.eval().requires_grad_(False)


def discover_local_precision_profiles(space, *, output_template, validation_inputs,
                                      validate, score, record, max_trials=64, progress=None):
    """Expand precision from zero, then refine only the newly selected regions.

    Coarse proposals are disjoint discovered branches. Once a complete-output
    pass is found, divide-and-test can restore native subregions. Failed native
    screens are not proofs about compiled accuracy. No source-model graph or
    hardcoded layer whitelist is used, and no probe is an admitted route.
    """
    from ._whole_model_selection import output_cast_like
    if type(max_trials) is not int or max_trials < 1 or not callable(score):
        raise ValueError('A finite trial budget and external precision score are required')
    started = time.perf_counter(); cache = {}; names = list(space.groups)
    def evaluate(selected, phase):
        key = tuple(sorted(space.close_selected(selected)))
        if key in cache:
            return cache[key]
        if len(record['trials']) >= max_trials:
            return None
        candidate = space.build(key)
        name = 'local_precision_probe_'+str(len(record['trials']))
        passed = True; errors = []; ranks = []
        for args, kwargs in validation_inputs:
            prepared, keywords = space.prepare(*args, **kwargs)
            output = output_cast_like(candidate(*prepared, **keywords), output_template)
            try:
                validate(name, output, args, kwargs)
            except (AssertionError, ValueError) as exc:
                passed = False; errors.append(str(exc)[:1000])
            rank = tuple(float(x) for x in score(name))
            if not rank or not all(math.isfinite(x) and x >= 0 for x in rank):
                raise ValueError('Precision ranking must contain finite nonnegative metrics')
            ranks.append(rank)
            del output
        row = dict(name=name, phase=phase, selected=list(key), passed=passed,
                   rank=list(max(ranks)), errors=errors,
                   promoted_state_elements=sum(space.original[n].numel() for k in key for n in space.groups[k]['tensors']))
        record['trials'].append(row);cache[key]=row
        del candidate
        if progress: progress()
        return row
    accepted = set(); initial = evaluate(accepted, 'original_dtype')
    record['original_passed'] = initial['passed']
    if initial['passed']:
        record.update(state='original_dtype_already_passed', seconds=time.perf_counter()-started)
        return []
    coarse = {}
    for key, group in space.groups.items():
        coarse.setdefault(group['coarse'], set()).add(key)
    passing = None
    while coarse and len(record['trials']) < max_trials:
        proposals = []
        for coarse_key, group in coarse.items():
            if group <= accepted:continue
            row = evaluate(accepted | group, 'expand_from_original_dtype')
            if row is not None: proposals.append((coarse_key, group, row))
        if not proposals: break
        chosen, group, row = min(proposals, key=lambda x:(not x[2]['passed'], x[2]['rank'], x[2]['promoted_state_elements']))
        accepted = set(row['selected']); coarse.pop(chosen)
        record.setdefault('expansions', []).append(dict(group=chosen, selected=sorted(accepted), passed=row['passed']))
        if row['passed']:
            passing = row;break
    if passing is None:
        record.update(state='no_passing_local_profile', selected_regions=sorted(accepted),
                      seconds=time.perf_counter()-started, budget_exhausted=len(record['trials']) >= max_trials)
        return []
    pending = [tuple(k for k in names if k in accepted)]
    while pending and len(record['trials']) < max_trials:
        group = pending.pop(0)
        row = evaluate(space.remove_regions(accepted,group), 'restore_native_subregions')
        if row is None: break
        if row['passed']:
            accepted = set(row['selected']);passing=row
        elif len(group)>1:
            middle = len(group)//2;pending[:0]=[group[:middle],group[middle:]]
    record.update(state='local_profiles_ready', selected_regions=sorted(accepted),
                  seconds=time.perf_counter()-started, budget_exhausted=bool(pending),
                  screening_failure_is_not_compiled_rejection=True)
    return [dict(selected=sorted(accepted), origin=passing['name'],
                 source='original_dtype_with_local_source_bindings')]


def refine_local_precision_profiles(space, seed, *, output_template, validation_inputs,
                                    validate, score, record, max_trials=64, progress=None):
    """Restore smaller native regions, retaining one failed screen for compilation.

    A native screen is a search heuristic, not the compiled numerical contract.
    The caller retains its original coarse route and validates every resulting
    compiled route. Q/K/V and other captured same-dtype joins remain indivisible.
    """
    from ._whole_model_selection import output_cast_like
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError('A finite refinement budget is required')
    record['trials'] = []
    cache = {}
    started = time.perf_counter()

    def evaluate(selected):
        selected = space.close_selected(selected)
        key = tuple(sorted(selected))
        if key in cache:
            return cache[key]
        if len(record['trials']) >= max_trials:
            return None
        name = 'operator_precision_probe_' + str(len(record['trials']))
        candidate = space.build(selected)
        passed, ranks, errors = True, [], []
        for args, kwargs in validation_inputs:
            values, keywords = space.prepare(*args, **kwargs)
            output = output_cast_like(candidate(*values, **keywords), output_template)
            try:
                validate(name, output, args, kwargs)
            except (AssertionError, ValueError) as exc:
                passed = False
                errors.append(str(exc)[:1000])
            rank = tuple(float(x) for x in score(name))
            if not rank or not all(math.isfinite(x) and x >= 0 for x in rank):
                raise ValueError('Precision ranking must be finite and nonnegative')
            ranks.append(rank)
            del output
        row = dict(name=name, selected=list(key), passed=passed, rank=list(max(ranks)), errors=errors,
                   promoted_state_elements=sum(space.original[n].numel() for k in key for n in space.groups[k]['tensors']))
        record['trials'].append(row)
        cache[key] = row
        del candidate
        if progress:
            progress()
        return row

    accepted = space.close_selected(seed)
    passing = evaluate(accepted)
    if not passing['passed']:
        record.update(state='refined_seed_failed', seconds=time.perf_counter()-started)
        return []
    pending = [tuple(sorted(accepted))]
    while pending and len(record['trials']) < max_trials:
        group = pending.pop(0)
        row = evaluate(space.remove_regions(accepted, group))
        if row is None:
            break
        if row['passed']:
            accepted, passing = set(row['selected']), row
        elif len(group) > 1:
            middle = len(group) // 2
            pending[:0] = [group[:middle], group[middle:]]
    # Always keep the best passing refined profile. Also compile the nearest
    # failed native screen: fusion may remove the rounding that made it fail.
    # It receives exactly the same complete-output gate as all other routes.
    proposals = _refinement_proposals(record['trials'], passing)
    record.update(state='refined_profiles_ready', selected_regions=sorted(accepted),
                  seconds=time.perf_counter()-started, budget_exhausted=bool(pending),
                  compiled_validation_required=True)
    return [dict(selected=row['selected'], origin=row['name'], native_screen_passed=row['passed'],
                 source='original_dtype_operator_regions', compiled_validation_required=True)
            for row in proposals]


def _refinement_proposals(trials, passing):
    proposals = [passing]
    failed = [r for r in trials if not r['passed']]
    if failed:
        proposals.append(min(failed, key=lambda r:(r['rank'], r['promoted_state_elements'])))
    # A last-rounding allowance can make a native screen pass right on its
    # boundary. Preserve one earlier passing precision assignment with a
    # better violation rank, rather than only compiling that fragile endpoint.
    # This does not change any tolerance or replace compiled validation.
    guarded = min((r for r in trials if r['passed']),
                  key=lambda r:(r['rank'][:2], r['promoted_state_elements']))
    if guarded['selected'] != passing['selected']:
        proposals.append(guarded)
    return proposals
