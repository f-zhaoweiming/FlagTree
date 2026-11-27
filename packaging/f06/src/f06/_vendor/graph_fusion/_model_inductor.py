"""Inference whole-graph scheduling, with P800 pattern exposure and host timing.

Rewrites happen before AOTAutograd; Inductor owns allocation, scheduling and
kernel generation. No Python region module remains on the warm execution path.
"""
import copy
from contextlib import nullcontext
import math
import operator
import statistics
import threading

import torch
from torch import fx
import torch.nn.functional as F


def _value(node):
    return node.meta.get('example_value', node.meta.get('val')) if isinstance(node, fx.Node) else None


def _is_call(node, targets):
    return isinstance(node, fx.Node) and node.op == 'call_function' and node.target in targets


def expose_linear(gm):
    records = []
    for node in gm.graph.nodes:
        if node.op == 'call_function' and str(node.target) in (
                '_dynamo_workaround.linear', '_dynamo_workaround.linear.default'):
            node.target = torch.ops.aten.linear.default
            records.append(dict(kind='linear', node=node.name, state='exposed_to_inductor'))
    gm.graph.lint()
    gm.recompile()
    return records


def expose_fp32_patterns(gm):
    """Keep live residual outputs; eliminate only internal low precision casts.

    New nodes occupy the old operation's position. No topological reordering
    across masks or other mutations is performed. Inductor may choose multiple
    kernels when a supported reduction schedule is more profitable.
    """
    graph = gm.graph
    records = []
    for node in list(graph.nodes):
        if _is_call(node, (F.layer_norm, torch.ops.aten.layer_norm.default)):
            names = ('input', 'normalized_shape', 'weight', 'bias', 'eps', 'cudnn_enable')
            bound = dict(zip(names, node.args), **node.kwargs)
            add = bound.get('input')
            if not _is_call(add, (operator.add, torch.add, torch.ops.aten.add.Tensor)):
                continue
            if len(add.args) != 2 or add.kwargs.get('alpha', 1) != 1 or set(add.kwargs) - {'alpha'}:
                continue
            # Adjacent pure operations only: no recomputation after intervening mutations.
            if add.next is not node:
                continue
            vals = [_value(x) for x in add.args]
            if not all(isinstance(v, torch.Tensor) for v in vals):
                continue
            dtype = vals[0].dtype
            if dtype not in (torch.float32, torch.float16, torch.bfloat16) or vals[1].dtype != dtype:
                continue
            external = [u for u in add.users if u is not node]
            record = dict(kind='residual_layernorm', node=node.name, outputs=2 if external else 1,
                          broadcast=tuple(vals[0].shape) != tuple(vals[1].shape), state='exposed_to_inductor')
            records.append(record)
            if dtype == torch.float32:
                continue
            with graph.inserting_before(add):
                a32 = graph.call_method('to', (add.args[0], torch.float32))
                b32 = graph.call_method('to', (add.args[1], torch.float32))
                sum32 = graph.call_function(operator.add, (a32, b32))
                if external:
                    residual = graph.call_method('to', (sum32, dtype))
                    residual.meta = dict(add.meta)
                    for user in external:
                        user.replace_input_with(add, residual)
            with graph.inserting_before(node):
                w, b = bound.get('weight'), bound.get('bias')
                w32 = graph.call_method('to', (w, torch.float32)) if w is not None else None
                b32 = graph.call_method('to', (b, torch.float32)) if b is not None else None
                norm32 = graph.call_function(F.layer_norm, (sum32, bound['normalized_shape'],
                    w32, b32, bound.get('eps', 1e-5)))
                result = graph.call_method('to', (norm32, dtype))
                result.meta = dict(node.meta)
            node.replace_all_uses_with(result)
            graph.erase_node(node)
            graph.erase_node(add)
        elif _is_call(node, (operator.mul, torch.mul, torch.ops.aten.mul.Tensor)):
            if len(node.args) != 2 or node.kwargs:
                continue
            x, sig = node.args
            if not _is_call(sig, (torch.sigmoid, torch.ops.aten.sigmoid.default)):
                x, sig = sig, x
            if (not _is_call(sig, (torch.sigmoid, torch.ops.aten.sigmoid.default))
                    or len(sig.args) != 1 or sig.kwargs):
                continue
            scale = sig.args[0]
            # Inner operations must also be pure. Replacing an out= call
            # would discard its write even when the return has one user.
            if (not _is_call(scale, (operator.mul, torch.mul, torch.ops.aten.mul.Tensor))
                    or len(scale.args) != 2 or scale.kwargs):
                continue
            if not ((scale.args[0] is x and scale.args[1] == 1.702) or
                    (scale.args[1] is x and scale.args[0] == 1.702)):
                continue
            if len(scale.users) != 1 or len(sig.users) != 1 or scale.next is not sig or sig.next is not node:
                continue
            value = _value(x)
            if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue
            records.append(dict(kind='quick_gelu', node=node.name, state='exposed_to_inductor'))
            if value.dtype == torch.float32:
                continue
            with graph.inserting_before(scale):
                x32 = graph.call_method('to', (x, torch.float32))
                mul = graph.call_function(operator.mul, (x32, 1.702))
                sigmoid = graph.call_function(torch.sigmoid, (mul,))
                product = graph.call_function(operator.mul, (x32, sigmoid))
                result = graph.call_method('to', (product, value.dtype))
                result.meta = dict(node.meta)
            node.replace_all_uses_with(result)
            for dead in (node, sig, scale):
                graph.erase_node(dead)
    graph.lint()
    gm.recompile()
    return records


def _fp32_reference(gm):
    reference = copy.deepcopy(gm).float()
    def promote(value):
        if isinstance(value, torch.dtype) and value in (torch.float16, torch.bfloat16):
            return torch.float32
        return value
    from torch.utils._pytree import tree_map
    for node in reference.graph.nodes:
        node.args = tree_map(promote, node.args)
        node.kwargs = tree_map(promote, node.kwargs)
    reference.recompile()
    def call(*args):
        return reference(*(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in args))
    return call


def refresh_graph_metadata(gm, example_inputs=None):
    """Infer rewritten graph values without retaining pre-rewrite dtype facts."""
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from torch.fx.passes.shape_prop import _extract_tensor_metadata
    from torch.utils._pytree import tree_map

    if example_inputs is None:
        placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
        if any('example_value' not in n.meta and 'val' not in n.meta for n in placeholders):
            raise RuntimeError('Rewritten graph metadata requires example inputs')
        example_inputs = [_value(n) for n in placeholders]
    mode = detect_fake_mode(example_inputs)
    if mode is None:
        mode = FakeTensorMode(allow_non_fake_inputs=True)

    class Values(FakeTensorProp):
        def fetch_attr(self, target):
            value = super().fetch_attr(target)
            return mode.from_tensor(value) if isinstance(value, torch.Tensor) else value

        def run_node(self, node):
            value = super().run_node(node)
            # FakeTensorProp stores val, but the region passes prefer
            # example_value. Update both, including nested/non-tensor results.
            node.meta['example_value'] = node.meta.get('val', value)
            if 'tensor_meta' in node.meta:
                node.meta['tensor_meta'] = tree_map(
                    lambda x: _extract_tensor_metadata(x) if isinstance(x, torch.Tensor) else x,
                    node.meta['example_value'])
            return value

    Values(gm, mode=mode).propagate(*example_inputs)
    return gm


def promote_whole_graph(gm, example_inputs=None):
    """Optional strict FP32 arithmetic candidate; cast model outputs once."""
    from torch.utils._pytree import tree_map
    from ._p800_model_layernorm import _final_cast
    graph=gm.graph
    output=next(n for n in graph.nodes if n.op=='output')
    dtypes={n: _value(n).dtype for n in graph.nodes if isinstance(_value(n),torch.Tensor)}
    def dtype32(v):
        return torch.float32 if isinstance(v,torch.dtype) and v in (torch.float16,torch.bfloat16) else v
    for node in list(graph.nodes):
        node.args=tree_map(dtype32,node.args);node.kwargs=tree_map(dtype32,node.kwargs)
    first=next(n for n in graph.nodes if n.op not in ('placeholder','get_attr'))
    for node in list(graph.nodes):
        if node.op in ('placeholder','get_attr') and dtypes.get(node) in (torch.float16,torch.bfloat16):
            users=list(node.users)
            insertion = graph.inserting_after(node) if node.op == 'get_attr' else graph.inserting_before(first)
            with insertion:
                cast=graph.call_method('to',(node,torch.float32))
            for user in users:user.replace_input_with(node,cast)
    def final_cast(n):
        if isinstance(n,fx.Node) and dtypes.get(n) in (torch.float16,torch.bfloat16):
            with graph.inserting_before(output):
                return graph.call_function(torch.ops.flagtree_p800.final_cast.default,(n,dtypes[n]))
        return n
    output.args=tree_map(final_cast,output.args)
    graph.lint();gm.recompile()
    return refresh_graph_metadata(gm, example_inputs)


def _compare(got, expected, tolerance):
    from ._model_precision import compare_model_outputs
    compare_model_outputs(got, expected, tolerance, allow_boxed_sequence=True)


class _ParameterPack:
    """Cache only immutable-version parameter conversions, never activations."""
    def __init__(self, fn, indices, transform=None):
        self.fn, self.indices = fn, tuple(indices)
        self.transform = transform
        self.entries = {}
        self.lock = threading.RLock()
        self.ready = None
        self.streams = set()

    @staticmethod
    def key(x):
        # Inference tensors have no version counter. A stable substitute would
        # hide legal updates made inside inference_mode, so never reuse them.
        if torch.is_inference(x):
            return None
        return (id(x), x.data_ptr(), x._version, x.dtype, x.device, tuple(x.shape), tuple(x.stride()))

    def __call__(self, *args):
        with self.lock:
            values=list(args);changed=False;device=None
            # A partial refresh on a new stream must also wait for the cached,
            # unchanged weights. The new ready event must cover both batches.
            if self.ready is not None and self.entries:
                previous=next(iter(self.entries.values()))[2]
                stream=torch.cuda.current_stream(previous.device)
                if stream.cuda_stream not in self.streams:
                    stream.wait_event(self.ready)
                    self.streams.add(stream.cuda_stream)
            for i in self.indices:
                x=args[i]
                if not isinstance(x,torch.nn.Parameter) or x.requires_grad:
                    raise RuntimeError('Cached inference weights must remain frozen Parameters')
                key=self.key(x);entry=self.entries.get(i)
                if key is None or entry is None or entry[0]!=key:
                    entry=(key,x,x.float() if self.transform is None else self.transform(x))
                    self.entries[i]=entry;changed=True
                values[i]=entry[2]
                if x.device.type=='cuda':device=x.device
            if device is not None:
                stream=torch.cuda.current_stream(device)
                if changed:
                    self.ready=torch.cuda.Event()
                    self.ready.record(stream)
                    self.streams={stream.cuda_stream}
                elif stream.cuda_stream not in self.streams:
                    stream.wait_event(self.ready)
                    self.streams.add(stream.cuda_stream)
                for i in self.indices:values[i].record_stream(stream)
            return self.fn(*values)


class _SelectGraph:
    def __init__(self, candidates, reference, tolerance, record):
        self.candidates, self.reference = candidates, reference
        self.tolerance, self.record = tolerance, record
        self.selected = None
        self.lock = threading.RLock()

    def __call__(self, *args):
        if self.selected is None:
            with self.lock:
                if self.selected is None:
                    self.prepare(args)
        return self.selected(*args)

    def prepare(self, args):
        from ._resource_benchmark import paired_end_to_end
        if torch.is_grad_enabled():
            raise RuntimeError('F06 whole-graph selection is inference-only')
        expected = self.reference(*args)
        valid = {}
        for name, fn in self.candidates.items():
            self.record['preparing'] = 'validate:' + name
            try:
                _compare(fn(*args), expected, self.tolerance)
            except Exception as exc:
                self.record['rejected'].append(dict(route=name, reason='FP32 validation', error=str(exc)))
            else:
                valid[name] = fn
        if not valid:
            raise RuntimeError('No whole-graph candidate passed FP32 validation: ' + str(self.record['rejected']))
        timing = paired_end_to_end(valid, args, count=32, warmup=5, seed=719)
        medians = {name: statistics.median(samples) for name, samples in timing.items()}
        if any(not math.isfinite(t) or t <= 0 for t in medians.values()):
            raise RuntimeError('Invalid whole-graph host timing; selection was not cached')
        route = min(medians, key=medians.get)
        self.selected = valid[route]
        self.record.update(state='selected', route=route, calibration_us=timing,
                           median_us=medians, validated_against='FP32 eager, no intermediate narrowing')
        self.record['regions'] = self.record['candidate_regions'].get(route, [])
        self.record.pop('preparing', None)
        self.candidates = None
        self.reference = None


def inductor_backend(*, tolerance, environment_key, audit, precision_policy='fp32_reference',
                     precision_options=None, resource_policy='auto', graph_choice=None,
                     _cost_graph_observer=None, _capture_cost_graph=False):
    """Compile complete graph candidates and select by synchronized host latency."""
    if precision_policy not in ('fp32_reference', 'preserve_operator', 'fuse_fp32_internals', 'source_fp32_mixed'):
        raise ValueError('Unknown operator precision policy: ' + str(precision_policy))
    if precision_options and precision_policy not in ('source_fp32_mixed', 'preserve_operator', 'fuse_fp32_internals'):
        raise ValueError('Precision module scopes require the original FP32 mixed policy')
    if _capture_cost_graph and (_cost_graph_observer is None or precision_policy not in (
            'source_fp32_mixed', 'preserve_operator', 'fuse_fp32_internals')):
        raise ValueError('Cost capture requires an observer and an explicit precision policy')
    if resource_policy not in ('auto','cluster','sdnn'):
        raise ValueError('Unknown whole-graph pointwise resource policy')
    if graph_choice is not None and (precision_policy != 'fp32_reference' or resource_policy == 'auto'):
        raise ValueError('An explicit resource candidate is required for a fixed graph choice')
    def backend(gm, example_inputs):
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.lowering import make_fallback
        # XPU's embedding bounds assertion currently has no __assert_fail device
        # symbol. Keep this op in the vendor runtime, including its bounds checks.
        # Registration is local to this explicitly activated inference process.
        from ._p800_inductor_gelu import _on_p800
        on_p800 = any(isinstance(x, torch.Tensor) and x.device.type == 'cuda' and _on_p800(x.device)
                       for x in example_inputs)
        if on_p800:
            make_fallback(torch.ops.aten.embedding.default, warn=False)
            make_fallback(torch.ops.aten.index.Tensor, warn=False)
            # Noncontiguous transpose + cat is miscompiled by this XPU backend.
            make_fallback(torch.ops.aten.cat.default, warn=False, override_decomp=True)
        if resource_policy == 'sdnn' and not on_p800:
            raise ValueError('SDNN whole-graph candidates require a P800 device')
        original = copy.deepcopy(gm)
        if precision_policy == 'source_fp32_mixed':
            from ._source_precision import source_mixed_candidate, SourceParameterPack
            from ._operator_precision import preserve_native_compute, _linear
            from ._p800_model_layernorm import rewrite, preserve_unmatched_norms, preserve_vendor_gelu, preserve_vendor_tanh
            options = dict(precision_options or {})
            schedule = options.pop('fp32_schedule', 'native')
            pointwise_policy = options.pop('pointwise_policy', 'generated')
            if schedule not in ('native', 'inductor', 'grouped'):
                raise ValueError('Unknown source FP32 schedule: ' + str(schedule))
            mixed, examples, indices, scopes = source_mixed_candidate(
                copy.deepcopy(gm), example_inputs, **options)
            regions = rewrite(mixed, fp32_only=True)
            refresh_graph_metadata(mixed, examples)
            preserve_unmatched_norms(mixed)
            preserve_vendor_gelu(mixed)
            preserve_vendor_tanh(mixed)
            contracts = preserve_native_compute(mixed)
            packs, groups = [], []
            if schedule != 'native':
                # Expose only FP32 contractions to the already validated
                # Inductor layouts. Half contractions retain native matrix mode.
                for node in mixed.graph.nodes:
                    if node.op == 'call_function' and node.target is _linear._opoverload and (
                            _value(node.args[0]).dtype == torch.float32):
                        node.target = torch.ops.aten.linear.default
                mixed.recompile()
                from ._p800_model_schedule import preserve_block_transpose, preserve_short_softmax
                regions += preserve_block_transpose(mixed) + preserve_short_softmax(mixed)
                if schedule == 'grouped':
                    from ._p800_model_gemm import packed_weight_candidate, grouped_projection_candidate
                    packed = packed_weight_candidate(mixed, examples)
                    if packed is not None:
                        mixed, examples, packs, additions = packed
                        regions += additions
                    grouped = grouped_projection_candidate(mixed, examples)
                    if grouped is not None:
                        mixed, examples, groups, additions = grouped
                        regions += additions
            from ._pointwise_candidates import preserve_pointwise
            if _cost_graph_observer is not None:
                _cost_graph_observer(mixed, examples)
            pointwise = preserve_pointwise(mixed, pointwise_policy)
            record = dict(backend='inductor_whole_graph', environment_key=environment_key,
                precision_policy=precision_policy, precision_options=precision_options,
                precision_contracts=contracts, source_precision_scopes=scopes,
                fp32_compute_regions=regions, cached_half_parameter_indices=list(indices),
                original_fp32_sources_retained=True, graph_code=mixed.code, regions=regions,
                rejected=[], state='awaiting_complete_model_validation',
                route='inductor_source_fp32_mixed_'+schedule, requires_original_fp32_model_validation=True)
            record.update(pointwise_implementation=pointwise_policy, native_pointwise_regions=pointwise)
            record.update(pointwise_resource_policy=resource_policy, pointwise_resource_candidates=[])
            audit.append(record)
            unresolved = [r['node'] for r in contracts if r['action']=='native_candidate_required']
            if unresolved:raise ValueError('Unreviewed mixed boundary: '+', '.join(unresolved))
            from ._p800_model_schedule import p800_lm_schedule
            with p800_lm_schedule(buffer_bytes=2048, split_addmm=True,
                    resource_kind=resource_policy, resource_audit=record['pointwise_resource_candidates']):
                body=mixed.forward if _capture_cost_graph else compile_fx(mixed, examples, config_patches={
                    # CUDA graph execution is disabled. Its independent
                    # partition pass recursively revisits shared fallback IR
                    # in residual DAGs and is unnecessary for this entry.
                    'triton.cudagraphs': False, 'graph_partition': False,
                    'compile_threads': 1, 'max_autotune': False,
                    'max_autotune_gemm': False, 'triton.autotune_pointwise': False, 'shape_padding': False})
            if groups:
                from ._p800_model_gemm import GroupedProjectionPack
                body = GroupedProjectionPack(body, groups)
            if packs:
                from ._p800_model_gemm import pack_column_major
                body = _ParameterPack(body, packs, transform=pack_column_major)
            return SourceParameterPack(body, len(example_inputs), indices) if indices else body
        if precision_policy in ('preserve_operator', 'fuse_fp32_internals'):
            # This is a separate original-dtype candidate, never a renamed
            # FP32-source graph. Protected operator semantics are tested
            # separately; original-FP32 accuracy and timing are evaluated at the
            # complete model entry, before selecting or caching this candidate.
            from ._operator_precision import preserve_native_compute
            from ._p800_model_layernorm import preserve_unmatched_norms, preserve_vendor_gelu, preserve_vendor_tanh
            options = dict(precision_options or {})
            pointwise_policy = options.pop('pointwise_policy', 'generated')
            require_compiled = options.pop('require_compiled', False)
            linear_schedule = options.pop('linear_schedule', 'native')
            matmul_policy = options.pop('matmul_policy', 'native')
            if matmul_policy not in ('native', 'tuned', 'full_k'):
                raise ValueError('Unknown captured matrix policy')
            if matmul_policy in ('tuned', 'full_k') and (not on_p800 or linear_schedule not in ('inductor', 'grouped')):
                raise ValueError('Tuned matrix candidates require the P800 Inductor schedule')
            pointwise_buffer_bytes = options.pop('pointwise_buffer_bytes', 2048)
            if pointwise_buffer_bytes not in (512, 1024, 2048):
                raise ValueError('Unsupported captured pointwise buffer size')
            if linear_schedule not in ('native', 'inductor', 'grouped'):
                raise ValueError('Unknown captured Linear schedule: '+str(linear_schedule))
            if options:
                raise ValueError('Unknown captured-precision options: ' + str(sorted(options)))
            protected = copy.deepcopy(gm)
            fp32_regions = []
            if precision_policy == 'fuse_fp32_internals':
                from ._fp32_compute_regions import fuse_fp32_compute_regions
                fp32_regions = fuse_fp32_compute_regions(protected, example_inputs)
                from ._p800_model_layernorm import rewrite as rewrite_layernorm
                fp32_regions += rewrite_layernorm(protected, fp32_only=True)
                refresh_graph_metadata(protected, example_inputs)
                from ._fp32_compute_regions import fold_layernorm_boundary_casts
                fp32_regions += fold_layernorm_boundary_casts(protected, example_inputs)
            preserve_unmatched_norms(protected)
            preserve_vendor_gelu(protected)
            preserve_vendor_tanh(protected)
            precision = preserve_native_compute(protected)
            exposed = []
            if linear_schedule in ('inductor', 'grouped'):
                from ._operator_precision import _linear
                for node in protected.graph.nodes:
                    if node.op == 'call_function' and node.target is _linear._opoverload:
                        if matmul_policy in ('tuned', 'full_k') and _value(node.args[0]).dtype != torch.float32:
                            continue
                        # Packing is an implementation choice for an existing
                        # FP32 contraction. Never widen a captured low-precision
                        # contraction to make it eligible for this schedule.
                        if linear_schedule == 'grouped' and _value(node.args[0]).dtype != torch.float32:
                            continue
                        # Keep the captured input/weight/output dtypes. The
                        # candidate changes lowering, not the precision graph.
                        exposed.append(dict(node=node.name, input_dtype=str(_value(node.args[0]).dtype),
                                            output_dtype=str(_value(node).dtype)))
                        node.target = torch.ops.aten.linear.default
                protected.recompile()
            packs, groups, layout_regions = [], [], []
            if linear_schedule == 'grouped':
                from ._p800_model_gemm import packed_weight_candidate, grouped_projection_candidate
                packed = packed_weight_candidate(protected, example_inputs)
                if packed is not None:
                    protected, example_inputs, packs, additions = packed
                    layout_regions += additions
                grouped = grouped_projection_candidate(protected, example_inputs)
                if grouped is not None:
                    protected, example_inputs, groups, additions = grouped
                    layout_regions += additions
            from ._pointwise_candidates import preserve_pointwise
            if _cost_graph_observer is not None:
                _cost_graph_observer(protected, example_inputs)
            pointwise_domain = preserve_pointwise(protected, 'native', dry_run=True)
            pointwise = preserve_pointwise(protected, pointwise_policy)
            candidate_name = ('inductor_fp32_compute_regions' if precision_policy == 'fuse_fp32_internals'
                              else 'inductor_preserve_precision')
            record = dict(backend='inductor_whole_graph', environment_key=environment_key,
                          precision_policy=precision_policy, precision_contracts=precision,
                          fp32_compute_regions=fp32_regions,
                          compute_nodes=sum(n.op.startswith('call_') for n in gm.graph.nodes),
                          regions=[], rejected=[], candidate_regions={'original': [],
                          candidate_name: precision + fp32_regions}, graph_code=protected.code,
                          requires_original_fp32_model_validation=True)
            record.update(pointwise_implementation=pointwise_policy,
                          native_pointwise_regions=pointwise,
                          pointwise_choice_domains=pointwise_domain,
                          execution_precision='captured_per_operator',
                          forced_fp32_inputs=False, capture_source='original_dtype',
                          captured_input_dtypes=[str(x.dtype) for x in example_inputs if isinstance(x, torch.Tensor)],
                          original_graph_code=original.code,
                          linear_schedule=linear_schedule, exposed_linears=exposed,
                          matmul_policy=matmul_policy, matrix_candidates=[],
                          layout_regions=layout_regions)
            audit.append(record)
            prepared = original.forward
            record.update(state='awaiting_complete_model_validation', route='original')
            try:
                unresolved = [r['node'] for r in precision if r['action'] == 'native_candidate_required']
                if unresolved:
                    raise ValueError('Native precision boundary has no reviewed lowering: ' + ', '.join(unresolved))
                from ._p800_model_schedule import p800_lm_schedule
                # These graphs retain native contraction precision, but their
                # generated pointwise kernels need the same P800 tiling as the
                # FP32 candidates. Do not split addmm across precision boundaries.
                record.update(pointwise_resource_policy=resource_policy, pointwise_resource_candidates=[])
                record.update(pointwise_buffer_bytes=pointwise_buffer_bytes, fp16_buffer_kernels=[])
                schedule = (p800_lm_schedule(buffer_bytes=pointwise_buffer_bytes, fp16_buffer_bytes=pointwise_buffer_bytes, split_addmm=linear_schedule == 'grouped',
                            resource_kind=resource_policy, resource_audit=record['pointwise_resource_candidates'],
                            buffer_audit=record['fp16_buffer_kernels'])
                            if on_p800 else nullcontext())
                matrix_scope = nullcontext()
                if matmul_policy in ('tuned', 'full_k'):
                    from ._p800_gemm_candidates import p800_gemm_candidates
                    device = next(x.device for x in example_inputs
                                  if isinstance(x, torch.Tensor) and x.device.type == 'cuda')
                    matrix_scope = p800_gemm_candidates(device, record['matrix_candidates'],
                        full_k=matmul_policy == 'full_k', native_bias=linear_schedule == 'grouped')
                with schedule, matrix_scope:
                    prepared = protected.forward if _capture_cost_graph else compile_fx(protected, example_inputs,
                        config_patches={'triton.cudagraphs': False, 'graph_partition': False, 'compile_threads': 1,
                                        'max_autotune': False, 'max_autotune_gemm': False,
                                        'triton.autotune_pointwise': False, 'shape_padding': False})
                if groups:
                    from ._p800_model_gemm import GroupedProjectionPack
                    prepared = GroupedProjectionPack(prepared, groups)
                if packs:
                    from ._p800_model_gemm import pack_column_major
                    prepared = _ParameterPack(prepared, packs, transform=pack_column_major)
                record['pointwise_schedule'] = 'p800_lm' if on_p800 else 'default'
                record['route'] = candidate_name
            except Exception as exc:
                record['rejected'].append(dict(route=candidate_name, reason='compile', error=str(exc)))
                if resource_policy != 'auto' or require_compiled:
                    raise
            # The original native route is already an independent whole-model
            # candidate. Do not choose between them using an inner FP16 gate or
            # subgraph timing: that could hide a faster FP32-accurate candidate.
            return prepared
        reference = _fp32_reference(original)
        native = copy.deepcopy(gm)
        linear = expose_linear(native)
        optimized = copy.deepcopy(native)
        from ._p800_model_layernorm import rewrite as rewrite_layernorm, preserve_unmatched_norms, preserve_vendor_gelu, preserve_vendor_tanh
        preserve_vendor_tanh(native)
        preserve_vendor_tanh(optimized)
        patterns = rewrite_layernorm(optimized)
        patterns += expose_fp32_patterns(optimized)
        vendor_norms = preserve_unmatched_norms(optimized)
        record = dict(backend='inductor_whole_graph', environment_key=environment_key,
                      compute_nodes=sum(n.op.startswith('call_') for n in gm.graph.nodes),
                      regions=[], rejected=[], candidate_regions={
                          'original': [], 'inductor': linear, 'inductor_f06': linear + patterns},
                      graph_code=optimized.code, vendor_norm_fallbacks=vendor_norms)
        audit.append(record)
        record.update(pointwise_resource_policy=resource_policy, pointwise_resource_candidates=[])
        candidates = {'original': original.forward}
        patches = {'triton.cudagraphs': False, 'graph_partition': False, 'compile_threads': 1,
                   'max_autotune': False, 'max_autotune_gemm': False,
                   # The XPU event/cache-flush pointwise calibration can stall
                   # at synchronization even with valid-timestamp guards (F41).
                   # Compile one native tile per candidate; whole-graph host
                   # timing still compares the ordinary and LM schedules.
                   'triton.autotune_pointwise': False,
                   # pad_mm performs a separate GPU-graph benchmark even when
                   # GEMM autotuning is off; that SDK capture can stall on P800.
                   'shape_padding': False}
        # Identical FP32 graphs share generated code but still carry the pattern
        # audit. Avoid treating measurement noise as a choice between aliases.
        graphs = [('inductor_f06', optimized)]
        if native.code != optimized.code:
            graphs.insert(0, ('inductor', native))
        sdk_norm = copy.deepcopy(native)
        preserve_unmatched_norms(sdk_norm)
        preserve_vendor_gelu(sdk_norm)
        graphs.append(('inductor_sdk_norm', sdk_norm))
        record['candidate_regions']['inductor_sdk_norm'] = linear
        sdk_math = copy.deepcopy(optimized)
        if preserve_vendor_gelu(sdk_math):
            graphs.append(('inductor_f06_sdk_math', sdk_math))
            record['candidate_regions']['inductor_f06_sdk_math'] = linear + patterns
        from ._p800_inductor_gelu import _on_p800
        fp32_inputs = [x for x in example_inputs if isinstance(x, torch.Tensor) and x.is_floating_point()]
        if fp32_inputs and all(x.dtype == torch.float32 for x in fp32_inputs) and _on_p800(fp32_inputs[0].device):
            from ._p800_model_schedule import schedule_variants
            for name, scheduled, regions in schedule_variants([
                    ('inductor_p800_lm', sdk_math, linear + patterns),
                    ('inductor_p800_lm_sdk_norm', sdk_norm, linear)]):
                graphs.append((name, scheduled))
                record['candidate_regions'][name] = regions
        if any(isinstance(x, torch.Tensor) and x.dtype in (torch.float16, torch.bfloat16) for x in example_inputs):
            try:
                promoted = promote_whole_graph(copy.deepcopy(sdk_math), example_inputs)
                graphs.append(('inductor_f06_fp32', promoted))
                record['candidate_regions']['inductor_f06_fp32'] = linear + patterns
            except Exception as exc:
                record['rejected'].append(dict(route='inductor_f06_fp32', reason='metadata inference', error=str(exc)))
        for name, graph in graphs:
            if graph_choice is not None:
                related = {name}
                if name.startswith('inductor_p800_lm') and name.endswith('2048'):
                    related.add(name[:-4]+'_packed_gemm2048')
                    if '_native_bias_mm' in name:
                        related.add(name[:-4]+'_grouped_proj2048')
                if graph_choice not in related:
                    continue
            record['preparing'] = 'compile:' + name
            try:
                if name.startswith('inductor_p800_lm'):
                    from ._p800_model_schedule import p800_lm_schedule
                    with p800_lm_schedule(buffer_bytes=2048 if name.endswith('2048') else None,
                                          split_addmm='_native_bias_mm' in name,
                                          resource_kind=resource_policy, resource_audit=record['pointwise_resource_candidates']):
                        candidates[name] = compile_fx(graph, example_inputs, config_patches=patches)
                elif resource_policy != 'auto' and on_p800:
                    from ._p800_model_schedule import p800_lm_schedule
                    with p800_lm_schedule(buffer_bytes=2048, resource_kind=resource_policy,
                                          resource_audit=record['pointwise_resource_candidates']):
                        candidates[name] = compile_fx(graph, example_inputs, config_patches=patches)
                else:
                    candidates[name] = compile_fx(graph, example_inputs, config_patches=patches)
            except Exception as exc:
                record['rejected'].append(dict(route=name, reason='compile', error=str(exc)))
            if name.startswith('inductor_p800_lm') and name.endswith('2048'):
                packed_name = name[:-4] + '_packed_gemm2048'
                try:
                    from ._p800_model_gemm import packed_weight_candidate, pack_column_major
                    packed = packed_weight_candidate(graph, example_inputs)
                    if packed is not None:
                        packed_graph, examples, indices, regions = packed
                        record['preparing'] = 'compile:' + packed_name
                        from ._p800_model_schedule import p800_lm_schedule
                        with p800_lm_schedule(buffer_bytes=2048, split_addmm='_native_bias_mm' in name,
                                resource_kind=resource_policy, resource_audit=record['pointwise_resource_candidates']):
                            body = compile_fx(packed_graph, examples, config_patches=patches)
                        candidates[packed_name] = _ParameterPack(body, indices, transform=pack_column_major)
                        record['candidate_regions'][packed_name] = record['candidate_regions'][name] + regions
                except Exception as exc:
                    record['rejected'].append(dict(route=packed_name, reason='compile', error=str(exc)))
                if '_native_bias_mm' in name:
                    grouped_name = name[:-4] + '_grouped_proj2048'
                    try:
                        from ._p800_model_gemm import grouped_projection_candidate, GroupedProjectionPack
                        # Reuse the validated contraction layout where eligible.
                        base_graph, base_examples, packed_indices = graph, example_inputs, []
                        base_regions = record['candidate_regions'][name]
                        packed_base = packed_weight_candidate(graph, example_inputs)
                        if packed_base is not None:
                            base_graph, base_examples, packed_indices, pack_regions = packed_base
                            base_regions = base_regions + pack_regions
                        grouped = grouped_projection_candidate(base_graph, base_examples)
                        if grouped is not None:
                            grouped_graph, examples, groups, regions = grouped
                            record['preparing'] = 'compile:' + grouped_name
                            with p800_lm_schedule(buffer_bytes=2048, split_addmm=True,
                                    resource_kind=resource_policy, resource_audit=record['pointwise_resource_candidates']):
                                body = compile_fx(grouped_graph, examples, config_patches=patches)
                            body = GroupedProjectionPack(body, groups)
                            if packed_indices:
                                body = _ParameterPack(body, packed_indices, transform=pack_column_major)
                            candidates[grouped_name] = body
                            record['candidate_regions'][grouped_name] = base_regions + regions
                    except Exception as exc:
                        record['rejected'].append(dict(route=grouped_name, reason='compile', error=str(exc)))
            if name == 'inductor_f06_fp32':
                # Hoist only parameter casts. Dynamo still guards the original
                # model dtypes; this private callable receives guarded FP32 packs.
                indices=[i for i,x in enumerate(example_inputs) if isinstance(x,torch.nn.Parameter)
                         and not x.requires_grad and x.dtype in (torch.float16,torch.bfloat16)]
                if indices:
                    try:
                        record['preparing'] = 'compile:inductor_f06_fp32_packed'
                        from torch._guards import detect_fake_mode
                        mode=detect_fake_mode()
                        if mode is None:continue
                        with mode:
                            examples=[torch.empty_strided(x.shape,x.stride(),dtype=torch.float32,device=x.device)
                                      if i in indices else x for i,x in enumerate(example_inputs)]
                        packed_name='inductor_f06_fp32_packed'
                        packed_graph=copy.deepcopy(graph)
                        # AOTAutograd re-executes the graph with these examples.
                        # Replace the input metadata, leaving intermediate
                        # inference to AOT rather than reusing half input facts.
                        inputs=[n for n in packed_graph.graph.nodes if n.op=='placeholder']
                        for node,value in zip(inputs,examples):
                            node.meta['example_value']=value
                            node.meta.pop('val',None)
                        refresh_graph_metadata(packed_graph, examples)
                        body=compile_fx(packed_graph,examples,config_patches=patches)
                        candidates[packed_name]=_ParameterPack(body,indices)
                        record['candidate_regions'][packed_name]=linear+patterns
                        record['packed_parameters']=len(indices)
                    except Exception as exc:
                        record['rejected'].append(dict(route='inductor_f06_fp32_packed',reason='compile',error=str(exc)))
        if graph_choice is not None:
            if graph_choice not in candidates:
                raise RuntimeError('Requested whole-graph resource candidate did not compile: '+str(record['rejected']))
            record.update(route=graph_choice, state='awaiting_complete_model_validation',
                          requires_original_fp32_model_validation=True)
            record.pop('preparing', None)
            return candidates[graph_choice]
        return _SelectGraph(candidates, reference, tolerance, record)
    return backend
