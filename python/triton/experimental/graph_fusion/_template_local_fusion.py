"""Use the existing template registry to propose local regions in full models.

The surrounding captured FX operations keep their order and native targets.
Regions may return several live outputs. Discovery is not a selectable eager
route; every proposed compiled model needs the caller's complete-output gate.
"""
import copy
import hashlib
import operator
from pathlib import Path
import types

import torch
from torch import fx

from ._patterns import FusionTemplate, Match, Registry, default_registry, _pair, _bind
from ._local_fusion import _ambient_supported, _p800_model, _signature


class _NoCompiledRegion(ValueError):
    """A recognized graph has no region with an admitted compilation ABI."""


def _quickgelu(node):
    def mul(n):
        return (isinstance(n, fx.Node) and n.op == 'call_function'
                and n.target in (operator.mul, torch.mul) and len(n.args) == 2 and not n.kwargs)
    if not mul(node):
        return None
    for x, sigmoid in (node.args, node.args[::-1]):
        if (not isinstance(x, fx.Node) or not isinstance(sigmoid, fx.Node)
                or sigmoid.op != 'call_function' or sigmoid.target is not torch.sigmoid
                or len(sigmoid.args) != 1 or sigmoid.kwargs):
            continue
        scale = sigmoid.args[0]
        if mul(scale) and any(scale.args[i] is x and type(scale.args[1-i]) in (int, float)
                             and scale.args[1-i] == 1.702 for i in (0, 1)):
            return Match('quick_gelu', (scale, sigmoid, node), (x,))
    return None


def discover_regions(gm, templates=None):
    """Reuse registry matchers, admitting live intermediates as extra outputs.

    Contiguous pure calls are replaced at their original last call. No outside
    computation, mutation, or use may be moved across that boundary.
    """
    registry = Registry((*default_registry().templates,
                         FusionTemplate('quick_gelu', 1, 40, _quickgelu)))
    calls = [n for n in gm.graph.nodes if n.op.startswith('call_')]
    position = {n:i for i,n in enumerate(calls)}
    occupied, selected, rejected = set(), [], []
    for template in registry.templates:
        if templates is not None and template.template_id not in templates:
            continue
        for node in calls:
            match = template.matcher(node)
            if match is None or occupied.intersection(match.nodes):
                continue
            indices = [position[n] for n in match.nodes]
            members = set(match.nodes)
            if indices != list(range(indices[0], indices[0] + len(indices))):
                rejected.append(dict(template=template.template_id, output=node.name,
                                     reason='intervening_computation_or_side_effect'))
                continue
            if any(n in members for n in match.inputs):
                rejected.append(dict(template=template.template_id, output=node.name,
                                     reason='region_input_depends_on_internal_node'))
                continue
            outputs = tuple(n for n in match.nodes if any(u not in members for u in n.users))
            if not outputs:
                continue
            selected.append((template, match, outputs))
            occupied.update(members)
    selected.sort(key=lambda item:position[item[1].nodes[-1]])
    return selected, rejected


def partition_regions(gm, templates, make_region, record):
    """Extract matches without modifying the input graph or dropping outputs."""
    from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule
    # Parameter/buffer values stay live; copy graph structure, not tensor data.
    copied = copy.deepcopy(gm, {id(x):x for x in (*gm.parameters(), *gm.buffers())})
    selected, rejected = discover_regions(copied, templates)
    record.update(regions=[], rejected=rejected)
    replacements, internal = {}, set()
    for index, (template, match, expected_outputs) in enumerate(selected):
        name = '_f06_template_region_' + str(index)
        assert not hasattr(copied, name)
        region, inputs, outputs = fuse_as_graphmodule(copied, list(match.nodes), name,
            {n:None for n in match.nodes}, always_return_tuple=True)
        assert set(outputs) == set(expected_outputs)
        row = dict(template=template.template_id, kind=match.kind,
            nodes=[n.name for n in match.nodes], inputs=[n.name for n in inputs],
            outputs=[n.name for n in outputs], options=dict(match.options))
        record['regions'].append(row)
        copied.add_module(name, make_region(region, row))
        replacements[match.nodes[-1]] = (name, inputs, outputs)
        internal.update(match.nodes)
    graph, values = fx.Graph(), {}
    for node in copied.graph.nodes:
        if node in replacements:
            name, inputs, outputs = replacements[node]
            result = graph.call_module(name, tuple(values[n] for n in inputs))
            for i, old in enumerate(outputs):
                values[old] = graph.call_function(operator.getitem, (result, i))
                values[old].meta = dict(old.meta)
        elif node not in internal:
            values[node] = graph.node_copy(node, lambda n:values[n])
    copied.graph = graph
    graph.lint()
    copied.recompile()
    record['graph_code'] = copied.code
    return copied


def _supported_args(args):
    if not args or torch.is_grad_enabled() or not _ambient_supported():
        return False
    for x in args:
        if (type(x) not in (torch.Tensor, torch.nn.Parameter) or x.device.type != 'cuda'
                or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or x.layout != torch.strided or not x.is_contiguous() or not x.numel()
                or x.requires_grad or x.storage_offset() != 0 or x.data_ptr() % 64
                or x.is_neg() or x.is_conj()):
            return False
    # Tracing can specialize input alias relationships. Keep aliases on their
    # original operator path instead of reusing a non-aliasing specialization.
    return len({x.untyped_storage().data_ptr() for x in args}) == len(args)


def _key(args):
    return tuple(_signature(x) for x in args)


def _matches_args(args, signature):
    """Check an already admitted layout without repeating its static facts.

    Equal sizes/strides/dtype/device prove the original nonempty contiguous
    supported layout. Storage offset/alignment and aliasing remain live checks.
    With offset zero, data_ptr is also the storage base; query it only once.
    """
    if (len(args) != len(signature) or torch.is_grad_enabled() or not _ambient_supported()
            or any(type(x) not in (torch.Tensor, torch.nn.Parameter)
                   or x.layout != torch.strided or x.storage_offset() != 0 for x in args)
            or _key(args) != signature):
        return False
    pointers = tuple(x.data_ptr() for x in args)
    return not any(p % 64 for p in pointers) and len(set(pointers)) == len(pointers)


def _make_args_guard(args):
    """Let Torch check the admitted tensor metadata in one native operation.

    Contents and addresses remain live. TensorGuards does not check offsets,
    aliases or inference mode, so those contracts are retained explicitly.
    """
    guards = torch._C._dynamo.guards
    check = guards.TensorGuards(*args, dynamic_dims_sizes=None, dynamic_dims_strides=None).check
    inference = torch.is_inference_mode_enabled()
    types_ = tuple(type(x) for x in args)

    def matches(values):
        if (len(values) != len(types_) or torch.is_grad_enabled()
                or torch.is_inference_mode_enabled() != inference or not _ambient_supported()
                or any(type(x) is not ty for x, ty in zip(values, types_))
                or not check(*values)
                or any(x.layout != torch.strided or x.storage_offset() != 0
                       or x.is_conj() or x.is_neg() for x in values)):
            return False
        pointers = tuple(x.data_ptr() for x in values)
        return not any(p % 64 for p in pointers) and len(set(pointers)) == len(pointers)
    return matches


def _protect_contractions(gm):
    """Retain vendor matrix and ReLU arithmetic, including low-dtype modes."""
    from ._operator_precision import _matmul, _relu, _lower_reviewed
    mm = {torch.mm:'mm', torch.bmm:'bmm', torch.matmul:'matmul', operator.matmul:'matmul',
          torch.ops.aten.mm.default:'mm', torch.ops.aten.bmm.default:'bmm',
          torch.ops.aten.matmul.default:'matmul'}
    protected = []
    for n in gm.graph.nodes:
        kind = mm.get(n.target) if n.op == 'call_function' else (
            n.target if n.op == 'call_method' and n.target in ('mm', 'matmul') else None)
        if kind is not None:
            operands = _pair(n, 'matmul')
            assert operands is not None
            _lower_reviewed(_matmul._opoverload, _matmul)
            n.op, n.target, n.args, n.kwargs = 'call_function', _matmul._opoverload, (*operands, kind), {}
            protected.append(dict(node=n.name, native=kind))
        elif ((n.op == 'call_function' and n.target in (torch.relu, torch.nn.functional.relu))
              or (n.op == 'call_method' and n.target == 'relu')):
            bound = _bind(n, ('input','inplace') if n.target is torch.nn.functional.relu else ('input',),
                          {'inplace':False})
            assert bound is not None and bound['inplace'] is False
            _lower_reviewed(_relu._opoverload, _relu)
            n.op, n.target, n.args, n.kwargs = 'call_function', _relu._opoverload, (bound['input'],), {}
            protected.append(dict(node=n.name, native='relu'))
    gm.graph.lint()
    gm.recompile()
    return protected


def compile_region(gm, args, buffer_bytes, record):
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx_inner, BoxedBool
    from ._inference import _trace_compiler_graph
    from ._p800_model_schedule import p800_lm_schedule
    from ._resource_codegen import _bound_module, _record_binary
    protected = copy.deepcopy(gm)
    record['native_boundaries'] = _protect_contractions(protected)
    row_options = dict(record.get('row_options', {}))
    if row_options.pop('ordered_moments', False):
        from ._p800_ordered_layernorm import rewrite
        record['ordered_layernorm'] = rewrite(protected, args)
    # The established tracer annotates dtype-rounding boundaries before ATen
    # decomposition. There is no FP32 promotion or dropped cast in this path.
    patches = {'emulate_precision_casts':True, 'triton.cudagraphs':False,
        'graph_partition':False, 'compile_threads':1, 'max_autotune':False,
        'max_autotune_gemm':False, 'triton.autotune_pointwise':False,
        'coordinate_descent_tuning':False, 'shape_padding':False, 'pattern_matcher':False}
    with config.patch(patches), p800_lm_schedule(buffer_bytes=buffer_bytes,
                                                fp16_buffer_bytes=buffer_bytes,
                                                **row_options):
        lowered = _trace_compiler_graph(protected, args, emulate_precision=True)
        compiled = compile_fx_inner(lowered, list(args), is_inference=True, cudagraphs=BoxedBool(False))
    module = _bound_module(compiled)
    kernels = [(n,k) for n,k in vars(module).items() if n.startswith('triton_') and hasattr(k,'run')]
    evidence = []
    for name, kernel in kernels:
        kernel.precompile()
        for launcher in kernel.launchers:
            binary = getattr(launcher, '__globals__', {}).get('bin')
            if binary is None:
                raise RuntimeError('Template region launcher has no bound binary')
            actual = 'sdnn' if binary.metadata.is_sdnn else 'cluster'
            evidence.append(dict(_record_binary(binary, actual), wrapper_binding=name))
    if not kernels:
        raise _NoCompiledRegion('Template region produced no generated kernel')
    # make_fx/compile_fx_inner keep these explicit tensor inputs in order.
    # No AOT parameter lifting, hidden weights, backward or saved tensors.
    def call(*values):
        return module.call(list(values))
    record.update(state='compiled', buffer_bytes=buffer_bytes, signature=repr(_key(args)),
        graph_code=gm.code, lowered_code=lowered.code, wrapper_path=module.__file__,
        wrapper_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        kernel_count=len(kernels), binary_evidence=evidence,
        row_search=[dict(wrapper_binding=name,
                         candidates=getattr(kernel,'p800_row_candidates',[]),
                         interfaces=getattr(kernel,'p800_row_interfaces',[]),
                         timings=getattr(kernel,'p800_row_timings',[]))
                    for name,kernel in kernels if hasattr(kernel,'p800_row_candidates')],
        call_boundary='guarded_generated_wrapper', precision='preserve_operator_boundaries')
    from ._p800_region_launch import prepare
    return prepare(module, args, record) or call


class _Region(torch.nn.Module):
    def __init__(self, graph, pipeline, row):
        super().__init__()
        self.graph = graph
        # A plain closure reference avoids registering the complete parent as
        # a child module (which would create a recursive module hierarchy).
        self._owner = lambda:pipeline
        self.row = row
        self.last = None

    def forward(self, *args):
        hit = self.last
        if hit is not None and hit[2](args):
            return hit[1](*args)
        owner = self._owner()
        if not _supported_args(args):
            return self.graph(*args)
        signature = _key(args)
        key = (self.graph.code, signature)
        call = owner.bindings.get(key)
        if call is None:
            if owner.prepared:
                return self.graph(*args)
            proof = dict(template=self.row['template'], kind=self.row['kind'],
                         output_count=len(self.row['outputs']), row_options=owner.row_options)
            call = compile_region(self.graph, args, owner.buffer_bytes, proof)
            owner.bindings[key] = call
            owner.record['compiled_regions'].append(proof)
        self.last = (signature, call, _make_args_guard(args))
        return call(*args)


def _capture_boundary(model):
    class Capture(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)
    fn = Capture.forward
    Capture.forward = types.FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__,
                                        fn.__defaults__, fn.__closure__)
    return Capture(model).eval()


class _Pipeline:
    def __init__(self, model, prepare, templates, buffer_bytes, record, row_options=None):
        private = copy.deepcopy(model, {id(x):x for x in (*model.parameters(), *model.buffers())})
        self.prepare, self.buffer_bytes, self.record = prepare, buffer_bytes, record
        self.row_options = dict(row_options or {})
        self.prepared, self.bindings = False, {}
        self._f06_audit = [record]
        self._f06_scope = 'complete_model_with_local_templates'
        def backend(gm, examples):
            row = {}
            graph = partition_regions(gm, templates, lambda region, info:_Region(region, self, info), row)
            record['graphs'].append(row)
            return graph.forward
        self.compiled = torch.compile(_capture_boundary(private), backend=backend, fullgraph=True, dynamic=False)

    def __call__(self, *args, **kwargs):
        args, kwargs = self.prepare(*args, **kwargs)
        return self.compiled(*args, **kwargs)

    def prepare_examples(self, examples):
        for args, kwargs in examples:
            output = self(*args, **kwargs)
            del output
        self.prepared = True
        if not self.bindings:
            raise _NoCompiledRegion('No eligible template region executed; this is not a compiled candidate')


def _template_schedules(templates, buffer_sizes):
    """Keep baseline schedules and add independently validated row algorithms.

    Shared columns require the out-of-place pure-region and no-alias runtime
    guards above. Reordered sums remain separate complete-model candidates.
    """
    result = []
    for size in buffer_sizes:
        if size not in (512, 1024, 2048):
            raise ValueError('Unsupported local template buffer size')
        result.append(('lm'+str(size), size, {}))
    if 'add_layernorm' in templates:
        result += [
            ('ordered_moments', 512, dict(ordered_moments=True)),
            ('lm512_reuse', 512, dict(reduction_reuse=True)),
            ('lm512_reuse_tree', 512, dict(reduction_reuse=True,reduction_tree=True)),
            ('lm2048_shared', 2048, dict(row_shared=True)),
            ('lm2048_shared_tree', 2048, dict(row_shared=True,reduction_tree=True)),
            ('lm2048_shared_reuse', 2048, dict(row_shared=True,reduction_reuse=True)),
            ('lm2048_shared_reuse_tree', 2048,
             dict(row_shared=True,reduction_reuse=True,reduction_tree=True)),
            ('lm2048_shared_half_cache', 2048,
             dict(row_shared=True,reduction_reuse=True,reduction_half_cache=True)),
            ('lm2048_shared_half_cache_tree', 2048,
             dict(row_shared=True,reduction_reuse=True,reduction_half_cache=True,reduction_tree=True)),
        ]
    return result


def template_local_candidate_routes(model, *, prepare, validation_inputs, record, buffer_sizes=(512, 1024)):
    from ._whole_model_selection import ModelRoute
    from ._model import enable_vendor_capture
    from torch.utils._pytree import tree_leaves
    from ._p800_inductor_gelu import _on_p800
    if not (_p800_model(model) or any(isinstance(x, torch.Tensor) and x.device.type=='cuda' and _on_p800(x.device)
                                     for x in tree_leaves(validation_inputs))):
        record.update(state='not_p800', implementations={})
        return {}
    if model.training or torch.is_grad_enabled() or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Local templates require frozen inference')
    enable_vendor_capture()
    discovery = []
    def observer(gm, examples):
        selected, rejected = discover_regions(gm)
        discovery.append(dict(graph_code=gm.code, rejected=rejected,
            regions=[dict(template=t.template_id, kind=m.kind, nodes=[n.name for n in m.nodes],
                          outputs=[n.name for n in out]) for t,m,out in selected]))
        return gm.forward
    private = copy.deepcopy(model, {id(x):x for x in (*model.parameters(), *model.buffers())})
    capture = torch.compile(_capture_boundary(private), backend=observer, fullgraph=True, dynamic=False)
    try:
        for args, kwargs in validation_inputs:
            pargs, pkwargs = prepare(*args, **kwargs)
            output = capture(*pargs, **pkwargs)
            del output
    except torch._dynamo.exc.Unsupported as exc:
        record.update(state='capture_unsupported', error=str(exc), implementations={})
        return {}
    families = sorted({r['template'] for g in discovery for r in g['regions']} - {'quick_gelu'})
    quick = any(r['template']=='quick_gelu' for g in discovery for r in g['regions'])
    record.update(state='proposed' if families else 'no_supported_regions', discovery=discovery,
        registry_templates=[t.template_id for t in default_registry().templates],
        capture_source='original_dtype', implementations={}, rejected_implementations=[],
        scope='contiguous pure registry regions; live intermediates retained as extra outputs')
    plans = [(family, (family,)) for family in families]
    if len(families)>1:
        plans.append(('all', tuple(families)))
    if families and quick:
        plans.append(('all_quickgelu', (*families, 'quick_gelu')))
    routes = {}
    for plan, templates in plans:
        for suffix, size, row_options in _template_schedules(templates, buffer_sizes):
            name = 'local_templates_'+plan+'_'+suffix
            audit = dict(route='inductor_local_templates', templates=list(templates), buffer_bytes=size,
                row_options=row_options,
                interface='guarded_generated_wrapper',
                graphs=[], compiled_regions=[], capture_source='original_dtype',
                complete_output_validation_required=True, outside_regions='original captured FX targets')
            call = _Pipeline(model, prepare, templates, size, audit, row_options)
            try:
                call.prepare_examples(validation_inputs)
            except (_NoCompiledRegion, torch._dynamo.exc.Unsupported) as exc:
                record['rejected_implementations'].append(dict(name=name, reason=str(exc), audit=audit))
                continue
            routes[name] = ModelRoute(call)
            record['implementations'][name] = audit
    return routes
