"""Guarded inference capture and whole-graph Inductor entry.

The legacy region planner remains available for comparison. ``compile_model``
uses the whole-graph backend; unsupported operators retain vendor fallbacks.
"""
import importlib
import math
import statistics
import threading

import torch
from torch import fx

from ._patterns import default_registry

_CAPTURE_LOCK = threading.RLock()
_CAPTURE_POLYFILLS = {}


def enable_vendor_capture():
    """Use the vendor's existing opaque Linear op only during Dynamo capture.

    Its real CUDA implementation is the same _linear used by eager, including
    Hydra's stateful context. No environment flag or installed file is changed.
    """
    try:
        vendor = importlib.import_module('torch_xmlir.nn.linear')
    except ImportError:
        return 'not_applicable'
    original = getattr(vendor, 'linear', None)
    if not callable(original) or not hasattr(torch.ops._dynamo_workaround, 'linear'):
        return 'unavailable'
    with _CAPTURE_LOCK:
        if original not in _CAPTURE_POLYFILLS:
            def captured_linear(input, weight, bias=None, *, out=None):
                if out is not None:
                    return vendor._linear(input, weight, bias, out=out)
                return torch.ops._dynamo_workaround.linear(input, weight, bias)
            _CAPTURE_POLYFILLS[original] = torch.compiler.substitute_in_graph(
                original, skip_signature_check=True)(captured_linear)
    return 'vendor_opaque_linear'


def find_regions(gm):
    """Choose disjoint pure patterns without changing any external dependency."""
    chosen, occupied, rejected = [], set(), []
    calls = [n for n in gm.graph.nodes if n.op.startswith('call_')]
    positions = {n:i for i,n in enumerate(calls)}
    for node in reversed(tuple(gm.graph.nodes)):
        for template in default_registry().templates:
            match = template.matcher(node)
            if match is None:
                continue
            members = set(match.nodes)
            if members & occupied:
                continue
            # The original intermediate may be live across the region (e.g.
            # pre-norm residuals). Keep that pattern unchanged unless its ABI
            # explicitly supports returning both values.
            if any(user not in members for n in match.nodes[:-1] for user in n.users):
                rejected.append(dict(kind=match.kind, node=node.name, reason='intermediate has external users'))
                continue
            first,last=min(positions[n] for n in members),max(positions[n] for n in members)
            if set(calls[first:last+1])!=members:
                rejected.append(dict(kind=match.kind,node=node.name,reason='intervening original operation'))
                continue
            chosen.append(match)
            occupied.update(members)
            break
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    chosen.sort(key=lambda m: order[m.nodes[-1]])
    return chosen, rejected


class _Region(torch.nn.Module):
    def __init__(self, gm, match, input_order, tolerance, environment_key, record):
        super().__init__()
        self.graph = gm
        self.kind = match.kind
        self.options = dict(match.options)
        self.input_order = input_order
        self.tolerance = tolerance
        self.environment_key = environment_key
        self.record = record
        self._selected = None
        self._lock = threading.RLock()

    def forward(self, *args):
        if self._selected is None:
            with self._lock:
                if self._selected is None:
                    self._prepare(args)
        return self._selected(*args)

    def _prepare(self, args):
        original = self.graph.forward
        self._selected = original
        self.record.update(state='original', route='original', reason='unsupported runtime layout')
        if (torch.is_grad_enabled() or not all(type(x) in (torch.Tensor,torch.nn.Parameter) for x in args)
                or any(x.requires_grad for x in args)):
            return
        if not args or args[0].device.type != 'cuda':
            return
        # Shape guards belong to the surrounding Dynamo graph. The candidate
        # additionally checks its live pointer/layout ABI before a warm launch.
        from ._p800_residual import prepare as prepare_residual
        ordered = tuple(args[i] for i in self.input_order)
        custom = prepare_residual(self.kind, self.options, ordered)
        if custom is not None:
            execute, reference, proof = custom
            def candidate(*values):
                return execute(*(values[i] for i in self.input_order))
            want = reference(*ordered)
            got = candidate(*args)
            torch.cuda.synchronize()
            from ._resource_benchmark import assert_close, paired_end_to_end
            try:
                assert_close(got, want, self.tolerance, output_dtype=args[0].dtype)
            except AssertionError as exc:
                self.record.update(reason='candidate failed FP32 numerical validation',
                                   validated=False,error=str(exc),proof=proof)
                return
            # Compare the actual region entry, including its Python/launch
            # overhead. No device event, CUDA Graph, or persistent timing cache.
            timing = paired_end_to_end({'original': original, 'f06': candidate}, args,
                                      count=48, warmup=8, seed=371)
            self.record.update(proof=proof, calibration=timing, validated=True)
            if statistics.median(timing['f06']) < statistics.median(timing['original']):
                self._selected = candidate
                self.record.update(state='fused', route='residual_layernorm', reason='validated region end-to-end win')
            else:
                self.record.update(reason='validated candidate did not beat original region')
            return
        if self.kind.startswith('add_layernorm'):
            # Keep wider, tail-width, non-dense, mixed-dtype or partially affine model
            # regions on their original path. In particular, do not route an
            # unsupported 3-D broadcast through an unrelated 2-D bias kernel.
            return
        # Existing small-graph machinery remains bounded to the matched region.
        from ._resource_autotune import autotune_resource_backend
        decisions = []
        backend = autotune_resource_backend(tolerance=self.tolerance,
                    environment_key=self.environment_key, audit=decisions)
        compiled = backend(self.graph, args)
        compiled(*args)
        torch.cuda.synchronize()
        self._selected = compiled
        self.record.update(state='prepared', route='resource_backend', resource_audit=decisions)


def partition_backend(*, tolerance, environment_key, audit):
    def backend(gm, example_inputs):
        from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule
        matches, rejected = find_regions(gm)
        graph_record = dict(compute_nodes=sum(n.op.startswith('call_') for n in gm.graph.nodes),
                            regions=[], rejected=rejected, unmatched_execution='original FX targets')
        audit.append(graph_record)
        replacements,internal={},set()
        # First extract against the unchanged original graph. Then copy it in
        # exactly its original execution order. Generic topological sorting
        # does not encode read-after-mutation dependencies through aliases
        # (e.g. masked_fill_ followed by a dtype-changing to()).
        for index, match in enumerate(matches):
            name = '_f06_region_' + str(index)
            nodes = list(match.nodes)
            subgraph, inputs, outputs = fuse_as_graphmodule(gm, nodes, name,
                {n: None for n in nodes}, always_return_tuple=False)
            assert len(outputs) == 1
            mapping = tuple(inputs.index(n) for n in match.inputs)
            entry = dict(name=name, kind=match.kind, nodes=[n.name for n in match.nodes], state='pending')
            graph_record['regions'].append(entry)
            gm.add_module(name, _Region(subgraph, match, mapping, tolerance, environment_key, entry))
            replacements[match.nodes[-1]]=(name,inputs)
            internal.update(match.nodes[:-1])
        graph=fx.Graph();copied={}
        for node in gm.graph.nodes:
            if node in replacements:
                name,inputs=replacements[node]
                copied[node]=graph.call_module(name,tuple(copied[n] for n in inputs))
                copied[node].meta=dict(node.meta)
            elif node not in internal:
                copied[node]=graph.node_copy(node,lambda n:copied[n])
        gm.graph=graph
        gm.graph.lint()
        gm.recompile()
        graph_record['graph_code'] = gm.code
        return gm.forward
    return backend


def compile_model(model, *, tolerance=None, environment_key='f06-model-v1',
                  audit=None, fullgraph=True, dynamic=False, vendor_capture=True,
                  precision_policy='fp32_reference', precision_options=None,
                  resource_policy='auto', graph_choice=None,
                  _cost_graph_observer=None, _capture_cost_graph=False):
    """Compile an inference model through Inductor and select its full-graph route."""
    if model.training:
        raise ValueError('F06 whole-model compilation requires model.eval()')
    if dynamic is not False:
        raise ValueError('F06 model regions currently require static Dynamo guards')
    records = [] if audit is None else audit
    capture = enable_vendor_capture() if vendor_capture else 'disabled'
    if tolerance is None:
        dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
        if len(dtypes) != 1:
            raise ValueError('Provide tolerance for a model without one floating parameter dtype')
        dtype, = dtypes
        tolerance = dict(atol=.0003, rtol=.0003) if dtype == torch.float32 else dict(
            atol=.03 if dtype == torch.bfloat16 else .003, rtol=.003)
    from ._model_inductor import inductor_backend
    if resource_policy != 'auto' or precision_policy == 'source_fp32_mixed' or precision_options:
        # Dynamo caches by Python code object. Independent resource backends
        # applied to the same model.forward otherwise consume its shared
        # recompile budget (CLIP has more than eight complete alternatives).
        # Give each cold candidate its own capture boundary, not a larger
        # global budget or an unguarded callable. Dynamo inlines the wrapper.
        import types
        class ResourceCapture(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.model = original

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        forward = ResourceCapture.forward
        ResourceCapture.forward = types.FunctionType(
            forward.__code__.replace(), forward.__globals__, forward.__name__,
            forward.__defaults__, forward.__closure__)
        model = ResourceCapture(model).eval()
    compiled = torch.compile(model, backend=inductor_backend(tolerance=tolerance,
        environment_key=environment_key, audit=records, precision_policy=precision_policy,
        precision_options=precision_options, resource_policy=resource_policy, graph_choice=graph_choice,
        _cost_graph_observer=_cost_graph_observer, _capture_cost_graph=_capture_cost_graph),
        fullgraph=fullgraph, dynamic=dynamic)
    compiled._f06_audit = records
    compiled._f06_capture = capture
    return compiled
