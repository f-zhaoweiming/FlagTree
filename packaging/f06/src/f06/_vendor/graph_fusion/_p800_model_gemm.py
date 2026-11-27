"""FP32 GEMM layout candidates for complete inference graphs."""
import copy
import math
import threading
import operator

import torch
import torch.nn.functional as F


def pack_column_major(weight):
    """Same values/shape/dtype, with the GEMM's right operand contiguous."""
    return weight.t().contiguous().t()


def packed_weight_candidate(gm, example_inputs):
    from ._model_inductor import _value, refresh_graph_metadata
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    indices, records = set(), []
    linear = (F.linear, torch.ops.aten.linear.default)
    for node in gm.graph.nodes:
        if node.op != 'call_function' or node.target not in linear or len(node.args) < 2:
            continue
        x, w = node.args[:2]
        if w not in placeholders:
            continue
        index = placeholders.index(w)
        parameter = example_inputs[index]
        if (not isinstance(parameter, torch.nn.Parameter) or parameter.requires_grad
                or any(u.op != 'call_function' or u.target not in linear
                       or len(u.args) < 2 or u.args[1] is not w for u in w.users)):
            continue
        a, b = _value(x), _value(w)
        if not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                   and v.layout == torch.strided and v.is_contiguous() for v in (a, b)):
            continue
        if a.ndim < 2 or b.ndim != 2 or not all(type(s) is int for s in (*a.shape, *b.shape)):
            continue
        m, k, n = math.prod(a.shape[:-1]), a.shape[-1], b.shape[0]
        if b.shape[1] != k or m < 4096 or n < 128 or k < 4 * n or m < 4 * n:
            continue
        indices.add(index)
        records.append(dict(kind='fp32_column_major_weight', node=node.name,
                            m=m, n=n, k=k, parameter_index=index, state='candidate_in_inductor_graph'))
    if not indices:
        return None
    mode = detect_fake_mode(example_inputs) or FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        examples = [torch.empty_strided(x.shape, (1, x.shape[0]), dtype=x.dtype, device=x.device)
                    if i in indices else x for i, x in enumerate(example_inputs)]
    graph = copy.deepcopy(gm)
    for node, value in zip((n for n in graph.graph.nodes if n.op == 'placeholder'), examples):
        node.meta['example_value'] = value
        node.meta.pop('val', None)
    refresh_graph_metadata(graph, examples)
    return graph, examples, sorted(indices), records


def _projection_motion_is_safe(node):
    """Prove that moving projections across this node cannot skip a write.

    FX's is_impure() is a useful positive check, not a purity guarantee for
    Python functions/methods. In particular torch.add(..., out=x) can be
    reported pure even when x aliases a projection input. Keep unknown calls
    and modules as barriers; accept schema-declared pure operators and the
    small set of functional shape/arithmetic operations used by these graphs.
    """
    if node.is_impure() or node.kwargs.get('out') is not None:
        return False
    if node.kwargs.get('inplace', False) is not False:
        return False
    if node.op == 'call_method':
        return node.target in ('view', 'reshape', 'transpose', 'permute',
                               'flatten', 'unsqueeze', 'squeeze', 'contiguous',
                               'clone', 'detach', 'size', 'dim', 'numel', 'stride')
    if node.op == 'call_function':
        schema = getattr(node.target, '_schema', None)
        if schema is not None:
            return not schema.is_mutable
        if node.target is F.linear:
            return len(node.args) in (2, 3) and not node.kwargs
        return node.target in (operator.getitem, operator.add, operator.sub,
                               operator.mul, operator.truediv, operator.floordiv,
                               operator.mod, operator.neg)
    return False


def grouped_projection_candidate(gm, example_inputs):
    """Combine independent frozen FP32 projections of the same activation.

    Packing changes only the output-column dimension. No operand narrowing,
    K partitioning or activation cache is introduced. Complete-model precision
    and timing still decide whether this candidate is admitted.
    """
    from ._model_inductor import _value, refresh_graph_metadata
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from . import _p800_model_schedule  # Register the existing native layout copy.
    graph = copy.deepcopy(gm)
    placeholders = [n for n in graph.graph.nodes if n.op == 'placeholder']
    positions = {n: i for i, n in enumerate(placeholders)}
    linear = (F.linear, torch.ops.aten.linear.default)
    by_input = {}
    for node in graph.graph.nodes:
        if node.op != 'call_function' or node.target not in linear or node.kwargs:
            continue
        if len(node.args) not in (2, 3):
            continue
        x, w = node.args[:2]
        bias = node.args[2] if len(node.args) == 3 else None
        if w not in positions or (bias is not None and bias not in positions):
            continue
        params = [w] + ([bias] if bias is not None else [])
        if any(not isinstance(example_inputs[positions[p]], torch.nn.Parameter)
               or example_inputs[positions[p]].requires_grad or len(p.users) != 1 for p in params):
            continue
        a, b = _value(x), _value(w)
        if not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                   and v.layout == torch.strided and v.is_contiguous() for v in (a, b)):
            continue
        if a.ndim < 2 or b.ndim != 2 or not all(type(s) is int for s in (*a.shape, *b.shape)):
            continue
        if a.shape[-1] != b.shape[1]:
            continue
        if bias is not None:
            bv = _value(bias)
            if (not isinstance(bv, torch.Tensor) or bv.dtype != torch.float32
                    or bv.shape != (b.shape[0],) or not bv.is_contiguous()):
                continue
        by_input.setdefault((x, bias is not None), []).append(node)
    groups, records = [], []
    examples = list(example_inputs)
    mode = detect_fake_mode(example_inputs) or FakeTensorMode(allow_non_fake_inputs=True)
    for (x, has_bias), nodes in by_input.items():
        if len(nodes) < 2:
            continue
        weights = [n.args[1] for n in nodes]
        widths = [_value(w).shape[0] for w in weights]
        # Larger row sets lost to layout traffic in complete-model validation.
        # This only bounds eligibility; measured complete-graph timing selects.
        av = _value(x)
        if (len(nodes) != 3 or len(set(widths)) != 1 or av.ndim != 3
                or math.prod(av.shape[:-1]) > 2 * widths[0]):
            continue
        endpoints, head_shapes = [], []
        for node in nodes:
            if len(node.users) != 1:
                break
            view = next(iter(node.users))
            if not (view.op == 'call_method' and view.target in ('view', 'reshape')
                    and len(view.users) == 1):
                break
            endpoint = next(iter(view.users))
            vv = _value(view)
            if not (endpoint.op == 'call_method' and endpoint.target == 'transpose'
                    and endpoint.args[1:] == (1, 2) and not endpoint.kwargs
                    and isinstance(vv, torch.Tensor) and vv.ndim == 4
                    and all(type(s) is int for s in vv.shape)
                    and tuple(vv.shape[:2]) == tuple(av.shape[:2])
                    and vv.shape[2] * vv.shape[3] == widths[0]
                    and not any(u.op == 'output' for u in endpoint.users)):
                break
            endpoints.append(endpoint)
            head_shapes.append(tuple(vv.shape))
        if len(endpoints) != 3 or len(set(head_shapes)) != 1:
            continue
        order = list(graph.graph.nodes)
        between = order[order.index(nodes[0]):order.index(nodes[-1])]
        if any(not _projection_motion_is_safe(n) for n in between):
            continue
        indices = tuple(positions[w] for w in weights)
        groups.append(indices)
        with mode:
            examples[indices[0]] = torch.empty((sum(widths), _value(weights[0]).shape[1]),
                                               dtype=torch.float32, device=_value(weights[0]).device)
        bias = None
        if has_bias:
            biases = [n.args[2] for n in nodes]
            bias_indices = tuple(positions[b] for b in biases)
            groups.append(bias_indices)
            bias = biases[0]
            with mode:
                examples[bias_indices[0]] = torch.empty(sum(widths), dtype=torch.float32,
                                                       device=_value(bias).device)
        with graph.graph.inserting_before(nodes[0]):
            combined = graph.graph.call_function(torch.ops.aten.linear.default, (x, weights[0], bias))
            batch, seq, heads, dim = head_shapes[0]
            viewed = graph.graph.call_function(torch.ops.aten.reshape.default,
                                              (combined, [batch, seq, 3, heads, dim]))
            permuted = graph.graph.call_function(torch.ops.aten.permute.default,
                                                (viewed, [2, 0, 1, 3, 4]))
            # One native copy preserves eager B,S,H,D storage for all three projections.
            # The generic strided views and generated three-copy alternatives
            # were slower and are deliberately not offered as candidates.
            dense = graph.graph.call_function(torch.ops.flagtree_p800.model_contiguous.default, (permuted,))
            for i, endpoint in enumerate(endpoints):
                item = graph.graph.call_function(operator.getitem, (dense, i))
                item = graph.graph.call_function(torch.ops.aten.transpose.int, (item, 1, 2))
                endpoint.replace_all_uses_with(item)
        records.append(dict(kind='fp32_grouped_projections', nodes=[n.name for n in nodes],
                            widths=widths, parameter_indices=list(indices),
                            layout='one_native_copy_to_3_B_S_H_D_then_view',
                            state='candidate_in_inductor_graph'))
        # Dynamo graphs can contain unused-result mutations (for example the
        # causal mask's masked_fill_). Generic FX dead-code elimination does
        # not recognize every such call_method. Remove only the pure nodes
        # replaced above; preserve all unrelated graph operations verbatim.
        for node, endpoint in zip(nodes, endpoints):
            view = endpoint.args[0]
            for retired in (endpoint, view, node):
                assert not retired.users
                graph.graph.erase_node(retired)
    if not groups:
        return None
    for node, value in zip(placeholders, examples):
        node.meta['example_value'] = value
        node.meta.pop('val', None)
    graph.graph.lint()
    graph.recompile()
    refresh_graph_metadata(graph, examples)
    return graph, examples, groups, records


class GroupedProjectionPack:
    """Versioned frozen-weight concatenations, safe across inference streams."""
    def __init__(self, fn, groups):
        self.fn, self.groups = fn, tuple(tuple(g) for g in groups)
        self.entries = {}
        self.lock = threading.RLock()
        self.ready = None
        self.streams = set()

    def __call__(self, *args):
        from ._model_inductor import _ParameterPack
        with self.lock:
            values = list(args)
            device = args[self.groups[0][0]].device
            stream = torch.cuda.current_stream(device) if device.type == 'cuda' else None
            if self.ready is not None and stream.cuda_stream not in self.streams:
                stream.wait_event(self.ready)
                self.streams.add(stream.cuda_stream)
            changed = False
            for group in self.groups:
                source = tuple(args[i] for i in group)
                if any(not isinstance(x, torch.nn.Parameter) or x.requires_grad for x in source):
                    raise RuntimeError('Grouped projection weights must remain frozen Parameters')
                key = tuple(_ParameterPack.key(x) for x in source)
                entry = self.entries.get(group)
                if any(k is None for k in key) or entry is None or entry[0] != key:
                    entry = (key, source, torch.cat(source, dim=0))
                    self.entries[group] = entry
                    changed = True
                values[group[0]] = entry[2]
                if stream is not None:
                    entry[2].record_stream(stream)
            if stream is not None and changed:
                self.ready = torch.cuda.Event()
                self.ready.record(stream)
                self.streams = {stream.cuda_stream}
            return self.fn(*values)
