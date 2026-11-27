"""Remove low-precision round trips inside proven FP32 compute regions.

Eligibility is an arithmetic contract, not a model, shape, or template match.
Native FP16 matrix modes and FP16 ReLU/clamp are deliberately absent. New
operator/dtype contracts require independent native-code evidence first.
"""
import operator

import torch
import torch.nn.functional as F
from torch import fx


def _value(node):
    return node.meta.get('example_value', node.meta.get('val')) if isinstance(node, fx.Node) else None


def _dtype(node):
    return getattr(_value(node), 'dtype', None)


_ADD = (operator.add, torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
_MUL = (operator.mul, torch.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
_SIGMOID = (torch.sigmoid, torch.ops.aten.sigmoid.default)
_NORM = (F.layer_norm, torch.ops.aten.layer_norm.default)
_VIEWS = {'view', 'reshape', 'transpose', 'permute', 'flatten', 'unsqueeze', 'squeeze', 'expand'}
_CASTS = {'float', 'half', 'to'}


def _scaled_add_args(node):
    """Return the inputs and scale of an add outside the unit-scale contract."""
    if not (node.op == 'call_function' and node.target in _ADD or
            node.op == 'call_method' and node.target == 'add'):
        return None
    args = node.args
    if len(args) == 3:
        # ATen add.Scalar(self, other, alpha) and the legacy Python
        # add(input, alpha, other) use different positional conventions.
        if node.target is torch.ops.aten.add.Scalar:
            left, other, alpha = args
        else:
            left, alpha, other = args
    else:
        alpha = node.kwargs.get('alpha', 1)
        left = args[0] if args else node.kwargs.get('input', node.kwargs.get('self'))
        other = args[1] if len(args) > 1 else node.kwargs.get('other')
    if type(alpha) in (int, float) and alpha == 1:
        return None
    return left, other, alpha


def _kind(node):
    value = _value(node)
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float16, torch.float32):
        return None
    if node.kwargs.get('inplace', False) or 'out' in node.kwargs:
        return None
    if node.kwargs.get('dtype') not in (None, torch.float16, torch.float32):
        return None
    if _scaled_add_args(node) is not None:
        return None
    if node.op == 'call_function':
        if node.target in _ADD + _MUL + _SIGMOID + _NORM:
            return 'compute'
    elif node.op == 'call_method':
        if node.target in ('add', 'mul', 'sigmoid'):
            return 'compute'
        if node.target in _VIEWS:
            return 'view'
        if (node.target in ('float', 'half') and len(node.args) == 1 and not node.kwargs
                or node.target == 'to' and (len(node.args) == 2 and not node.kwargs
                    and isinstance(node.args[1], torch.dtype) and node.args[1] in (torch.float16, torch.float32)
                    or len(node.args) == 1 and set(node.kwargs) == {'dtype'}
                    and node.kwargs['dtype'] in (torch.float16, torch.float32))):
            return 'cast'
    return None


def _effect(node):
    if node.op in ('placeholder', 'get_attr', 'output'):
        return False
    if (node.kwargs.get('inplace', False) or 'out' in node.kwargs
            or node.op == 'call_method' and str(node.target).endswith('_')
            or getattr(getattr(node.target, '_schema', None), 'is_mutable', False)):
        return True
    if node.op == 'call_module':
        return True
    if _kind(node) is not None:
        return False
    if node.op == 'call_function' and hasattr(node.target, '_schema'):
        return node.is_impure()
    if node.op == 'call_function' and node.target in (
            F.linear, torch.mm, torch.bmm, torch.matmul, F.scaled_dot_product_attention,
            F.conv2d, torch.cat, torch.relu, F.relu, torch.clamp, F.gelu, F.silu,
            F.softmax, torch.softmax, torch.exp, torch.div):
        return False
    if node.op == 'call_function' and node.target is F.embedding:
        return node.kwargs.get('max_norm', node.args[3] if len(node.args) > 3 else None) is not None
    if str(node.target) in ('_dynamo_workaround.linear', '_dynamo_workaround.linear.default'):
        return False
    if (node.op == 'call_method' and node.target in ('size', 'stride', 'numel', 'dim')
            or node.op == 'call_function' and node.target is getattr and len(node.args) == 2
            and node.args[1] in ('shape', 'dtype', 'device', 'ndim', 'layout')
            or node.op == 'call_function' and node.target is operator.getitem
            and not isinstance(_value(node), torch.Tensor)):
        return False
    # Unknown Python calls and methods may observe or mutate an alias. Do not
    # move an ingress cast or bypass an observable rounding across them.
    return True


def fuse_fp32_compute_regions(gm, example_inputs=None):
    """Widen connected pure regions and round only their observable exits.

FP16 add/multiply/sigmoid and LayerNorm have decoded FP32 arithmetic in the
reviewed P800 SDK (M06). LayerNorm moments also use FP32 accumulation. BF16
does not inherit these contracts just because its storage is also 16 bits.
Unverified operators stay outside the region and keep their native dtype.
"""
    from ._model_inductor import refresh_graph_metadata
    from ._p800_model_layernorm import _final_cast

    graph = gm.graph
    nodes = list(graph.nodes)
    positions = {n: i for i, n in enumerate(nodes)}
    kinds = {n: _kind(n) for n in nodes}
    dtypes = {n: _dtype(n) for n in nodes}
    original_args = {n: (n.args, dict(n.kwargs)) for n in nodes}
    segments, segment = {}, 0
    for node in nodes:
        if _effect(node):
            segment += 1
        segments[node] = segment
    eligible = {n for n in nodes if kinds[n] is not None}
    # Never reinterpret integer/bool inputs, BF16, or mixed unreviewed types.
    eligible = {n for n in eligible if all(_dtype(a) in (None, torch.float16, torch.float32)
                for a in n.all_input_nodes)}
    groups, visited = [], set()
    for node in nodes:
        if node not in eligible or node in visited:
            continue
        group, pending = set(), [node]
        while pending:
            current = pending.pop()
            if current in group:
                continue
            group.add(current)
            pending.extend(n for n in (*current.all_input_nodes, *current.users)
                           if n in eligible and n not in group and segments[n] == segments[current])
        visited.update(group)
        if (sum(kinds[n] == 'compute' for n in group) >= 2
                and any(dtypes[n] == torch.float16 and any(u in group for u in n.users) for n in group)):
            groups.append(group)
    if not groups:
        return []
    group_of = {n: i for i, group in enumerate(groups) for n in group}
    first = {i: min(group, key=positions.get) for i, group in enumerate(groups)}
    ingress, exports = {}, {}
    records = []
    for i, group in enumerate(groups):
        records.append(dict(kind='fp32_compute_region', nodes=[n.name for n in nodes if n in group],
            compute_nodes=[n.name for n in nodes if n in group and kinds[n] == 'compute'],
            internal_low_edges=sum(dtypes[n] == torch.float16 for n in group for u in n.users if u in group),
            compute_dtype='torch.float32', accumulator='FP32 for reviewed LayerNorm; no reduction in pointwise ops',
            evidence='M06 P800 decoded FP16 add/multiply/sigmoid and LayerNorm; no matrix or half-arithmetic promotion',
            input_conversions=[], output_conversions=[], state='candidate_requires_complete_FP32_model_validation'))
    for node in nodes:
        if node not in group_of:
            continue
        i = group_of[node]
        def widen(arg):
            if group_of.get(arg) == i or dtypes.get(arg) != torch.float16:
                return arg
            key = (i, arg)
            if key not in ingress:
                # Each component stays within one effect-free interval. An
                # already defined input can be widened once at region entry.
                anchor = first[i] if positions[arg] < positions[first[i]] else node
                with graph.inserting_before(anchor):
                    ingress[key] = graph.call_method('float', (arg,))
                records[i]['input_conversions'].append(ingress[key].name)
            return ingress[key]
        node.args = fx.map_arg(node.args, widen)
        node.kwargs = fx.map_arg(node.kwargs, widen)
        if kinds[node] == 'cast':
            node.op, node.target, node.args, node.kwargs = 'call_method', 'to', (node.args[0], torch.float32), {}
        node.meta['f06_fp32_compute_region'] = i

    def export(node):
        if dtypes[node] != torch.float16:
            return node
        if node in exports:
            return exports[node]
        i = group_of[node]
        args, kwargs = original_args[node]
        source = args[0] if args and isinstance(args[0], fx.Node) else None
        # Reconstruct observable views from one rounded base, preserving shared
        # storage and identity casts rather than rounding each alias separately.
        if kinds[node] in ('view', 'cast') and dtypes.get(source) == dtypes[node]:
            base = export(source) if source in group_of else source
            if kinds[node] == 'cast':
                exports[node] = base
                return base
            with graph.inserting_after(node):
                out = graph.call_method(node.target, (base, *args[1:]), kwargs)
        else:
            # M06: when a live FP32 intermediate also escapes as FP16, keep its
            # boundary conversion separate from mixed-output pointwise stores.
            def needs_wide(current, seen):
                for user in current.users:
                    if group_of.get(user) != i or user in seen:
                        continue
                    seen.add(user)
                    if (kinds[user] == 'compute' or dtypes[user] == torch.float32
                            and any(group_of.get(u) != i for u in user.users)
                            or needs_wide(user, seen)):
                        return True
                return False
            live_wide = needs_wide(node, set())
            with graph.inserting_after(node):
                out = graph.call_function(torch.ops.flagtree_p800.final_cast.default,
                                          (node, torch.float16)) if live_wide else graph.call_method('half', (node,))
            records[i]['output_conversions'].append(out.name)
        out.meta['f06_region_output_dtype'] = 'torch.float16'
        exports[node] = out
        return out

    # Use the original users, not ingress casts or newly created export nodes.
    # A later region must re-read a rounded value after an intervening effect.
    for user in nodes:
        def boundary(arg):
            if arg in group_of and group_of.get(user) != group_of[arg] and dtypes[arg] == torch.float16:
                return export(arg)
            return arg
        if user not in group_of:
            user.args = fx.map_arg(user.args, boundary)
            user.kwargs = fx.map_arg(user.kwargs, boundary)
    for (i, source), cast in ingress.items():
        if source in group_of and group_of[source] != i:
            cast.args = (export(source),)
    graph.lint()
    gm.recompile()
    refresh_graph_metadata(gm, example_inputs)
    return records


def fold_layernorm_boundary_casts(gm, example_inputs=None):
    """Let the reviewed template load/store FP16 around its FP32 arithmetic.

    Only collapse complete template boundaries. A live FP32 residual or norm
    user prevents folding, so a larger FP32 region never gains a new rounding.
    The existing template widens both rows before adding and computing moments.
    """
    from ._model_inductor import refresh_graph_metadata
    graph = gm.graph
    records = []
    for node in list(graph.nodes):
        if node.op != 'call_function' or node.target != torch.ops.flagtree_p800.model_layernorm.default:
            continue
        args = list(node.args)
        if len(args) != 7 or node.kwargs:
            continue
        casts = []
        valid = True
        for value in args[:4]:
            if value is None:
                continue
            if (not isinstance(value, fx.Node) or _kind(value) != 'cast'
                    or _dtype(value) != torch.float32 or _dtype(value.args[0]) != torch.float16):
                valid = False
                break
            casts.append(value)
        if not valid or len(casts) not in (2, 4):
            continue
        casts = list(dict.fromkeys(casts))
        outputs, exits = list(node.users), []
        if not outputs:
            continue
        for output in outputs:
            if (output.op != 'call_function' or output.target is not operator.getitem
                    or output.args[1] not in (0, 1) or not output.users):
                valid = False
                break
            for user in output.users:
                if (_dtype(user) != torch.float16 or not (
                        _kind(user) == 'cast' and user.args[0] is output
                        or user.op == 'call_function'
                        and user.target == torch.ops.flagtree_p800.final_cast.default
                        and user.args == (output, torch.float16))):
                    valid = False
                    break
                exits.append((user, output))
        if not valid:
            continue
        # Do not hoist reads across a mutation or effect between the casts and
        # the template, even if the value has the same storage dtype.
        positions = {n: i for i, n in enumerate(graph.nodes)}
        ordered = list(graph.nodes)
        if any(_effect(n) for cast in casts for n in ordered[positions[cast]+1:positions[node]]):
            continue
        args[:4] = [x.args[0] if x in casts else x for x in args[:4]]
        node.args = tuple(args)
        for cast, output in exits:
            cast.replace_all_uses_with(output)
            graph.erase_node(cast)
        for cast in reversed(casts):
            if not cast.users:
                graph.erase_node(cast)
        records.append(dict(kind='fp32_layernorm_fused_io', node=node.name,
            storage_dtype='torch.float16', compute_dtype='torch.float32',
            accumulator_dtype='torch.float32', input_casts_folded=len(casts),
            output_casts_folded=len(exits), intermediate_rounding=False))
    if records:
        graph.lint()
        gm.recompile()
        refresh_graph_metadata(gm, example_inputs)
    return records
