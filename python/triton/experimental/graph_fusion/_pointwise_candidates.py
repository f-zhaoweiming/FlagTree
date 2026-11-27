"""Expose native pointwise implementations as independent graph candidates."""
import operator
from collections.abc import Mapping

import torch
from torch import fx


def preserve_pointwise(gm, policy, *, dry_run=False):
    assignments = dict(policy) if isinstance(policy, Mapping) else None
    if assignments is not None:
        if set(assignments.values()) - {'generated', 'native'}:
            raise ValueError('Per-node pointwise choices must be generated or native')
        missing = set(assignments) - {node.name for node in gm.graph.nodes}
        if missing:
            raise ValueError('Pointwise plan references missing graph nodes: ' + str(sorted(missing)))
    elif policy not in ('generated', 'native_mul', 'native'):
        raise ValueError('Unknown pointwise implementation policy: ' + str(policy))
    if policy == 'generated':
        return []
    from ._operator_precision import _native_schema, _lower_direct, _value
    from torch._inductor.lowering import make_fallback
    records = []
    applied = set()
    for node in gm.graph.nodes:
        choice = assignments.get(node.name, 'generated') if assignments is not None else policy
        if choice == 'generated':
            continue
        if node.op not in ('call_function', 'call_method'):
            continue
        if node.kwargs.get('out') is not None:
            continue
        native = None
        args = node.args
        kwargs = dict(node.kwargs)
        mul = (node.target in (operator.mul, torch.mul, torch.ops.aten.mul.Tensor,
                              torch.ops.aten.mul.Scalar) or
               node.op == 'call_method' and node.target == 'mul')
        add = (node.target in (operator.add, torch.add, torch.ops.aten.add.Tensor,
                              torch.ops.aten.add.Scalar) or
               node.op == 'call_method' and node.target == 'add')
        if (mul or add and choice == 'native') and len(args) == 2:
            if set(kwargs) - ({'alpha'} if add else set()):
                continue
            a, b = args
            if not isinstance(_value(a), torch.Tensor):
                # Only multiplication may reorder a scalar and a tensor.
                if not mul or not isinstance(_value(b), torch.Tensor):
                    continue
                a, b = b, a
            if not isinstance(_value(b), torch.Tensor) and type(b) not in (float, int):
                continue
            kind = 'Tensor' if isinstance(_value(b), torch.Tensor) else 'Scalar'
            native = getattr(torch.ops.aten.mul if mul else torch.ops.aten.add, kind)
            args = (a, b)
        elif choice == 'native' and len(args) == 1 and not kwargs and (
                node.target in (torch.sigmoid, torch.ops.aten.sigmoid.default) or
                node.op == 'call_method' and node.target == 'sigmoid'):
            native = torch.ops.aten.sigmoid.default
        if native is None:
            continue
        value = _value(node)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        if dry_run:
            records.append(dict(node=node.name, kind='pointwise_choice_domain',
                                original=str(node.target), native=str(native), dtype=str(value.dtype),
                                choices=['generated', 'native'], preserve_output_dtype=True))
            applied.add(node.name)
            continue
        target = _native_schema(native)
        if target is None:
            continue
        make_fallback(target, warn=False)
        _lower_direct(target, native)
        original = str(node.target)
        node.op, node.target, node.args, node.kwargs = 'call_function', target, args, kwargs
        records.append(dict(node=node.name, kind='native_pointwise_candidate',
                            original=original, native=str(native), dtype=str(value.dtype)))
        applied.add(node.name)
    if assignments is not None:
        unsupported = {name for name, choice in assignments.items() if choice == 'native'} - applied
        if unsupported:
            raise ValueError('Pointwise plan requests unsupported native nodes: ' + str(sorted(unsupported)))
    if not dry_run:
        gm.graph.lint()
        gm.recompile()
    return records
