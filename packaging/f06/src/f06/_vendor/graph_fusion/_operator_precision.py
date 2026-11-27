"""Precision boundaries for an original-dtype graph, before fusion rewrites.

Storage dtype is not an arithmetic or accumulator contract. In particular a
native half Linear must not be replaced by a widened aten.mm merely because
some pointwise kernels compute in float. This policy keeps native compute
nodes opaque and lets Inductor schedule explicit conversions around them.
"""
import operator
import hashlib
import threading

import torch
import torch.nn.functional as F
from torch import fx
from torch.utils._pytree import tree_leaves


LOW = (torch.float16, torch.bfloat16)
_LIBRARIES = {}
_REGISTRATION_LOCK = threading.RLock()


def _lower_direct(target, native):
    # The custom schema is an AOT boundary, not a runtime Python wrapper.
    # Emit the exact native ATen entry after scheduling has kept that boundary.
    from torch._inductor.lowering import register_lowering, fallback_handler
    @register_lowering(target, type_promotion_kind=None)
    def lower(*args, **kwargs):
        return fallback_handler(native, add_to_fallback_set=False)(*args, **kwargs)


def _lower_reviewed(target, op):
    from torch._inductor.lowering import register_lowering, fallback_handler
    if op is _relu:
        _lower_direct(target, torch.ops.aten.relu.default)
    elif op is _clamp:
        _lower_direct(target, torch.ops.aten.clamp.default)
    elif op is _matmul:
        @register_lowering(target, type_promotion_kind=None)
        def lower(x, y, kind):
            native = getattr(torch.ops.aten, kind).default
            return fallback_handler(native, add_to_fallback_set=False)(x, y)
    elif op is _linear and hasattr(torch.ops._dynamo_workaround, 'linear'):
        @register_lowering(target, type_promotion_kind=None)
        def lower(x, w, b):
            # Only the vendor entry preserves P800 Linear's matrix mode and
            # bias behavior. aten.linear is not an interchangeable substitute.
            native = torch.ops._dynamo_workaround.linear.default if x.get_device().type == 'cuda' else target
            return fallback_handler(native, add_to_fallback_set=False)(x, w, b)


def _native_schema(op):
    """Keep an unreviewed pure ATen operation opaque instead of promoting it."""
    schema = getattr(op, '_schema', None)
    if schema is None or any(a.alias_info is not None for a in (*schema.arguments, *schema.returns)):
        return None
    with _REGISTRATION_LOCK:
        if op not in _LIBRARIES:
            text = str(schema)
            name = 'precision_native_' + hashlib.sha256(text.encode()).hexdigest()[:24]
            lib = torch.library.Library('flagtree_p800', 'FRAGMENT')
            lib.define(name + text[text.index('('):])
            lib.impl(name, op, 'CompositeExplicitAutograd')
            torch.library.register_fake('flagtree_p800::' + name)(lambda *args, **kwargs: op(*args, **kwargs))
            _LIBRARIES[op] = (lib, getattr(torch.ops.flagtree_p800, name).default)
        return _LIBRARIES[op][1]


def _value(node):
    return node.meta.get('example_value', node.meta.get('val')) if isinstance(node, fx.Node) else node


def _tensors(tree):
    return [v for n in tree_leaves(tree) for v in tree_leaves(_value(n)) if isinstance(v, torch.Tensor)]


def review_precision(gm):
    """Audit every compute node, without inferring arithmetic from its name."""
    records = []
    for node in gm.graph.nodes:
        if not node.op.startswith('call_'):
            continue
        inputs, outputs = _tensors((node.args, node.kwargs)), _tensors(node)
        low = any(v.dtype in LOW for v in inputs + outputs)
        known = bool(outputs) and all(not v.is_floating_point() or v.dtype == torch.float32
                                      for v in inputs + outputs)
        row = dict(node=node.name, operator=str(node.target),
                   input_dtypes=[str(v.dtype) for v in inputs],
                   output_dtypes=[str(v.dtype) for v in outputs],
                   consumer_dtypes={n.name: [str(v.dtype) for v in _tensors(n)] for n in node.users},
                   internal_arithmetic='not_established', accumulator='not_established',
                   rounding='preserve_graph_casts',
                   action='review_fp32_region' if known else 'preserve_native_precision',
                   contains_low_precision=low)
        # These statements refer to the decoded P800 samples, not a claim
        # about every shape, dtype, SDK version or internal accumulator stage.
        if low and node.target in (torch.relu, F.relu, torch.clamp, torch.ops.aten.relu.default,
                                   torch.ops.aten.clamp.default):
            row['evidence'] = 'P800 FP16 ReLU/clamp samples contain vmax.hf/vmin.hf (M06); preserve native for both low dtypes'
        elif low and ('linear' in str(node.target) or node.target in
                      (torch.mm, torch.matmul, torch.bmm, torch.addmm)):
            row['evidence'] = 'P800 sampled Linear modes 37/8 differ; FP16 mm mode 5; accumulator not inferred (M60)'
        node.meta['f06_precision_contract'] = dict(row)
        records.append(row)
    return records


# Explicit op boundaries survive AOT decomposition. The implementations use
# the original vendor entry and original dtype, including its rounding. No
# tensor values, model parameters or outputs are cached by these wrappers.
@torch.library.custom_op('flagtree_p800::precision_linear', mutates_args=())
def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    return F.linear(x, w, b)


@_linear.register_fake
def _linear_fake(x, w, b):
    return torch.empty((*x.shape[:-1], w.shape[0]), dtype=x.dtype, device=x.device)


@torch.library.custom_op('flagtree_p800::precision_matmul', mutates_args=())
def _matmul(x: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    return {'mm': torch.mm, 'bmm': torch.bmm, 'matmul': torch.matmul}[kind](x, y)


@_matmul.register_fake
def _matmul_fake(x, y, kind):
    return {'mm': torch.mm, 'bmm': torch.bmm, 'matmul': torch.matmul}[kind](x, y)


@torch.library.custom_op('flagtree_p800::precision_relu', mutates_args=())
def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@_relu.register_fake
def _relu_fake(x):
    return torch.relu(x)


@torch.library.custom_op('flagtree_p800::precision_clamp', mutates_args=())
def _clamp(x: torch.Tensor, minimum: float | None, maximum: float | None) -> torch.Tensor:
    return torch.clamp(x, min=minimum, max=maximum)


@_clamp.register_fake
def _clamp_fake(x, minimum, maximum):
    return torch.clamp(x, min=minimum, max=maximum)


@torch.library.custom_op('flagtree_p800::precision_conv2d', mutates_args=())
def _conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
            stride: list[int], padding: list[int], dilation: list[int], groups: int) -> torch.Tensor:
    return F.conv2d(x, w, b, stride, padding, dilation, groups)


@_conv2d.register_fake
def _conv2d_fake(x, w, b, stride, padding, dilation, groups):
    return F.conv2d(x, w, b, stride, padding, dilation, groups)


@torch.library.custom_op('flagtree_p800::precision_embedding', mutates_args=())
def _embedding(ids: torch.Tensor, w: torch.Tensor, padding: int | None,
               norm_type: float, scale_grad: bool, sparse: bool) -> torch.Tensor:
    return F.embedding(ids, w, padding, None, norm_type, scale_grad, sparse)


@_embedding.register_fake
def _embedding_fake(ids, w, padding, norm_type, scale_grad, sparse):
    return F.embedding(ids, w, padding, None, norm_type, scale_grad, sparse)


@torch.library.custom_op('flagtree_p800::precision_norm', mutates_args=())
def _norm(x: torch.Tensor, p: float, dim: list[int] | None, keepdim: bool) -> torch.Tensor:
    return x.norm(p=p, dim=dim, keepdim=keepdim)


@_norm.register_fake
def _norm_fake(x, p, dim, keepdim):
    return x.norm(p=p, dim=dim, keepdim=keepdim)


@torch.library.custom_op('flagtree_p800::precision_normalize', mutates_args=())
def _normalize(x: torch.Tensor, p: float, dim: int, eps: float) -> torch.Tensor:
    return F.normalize(x, p=p, dim=dim, eps=eps)


@_normalize.register_fake
def _normalize_fake(x, p, dim, eps):
    return F.normalize(x, p=p, dim=dim, eps=eps)


def preserve_native_compute(gm):
    """Protect reviewed pure native entries; retain all other original nodes.

No cast is removed here. In-place/out= forms are never replaced with pure
ops. Unknown arithmetic is not authorized for FP32 promotion by this pass.
"""
    from torch._inductor.lowering import make_fallback
    records = review_precision(gm)
    by_name = {r['node']: r for r in records}
    mm = {torch.mm: 'mm', torch.bmm: 'bmm', torch.matmul: 'matmul', operator.matmul: 'matmul',
          torch.ops.aten.mm.default: 'mm', torch.ops.aten.bmm.default: 'bmm',
          torch.ops.aten.matmul.default: 'matmul'}
    pure = {torch.sigmoid: torch.ops.aten.sigmoid.default, torch.exp: torch.ops.aten.exp.default,
            torch.sqrt: torch.ops.aten.sqrt.default, torch.cat: torch.ops.aten.cat.default,
            torch.stack: torch.ops.aten.stack.default,
            F.scaled_dot_product_attention: torch.ops.aten.scaled_dot_product_attention.default}
    views = {'view', 'reshape', 'flatten', 'transpose', 'permute', 'expand', 'unsqueeze',
             'squeeze', 't', 'contiguous', 'clone', 'detach', 'to', 'type_as', 'float', 'half', 'bfloat16'}
    widened_pointwise = (operator.add, operator.mul, torch.add, torch.mul)
    for node in gm.graph.nodes:
        row = by_name.get(node.name)
        if row is None:
            continue
        # The captured vendor entry needs the reviewed fake ABI for FP32 too.
        # Otherwise AOT can execute Hydra's data_ptr checks on FakeTensors.
        # Preserve only an already captured vendor/custom Linear here; this
        # must not turn an ordinary FP32 aten.linear into a forced fallback.
        if node.op == 'call_function' and (node.target is _linear._opoverload or
                str(node.target) in ('_dynamo_workaround.linear', '_dynamo_workaround.linear.default')):
            bound = dict(zip(('input', 'weight', 'bias'), node.args), **node.kwargs)
            if len(node.args) <= 3 and not set(bound) - {'input', 'weight', 'bias'} and {'input', 'weight'} <= set(bound):
                node.target = _linear._opoverload
                node.args = (bound['input'], bound['weight'], bound.get('bias'))
                node.kwargs = {}
                make_fallback(node.target, warn=False)
                _lower_reviewed(node.target, _linear)
                row.update(action='native_compute_boundary', retained_entry=str(node.target))
            else:
                row['action'] = 'native_candidate_required'
            node.meta['f06_precision_contract'] = dict(row)
            continue
        if not row['contains_low_precision']:
            continue
        op, args = None, None
        if node.op == 'call_function' and node.target is F.dropout:
            bound = dict(zip(('input', 'p', 'training', 'inplace'), node.args), **node.kwargs)
            if (len(node.args) <= 4 and not set(bound)-{'input', 'p', 'training', 'inplace'}
                    and 'input' in bound and bound.get('training', True) is False):
                # Evaluation dropout aliases its input. Leave the original
                # operation for AOT, rather than pretending it returns a copy.
                row['action'] = 'retain_inference_identity'
                node.meta['f06_precision_contract'] = dict(row)
                continue
        if node.op == 'call_function' and node.target is operator.iadd:
            if (len(node.args) == 2 and not node.kwargs and
                    all(isinstance(_value(n), torch.Tensor) for n in node.args)):
                # AOT functionalization must see the mutation and its aliases.
                # Do not replace this with the pure add custom operator.
                row['action'] = 'retain_dtype_mutation'
                node.meta['f06_precision_contract'] = dict(row)
                continue
        if ((node.op == 'call_function' and node.target is torch.sum) or
                (node.op == 'call_method' and node.target == 'sum')):
            bound = dict(zip(('input', 'dim', 'keepdim'), node.args), **node.kwargs)
            dim = bound.get('dim')
            if (len(node.args) <= 3 and not set(bound)-{'input', 'dim', 'keepdim', 'dtype'}
                    and 'input' in bound and type(bound.get('keepdim', False)) is bool
                    and (dim is None or type(dim) is int or isinstance(dim, (tuple, list))
                         and all(type(d) is int for d in dim))):
                target = torch.ops.aten.sum.dim_IntList
                opaque = _native_schema(target)
                make_fallback(opaque, warn=False)
                _lower_direct(opaque, target)
                node.op, node.target = 'call_function', opaque
                node.args = (bound['input'], [dim] if type(dim) is int else dim,
                             bound.get('keepdim', False))
                node.kwargs = {'dtype': bound['dtype']} if 'dtype' in bound else {}
                row.update(action='native_compute_boundary', retained_entry=str(opaque))
                node.meta['f06_precision_contract'] = dict(row)
                continue
        if node.op == 'call_function' and str(node.target) == 'flagtree_p800.model_layernorm.default':
            row.update(action='reviewed_fp32_template', internal_arithmetic='FP32', accumulator='FP32',
                       rounding='template final stores only', retained_entry=str(node.target))
            node.meta['f06_precision_contract'] = dict(row)
            continue
        if node.op == 'call_function' and str(node.target).startswith('flagtree_p800.vendor_'):
            row['action'] = 'native_compute_boundary'
            row['retained_entry'] = str(node.target)
            node.meta['f06_precision_contract'] = dict(row)
            continue
        if node.op == 'call_function' and (node.target in (F.linear, torch.ops.aten.linear.default)
                or str(node.target) in ('_dynamo_workaround.linear', '_dynamo_workaround.linear.default')):
            bound = dict(zip(('input', 'weight', 'bias'), node.args), **node.kwargs)
            if len(node.args) <= 3 and not set(bound) - {'input', 'weight', 'bias'}:
                if 'input' in bound and 'weight' in bound:
                    op, args = _linear, (bound['input'], bound['weight'], bound.get('bias'))
        elif node.op == 'call_function' and node.target in mm and len(node.args) == 2 and not node.kwargs:
            op, args = _matmul, (*node.args, mm[node.target])
        elif ((node.op == 'call_function' and node.target in (torch.relu, F.relu, torch.ops.aten.relu.default))
              or (node.op == 'call_method' and node.target == 'relu')):
            if len(node.args) == 1 and (not node.kwargs or node.kwargs == {'inplace': False}):
                op, args = _relu, node.args
        elif ((node.op == 'call_function' and node.target in (torch.clamp, torch.ops.aten.clamp.default))
              or (node.op == 'call_method' and node.target == 'clamp')):
            bound = dict(zip(('input', 'min', 'max'), node.args), **node.kwargs)
            if (len(node.args) <= 3 and 'input' in bound and not set(bound) - {'input', 'min', 'max'}
                    and all(bound.get(k) is None or type(bound[k]) in (int, float) for k in ('min', 'max'))):
                op, args = _clamp, (bound['input'], bound.get('min'), bound.get('max'))
        elif node.op == 'call_function' and node.target is F.conv2d:
            names = ('input', 'weight', 'bias', 'stride', 'padding', 'dilation', 'groups')
            bound = dict(zip(names, node.args), **node.kwargs)
            def pair(value):
                return [value, value] if type(value) is int else list(value)
            if (len(node.args) <= 7 and not set(bound) - set(names) and 'input' in bound and 'weight' in bound
                    and all(type(bound.get(k, default)) is int or
                            (isinstance(bound.get(k), (tuple, list)) and len(bound[k]) == 2
                             and all(type(v) is int for v in bound[k]))
                            for k, default in (('stride', 1), ('padding', 0), ('dilation', 1)))):
                op, args = _conv2d, (bound['input'], bound['weight'], bound.get('bias'),
                    pair(bound.get('stride', 1)), pair(bound.get('padding', 0)),
                    pair(bound.get('dilation', 1)), bound.get('groups', 1))
        elif node.op == 'call_function' and node.target is F.embedding:
            names = ('input', 'weight', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse')
            bound = dict(zip(names, node.args), **node.kwargs)
            if (len(node.args) <= 7 and not set(bound) - set(names) and 'input' in bound and 'weight' in bound
                    and bound.get('max_norm') is None):
                op, args = _embedding, (bound['input'], bound['weight'], bound.get('padding_idx'),
                    float(bound.get('norm_type', 2.)), bound.get('scale_grad_by_freq', False), bound.get('sparse', False))
        elif node.op == 'call_method' and node.target == 'norm':
            names = ('input', 'p', 'dim', 'keepdim')
            bound = dict(zip(names, node.args), **node.kwargs)
            dim = bound.get('dim')
            if (len(node.args) <= 4 and not set(bound) - set(names) and type(bound.get('p', 2)) in (int, float)
                    and (dim is None or type(dim) is int or
                         isinstance(dim, (list, tuple)) and all(type(v) is int for v in dim))):
                op, args = _norm, (bound['input'], float(bound.get('p', 2)),
                    [dim] if type(dim) is int else list(dim) if dim is not None else None, bound.get('keepdim', False))
        elif node.op == 'call_function' and node.target is F.normalize:
            bound = dict(zip(('input', 'p', 'dim', 'eps', 'out'), node.args), **node.kwargs)
            if (len(node.args) <= 5 and not set(bound)-{'input', 'p', 'dim', 'eps', 'out'}
                    and 'input' in bound and bound.get('out') is None
                    and type(bound.get('p', 2.)) in (int, float)
                    and type(bound.get('dim', 1)) is int
                    and type(bound.get('eps', 1e-12)) in (int, float)):
                op, args = _normalize, (bound['input'], float(bound.get('p', 2.)),
                                       bound.get('dim', 1), float(bound.get('eps', 1e-12)))
        if op is not None:
            target = op._opoverload
            make_fallback(target, warn=False)
            _lower_reviewed(target, op)
            node.op, node.target, node.args, node.kwargs = 'call_function', target, args, {}
            row['action'] = 'native_compute_boundary'
            row['retained_entry'] = str(target)
        elif node.op == 'call_function' and (node.target in pure or hasattr(node.target, '_schema')):
            target = pure.get(node.target, node.target)
            opaque = _native_schema(target)
            if opaque is not None:
                make_fallback(opaque, warn=False)
                _lower_direct(opaque, target)
                node.target = opaque
                row.update(action='native_compute_boundary', retained_entry=str(opaque))
            else:
                row['action'] = 'native_candidate_required'
        elif ((node.op == 'call_method' and node.target in ('exp', 'masked_fill'))
              or (node.op == 'call_function' and node.target in (operator.sub, operator.truediv))):
            if node.target == 'exp' and len(node.args) == 1 and not node.kwargs:
                target = torch.ops.aten.exp.default
            elif node.target == 'masked_fill' and len(node.args) == 3 and not node.kwargs:
                target = (torch.ops.aten.masked_fill.Tensor if isinstance(_value(node.args[2]), torch.Tensor)
                          else torch.ops.aten.masked_fill.Scalar)
            elif node.target in (operator.sub, operator.truediv) and len(node.args) == 2 and not node.kwargs:
                packet = torch.ops.aten.sub if node.target is operator.sub else torch.ops.aten.div
                target = packet.Tensor if isinstance(_value(node.args[1]), torch.Tensor) else packet.Scalar
                if not isinstance(_value(node.args[0]), torch.Tensor):
                    if node.target is operator.sub and isinstance(_value(node.args[1]), torch.Tensor):
                        target = torch.ops.aten.rsub.Scalar
                        node.args = (node.args[1], node.args[0])
                    else:
                        target = None
            else:
                target = None
            if target is None:
                row['action'] = 'native_candidate_required'
            else:
                opaque = _native_schema(target)
                make_fallback(opaque, warn=False)
                _lower_direct(opaque, target)
                node.op, node.target = 'call_function', opaque
                row.update(action='native_compute_boundary', retained_entry=str(opaque))
        elif node.op == 'call_method' and node.target in views:
            row['action'] = 'retain_layout_or_conversion'
        elif node.op == 'call_function' and node.target is operator.getitem:
            row['action'] = 'retain_indexing'
        elif (node.op == 'call_function' and node.target in widened_pointwise
              and all(v.dtype in (torch.float16, torch.float32) for v in _tensors((node.args, node.kwargs)))):
            # This permits scheduling only. It does not authorize removing an
            # intervening cast or changing the result dtype. The complete-model FP32
            # reference gate decides whether the resulting candidate is admissible.
            row['action'] = 'retain_dtype_pointwise_candidate'
        else:
            row['action'] = 'native_candidate_required'
        node.meta['f06_precision_contract'] = dict(row)
    gm.graph.lint()
    gm.recompile()
    return records
