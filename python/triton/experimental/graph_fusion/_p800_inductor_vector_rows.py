"""Inference-only Inductor templates for dense P800 broadcast rows."""
import hashlib
import operator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._inductor import ir
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel
from torch._inductor.virtualized import V

from ._p800_inductor_gelu import _on_p800
from ._p800_vector_rows import configuration


@torch.library.custom_op('flagtree_p800::vector_rows', mutates_args=())
def _rows(matrix: torch.Tensor, bias: torch.Tensor, kind: str, keepdim: bool, revision: str) -> torch.Tensor:
    if kind == 'mul_sum':
        return (matrix * bias).sum(-1, keepdim=keepdim)
    value = matrix + bias
    if kind == 'bias_add':
        return value
    if kind == 'relu':
        return torch.relu(value)
    if kind == 'silu':
        return F.silu(value)
    if kind == 'gelu_tanh':
        return F.gelu(value, approximate='tanh')
    return torch.softmax(value, -1)


@_rows.register_fake
def _fake(matrix, bias, kind, keepdim, revision):
    if kind == 'mul_sum':
        shape = (matrix.shape[0], 1) if keepdim else (matrix.shape[0],)
        return matrix.new_empty(shape)
    return torch.empty_like(matrix)


class _Kernel(TritonTemplateKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(self.write_rows)

    def write_rows(self, mode, dtype, cross):
        with self.create_subgraph_body("<STORE_OUTPUT>"):
            output = self.args.output(self.output_node.get_name())
        suffix = ('float', 'half', 'bf')[dtype]
        n = 'tl.full((), N, tl.int32)'
        tile = 'tl.full((), TILE, tl.int32)'
        if mode == 8:
            target = 'kl3_silu_balanced'
            args = 'tl.program_id(0)'
        elif mode == 7:
            target = 'kl3_silu_pipeline'
            args = f'tl.program_id(0), {n}, tl.full((), M * N, tl.int32), {tile}, tl.full((), PROGS, tl.int32)'
        elif mode < 4:
            target = 'kl3_point_' + suffix
            args = f'tl.program_id(0), tl.full((), M * N, tl.int32), {n}, {tile}, tl.full((), KIND, tl.int32)'
        elif cross:
            target = 'kl3_sum_cross'
            args = 'first, count'
        elif mode == 4:
            target = 'kl3_sum_' + suffix
            args = f'first, count, {n}'
        elif mode == 5 and suffix in ('half', 'bf'):
            target = 'kl3_soft_' + suffix
            args = f'first, count, {n}, {tile}'
        else:
            raise ValueError(f'unsupported P800 vector row schedule {mode} for {suffix}')
        call = f'tle.raw.call({target}, (A, B, {output}, {args}))'
        return self._register_hook("<STORE_OUTPUT>", lambda: call)


class _Template(TritonTemplate):
    kernel_type = _Kernel


def _grid(*args):
    return (args[-1]['PROGS'], 1, 1)


_template = _Template('flagtree_vector_rows', _grid, """
import triton.experimental.tle as tle
from triton.experimental.graph_fusion._p800_vector_rows import (kl3_point_bf, kl3_point_half, kl3_point_float,
    kl3_sum_bf, kl3_sum_half, kl3_sum_float, kl3_sum_cross, kl3_soft_bf, kl3_soft_half,
    kl3_silu_balanced, kl3_silu_pipeline)
{{def_kernel("A", "B")}}
{% if KIND == 4 or KIND == 5 %}
    pid = tl.program_id(0)
{% if CROSS %}
    first = pid * (M // 12) + tl.minimum(pid, M % 12)
    count = M // 12 + (pid < M % 12).to(tl.int32)
{% else %}
    first = pid * ROWS
    count = tl.minimum(ROWS, M - first)
{% endif %}
{% endif %}
    {{write_rows(KIND, DTYPE, CROSS)}}
""")
_here = Path(__file__)
_revision = hashlib.sha256(_here.read_bytes() + _here.with_name('_p800_vector_rows.py').read_bytes()
                           + _here.with_name('_p800_vector_rows.xpu').read_bytes()).hexdigest()


@register_lowering(torch.ops.flagtree_p800.vector_rows.default, type_promotion_kind=None)
def _lower(matrix, bias, kind, keepdim, revision):
    m, n = map(int, matrix.get_size())
    mode, width, rows, programs = configuration(kind, str(matrix.get_dtype()).removeprefix('torch.'), m, n)
    size = ([m, 1] if keepdim else [m]) if kind == 'mul_sum' else [m, n]
    layout = ir.FixedLayout(matrix.get_device(), matrix.get_dtype(), size, ir.FlexibleLayout.contiguous_strides(size))
    inputs = [ir.ExternKernel.require_exact_strides(matrix, [n, 1]), ir.ExternKernel.require_exact_strides(bias, [1])]
    choice = _template.generate(inputs, layout, num_stages=1, num_warps=4, M=m, N=n, KIND=mode,
                                TILE=width, ROWS=rows, PROGS=programs,
                                DTYPE=(torch.float32, torch.float16, torch.bfloat16).index(matrix.get_dtype()),
                                CROSS=(mode == 4 and n == 1024 and matrix.get_dtype() == torch.float32))
    output = choice.output_node()
    V.graph.no_fuse_buffer_names.add(output.get_name())
    return output


@torch.library.custom_op('flagtree_p800::matmul_rows', mutates_args=())
def _matmul_rows(left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor, kind: str,
                 revision: str) -> torch.Tensor:
    product = left @ right
    if kind == 'gelu_exact':
        return F.gelu(product + bias)
    return _rows(product, bias, kind, False, revision)


@_matmul_rows.register_fake
def _fake_matmul(left, right, bias, kind, revision):
    return left.new_empty((left.shape[0], right.shape[1]))


@register_lowering(torch.ops.flagtree_p800.matmul_rows.default, type_promotion_kind=None)
def _lower_matmul(left, right, bias, kind, revision):
    # Preserve the vendor MatMul output and its rounding before bias addition.
    product = fallback_handler(torch.ops.aten.mm.default, add_to_fallback_set=False)(left, right)
    if kind == 'gelu_exact':
        from ._p800_inductor_gelu import _lower as lower_gelu
        return lower_gelu(product, bias, 'none', revision)
    return _lower(product, bias, kind, False, revision)


def _rewrite_matmul(graph):
    from ._p800_native_mm import match
    gm = graph.owning_module
    args = [node for node in graph.nodes if node.op == 'placeholder']
    if gm is None or len(args) != 3:
        return
    values = [node.meta.get('example_value') for node in args]
    if not all(isinstance(x, torch.Tensor) and x.device.type == 'cuda' and x.device == values[0].device
               and x.layout == torch.strided and all(type(d) is int for d in x.shape) for x in values):
        return
    if not _on_p800(values[0].device):
        return
    spec = match(gm, values, vendor_product=True)
    if spec is None or spec['K'] != 64:
        return
    ops = [n for n in graph.nodes if n.op not in ('placeholder', 'output')]
    if (len(ops) != (3 if spec['act'] else 2) or any(len(n.users) != 1 for n in ops)
            or any(a.next is not b for a, b in zip(ops, ops[1:]))):
        return
    if any(x.requires_grad for x in values) and (torch.is_grad_enabled() or any(
            not isinstance(n.meta.get('example_value'), torch.Tensor) or n.meta['example_value'].requires_grad
            for n in ops)):
        return
    kind = ('bias_add', 'silu', 'relu', 'gelu_tanh', 'gelu_exact')[spec['act']]
    if kind != 'gelu_exact' and configuration(kind, spec['dtype'], spec['M'], spec['N']) is None:
        return
    terminal = ops[-1]
    terminal.op, terminal.target = 'call_function', torch.ops.flagtree_p800.matmul_rows.default
    terminal.args, terminal.kwargs = (*args, kind, _revision), {}
    for old in reversed(ops[:-1]):
        graph.erase_node(old)


def _operation(node):
    target = node.target
    args, kw = node.args, node.kwargs
    if not args or node.op not in ('call_method', 'call_function'):
        return None
    if target in ('sum', torch.sum, torch.ops.aten.sum.dim_IntList, 'softmax', torch.softmax,
                  F.softmax, torch.ops.aten.softmax.int):
        is_sum = target in ('sum', torch.sum, torch.ops.aten.sum.dim_IntList)
        keys = {'dim', 'dtype', 'keepdim'} if is_sum else {'dim', 'dtype', '_stacklevel'}
        if len(args) > (3 if is_sum else 2) or set(kw) - keys or kw.get('dtype') is not None:
            return None
        dim = kw.get('dim', args[1] if len(args) > 1 else None)
        if isinstance(dim, (tuple, list)) and len(dim) == 1:
            dim = dim[0]
        if type(dim) is not int or dim not in (-1, 1):
            return None
        keep = kw.get('keepdim', args[2] if len(args) > 2 else False)
        if type(keep) is bool:
            return ('mul_sum' if is_sum else 'add_softmax', keep)
    elif target in (F.silu, torch.ops.aten.silu.default) and len(args) == 1 and not (set(kw) - {'inplace'}) and not kw.get('inplace', False):
        return ('silu', False)
    elif target in (F.gelu, torch.ops.aten.gelu.default) and len(args) == 1 and dict(kw) == {'approximate': 'tanh'}:
        return ('gelu_tanh', False)
    return None


def rewrite(graph):
    _rewrite_matmul(graph)
    for node in list(graph.nodes):
        op = _operation(node)
        if op is None:
            continue
        kind, keep = op
        binary = node.args[0]
        if not isinstance(binary, torch.fx.Node) or binary.next is not node or len(binary.users) != 1:
            continue
        if kind == 'mul_sum':
            allowed = (operator.mul, torch.mul, torch.ops.aten.mul.Tensor, 'mul')
            if binary.kwargs:
                continue
        else:
            allowed = (operator.add, torch.add, torch.ops.aten.add.Tensor, 'add')
            if set(binary.kwargs) - {'alpha'} or binary.kwargs.get('alpha', 1) != 1:
                continue
        if binary.op not in ('call_method', 'call_function') or binary.target not in allowed or len(binary.args) != 2:
            continue
        operands = list(binary.args)
        if not all(isinstance(x, torch.fx.Node) for x in operands):
            continue
        vals = [x.meta.get('example_value') for x in operands]
        if not all(isinstance(x, torch.Tensor) and x.device.type == 'cuda' and x.layout == torch.strided
                   and x.is_contiguous() and all(type(d) is int for d in x.shape) for x in vals):
            continue
        if vals[0].ndim == 1:
            vals.reverse()
            operands.reverse()
        a, b = vals
        if a.ndim != 2 or b.shape != (a.shape[1],) or a.device != b.device or a.dtype != b.dtype:
            continue
        if any(x.requires_grad for x in vals):
            outputs = [x.meta.get('example_value') for x in (binary, node)]
            if torch.is_grad_enabled() or any(not isinstance(x, torch.Tensor) or x.requires_grad for x in outputs):
                continue
        if _on_p800(a.device) and configuration(kind, str(a.dtype).removeprefix('torch.'), *a.shape):
            node.op, node.target = 'call_function', torch.ops.flagtree_p800.vector_rows.default
            node.args, node.kwargs = (*operands, kind, keep, _revision), {}
            graph.erase_node(binary)
