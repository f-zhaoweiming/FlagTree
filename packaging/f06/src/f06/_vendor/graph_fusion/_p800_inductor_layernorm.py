"""Inductor lowering for inference-only P800 broadcast Add + LayerNorm."""
import hashlib
import operator
from pathlib import Path

import torch
from torch._inductor import ir
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel
from torch._inductor.virtualized import V

from ._p800_inductor_gelu import _on_p800
from ._p800_layernorm import rows_per_program, segment_width, toolchain_available


@torch.library.custom_op("flagtree_p800::add_layernorm", mutates_args=())
def _add_layernorm(matrix: torch.Tensor, bias: torch.Tensor, weight: torch.Tensor | None,
                   shift: torch.Tensor | None, eps: float, revision: str) -> torch.Tensor:
    return torch.nn.functional.layer_norm(matrix + bias, (matrix.shape[1],), weight, shift, eps)


@_add_layernorm.register_fake
def _fake(matrix, bias, weight, shift, eps, revision):
    return torch.empty_like(matrix)


class _LayerNormKernel(TritonTemplateKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(self.store_layernorm)

    def store_layernorm(self):
        with self.create_subgraph_body("<STORE_OUTPUT>"):
            out = self.args.output(self.output_node.get_name())
        call = (f"tle.raw.call(kl3_add_layernorm, (A, B, weight, shift, {out}, tl.program_id(0) * ROWS, "
                "count, tl.full((), N, tl.int32), tl.full((), SEGMENT, tl.int32), "
                "tl.full((), EPS, tl.float32), tl.full((), AFFINE, tl.int32), "
                "tl.full((), 2 if A.dtype.element_ty == tl.bfloat16 else A.dtype.element_ty == tl.float16, tl.int32)))")
        return self._register_hook("<STORE_OUTPUT>", lambda: call)


class _LayerNormTemplate(TritonTemplate):
    kernel_type = _LayerNormKernel


def _grid(m, n, meta):
    return ((m + meta['ROWS'] - 1) // meta['ROWS'], 1, 1)


_SOURCE = """
import triton.experimental.tle as tle
from triton.experimental.graph_fusion._p800_layernorm import kl3_add_layernorm
{% if AFFINE %}
{{def_kernel("A", "B", "W", "SHIFT")}}
    weight = W
    shift = SHIFT
{% else %}
{{def_kernel("A", "B")}}
    weight = B
    shift = B
{% endif %}
{% if M % ROWS %}
    count = tl.minimum(ROWS, M - tl.program_id(0) * ROWS)
{% else %}
    count = tl.full((), ROWS, tl.int32)
{% endif %}
    {{store_layernorm()}}
"""
_template = _LayerNormTemplate("flagtree_add_layernorm", _grid, _SOURCE)
_here = Path(__file__)
_revision = hashlib.sha256(_here.read_bytes() + _here.with_name('_p800_layernorm.py').read_bytes()
                           + _here.with_name('_p800_layernorm.xpu').read_bytes()).hexdigest()


@register_lowering(torch.ops.flagtree_p800.add_layernorm.default, type_promotion_kind=None)
def _lower(matrix, bias, weight, shift, eps, revision):
    m, n = map(int, matrix.get_size())
    layout = ir.FixedLayout(matrix.get_device(), matrix.get_dtype(), [m, n], [n, 1])
    matrix = ir.ExternKernel.require_exact_strides(matrix, [n, 1])
    vectors = [bias] if weight is None else [bias, weight, shift]
    inputs = [matrix, *(ir.ExternKernel.require_exact_strides(v, [1]) for v in vectors)]
    dtype = str(matrix.get_dtype()).removeprefix('torch.')
    segment = segment_width(m, n, dtype)
    affine = weight is not None
    choice = _template.generate(inputs, layout, num_stages=1, num_warps=4, M=m, N=n, SEGMENT=segment,
                                EPS=eps, AFFINE=affine, ROWS=rows_per_program(m, allow_tail=True, dtype=dtype))
    output = choice.output_node()
    # The inlined device function writes the output buffer directly.
    V.graph.no_fuse_buffer_names.add(output.get_name())
    return output


def rewrite(graph):
    import torch.nn.functional as F
    for norm in list(graph.nodes):
        if norm.op != 'call_function' or norm.target not in (F.layer_norm, torch.ops.aten.layer_norm.default):
            continue
        # Bind optional parameters in the same order as aten.layer_norm.
        names = ('input', 'normalized_shape', 'weight', 'bias', 'eps', 'cudnn_enable')
        if len(norm.args) > len(names) or set(norm.kwargs) - set(names):
            continue
        args = dict(zip(names, norm.args), **norm.kwargs)
        add = args.get('input')
        if not isinstance(add, torch.fx.Node) or add.next is not norm or len(add.users) != 1:
            continue
        if not ((add.op == 'call_function' and add.target in (operator.add, torch.add, torch.ops.aten.add.Tensor))
                or (add.op == 'call_method' and add.target == 'add')):
            continue
        if len(add.args) != 2 or set(add.kwargs) - {'alpha'} or add.kwargs.get('alpha', 1) != 1:
            continue
        a, b = add.args
        w, shift = args.get('weight'), args.get('bias')
        if (w is None) != (shift is None):
            continue
        nodes = (a, b) if w is None else (a, b, w, shift)
        if not all(isinstance(x, torch.fx.Node) for x in nodes):
            continue
        vals = [x.meta.get('example_value') for x in nodes]
        if not all(isinstance(x, torch.Tensor) and x.device.type == 'cuda' and x.layout == torch.strided
                   and x.dtype in (torch.float16, torch.float32, torch.bfloat16) and x.is_contiguous()
                   and all(type(d) is int for d in x.shape) for x in vals):
            continue
        if vals[0].ndim == 1:
            a, b = b, a
            vals[:2] = vals[1::-1]
        matrix = vals[0]
        if matrix.ndim != 2 or any(x.device != matrix.device or x.dtype != matrix.dtype or x.shape != (matrix.shape[1],)
                                   for x in vals[1:]):
            continue
        shape, eps = args.get('normalized_shape'), args.get('eps', 1e-5)
        if not isinstance(shape, (list, tuple)) or not all(type(d) is int for d in shape) or tuple(shape) != (matrix.shape[1],):
            continue
        if type(eps) not in (int, float) or segment_width(*matrix.shape, str(matrix.dtype).removeprefix('torch.')) is None:
            continue
        if any(x.requires_grad for x in vals):
            outputs = [node.meta.get('example_value') for node in (add, norm)]
            if torch.is_grad_enabled() or any(not isinstance(x, torch.Tensor) or x.requires_grad for x in outputs):
                continue
        if _on_p800(matrix.device) and toolchain_available():
            norm.target = torch.ops.flagtree_p800.add_layernorm.default
            norm.args = (a, b, w, shift, float(eps), _revision)
            norm.kwargs = {}
            graph.erase_node(add)
