"""P800 residual LayerNorm as an Inductor multi-output template."""
import hashlib
import math
import operator
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.experimental.tle as tle
from torch._inductor import ir
from torch._inductor.lowering import register_lowering, empty_strided, make_fallback
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel
from torch._inductor.virtualized import V

from ._p800_inductor_gelu import _on_p800
from ._p800_layernorm import toolchain_available
from ._model_inductor import _value, _is_call


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_model_layernorm(a, b, w, s, out, residual, first, rows, n, brows, eps, affine, dual, dtype):
    """FP32 SDK-order moments with optional residual writeback."""


@torch.library.custom_op('flagtree_p800::model_layernorm', mutates_args=())
def _op(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor | None, s: torch.Tensor | None,
        eps: float, dual: bool, revision: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = a.float() + b.float()
    norm = F.layer_norm(z, (a.shape[-1],), None if w is None else w.float(),
                        None if s is None else s.float(), eps).to(a.dtype)
    return norm, z.to(a.dtype) if dual else torch.empty_like(a)


@_op.register_fake
def _fake(a, b, w, s, eps, dual, revision):
    return torch.empty_like(a), torch.empty_like(a)


class _Kernel(TritonTemplateKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(self.store_model_norm)

    def store_model_norm(self):
        with self.create_subgraph_body('<STORE_OUTPUT>'):
            out = self.args.output(self.output_node.get_name())
        call = (f'tle.raw.call(kl3_model_layernorm, (A, B, weight, shift, {out}, R, '
                'tl.program_id(0) * ROWS, count, tl.full((), N, tl.int32), '
                'tl.full((), BROWS, tl.int32), tl.full((), EPS, tl.float32), '
                'tl.full((), AFFINE, tl.int32), tl.full((), DUAL, tl.int32), '
                'tl.full((), 2 if A.dtype.element_ty == tl.bfloat16 else A.dtype.element_ty == tl.float16, tl.int32)))')
        return self._register_hook('<STORE_OUTPUT>', lambda: call)


class _Template(TritonTemplate):
    kernel_type = _Kernel


def _grid(*args):
    *sizes, meta = args
    m = math.prod(sizes[:-1])
    return ((m + meta['ROWS'] - 1) // meta['ROWS'], 1, 1)


_SOURCE = '''
import triton.experimental.tle as tle
from triton.experimental.graph_fusion._p800_model_layernorm import kl3_model_layernorm
{% if AFFINE %}
{{def_kernel("A", "B", "R", "W", "S")}}
    weight = W
    shift = S
{% else %}
{{def_kernel("A", "B", "R")}}
    weight = B
    shift = B
{% endif %}
    count = tl.minimum(ROWS, M - tl.program_id(0) * ROWS)
    {{store_model_norm()}}
'''
_template = _Template('flagtree_model_layernorm', _grid, _SOURCE)
_path = Path(__file__)
_revision = hashlib.sha256(_path.read_bytes() + _path.with_name('_p800_layernorm.xpu').read_bytes()).hexdigest()


class _ResidualBuffer(ir.TritonTemplateBuffer):
    """Extend Torch 2.9's template ABI only for a freshly allocated output.

    Upstream's constructor gates mutation outputs to FlexAttention. We use the
    same scheduler representation for our private buffer; user inputs can never
    be listed as mutation outputs by this template.
    """
    def __init__(self, choice, residual):
        assert isinstance(residual, ir.TensorBox)
        assert isinstance(residual.data.data, ir.ComputedBuffer)
        super().__init__(choice.layout, choice.input_nodes, choice.make_kernel_render,
                         mutated_inputs=None, allowed_prologue_inps=choice.allowed_prologue_inps)
        self.mutated_inputs=[residual]
        self.outputs.append(ir.MutationOutput(ir.NoneLayout(device=residual.get_device()), residual, self))


def _strides(size):
    return [math.prod(size[i+1:]) for i in range(len(size))]


@register_lowering(torch.ops.flagtree_p800.model_layernorm.default, type_promotion_kind=None)
def _lower(a, b, w, s, eps, dual, revision):
    shape = list(map(int, a.get_size()));strides = _strides(shape)
    n = shape[-1];m = math.prod(shape[:-1]);brows = math.prod(map(int, b.get_size())) // n
    a = ir.ExternKernel.require_exact_strides(a, strides)
    b = ir.ExternKernel.require_exact_strides(b, _strides(b.get_size()))
    residual = empty_strided(shape, strides, dtype=a.get_dtype(), device=a.get_device())
    residual.realize()
    inputs = [a, b, residual]
    if w is not None:
        inputs += [ir.ExternKernel.require_exact_strides(v, [1]) for v in (w,s)]
    layout = ir.FixedLayout(a.get_device(), a.get_dtype(), shape, strides)
    choice = _template.generate(inputs, layout, num_stages=1, num_warps=4,
        mutated_inputs=[residual], M=m, N=n, BROWS=brows, EPS=eps, AFFINE=w is not None,
        DUAL=dual, ROWS=64)
    norm = ir.TensorBox.create(_ResidualBuffer(choice, residual))
    V.graph.no_fuse_buffer_names.add(norm.get_name())
    V.graph.no_fuse_buffer_names.add(residual.get_name())
    return norm, residual


def rewrite(gm, *, fp32_only=False):
    graph=gm.graph;records=[]
    for norm in list(graph.nodes):
        if not _is_call(norm,(F.layer_norm,torch.ops.aten.layer_norm.default)):
            continue
        names=('input','normalized_shape','weight','bias','eps','cudnn_enable')
        args=dict(zip(names,norm.args),**norm.kwargs);add=args.get('input')
        if not _is_call(add,(operator.add,torch.add,torch.ops.aten.add.Tensor)) or add.next is not norm:
            continue
        if len(add.args)!=2 or set(add.kwargs)-{'alpha'} or add.kwargs.get('alpha',1)!=1:
            continue
        a,b=add.args;av,bv=_value(a),_value(b);w,s=args.get('weight'),args.get('bias')
        if fp32_only and any(not isinstance(v,torch.Tensor) or v.dtype != torch.float32 for v in (av,bv)):
            continue
        if not all(isinstance(v,torch.Tensor) and v.device.type=='cuda' and v.layout==torch.strided
                   and v.dtype in (torch.float32,torch.float16,torch.bfloat16)
                   and all(type(d) is int for d in v.shape) for v in (av,bv)):
            continue
        if av.numel()<bv.numel():a,b,av,bv=b,a,bv,av
        if av.ndim<2 or av.dtype!=bv.dtype or av.device!=bv.device or bv.ndim>av.ndim:
            continue
        n=av.shape[-1];bs=(1,)*(av.ndim-bv.ndim)+tuple(bv.shape)
        if n<=0 or n>768 or n%64 or bs[-1]!=n or not av.numel() or not bv.numel() or math.prod(av.shape)>2**31-1:
            continue
        # Modulo-row addressing admits broadcast leading dimensions only.
        active=False;valid=True
        for x,y in zip(av.shape,bs):
            active |= y!=1
            if (active and x!=y) or (not active and y!=1):valid=False
        if not valid or tuple(args['normalized_shape'])!=(n,) or (w is None)!=(s is None):
            continue
        if w is not None and not all(isinstance(_value(v),torch.Tensor) and _value(v).shape==(n,)
             and _value(v).dtype==av.dtype and _value(v).device==av.device and _value(v).is_contiguous() for v in (w,s)):
            continue
        if not _on_p800(av.device) or not toolchain_available():continue
        users=[u for u in add.users if u is not norm];dual=bool(users)
        with graph.inserting_before(norm):
            call=graph.call_function(torch.ops.flagtree_p800.model_layernorm.default,
                                    (a,b,w,s,float(args.get('eps',1e-5)),dual,_revision))
            output=graph.call_function(operator.getitem,(call,0));output.meta=dict(norm.meta)
            if dual:
                residual=graph.call_function(operator.getitem,(call,1));residual.meta=dict(add.meta)
                for user in users:user.replace_input_with(add,residual)
        norm.replace_all_uses_with(output)
        records.append(dict(kind='residual_layernorm',node=norm.name,outputs=2 if dual else 1,
                            broadcast=av.shape!=bv.shape,state='inductor_template',template='sdk_order_residual'))
        graph.erase_node(norm);graph.erase_node(add)
    graph.lint();gm.recompile();return records


@torch.library.custom_op('flagtree_p800::vendor_layernorm', mutates_args=())
def _vendor_norm(x: torch.Tensor, shape: list[int], w: torch.Tensor | None,
                 b: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.layer_norm(x, shape, w, b, eps)


@_vendor_norm.register_fake
def _vendor_norm_fake(x, shape, w, b, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


make_fallback(torch.ops.flagtree_p800.vendor_layernorm.default, warn=False)


def preserve_unmatched_norms(gm):
    # A standalone vendor LayerNorm must retain the SDK's reduction order.
    # In particular the embedding norm may follow an in-place add, which the
    # pure residual matcher deliberately does not move or delete.
    count=0
    for node in gm.graph.nodes:
        if not _is_call(node,(F.layer_norm,torch.ops.aten.layer_norm.default)):
            continue
        names=('input','normalized_shape','weight','bias','eps','cudnn_enable')
        args=dict(zip(names,node.args),**node.kwargs)
        node.target=torch.ops.flagtree_p800.vendor_layernorm.default
        node.args=(args['input'],list(args['normalized_shape']),args.get('weight'),args.get('bias'),float(args.get('eps',1e-5)))
        node.kwargs={};count+=1
    gm.recompile();return count


@torch.library.custom_op('flagtree_p800::vendor_gelu', mutates_args=())
def _vendor_gelu(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return F.gelu(x, approximate=approximate)


@_vendor_gelu.register_fake
def _vendor_gelu_fake(x, approximate):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


make_fallback(torch.ops.flagtree_p800.vendor_gelu.default, warn=False)


def preserve_vendor_gelu(gm):
    count=0
    for node in gm.graph.nodes:
        if _is_call(node,(F.gelu,torch.ops.aten.gelu.default)):
            approximate=node.kwargs.get('approximate',node.args[1] if len(node.args)>1 else 'none')
            node.target=torch.ops.flagtree_p800.vendor_gelu.default
            node.args=(node.args[0],approximate);node.kwargs={};count+=1
    gm.recompile();return count


@torch.library.custom_op('flagtree_p800::vendor_tanh', mutates_args=())
def _vendor_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


@_vendor_tanh.register_fake
def _vendor_tanh_fake(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


make_fallback(torch.ops.flagtree_p800.vendor_tanh.default, warn=False)


def preserve_vendor_tanh(gm):
    count=0
    for node in gm.graph.nodes:
        if (_is_call(node,(torch.tanh,torch.ops.aten.tanh.default)) or
                (node.op=='call_method' and node.target=='tanh')):
            node.op='call_function';node.target=torch.ops.flagtree_p800.vendor_tanh.default
            node.args=(node.args[0],);node.kwargs={};count+=1
    gm.recompile();return count


@torch.library.custom_op('flagtree_p800::final_cast', mutates_args=())
def _final_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # Keep the model's final narrowing separate from live FP32 intermediates.
    # This XPU backend cannot lower a fused fp16/fp32 dual-output pointwise op.
    return x.to(dtype=dtype, copy=True)


@_final_cast.register_fake
def _final_cast_fake(x, dtype):
    return torch.empty_like(x, dtype=dtype)


make_fallback(torch.ops.flagtree_p800.final_cast.default, warn=False)
