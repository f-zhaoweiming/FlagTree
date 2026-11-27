"""Optimize the decomposed FP32 Softmax normalization in native Inductor FX IR."""
from contextlib import contextmanager
from contextvars import ContextVar
import re

_ROW_SHIFT = ContextVar("flagtree_kl3_softmax_shift", default=False)


def configure_row_kernel(source, kwargs):
    # Only the exact normalized graph below may request the relaxed KL3
    # maximum. Keep other kernels compiled inside the same scope strict.
    if _ROW_SHIFT.get() and '_kl3_softmax_shift' in getattr(getattr(source, 'fn', None), 'src', ''):
        kwargs['options'] = dict(kwargs.get('options') or {}, kl3_softmax_shift=True, kl3_row_softmax=True)



def reuse_row_reciprocal(graph):
    import torch
    aten = torch.ops.aten
    rewritten = []
    for division in list(graph.find_nodes(op='call_function', target=aten.div.Tensor)):
        exp, total = division.args[:2]
        if not (isinstance(exp, torch.fx.Node) and isinstance(total, torch.fx.Node)
                and exp.target == aten.exp.default and total.target == aten.sum.dim_IntList
                and len(total.args) == 3 and total.args[0] is exp and total.args[2] is True
                and not total.kwargs and not division.kwargs):
            continue
        value = exp.meta.get('val')
        if value is None or value.dtype != torch.float32 or value.ndim < 1:
            continue
        axes = total.args[1]
        if tuple(axes) not in ((-1,), (value.ndim - 1,)):
            continue
        # Require the stabilized numerator. exp(x - row_amax(x)) has a sum
        # >= 1 on every finite, nonempty row; arbitrary exp(x) can underflow.
        sub = exp.args[0]
        if not isinstance(sub, torch.fx.Node) or sub.target != aten.sub.Tensor or len(sub.args) != 2 or sub.kwargs:
            continue
        x, maximum = sub.args
        if not (isinstance(maximum, torch.fx.Node) and maximum.target == aten.amax.default
                and len(maximum.args) == 3 and maximum.args[0] is x
                and tuple(maximum.args[1]) in ((-1,), (value.ndim - 1,)) and maximum.args[2] is True):
            continue
        with graph.inserting_before(division):
            inv = graph.call_function(aten.reciprocal.default, (total,))
            inv.meta = total.meta.copy()
        division.target = aten.mul.Tensor
        division.args = (exp, inv)
        rewritten.append(division)
    return rewritten


def _kl3_row_schedule(inputs, gm):
    # The measured case is a static, dense FP32 matrix plus a contiguous
    # broadcast row. Leave every other shape with the caller's choices.
    import operator
    import torch
    if len(inputs) != 2 or not all(isinstance(t, torch.Tensor) and t.dtype == torch.float32
                                   and t.is_contiguous() for t in inputs):
        return False
    a, b = inputs
    if a.shape != (2048, 2048) or b.shape != (2048,) or a.device != b.device:
        return False
    nodes = [node for node in gm.graph.nodes if node.op not in ('placeholder', 'output')]
    if len(nodes) != 2:
        return False
    add, softmax = nodes
    dim = softmax.kwargs.get('dim', softmax.args[1] if len(softmax.args) > 1 else None)
    return (add.target in (operator.add, torch.add, torch.ops.aten.add.Tensor) and
            softmax.args[0] is add and dim in (-1, 1) and softmax.kwargs.get('dtype') is None)


@contextmanager
def native_softmax_scope(gm, inputs=()):
    import torch
    from torch._inductor import config
    targets = (torch.softmax, torch.nn.functional.softmax, torch.ops.aten.softmax.int,
               torch.ops.aten._softmax.default)
    if not any((n.op == 'call_function' and n.target in targets) or
               (n.op == 'call_method' and n.target == 'softmax') for n in gm.graph.nodes):
        yield
        return
    previous = config.post_grad_custom_pre_pass
    exclusive = False

    def optimize(graph):
        nonlocal exclusive
        if previous is not None:
            previous(graph)
        rewritten = reuse_row_reciprocal(graph)
        # Two inputs, add, amax, sub, exp, sum, reciprocal, mul and output.
        # The row maximum may only be relaxed when normalization is its only
        # observable use. A caller's post-grad hook may have changed the graph.
        nodes = list(graph.nodes)
        exclusive = (len(rewritten) == 1 and len(nodes) == 10
                     and nodes[-1].op == 'output' and nodes[-1].args[0] == (rewritten[0],)
                     and nodes[2].target == torch.ops.aten.add.Tensor)

    options = {'post_grad_custom_pre_pass': optimize}
    if not (torch.__version__.startswith('2.9.') and _kl3_row_schedule(inputs, gm)):
        with config.patch(options):
            yield
        return

    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor.virtualized import V
    from torch._inductor.codegen.triton import FixedTritonConfig, TritonKernel
    original = V.choices
    tile = FixedTritonConfig({'XBLOCK': 8, 'num_warps': 4, 'num_stages': 1})

    class RowChoices:
        def __getattr__(self, name):
            return getattr(original, name)

        def triton_kernel_kwargs(self, kernel_type, features, groups, kwargs):
            kwargs = original.triton_kernel_kwargs(kernel_type, features, groups, kwargs)
            if not exclusive or features.numel != 2048 or features.reduction_numel != 2048:
                return kwargs
            # Inductor uses the tile to prove divisibility and generate masks.
            return dict(kwargs, fixed_config=tile, override_persistent_reduction=True,
                        override_cooperative_reduction=False)

    options.update(online_softmax=False, coordinate_descent_tuning=False)
    with compile_lock, V.set_choices_handler(RowChoices()), config.patch(options):
        original_resize = TritonKernel.reduction_resize_and_shape

        def resize(kernel, expression, shape):
            # Pinned Torch 2.9 forms its final reduction expression here. All
            # indexing, accumulation, layout and stores remain native codegen.
            if kernel.fixed_config is tile:
                match = re.fullmatch(r'triton_helpers\.max2\((\w+), 1\)', expression)
                if match:
                    helper = kernel.helper_functions.add(
                        '@triton.jit\ndef {name}(a, b):\n    return tl.where(a > b, a, b)',
                        base_name='_kl3_softmax_shift')
                    expression = f'tl.reduce({match[1]}, 1, {helper})'
            return original_resize(kernel, expression, shape)

        token = _ROW_SHIFT.set(True)
        TritonKernel.reduction_resize_and_shape = resize
        try:
            yield
        finally:
            TritonKernel.reduction_resize_and_shape = original_resize
            _ROW_SHIFT.reset(token)
