"""P800 inference precision policy: promote arithmetic edges, round observable exits."""
import copy
import operator

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

_LOW = (torch.float16, torch.bfloat16)
_MATMUL = (operator.matmul, torch.matmul, torch.mm, torch.ops.aten.matmul.default, torch.ops.aten.mm.default,
           'matmul', 'mm')
_ARITHMETIC = {
    operator.add, operator.mul, operator.matmul, operator.truediv,
    torch.add, torch.mul, torch.div, torch.matmul, torch.mm, torch.sum, torch.amax, torch.exp,
    torch.relu, torch.softmax, torch.clone, torch.ops.aten.clone.default, F.relu, F.silu, F.gelu, F.softmax, F.layer_norm,
    torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar, torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar,
    torch.ops.aten.matmul.default, torch.ops.aten.mm.default, torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList, torch.ops.aten.amax.default, torch.ops.aten.exp.default,
    torch.ops.aten.relu.default, torch.ops.aten.softmax.int, torch.ops.aten.gelu.default,
    torch.ops.aten.silu.default, torch.ops.aten.layer_norm.default,
}
_METHODS = {'add', 'mul', 'matmul', 'mm', 'div', 'truediv', 'sum', 'amax', 'exp', 'relu', 'softmax', 'clone'}
_CAST = 'f06_precision_boundary'


@torch.library.custom_op('flagtree_p800::round_output', mutates_args=())
def round_output(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # The SDNN FP32 -> BF16 DMA truncates. The SDK conversion implements RNE.
    return value.to(dtype=dtype, copy=True)


@round_output.register_fake
def _fake_output(value, dtype):
    return torch.empty_like(value, dtype=dtype)


@torch.library.custom_op('flagtree_p800::fp32_input', mutates_args=())
def fp32_input(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float32, copy=True)


@fp32_input.register_fake
def _fake_input(value):
    return torch.empty_like(value, dtype=torch.float32)


@torch.library.custom_op('flagtree_p800::fp32_product', mutates_args=())
def fp32_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # Preserve the SDK FP32 product if compiling the widened graph. Absorbing
    # its input casts into a tuned SDNN GEMM can change FP32 rounding or fault
    # while benchmarking a tile. The separately verified mixed kernel can
    # consume the original low-precision matrices directly.
    with torch.autocast(left.device.type, enabled=False):
        return torch.matmul(left, right)


@fp32_product.register_fake
def _fake_product(left, right):
    with torch.autocast(left.device.type, enabled=False):
        return torch.matmul(left, right)


@torch.library.custom_op('flagtree_p800::fp32_epilogue', mutates_args=())
def fp32_epilogue(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'relu':
        return torch.relu(value)
    if mode == 'silu':
        return F.silu(value)
    return F.gelu(value, approximate='tanh')


@fp32_epilogue.register_fake
def _fake_epilogue(value, mode):
    return torch.empty_like(value)


@torch.library.custom_op('flagtree_p800::fp32_row_sum', mutates_args=())
def fp32_row_sum(value: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    # At BF16 ties even a single FP32 ULP in the reduction can change the
    # returned BF16 bin. Preserve the SDK's FP32 reduction order.
    return torch.sum(value, dim=dim, keepdim=keepdim)


@fp32_row_sum.register_fake
def _fake_row_sum(value, dim, keepdim):
    return torch.sum(value, dim=dim, keepdim=keepdim)


def _bf16_product_sum(node):
    if node.target not in ('sum', torch.sum, torch.ops.aten.sum.dim_IntList) or not node.args:
        return None
    value = node.args[0]
    if (not isinstance(value, Node) or _dtype(value) != torch.bfloat16
            or value.target not in ('mul', operator.mul, torch.mul, torch.ops.aten.mul.Tensor)):
        return None
    dim = node.kwargs.get('dim', node.args[1] if len(node.args) > 1 else None)
    if isinstance(dim, (tuple, list)) and len(dim) == 1:
        dim = dim[0]
    keepdim = node.kwargs.get('keepdim', node.args[2] if len(node.args) > 2 else False)
    if type(dim) is int and type(keepdim) is bool:
        return dim, keepdim
    return None


def _mm_epilogue(node):
    # A mixed-input SDNN MatMul with an inlined nonlinear epilogue can
    # fault during coordinate descent, before numerical admission can run.
    if not node.args or not isinstance(node.args[0], Node):
        return None
    bias = node.args[0]
    if bias.target not in (operator.add, torch.add, torch.ops.aten.add.Tensor, 'add'):
        return None
    if not any(n.target in _MATMUL for n in bias.all_input_nodes):
        return None
    if node.target in (F.relu, torch.relu, 'relu', torch.ops.aten.relu.default):
        return 'relu'
    if node.target in (F.silu, torch.ops.aten.silu.default):
        return 'silu'
    if node.target in (F.gelu, torch.ops.aten.gelu.default) and node.kwargs.get('approximate') == 'tanh':
        return 'gelu_tanh'
    return None


def _value(node):
    return node.meta.get('example_value', node.meta.get('val'))


def _dtype(node):
    return getattr(_value(node), 'dtype', None)


def _cast_dtype(node):
    if node.op != 'call_method' or node.kwargs or not node.args:
        if (node.op == 'call_method' and node.target == 'to' and len(node.args) == 1
                and set(node.kwargs) == {'dtype'}):
            return node.kwargs['dtype']
        return None
    if len(node.args) == 1:
        return {'float': torch.float32, 'half': torch.float16, 'bfloat16': torch.bfloat16}.get(node.target)
    if node.target == 'to' and len(node.args) == 2 and isinstance(node.args[1], torch.dtype):
        return node.args[1]
    return None


def _arithmetic(node):
    if ((node.op != 'call_method' or node.target not in _METHODS)
            and (node.op != 'call_function' or node.target not in _ARITHMETIC)):
        return False
    if node.kwargs.get('inplace', False) or 'out' in node.kwargs or node.kwargs.get('dtype') not in (None, torch.float32):
        return False
    if node.target in ('softmax', torch.softmax, torch.ops.aten.softmax.int) and len(node.args) > 2:
        return node.args[2] in (None, torch.float32)
    if node.target is F.softmax and len(node.args) > 3:
        return node.args[3] in (None, torch.float32)
    if node.target in (F.silu, F.relu) and len(node.args) > 1:
        return node.args[1] is False
    return True


def _meta_cast(node, source, dtype):
    # Dynamo/pre-grad supplies FakeTensors. Never allocate a matrix to update metadata.
    from torch._subclasses.fake_tensor import FakeTensor
    node.meta = source.meta.copy()
    for key in ('example_value', 'val'):
        v = node.meta.get(key)
        if isinstance(v, FakeTensor) or (isinstance(v, torch.Tensor) and v.device.type == 'meta'):
            node.meta[key] = v.to(dtype)
    if 'tensor_meta' in node.meta:
        node.meta['tensor_meta'] = node.meta['tensor_meta']._replace(dtype=dtype)


def _is_native_chain(a, b):
    add = (operator.add, torch.add, torch.ops.aten.add.Tensor, 'add')
    return ((a.target in _MATMUL and b.target in add)
            or (a.target in (operator.mul, torch.mul, torch.ops.aten.mul.Tensor, 'mul')
                and b.target in ('sum', torch.sum, torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default))
            or (a.target in add and b.target in (F.gelu, F.silu, F.layer_norm, F.softmax, F.relu, torch.relu,
                torch.softmax, 'softmax', 'relu', torch.ops.aten.softmax.int, torch.ops.aten.relu.default,
                torch.ops.aten.gelu.default, torch.ops.aten.silu.default, torch.ops.aten.layer_norm.default)))


def rewrite(graph, *, native=False, shared=False):
    nodes = list(graph.nodes)
    if any(_CAST in n.meta for n in nodes):
        return False
    selected = {n for n in nodes if _arithmetic(n) and _dtype(n) in _LOW}
    if not selected:
        return False
    if native:
        # The default Inductor hook applies specifically to connected F06 patterns.
        if any(n.op == 'call_method' and str(n.target).endswith('_')
               or n.op == 'call_function' and (getattr(getattr(n.target, '_schema', None), 'is_mutable', False)
                                               or n.kwargs.get('inplace', False) or 'out' in n.kwargs)
               for n in nodes):
            return False
        seeds = set()
        for a in selected:
            for user in a.users:
                successors = user.users if _cast_dtype(user) is not None else (user,)
                for b in successors:
                    if _arithmetic(b) and _dtype(b) in (*_LOW, torch.float32) and _is_native_chain(a, b):
                        seeds.update((a, b))
        if not seeds:
            return False
        selected = seeds
    # An FP32 consumer (for example a sum with dtype=float32) can use the
    # widened edge directly. Its own dtype and explicit precision are retained.
    for n in nodes:
        if _dtype(n) in (*_LOW, torch.float32) and _arithmetic(n) and any(a in selected for a in n.all_input_nodes):
            selected.add(n)
    # Mutation or an intervening Python effect can observe/update a tensor.
    # Never reuse an ingress conversion across one in the general pre-grad graph.
    cache_inputs = all(n.op in ('placeholder', 'get_attr', 'output') or _arithmetic(n) for n in nodes)
    first = next(n for n in nodes if n in selected)
    early_sources = set(nodes[:nodes.index(first)])
    original = {n: _dtype(n) for n in selected}
    epilogues = {n: _mm_epilogue(n) for n in selected}
    sums = {n: _bf16_product_sum(n) for n in selected}
    casts = {}
    for node in nodes:
        if node not in selected:
            continue

        def widen(arg):
            if arg in selected or _dtype(arg) not in _LOW:
                return arg
            if arg not in casts:
                anchor = first if cache_inputs and arg in early_sources else node
                with graph.inserting_before(anchor):
                    if shared:
                        wide = graph.call_function(torch.ops.flagtree_p800.fp32_input.default, (arg,))
                    else:
                        wide = graph.call_method('float', (arg,))
                _meta_cast(wide, arg, torch.float32)
                wide.meta[_CAST] = 'input'
                casts[arg] = wide
            return casts[arg]

        node.args = torch.fx.map_arg(node.args, widen)
        node.kwargs = torch.fx.map_arg(node.kwargs, widen)
        value = _value(node)
        if node.target in _MATMUL and (not shared and original[node] in _LOW or torch.is_autocast_enabled(value.device.type)):
            args = tuple(node.args)
            if not args:
                args = (node.kwargs.get('input', node.kwargs.get('self')),)
            if len(args) == 1:
                args += (node.kwargs.get('other', node.kwargs.get('mat2')),)
            node.op, node.target = 'call_function', torch.ops.flagtree_p800.fp32_product.default
            node.args, node.kwargs = args, {}
        elif not shared and epilogues[node]:
            node.op, node.target = 'call_function', torch.ops.flagtree_p800.fp32_epilogue.default
            node.args, node.kwargs = (node.args[0], epilogues[node]), {}
        elif not shared and sums[node] is not None:
            dim, keepdim = sums[node]
            node.op, node.target = 'call_function', torch.ops.flagtree_p800.fp32_row_sum.default
            node.args, node.kwargs = (node.args[0], dim, keepdim), {}
        _meta_cast(node, node, torch.float32)
        if not cache_inputs:
            casts.clear()
    for node in nodes:
        if node not in selected:
            continue
        if original[node] not in _LOW:
            continue
        observers = []
        for user in tuple(node.users):
            if user in selected:
                continue
            target_dtype = _cast_dtype(user)
            if target_dtype in _LOW:
                # This is an explicit boundary even when it was an identity
                # cast in the original graph. Prevent SDNN from erasing it.
                user.op, user.target = 'call_function', torch.ops.flagtree_p800.round_output.default
                user.args, user.kwargs = (node, target_dtype), {}
                user.meta[_CAST] = 'explicit'
            elif target_dtype is None:
                observers.append(user)
        if observers:
            with graph.inserting_after(node):
                out = graph.call_function(torch.ops.flagtree_p800.round_output.default, (node, original[node]))
            _meta_cast(out, node, original[node])
            out.meta[_CAST] = 'output'
            for user in observers:
                user.replace_input_with(node, out)
    if not any(_CAST in n.meta for n in graph.nodes):
        # Under autocast, FP32 inputs and a promoted FP32 output can leave a
        # rewritten MatMul without any explicit ingress/egress FX nodes.
        first.meta[_CAST] = 'compute'
    graph.lint()
    return True


def rewrite_native(graph):
    if any(_CAST in n.meta for n in graph.nodes):
        return False
    from ._p800_inductor_gelu import _on_p800
    operands = [_value(n) for n in graph.nodes if n.op in ('placeholder', 'get_attr')]
    tensors = [x for x in operands if isinstance(x, torch.Tensor)]
    if (not tensors or tensors[0].device.type != 'cuda'
            or not (any(x.dtype in _LOW for x in tensors) or torch.is_autocast_enabled('cuda'))
            or any(x.device != tensors[0].device for x in tensors)
            or (torch.is_grad_enabled() and any(x.requires_grad for x in tensors))):
        return False
    from ._p800_shared_fp32 import template_graph
    return _on_p800(tensors[0].device) and rewrite(graph, native=True, shared=template_graph(graph))


def normalized(gm, inputs):
    if any(_CAST in node.meta for node in gm.graph.nodes):
        return gm
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    if (not tensors or tensors[0].device.type != 'cuda'
            or any(x.device != tensors[0].device for x in tensors)
            or not (any(x.dtype in _LOW for x in tensors) or torch.is_autocast_enabled('cuda'))
            or (torch.is_grad_enabled() and any(x.requires_grad for x in (*tensors, *gm.parameters())))):
        return gm
    from ._resource_codegen import baidu_target
    target = baidu_target()
    if not target or target['arch'] != 3 or target['warp_size'] != 1:
        return gm
    from ._resource_fx import _analyze_graph
    _analyze_graph(gm)
    clone = GraphModule(gm, copy.deepcopy(gm.graph))
    from torch._subclasses.fake_tensor import FakeTensor
    if any(not isinstance(_value(n), FakeTensor) for n in clone.graph.nodes
           if _arithmetic(n) or n.op in ('placeholder', 'get_attr')):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        FakeTensorProp(clone, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(*inputs)
        for n in clone.graph.nodes:
            if 'val' in n.meta:
                n.meta['example_value'] = n.meta['val']
    if not rewrite(clone.graph):
        return gm
    clone.recompile()
    return clone


def is_wide(gm):
    return any(_CAST in n.meta for n in gm.graph.nodes)


def inner_graph(gm):
    """Extract one FP32 region with a single result for an existing hardware shortcut."""
    graph = torch.fx.Graph()
    env = {}
    exit_node = None
    for n in gm.graph.nodes:
        boundary = n.meta.get(_CAST)
        if boundary == 'input':
            src, = n.args
            if src.op != 'placeholder':
                return None
            env[n] = env[src]
            env[n].meta = {k: v for k, v in n.meta.items() if k != _CAST}
        elif boundary == 'output':
            if exit_node is not None or any(u.op != 'output' for u in n.users):
                return None
            env[n] = env[n.args[0]]
            exit_node = n
        elif n.op == 'output':
            returned = n.args[0]
            leaves = returned if isinstance(returned, (list, tuple)) else (returned,)
            if exit_node is None or len(leaves) != 1 or leaves[0] is not exit_node:
                return None
            env[n] = graph.node_copy(n, env.__getitem__)
        elif n.target == torch.ops.flagtree_p800.fp32_epilogue.default:
            src, mode = n.args
            act = {'relu': torch.relu, 'silu': F.silu, 'gelu_tanh': F.gelu}[mode]
            kwargs = {'approximate': 'tanh'} if mode == 'gelu_tanh' else {}
            env[n] = graph.call_function(act, (env[src],), kwargs)
            env[n].meta = n.meta.copy()
        elif n.target == torch.ops.flagtree_p800.fp32_product.default:
            env[n] = graph.call_function(torch.matmul, tuple(env[a] for a in n.args))
            env[n].meta = n.meta.copy()
        elif n.op == 'placeholder' or _arithmetic(n):
            if n.op == 'placeholder' and any(u.meta.get(_CAST) != 'input' for u in n.users):
                return None
            env[n] = graph.node_copy(n, env.__getitem__)
        else:
            return None
    if exit_node is None or exit_node.args[1] not in _LOW:
        return None
    return GraphModule(gm, graph), exit_node.args[1]


def compile_wide(gm, inputs, *, single_kernel=False):
    from . import _p800_flat, _p800_native_mm, _p800_reduce

    region = inner_graph(gm)
    if region is None:
        raise ValueError('requires a single FP32 region with a low-precision result')
    inner, dtype = region
    if not all(isinstance(t, torch.Tensor) and t.dtype == dtype for t in inputs):
        raise ValueError('mixed external layouts are handled by the graph backend')
    samples = tuple(t.float() for t in inputs)
    mm = _p800_native_mm.match(inner, samples)
    flat = _p800_flat.match(inner, samples)
    flat_mixed = flat is not None and all(t.is_contiguous() for t in inputs)
    if single_kernel and not flat_mixed:
        raise ValueError('the hardware schedule requires additional conversion launches')
    mixed = (mm is not None and mm['K'] == 64
             and (mm['M'], mm['N']) in ((64, 1024), (512, 1024), (2048, 2048))
             and all(t.is_contiguous() for t in inputs))
    if flat_mixed:
        flat.update(dtype=str(dtype).removeprefix('torch.'), final_round_only=True)
        kernel, proof = _p800_flat.compile(flat, inputs)
    elif mixed:
        from ._p800_fp32_mm import prepare
        kernel, proof = prepare(mm)
    elif mm is not None:
        native, proof = _p800_native_mm.compile(mm, samples)
        def kernel(xs):
            return native(*xs)
    else:
        for module in (_p800_flat, _p800_reduce):
            spec = module.match(inner, samples)
            if spec is not None:
                kernel, proof = module.compile(spec, samples)
                break
        else:
            raise ValueError('no hardware shortcut for the FP32 region')

    output = next(n for n in gm.graph.nodes if n.op == 'output').args[0]
    proof['final_round_only'] = True
    proof['compute_dtype'] = 'float32'
    proof['output_dtype'] = str(dtype)
    proof['input_conversion'] = 'in_kernel' if mixed or flat_mixed else 'before_kernel'
    # The SDK cast is a real launch, as are any input casts or separate FP32 activation.
    proof['kernel_count'] = 1 if flat_mixed else None
    def run(*xs):
        args = xs if mixed or flat_mixed else tuple(x.float() for x in xs)
        value, = kernel(args)
        if mixed and mm['act']:
            act = mm['act']
            if act == 1:
                value = F.silu(value)
            elif act == 2:
                value = torch.relu(value)
            else:
                value = F.gelu(value, approximate='tanh' if act == 3 else 'none')
        result = value if flat_mixed else value.to(dtype)
        if isinstance(output, tuple):
            return (result,)
        if isinstance(output, list):
            return [result]
        return result
    return run, proof
