"""Keep the original BF16 Add in FX and compile its GELU consumer for cluster."""
import copy


def eligible(gm, inputs, add):
    import torch
    placeholders = [n for n in gm.graph.nodes if n.op == 'placeholder']
    if (len(inputs) != 2 or len(placeholders) != 2 or set(add.all_input_nodes) != set(placeholders)
            or add.kwargs.get('out') is not None or add.kwargs.get('alpha', 1) != 1):
        return False
    if not all(isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16
               and x.layout == torch.strided and x.is_contiguous() for x in inputs):
        return False
    matrix, bias = sorted(inputs, key=lambda x: x.ndim, reverse=True)
    return (matrix.ndim == 2 and bias.ndim == 1 and matrix.device == bias.device
            and matrix.shape[1] == bias.shape[0] and 0 < matrix.numel() <= 2**31 - 1)


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['src', 'out'], do_not_specialize_on_alignment=['src', 'out'])
    def _bf16_gelu(src, out, N: tl.constexpr, BLOCK: tl.constexpr, TANH: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        # A redundant mask changes the P800 lowering even for complete tiles.
        if N % BLOCK:
            x = tl.load(src + i, mask=i < N, other=0).to(tl.float32)
        else:
            x = tl.load(src + i).to(tl.float32)
        if TANH:
            u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            result = x * tl.sigmoid(2.0 * u)
        else:
            result = (0.5 * x) * (1.0 + tl.math.erf(x * 0.7071067811865476))
        if N % BLOCK:
            tl.store(out + i, result, mask=i < N)
        else:
            tl.store(out + i, result)

    return _bf16_gelu


def compile(gm, inputs, candidate, mode):
    import torch
    import triton
    from ._resource_codegen import bind_evidence

    nodes = [n for n in gm.graph.nodes if n.op in ('call_method', 'call_function')]
    if (mode not in ('none', 'tanh') or candidate.single_kernel
            or not eligible(gm, inputs, nodes[0])):
        raise ValueError('unsupported BF16 Add + GELU partition')

    count = max(x.numel() for x in inputs)
    size = triton.next_power_of_2(count)
    block = min(size, max(2048, min(8192, size // 128)))
    launch = _kernel()[(triton.cdiv(count, block),)]
    unary = dict(route='bf16_gelu', resource='cluster', direct_launches=1,
                 resource_verified=False, external_calls=0, tiles=dict(xblock=block, warps=4))

    def cluster_gelu(summed):
        out = torch.empty_like(summed)
        binary = launch(summed, out, N=count, BLOCK=block, TANH=mode == 'tanh', num_warps=4, num_stages=1)
        if not unary['resource_verified']:
            bind_evidence(unary, binary)
        return out

    # Preserve the Add call, its argument order, and the original output pytree.
    graph = copy.deepcopy(gm.graph)
    add, gelu = [n for n in graph.nodes if n.op in ('call_method', 'call_function')]
    gelu.target = cluster_gelu
    gelu.args = (add,)
    gelu.kwargs = {}
    fn = torch.fx.GraphModule(gm, graph).forward
    producer = dict(route='eager_add', resource='eager', external_calls=1, resource_verified=False,
                    node_ids=[add.name], kernel_evidence=[])
    # The vendor dispatch is opaque to Triton. Only device tracing can count
    # or identify its launches. Certify the bound consumer independently.
    return fn, dict(route=candidate.name, regions=[producer, unary], kernel_count=None)
