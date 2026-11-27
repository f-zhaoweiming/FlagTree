"""Inference-only BF16 bias + GELU templates for the P800 Torch 2.9 Inductor."""
import hashlib
import operator
from pathlib import Path

import torch
from torch._inductor import ir
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel
from torch._inductor.virtualized import V


@torch.library.custom_op("flagtree_p800::bias_gelu", mutates_args=())
def _bias_gelu(matrix: torch.Tensor, bias: torch.Tensor, mode: str, revision: str) -> torch.Tensor:
    return torch.nn.functional.gelu(matrix + bias, approximate=mode)


@_bias_gelu.register_fake
def _fake(matrix, bias, mode, revision):
    return torch.empty_like(matrix)


class _LocalOutputKernel(TritonTemplateKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(self.copy_output, self.vector_output)

    def vector_output(self):
        with self.create_subgraph_body("<STORE_OUTPUT>"):
            output = self.args.output(self.output_node.get_name())
        call = (f"tle.raw.call(kl3_bf16_bias_gelu, (A, B, {output}, tl.program_id(0), "
                "tl.full((), PROGS, tl.int32), tl.full((), M * N, tl.int32), "
                "tl.full((), N, tl.int32), tl.full((), WIDTH, tl.int32)))")
        return self._register_hook("<STORE_OUTPUT>", lambda: call)

    def copy_output(self):
        # This template materializes its BF16 result with DMA. The lowering
        # keeps consumers outside this template so they read the actual result.
        with self.create_subgraph_body("<STORE_OUTPUT>"):
            output = self.args.output(self.output_node.get_name())
        code = (f"dst = tl.make_tensor_descriptor({output}, shape=[M, N], "
                "strides=[N, 1], block_shape=[ROWS, N])\n"
                "    tle.gpu.copy(tile, dst, [ROWS, N], [first, 0])")
        return self._register_hook("<STORE_OUTPUT>", lambda: code)


class _LocalOutputTemplate(TritonTemplate):
    kernel_type = _LocalOutputKernel


def _grid(m, n, meta):
    return (m // meta["ROWS"], 1, 1)


def _unary_grid(m, n, meta):
    return (m * n // meta["BLOCK"], 1, 1)


_ACTIVATION = """
    if TANH:
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        y = z * tl.sigmoid(2.0 * u)
    else:
        y = (0.5 * z) * (1.0 + tl.math.erf(z * 0.7071067811865476))
"""
_FUSED = """
import triton.experimental.tle.language as tle
{{def_kernel("A", "B")}}
    src = tl.make_tensor_descriptor(A, shape=[M, N], strides=[N, 1], block_shape=[ROWS, N])
    bias = tl.make_tensor_descriptor(B, shape=[N], strides=[1], block_shape=[N])
    first = tl.program_id(0) * ROWS
    tile = tle.gpu.alloc([ROWS, N], dtype=tl.bfloat16, scope=tle.gpu.lmem)
    row_bias = tle.gpu.alloc([N], dtype=tl.bfloat16, scope=tle.gpu.lmem)
    rows = tl.broadcast_to(tl.arange(0, ROWS)[:, None], (ROWS, N))
    cols = tl.broadcast_to(tl.arange(0, N)[None, :], (ROWS, N))
    p = tle.gpu.local_ptr(tile, (rows, cols))
    bp = tle.gpu.local_ptr(row_bias, (tl.arange(0, N),))
    tle.gpu.copy(src, tile, [ROWS, N], [first, 0])
    tle.gpu.copy(bias, row_bias, [N], [0])
    summed = tl.load(p).to(tl.float32) + tl.load(bp)[None, :].to(tl.float32)
    # Eager rounds Add to BF16 before GELU. Keep that boundary in local memory.
    tl.store(p, summed.to(tl.bfloat16))
    z = tl.load(p).to(tl.float32)
""" + _ACTIVATION + """
    tl.store(p, y.to(tl.bfloat16))
    {{copy_output()}}
"""
_UNARY = """
{{def_kernel("A")}}
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(A + idx).to(tl.float32)
""" + _ACTIVATION + """
    row = idx // N
    col = idx % N
    {{store_output(("row", "col"), "y")}}
"""


def _vector_grid(m, n, meta):
    return (meta["PROGS"], 1, 1)


_VECTOR = """
import triton.experimental.tle as tle
from triton.experimental.graph_fusion._p800_bf16_gelu import kl3_bf16_bias_gelu
{{def_kernel("A", "B")}}
    {{vector_output()}}
"""
_vector_template = _LocalOutputTemplate("flagtree_bf16_vector_gelu", _vector_grid, _VECTOR)
_fused_template = _LocalOutputTemplate("flagtree_bf16_bias_gelu", _grid, _FUSED)
_unary_template = TritonTemplate("flagtree_bf16_gelu", _unary_grid, _UNARY)

# Include lowering and matcher changes in the FX graph cache key.
_here = Path(__file__)
_REVISION = hashlib.sha256(_here.read_bytes() + _here.with_name('_p800_bf16_gelu.py').read_bytes()
                           + _here.with_name('_p800_bf16_gelu.xpu').read_bytes()).hexdigest()[:16]
_SMALL = {(64, 1024), (512, 1024)}
_LARGE = (2048, 2048)


@register_lowering(torch.ops.flagtree_p800.bias_gelu.default, type_promotion_kind=None)
def _lower(matrix, bias, mode, revision):
    m, n = map(int, matrix.get_size())
    layout = ir.FixedLayout(matrix.get_device(), matrix.get_dtype(), [m, n], [n, 1])
    if mode == "none":
        from ._p800_layernorm import toolchain_available
        if toolchain_available():
            from ._p800_bf16_gelu import PROGRAMS, tile_width
            width = tile_width(m, n)
            if width is not None:
                matrix = ir.ExternKernel.require_exact_strides(matrix, [n, 1])
                bias = ir.ExternKernel.require_exact_strides(bias, [1])
                choice = _vector_template.generate([matrix, bias], layout, num_stages=1, num_warps=4,
                                                    M=m, N=n, WIDTH=width, PROGS=PROGRAMS)
                output = choice.output_node()
                V.graph.no_fuse_buffer_names.add(output.get_name())
                return output
    if (m, n) in _SMALL:
        rows = 2 if m == 64 and mode == "tanh" else 8
        # DMA descriptors below use dense element strides, including for
        # producers whose layout Inductor has not chosen yet.
        matrix = ir.ExternKernel.require_exact_strides(matrix, [n, 1])
        bias = ir.ExternKernel.require_exact_strides(bias, [1])
        choice = _fused_template.generate([matrix, bias], layout, num_stages=1, num_warps=4,
                                          M=m, N=n, ROWS=rows, TANH=mode == "tanh")
        output = choice.output_node()
        # copy_output writes the rounded tile directly. A fused epilogue
        # would be discarded by that store hook. Only fence this one buffer.
        V.graph.no_fuse_buffer_names.add(output.get_name())
        return output

    assert (m, n) == _LARGE
    summed = fallback_handler(torch.ops.aten.add.Tensor, add_to_fallback_set=False)(matrix, bias)
    if mode == "none":
        return fallback_handler(torch.ops.aten.gelu.default, add_to_fallback_set=False)(summed)
    return _unary_template.generate([summed], layout, num_stages=1, num_warps=4,
                                    N=n, BLOCK=8192, TANH=True).output_node()


def _match(gelu, shapes):
    if (gelu.op != "call_function" or gelu.target not in (torch.nn.functional.gelu, torch.ops.aten.gelu.default)
            or len(gelu.args) != 1 or set(gelu.kwargs) - {"approximate"}):
        return
    add = gelu.args[0]
    if not isinstance(add, torch.fx.Node):
        return
    if not ((add.op == "call_function" and add.target in (operator.add, torch.add, torch.ops.aten.add.Tensor))
            or (add.op == "call_method" and add.target == "add")):
        return
    # Replacing the pair at GELU must not reread inputs across mutations.
    if (add.next is not gelu or len(add.users) != 1 or len(add.args) != 2
            or set(add.kwargs) - {"alpha"} or add.kwargs.get("alpha", 1) != 1):
        return
    mode = gelu.kwargs.get("approximate", "none")
    if mode not in ("none", "tanh") or not all(isinstance(arg, torch.fx.Node) for arg in add.args):
        return
    operands = add.args
    values = [x.meta.get("example_value") for x in operands]
    if not all(isinstance(v, torch.Tensor) and v.device.type == "cuda" and v.dtype == torch.bfloat16
               and v.layout == torch.strided and all(type(d) is int for d in v.shape) and v.is_contiguous()
               for v in values):
        return
    if any(v.requires_grad for v in values):
        # A live module Parameter keeps requires_grad in inference. Check
        # both the tracing context and the recorded results before lowering
        # it through an inference-only custom operator.
        traced = [node.meta.get("example_value") for node in (add, gelu)]
        if torch.is_grad_enabled() or not all(isinstance(v, torch.Tensor) and not v.requires_grad for v in traced):
            return
    if values[0].ndim == 1:
        operands = operands[::-1]
        values.reverse()
    matrix, bias = values
    if (tuple(matrix.shape) not in shapes or tuple(bias.shape) != (matrix.shape[1],)
            or matrix.device != bias.device):
        return
    return add, operands, mode, matrix.device


def _on_p800(device):
    from triton.runtime import driver
    from triton.backends.xpu.driver import XPUDriver
    active = driver.active
    if not isinstance(active, XPUDriver):
        return False
    # Query the graph operand's logical device, which can differ from the
    # current device during compilation. Torch itself advertises SM86 here.
    index = device.index if device.index is not None else active.get_current_device()
    properties = active.utils.get_device_properties(index)
    return (properties["device_model"], properties["warpSize"]) == (3, 1)


def rewrite(graph, *, small_only=False, vector_exact=False):
    """Lower adjacent, single-use BF16 Add/GELU pairs at compile time."""
    shapes = _SMALL if small_only and not vector_exact else _SMALL | {_LARGE}
    devices = {}
    for gelu in list(graph.nodes):
        match = _match(gelu, shapes)
        if match is None:
            continue
        add, operands, mode, device = match
        # The historical large two-kernel policy is opt-in and was tuned
        # for terminal pairs with graph inputs.
        if tuple(operands[0].meta["example_value"].shape) == _LARGE:
            if small_only:
                from ._p800_layernorm import toolchain_available
                if mode != "none" or not toolchain_available():
                    continue
            if (any(x.op != "placeholder" for x in operands)
                    or [node.op for node in graph.nodes if node.op != "placeholder"] != [add.op, gelu.op, "output"]
                    or gelu.next.all_input_nodes != [gelu]):
                continue
        if device not in devices:
            devices[device] = _on_p800(device)
        if devices[device]:
            gelu.target = torch.ops.flagtree_p800.bias_gelu.default
            gelu.args = (*operands, mode, _REVISION)
            gelu.kwargs = {}
            graph.erase_node(add)
