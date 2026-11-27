"""Opt-in static inference plans for constrained single-template FX graphs.

The public entry point, :func:`compile_static`, accepts graph modules whose
whole computational body is covered by exactly one template from
:func:`._patterns.default_registry` and returns a callable plan that avoids the
outer Dynamo/AOT/FX interpreter on the steady-state path.  The original
``GraphModule`` is retained for live attribute lookup; all ``get_attr`` tensor
nodes are lifted to extra placeholders before tracing and native Inductor
compilation.

On NVIDIA SM80 and newer and positively identified P800, bounded dense GEMM+bias epilogues enable Inductor's
automatic GEMM template selection and epilogue fusion. Matmul and addition
output-dtype rounding are retained; no project arithmetic kernel is supplied.
Larger dense NVIDIA activation epilogues also autotune, retaining both library
and template candidates and benchmarking whether epilogue fusion is beneficial.

The optional ``avoid_bias_materialization`` mode recognizes the narrow pattern
``torch.add(torch.matmul(x, w), bias)`` (either add operand order) and executes the
same eager operators every call::

    result = torch.mm(x, w)
    torch.add(result, bias, out=result)

This native reuse path does not use ``addmm`` or cache a bias value. It is
considered after single-kernel small-matrix eligibility and is CUDA-only;
other devices use the native Inductor path.

Scope is intentionally narrow inference: all tensor arguments and live
attributes must be exact ``torch.Tensor``/``torch.nn.Parameter`` instances.
Tensors requiring gradients are accepted only while gradient recording is
disabled; enabling gradients later routes the call to the fallback.
Operator-only FX graphs do not depend on the training
flag, which is therefore not guarded, but this plan is only intended for
inference calls. All positional arguments, including optional ones, must be
supplied explicitly. Any guard miss or omitted/keyword/gradient call is routed to
the caller-supplied ``fallback`` callable.  Initial graph/state validation
failures raise :class:`UnsupportedStaticInference`; errors from the native
compiler are not caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import ast
import operator
from typing import Any, Callable, Optional

import torch
from torch import fx

from ._patterns import default_registry, match_graph

__all__ = ["StaticInferencePlan", "UnsupportedStaticInference", "compile_static"]


class UnsupportedStaticInference(RuntimeError):
    """Initial graph or mode is outside the static-inference contract.

    ``reason`` is a stable short code; ``detail`` may carry additional
    diagnostic text.  The caller is expected to use its normal
    ``torch.compile`` path for this graph.
    """

    def __init__(self, reason: str, detail: Optional[str] = None):
        self.reason = str(reason)
        self.detail = detail
        message = self.reason if detail is None else f"{self.reason}: {detail}"
        super().__init__(message)


_KIND_INDUCTOR = "inductor_static_inference"
_KIND_NATIVE_MM_ADD = "native_mm_add_out_reuse"
_KIND_SMALL_MATMUL = "inductor_small_matmul_single_kernel"

# Scoped to compilation of eligible small graphs. Inductor chooses the tile and
# generates the complete epilogue from the FX operations. Preserve mm/add dtype
# boundaries: addmm rewriting and dropping intermediate casts change BF16 output.
_SMALL_MATMUL_OPTIONS = {
    "max_autotune": True,
    "max_autotune_gemm_backends": "TRITON",
    "epilogue_fusion": True,
    "benchmark_epilogue_fusion": False,
    "pattern_matcher": False,
    "emulate_precision_casts": True,
    # Inductor's static CUDA launcher bypasses Triton launch hooks. Use its
    # Triton launcher so hooks added after compilation remain observable.
    "use_static_cuda_launcher": False,
    # TensorGuards in every public call already check sizes and strides.
    "size_asserts": False,
    "triton.cudagraphs": False,
}

# Larger NVIDIA epilogues must also search GEMM implementations. Unlike the
# bounded small path, retain the library candidate and benchmark fusion: one
# kernel is not necessarily faster than a library GEMM plus pointwise work.
_LARGE_MATMUL_OPTIONS = {
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "epilogue_fusion": True,
    "benchmark_epilogue_fusion": True,
    "pattern_matcher": False,
    "emulate_precision_casts": True,
    "triton.cudagraphs": False,
}


def _small_matmul_options():
    from torch._inductor import config
    options = dict(_SMALL_MATMUL_OPTIONS)
    # Older Inductor already uses Triton's launcher and has no static-CUDA
    # launcher switch. Keep all precision/fusion options mandatory.
    if "use_static_cuda_launcher" not in config.get_config_copy():
        options.pop("use_static_cuda_launcher")
    return options


_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)
_MUTATING_CALLABLES = frozenset({
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.imatmul,
    operator.itruediv,
    operator.ifloordiv,
    operator.imod,
    operator.ipow,
    operator.ilshift,
    operator.irshift,
    operator.iand,
    operator.ior,
    operator.ixor,
    operator.setitem,
    operator.delitem,
})
_ALLOWED_NODE_OPS = frozenset({
    "placeholder",
    "get_attr",
    "call_function",
    "call_method",
    "output",
})
_MISSING = object()


@dataclass(frozen=True)
class _TensorSpec:
    """Static tensor metadata; never contains tensor values."""

    shape: tuple
    stride: tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    requires_grad: bool

    def as_dict(self):
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "layout": str(self.layout),
            "requires_grad": self.requires_grad,
            "conjugate_view": False,
            "negative_view": False,
        }


@dataclass(frozen=True)
class _MmAddReuse:
    x_index: int
    w_index: int
    bias_attr_index: int
    x_name: str
    w_name: str
    bias_target: str
    rows: int
    width: int
    matmul_name: str
    bias_first: bool


@dataclass(frozen=True)
class _SmallMatmul:
    indices: tuple
    shape: tuple
    bias_strides: tuple
    bias_first: bool
    activation: str


@dataclass(frozen=True)
class _Analysis:
    gm: fx.GraphModule
    placeholders: tuple
    example_inputs: tuple
    input_specs: tuple
    attr_targets: tuple
    attr_values: tuple
    attr_specs: tuple
    get_attr_nodes: tuple
    template: Any
    match: Any
    output_node: fx.Node
    forward_code: Any
    precision_state: tuple


_PRECISION_FIELDS = (
    "float32_matmul_precision",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "cuda_matmul_allow_fp16_reduced_precision_reduction",
    "cuda_matmul_allow_bf16_reduced_precision_reduction",
    "deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "cudnn_deterministic",
)

# ---------------------------------------------------------------------------
# Static metadata and process-mode guards


def _tensor_type_ok(value: Any) -> bool:
    """Exact Tensor/Parameter check; Tensor subclasses are outside the contract."""
    return type(value) in _TENSOR_TYPES


def _tensor_spec(value: torch.Tensor) -> _TensorSpec:
    return _TensorSpec(
        shape=tuple(value.shape),
        stride=tuple(value.stride()),
        storage_offset=value.storage_offset(),
        dtype=value.dtype,
        device=value.device,
        layout=value.layout,
        requires_grad=value.requires_grad,
    )


def _metadata_matches(value: Any, spec: _TensorSpec) -> bool:
    """Return a Python bool; never converts a tensor or comparison result."""
    if not _tensor_type_ok(value):
        return False
    if torch.is_grad_enabled() and value.requires_grad:
        return False
    if value.is_conj() or value.is_neg():
        return False
    if tuple(value.shape) != spec.shape:
        return False
    if tuple(value.stride()) != spec.stride:
        return False
    if value.storage_offset() != spec.storage_offset:
        return False
    if value.dtype != spec.dtype:
        return False
    if value.device != spec.device:
        return False
    if value.layout != spec.layout:
        return False
    return True


def _non_aliased(values: tuple) -> bool:
    """True only when every pair of live tensors has disjoint storage."""
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if torch._C._is_alias_of(values[left], values[right]):
                return False
    return True


def _autocast_active() -> bool:
    # torch.is_autocast_enabled(device_type) is the current native API.
    return bool(torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"))


def _torch_function_mode_active() -> bool:
    checker = getattr(getattr(torch, "overrides", None), "_is_torch_function_mode_enabled", None)
    if checker is None:
        checker = getattr(torch._C, "_is_torch_function_mode_enabled", None)
    if checker is None:
        # Fail closed when the native query is unavailable.
        return True
    return bool(checker())


def _dispatch_mode_active() -> bool:
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    except (ImportError, AttributeError):
        return True
    return bool(_get_current_dispatch_mode_stack())


def _functorch_active() -> bool:
    checker = getattr(torch._C, "_are_functorch_transforms_active", None)
    if checker is None:
        return True
    return bool(checker())


def _mode_reason() -> Optional[str]:
    """Return a guard-miss reason for active transform/precision contexts."""
    if _autocast_active():
        return "autocast_active"
    if _FUNCTION_MODE_ENABLED is None or _FUNCTION_MODE_ENABLED():
        return "torch_function_mode_active"
    if (_DISPATCH_STACK_LENGTH() if _DISPATCH_STACK_LENGTH is not None else _dispatch_mode_active()):
        return "dispatch_mode_active"
    if _FUNCTORCH_ACTIVE is None or _FUNCTORCH_ACTIVE():
        return "functorch_transform_active"
    return None


def _forward_hooks_active(gm: fx.GraphModule) -> bool:
    if getattr(gm, "_forward_hooks", None):
        return True
    if getattr(gm, "_forward_pre_hooks", None):
        return True
    module_impl = torch.nn.modules.module
    if getattr(module_impl, "_global_forward_hooks", None):
        return True
    if getattr(module_impl, "_global_forward_pre_hooks", None):
        return True
    return False


def _public_reduction_setting(name: str):
    matmul = torch.backends.cuda.matmul
    value = getattr(matmul, name)
    # Newer PyTorch's native query includes split-K, exposed separately by the
    # public facade. Older versions return only the reduction boolean.
    try:
        return value, getattr(matmul, name + "_split_k")
    except AttributeError:
        return value


def _python_precision_state() -> tuple:
    cuda_matmul = torch.backends.cuda.matmul
    return (
        torch.get_float32_matmul_precision(),
        bool(cuda_matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        _public_reduction_setting("allow_fp16_reduced_precision_reduction"),
        _public_reduction_setting("allow_bf16_reduced_precision_reduction"),
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        bool(torch.backends.cudnn.deterministic),
    )


# Bind queries, never state values: these functions still read current TLS and
# process settings on every invocation. Avoid repeated Python property dispatch
# and construction of a list containing the dispatch-mode stack on the hot path.
_FUNCTION_MODE_ENABLED = getattr(torch._C, "_is_torch_function_mode_enabled", None)
_DISPATCH_STACK_LENGTH = getattr(torch._C, "_len_torch_dispatch_stack", None)
_FUNCTORCH_ACTIVE = getattr(torch._C, "_are_functorch_transforms_active", None)
_PRECISION_GETTER_NAMES = (
    "_get_float32_matmul_precision",
    "_get_cublas_allow_tf32",
    "_get_cudnn_allow_tf32",
    "_get_cublas_allow_fp16_reduced_precision_reduction",
    "_get_cublas_allow_bf16_reduced_precision_reduction",
    "_get_deterministic_algorithms",
    "_get_deterministic_algorithms_warn_only",
    "_get_cudnn_deterministic",
)
_PRECISION_GETTERS = tuple(getattr(torch._C, name, None) for name in _PRECISION_GETTER_NAMES)
if all(callable(fn) for fn in _PRECISION_GETTERS):
    (_get_matmul_precision, _get_tf32, _get_cudnn_tf32, _get_fp16_reduction, _get_bf16_reduction, _get_deterministic,
     _get_warn_only, _get_cudnn_deterministic) = _PRECISION_GETTERS

    def _precision_state() -> tuple:
        return (_get_matmul_precision(), _get_tf32(), _get_cudnn_tf32(), _get_fp16_reduction(), _get_bf16_reduction(),
                _get_deterministic(), _get_warn_only(), _get_cudnn_deterministic())
else:
    _precision_state = _python_precision_state


def _precision_info(state: tuple) -> dict:
    return dict(zip(_PRECISION_FIELDS, state))


# ---------------------------------------------------------------------------
# Initial graph validation


def _resolve_attr(module: torch.nn.Module, target: str):
    """Resolve a dotted FX get_attr path, returning _MISSING on absence."""
    if not isinstance(target, str) or not target:
        return _MISSING
    atoms = target.split(".")
    if any(not atom for atom in atoms):
        return _MISSING
    value = module
    for atom in atoms:
        try:
            value = getattr(value, atom)
        except AttributeError:
            return _MISSING
    return value


def _call_name(target: Any) -> str:
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    return str(target)


def _mutation_reason(node: fx.Node) -> Optional[str]:
    """Reject mutating calls, including out=/inplace= and in-place overloads."""
    if node.op not in ("call_function", "call_method"):
        return None
    if node.target in _MUTATING_CALLABLES:
        return "mutable_operator"
    name = _call_name(node.target)
    short = name.split(".", 1)[0]
    if short.endswith("_"):
        return "inplace_call"
    if node.kwargs.get("inplace") is True:
        return "inplace_keyword"
    if node.kwargs.get("out") is not None:
        return "out_keyword"
    return None


def _check_placeholder(node: fx.Node) -> None:
    # Defaults do not affect a call that explicitly supplies every argument.
    # Arity/keyword guards retain the original callable for omitted arguments.
    if len(node.args) > 1 or node.kwargs:
        raise UnsupportedStaticInference("non_flat_placeholder", detail=f"placeholder {node.target!r} has defaults")
    if not isinstance(node.target, str) or not node.target.isidentifier():
        raise UnsupportedStaticInference("non_flat_placeholder")


def _check_live_tensor(value: Any, label: str) -> _TensorSpec:
    if not _tensor_type_ok(value):
        raise UnsupportedStaticInference("unsupported_tensor_type", detail=f"{label} is {type(value).__name__}")
    if torch.is_grad_enabled() and value.requires_grad:
        raise UnsupportedStaticInference("requires_grad_not_supported", detail=label)
    if value.is_conj() or value.is_neg():
        raise UnsupportedStaticInference("conjugate_or_negative_view", detail=label)
    return _tensor_spec(value)


def _analyze(gm: fx.GraphModule, example_inputs: Any) -> _Analysis:
    if not isinstance(gm, fx.GraphModule):
        raise UnsupportedStaticInference("not_fx_graph_module", detail=type(gm).__name__)

    if isinstance(example_inputs, torch.Tensor):
        raise UnsupportedStaticInference("example_inputs_must_be_a_sequence")
    try:
        inputs = tuple(example_inputs)
    except TypeError as exc:
        raise UnsupportedStaticInference("invalid_example_inputs", detail=str(exc)) from exc

    forward = getattr(gm, "forward", None)
    forward_code = getattr(forward, "__code__", None)
    if forward_code is None:
        raise UnsupportedStaticInference("missing_forward_code")
    if _forward_hooks_active(gm):
        raise UnsupportedStaticInference("forward_hooks_active")
    initial_mode = _mode_reason()
    if initial_mode is not None:
        raise UnsupportedStaticInference(initial_mode)

    nodes = list(gm.graph.nodes)
    output_nodes = [node for node in nodes if node.op == "output"]
    if len(output_nodes) != 1 or output_nodes[0] is not nodes[-1]:
        raise UnsupportedStaticInference("missing_single_output")

    placeholders = []
    get_attr_nodes = []
    attr_targets = []
    attr_values_by_target = {}
    for node in nodes:
        if node.op == "call_module":
            raise UnsupportedStaticInference("call_module_not_allowed", detail=node.name)
        if node.op not in _ALLOWED_NODE_OPS:
            raise UnsupportedStaticInference("unsupported_node_op", detail=f"{node.op} ({node.name})")
        mutation = _mutation_reason(node)
        if mutation is not None:
            raise UnsupportedStaticInference("mutation_not_allowed", detail=f"{mutation} ({node.name})")
        if node.op == "placeholder":
            _check_placeholder(node)
            placeholders.append(node)
        elif node.op == "get_attr":
            if not isinstance(node.target, str) or not node.target:
                raise UnsupportedStaticInference("non_string_get_attr", detail=node.name)
            if node.target not in attr_values_by_target:
                value = _resolve_attr(gm, node.target)
                if value is _MISSING:
                    raise UnsupportedStaticInference("attribute_resolution_failed", detail=node.target)
                if not _tensor_type_ok(value):
                    raise UnsupportedStaticInference("non_tensor_get_attr", detail=node.target)
                attr_targets.append(node.target)
                attr_values_by_target[node.target] = value
            get_attr_nodes.append(node)

    if len(placeholders) != len(inputs):
        raise UnsupportedStaticInference(
            "placeholder_arity_mismatch",
            detail=f"graph has {len(placeholders)}, example_inputs has {len(inputs)}",
        )

    input_specs = tuple(_check_live_tensor(value, f"example_inputs[{index}]") for index, value in enumerate(inputs))
    attr_specs = []
    attr_values = []
    for target in attr_targets:
        value = attr_values_by_target[target]
        attr_specs.append(_check_live_tensor(value, f"get_attr {target!r}"))
        attr_values.append(value)
    attr_specs = tuple(attr_specs)
    attr_values = tuple(attr_values)

    all_live = inputs + attr_values
    if not _non_aliased(all_live):
        raise UnsupportedStaticInference("aliasing_tensors", detail="arguments and live attributes must not alias")

    selected, rejected = match_graph(gm, default_registry())
    if not selected:
        raise UnsupportedStaticInference("no_supported_template", detail=str(tuple(rejected)))
    if len(selected) != 1:
        raise UnsupportedStaticInference("multiple_supported_templates", detail=str(len(selected)))
    template, match = selected[0]

    computational = {node for node in nodes if node.op not in ("placeholder", "get_attr", "output")}
    if computational != set(match.nodes):
        raise UnsupportedStaticInference(
            "uncovered_computational_nodes",
            detail=str(sorted(node.name for node in computational - set(match.nodes))),
        )

    output_node = nodes[-1]
    if output_node.kwargs or len(output_node.args) != 1:
        raise UnsupportedStaticInference("output_not_single_tensor")
    output_value = output_node.args[0]
    if not isinstance(output_value, fx.Node):
        raise UnsupportedStaticInference("output_not_single_tensor")
    if output_value is not match.nodes[-1]:
        raise UnsupportedStaticInference("output_not_region_end", detail=output_value.name)

    return _Analysis(
        gm=gm,
        placeholders=tuple(placeholders),
        example_inputs=inputs,
        input_specs=input_specs,
        attr_targets=tuple(attr_targets),
        attr_values=attr_values,
        attr_specs=attr_specs,
        get_attr_nodes=tuple(get_attr_nodes),
        template=template,
        match=match,
        output_node=output_node,
        forward_code=forward_code,
        precision_state=_precision_state(),
    )


# ---------------------------------------------------------------------------
# Narrow native mm + bias(out=) reuse recognition


def _dense_contiguous_float(value: torch.Tensor) -> bool:
    return (_tensor_type_ok(value) and value.layout == torch.strided and value.is_floating_point()
            and value.is_contiguous())


def _find_mm_add_reuse(analysis: _Analysis):
    """Return ``(_MmAddReuse, None)`` or ``(None, skip_reason)``.

    This function only recognizes the exact validated two-dimensional pattern.  A None result
    means the caller should use the normal native compiler path.
    """
    match = analysis.match
    if match.kind != "matmul_bias":
        return None, "template_is_not_matmul_bias"
    if len(match.nodes) != 2 or len(match.inputs) != 3:
        return None, "unexpected_match_shape"

    product, addition = match.nodes
    x_node, w_node, bias_node = match.inputs
    if not all(isinstance(node, fx.Node) for node in (product, addition, x_node, w_node, bias_node)):
        return None, "non_node_match_operand"

    if x_node.op != "placeholder" or w_node.op != "placeholder":
        return None, "mm_operands_must_be_placeholders"
    if bias_node.op != "get_attr":
        return None, "bias_must_be_a_get_attr"
    if len(analysis.placeholders) != 2:
        return None, "mm_bias_requires_two_placeholders"
    if len(analysis.get_attr_nodes) != 1:
        return None, "mm_bias_requires_one_get_attr"

    mm_target_ok = ((product.op == "call_function" and product.target in (torch.mm, torch.matmul, operator.matmul))
                    or (product.op == "call_method" and product.target in ("mm", "matmul")))
    if not mm_target_ok:
        return None, "matmul_target_not_supported"
    if product.kwargs or len(product.args) != 2:
        return None, "unexpected_mm_arguments"

    add_target_ok = ((addition.op == "call_function" and addition.target in (torch.add, operator.add))
                     or (addition.op == "call_method" and addition.target == "add"))
    if not add_target_ok:
        return None, "add_target_not_supported"
    if len(addition.args) != 2:
        return None, "unexpected_add_arguments"
    if "out" in addition.kwargs:
        return None, "add_out_keyword_present"
    alpha = addition.kwargs.get("alpha", 1)
    if type(alpha) not in (int, float) or alpha != 1:
        return None, "add_alpha_not_one"

    if tuple(product.users) != (addition, ):
        return None, "mm_output_has_other_users"

    placeholder_indices = {node: index for index, node in enumerate(analysis.placeholders)}
    x_index = placeholder_indices.get(x_node)
    w_index = placeholder_indices.get(w_node)
    if x_index is None or w_index is None or x_index == w_index:
        return None, "mm_placeholder_mapping_failed"

    if bias_node.target not in analysis.attr_targets:
        return None, "bias_attribute_not_lifted"
    bias_attr_index = analysis.attr_targets.index(bias_node.target)

    x = analysis.example_inputs[x_index]
    w = analysis.example_inputs[w_index]
    bias = analysis.attr_values[bias_attr_index]

    if not _dense_contiguous_float(x):
        return None, "x_not_dense_contiguous_float"
    if not _dense_contiguous_float(w):
        return None, "w_not_dense_contiguous_float"
    if not _dense_contiguous_float(bias):
        return None, "bias_not_dense_contiguous_float"
    if x.dim() != 2 or w.dim() != 2 or bias.dim() != 1:
        return None, "unexpected_operand_rank"
    if tuple(x.shape[1:]) != (int(w.shape[0]), ):
        return None, "inner_dimension_mismatch"
    if int(bias.shape[0]) != int(w.shape[1]):
        return None, "bias_width_mismatch"
    if x.dtype != w.dtype or x.dtype != bias.dtype:
        return None, "dtype_mismatch"
    if x.device != w.device or x.device != bias.device:
        return None, "device_mismatch"

    return _MmAddReuse(
        x_index=int(x_index),
        w_index=int(w_index),
        bias_attr_index=int(bias_attr_index),
        x_name=x_node.target,
        w_name=w_node.target,
        bias_target=bias_node.target,
        rows=int(x.shape[0]),
        width=int(w.shape[1]),
        matmul_name="mm" if product.target in (torch.mm, "mm") else "matmul",
        bias_first=addition.args[0] is bias_node,
    ), None


# ---------------------------------------------------------------------------
# Plan construction


def _find_small_matmul(analysis: _Analysis):
    return _find_matmul_epilogue(analysis, small=True)


def _find_matmul_epilogue(analysis: _Analysis, *, small: bool):
    """Recognize dense GEMM epilogues without reading tensor values.

    The small policy retains its bounds and diagnostic codes. The large policy
    shares the same dtype, layout, broadcast and precision requirements.
    """
    activations = {
        "matmul_bias": "none",
        "matmul_bias_relu": "relu",
        "matmul_bias_gelu_exact": "gelu_exact",
        "matmul_bias_gelu_tanh": "gelu_tanh",
        "matmul_bias_silu": "silu",
    }
    match = analysis.match
    if match.kind not in activations or len(match.inputs) != 3:
        return None, "not_small_matmul_template"
    # _analyze has already required the match to cover the entire graph. Bind
    # operands by FX identity, including reordered inputs and live attributes.
    bindings = {node: i for i, node in enumerate(analysis.placeholders)}
    for node in analysis.get_attr_nodes:
        bindings[node] = len(analysis.example_inputs) + analysis.attr_targets.index(node.target)
    if any(node not in bindings for node in match.inputs):
        return None, "matmul_operands_must_be_inputs_or_attrs"
    indices = tuple(bindings[node] for node in match.inputs)
    values = analysis.example_inputs + analysis.attr_values
    x, w, bias = (values[i] for i in indices)
    if not all(_dense_contiguous_float(t) for t in (x, w, bias)):
        return None, "small_matmul_requires_dense_contiguous"
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None, "small_matmul_dtype"
    if any(t.dtype != x.dtype or t.device != x.device for t in (w, bias)):
        return None, "small_matmul_dtype_or_device_mismatch"
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        return None, "small_matmul_rank_or_inner_dimension"
    m, k = x.shape
    n = w.shape[1]
    if min(m, n, k) <= 0 or (small and not (m <= 64 and n <= 1024 and k <= 128)):
        return None, "outside_small_matmul_bounds"
    if bias.ndim == 0:
        strides = (0, 0)
    elif bias.ndim == 1 and bias.shape[0] in (1, n):
        strides = (0, bias.stride(0) if bias.shape[0] == n else 0)
    elif bias.ndim == 2 and bias.shape[0] in (1, m) and bias.shape[1] in (1, n):
        strides = (bias.stride(0) if bias.shape[0] == m else 0, bias.stride(1) if bias.shape[1] == n else 0)
    else:
        return None, "small_matmul_bias_broadcast"
    if analysis.precision_state[1]:
        return None, "small_matmul_requires_tf32_disabled"
    if x.dtype != torch.float32:
        reduction = analysis.precision_state[3 if x.dtype == torch.float16 else 4]
        if reduction[0] if isinstance(reduction, tuple) else reduction:
            return None, "small_matmul_requires_full_precision_reduction"
    product, addition = match.nodes[:2]
    # Preserve addition order, including keyword binding accepted by the matcher.
    first = addition.args[0] if addition.args else addition.kwargs.get("input")
    return _SmallMatmul(indices, (m, n, k), strides, first is match.inputs[2], activations[match.kind]), None


def _small_matmul_device_supported(value):
    if value.device.type != "cuda":
        return False
    from triton.runtime import driver
    # P800 exposes CUDA tensors too, so tensor.device alone is insufficient.
    if type(driver.active).__module__ == "triton.backends.nvidia.driver":
        return torch.cuda.get_device_capability(value.device)[0] >= 8
    from ._p800_matmul import is_p800
    return is_p800(value)


def _large_matmul_device_supported(value):
    if value.device.type != "cuda":
        return False
    from triton.runtime import driver
    return (type(driver.active).__module__ == "triton.backends.nvidia.driver"
            and torch.cuda.get_device_capability(value.device)[0] >= 8)


def _make_large_matmul_plan(analysis, spec, fallback):
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import config

    options = dict(_LARGE_MATMUL_OPTIONS)
    with compile_lock, config.patch(options):
        plan = _make_compiler_plan(analysis, fallback, None)
    plan.plan_info.update(
        large_matmul_compiler_options=options,
        kernel_provider="native_inductor_codegen",
        shape=list(spec.shape),
        activation=spec.activation,
        rounding_boundaries=["matmul_output_dtype", "add_output_dtype"],
        fusion_policy="native_autotune_with_library_and_template_candidates",
    )
    return plan


def _single_inductor_kernel(sources):
    """Audit the generated wrapper, failing closed on unknown/multiple kernels.

    The admitted graph contains only one GEMM and pointwise epilogue. Inspect
    the compiler's declarations AND calls, not a requested kernel-count label.
    Device tests additionally verify the actual launch count with a profiler.
    """
    if len(sources) != 1:
        return None
    try:
        tree = ast.parse(sources[0])
    except SyntaxError:
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            target = node.value.func
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "async_compile" and target.attr == "triton"):
                names.update(n.id for n in node.targets if isinstance(n, ast.Name))
    launches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        fn = node.func
        if isinstance(fn.value, ast.Name):
            if fn.value.id == "extern_kernels" or (fn.value.id == "async_compile" and fn.attr == "multi_kernel"):
                return None
            if fn.value.id in names and fn.attr == "run":
                launches.append(fn.value.id)
    return launches[0] if len(names) == len(launches) == 1 else None


def _make_small_matmul_plan(analysis, spec, fallback):
    from torch._inductor import config
    from ._p800_matmul import is_p800, native_template_eligibility
    value = (analysis.example_inputs + analysis.attr_values)[spec.indices[0]]
    p800 = is_p800(value)
    options = _small_matmul_options()
    if p800:
        # The temporary NVIDIA heuristic override is not part of native graph
        # cache identity. Do not reuse a graph produced without this policy.
        options.update(fx_graph_cache=False, fx_graph_remote_cache=False)
    scope = native_template_eligibility(value.device) if p800 else nullcontext()
    with scope, config.patch(options):
        plan = _make_compiler_plan(analysis, fallback, None, prepare_launch=not p800, require_p800_kernel=p800)
    kernel_name = _single_inductor_kernel(plan.generated_sources)
    plan.plan_info.update(
        small_matmul_compiler_options=options,
        kernel_provider="native_inductor_codegen",
        shape=list(spec.shape),
        activation=spec.activation,
        rounding_boundaries=["matmul_output_dtype", "add_output_dtype"],
    )
    if kernel_name is not None:
        plan.kind = _KIND_SMALL_MATMUL
        plan.plan_info.update(kind=plan.kind, kernel_name=kernel_name, kernel_count=1)
    else:
        # Keep the correct compiled graph, but never claim a single-kernel plan
        # when this compiler version/layout produced a different wrapper.
        plan.plan_info["small_matmul_skipped_reason"] = "compiler_single_kernel_not_verified"
    return plan


def _placeholder_name(used: set, preferred: Any, fallback: str) -> str:
    base = preferred if isinstance(preferred, str) and preferred.isidentifier() else fallback
    candidate = base
    suffix = 0
    while candidate in used:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used.add(candidate)
    return candidate


def _lift_graph_module(
    gm: fx.GraphModule,
    placeholders: tuple,
    attr_targets: tuple,
) -> fx.GraphModule:
    """Copy ``gm`` into a graph with get_attr tensors as extra placeholders.

    The supplied ``gm`` is not modified.  All get_attr nodes targeting the same
    live attribute share one lifted placeholder.  The copied output is a
    one-element tuple, and node args/kwargs constants are copied verbatim.
    """
    graph = fx.Graph()
    used_names = set()
    env = {}

    for index, node in enumerate(placeholders):
        name = _placeholder_name(used_names, getattr(node, "target", None), f"_f06_input_{index}")
        env[node] = graph.placeholder(name)

    attr_placeholders = []
    target_to_placeholder = {}
    for index, target in enumerate(attr_targets):
        name = _placeholder_name(used_names, f"_f06_attr_{index}", f"_f06_attr_{index}")
        placeholder = graph.placeholder(name)
        attr_placeholders.append(placeholder)
        target_to_placeholder[target] = placeholder

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            env[node] = target_to_placeholder[node.target]

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":
            graph.output((env[node.args[0]], ))
            continue
        env[node] = graph.node_copy(node, lambda item: env[item])

    graph.lint()
    return fx.GraphModule(torch.nn.Module(), graph)


def _call_compile_fx_inner(aten_gm: fx.GraphModule, flat_inputs: tuple):
    # Native Inductor interface.  Compilation errors intentionally propagate.
    from torch._inductor.compile_fx import BoxedBool, compile_fx_inner

    return compile_fx_inner(
        aten_gm,
        list(flat_inputs),
        is_inference=True,
        cudagraphs=BoxedBool(False),
    )


def _compiled_runner(compiled: Any):

    def run(values: tuple) -> torch.Tensor:
        # CompiledFxGraph is a boxed callable: one list argument in, one
        # sequence of graph outputs out.  The lifted graph has one output.
        result = compiled(list(values))
        if isinstance(result, (tuple, list)):
            if len(result) != 1:
                raise RuntimeError("static inference compiled plan returned "
                                   f"{len(result)} outputs, expected 1")
            result = result[0]
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("static inference compiled plan did not return a Tensor: "
                               f"{type(result).__name__}")
        return result

    return run


def _base_plan_info(
    analysis: _Analysis,
    kind: str,
    fallback: Optional[Callable],
    generated_sources: list,
) -> dict:
    return {
        "kind":
        kind,
        "template":
        analysis.template.template_id,
        "match_kind":
        analysis.match.kind,
        "calls":
        tuple(node.name for node in analysis.match.nodes),
        "placeholders":
        tuple(node.target for node in analysis.placeholders),
        "input_count":
        len(analysis.example_inputs),
        "attr_targets":
        tuple(analysis.attr_targets),
        "inputs":
        tuple(spec.as_dict() for spec in analysis.input_specs),
        "attrs":
        tuple(spec.as_dict() for spec in analysis.attr_specs),
        "output_count":
        1,
        "inference_only":
        True,
        "training_flag_guarded":
        False,
        "fallback_supplied":
        fallback is not None,
        "generated_source_count":
        len(generated_sources),
        "precision_state":
        _precision_info(analysis.precision_state),
        "autocast":
        "required_disabled",
        "transform_modes":
        "required_none",
        "cuda_graphs":
        False,
        "inference_scope": ("operator-only FX graph; module.training is not consulted"),
        "guard_checks": (
            "call_arity",
            "keyword_arguments",
            "forward_code_identity",
            "forward_hooks",
            "autocast",
            "torch_function_mode",
            "dispatch_mode",
            "functorch_transform",
            "precision_state",
            "tensor_type",
            "shape",
            "stride",
            "storage_offset",
            "dtype",
            "device",
            "layout",
            "requires_grad",
            "conjugate_negative_view",
            "tensor_aliasing",
            "live_attr_lookup",
        ),
    }


def _trace_compiler_graph(lifted, flat, *, emulate_precision):
    from torch._inductor import config
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._dynamo.convert_frame import compile_lock

    # Native tracing annotates eager pointwise output boundaries. Enabling the
    # option only during lowering is too late: add -> LayerNorm would consume
    # an unrounded FP32 sum. Internal decomposition arithmetic stays unmarked.
    scope = config.patch({'emulate_precision_casts': True}) if emulate_precision else nullcontext()
    with compile_lock, scope:
        return make_fx(lifted, decomposition_table=select_decomp_table(), tracing_mode='fake',
                       _allow_non_fake_inputs=True)(*flat)


def _make_compiler_plan(
    analysis: _Analysis,
    fallback: Optional[Callable],
    bias_reuse_skipped_reason: Optional[str],
    *,
    prepare_launch: bool = False,
    require_p800_kernel: bool = False,
) -> StaticInferencePlan:
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import config

    # Eager-visible low-precision boundaries are a graph semantic contract,
    # independent of the device and of matrix/vector resource selection.
    # Keep the native option active before tracing and through lowering/cache
    # lookup; enabling it only after make_fx loses the boundary annotations.
    with compile_lock, config.patch({"emulate_precision_casts": True}):
        return _make_compiler_plan_with_precision(
            analysis,
            fallback,
            bias_reuse_skipped_reason,
            prepare_launch=prepare_launch,
            require_p800_kernel=require_p800_kernel,
        )


def _make_compiler_plan_with_precision(
    analysis: _Analysis,
    fallback: Optional[Callable],
    bias_reuse_skipped_reason: Optional[str],
    *,
    prepare_launch: bool,
    require_p800_kernel: bool,
) -> StaticInferencePlan:
    lifted = _lift_graph_module(analysis.gm, analysis.placeholders, analysis.attr_targets)
    flat = tuple(analysis.example_inputs) + tuple(analysis.attr_values)
    from ._p800_matmul import is_p800
    p800 = bool(flat) and flat[0].device.type == 'cuda' and is_p800(flat[0])
    aten_gm = _trace_compiler_graph(lifted, flat, emulate_precision=True)
    if not isinstance(aten_gm, fx.GraphModule):
        raise RuntimeError("make_fx did not return a GraphModule: "
                           f"{type(aten_gm).__name__}")
    aten_placeholders = tuple(node for node in aten_gm.graph.nodes if node.op == "placeholder")
    if len(aten_placeholders) != len(flat):
        raise RuntimeError("traced static inference graph placeholder count changed: "
                           f"{len(aten_placeholders)} != {len(flat)}")

    resource_info = None
    if p800:
        from . import _resource_codegen as codegen
        from torch._inductor import config
        resource = 'sdnn' if require_p800_kernel else 'cluster'
        with codegen.generated_resource(resource,
                                        scalar_reduction=analysis.template.template_id == 'mul_sum'), config.patch({
                                            'epilogue_fusion':
                                            True,
                                            'benchmark_epilogue_fusion':
                                            False,
                                            'pattern_matcher':
                                            False,
                                            'emulate_precision_casts':
                                            True,
                                            'size_asserts':
                                            False,
                                            'triton.cudagraphs':
                                            False,
                                        }):
            compiled = _call_compile_fx_inner(aten_gm, flat)
        audit = codegen.audit_wrapper(compiled.source_code, resource)
        proofs = codegen._bound_kernel_evidence(compiled, resource)
        resource_info = dict(resource=resource, wrapper_audit=audit,
                             scalar_reduction_layout=analysis.template.template_id == 'mul_sum', kernel_evidence=proofs,
                             source_code=compiled.source_code, external_calls=len(audit['external_call_targets']),
                             resource_verified=not audit['external_call_targets'] and bool(proofs))
        if not resource_info['resource_verified']:
            raise RuntimeError('P800 compiled resource or external computation is unverified')
    else:
        compiled = _call_compile_fx_inner(aten_gm, flat)
    if not callable(compiled):
        raise TypeError("compile_fx_inner result is not callable: "
                        f"{type(compiled).__name__}")

    source_code = getattr(compiled, "source_code", None)
    generated_sources = ([source_code] if isinstance(source_code, str) and source_code else [])

    plan_info = _base_plan_info(analysis, _KIND_INDUCTOR, fallback, generated_sources)
    plan_info["compiler"] = ("torch._inductor.compile_fx.compile_fx_inner")
    plan_info["tracer"] = ("torch.fx.experimental.proxy_tensor.make_fx")
    plan_info["precision_policy"] = "eager_materialization_boundaries"
    plan_info["decomposition_table"] = ("torch._inductor.decomposition.select_decomp_table")
    plan_info["graph_lifted_get_attrs"] = len(analysis.attr_targets)
    plan_info["traced_placeholders"] = len(aten_placeholders)
    if resource_info is not None:
        plan_info['p800_resource'] = resource_info
    if bias_reuse_skipped_reason is not None:
        plan_info["bias_reuse_skipped_reason"] = bias_reuse_skipped_reason

    if require_p800_kernel:
        from ._p800_matmul import inspect_bound_binary
        kernel_name = _single_inductor_kernel(generated_sources)
        if kernel_name is None:
            raise RuntimeError("P800 compiler did not produce a single GEMM epilogue kernel")
        plan_info["p800_binary"] = inspect_bound_binary(compiled, kernel_name)
        plan_info["prepared_launch"] = "native"
        plan_info["native_template_eligibility"] = "p800_scoped_sm_heuristic_override"

    if prepare_launch:
        kernel_name = _single_inductor_kernel(generated_sources)
        if kernel_name is not None:
            from ._inductor_launch import prepared_inductor_call
            compiled = prepared_inductor_call(compiled, kernel_name, plan_info)

    plan = StaticInferencePlan(
        kind=_KIND_INDUCTOR,
        runner=_compiled_runner(compiled),
        gm=analysis.gm,
        num_inputs=len(analysis.example_inputs),
        input_specs=analysis.input_specs,
        attr_targets=analysis.attr_targets,
        attr_specs=analysis.attr_specs,
        forward_code=analysis.forward_code,
        precision_state=analysis.precision_state,
        fallback=fallback,
        plan_info=plan_info,
        generated_sources=generated_sources,
        guard_examples=tuple(analysis.example_inputs) + tuple(analysis.attr_values),
    )
    # Retain compilation provenance for inspection/profiling, never checked in
    # the timed call. The public callable retains its ordinary runtime guards.
    plan._compiled_graph = compiled
    return plan


def _mm_add_sources(matmul_name="mm", bias_first=False) -> list:
    # Plain Python operator descriptions, not generated/compiled kernel source.
    return [
        f"python_operator: result = torch.{matmul_name}(x, w)",
        "python_operator: torch.add(bias, result, out=result)"
        if bias_first else "python_operator: torch.add(result, bias, out=result)",
        "python_operator: return result",
    ]


def _make_mm_add_plan(
    analysis: _Analysis,
    reuse: _MmAddReuse,
    fallback: Optional[Callable],
) -> StaticInferencePlan:
    num_inputs = len(analysis.example_inputs)
    x_index = reuse.x_index
    w_index = reuse.w_index
    bias_index = num_inputs + reuse.bias_attr_index
    matmul_op = torch.mm if reuse.matmul_name == "mm" else torch.matmul
    bias_first = reuse.bias_first

    def run(values: tuple) -> torch.Tensor:
        x = values[x_index]
        w = values[w_index]
        bias = values[bias_index]
        result = matmul_op(x, w)
        if bias_first:
            torch.add(bias, result, out=result)
        else:
            torch.add(result, bias, out=result)
        return result

    generated_sources = _mm_add_sources(reuse.matmul_name, bias_first)
    plan_info = _base_plan_info(analysis, _KIND_NATIVE_MM_ADD, fallback, generated_sources)
    plan_info.update({
        "source_kind":
        "native_python_operator_sequence",
        "operator_sequence": (
            f"torch.{reuse.matmul_name}(x, w)",
            "torch.add(bias, result, out=result)" if bias_first else "torch.add(result, bias, out=result)",
        ),
        "x_placeholder":
        reuse.x_name,
        "w_placeholder":
        reuse.w_name,
        "bias_attr_target":
        reuse.bias_target,
        "output_rows":
        reuse.rows,
        "output_width":
        reuse.width,
        "avoid_bias_materialization":
        True,
        "cuda_graphs":
        False,
    })

    return StaticInferencePlan(
        kind=_KIND_NATIVE_MM_ADD,
        runner=run,
        gm=analysis.gm,
        num_inputs=num_inputs,
        input_specs=analysis.input_specs,
        attr_targets=analysis.attr_targets,
        attr_specs=analysis.attr_specs,
        forward_code=analysis.forward_code,
        precision_state=analysis.precision_state,
        fallback=fallback,
        plan_info=plan_info,
        generated_sources=generated_sources,
        guard_examples=tuple(analysis.example_inputs) + tuple(analysis.attr_values),
    )


class StaticInferencePlan:
    """Callable steady-state plan with metadata/live-attribute guards.

    A guard miss invokes ``fallback(*args, **kwargs)``.  If no fallback was
    supplied, :class:`UnsupportedStaticInference` is raised.  Errors raised by
    the compiled graph or by the caller's fallback are never swallowed.
    """

    def __init__(
        self,
        *,
        kind: str,
        runner: Callable,
        gm: fx.GraphModule,
        num_inputs: int,
        input_specs: tuple,
        attr_targets: tuple,
        attr_specs: tuple,
        forward_code: Any,
        precision_state: tuple,
        fallback: Optional[Callable],
        plan_info: dict,
        generated_sources: list,
        guard_examples: tuple,
    ) -> None:
        self.kind = kind
        self.generated_sources = generated_sources
        self.plan_info = plan_info
        self._runner = runner
        self._gm = gm
        self._num_inputs = num_inputs
        self._input_specs = tuple(input_specs)
        self._flat_specs = self._input_specs + tuple(attr_specs)
        self._attr_targets = tuple(attr_targets)
        self._forward_code = forward_code
        self._precision_state = tuple(precision_state)
        self._fallback = fallback
        from torch._C._dynamo.guards import TensorGuards
        self._tensor_guard = TensorGuards(
            *guard_examples,
            dynamic_dims_sizes=[list(spec.shape) for spec in self._flat_specs],
            dynamic_dims_strides=[list(spec.stride) for spec in self._flat_specs],
        )
        self._tensor_names = [f"tensor_{i}" for i in range(len(guard_examples))]
        self._attr_getters = tuple(operator.attrgetter(target) for target in self._attr_targets)
        self._alias_pairs = tuple(
            (left, right) for left in range(len(guard_examples)) for right in range(left + 1, len(guard_examples)))
        self._is_alias_of = torch._C._is_alias_of
        self.fast_calls = 0
        self.fallback_calls = 0
        self.last_fallback_reason = None
        self.plan_info["tensor_guard"] = "native PyTorch TensorGuards plus storage/view/grad checks"

    def __call__(self, *args, **kwargs):
        if kwargs:
            return self._fallback_or_raise("keyword_arguments", args, kwargs)
        values, reason = self._prepare(args)
        if values is not None:
            self.fast_calls += 1
            return self._runner(values)
        return self._fallback_or_raise(reason, args, kwargs)

    def _prepare(self, args: tuple):
        if len(args) != self._num_inputs:
            return None, "call_arity_mismatch"
        forward = getattr(self._gm, "forward", None)
        if (getattr(forward, "__code__", None) is not self._forward_code
                or getattr(forward, "__self__", None) is not self._gm):
            return None, "forward_code_or_binding_changed"
        if _forward_hooks_active(self._gm):
            return None, "forward_hooks_active"
        state_reason = _mode_reason()
        if state_reason is not None:
            return None, state_reason
        if _precision_state() != self._precision_state:
            return None, "precision_state_changed"

        if self._attr_getters:
            attrs, reason = self._live_attrs()
            if attrs is None:
                return None, reason
            values = args + attrs
        else:
            values = args

        if not self._tensor_guard.check(*values, tensor_check_names=self._tensor_names):
            return None, "tensor_metadata_changed"
        grad_enabled = torch.is_grad_enabled()
        for value, spec in zip(values, self._flat_specs):
            if ((grad_enabled and value.requires_grad) or value.is_conj() or value.is_neg()
                    or value.storage_offset() != spec.storage_offset or value.layout != spec.layout):
                return None, "tensor_storage_or_view_changed"
        for left, right in self._alias_pairs:
            if self._is_alias_of(values[left], values[right]):
                return None, "tensor_aliasing"
        return values, None

    def _live_attrs(self):
        values = []
        for getter in self._attr_getters:
            try:
                value = getter(self._gm)
            except AttributeError:
                return None, "live_attribute_missing"
            values.append(value)
        return tuple(values), None

    def _fallback_or_raise(self, reason: Optional[str], args: tuple, kwargs: dict):
        self.fallback_calls += 1
        self.last_fallback_reason = reason
        if self._fallback is None:
            raise UnsupportedStaticInference("guard_miss_without_fallback", detail=reason)
        return self._fallback(*args, **kwargs)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(kind={self.kind!r}, "
                f"template={self.plan_info.get('template')!r})")


# ---------------------------------------------------------------------------
# Public factory


def _validate_fallback(fallback: Optional[Callable]) -> None:
    if fallback is not None and not callable(fallback):
        raise TypeError("fallback must be callable or None")


def compile_static(
    gm: fx.GraphModule,
    example_inputs,
    *,
    avoid_bias_materialization: bool = False,
    fallback: Optional[Callable] = None,
) -> StaticInferencePlan:
    """Build an opt-in static inference plan for one accepted FX template.

    Initial graph/mode violations raise :class:`UnsupportedStaticInference`.
    Native tracing/compiler failures are allowed to propagate.  On later guard
    misses the plan calls ``fallback(*args, **kwargs)``; the parent runtime shim
    owns that original ``torch.compile`` fallback path.
    """
    if type(avoid_bias_materialization) is not bool:
        raise TypeError("avoid_bias_materialization must be a bool")
    _validate_fallback(fallback)

    analysis = _analyze(gm, example_inputs)

    small, small_reason = _find_small_matmul(analysis)
    if small is not None:
        if _small_matmul_device_supported((analysis.example_inputs + analysis.attr_values)[small.indices[0]]):
            return _make_small_matmul_plan(analysis, small, fallback)
        small_reason = "small_matmul_requires_nvidia_sm80_or_p800"

    # Only activation epilogues outside the small bounds use this policy.
    # Preserve the existing bias-only reuse path and all non-NVIDIA behavior.
    if small_reason == "outside_small_matmul_bounds":
        large, _ = _find_matmul_epilogue(analysis, small=False)
        if large is not None and large.activation != "none":
            value = (analysis.example_inputs + analysis.attr_values)[large.indices[0]]
            if _large_matmul_device_supported(value):
                plan = _make_large_matmul_plan(analysis, large, fallback)
                plan.plan_info["small_matmul_skipped_reason"] = small_reason
                return plan

    bias_skip_reason = None
    if avoid_bias_materialization:
        reuse, reason = _find_mm_add_reuse(analysis)
        if reuse is not None:
            # Production gate: the reuse plan is for the CUDA/P800 path only.
            x = analysis.example_inputs[reuse.x_index]
            if x.device.type == "cuda":
                return _make_mm_add_plan(analysis, reuse, fallback)
            bias_skip_reason = "requires_cuda_device"
        else:
            bias_skip_reason = reason or "pattern_not_eligible"

    plan = _make_compiler_plan(analysis, fallback, bias_skip_reason)
    plan.plan_info["small_matmul_skipped_reason"] = small_reason
    return plan


def _compile_native_mm_add(
    gm: fx.GraphModule,
    example_inputs,
    fallback: Optional[Callable] = None,
    *,
    allow_non_cuda: bool = False,
) -> StaticInferencePlan:
    """Private test/integration hook for the native mm+add(out=) plan.

    Production callers use :func:`compile_static`; ``allow_non_cuda`` exists so
    CPU tests can exercise the identical guard and operator path without
    weakening the public CUDA gate.
    """
    if type(allow_non_cuda) is not bool:
        raise TypeError("allow_non_cuda must be a bool")
    _validate_fallback(fallback)

    analysis = _analyze(gm, example_inputs)
    reuse, reason = _find_mm_add_reuse(analysis)
    if reuse is None:
        raise UnsupportedStaticInference("bias_reuse_pattern_unsupported", detail=reason)
    x = analysis.example_inputs[reuse.x_index]
    if not allow_non_cuda and x.device.type != "cuda":
        raise UnsupportedStaticInference("bias_reuse_cuda_required", detail=str(x.device))
    return _make_mm_add_plan(analysis, reuse, fallback)
