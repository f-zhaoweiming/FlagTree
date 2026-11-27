"""Offline FX subgraph construction and execution glue for resource planning.

This module is experimental scheduling/calibration glue only.  It copies and
executes existing pure FX nodes; it does not generate kernels, autotune, supply
runtime guards, or claim production certification.  A parent integration using
``torch.compile`` remains responsible for its native guards and full-plan
end-to-end numerical/performance validation.

All helpers reject autograd tensors. Offline example collection refuses device
graph capture; metadata and prepared execution support CPU, GPU and fake tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx
import torch.nn.functional as F
from torch.fx.node import map_arg

from torch.fx.immutable_collections import (
    immutable_dict as _FxImmutableDict,
    immutable_list as _FxImmutableList,
)

from ._resource_planner import PartitionPlan

try:
    from torch._ops import OpOverload as _OpOverload
except Exception:  # pragma: no cover - defensive for unusual Torch builds
    _OpOverload = ()  # type: ignore[assignment]

__all__ = [
    "PartitionExecutionError",
    "Region",
    "UnsupportedGraph",
    "build_all_regions",
    "build_region",
    "collect_example_values",
    "context_fingerprint",
    "execute_partition",
]

_MAX_COMPUTE_NODES = 16
_ALLOWED_METHODS = frozenset({
    "add",
    "mul",
    "div",
    "truediv",
    "matmul",
    "mm",
    "sum",
    "amax",
    "exp",
    "relu",
    "clone",
    "softmax",
})
_ALLOWED_ATEN_OVERLOADS = frozenset({
    ("add", "Tensor"),
    ("add", "Scalar"),
    ("mul", "Tensor"),
    ("mul", "Scalar"),
    ("div", "Tensor"),
    ("div", "Scalar"),
    ("matmul", "default"),
    ("mm", "default"),
    ("sum", "default"),
    ("sum", "dim_IntList"),
    ("amax", "default"),
    ("exp", "default"),
    ("relu", "default"),
    ("clone", "default"),
    ("softmax", "int"),
    ("gelu", "default"),
    ("silu", "default"),
    ("layer_norm", "default"),
})


def _optional_function(owner: Any, name: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError:
        return None


_C_NN = getattr(torch._C, "_nn", None)
_candidate_targets = [
    torch.add,
    torch.mul,
    torch.div,
    torch.matmul,
    torch.mm,
    torch.sum,
    torch.amax,
    torch.exp,
    torch.relu,
    torch.clone,
    torch.softmax,
    operator.add,
    operator.mul,
    operator.truediv,
    operator.matmul,
    F.relu,
    F.gelu,
    F.silu,
    F.softmax,
    F.layer_norm,
    getattr(torch, "truediv", None),
    _optional_function(_C_NN, "gelu"),
    _optional_function(_C_NN, "silu"),
    _optional_function(_C_NN, "layer_norm"),
]
_ALLOWED_FUNCTION_TARGETS = frozenset(target for target in _candidate_targets if target is not None)


class UnsupportedGraph(Exception):
    """The FX graph or an offline value is outside the supported pure contract."""


class PartitionExecutionError(ValueError):
    """A plan, compiled-region map, or region output is inconsistent."""


@dataclass(frozen=True)
class Region:
    """A standalone FX subgraph for one contiguous compute-node interval.

    ``gm`` has placeholders in ``input_names`` order and always returns a tuple
    of boundary values in ``output_names`` order.  Live get_attr values are not
    embedded; callers supply them as inputs on every invocation.
    """

    gm: torch.fx.GraphModule
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    start: int
    end: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class _GraphInfo:
    gm: torch.fx.GraphModule
    nodes: tuple[torch.fx.Node, ...]
    node_index: dict[torch.fx.Node, int]
    compute_nodes: tuple[torch.fx.Node, ...]
    compute_ids: tuple[str, ...]
    placeholders: tuple[torch.fx.Node, ...]
    get_attrs: tuple[torch.fx.Node, ...]
    output_node: torch.fx.Node
    output_structure: Any


class _RecordingInterpreter(torch.fx.Interpreter):

    def __init__(self, module: torch.fx.GraphModule) -> None:
        super().__init__(module)
        self.values: dict[str, Any] = {}

    def run_node(self, node: torch.fx.Node) -> Any:
        result = super().run_node(node)
        self.values[node.name] = result
        return result


def _aten_parts(target: Any) -> tuple[str, str] | None:
    if not isinstance(target, _OpOverload):
        return None
    schema = getattr(target, "_schema", None)
    full_name = getattr(schema, "name", None)
    overload_name = getattr(target, "__name__", None)
    if not isinstance(full_name, str) or not isinstance(overload_name, str):
        return None
    base_name = full_name.split("::")[-1]
    if "." in overload_name:
        overload = overload_name.split(".", 1)[1]
    else:
        overload = ""
    return base_name, overload


def _is_supported_function(target: Any) -> bool:
    aten = _aten_parts(target)
    if aten is not None:
        return aten in _ALLOWED_ATEN_OVERLOADS
    try:
        return target in _ALLOWED_FUNCTION_TARGETS
    except TypeError:
        return False


def _target_identity(target: Any) -> str:
    aten = _aten_parts(target)
    if aten is not None:
        module = getattr(target, "__module__", None)
        name = getattr(target, "__name__", None)
        if not isinstance(module, str) or not isinstance(name, str):
            raise UnsupportedGraph("aten target has no stable identity")
        return f"{module}.{name}"
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise UnsupportedGraph("call target has no stable module/qualname identity")
    return f"{module}.{qualname}"


def _validate_call_target(node: torch.fx.Node) -> None:
    kwargs = node.kwargs
    if not isinstance(kwargs, Mapping):
        raise UnsupportedGraph(f"{node.name!r}: call kwargs are not a mapping")
    if "out" in kwargs:
        raise UnsupportedGraph(f"{node.name!r}: out= call variants/mutations are unsupported")
    if "inplace" in kwargs and kwargs["inplace"] is not False:
        raise UnsupportedGraph(f"{node.name!r}: inplace=True mutations are unsupported")

    if node.op == "call_method":
        target = node.target
        if not isinstance(target, str):
            raise UnsupportedGraph(f"{node.name!r}: method target is not a string")
        if target.endswith("_") or target not in _ALLOWED_METHODS:
            raise UnsupportedGraph(f"{node.name!r}: unsupported or mutating tensor method {target!r}")
        return

    if node.op != "call_function":
        raise UnsupportedGraph(f"{node.name!r}: unsupported call node kind")
    if not callable(node.target):
        raise UnsupportedGraph(f"{node.name!r}: call target is not callable")

    aten = _aten_parts(node.target)
    target_name = getattr(node.target, "__name__", None)
    if aten is not None:
        base_name, overload = aten
        if base_name.endswith("_") or "out" in overload:
            raise UnsupportedGraph(f"{node.name!r}: inplace/out aten overloads are unsupported")
    elif isinstance(target_name, str) and target_name.endswith("_"):
        raise UnsupportedGraph(f"{node.name!r}: inplace/out function targets are unsupported")
    if not _is_supported_function(node.target):
        safe_target = (_target_identity(node.target) if aten is None else f"aten.{aten[0]}.{aten[1]}")
        raise UnsupportedGraph(f"{node.name!r}: unsupported or side-effecting call target "
                               f"{safe_target!r}")


def _analyze_graph(gm: torch.fx.GraphModule) -> _GraphInfo:
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError("gm must be a torch.fx.GraphModule")

    nodes = tuple(gm.graph.nodes)
    for node in nodes:
        if node.op == "call_module":
            raise UnsupportedGraph(f"{node.name!r}: call_module is unsupported; trace pure functions")
        if node.op not in (
                "placeholder",
                "get_attr",
                "call_function",
                "call_method",
                "output",
        ):
            raise UnsupportedGraph(f"{node.name!r}: unsupported FX node kind {node.op!r}")

    compute_nodes = tuple(node for node in nodes if node.op in ("call_function", "call_method"))
    if len(compute_nodes) > _MAX_COMPUTE_NODES:
        raise UnsupportedGraph(f"graph has {len(compute_nodes)} compute nodes; "
                               f"offline enumeration supports at most {_MAX_COMPUTE_NODES}")
    if not compute_nodes:
        raise UnsupportedGraph("graph has no call_function/call_method compute nodes")

    output_nodes = [node for node in nodes if node.op == "output"]
    if len(output_nodes) != 1 or nodes[-1].op != "output":
        raise UnsupportedGraph("graph must have exactly one final output node")
    output_node = output_nodes[0]
    if len(output_node.args) != 1:
        raise UnsupportedGraph("graph output must contain one output structure")

    for node in nodes:
        if node.op in ("call_function", "call_method"):
            _validate_call_target(node)
        elif node.op in ("placeholder", "get_attr"):
            if not isinstance(node.target, str) or not node.target:
                raise UnsupportedGraph(f"{node.name!r}: {node.op} target must be a nonempty string")

    for node in compute_nodes:
        if not node.users:
            raise UnsupportedGraph(f"{node.name!r}: dead compute nodes are unsupported; "
                                   "every compute node must reach an outside use or final output")

    node_index = {node: index for index, node in enumerate(nodes)}
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    get_attrs = tuple(node for node in nodes if node.op == "get_attr")
    return _GraphInfo(
        gm=gm,
        nodes=nodes,
        node_index=node_index,
        compute_nodes=compute_nodes,
        compute_ids=tuple(node.name for node in compute_nodes),
        placeholders=placeholders,
        get_attrs=get_attrs,
        output_node=output_node,
        output_structure=output_node.args[0],
    )


def _validate_interval(start: int, end: int, node_count: int) -> None:
    if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int)):
        raise TypeError("region start/end must be integer positional indices")
    if start < 0 or end > node_count:
        raise ValueError(f"region [{start}, {end}) is outside [0, {node_count})")
    if start >= end:
        raise ValueError(f"region [{start}, {end}) is empty; empty regions are unsupported")


def _boundary_outputs(
    interval_nodes: tuple[torch.fx.Node, ...],
    interval: frozenset[torch.fx.Node],
) -> tuple[torch.fx.Node, ...]:
    return tuple(node for node in interval_nodes if any(user not in interval for user in node.users))


def _build_region_from_info(info: _GraphInfo, start: int, end: int) -> Region:
    node_count = len(info.compute_nodes)
    _validate_interval(start, end, node_count)
    interval_nodes = info.compute_nodes[start:end]
    interval = frozenset(interval_nodes)
    if not interval:
        raise ValueError(f"region [{start}, {end}) has no compute nodes; empty regions are unsupported")

    external: set[torch.fx.Node] = set()

    def record_dependency(value: torch.fx.Node) -> None:
        if value not in interval:
            external.add(value)

    for node in interval_nodes:
        map_arg(node.args, record_dependency)
        map_arg(node.kwargs, record_dependency)
    external_nodes = tuple(sorted(external, key=lambda node: info.node_index[node]))

    graph = torch.fx.Graph()
    mapping: dict[torch.fx.Node, torch.fx.Node] = {}
    for dependency in external_nodes:
        mapping[dependency] = graph.placeholder(dependency.name)

    for original in interval_nodes:
        args = map_arg(original.args, lambda node: mapping[node])
        kwargs = map_arg(original.kwargs, lambda node: mapping[node])
        if original.op == "call_function":
            copied = graph.call_function(original.target, args=args, kwargs=kwargs)
        else:
            copied = graph.call_method(original.target, args=args, kwargs=kwargs)
        mapping[original] = copied

    boundary = _boundary_outputs(interval_nodes, interval)
    if not boundary:
        raise UnsupportedGraph(f"region [{start}, {end}) has no boundary outputs; "
                               "dead regions are unsupported")
    graph.output(tuple(mapping[node] for node in boundary))
    region_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return Region(
        gm=region_gm,
        input_names=tuple(node.name for node in external_nodes),
        output_names=tuple(node.name for node in boundary),
        start=start,
        end=end,
        node_ids=tuple(node.name for node in interval_nodes),
    )


def build_region(gm: torch.fx.GraphModule, start: int, end: int) -> Region:
    """Build the standalone subgraph for compute-node interval ``[start, end)``."""
    info = _analyze_graph(gm)
    return _build_region_from_info(info, start, end)


def build_all_regions(gm: torch.fx.GraphModule) -> tuple[Region, ...]:
    """Enumerate all nonempty contiguous intervals in stable ``(start, end)`` order."""
    info = _analyze_graph(gm)
    regions: list[Region] = []
    for start in range(len(info.compute_nodes)):
        for end in range(start + 1, len(info.compute_nodes) + 1):
            regions.append(_build_region_from_info(info, start, end))
    return tuple(regions)


def _fetch_attr(gm: torch.fx.GraphModule, target: str) -> Any:
    current: Any = gm
    for atom in target.split("."):
        try:
            current = getattr(current, atom)
        except AttributeError as exc:
            raise UnsupportedGraph(f"live get_attr target {target!r} is missing") from exc
    return current


def _validate_runtime_value(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise UnsupportedGraph(f"{label}: autograd tensors are unsupported; requires_grad=True")
        return
    if isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            _validate_runtime_value(key, f"{label}.key[{index}]")
            _validate_runtime_value(item, f"{label}.value[{index}]")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_runtime_value(item, f"{label}[{index}]")
        return
    if isinstance(
            value,
        (bool, int, float, complex, str, bytes, type(None), slice, torch.dtype, torch.device, torch.layout),
    ):
        return
    memory_format = getattr(torch, "memory_format", None)
    if memory_format is not None and isinstance(value, memory_format):
        return
    raise UnsupportedGraph(f"{label}: unsupported runtime value type {type(value).__name__}")


def _normalize_example_inputs(info: _GraphInfo, example_inputs: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(example_inputs, (str, bytes, torch.Tensor)) or not isinstance(example_inputs, Sequence):
        raise TypeError("example_inputs must be an outer sequence of placeholder values")
    values = tuple(example_inputs)
    if len(values) != len(info.placeholders):
        raise ValueError(f"expected {len(info.placeholders)} example inputs, got {len(values)}")
    for index, value in enumerate(values):
        _validate_runtime_value(value, f"example_inputs[{index}]")
    return values


def _live_attr_values(info: _GraphInfo) -> dict[torch.fx.Node, Any]:
    values: dict[torch.fx.Node, Any] = {}
    for node in info.get_attrs:
        value = _fetch_attr(info.gm, node.target)
        _validate_runtime_value(value, f"get_attr[{node.target}]")
        values[node] = value
    return values


def _validate_graph_tensor_literals(info: _GraphInfo) -> None:

    def scan(value: Any, label: str) -> None:
        if isinstance(value, torch.fx.Node):
            return
        if isinstance(value, torch.Tensor):
            _validate_runtime_value(value, label)
            return
        if isinstance(value, Mapping):
            for index, (key, item) in enumerate(value.items()):
                scan(key, f"{label}.key[{index}]")
                scan(item, f"{label}.value[{index}]")
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                scan(item, f"{label}[{index}]")
        # Non-tensor literals are target arguments, not executable graph data
        # sources; they are handled by target/argument preservation.

    for node in info.nodes:
        if node.op in ("call_function", "call_method"):
            scan(node.args, f"{node.name}.args")
            scan(node.kwargs, f"{node.name}.kwargs")
    scan(info.output_structure, "output")


def _external_value_map(info: _GraphInfo, external_values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(external_values, Mapping):
        raise TypeError("external_values must be a mapping of node name to value")
    known_names = {node.name for node in info.nodes}
    for name in external_values:
        if not isinstance(name, str):
            raise ValueError("external_values keys must be string node names")
        if name not in known_names:
            raise ValueError(f"external_values contains unknown node name {name!r}")
    return {node.name: external_values[node.name] for node in info.nodes if node.name in external_values}


def _capture_active(values):
    from torch.utils._pytree import tree_leaves
    real_gpu = any(
        isinstance(v, torch.Tensor) and v.device.type == "cuda" and getattr(v, "fake_mode", None) is None
        for v in tree_leaves(values))
    return real_gpu and torch.cuda.is_current_stream_capturing()


def collect_example_values(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any] | None = None,
    *,
    external_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record original FX node values offline, or accept an external metadata map.

    Offline execution validates the graph, all example values, and every live
    get_attr before invoking ``torch.fx.Interpreter`` on the original targets.
    If ``external_values`` is supplied, it takes precedence and no PyTorch or
    generated code is executed; ``example_inputs`` is then ignored.
    """
    info = _analyze_graph(gm)
    if external_values is not None:
        return _external_value_map(info, external_values)
    if example_inputs is None:
        raise ValueError("example_inputs are required when external_values is omitted")

    inputs = _normalize_example_inputs(info, example_inputs)
    attrs = _live_attr_values(info)
    _validate_graph_tensor_literals(info)
    if _capture_active((inputs, tuple(attrs.values()))):
        raise UnsupportedGraph("offline example collection is forbidden during graph capture")
    interpreter = _RecordingInterpreter(gm)
    with torch.no_grad():
        interpreter.run(*inputs)
    return dict(interpreter.values)


def _validate_plan(plan: Any, info: _GraphInfo) -> tuple[Any, ...]:
    if not isinstance(plan, PartitionPlan):
        raise PartitionExecutionError("plan must be an accepted resource_planner.PartitionPlan")
    context_key = plan.context_key
    if not isinstance(context_key, str) or not context_key.strip():
        raise PartitionExecutionError("plan context_key must be nonempty")
    scope = plan.validated_scope
    if scope.context_key != context_key:
        raise PartitionExecutionError("plan validated_scope context does not match plan context")
    if scope.start != 0 or scope.end != len(info.compute_nodes):
        raise PartitionExecutionError("plan validated_scope does not span the graph's compute nodes")
    if tuple(scope.node_ids) != info.compute_ids:
        raise PartitionExecutionError("plan node IDs do not match the graph compute-node order")

    regions = tuple(sorted(plan.regions, key=lambda region: (region.start, region.end)))
    if not regions:
        raise PartitionExecutionError("plan contains no regions")
    next_start = 0
    for region in regions:
        if region.context_key != context_key:
            raise PartitionExecutionError("every plan region context must match the plan context")
        if region.start != next_start or region.end <= region.start:
            raise PartitionExecutionError("plan regions must be nonempty, ordered, and contiguous")
        if region.end > len(info.compute_nodes):
            raise PartitionExecutionError("plan region extends beyond the graph's compute nodes")
        expected_ids = info.compute_ids[region.start:region.end]
        if tuple(region.node_ids) != expected_ids:
            raise PartitionExecutionError(f"plan region [{region.start}, {region.end}) node IDs do not "
                                          "match the graph")
        next_start = region.end
    if next_start != len(info.compute_nodes):
        raise PartitionExecutionError("plan does not fully cover the graph's compute nodes")
    return regions


def _thaw_fx_output_structure(value: Any) -> Any:
    """Use runtime-style list/dict literals instead of FX's immutable wrappers."""
    if isinstance(value, _FxImmutableDict):
        return {key: _thaw_fx_output_structure(item) for key, item in value.items()}
    if isinstance(value, _FxImmutableList):
        return [_thaw_fx_output_structure(item) for item in value]
    if type(value) is tuple:
        return tuple(_thaw_fx_output_structure(item) for item in value)
    return value


def execute_partition(
    gm: torch.fx.GraphModule,
    plan: PartitionPlan,
    compiled_regions: Mapping[tuple[int, int, str], Any],
    *inputs: Any,
) -> Any:
    """Execute a fully compiled contiguous partition using explicit region cuts.

    ``compiled_regions`` must contain one callable per plan region keyed by
    ``(start, end, resource)``.  Every callable is preflighted before execution;
    there is no original-target fallback.  The helper rechecks graph/plan node
    identity, coverage, and plan-internal context alignment, but it cannot
    recompute environment guards or provide runtime numerical validation.
    """
    info = _analyze_graph(gm)
    regions = _validate_plan(plan, info)
    if not isinstance(compiled_regions, Mapping):
        raise TypeError("compiled_regions must be a mapping")

    build_cache: dict[tuple[int, int], Region] = {}
    call_specs: list[tuple[Any, tuple[str, ...], tuple[str, ...], tuple[int, int, str]]] = []
    for plan_region in regions:
        region_key = (plan_region.start, plan_region.end)
        region = build_cache.get(region_key)
        if region is None:
            region = _build_region_from_info(info, plan_region.start, plan_region.end)
            build_cache[region_key] = region
        compiled_key = (
            plan_region.start,
            plan_region.end,
            plan_region.resource,
        )
        if compiled_key not in compiled_regions:
            raise PartitionExecutionError(f"missing compiled region for key {compiled_key!r}")
        compiled = compiled_regions[compiled_key]
        if not callable(compiled):
            raise PartitionExecutionError(f"compiled region for key {compiled_key!r} is not callable")
        call_specs.append((
            compiled,
            tuple(region.input_names),
            tuple(region.output_names),
            compiled_key,
        ))

    if len(inputs) != len(info.placeholders):
        raise PartitionExecutionError(f"expected {len(info.placeholders)} runtime inputs, got {len(inputs)}")
    for index, value in enumerate(inputs):
        _validate_runtime_value(value, f"inputs[{index}]")
    attr_values = _live_attr_values(info)
    _validate_graph_tensor_literals(info)

    node_by_name = {node.name: node for node in info.nodes}
    values: dict[torch.fx.Node, Any] = {}
    for placeholder, value in zip(info.placeholders, inputs):
        values[placeholder] = value
    values.update(attr_values)

    for compiled, input_names, output_names, compiled_key in call_specs:
        region_inputs = tuple(values[node_by_name[name]] for name in input_names)
        outputs = compiled(*region_inputs)
        if not isinstance(outputs, tuple):
            raise PartitionExecutionError(f"compiled region {compiled_key!r} must return a tuple")
        if len(outputs) != len(output_names):
            raise PartitionExecutionError(f"compiled region {compiled_key!r} returned {len(outputs)} outputs; "
                                          f"expected {len(output_names)}")
        for name, value in zip(output_names, outputs):
            values[node_by_name[name]] = value

    try:
        mapped_output = map_arg(info.output_structure, lambda node: values[node])
        return _thaw_fx_output_structure(mapped_output)
    except KeyError as exc:
        raise PartitionExecutionError("execution did not produce a value required by the final output") from exc


def _encode_tensor_metadata(value: torch.Tensor, path: str) -> dict[str, Any]:
    if value.requires_grad:
        raise UnsupportedGraph(f"{path}: autograd tensor metadata is unsupported")
    try:
        shape = [int(item) for item in value.shape]
        stride = [int(item) for item in value.stride()]
        storage_offset = int(value.storage_offset())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise UnsupportedGraph(f"{path}: tensor layout metadata is unsupported") from exc
    return {
        "kind": "tensor",
        "shape": shape,
        "stride": stride,
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "storage_offset": storage_offset,
    }


def _encode_value(
    value: Any,
    node_index: Mapping[torch.fx.Node, int] | None = None,
    path: str = "value",
) -> Any:
    if isinstance(value, torch.fx.Node):
        if node_index is None or value not in node_index:
            raise UnsupportedGraph(f"{path}: FX node reference outside graph")
        return {"kind": "node_ref", "index": node_index[value]}
    if isinstance(value, torch.Tensor):
        return _encode_tensor_metadata(value, path)
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            text = "nan"
        elif math.isinf(value):
            text = "inf" if value > 0 else "-inf"
        else:
            return {"kind": "float", "value": value}
        return {"kind": "float", "value": text}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _encode_value(value.real, node_index, path + ".real"),
            "imag": _encode_value(value.imag, node_index, path + ".imag"),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": value.hex()}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, torch.Size):
        return {
            "kind": "size",
            "items": [_encode_value(item, node_index, path + f"[{i}]") for i, item in enumerate(value)],
        }
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"kind": "device", "value": str(value)}
    if isinstance(value, torch.layout):
        return {"kind": "layout", "value": str(value)}
    memory_format = getattr(torch, "memory_format", None)
    if memory_format is not None and isinstance(value, memory_format):
        return {"kind": "memory_format", "value": str(value)}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _encode_value(value.start, node_index, path + ".start"),
            "stop": _encode_value(value.stop, node_index, path + ".stop"),
            "step": _encode_value(value.step, node_index, path + ".step"),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_value(item, node_index, path + f"[{i}]") for i, item in enumerate(value)],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_value(item, node_index, path + f"[{i}]") for i, item in enumerate(value)],
        }
    if isinstance(value, Mapping):
        encoded_items = []
        for index, (key, item) in enumerate(value.items()):
            encoded_key = _encode_value(key, node_index, path + f".key[{index}]")
            encoded_value = _encode_value(item, node_index, path + f".value[{index}]")
            encoded_items.append((encoded_key, encoded_value))
        encoded_items.sort(key=lambda pair: json.dumps(
            pair[0],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
        return {
            "kind": "mapping",
            "items": [{"key": key, "value": item} for key, item in encoded_items],
        }
    raise UnsupportedGraph(f"{path}: unsupported value type {type(value).__name__}")


def _encode_node(info: _GraphInfo, node: torch.fx.Node) -> dict[str, Any]:
    data: dict[str, Any] = {
        "op": node.op,
        "args": _encode_value(node.args, info.node_index, f"{node.name}.args"),
        "kwargs": _encode_value(node.kwargs, info.node_index, f"{node.name}.kwargs"),
    }
    if node.op == "call_function":
        data["target_kind"] = "function"
        data["target"] = _target_identity(node.target)
    elif node.op == "call_method":
        data["target_kind"] = "method"
        data["target"] = node.target
    elif node.op in ("placeholder", "get_attr"):
        data["target_kind"] = "data"
        data["target"] = node.target
    elif node.op == "output":
        data["target_kind"] = "output"
        data["target"] = "output"
    else:
        raise UnsupportedGraph(f"{node.name!r}: unsupported FX node kind")
    return data


def context_fingerprint(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    environment_key: str,
) -> str:
    """Hash canonical graph/input/live-attr metadata and an environment key.

    Tensor contents, data pointers, and storage contents are never read.  The
    parent supplies ``environment_key`` (hardware/compiler/precision); this
    function does not execute or validate the full partition numerically.
    """
    if not isinstance(environment_key, str) or not environment_key.strip():
        raise ValueError("environment_key must be a nonempty string")
    info = _analyze_graph(gm)
    inputs = _normalize_example_inputs(info, example_inputs)
    attr_values = _live_attr_values(info)

    input_payload = [{
        "placeholder": placeholder.name,
        "value": _encode_value(value, None, f"example_inputs[{index}]"),
    } for index, (placeholder, value) in enumerate(zip(info.placeholders, inputs))]
    attr_payload = [{
        "target": node.target,
        "value": _encode_value(attr_values[node], None, f"get_attr[{node.target}]"),
    } for node in info.get_attrs]
    payload = {
        "schema": "resource_fx_fingerprint_v1",
        "environment_key": environment_key,
        "nodes": [_encode_node(info, node) for node in info.nodes],
        "compute_node_indices": [info.node_index[node] for node in info.compute_nodes],
        "output_structure": _encode_value(info.output_structure, info.node_index, "output"),
        "inputs": input_payload,
        "get_attrs": attr_payload,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
