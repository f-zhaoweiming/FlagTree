"""Structural matching only: never execute the model during planning."""

from dataclasses import dataclass
import math
import operator

import torch
from torch import fx
import torch.nn.functional as F


@dataclass(frozen=True)
class Match:
    kind: str
    nodes: tuple
    inputs: tuple
    # Immutable semantic options that are not tensor operands (for example the
    # LayerNorm epsilon). Tuning configurations can never override these.
    options: tuple = ()


@dataclass(frozen=True)
class FusionTemplate:
    template_id: str
    version: int
    priority: int
    matcher: object


class Registry:
    """Per-compilation template registry, independent of the kernel dispatcher."""

    def __init__(self, templates=()):
        self._templates = {}
        for template in templates:
            self.register(template)

    def register(self, template):
        if template.template_id in self._templates:
            raise ValueError(f"Duplicate fusion template: {template.template_id}")
        self._templates[template.template_id] = template

    @property
    def templates(self):
        return tuple(sorted(self._templates.values(), key=lambda item: (-item.priority, item.template_id)))


def _bind(node, names, defaults=None):
    if len(node.args) > len(names) or any(key not in names for key in node.kwargs):
        return None
    bound = dict(defaults or {})
    for name, value in zip(names, node.args):
        if name in node.kwargs:
            return None
        bound[name] = value
    bound.update(node.kwargs)
    return bound if all(name in bound for name in names) else None


def _pair(node, kind):
    if not isinstance(node, fx.Node):
        return None
    binary = operator.add if kind == "add" else operator.matmul
    functions = (torch.add,) if kind == "add" else (torch.matmul, torch.mm)
    methods = ("add",) if kind == "add" else ("matmul", "mm")
    if node.op == "call_function" and node.target is binary:
        values = node.args if len(node.args) == 2 and not node.kwargs else ()
    elif ((node.op == "call_function" and node.target in functions)
          or (node.op == "call_method" and node.target in methods)):
        if len(node.args) > 2 or (node.op == "call_method" and "input" in node.kwargs):
            return None
        other = "mat2" if node.target in (torch.mm, "mm") else "other"
        names = ("input", other, "alpha", "out") if kind == "add" else ("input", other, "out")
        args = _bind(node, names, {"alpha": 1, "out": None})
        if args is None or args["out"] is not None:
            return None
        if kind == "add" and (type(args["alpha"]) not in (int, float) or args["alpha"] != 1):
            return None
        values = (args["input"], args[other])
    else:
        return None
    return tuple(values) if len(values) == 2 and all(isinstance(x, fx.Node) for x in values) else None


def _mul_pair(node):
    if not isinstance(node, fx.Node):
        return None
    binary = operator.mul
    functions = (torch.mul,)
    methods = ("mul",)
    if node.op == "call_function" and node.target is binary:
        values = node.args if len(node.args) == 2 and not node.kwargs else ()
    elif ((node.op == "call_function" and node.target in functions)
          or (node.op == "call_method" and node.target in methods)):
        if len(node.args) > 3 or (node.op == "call_method" and "input" in node.kwargs):
            return None
        args = _bind(node, ("input", "other", "out"), {"out": None})
        if args is None or args["out"] is not None:
            return None
        values = (args["input"], args["other"])
    else:
        return None
    return tuple(values) if len(values) == 2 and all(isinstance(x, fx.Node) for x in values) else None


def _matmul_bias(node):
    added = _pair(node, "add")
    if added:
        for product, bias in (added, added[::-1]):
            operands = _pair(product, "matmul")
            if operands:
                return Match("matmul_bias", (product, node), operands + (bias,))
    return None


def _matmul_bias_relu(node):
    if node.op == "call_function" and node.target in (F.relu, torch.relu):
        names = ("input", "inplace") if node.target is F.relu else ("input",)
        args = _bind(node, names, {"inplace": False})
        if args is None or args["inplace"] is not False:
            return None
        source = args["input"]
    elif node.op == "call_method" and node.target == "relu" and len(node.args) == 1 and not node.kwargs:
        source = node.args[0]
    else:
        return None
    match = _matmul_bias(source)
    return Match("matmul_bias_relu", match.nodes + (node,), match.inputs) if match else None


def _add_softmax(node):
    if node.op == "call_function" and node.target is F.softmax:
        args = _bind(node, ("input", "dim", "_stacklevel", "dtype"),
                     {"dim": None, "_stacklevel": 3, "dtype": None})
        if args is not None and type(args["_stacklevel"]) is not int:
            return None
    elif node.op == "call_function" and node.target is torch.softmax:
        if len(node.args) > 3:
            return None
        args = _bind(node, ("input", "dim", "dtype", "out"), {"dtype": None, "out": None})
        if args is not None and args["out"] is not None:
            return None
    elif node.op == "call_method" and node.target == "softmax" and "input" not in node.kwargs:
        args = _bind(node, ("input", "dim", "dtype"), {"dtype": None})
    else:
        return None
    if args is None or type(args["dim"]) is not int or args["dim"] not in (-1, 1) or args["dtype"] is not None:
        return None
    pair = _pair(args["input"], "add")
    return Match("add_softmax", (args["input"], node), pair) if pair else None


def _activation(node, activation):
    """Parse a functional or Tensor-method activation without executing it."""
    if activation == "gelu":
        if node.op == "call_function" and node.target in (F.gelu,):
            args = _bind(node, ("input", "approximate"), {"approximate": "none"})
        elif node.op == "call_method" and node.target == "gelu":
            if "input" in node.kwargs or len(node.args) > 2:
                return None
            args = _bind(node, ("input", "approximate"), {"approximate": "none"})
        else:
            return None
        if args is None:
            return None
        approximate = args["approximate"]
        if approximate not in ("none", "tanh"):
            return None
        return args["input"], ("exact" if approximate == "none" else "tanh")
    if activation == "silu":
        functions = (F.silu, torch.nn.functional.silu)
        if node.op == "call_function" and node.target in functions:
            args = _bind(node, ("input", "inplace"), {"inplace": False})
        elif node.op == "call_method" and node.target == "silu":
            if "input" in node.kwargs or len(node.args) != 1 or node.kwargs:
                return None
            args = {"input": node.args[0], "inplace": False}
        else:
            return None
        if args is None or args["inplace"] is not False:
            return None
        return args["input"], ""
    return None


def _matmul_bias_activation(node, activation):
    parsed = _activation(node, activation)
    if parsed is None:
        return None
    source, variant = parsed
    match = _matmul_bias(source)
    if match is None:
        return None
    suffix = f"_{variant}" if variant else ""
    return Match(f"matmul_bias_{activation}{suffix}", match.nodes + (node,), match.inputs)


def _add_activation(node, activation):
    parsed = _activation(node, activation)
    if parsed is None:
        return None
    source, variant = parsed
    pair = _pair(source, "add")
    if pair is None:
        return None
    suffix = f"_{variant}" if variant else ""
    return Match(f"add_{activation}{suffix}", (source, node), pair)


def _add_layernorm(node):
    if node.op == "call_function" and node.target in (F.layer_norm,):
        args = _bind(node, ("input", "normalized_shape", "weight", "bias", "eps"),
                     {"weight": None, "bias": None, "eps": 1e-5})
    elif node.op == "call_function" and getattr(torch, "layer_norm", None) is node.target:
        args = _bind(node, ("input", "normalized_shape", "weight", "bias", "eps"),
                     {"weight": None, "bias": None, "eps": 1e-5})
    elif node.op == "call_method" and node.target == "layer_norm":
        if "input" in node.kwargs:
            return None
        args = _bind(node, ("input", "normalized_shape", "weight", "bias", "eps"),
                     {"weight": None, "bias": None, "eps": 1e-5})
    else:
        return None
    if args is None:
        return None
    shape = args["normalized_shape"]
    if isinstance(shape, int) and not isinstance(shape, bool):
        shape = (shape,)
    elif isinstance(shape, (tuple, list)) and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in shape):
        shape = tuple(shape)
    else:
        return None
    if len(shape) != 1:
        return None
    eps = args["eps"]
    if type(eps) not in (int, float) or not math.isfinite(eps):
        return None
    for name in ("weight", "bias"):
        if args[name] is not None and not isinstance(args[name], fx.Node):
            return None
    pair = _pair(args["input"], "add")
    if pair is None:
        return None
    weight, bias = args["weight"], args["bias"]
    inputs = pair + tuple(value for value in (weight, bias) if value is not None)
    kind = ("add_layernorm_affine" if weight is not None and bias is not None else
            "add_layernorm_weight" if weight is not None else
            "add_layernorm_bias" if bias is not None else "add_layernorm")
    return Match(kind, (args["input"], node), inputs,
                 (("eps", float(eps)), ("normalized_shape", shape)))


def _mul_sum(node):
    if node.op == "call_function" and node.target is torch.sum:
        args = _bind(node, ("input", "dim", "keepdim", "dtype", "out"),
                     {"dim": None, "keepdim": False, "dtype": None, "out": None})
    elif node.op == "call_method" and node.target == "sum":
        if "input" in node.kwargs or len(node.args) > 4:
            return None
        args = _bind(node, ("input", "dim", "keepdim", "dtype"),
                     {"dim": None, "keepdim": False, "dtype": None})
    else:
        return None
    if args is None or args["dtype"] is not None or args.get("out") is not None:
        return None
    if type(args["dim"]) is not int or args["dim"] not in (-1, 1):
        return None
    if type(args["keepdim"]) is not bool:
        return None
    pair = _mul_pair(args["input"])
    if pair is None:
        return None
    kind = "mul_sum_keepdim" if args["keepdim"] else "mul_sum"
    return Match(kind, (args["input"], node), pair)


def composition_candidates(graph_module, registry=None):
    """Return planning-only, constrained cross-call composition candidates.

    A candidate is a chain of adjacent FX operator calls selected by one
    template. This is graph-level composition, not kernel-call capture and not
    fusion across separate compiled-plan invocations.
    """
    selected, rejected = match_graph(graph_module, registry or default_registry())
    records = []
    for template, match in selected:
        records.append({
            "template": template.template_id,
            "kind": match.kind,
            "output": match.nodes[-1].name,
            "calls": tuple(node.name for node in match.nodes),
            "inputs": tuple(node.name for node in match.inputs),
            "cross_call": len(match.nodes) > 1,
            "options": dict(match.options),
            "eligibility": "adjacent_single_consumer_shared_static_inference_metadata",
        })
    return {"selected": records, "rejected": list(rejected),
            "scope": "single_fx_graph_operator_calls_only"}


def default_registry():
    return Registry((
        FusionTemplate("matmul_bias_gelu", 1, 40,
                       lambda node: _matmul_bias_activation(node, "gelu")),
        FusionTemplate("matmul_bias_silu", 1, 40,
                       lambda node: _matmul_bias_activation(node, "silu")),
        FusionTemplate("add_layernorm", 1, 35, _add_layernorm),
        FusionTemplate("add_gelu", 1, 34,
                       lambda node: _add_activation(node, "gelu")),
        FusionTemplate("add_silu", 1, 34,
                       lambda node: _add_activation(node, "silu")),
        FusionTemplate("mul_sum", 1, 33, _mul_sum),
        FusionTemplate("matmul_bias_relu", 1, 30, _matmul_bias_relu),
        FusionTemplate("matmul_bias", 1, 20, _matmul_bias),
        FusionTemplate("add_softmax", 1, 20, _add_softmax)))


def match_graph(graph_module, registry):
    computational = [n for n in graph_module.graph.nodes if n.op not in ("placeholder", "get_attr", "output")]
    positions = {node: index for index, node in enumerate(computational)}
    selected, rejected, occupied = [], [], set()
    for template in registry.templates:
        for node in computational:
            match = template.matcher(node)
            if match is None or occupied.intersection(match.nodes):
                continue
            indices = [positions[n] for n in match.nodes]
            reason = None
            if indices != list(range(indices[0], indices[0] + len(indices))):
                reason = "intervening_computation_or_side_effect"
            elif any(tuple(a.users) != (b,) for a, b in zip(match.nodes, match.nodes[1:])):
                reason = "intermediate_has_external_users"
            elif any(value in match.nodes for value in match.inputs):
                reason = "region_input_depends_on_internal_node"
            if reason:
                rejected.append({"template": template.template_id, "output": node.name, "reason": reason})
            else:
                occupied.update(match.nodes)
                selected.append((template, match))
    selected.sort(key=lambda item: positions[item[1].nodes[-1]])
    return tuple(selected), tuple(rejected)
