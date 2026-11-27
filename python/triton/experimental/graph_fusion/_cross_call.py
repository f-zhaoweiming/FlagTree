"""Constrained composition of separately supplied FX GraphModules.

This is graph inlining, not kernel-call capture. Stages are pure FX graphs; the
first consumes the example inputs, every later stage consumes only the previous
single tensor output, and the combined chain must be covered by exactly one
supported fusion template. Names are bound to the original stage modules so
live parameter/buffer changes remain visible.
"""

from dataclasses import dataclass

import torch
from torch import fx

from ._patterns import default_registry, match_graph


@dataclass(frozen=True)
class CrossCallCandidate:
    accepted: bool
    stages: tuple
    graph_module: object = None
    template: str = None
    kind: str = None
    calls: tuple = ()
    reason: str = None
    provenance: tuple = ()

    def compile(self, example_inputs, **kwargs):
        if not self.accepted:
            raise ValueError(f"rejected cross-call candidate: {self.reason}")
        from ._runtime import compile as compile_plan
        return compile_plan(self.graph_module, example_inputs, **kwargs)

    def as_dict(self):
        return {"accepted": self.accepted, "stages": len(self.stages),
                "template": self.template, "kind": self.kind,
                "calls": self.calls, "reason": self.reason,
                "provenance": dict(self.provenance)}


def _reject(stages, reason):
    return CrossCallCandidate(False, tuple(stages), reason=reason,
                              provenance=(("source", "graph_inlining"),))


def _node_output(stage):
    output = next((n for n in stage.graph.nodes if n.op == "output"), None)
    if output is None or len(output.args) != 1 or not isinstance(output.args[0], fx.Node):
        return None
    node = output.args[0]
    if node.op in ("placeholder", "get_attr", "output"):
        return None
    return node


def _validate_stage(stage, index):
    for node in stage.graph.nodes:
        if node.op not in ("placeholder", "get_attr", "call_function",
                           "call_method", "output"):
            return f"unsupported_stage_op:{node.op}"
        if node.op == "call_module":
            return "call_module_not_allowed"
        target = node.target
        if node.op == "call_method" and str(target).endswith("_"):
            return "mutable_call_method"
        if node.op == "call_function":
            name = getattr(target, "__name__", str(target)).lower()
            if "inplace" in name or name in {
                    "set_", "copy_", "resize_", "add_", "mul_", "sub_",
                    "div_", "zero_", "fill_", "normal_", "uniform_"}:
                return "mutable_call_function"
        if node.op in ("call_function", "call_method"):
            if node.kwargs.get("inplace") is True:
                return "inplace_keyword"
    return None


def cross_call_candidate(stages, example_inputs):
    """Return an accepted inlined candidate or a rejected record.

    Structural validation only; no example value is executed. The accepted
    ``graph_module`` can be handed to :func:`graph_fusion.compile`.
    """
    stages = tuple(stages)
    inputs = tuple(example_inputs)
    if len(stages) < 2:
        return _reject(stages, "requires_at_least_two_stages")
    if not all(isinstance(stage, fx.GraphModule) for stage in stages):
        return _reject(stages, "stages_must_be_fx_graphmodules")
    if not inputs:
        return _reject(stages, "requires_example_inputs")

    first_placeholders = [n for n in stages[0].graph.nodes if n.op == "placeholder"]
    if len(first_placeholders) != len(inputs):
        return _reject(stages, "first_stage_input_arity_mismatch")
    outputs = []
    for index, stage in enumerate(stages):
        reason = _validate_stage(stage, index)
        if reason:
            return _reject(stages, reason)
        placeholders = [n for n in stage.graph.nodes if n.op == "placeholder"]
        if index and len(placeholders) != 1:
            return _reject(stages, "consumer_must_have_one_placeholder")
        output = _node_output(stage)
        if output is None:
            return _reject(stages, "stage_must_return_one_tensor")
        outputs.append(output)

    graph = fx.Graph()
    root = torch.nn.Module()
    prefixes = []
    previous = None
    for index, (stage, output) in enumerate(zip(stages, outputs)):
        prefix = f"_f06_stage{index}"
        prefixes.append(prefix)
        setattr(root, prefix, stage)
        placeholders = [n for n in stage.graph.nodes if n.op == "placeholder"]
        env = {}
        if index == 0:
            for number, placeholder in enumerate(placeholders):
                env[placeholder] = graph.placeholder(f"stage0_{number}")
        else:
            env[placeholders[0]] = previous
        stage_output = None
        for node in stage.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                stage_output = env[node.args[0]]
                continue
            if node.op == "get_attr":
                if not isinstance(node.target, str):
                    return _reject(stages, "non_string_get_attr")
                env[node] = graph.get_attr(f"{prefix}.{node.target}")
            else:
                env[node] = graph.node_copy(node, lambda n: env[n])
        if stage_output is None:
            return _reject(stages, "stage_output_not_mapped")
        previous = stage_output
    graph.output(previous)
    graph.lint()
    composed = fx.GraphModule(root, graph)
    # GraphModule copies get_attr hierarchies. Keep the supplied stage objects
    # so replacing a parameter/buffer remains visible, not only in-place edits.
    for prefix, stage in zip(prefixes, stages):
        setattr(composed, prefix, stage)

    selected, rejected = match_graph(composed, default_registry())
    if not selected:
        return CrossCallCandidate(False, stages, graph_module=composed,
                                  reason="no_supported_template_covers_chain",
                                  provenance=(("source", "graph_inlining"),
                                              ("rejected", tuple(rejected))))
    if len(selected) != 1:
        return CrossCallCandidate(False, stages, graph_module=composed,
                                  reason="multiple_fusion_regions",
                                  provenance=(("source", "graph_inlining"),))
    template, match = selected[0]
    computational = {n for n in composed.graph.nodes
                     if n.op not in ("placeholder", "get_attr", "output")}
    if not computational.issubset(set(match.nodes)):
        return CrossCallCandidate(False, stages, graph_module=composed,
                                  reason="uncovered_operator_in_chain",
                                  provenance=(("source", "graph_inlining"),))
    output_node = next(n for n in composed.graph.nodes if n.op == "output")
    if output_node.args != (match.nodes[-1],):
        return CrossCallCandidate(False, stages, graph_module=composed,
                                  reason="output_not_region_end",
                                  provenance=(("source", "graph_inlining"),))
    return CrossCallCandidate(
        True, stages, graph_module=composed, template=template.template_id,
        kind=match.kind, calls=tuple(node.name for node in match.nodes),
        provenance=(("source", "graph_inlining"),
                    ("stages", len(stages)),
                    ("options", tuple(sorted(dict(match.options).items())))))
