"""CPU tests for unchanged patterns and supported MatMul+Bias -> ReLU inlining."""

import operator
import unittest
from unittest import mock

import torch
import torch.fx

from triton.experimental.graph_fusion import (
    FusionTemplate,
    Registry,
    composition_candidates,
    cross_call_candidate,
    default_registry,
)


def _never_match(*args, **kwargs):
    return False


class _MatMulBias(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        return x @ self.weight + self.bias


class _Relu(torch.nn.Module):

    def forward(self, x):
        return torch.relu(x)


class _ReluMutating(torch.nn.Module):

    def forward(self, x):
        return x.relu_()


class _TupleOutput(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        return x @ self.weight + self.bias, x


class PatternRegistryTests(unittest.TestCase):

    def test_template_and_registry_contract(self):
        template = FusionTemplate("matmul-bias-relu", 1, 0, _never_match)
        registry = Registry(templates=())
        registry.register(template)
        self.assertIn(template, registry.templates)
        self.assertIsInstance(default_registry(), Registry)
        self.assertTrue(callable(composition_candidates))

    def test_default_registry_is_a_factory(self):
        first = default_registry()
        second = default_registry()
        self.assertIsInstance(first, Registry)
        self.assertIsInstance(second, Registry)


class CrossCallTests(unittest.TestCase):

    def test_supported_chain_preserves_targets_edges_and_live_replaced_parameters(self):
        stage0 = torch.fx.symbolic_trace(_MatMulBias())
        stage1 = torch.fx.symbolic_trace(_Relu())
        example_inputs = (torch.randn(2, 3), )

        candidate = cross_call_candidate([stage0, stage1], example_inputs)
        self.assertTrue(candidate.accepted, candidate.reason)
        combined = candidate.graph_module
        self.assertIsInstance(combined, torch.fx.GraphModule)

        targets = [node.target for node in combined.graph.nodes if node.op == "call_function"]
        self.assertIn(operator.matmul, targets)
        self.assertIn(operator.add, targets)
        self.assertIn(torch.relu, targets)

        add_node = next(n for n in combined.graph.nodes if n.op == "call_function" and n.target is operator.add)
        matmul_node = next(n for n in combined.graph.nodes if n.op == "call_function" and n.target is operator.matmul)
        self.assertTrue(any(arg is matmul_node for arg in add_node.args))

        replacement_weight = torch.nn.Parameter(torch.randn(3, 4))
        replacement_bias = torch.nn.Parameter(torch.randn(4))
        stage0.weight = replacement_weight
        stage0.bias = replacement_bias
        self.assertTrue(any(p is replacement_weight for p in combined.parameters()))
        self.assertTrue(any(p is replacement_bias for p in combined.parameters()))

        with torch.no_grad():
            expected = torch.relu(stage0(example_inputs[0]))
            got = combined(example_inputs[0])
        torch.testing.assert_close(got, expected)

        sentinel = object()
        with mock.patch("triton.experimental.fusion.compile", return_value=sentinel) as compile_mock:
            compiled = candidate.compile(example_inputs)
        self.assertIs(compiled, sentinel)
        compile_mock.assert_called_once()
        self.assertIs(compile_mock.call_args.args[0], combined)

    def test_mutating_second_stage_rejected_with_specific_reason(self):
        stage0 = torch.fx.symbolic_trace(_MatMulBias())
        stage1 = torch.fx.symbolic_trace(_ReluMutating())
        candidate = cross_call_candidate([stage0, stage1], (torch.randn(2, 3), ))
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.reason, "mutable_call_method")

    def test_multioutput_first_stage_rejected_with_specific_reason(self):
        stage0 = torch.fx.symbolic_trace(_TupleOutput())
        stage1 = torch.fx.symbolic_trace(_Relu())
        candidate = cross_call_candidate([stage0, stage1], (torch.randn(2, 3), ))
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.reason, "stage_must_return_one_tensor")


if __name__ == "__main__":
    unittest.main()
