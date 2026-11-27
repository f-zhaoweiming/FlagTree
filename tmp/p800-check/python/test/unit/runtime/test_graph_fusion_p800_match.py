"""CPU match-guard tests for the P800 whole-kernel replacements.

These pin the host-side admission rules that keep a loose FX match from
replacing an unsupported region with a kernel that computes something else:
the reduction must be the region output, on the last axis, and it must fold
exactly the supported inputs.
"""
import importlib.util
import pathlib
import sys
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - the device environment has torch
    torch = None

_HERE = pathlib.Path(__file__).resolve().parents[3] / "triton" / "experimental" / "graph_fusion"


def _load(name):
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_p800_match_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is required")
class P800MatchGuardTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reduce = _load("_p800_reduce")
        cls.dsa = _load("_p800_dsa")
        cls.flat = _load("_p800_flat")

    def _gm(self, build):
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        graph.output(build(graph, a, b))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    @staticmethod
    def _inputs(rows=8, cols=16):
        return (torch.randn(rows, cols), torch.randn(cols))

    @staticmethod
    def _dsa_inputs(rows=64, cols=16):
        return (torch.randn(rows, cols), torch.randn(cols))

    @staticmethod
    def _flat_inputs(rows=64, cols=16):
        return (torch.randn(rows, cols), torch.randn(cols))

    # -- _p800_reduce -----------------------------------------------------

    def test_reduce_softmax_requires_add_and_last_dim(self):
        def good(g, a, b):
            return g.call_function(torch.softmax, (g.call_function(torch.add, (a, b)), -1))

        spec = self.reduce.match(self._gm(good), self._inputs())
        self.assertEqual(spec["kind"], "add_softmax")

        def missing_add(g, a, b):
            return g.call_function(torch.softmax, (a, -1))

        self.assertIsNone(self.reduce.match(self._gm(missing_add), self._inputs()))

        def wrong_dim(g, a, b):
            return g.call_function(torch.softmax, (g.call_function(torch.add, (a, b)), 0))

        self.assertIsNone(self.reduce.match(self._gm(wrong_dim), self._inputs()))

    def test_reduce_layernorm_requires_add_and_matching_shape(self):
        import torch.nn.functional as F

        def good(g, a, b):
            return g.call_function(F.layer_norm, (g.call_function(torch.add, (a, b)), (16,)))

        self.assertEqual(self.reduce.match(self._gm(good), self._inputs())["kind"],
                         "add_layernorm")

        def wrong_shape(g, a, b):
            return g.call_function(F.layer_norm, (g.call_function(torch.add, (a, b)), (8,)))

        self.assertIsNone(self.reduce.match(self._gm(wrong_shape), self._inputs()))

    def test_reduce_mul_sum_requires_last_axis_and_terminal_sum(self):
        def good(g, a, b):
            product = g.call_function(torch.mul, (a, b))
            return g.call_function(torch.sum, (product, -1))

        self.assertEqual(self.reduce.match(self._gm(good), self._inputs())["kind"], "mul_sum")

        def wrong_dim(g, a, b):
            product = g.call_function(torch.mul, (a, b))
            return g.call_function(torch.sum, (product, 0))

        self.assertIsNone(self.reduce.match(self._gm(wrong_dim), self._inputs()))

        def extra_op(g, a, b):
            z = g.call_function(torch.add, (a, b))
            softmax = g.call_function(torch.softmax, (z, -1))
            return g.call_function(torch.add, (softmax, 1.0))

        self.assertIsNone(self.reduce.match(self._gm(extra_op), self._inputs()))

    # -- _p800_dsa --------------------------------------------------------

    def test_dsa_does_not_steal_a_softmax_denominator(self):
        def build(g, a, b):
            z = g.call_function(torch.add, (a, b))
            e = g.call_function(torch.exp, (z,))
            return g.call_function(torch.sum, (e, -1))

        self.assertIsNone(self.dsa.match(self._gm(build), self._dsa_inputs()))

    def test_dsa_rowsum_and_mulsum_require_last_axis_and_terminal_sum(self):
        def rowsum(g, a, b):
            return g.call_function(torch.sum, (a, -1))

        self.assertEqual(self.dsa.match(self._gm(rowsum), self._dsa_inputs())["kind"], "rowsum")

        def wrong_dim(g, a, b):
            return g.call_function(torch.sum, (a, 0))

        self.assertIsNone(self.dsa.match(self._gm(wrong_dim), self._dsa_inputs()))

        def mulsum(g, a, b):
            product = g.call_function(torch.mul, (a, b))
            return g.call_function(torch.sum, (product, -1))

        self.assertEqual(self.dsa.match(self._gm(mulsum), self._dsa_inputs())["kind"], "mulsum")

        def extra_op(g, a, b):
            product = g.call_function(torch.mul, (a, b))
            return g.call_function(torch.add, (product, 1.0))

        self.assertIsNone(self.dsa.match(self._gm(extra_op), self._dsa_inputs()))

    # -- _p800_flat -------------------------------------------------------

    def test_flat_requires_add_and_terminal_activation(self):
        import torch.nn.functional as F

        def good(g, a, b):
            return g.call_function(F.silu, (g.call_function(torch.add, (a, b)),))

        self.assertEqual(self.flat.match(self._gm(good), self._flat_inputs())["kind"], "silu")

        def missing_add(g, a, b):
            return g.call_function(F.silu, (a,))

        self.assertIsNone(self.flat.match(self._gm(missing_add), self._flat_inputs()))

        def extra_op(g, a, b):
            z = g.call_function(torch.add, (a, b))
            act = g.call_function(F.silu, (z,))
            return g.call_function(torch.add, (act, 1.0))

        self.assertIsNone(self.flat.match(self._gm(extra_op), self._flat_inputs()))

    def test_flat_rejects_unaligned_total(self):
        import torch.nn.functional as F

        def build(g, a, b):
            return g.call_function(F.silu, (g.call_function(torch.add, (a, b)),))

        self.assertIsNone(self.flat.match(self._gm(build), (torch.randn(3, 5), torch.randn(5))))


if __name__ == "__main__":
    unittest.main()