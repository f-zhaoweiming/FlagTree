"""Common compiler entry preserves eager low-precision boundaries."""
import unittest
from unittest import mock

from test_graph_fusion_static import InferenceCPUTestBase, _FakeCompiled, torch


@unittest.skipIf(torch is None, "torch is required")
class PrecisionBoundaryTests(InferenceCPUTestBase):

    def graph_and_inputs(self, dtype, normalization=False):

        class Product(torch.nn.Module):

            def forward(self, x, y):
                return (x * y).sum(dim=-1)

        class Normalization(torch.nn.Module):

            def forward(self, x, y):
                return torch.nn.functional.layer_norm(x + y, (128, ), eps=1e-3)

        gm = torch.fx.symbolic_trace(Normalization() if normalization else Product())
        return gm, (torch.randn(2, 128, dtype=dtype), torch.randn(2, 128, dtype=dtype))

    def test_public_static_entry_marks_boundaries_before_lowering(self):
        from torch._inductor import config
        from torch._dynamo.convert_frame import compile_lock
        for dtype in (torch.float16, torch.bfloat16):
            for normalization in (False, True):
                with self.subTest(dtype=dtype, normalization=normalization):
                    gm, args = self.graph_and_inputs(dtype, normalization)
                    boundary = torch.ops.aten.add.Tensor if normalization else torch.ops.aten.mul.Tensor
                    with config.patch({"emulate_precision_casts": False}):
                        original_hash = config.get_hash()

                        def lower(traced, flat):
                            self.assertTrue(compile_lock._is_owned())
                            self.assertTrue(config.emulate_precision_casts)
                            self.assertNotEqual(config.get_hash(), original_hash)
                            marked = [n for n in traced.graph.nodes if n.meta.get("low_precision_pointwise_barrier")]
                            # The final eager output may also carry a boundary.
                            # Only the producer feeding the next fused operator
                            # is the regression under test here.
                            self.assertEqual(sum(n.target == boundary for n in marked), 1)
                            if normalization:
                                internal = [n for n in traced.graph.nodes if n.target == torch.ops.aten.mul.Tensor]
                                self.assertTrue(internal)
                                self.assertTrue(all(n not in marked for n in internal))
                            return _FakeCompiled(traced, "mock_inductor_source")

                        with mock.patch.object(self.inference, "_call_compile_fx_inner", side_effect=lower):
                            plan = self.inference.compile_static(gm, args)
                        self.assertFalse(config.emulate_precision_casts)
                        self.assertEqual(config.get_hash(), original_hash)
                        self.assertEqual(plan.plan_info["precision_policy"], "eager_materialization_boundaries")
                        torch.testing.assert_close(plan(*args), gm(*args))

    def test_compilation_failure_restores_caller_config_and_cache_identity(self):
        from torch._inductor import config
        gm, args = self.graph_and_inputs(torch.bfloat16)
        for original in (False, True):
            for stage in ("_trace_compiler_graph", "_call_compile_fx_inner"):
                with self.subTest(original=original, stage=stage), \
                     config.patch({"emulate_precision_casts": original}):
                    original_hash = config.get_hash()

                    def fail(*unused, **kwargs):
                        self.assertTrue(config.emulate_precision_casts)
                        raise LookupError("precision compilation failed")

                    with mock.patch.object(self.inference, stage, side_effect=fail):
                        with self.assertRaisesRegex(LookupError, "precision compilation failed"):
                            self.inference.compile_static(gm, args)
                    self.assertEqual(config.emulate_precision_casts, original)
                    self.assertEqual(config.get_hash(), original_hash)


if __name__ == "__main__":
    unittest.main()
