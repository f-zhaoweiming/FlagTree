"""Experimental tanh helper contracts and native compiler-path isolation."""
import ast
from contextvars import Context
import types
import unittest
from unittest import mock

from test_graph_fusion_static import InferenceCPUTestBase, _FakeCompiled, _load_module, torch


@unittest.skipIf(torch is None, "torch is required")
class CudaTanhTests(InferenceCPUTestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = _load_module("_cuda_tanh")

    def test_expression_survives_inductor_source_embedding(self):
        source = "def kernel(x):\n    return " + self.policy.tanh_expression("x") + "\n"
        wrapper = "source = '''" + source + "'''"
        embedded = ast.literal_eval(ast.parse(wrapper).body[0].value)
        parsed = ast.parse(embedded)
        call = parsed.body[0].body[0].value
        self.assertEqual(ast.literal_eval(call.args[0]).split(), self.policy.ASM.split())

    def test_nested_scope_context_cache_and_exception_restoration(self):
        from torch._inductor import config
        from torch._inductor.codegen.triton import TritonOverrides

        original = mock.Mock(return_value="original")
        with mock.patch("torch._inductor.codegen.triton.maybe_upcast_float32",
                        return_value=lambda fn: fn), \
             mock.patch.object(TritonOverrides, "tanh", staticmethod(original)), \
             config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": True}):
            descriptor = TritonOverrides.__dict__["tanh"]
            initial_hash = config.get_hash()
            with self.assertRaisesRegex(ValueError, "stop"):
                with self.policy.precise_tanh_codegen():
                    self.assertEqual(TritonOverrides.tanh("x"), self.policy.tanh_expression("x"))
                    self.assertEqual(Context().run(TritonOverrides.tanh, "x"), "original")
                    self.assertFalse(config.fx_graph_cache)
                    self.assertFalse(config.fx_graph_remote_cache)
                    with self.policy.precise_tanh_codegen():
                        self.assertEqual(TritonOverrides.tanh("x"), self.policy.tanh_expression("x"))
                    self.assertTrue(self.policy._ACTIVE.get())
                    raise ValueError("stop")
            self.assertIs(TritonOverrides.__dict__["tanh"], descriptor)
            self.assertEqual(config.get_hash(), initial_hash)
            self.assertFalse(self.policy._ACTIVE.get())

    def test_only_validated_target_library_gelu_and_dtypes(self):
        nvidia = type("Driver", (), {"__module__": "triton.backends.nvidia.driver"})()
        p800 = type("Driver", (), {"__module__": "triton.backends.xpu.driver"})()
        value = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"), dtype=torch.float32)
        with mock.patch("triton.runtime.driver", types.SimpleNamespace(active=nvidia)), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)) as capability, \
             mock.patch.object(self.policy, "_libdevice_sha256", return_value=self.policy.LIBDEVICE_SHA256) as digest:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                value.dtype = dtype
                for kind in ("matmul_bias_gelu_tanh", "add_gelu_tanh"):
                    self.assertTrue(self.policy.eligible(kind, [value]))
            for kind in ("matmul_bias_gelu", "add_gelu", "matmul_bias_relu", "softmax"):
                self.assertFalse(self.policy.eligible(kind, [value]))
            value.dtype = torch.float64
            self.assertFalse(self.policy.eligible("add_gelu_tanh", [value]))
            value.dtype = torch.float32
            capability.return_value = (9, 0)
            self.assertFalse(self.policy.eligible("add_gelu_tanh", [value]))
            capability.return_value = (8, 9)
            digest.return_value = "different-library"
            self.assertFalse(self.policy.eligible("add_gelu_tanh", [value]))
            digest.side_effect = FileNotFoundError("no libdevice")
            self.assertFalse(self.policy.eligible("add_gelu_tanh", [value]))
        with mock.patch("triton.runtime.driver", types.SimpleNamespace(active=p800)):
            self.assertFalse(self.policy.eligible("add_gelu_tanh", [value]))
        value.device.type = "cpu"
        self.assertFalse(self.policy.eligible("add_gelu_tanh", [value]))
        self.assertFalse(self.policy.eligible("add_gelu_tanh", []))

    def test_compiler_keeps_native_tanh_materialization_precision_and_caches(self):
        from torch._inductor import config
        from torch._dynamo.convert_frame import compile_lock
        from torch._inductor.codegen.triton import TritonOverrides

        class AddGelu(torch.nn.Module):

            def forward(self, x, b):
                return torch.nn.functional.gelu(x + b, approximate="tanh")

        gm = torch.fx.symbolic_trace(AddGelu())
        args = (torch.randn(2, 128, dtype=torch.bfloat16), torch.randn(128, dtype=torch.bfloat16))
        descriptor = TritonOverrides.__dict__["tanh"]
        with config.patch({"emulate_precision_casts": False, "fx_graph_cache": True}), \
             mock.patch.object(self.policy, "eligible", side_effect=AssertionError("experimental policy used")):

            def lower(traced, values):
                self.assertTrue(compile_lock._is_owned())
                self.assertTrue(config.emulate_precision_casts)
                self.assertTrue(config.fx_graph_cache)
                self.assertFalse(self.policy._ACTIVE.get())
                self.assertIs(TritonOverrides.__dict__["tanh"], descriptor)
                self.assertTrue(
                    any(n.target == torch.ops.aten.add.Tensor and n.meta.get("low_precision_pointwise_barrier")
                        for n in traced.graph.nodes))
                return _FakeCompiled(traced, "mock_inductor_source")

            with mock.patch.object(self.inference, "_call_compile_fx_inner", side_effect=lower):
                plan = self.inference.compile_static(gm, args)
            self.assertNotIn("activation_math_policy", plan.plan_info)
            self.assertNotIn("tanh_bitwise_equivalent", plan.plan_info)
            self.assertNotIn("tanh_validation", plan.plan_info)
            self.assertFalse(config.emulate_precision_casts)
            self.assertTrue(config.fx_graph_cache)
            self.assertFalse(self.policy._ACTIVE.get())
            self.assertIs(TritonOverrides.__dict__["tanh"], descriptor)
            with mock.patch.object(self.inference, "_call_compile_fx_inner", side_effect=LookupError("failed")):
                with self.assertRaises(LookupError):
                    self.inference.compile_static(gm, args)
            self.assertFalse(config.emulate_precision_casts)
            self.assertTrue(config.fx_graph_cache)
            self.assertFalse(self.policy._ACTIVE.get())


if __name__ == "__main__":
    unittest.main()
