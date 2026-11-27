"""CPU contracts for P800 admission, isolation and fail-closed binary evidence."""
from contextlib import nullcontext
from contextvars import Context
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from test_graph_fusion_static import InferenceCPUTestBase, torch, _load_module


@unittest.skipIf(torch is None, "torch is required")
class P800RoutingTests(InferenceCPUTestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.p800 = _load_module("_p800_matmul")

    def test_backend_identity_uses_xpu_arch_not_cuda_alias(self):
        value = types.SimpleNamespace(device=torch.device("cuda:0"))
        target = types.SimpleNamespace(arch=3, warp_size=1)
        driver_type = type("XPUDriver", (), {
            "__module__": "triton.backends.xpu.driver",
            "get_current_target": lambda self: target,
        })
        runtime = types.ModuleType("triton.runtime")
        runtime.driver = types.SimpleNamespace(active=driver_type())
        with mock.patch.dict(sys.modules, {"triton.runtime": runtime}), \
             mock.patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()):
            self.assertTrue(self.p800.is_p800(value))
            self.assertTrue(self.inference._small_matmul_device_supported(value))
            target.arch = 4
            self.assertFalse(self.p800.is_p800(value))
            target.arch = 3
            target.warp_size = 32
            self.assertFalse(self.p800.is_p800(value))
            target.warp_size = 1
            runtime.driver.active = types.SimpleNamespace(get_current_target=lambda: target)
            self.assertFalse(self.p800.is_p800(value))
        self.assertFalse(self.p800.is_p800(torch.randn(1)))

    def test_sm_heuristic_override_is_context_and_device_local_and_restored(self):
        from torch._inductor import utils
        original = mock.Mock(return_value=False)
        with mock.patch.object(utils, "is_big_gpu", original):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                with self.p800.native_template_eligibility(torch.device("cuda:0")):
                    self.assertTrue(utils.is_big_gpu(0))
                    self.assertTrue(utils.is_big_gpu(torch.device("cuda:0")))
                    self.assertFalse(utils.is_big_gpu(torch.device("cuda:1")))
                    self.assertFalse(utils.is_big_gpu(torch.device("cpu")))
                    with mock.patch.object(torch.cuda, "current_device", return_value=0):
                        self.assertTrue(utils.is_big_gpu(torch.device("cuda")))
                    self.assertFalse(utils.is_big_gpu(1))
                    self.assertFalse(Context().run(utils.is_big_gpu, 0))
                    self.assertFalse(Context().run(utils.is_big_gpu, torch.device("cuda:0")))
                    with self.assertRaisesRegex(RuntimeError, "nested"):
                        with self.p800.native_template_eligibility(torch.device("cuda:0")):
                            pass
                    raise RuntimeError("compile failed")
            self.assertIs(utils.is_big_gpu, original)
            self.assertFalse(utils.is_big_gpu(0))
        self.assertIsNone(self.p800._DEVICE.get())

    def test_p800_candidate_uses_native_options_and_requires_single_kernel(self):
        gm = self._make_symbolic_mm_bias()
        args = self._sample_inputs()
        source = "kernel = async_compile.triton('kernel', '')\ndef call(args):\n    return kernel.run(*args)\n"
        capture = self._install_mock_compiler(source)
        from torch._inductor import config
        original = config.get_config_copy()
        proof = {"resource": "sdnn", "binary_hash": "test-fixture"}
        with mock.patch.object(self.inference, "_small_matmul_device_supported", return_value=True), \
             mock.patch.object(self.p800, "is_p800", return_value=True), \
             mock.patch.object(self.p800, "native_template_eligibility", side_effect=lambda _: nullcontext()), \
             mock.patch.object(self.p800, "inspect_bound_binary", return_value=proof) as inspect_binary:
            plan = self.inference.compile_static(gm, args)
        inspect_binary.assert_called_once_with(capture["compiled"][0], "kernel")
        self.assertEqual(plan.plan_info["p800_binary"], proof)
        self.assertEqual(plan.plan_info["prepared_launch"], "native")
        self.assertEqual(plan.plan_info["kernel_count"], 1)
        options = plan.plan_info["small_matmul_compiler_options"]
        self.assertEqual(options["max_autotune_gemm_backends"], "TRITON")
        self.assertFalse(options["pattern_matcher"])
        self.assertTrue(options["emulate_precision_casts"])
        self.assertFalse(options["fx_graph_cache"])
        self.assertEqual({k: original[k] for k in options}, {k: config.get_config_copy()[k] for k in options})
        torch.testing.assert_close(plan(*args), gm(*args))

    def test_p800_rejects_external_addmm_or_unverified_binary(self):
        gm = self._make_symbolic_mm_bias()
        args = self._sample_inputs()
        with mock.patch.object(self.inference, "_small_matmul_device_supported", return_value=True), \
             mock.patch.object(self.p800, "is_p800", return_value=True), \
             mock.patch.object(self.p800, "native_template_eligibility", side_effect=lambda _: nullcontext()):
            self._install_mock_compiler("def call(args):\n    return extern_kernels.addmm(*args)\n")
            with self.assertRaisesRegex(RuntimeError, "single GEMM epilogue"):
                self.inference.compile_static(gm, args)

    def test_bound_binary_evidence_is_actual_not_requested(self):
        from torch._inductor.codecache import PyCodeCache
        source = "native wrapper fixture"
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "wrapper.py"
            file.write_text(source)
            compiled = types.SimpleNamespace(cache_key="p800-unit-fixture", source_code=source)
            metadata = types.SimpleNamespace(arch=3, is_sdnn=True, name="native_kernel")
            binary = types.SimpleNamespace(metadata=metadata, hash="fixture", asm={"llir": "fixture"})
            launcher = types.FunctionType((lambda: None).__code__, {"bin": binary})
            tuner = types.SimpleNamespace(precompile=mock.Mock(), launchers=[launcher])
            module = types.SimpleNamespace(__file__=str(file), kernel=tuner)
            # P800's Torch 2.5 cache contract is modeled explicitly; newer
            # NVIDIA Torch releases no longer expose this class attribute.
            with mock.patch.object(PyCodeCache, "cache", {compiled.cache_key: module}, create=True):
                self.assertEqual(self.p800.inspect_bound_binary(compiled, "kernel")["resource"], "sdnn")
                metadata.is_sdnn = False
                self.assertEqual(self.p800.inspect_bound_binary(compiled, "kernel")["resource"], "cluster")
                metadata.arch = 4
                with self.assertRaisesRegex(RuntimeError, "unverified"):
                    self.p800.inspect_bound_binary(compiled, "kernel")
                metadata.arch = 3
                tuner.launchers.append(launcher)
                with self.assertRaisesRegex(RuntimeError, "one launcher"):
                    self.p800.inspect_bound_binary(compiled, "kernel")
            with mock.patch.object(PyCodeCache, "cache", {}, create=True):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    self.p800.inspect_bound_binary(compiled, "kernel")


if __name__ == "__main__":
    unittest.main()
