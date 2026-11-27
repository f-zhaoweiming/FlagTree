"""Explicit GPU acceptance: F06_RUN_CUDA_TESTS=1 python -m unittest ... .

Ordinary CPU discovery skips this suite. Run on an otherwise idle NVIDIA GPU;
the report benchmark and its occupancy monitor provide that isolation locally.
"""
import os
import inspect
import unittest
from unittest import mock

import torch
import triton
from triton.experimental.graph_fusion import compile as graph_compile


class Matrix(torch.nn.Module):

    def __init__(self, activation="none", bias_first=False):
        super().__init__()
        self.activation = activation
        self.bias_first = bias_first

    def forward(self, x, w, bias=None):
        product = x @ w
        value = bias + product if self.bias_first else product + bias
        if self.activation == "relu":
            return torch.relu(value)
        if self.activation.startswith("gelu"):
            return torch.nn.functional.gelu(value, approximate="tanh" if self.activation == "gelu_tanh" else "none")
        if self.activation == "silu":
            return torch.nn.functional.silu(value)
        return value


@unittest.skipUnless(os.environ.get("F06_RUN_CUDA_TESTS") == "1", "explicit idle-GPU acceptance only")
class SmallMatmulCUDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from triton.runtime import driver
        if not torch.cuda.is_available() or type(driver.active).__module__ != "triton.backends.nvidia.driver":
            raise unittest.SkipTest("NVIDIA CUDA required")
        if torch.cuda.get_device_capability()[0] < 8:
            raise unittest.SkipTest("SM80+ required")
        cls.original_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.original_reductions = (torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
                                   torch._C._get_cublas_allow_bf16_reduced_precision_reduction())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.manual_seed(61)

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.original_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = cls.original_reductions[0]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = cls.original_reductions[1]

    def make(self, dtype, shape, activation="none", bias_shape=None, bias_first=False):
        m, n, k = shape
        gm = torch.fx.symbolic_trace(Matrix(activation, bias_first)).cuda()
        bias_shape = (n, ) if bias_shape is None else bias_shape
        xs = (torch.randn(m, k, dtype=dtype, device="cuda"), torch.randn(k, n, dtype=dtype, device="cuda"),
              torch.randn(bias_shape, dtype=dtype, device="cuda"))
        plan = graph_compile(gm, xs, static_inference=True, fullgraph=True, dynamic=False)
        self.assertEqual(plan.kind, "inductor_small_matmul_single_kernel")
        self.assertEqual(plan.plan_info["kernel_provider"], "native_inductor_codegen")
        return gm, xs, plan

    def close(self, actual, expected, dtype):
        tol = {torch.float32: 3e-4, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
        torch.testing.assert_close(actual, expected, atol=tol, rtol=3e-4 if dtype == torch.float32 else 3e-3,
                                   equal_nan=True)

    @torch.no_grad()
    def test_native_tanh_for_matrix_and_pointwise_gelu(self):
        from torch._inductor import config
        from torch._inductor.codegen.triton import TritonOverrides

        class AddGelu(torch.nn.Module):

            def forward(self, x, b):
                return torch.nn.functional.gelu(x + b, approximate="tanh")

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for m, n in ((16, 128), (32, 256), (64, 1024)):
                gm, xs, plan = self.make(dtype, (m, n, 64), "gelu_tanh")
                pointwise = torch.fx.symbolic_trace(AddGelu()).cuda()
                args = (torch.randn(m, n, device="cuda", dtype=dtype), torch.randn(n, device="cuda", dtype=dtype))
                pointwise_plan = graph_compile(pointwise, args, static_inference=True, fullgraph=True, dynamic=False)
                for graph, inputs, compiled in ((gm, xs, plan), (pointwise, args, pointwise_plan)):
                    with self.subTest(dtype=dtype, shape=(m, n), kind=compiled.kind):
                        source = "\n".join(compiled.generated_sources)
                        self.assertNotIn("activation_math_policy", compiled.plan_info)
                        self.assertNotIn("tanh_validation", compiled.plan_info)
                        self.assertIn("libdevice.tanh(", source)
                        self.assertNotIn("use_small", source)
                        self.assertNotIn("tl.exp(2.0", source)
                        self.close(compiled(*inputs), graph(*inputs), dtype)
                        self.assertEqual(compiled.fallback_calls, 0)
                        # Recompilation with the native FX cache retains tanh
                        # and its original code generator.
                        descriptor = TritonOverrides.__dict__["tanh"]
                        with config.patch({"fx_graph_cache": True}):
                            native = graph_compile(graph, inputs, static_inference=True, fullgraph=True, dynamic=False)
                        self.assertIs(TritonOverrides.__dict__["tanh"], descriptor)
                        self.assertIn("libdevice.tanh(", "\n".join(native.generated_sources))
                        self.assertNotIn("activation_math_policy", native.plan_info)
                        actual, original = compiled(*inputs), native(*inputs)
                        self.close(actual, original, dtype)
                        if graph is pointwise:
                            self.assertTrue(torch.equal(actual.view(torch.uint8), original.view(torch.uint8)))
                args[0].zero_()
                args[1].zero_()
                args[1][:7] = torch.tensor([
                    float("nan"),
                    float("inf"), -float("inf"), 0., -0.,
                    torch.finfo(dtype).tiny, -torch.finfo(dtype).tiny
                ], device="cuda", dtype=dtype)
                got, expected = pointwise_plan(*args), pointwise(*args)
                self.close(got, expected, dtype)
                self.assertTrue(torch.equal(torch.isnan(got), torch.isnan(expected)))
                self.assertTrue(torch.equal(torch.isinf(got), torch.isinf(expected)))

    @torch.no_grad()
    def test_report_shapes_all_epilogues_one_launch_and_fresh_outputs(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((16, 128, 64), (32, 256, 64), (64, 1024, 64)):
                for act in ("none", "relu", "gelu_exact", "gelu_tanh", "silu"):
                    with self.subTest(dtype=dtype, shape=shape, activation=act):
                        gm, xs, plan = self.make(dtype, shape, act)
                        expected = gm(*xs)
                        held = [plan(*xs) for _ in range(3)]
                        self.assertEqual(plan.plan_info.get("prepared_launch"), "enabled", plan.plan_info)
                        self.assertEqual(len({t.data_ptr() for t in held}), 3)
                        for value in held:
                            self.close(value, expected, dtype)
                            self.assertTrue(all(value.data_ptr() != t.data_ptr() for t in xs))
                        changed = tuple(t + 0.015 for t in xs)
                        self.close(plan(*changed), gm(*changed), dtype)
                        for value in held:
                            self.close(value, expected, dtype)
                        torch.cuda.synchronize()
                        with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
                                            ]) as prof:
                            plan(*xs)
                            torch.cuda.synchronize()
                        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        self.assertEqual(len(events), 1)
                        self.assertTrue(events[0].name.startswith("triton_"), events[0].name)
                        self.assertEqual(events[0].name, plan.plan_info["kernel_name"])
                        self.assertTrue(any("Call CompiledFxGraph" in e.name for e in prof.events()))
                        self.assertEqual(plan.fallback_calls, 0)

    @torch.no_grad()
    def test_tails_bounds_bias_broadcast_and_operand_order(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((1, 1, 1), (7, 33, 17), (31, 255, 63), (63, 1023, 65), (64, 1024, 128)):
                m, n, _ = shape
                for bias_shape in ((), (1, ), (n, ), (1, n), (m, 1), (m, n)):
                    with self.subTest(dtype=dtype, shape=shape, bias_shape=bias_shape):
                        gm, xs, plan = self.make(dtype, shape, "relu", bias_shape, bias_first=True)
                        self.close(plan(*xs), gm(*xs), dtype)

    @torch.no_grad()
    def test_special_values_and_materialized_low_precision_boundaries(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for act in ("none", "relu", "gelu_exact", "gelu_tanh", "silu"):
                gm, xs, plan = self.make(dtype, (16, 128, 64), act)
                x, w, b = xs
                x.zero_()
                w.zero_()
                values = [
                    float("nan"),
                    float("inf"), -float("inf"), 0., -0.,
                    torch.finfo(dtype).tiny, -torch.finfo(dtype).tiny,
                    torch.finfo(dtype).tiny / 2,
                    torch.finfo(dtype).max
                ]
                b.zero_()
                b[:len(values)] = torch.tensor(values, dtype=dtype, device="cuda")
                got, expected = plan(*xs), gm(*xs)
                self.assertTrue(torch.equal(torch.isnan(got), torch.isnan(expected)))
                self.assertTrue(torch.equal(torch.isinf(got), torch.isinf(expected)))
                self.assertTrue(
                    torch.equal(
                        torch.signbit(got)[torch.isinf(expected)],
                        torch.signbit(expected)[torch.isinf(expected)]))
                zeros = expected == 0
                self.assertTrue(torch.equal(torch.signbit(got)[zeros], torch.signbit(expected)[zeros]))
                self.close(got, expected, dtype)
                # Propagate nonfinite matrix operands too, including 0*inf.
                b.zero_()
                for value in (float("nan"), float("inf"), -float("inf")):
                    x.zero_()
                    w.fill_(1)
                    x[0, 0] = value
                    self.close(plan(*xs), gm(*xs), dtype)
                # One-hot mm has an exact accumulation. Test lowp mm and add
                # boundaries separately from differences in GEMM reduction order.
                x.zero_()
                w.normal_()
                x[:, 0] = 0.3
                b.normal_()
                self.close(plan(*xs), gm(*xs), dtype)

    @torch.no_grad()
    def test_stream_and_dynamic_launch_hooks(self):
        gm, xs, plan = self.make(torch.float32, (32, 256, 64), "relu")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        seen = []
        exited = []

        def hook(metadata):
            seen.append(metadata.get())

        def exit_hook(metadata):
            exited.append(metadata.get())

        triton.knobs.runtime.launch_enter_hook.add(hook)
        triton.knobs.runtime.launch_exit_hook.add(exit_hook)
        try:
            with torch.cuda.stream(stream):
                got = plan(*xs)
        finally:
            triton.knobs.runtime.launch_enter_hook.remove(hook)
            triton.knobs.runtime.launch_exit_hook.remove(exit_hook)
        stream.synchronize()
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(exited), 1)
        self.assertEqual(seen[0]["stream"], stream.cuda_stream)
        self.close(got, gm(*xs), torch.float32)
        self.close(plan(*xs), gm(*xs), torch.float32)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(exited), 1)

    @torch.no_grad()
    def test_prepared_launch_preserves_alignment_repair(self):
        gm, xs, plan = self.make(torch.float32, (16, 128, 64), "relu")
        plan(*xs)
        self.assertEqual(plan.plan_info["prepared_launch"], "enabled", plan.plan_info)
        backing = torch.empty(xs[0].numel() + 1, device="cuda", dtype=torch.float32)

        class External:
            __cuda_array_interface__ = dict(
                shape=tuple(xs[0].shape),
                strides=None,
                typestr="<f4",
                version=3,
                data=(backing.data_ptr() + 4, False),
            )

        unaligned = torch.as_tensor(External(), device="cuda")
        self.assertEqual(unaligned.storage_offset(), 0)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        unaligned.copy_(xs[0])
        values = (unaligned, *xs[1:])
        self.close(plan(*values), gm(*values), torch.float32)
        self.assertEqual(plan.fallback_calls, 0)

    @torch.no_grad()
    def test_prepared_launch_keeps_native_instrumentation_and_invalidation(self):
        from torch._inductor import debug
        from torch._dynamo.utils import get_runtime_metrics_context
        gm, xs, plan = self.make(torch.float32, (16, 128, 64), "relu")
        plan(*xs)
        self.assertEqual(plan.plan_info["prepared_launch"], "enabled", plan.plan_info)
        adapter = inspect.getclosurevars(plan._runner).nonlocals["compiled"]
        compiled = inspect.getclosurevars(adapter).nonlocals["compiled"]
        original = compiled.current_callable
        with mock.patch.object(compiled, "current_callable", wraps=original) as changed:
            self.close(plan(*xs), gm(*xs), torch.float32)
            changed.assert_called_once()
        order = []
        with mock.patch.object(debug, "RECORD_GRAPH_EXECUTION", True), \
                mock.patch.object(debug, "GRAPH_EXECUTION_ORDER", order):
            self.close(plan(*xs), gm(*xs), torch.float32)
        self.assertEqual(len(order), 1)
        context = get_runtime_metrics_context()
        with mock.patch.object(context, "_metrics", {"test_pending": 1}), \
                mock.patch.object(context, "finish") as finish:
            self.close(plan(*xs), gm(*xs), torch.float32)
            finish.assert_called_once()

    def test_live_parameter_replacement_and_autograd_fallback(self):

        class Linear(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(64, 128, device="cuda"))
                self.b = torch.nn.Parameter(torch.randn(128, device="cuda"))

            def forward(self, x):
                return torch.relu(x @ self.w + self.b)

        gm = torch.fx.symbolic_trace(Linear())
        x = torch.randn(16, 64, device="cuda", requires_grad=True)
        with torch.no_grad():
            plan = graph_compile(gm, (x, ), static_inference=True, fullgraph=True, dynamic=False)
            self.assertEqual(plan.kind, "inductor_small_matmul_single_kernel")
            self.close(plan(x), gm(x), torch.float32)
            gm.w = torch.nn.Parameter(torch.randn_like(gm.w))
            gm.b = torch.nn.Parameter(torch.randn_like(gm.b))
            self.close(plan(x), gm(x), torch.float32)
        plan(x).sum().backward()
        self.assertEqual(plan.fallback_calls, 1)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(gm.w.grad)
        self.assertIsNotNone(gm.b.grad)


if __name__ == "__main__":
    unittest.main()
