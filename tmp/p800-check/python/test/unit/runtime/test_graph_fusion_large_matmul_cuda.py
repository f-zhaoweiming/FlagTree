"""Large NVIDIA epilogues: precision boundaries, live state, streams and cache."""
import os
import unittest

import torch
from triton.experimental.graph_fusion import compile as graph_compile


class Matrix(torch.nn.Module):
    def __init__(self, activation, bias):
        super().__init__()
        self.activation = activation
        self.register_buffer("bias", bias)

    def forward(self, x, w):
        value = x @ w + self.bias
        if self.activation == "relu":
            return torch.relu(value)
        if self.activation == "silu":
            return torch.nn.functional.silu(value)
        return torch.nn.functional.gelu(value, approximate="tanh" if self.activation == "gelu_tanh" else "none")


@unittest.skipUnless(os.environ.get("F06_RUN_CUDA_TESTS") == "1", "explicit idle-GPU acceptance only")
class LargeMatmulCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from triton.runtime import driver
        if (not torch.cuda.is_available() or type(driver.active).__module__ != "triton.backends.nvidia.driver"
                or torch.cuda.get_device_capability()[0] < 8):
            raise unittest.SkipTest("NVIDIA SM80+ required")
        cls.original = (torch.backends.cuda.matmul.allow_tf32,
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    @classmethod
    def tearDownClass(cls):
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction) = cls.original

    @torch.no_grad()
    def test_rounding_fresh_outputs_live_bias_and_cached_compilation(self):
        from torch._inductor import config
        for dtype in (torch.float16, torch.bfloat16):
            for activation in ("relu", "gelu_exact", "gelu_tanh", "silu"):
                with self.subTest(dtype=dtype, activation=activation):
                    eps = torch.finfo(dtype).eps
                    x = torch.zeros(128, 64, device="cuda", dtype=dtype)
                    w = torch.zeros(64, 128, device="cuda", dtype=dtype)
                    x[:, 0] = 1 + eps
                    w[0, :] = 1 + eps
                    bias = torch.full((128,), -(1 + 2 * eps), device="cuda", dtype=dtype)
                    gm = torch.fx.symbolic_trace(Matrix(activation, bias))
                    expected = gm(x, w)
                    self.assertTrue(torch.equal(expected, torch.zeros_like(expected)))
                    # Lost mm rounding would leave eps**2 before activation.
                    before = config.get_config_copy()
                    for _ in range(2):
                        plan = graph_compile(gm, (x, w), static_inference=True, fullgraph=True, dynamic=False)
                        self.assertIn("large_matmul_compiler_options", plan.plan_info)
                        self.assertEqual(config.get_config_copy(), before)
                        held = [plan(x, w) for _ in range(3)]
                        self.assertEqual(len({v.data_ptr() for v in held}), 3)
                        for v in held:
                            torch.testing.assert_close(v, expected, atol=0, rtol=0)
                    gm.bias = torch.full_like(bias, eps / 2)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = plan(x, w)
                    stream.synchronize()
                    torch.testing.assert_close(actual, gm(x, w), atol=3e-3, rtol=3e-3)
                    for v in held:
                        torch.testing.assert_close(v, expected, atol=0, rtol=0)
                    # No nonfinite approximation or stale parameter contents.
                    gm.bias[:5] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0.],
                                               device="cuda", dtype=dtype)
                    actual, expected = plan(x, w), gm(x, w)
                    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3, equal_nan=True)
                    self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
                    self.assertTrue(torch.equal(torch.isinf(actual), torch.isinf(expected)))
                    self.assertEqual(plan.fallback_calls, 0)


if __name__ == "__main__":
    unittest.main()
