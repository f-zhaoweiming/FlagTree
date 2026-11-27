"""Adversarial numerical and native-cache regression on an explicitly idle GPU."""
import os
import unittest

import torch
from triton.experimental.graph_fusion import compile as graph_compile
from triton.experimental.graph_fusion import _inference


@unittest.skipUnless(os.environ.get("F06_RUN_CUDA_TESTS") == "1", "explicit idle-GPU acceptance only")
class PrecisionBoundaryCUDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from triton.runtime import driver
        if not torch.cuda.is_available() or type(driver.active).__module__ != "triton.backends.nvidia.driver":
            raise unittest.SkipTest("NVIDIA CUDA required")

    @torch.no_grad()
    def test_product_rounding_cancellation_and_shared_native_cache(self):
        from torch._inductor import config

        class Product(torch.nn.Module):

            def __init__(self, keepdim):
                super().__init__()
                self.keepdim = keepdim

            def forward(self, x, y):
                return torch.sum(x * y, dim=-1, keepdim=self.keepdim)

        for dtype in (torch.float16, torch.bfloat16):
            for keepdim in (False, True):
                with self.subTest(dtype=dtype, keepdim=keepdim), \
                     config.patch({"emulate_precision_casts": False, "fx_graph_cache": True}):
                    eps = torch.finfo(dtype).eps
                    # Eager products cancel exactly after the visible low-precision
                    # rounding. FP32 products instead leave eps**2 per pair.
                    x = torch.tensor([1 + eps, -(1 + 2 * eps)], device="cuda", dtype=dtype).repeat(16, 64)
                    y = torch.tensor([1 + eps, 1], device="cuda", dtype=dtype).repeat(16, 64)
                    gm = torch.fx.symbolic_trace(Product(keepdim))
                    expected = gm(x, y)
                    self.assertTrue(torch.equal(expected, torch.zeros_like(expected)))
                    unprotected = torch.compile(gm, fullgraph=True, dynamic=False)
                    self.assertTrue((unprotected(x, y) != expected).all())
                    for _ in range(2):
                        # The second plan uses the same native cache directory;
                        # neither may reuse the unprotected computation above.
                        plan = graph_compile(gm, (x, y), static_inference=True, fullgraph=True, dynamic=False)
                        self.assertFalse(config.emulate_precision_casts)
                        self.assertIsNotNone(_inference._single_inductor_kernel(plan.generated_sources))
                        outputs = [plan(x, y) for _ in range(3)]
                        for result in outputs:
                            torch.testing.assert_close(result, expected, atol=0, rtol=0)
                        self.assertEqual(len({v.data_ptr() for v in outputs}), 3)
                        self.assertEqual(plan.fallback_calls, 0)
                        source = "\n".join(plan.generated_sources)
                        self.assertIn(".to(tl." + ("float16" if dtype == torch.float16 else "bfloat16") + ")", source)
                        self.assertIn(".to(tl.float32)", source)
                    self.assertTrue((unprotected(x, y) != expected).all())


if __name__ == "__main__":
    unittest.main()
