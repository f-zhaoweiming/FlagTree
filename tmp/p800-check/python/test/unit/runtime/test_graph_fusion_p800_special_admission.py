"""The safe entry must reject unproven math before compiling or scanning inputs."""
import unittest

import torch
from triton.experimental.graph_fusion import p800_special_values as public


class SpecialAdmission(unittest.TestCase):
    @staticmethod
    def reduction(keepdim=False):
        return torch.fx.symbolic_trace(lambda x, y: (x * y).sum(-1, keepdim=keepdim))

    def test_extreme_value_failure_prevents_compilation(self):
        for dtype in (torch.float32, torch.bfloat16):
            for keepdim in (False, True):
                xs = (torch.empty(32, 256, dtype=dtype), torch.empty(256, dtype=dtype))
                with self.assertRaisesRegex(ValueError, 'extreme-value'):
                    public.prepare_special_candidate(self.reduction(keepdim), xs, tolerance=dict(atol=0, rtol=0))

    def test_reject_risky_reduction_shape(self):
        xs = (torch.empty(64, 1024, dtype=torch.float16), torch.empty(1024, dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, 'shape'):
            public._admit(self.reduction(), xs)

    def test_oracle_detects_infinite_bias_silu_bug(self):
        from torch.nn.functional import silu
        xs = (torch.zeros(16, 64), torch.zeros(64, 128), torch.zeros(128))
        gm = torch.fx.symbolic_trace(lambda x, w, b: silu(x @ w + b))
        self.assertEqual(public._admit(gm, xs), 'matmul_bias_silu')
        def bad(x, w, b):
            result = gm(x, w, b)
            return torch.where(torch.isposinf(result), float('nan'), result)
        with self.assertRaises(AssertionError):
            public._special_oracle('matmul_bias_silu', bad, xs, dict(atol=3e-4, rtol=3e-4))

    def test_oracle_checks_both_multiply_orders(self):
        gm = self.reduction(True)
        xs = (torch.zeros(32, 256, dtype=torch.float16), torch.zeros(256, dtype=torch.float16))
        tolerance = dict(atol=.003, rtol=.003)
        self.assertEqual(public._admit(gm, xs), 'mul_sum_keepdim')
        public._special_oracle('mul_sum_keepdim', gm, xs, tolerance)
        for force_zero in (lambda x,y:x==0, lambda x,y:y==0):
            def broken(x, y):
                return torch.where(force_zero(x,y), 0., x*y).sum(-1, keepdim=True)
            with self.assertRaises(AssertionError):
                public._special_oracle('mul_sum_keepdim', broken, xs, tolerance)


if __name__ == '__main__':
    unittest.main()
