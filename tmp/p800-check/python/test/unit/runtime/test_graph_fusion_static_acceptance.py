import types
import unittest
import torch
from test_graph_fusion_static import _load_under_test, _bit_equal, _Backstop


class ParentAcceptance(unittest.TestCase):

    def test_original_matmul_graph_rounding_and_live_bias(self):
        m = _load_under_test()['inference']

        class M(torch.nn.Module):

            def __init__(self, b):
                super().__init__()
                self.register_buffer('bias', b)

            def forward(self, x, w):
                return torch.matmul(x, w) + self.bias

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.manual_seed(61)
            x = (torch.randn(4, 4) * .25).to(dtype)
            w = (torch.randn(4, 4) * .25).to(dtype)
            bias = (torch.randn(4) * .25).to(dtype)
            gm = torch.fx.symbolic_trace(M(bias))
            fallback = _Backstop()
            p = m._compile_native_mm_add(gm, (x, w), fallback, allow_non_cuda=True)
            self.assertEqual(p.plan_info['operator_sequence'][0], 'torch.matmul(x, w)')
            for value in [
                    float('nan'),
                    float('inf'),
                    float('-inf'), 0., -0.,
                    torch.finfo(dtype).tiny,
                    torch.finfo(dtype).tiny / 2
            ]:
                new = bias.clone()
                new[0] = value
                gm.bias = new
                got = p(x, w)
                expected = gm(x, w)
                self.assertTrue(_bit_equal(got, expected))
                self.assertEqual(p.fallback_calls, 0)

    def test_rebound_forward_cannot_use_original_bias(self):
        m = _load_under_test()['inference']

        class M(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.register_buffer('bias', torch.randn(4))

            def forward(self, x, w):
                return torch.mm(x, w) + self.bias

        gm = torch.fx.symbolic_trace(M())
        x = torch.randn(4, 4)
        w = torch.randn(4, 4)
        fallback = _Backstop()
        p = m._compile_native_mm_add(gm, (x, w), fallback, allow_non_cuda=True)
        other = M()
        gm.forward = types.MethodType(gm.forward.__func__, other)
        self.assertEqual(p(x, w), 'fallback')
        self.assertEqual(p.fallback_calls, 1)


if __name__ == '__main__': unittest.main()
