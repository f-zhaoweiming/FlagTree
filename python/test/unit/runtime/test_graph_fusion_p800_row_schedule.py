"""Post-grad Softmax rewrites and the bounded P800 Inductor schedule."""
from test_graph_fusion_static import InferenceCPUTestBase, _load_module, torch


class P800NativeSoftmaxTests(InferenceCPUTestBase):
    def test_only_stabilized_fp32_last_dim_softmax_reuses_the_row_reciprocal(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        policy = _load_module('_p800_inductor_softmax')
        for axis, dtype, stabilized, same_exp in (
            (-1, torch.float32, True, True), (0, torch.float32, True, True),
            (-1, torch.float64, True, True), (-1, torch.float32, False, True),
            (-1, torch.float32, True, False),
        ):
            with self.subTest(axis=axis, dtype=dtype, stabilized=stabilized, same_exp=same_exp):
                def fn(x):
                    numerator = (x-x.amax(axis, keepdim=True)).exp() if stabilized else x.exp()
                    denominator = numerator if same_exp else numerator + 1
                    return numerator / denominator.sum(axis, keepdim=True)

                x = torch.randn(8, 16, dtype=dtype)
                x[0, 0] = float('nan')
                x[1, 1] = float('inf')
                x[2, :] = -float('inf')
                gm = make_fx(fn)(x)
                policy.reuse_row_reciprocal(gm.graph)
                gm.graph.lint()
                gm.recompile()
                targets = [node.target for node in gm.graph.nodes]
                applied = axis == -1 and dtype == torch.float32 and stabilized and same_exp
                self.assertEqual(targets.count(torch.ops.aten.reciprocal.default), int(applied))
                self.assertEqual(targets.count(torch.ops.aten.div.Tensor), int(not applied))
                torch.testing.assert_close(gm(x), fn(x), equal_nan=True)

    def test_fixed_tile_applies_only_to_the_measured_case(self):
        policy = _load_module('_p800_inductor_softmax')
        mod = torch.fx.symbolic_trace(lambda a, b: torch.softmax(a + b, -1))
        a = torch.empty(2048, 2048, device='meta')
        b = torch.empty(2048, device='meta')
        self.assertTrue(policy._kl3_row_schedule((a, b), mod))
        for args in [(a.half(), b.half()), (a[:64], b), (a.T, b), (a, b[None, :])]:
            self.assertFalse(policy._kl3_row_schedule(args, mod))
        wrong_axis = torch.fx.symbolic_trace(lambda a, b: torch.softmax(a + b, 0))
        self.assertFalse(policy._kl3_row_schedule((a, b), wrong_axis))
