"""Precision-region graph rewrites independent of model names and shapes."""
import unittest
import torch
import torch.nn.functional as F
from torch import fx
from triton.experimental.graph_fusion._fp32_compute_regions import fuse_fp32_compute_regions, fold_layernorm_boundary_casts


def capture(model, args):
    gm = fx.symbolic_trace(model)
    class Values(fx.Interpreter):
        def run_node(self, n):
            value = super().run_node(n)
            n.meta['example_value'] = value
            return value
    Values(gm).run(*args)
    return gm


class ComputeRegionTests(unittest.TestCase):
    def test_chain_removes_intermediate_rounding(self):
        class Model(torch.nn.Module):
            def forward(self, x, a, b):
                return (x + a) + b
        args = tuple(torch.tensor([v], dtype=torch.float16) for v in (2048, 1, -2048))
        gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(Model()(*args).item(), 0)
        self.assertEqual(gm(*args).item(), 1)
        self.assertEqual(gm(*args).dtype, torch.float16)
        self.assertEqual(len(rows[0]['output_conversions']), 1)
        for n in gm.graph.nodes:
            if 'f06_fp32_compute_region' in n.meta:
                self.assertEqual(n.meta['example_value'].dtype, torch.float32)

    def test_arbitrary_broadcast_shape_and_long_quickgelu_chain(self):
        class Model(torch.nn.Module):
            def forward(self, x, b):
                y = x + b
                z = y * torch.sigmoid(y * 1.702)
                return z + b
        for shape in ((2, 3, 5), (17, 129)):
            args = (torch.randn(*shape).half(), torch.randn(shape[-1]).half())
            gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]['compute_nodes']), 5)
            reference = Model()(*(v.float() for v in args)).half()
            torch.testing.assert_close(gm(*args), reference, atol=0, rtol=0)

    def test_public_branches_and_views_share_one_rounded_base(self):
        class Model(torch.nn.Module):
            def forward(self, x, a, b):
                y = (x + a) + b
                view = y.view(-1)
                return y, view, view * 2 + 1
        args = tuple(torch.full((2, 3), v, dtype=torch.float16) for v in (2048, 1, -2048))
        gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 1)
        actual = gm(*args)
        reference = tuple(v.half() for v in Model()(*(v.float() for v in args)))
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        self.assertEqual(actual[0].data_ptr(), actual[1].data_ptr())
        self.assertEqual(actual[1].stride(), (1,))

    def test_real_half_arithmetic_stays_between_regions(self):
        class Model(torch.nn.Module):
            def forward(self, x, a, b):
                y = torch.relu((x + a) + b)
                return (y * 2) * 3
        args = tuple(torch.tensor([v], dtype=torch.float16) for v in (2048, 1, -2048))
        gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 2)
        relu = next(n for n in gm.graph.nodes if n.target is torch.relu)
        self.assertEqual(relu.meta['example_value'].dtype, torch.float16)
        self.assertEqual(relu.args[0].meta['example_value'].dtype, torch.float16)
        self.assertEqual(gm(*args).item(), 6)

    def test_escaping_input_view_keeps_original_input_storage(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.view(-1).half()
                return x, y, (y + 1) * 2
        args = (torch.randn(2, 3).half(),)
        gm = capture(Model(), args)
        self.assertEqual(len(fuse_fp32_compute_regions(gm, args)), 1)
        actual = gm(*args)
        self.assertEqual(actual[0].data_ptr(), args[0].data_ptr())
        self.assertEqual(actual[1].data_ptr(), args[0].data_ptr())
        torch.testing.assert_close(actual[2], ((args[0].view(-1).float()+1)*2).half(), atol=0, rtol=0)
        actual[1].add_(1)
        torch.testing.assert_close(actual[1], args[0].view(-1), atol=0, rtol=0)

    def test_low_matrix_compute_is_not_promoted(self):
        class Model(torch.nn.Module):
            def forward(self, x, a, b, w):
                y = (x + a) + b
                z = torch.mm(y, w)
                return (y + z) * 2
        args = (torch.full((2, 3), 2048.).half(), torch.ones(3).half(),
                torch.full((3,), -2048.).half(), torch.eye(3).half())
        gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 1)
        mm = next(n for n in gm.graph.nodes if n.target is torch.mm)
        self.assertEqual(mm.meta['example_value'].dtype, torch.float16)
        self.assertTrue(all(v.meta['example_value'].dtype == torch.float16 for v in mm.args))
        torch.testing.assert_close(gm(*args), torch.full((2, 3), 4.).half(), atol=0, rtol=0)

    def test_mutation_splits_regions_and_rereads_the_observable_half_value(self):
        class Model(torch.nn.Module):
            def forward(self, x, b):
                y = x + b
                before = y + 1
                y.add_(2)
                after = (y + 3) + 1
                return before, after, y
        args = (torch.tensor([2048.]).half(), torch.tensor([1.]).half())
        gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 2)
        self.assertEqual([v.item() for v in gm(*args)], [2050., 2054., 2050.])
        self.assertTrue(any(n.target == 'add_' for n in gm.graph.nodes))

    def test_explicit_half_float_roundtrip_inside_region_disappears(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return ((x.float() + 1).half().float() - 2048).half()
        # Subtract is deliberately not a reviewed low-precision contract. Use
        # the equivalent addition so both compute nodes have reviewed evidence.
        class AddModel(torch.nn.Module):
            def forward(self, x):
                return ((x.float() + 1).half().float() + -2048).half()
        args = (torch.tensor([2048.]).half(),)
        gm = capture(AddModel(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(gm(*args).item(), 1)
        self.assertEqual(AddModel()(*args).item(), 0)

    def test_add_layernorm_uses_fp32_moments_and_final_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x, b, weight, shift):
                return F.layer_norm(x + b, (5,), weight, shift)
        args = (torch.randn(2, 3, 5).half(), torch.randn(5).half(),
                torch.randn(5).half(), torch.randn(5).half())
        gm = capture(Model(), args); rows = fuse_fp32_compute_regions(gm, args)
        self.assertEqual(len(rows), 1)
        torch.testing.assert_close(gm(*args), Model()(*(v.float() for v in args)).half(), atol=0, rtol=0)

    def test_bf16_does_not_inherit_fp16_machine_code_evidence(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1) * 2
        args = (torch.ones(3, dtype=torch.bfloat16),)
        gm = capture(Model(), args); before = gm.code
        self.assertEqual(fuse_fp32_compute_regions(gm, args), [])
        self.assertEqual(gm.code, before)

    def test_template_casts_fold_only_when_no_wide_output_is_live(self):
        from triton.experimental.graph_fusion._p800_model_layernorm import _op
        class Model(torch.nn.Module):
            def __init__(self, wide):
                super().__init__()
                self.wide = wide
            def forward(self, x, b, w, s):
                y, residual = torch.ops.flagtree_p800.model_layernorm.default(
                    x.float(), b.float(), w.float(), s.float(), 1e-5, True, 'test')
                return y.half(), residual if self.wide else residual.half()
        args = (torch.randn(2, 512).half(), *(torch.randn(512).half() for _ in range(3)))
        for wide in (False, True):
            model = Model(wide)
            gm = capture(model, args)
            records = fold_layernorm_boundary_casts(gm, args)
            self.assertEqual(len(records), 0 if wide else 1)
            torch.testing.assert_close(gm(*args), model(*args), atol=0, rtol=0)
            if not wide:
                from triton.experimental.graph_fusion._operator_precision import preserve_native_compute
                audit = preserve_native_compute(gm)
                self.assertTrue(any(r['action']=='reviewed_fp32_template' for r in audit))
                self.assertTrue(any(n.target==torch.ops.flagtree_p800.model_layernorm.default for n in gm.graph.nodes))

    def test_template_cast_fold_does_not_cross_input_mutation(self):
        from triton.experimental.graph_fusion._p800_model_layernorm import _op
        class Model(torch.nn.Module):
            def forward(self, x, b):
                xf = x.float()
                x.add_(1)
                y, residual = torch.ops.flagtree_p800.model_layernorm.default(
                    xf, b.float(), None, None, 1e-5, False, 'test')
                return y.half()
        args = (torch.randn(2, 512).half(), torch.randn(512).half())
        gm = capture(Model(), args)
        self.assertEqual(fold_layernorm_boundary_casts(gm, args), [])


if __name__ == '__main__':
    unittest.main()
