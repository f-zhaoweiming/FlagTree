"""Operator boundaries retain native dtype semantics before compiler lowering."""
import copy
import operator
import unittest

import torch
import torch.nn.functional as F
from torch import fx
from triton.experimental.graph_fusion._operator_precision import preserve_native_compute, review_precision


def capture(model, args):
    gm = fx.symbolic_trace(model)
    class Values(fx.Interpreter):
        def run_node(self, n):
            value = super().run_node(n)
            n.meta['example_value'] = value
            return value
    Values(gm).run(*args)
    return gm


class PrecisionBoundaryTests(unittest.TestCase):
    def test_native_low_linear_keeps_casts_and_both_public_outputs(self):
        class Model(torch.nn.Module):
            def forward(self, x, w, b):
                z = torch.ops.aten.linear.default(x.half(), w, b)
                return z.float() * 1.5, z
        args = (torch.randn(2, 4), torch.randn(3, 4).half(), torch.randn(3).half())
        gm = capture(Model(), args)
        records = preserve_native_compute(gm)
        for x in (args[0], args[0] + 1):
            torch.testing.assert_close(gm(x, *args[1:]), Model()(x, *args[1:]), atol=0, rtol=0)
        linear = next(r for r in records if 'linear' in r['operator'])
        self.assertEqual(linear['input_dtypes'], ['torch.float16'] * 3)
        self.assertEqual(linear['output_dtypes'], ['torch.float16'])
        self.assertEqual(linear['accumulator'], 'not_established')
        self.assertEqual(linear['action'], 'native_compute_boundary')
        self.assertIn('half()', gm.code)
        self.assertIn('float()', gm.code)
        self.assertEqual(gm(*args)[1].dtype, torch.float16)

    def test_fp16_and_bf16_relu_clamp_preserve_edge_values(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x), torch.clamp(x, min=-1., max=1.)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.tensor([-float('inf'), -1., -0., 0., 1., float('inf'), float('nan')], dtype=dtype)
            gm = capture(Model(), (x,))
            rows = preserve_native_compute(gm)
            actual, expected = gm(x), Model()(x)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)
                self.assertTrue(torch.equal(torch.signbit(got), torch.signbit(want)))
            self.assertTrue(all(r['action'] == 'native_compute_boundary' for r in rows))

    def test_unknown_pure_aten_retains_implementation_and_metadata(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.sin.default(x)
        x = torch.randn(3, 4).half().t()
        gm = capture(Model(), (x,)); rows = preserve_native_compute(gm)
        self.assertEqual(rows[0]['internal_arithmetic'], 'not_established')
        self.assertEqual(rows[0]['action'], 'native_compute_boundary')
        torch.testing.assert_close(gm(x), Model()(x), atol=0, rtol=0)
        from triton.experimental.graph_fusion._model_inductor import refresh_graph_metadata
        refresh_graph_metadata(gm, (x,))
        op = next(n for n in gm.graph.nodes if n.op == 'call_function')
        self.assertEqual(op.meta['example_value'].stride(), gm(x).stride())

    def test_unknown_python_compute_requires_native_candidate(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.special.erfcx(x)
        # Only capture shape/type metadata; the CPU half kernel need not exist.
        gm = fx.symbolic_trace(Model()); x = torch.empty(2, dtype=torch.float16)
        for n in gm.graph.nodes:
            n.meta['example_value'] = x
        rows = preserve_native_compute(gm)
        self.assertEqual(rows[0]['action'], 'native_candidate_required')

    def test_inplace_and_out_are_not_replaced_by_pure_wrappers(self):
        class Model(torch.nn.Module):
            def forward(self, x, out):
                torch.clamp(x, min=-1., out=out)
                x.relu_()
                return out, x
        args = (torch.tensor([-2., 3.]).half(), torch.empty(2).half())
        gm = capture(Model(), tuple(x.clone() for x in args)); before = gm.code
        rows = preserve_native_compute(gm)
        self.assertEqual(gm.code, before)
        self.assertTrue(all(r['action'] == 'native_candidate_required' for r in rows))

    def test_fp32_graph_is_not_narrowed_by_policy(self):
        class Model(torch.nn.Module):
            def forward(self, x, w):
                return torch.ops.aten.linear.default(torch.relu(x), w)
        args = (torch.randn(2, 4), torch.randn(3, 4))
        gm = capture(Model(), args); before = gm.code
        rows = preserve_native_compute(gm)
        self.assertEqual(gm.code, before)
        self.assertTrue(all(not r['contains_low_precision'] for r in rows))

    def test_matrix_input_dtype_does_not_imply_accumulator_precision(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)
        args = (torch.randn(2, 4).half(), torch.randn(4, 3).half())
        gm = capture(Model(), args); rows = preserve_native_compute(gm)
        self.assertEqual(rows[0]['accumulator'], 'not_established')
        torch.testing.assert_close(gm(*args), Model()(*args), atol=0, rtol=0)

    def test_low_candidate_cannot_bypass_complete_fp32_gate(self):
        from triton.experimental.graph_fusion._model_precision import compare_model_outputs
        x = torch.tensor([2048., 1., -2048.])
        original_fp32 = (x[0] + x[1]) + x[2]
        half = x.half(); low = (half[0] + half[1]) + half[2]
        self.assertEqual(low.item(), 0.)
        self.assertEqual(original_fp32.item(), 1.)
        with self.assertRaises(AssertionError):
            compare_model_outputs(low, original_fp32, dict(atol=.003, rtol=.003))

    def test_preserved_candidate_requires_outer_fp32_gate_and_no_inner_selector(self):
        from unittest.mock import patch
        from triton.experimental.graph_fusion._model_inductor import inductor_backend
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        x = torch.randn(4).half(); gm = capture(Model(), (x,)); audit = []
        prepared = lambda value: value
        with patch('torch._inductor.compile_fx.compile_fx', return_value=prepared), patch(
                'triton.experimental.graph_fusion._model_inductor._SelectGraph',
                side_effect=AssertionError('must not select using inner FP16 reference')):
            result = inductor_backend(tolerance=dict(atol=.003, rtol=.003),
                environment_key='test', audit=audit, precision_policy='preserve_operator')(gm, (x,))
        self.assertIs(result, prepared)
        self.assertTrue(audit[0]['requires_original_fp32_model_validation'])
        self.assertEqual(audit[0]['state'], 'awaiting_complete_model_validation')
        self.assertNotIn('validated_against', audit[0])

    def test_native_conv_embedding_norm_and_reverse_subtraction(self):
        class Model(torch.nn.Module):
            def forward(self, x, w, ids, embedding):
                image = F.conv2d(x, w, stride=2)
                text = F.embedding(ids, embedding)
                return 1. - image, text / text.norm(p=2, dim=-1, keepdim=True)
        args = (torch.randn(1, 3, 8, 8).half(), torch.randn(4, 3, 2, 2).half(),
                torch.tensor([1, 2]), torch.randn(4, 8).half())
        gm = capture(Model(), args); rows = preserve_native_compute(gm)
        self.assertFalse([r for r in rows if r['action'] == 'native_candidate_required'])
        torch.testing.assert_close(gm(*args), Model()(*args), atol=0, rtol=0)

    def test_native_boundary_lowering_calls_original_entry_without_python_wrapper(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from torch._inductor.lowering import lowerings
        from triton.experimental.graph_fusion._operator_precision import _lower_reviewed, _linear, _relu, _matmul
        called = []
        def fallback(op, **kwargs):
            def run(*args, **kwargs):
                called.append(op)
                return 'native'
            return run
        x = SimpleNamespace(get_device=lambda: torch.device('cuda'))
        # This checks dispatch with mock IR values; real IR is checked in the
        # compiler/device test. Do not ask validate_ir to accept the marker.
        with patch('torch._inductor.lowering.fallback_handler', fallback), patch(
                'torch._inductor.lowering.validate_ir'):
            _lower_reviewed(_relu._opoverload, _relu)
            self.assertEqual(lowerings[_relu._opoverload](x), 'native')
            self.assertIs(called[-1], torch.ops.aten.relu.default)
            _lower_reviewed(_matmul._opoverload, _matmul)
            self.assertEqual(lowerings[_matmul._opoverload](x, x, 'mm'), 'native')
            self.assertIs(called[-1], torch.ops.aten.mm.default)
            if hasattr(torch.ops._dynamo_workaround, 'linear'):
                _lower_reviewed(_linear._opoverload, _linear)
                self.assertEqual(lowerings[_linear._opoverload](x, x, None), 'native')
                self.assertIs(called[-1], torch.ops._dynamo_workaround.linear.default)

    def test_original_dtype_pooling_keeps_reductions_and_explicit_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = F.dropout(x, p=.5, training=False)
                return torch.stack((F.normalize(y, p=2., dim=-1), y)), torch.sum(y, dim=-1), y.sum(), y.sum(dim=0, dtype=torch.float32), y
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(3, 4).to(dtype).t()
            gm = capture(Model(), (x,)); rows = preserve_native_compute(gm)
            self.assertFalse([r for r in rows if r['action'] == 'native_candidate_required'])
            got, want = gm(x), Model()(x)
            torch.testing.assert_close(got, want, atol=0, rtol=0)
            self.assertEqual(got[0].dtype, dtype)
            self.assertEqual(got[1].dtype, dtype)
            self.assertEqual(got[3].dtype, torch.float32)
            self.assertEqual(got[4].data_ptr(), x.data_ptr())

    def test_iadd_keeps_mutation_and_aliases(self):
        graph = fx.Graph(); x = graph.placeholder('x'); y = graph.placeholder('y')
        z = graph.call_function(operator.iadd, (x, y)); graph.output((x, z))
        gm = fx.GraphModule({}, graph)
        for n in (x, y, z): n.meta['example_value'] = torch.ones(4).half()
        before = gm.code
        rows = preserve_native_compute(gm)
        self.assertEqual(before, gm.code)
        self.assertEqual(rows[0]['action'], 'retain_dtype_mutation')
        value = torch.ones(4).half(); got = gm(value, value.clone())
        self.assertTrue(torch.equal(value, torch.full_like(value, 2)))
        self.assertEqual(got[0].data_ptr(), got[1].data_ptr())


if __name__ == '__main__':
    unittest.main()
