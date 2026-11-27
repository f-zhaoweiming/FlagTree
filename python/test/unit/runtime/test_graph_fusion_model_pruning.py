import math
import operator
import unittest

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp

from triton.experimental.graph_fusion._model_pruning import extract_regions, choose_policies, validate_survivors


class PruningTests(unittest.TestCase):
    def graph(self, dtype=torch.float32, transposed=False):
        class Region(torch.nn.Module):
            def forward(self, x, y):
                return (x * torch.sigmoid(1.702 * x) + y).half()
        gm = fx.symbolic_trace(Region())
        x = torch.randn(2, 4, dtype=dtype)
        if transposed: x = x.t().contiguous().t()
        class Values(fx.Interpreter):
            def run_node(self, n):
                value = super().run_node(n)
                n.meta['example_value'] = value
                return value
        Values(gm).run(x, x)
        return gm, x

    def test_expression_keeps_output_conversion_and_both_inputs(self):
        gm, x = self.graph()
        rows, domain = extract_regions(gm)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(domain), 4)
        self.assertIn('half()', rows[0]['identity']['code'])
        torch.testing.assert_close(rows[0]['module'](x, x)[0], gm(x, x))

    def test_signature_includes_dtype_and_strides(self):
        keys = [extract_regions(self.graph(dtype, transpose)[0])[0][0]['signature']
                for dtype, transpose in [(torch.float32, False), (torch.float16, False), (torch.float32, True)]]
        self.assertEqual(len(set(keys)), 3)

    def test_repeated_node_names_do_not_split_calibration_cache(self):
        gm, _ = self.graph()
        key = extract_regions(gm)[0][0]['signature']
        for n in gm.graph.nodes:
            if n.op != 'output': n.name = 'renamed_' + n.name
        gm.recompile()
        self.assertEqual(extract_regions(gm)[0][0]['signature'], key)

    def test_resource_disagreement_preserves_both_winners(self):
        p = ['generated', 'native_mul', 'native']
        scores = {'cluster': dict(zip(p, [1., 2., 10.])), 'sdnn': dict(zip(p, [1., 10., 2.]))}
        self.assertEqual(choose_policies(p, scores), p)

    def test_unknown_invalid_and_near_ties_keep_choices(self):
        p = ['generated', 'native_mul', 'native']
        for scores in ({}, {'auto':dict(zip(p,[1.,1.01,1.02]))}, {'auto':dict(zip(p,[1.,2.,math.nan]))}):
            self.assertEqual(choose_policies(p, scores), p)

    def test_clear_cost_separation_can_prune(self):
        p = ['generated', 'native_mul', 'native']
        self.assertEqual(choose_policies(p, {'auto':dict(zip(p,[1.,2.,4.]))}), p[:2])

    def test_tied_slow_runners_up_do_not_expand_fastest_uncertainty_band(self):
        p = ['generated', 'native_mul', 'native']
        self.assertEqual(choose_policies(p, {'auto':dict(zip(p,[1.,2.,2.01]))}),p[:2])

    def test_small_space_skips_pruning(self):
        self.assertEqual(choose_policies(['generated','native'], {}), ['generated','native'])

    def test_resource_shortlist_is_not_expanded_back_to_cartesian_product(self):
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._model_pipeline import _CompiledPipeline
        parent = _CompiledPipeline(lambda: ((),{}),SimpleNamespace(_f06_audit=[],_f06_capture='test'))
        calls=[]
        def factory():
            calls.append(True)
            return {'cluster':object(),'sdnn':object()}
        parent._f06_resource_factory=factory
        parent._f06_resource_allowlist=('sdnn',)
        self.assertEqual(set(parent._f06_resource_variants()),{'sdnn'})
        self.assertEqual(set(parent._f06_resource_variants()),{'sdnn'})
        self.assertEqual(len(calls),1)
        self.assertIn('cluster',parent._f06_resource_candidates)

    def test_default_pipeline_retains_all_resource_candidates(self):
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._model_pipeline import _CompiledPipeline
        parent=_CompiledPipeline(lambda: ((),{}),SimpleNamespace(_f06_audit=[],_f06_capture='test'))
        parent._f06_resource_factory=lambda: {'cluster':1,'sdnn':2}
        self.assertEqual(parent._f06_resource_variants(),{'cluster':1,'sdnn':2})

    def test_failed_survivors_require_reopening(self):
        def validate(name, output, args, kwargs):
            if output != 1: raise AssertionError('full FP32 contract')
        rows=[]
        self.assertFalse(validate_survivors([('a',lambda:0),('b',lambda:0)],
            validation_inputs=[((),{})],validate=validate,record=rows))
        self.assertEqual(len(rows),2)
        rows=[]
        self.assertTrue(validate_survivors([('a',lambda:0),('b',lambda:1),('c',lambda:0)],
            validation_inputs=[((),{})],validate=validate,record=rows))
        self.assertEqual(len(rows),2)

    def test_survivor_execution_failure_is_not_precision_recovery(self):
        def broken(): raise RuntimeError('device failure')
        with self.assertRaisesRegex(RuntimeError,'device failure'):
            validate_survivors([('a',broken)],validation_inputs=[((),{})],
                               validate=lambda *args:None,record=[])

    def test_precision_rejected_parent_is_still_prepared_for_resource_variants(self):
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._model_pipeline import _CompiledPipeline
        class Compiled:
            _f06_capture='test'
            def __init__(self):self._f06_audit=[]
            def __call__(self):
                self._f06_audit.append({'route':'compiled'})
                return 0
        parent=_CompiledPipeline(lambda: ((),{}),Compiled())
        def factory():
            self.assertTrue(parent._f06_audit)
            return {'sdnn':lambda:1}
        parent._f06_resource_factory=factory
        def validate(name,out,args,kwargs):
            if out!=1:raise AssertionError('original FP32 accuracy')
        self.assertFalse(validate_survivors([('parent',parent)],validation_inputs=[((),{})],
                                           validate=validate,record=[]))
        self.assertTrue(validate_survivors([('sdnn',parent._f06_resource_variants()['sdnn'])],
            validation_inputs=[((),{})],validate=validate,record=[]))

    def test_integer_mask_cast_is_part_of_region(self):
        graph = fx.Graph(); mask = graph.placeholder('mask')
        cast = graph.call_method('to', (mask,), {'dtype': torch.float32})
        scaled = graph.call_function(operator.mul, (cast, -10000.))
        graph.output(scaled)
        mask.meta['example_value'] = torch.ones(2,4,dtype=torch.int64)
        cast.meta['example_value'] = mask.meta['example_value'].float()
        scaled.meta['example_value'] = cast.meta['example_value'] * -10000.
        rows, _ = extract_regions(fx.GraphModule({},graph))
        self.assertEqual(rows[0]['identity']['inputs'][0]['dtype'], 'torch.int64')
        self.assertIn('torch.float32', rows[0]['identity']['code'])
        torch.testing.assert_close(rows[0]['module'](mask.meta['example_value'])[0],scaled.meta['example_value'])

    def test_residual_edge_cannot_cross_opaque_dependency(self):
        graph=fx.Graph();x=graph.placeholder('x');y=graph.placeholder('y')
        first=graph.call_function(operator.add,(x,y))
        opaque=graph.call_function(torch.softmax,(first,),{'dim':-1})
        second=graph.call_function(operator.add,(first,opaque));graph.output(second)
        for n in (x,y,first,opaque,second): n.meta['example_value']=torch.randn(2,4)
        rows,_=extract_regions(fx.GraphModule({},graph))
        self.assertEqual([r['nodes'] for r in rows],[[first.name],[second.name]])

    def test_long_outside_path_prevents_indirect_region_merge(self):
        graph=fx.Graph();x=graph.placeholder('x');y=graph.placeholder('y')
        a=graph.call_function(operator.add,(x,y))
        b=graph.call_function(operator.mul,(a,2.))
        outside=graph.call_function(torch.softmax,(a,),{'dim':-1})
        c=graph.call_function(operator.add,(b,outside));graph.output(c)
        for n in (x,y,a,b,outside,c): n.meta['example_value']=torch.randn(2,4)
        rows,_=extract_regions(fx.GraphModule({},graph))
        self.assertEqual([r['nodes'] for r in rows],[[a.name,b.name],[c.name]])


if __name__ == '__main__':
    unittest.main()
