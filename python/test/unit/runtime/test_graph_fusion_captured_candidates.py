import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from triton.experimental.graph_fusion._model_candidates import model_candidate_routes
from triton.experimental.graph_fusion._model_inductor import inductor_backend


class CapturedCandidateTests(unittest.TestCase):
    def test_default_search_uses_original_dtype_without_fp32_discovery(self):
        for dtype in (torch.float16, torch.bfloat16):
            model=torch.nn.Linear(3,4).to(dtype=dtype).eval().requires_grad_(False)
            original=model.weight.detach().clone();calls=[];record={}
            def build(value, **kw):
                calls.append((value,kw))
                return SimpleNamespace(_f06_audit=[],_f06_resource_allowlist=None)
            with torch.no_grad(), patch('triton.experimental.graph_fusion._model_pipeline.compile_pipeline',build), patch(
                    'triton.experimental.graph_fusion._model_candidates.discover_precision_partitions',
                    side_effect=AssertionError('FP32-rooted search must not run')):
                routes=model_candidate_routes(model,prepare=lambda x:((x,),{}),
                    output_template=torch.empty(2,4,dtype=dtype),
                    validation_inputs=[((torch.ones(2,3,dtype=dtype),),{})],
                    validate=lambda *a:None,record=record,tolerance=dict(atol=.003,rtol=.003))
            self.assertEqual(len(routes),12)
            self.assertEqual(record['capture_source'],'original_dtype')
            self.assertFalse(record['forced_fp32_inputs'])
            self.assertTrue(all(m is model and kw.get('source_precision') is None for m,kw in calls))
            self.assertEqual({kw['precision_policy'] for _,kw in calls},{'preserve_operator','fuse_fp32_internals'})
            self.assertEqual(model.weight.dtype,dtype)
            torch.testing.assert_close(model.weight,original,atol=0,rtol=0)

    def test_reference_is_not_executed_during_unpruned_construction(self):
        model=torch.nn.Linear(2,2).half().eval().requires_grad_(False)
        with torch.no_grad(), patch('triton.experimental.graph_fusion._model_pipeline.compile_pipeline',
                return_value=SimpleNamespace(_f06_audit=[],_f06_resource_allowlist=None)):
            routes=model_candidate_routes(model,prepare=lambda x:((x,),{}),
                output_template=torch.empty(1,2).half(),validation_inputs=[((torch.ones(1,2).half(),),{})],
                validate=lambda *a:(_ for _ in ()).throw(AssertionError('external validator only')),
                record={},tolerance=dict(atol=.003,rtol=.003),linear_schedules=('native',),pointwise_policies=('native',))
        self.assertEqual(len(routes),2)

    def test_captured_linear_keeps_low_input_and_output_in_both_lowerings(self):
        from triton.experimental.graph_fusion._model import enable_vendor_capture
        enable_vendor_capture()
        class Chain(torch.nn.Module):
            def __init__(self):
                super().__init__();self.a=torch.nn.Linear(3,4);self.b=torch.nn.Linear(4,2)
            def forward(self,x):return self.b(torch.relu(self.a(x)))
        for schedule in ('native','inductor','grouped'):
            model=Chain().half().eval().requires_grad_(False);x=torch.randn(2,3).half()
            graphs=[];audit=[]
            backend=inductor_backend(tolerance=dict(atol=.003,rtol=.003),environment_key='native-source-test',
                audit=audit,precision_policy='preserve_operator',
                precision_options=dict(linear_schedule=schedule,pointwise_policy='native',require_compiled=True),
                _cost_graph_observer=lambda gm,args:graphs.append(gm),_capture_cost_graph=True)
            with torch.no_grad(),patch('triton.experimental.graph_fusion._model_inductor._fp32_reference',
                                     side_effect=AssertionError('no FP32 graph construction')):
                compiled=torch.compile(model,backend=backend,fullgraph=True)
                actual=compiled(x)
                torch.testing.assert_close(actual,model(x),atol=0,rtol=0)
            self.assertEqual(actual.dtype,torch.float16)
            self.assertEqual(audit[0]['capture_source'],'original_dtype')
            self.assertNotIn('torch.float32',audit[0]['captured_input_dtypes'])
            nodes=[n for n in graphs[0].graph.nodes if 'linear' in str(n.target)]
            self.assertEqual(len(nodes),2)
            self.assertTrue(all(n.meta['example_value'].dtype==torch.float16 for n in nodes))
            self.assertFalse(any(n.op=='call_method' and n.target=='float' for n in graphs[0].graph.nodes))

    def test_grouped_schedule_preserves_mixed_precision_and_refreshes_weights(self):
        from triton.experimental.graph_fusion._model import enable_vendor_capture
        enable_vendor_capture()
        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q=torch.nn.Linear(16,12)
                self.k=torch.nn.Linear(16,12)
                self.v=torch.nn.Linear(16,12)
                self.low=torch.nn.Linear(16,4).half()
            def forward(self,x,y):
                q=self.q(x).view(2,7,3,4).transpose(1,2)
                k=self.k(x).view(2,7,3,4).transpose(1,2)
                v=self.v(x).view(2,7,3,4).transpose(1,2)
                return q+k,v.sigmoid(),self.low(y)
        model=Mixed().eval().requires_grad_(False)
        args=(torch.randn(2,7,16),torch.randn(2,7,16).half())
        audit=[];graphs=[]
        backend=inductor_backend(tolerance=dict(atol=.003,rtol=.003),environment_key='mixed-grouped-test',
            audit=audit,precision_policy='preserve_operator',
            precision_options=dict(linear_schedule='grouped',require_compiled=True),
            _cost_graph_observer=lambda gm,args:graphs.append(gm),_capture_cost_graph=True)
        with torch.no_grad(),patch('triton.experimental.graph_fusion._model_inductor._fp32_reference',
                                  side_effect=AssertionError('no FP32-rooted graph')):
            compiled=torch.compile(model,backend=backend,fullgraph=True)
            actual=compiled(*args)
            torch.testing.assert_close(actual,model(*args))
            model.k.weight.add_(.25)
            fresh=(args[0]+1,args[1]+1)
            torch.testing.assert_close(compiled(*fresh),model(*fresh))
        self.assertEqual(actual[2].dtype,torch.float16)
        self.assertEqual([r['kind'] for r in audit[0]['layout_regions']],['fp32_grouped_projections'])
        linears=[n for n in graphs[0].graph.nodes if 'linear' in str(n.target)]
        self.assertEqual(len(linears),2)
        self.assertTrue(any('precision_linear' in str(n.target) and
            n.meta['example_value'].dtype==torch.float16 for n in linears))
        self.assertEqual(model.low.weight.dtype,torch.float16)

    def test_invalid_capture_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'capture source'):
            model_candidate_routes(None,prepare=None,output_template=None,validation_inputs=[],
                validate=None,record={},tolerance={},capture_source='implicit_fp32')


if __name__=='__main__':unittest.main()
