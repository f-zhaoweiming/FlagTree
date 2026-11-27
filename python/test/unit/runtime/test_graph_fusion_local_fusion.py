"""Candidate granularity, shared parameters, guards and complete-route admission."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from triton.experimental.graph_fusion import _local_fusion as local
from triton.experimental.graph_fusion._captured_candidates import captured_model_candidate_routes
from triton.experimental.graph_fusion._whole_model_selection import ModelRoute, select_whole_model


class QuickGELUActivation(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.act = QuickGELUActivation()
        self.shared = self.act
    def forward(self, x):
        y = self.linear(x)
        return {'y': self.act(y), 'residual': self.shared(y) + x}


class LocalFusionTests(unittest.TestCase):
    def model(self):
        return Model().eval().requires_grad_(False)

    def patches(self):
        return (patch.object(local, '_p800_model', return_value=True),
                patch.object(local, '_quickgelu_types', return_value=(QuickGELUActivation,)),
                patch.object(local, '_supported_tensor', side_effect=lambda x:
                             type(x) is torch.Tensor and not torch.is_grad_enabled() and not x.requires_grad),
                patch.object(local, '_compile_region', side_effect=self.compile))

    def compile(self, graph, x, size, record):
        record.update(state='compiled', signature=repr(local._signature(x)), buffer_bytes=size)
        return graph.forward

    def test_graph_requires_exact_pure_quickgelu(self):
        self.assertIsNotNone(local._quickgelu_graph(QuickGELUActivation().eval()))
        class Wrong(torch.nn.Module):
            def forward(self, x):return x * torch.sigmoid(1.7 * x)
        self.assertIsNone(local._quickgelu_graph(Wrong().eval()))
        act=QuickGELUActivation().eval();act.register_buffer('extra',torch.ones(1))
        self.assertIsNone(local._quickgelu_graph(act))
        act=QuickGELUActivation().eval();act.register_forward_hook(lambda *a:None)
        self.assertIsNone(local._quickgelu_graph(act))
        self.assertIsNone(local._quickgelu_graph(QuickGELUActivation()))

    def test_no_regions_or_non_p800_does_not_create_routes(self):
        model=self.model()
        with torch.no_grad(),patch.object(local,'_p800_model',return_value=False):
            self.assertEqual(local.local_fusion_candidate_routes(model,prepare=None,validation_inputs=[],record={}),{})

    def test_hot_match_preserves_offset_shape_context_and_autocast_guards(self):
        with torch.no_grad():
            x=torch.ones(4);signature=local._signature(x)
            self.assertTrue(local._matches(x,signature))
            self.assertFalse(local._matches(torch.ones(5),signature))
            self.assertFalse(local._matches(torch.ones(5)[1:],signature))
            self.assertFalse(local._matches(torch.ones(8)[::2],signature))
            self.assertFalse(local._matches(x.double(),signature))
            negative=torch._neg_view(x)
            self.assertTrue(negative.is_neg())
            self.assertEqual(negative.data_ptr(),x.data_ptr())
            self.assertFalse(local._matches(negative,signature))
            with torch.autocast('cpu'):
                self.assertFalse(local._matches(x,signature))
            with torch.inference_mode():
                self.assertFalse(local._matches(x,signature))
        self.assertFalse(local._matches(x,signature))

    def test_private_models_share_parameters_and_tied_modules_without_mutating_input(self):
        model=self.model();x=torch.randn(2,4);before=x.clone();original=model.act.forward
        p=self.patches()
        with torch.no_grad(),p[0],p[1],p[2],p[3] as compile_call:
            routes=local.local_fusion_candidate_routes(model,prepare=lambda x:((x,),{}),
                validation_inputs=[((x,),{})],record={})
            self.assertEqual(set(routes),{'local_quickgelu_lm512','local_quickgelu_lm1024'})
            self.assertEqual(compile_call.call_count,2)
            for route in routes.values():
                private=route.call.model
                self.assertIsNot(private,model)
                self.assertIs(private.linear.weight,model.linear.weight)
                self.assertIs(private.act,private.shared)
                self.assertIsNot(private.act,model.act)
                a=route.call(x);b=route.call(x+1)
                torch.testing.assert_close(a,model(x),atol=0,rtol=0)
                torch.testing.assert_close(b,model(x+1),atol=0,rtol=0)
                self.assertNotEqual(a['y'].data_ptr(),b['y'].data_ptr())
            model.linear.weight.add_(.25)
            torch.testing.assert_close(routes['local_quickgelu_lm512'].call(x),model(x))
            torch.testing.assert_close(x,before,atol=0,rtol=0)
            self.assertEqual(model.act.forward,original)

    def test_prepared_miss_keeps_native_semantics_without_recompilation(self):
        model=self.model();p=self.patches();x=torch.randn(2,4)
        with torch.no_grad(),p[0],p[1],p[2],p[3] as compiler:
            routes=local.local_fusion_candidate_routes(model,prepare=lambda x:((x,),{}),
                validation_inputs=[((x,),{})],record={},buffer_sizes=(512,))
            call=next(iter(routes.values())).call
            for shape in [(3,4),(1,4)]:
                y=torch.randn(*shape);torch.testing.assert_close(call(y),model(y))
            self.assertEqual(compiler.call_count,1)
        with torch.enable_grad():
            y=x.clone().requires_grad_();out=call(y)['y'];out.sum().backward()
            self.assertIsNotNone(y.grad)

    def test_full_candidate_factory_and_selector_include_local_route(self):
        model=self.model();x=torch.randn(2,4);p=self.patches();record={}
        def whole(*args,**kw):
            return SimpleNamespace(_f06_audit=[],_f06_resource_allowlist=None)
        with torch.no_grad(),p[0],p[1],p[2],p[3],patch(
                'triton.experimental.graph_fusion._model_pipeline.compile_pipeline',whole):
            routes=captured_model_candidate_routes(model,prepare=lambda x:((x,),{}),
                output_template=model(x),validation_inputs=[((x,),{})],validate=lambda *a:None,
                record=record,tolerance=dict(atol=.003,rtol=.003))
            self.assertIn('local_quickgelu_lm1024',routes)
            self.assertTrue(record['implementations']['local_quickgelu_lm1024']['requires_complete_compiled_validation'])
            calls={n:r for n,r in routes.items() if n.startswith('local_')}
            calls['eager']=ModelRoute(model)
            selection={}
            winner=select_whole_model(calls,validation_inputs=[((x,),{})],
                validate=lambda n,out,*a:torch.testing.assert_close(out,model(x)),
                measure=lambda fn:[1.0 if fn is model else .8 if fn.record['buffer_bytes']==512 else .7],
                record=selection,preferred='eager')
            self.assertEqual(selection['route'],'local_quickgelu_lm1024')
            torch.testing.assert_close(winner(x),model(x))

    def test_bad_complete_output_rejects_fast_local_route(self):
        model=self.model();x=torch.randn(2,4);p=self.patches()
        with torch.no_grad(),p[0],p[1],p[2],p[3]:
            routes=local.local_fusion_candidate_routes(model,prepare=lambda x:((x,),{}),
                validation_inputs=[((x,),{})],record={},buffer_sizes=(512,))
            routes['eager']=ModelRoute(model);record={}
            def validate(name,out,*a):
                if name!='eager':raise AssertionError('A public residual output got worse')
            winner=select_whole_model(routes,validation_inputs=[((x,),{})],validate=validate,
                measure=lambda fn:[1.],record=record,preferred='eager')
            self.assertIs(winner,model)
            self.assertEqual(record['rejected'][0]['stage'],'full_output_validation')


if __name__=='__main__':unittest.main()
