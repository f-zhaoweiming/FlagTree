"""CPU checks for whole-model partition semantics; no accelerator initialization."""
import copy
import operator
import unittest

import torch
import torch.nn.functional as F
from torch import fx

from triton.experimental.graph_fusion._model import find_regions, partition_backend, enable_vendor_capture


class ModelPartitionTests(unittest.TestCase):
    def lower(self, model, inputs):
        gm=fx.symbolic_trace(model)
        expected=gm(*inputs)
        audit=[]
        run=partition_backend(tolerance=dict(atol=.0003,rtol=.0003),environment_key='cpu-test',audit=audit)(gm,inputs)
        with torch.no_grad():actual=run(*inputs)
        torch.testing.assert_close(actual,expected)
        return gm,audit

    def test_large_graph_and_indexing_stay_original(self):
        class Model(torch.nn.Module):
            def forward(self,x,y):
                x=x[:,1:]
                for _ in range(20):x=torch.sin(x)
                return F.layer_norm(x+y,(8,),eps=1e-5),x
        gm,audit=self.lower(Model(),(torch.randn(3,5,8),torch.randn(3,4,8)))
        self.assertGreater(audit[0]['compute_nodes'],16)
        self.assertEqual(len(audit[0]['regions']),1)
        self.assertTrue(any(n.target is operator.getitem for n in gm.graph.nodes))
        self.assertEqual(sum(n.target is torch.sin for n in gm.graph.nodes),20)

    def test_shared_intermediate_preserves_extra_output(self):
        class Model(torch.nn.Module):
            def forward(self,a,b):
                z=a+b
                return F.layer_norm(z,(8,)),z
        gm=fx.symbolic_trace(Model());matches,rejected=find_regions(gm)
        self.assertEqual(matches,[])
        self.assertEqual(rejected[0]['reason'],'intermediate has external users')
        self.lower(Model(),(torch.randn(2,8),torch.randn(2,8)))

    def test_two_regions_live_parameters_and_output_structure(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.ones(8),requires_grad=False)
                self.b=torch.nn.Parameter(torch.zeros(8),requires_grad=False)
            def forward(self,a,b):
                x=F.layer_norm(a+b,(8,),self.w,self.b)
                return {'norm':x,'softmax':torch.softmax(x+b,-1)}
        gm,audit=self.lower(Model(),(torch.randn(2,8),torch.randn(2,8)))
        self.assertEqual(len(audit[0]['regions']),2)
        gm.w.fill_(2)
        a,b=torch.randn(2,8),torch.randn(2,8)
        expected=F.layer_norm(a+b,(8,),gm.w,gm.b)
        with torch.no_grad():actual=gm(a,b)
        torch.testing.assert_close(actual['norm'],expected)

    def test_vendor_linear_capture_without_environment_flag(self):
        try:import torch_xmlir
        except ImportError:self.skipTest('Vendor environment only')
        import importlib
        module=importlib.import_module('torch_xmlir.nn.linear')
        before=module._dynamo_workaround_enabled
        self.assertEqual(enable_vendor_capture(),'vendor_opaque_linear')
        self.assertEqual(module._dynamo_workaround_enabled,before)
        model=torch.nn.Linear(8,8).eval().requires_grad_(False)
        graphs=[]
        def backend(gm,inputs):graphs.append(gm);return gm.forward
        with torch.no_grad():
            compiled=torch.compile(model,backend=backend,fullgraph=True)
            x=torch.randn(2,8);torch.testing.assert_close(compiled(x),model(x))
        self.assertEqual(len(graphs),1)

    def test_mask_mutation_precedes_dtype_copy(self):
        # CLIP's causal-mask sequence: topological sorting can illegally move
        # to(float16) ahead of masked_fill_, whereas to(float32) aliases mask
        # and happens to hide that bug. Test both dtype behaviors.
        for dtype in (torch.float32,torch.float16):
            graph=fx.Graph();a=graph.placeholder('a');b=graph.placeholder('b')
            mask=graph.call_function(torch.full,((4,4),-1000.0))
            row=graph.call_function(torch.arange,(4,))
            limit=graph.call_function(operator.add,(row,1))
            view=graph.call_method('view',(limit,4,1))
            cond=graph.call_function(operator.lt,(row,view))
            graph.call_method('masked_fill_',(mask,cond,0))
            narrow=graph.call_method('to',(mask,dtype))
            add=graph.call_function(operator.add,(a,b))
            norm=graph.call_function(F.layer_norm,(add,(8,)))
            graph.output((narrow,norm));gm=fx.GraphModule(torch.nn.Module(),graph)
            inputs=(torch.randn(2,8),torch.randn(2,8));expected=gm(*inputs);audit=[]
            fn=partition_backend(tolerance=dict(atol=.0003,rtol=.0003),environment_key='mutation',audit=audit)(gm,inputs)
            with torch.no_grad():actual=fn(*inputs)
            torch.testing.assert_close(actual,expected)
            targets=[n.target for n in gm.graph.nodes]
            self.assertLess(targets.index('masked_fill_'),targets.index('to'))
            self.assertEqual(len(audit[0]['regions']),1)

    def test_intervening_mutation_prevents_fusion(self):
        class Model(torch.nn.Module):
            def forward(self,a,b):
                z=a+b
                a.mul_(2)
                return F.layer_norm(z,(8,))
        gm=fx.symbolic_trace(Model());matches,rejected=find_regions(gm)
        self.assertEqual(matches,[])
        self.assertEqual(rejected[0]['reason'],'intervening original operation')


if __name__=='__main__':unittest.main()
