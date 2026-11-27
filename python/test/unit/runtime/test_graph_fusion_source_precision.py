import unittest
import torch
import torch.nn.functional as F
from torch import fx
from triton.experimental.graph_fusion._source_precision import source_mixed_candidate, SourceParameterPack, matches


def graph(args,paths=('model.vision_model.encoder.layers.11.fc','model.vision_model.encoder.layers.1.fc')):
    g=fx.Graph();x,w,b=[g.placeholder(n) for n in ('x','w','b')]
    a=g.call_function(F.linear,(x,w,b));z=g.call_function(F.linear,(x,w,b));g.output((a,z,x,w))
    gm=fx.GraphModule({},g)
    for n,p in zip((a,z),paths):n.meta['nn_module_stack']={p:(p,torch.nn.Linear)}
    class Values(fx.Interpreter):
        def run_node(self,n):
            y=super().run_node(n);n.meta['example_value']=y;return y
    Values(gm).run(*args)
    return gm


class SourcePrecisionTests(unittest.TestCase):
    def args(self):
        return (torch.tensor([[1.0001,2.0002]]),torch.nn.Parameter(torch.eye(2),requires_grad=False),
                torch.nn.Parameter(torch.zeros(2),requires_grad=False))

    def test_fp32_ingress_and_protected_scope_not_restored_from_half(self):
        args=self.args();gm=graph(args)
        # A public parameter alias stays FP32 and cannot be cached as low.
        ret=next(n for n in gm.graph.nodes if n.op=='output');ret.args=((ret.args[0][0],ret.args[0][1],ret.args[0][2]),)
        gm.recompile()
        _,ex,indices,records=source_mixed_candidate(gm,args,fp16_modules=('vision_model.encoder',),fp32_modules=('vision_model.encoder.layers.11',))
        wrapped=SourceParameterPack(gm.forward,len(args),indices)
        a,z,x=wrapped(*args)
        torch.testing.assert_close(a,F.linear(*args),atol=0,rtol=0)
        torch.testing.assert_close(z,F.linear(*(v.half() for v in args)).float(),atol=0,rtol=0)
        self.assertFalse(torch.equal(a,z));self.assertIs(x,args[0])
        self.assertEqual(len(records),1);self.assertEqual(len(indices),2)
        self.assertEqual(args[1].dtype,torch.float32)
        self.assertTrue(all(x.dtype==torch.float16 for x in ex[-2:]))
        old=wrapped.pack.entries[indices.index(1)+len(args)][2]
        wrapped(*args);self.assertIs(old,wrapped.pack.entries[len(args)][2])
        args[1].add_(1)
        a,z,_=wrapped(*args);self.assertIsNot(old,wrapped.pack.entries[len(args)][2])
        torch.testing.assert_close(z,F.linear(*(v.half() for v in args)).float(),atol=0,rtol=0)

    def test_strict_fake_mode_with_real_frozen_parameters(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        args=self.args();gm=graph(args)
        out=next(n for n in gm.graph.nodes if n.op=='output');out.args=((out.args[0][0],out.args[0][1]),);gm.recompile()
        mode=FakeTensorMode()
        examples=(mode.from_tensor(args[0]),*args[1:])
        _,result,idx,rows=source_mixed_candidate(gm,examples,fp16_modules=('vision_model.encoder',))
        self.assertEqual(len(rows),2);self.assertEqual(len(idx),2)
        self.assertEqual(result[-1].dtype,torch.float16)

    def test_escaping_weight_alias_kept_and_not_demoted(self):
        args=self.args();gm=graph(args)
        _,ex,idx,rows=source_mixed_candidate(gm,args,fp16_modules=('vision_model.encoder',))
        self.assertEqual(rows,[]);self.assertEqual(idx,());self.assertEqual(len(ex),3)
        self.assertIs(gm(*args)[3],args[1])

    def test_already_rounded_source_rejected(self):
        args=self.args();gm=graph(args)
        with self.assertRaisesRegex(ValueError,'original FP32'):
            source_mixed_candidate(gm,[args[0].half(),*args[1:]],fp16_modules=('vision_model.encoder',))



    def test_scope_boundaries_and_unknown_module(self):
        self.assertTrue(matches('model.encoder.layers.11.fc',('encoder.layers.11',)))
        self.assertFalse(matches('model.encoder.layers.110.fc',('encoder.layers.11',)))
        args=self.args();gm=graph(args,('unknown','unknown'))
        _,_,idx,rows=source_mixed_candidate(gm,args,fp16_modules=('encoder',))
        self.assertEqual(rows,[]);self.assertEqual(idx,())

    def test_activation_changes_and_parameter_replacement_invalidate_copies(self):
        args=self.args();gm=graph(args);o=next(n for n in gm.graph.nodes if n.op=='output');o.args=((o.args[0][0],o.args[0][1]),);gm.recompile()
        _,_,idx,_=source_mixed_candidate(gm,args,fp16_modules=('vision_model.encoder',))
        wrapped=SourceParameterPack(gm.forward,3,idx);a=wrapped(*args)[0]
        x=args[0]+1;b=wrapped(x,*args[1:])[0];self.assertFalse(torch.equal(a,b))
        w=torch.nn.Parameter(torch.eye(2)*2,requires_grad=False)
        got=wrapped(x,w,args[2])[0];want=F.linear(x.half(),w.half(),args[2].half()).float()
        torch.testing.assert_close(got,want,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
