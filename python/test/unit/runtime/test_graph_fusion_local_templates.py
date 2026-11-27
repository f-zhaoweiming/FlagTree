"""Reuse old matchers without losing live outputs, mutations or dtype edges."""
import operator
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import fx
import torch.nn.functional as F

from triton.experimental.graph_fusion import _template_local_fusion as local
from triton.experimental.graph_fusion._patterns import default_registry
from triton.experimental.graph_fusion._whole_model_selection import select_whole_model


class Residual(torch.nn.Module):
    def forward(self, x, bias):
        s = x + bias
        y = F.layer_norm(s, (4,))
        return {'y':y, 'residual':s, 'reuse':s * 2}


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    def forward(self, x):
        s = x + self.weight
        return F.layer_norm(s, (4,)), s


class TemplateLocalTests(unittest.TestCase):
    def test_region_packet_rejects_mutation_extra_launch_and_escaping_view(self):
        from triton.experimental.graph_fusion._p800_region_launch import _plan
        source='''def call(args):
    a,b,w,t=args
    args.clear()
    assert_size_stride(a,(3,4),(4,1))
    assert_size_stride(b,(3,4),(4,1))
    assert_size_stride(w,(4,),(1,))
    assert_size_stride(t,(4,),(1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        out=empty_strided_cuda((2,3,4),(12,4,1),torch.float16)
        stream=get_raw_stream(0)
        kernel.run(a,b,w,t,out,1,1,1,stream=stream)
        del a
        del b
        del w
        del t
    return (reinterpret_tensor(out,(3,4),(4,1),0),reinterpret_tensor(out,(3,4),(4,1),12))
'''
        self.assertIsNotNone(_plan(source,'kernel'))
        for changed in (source.replace('    args.clear()', '    args.clear()\n    a.add_(1)'),
                        source.replace('        del a', '        kernel.run(a,b,w,t,out,1,1,1,stream=stream)\n        del a'),
                        source.replace('(4,1),12)', '(4,1),13)'),
                        source.replace('torch.float16','torch.int16'),
                        source.replace('kernel.run(a,b,w,t', 'kernel.run(b,a,w,t')):
            self.assertIsNone(_plan(changed,'kernel'))

    def graph(self, fn):
        return fx.symbolic_trace(fn)

    def test_all_registry_families_are_discovered(self):
        cases = {
            'matmul_bias':lambda x,w,b:x @ w + b,
            'matmul_bias_relu':lambda x,w,b:torch.relu(x @ w + b),
            'matmul_bias_gelu':lambda x,w,b:F.gelu(x @ w + b),
            'matmul_bias_silu':lambda x,w,b:F.silu(x @ w + b),
            'add_gelu':lambda x,b:F.gelu(x+b),
            'add_silu':lambda x,b:F.silu(x+b),
            'add_softmax':lambda x,b:F.softmax(x+b,dim=-1),
            'add_layernorm':lambda x,b:F.layer_norm(x+b,(4,)),
            'mul_sum':lambda x,b:torch.sum(x*b,dim=-1),
        }
        self.assertEqual(set(cases), {t.template_id for t in default_registry().templates})
        for expected, fn in cases.items():
            with self.subTest(template=expected):
                selected,_ = local.discover_regions(self.graph(fn))
                self.assertEqual([t.template_id for t,_,_ in selected],[expected])

    def test_semantic_variants_keep_options(self):
        g=self.graph(lambda x,b:F.gelu(x+b,approximate='tanh'))
        self.assertEqual(local.discover_regions(g)[0][0][1].kind,'add_gelu_tanh')
        g=self.graph(lambda x,b:torch.sum(x*b,dim=-1,keepdim=True))
        self.assertEqual(local.discover_regions(g)[0][0][1].kind,'mul_sum_keepdim')
        g=self.graph(lambda x,b,w,s:F.layer_norm(x+b,(4,),w,s,eps=.01))
        match=local.discover_regions(g)[0][0][1]
        self.assertEqual(match.kind,'add_layernorm_affine')
        self.assertEqual(dict(match.options)['eps'],.01)

    def test_external_intermediate_is_returned_and_all_consumers_preserved(self):
        model=Residual().eval();gm=fx.symbolic_trace(model);original=gm.code;record={}
        changed=local.partition_regions(gm,('add_layernorm',),lambda g,_:g,record)
        self.assertEqual(gm.code,original)
        self.assertEqual(len(record['regions']),1)
        self.assertEqual(len(record['regions'][0]['outputs']),2)
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            for shape in ((2,4),(2,3,4)):
                x=torch.randn(shape).to(dtype);b=torch.randn(4).to(dtype)
                torch.testing.assert_close(changed(x,b),model(x,b),atol=0,rtol=0)

    def test_no_reordering_across_external_mutation(self):
        def fn(x,b):
            s=x+b
            x.add_(1)
            return F.layer_norm(s,(4,)),x
        gm=self.graph(fn);selected,rejected=local.discover_regions(gm)
        self.assertEqual(selected,[])
        self.assertEqual(rejected[0]['reason'],'intervening_computation_or_side_effect')

    def test_side_effect_after_region_keeps_original_order(self):
        def fn(x,b):
            s=x+b
            n=F.layer_norm(s,(4,))
            s.add_(3)
            return n,s,x
        gm=self.graph(fn);record={}
        changed=local.partition_regions(gm,('add_layernorm',),lambda g,_:g,record)
        x=torch.randn(3,4);b=torch.randn(4)
        torch.testing.assert_close(changed(x.clone(),b),fn(x.clone(),b),atol=0,rtol=0)
        self.assertEqual(len(record['regions'][0]['outputs']),2)

    def test_no_out_or_inplace_activation_is_replaced(self):
        for fn in (lambda x,b:F.silu(x+b,inplace=True), lambda x,b:F.relu(x@b,inplace=True)):
            self.assertFalse(local.discover_regions(self.graph(fn))[0])

    def test_quickgelu_can_join_registry_regions(self):
        def fn(x,b):
            s=x+b
            y=F.layer_norm(s,(4,))
            return y*torch.sigmoid(1.702*y),s
        gm=self.graph(fn);selected,_=local.discover_regions(gm)
        self.assertEqual({t.template_id for t,_,_ in selected},{'add_layernorm','quick_gelu'})
        changed=local.partition_regions(gm,('add_layernorm','quick_gelu'),lambda g,_:g,{})
        x=torch.randn(2,4);b=torch.randn(4)
        torch.testing.assert_close(changed(x,b),fn(x,b),atol=0,rtol=0)

    def test_matrix_and_relu_keep_native_boundaries(self):
        from triton.experimental.graph_fusion._operator_precision import _matmul,_relu
        gm=self.graph(lambda x,w,b:torch.relu(x@w+b))
        proof=local._protect_contractions(gm)
        self.assertEqual([x['native'] for x in proof],['matmul','relu'])
        targets={n.target for n in gm.graph.nodes}
        self.assertIn(_matmul._opoverload,targets);self.assertIn(_relu._opoverload,targets)
        for dtype in (torch.float16,torch.float32,torch.bfloat16):
            x=torch.randn(2,4).to(dtype);w=torch.randn(4,4).to(dtype);b=torch.randn(4).to(dtype)
            torch.testing.assert_close(gm(x,w,b),torch.relu(x@w+b),atol=0,rtol=0)

    def test_keyword_matrix_and_relu_keep_the_same_registry_contract(self):
        gm=self.graph(lambda x,w,b:F.relu(input=torch.mm(input=x,mat2=w)+b))
        self.assertEqual(local.discover_regions(gm)[0][0][0].template_id,'matmul_bias_relu')
        local._protect_contractions(gm)
        x=torch.randn(2,4);w=torch.randn(4,4);b=torch.randn(4)
        torch.testing.assert_close(gm(x,w,b),torch.relu(torch.mm(x,w)+b),atol=0,rtol=0)

    def test_args_guard_rejects_special_context_before_device_or_pointer_use(self):
        with torch.enable_grad():
            self.assertFalse(local._supported_args((torch.ones(4),)))
        with torch.no_grad():
            self.assertFalse(local._supported_args((torch.ones(4),))) # CPU
            self.assertFalse(local._supported_args(()))
            self.assertFalse(local._supported_args((3,)))

    def patches(self):
        def supported(args):
            return (all(type(x) in (torch.Tensor,torch.nn.Parameter) and not x.requires_grad for x in args)
                    and not torch.is_grad_enabled())
        def compile(gm,args,size,record):
            record.update(state='compiled',kernel_count=1,buffer_bytes=size)
            return gm.forward
        return (patch.object(local,'_p800_model',return_value=True),
                patch.object(local,'_supported_args',side_effect=supported),
                patch.object(local,'compile_region',side_effect=compile))

    def test_cached_layout_checks_live_aliases_offsets_and_context(self):
        # CPU tensors stand in for a previously admitted layout. Production
        # admission remains CUDA-only; this exercises the hit predicate itself.
        with torch.no_grad():
            a,b=torch.randn(4,8),torch.randn(4,8)
            signature=local._key((a,b))
            self.assertTrue(local._matches_args((a,b),signature))
            self.assertTrue(local._matches_args((a+1,b+1),signature))
            self.assertFalse(local._matches_args((a,a),signature))
            self.assertFalse(local._matches_args((a.flatten()[1:].view(1,31),b),signature))
            self.assertFalse(local._matches_args((a.t(),b),signature))
            self.assertFalse(local._matches_args((a.double(),b),signature))
            self.assertFalse(local._matches_args((a,),signature))
            self.assertFalse(local._matches_args((a,3),signature))
            with patch.object(local,'_ambient_supported',return_value=False):
                self.assertFalse(local._matches_args((a,b),signature))
            with torch.inference_mode():
                self.assertFalse(local._matches_args((a,b),signature))
        with torch.enable_grad():
            self.assertFalse(local._matches_args((a,b),signature))

    def test_factory_makes_generated_routes_and_keeps_live_weights_and_inputs(self):
        model=Layer().eval();x=torch.randn(2,4);record={};p=self.patches()
        with torch.no_grad(),p[0],p[1],p[2] as compile:
            routes=local.template_local_candidate_routes(model,prepare=lambda x:((x,),{}),
                validation_inputs=[((x,),{})],record=record)
            self.assertEqual(set(routes), {
                'local_templates_add_layernorm_'+suffix
                for suffix,_,_ in local._template_schedules(('add_layernorm',),(512,1024))})
            self.assertEqual(len(routes),11)
            self.assertTrue(all(v['complete_output_validation_required']
                                for v in record['implementations'].values()))
            before=compile.call_count
            for route in routes.values():
                torch.testing.assert_close(route.call(x),model(x),atol=0,rtol=0)
            model.weight.add_(1)
            for route in routes.values():
                fresh=x+1
                torch.testing.assert_close(route.call(fresh),model(fresh),atol=0,rtol=0)
                changed=torch.randn(3,4)
                torch.testing.assert_close(route.call(changed),model(changed),atol=0,rtol=0)
            self.assertEqual(compile.call_count,before)
            self.assertNotIn('native_original_dtype',routes)

    def test_native_cached_guard_retains_alias_offset_mode_and_live_inputs(self):
        with torch.no_grad():
            a,b=torch.randn(4,8),torch.randn(4,8)
            check=local._make_args_guard((a,b))
            self.assertTrue(check((a,b)))
            self.assertTrue(check((a+1,b+1)))
            self.assertFalse(check((a,a)))
            self.assertFalse(check((torch.randn(64)[32:].view(4,8),b)))
            self.assertFalse(check((a.t(),b)))
            self.assertFalse(check((a.double(),b)))
            self.assertFalse(check((a,)))
            self.assertFalse(check((a,3)))
            self.assertFalse(check((a.requires_grad_(),b)))
            a.requires_grad_(False)
            self.assertFalse(check((torch._neg_view(a),b)))
            with torch.inference_mode():
                self.assertFalse(check((a,b)))
            with patch.object(local,'_ambient_supported',return_value=False):
                self.assertFalse(check((a,b)))
        with torch.enable_grad():
            self.assertFalse(check((a,b)))

    def test_row_alternatives_keep_precision_changes_separate(self):
        self.assertEqual(local._template_schedules(('add_gelu',),(512,)), [('lm512',512,{})])
        choices=local._template_schedules(('add_layernorm',),(512,1024))
        self.assertEqual(choices[:2],[('lm512',512,{}),('lm1024',1024,{})])
        self.assertTrue(any(v.get('row_shared') and n==2048 for _,n,v in choices))
        self.assertTrue(any(v.get('reduction_reuse') and not v.get('reduction_tree') for _,_,v in choices))
        self.assertTrue(any(v.get('reduction_tree') for _,_,v in choices))
        self.assertTrue(any(v.get('row_shared') and v.get('reduction_tree')
                            and not v.get('reduction_reuse') for _,_,v in choices))
        self.assertTrue(any(v.get('reduction_reuse') and v.get('reduction_half_cache')
                            for _,_,v in choices))
        with self.assertRaises(ValueError):local._template_schedules(('add_layernorm',),(4096,))

    def test_ordered_moments_preserves_dual_outputs_and_rounding(self):
        from triton.experimental.graph_fusion._p800_ordered_layernorm import rewrite
        # A single reused Add is the actual region contract.
        class Reused(torch.nn.Module):
            def forward(self,a,b,w,t):
                z=a+b
                return F.layer_norm(z,(64,),w,t,1e-5),z
        gm=fx.symbolic_trace(Reused())
        args=(torch.randn(2,3,64).half(),torch.randn(64).half(),torch.randn(64).half(),torch.randn(64).half())
        want=gm(*args)
        proof=rewrite(gm,args)
        self.assertTrue(proof['dual_output'])
        actual=gm(*args)
        for a,b in zip(actual,want):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(sum(n.target is torch.ops.flagtree_p800.ordered_add_layernorm.default for n in gm.graph.nodes),1)

    def test_ordered_moments_rejects_unsupported_precision_and_broadcast(self):
        from triton.experimental.graph_fusion._p800_ordered_layernorm import rewrite
        for dtype,shape in ((torch.float32,(64,)),(torch.float16,(2,1,64))):
            gm=fx.symbolic_trace(lambda a,b,w,t:F.layer_norm(a+b,(64,),w,t,1e-5))
            args=(torch.randn(2,3,64).to(dtype),torch.randn(shape).to(dtype),torch.ones(64).to(dtype),torch.zeros(64).to(dtype))
            before=gm.code
            self.assertIsNone(rewrite(gm,args))
            self.assertEqual(gm.code,before)

    def test_full_output_rejection_does_not_replace_with_eager(self):
        model=Layer().eval();x=torch.randn(2,4);p=self.patches();record={}
        with torch.no_grad(),p[0],p[1],p[2]:
            routes=local.template_local_candidate_routes(model,prepare=lambda x:((x,),{}),
                validation_inputs=[((x,),{})],record={},buffer_sizes=(512,))
            def reject(name,out,*args):raise AssertionError('public residual error increased')
            with self.assertRaisesRegex(RuntimeError,'No complete model route'):
                select_whole_model(routes,validation_inputs=[((x,),{})],validate=reject,
                    measure=lambda _:self.fail('rejected output must not be timed'),record=record)
            self.assertEqual(record['state'],'no_valid_route')

    def test_recognized_but_unsupported_runtime_is_not_a_native_candidate(self):
        model=Layer().eval();x=torch.randn(2,4);record={}
        with torch.no_grad(),patch.object(local,'_p800_model',return_value=True),patch.object(
                local,'_supported_args',return_value=False):
            routes=local.template_local_candidate_routes(model,prepare=lambda x:((x,),{}),
                validation_inputs=[((x,),{})],record=record)
        self.assertEqual(routes,{})
        self.assertEqual({v['name'] for v in record['rejected_implementations']}, {
            'local_templates_add_layernorm_'+suffix
            for suffix,_,_ in local._template_schedules(('add_layernorm',),(512,1024))})

    def test_explicit_narrowing_survives_precision_preserving_trace(self):
        from triton.experimental.graph_fusion._inference import _trace_compiler_graph
        class Roundtrip(torch.nn.Module):
            def forward(self, x):
                return x.half().float(), x.bfloat16().float()
        x=torch.tensor([1.00048828125, 1.00390625, -1.00048828125])
        with torch.no_grad():
            graph=_trace_compiler_graph(Roundtrip(),(x,),emulate_precision=True)
            casts=[n for n in graph.graph.nodes if n.op=='call_function'
                   and n.target is torch.ops.prims.convert_element_type.default]
            narrow=[n for n in casts if n.args[1] in (torch.float16,torch.bfloat16)]
            self.assertEqual(len(narrow),2)
            self.assertTrue(all(n.meta.get('low_precision_pointwise_barrier') for n in narrow))
            self.assertTrue(all(not n.meta.get('low_precision_pointwise_barrier')
                                for n in casts if n.args[1]==torch.float32))
            torch.testing.assert_close(graph(x),Roundtrip()(x),atol=0,rtol=0)
            self.assertTrue(all(not torch.equal(y,x) for y in graph(x)))


if __name__=='__main__':unittest.main()
