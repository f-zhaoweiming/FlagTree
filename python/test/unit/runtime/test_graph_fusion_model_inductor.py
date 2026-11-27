"""Graph semantics for inference whole-graph exposure, executable without P800."""
import copy
import operator
import unittest

import torch
import torch.nn.functional as F
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp
from triton.experimental.graph_fusion._model_inductor import expose_fp32_patterns, expose_linear, _fp32_reference


def trace(model, args):
    gm = fx.symbolic_trace(model)
    class Values(fx.Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if isinstance(out, torch.Tensor):
                n.meta['example_value'] = out
            return out
    Values(gm).run(*args)
    return gm


class WholeGraphTests(unittest.TestCase):
    def test_native_row_bias_candidate_keeps_fp32_order_and_strided_weights(self):
        from triton.experimental.graph_fusion._p800_gemm_candidates import _native_row_bias_mm
        torch.manual_seed(53)
        a = torch.randn(17, 32)
        w = torch.randn(23, 32)
        bias = torch.randn(23)
        for rhs in (w.t(), w.t().contiguous()):
            out = torch.empty(17, 23)
            expected = torch.mm(a, rhs) + bias
            actual = _native_row_bias_mm(bias.expand(17, 23), a, rhs, out=out)
            self.assertIs(actual, out)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaises(AssertionError):
            _native_row_bias_mm(bias.expand(17, 23), a, w.t(), out=out, beta=0)

    def test_sdnn_matrix_epilogue_retains_nonlinear_cluster_boundary(self):
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._p800_gemm_candidates import template_boundary_eligibility
        def candidate(operations):
            body=SimpleNamespace(get_nodes=lambda:[SimpleNamespace(op='call_method',target=x) for x in operations])
            return SimpleNamespace(get_nodes=lambda:[SimpleNamespace(_body=body)])
        self.assertTrue(template_boundary_eligibility(candidate(['load','add','add','to_dtype','store']))[0])
        self.assertFalse(template_boundary_eligibility(candidate(['load','mul','sigmoid','mul','store']))[0])
        self.assertFalse(template_boundary_eligibility(candidate(['load','exp','store']))[0])
        self.assertFalse(template_boundary_eligibility(candidate(['unknown_future_op']))[0])
        self.assertFalse(template_boundary_eligibility(SimpleNamespace(get_nodes=lambda:[object()]))[0])

    def test_sdnn_epilogue_accepts_real_inductor_index_bookkeeping(self):
        import sympy
        from types import SimpleNamespace
        from torch._inductor.loop_body import LoopBody
        from torch._inductor.virtualized import V, ops
        from torch._inductor.sizevars import SizeVarAllocator
        from triton.experimental.graph_fusion._p800_gemm_candidates import template_boundary_eligibility
        i=sympy.Symbol('i',integer=True)
        def body(index,reduction):
            x=ops.load('a',index[0])
            ops.store('b',index[0],ops.add(x,ops.constant(1,torch.float32)))
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            loop=LoopBody(body,([i],[]),{i:16},[i],[])
        node=SimpleNamespace(get_nodes=lambda:[SimpleNamespace(_body=loop)])
        self.assertTrue(template_boundary_eligibility(node)[0])

    def test_p800_gemm_candidates_do_not_change_low_precision_or_small_matrices(self):
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._p800_gemm_candidates import eligible_shape
        def node(shape, dtype=torch.float32):
            return SimpleNamespace(get_size=lambda:shape,get_dtype=lambda:dtype)
        self.assertEqual(eligible_shape(node([5770,1024]),node([1024,4096])),(5770,4096,1024))
        self.assertIsNone(eligible_shape(node([5770,1024],torch.float16),node([1024,4096],torch.float16)))
        self.assertIsNone(eligible_shape(node([64,1024]),node([1024,4096])))
        self.assertIsNone(eligible_shape(node([10,577,1024]),node([10,1024,577])))
        self.assertIsNone(eligible_shape(node([5770,1024]),node([512,4096])))

    def test_p800_gemm_tiles_preserve_existing_heuristic_state(self):
        from dataclasses import dataclass
        from triton.experimental.graph_fusion._p800_gemm_candidates import extended_configs
        @dataclass(frozen=True)
        class Config:
            block_m:int=128
            block_n:int=128
            block_k:int=64
            num_stages:int=3
            num_warps:int=8
        original=[Config()];expanded=extended_configs(original)
        self.assertEqual(original,[Config()])
        self.assertIs(expanded[0],original[0])
        self.assertEqual(len(expanded),8)
        self.assertIn(Config(256,256,128),expanded)
        self.assertEqual(extended_configs(expanded),expanded)

    def test_p800_gemm_scope_restores_lowerings_and_memory_heuristic_on_error(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        from types import SimpleNamespace
        from torch._inductor import config
        from torch._inductor.lowering import lowerings
        from torch._inductor.kernel import mm as registration
        from torch._inductor.scheduler import Scheduler
        from torch._inductor.template_heuristics.triton import BaseConfigHeuristic
        from triton.experimental.graph_fusion._p800_gemm_candidates import p800_gemm_candidates
        nodes=[SimpleNamespace(get_size=lambda:[1024,1024],get_dtype=lambda:torch.float32)]*2
        original_checker=BaseConfigHeuristic._get_exceeding_shared_memory_checker
        original_configs=BaseConfigHeuristic.get_mm_configs
        original_fuse=Scheduler.can_fuse
        original_speedup=Scheduler.speedup_by_fusion
        def fail(*args,**kw):
            self.assertTrue(config.max_autotune_gemm)
            self.assertEqual(config.max_autotune_gemm_backends,'ATEN,TRITON')
            self.assertIsNone(BaseConfigHeuristic._get_exceeding_shared_memory_checker(None))
            raise ValueError('deliberate lowering failure')
        audit=[]
        with patch.dict(lowerings,{torch.ops.aten.mm.default:fail}), \
             patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',return_value=True), \
             patch('triton.experimental.graph_fusion._p800_matmul.native_template_eligibility',return_value=nullcontext()), \
             config.patch({'max_autotune_gemm':False}):
            with self.assertRaisesRegex(ValueError,'deliberate'):
                with p800_gemm_candidates(torch.device('cuda:0'),audit):
                    template=SimpleNamespace(is_template=lambda:True)
                    nonlinear=SimpleNamespace(is_template=lambda:False,get_nodes=lambda:[SimpleNamespace(_body=SimpleNamespace(
                        get_nodes=lambda:[SimpleNamespace(op='call_method',target='sigmoid')]))])
                    # Even a template shortcut in the original scheduler must
                    # not reach its nonlinear epilogue path.
                    self.assertFalse(Scheduler.can_fuse(object(),template,nonlinear))
                    self.assertFalse(Scheduler.can_fuse(object(),nonlinear,template))
                    lowerings[torch.ops.aten.mm.default](*nodes)
            self.assertIs(lowerings[torch.ops.aten.mm.default],fail)
            self.assertFalse(config.max_autotune_gemm)
        self.assertEqual(len(audit),3)
        self.assertIs(Scheduler.can_fuse,original_fuse)
        self.assertIs(Scheduler.speedup_by_fusion,original_speedup)
        self.assertIs(BaseConfigHeuristic._get_exceeding_shared_memory_checker,original_checker)
        self.assertIs(BaseConfigHeuristic.get_mm_configs,original_configs)

    def test_p800_full_k_retains_one_dot_after_native_preprocessing(self):
        from types import SimpleNamespace
        from torch._inductor import config
        from torch._inductor.virtualized import V
        from torch._inductor.sizevars import SizeVarAllocator
        from triton.experimental.graph_fusion._p800_gemm_candidates import full_k_configs
        from torch._inductor.template_heuristics.triton import BaseConfigHeuristic
        heuristic = BaseConfigHeuristic()
        for k in (512, 1024):
            raw = full_k_configs(heuristic.mm_configs, k)
            self.assertTrue(raw)
            with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())), \
                    config.patch({'max_autotune_prune_choices_based_on_shared_mem': False}):
                configs = list(heuristic.preprocess_mm_configs(5770, 1024, k, raw))
            self.assertTrue(configs)
            self.assertTrue(all(c.kwargs['BLOCK_K'] >= k for c in configs))
            pipelined_groups = {c.kwargs['GROUP_M'] for c in configs
                                if c.kwargs['BLOCK_M'] == 64
                                and c.kwargs['BLOCK_N'] == 256
                                and c.num_stages == 2}
            self.assertEqual(pipelined_groups, {8, 32, 64})
        self.assertEqual(full_k_configs([], 1024), [])
        for k in (768, 3072, 4096):
            self.assertEqual(full_k_configs(heuristic.mm_configs, k), [])

    def test_full_k_epilogue_benchmarks_tiles_excluded_by_unfused_order(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from torch._inductor import ir, config
        from triton.experimental.graph_fusion._p800_gemm_candidates import benchmark_full_k_epilogue
        first, second, invalid = [Mock(spec=ir.TritonTemplateCallerBase) for _ in range(3)]
        for c, name in zip((first, second, invalid), ('first', 'second', 'invalid')):
            c.name=name; c.description=name
        multi=Mock(spec=ir.MultiTemplateBuffer)
        multi.get_dtype.return_value=torch.float32;multi.get_size.return_value=[5770,1024]
        multi.choice_timings.return_value={object():.10,first:.30,second:.40,invalid:float('nan')}
        multi.swap_as_triton_caller.side_effect=lambda _:nullcontext()
        multi._choice_timings={}
        n1=SimpleNamespace(is_template=lambda:True,get_template_node=lambda:multi,
            get_nodes=lambda:[SimpleNamespace(get_device=lambda:torch.device('cuda:0'))])
        n2=SimpleNamespace(is_template=lambda:False,get_device=lambda:torch.device('cuda:0'),get_nodes=lambda:[SimpleNamespace(_body=SimpleNamespace(
            get_nodes=lambda:[SimpleNamespace(op='call_method',target='add')]))])
        scheduler=Mock();scheduler.benchmark_fused_nodes.return_value=(.05,'pointwise.py')
        scheduler.benchmark_codegened_module.side_effect=[(.05,'pointwise.py'),(.20,'first.py'),(.09,'second.py')]
        audit=[]
        with config.patch({'multi_kernel_hints':[]}),patch('torch._inductor.codecache.PyCodeCache.load'):
            self.assertTrue(benchmark_full_k_epilogue(scheduler,n1,n2,audit))
        self.assertEqual(scheduler.benchmark_codegened_module.call_count,3)
        multi.finalize_as_triton_caller.assert_called_once_with(second)
        self.assertTrue(all(r['excluded_by_old_bound'] for r in audit[0]['candidates']))
        self.assertEqual(audit[0]['selected'],'second')
        self.assertEqual(multi._choice_timings[None],{first:.20,second:.09})

    def test_full_k_epilogue_rejects_slow_or_invalid_fused_results(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from torch._inductor import ir,config
        from triton.experimental.graph_fusion._p800_gemm_candidates import benchmark_full_k_epilogue
        c=Mock(spec=ir.TritonTemplateCallerBase);c.name='candidate';c.description='candidate'
        multi=Mock(spec=ir.MultiTemplateBuffer);multi.get_dtype.return_value=torch.float32
        multi.get_size.return_value=[5770,1024];multi.choice_timings.return_value={object():.10,c:.30}
        multi.swap_as_triton_caller.side_effect=lambda _:nullcontext();multi._choice_timings={}
        n1=SimpleNamespace(is_template=lambda:True,get_template_node=lambda:multi,
            get_nodes=lambda:[SimpleNamespace(get_device=lambda:torch.device('cuda:0'))])
        body=SimpleNamespace(get_nodes=lambda:[SimpleNamespace(op='call_method',target='to_dtype')])
        n2=SimpleNamespace(is_template=lambda:False,get_device=lambda:torch.device('cuda:0'),get_nodes=lambda:[SimpleNamespace(_body=body)])
        s=Mock();s.benchmark_fused_nodes.return_value=(.05,'pointwise.py')
        with config.patch({'multi_kernel_hints':[]}),patch('torch._inductor.codecache.PyCodeCache.load'):
            for value in (.2,0.,-1.,float('nan'),float('inf')):
                s.benchmark_codegened_module.side_effect=[(.05,'pointwise.py')]+[(value,'candidate.py')]*3
                self.assertFalse(benchmark_full_k_epilogue(s,n1,n2,[]))
            body.get_nodes=lambda:[SimpleNamespace(op='call_method',target='sigmoid')]
            self.assertIsNone(benchmark_full_k_epilogue(s,n1,n2,[]))
        multi.finalize_as_triton_caller.assert_not_called()
        self.assertEqual(multi._choice_timings,{})

    def test_full_k_epilogue_invalid_measurements_are_retried_and_evicted(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from triton.experimental.graph_fusion._p800_gemm_candidates import _valid_epilogue_timing
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'kernel.py';source.write_text('')
            cache=source.with_suffix('.kernel_perf');values=iter([-4.,float('nan'),.12])
            def measure(module,device):
                self.assertFalse(cache.exists())
                value=next(values);cache.write_text(str(value));return value,str(source)
            scheduler=SimpleNamespace(benchmark_codegened_module=measure);record={}
            with patch('torch._inductor.codecache.cache_dir',return_value=directory):
                value,_=_valid_epilogue_timing(scheduler,SimpleNamespace(__file__=str(source)),None,record)
            self.assertEqual(value,.12);self.assertEqual(float(cache.read_text()),.12)
            self.assertEqual(record['invalid_measurements'],['-4.0','nan'])

    def test_fixed_resource_graph_is_not_eliminated_by_inner_tensor_timing(self):
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from triton.experimental.graph_fusion._model_inductor import inductor_backend
        mode=FakeTensorMode()
        value=FakeTensor(mode,torch.empty(4,device='meta'),torch.device('cuda',0))
        audit=[];prepared=lambda x:x
        with patch('torch._inductor.compile_fx.compile_fx',return_value=prepared), \
             patch('torch._inductor.lowering.make_fallback'), \
             patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',return_value=True), \
             patch('triton.experimental.graph_fusion._model_inductor._SelectGraph',side_effect=AssertionError('inner selection')):
            result=inductor_backend(tolerance=dict(atol=.003,rtol=.003),environment_key='resource',audit=audit,
                resource_policy='sdnn',graph_choice='inductor_p800_lm2048')(fx.symbolic_trace(torch.nn.Identity()),(value,))
        self.assertIs(result,prepared)
        self.assertEqual(audit[0]['state'],'awaiting_complete_model_validation')
        self.assertTrue(audit[0]['requires_original_fp32_model_validation'])

    def test_sdnn_resource_eligibility_keeps_extent_masks_and_memory_layout(self):
        from triton.experimental.graph_fusion._p800_model_schedule import sdnn_pointwise_eligibility
        source='''
@triton_heuristics.fixed_config(config={'XBLOCK':32768})
@triton.jit
def generated(in_ptr0,out_ptr0,xnumel,XBLOCK):
    xoffset=tl.program_id(0)*XBLOCK
    xindex=xoffset+tl.arange(0,XBLOCK)[:]
    xmask=xindex<xnumel
    x0=xindex
    a=tl.load(in_ptr0+x0,xmask).to(tl.float32)
    b=a*tl.sigmoid(a*1.702)
    tl.store(out_ptr0+x0,b,xmask)
'''
        signature={'in_ptr0':'*fp16','out_ptr0':'*fp16','xnumel':'i32'}
        self.assertTrue(sdnn_pointwise_eligibility(source,signature)[0])
        from torch._inductor.utils import Placeholder
        template=source.replace('def generated(', 'def '+str(Placeholder.KERNEL_NAME)+'(')
        self.assertTrue(sdnn_pointwise_eligibility(template,signature)[0])
        self.assertIn(str(Placeholder.KERNEL_NAME),template)
        for changed in (source.replace('x0=xindex','x0=xindex%1024'),
                        source.replace('in_ptr0+x0,xmask','in_ptr0+x0'),
                        source.replace('tl.store(out_ptr0+x0,b,xmask)','tl.store(out_ptr0+x0,b,a>0)'),
                        source.replace('tl.sigmoid','libdevice.unreviewed')):
            self.assertFalse(sdnn_pointwise_eligibility(changed,signature)[0])
        self.assertFalse(sdnn_pointwise_eligibility(source,dict(signature,in_ptr0='*i64'))[0])

    def test_sdnn_option_survives_deferred_compile_and_restores_global_compile(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import triton
        from triton.experimental.graph_fusion._p800_model_schedule import p800_sdnn_config
        fn=object();seen=[]
        binary=SimpleNamespace(metadata=SimpleNamespace(buffer_size_limit=512,is_sdnn=True,effective_resource_kind='sdnn'))
        def compile(source,**kwargs):
            seen.append(kwargs['options']);return binary
        def fixed_config(**kwargs):
            self.assertEqual(kwargs['config']['XBLOCK'],131072)
            def decorate(function):
                return SimpleNamespace(_precompile_config=lambda:triton.compile(SimpleNamespace(fn=function),options={}))
            return decorate
        with patch('triton.compile',compile),patch('torch._inductor.runtime.triton_heuristics.fixed_config',fixed_config):
            tuner=p800_sdnn_config(config={'XBLOCK':32768},triton_meta={})(fn)
            self.assertEqual(seen,[])
            self.assertIs(tuner._precompile_config(),binary)
            self.assertEqual(seen,[dict(resource_kind='sdnn',buffer_size_limit=512)])
            self.assertIs(triton.compile,compile)

    def test_original_dtype_candidates_use_scoped_schedule_only_on_p800(self):
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from torch._inductor.virtualized import V
        from torch._inductor.codegen.triton import TritonKernel
        from triton.experimental.graph_fusion._model_inductor import inductor_backend
        mode = FakeTensorMode()
        fake = FakeTensor(mode, torch.empty(4, device='meta', dtype=torch.float16),
                          torch.device('cuda', 0))
        cpu = torch.empty(4, dtype=torch.float16)
        for policy in ('preserve_operator', 'fuse_fp32_internals'):
            for value, detected, expected in ((fake, True, True), (fake, False, False),
                                               (cpu, True, False)):
                with self.subTest(policy=policy, device=value.device, p800=detected):
                    gm = fx.symbolic_trace(torch.nn.Identity())
                    audit, observed = [], []
                    before, body = V.choices, TritonKernel.codegen_body
                    prepared = lambda x: x
                    def compile_candidate(graph, inputs, **kwargs):
                        observed.append(V.choices is not before)
                        self.assertEqual(inputs, (value,))
                        return prepared
                    with patch('torch._inductor.compile_fx.compile_fx', compile_candidate), \
                            patch('torch._inductor.lowering.make_fallback'), \
                            patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',
                                  return_value=detected):
                        result = inductor_backend(tolerance=dict(atol=.003, rtol=.003),
                            environment_key='schedule-dispatch', audit=audit,
                            precision_policy=policy)(gm, (value,))
                    self.assertEqual(observed, [expected])
                    self.assertIs(result, prepared)
                    self.assertIs(V.choices, before)
                    self.assertIs(TritonKernel.codegen_body, body)
                    self.assertEqual(audit[0]['pointwise_schedule'], 'p800_lm' if expected else 'default')
                    self.assertTrue(audit[0]['requires_original_fp32_model_validation'])
                    self.assertEqual(audit[0]['state'], 'awaiting_complete_model_validation')

    def test_original_dtype_compile_failure_restores_schedule_and_keeps_native_candidate(self):
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from torch._inductor.virtualized import V
        from torch._inductor.codegen.triton import TritonKernel
        from triton.experimental.graph_fusion._model_inductor import inductor_backend
        mode = FakeTensorMode()
        value = FakeTensor(mode, torch.empty(4, device='meta', dtype=torch.float16),
                           torch.device('cuda', 0))
        for policy in ('preserve_operator', 'fuse_fp32_internals'):
            with self.subTest(policy=policy):
                before, body = V.choices, TritonKernel.codegen_body
                audit, observed = [], []
                def fail_compile(*args, **kwargs):
                    observed.append(V.choices is not before)
                    raise RuntimeError('controlled compile failure')
                with patch('torch._inductor.compile_fx.compile_fx', fail_compile), \
                        patch('torch._inductor.lowering.make_fallback'), \
                        patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',
                              return_value=True):
                    result = inductor_backend(tolerance=dict(atol=.003, rtol=.003),
                        environment_key='schedule-failure', audit=audit,
                        precision_policy=policy)(fx.symbolic_trace(torch.nn.Identity()), (value,))
                self.assertEqual(observed, [True])
                self.assertIs(result(value), value)
                self.assertIs(V.choices, before)
                self.assertIs(TritonKernel.codegen_body, body)
                self.assertEqual(audit[0]['route'], 'original')
                self.assertIn('controlled compile failure', audit[0]['rejected'][0]['error'])

    def test_grouped_projection_complete_outputs_and_versioned_weights(self):
        from triton.experimental.graph_fusion._p800_model_gemm import grouped_projection_candidate, GroupedProjectionPack
        class Model(torch.nn.Module):
            def forward(self, x, w1, w2, w3, b1, b2, b3):
                q=torch.ops.aten.linear.default(x,w1,b1);k=torch.ops.aten.linear.default(x,w2,b2);v=torch.ops.aten.linear.default(x,w3,b3)
                q=q.view(2,7,3,4).transpose(1,2)
                k=k.view(2,7,3,4).transpose(1,2)
                v=v.view(2,7,3,4).transpose(1,2)
                return q.sin()+k.cos(),v.sigmoid(),x
        x=torch.randn(2,7,16)
        params=[torch.nn.Parameter(torch.randn(*shape),requires_grad=False)
                for shape in [(12,16)]*3+[(12,)]*3]
        original=Model();gm=trace(original,(x,*params));before=gm.code
        candidate=grouped_projection_candidate(gm,(x,*params))
        self.assertIsNotNone(candidate)
        body,examples,groups,records=candidate
        packed=GroupedProjectionPack(body.forward,groups)
        for fresh in (x,x+1):
            torch.testing.assert_close(packed(fresh,*params),original(fresh,*params))
        saved={key:value[2] for key,value in packed.entries.items()}
        params[1].add_(.25);params[5].sub_(.5)
        torch.testing.assert_close(packed(x,*params),original(x,*params))
        self.assertTrue(all(value[2] is not saved[key] for key,value in packed.entries.items()))
        self.assertEqual(gm.code,before)
        self.assertEqual(len(records),1)
        self.assertTrue(all(v[2].dtype==torch.float32 for v in packed.entries.values()))

    def test_grouped_projection_preserves_unused_result_mask_mutation(self):
        from triton.experimental.graph_fusion._p800_model_gemm import grouped_projection_candidate, GroupedProjectionPack
        class Model(torch.nn.Module):
            def forward(self, x, w1, w2, w3):
                mask = torch.ones_like(x[:, :, 0])
                mask.masked_fill_(mask > 0, 0)
                q = torch.ops.aten.linear.default(x, w1).view(2, 7, 3, 4).transpose(1, 2)
                k = torch.ops.aten.linear.default(x, w2).view(2, 7, 3, 4).transpose(1, 2)
                v = torch.ops.aten.linear.default(x, w3).view(2, 7, 3, 4).transpose(1, 2)
                return q + k + v + mask[:, None, :, None]
        x = torch.randn(2, 7, 16)
        params = [torch.nn.Parameter(torch.randn(12, 16), requires_grad=False) for _ in range(3)]
        model = Model()
        graph, examples, groups, records = grouped_projection_candidate(trace(model, (x, *params)), (x, *params))
        self.assertTrue(any(n.op == 'call_method' and n.target == 'masked_fill_' for n in graph.graph.nodes))
        torch.testing.assert_close(GroupedProjectionPack(graph.forward, groups)(x, *params), model(x, *params))

    def test_grouped_projection_keeps_shared_weights_and_public_layout(self):
        from triton.experimental.graph_fusion._p800_model_gemm import grouped_projection_candidate
        class Model(torch.nn.Module):
            def forward(self,x,w1,w2):return torch.ops.aten.linear.default(x,w1),torch.ops.aten.linear.default(x,w2)
        x=torch.randn(3,16);w1=torch.nn.Parameter(torch.randn(12,16),requires_grad=False)
        w2=torch.nn.Parameter(torch.randn(12,16),requires_grad=False)
        gm=trace(Model(),(x,w1,w2))
        self.assertIsNone(grouped_projection_candidate(gm,(x,w1,w2)))
        class Shared(torch.nn.Module):
            def forward(self,x,w1,w2):return torch.ops.aten.linear.default(x,w1).sin()+torch.ops.aten.linear.default(x,w1).cos()+w2.sum()
        gm=trace(Shared(),(x,w1,w2))
        self.assertIsNone(grouped_projection_candidate(gm,(x,w1,w2)))

    def test_split_addmm_restores_lowering_and_preserves_other_semantics(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._inductor.lowering import lowerings
        from torch._inductor.kernel import mm as _mm_registration
        from triton.experimental.graph_fusion._p800_model_schedule import _split_fp32_addmm
        op=torch.ops.aten.addmm.default;mm=torch.ops.aten.mm.default;add=torch.ops.aten.add.Tensor
        fp32=SimpleNamespace(get_dtype=lambda:torch.float32)
        half=SimpleNamespace(get_dtype=lambda:torch.float16)
        original=lambda *a,**kw:('original',kw)
        def native_add(target, **kwargs):
            self.assertIs(target, add)
            self.assertFalse(kwargs['add_to_fallback_set'])
            return lambda a,b:('sum',a,b)
        with patch.dict(lowerings,{op:original,mm:lambda *a:'product'}), \
                patch('torch._inductor.lowering.fallback_handler',side_effect=native_add):
            with self.assertRaisesRegex(RuntimeError,'restore'):
                with _split_fp32_addmm(True):
                    self.assertEqual(lowerings[op](fp32,fp32,fp32),('sum','product',fp32))
                    for args,kwargs in [((half,half,half),{}),((fp32,fp32,fp32),{'beta':0}),
                                        ((fp32,fp32,fp32),{'alpha':.5})]:
                        self.assertEqual(lowerings[op](*args,**kwargs)[0],'original')
                    raise RuntimeError('restore')
            self.assertIs(lowerings[op],original)

    def test_packed_contraction_preserves_bias_outputs_and_fresh_inputs(self):
        from triton.experimental.graph_fusion._p800_model_gemm import packed_weight_candidate, pack_column_major
        class Model(torch.nn.Module):
            def forward(self, x, w, b):
                y = torch.ops.aten.linear.default(x, w, b)
                return y, y.sigmoid(), x
        x = torch.randn(8,512,512)
        w = torch.nn.Parameter(torch.randn(128,512),requires_grad=False)
        b = torch.randn(128)
        gm = trace(Model(), (x,w,b))
        candidate=packed_weight_candidate(gm,(x,w,b))
        self.assertIsNotNone(candidate)
        gm,examples,indices,records=candidate
        self.assertEqual(indices,[1]);self.assertEqual(len(records),1)
        self.assertEqual(examples[1].stride(),(1,128))
        for args in ((x,w,b),(x+1,w*.5,b-2)):
            packed=(args[0],pack_column_major(args[1]),args[2])
            torch.testing.assert_close(gm(*packed),Model()(*args),atol=1e-4,rtol=1e-4)
            self.assertEqual(gm(*args)[2].data_ptr(),args[0].data_ptr())
            self.assertTrue(gm(*args)[0].is_contiguous())
            self.assertEqual(gm(*args)[0].view(-1).numel(),8*512*128)

    def test_backend_retune_requires_forwarding_proof(self):
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._p800_model_schedule import _validate_buffer_option
        for actual in (16,512,1024,2048):
            binary=SimpleNamespace(metadata=SimpleNamespace(buffer_size_limit=actual))
            with self.assertRaisesRegex(RuntimeError,'not applied'):
                _validate_buffer_option(binary,2048,[])
            result=_validate_buffer_option(binary,2048,[binary])
            self.assertEqual(result['actual'],actual)
            self.assertEqual(result['retuned'],actual!=2048)
            self.assertEqual(_validate_buffer_option(binary,2048,[]),result)
        for actual in (None,0,8,1000,4096,True):
            binary=SimpleNamespace(metadata=SimpleNamespace(buffer_size_limit=actual))
            with self.assertRaisesRegex(RuntimeError,'retuned legally'):
                _validate_buffer_option(binary,2048,[binary])

    def test_packed_contraction_keeps_other_dtypes_and_small_shapes(self):
        from triton.experimental.graph_fusion._p800_model_gemm import packed_weight_candidate
        class Model(torch.nn.Module):
            def forward(self, x, w):return torch.ops.aten.linear.default(x,w)
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            x=torch.randn(32,64).to(dtype);w=torch.randn(16,64).to(dtype)
            gm=trace(Model(),(x,w));before=gm.code
            self.assertIsNone(packed_weight_candidate(gm,(x,w)))
            self.assertEqual(gm.code,before)

    def test_weight_layout_cache_invalidates_without_caching_activations(self):
        from triton.experimental.graph_fusion._model_inductor import _ParameterPack
        from triton.experimental.graph_fusion._p800_model_gemm import pack_column_major
        x=torch.randn(3,8);w=torch.nn.Parameter(torch.randn(16,8),requires_grad=False)
        pack=_ParameterPack(lambda a,b:a@b.t(),[1],transform=pack_column_major)
        torch.testing.assert_close(pack(x,w),x@w.t())
        saved=pack.entries[1][2]
        self.assertEqual(saved.stride(),(1,16));self.assertEqual(saved.dtype,torch.float32)
        torch.testing.assert_close(saved,w,atol=0,rtol=0)
        torch.testing.assert_close(pack(x+1,w),(x+1)@w.t())
        self.assertIs(pack.entries[1][2],saved)
        w.copy_(w+2)
        torch.testing.assert_close(pack(x,w),x@w.t())
        self.assertIsNot(pack.entries[1][2],saved)

    def test_aot_boxed_return_and_fx_tuple_match(self):
        from triton.experimental.graph_fusion._model_inductor import _compare
        x=torch.randn(4)
        _compare([x], (x,), dict(atol=0,rtol=0))
        with self.assertRaises(AssertionError):
            _compare([x,x], (x,), dict(atol=0,rtol=0))

    def test_dual_output_broadcast_fp32_internal_and_final_dtype(self):
        class Model(torch.nn.Module):
            def forward(self,a,b):
                z=a+b
                return F.layer_norm(z,(64,)),z
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            args=(torch.randn(2,3,64).to(dtype),torch.randn(1,3,64).to(dtype))
            gm=trace(Model(),args)
            records=expose_fp32_patterns(gm)
            self.assertEqual(records[0]['outputs'],2)
            self.assertTrue(records[0]['broadcast'])
            out,residual=gm(*args)
            total=args[0].float()+args[1].float()
            torch.testing.assert_close(residual,total.to(dtype),atol=0,rtol=0)
            torch.testing.assert_close(out,F.layer_norm(total,(64,)).to(dtype),atol=0,rtol=0)
            self.assertEqual(out.dtype,dtype)

    def test_quick_gelu_does_not_narrow_intermediates(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x*torch.sigmoid(1.702*x)
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            x=torch.linspace(-9,9,1024).to(dtype)
            gm=trace(Model(),(x,));records=expose_fp32_patterns(gm)
            self.assertEqual(len(records),1)
            expected=(x.float()*torch.sigmoid(1.702*x.float())).to(dtype)
            torch.testing.assert_close(gm(x),expected,atol=0,rtol=0)

    def test_mutation_order_unchanged(self):
        class Model(torch.nn.Module):
            def forward(self,a,b):
                z=a+b
                a.mul_(2)
                return F.layer_norm(z,(64,)),a
        args=(torch.randn(2,64).half(),torch.randn(2,64).half())
        gm=trace(Model(),tuple(x.clone() for x in args));before=gm.code
        self.assertEqual(expose_fp32_patterns(gm),[])
        self.assertEqual(gm.code,before)

    def test_other_sigmoid_constants_are_not_quick_gelu(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x*torch.sigmoid(1.7*x)
        gm=trace(Model(),(torch.randn(4).half(),));before=gm.code
        self.assertEqual(expose_fp32_patterns(gm),[]);self.assertEqual(gm.code,before)

    def test_quick_gelu_preserves_inner_output_writes(self):
        class ScaleOutput(torch.nn.Module):
            def forward(self, x, out):
                return x * torch.sigmoid(torch.mul(x, 1.702, out=out)), out

        class SigmoidOutput(torch.nn.Module):
            def forward(self, x, out):
                return x * torch.sigmoid(1.702 * x, out=out), out

        for model in (ScaleOutput(), SigmoidOutput()):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for aliases_input in (False, True):
                    with self.subTest(model=type(model).__name__, dtype=dtype, aliases_input=aliases_input):
                        x = torch.linspace(-9, 9, 128).to(dtype)
                        out = x if aliases_input else torch.full_like(x, -123)
                        gm = trace(model, (x.clone(), out.clone()))
                        expose_fp32_patterns(gm)
                        ref_x = x.clone()
                        ref_out = ref_x if aliases_input else out.clone()
                        expected = model(ref_x, ref_out)
                        actual = gm(x, out)
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got, want, atol=0, rtol=0)
                        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
                        torch.testing.assert_close(x, ref_x, atol=0, rtol=0)
                        self.assertEqual(actual[1].data_ptr(), out.data_ptr())

    def test_parameter_pack_invalidates_and_never_caches_input(self):
        from triton.experimental.graph_fusion._model_inductor import _ParameterPack
        w=torch.nn.Parameter(torch.randn(8).half(),requires_grad=False)
        pack=_ParameterPack(lambda x,p:x.float()+p,[1])
        x=torch.randn(8).half()
        torch.testing.assert_close(pack(x,w),x.float()+w.float(),atol=0,rtol=0)
        saved=pack.entries[1][2]
        pack(x+1,w);self.assertIs(pack.entries[1][2],saved)
        w.copy_(w+2)
        torch.testing.assert_close(pack(x,w),x.float()+w.float(),atol=0,rtol=0)
        self.assertIsNot(pack.entries[1][2],saved)
        w.data=torch.randn_like(w)
        torch.testing.assert_close(pack(x*2,w),x.float()*2+w.float(),atol=0,rtol=0)
        other=torch.nn.Parameter(torch.randn_like(w),requires_grad=False)
        torch.testing.assert_close(pack(x,other),x.float()+other.float(),atol=0,rtol=0)

    def test_whole_graph_fp32_candidate_preserves_final_dtype(self):
        from triton.experimental.graph_fusion._model_inductor import promote_whole_graph
        class Model(torch.nn.Module):
            def forward(self,a,b):return (a+b).sigmoid(),a+b
        a=torch.randn(2,8).half();b=torch.randn(2,8).half()
        gm=trace(Model(),(a,b));promote_whole_graph(gm)
        actual=gm(a,b)
        for value,want in zip(actual,((a.float()+b.float()).sigmoid(),a.float()+b.float())):
            self.assertEqual(value.dtype,torch.float16)
            torch.testing.assert_close(value,want.half(),atol=0,rtol=0)

    def test_fp32_reference_promotes_explicit_dtype(self):
        class Model(torch.nn.Module):
            def forward(self,a):return (a.to(torch.float16)*1.702).sigmoid()
        x=torch.randn(512).half();gm=trace(Model(),(x,))
        reference=_fp32_reference(gm)
        torch.testing.assert_close(reference(x),(x.float()*1.702).sigmoid(),atol=0,rtol=0)

    def test_promoted_pool_projection_survives_region_pass(self):
        from triton.experimental.graph_fusion._model_inductor import promote_whole_graph, refresh_graph_metadata
        from triton.experimental.graph_fusion._p800_fp32_region import rewrite
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        from torch._subclasses.fake_tensor import FakeTensorMode
        class Model(torch.nn.Module):
            def forward(self,x,mask,weight):
                pooled=(x*mask).sum(1)/mask.sum(1)
                return torch.ops.aten.linear.default(torch.cat([pooled],1),weight,None)
        for dtype in (torch.float16,torch.bfloat16):
            args=(torch.randn(2,5,8).to(dtype),torch.ones(2,5,1).to(dtype),torch.randn(8,8).to(dtype))
            gm=fx.symbolic_trace(Model())
            FakeTensorProp(gm,mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(*args)
            for node in gm.graph.nodes:
                if 'val' in node.meta:node.meta['example_value']=node.meta['val']
            with torch.no_grad():
                promote_whole_graph(gm,args)
                # This used to insert an FP16/BF16 cast before the FP32 weight.
                self.assertFalse(rewrite(gm.graph,native=True))
                actual=gm(*args)
                expected=Model()(*(x.float() for x in args)).to(dtype)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                # Packed parameters change placeholder metadata as well.
                packed=(args[0],args[1],args[2].float())
                refresh_graph_metadata(gm,packed)
                self.assertFalse(rewrite(gm.graph,native=True))
                torch.testing.assert_close(gm(*packed),expected,atol=0,rtol=0)
                changed=(args[0]+1,args[1],args[2]*2)
                torch.testing.assert_close(gm(*changed),Model()(*(x.float() for x in changed)).to(dtype),atol=0,rtol=0)

    def test_metadata_refresh_does_not_mutate_real_inputs_or_parameters(self):
        from triton.experimental.graph_fusion._model_inductor import refresh_graph_metadata
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('offset',torch.ones(8))
            def forward(self,x):
                x.add_(self.offset)
                return x,x.shape[0]
        gm=fx.symbolic_trace(Model());x=torch.zeros(2,8);saved=x.clone()
        refresh_graph_metadata(gm,(x,))
        torch.testing.assert_close(x,saved,atol=0,rtol=0)
        torch.testing.assert_close(gm.offset,torch.ones(8),atol=0,rtol=0)

    def test_vendor_short_rows_and_transpose_preserve_masks_and_values(self):
        from triton.experimental.graph_fusion._p800_model_schedule import preserve_short_softmax, preserve_block_transpose
        class Model(torch.nn.Module):
            def forward(self,x):
                return F.softmax(x,dim=-1),x.transpose(1,2).contiguous()
        for n in (3,45,77,128):
            x=torch.randn(2,7,n)
            x[0,0,:]=float('-inf')
            x[0,1,0]=float('inf')
            x[0,2,-1]=float('nan')
            gm=trace(Model(),(x,))
            self.assertEqual(len(preserve_short_softmax(gm)),1)
            self.assertEqual(len(preserve_block_transpose(gm)),1)
            actual=gm(x)
            expected=Model()(x)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
            self.assertTrue(actual[1].is_contiguous())
            self.assertNotEqual(actual[1].data_ptr(),x.data_ptr())
        class Other(torch.nn.Module):
            def forward(self,x):
                return F.softmax(x,dim=1,dtype=torch.float64)
        gm=trace(Other(),(torch.randn(2,3,45),))
        self.assertFalse(preserve_short_softmax(gm))

    def test_lm_options_cover_homogeneous_half_without_changing_mixed_conversions(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._inductor.codegen.triton import TritonKernel
        from torch._inductor.virtualized import V
        from triton.experimental.graph_fusion._p800_model_schedule import p800_lm_schedule
        original = V.choices
        with patch.object(TritonKernel,'codegen_body',return_value=None) as body:
            with self.assertRaisesRegex(RuntimeError,'restore'):
                with p800_lm_schedule(buffer_bytes=2048):
                    for signature, fixed, expected in [({'a':'*fp32','b':'*fp32'},True,2048),
                                                      ({'a':'*fp16','b':'*fp16'},True,2048),
                                                      ({'a':'*fp32','b':'*i64'},True,None),
                                                      ({'a':'*fp32','b':'*fp16'},True,None),
                                                      ({'a':'*fp32'},False,None)]:
                        kernel=SimpleNamespace(triton_meta={'signature':signature},fixed_config=fixed)
                        TritonKernel.codegen_body(kernel)
                        self.assertEqual(kernel.triton_meta.get('buffer_size_limit'),expected)
                    raise RuntimeError('restore')
            self.assertIs(TritonKernel.codegen_body,body)
        self.assertIs(V.choices,original)

    def test_half_tile_selected_before_mask_generation(self):
        from types import SimpleNamespace
        from torch._inductor.virtualized import V
        from triton.experimental.graph_fusion._p800_model_schedule import p800_lm_schedule
        from unittest.mock import patch
        feature = SimpleNamespace(numel=98304, reduction_numel=1, buf_accesses=lambda: {'a': [], 'b': []})
        before = V.choices
        for dtypes, size, expected in [((torch.float16,torch.float16),2048,65536),
                                      ((torch.float16,torch.float16),1024,32768),
                                      ((torch.float16,torch.float32),2048,32768),
                                      ((torch.float32,torch.float32),2048,32768)]:
            graph = SimpleNamespace(sizevars=SimpleNamespace(size_hint=lambda x,**kw:x),
                get_dtype=lambda name:dict(zip(('a','b'),dtypes))[name])
            with V.set_graph_handler(graph), patch.object(before,'triton_kernel_kwargs',side_effect=lambda *args:args[-1]):
                with p800_lm_schedule(buffer_bytes=size):
                    result = V.choices.triton_kernel_kwargs(None,feature,None,{})
                    self.assertEqual(result['fixed_config']['XBLOCK'],expected)
        self.assertIs(V.choices,before)

    def test_fp32_packet_tile_matches_buffer_and_is_audited(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._inductor.virtualized import V
        from torch._inductor.codegen.triton import TritonKernel
        from triton.experimental.graph_fusion._p800_model_schedule import p800_lm_schedule
        feature=SimpleNamespace(numel=98305,reduction_numel=1,buf_accesses=lambda:{'a':[]})
        graph=SimpleNamespace(sizevars=SimpleNamespace(size_hint=lambda x,**kw:x),get_dtype=lambda _:torch.float32)
        before=V.choices
        with patch.object(TritonKernel,'codegen_body',return_value=None), V.set_graph_handler(graph), patch.object(before,'triton_kernel_kwargs',side_effect=lambda *a:a[-1]):
            for size in (512,1024,2048):
                audit=[]
                with p800_lm_schedule(buffer_bytes=size,buffer_audit=audit):
                    result=V.choices.triton_kernel_kwargs(None,feature,None,{})
                    self.assertEqual(result['fixed_config']['XBLOCK'],size*16)
                    kernel=SimpleNamespace(triton_meta={'signature':{'a':'*fp32'}},fixed_config=True)
                    TritonKernel.codegen_body(kernel)
                    self.assertEqual(audit,[dict(storage='float32',requested=size)])

    def test_fusion_benchmark_uses_execution_tile_before_mask_generation(self):
        from types import SimpleNamespace
        from unittest.mock import patch, Mock
        from torch._inductor.virtualized import V
        from torch._inductor.codegen.triton import TritonScheduling
        from triton.experimental.graph_fusion._p800_model_schedule import p800_lm_schedule
        features = SimpleNamespace(numel=5908480, reduction_numel=1, buf_accesses=lambda: {'a': []})
        graph = SimpleNamespace(sizevars=SimpleNamespace(size_hint=lambda x, **kw: x), get_dtype=lambda _: torch.float32)
        node = SimpleNamespace(is_template=lambda: False, is_reduction=lambda: False, group=('cuda', (5908480, 1)))
        kernel = SimpleNamespace(codegen_kernel=lambda: 'verified schedule')
        constructor = Mock(return_value=kernel)
        scheduler = SimpleNamespace(generate_node_schedule=lambda *a: [node], select_tiling=lambda *a: {'x': 5908480},
            kernel_type=constructor, codegen_node_schedule_with_kernel=Mock())
        original = TritonScheduling.generate_kernel_code_from_nodes
        with V.set_graph_handler(graph), patch('torch._inductor.codegen.simd_kernel_features.SIMDKernelFeatures', return_value=features):
            with p800_lm_schedule(buffer_bytes=1024):
                source = TritonScheduling.generate_kernel_code_from_nodes(scheduler, [node], True)
                self.assertEqual(source, 'verified schedule')
                self.assertEqual(constructor.call_args.kwargs['fixed_config']['XBLOCK'], 16384)
                scheduler.codegen_node_schedule_with_kernel.assert_called_once_with([node], kernel)
        self.assertIs(TritonScheduling.generate_kernel_code_from_nodes, original)

    def test_half_candidates_keep_existing_fp32_buffer(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._inductor.codegen.triton import TritonKernel
        from triton.experimental.graph_fusion._p800_model_schedule import p800_lm_schedule
        with patch.object(TritonKernel,'codegen_body',return_value=None):
            for size in (512,1024,2048):
                with p800_lm_schedule(buffer_bytes=2048,fp16_buffer_bytes=size):
                    for dtype,want in [('fp16',size),('fp32',2048)]:
                        kernel=SimpleNamespace(triton_meta={'signature':{'a':'*'+dtype,'b':'*'+dtype}},fixed_config=True)
                        TritonKernel.codegen_body(kernel)
                        self.assertEqual(kernel.triton_meta['buffer_size_limit'],want)

    def test_short_softmax_preserves_explicit_output_and_aliases(self):
        from triton.experimental.graph_fusion._p800_model_schedule import preserve_short_softmax

        class Model(torch.nn.Module):
            def forward(self, x, out):
                value = torch.softmax(x, dim=-1, out=out)
                return value, out

        for width in (3, 45, 77, 128):
            x = torch.randn(2, 7, width)
            out = torch.full_like(x, -12345.)
            gm = trace(Model(), (x, out))
            preserve_short_softmax(gm)
            out.fill_(-12345.)
            value, alias = gm(x, out)
            torch.testing.assert_close(out, torch.softmax(x, dim=-1), atol=0, rtol=0)
            self.assertEqual(value.data_ptr(), out.data_ptr())
            self.assertEqual(alias.data_ptr(), out.data_ptr())

    def test_backend_buffer_option_survives_whitelist_and_restores_after_error(self):
        import triton
        from types import SimpleNamespace
        from unittest.mock import patch
        from triton.experimental.graph_fusion._p800_model_schedule import _compile_buffer_option
        fn=object()
        with patch.object(triton,'compile',side_effect=lambda source,**kwargs:kwargs) as original:
            with self.assertRaisesRegex(RuntimeError,'restore'):
                with _compile_buffer_option(fn,2048):
                    actual=triton.compile(SimpleNamespace(fn=fn),options={'debug':True})
                    self.assertEqual(actual['options'],{'debug':True,'buffer_size_limit':2048})
                    other=triton.compile(SimpleNamespace(fn=object()),options={'debug':True})
                    self.assertEqual(other['options'],{'debug':True})
                    raise RuntimeError('restore')
            self.assertIs(triton.compile,original)

    def test_pointwise_workaround_does_not_disable_reduction_choices(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._inductor import config
        from torch._inductor.codegen.triton import TritonKernel
        from triton.experimental.graph_fusion._p800_model_schedule import p800_lm_schedule
        with config.patch({'triton.autotune_pointwise': False}):
            with patch.object(TritonKernel,'codegen_kernel',lambda *a,**k: str(config.triton.autotune_pointwise)):
                with p800_lm_schedule():
                    for reduction in (False,True):
                        k=SimpleNamespace(inside_reduction=reduction,triton_meta={})
                        self.assertEqual(TritonKernel.codegen_kernel(k),str(reduction))
                        self.assertFalse(config.triton.autotune_pointwise)

    def test_vendor_norm_remains_eligible_for_fast_schedules_without_aliases(self):
        from triton.experimental.graph_fusion._p800_model_schedule import schedule_variants
        from triton.experimental.graph_fusion._p800_model_layernorm import preserve_unmatched_norms
        class Model(torch.nn.Module):
            def forward(self, x):
                y = F.layer_norm(x, (8,))
                return y * torch.sigmoid(1.702 * y)
        x = torch.randn(2, 8)
        fused = trace(Model(), (x,))
        vendor = copy.deepcopy(fused)
        self.assertEqual(preserve_unmatched_norms(vendor), 1)
        originals = (fused.code, vendor.code)
        variants = schedule_variants([('inductor_p800_lm', fused, []),
            ('inductor_p800_lm_sdk_norm', vendor, []), ('duplicate', copy.deepcopy(vendor), [])])
        self.assertEqual([x[0] for x in variants], ['inductor_p800_lm', 'inductor_p800_lm2048',
            'inductor_p800_lm_sdk_norm', 'inductor_p800_lm_sdk_norm2048'])
        for name, gm, regions in variants:
            torch.testing.assert_close(gm(x), Model()(x), atol=0, rtol=0)
            targets = [n.target for n in gm.graph.nodes if n.op == 'call_function']
            self.assertEqual(torch.ops.flagtree_p800.vendor_layernorm.default in targets, 'sdk_norm' in name)
        self.assertEqual((fused.code, vendor.code), originals)

    def test_final_conversion_keeps_dtype_layout_and_fresh_storage(self):
        from triton.experimental.graph_fusion._p800_model_layernorm import _final_cast
        x=torch.randn(7,13).t()
        x[0,0]=float('inf');x[1,0]=float('nan')
        for dtype in (torch.float16,torch.bfloat16):
            result=_final_cast(x,dtype)
            expected=x.to(dtype=dtype,copy=True)
            self.assertEqual(result.dtype,dtype)
            self.assertEqual(result.stride(),expected.stride())
            self.assertNotEqual(result.data_ptr(),x.data_ptr())
            torch.testing.assert_close(result,expected,atol=0,rtol=0,equal_nan=True)

    def test_sdnn_tensor_spans_are_conservative_and_reject_unknown_layouts(self):
        from types import SimpleNamespace
        from torch._inductor.codegen.common import TensorArg
        from triton.experimental.graph_fusion._p800_gemm_candidates import static_tensor_spans
        def node(size,stride,offset=0):
            return SimpleNamespace(get_size=lambda:size,get_stride=lambda:stride,
                                   get_layout=lambda:SimpleNamespace(offset=offset))
        args=[TensorArg('arg0','x',torch.float32),TensorArg('arg1','w',torch.float16)]
        nodes={'x':node((3,5),(8,1),2),'w':node((5,7),(1,5))}
        self.assertEqual(static_tensor_spans(args,nodes.get),(92,70))
        for bad in (node((3,5),(-5,1)),node((0,5),(5,1)),node((3,5),(5,1),-1),None):
            nodes['x']=bad
            self.assertEqual(static_tensor_spans(args,nodes.get),())
        self.assertEqual(static_tensor_spans([object()],nodes.get),())

    def test_sdnn_autotune_output_need_not_be_in_global_graph(self):
        from types import SimpleNamespace
        from torch._inductor.codegen.common import TensorArg
        from triton.experimental.graph_fusion._p800_gemm_candidates import template_tensor_spans
        output=SimpleNamespace(get_name=lambda:'buf_out',get_size=lambda:(7,11),
            get_stride=lambda:(11,1),get_layout=lambda:SimpleNamespace(offset=0))
        kernel=SimpleNamespace(input_nodes=[],output_node=output)
        def lookup(name):
            raise RuntimeError('Failed to find buffer matching name '+name)
        self.assertEqual(template_tensor_spans(kernel,[TensorArg('out','buf_out',torch.float32)],lookup),(308,))
        self.assertEqual(template_tensor_spans(kernel,[TensorArg('other','missing',torch.float32)],lookup),())
        larger=SimpleNamespace(get_name=lambda:'buf_out',get_size=lambda:(11,11),
            get_stride=lambda:(11,1),get_layout=lambda:SimpleNamespace(offset=0))
        kernel.input_nodes=[larger]
        self.assertEqual(template_tensor_spans(kernel,[TensorArg('out','buf_out',torch.float32)],lookup),(484,))

    def test_sdnn_bounds_reach_only_matching_deferred_compile(self):
        import triton
        from types import SimpleNamespace
        from unittest.mock import patch
        from triton.experimental.graph_fusion._p800_model_schedule import _compile_buffer_option
        fn=object();bounds=(4096,8192,2048)
        def compile(source,**kw):return kw
        with patch.object(triton,'compile',compile):
            with _compile_buffer_option(fn,512,sdnn_tensor_bytes=bounds):
                result=triton.compile(SimpleNamespace(fn=fn),options={'debug':True})
                self.assertEqual(result['options']['xpu_sdnn_tensor_bytes'],bounds)
                other=triton.compile(SimpleNamespace(fn=object()),options={'debug':True})
                self.assertEqual(other['options'],{'debug':True})
            self.assertIs(triton.compile,compile)

    def test_sdnn_deferred_compile_binding_can_cross_process_boundary(self):
        import pickle
        from functools import partial
        from types import SimpleNamespace
        from triton.experimental.graph_fusion._p800_gemm_candidates import _precompile_sdnn_config
        tuner=SimpleNamespace(triton_meta={'xpu_sdnn_tensor_bytes':(2048,4096)})
        tuner._precompile_config=partial(_precompile_sdnn_config,tuner)
        restored=pickle.loads(pickle.dumps(tuner))
        self.assertIs(restored._precompile_config.func,_precompile_sdnn_config)
        self.assertIs(restored._precompile_config.args[0],restored)

    def test_sdnn_template_imports_preserve_decorator_indentation(self):
        import ast
        from pathlib import Path
        from triton.experimental.graph_fusion._p800_gemm_candidates import _sdnn_template_source
        original = '\n            @triton_heuristics.template(\n                triton_meta={},\n            )\n            @triton.jit\n'
        for text in (original, __import__('textwrap').dedent(original)):
            source = _sdnn_template_source(text, (64,128), Path(__file__))
            module = ast.parse(source + 'def kernel(x, y):\n    return x + y\n')
            self.assertEqual(len(module.body[-1].decorator_list),2)
            self.assertIn("'xpu_sdnn_tensor_bytes': (64, 128)",source)

    def test_backend_options_do_not_wait_for_parent_graph_lock(self):
        import threading
        import triton
        from types import SimpleNamespace
        from unittest.mock import patch
        from torch._dynamo.convert_frame import compile_lock
        from triton.experimental.graph_fusion._p800_model_schedule import _compile_buffer_option
        done=threading.Event();errors=[];results=[];fn=object()
        def worker():
            try:
                with _compile_buffer_option(fn,512,sdnn_tensor_bytes=(64,128)):
                    results.append(triton.compile(SimpleNamespace(fn=fn)))
            except BaseException as exc:errors.append(exc)
            finally:done.set()
        with patch.object(triton,'compile',lambda source,**kw:kw) as original:
            with compile_lock:
                t=threading.Thread(target=worker);t.start()
                finished=done.wait(5)
            t.join(5)
            self.assertTrue(finished,'worker waited for parent compile_lock')
            self.assertFalse(errors)
            self.assertEqual(results[0]['options']['xpu_sdnn_tensor_bytes'],(64,128))
            self.assertIs(triton.compile,original)

    def test_backend_options_do_not_leak_to_unrelated_compile_thread(self):
        import threading
        import triton
        from types import SimpleNamespace
        from unittest.mock import patch
        from triton.experimental.graph_fusion._p800_model_schedule import _compile_buffer_option
        fn=object();seen=[]
        with patch.object(triton,'compile',lambda source,**kw:kw):
            with _compile_buffer_option(fn,512,sdnn_tensor_bytes=(64,128)):
                t=threading.Thread(target=lambda:seen.append(triton.compile(SimpleNamespace(fn=fn))))
                t.start();t.join(5)
                self.assertFalse(t.is_alive())
                self.assertEqual(seen,[{}])


if __name__=='__main__':unittest.main()
