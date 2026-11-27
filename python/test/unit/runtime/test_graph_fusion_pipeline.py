"""Check the actual Dynamo boundary without accelerator compilation."""
import unittest
from unittest.mock import patch

import torch
from triton.experimental.graph_fusion._model_pipeline import compile_pipeline


class PipelineTests(unittest.TestCase):
    def test_matrix_candidates_are_cold_scoped_and_keep_full_pipeline(self):
        for schedule in ('inductor', 'grouped'):
            from types import SimpleNamespace
            options=[];prepared=[]
            class Model:
                def parameters(self):return [SimpleNamespace(device=torch.device('cuda:0'))]
                def buffers(self):return []
            class Compiled:
                _f06_capture='test'
                def __init__(self):
                    self._f06_audit=[dict(exposed_linears=[dict(input_dtype='torch.float32')])]
                def __call__(self,x):return x+1
            def compile_model(model,**kwargs):
                options.append(kwargs);return Compiled()
            def prepare(x):prepared.append(x);return ((x,),{})
            with patch('triton.experimental.graph_fusion._model.compile_model',compile_model), \
                 patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',return_value=True):
                base=compile_pipeline(Model(),prepare=prepare,precision_policy='fuse_fp32_internals',
                    precision_options=dict(linear_schedule=schedule))
                self.assertEqual(len(options),1)
                variants=base._f06_gemm_variants();self.assertIs(base._f06_gemm_variants(),variants)
                self.assertEqual(set(variants),{'gemm_tuned','gemm_full_k'})
                new=variants['gemm_tuned'];self.assertEqual(new(3),4)
                self.assertEqual(prepared,[3])
                self.assertEqual(options[-2]['precision_options']['matmul_policy'],'tuned')
                self.assertEqual(options[-1]['precision_options']['matmul_policy'],'full_k')
                self.assertEqual(new._f06_gemm_variants(),{})
                self.assertEqual(new._f06_resource_variants(),{})
                self.assertEqual(new._f06_buffer_variants(),{})
                base._f06_gemm_candidates=None
                base._f06_audit[0]['exposed_linears'][0]['input_dtype']='torch.float16'
                self.assertEqual(base._f06_gemm_variants(),{})
                base._f06_gemm_candidates=None
                base._f06_audit[0]['exposed_linears'][0]['input_dtype']='torch.float32'
                with patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',return_value=False):
                    self.assertEqual(base._f06_gemm_variants(),{})
            self.assertEqual(len(options),3)

    def test_resource_candidates_keep_independent_guarded_capture_caches(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0004))
            def forward(self, x):
                return {'value': x * self.weight - 1000}
        model = Model().eval()
        template = {'value': torch.empty(1, dtype=torch.float16)}
        limit = torch._dynamo.config.recompile_limit
        with torch._dynamo.config.patch(recompile_limit=2), \
             patch('triton.experimental.graph_fusion._model_inductor.inductor_backend', self.backend):
            routes = [compile_pipeline(model, prepare=lambda x: ((x,), {}),
                        source_precision='float32', output_template=template,
                        resource_policy=('cluster' if i % 2 else 'sdnn'),
                        vendor_capture=False) for i in range(10)]
            with torch.no_grad():
                for n in (3, 5):
                    x = torch.full((n,), 1000.)
                    for run in routes:
                        torch.testing.assert_close(run(x)['value'], model(x)['value'].half(), atol=0, rtol=0)
                self.assertEqual(len(self.graphs), 20)
                model.weight.add_(.1)
                for run in reversed(routes):
                    torch.testing.assert_close(run(x)['value'], model(x)['value'].half(), atol=0, rtol=0)
                self.assertEqual(len(self.graphs), 20)
        self.assertEqual(torch._dynamo.config.recompile_limit, limit)

    def test_resource_expansion_mirrors_prepared_graph_and_keeps_preprocessing_live(self):
        from types import SimpleNamespace
        prepared=[];options=[]
        class Model:
            def parameters(self):return [SimpleNamespace(device=torch.device('cuda',0))]
            def buffers(self):return []
        class Compiled:
            _f06_capture='test'
            def __init__(self,audit):self._f06_audit=audit
            def __call__(self,x):return x+1
        def compile_model(model,**kwargs):
            options.append(kwargs)
            audit=kwargs.get('audit',[])
            if kwargs.get('resource_policy','auto')=='auto':audit.append(dict(route='inductor_p800_lm2048'))
            return Compiled(audit)
        def prepare(x):prepared.append(x);return ((x,),{})
        with patch('triton.experimental.graph_fusion._model.compile_model',compile_model), \
             patch('triton.experimental.graph_fusion._p800_inductor_gelu._on_p800',return_value=True):
            base=compile_pipeline(Model(),prepare=prepare,resource_policy='auto')
            self.assertEqual(base(3),4)
            variants=base._f06_resource_variants()
            self.assertIs(base._f06_resource_variants(),variants)
            self.assertEqual(set(variants),{'cluster','sdnn'})
            for kind,call in variants.items():
                self.assertEqual(call(4),5)
                self.assertEqual(call._f06_resource_variants(),{})
            for option in options[1:]:
                self.assertEqual(option['graph_choice'],'inductor_p800_lm2048')
                self.assertTrue(option['fullgraph'])
            self.assertEqual(prepared,[3,4,4])
            # An unsupported native-precision boundary retains eager. There
            # are no generated pointwise kernels to turn into SDNN candidates.
            base._f06_audit[:]=[dict(route='original',rejected=[dict(reason='unreviewed boundary')])]
            base._f06_resource_candidates=None
            self.assertEqual(base._f06_resource_variants(),{})
            self.assertEqual(len(options),3)
            self.assertEqual(base._f06_audit[0]['resource_enumeration'],'skipped_original_vendor_fallback')

    def setUp(self):
        torch._dynamo.reset()
        self.graphs = []

    def backend(self, **options):
        def capture(gm, inputs):
            self.graphs.append(gm)
            return gm.forward
        return capture

    def compile(self, model, prepare):
        with patch('triton.experimental.graph_fusion._model_inductor.inductor_backend', self.backend):
            return compile_pipeline(model.eval(), prepare=prepare,
                                    tolerance=dict(atol=0, rtol=0), vendor_capture=False)

    def test_direct_forward_pooling_normalization_and_output_order_in_one_graph(self):
        class Pipeline(torch.nn.Module):
            class Linear(torch.nn.Module):
                def __init__(self, n, m):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.randn(n, m))
                def forward(self, x):
                    return x @ self.weight

            def __init__(self):
                super().__init__()
                self.encoder = self.Linear(4, 8)
                self.projection = self.Linear(8, 3)

            def forward(self, x, order):
                # Direct .forward calls used by encode must not hide the root.
                hidden = self.encoder.forward(x).relu()
                pooled = hidden.mean(1)
                projected = self.projection.forward(pooled)
                return torch.nn.functional.normalize(projected, dim=-1).index_select(0, order)

        model = Pipeline().eval()
        calls = []

        def prepare(x, order):
            calls.append(x.shape)
            return ((x, torch.tensor(order)), {})

        run = self.compile(model, prepare)
        with torch.no_grad():
            for n in (3, 3, 5):
                x = torch.randn(n, 7, 4)
                order = list(reversed(range(n)))
                torch.testing.assert_close(run(x, order), model(x, torch.tensor(order)))
        self.assertEqual(len(calls), 3)
        # New shape specializes another complete graph, not separate modules.
        self.assertEqual(len(self.graphs), 2)
        for graph in self.graphs:
            code = graph.code
            for op in ('@', 'mean', 'normalize', 'index_select'):
                self.assertIn(op, code)
        self.assertEqual(run._f06_scope, 'complete_tensor_forward')

    def test_graph_break_is_explicit_failure(self):
        class Broken(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                torch._dynamo.graph_break()
                return x.relu()
        run = self.compile(Broken(), lambda x: ((x,), {}))
        with torch.no_grad(), self.assertRaises(torch._dynamo.exc.Unsupported):
            run(torch.randn(3))
        self.assertEqual(self.graphs, [])

    def test_preprocessing_and_output_contract_remain_live(self):
        class Pipeline(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(3))
            def forward(self, x, *, scale):
                y = (x + self.bias) * scale
                return {'value': y, 'residual': (x,)}
        model = Pipeline().eval()
        run = self.compile(model, lambda values: ((torch.tensor(values),), {'scale': 2}))
        with torch.no_grad():
            a = run([1., 2., 3.])
            model.bias.add_(3)
            b = run([3., 1., 4.])
        torch.testing.assert_close(a['value'], torch.tensor([4., 6., 8.]))
        torch.testing.assert_close(b['value'], torch.tensor([14., 10., 16.]))
        torch.testing.assert_close(b['residual'][0], torch.tensor([3., 1., 4.]))

    def test_reject_partial_capture_and_malformed_preparation(self):
        with self.assertRaises(ValueError):
            compile_pipeline(torch.nn.Identity().eval(), prepare=lambda x: ((x,), {}), fullgraph=False)
        run = self.compile(torch.nn.Identity(), lambda x: x)
        with self.assertRaises(TypeError):
            run(torch.ones(3))

    def test_original_fp32_capture_and_final_cast_preserve_unrounded_values(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0004))

            def forward(self, x):
                y = x * self.weight - 1000
                if x.dtype == torch.float16:
                    y = y.clamp(-10, 10)
                return {'value': y, 'ids': torch.arange(x.shape[0])}

        model = Model().eval()
        template = {'value': torch.empty(1, 1, dtype=torch.float16), 'ids': torch.zeros(1, dtype=torch.int64)}
        with patch('triton.experimental.graph_fusion._model_inductor.inductor_backend', self.backend):
            run = compile_pipeline(model, prepare=lambda x: ((x,), {}),
                source_precision='float32', output_template=template,
                tolerance=dict(atol=0, rtol=0), vendor_capture=False)
        with torch.no_grad():
            for n in (1, 3):
                x = torch.full((n, 1), 1000.)
                result = run(x)
                torch.testing.assert_close(result['value'], model(x)['value'].half(), atol=0, rtol=0)
                self.assertGreater(result['value'].float().min().item(), .39)
                self.assertEqual(result['ids'].dtype, torch.int64)
        self.assertEqual(len(self.graphs), 2)
        for graph in self.graphs:
            self.assertNotIn('clamp', graph.code)
            casts = [x for x in graph.graph.nodes if x.op == 'call_method' and x.target == 'to']
            self.assertEqual(len(casts), 1)
            self.assertEqual(casts[0].kwargs['dtype'], torch.float16)
        with self.assertRaisesRegex(ValueError, 'original FP32 prepared inputs'):
            run(torch.ones(1, 1, dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, 'original FP32 parameters'):
            compile_pipeline(Model().half().eval(), prepare=lambda x: ((x,), {}), source_precision='float32')


if __name__ == '__main__':
    unittest.main()
