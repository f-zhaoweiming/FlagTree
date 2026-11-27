"""CPU contracts for whole-plan native Inductor resource selection."""
import copy
import types
import unittest
from unittest import mock

from test_graph_fusion_static import InferenceCPUTestBase, torch, _load_module


@unittest.skipIf(torch is None, 'torch is required')
class ResourceAutotuneTests(InferenceCPUTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.auto = _load_module('_resource_autotune')
        cls.bench = _load_module('_resource_benchmark')

    def graph(self):
        return torch.fx.symbolic_trace(lambda a, b, c: torch.relu(a @ b + c))

    def record(self, candidate, us):
        proof = {'kernel_count': len(candidate.regions), 'regions': [
            {'resource_verified': True, 'external_calls': 0,
             'kernel_evidence': [{'effective': region.resource}]} for region in candidate.regions]}
        return {'name': candidate.name, 'context_key': 'context', 'tolerance': {'atol': 3e-4, 'rtol': 3e-4},
                'status': 'passed', 'checks': {k: True for k in self.auto.CHECKS},
                'gpu_samples_us': [us, us * 1.01, us * 0.99], 'proof': proof,
                'e2e_protocol': 'randomized_ab_ba_v1', 'e2e_round_samples_us': [[us*2]*128, [us*2]*128]}

    def test_small_has_only_single_kernel_resource_candidates(self):
        plans = self.auto.candidate_plans(self.graph(), (torch.randn(16, 64), torch.randn(64, 128), torch.randn(128)))
        self.assertEqual([p.name for p in plans], ['eager', 'inductor', 'sdnn'])
        self.assertTrue(plans[-1].single_kernel)

    def test_large_preserves_sdnn_dot_and_all_epilogue_cut_options(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=False)
        self.assertEqual([p.name for p in plans], ['eager', 'inductor', 'sdnn', 'sdnn_cluster_at_1', 'sdnn_cluster_at_2'])
        for p in plans[2:]:
            self.assertEqual(p.regions[0].resource, 'sdnn')
            self.assertEqual(p.regions[-1].end, 3)

    def test_norm_never_invents_sdnn_support(self):
        gm = torch.fx.symbolic_trace(lambda x, b: torch.nn.functional.layer_norm(x + b, (128,)))
        plans = self.auto.candidate_plans(gm, (torch.randn(16, 128), torch.randn(128)))
        self.assertEqual([p.name for p in plans], ['eager', 'inductor', 'cluster'])

    def test_opt_in_reduction_split_preserves_product_dtype_and_keepdim(self):
        from test_graph_fusion_static import _load_module
        fx = _load_module('_resource_fx')
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for keep in (False, True):
                gm = torch.fx.symbolic_trace(lambda a, b: (a * b).sum(-1, keepdim=keep))
                inputs = (torch.randn(4, 128).to(dtype), torch.randn(128).to(dtype))
                plans = self.auto.candidate_plans(gm, inputs, single_kernel=False, split_reductions=True)
                self.assertEqual([p.name for p in plans[-4:]], ['cluster_cluster_at_1', 'cluster_sdnn_at_1',
                                                             'sdnn_cluster_at_1', 'sdnn_sdnn_at_1'])
                first, second = (fx.build_region(gm, a, b) for a, b in ((0, 1), (1, 2)))
                product, = first.gm(*inputs)
                self.assertEqual(product.dtype, dtype)
                output, = second.gm(product)
                torch.testing.assert_close(output, gm(*inputs), atol=0, rtol=0, equal_nan=True)
                self.assertEqual(output.shape, (4, 1) if keep else (4,))

    def test_reduction_split_does_not_change_default_or_single_kernel_plans(self):
        gm = torch.fx.symbolic_trace(lambda a, b: (a * b).sum(-1))
        xs = (torch.randn(16, 128), torch.randn(128))
        default = self.auto.candidate_plans(gm, xs)
        self.assertEqual(default, self.auto.candidate_plans(gm, xs, split_reductions=True))
        self.assertEqual([p.name for p in self.auto.candidate_plans(gm, xs, single_kernel=False)],
                         ['eager', 'inductor', 'cluster', 'sdnn'])
        add_sum = torch.fx.symbolic_trace(lambda a, b: (a + b).sum(-1))
        self.assertEqual(len(self.auto.candidate_plans(add_sum, xs, single_kernel=False, split_reductions=True)), 4)

    def test_selects_full_plan_time_without_double_counted_transfer(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=False)
        rows = [self.record(p, t) for p, t in zip(plans, [20, 15, 12, 7, 9])]
        p, result = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(p.name, 'sdnn_cluster_at_1')
        self.assertEqual(result['calibration_gpu_us'], 7)
        self.assertFalse(result['fallback'])

    def test_failed_fast_candidate_falls_back_to_valid_native_baseline(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=True)
        rows = [self.record(p, t) for p, t in zip(plans, [8, 6, 1])]
        for bad in ['special', 'immutable', 'graph_fresh_data']:
            trial = copy.deepcopy(rows)
            trial[2]['checks'][bad] = False
            p, result = self.auto.select_measurement(plans, trial, 'context', rows[0]['tolerance'])
            self.assertEqual(p.name, 'inductor')
            self.assertTrue(result['fallback'])

    def test_wrong_context_precision_binary_and_launch_count_are_rejected(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=True)
        base = [self.record(p, t) for p, t in zip(plans, [8, 6, 1])]
        for mutation in ['context', 'precision', 'binary', 'launch', 'timing']:
            rows = copy.deepcopy(base)
            r = rows[-1]
            if mutation == 'context': r['context_key'] = 'another GPU/layout'
            if mutation == 'precision': r['tolerance']['atol'] = 0.1
            if mutation == 'binary': r['proof']['regions'][0]['kernel_evidence'][0]['effective'] = 'cluster'
            if mutation == 'launch': r['proof']['kernel_count'] = 2
            if mutation == 'timing': r['gpu_samples_us'] = [float('nan')]
            p, _ = self.auto.select_measurement(plans, rows, 'context', base[0]['tolerance'])
            self.assertEqual(p.name, 'inductor', mutation)

    def test_duplicate_profiles_do_not_cherry_pick_fast_sample(self):
        p = self.auto.Candidate('eager')
        r = self.record(p, 3)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.auto.select_measurement([p], [r, r], 'context', r['tolerance'])

    def test_gpu_winner_with_slow_ordinary_calls_is_rejected(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=True)
        rows = [self.record(p, us) for p, us in zip(plans, [8, 6, 1])]
        rows[1]['e2e_round_samples_us'] = [[30]*128]*2
        rows[2]['e2e_round_samples_us'] = [[20]*128]*2
        p, decision = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(p.name, 'eager')
        self.assertEqual(decision['e2e_acceptance']['accepted'], ['eager'])

    def test_both_rounds_need_gain_over_fastest_stable_baseline(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=True)
        rows = [self.record(p, us) for p, us in zip(plans, [8, 6, 1])]
        rows[2]['e2e_round_samples_us'] = [[8]*128, [13]*128]
        p, decision = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(p.name, 'inductor')
        self.assertEqual(decision['e2e_acceptance']['baseline'], 'inductor')

    def test_missing_or_nonfinite_e2e_cannot_pass(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=True)
        base = [self.record(p, us) for p, us in zip(plans, [8, 6, 1])]
        for field, value in [('e2e_protocol','old_blocked'), ('e2e_round_samples_us',[[float('nan')]*128]*2),
                             ('e2e_round_samples_us', [[1]*127]*2)]:
            rows = copy.deepcopy(base); rows[2][field] = value
            p, _ = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
            self.assertEqual(p.name, 'inductor')
        rows = copy.deepcopy(base);rows[0].pop('e2e_protocol')
        with self.assertRaisesRegex(RuntimeError, 'eager'):
            self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])

    def test_fast_entry_preserves_profiler_and_cleanup_after_failure(self):
        from torch._inductor.output_code import CompiledFxGraph
        runtime = _load_module('_resource_runtime')
        binary = object.__new__(CompiledFxGraph)
        binary.compiled_fn_runner = None
        binary._fx_graph_cache_key = 'unit-fast-entry'
        binary.current_callable = lambda xs: xs[0] + 1
        run, enabled = runtime.inference_entry(binary)
        self.assertTrue(enabled)
        with torch.no_grad():
            self.assertEqual(run([2]), 3)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                self.assertEqual(run([2]), 3)
            self.assertTrue(any('Call CompiledFxGraph' in e.key for e in prof.key_averages()))
            metrics = types.SimpleNamespace(_metrics={'pending': 1}, finish=mock.Mock())
            from torch._inductor.runtime.autotune_cache import AutotuneCacheBundler
            binary.current_callable = mock.Mock(side_effect=RuntimeError('kernel failed'))
            with mock.patch('torch._dynamo.utils.get_runtime_metrics_context', return_value=metrics):
                run, _ = runtime.inference_entry(binary)
            with mock.patch.object(AutotuneCacheBundler, '_bundler', object()), \
                 mock.patch.object(AutotuneCacheBundler, 'end_compile') as finish:
                with self.assertRaisesRegex(RuntimeError, 'kernel failed'):run([2])
                metrics.finish.assert_called_once()
                finish.assert_called_once()

    def test_fast_entry_unknown_type_keeps_original(self):
        runtime = _load_module('_resource_runtime')
        fn = lambda x:x
        run, enabled = runtime.inference_entry(fn)
        self.assertFalse(enabled)
        self.assertIs(run, fn)

    def test_unknown_torch_version_does_not_require_private_entry_api(self):
        import sys
        runtime = _load_module('_resource_runtime')
        fn = lambda x:x
        with mock.patch.object(torch, '__version__', '2.8.0'), \
             mock.patch.dict(sys.modules, {'torch._inductor.output_code': None}):
            run, enabled = runtime.inference_entry(fn)
        self.assertFalse(enabled)
        self.assertIs(run, fn)

    def test_nonuniform_change_detects_translation_invariant_softmax(self):
        xs = (torch.randn(16, 128), torch.randn(128))
        ys = self.bench._changed(xs)
        self.assertFalse(torch.allclose(torch.softmax(sum(xs), -1), torch.softmax(sum(ys), -1)))

    def test_validation_preserves_inputs_and_detects_numerical_failure(self):
        gm = self.graph()
        xs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
        with torch.no_grad():
            checks = self.bench.validate(gm, gm, xs, {'atol': 3e-4, 'rtol': 3e-4})
            self.assertTrue(all(checks.values()))
            with self.assertRaises(AssertionError):
                self.bench.validate(lambda *args: gm(*args) + 1, gm, xs, {'atol': 3e-4, 'rtol': 3e-4})

    def test_rejected_mutating_candidate_does_not_poison_next_candidate(self):
        gm = self.graph()
        xs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
        original = xs[0].clone()
        def broken(*args):
            args[0].add_(1)
            raise RuntimeError('candidate failed')
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, 'candidate failed'):
                self.bench.validate(broken, gm, xs, {'atol': 3e-4, 'rtol': 3e-4})
        self.assertTrue(torch.equal(xs[0], original))

    def test_backend_calibrates_once_and_reuses_warm_callable(self):
        codegen = _load_module('_resource_codegen')
        gm = self.graph()
        inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
        with mock.patch.object(codegen, 'baidu_target', return_value={'arch': 3}), \
             mock.patch.object(self.auto, 'calibrate_resources', return_value=(gm, {})) as calibrate:
            backend = self.auto.autotune_resource_backend(tolerance={'atol': 3e-4, 'rtol': 3e-4},
                                                          environment_key='unit-context')
            run = backend(gm, inputs)
            self.assertEqual(calibrate.call_count, 0)
            torch.testing.assert_close(run(*inputs), gm(*inputs))
            torch.testing.assert_close(run(*inputs), gm(*inputs))
            self.assertEqual(calibrate.call_count, 1)

    def test_native_baseline_compilation_cannot_mutate_reference_graph(self):
        from contextlib import nullcontext
        compiler = _load_module('_p800_compiler')
        matmul = _load_module('_p800_matmul')
        gm = self.graph()
        inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
        before = str(gm.graph)
        def compile_native(model, args):
            node = next(n for n in model.graph.nodes if n.op == 'output')
            original = node.args[0]
            node.args = ([original],)
            model.recompile()
            return lambda *xs: model(*xs)[0]
        with mock.patch.object(compiler, 'target_scope', side_effect=lambda: nullcontext()), \
             mock.patch.object(matmul, 'native_template_eligibility', side_effect=lambda _: nullcontext()), \
             mock.patch.object(self.auto, 'compiler_options', return_value={}), \
             mock.patch('torch._inductor.compile_fx.compile_fx', side_effect=compile_native):
            fn, proof = self.auto.compile_candidate(gm, inputs, self.auto.Candidate('inductor'))
        self.assertEqual(str(gm.graph), before)
        torch.testing.assert_close(fn(*inputs), gm(*inputs))
        self.assertEqual(proof['mode'], 'max-autotune-no-cudagraphs')

    def test_offline_costs_cannot_select_a_different_retuned_binary(self):
        codegen = _load_module('_resource_codegen')
        gm = self.graph()
        inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
        plans = self.auto.candidate_plans(gm, inputs)
        key = self.auto.context_key(gm, inputs, 'unit-environment')
        rows = [self.record(p, us) for p, us in zip(plans, [8, 5, 1])]
        for row in rows: row['context_key'] = key
        functions = {p.name: (lambda *xs: gm(*xs)) for p in plans}
        costs = {id(functions['eager']): 8, id(functions['inductor']): 5, id(functions['sdnn']): 10}
        def compile_candidate(g, xs, plan):
            return functions[plan.name], self.record(plan, 1)['proof']
        def measure_candidates(fns, *args, **kwargs):
            return {name: dict(status='passed', checks={k: True for k in self.auto.CHECKS},
                        gpu_samples_us=[costs[id(fn)]] * 3, e2e_protocol='randomized_ab_ba_v1',
                        e2e_round_samples_us=[[costs[id(fn)]*2]*128]*2) for name,fn in fns.items()}
        prepared = {}
        with torch.no_grad(), mock.patch.object(codegen, 'baidu_target', return_value={'arch': 3}), \
             mock.patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
             mock.patch.object(self.auto, 'compile_candidate', side_effect=compile_candidate), \
             mock.patch.object(self.bench, 'measure_candidates', side_effect=measure_candidates):
            fn, decision = self.auto.calibrate_resources(gm, inputs, tolerance=rows[0]['tolerance'],
                environment_key='unit-environment', records=rows, prepared_candidates=prepared)
        self.assertEqual(decision['selected'], 'inductor')
        self.assertIs(fn, functions['inductor'])
        self.assertIs(fn, prepared['inductor'][0])

    def test_candidate_captures_have_common_batch_and_are_retained(self):
        constructed = []
        class FakeGraph:
            def __init__(self, fn, *args, calls=None):
                self.calls = calls
                self.us = fn()
                self.fresh_data_passed = True
                self.closed = False
                constructed.append(self)
            def sample(self): return self.us
            def close(self): self.closed = True
        retained = {}
        with mock.patch.object(self.bench, 'validate', return_value={'ordinary': True}), \
             mock.patch.object(self.bench, 'GraphMeasurement', FakeGraph), \
             mock.patch.object(self.bench, 'paired_end_to_end', return_value={'eager':[10]*128,'sdnn':[20]*128}):
            records = self.bench.measure_candidates({'eager': lambda:10, 'sdnn': lambda:20},
                                                   None, (), {}, retain=retained)
        self.assertEqual(len(retained['eager'].graphs), 3)
        self.assertEqual(len(retained['sdnn'].graphs), 3)
        self.assertTrue(all(g.calls == 400 for v in retained.values() for g in v.graphs))
        self.assertTrue(all(not g.closed for v in retained.values() for g in v.graphs))
        self.assertEqual(records['eager']['gpu_us'], 10)
        self.assertEqual(records['sdnn']['gpu_us'], 20)
        self.assertEqual(len(records['eager']['gpu_samples_us']), 18)
        self.assertTrue(all(g.closed for g in constructed[:2]))

    def test_launch_count_checks_calls_not_only_kernel_definitions(self):
        self.assertEqual(self.auto.launch_count('def call(args):\n    a.run(args)\n    a.run(args)\n'), 2)
        self.assertEqual(self.auto.launch_count('class Runner:\n    def call(self, args):\n        a.run(args)\n'), 1)
        with self.assertRaisesRegex(RuntimeError, 'dynamic launch'):
            self.auto.launch_count('def call(args):\n    for x in args:\n        a.run(x)\n')

    def test_context_changes_with_shape_stride_dtype_environment(self):
        gm = self.graph()
        xs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8))
        initial = self.auto.context_key(gm, xs, 'gpu0-sdk')
        self.assertEqual(initial, self.auto.context_key(gm, tuple(x + 1 for x in xs), 'gpu0-sdk'))
        self.assertNotEqual(initial, self.auto.context_key(gm, xs, 'gpu3-sdk'))
        self.assertNotEqual(initial, self.auto.context_key(gm, tuple(x.half() for x in xs), 'gpu0-sdk'))
        ys = (torch.randn(3, 4), xs[1], xs[2])
        self.assertNotEqual(initial, self.auto.context_key(gm, ys, 'gpu0-sdk'))


if __name__ == '__main__':
    unittest.main()
