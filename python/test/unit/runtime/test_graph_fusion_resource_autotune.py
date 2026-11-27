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

    def test_shared_sdnn_large_fp32_matrix_allows_native_tile_selection(self):
        a, b, bias = torch.empty(2048, 64), torch.empty(64, 2048), torch.empty(2048)
        self.assertIsNone(self.auto._shared_mm_tile((a, b, bias)))
        self.assertIsNone(self.auto._shared_mm_tile((a[:512], b, bias)))
        self.assertEqual(self.auto._shared_mm_tile((a[:64], b[:, :1024], bias[:1024])), (64, 32, 128, 2, 8))

    def test_shared_bf16_boundary_witness_depends_on_fp32_accumulation(self):
        # The production admission probe quantizes seed 61 inputs before promoting to FP32.
        gen = torch.Generator().manual_seed(61)
        a, b, bias = [torch.randn(shape, generator=gen).bfloat16().float()
                      for shape in ((2048, 64), (64, 2048), (2048,))]
        terms = a[290] * b[:, 1263]
        serial = torch.zeros(())
        for value in terms:
            serial = serial + value
        tree = terms
        while tree.numel() > 1:
            tree = tree[::2] + tree[1::2]
        upper, lower = serial + bias[1263], tree[0] + bias[1263]
        self.assertEqual((upper.view(torch.int32).item(), lower.view(torch.int32).item()),
                         (0x411f8000, 0x411f7fff))
        self.assertEqual((upper.bfloat16().item(), lower.bfloat16().item()), (10., 9.9375))
        self.assertEqual(upper.half().item(), lower.half().item())
        self.assertFalse(torch.isclose(upper.bfloat16(), lower.bfloat16(), atol=.03, rtol=.003))

    def test_grouped_relu_keeps_generic_candidate_and_runs_one_complete_grid(self):
        gm = self.graph()
        inputs = (torch.empty(2048, 64), torch.empty(64, 2048), torch.empty(2048))
        name = 'native_mm_grouped_relu'
        plans = self.auto.candidate_plans(gm, inputs, single_kernel=True)
        self.assertIn('native_mm', [p.name for p in plans])
        grouped = next(p for p in plans if p.name == name)
        self.assertTrue(grouped.single_kernel)
        for graph, xs in [(gm, (inputs[0][:512], *inputs[1:])),
                          (torch.fx.symbolic_trace(lambda a, b, c: a @ b + c), inputs),
                          (gm, tuple(x.half() for x in inputs))]:
            self.assertNotIn(name, [p.name for p in self.auto.candidate_plans(graph, xs)])
        from importlib import import_module
        package = self.auto.__package__
        mm = import_module(package + '._p800_native_mm')
        codegen = import_module(package + '._resource_codegen')
        launch = import_module(package + '._p800_bound_launch')
        kernel = mock.MagicMock()
        submit = kernel.__getitem__.return_value
        submit.return_value.metadata.is_sdnn = True
        mapped = {'effective': 'sdnn'}
        with mock.patch.object(mm, '_kernel', return_value=(None,) * 5), \
                mock.patch.object(mm, '_aligned_bias_kernel', return_value=kernel), \
                mock.patch.object(codegen, '_record_binary', return_value=mapped), \
                mock.patch.object(launch, 'bind', return_value=None):
            fn, proof = self.auto.compile_candidate(gm, inputs, grouped)
            self.assertEqual(tuple(fn(*inputs).shape), (2048, 2048))
        kernel.__getitem__.assert_called_once_with((128,))
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(submit.call_args.args[4:], (2048, 64, 128, 256))
        self.assertEqual(submit.call_args.kwargs, dict(RELU=True, num_warps=4, num_stages=3, is_sdnn=True))
        self.assertEqual((proof['route'], proof['schedule'], proof['kernel_count'], proof['kernel_evidence']),
                         (name, 'grouped_complete_tiles', 1, [mapped]))
        rows = [self.record(plans[0], 10), self.record(grouped, 7)]
        rows[1]['proof'] = proof
        winner, _ = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(winner.name, name)
        rows[1]['proof'] = dict(proof, kernel_evidence=[{'effective': 'cluster'}])
        winner, _ = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(winner.name, 'eager')

    def record(self, candidate, us):
        proof = {'kernel_count': len(candidate.regions), 'regions': [
            {'resource_verified': True, 'external_calls': 0,
             'kernel_evidence': [{'effective': region.resource}]} for region in candidate.regions]}
        return {'name': candidate.name, 'context_key': 'context', 'tolerance': {'atol': 3e-4, 'rtol': 3e-4},
                'status': 'passed', 'checks': {k: True for k in self.auto.CHECKS},
                'gpu_samples_us': [us, us * 1.01, us * 0.99], 'proof': proof,
                'e2e_protocol': 'randomized_ab_ba_v1', 'e2e_round_samples_us': [[us*2]*128, [us*2]*128]}

    def test_small_matrix_admits_multikernel_and_partitioned_plans_by_default(self):
        gm = self.graph()
        inputs = (torch.randn(16, 64), torch.randn(64, 128), torch.randn(128))
        plans = self.auto.candidate_plans(gm, inputs)
        self.assertEqual([p.name for p in plans], ['eager', 'inductor', 'native_mm', 'sdnn',
                                                   'sdnn_cluster_at_1', 'sdnn_cluster_at_2'])
        self.assertFalse(any(p.single_kernel for p in plans))
        strict = self.auto.candidate_plans(gm, inputs, single_kernel=True)
        self.assertEqual([p.name for p in strict], ['eager', 'inductor', 'native_mm', 'sdnn'])
        self.assertTrue(strict[-1].single_kernel)
        # Both binaries in the whole SDNN region are verified. Admit it and
        # let its measured ordinary-call latency compete with eager.
        sdnn = next(p for p in plans if p.name == 'sdnn')
        rows = [self.record(plans[0], 8), self.record(sdnn, 4)]
        proof = rows[1]['proof']
        proof['kernel_count'] = 2
        proof['regions'][0]['kernel_evidence'].append({'effective': 'sdnn'})
        selected, _ = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(selected.name, 'sdnn')

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

    def test_add_gelu_compares_the_complete_plans(self):
        fx = _load_module('_resource_fx')
        for mode in ('none', 'tanh'):
            gm = torch.fx.symbolic_trace(lambda a, b: torch.nn.functional.gelu(a + b, approximate=mode))
            xs = (torch.randn(4, 32).bfloat16(), torch.randn(32).bfloat16())
            plans = self.auto.candidate_plans(gm, xs, split_gelu=True)
            names = [p.name for p in plans]
            self.assertIn('cluster', names)
            self.assertNotIn('sdnn', names)
            self.assertEqual('sdnn' in [p.name for p in self.auto.candidate_plans(gm, tuple(x.float() for x in xs))],
                             mode == 'none')
            self.assertIn('cluster_cluster_at_1', names)
            hybrid, = [p for p in self.auto.candidate_plans(gm, xs) if len(p.regions) > 1]
            self.assertEqual(hybrid.name, 'eager_add_cluster_gelu')
            self.assertEqual([r.resource for r in hybrid.regions], ['eager', 'cluster'])
            strict = self.auto.candidate_plans(gm, xs, single_kernel=True, split_gelu=True)
            self.assertFalse(any(len(p.regions) > 1 for p in strict))
            add, gelu = (fx.build_region(gm, i, i + 1) for i in (0, 1))
            summed, = add.gm(*xs)
            actual, = gelu.gm(summed)
            self.assertEqual(summed.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, gm(*xs), atol=0, rtol=0)

        gm = torch.fx.symbolic_trace(lambda a, b: (a + b, torch.nn.functional.gelu(b, approximate='tanh')))
        self.assertEqual([p.name for p in self.auto.candidate_plans(gm, xs, split_gelu=True)],
                         ['eager', 'inductor', 'cluster', 'sdnn'])

    def test_eager_add_cluster_gelu_requires_verified_consumer_and_original_producer(self):
        gm = torch.fx.symbolic_trace(lambda a, b: torch.nn.functional.gelu(b + a, approximate='tanh'))
        xs = (torch.zeros(8, 256, dtype=torch.bfloat16), torch.zeros(256, dtype=torch.bfloat16))
        plans = self.auto.candidate_plans(gm, xs)
        hybrid = next(p for p in plans if p.name == 'eager_add_cluster_gelu')
        for bad in [(xs[0].float(), xs[1]), (xs[0].T, xs[1]), (xs[0], xs[1][:8])]:
            self.assertNotIn(hybrid, self.auto.candidate_plans(gm, bad))
        eager = self.record(plans[0], 20)
        fast = self.record(hybrid, 10)
        producer = dict(route='eager_add', resource='eager', resource_verified=False,
                        external_calls=1, node_ids=hybrid.regions[0].node_ids)
        fast['proof']['regions'][0] = producer
        fast['proof']['regions'][1].update(route='bf16_gelu', direct_launches=1)
        def winner():
            return self.auto.select_measurement(plans, [eager, fast], 'context', eager['tolerance'])[0]
        self.assertEqual(winner(), hybrid)
        producer['external_calls'] = 2
        self.assertEqual(winner(), plans[0])
        producer['external_calls'] = 1
        fast['proof']['regions'][1]['kernel_evidence'] = []
        self.assertEqual(winner(), plans[0])

    def test_bf16_eager_add_cluster_gelu_tail_and_output_structure(self):
        if not torch.cuda.is_available() or _load_module('_resource_codegen').baidu_target() is None:
            self.skipTest('requires P800')
        def graph(bias, x):
            summed = torch.add(input=bias, other=x)
            result = torch.nn.functional.gelu(input=summed, approximate='tanh')
            return summed, {'gelu': result, 'alias': result}
        gm = torch.fx.symbolic_trace(graph)
        with torch.no_grad():
            for rows, cols in [(8, 1024), (65, 1025)]:
                # Offset views keep stride one and exercise unaligned inputs.
                xs = (torch.randn(cols + 1).bfloat16().cuda()[1:],
                      torch.randn(rows * cols + 1).bfloat16().cuda()[1:].view(rows, cols))
                plan = next(p for p in self.auto.candidate_plans(gm, xs) if p.name == 'eager_add_cluster_gelu')
                fn, proof = self.auto.compile_candidate(gm, xs, plan)
                actual, output = fn(*xs)
                expected, reference = gm(*xs)
                torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
                torch.testing.assert_close(output['gelu'].cpu(), reference['gelu'].cpu(), atol=.03, rtol=.003)
                self.assertIs(output['alias'], output['gelu'])
                self.assertNotEqual(output['gelu'].data_ptr(), fn(*xs)[1]['gelu'].data_ptr())
                self.assertEqual(proof['regions'][1]['kernel_evidence'][0]['effective'], 'cluster')

    def test_bf16_vendor_matmul_preserves_live_operands_and_output(self):
        flat = _load_module('_p800_flat')
        gm = torch.fx.symbolic_trace(lambda a, b, bias: torch.nn.functional.silu((a @ b) + bias))
        xs = (torch.randn(64, 64).bfloat16(), torch.randn(64, 1024).bfloat16(), torch.randn(1024).bfloat16())
        plans = self.auto.candidate_plans(gm, xs)
        choice, = [p for p in plans if p.name == 'eager_mm_cluster_activation']
        self.assertEqual([(r.resource, r.start, r.end) for r in choice.regions], [('eager', 0, 1), ('cluster', 1, 3)])
        self.assertNotIn(choice.name, [p.name for p in self.auto.candidate_plans(gm, xs, single_kernel=True)])

        def reference(spec, inputs):
            self.assertEqual(spec['kind'], 'silu')
            return lambda pair: (torch.nn.functional.silu(pair[0] + pair[1]),), dict(route='flat_add', direct_launches=1)

        with mock.patch.object(flat, 'compile', side_effect=reference):
            fn, proof = self.auto.compile_candidate(gm, xs, choice)
        self.assertEqual(proof['regions'][0]['external_calls'], 1)
        for offset in (0, 1):
            values = tuple((x + offset).contiguous() for x in xs)
            torch.testing.assert_close(fn(*values), gm(*values), atol=0, rtol=0)
        supported = (torch.empty(512, 64, dtype=torch.bfloat16), *xs[1:])
        with mock.patch.object(_load_module('_p800_layernorm'), 'toolchain_available', return_value=True):
            self.assertIn(choice.name, [p.name for p in self.auto.candidate_plans(gm, supported)])
        unsupported = (torch.empty(513, 64, dtype=torch.bfloat16), *xs[1:])
        self.assertNotIn(choice.name, [p.name for p in self.auto.candidate_plans(gm, unsupported)])

    def test_flat_tile_timing_uses_the_same_captured_launch(self):
        flat = _load_module('_p800_flat')
        launch = mock.Mock()
        inputs = (mock.sentinel.a, mock.sentinel.b, mock.sentinel.output, {'xblock': 4096, 'warps': 8})

        def capture(run, args, calls):
            self.assertEqual(args, ())
            self.assertEqual(calls, 32)
            run()
            return mock.sentinel.graph, None

        with mock.patch.object(self.bench, '_capture', side_effect=capture), mock.patch.object(
                self.bench, 'replay_us', side_effect=[8.0, 3.0, 5.0, 4.0, 100.0]) as replay:
            self.assertEqual(flat._device_tile_us(launch, *inputs), 5.0)
        launch.assert_called_once_with(*inputs)
        self.assertEqual(replay.call_args_list, [mock.call(mock.sentinel.graph, 32)] * 5)

    def test_bf16_flat_activations_require_supported_operands(self):
        flat = _load_module('_p800_flat')
        x = torch.randn(2, 1024).bfloat16()
        b = torch.randn(1024).bfloat16()
        gelu = torch.fx.symbolic_trace(lambda a, b: torch.nn.functional.gelu(a + b))
        silu = torch.fx.symbolic_trace(lambda a, b: torch.nn.functional.silu(a + b))
        self.assertEqual(flat.match(gelu, (x, b))['dtype'], 'bfloat16')
        self.assertEqual(flat.match(silu, (x, b))['kind'], 'silu')
        with mock.patch.object(_load_module('_p800_layernorm'), 'toolchain_available', return_value=True):
            self.assertEqual(flat.match(silu, (torch.empty(512, 1024, dtype=x.dtype), b))['kind'], 'silu')
        self.assertIsNone(flat.match(silu, (torch.empty(513, 1024, dtype=x.dtype), b)))
        self.assertIsNone(flat.match(gelu, (x, b.float())))
        self.assertIsNone(flat.match(gelu, (x[:, ::2], b[::2])))

    def test_opt_in_reduction_split_preserves_product_dtype_and_keepdim(self):
        from test_graph_fusion_static import _load_module
        fx = _load_module('_resource_fx')
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for keep in (False, True):
                gm = torch.fx.symbolic_trace(lambda a, b: (a * b).sum(-1, keepdim=keep))
                inputs = (torch.randn(4, 128).to(dtype), torch.randn(128).to(dtype))
                plans = self.auto.candidate_plans(gm, inputs, single_kernel=False, split_reductions=True)
                names = [p.name for p in plans]
                consumers = ('cluster', 'sdnn') if dtype == torch.float32 else ('cluster',)
                self.assertEqual([n for n in names if '_at_' in n],
                                 [f'{p}_{c}_at_1' for p in ('cluster', 'sdnn') for c in consumers])
                self.assertEqual('sdnn' in names, dtype == torch.float32)
                if dtype != torch.float32:
                    runtime = _load_module('_resource_runtime')
                    with self.assertRaisesRegex(RuntimeError, 'no safe native lowering'):
                        native = self.auto.Segment(0, 2, 'sdnn', tuple(n.name for n in gm.graph.nodes
                                                                       if n.op in ('call_method', 'call_function')))
                        runtime.prepare_regions(gm, inputs, [native])
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
        self.assertEqual([p.name for p in default], ['eager', 'inductor', 'cluster', 'sdnn'])
        self.assertFalse(any(p.single_kernel for p in default))
        self.assertEqual(self.auto.candidate_plans(gm, xs, single_kernel=True),
                         self.auto.candidate_plans(gm, xs, single_kernel=True, split_reductions=True))
        add_sum = torch.fx.symbolic_trace(lambda a, b: (a + b).sum(-1))
        self.assertEqual(len(self.auto.candidate_plans(add_sum, xs, single_kernel=False, split_reductions=True)), 4)

    def test_selects_full_plan_time_without_double_counted_transfer(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=False)
        rows = [self.record(p, t) for p, t in zip(plans, [20, 15, 12, 7, 9])]
        p, result = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(p.name, 'sdnn_cluster_at_1')
        self.assertEqual(result['calibration_gpu_us'], 7)
        self.assertFalse(result['fallback'])

    def test_public_wall_time_ranks_admitted_plans_with_opposite_device_order(self):
        plans = self.auto.candidate_plans(self.graph(), (), single_kernel=False)
        rows = [self.record(plan, t) for plan, t in zip(plans, [20, 30, 1, 7, 9])]
        for r in rows:
            r['e2e_protocol'] = 'public_module_ab_ba_v1'
        rows[2]['e2e_round_samples_us'] = [[19] * 128, [17] * 128]
        plan, decision = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'],
                                                      e2e_protocol='public_module_ab_ba_v1')
        self.assertEqual(plan.name, plans[3].name)
        self.assertEqual(decision['calibration_gpu_us'], 7)
        self.assertEqual(decision['e2e_acceptance']['ranking'], 'mean_of_two_round_medians_us')
        rows[0]['e2e_protocol'] = 'randomized_ab_ba_v1'
        with self.assertRaises(RuntimeError):
            self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'],
                                         e2e_protocol='public_module_ab_ba_v1')

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

    def test_public_route_prefers_repeatable_gain_over_direct_module(self):
        plans = [self.auto.Candidate(name) for name in ('eager', 'sdnn', 'cluster')]
        rows = [self.record(plan, us) for plan, us in zip(plans, (15, 8, 6))]
        # Cluster wins on average but loses the second round to direct eager.
        rows[2]['e2e_round_samples_us'] = [[8]*128, [20]*128]
        rows[0]['direct_eager_round_samples_us'] = [[19]*128, [19]*128]
        plan, decision = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(plan.name, 'sdnn')
        self.assertEqual(decision['e2e_acceptance']['direct_eager']['faster_routes'], ['sdnn'])
        # The primary ranking and the direct reference have independent
        # series. Cluster clears the direct threshold in its own paired run.
        rows[2]['vs_direct_round_samples_us'] = [[15]*128, [16]*128]
        plan, decision = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(plan.name, 'cluster')
        self.assertEqual(decision['e2e_acceptance']['direct_eager']['faster_routes'], ['cluster', 'sdnn'])
        rows[0]['direct_eager_round_samples_us'] = [[7]*128, [7]*128]
        plan, decision = self.auto.select_measurement(plans, rows, 'context', rows[0]['tolerance'])
        self.assertEqual(plan.name, 'cluster')
        self.assertEqual(decision['e2e_acceptance']['direct_eager']['faster_routes'], [])

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

    def test_single_infinity_in_middle_broadcast_operand_cannot_hide_behind_nan(self):
        gm = torch.fx.symbolic_trace(lambda x, b, c: (x + b + c).sum(-1))
        xs = (torch.ones(4, 8), torch.ones(8), torch.ones(4, 8))

        def broken(x, b, c):
            result = gm(x, b, c)
            return torch.where(result.isposinf(), float('nan'), result) if b.isposinf().any() else result

        with torch.no_grad():
            with self.assertRaisesRegex(AssertionError, r'special input 1 \(\+Inf\)'):
                self.bench.validate(broken, gm, xs, {'atol': 3e-4, 'rtol': 3e-4})

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
        costs = {id(fn): {'eager': 8, 'inductor': 5}.get(name, 10) for name, fn in functions.items()}
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

    def test_invalid_replay_sample_rejects_only_that_candidate(self):
        class Graph:
            def __init__(self, fn, *args, calls=None):
                self.calls = calls
                self.invalid = fn() == 20 and calls != 8
                self.fresh_data_passed = True

            def sample(self):
                if self.invalid:
                    raise RuntimeError('invalid replay elapsed time')
                return 10

            def close(self):
                pass

        def paired(functions, inputs, **kwargs):
            return {name: [10] * 128 for name in functions}

        with mock.patch.object(self.bench, 'validate', return_value={'ordinary': True}), \
             mock.patch.object(self.bench, 'GraphMeasurement', Graph), \
             mock.patch.object(self.bench, 'paired_end_to_end', side_effect=paired):
            records = self.bench.measure_candidates({'eager': lambda: 10, 'sdnn': lambda: 20},
                                                     None, (), {})
        self.assertEqual(records['eager']['status'], 'passed')
        self.assertEqual(records['sdnn']['status'], 'rejected')
        self.assertIn('invalid replay elapsed time', records['sdnn']['error'])

    def test_replay_replaces_unusable_event_pair_and_normalizes_batch(self):
        graph = mock.Mock()
        events = [mock.Mock() for _ in range(4)]
        events[0].elapsed_time.return_value = float('nan')
        events[2].elapsed_time.return_value = .160
        with mock.patch.object(torch.cuda, 'synchronize'), mock.patch.object(torch.cuda, 'Event', side_effect=events):
            self.assertEqual(self.bench.replay_us(graph, 8), 5.)
        self.assertEqual(graph.replay.call_count, 5)
        events[0].elapsed_time.assert_called_once_with(events[1])
        events[2].elapsed_time.assert_called_once_with(events[3])

    def test_calibration_retries_only_invalid_event_time(self):
        invalid = self.bench.InvalidReplayTiming('invalid replay elapsed time')
        record = {'timing_retries': 0}
        graph = mock.Mock()
        graph.sample.side_effect = [invalid, 7., invalid, invalid, RuntimeError('launch failed')]
        self.assertEqual(self.bench._calibration_sample(graph, record), 7.)
        with self.assertRaises(self.bench.InvalidReplayTiming):
            self.bench._calibration_sample(graph, record)
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            self.bench._calibration_sample(graph, record)
        self.assertEqual(record['timing_retries'], 2)

    def test_launch_count_checks_calls_not_only_kernel_definitions(self):
        self.assertEqual(self.auto.launch_count('def call(args):\n    a.run(args)\n    a.run(args)\n'), 2)
        self.assertEqual(self.auto.launch_count('class Runner:\n    def call(self, args):\n        a.run(args)\n'), 1)
        self.assertIsNone(self.auto.launch_count('def call(args):\n    for x in args:\n        a.run(x)\n'))

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
