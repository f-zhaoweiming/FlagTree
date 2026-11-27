import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from triton.experimental.graph_fusion._arithmetic_boundaries import (
    ArithmeticBoundaryError, arithmetic_contract, check_arithmetic_boundaries)
from triton.experimental.graph_fusion._local_precision import (
    LocalPrecisionSpace, discover_local_precision_profiles, _refinement_proposals)


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4,), 1.0003))

    def forward(self, x):
        return x * self.weight


class Chain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = Scale()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(self.scale(x))


def valued_graph(model, x):
    graph = torch.fx.symbolic_trace(model)

    class Values(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            node.meta['example_value'] = result
            return result

    Values(graph).run(x)
    return graph


class ArithmeticBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        torch.manual_seed(41)

    def space(self, model, reviewed=True):
        master = model.eval().requires_grad_(False)
        low = copy.deepcopy(master).half()
        x = torch.randn(2, 4)
        record = {}
        space = LocalPrecisionSpace(low, source_tensors=dict(master.named_parameters()),
            prepare=lambda x: ((x.half(),), {}), source_prepare=lambda x: ((x,), {}),
            validation_inputs=[((x,), {})], record=record, arithmetic_reviewed_sdk=reviewed)
        return master, low, x, space, record

    def test_unknown_matrix_cannot_be_promoted_by_local_source_binding(self):
        with torch.no_grad():
            master, low, x, space, record = self.space(Chain())
            with self.assertRaisesRegex(ArithmeticBoundaryError, 'native arithmetic boundary changed'):
                space.build(('linear',))
            self.assertEqual(low.linear.weight.dtype, torch.float16)
            self.assertFalse(record['arithmetic_boundary_checks'][-1]['passed'])

    def test_proven_fp32_region_still_promotes_before_native_matrix_boundary(self):
        with torch.no_grad():
            master, low, x, space, record = self.space(Chain())
            candidate = space.build(('scale',))
            self.assertIs(candidate.scale.region.weight, master.scale.weight)
            self.assertEqual(candidate.linear.region.weight.dtype, torch.float16)
            expected = low.linear(master.scale(x).half())
            torch.testing.assert_close(candidate(x), expected, atol=0, rtol=0)
            fine, seed = space.refined(('scale',), {})
            self.assertEqual(seed, {'scale'})
            with self.assertRaises(ArithmeticBoundaryError):
                fine.build(('linear',))

    def test_unwrapped_native_half_operator_sees_original_precision(self):
        class HalfRelu(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale = Scale()
            def forward(self, x):
                return torch.relu(self.scale(x))
        with torch.no_grad():
            _, _, _, space, _ = self.space(HalfRelu())
            with self.assertRaisesRegex(ArithmeticBoundaryError, 'native arithmetic boundary changed'):
                space.build(('scale',))

    def test_unreviewed_sdk_does_not_inherit_fp32_region_permission(self):
        with torch.no_grad():
            _, _, _, space, _ = self.space(Chain(), reviewed=False)
            with self.assertRaises(ArithmeticBoundaryError):
                space.build(('scale',))

    def test_unsafe_profiles_never_reach_precision_scoring_or_failed_screen_reopening(self):
        with torch.no_grad():
            _, low, x, space, record = self.space(Chain())
            validated = []
            def validate(name, output, args, kwargs):
                validated.append(name)
                raise AssertionError('Deliberately force search; not a numerical acceptance threshold')
            profiles = discover_local_precision_profiles(space, output_template=low(x.half()),
                validation_inputs=[((x,), {})], validate=validate, score=lambda name: (1,),
                record=record, max_trials=6)
            self.assertEqual(profiles, [])
            rejected = [r for r in record['trials'] if not r.get('arithmetic_boundary_passed', True)]
            self.assertTrue(rejected)
            self.assertTrue(all(r['name'] not in validated for r in rejected))
            passing = dict(selected=[], passed=True, rank=[0], promoted_state_elements=0)
            self.assertEqual(_refinement_proposals([passing, *rejected], passing), [passing])

    def test_candidate_reclassification_cannot_authorize_half_relu_widening(self):
        model = torch.nn.ReLU()
        x = torch.randn(2, 4)
        original = arithmetic_contract(valued_graph(model, x.half()), reviewed_sdk=True)
        candidate = valued_graph(model, x)
        self.assertEqual(original[0]['arithmetic'], 'fp16')
        self.assertEqual(arithmetic_contract(candidate, reviewed_sdk=True)[0]['arithmetic'], 'fp32')
        with self.assertRaises(ArithmeticBoundaryError):
            check_arithmetic_boundaries(candidate, [original], reviewed_sdk=True)

    def test_new_shape_and_new_arithmetic_branch_require_revalidation(self):
        class Branch(torch.nn.Module):
            def __init__(self, extra=False):
                super().__init__(); self.extra = extra
            def forward(self, x):
                return x.sigmoid().sin() if self.extra else x.sigmoid()
        x = torch.randn(2, 4)
        original = arithmetic_contract(valued_graph(Branch(), x.half()), reviewed_sdk=True)
        for candidate in (valued_graph(Branch(True), x.half()),
                          valued_graph(Branch(), torch.randn(3, 4).half())):
            with self.assertRaises(ArithmeticBoundaryError):
                check_arithmetic_boundaries(candidate, [original], reviewed_sdk=True)
        check_arithmetic_boundaries(valued_graph(Branch(), x), [original], reviewed_sdk=True)

    def test_formal_search_splits_coarse_region_and_passes_proof_to_compiler(self):
        from triton.experimental.graph_fusion._arithmetic_search import graph_arithmetic, choose_domain
        from triton.experimental.graph_fusion._captured_candidates import captured_model_candidate_routes
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = torch.nn.Sequential(Scale(), torch.nn.Linear(4, 4, bias=False))
            def forward(self, x):
                return self.block(x)
        with torch.no_grad():
            master = Nested().eval().requires_grad_(False)
            master.block[1].weight.copy_(torch.eye(4))
            low = copy.deepcopy(master).half()
            x = torch.full((2, 4), 1.0003)
            wanted = low.block[1](master.block[0](x).half())
            scores, record, builds = {}, {}, []
            def inspect(model, *, prepare, validation_inputs, record):
                from triton.experimental.graph_fusion._model import enable_vendor_capture
                enable_vendor_capture()
                graphs = []
                def capture(gm, inputs):
                    graphs.append(gm); return gm.forward
                compiled = torch.compile(model, backend=capture, fullgraph=True)
                for args, kwargs in validation_inputs:
                    a, k = prepare(*args, **kwargs); compiled(*a, **k)
                rows = [r for gm in graphs for r in graph_arithmetic(gm, reviewed_sdk=True)]
                record.update(domain=choose_domain(rows), runtime=dict(matched=True))
                return graphs
            def validate(name, output, *args):
                scores[name] = (float((output.float() - wanted.float()).abs().sum()),)
                torch.testing.assert_close(output, wanted, atol=0, rtol=0)
            def build(model, **options):
                builds.append((model, options))
                return SimpleNamespace(_f06_audit=[], _f06_resource_allowlist=None)
            with patch('triton.experimental.graph_fusion._arithmetic_search.inspect_model_arithmetic', inspect), patch(
                    'triton.experimental.graph_fusion._model_pipeline.compile_pipeline', build):
                routes = captured_model_candidate_routes(low, prepare=lambda x: ((x.half(),), {}),
                    source_prepare=lambda x: ((x,), {}), source_tensors=dict(master.named_parameters()),
                    output_template=low(x.half()), validation_inputs=[((x,), {})], validate=validate,
                    precision_score=lambda name: scores[name], record=record,
                    tolerance=dict(atol=3e-4, rtol=3e-4), linear_schedules=('native',),
                    pointwise_policies=('native',), max_precision_trials=12)
            self.assertTrue(routes)
            self.assertEqual(record['arithmetic_search']['domain'], 'unresolved')
            self.assertTrue(record['local_precision_search']['arithmetic_splits'])
            profiles = record['local_precision_profiles']
            self.assertEqual(profiles[0]['selected'], ['block.0'])
            checked = [(m, o) for m, o in builds if 'arithmetic_boundaries' in o['precision_options']]
            self.assertTrue(checked)
            for candidate, options in checked:
                block = getattr(candidate.block, 'region', candidate.block)
                self.assertEqual(block[1].region.weight.dtype, torch.float16)
                self.assertTrue(options['precision_options']['arithmetic_boundaries']['contracts'])

    def test_inductor_entry_rechecks_contract_before_lowering(self):
        from triton.experimental.graph_fusion._model_inductor import inductor_backend
        model = torch.nn.ReLU()
        x = torch.randn(2, 4)
        contracts = [arithmetic_contract(valued_graph(model, x.half()), reviewed_sdk=True)]
        backend = inductor_backend(tolerance=dict(atol=3e-4, rtol=3e-4),
            environment_key='arithmetic-contract-test', audit=[], precision_policy='preserve_operator',
            precision_options=dict(arithmetic_boundaries=dict(contracts=contracts, reviewed_sdk=True)),
            _capture_cost_graph=True, _cost_graph_observer=lambda *args: None)
        with torch.no_grad(), self.assertRaisesRegex(ArithmeticBoundaryError, 'native arithmetic boundary changed'):
            backend(valued_graph(model, x), [x])
        audit = []
        backend = inductor_backend(tolerance=dict(atol=3e-4, rtol=3e-4),
            environment_key='arithmetic-contract-pass', audit=audit, precision_policy='preserve_operator',
            precision_options=dict(arithmetic_boundaries=dict(contracts=contracts, reviewed_sdk=True)),
            _capture_cost_graph=True, _cost_graph_observer=lambda *args: None)
        with torch.no_grad():
            call = backend(valued_graph(model, x.half()), [x.half()])
            torch.testing.assert_close(call(x.half()), model(x.half()), atol=0, rtol=0)
        self.assertEqual(audit[0]['arithmetic_boundaries']['protected_nodes'], 1)

    def test_independent_candidates_keep_guards_without_global_recompile_budget_change(self):
        with torch.no_grad(), torch._dynamo.config.patch(recompile_limit=2):
            _, low, x, space, record = self.space(Chain())
            for i in range(12):
                candidate = space.build(('scale',) if i % 2 else ())
                self.assertEqual(candidate(x).dtype, torch.float16)
            self.assertEqual(torch._dynamo.config.recompile_limit, 2)
            self.assertTrue(all(r['passed'] for r in record['arithmetic_boundary_checks']))

    def test_public_final_cast_is_a_conversion_not_new_unknown_arithmetic(self):
        from triton.experimental.graph_fusion._p800_model_layernorm import _final_cast
        class Public(torch.nn.Module):
            def forward(self, x):
                return _final_cast(x.sigmoid(), torch.float16)
        x = torch.randn(2, 4)
        # Use the same sigmoid target form in the original and candidate.
        class Body(torch.nn.Module):
            def forward(self, x): return x.sigmoid()
        original = [arithmetic_contract(valued_graph(Body(), x.half()), reviewed_sdk=True)]
        candidate = valued_graph(Public(), x)
        check_arithmetic_boundaries(candidate, original, reviewed_sdk=True)


if __name__ == '__main__':
    unittest.main()
