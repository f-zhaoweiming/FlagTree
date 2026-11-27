import unittest

import torch
from torch import fx

from triton.experimental.graph_fusion._model_candidates import (
    discover_precision_partitions, _refine_compiled_partition, _profile_groups)
from triton.experimental.graph_fusion._pointwise_candidates import preserve_pointwise


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sensitive = torch.nn.Linear(2, 2, bias=False)
        self.safe = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.sensitive.weight.copy_(torch.eye(2) * 1.0001)
            self.safe.weight.copy_(torch.eye(2))
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.sensitive(x), self.safe(x)


class CandidateTests(unittest.TestCase):
    def test_captured_vendor_fp32_linear_uses_reviewed_fake_boundary(self):
        native=getattr(torch.ops._dynamo_workaround,'linear',None)
        if native is None:
            self.skipTest('Vendor capture entry is not installed')
        from triton.experimental.graph_fusion._operator_precision import _linear, preserve_native_compute
        from torch._subclasses.fake_tensor import FakeTensorMode
        graph=fx.Graph();x=graph.placeholder('x');w=graph.placeholder('w')
        y=graph.call_function(native,(x,w));graph.output(y)
        gm=fx.GraphModule({},graph)
        x.meta['example_value']=torch.randn(2,3);w.meta['example_value']=torch.randn(4,3)
        y.meta['example_value']=torch.randn(2,4)
        preserve_native_compute(gm)
        self.assertIs(y.target,_linear._opoverload)
        with FakeTensorMode() as mode:
            out=gm(mode.from_tensor(x.meta['example_value']),mode.from_tensor(w.meta['example_value']))
        self.assertEqual(out.shape,(2,4));self.assertEqual(out.dtype,torch.float32)

    def test_fp32_reviewed_linear_boundary_retains_fake_abi(self):
        from triton.experimental.graph_fusion._operator_precision import _linear, preserve_native_compute
        graph=fx.Graph()
        x=graph.placeholder('x');w=graph.placeholder('w')
        y=graph.call_function(_linear._opoverload,(x,w,None));graph.output(y)
        gm=fx.GraphModule({},graph)
        x.meta['example_value']=torch.randn(2,3)
        w.meta['example_value']=torch.randn(4,3)
        y.meta['example_value']=torch.randn(2,4)
        rows=preserve_native_compute(gm)
        self.assertEqual(rows[0]['action'],'native_compute_boundary')
        self.assertFalse(rows[0]['contains_low_precision'])
        self.assertIs(y.target,_linear._opoverload)
        expected=torch.nn.functional.linear(x.meta['example_value'],w.meta['example_value'])
        torch.testing.assert_close(gm(x.meta['example_value'],w.meta['example_value']),expected)

    def test_plain_fp32_linear_is_not_forced_to_vendor_boundary(self):
        from triton.experimental.graph_fusion._operator_precision import preserve_native_compute
        graph=fx.Graph();x=graph.placeholder('x');w=graph.placeholder('w')
        y=graph.call_function(torch.nn.functional.linear,(x,w));graph.output(y)
        gm=fx.GraphModule({},graph)
        for n,v in ((x,torch.randn(2,3)),(w,torch.randn(4,3)),(y,torch.randn(2,4))):
            n.meta['example_value']=v
        preserve_native_compute(gm)
        self.assertIs(y.target,torch.nn.functional.linear)

    def test_captured_graph_backend_accepts_implementation_options(self):
        from triton.experimental.graph_fusion._model_inductor import inductor_backend
        backend = inductor_backend(tolerance=dict(atol=.003, rtol=.003),
            environment_key='captured-domain', audit=[], precision_policy='preserve_operator',
            precision_options=dict(pointwise_policy='native', require_compiled=True))
        self.assertTrue(callable(backend))

    def test_per_node_implementation_choices_retain_unselected_rounding_boundaries(self):
        class Points(torch.nn.Module):
            def forward(self, x, y):
                return x * y + y
        args = (torch.randn(4).half(), torch.randn(4).half())
        gm = fx.symbolic_trace(Points())
        class Values(fx.Interpreter):
            def run_node(self, node):
                value = super().run_node(node)
                node.meta['example_value'] = value
                return value
        expected = Values(gm).run(*args)
        import operator
        mul = next(n for n in gm.graph.nodes if n.target is operator.mul)
        add = next(n for n in gm.graph.nodes if n.target is operator.add)
        records = preserve_pointwise(gm, {mul.name: 'native', add.name: 'generated'})
        self.assertEqual([r['node'] for r in records], [mul.name])
        self.assertIs(add.target, operator.add)
        self.assertEqual(gm(*args).dtype, torch.float16)
        torch.testing.assert_close(gm(*args), expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'missing graph nodes'):
            preserve_pointwise(gm, {'missing': 'native'})

    def test_per_node_native_choice_cannot_silently_skip_mutation(self):
        class Mutation(torch.nn.Module):
            def forward(self, x):
                return x.add_(1)
        gm = fx.symbolic_trace(Mutation())
        with self.assertRaisesRegex(ValueError, 'unsupported native nodes'):
            preserve_pointwise(gm, {'add_': 'native'})

    def qkv_model(self, *, sensitive=False, different_input=False, mutate=False):
        class QKV(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for name in ('q_proj', 'k_proj', 'v_proj', 'safe'):
                    layer = torch.nn.Linear(2, 2, bias=False)
                    with torch.no_grad():
                        layer.weight.copy_(torch.eye(2) * (1.0001 if sensitive and name == 'v_proj' else 1.))
                    setattr(self, name, layer)
                self.requires_grad_(False).eval()

            def forward(self, x):
                x = x.clone()
                q = self.q_proj(x)
                if mutate:
                    x.add_(1.)
                k = self.k_proj(x.clone() if different_input else x)
                return q, k, self.v_proj(x), self.safe(x)
        return QKV()

    def test_joint_qkv_fails_atomically_without_discarding_safe_linear(self):
        profiles, record = self.discover(self.qkv_model(sensitive=True))
        self.assertEqual(record['linear_modules'], 4)
        self.assertEqual(record['linear_groups'], 2)
        self.assertTrue(record['joint_qkv'][0]['joint'])
        self.assertIn(['safe'], [p['fp16_modules'] for p in profiles])
        for row in record['trials']:
            narrowed = set(row['fp16_modules']) & {'q_proj', 'k_proj', 'v_proj'}
            self.assertIn(len(narrowed), (0, 3))

    def test_joint_qkv_can_execute_all_three_directly_in_fp16(self):
        profiles, record = self.discover(self.qkv_model())
        self.assertIn(['k_proj', 'q_proj', 'safe', 'v_proj'], [p['fp16_modules'] for p in profiles])
        self.assertEqual(record['linear_groups'], 2)

    def test_qkv_names_do_not_group_distinct_inputs_or_mutated_versions(self):
        for options in ({'different_input': True}, {'mutate': True}):
            _, record = self.discover(self.qkv_model(**options))
            self.assertFalse(record['joint_qkv'][0]['joint'])
            self.assertEqual(record['linear_groups'], 4)

    def test_joint_qkv_preserves_tied_alias_and_compiled_refinement_group(self):
        model = self.qkv_model()
        model.v_alias = model.v_proj
        profiles, record = self.discover(model)
        profile = profiles[1]['fp16_modules']
        groups = _profile_groups(profile, record['precision_groups'])
        joint = ['k_proj', 'q_proj', 'v_alias', 'v_proj']
        self.assertIn(joint, groups)
        seen = []
        def build(paths, name):
            seen.append(paths)
            return lambda x: paths
        def validate(name, output, args, kwargs):
            if 'v_proj' in output:
                raise AssertionError('compiled QKV fails')
        paths, _ = _refine_compiled_partition(groups, build,
            validation_inputs=[((1,), {})], validate=validate, record={}, max_trials=8)
        self.assertEqual(paths, ['safe'])
        self.assertTrue(all(len(set(paths).intersection(joint)) in (0, 4) for paths in seen))
        with self.assertRaisesRegex(ValueError, 'splits'):
            _profile_groups(['v_proj'], record['precision_groups'])

    def test_joint_qkv_checks_all_validation_examples(self):
        class Conditional(self.qkv_model().__class__):
            def forward(self, x):
                return self.q_proj(x), self.k_proj(x if x[0, 0] == 1 else x.clone()), self.v_proj(x)
        model = Conditional()
        record = {}
        with torch.no_grad():
            discover_precision_partitions(model, prepare=lambda x: ((x,), {}),
                output_template=model(torch.ones(1, 2)),
                validation_inputs=[((torch.ones(1, 2),), {}), ((torch.ones(1, 2)*2,), {})],
                validate=lambda *args: None, record=record)
        self.assertFalse(record['joint_qkv'][0]['joint'])

    def test_compiled_failures_refine_native_passing_partition(self):
        calls, record = [], {}
        def build(paths, name):
            calls.append(tuple(paths))
            return lambda x: (paths, x)
        def validate(name, output, args, kwargs):
            if 'sensitive' in output[0]:
                raise AssertionError('compiled normalization consumes remaining error budget')
        paths, selected = _refine_compiled_partition([['sensitive'], ['safe', 'safe_alias']], build,
            validation_inputs=[((1,), {}), ((2,), {})], validate=validate, record=record,
            max_trials=8)
        self.assertEqual(paths, ['safe', 'safe_alias'])
        self.assertEqual(selected(3), (paths, 3))
        self.assertEqual(calls, [('safe', 'safe_alias', 'sensitive'), ('sensitive',), ('safe', 'safe_alias')])
        self.assertFalse(record['budget_exhausted'])

    def test_compiled_refinement_budget_cannot_admit_failed_output(self):
        record = {}
        def reject(*args):
            raise AssertionError('compiled output fails')
        paths, selected = _refine_compiled_partition([['a'], ['b']],
            lambda paths, name: lambda x: x, validation_inputs=[((1,), {})],
            validate=reject, record=record, max_trials=1)
        self.assertEqual(paths, [])
        self.assertIsNone(selected)
        self.assertTrue(record['budget_exhausted'])

    def discover(self, model, **kwargs):
        x = torch.ones(1, 2, dtype=next(model.parameters()).dtype)
        reference = model(x)
        def validate(name, output, args, keywords):
            for a, b in zip(output, reference):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        record = {}
        with torch.no_grad():
            profiles = discover_precision_partitions(model,
                prepare=lambda value: ((value,), {}), output_template=reference,
                validation_inputs=[((x,), {})], validate=validate, record=record, **kwargs)
        return profiles, record

    def test_sensitive_group_refined_without_mutating_source(self):
        model = Model()
        snapshot = {name: value.clone() for name, value in model.state_dict().items()}
        modules = {name: id(value) for name, value in model.named_modules()}
        profiles, record = self.discover(model)
        self.assertIn(['safe'], [x['fp16_modules'] for x in profiles])
        self.assertNotIn(['safe', 'sensitive'], [x['fp16_modules'] for x in profiles])
        self.assertFalse(record['budget_exhausted'])
        self.assertEqual(modules, {name: id(value) for name, value in model.named_modules()})
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], atol=0, rtol=0)
            self.assertEqual(value.dtype, torch.float32)

    def test_budget_is_explicit_and_does_not_admit_failed_profile(self):
        profiles, record = self.discover(Model(), max_trials=1)
        self.assertEqual(len(record['trials']), 1)
        self.assertTrue(record['budget_exhausted'])
        self.assertEqual([x['fp16_modules'] for x in profiles], [[]])

    def test_tied_module_aliases_are_one_group(self):
        model = Model()
        model.alias = model.safe
        profiles, record = self.discover(model)
        self.assertEqual(record['linear_groups'], 2)
        self.assertIn(['alias', 'safe'], [x['fp16_modules'] for x in profiles])

    def test_validation_covers_every_example(self):
        model = Model()
        record = {}
        calls = []
        def validate(name, output, args, kwargs):
            calls.append(float(args[0][0, 0]))
            if args[0][0, 0] == 2:
                raise AssertionError('second example rejects every partition')
        with torch.no_grad():
            profiles = discover_precision_partitions(model, prepare=lambda x: ((x,), {}),
                output_template=model(torch.ones(1, 2)),
                validation_inputs=[((torch.ones(1, 2),), {}), ((torch.ones(1, 2)*2,), {})],
                validate=validate, record=record)
        self.assertEqual([x['fp16_modules'] for x in profiles], [[]])
        self.assertEqual(calls, [1., 2.] * len(record['trials']))

    def test_device_failures_are_not_precision_rejections(self):
        model = Model()
        def validate(*args):
            raise RuntimeError('device failure')
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'device failure'):
            discover_precision_partitions(model, prepare=lambda x: ((x,), {}),
                output_template=model(torch.ones(1, 2)), validation_inputs=[((torch.ones(1, 2),), {})],
                validate=validate, record={})

    def test_rounded_sources_and_unfrozen_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original FP32'):
            self.discover(Model().half())
        with self.assertRaisesRegex(ValueError, 'frozen'):
            self.discover(Model().requires_grad_(True))

    def test_native_pointwise_keeps_broadcast_scalar_and_alpha(self):
        class Points(torch.nn.Module):
            def forward(self, x, bias):
                y = 1.702 * x
                return torch.add(y * torch.sigmoid(y), bias, alpha=2.)
        args = (torch.randn(3, 5), torch.randn(5))
        gm = fx.symbolic_trace(Points())
        class Values(fx.Interpreter):
            def run_node(self, node):
                value = super().run_node(node)
                node.meta['example_value'] = value
                return value
        expected = Values(gm).run(*args)
        rows = preserve_pointwise(gm, 'native')
        self.assertEqual(len(rows), 4)
        torch.testing.assert_close(gm(*args), expected, atol=0, rtol=0)
        self.assertTrue(any(row['native'] == 'aten.mul.Scalar' for row in rows))

    def test_native_mul_does_not_replace_add_sigmoid_or_mutation(self):
        class Points(torch.nn.Module):
            def forward(self, x):
                y = x.clone()
                y.add_(1)
                return y * torch.sigmoid(y) + 2
        gm = fx.symbolic_trace(Points())
        # Populate example tensors without replacing mutation in the graph.
        class Values(fx.Interpreter):
            def run_node(self, node):
                value = super().run_node(node)
                node.meta['example_value'] = value
                return value
        x = torch.randn(3)
        expected = Values(gm).run(x)
        rows = preserve_pointwise(gm, 'native_mul')
        self.assertEqual(len(rows), 1)
        self.assertTrue(any(n.target == 'add_' for n in gm.graph.nodes))
        torch.testing.assert_close(gm(x), expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
