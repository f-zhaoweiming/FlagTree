"""Whole-output admission and bounded noisy-block remeasurement."""
import unittest

from triton.experimental.graph_fusion._whole_model_selection import ModelRoute, select_whole_model


class SelectionTests(unittest.TestCase):
    def test_buffer_variants_require_full_outputs_and_complete_latency(self):
        good = lambda value: (value, value)
        bad = lambda value: (value, value + 1)
        class Pipeline:
            def __call__(self, value): return value, value
            def _f06_buffer_variants(self): return {'lm512': bad, 'lm1024': good}
        base = Pipeline(); record = {}; measured = []
        def validate(name, out, args, kwargs):
            if out != (args[0], args[0]): raise AssertionError('auxiliary output')
        def measure(fn):
            measured.append(fn)
            return [.005 if fn is good else .01]
        selected = select_whole_model({'graph': ModelRoute(base)},
            validation_inputs=[((2,), {}), ((3,), {})], validate=validate,
            measure=measure, record=record)
        self.assertIs(selected, good)
        self.assertNotIn(bad, measured)
        self.assertEqual(record['route'], 'graph__lm1024')
        self.assertEqual(record['resource_variants']['graph__lm1024']['dimension'], 'buffer')

    def test_resource_near_tie_keeps_valid_parent_when_native_precision_fails(self):
        class Pipeline:
            def __call__(self):return 'parent'
            def _f06_resource_variants(self):return {'sdnn':alternative}
        base=Pipeline();alternative=lambda:'sdnn';record={}
        def validate(name,out,args,kwargs):
            if name=='native':raise AssertionError('FP32 reference failed')
        result=select_whole_model({'native':ModelRoute(lambda:'bad'),'graph':ModelRoute(base)},
            validation_inputs=[((),{})],validate=validate,measure=lambda fn:[.01 if fn is base else .00995],
            record=record,preferred='native')
        self.assertIs(result,base)
        self.assertEqual(record['route'],'graph')

    def test_resource_variants_are_validated_as_complete_calls_before_timing(self):
        measured=[];expanded={};record={};calls=[]
        class Pipeline:
            def __init__(self, kind):self.kind=kind
            def __call__(self, value):
                calls.append((self.kind,value))
                return {'main':value, 'aux':0 if self.kind=='sdnn' else 1}
            def _f06_resource_variants(self):
                self.assert_prepared = any(k==self.kind for k,_ in calls)
                if self.kind=='base':return {'cluster':cluster,'sdnn':sdnn}
                return {}
        base=Pipeline('base');cluster=Pipeline('cluster');sdnn=Pipeline('sdnn')
        def validate(name, output, args, kwargs):
            if output['aux']!=1:raise AssertionError('Full auxiliary output failed')
        def measure(fn):
            measured.append(fn.kind)
            return [.002 if fn.kind=='cluster' else .01]
        selected=select_whole_model({'graph':ModelRoute(base)},
            validation_inputs=[((2,),{}),((3,),{})], validate=validate,measure=measure,
            record=record,rounds=4,expanded_routes=expanded)
        self.assertIs(selected,cluster)
        self.assertTrue(base.assert_prepared)
        self.assertNotIn('sdnn',measured)
        self.assertEqual(record['route'],'graph__cluster')
        self.assertEqual(record['rejected'][0]['route'],'graph__sdnn')
        self.assertEqual(set(expanded),{'graph','graph__cluster','graph__sdnn'})
        self.assertEqual(record['candidates']['graph__cluster']['validated_examples'],2)
        self.assertEqual(calls,[('base',2),('base',3),('cluster',2),('cluster',3),('sdnn',2)])

    def test_rejected_baseline_does_not_hide_valid_sdnn_variant(self):
        good=lambda x:x
        class Bad:
            def __call__(self,x):return x+1
            def _f06_resource_variants(self):return {'sdnn':good}
        def validate(name,out,args,kwargs):
            if out!=args[0]:raise AssertionError('original FP32 mismatch')
        record={}
        selected=select_whole_model({'graph':ModelRoute(Bad())},validation_inputs=[((3,),{})],
            validate=validate,measure=lambda fn:[.01],record=record)
        self.assertIs(selected,good)
        self.assertEqual(record['route'],'graph__sdnn')

    def select(self, native, fused, *, record=None, **options):
        record = {} if record is None else record
        routes = {name: ModelRoute(lambda n=name: n) for name in ('native', 'fused', 'incorrect')}
        counts = dict.fromkeys(routes, 0)

        def validate(name, output, args, kwargs):
            if output == 'incorrect':
                raise AssertionError('Incorrect complete output')

        def measure(fn):
            name = next(n for n, route in routes.items() if route.call is fn)
            self.assertNotEqual(name, 'incorrect')
            index = counts[name]
            counts[name] += 1
            value = (native if name == 'native' else fused)[index]
            if isinstance(value, Exception):
                raise value
            return [value] * 10

        chosen = select_whole_model(routes, validation_inputs=[((), {})], validate=validate,
                                    measure=measure, record=record, preferred='native', **options)
        self.assertIs(chosen, routes[record['route']].call)
        self.assertEqual(counts['incorrect'], 0)
        return record, counts

    def test_full_noisy_block_replaced_and_preserved_for_audit(self):
        native = [.0093712, .0102020, .0093477, .0093467]
        fused = [.0053311, .0053118, .0053047, .0053366]
        before, _ = self.select(native, fused, max_remeasurements=0)
        self.assertEqual(before['route'], 'native')
        after, counts = self.select(native + [.00935]*4, fused + [.00532]*4)
        self.assertEqual(after['route'], 'fused')
        self.assertEqual(counts['native'], 8)
        old = after['timing_attempts'][0]['candidates']['native']['raw_seconds']
        self.assertEqual(old, [x for value in native for x in [value]*10])
        self.assertEqual(after['candidates']['native']['raw_seconds'], [.00935]*40)
        self.assertTrue(after['gain_check']['stable'])

    def test_persistent_drift_is_bounded_and_rejected(self):
        result, counts = self.select([.025, .03, .045, .055]*2, [.012]*8)
        self.assertEqual(result['route'], 'native')
        self.assertEqual(counts['native'], 8)

    def test_fresh_tie_or_reversed_winner_keeps_native(self):
        for fresh in (.00995, .012):
            result, _ = self.select([.01, .012, .01, .01] + [.01]*4, [.005]*4 + [fresh]*4)
            self.assertEqual(result['route'], 'native')

    def test_single_spike_cannot_promote_slower_route(self):
        result, counts = self.select([.01, .01, .04, .01], [.012]*4)
        self.assertEqual(result['route'], 'native')
        self.assertEqual(counts['native'], 4)

    def test_stable_winner_does_not_trigger_remeasurement(self):
        result, counts = self.select([.01]*4, [.005]*4)
        self.assertEqual(result['route'], 'fused')
        self.assertEqual(counts['native'], 4)
        audit=result['interface_audit']
        self.assertEqual(audit['native']['status'],'slower_measured')
        self.assertEqual(audit['native']['replacement'],'fused')
        self.assertEqual(audit['native']['relative_to_selected'],2.)
        self.assertEqual(audit['fused']['status'],'selected')
        self.assertNotIn('incorrect',audit)

    def test_invalid_fresh_timing_cannot_publish_a_winner(self):
        for bad in (float('nan'), float('inf'), 0, -1):
            record = {'route': 'stale', 'mean_seconds': {'stale': .001},
                      'interface_audit': {'stale': {'status': 'selected'}}}
            with self.assertRaisesRegex(RuntimeError, 'Invalid complete-model timing'):
                self.select([.01, .012, .01, .01, bad], [.005]*8, record=record)
            self.assertNotIn('route', record)
            self.assertNotIn('mean_seconds', record)
            self.assertNotIn('interface_audit', record)
            self.assertEqual(record['state'], 'invalid_timing')

    def test_fresh_execution_failure_is_not_swallowed(self):
        record = {}
        with self.assertRaisesRegex(RuntimeError, 'device failure'):
            self.select([.01, .012, .01, .01, RuntimeError('device failure')], [.005]*8,
                        record=record)
        self.assertNotIn('route', record)

    def test_retry_limit_validation(self):
        for value in (-1, 3, True, .5):
            with self.assertRaises(ValueError):
                self.select([.01]*4, [.005]*4, max_remeasurements=value)


if __name__ == '__main__':
    unittest.main()
