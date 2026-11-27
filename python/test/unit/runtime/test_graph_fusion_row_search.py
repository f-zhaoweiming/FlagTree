"""P800 row schedules retain default candidates, masks and bounded calibration."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import triton
from torch._inductor.runtime.hints import ReductionHint
from triton.experimental.graph_fusion._p800_model_schedule import _row_reduction_configs, _row_host_bench, _compile_buffer_option


class RowSearchTests(unittest.TestCase):
    def test_row_backend_options_are_forwarded_only_to_the_owning_jit(self):
        fn,other=object(),object()
        source=SimpleNamespace(fn=fn); foreign=SimpleNamespace(fn=other)
        calls=[]
        def compile(src,**kw):
            calls.append((src,kw));return object()
        with patch.object(triton,'compile',side_effect=compile):
            with _compile_buffer_option(fn,2048,reduction_reuse=True,reduction_tree=False,
                                         row_shared=True,reduction_half_cache=True) as forwarded:
                a=triton.compile(source,options={'debug':True})
                triton.compile(foreign,options={'debug':False})
        self.assertEqual(calls[0][1]['options'],dict(debug=True,buffer_size_limit=2048,
            xpu_reduction_reuse=True,xpu_reduction_tree=False,xpu_row_shared=True,
            xpu_reduction_half_cache=True))
        self.assertEqual(calls[1][1]['options'],{'debug':False})
        self.assertEqual(forwarded,[a])

    def test_search_retains_baseline_and_covers_missing_rows(self):
        base = [triton.Config({'XBLOCK': 1}, num_warps=8, num_stages=1)]
        for width, expected in ((256, [1,4,16,64]), (1024, [1,4,16,64]), (2048, [1,4,16])):
            configs = _row_reduction_configs({'x':8192,'r0_':width}, base,
                ReductionHint.INNER, {'x':'*fp16','y':'*fp32'}, {})
            self.assertEqual([c.kwargs['XBLOCK'] for c in configs], expected)
            self.assertIs(configs[0],base[0])
            self.assertEqual(len(base),1)
            self.assertEqual(_row_reduction_configs({'x':8192,'r0_':width},configs,
                ReductionHint.INNER,{'x':'*fp16'},{}),configs)

    def test_unsupported_layouts_and_mutation_keep_original_choices(self):
        base = [triton.Config({'XBLOCK':1})]
        for hints,hint,signature,meta in (
            ({'x':2,'r0_':1024},ReductionHint.INNER,{'a':'*fp16'},{}),
            ({'x':1024,'r0_':8192},ReductionHint.INNER,{'a':'*fp16'},{}),
            ({'x':1024,'y':2,'r0_':1024},ReductionHint.INNER,{'a':'*fp16'},{}),
            ({'x':1024,'r0_':1024},ReductionHint.OUTER,{'a':'*fp16'},{}),
            ({'x':1024,'r0_':1024},ReductionHint.INNER,{'a':'*i64'},{}),
            ({'x':1024,'r0_':1024},ReductionHint.INNER,{'a':'*fp16'},{'no_x_dim':True}),
            ({'x':1024,'r0_':1024},ReductionHint.INNER,{'a':'*fp16'},{'mutated_arg_names':['a']}),
        ):
            self.assertIs(_row_reduction_configs(hints,base,hint,signature,meta),base)

    def test_calibration_restores_arguments_and_rejects_invalid_clock(self):
        from unittest.mock import Mock
        device=Mock();device.current_device.return_value=0;device.get_raw_stream.return_value=7
        tuner=SimpleNamespace(get_device_interface=lambda:device,
            copy_args_to_cpu_if_needed=Mock(return_value=[]),
            maybe_clone_args=Mock(return_value=((42,),{})),reset_to_zero_args=Mock(),
            restore_args_from_cpu=Mock(),p800_row_timings=[])
        launch=Mock();launch.config=SimpleNamespace(kwargs={'XBLOCK':4})
        with patch('time.perf_counter_ns',side_effect=[1000000,1000000]):
            with self.assertRaisesRegex(RuntimeError,'Invalid P800'):
                _row_host_bench(tuner,launch,42)
        self.assertEqual(tuner.restore_args_from_cpu.call_count,4)
        self.assertEqual(tuner.p800_row_timings,[])
        with patch('time.perf_counter_ns',side_effect=[v for i in range(7) for v in (0,2000000)]):
            self.assertEqual(_row_host_bench(tuner,launch,42),2.0)
        self.assertEqual(tuner.p800_row_timings[0]['config'],{'XBLOCK':4})


if __name__ == '__main__':
    unittest.main()
