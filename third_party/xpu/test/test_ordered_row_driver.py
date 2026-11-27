"""Keep unproved row layouts compilable without bypassing resource guards."""
import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    'dma_driver', Path(__file__).parents[1] / 'backend/dma_pipeline.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DriverTests(unittest.TestCase):
    def test_unproved_layout_retries_original_ir(self):
        self.check(0, 2)

    def test_proved_layout_does_not_retry(self):
        self.check(1, 1)

    def check(self, hits, calls):
        ir = 'llvm.xpu.gm2lm_v3 llvm.xpu.lm2gm_v3 original'
        metadata = dict(arch=3, core_num=64, enable_fp_fusion=False,
                        xpu_ordered_row_reuse=True, hash='test', name='row')
        cache = Mock()
        module = types.ModuleType('triton.runtime.cache')
        module.get_cache_manager = lambda _: cache
        results = [types.SimpleNamespace(returncode=0, stdout='resident',
                   stderr=json.dumps(dict(ordered_resident_rows=hits))),
                   types.SimpleNamespace(returncode=0, stdout='generic',
                   stderr=json.dumps(dict(ordered_resident_rows=0)))]
        with patch.object(driver.subprocess, 'run', side_effect=results) as run, \
                patch.object(driver, 'tool_path', return_value=Path(__file__)), \
                patch.object(driver, 'codegen_key', return_value='test'), \
                patch.dict('sys.modules', {'triton.runtime.cache': module}):
            result = driver.optimize_ir(ir, metadata, 2)
        self.assertEqual(run.call_count, calls)
        self.assertTrue(metadata['dma_pipeline']['ordered_row_requested'])
        if not hits:
            self.assertEqual(result, 'generic')
            self.assertEqual(run.call_args_list[1].kwargs['input'], ir)
            self.assertNotIn('--ordered-row', run.call_args_list[1].args[0])
            self.assertEqual(metadata['dma_pipeline']['ordered_row_fallback'], 'unproved_layout')
        else:
            self.assertEqual(result, 'resident')
            self.assertNotIn('ordered_row_fallback', metadata['dma_pipeline'])

    def test_invalid_resource_is_not_a_layout_fallback(self):
        with patch.object(driver.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                driver.optimize_ir('llvm.xpu.gm2lm_v3 llvm.xpu.lm2gm_v3',
                    dict(arch=3, core_num=16, xpu_ordered_row_reuse=True), 2)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
