"""CPU contracts for native SDNN RNE partitioning and event handoff.

The reduced native fixture is test data, never a fusion implementation.
Actual device ordering and rounding require the separate P800 bit-level gate.
"""
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

_SOURCE = (Path(__file__).resolve().parents[3] / "triton" / "experimental" / "graph_fusion" / "_p800_parallel_rne.py")
_SPEC = importlib.util.spec_from_file_location("_p800_rne_contracts", _SOURCE)
rne = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rne)
_FIXTURE = Path(__file__).with_name("fixtures") / "p800_rne.mlir"


class ParallelRneSafetyTests(unittest.TestCase):

    def setUp(self):
        self.source = _FIXTURE.read_text()

    def test_each_element_has_one_writer(self):
        _, records = rne.rewrite(self.source)
        self.assertEqual(len(records), 2)
        for record in records:
            rows = [row for _, lo, hi in record['partitions'] for row in range(lo, hi)]
            self.assertEqual(rows, list(range(record['rows'])))
            self.assertEqual([core for core, _, _ in record['partitions']], list(range(8)))

    def test_ready_requires_event_completion_before_cross_core_barrier(self):
        changed, records = rne.rewrite(self.source)
        for i, record in enumerate(records):
            begin = changed.index(f'%f06_rne_sync_mask_{i} =')
            barrier = f'llvm.call @llvm.xpu.sync_core(%f06_rne_sync_mask_{i})'
            ready = changed.index(barrier, begin)
            done = changed.index(barrier, ready + len(barrier))
            handoff = changed[begin:ready]
            waits = [int(v) for v in re.findall(r'sdnn.core_wait\(3, (\d+)\)', handoff)]
            self.assertEqual(waits, record['original_waits'])
            self.assertLess(handoff.rindex('sdnn.core_wait'), handoff.index('llvm.call @llvm.xpu.xfence'))
            self.assertLess(handoff.index('llvm.call @llvm.xpu.xfence'), handoff.index('llvm.call @llvm.xpu.mfence.v2'))
            self.assertIn(f'%f06_rne_mem_mask_{i} = arith.constant 7', handoff)
            self.assertEqual(changed[ready:done].count('attributes {little_core, core_id ='), 8)
            for event in record['original_signals']:
                self.assertIn(f'sdnn.core_arrive(3, {event})', changed[done:done + 300])

    def test_arithmetic_and_special_value_body_are_unchanged(self):
        changed, _ = rne.rewrite(self.source)
        pattern = r'^      sdnn\.core_specialize ins\([^\n]+\n(.*?)^      \}'
        original = [b for b in re.findall(pattern, self.source, re.M | re.S) if 'arith.shrui' in b]
        workers = [b for b in re.findall(pattern, changed, re.M | re.S) if 'arith.shrui' in b]
        self.assertEqual(len(workers), 8 * len(original))
        for i, body in enumerate(original):
            outer = re.search(r'scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+) \{', body)
            for worker in workers[8 * i:8 * i + 8]:
                restored = re.sub(
                    r'%f06_row_begin = arith.constant \d+ : index\n        %f06_row_end = arith.constant \d+ : index\n        ',
                    '', worker)
                restored = restored.replace(f'scf.for {outer[1]} = %f06_row_begin to %f06_row_end step {outer[4]} {{',
                                            outer[0])
                self.assertEqual(restored, body)

    def test_overlapping_rows_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'contiguous non-overlapping'):
            rne.rewrite(self.source.replace('d0 * 32 + d1', 'd0 * 0 + d1'))

    def test_conditional_execution_keeps_original_lowering(self):
        for conditional in ('scf.if ', 'cf.cond_br '):
            source = conditional + self.source
            self.assertEqual(rne.rewrite(source), (source, []))

    def test_hidden_effects_are_rejected(self):
        source = self.source.replace('sdnn.core_specialize_return',
                                     'func.call @unexpected() : () -> ()\n        sdnn.core_specialize_return', 1)
        with self.assertRaises(AssertionError):
            rne.rewrite(source)

    def test_non_eight_core_launch_does_not_enter_native_transform(self):
        for cores in (None, -1, 8):
            self.assertTrue(rne.supported_launch(SimpleNamespace(launch_cores=cores)))
        sentinel = object()
        for cores in (0, 1, 2, 4, 16, 64):
            options = SimpleNamespace(launch_cores=cores)
            self.assertFalse(rne.supported_launch(options))
            self.assertEqual(rne.lower(sentinel, '.', options), (sentinel, []))


if __name__ == '__main__':
    unittest.main()
