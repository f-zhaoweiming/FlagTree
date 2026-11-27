"""Proof-boundary tests using the LLVM emitted by the ordered row template."""
import json
import os
from pathlib import Path
import subprocess
import unittest


class OrderedRowTests(unittest.TestCase):
    source = Path(__file__).with_name('dma-ordered-row.ll').read_text()

    def run_ir(self, source, enabled=True):
        command = [os.environ['XPU_DMA_OPT'], '--pipeline']
        if enabled:
            command.append('--ordered-row')
        result = subprocess.run(command, input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout, json.loads(result.stderr.splitlines()[-1])

    def test_complete_two_pass_residency(self):
        after, stats = self.run_ir(self.source)
        self.assertEqual(stats['ordered_resident_rows'], 1)
        self.assertEqual(stats['skipped_empty_rows'], 1)
        self.assertEqual(after.count('call void @llvm.xpu.gm2lm_v3'), 3)
        self.assertEqual(stats['pipelined_loops'], 1)
        self.assertIn('row.pingpong', after)
        self.assertIn('row.next.ready', after)
        self.assertEqual(after.count('call void @llvm.xpu.lm2gm_v3'), 2)
        self.assertEqual(after.count('call void @llvm.xpu.gm2sm_v3'), 2)
        self.assertEqual(after.count('call void @llvm.xpu.csr_set_sync_group'), 2)
        # No changed arithmetic instruction/order; only dataflow replaces the
        # redundant affine add/round. Later DCE removes those dead operations.
        for instruction in ('fadd <16 x float>', 'fmul <16 x float>', 'vmac.f.mz.rn'):
            self.assertEqual(after.count(instruction), self.source.count(instruction))
        self.assertEqual(stats['ordered_compact_rounds'], 1)
        self.assertEqual(after.count('vfloat2fp16_h.rn'), self.source.count('vfloat2fp16_h.rn'))
        self.assertNotIn('extractelement <32 x half>', after)
        self.assertIn('addrspace(2)', after)

    def test_collective_between_phases_prevents_row_skip(self):
        source = self.source.replace('  %246 = fdiv',
            '  call void @collective()\n  %246 = fdiv') + '\ndeclare void @collective()\n'
        _, stats = self.run_ir(source)
        self.assertEqual(stats['ordered_resident_rows'], 1)
        self.assertEqual(stats['skipped_empty_rows'], 0)
        self.assertEqual(stats['pipelined_loops'], 0)

    def test_prefetch_declines_when_extra_buffer_exceeds_budget(self):
        source=self.source.replace('  %9 = tail call',
            '  %extra = alloca [1024 x i8], align 64\n  %9 = tail call')
        after,stats=self.run_ir(source)
        self.assertEqual(stats['ordered_resident_rows'], 1)
        self.assertEqual(stats['pipelined_loops'], 0)
        self.assertNotIn('row.pingpong',after)

    def test_sync_only_mode_does_not_prefetch(self):
        result=subprocess.run([os.environ['XPU_DMA_OPT'],'--ordered-row'],
            input=self.source,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        stats=json.loads(result.stderr.splitlines()[-1])
        self.assertEqual(stats['ordered_resident_rows'],1)
        self.assertEqual(stats['pipelined_loops'],0)

    def test_disabled_retains_generic_scheduler(self):
        after, stats = self.run_ir(self.source, False)
        self.assertEqual(stats['ordered_resident_rows'], 0)
        self.assertEqual(stats['loop_cached_reads'], 8)
        self.assertNotIn('call void @llvm.xpu.gm2sm_v3', after)

    def test_contract_and_layout_changes_rejected(self):
        changes = [
            self.source.replace('vfloat2fp16_h.rn', 'vfloat2fp16_h.rz'),
            self.source.replace('i32 0, i32 64)', 'i32 0, i32 32)'),
            self.source.replace('i32 0, i32 32)', 'i32 1, i32 32)'),
            self.source.replace('store <32 x half>', 'store volatile <32 x half>'),
            self.source.replace('  %9 = tail call', '  %extra = alloca [7168 x i8], align 64\n  %9 = tail call'),
            self.source.replace('target triple = "xpu3-baidu-none-gnu"',
                                'target triple = "xpu3-baidu-none-gnu"\n@shared = internal addrspace(2) global [64 x i8] undef'),
            self.source.replace('  call void @llvm.xpu.lm2gm_v3',
                                '  call void @unknown()\n  call void @llvm.xpu.lm2gm_v3')+'\ndeclare void @unknown()\n',
        ]
        for source in changes:
            self.assertNotEqual(source, self.source)
            with self.subTest(change=changes.index(source)):
                _, stats = self.run_ir(source)
                self.assertEqual(stats['ordered_resident_rows'], 0)


if __name__ == '__main__':
    unittest.main()
