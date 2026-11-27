"""Complete-span proof and rejection controls for the readonly DMA cache."""
import json
import os
from pathlib import Path
import subprocess
import unittest


class LoopReadCacheTests(unittest.TestCase):
    source = Path(__file__).with_name('dma-loop-read-cache.ll').read_text()

    def run_ir(self, source):
        p = subprocess.run([os.environ['XPU_DMA_OPT'], '--sync'], input=source,
                           text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, json.loads(p.stderr.splitlines()[-1])

    def test_complete_guarded_rows_keep_math(self):
        after, stats = self.run_ir(self.source)
        self.assertEqual(stats['loop_cached_reads'], 4)
        self.assertEqual(after.count('call void @llvm.xpu.gm2lm_v3'), 2)
        self.assertEqual(after.count('fadd float'), self.source.count('fadd float'))
        self.assertIn('i32 1024)', after)
        # All original waits remain. The new preload has its own completion.
        self.assertEqual(after.count('call void @llvm.xpu.mfence.v2'), 3)
        self.assertIn('br i1 %valid', after)

    def test_inverted_row_guard_is_preserved(self):
        source = self.source.replace('label %read, label %join', 'label %join, label %read')
        after, stats = self.run_ir(source)
        self.assertEqual(stats['loop_cached_reads'], 4)
        self.assertIn('xor i1 %valid, true', after)

    def test_unsafe_or_incomplete_spans_are_rejected(self):
        changes = [
            self.source.replace('i64 32', 'i64 16'),  # overlap and hole
            self.source.replace('i64 %i, 64', 'i64 %i, 128'),  # gaps
            self.source.replace('entry:\n', 'entry:\n  %big = alloca [7168 x i8], align 64\n'),
            self.source.replace('  %v = load', '  store float 0.0, ptr addrspace(1) %a\n  %v = load'),
            self.source.replace('  %v = load', '  call void @unknown()\n  %v = load')+'\ndeclare void @unknown()\n',
            self.source.replace('load float', 'load volatile float'),
            self.source.replace('  br i1 %valid, label %read',
                                '  %partial = icmp ult i64 %i, 8\n  br i1 %partial, label %read'),
            self.source.replace('i32 0, i32 32)', 'i32 1, i32 32)'),
        ]
        for source in changes:
            with self.subTest(source=source):
                _, stats = self.run_ir(source)
                self.assertEqual(stats['loop_cached_reads'], 0)


if __name__ == '__main__':
    unittest.main()
