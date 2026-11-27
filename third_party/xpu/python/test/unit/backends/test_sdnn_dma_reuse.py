"""Dependency and synchronization regressions for the standalone LLVM pass."""
import json
import os
from pathlib import Path
import re
import subprocess

import unittest

HERE = Path(__file__).parent
SOURCE = (HERE / 'fixtures/sdnn-full-k.ll').read_text()


def transform(source, bounds=True, weights_only=False):
    tool = Path(os.environ.get('XPU_DMA_OPT', HERE.parents[3] / 'backend/bin/xpu-dma-opt'))
    if not tool.exists():
        raise unittest.SkipTest('build xpu-dma-opt first')
    args = [str(tool)]
    if bounds:
        args.append('--sdnn-bounds=23633920,4194304,23633920,4096,23633920')
    if weights_only:
        args.append('--sdnn-weights-only')
    result = subprocess.run(args, input=source, text=True, capture_output=True, check=True)
    return result.stdout, json.loads(result.stderr)


def test_same_weight_keeps_original_tokens_and_arithmetic():
    out, stats = transform(SOURCE)
    assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 2
    for op in ('xwait', 'xsignal', 'mm', 'dsmadd'):
        pattern = r'call void @llvm.xpu.' + op + r'\([^\n]*?\)'
        assert re.findall(pattern, re.sub(r'%\d+', '%ssa', out)) == re.findall(pattern, re.sub(r'%\d+', '%ssa', SOURCE))
    assert 'sdnn.reuse.valid' in out
    assert 'icmp ule i64' in out  # overlap fallback, including shifted views


def test_unproven_layout_or_write_keeps_original_code():
    changes = [
        ('dma_cfg(i32 4096, i32 4096, i32 %71, i32 4)', 'dma_cfg(i32 4096, i32 8192, i32 %71, i32 4)'),
        ('dma_cfg(i32 1048576, i32 %99', 'dma_cfg(i32 524288, i32 %99'),
        ('ptr addrspace(1) %4, i64 %79', 'ptr addrspace(1) %1, i64 %79'),
        ('i32 1024, i32 1)', 'i32 1024, i32 7)'),
    ]
    for old,new in changes:
        source = SOURCE.replace(old, new)
        assert source != SOURCE
        out, stats = transform(source, weights_only=True)
        assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 0
        assert out == source


def test_rotating_weight_destination_is_not_reused():
    source = SOURCE
    marker = '  tail call void @llvm.xpu.shuffle_batch.v2(ptr null, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 524288'
    replacement = '  %rotating_l1w = inttoptr i32 %39 to ptr\n' + marker.replace('v2(ptr null,', 'v2(ptr %rotating_l1w,')
    source = source.replace(marker, replacement)
    assert source != SOURCE
    out, stats = transform(source, weights_only=True)
    assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 0
    assert out == source


def test_input_cache_is_independent_of_weight_qualification():
    source = SOURCE.replace('dma_cfg(i32 4096, i32 4096, i32 %71, i32 4)',
                            'dma_cfg(i32 4096, i32 8192, i32 %71, i32 4)')
    assert source != SOURCE
    _, stats = transform(source)
    assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 1


def test_two_slot_pipeline_keeps_per_slot_validity_and_original_tokens():
    source = (HERE / 'fixtures/sdnn-full-k-two-slot.ll').read_text()
    out, stats = transform(source)
    assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 2
    for op in ('xwait', 'xsignal', 'mm', 'dsmadd'):
        pattern = r'call void @llvm.xpu.' + op + r'\([^\n]*?\)'
        assert re.findall(pattern, re.sub(r'%\d+', '%ssa', out)) == re.findall(pattern, re.sub(r'%\d+', '%ssa', source))
    assert 'sdnn.reuse.valid' in out
    # Bounded output addresses must remain beyond both cached UB packets.
    unsafe = source.replace('or disjoint i32 %124, 819712', 'or disjoint i32 %124, 0')
    assert unsafe != source
    result, stats = transform(unsafe)
    assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 0
    assert result == unsafe
    # An output write into the input argument must disable its cache. The
    # independent read-only weight packet can still be qualified.
    source = SOURCE.replace('ptr addrspace(1) %4, i64 %79', 'ptr addrspace(1) %0, i64 %79')
    assert source != SOURCE
    _, stats = transform(source)
    assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 1


def test_unknown_side_effect_and_missing_bounds_are_not_optimized():
    source = SOURCE.replace('  tail call void @llvm.xpu.xsignal_mode_cfg(i32 1)',
                            '  call void @unknown()\n  tail call void @llvm.xpu.xsignal_mode_cfg(i32 1)')
    source += '\ndeclare void @unknown()\n'
    for text, bounds in ((source, True), (SOURCE, False)):
        out, stats = transform(text, bounds)
        assert stats['sdnn_cached_dma'] == stats['sdnn_cached_ds'] == 0
        assert out == text


if __name__ == '__main__':
    test_same_weight_keeps_original_tokens_and_arithmetic()
    test_unproven_layout_or_write_keeps_original_code()
    test_unknown_side_effect_and_missing_bounds_are_not_optimized()
    test_rotating_weight_destination_is_not_reused()
    test_input_cache_is_independent_of_weight_qualification()
    test_two_slot_pipeline_keeps_per_slot_validity_and_original_tokens()
    print('12 SDNN reuse checks passed')
