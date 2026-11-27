"""Run the structural KL3 DMA pass before the LLVM 19 code generator."""
import functools
import hashlib
import json
from pathlib import Path
import subprocess


def tool_path():
    return Path(__file__).with_name('bin') / 'xpu-dma-opt'


@functools.lru_cache(None)
def codegen_key():
    path = tool_path()
    return 'dma-v1:' + (hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'unavailable')


def optimize_ir(ir, metadata, mode):
    if mode not in (0, 1, 2):
        raise ValueError('xpu_dma_pipeline must be 0 (off), 1 (sync), or 2 (prefetch)')
    if not mode or metadata.get('arch') != 3:
        return ir
    if metadata.get('is_sdnn'):
        bounds = metadata.get('xpu_sdnn_tensor_bytes', ())
        if not bounds or not all(type(n) is int and 0 < n <= 1 << 40 for n in bounds):
            return ir
        args = ['--sdnn-bounds=' + ','.join(map(str, bounds))]
    else:
        if 'llvm.xpu.gm2lm_v3' not in ir or 'llvm.xpu.lm2gm_v3' not in ir:
            return ir
        args = ['--pipeline' if mode == 2 else '--sync']
        if metadata.get('xpu_ordered_row_reuse'):
            if metadata.get('core_num') != 64 or metadata.get('enable_fp_fusion'):
                raise ValueError('Ordered row reuse requires 64 cores and explicit contraction')
            args.append('--ordered-row')
    path = tool_path()
    if not path.is_file():
        raise RuntimeError('XPU DMA pass is enabled but xpu-dma-opt is not packaged')
    result = subprocess.run([str(path), *args],
                            input=ir, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('XPU DMA scheduling failed: ' + result.stderr[-4000:])
    record = json.loads(result.stderr.strip().splitlines()[-1])
    if metadata.get('xpu_ordered_row_reuse'):
        # Small/constant rows can simplify into a different loop layout. A
        # failed residency proof must retain the original scheduling path.
        if record.get('ordered_resident_rows') != 1:
            result = subprocess.run([str(path), *(a for a in args if a != '--ordered-row')],
                                    input=ir, text=True, capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError('XPU DMA scheduling failed: ' + result.stderr[-4000:])
            record = json.loads(result.stderr.strip().splitlines()[-1])
            record['ordered_row_fallback'] = 'unproved_layout'
        record['ordered_row_requested'] = True
    metadata['dma_pipeline'] = dict(record, mode=mode, tool=codegen_key())
    from triton.runtime.cache import get_cache_manager
    cache = get_cache_manager(metadata['hash'])
    cache.put(result.stdout, metadata['name'] + '.dma.llir')
    cache.put(json.dumps(metadata['dma_pipeline']), metadata['name'] + '.dma.json')
    return result.stdout
