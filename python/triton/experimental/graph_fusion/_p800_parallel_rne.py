"""Partition native SDNN pointwise RNE across its eight scalar control cores.

The arithmetic body, buffers, GEMM, and activation remain compiler generated.
Native sync_core(255) barriers bracket the disjoint writes. Existing event
waits/signals and per-core memory fences are preserved. Native engine and
memory completion precede broadcasting input readiness. No new event slots.
"""
import re
from pathlib import Path


def supported_launch(options):
    # The native SDNN default is eight control cores; num_warps/core_num are
    # unrelated compiler options and must not be used as the physical count.
    return getattr(options, 'launch_cores', -1) in (None, -1, 8)


def rewrite(source):
    if 'scf.if ' in source or 'cf.cond_br ' in source or '@llvm.xpu.sync_core' in source:
        return source, []  # Cannot prove every core reaches the barriers.
    records = []
    pattern = re.compile(r'^      sdnn\.core_specialize ins\([^\n]+\n.*?^      \}', re.M | re.S)

    def change(match):
        region = match[0]
        header, body = region.split('\n', 1)
        is_rne = 'arith.shrui' in body
        is_bounds = 'arith.maximumf' in body and 'arith.minimumf' in body
        is_multiply = 'c2143289344_i32' in header and 'c2139095040_i32' in header and 'arith.cmpi' in body and 'arith.select' in body
        is_integer_bounds = 'c1098907648_i32' in header and 'arith.cmpi' in body and 'arith.select' in body
        is_nan_restore = not is_rne and 'c4194304_i32' in header and 'c2139095040_i32' in header and 'arith.select' in body
        is_bitcast_copy = ('xf32,' in header and 'xi32,' in header and 'arith.bitcast' in body
                           and 'arith.cmpi' not in body and 'arith.select' not in body and not is_rne)
        is_guard = is_multiply or is_integer_bounds or is_nan_restore or is_bitcast_copy
        if 'attributes {little_core,' not in header or not (is_rne or is_bounds or is_guard):
            return region
        # Fail closed on new scheduling or dataflow forms. This pass only
        # distributes the existing, independent, rectangular RNE loop nest.
        assert '_cached' not in header and 'core_id' not in header
        loops = list(re.finditer(r'scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+) \{', body))
        if len(loops) != 2:
            return region  # Keep native serial arithmetic for a different rank.
        constants = dict(re.findall(r'(%\w+) = arith.constant (\d+) : index', body))
        outer, inner = loops
        row, lo, hi, step = outer.groups()
        col, clo, chi, cstep = inner.groups()
        assert constants[lo] == constants[clo] == '0'
        assert constants[step] == constants[cstep] == '1'
        rows, cols = int(constants[hi]), int(constants[chi])
        if rows < 8 or rows % 8 != 0:
            return region  # No disjoint eight-way row partition is available.
        if 'sdnn.layout = "CR_RowMajor"' not in header:
            return region  # Preserve native lowering for other memory layouts.
        buffers = re.findall(r'memref<(\d+)x(\d+)x(f32|i32), (#[\w]+)>', header)
        if len(buffers) != header.count('memref<'):
            return region  # Precision boundary loops may also carry narrow buffers.
        if is_guard:
            assert len(buffers) >= 2
            assert cols % 16 == 0, 'guard row must span complete 64-byte cache lines'
        else:
            if len(buffers) != (2 if is_rne else 4) or any(dtype != 'f32' for _, _, dtype, _ in buffers):
                return region  # Separate integer stages are not this FP32-to-FP32 loop.
        for m, n, dtype, layout in buffers:
            assert (int(m), int(n)) == (rows, cols)
            definition = f'{layout} = affine_map<(d0, d1) -> (d0 * {cols} + d1)>'
            assert definition in source, 'parallel RNE requires contiguous non-overlapping rows'
        accesses = re.findall(r'memref\.(?:load|store)[^\n]*\[([^]]+)\]', body)
        assert accesses == [f'{row}, {col}'] * len(buffers)
        # No hidden side effects, cross-element values, nested regions, or calls.
        operations = re.findall(r'^\s+(?:%\w+ = )?([a-z_]+\.[a-z_]+)', body, re.M)
        allowed = {
            'arith.constant', 'scf.for', 'memref.load', 'memref.store', 'arith.bitcast', 'arith.andi', 'arith.ori',
            'arith.cmpi', 'arith.select', 'arith.addi', 'arith.shrui', 'sdnn.core_specialize_return'
        }
        if is_guard:
            assert set(operations) <= allowed, operations
            assert operations.count('memref.store') == 1
            assert operations.count('memref.load') == len(buffers) - 1
            if is_bitcast_copy:
                assert len(buffers) == 2 and {b[2] for b in buffers} == {'f32', 'i32'}
                if not set(operations) <= {
                    'arith.constant', 'scf.for', 'memref.load', 'memref.store', 'arith.bitcast',
                    'sdnn.core_specialize_return'
                }:
                    return region  # A pure fused epilogue, not a standalone bitcast copy.
                assert operations.count('arith.bitcast') == 1
            else:
                assert operations.count('arith.select') >= 1
        elif is_rne:
            assert set(operations) <= allowed, operations
            assert operations.count('memref.load') == operations.count('memref.store') == 1
            assert operations.count('arith.select') == 2 and operations.count('arith.shrui') == 1
            assert 'c32767_i32' in header and 'c-65536_i32' in header and 'c2139095040_i32' in header
        else:
            allowed_bounds = {
                'arith.constant', 'scf.for', 'memref.load', 'memref.store', 'arith.maximumf', 'arith.minimumf',
                'sdnn.core_specialize_return'
            }
            assert set(operations) <= allowed_bounds, operations
            assert operations.count('memref.load') == 3 and operations.count('memref.store') == 1
            assert operations.count('arith.maximumf') == operations.count('arith.minimumf') == 1
        waits = [int(v.strip()) for v in re.search(r'wait = array<i32: ([^>]+)>', header)[1].split(',')]
        signals = [int(v.strip()) for v in re.search(r'signal = array<i32: ([^>]+)>', header)[1].split(',')]
        index = len(records)
        mask = f'%f06_rne_sync_mask_{index}'
        barrier = f'      llvm.call @llvm.xpu.sync_core({mask}) : (i32) -> ()'
        lines = [f'      {mask} = arith.constant 255 : i32']
        lines += [f'      sdnn.core_wait(3, {event})' for event in waits]
        # Event waits were moved out of the original worker. Complete the
        # producer operations before other cores observe the ready barrier;
        # a barrier alone permits stale first-cache-line reads across tiles.
        lines += [
            f'      %f06_rne_mem_mask_{index} = arith.constant 7 : i32',
            '      llvm.call @llvm.xpu.xfence() : () -> ()',
            f'      llvm.call @llvm.xpu.mfence.v2(%f06_rne_mem_mask_{index}) : (i32) -> ()'
        ]
        lines.append(barrier)
        clean = re.sub(r'(?:signal|wait) = array<i32: [^>]+>,? ?', '', header).replace(', }', '}')
        partitions = []
        for core in range(8):
            start, end = rows * core // 8, rows * (core + 1) // 8
            partitions.append([core, start, end])
            attrs = f'little_core, core_id = {core} : i32,'
            worker_header = clean.replace('little_core,', attrs)
            worker_body = body[:outer.start(
            )] + f'%f06_row_begin = arith.constant {start} : index\n        %f06_row_end = arith.constant {end} : index\n        ' + body[
                outer.start():]
            worker_body = worker_body.replace(outer[0],
                                              f'scf.for {row} = %f06_row_begin to %f06_row_end step {step} {{', 1)
            lines.append(worker_header + '\n' + worker_body)
        lines.append(barrier)
        lines += [f'      sdnn.core_arrive(3, {event})' for event in signals]
        kind = ('rne_ieee_multiply' if is_rne else 'ieee_multiply') if is_multiply else (
            'rne' if is_rne else 'nan_preserving_bounds')
        if is_integer_bounds: kind = 'integer_tanh_bounds'
        if is_nan_restore: kind = 'nan_restore'
        if is_bitcast_copy: kind = 'bitcast_copy'
        records.append({
            'kind': kind, 'buffer_types': [b[2] for b in buffers], 'rows': rows, 'columns': cols, 'partitions':
            partitions, 'leader': 3, 'synchronization': 'llvm.xpu.sync_core', 'mask': 255, 'original_waits': waits,
            'original_signals': signals, 'arithmetic': 'unchanged', 'input_handoff_fence': 'xfence+mfence(7)'
        })
        return '\n'.join(lines)

    changed = pattern.sub(change, source)
    if records:
        assert '@llvm.xpu.sync_core' not in source
        changed = changed.replace(
            'module {',
            'module {\n  llvm.func @llvm.xpu.sync_core(i32) attributes {passthrough = ["convergent", "noduplicate"]}\n  llvm.func @llvm.xpu.xfence()\n  llvm.func @llvm.xpu.mfence.v2(i32)',
            1)
    return changed, records


def lower(mod, directory, options):
    if not supported_launch(options):
        return mod, []  # Preserve original lowering for a non-eight-core launch.
    from ._p800_native_activation import serialize, parse
    source = serialize(mod, directory, 'before_parallel_rne')
    changed, records = rewrite(source)
    if not records:
        return mod, records
    return parse(mod, changed, Path(directory) / 'parallel-rne.mlir'), records


def rewrite_single_multiply(source):
    """Partition a native one-row multiply repair into disjoint cache lines.

    Explicit experimental caller only. Keep all producer/consumer event edges
    and the scalar arithmetic unchanged. Each core owns complete 64-byte lines;
    a readiness fence and two collective barriers hand off the SRAM buffers.
    Unknown layouts or side effects retain the original native implementation.
    """
    if any(s in source for s in ('scf.if ', 'cf.cond_br ', '@llvm.xpu.sync_core')):
        return source, []
    records = []
    pattern = re.compile(r'^      sdnn\.core_specialize ins\([^\n]+\n.*?^      \}', re.M | re.S)

    def change(match):
        header, body = match[0].split('\n', 1)
        if not all(s in header for s in ('little_core,', 'c2143289344_i32', 'c2139095040_i32')):
            return match[0]
        if '_cached' in header or 'core_id' in header:
            return match[0]
        buffers = re.findall(r'memref<1x(128|256)x(f32|i32), (#[\w]+)>', header)
        if not buffers or len(buffers) != header.count('memref<'):
            return match[0]
        cols = int(buffers[0][0])
        for n, _, layout in buffers:
            if int(n) != cols or f'{layout} = affine_map<(d0, d1) -> (d0 * {cols} + d1)>' not in source:
                return match[0]
        loops = list(re.finditer(r'scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+) \{', body))
        if len(loops) != 1:
            return match[0]
        loop, = loops
        col, lo, hi, step = loop.groups()
        constants = dict(re.findall(r'(%\w+) = arith.constant (\d+) : index', body))
        if (constants.get(lo), constants.get(hi), constants.get(step)) != ('0', str(cols), '1'):
            return match[0]
        accesses = re.findall(r'memref\.(?:load|store)[^\n]*\[([^]]+)\]', body)
        if accesses != [f'{lo}, {col}'] * len(buffers):
            return match[0]
        operations = re.findall(r'^\s+(?:%\w+ = )?([a-z_]+\.[a-z_]+)', body, re.M)
        allowed = {'arith.constant', 'scf.for', 'memref.load', 'memref.store', 'arith.bitcast',
                   'arith.andi', 'arith.ori', 'arith.xori', 'arith.cmpi', 'arith.select',
                   'arith.addi', 'arith.shrui', 'sdnn.core_specialize_return'}
        if not set(operations) <= allowed or operations.count('memref.store') != 1:
            return match[0]
        if operations.count('memref.load') != len(buffers) - 1:
            return match[0]
        waits = re.search(r'wait = array<i32: ([^>]+)>', header)
        signals = re.search(r'signal = array<i32: ([^>]+)>', header)
        if waits is None or signals is None:
            return match[0]
        waits, signals = ([int(v.strip()) for v in m[1].split(',')] for m in (waits, signals))
        index = len(records)
        mask = f'%f06_mul_sync_{index}'
        barrier = f'      llvm.call @llvm.xpu.sync_core({mask}) : (i32) -> ()'
        lines = [f'      {mask} = arith.constant 255 : i32']
        lines += [f'      sdnn.core_wait(3, {event})' for event in waits]
        lines += [f'      %f06_mul_fence_{index} = arith.constant 7 : i32',
                  '      llvm.call @llvm.xpu.xfence() : () -> ()',
                  f'      llvm.call @llvm.xpu.mfence.v2(%f06_mul_fence_{index}) : (i32) -> ()', barrier]
        clean = re.sub(r'(?:signal|wait) = array<i32: [^>]+>,? ?', '', header).replace(', }', '}')
        partitions = []
        for core in range(8):
            start, end = cols * core // 8, cols * (core + 1) // 8
            assert start % 16 == end % 16 == 0
            partitions.append([core, start, end])
            worker = body[:loop.start()] + (
                f'%f06_col_begin = arith.constant {start} : index\n'
                f'        %f06_col_end = arith.constant {end} : index\n        ') + body[loop.start():]
            worker = worker.replace(loop[0], f'scf.for {col} = %f06_col_begin to %f06_col_end step {step} {{', 1)
            lines.append(clean.replace('little_core,', f'little_core, core_id = {core} : i32,') + '\n' + worker)
        lines.append(barrier)
        lines += [f'      sdnn.core_arrive(3, {event})' for event in signals]
        records.append(dict(kind='single_multiply_columns', columns=cols, partitions=partitions,
                            original_waits=waits, original_signals=signals, arithmetic='unchanged',
                            barriers=2, input_handoff_fence='xfence+mfence(7)'))
        return '\n'.join(lines)

    changed = pattern.sub(change, source)
    if records:
        changed = changed.replace('module {', 'module {\n  llvm.func @llvm.xpu.sync_core(i32) attributes '
                                  '{passthrough = ["convergent", "noduplicate"]}\n'
                                  '  llvm.func @llvm.xpu.xfence()\n  llvm.func @llvm.xpu.mfence.v2(i32)', 1)
    return changed, records
