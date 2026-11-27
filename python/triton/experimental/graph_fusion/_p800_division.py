"""Legalize full tensor SDNN division into per-row SDNN ops.

Native division loads one 32-value denominator coefficient vector and applies
it to every row. Splitting the tensor rows retains native instruction lowering
and synchronization while making each coefficient vector belong to one row.
This is a generic scalar operation repair, with no GEMM or fusion template.
"""
import os
import re
from pathlib import Path


def split_division_rows(mod, output_dir):
    from triton._C.libtriton import ir
    # This pinned native build returns an empty string from module.__str__.
    # Use its pass reproducer to serialize the real module under the compile lock.
    previous = os.environ.get('TRITON_REPRODUCER_PATH')
    prefix = str(Path(output_dir) / 'division-input')
    os.environ['TRITON_REPRODUCER_PATH'] = prefix
    try:
        pm = ir.pass_manager(mod.context)
        pm.run(mod, 'before_division_repair')
        source = Path(prefix + '.before_division_repair.repro.mlir').read_text()
    finally:
        if previous is None:
            os.environ.pop('TRITON_REPRODUCER_PATH', None)
        else:
            os.environ['TRITON_REPRODUCER_PATH'] = previous
    assert source.strip(), 'cannot inspect native SDNN module'
    records = []
    pattern = re.compile(
        r'^(?P<indent>\s*)sdnn\.ew div (?P<dst>%[\w]+) src0 (?P<a>%[\w]+) src1 (?P<b>%[\w]+) '
        r'\{(?P<attrs>[^\n]+)\} : \((?P<type>memref<(?P<m>\d+)x(?P<n>\d+)xf32, #[\w]+>), '
        r'(?P=type), (?P=type)\) -> \(\)$', re.M)

    def rewrite(match):
        m, n = int(match['m']), int(match['n'])
        if m == 1:
            return match[0]
        attrs = match['attrs']
        ordinary = re.sub(r'(?:signal|wait)(?:_cached)? = array<i32: [^>]+>,? ?', '', attrs)
        ordinary = ordinary.rstrip(', ')
        signals = re.findall(r'(?:signal|signal_cached) = array<i32: [^>]+>', attrs)
        waits = re.findall(r'(?:wait|wait_cached) = array<i32: [^>]+>', attrs)
        prefix = f'%f06_div_{len(records)}_'
        lines = []
        for row in range(m):
            ty = f'memref<1x{n}xf32, strided<[{n}, 1], offset: {row*n}>>'
            views = []
            by_operand = {}
            for role in ('dst', 'a', 'b'):
                operand = match[role]
                if operand not in by_operand:
                    name = f'{prefix}{row}_{role}'
                    lines.append(
                        f'{name} = memref.subview {operand}[{row}, 0] [1, {n}] [1, 1] : {match["type"]} to {ty}')
                    by_operand[operand] = name
                views.append(by_operand[operand])
            row_attrs = ', '.join([ordinary] + (waits if row == 0 else []) + (signals if row == m - 1 else []))
            lines.append(
                f'sdnn.ew div {views[0]} src0 {views[1]} src1 {views[2]} {{{row_attrs}}} : ({ty}, {ty}, {ty}) -> ()')
        records.append({'rows': m, 'columns': n, 'dtype': 'f32', 'original': match[0].strip(), 'operations_after': m})
        return '\n'.join(match['indent'] + line for line in lines)

    changed = pattern.sub(rewrite, source)
    if not records:
        return mod, records
    path = Path(output_dir) / 'sdnn-div-row-repair.mlir'
    path.write_text(changed)
    result = ir.parse_mlir_module(str(path), mod.context)
    result.context = mod.context
    assert result.verify()
    return result, records
