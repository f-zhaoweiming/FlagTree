"""Experimental native SDNN activation legalization; no GEMM or kernel template."""
import os
import re
from pathlib import Path
import json
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar

_ENABLED = ContextVar("p800_native_activation_enabled", default=False)


@contextmanager
def activation_scope():
    if _ENABLED.get():
        raise RuntimeError("nested native activation lowering is unsupported")
    token = _ENABLED.set(True)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def serialize(mod, directory, name):
    from triton._C.libtriton import ir
    saved = os.environ.get('TRITON_REPRODUCER_PATH')
    prefix = str(Path(directory) / name)
    os.environ['TRITON_REPRODUCER_PATH'] = prefix
    try:
        ir.pass_manager(mod.context).run(mod, name)
        return Path(prefix + '.' + name + '.repro.mlir').read_text()
    finally:
        if saved is None:
            os.environ.pop('TRITON_REPRODUCER_PATH', None)
        else:
            os.environ['TRITON_REPRODUCER_PATH'] = saved


def parse(mod, source, path):
    from triton._C.libtriton import ir, passes
    Path(path).write_text(source)
    result = ir.parse_mlir_module(str(path), mod.context)
    result.context = mod.context
    pm = ir.pass_manager(mod.context)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(result, 'candidate_cleanup')
    assert result.verify()
    return result


def relu(mod, directory, mode):
    text = serialize(mod, directory, 'before_relu')
    zero = set(re.findall(r'(%[\w]+) = arith.constant dense<0\.000000e\+00> : tensor<[^>]+>', text))
    compares = {
        name: (left, right, ty)
        for name, left, right, ty in re.findall(r'(%[\w]+) = arith.cmpf ogt, (%[\w]+), (%[\w]+) : (tensor<[^>]+>)',
                                                text)
    }
    records = []

    def replace(m):
        dst, cond, left, right = m['dst'], m['cond'], m['left'], m['right']
        if cond not in compares or compares[cond][:2] != (left, right) or left not in zero:
            return m[0]
        ty = compares[cond][2]
        records.append({'input': right, 'zero': left, 'type': ty})
        op = {'maximum': 'arith.maximumf', 'native_relu_placeholder': 'arith.maxnumf'}[mode]
        exact = re.search(re.escape(right) + r' = arith.extf %[\w]+ : tensor<[^>]+xbf16> to ' + re.escape(ty), text)
        annotation = ' {f06.bf16_exact = true}' if exact else ''
        return f'{dst} = {op} {right}, {left}{annotation} : {ty}'

    changed = re.sub(r'(?P<dst>%[\w]+) = arith.select (?P<cond>%[\w]+), (?P<left>%[\w]+), (?P<right>%[\w]+) : [^\n]+',
                     replace, text)
    if not records: return mod, []
    assert len(records) == 1, records
    return parse(mod, changed, Path(directory) / 'relu-candidate.mlir'), records


def exact_gelu_to_placeholder(mod, directory):
    text = serialize(mod, directory, 'before_gelu')
    # Reserve exp for the temporary opcode only when no real exp exists.
    # This fail-closed prototype does not touch tanh-GELU or arbitrary erf.
    if re.search(r'\bmath\.exp\b', text): return mod, []
    consts = {
        n: float(v)
        for n, v in re.findall(r'(%[\w]+) = arith.constant dense<([-+0-9.e]+)> : tensor<[^>]+>', text)
    }
    muls = {
        n: (a, b, t)
        for n, a, b, t in re.findall(r'(%[\w]+) = arith.mulf (%[\w]+), (%[\w]+) : (tensor<[^>]+>)', text)
    }
    adds = {
        n: (a, b, t)
        for n, a, b, t in re.findall(r'(%[\w]+) = arith.addf (%[\w]+), (%[\w]+) : (tensor<[^>]+>)', text)
    }
    erfs = re.findall(r'(%[\w]+) = tt.extern_elementwise (%[\w]+) \{[^\n]*?symbol = "(?:_ZN3xpu3erfEf|__nv_erff)"',
                      text)
    records = []

    def other_const(pair, value):
        a, b = pair[:2]
        if a in consts and abs(consts[a] - value) < 1e-8: return b
        if b in consts and abs(consts[b] - value) < 1e-8: return a
        return None

    for erf, scaled in erfs:
        if scaled not in muls: continue
        x = other_const(muls[scaled], 0.707106769)
        if x is None: continue
        halves = [n for n, pair in muls.items() if other_const(pair, 0.5) == x]
        plus = [n for n, pair in adds.items() if other_const(pair, 1.0) == erf]
        for n, (a, b, ty) in muls.items():
            if (a in halves and b in plus) or (b in halves and a in plus):
                old = f'{n} = arith.mulf {a}, {b} : {ty}'
                assert old in text
                suffix = n[1:]
                new = (f'%f06_gelu_{suffix} = math.exp {x} : {ty}\n'
                       f'    %f06_lo_{suffix} = arith.constant dense<-3.40282347e+38> : {ty}\n'
                       f'    %f06_hi_{suffix} = arith.constant dense<3.40282347e+38> : {ty}\n'
                       f'    %f06_max_{suffix} = arith.maxnumf {x}, %f06_lo_{suffix} : {ty}\n'
                       f'    %f06_clip_{suffix} = arith.minnumf %f06_max_{suffix}, %f06_hi_{suffix} : {ty}\n'
                       f'    %f06_guard_{suffix} = arith.subf {x}, %f06_clip_{suffix} : {ty}\n'
                       f'    {n} = arith.addf %f06_gelu_{suffix}, %f06_guard_{suffix} : {ty}')
                text = text.replace(old, new)
                records.append({'result': n, 'input': x, 'erf': erf, 'semantic': 'exact GELU'})
    if not records: return mod, []
    assert len(records) == 1, {'erfs': erfs, 'records': records}
    return parse(mod, text, Path(directory) / 'gelu-placeholder.mlir'), records


def restore_gelu(mod, directory):
    text = serialize(mod, directory, 'before_gelu_restore')
    lines = text.splitlines()
    changed = []
    for i, line in enumerate(lines):
        if 'sdnn.ew ' not in line or 'lut_func exp ' not in line: continue
        assert 'lut_func gelu' not in text
        # The native pass reselects interpolation size, range and overflow data.
        # Keep all storage operands and already scheduled wait/signal events.
        after = line.replace('lut_func exp ', 'lut_func gelu ')
        after = re.sub(r'(?:kOverflow|kUnderflow|lut_max|lut_min) = [-+0-9.e]+ : f32,? ?', '', after)
        lines[i] = after
        changed.append({'before': line, 'after': after})
    assert len(changed) == 1, changed
    result = parse(mod, '\n'.join(lines) + '\n', Path(directory) / 'gelu-native.mlir')
    return result, changed


def bound_tanh_inputs(mod, directory, *, triton_bitcast=False):
    text = serialize(mod, directory, 'before_tanh_guard')
    pattern = re.compile(
        r'(?P<dst>%[\w]+) = tt.extern_elementwise (?P<x>%[\w]+)(?P<attrs> \{[^\n]*symbol = "_ZN3xpu5tanhfEf"\}) : \((?P<ty>tensor<[^>]+xf32>)\) -> (?P=ty)'
    )

    def replace(m):
        dst, x, ty = m['dst'], m['x'], m['ty']
        tag = dst[1:]
        p = f'%tanh_guard_{tag}_'
        it = ty.replace('f32>', 'i32>')
        bt = ty.replace('f32>', 'i1>')
        lines = [
            f'{p}bits = arith.bitcast {x} : {ty} to {it}',
            f'{p}mask = arith.constant dense<2147483647> : {it}',
            f'{p}signmask = arith.constant dense<2147483648> : {it}',
            f'{p}limit = arith.constant dense<1098907648> : {it}',
            f'{p}inf = arith.constant dense<2139095040> : {it}',
            f'{p}quiet = arith.constant dense<4194304> : {it}',
            f'{p}fractionmask = arith.constant dense<8388607> : {it}',
            f'{p}zero = arith.constant dense<0> : {it}',
            f'{p}mag = arith.andi {p}bits, {p}mask : {it}',
            f'{p}sign = arith.andi {p}bits, {p}signmask : {it}',
            f'{p}large = arith.cmpi ugt, {p}mag, {p}limit : {it}',
            f'{p}limited = arith.ori {p}sign, {p}limit : {it}',
            f'{p}safe_bits = arith.select {p}large, {p}limited, {p}bits : {bt}, {it}',
            f'{p}safe = arith.bitcast {p}safe_bits : {it} to {ty}',
            f'{p}raw = tt.extern_elementwise {p}safe{m["attrs"]} : ({ty}) -> {ty}',
            f'{p}rawbits = arith.bitcast {p}raw : {ty} to {it}',
            f'{p}exponent = arith.andi {p}bits, {p}inf : {it}',
            f'{p}fraction = arith.andi {p}bits, {p}fractionmask : {it}',
            f'{p}expmax = arith.cmpi eq, {p}exponent, {p}inf : {it}',
            f'{p}nonzero_fraction = arith.cmpi ne, {p}fraction, {p}zero : {it}',
            f'{p}isnan = arith.andi {p}expmax, {p}nonzero_fraction : {bt}',
            f'{p}nanbits = arith.ori {p}bits, {p}quiet : {it}',
            f'{p}result = arith.select {p}isnan, {p}nanbits, {p}rawbits : {bt}, {it}',
            f'{dst} = arith.bitcast {p}result : {it} to {ty}',
        ]
        body = '\n    '.join(lines)
        if triton_bitcast:
            body = re.sub(r'arith.bitcast (%[\w]+) : (tensor<[^>]+>) to (tensor<[^>]+>)', r'tt.bitcast \1 : \2 -> \3',
                          body)
        return body

    text, count = pattern.subn(replace, text)
    if not count: return mod, []
    return parse(mod, text,
                 Path(directory) / 'tanh-guard.mlir'), [{
                     'count': count, 'bounds': [-16.0, 16.0], 'nan': 'integer classify and restore quiet payload',
                     'scope': 'native SDNN f32 tanh input/output'
                 }]


def preserve_multiply_specials(mod, directory):
    """Repair SDNN multiplication zero suppression without changing finite results.

    Native EW multiplication can erase NaN/Inf when its second input is zero.
    Keep its arithmetic, and use integer classifications to restore IEEE NaN
    results. A nonzero normal finite constant or a square cannot encounter that
    invalid operand pair. No matrix, layout, or launch template is supplied.
    """
    import math
    source = serialize(mod, directory, 'before_ieee_multiply')
    constants = {}
    for name, value in re.findall(r'(%\w+) = arith.constant dense<([^>]+)> : tensor<(?:\d+x)+f32>', source):
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number) and 2**-126 <= abs(number) <= (2 - 2**-23) * 2**127:
            constants[name] = number
    records = []
    pattern = re.compile(
        r'^(?P<indent>\s*)(?P<dst>%\w+) = arith\.mulf (?P<a>%\w+), (?P<b>%\w+) : (?P<ty>tensor<(?:\d+x)+f32>)$', re.M)

    def replace(m):
        if m['a'] == m['b'] or m['a'] in constants or m['b'] in constants:
            return m[0]
        i = len(records)
        prefix = f'%ieee_mul_{i}_'
        ty = m['ty']
        ity = ty.replace('f32>', 'i32>')
        bty = ty.replace('f32>', 'i1>')
        a = m['a']
        b = m['b']
        lines = [
            f'{prefix}raw = arith.mulf {a}, {b} : {ty}',
            f'{prefix}mask = arith.constant dense<2147483647> : {ity}',
            f'{prefix}inf = arith.constant dense<2139095040> : {ity}',
            f'{prefix}zero = arith.constant dense<0> : {ity}',
            f'{prefix}nan = arith.constant dense<2143289344> : {ity}',
            f'{prefix}abits = arith.bitcast {a} : {ty} to {ity}',
            f'{prefix}bbits = arith.bitcast {b} : {ty} to {ity}',
            f'{prefix}amag = arith.andi {prefix}abits, {prefix}mask : {ity}',
            f'{prefix}bmag = arith.andi {prefix}bbits, {prefix}mask : {ity}',
            f'{prefix}anan = arith.cmpi ugt, {prefix}amag, {prefix}inf : {ity}',
            f'{prefix}bnan = arith.cmpi ugt, {prefix}bmag, {prefix}inf : {ity}',
            f'{prefix}ainf = arith.cmpi eq, {prefix}amag, {prefix}inf : {ity}',
            f'{prefix}binf = arith.cmpi eq, {prefix}bmag, {prefix}inf : {ity}',
            f'{prefix}azero = arith.cmpi eq, {prefix}amag, {prefix}zero : {ity}',
            f'{prefix}bzero = arith.cmpi eq, {prefix}bmag, {prefix}zero : {ity}',
            f'{prefix}invalid_a = arith.andi {prefix}ainf, {prefix}bzero : {bty}',
            f'{prefix}invalid_b = arith.andi {prefix}binf, {prefix}azero : {bty}',
            f'{prefix}invalid = arith.ori {prefix}invalid_a, {prefix}invalid_b : {bty}',
            f'{prefix}any_nan = arith.ori {prefix}anan, {prefix}bnan : {bty}',
            f'{prefix}bad = arith.ori {prefix}invalid, {prefix}any_nan : {bty}',
            f'{prefix}rawbits = arith.bitcast {prefix}raw : {ty} to {ity}',
            f'{prefix}fixed = arith.select {prefix}bad, {prefix}nan, {prefix}rawbits : {bty}, {ity}',
            f"{m['dst']} = arith.bitcast {prefix}fixed : {ity} to {ty}",
        ]
        records.append({
            'result': m['dst'], 'type': ty, 'finite_arithmetic': 'unchanged', 'repair':
            'NaN propagation and zero-times-Inf'
        })
        return '\n'.join(m['indent'] + line for line in lines)

    changed = pattern.sub(replace, source)
    return (parse(mod, changed, Path(directory) / 'ieee-multiply.mlir') if records else mod), records


def prepare(mod, metadata):
    if not _ENABLED.get():
        metadata['native_exact_gelu'] = metadata['native_relu'] = '[]'
        return mod
    # Metadata prevents the temporary exp opcode from escaping to execution.
    # Any mixed real-exp graph keeps the original erf lowering instead.
    with tempfile.TemporaryDirectory(prefix='native-activation-', dir=os.environ['TRITON_CACHE_DIR']) as directory:
        mod, tanh_bounds = bound_tanh_inputs(mod, directory)
        mod, gelu = exact_gelu_to_placeholder(mod, directory)
        mod, relu_records = relu(mod, directory, 'native_relu_placeholder')
        mod, multiply = preserve_multiply_specials(mod, directory)
    metadata['native_tanh_bounds'] = json.dumps(tanh_bounds)
    metadata['native_ieee_multiply'] = json.dumps(multiply)
    metadata['native_exact_gelu'] = json.dumps(gelu)
    metadata['native_relu'] = json.dumps(relu_records)
    return mod


def restore_relu(mod, directory):
    """Standalone native maximum discards NaN; linear's ReLU propagates it."""
    text = serialize(mod, directory, 'before_relu_restore')
    zero = set(re.findall(r'(%[\w]+) = arith.constant 0\.000000e\+00 : f32', text))
    ones = re.findall(r'(%[\w]+) = arith.constant 1\.000000e\+00 : f32', text)
    records = []
    pattern = re.compile(
        r'^(?P<indent>\s*)sdnn\.ew maximum (?P<dst>%[\w]+) src0 (?P<src>%[\w]+) c (?P<c>%[\w]+) \{(?P<attrs>[^\n]+)\} : \((?P<dstty>memref<[^>]+>), (?P<srcty>memref<[^>]+>), f32\) -> \(\)$',
        re.M)

    def replace(match):
        if match['c'] not in zero: return match[0]
        assert ones, 'cannot identify exact linear coefficient'
        result = (f"{match['indent']}sdnn.ew linear {match['dst']} src0 {match['src']} a {ones[0]} "
                  f"{{activation = 1 : i32, {match['attrs']}}} : ({match['dstty']}, {match['srcty']}, f32) -> ()")
        records.append({'from': match[0].strip(), 'to': result.strip()})
        return result

    changed = pattern.sub(replace, text)
    assert records or 'activation = 1 : i32' in text, 'ReLU was not lowered to a native activation'
    return (parse(mod, changed, Path(directory) / 'relu-native.mlir') if records else mod), records


def retune_gelu(mod, metadata):
    """Preserve the native 512-entry ABI, refine the GELU interpolation grid."""
    from triton._C.libtriton import xpu
    with tempfile.TemporaryDirectory(prefix='gelu-lut-', dir=os.environ['TRITON_CACHE_DIR']) as directory:
        text = serialize(mod, directory, 'before_gelu_lut_refinement')
        records = []

        def replace(m):
            line = m[0]
            assert 'packedid = 887693 : ui32' in line, 'unknown native GELU table ABI'
            new = line.replace('packedid = 887693', 'packedid = 887629')
            for name, value in (('lut_min', '-8.000000e+00'), ('lut_max', '7.984375e+00'),
                                ('kOverflow', '1.000000e+00'), ('kUnderflow', '0.000000e+00')):
                new, count = re.subn(name + r' = [-+0-9.e]+ : f32', name + ' = ' + value + ' : f32', new)
                assert count == 1, (name, line)
            records.append({'before': line, 'after': new})
            return new

        changed = re.sub(r'^.*sdnn\.ew .*lut_func gelu .*$', replace, text, flags=re.M)
        assert len(records) == 1
        result = parse(mod, changed, Path(directory) / 'gelu-lut-refined.mlir')
        info = xpu.get_lut_info(result)
        assert list(info) == ['tt.gelu']
        gelu = info['tt.gelu']
        assert (gelu.dtype, gelu.mode, gelu.size, gelu.min, gelu.interval) == (0, 1, 512, -8.0, 0.015625)
        metadata['native_gelu_lut_refinement'] = json.dumps(records)
        return result


def install_stage(native):
    """Split the native pass pipeline once, before its activation table pass.

    The original scheduling, bufferization, native instruction selection and
    synchronization passes remain in their original order. Unknown pipeline
    versions fail closed instead of silently emitting a placeholder function.
    """
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(native)))
    tree.body[0].decorator_list = []

    class InsertStage(ast.NodeTransformer):
        count = 0
        table_count = 0

        def visit_If(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "opt.exp_range != ':0'":
                self.table_count += 1
                return [node] + ast.parse(
                    "if metadata.get('native_exact_gelu', '[]') != '[]':\n    pm.run(mod, 'after_native_activation_table')\n    mod = _native_retune_gelu(mod, metadata)\n    pm = ir.pass_manager(mod.context)\n    pm.enable_debug()"
                ).body
            return node

        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'add_tritonsdnn_fuse_relu_activation_pass'):
                self.count += 1
                return [node] + ast.parse(
                    "if metadata.get('native_exact_gelu', '[]') != '[]' or metadata.get('native_relu', '[]') != '[]':\n    pm.run(mod, 'before_native_activation')\n    mod = _native_restore_activation(mod, metadata)\n    pm = ir.pass_manager(mod.context)\n    pm.enable_debug()"
                ).body
            return node

    patch = InsertStage()
    tree = patch.visit(tree)
    assert (patch.count, patch.table_count) == (1, 1), 'unknown native SDNN activation pipeline'

    def restore(mod, metadata):
        with tempfile.TemporaryDirectory(prefix='native-gelu-', dir=os.environ['TRITON_CACHE_DIR']) as directory:
            if metadata.get('native_exact_gelu', '[]') != '[]':
                mod, records = restore_gelu(mod, directory)
                assert len(records) == len(json.loads(metadata['native_exact_gelu']))
                metadata['native_exact_gelu_restored'] = json.dumps(records)
            if metadata.get('native_relu', '[]') != '[]':
                mod, records = restore_relu(mod, directory)
                metadata['native_relu_restored'] = json.dumps(records)
        return mod

    namespace = dict(native.__globals__, _native_restore_activation=restore, _native_retune_gelu=retune_gelu)
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<native_sdnn_activation_stage>', 'exec'), namespace)
    return namespace['make_ttxir']
