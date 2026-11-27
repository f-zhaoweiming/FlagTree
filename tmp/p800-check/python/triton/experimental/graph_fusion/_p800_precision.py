"""Generic P800 scalar legalization and BF16 rounding.

No matrix code, tiling or launch template is supplied.
"""
import re
from contextlib import contextmanager
from contextvars import ContextVar

import hashlib
import os
import json
import tempfile
from pathlib import Path

_ACTIVE = ContextVar("flagtree_p800_precision", default=False)


def preserve_bf16_rounding(mod, special_values=True):
    from triton._C.libtriton import ir, passes
    import triton.language as tl
    exact_values = set()

    def classify(op):
        if op.get_name() == 'arith.maxnumf' and op.get_bool_attr('f06.bf16_exact'):
            exact_values.add(op.get_result(0).id())

    mod.walk(classify)
    casts = []
    mod.walk(lambda op: casts.append(op) if op.get_name() == 'arith.truncf' else None)
    rewritten = []
    for op in casts:
        source, dest = op.get_operand(0), op.get_result(0)
        if source.id() in exact_values:
            continue  # ReLU of an already BF16 value is exactly representable.
        src, dst = str(source.get_type()), str(dest.get_type())
        shape = re.fullmatch(r'tensor<((?:[0-9]+x)+)f32>', src)
        if not shape or dst != src.replace('f32>', 'bf16>'):
            continue
        dims = [int(x) for x in shape[1].rstrip('x').split('x')]
        b = ir.builder(mod.context)
        b.set_insertion_point_after(op)
        u32 = tl.block_type(tl.uint32, dims).to_ir(b)

        def constant(value):
            return b.create_splat(u32, b.get_uint32(value))

        bits = b.create_bitcast(source, u32)
        odd = b.create_and(b.create_lshr(bits, constant(16)), constant(1))
        rounded = b.create_and(b.create_add(b.create_add(bits, constant(0x7FFF)), odd), constant(0xFFFF0000))
        if special_values:
            magnitude = b.create_and(bits, constant(0x7FFFFFFF))
            is_special = b.create_icmpUGE(magnitude, constant(0x7F800000))
            is_nan = b.create_icmpUGT(magnitude, constant(0x7F800000))
            quiet_nan = b.create_or(bits, constant(0x00400000))
            special = b.create_select(is_nan, quiet_nan, bits)
            rounded = b.create_select(is_special, special, rounded)
        rounded_float = b.create_bitcast(rounded, source.get_type())
        new_cast = b.create_fp_trunc(rounded_float, dest.get_type())
        dest.replace_all_uses_with(new_cast)
        rewritten.append({'source': src, 'destination': dst})
    pm = ir.pass_manager(mod.context)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(mod, 'after_bf16_rounding')
    assert mod.verify()
    return rewritten


@contextmanager
def _compilation_scope():
    """Scope the passes to one serialized P800 compilation.

    The caller holds the existing native-template compile lock. Use an isolated
    Triton cache and force compilation because the pass is not in backend.hash.
    """
    from triton.backends.xpu.compiler import XPUBackend
    from triton._C.libtriton import xpu
    original = XPUBackend.make_ttir
    original_table = XPUBackend.make_ewtable
    original_llir = XPUBackend.make_llir
    from ._p800_division import split_division_rows
    from ._p800_activation import lower_scalar_ops
    from ._p800_tables import make_ewtable
    keys = ('TRITON_CACHE_DIR', 'TRITON_ALWAYS_COMPILE')
    saved = {key: os.environ.get(key) for key in keys}
    cache = saved['TRITON_CACHE_DIR']
    if not cache:
        from triton import knobs
        cache = knobs.cache.dir
    digest = hashlib.sha256(b''.join(
        (Path(__file__).parent / name).read_bytes()
        for name in ('_p800_precision.py', '_p800_activation.py', '_p800_tables.py', '_p800_division.py',
                     '_p800_native_activation.py', '_p800_parallel_rne.py'))).hexdigest()[:16]

    def make_ttir(mod, metadata, options):
        mod = original(mod, metadata, options)
        if not _ACTIVE.get() or options.arch != 3:
            return mod
        if not (options.is_sdnn or xpu.is_sdnn_kernel(mod)):
            from ._p800_native_activation import bound_tanh_inputs
            with tempfile.TemporaryDirectory(prefix='cluster-tanh-', dir=os.environ['TRITON_CACHE_DIR']) as directory:
                mod, records = bound_tanh_inputs(mod, directory, triton_bitcast=True)
            for record in records:
                record['scope'] = 'native cluster f32 tanh input/output'
            metadata['cluster_tanh_bounds'] = json.dumps(records)
            metadata['cluster_rounding_rewrites'] = json.dumps(preserve_bf16_rounding(mod, special_values=True))
            return mod
        from ._p800_native_activation import prepare
        mod = prepare(mod, metadata)
        lower_scalar_ops(mod, clamp_max=False)
        preserve_bf16_rounding(mod, special_values=True)
        return mod

    def make_llir(mod, metadata, options):
        if _ACTIVE.get() and options.arch == 3 and metadata.get('is_sdnn'):
            with tempfile.TemporaryDirectory(prefix='division-', dir=os.environ['TRITON_CACHE_DIR']) as directory:
                from ._p800_parallel_rne import lower
                mod, parallel = lower(mod, directory, options)
                metadata["parallel_rne"] = json.dumps(parallel)
                mod, repairs = split_division_rows(mod, directory)
                if parallel:
                    from ._p800_native_activation import serialize
                    text = serialize(mod, directory, "parallel_rne_final")
                    identity = hashlib.sha256(text.encode()).hexdigest()
                    audit = Path(os.environ["TRITON_CACHE_DIR"]) / "parallel-ir"
                    audit.mkdir(parents=True, exist_ok=True)
                    (audit / (identity + ".mlir")).write_text(text)
                    metadata["parallel_rne_ir_sha256"] = identity
            metadata['division_row_repair'] = json.dumps(repairs)
        return original_llir(mod, metadata, options)

    def table(mod, metadata, options):
        return make_ewtable(mod, metadata, options) if _ACTIVE.get() and options.arch == 3 else original_table(
            mod, metadata, options)

    XPUBackend.make_llir = staticmethod(make_llir)
    XPUBackend.make_ttir = staticmethod(make_ttir)
    XPUBackend.make_ewtable = staticmethod(table)
    os.environ['TRITON_CACHE_DIR'] = cache + '-p800-ops-' + digest
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    try:
        yield
    finally:
        XPUBackend.make_llir = staticmethod(original_llir)
        XPUBackend.make_ttir = staticmethod(original)
        XPUBackend.make_ewtable = staticmethod(original_table)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def compilation_scope():
    from torch._dynamo.convert_frame import compile_lock
    with compile_lock:
        if _ACTIVE.get():
            raise RuntimeError("nested P800 precision compilation is unsupported")
        token = _ACTIVE.set(True)
        try:
            with _compilation_scope():
                yield
        finally:
            _ACTIVE.reset(token)
