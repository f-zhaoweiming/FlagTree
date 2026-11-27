"""Experimental NVIDIA GELU(tanh) lowering; not used by compiler plans.

Keep libdevice's coefficients, rounding, FTZ, saturation and sign handling.
Evaluating both ranges before selection removes divergent control flow. The
validated SM89/libdevice pair is gated explicitly; other targets keep native
Inductor tanh. Retained for experiment reproduction; the default compiler path
uses native libdevice.tanh. This policy never changes materialization casts.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
from pathlib import Path

POLICY = "cuda_gelu_tanh_branchless_v1"
LIBDEVICE_SHA256 = "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee"
_ACTIVE = ContextVar("f06_cuda_precise_tanh", default=False)

# Matches the validated __nv_tanhf expansion, including unordered small-range
# selection for NaNs and FTZ for subnormal inputs. Do not reassociate its FMAs.
ASM = """{
.reg .f32 a, z, p, small, e, d, r, large;
.reg .b32 sign, bits;
.reg .pred use_small, saturated;
abs.ftz.f32 a, $1;
setp.ltu.f32 use_small, a, 0f3F19999A;
mul.f32 z, $1, $1;
fma.rn.ftz.f32 p, 0f3C80F082, z, 0fBD563CAE;
fma.rn.ftz.f32 p, p, z, 0f3E085941;
fma.rn.ftz.f32 p, p, z, 0fBEAAA9ED;
fma.rn.ftz.f32 p, p, z, 0f00000000;
fma.rn.ftz.f32 small, p, $1, $1;
mul.f32 e, a, 0f4038AA3B;
ex2.approx.ftz.f32 e, e;
add.f32 d, e, 0f3F800000;
rcp.approx.ftz.f32 r, d;
fma.rn.ftz.f32 large, r, 0fC0000000, 0f3F800000;
setp.ge.f32 saturated, a, 0f41102CB4;
selp.f32 large, 0f3F800000, large, saturated;
mov.b32 sign, $1;
and.b32 sign, sign, 0x80000000;
mov.b32 bits, large;
or.b32 bits, bits, sign;
mov.b32 large, bits;
selp.f32 $0, small, large, use_small;
}"""


@lru_cache(maxsize=16)
def _file_digest(path, size, mtime_ns, ctime_ns):
    # Include file identity in the memoization key so a replaced library does
    # not inherit the previous library's admission decision.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _libdevice_sha256():
    from triton import knobs
    from triton.backends.nvidia import compiler

    path = Path(knobs.nvidia.libdevice_path or Path(compiler.__file__).parent / "lib" / "libdevice.10.bc").resolve()
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def eligible(kind, inputs):
    import torch

    if kind not in ("matmul_bias_gelu_tanh", "add_gelu_tanh") or not inputs:
        return False
    if any(x.device.type != "cuda" or x.dtype not in (torch.float32, torch.float16, torch.bfloat16) for x in inputs):
        return False
    from triton.runtime import driver

    if type(driver.active).__module__ != "triton.backends.nvidia.driver":
        return False
    if any(torch.cuda.get_device_capability(x.device) != (8, 9) for x in inputs):
        return False
    try:
        return _libdevice_sha256() == LIBDEVICE_SHA256
    except (ImportError, AttributeError, OSError):
        return False


def tanh_expression(x):
    # Inductor embeds kernel source in another Python string. Avoid escaped
    # newlines being consumed by that outer layer before Triton imports it.
    asm = " ".join(ASM.split())
    return (f"tl.inline_asm_elementwise({asm!r}, constraints='=f,f', "
            f"args=[({x}).to(tl.float32)], dtype=tl.float32, is_pure=True, pack=1)")


@contextmanager
def precise_tanh_codegen():
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import config
    from torch._inductor.codegen.triton import TritonOverrides, maybe_upcast_float32

    with compile_lock:
        descriptor = TritonOverrides.__dict__["tanh"]
        original = TritonOverrides.tanh

        @maybe_upcast_float32()
        def expression(x):
            return tanh_expression(x)

        def scoped(x):
            return expression(x) if _ACTIVE.get() else original(x)

        token = _ACTIVE.set(True)
        TritonOverrides.tanh = staticmethod(scoped)
        try:
            # Native FX keys do not include this scoped codegen override.
            # Generated Triton source/binary caches still key the exact source.
            with config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False}):
                yield
        finally:
            TritonOverrides.tanh = descriptor
            _ACTIVE.reset(token)
