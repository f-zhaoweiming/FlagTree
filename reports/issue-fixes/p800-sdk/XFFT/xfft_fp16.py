"""One contiguous, interleaved FP16 C2C transform on the current P800 stream."""
import ctypes as C
from contextlib import contextmanager

import torch

NOT_IMPLEMENTED = 14
COMPLEX_HALF = 6
COMPLEX_FLOAT = 4


def _call(lib, name, argtypes):
    function = getattr(lib, name)
    function.argtypes = argtypes
    function.restype = C.c_int
    return function


def _check(name, status):
    if status:
        raise RuntimeError(f"{name} returned XFFT status {status}")


@contextmanager
def _handle(lib):
    handle = C.c_void_p()
    _check("cufftCreate", _call(lib, "cufftCreate", [C.POINTER(C.c_void_p)])(C.byref(handle)))
    try:
        yield handle
    finally:
        status = _call(lib, "cufftDestroy", [C.c_void_p])(handle)
        handle.value = None
        _check("cufftDestroy", status)


def _plan(lib, handle, length, dtype):
    dim = (C.c_longlong * 1)(length)
    dimptr = C.POINTER(C.c_longlong)
    wide = C.c_longlong
    size = C.c_size_t()
    plan = _call(lib, "cufftXtMakePlanMany", [
        C.c_void_p, C.c_int, dimptr, dimptr, wide, wide, C.c_int,
        dimptr, wide, wide, C.c_int, wide, C.POINTER(C.c_size_t), C.c_int,
    ])
    return plan(handle, 1, dim, dim, 1, length, dtype, dim, 1, length, dtype,
                1, C.byref(size), dtype)


@contextmanager
def fp16_c2c_plan(lib, length, device):
    """Yield (execute, route, native_status). Both directions use XFFT's unnormalized convention."""
    if length < 1:
        raise ValueError("length must be positive")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("expected an XPyTorch accelerator device")
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream()
        def bind(handle):
            if stream.cuda_stream:
                _check("cufftSetStream", _call(lib, "cufftSetStream", [C.c_void_p, C.c_void_p])(
                    handle, stream.cuda_stream))
        with _handle(lib) as native:
            status = _plan(lib, native, length, COMPLEX_HALF)
            if status == 0:
                bind(native)
                try:
                    yield _executor(lib, native, length, stream, False), "native_fp16", status
                finally:
                    stream.synchronize()
                return
            if status != NOT_IMPLEMENTED:
                _check("cufftXtMakePlanMany(FP16)", status)
        with _handle(lib) as promoted:
            _check("cufftXtMakePlanMany(FP32)", _plan(lib, promoted, length, COMPLEX_FLOAT))
            bind(promoted)
            try:
                yield _executor(lib, promoted, length, stream, True), "xfft_fp32_final_fp16", status
            finally:
                stream.synchronize()


def _executor(lib, handle, length, stream, promoted):
    name = "cufftExecC2C" if promoted else "cufftXtExec"
    run = _call(lib, name, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int])
    def execute(values, direction=-1):
        if not handle.value:
            raise RuntimeError("the XFFT plan has already been destroyed")
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if (values.device != stream.device or values.dtype != torch.float16 or
                tuple(values.shape) not in ((length, 2), (1, length, 2)) or not values.is_contiguous()):
            raise ValueError("expected contiguous FP16 real/imaginary pairs for one plan on its device")
        if torch.cuda.current_stream(values.device) != stream:
            raise ValueError("execute on the stream used to create the plan")
        source = values.float() if promoted else values
        output = torch.empty_like(source)
        _check(name, run(handle, source.data_ptr(), output.data_ptr(), direction))
        return output.half() if promoted else output
    return execute
