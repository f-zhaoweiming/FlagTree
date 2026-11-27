"""Admit P800 to native Inductor GEMM templates and inspect the actual binary.

No arithmetic, tile, grid, device code or backend pass is supplied here.
The NVIDIA SM-count heuristic is scoped to one identified P800 compilation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path
import threading

_DEVICE = ContextVar("flagtree_p800_template_device", default=None)
_LOCK = threading.RLock()


def is_p800(value):
    if value.device.type != "cuda":
        return False
    from triton.runtime import driver
    import torch
    active = driver.active
    cls = type(active)
    if cls.__module__ != "triton.backends.xpu.driver" or cls.__name__ != "XPUDriver":
        return False
    with torch.cuda.device(value.device):
        target = active.get_current_target()
    # The report P800 identifies itself as XPU architecture 3, warp_size=1.
    # Other XPU generations need their own validation, not a CUDA-alias guess.
    return type(target.arch) is int and target.arch == 3 and target.warp_size == 1


@contextmanager
def native_template_eligibility(device):
    import torch
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import utils

    index = device.index if device.index is not None else torch.cuda.current_device()
    with compile_lock, _LOCK:
        original = getattr(utils, "is_big_gpu", None)
        if not callable(original):
            raise RuntimeError("P800 native GEMM eligibility API is unavailable")
        if _DEVICE.get() is not None:
            raise RuntimeError("nested P800 template compilation is unsupported")
        token = _DEVICE.set(index)

        def eligible(index_or_device=0):
            requested = _DEVICE.get()
            # Torch 2.5 passes an index; 2.9 passes layout.device. Preserve
            # device locality for both forms, including another device type.
            if isinstance(index_or_device, torch.device):
                if index_or_device.type != device.type:
                    return original(index_or_device)
                queried_index = index_or_device.index
                if queried_index is None:
                    queried_index = torch.cuda.current_device()
            else:
                queried_index = index_or_device
            if requested is not None and queried_index == requested:
                return True
            return original(index_or_device)

        utils.is_big_gpu = eligible
        try:
            yield
        finally:
            if utils.is_big_gpu is eligible:
                utils.is_big_gpu = original
            _DEVICE.reset(token)


def inspect_bound_binary(compiled, kernel_name):
    """Require one real bound XPU binary; never infer its resource from mm."""
    from torch._inductor.codecache import PyCodeCache

    cache = getattr(PyCodeCache, "cache", {})
    module = cache.get(compiled.cache_key)
    if module is None or Path(module.__file__).read_text() != compiled.source_code:
        raise RuntimeError("P800 generated wrapper identity could not be verified")
    tuner = getattr(module, kernel_name, None)
    if tuner is None or not callable(getattr(tuner, "precompile", None)):
        raise RuntimeError("P800 generated kernel binding could not be verified")
    tuner.precompile()
    if len(tuner.launchers) != 1:
        raise RuntimeError("P800 native GEMM did not resolve to one launcher")
    binary = getattr(tuner.launchers[0], "__globals__", {}).get("bin")
    metadata = getattr(binary, "metadata", None)
    is_sdnn = getattr(metadata, "is_sdnn", None)
    if type(is_sdnn) is not bool or getattr(metadata, "arch", None) != 3:
        raise RuntimeError("P800 bound binary resource or architecture is unverified")
    return {
        "resource": "sdnn" if is_sdnn else "cluster",
        "binary_hash": str(binary.hash),
        "kernel_name": metadata.name,
        "asm_sha256": {
            key: hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()
            for key, value in binary.asm.items()
        },
        "scope": "generated_wrapper_and_bound_native_binary",
    }
