"""Test raw plan status and actual P800 output against CPU FP64 on identical FP16 samples."""
import argparse
import ctypes as C
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from xfft_fp16 import fp16_c2c_plan


def verify(lib_path, lengths=(128, 512, 1024)):
    lib = C.CDLL(str(lib_path))
    cases = []
    for n in lengths:
        gen = torch.Generator().manual_seed(20261126 + n)
        random = torch.randn(n, 2, generator=gen).half()
        impulse = torch.zeros_like(random)
        impulse[1, 0], impulse[1, 1] = 1.0, -0.5
        with fp16_c2c_plan(lib, n, torch.device("cuda", torch.cuda.current_device())) as (execute, route, status):
            for pattern, host in (("random", random), ("impulse", impulse)):
                samples = host.cuda()
                for sign in (-1, 1):
                    z = torch.view_as_complex(host.double())
                    expected = torch.fft.fft(z) if sign == -1 else torch.fft.ifft(z, norm="forward")
                    reference = torch.view_as_real(expected)
                    result = execute(samples, sign).cpu()
                    # One binary16 ULP (take the larger neighbor at powers of two).
                    rounded = reference.half()
                    up = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
                    down = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
                    ulp = torch.maximum(up.double()-rounded.double(), rounded.double()-down.double())
                    error = (result.double()-reference).abs()
                    # Account for FP32 roundoff near exact zeros.
                    tolerance = ulp + 4 * torch.finfo(torch.float32).eps * host.abs().double().sum()
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(5):
                        start = time.perf_counter_ns()
                        execute(samples, sign)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter_ns()-start)/1000)
                    cases.append(dict(length=n, pattern=pattern, direction=sign, route=route,
                                      native_status=status, dtype=str(result.dtype),
                                      samples=result.numel(), finite=bool(result.isfinite().all()),
                                      max_abs=error.max().item(), rmse=error.square().mean().sqrt().item(),
                                      different_from_fp64_rounded=int((result != rounded).sum()),
                                      beyond_tolerance=int((error > tolerance).sum()),
                                      median_wall_us=statistics.median(times),
                                      passed=bool(result.dtype==torch.float16 and result.isfinite().all() and (error<=tolerance).all())))
    return dict(library_sha256=hashlib.sha256(Path(lib_path).read_bytes()).hexdigest(), cases=cases)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("library", type=Path)
    a = p.parse_args()
    result = verify(a.library)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(row["passed"] for row in result["cases"]) else 1)
