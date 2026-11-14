# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compare serial, forced-parallel and automatic tile-dataflow scheduling.

Run with a FlagTree build containing enable_tile_dataflow, for example:
  python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --output /tmp/tile-dataflow.json

Measurements are per-launch GPU times from CUDA graph replay with hot inputs.
They exclude tracing/compilation and Python launch overhead. All variants use
the same kernel, inputs, tile dimensions and requested default-group warps.
The baseline suite reproduces the original balanced branch cases. The holdout
suite uses different shapes and an additional imbalanced nonlinear kernel.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import functools
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

import triton
import triton.language as tl
import triton.testing


VARIANTS = ("baseline", "forced", "automatic")
TIMING_ORDERS = (
    ("baseline", "forced", "automatic"),
    ("automatic", "forced", "baseline"),
    ("forced", "baseline", "automatic"),
    ("automatic", "baseline", "forced"),
    ("baseline", "automatic", "forced"),
    ("forced", "automatic", "baseline"),
)
SUITE_SHAPES = {
    "baseline": [(4, 128), (64, 1024), (128, 128), (4096, 128), (4096, 1024), (1027, 513)],
    "holdout": [(3, 65), (12, 257), (97, 769), (257, 255), (1536, 1009), (8192, 511)],
}


@triton.jit
def _tile_branches(X, Y, Out, M: tl.constexpr, N: tl.constexpr,
                   SHARED_INPUT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    if SHARED_INPUT:
        y = x
    else:
        y = tl.load(Y + offsets, valid, other=0).to(tl.float32)
    left = tl.sum(tl.where(valid, tl.exp(-(x * x)), 0.0), axis=1)
    right = tl.sum(tl.where(valid, tl.sqrt(tl.abs(y) + 1.0), 0.0), axis=1)
    tl.store(Out + rows, (left + right) / N, rows < M)


@triton.jit
def _imbalanced_branches(X, Y, Out, M: tl.constexpr, N: tl.constexpr,
                         SHARED_INPUT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         CHAIN_DEPTH: tl.constexpr, PREFIX_DEPTH: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    for _ in tl.static_range(PREFIX_DEPTH):
        x = x / (1.0 + x * x)
    if SHARED_INPUT:
        y = x
    else:
        y = tl.load(Y + offsets, valid, other=0).to(tl.float32)
    for _ in tl.static_range(CHAIN_DEPTH):
        x = x / (1.0 + x * x)
    right = y / (1.0 + tl.abs(y))
    tl.store(Out + offsets, x + right, valid)


@functools.lru_cache(maxsize=1)
def _cuda_driver():
    driver = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
    query = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    query.restype = ctypes.c_int
    return driver


def _resources(kernel, query_occupancy=False):
    resources = {
        "num_warps": kernel.metadata.num_warps,
        "shared_bytes": kernel.metadata.shared,
        "registers_per_thread": getattr(kernel, "n_regs", None),
        "spills": getattr(kernel, "n_spills", None),
        "max_threads": getattr(kernel, "n_max_threads", None),
        "scheduled_forks": len(re.findall(r"(?m)^[ \t]*%[^\n]* = ttg\.warp_specialize\(", kernel.asm["ttgir"])),
        "active_ctas_per_sm": None,
        "active_warps_per_sm": None,
        "occupancy_query_error": None,
    }
    if query_occupancy:
        try:
            active = ctypes.c_int()
            status = _cuda_driver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active), ctypes.c_void_p(kernel.function),
                kernel.metadata.num_warps * 32, kernel.metadata.shared,
            )
            if status:
                raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor returned CUDA error {status}")
            resources["active_ctas_per_sm"] = active.value
            resources["active_warps_per_sm"] = active.value * kernel.metadata.num_warps
        except Exception as error:
            resources["occupancy_query_error"] = f"{type(error).__name__}: {error}"
    return resources


def _decisions(kernel):
    raw = getattr(kernel.metadata, "tile_dataflow_cost_model", "")
    decisions = json.loads(raw or "[]")
    if not isinstance(decisions, list) or not all(isinstance(item, dict) for item in decisions):
        raise ValueError("tile_dataflow_cost_model must be a JSON array of decisions")
    return decisions


def _strip_cost_diagnostics(ttgir):
    # Remove exactly the diagnostic string attribute, including its MLIR string
    # escaping. All operations, types, other attributes and locations remain.
    pattern = r'"?tle\.tile_dataflow\.cost_model"?\s*=\s*"(?:\\.|[^"\\])*"'
    match = re.search(pattern, ttgir)
    if match is None:
        return ttgir
    before, after = ttgir[:match.start()], ttgir[match.end():]
    if before.endswith(", "):
        before = before[:-2]
    elif after.startswith(", "):
        after = after[2:]
    return before + after


def _environment():
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"),
                              capture_output=True, text=True, check=False)
    device = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "triton_path": triton.__file__,
        "device": device.name,
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessors": device.multi_processor_count,
        "total_memory_bytes": device.total_memory,
    }


def _case(rows, columns, shared_input, family, args):
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn((rows, columns), generator=generator).to(device="cuda", dtype=dtype)
    y = torch.randn((rows, columns), generator=generator).to(device="cuda", dtype=dtype)
    left = x.float()
    if family == "balanced":
        other = x.float() if shared_input else y.float()
        expected = (torch.exp(-(left * left)).sum(1) + torch.sqrt(other.abs() + 1).sum(1)) / columns
        output_shape = (rows,)
        entry = _tile_branches
        extra = {}
    else:
        for _ in range(args.prefix_depth):
            left = left / (1 + left * left)
        other = left if shared_input else y.float()
        for _ in range(args.chain_depth):
            left = left / (1 + left * left)
        expected = left + other / (1 + other.abs())
        output_shape = (rows, columns)
        entry = _imbalanced_branches
        extra = {"CHAIN_DEPTH": args.chain_depth, "PREFIX_DEPTH": args.prefix_depth}
    outputs = {}
    kernels = {}
    failed_variants = {}
    variants = {variant: {"correct": False, "status": "pending", "median_us": None, "rounds": [],
                          "resources": None, "selected_schedule": None, "cost_model_decisions": []}
                for variant in VARIANTS}

    def record_failure(variant, stage, error):
        failure = {"stage": stage, "error": f"{type(error).__name__}: {error}"}
        failed_variants[variant] = failure
        variants[variant].update(status="failed", failure=failure)

    def launch(variant):
        return entry[(triton.cdiv(rows, args.block_rows),)](
            x, y, outputs[variant], rows, columns, shared_input,
            BLOCK_M=args.block_rows, BLOCK_N=triton.next_power_of_2(columns),
            num_warps=args.num_warps, enable_fp_fusion=False,
            enable_tile_dataflow=variant != "baseline", tile_dataflow_force=variant == "forced",
            **extra,
        )

    for variant in VARIANTS:
        stage = "compile_and_launch"
        try:
            outputs[variant] = torch.empty(output_shape, device="cuda")
            kernels[variant] = launch(variant)
            stage = "correctness"
            torch.testing.assert_close(outputs[variant], expected, rtol=3e-5, atol=3e-6)
            variants[variant]["correct"] = True
            stage = "diagnostics"
            ttgir = kernels[variant].asm["ttgir"]
            variants[variant].update(
                resources=_resources(kernels[variant], query_occupancy=args.resources),
                selected_schedule="parallel" if "ttg.warp_specialize" in ttgir else "serial",
                cost_model_decisions=_decisions(kernels[variant]),
                status="validated",
            )
            if variant == "forced" and "ttg.warp_specialize" not in ttgir:
                raise RuntimeError("forced variant has no executable warp-specialized region")
            if variant == "baseline" and "ttg.warp_specialize" in ttgir:
                raise RuntimeError("baseline variant unexpectedly contains a scheduled fork")
        except Exception as error:
            record_failure(variant, stage, error)

    if args.dump_ir_dir:
        args.dump_ir_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{family}-{rows}x{columns}-shared{int(shared_input)}-{args.dtype}-bm{args.block_rows}-w{args.num_warps}"
        if family == "imbalanced":
            prefix += f"-chain{args.chain_depth}-prefix{args.prefix_depth}"
        for variant, kernel in kernels.items():
            for stage in ("ttgir", "ptx"):
                (args.dump_ir_dir / f"{prefix}-{variant}.{stage}").write_text(kernel.asm[stage])

    serial_code_equivalent = None
    if "automatic" not in failed_variants and variants["automatic"]["selected_schedule"] == "serial":
        if "baseline" in failed_variants:
            serial_code_equivalent = {"ttgir": None, "ptx": None, "reason": "baseline_failed"}
        else:
            serial_code_equivalent = {
                "ttgir": _strip_cost_diagnostics(kernels["automatic"].asm["ttgir"]) == kernels["baseline"].asm["ttgir"],
                "ptx": kernels["automatic"].asm["ptx"] == kernels["baseline"].asm["ptx"],
            }
            if not all(serial_code_equivalent.values()):
                record_failure("automatic", "serial_code_equivalence", RuntimeError(str(serial_code_equivalent)))

    for variant in VARIANTS:
        if variant in failed_variants:
            continue
        try:
            for _ in range(args.warmup):
                launch(variant)
            torch.cuda.synchronize()
        except Exception as error:
            record_failure(variant, "warmup", error)
    timing_orders = []
    for index in range(args.rounds):
        order = [variant for variant in TIMING_ORDERS[index % len(TIMING_ORDERS)] if variant not in failed_variants]
        timing_orders.append(order)
        for variant in order:
            try:
                p50, p20, p80 = triton.testing.do_bench_cudagraph(
                    lambda: launch(variant), rep=args.rep_ms, quantiles=[0.5, 0.2, 0.8],
                )
                variants[variant]["rounds"].append({"p50_us": p50 * 1000, "p20_us": p20 * 1000, "p80_us": p80 * 1000})
            except Exception as error:
                record_failure(variant, f"timing_round_{index}", error)
    for variant in VARIANTS:
        if variant not in failed_variants:
            variants[variant]["median_us"] = statistics.median(item["p50_us"] for item in variants[variant]["rounds"])
            variants[variant]["status"] = "ok"

    def speedup(variant):
        baseline, candidate = variants["baseline"]["median_us"], variants[variant]["median_us"]
        return baseline / candidate if baseline is not None and candidate is not None else None

    result = {
        "rows": rows,
        "columns": columns,
        "family": family,
        "chain_depth": args.chain_depth if family == "imbalanced" else None,
        "prefix_depth": args.prefix_depth if family == "imbalanced" else None,
        "shared_input": shared_input,
        "dtype": args.dtype,
        "block_rows": args.block_rows,
        "block_columns": triton.next_power_of_2(columns),
        "correct": not failed_variants,
        "failed_variants": failed_variants,
        "automatic_serial_code_equivalent": serial_code_equivalent,
        "timing_orders": timing_orders,
        "forced_speedup": speedup("forced"),
        "automatic_speedup": speedup("automatic"),
    }
    result.update(variants)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("baseline", "holdout"), default="baseline")
    parser.add_argument("--shape", type=int, nargs=2, action="append", metavar=("ROWS", "COLUMNS"))
    parser.add_argument("--family", choices=("balanced", "imbalanced"), action="append",
                        help="repeat to select families; holdout defaults to both")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--input-mode", choices=("independent", "shared", "both"), default="both")
    parser.add_argument("--block-rows", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--num-warps", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--chain-depth", type=int, default=8, help="left branch depth in imbalanced kernel")
    parser.add_argument("--prefix-depth", type=int, default=0, help="serial shared-input prefix in imbalanced kernel")
    parser.add_argument("--warmup", type=int, default=50, help="warmup launches of each variant")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=float, default=30, help="CUDA graph benchmark repetition budget")
    parser.add_argument("--seed", type=int, default=61017)
    parser.add_argument("--resources", action="store_true", help="also query CUDA driver occupancy; failures report null")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-ir-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("an NVIDIA GPU and a matching FlagTree build are required")
    if args.rounds < 1 or args.warmup < 0 or args.rep_ms <= 0 or args.chain_depth < 1 or args.prefix_depth < 0:
        parser.error("rounds, rep-ms and chain-depth must be positive; warmup and prefix-depth must be nonnegative")
    shapes = args.shape or SUITE_SHAPES[args.suite]
    if any(min(shape) <= 0 for shape in shapes):
        parser.error("shape dimensions must be positive")
    modes = {"independent": (False,), "shared": (True,), "both": (False, True)}[args.input_mode]
    families = args.family or (["balanced"] if args.suite == "baseline" else ["balanced", "imbalanced"])
    report = {
        "environment": _environment(),
        "suite": args.suite,
        "families": families,
        "variants": VARIANTS,
        "measurement": "cuda_graph_replay_per_launch_hot_inputs",
        "warmup_launches_per_variant": args.warmup,
        "rounds": args.rounds,
        "rep_ms": args.rep_ms,
        "seed": args.seed,
        "query_driver_occupancy": args.resources,
        "cases": [],
    }
    failed = False
    for rows, columns in shapes:
        for shared in modes:
            for family in families:
                try:
                    result = _case(rows, columns, shared, family, args)
                except Exception as error:
                    result = {"rows": rows, "columns": columns, "shared_input": shared, "family": family,
                              "correct": False, "error": f"{type(error).__name__}: {error}",
                              "failed_variants": {variant: {"stage": "case_setup", "error": f"{type(error).__name__}: {error}"}
                                                  for variant in VARIANTS}}
                failed |= not result["correct"]
                report["cases"].append(result)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
