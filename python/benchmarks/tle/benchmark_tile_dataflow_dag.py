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

"""Compare serial, forced and automatic schedules on multiway tile DAGs.

Example:
  python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --resources --output /tmp/tile-dag.json

CUDA graph replay times exclude compilation and Python launch overhead. The
same kernel, shape and default-group size are used for all three variants.
Independent failures remain visible while successful variants are measured.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.testing

from benchmark_tile_dataflow_schedule import (
    VARIANTS, TIMING_ORDERS, _decisions, _environment, _resources, _strip_cost_diagnostics,
)

TOPOLOGIES = ("three", "four", "disconnected", "fanout", "waves2", "waves3")
DEFAULT_SHAPES = ((5, 96), (96, 384), (1025, 768), (6144, 1536))


@triton.jit
def _wide_dag(A, B, C, D, Out, Aux, M: tl.constexpr, N: tl.constexpr,
              BRANCHES: tl.constexpr, SHARED: tl.constexpr, DISCONNECTED: tl.constexpr,
              FANOUT: tl.constexpr, DEPTH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    a = tl.load(A + offsets, mask, other=0).to(tl.float32)
    if SHARED:
        b, c, d = a, a, a
    else:
        b = tl.load(B + offsets, mask, other=0).to(tl.float32)
        c = tl.load(C + offsets, mask, other=0).to(tl.float32)
        if BRANCHES == 4:
            d = tl.load(D + offsets, mask, other=0).to(tl.float32)
    for _ in tl.static_range(DEPTH):
        a = tl.exp(-(a * a))
        b = tl.sqrt(tl.abs(b) + 1.0)
        c = c / (1.0 + c * c)
        if BRANCHES == 4:
            d = tl.sqrt(tl.abs(d) + 2.0)
    ra = tl.sum(tl.where(mask, a, 0.0), 1)
    rb = tl.sum(tl.where(mask, b, 0.0), 1)
    rc = tl.sum(tl.where(mask, c, 0.0), 1)
    if BRANCHES == 4:
        rd = tl.sum(tl.where(mask, d, 0.0), 1)
    if DISCONNECTED:
        tl.store(Out + rows, (ra + rb) / N, rows < M)
        tl.store(Aux + rows, (rc + rd) / N, rows < M)
    else:
        if BRANCHES == 4:
            joined = (ra + rb) + (rc + rd)
        else:
            joined = (ra + rb) + rc
        tl.store(Out + rows, joined / N, rows < M)
        if FANOUT:
            tl.store(Aux + rows, ra / N, rows < M)



@triton.jit
def _sequential_dag(A, B, Out, M: tl.constexpr, N: tl.constexpr,
                    WAVES: tl.constexpr, FIRST_DEPTH: tl.constexpr, LATER_DEPTH: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(A + offsets, mask, other=0).to(tl.float32)
    y = tl.load(B + offsets, mask, other=0).to(tl.float32)
    initial_x, initial_y = x, y
    for _ in tl.static_range(FIRST_DEPTH):
        x = tl.exp(-(x * x))
        y = tl.sqrt(tl.abs(y) + 1.0)
        if FIRST_DEPTH > 1:
            x = x + 0.03125 * initial_x
            y = y + 0.0625 * initial_y
    value = x + y
    # The joined tensor is a true shared ancestor of the next fork. Its
    # consumers cannot be placed in the previous wave or run before its join.
    for _ in tl.static_range(WAVES - 1):
        left, right = value, value
        for _ in tl.static_range(LATER_DEPTH):
            left = tl.exp(-(left * left))
            right = tl.sqrt(tl.abs(right) + 1.0)
            if LATER_DEPTH > 1:
                left = left + 0.03125 * value
                right = right + 0.0625 * value
        value = left + right
    result = tl.sum(tl.where(mask, value, 0.0), 1) / N
    tl.store(Out + rows, result, rows < M)



def _wide_reference(inputs, branches, shared, depth):
    a, b, c, d = (value.float() for value in inputs)
    if shared:
        b, c, d = a, a, a
    for _ in range(depth):
        a = torch.exp(-(a * a))
        b = torch.sqrt(b.abs() + 1)
        c = c / (1 + c * c)
        d = torch.sqrt(d.abs() + 2)
    results = [value.mean(1) for value in (a, b, c, d)]
    return results[:branches]


def _physical_wave_groups(ttgir):
    # A persistent worker region can execute several dependent logical waves.
    # Only operation headers count: the cost JSON may itself name this op.
    headers = re.findall(r"(?m)^[ \t]*%[^\n]* = ttg\.warp_specialize\([^\n]*", ttgir)
    groups = []
    for header in headers:
        encoded = re.search(r"tle\.tile_dataflow\.logical_waves = (\d+) : i32", header)
        waves = int(encoded[1]) if encoded else 1
        if waves < 1:
            raise RuntimeError(f"invalid logical wave count in {header}")
        groups.append(waves)
    return groups


def _case(rows, columns, topology, shared, args):
    generator = torch.Generator().manual_seed(args.seed)
    inputs = [torch.randn((rows, columns), generator=generator).to(device="cuda", dtype=getattr(torch, args.dtype))
              for _ in range(4)]
    sequential = topology.startswith("waves")
    branches = 3 if topology == "three" else 4
    waves = int(topology[-1]) if sequential else 1
    if sequential:
        x, y = inputs[0].float(), inputs[1].float()
        initial_x, initial_y = x, y
        for _ in range(args.first_wave_depth):
            x = torch.exp(-(x * x))
            y = torch.sqrt(y.abs() + 1)
            if args.first_wave_depth > 1:
                x = x + 0.03125 * initial_x
                y = y + 0.0625 * initial_y
        value = x + y
        for _ in range(waves - 1):
            left, right = value, value
            for _ in range(args.later_wave_depth):
                left = torch.exp(-(left * left))
                right = torch.sqrt(right.abs() + 1)
                if args.later_wave_depth > 1:
                    left = left + 0.03125 * value
                    right = right + 0.0625 * value
            value = left + right
        expected, auxiliary = value.mean(1), None
    else:
        values = _wide_reference(inputs, branches, shared, args.depth)
        expected = values[0] + values[1]
        auxiliary = values[2] + values[3] if topology == "disconnected" else None
        if topology != "disconnected":
            expected = expected + (values[2] if branches == 3 else values[2] + values[3])
        if topology == "fanout":
            auxiliary = values[0]

    outputs, kernels, failed = {}, {}, {}
    variants = {variant: {"correct": False, "status": "pending", "median_us": None, "rounds": [],
                          "resources": None, "cost_model_decisions": [], "scheduled_wave_widths": [],
                          "physical_wave_groups": [],
                          "selected_schedule": None, "schedule_reason": None}
                for variant in VARIANTS}

    def launch(variant):
        options = dict(BLOCK_M=args.block_rows, BLOCK_N=triton.next_power_of_2(columns),
                       num_warps=args.num_warps, enable_fp_fusion=False,
                       enable_tile_dataflow=variant != "baseline", tile_dataflow_force=variant == "forced",
                       tile_dataflow_max_partitions=args.max_partitions,
                       tile_dataflow_min_speedup=args.min_speedup,
                       tile_dataflow_coalesce=not args.no_coalesce)
        out, aux = outputs[variant]
        if sequential:
            return _sequential_dag[(triton.cdiv(rows, args.block_rows),)](
                inputs[0], inputs[1], out, rows, columns, waves,
                args.first_wave_depth, args.later_wave_depth, **options)
        return _wide_dag[(triton.cdiv(rows, args.block_rows),)](
            *inputs, out, aux, rows, columns, branches, shared, topology == "disconnected",
            topology == "fanout", args.depth, **options)

    def fail(variant, stage, error):
        failure = {"stage": stage, "error": f"{type(error).__name__}: {error}"}
        failed[variant] = failure
        variants[variant].update(status="failed", failure=failure)

    for variant in VARIANTS:
        stage = "compile_and_launch"
        try:
            outputs[variant] = (torch.empty((rows,), device="cuda"), torch.empty((rows,), device="cuda"))
            kernels[variant] = launch(variant)
            stage = "correctness"
            torch.testing.assert_close(outputs[variant][0], expected, rtol=4e-5, atol=5e-6)
            if auxiliary is not None:
                torch.testing.assert_close(outputs[variant][1], auxiliary, rtol=4e-5, atol=5e-6)
            variants[variant]["correct"] = True
            stage = "diagnostics"
            decisions = _decisions(kernels[variant])
            widths = [item["branches"] for item in decisions if item["decision"] in ("forced", "parallel")]
            wave_groups = _physical_wave_groups(kernels[variant].asm["ttgir"])
            actual_waves = sum(wave_groups)
            if actual_waves != len(widths):
                raise RuntimeError(f"diagnostics describe {len(widths)} waves, but TTGIR contains "
                                   f"{actual_waves} logical waves in {len(wave_groups)} physical regions")
            reason = ("disabled" if variant == "baseline" else
                      "scheduled" if widths else decisions[-1]["reason"] if decisions else "no_legal_wave")
            resources = _resources(kernels[variant], args.resources)
            resources["scheduled_forks"] = len(wave_groups)
            resources["logical_scheduled_forks"] = actual_waves
            variants[variant].update(status="validated", resources=resources,
                                     cost_model_decisions=decisions, scheduled_wave_widths=widths,
                                     physical_wave_groups=wave_groups,
                                     selected_schedule="parallel" if widths else "serial", schedule_reason=reason)
            if variant == "baseline" and "ttg.warp_specialize" in kernels[variant].asm["ttgir"]:
                raise RuntimeError("baseline unexpectedly contains warp specialization")
        except Exception as error:
            fail(variant, stage, error)

    serial_equivalence = None
    if "automatic" not in failed and not variants["automatic"]["scheduled_wave_widths"]:
        if "baseline" in failed:
            serial_equivalence = {"ttgir": None, "ptx": None, "reason": "baseline_failed"}
        else:
            serial_equivalence = {
                "ttgir": _strip_cost_diagnostics(kernels["automatic"].asm["ttgir"]) == kernels["baseline"].asm["ttgir"],
                "ptx": kernels["automatic"].asm["ptx"] == kernels["baseline"].asm["ptx"],
            }
            if not all(serial_equivalence.values()):
                fail("automatic", "serial_code_equivalence", RuntimeError(str(serial_equivalence)))
    if args.dump_ir_dir:
        args.dump_ir_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{topology}-{rows}x{columns}-shared{int(shared)}-{args.dtype}-depth{args.depth}"
        if sequential:
            prefix += f"-first{args.first_wave_depth}-later{args.later_wave_depth}"
        for variant, kernel in kernels.items():
            for stage in ("ttgir", "ptx"):
                (args.dump_ir_dir / f"{prefix}-{variant}.{stage}").write_text(kernel.asm[stage])
    for variant in VARIANTS:
        if variant in failed:
            continue
        try:
            for _ in range(args.warmup):
                launch(variant)
            torch.cuda.synchronize()
        except Exception as error:
            fail(variant, "warmup", error)
    orders = []
    for index in range(args.rounds):
        order = [variant for variant in TIMING_ORDERS[index % len(TIMING_ORDERS)] if variant not in failed]
        orders.append(order)
        for variant in order:
            try:
                p50, p20, p80 = triton.testing.do_bench_cudagraph(
                    lambda: launch(variant), rep=args.rep_ms, quantiles=[0.5, 0.2, 0.8])
                variants[variant]["rounds"].append({"p50_us": p50 * 1000, "p20_us": p20 * 1000, "p80_us": p80 * 1000})
            except Exception as error:
                fail(variant, f"timing_round_{index}", error)
    for variant in VARIANTS:
        if variant not in failed:
            variants[variant].update(status="ok", median_us=statistics.median(
                item["p50_us"] for item in variants[variant]["rounds"]))

    def speedup(variant):
        base, candidate = variants["baseline"]["median_us"], variants[variant]["median_us"]
        return base / candidate if base is not None and candidate is not None else None

    return dict(rows=rows, columns=columns, topology=topology, shared_input=shared,
                dtype=args.dtype, depth=None if sequential else args.depth, block_rows=args.block_rows,
                first_wave_depth=args.first_wave_depth if sequential else None,
                later_wave_depth=args.later_wave_depth if sequential else None,
                block_columns=triton.next_power_of_2(columns), correct=not failed, failed_variants=failed,
                automatic_serial_code_equivalent=serial_equivalence, timing_orders=orders,
                forced_speedup=speedup("forced"), automatic_speedup=speedup("automatic"), **variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, action="append", metavar=("ROWS", "COLUMNS"))
    parser.add_argument("--topology", choices=TOPOLOGIES, action="append")
    parser.add_argument("--input-mode", choices=("independent", "shared", "both"), default="independent")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--block-rows", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--num-warps", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--max-partitions", type=int, default=4)
    parser.add_argument("--min-speedup", type=float, default=1.1)
    parser.add_argument("--no-coalesce", action="store_true",
                        help="keep separate worker dispatches for an otherwise identical parallel-plan ablation")
    parser.add_argument("--depth", type=int, default=3, help="nonlinear chain depth for wide DAGs; waves use their topology depth")
    parser.add_argument("--first-wave-depth", type=int, default=1, help="nonlinear chain depth in the first sequential wave")
    parser.add_argument("--later-wave-depth", type=int, default=1, help="nonlinear chain depth in subsequent sequential waves")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=float, default=30)
    parser.add_argument("--seed", type=int, default=61018)
    parser.add_argument("--resources", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-ir-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("an NVIDIA GPU and a matching FlagTree build are required")
    if (args.rounds < 1 or args.warmup < 0 or args.rep_ms <= 0 or args.depth < 1 or args.max_partitions < 1
            or args.first_wave_depth < 1 or args.later_wave_depth < 1):
        parser.error("rounds, rep-ms, depths and max-partitions must be positive; warmup must be nonnegative")
    shapes = args.shape or DEFAULT_SHAPES
    if any(min(shape) <= 0 for shape in shapes):
        parser.error("shape dimensions must be positive")
    topologies = args.topology or TOPOLOGIES
    modes = {"independent": (False,), "shared": (True,), "both": (False, True)}[args.input_mode]
    report = dict(environment=_environment(), measurement="cuda_graph_replay_per_launch_hot_inputs",
                  topologies=topologies, variants=VARIANTS, warmup_launches_per_variant=args.warmup,
                  rounds=args.rounds, rep_ms=args.rep_ms, seed=args.seed,
                  max_partitions=args.max_partitions, min_speedup=args.min_speedup,
                  coalesce_waves=not args.no_coalesce,
                  query_driver_occupancy=args.resources, completed=False, cases=[])
    if hasattr(os, "sched_getaffinity"):
        report["environment"]["cpu_affinity"] = sorted(os.sched_getaffinity(0))

    def checkpoint():
        # Preserve completed measurements if a later native compilation fails.
        # Replace atomically so interrupted writes never resemble a valid run.
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)

    checkpoint()
    failed = False
    for rows, columns in shapes:
        for topology in topologies:
            # The sequential kernel already shares each joined tensor between
            # its next consumers; input-mode applies only to the wide DAGs.
            for shared in ((False,) if topology.startswith("waves") else modes):
                try:
                    result = _case(rows, columns, topology, shared, args)
                except Exception as error:
                    result = dict(rows=rows, columns=columns, topology=topology, shared_input=shared,
                                  correct=False, error=f"{type(error).__name__}: {error}",
                                  failed_variants={v: {"stage": "case_setup", "error": str(error)} for v in VARIANTS})
                failed |= not result["correct"]
                report["cases"].append(result)
                checkpoint()
    report["completed"] = True
    checkpoint()
    text = json.dumps(report, indent=2)
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


