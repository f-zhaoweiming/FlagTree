"""Executable tile-DAG scheduling through the normal NVIDIA compiler pipeline."""

import json
import math
import os
import re

import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or os.environ.get("TRITON_INTERPRET") == "1",
    reason="requires an NVIDIA GPU and the compiled FlagTree pipeline",
)


@triton.jit
def _tile_branches(X, Y, Out, M: tl.constexpr, N: tl.constexpr,
                   STRIDE_X: tl.constexpr, STRIDE_Y: tl.constexpr,
                   SHARED_INPUT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    x = tl.load(X + rows[:, None] * STRIDE_X + columns[None, :], valid, other=0).to(tl.float32)
    if SHARED_INPUT:
        y = x
    else:
        y = tl.load(Y + rows[:, None] * STRIDE_Y + columns[None, :], valid, other=0).to(tl.float32)
    # Each branch is a serial chain. The only independent expensive operands
    # meet at the final add; neither chain names a high-level operator pattern.
    left = tl.sum(tl.where(valid, tl.exp(-(x * x)), 0.0), axis=1)
    right = tl.sum(tl.where(valid, tl.sqrt(tl.abs(y) + 1.0), 0.0), axis=1)
    result = (left + right) / N
    tl.store(Out + rows, result, rows < M)


@pytest.mark.parametrize("shape", [(1, 17), (7, 129), (129, 513)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shared_input", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_tile_dataflow_correctness(shape, dtype, shared_input, padded):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    rows, columns = shape
    generator = torch.Generator().manual_seed(61017)
    storage_columns = columns + 7 if padded else columns
    x = torch.randn((rows, storage_columns), generator=generator).to(device="cuda", dtype=dtype)[:, :columns]
    y = torch.randn((rows, storage_columns), generator=generator).to(device="cuda", dtype=dtype)[:, :columns]
    source_y = x if shared_input else y
    reference = (torch.exp(-(x.float() * x.float())).sum(1) + torch.sqrt(source_y.float().abs() + 1).sum(1)) / columns
    originals = (x.clone(), y.clone())
    outputs = []
    kernels = []
    for enabled in (False, True):
        # Keep a guard immediately after the output to detect unmasked tail
        # stores from the partial final row tile.
        storage = torch.full((rows + 7,), 12345.0, device="cuda")
        output = storage[:rows]
        kernel = _tile_branches[(triton.cdiv(rows, 4),)](
            x, y, output, rows, columns, x.stride(0), y.stride(0), shared_input,
            BLOCK_M=4, BLOCK_N=triton.next_power_of_2(columns), num_warps=4,
            enable_fp_fusion=False, enable_tile_dataflow=enabled, tile_dataflow_force=enabled,
        )
        torch.testing.assert_close(output, reference, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(storage[rows:], torch.full_like(storage[rows:], 12345.0), rtol=0, atol=0)
        outputs.append(output)
        kernels.append(kernel)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=3e-5, atol=3e-6)
    for actual, expected in zip((x, y), originals):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    baseline, scheduled = kernels
    assert "tle.tile_dataflow" not in baseline.asm["ttgir"]
    assert "ttg.warp_specialize" not in baseline.asm["ttgir"]
    assert "tle.tile_dataflow" in scheduled.asm["ttgir"]
    assert "ttg.warp_specialize" in scheduled.asm["ttgir"]
    assert "ttg.local_store" in scheduled.asm["ttgir"]
    assert "ttg.local_load" in scheduled.asm["ttgir"]
    assert baseline.metadata.num_warps == 4
    assert 8 <= scheduled.metadata.num_warps <= 32
    assert scheduled.metadata.shared > 0


def test_tile_dataflow_nondefault_stream_and_repeated_arguments():
    stream = torch.cuda.Stream()
    retained = []
    with torch.cuda.stream(stream):
        for columns in (33, 129, 513):
            x = torch.randn((7, columns), device="cuda")
            y = torch.randn_like(x)
            output = torch.empty((7,), device="cuda")
            kernel = _tile_branches[(2,)](
                x, y, output, 7, columns, columns, columns, False,
                BLOCK_M=4, BLOCK_N=triton.next_power_of_2(columns), num_warps=4,
                enable_fp_fusion=False, enable_tile_dataflow=True, tile_dataflow_force=True,
            )
            expected = (torch.exp(-(x * x)).sum(1) + torch.sqrt(y.abs() + 1).sum(1)) / columns
            retained.append((output, expected, kernel))
    stream.synchronize()
    for actual, expected, kernel in retained:
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        assert "tle.tile_dataflow" in kernel.asm["ttgir"]


@pytest.mark.parametrize("num_warps", [8, 16])
def test_tile_dataflow_larger_warp_groups(num_warps):
    x = torch.randn((7, 129), device="cuda")
    output = torch.empty((7,), device="cuda")
    try:
        kernel = _tile_branches[(2,)](
            x, x, output, 7, 129, 129, 129, True, BLOCK_M=4, BLOCK_N=256,
            num_warps=num_warps, enable_fp_fusion=False, enable_tile_dataflow=True, tile_dataflow_force=True,
        )
    except OutOfResources as error:
        pytest.skip(f"device cannot hold both {num_warps}-warp groups: {error}")
    expected = (torch.exp(-(x * x)).sum(1) + torch.sqrt(x.abs() + 1).sum(1)) / 129
    torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
    assert "tle.tile_dataflow" in kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in kernel.asm["ttgir"]
    assert kernel.metadata.num_warps == 2 * num_warps


@triton.jit
def _aliased_write_between_reads(X, Alias, Out, M: tl.constexpr, N: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    before = tl.load(X + offsets, valid, other=0)
    left = tl.sum(tl.where(valid, tl.exp(-(before * before)), 0.0), axis=1)
    # The store's pointer is a different SSA argument. Runtime aliasing must
    # still preserve both the earlier read and the later read of this memory.
    tl.store(Alias + offsets, 3.0, valid)
    after = tl.load(X + offsets, valid, other=0)
    right = tl.sum(tl.where(valid, tl.sqrt(tl.abs(after) + 1.0), 0.0), axis=1)
    tl.store(Out + rows, (left + right) / N, rows < M)


def test_tile_dataflow_runtime_alias_hazard():
    initial = torch.randn((7, 129), device="cuda")
    expected = torch.exp(-(initial * initial)).mean(1) + 2.0
    outputs = []
    for enabled in (False, True):
        x = initial.clone()
        output = torch.empty((7,), device="cuda")
        kernel = _aliased_write_between_reads[(2,)](
            x, x, output, 7, 129, BLOCK_M=4, BLOCK_N=256, num_warps=4,
            enable_fp_fusion=False, enable_tile_dataflow=enabled, tile_dataflow_force=enabled,
        )
        torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(x, torch.full_like(x, 3.0), rtol=0, atol=0)
        assert "ttg.warp_specialize" not in kernel.asm["ttgir"]
        outputs.append(output)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-5, atol=3e-6)


@triton.jit
def _data_dependent_bool_capture(X, Y, Out, M: tl.constexpr, N: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(X + offsets, valid, other=0)
    y = tl.load(Y + offsets, valid, other=0)
    # This shared mask depends on loaded floating-point data. The worker must
    # receive it through shared memory rather than recomputing index arithmetic.
    positive = x > 0
    left = tl.sum(tl.where(positive, tl.exp(-(x * x)), 0.0), axis=1)
    right = tl.sum(tl.where(positive, tl.sqrt(tl.abs(y) + 1.0), 0.0), axis=1)
    tl.store(Out + rows, (left + right) / N, rows < M)


def test_tile_dataflow_boolean_capture():
    x = torch.randn((7, 129), device="cuda")
    y = torch.randn_like(x)
    positive = x > 0
    expected = (torch.where(positive, torch.exp(-(x * x)), 0).sum(1)
                + torch.where(positive, torch.sqrt(y.abs() + 1), 0).sum(1)) / 129
    for enabled in (False, True):
        output = torch.empty((7,), device="cuda")
        kernel = _data_dependent_bool_capture[(2,)](
            x, y, output, 7, 129, BLOCK_M=4, BLOCK_N=256, num_warps=4,
            enable_fp_fusion=False, enable_tile_dataflow=enabled, tile_dataflow_force=enabled,
        )
        torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
        if enabled:
            ttgir = kernel.asm["ttgir"]
            assert "tle.tile_dataflow" in ttgir
            assert "ttg.warp_specialize" in ttgir
            assert "arith.extui" in ttgir and "arith.trunci" in ttgir
            assert "xi8," in ttgir


@triton.jit
def _boolean_branch_result(X, Y, Out, M: tl.constexpr, N: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(X + offsets, valid, other=0)
    y = tl.load(Y + offsets, valid, other=0)
    left = x * x > 0.5
    right = tl.sqrt(tl.abs(y) + 1.0) > 1.5
    tl.store(Out + offsets, left ^ right, valid)


def test_tile_dataflow_boolean_branch_result():
    indices = torch.arange(7 * 129, device="cuda").reshape(7, 129)
    # Exact binary fractions stay well away from comparison thresholds, making
    # the boolean reference insensitive to approximate sqrt rounding.
    x = (indices.remainder(19).float() - 9) / 4
    y = (indices.remainder(23).float() - 11) / 2
    expected = (x * x > 0.5) ^ (torch.sqrt(y.abs() + 1) > 1.5)
    for enabled in (False, True):
        output = torch.empty((7, 129), device="cuda", dtype=torch.bool)
        kernel = _boolean_branch_result[(2,)](
            x, y, output, 7, 129, BLOCK_M=4, BLOCK_N=256, num_warps=4,
            enable_fp_fusion=False, enable_tile_dataflow=enabled, tile_dataflow_force=enabled,
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if enabled:
            ttgir = kernel.asm["ttgir"]
            assert "tle.tile_dataflow" in ttgir
            assert "ttg.warp_specialize" in ttgir
            assert "arith.extui" in ttgir and "arith.trunci" in ttgir
            assert "xi8," in ttgir


def _without_cost_diagnostics(ttgir):
    match = re.search(r'"?tle\.tile_dataflow\.cost_model"?\s*=\s*"(?:\\.|[^"\\])*"', ttgir)
    if match is None:
        return ttgir
    before, after = ttgir[:match.start()], ttgir[match.end():]
    if before.endswith(", "):
        before = before[:-2]
    elif after.startswith(", "):
        after = after[2:]
    return before + after


@pytest.mark.parametrize("rows,columns,shared_input,expected_parallel", [
    (32, 128, False, False),
    (128, 1024, False, True),
    (128, 1024, True, False),
    (129, 513, False, False),
])
def test_tile_dataflow_automatic_decision(rows, columns, shared_input, expected_parallel):
    x = torch.randn((rows, columns), device="cuda")
    y = torch.randn_like(x)
    other = x if shared_input else y
    expected = (torch.exp(-(x * x)).sum(1) + torch.sqrt(other.abs() + 1).sum(1)) / columns
    kernels = []
    outputs = []
    for enabled in (False, True):
        output = torch.empty((rows,), device="cuda")
        kernel = _tile_branches[(triton.cdiv(rows, 4),)](
            x, y, output, rows, columns, columns, columns, shared_input,
            BLOCK_M=4, BLOCK_N=triton.next_power_of_2(columns), num_warps=4,
            enable_fp_fusion=False, enable_tile_dataflow=enabled,
        )
        torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
        kernels.append(kernel)
        outputs.append(output)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=3e-5, atol=3e-6)
    baseline, automatic = kernels
    assert ("ttg.warp_specialize" in automatic.asm["ttgir"]) == expected_parallel
    decisions = json.loads(automatic.metadata.tile_dataflow_cost_model)
    assert decisions
    for decision in decisions:
        assert decision["decision"] in ("serial", "parallel")
        assert decision["reason"]
        for field in ("serial_cost", "parallel_cost", "sync_cost", "shared_cost", "occupancy_cost", "recompute_cost"):
            assert math.isfinite(decision[field]) and decision[field] >= 0, (field, decision)
        for field in ("baseline_shared_bytes", "parallel_shared_bytes", "baseline_registers", "parallel_registers",
                      "baseline_active_ctas", "parallel_active_ctas"):
            assert decision[field] >= 0, (field, decision)
        assert math.isfinite(decision["predicted_speedup"]) and decision["predicted_speedup"] > 0
    if "ttg.warp_specialize" in automatic.asm["ttgir"]:
        assert any(decision["decision"] == "parallel" for decision in decisions)
        assert automatic.metadata.num_warps >= 8
    else:
        assert all(decision["decision"] == "serial" for decision in decisions)
        assert _without_cost_diagnostics(automatic.asm["ttgir"]) == baseline.asm["ttgir"]
        assert automatic.asm["ptx"] == baseline.asm["ptx"]
        assert automatic.metadata.num_warps == baseline.metadata.num_warps
        assert automatic.metadata.shared == baseline.metadata.shared


def test_tile_dataflow_force_bypasses_only_profitability():
    x = torch.randn((128, 1024), device="cuda")
    y = torch.randn_like(x)
    expected = (torch.exp(-(x * x)).sum(1) + torch.sqrt(y.abs() + 1).sum(1)) / 1024
    kernels = []
    for enabled, force in ((False, False), (True, False), (True, True)):
        output = torch.empty((128,), device="cuda")
        kernel = _tile_branches[(32,)](
            x, y, output, 128, 1024, 1024, 1024, False,
            BLOCK_M=4, BLOCK_N=1024, num_warps=4, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_force=force,
            tile_dataflow_min_speedup=1000000.0,
        )
        torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
        kernels.append(kernel)
    baseline, rejected, forced = kernels
    assert "ttg.warp_specialize" not in rejected.asm["ttgir"]
    assert _without_cost_diagnostics(rejected.asm["ttgir"]) == baseline.asm["ttgir"]
    assert rejected.asm["ptx"] == baseline.asm["ptx"]
    rejected_decisions = json.loads(rejected.metadata.tile_dataflow_cost_model)
    assert rejected_decisions and all(item["decision"] == "serial" for item in rejected_decisions)
    assert "ttg.warp_specialize" in forced.asm["ttgir"]
    assert any(item["decision"] == "forced" for item in json.loads(forced.metadata.tile_dataflow_cost_model))
    # The separate alias-hazard regression runs with force enabled and requires
    # no fork, so this override cannot silently become a legality override.


@triton.jit
def _reverse_worker_branches(X, Y, Scratch, Out, M: tl.constexpr, N: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(X + offsets, valid, other=0)
    # This observable consumer keeps x outside either candidate branch. Moving
    # the right branch to a worker would therefore stage an entire float tile.
    tl.store(Scratch + offsets, x, valid)
    y = tl.load(Y + offsets, valid, other=0)
    left_value = tl.exp(-(y * y))
    left_value = tl.exp(-(left_value * left_value))
    right_value = tl.sqrt(tl.abs(x) + 1.0)
    right_value = tl.sqrt(right_value + 1.0)
    left = tl.sum(tl.where(valid, left_value, 0.0), axis=1)
    right = tl.sum(tl.where(valid, right_value, 0.0), axis=1)
    tl.store(Out + rows, (left + right) / N, rows < M)


def test_tile_dataflow_automatic_reverses_worker_to_reduce_transfer():
    rows, columns = 127, 1024
    generator = torch.Generator().manual_seed(61018)
    x = torch.randn((rows, columns), generator=generator).cuda()
    y = torch.randn((rows, columns), generator=generator).cuda()
    originals = (x.clone(), y.clone())
    first_exp = torch.exp(-(y * y))
    expected = (torch.exp(-(first_exp * first_exp)).sum(1)
                + torch.sqrt(torch.sqrt(x.abs() + 1) + 1).sum(1)) / columns
    kernels, outputs = [], []
    for enabled in (False, True):
        scratch_storage = torch.full((rows * columns + 7,), 12345.0, device="cuda")
        scratch = scratch_storage[:rows * columns].view(rows, columns)
        output_storage = torch.full((rows + 7,), 12345.0, device="cuda")
        output = output_storage[:rows]
        kernel = _reverse_worker_branches[(triton.cdiv(rows, 4),)](
            x, y, scratch, output, rows, columns, BLOCK_M=4, BLOCK_N=1024,
            num_warps=4, enable_fp_fusion=False, enable_tile_dataflow=enabled,
            tile_dataflow_min_speedup=1.0,
        )
        torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(scratch, x, rtol=0, atol=0)
        for guard in (scratch_storage[rows * columns:], output_storage[rows:]):
            torch.testing.assert_close(guard, torch.full_like(guard, 12345.0), rtol=0, atol=0)
        kernels.append(kernel)
        outputs.append(output)
    for actual, original in zip((x, y), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-5, atol=3e-6)
    baseline, automatic = kernels
    assert "ttg.warp_specialize" not in baseline.asm["ttgir"]
    assert "ttg.warp_specialize" in automatic.asm["ttgir"]
    decisions = json.loads(automatic.metadata.tile_dataflow_cost_model)
    chosen = [item for item in decisions if item["decision"] == "parallel"]
    assert len(chosen) == 1, decisions
    assert chosen[0]["worker_operand"] == 0, decisions
    # Four f32 row results are written and read; no full input tile crosses the
    # worker boundary. The original operand order would require 32 KiB more.
    assert chosen[0]["transfer_bytes"] == 2 * 4 * 4, decisions
    assert automatic.metadata.num_warps == 8
