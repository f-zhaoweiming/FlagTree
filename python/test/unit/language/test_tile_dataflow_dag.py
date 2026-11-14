"""Multiway tile-DAG scheduling, including disconnected and dependent waves."""

import json
import math
import os
import re

import pytest
import torch

import triton
import triton.language as tl


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or os.environ.get("TRITON_INTERPRET") == "1",
    reason="requires an NVIDIA GPU and the compiled FlagTree pipeline",
)


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
                    WAVES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(A + offsets, mask, other=0).to(tl.float32)
    y = tl.load(B + offsets, mask, other=0).to(tl.float32)
    value = tl.exp(-(x * x)) + tl.sqrt(tl.abs(y) + 1.0)
    # The joined tensor is a true shared ancestor of the next fork. Its
    # consumers cannot be placed in the previous wave or run before its join.
    for _ in tl.static_range(WAVES - 1):
        left = tl.exp(-(value * value))
        right = tl.sqrt(tl.abs(value) + 1.0)
        value = left + right
    result = tl.sum(tl.where(mask, value, 0.0), 1) / N
    tl.store(Out + rows, result, rows < M)


def _wave_regions(ttgir):
    # Anchor on operation lines: the cost JSON can itself name a
    # "ttg.warp_specialize" insertion point, which is not another operation.
    starts = list(re.finditer(r"(?m)^([ \t]*)%[^\n]* = ttg\.warp_specialize\(", ttgir))
    regions = []
    for start in starts:
        indent = start[1]
        tail = ttgir[start.start():]
        end = re.search(r"(?m)^" + re.escape(indent) + r"\} :[^\n]*", tail)
        assert end is not None, tail
        text = tail[:end.end()]
        parts = list(re.finditer(r"(?m)^" + re.escape(indent) + r"(?:default \{|partition\d+\()", text))
        assert len(parts) >= 2, text
        regions.append((text, [text[part.start():parts[index + 1].start() if index + 1 < len(parts) else len(text)]
                               for index, part in enumerate(parts)]))
    return regions


def _logical_waves(text):
    encoded = re.search(r"tle\.tile_dataflow\.logical_waves = (\d+) : i32", text.splitlines()[0])
    count = int(encoded[1]) if encoded else 1
    assert count >= 1, text
    return count


def _wave_costs(text):
    header = text.splitlines()[0]
    encoded = re.search(r"tle\.tile_dataflow\.wave_costs = \[([^\]]+)\]", header)
    if encoded:
        components = re.findall(r"array<f64: ([^>]+)>", encoded[1])
    else:
        encoded = re.search(r"tle\.tile_dataflow\.costs = array<f64: ([^>]+)>", header)
        assert encoded is not None, text
        components = [encoded[1]]
    assert len(components) == _logical_waves(text), text
    return [[float(value) for value in item.split(",")] for item in components]


def _schedule(kernel, widths, default_warps=4):
    regions = _wave_regions(kernel.asm["ttgir"])
    # Force may additionally schedule independent indexing or small output
    # arithmetic. They cannot satisfy the required nonlinear DAG waves.
    main = [(text, parts) for text, parts in regions if re.search(r"\bmath\.", text)]
    logical_widths = [len(parts) for text, parts in main for _ in range(_logical_waves(text))]
    assert logical_widths == widths, kernel.asm["ttgir"]
    for text, parts in main:
        assert all(re.search(r"\b(?:math\.|arith\.(?:addf|subf|mulf|divf))", part) for part in parts), text
    for text, parts in regions:
        if not re.search(r"\bmath\.", text):
            assert "tt.reduce" not in text and not re.search(r"\btt\.load\b", text), text
    assert kernel.metadata.num_warps == default_warps * max(len(parts) for _, parts in regions)
    decisions = json.loads(kernel.metadata.tile_dataflow_cost_model)
    chosen = [item for item in decisions if item["decision"] in ("forced", "parallel")]
    assert len(chosen) == sum(_logical_waves(text) for text, _ in regions), decisions
    assert sorted(item["branches"] for item in chosen) == sorted(
        len(parts) for text, parts in regions for _ in range(_logical_waves(text))), decisions
    assert len({item["wave"] for item in chosen}) == len(chosen), decisions
    unmatched, matched = list(chosen), []
    for text, parts in main:
        for components in _wave_costs(text):
            assert len(components) >= 6, components
            assert 0 <= components[1] <= components[0] and components[2] >= 0, components
            assert components[3] >= 0 and components[4] == len(parts), components
            candidates = [item for item in unmatched if item["branches"] == len(parts)
                          and math.isclose(sum(item["branch_costs"]), components[0], rel_tol=1e-6, abs_tol=1e-6)]
            assert candidates, (components, unmatched)
            # A merged candidate records the marginal handoff traffic after
            # reusing its predecessor; wave_costs retain the original wave.
            assert candidates[0]["transfer_bytes"] >= 0, candidates[0]
            matched.append(candidates[0])
            unmatched.remove(candidates[0])
    return matched


def _check_persistent_handoffs(kernel, retain_worker_results=False):
    regions = [(text, parts) for text, parts in _wave_regions(kernel.asm["ttgir"])
               if re.search(r"\bmath\.", text)]
    merged = [(text, parts) for text, parts in regions if _logical_waves(text) > 1]
    assert merged, kernel.asm["ttgir"]
    for text, parts in merged:
        boundaries = _logical_waves(text) - 1
        for part in parts:
            barriers = len(re.findall(r"\bttng\.wait_barrier_named\b", part))
            assert boundaries <= barriers <= 2 * boundaries, text
        if retain_worker_results:
            for worker in parts[1:]:
                # The worker's completed tile remains an SSA value for the
                # next join instead of being loaded back from shared memory.
                stores = list(re.finditer(r"ttg\.local_store\s+(%[a-zA-Z0-9_]+),", worker))
                retained = sum(bool(re.search(re.escape(store[1]) + r"(?![a-zA-Z0-9_])", worker[store.end():]))
                               for store in stores)
                assert retained >= boundaries, text


def _guarded_output(rows):
    storage = torch.full((rows + 11,), 12345.0, device="cuda")
    return storage, storage[:rows]


def _check_output(storage, expected):
    rows = expected.numel()
    torch.testing.assert_close(storage[:rows], expected, rtol=4e-5, atol=5e-6)
    torch.testing.assert_close(storage[rows:], torch.full_like(storage[rows:], 12345.0), rtol=0, atol=0)


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


@pytest.mark.parametrize("branches", [3, 4])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tile_dataflow_multiway_binary_join_tree(branches, shared, dtype):
    rows, columns = 7, 129
    inputs = [torch.randn((rows, columns), device="cuda", dtype=dtype) for _ in range(4)]
    originals = [value.clone() for value in inputs]
    results = _wide_reference(inputs, branches, shared, 1)
    expected = (results[0] + results[1]) + (results[2] if branches == 3 else results[2] + results[3])
    outputs = []
    for enabled in (False, True):
        storage, output = _guarded_output(rows)
        kernel = _wide_dag[(triton.cdiv(rows, 4),)](
            *inputs, output, output, rows, columns, branches, shared, False, False, 1,
            BLOCK_M=4, BLOCK_N=256, num_warps=4, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_force=enabled, tile_dataflow_max_partitions=4,
        )
        _check_output(storage, expected)
        outputs.append(output)
        if enabled:
            _schedule(kernel, [branches])
        else:
            assert "ttg.warp_specialize" not in kernel.asm["ttgir"]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=4e-5, atol=5e-6)
    for actual, original in zip(inputs, originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_tile_dataflow_four_groups_at_cta_warp_limit():
    rows, columns = 7, 129
    inputs = [torch.randn((rows, columns), device="cuda") for _ in range(4)]
    originals = [value.clone() for value in inputs]
    values = _wide_reference(inputs, 4, False, 1)
    expected = (values[0] + values[1]) + (values[2] + values[3])
    outputs = []
    for enabled in (False, True):
        storage, output = _guarded_output(rows)
        kernel = _wide_dag[(triton.cdiv(rows, 4),)](
            *inputs, output, output, rows, columns, 4, False, False, False, 1,
            BLOCK_M=4, BLOCK_N=256, num_warps=8, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_force=enabled, tile_dataflow_max_partitions=4,
        )
        _check_output(storage, expected)
        outputs.append(output)
        if enabled:
            _schedule(kernel, [4], default_warps=8)
            assert kernel.metadata.num_warps == 32
            assert all(int(warps) == 8 for warps in re.findall(r"num_warps\((\d+)\)", kernel.asm["ttgir"]))
        else:
            assert kernel.metadata.num_warps == 8
            assert "ttg.warp_specialize" not in kernel.asm["ttgir"]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=4e-5, atol=5e-6)
    for actual, original in zip(inputs, originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("disconnected", [False, True])
def test_tile_dataflow_disconnected_outputs_and_fanout(shared, disconnected):
    rows, columns = 11, 65
    inputs = [torch.randn((rows, columns), device="cuda") for _ in range(4)]
    results = _wide_reference(inputs, 4, shared, 1)
    expected = results[0] + results[1]
    auxiliary = results[2] + results[3] if disconnected else results[0]
    if not disconnected:
        expected = expected + (results[2] + results[3])
    for enabled in (False, True):
        storage, output = _guarded_output(rows)
        auxiliary_storage, auxiliary_output = _guarded_output(rows)
        kernel = _wide_dag[(triton.cdiv(rows, 4),)](
            *inputs, output, auxiliary_output, rows, columns, 4, shared, disconnected, not disconnected, 1,
            BLOCK_M=4, BLOCK_N=128, num_warps=4, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_force=enabled, tile_dataflow_max_partitions=4,
        )
        _check_output(storage, expected)
        _check_output(auxiliary_storage, auxiliary)
        if enabled:
            _schedule(kernel, [4])


@pytest.mark.parametrize("waves", [2, 3])
@pytest.mark.parametrize("columns", [65, 257])
def test_tile_dataflow_dependent_fork_waves(waves, columns):
    rows = 9
    x = torch.randn((rows, columns), device="cuda")
    y = torch.randn_like(x)
    value = torch.exp(-(x * x)) + torch.sqrt(y.abs() + 1)
    for _ in range(waves - 1):
        value = torch.exp(-(value * value)) + torch.sqrt(value.abs() + 1)
    expected = value.mean(1)
    for enabled in (False, True):
        storage, output = _guarded_output(rows)
        kernel = _sequential_dag[(triton.cdiv(rows, 4),)](
            x, y, output, rows, columns, waves, BLOCK_M=4,
            BLOCK_N=triton.next_power_of_2(columns), num_warps=4, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_force=enabled, tile_dataflow_max_partitions=4,
        )
        _check_output(storage, expected)
        if enabled:
            _schedule(kernel, [2] * waves)
            _check_persistent_handoffs(kernel)


@pytest.mark.parametrize("branches", [3, 4])
def test_tile_dataflow_multiway_automatic_numerics(branches):
    rows, columns = 31, 1024
    inputs = [torch.randn((rows, columns), device="cuda") for _ in range(4)]
    results = _wide_reference(inputs, branches, False, 3)
    expected = (results[0] + results[1]) + (results[2] if branches == 3 else results[2] + results[3])
    storage, output = _guarded_output(rows)
    kernel = _wide_dag[(triton.cdiv(rows, 4),)](
        *inputs, output, output, rows, columns, branches, False, False, False, 3,
        BLOCK_M=4, BLOCK_N=1024, num_warps=4, enable_fp_fusion=False,
        enable_tile_dataflow=True, tile_dataflow_max_partitions=4, tile_dataflow_min_speedup=1.0,
    )
    _check_output(storage, expected)
    decisions = json.loads(kernel.metadata.tile_dataflow_cost_model)
    assert decisions
    assert all(item["decision"] in ("serial", "parallel") for item in decisions)
    chosen = [item for item in decisions if item["decision"] == "parallel"]
    if chosen:
        assert max(item["branches"] for item in chosen) >= 3, decisions
        assert kernel.metadata.num_warps >= 12
    else:
        assert "ttg.warp_specialize" not in kernel.asm["ttgir"]


@triton.jit
def _alias_boundaries(X, Alias, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    before = tl.load(X + offsets, mask, other=0)
    first = tl.exp(-(before * before))
    tl.store(Alias + offsets, 2.0, mask)
    middle = tl.load(X + offsets, mask, other=0)
    second = tl.sqrt(tl.abs(middle) + 1.0)
    tl.store(Alias + offsets, 3.0, mask)
    after = tl.load(X + offsets, mask, other=0)
    third = after / (1.0 + after * after)
    tl.store(Out + offsets, (first + second) + third, mask)


def test_tile_dataflow_global_memory_dependencies():
    columns = 137
    initial = torch.randn((columns,), device="cuda")
    expected = torch.exp(-(initial * initial)) + 3.0 ** 0.5 + 0.3
    for enabled in (False, True):
        x = initial.clone()
        storage, output = _guarded_output(columns)
        kernel = _alias_boundaries[(1,)](
            x, x, output, columns, BLOCK=256, num_warps=4, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_force=enabled, tile_dataflow_max_partitions=4,
        )
        _check_output(storage, expected)
        torch.testing.assert_close(x, torch.full_like(x, 3.0), rtol=0, atol=0)
        assert "ttg.warp_specialize" not in kernel.asm["ttgir"]


@triton.jit
def _symmetric_dag(A, B, C, D, Out, M: tl.constexpr, N: tl.constexpr,
                   BRANCHES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    a = tl.load(A + offsets, mask, other=0)
    b = tl.load(B + offsets, mask, other=0)
    if BRANCHES == 4:
        c = tl.load(C + offsets, mask, other=0)
        d = tl.load(D + offsets, mask, other=0)
    for _ in tl.static_range(3):
        a = tl.exp(-(a * a))
        b = tl.exp(-(b * b))
        if BRANCHES == 4:
            c = tl.exp(-(c * c))
            d = tl.exp(-(d * d))
    ra = tl.sum(tl.where(mask, a, 0.0), 1)
    rb = tl.sum(tl.where(mask, b, 0.0), 1)
    result = ra + rb
    if BRANCHES == 4:
        rc = tl.sum(tl.where(mask, c, 0.0), 1)
        rd = tl.sum(tl.where(mask, d, 0.0), 1)
        result = result + (rc + rd)
    tl.store(Out + rows, result / N, rows < M)


def test_tile_dataflow_register_pressure_uses_worker_peak():
    rows, columns = 7, 1024
    inputs = [torch.randn((rows, columns), device="cuda") for _ in range(4)]
    values = inputs
    for _ in range(3):
        values = [torch.exp(-(value * value)) for value in values]
    costs = []
    for branches in (2, 4):
        storage, output = _guarded_output(rows)
        kernel = _symmetric_dag[(triton.cdiv(rows, 4),)](
            *inputs, output, rows, columns, branches, BLOCK_M=4, BLOCK_N=1024,
            num_warps=4, enable_fp_fusion=False, enable_tile_dataflow=True,
            tile_dataflow_force=True, tile_dataflow_max_partitions=4,
        )
        expected = values[0].mean(1) + values[1].mean(1)
        if branches == 4:
            expected = expected + (values[2].mean(1) + values[3].mean(1))
        _check_output(storage, expected)
        costs.append(_schedule(kernel, [branches])[0])
    # Additional workers have separate register files. Four row results may
    # add a few join registers, but must not add another whole live tile.
    assert costs[1]["parallel_registers"] <= costs[0]["parallel_registers"] + 4, costs


@triton.jit
def _heavy_sequential_dag(A, B, Out, M: tl.constexpr, N: tl.constexpr,
                          WAVES: tl.constexpr, FIRST_DEPTH: tl.constexpr, LATER_DEPTH: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(A + offsets, mask, other=0)
    y = tl.load(B + offsets, mask, other=0)
    left, right = x, y
    # Retain input dependence through long chains instead of converging toward
    # an input-independent fixed point that could hide synchronization errors.
    for _ in tl.static_range(FIRST_DEPTH):
        left = tl.exp(-(left * left)) + 0.03125 * x
        right = tl.sqrt(tl.abs(right) + 1.0) + 0.0625 * y
    value = left + right
    for _ in tl.static_range(WAVES - 1):
        left, right = value, value
        for _ in tl.static_range(LATER_DEPTH):
            left = tl.exp(-(left * left)) + 0.03125 * value
            right = tl.sqrt(tl.abs(right) + 1.0) + 0.0625 * value
        value = left + right
    tl.store(Out + rows, tl.sum(tl.where(mask, value, 0.0), 1) / N, rows < M)


@pytest.mark.parametrize("waves", [2, 3])
@pytest.mark.parametrize("later_depth", [6, 12])
def test_tile_dataflow_automatic_multiple_profitable_waves(waves, later_depth):
    rows, columns = 15, 512
    first_depth = 40 * (waves - 1) * (later_depth // 6)
    x = torch.randn((rows, columns), device="cuda")
    y = torch.randn_like(x)
    left, right = x, y
    for _ in range(first_depth):
        left = torch.exp(-(left * left)) + 0.03125 * x
        right = torch.sqrt(right.abs() + 1) + 0.0625 * y
    value = left + right
    for _ in range(waves - 1):
        left, right = value, value
        for _ in range(later_depth):
            left = torch.exp(-(left * left)) + 0.03125 * value
            right = torch.sqrt(right.abs() + 1) + 0.0625 * value
        value = left + right
    for enabled in (False, True):
        storage, output = _guarded_output(rows)
        kernel = _heavy_sequential_dag[(triton.cdiv(rows, 4),)](
            x, y, output, rows, columns, waves, first_depth, later_depth,
            BLOCK_M=4, BLOCK_N=512, num_warps=4, enable_fp_fusion=False,
            enable_tile_dataflow=enabled, tile_dataflow_max_partitions=4,
            tile_dataflow_min_speedup=1.0,
        )
        _check_output(storage, value.mean(1))
        if enabled:
            # Persistent workers reduce the handoff cost enough to retain
            # the shorter tails as real parallel waves as well.
            chosen = _schedule(kernel, [2] * waves)
            _check_persistent_handoffs(kernel)
            assert all(item["decision"] == "parallel" for item in chosen), chosen
            assert all(item["marginal_saving"] >= item["required_saving"] for item in chosen), chosen


@triton.jit
def _buffer_overwrite_dag(A, B, Out, M: tl.constexpr, N: tl.constexpr,
                          WAVES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    x = tl.load(A + offsets, mask, other=0).to(tl.float32)
    y = tl.load(B + offsets, mask, other=0).to(tl.float32)
    left = tl.exp(-(x * x)) + 0.03125 * x
    right = tl.sqrt(tl.abs(y) + 1.0) + 0.0625 * y
    value = left + right
    for _ in tl.static_range(WAVES - 1):
        left = tl.exp(-(value * value)) + 0.03125 * value
        right = tl.sqrt(tl.abs(value) + 1.0) + 0.0625 * value
        value = left + right
    # Compare every element instead of a reduction that could hide corrupt
    # lanes through cancellation after a staged input is overwritten.
    tl.store(Out + offsets, value, mask)


@pytest.mark.parametrize("waves", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tile_dataflow_private_capture_buffer_overwrite(waves, dtype):
    rows, columns = 9, 513
    for iteration in range(3):
        x = torch.randn((rows, columns), device="cuda", dtype=dtype) + 0.125 * iteration
        y = torch.randn_like(x) - 0.25 * iteration
        originals = [x.clone(), y.clone()]
        xf, yf = x.float(), y.float()
        expected = torch.exp(-(xf * xf)) + 0.03125 * xf
        expected = expected + (torch.sqrt(yf.abs() + 1) + 0.0625 * yf)
        for _ in range(waves - 1):
            left = torch.exp(-(expected * expected)) + 0.03125 * expected
            right = torch.sqrt(expected.abs() + 1) + 0.0625 * expected
            expected = left + right
        outputs = []
        for enabled in (False, True):
            storage = torch.full((rows * columns + 18,), 12345.0, device="cuda")
            output = storage[7:-11].view(rows, columns)
            kernel = _buffer_overwrite_dag[(triton.cdiv(rows, 4),)](
                x, y, output, rows, columns, waves, BLOCK_M=4, BLOCK_N=1024,
                num_warps=4, enable_fp_fusion=False, enable_tile_dataflow=enabled,
                tile_dataflow_force=enabled, tile_dataflow_max_partitions=2,
            )
            torch.testing.assert_close(output, expected, rtol=4e-5, atol=5e-6)
            for guard in (storage[:7], storage[-11:]):
                torch.testing.assert_close(guard, torch.full_like(guard, 12345.0), rtol=0, atol=0)
            outputs.append(output)
            if enabled:
                _schedule(kernel, [2] * waves)
                _check_persistent_handoffs(kernel, retain_worker_results=True)
                main = [(text, parts) for text, parts in _wave_regions(kernel.asm["ttgir"])
                        if re.search(r"\bmath\.", text)]
                for text, parts in main[1:]:
                    if _logical_waves(text) > 1:
                        continue
                    assert len(parts) == 2, text
                    worker = parts[1]
                    loads = re.findall(r"ttg\.local_load\s+(%[a-zA-Z0-9_]+)", worker)
                    stores = re.findall(r"ttg\.local_store\s+%[a-zA-Z0-9_]+,\s*(%[a-zA-Z0-9_]+)", worker)
                    assert len(stores) == 1 and stores[0] in loads, text
            else:
                assert "ttg.warp_specialize" not in kernel.asm["ttgir"]
        torch.testing.assert_close(outputs[0], outputs[1], rtol=4e-5, atol=5e-6)
        for actual, original in zip((x, y), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)


@triton.jit
def _persistent_broadcast_dag(A, B, C, D, Out, M: tl.constexpr, N: tl.constexpr,
                              BRANCHES: tl.constexpr, WAVES: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    offsets = rows[:, None] * N + columns[None, :]
    a = tl.load(A + offsets, mask, other=0).to(tl.float32)
    b = tl.load(B + offsets, mask, other=0).to(tl.float32)
    c = tl.load(C + offsets, mask, other=0).to(tl.float32)
    d = tl.load(D + offsets, mask, other=0).to(tl.float32)
    for _ in tl.static_range(WAVES):
        initial_a, initial_b, initial_c, initial_d = a, b, c, d
        # Unequal chains let an early worker reach its output store while
        # another worker is still reading the common previous default tile.
        for _ in tl.static_range(4):
            a = tl.exp(-(a * a)) + 0.03125 * initial_a
        b = tl.sqrt(tl.abs(b) + 1.0) + 0.0625 * initial_b
        for _ in tl.static_range(2):
            c = tl.sqrt(tl.abs(c) + 2.0) + 0.125 * initial_c
        if BRANCHES == 4:
            for _ in tl.static_range(3):
                d = tl.exp(-(d * d + 0.5)) + 0.0625 * initial_d
            value = (a + b) + (c + d)
        else:
            value = (a + b) + c
        a, b, c, d = value, value, value, value
    tl.store(Out + offsets, value, mask)


@pytest.mark.parametrize("branches", [3, 4])
@pytest.mark.parametrize("waves", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tile_dataflow_persistent_broadcast_handoffs(branches, waves, dtype):
    rows, columns = 9, 129
    for iteration in range(3):
        inputs = [torch.randn((rows, columns), device="cuda", dtype=dtype) + 0.125 * (iteration - index)
                  for index in range(4)]
        originals = [value.clone() for value in inputs]
        a, b, c, d = (value.float() for value in inputs)
        for _ in range(waves):
            initial_a, initial_b, initial_c, initial_d = a, b, c, d
            for _ in range(4):
                a = torch.exp(-(a * a)) + 0.03125 * initial_a
            b = torch.sqrt(b.abs() + 1) + 0.0625 * initial_b
            for _ in range(2):
                c = torch.sqrt(c.abs() + 2) + 0.125 * initial_c
            if branches == 4:
                for _ in range(3):
                    d = torch.exp(-(d * d + 0.5)) + 0.0625 * initial_d
                expected = (a + b) + (c + d)
            else:
                expected = (a + b) + c
            a, b, c, d = expected, expected, expected, expected
        outputs = []
        for enabled in (False, True):
            storage = torch.full((rows * columns + 18,), 12345.0, device="cuda")
            output = storage[7:-11].view(rows, columns)
            kernel = _persistent_broadcast_dag[(triton.cdiv(rows, 4),)](
                *inputs, output, rows, columns, branches, waves, BLOCK_M=4, BLOCK_N=256,
                num_warps=4, enable_fp_fusion=False, enable_tile_dataflow=enabled,
                tile_dataflow_force=enabled, tile_dataflow_max_partitions=branches,
            )
            torch.testing.assert_close(output, expected, rtol=4e-5, atol=5e-6)
            for guard in (storage[:7], storage[-11:]):
                torch.testing.assert_close(guard, torch.full_like(guard, 12345.0), rtol=0, atol=0)
            outputs.append(output)
            if enabled:
                _schedule(kernel, [branches] * waves)
                _check_persistent_handoffs(kernel, retain_worker_results=True)
            else:
                assert "ttg.warp_specialize" not in kernel.asm["ttgir"]
        torch.testing.assert_close(outputs[0], outputs[1], rtol=4e-5, atol=5e-6)
        for actual, original in zip(inputs, originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
