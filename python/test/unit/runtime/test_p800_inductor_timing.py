"""Regression tests for the Torch 2.9 Inductor patch shipped with the P800 backend."""
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
from torch._inductor import autotune_process, codecache, config
from torch._inductor.codecache import CacheBase, PersistentCache

pytestmark = pytest.mark.skipif(not hasattr(codecache, "_valid_benchmark_ms"), reason="requires patched Torch 2.9 cache")


@dataclass(frozen=True)
class Choice:
    key: str

    def hash_key(self):
        return self.key

    def info_dict(self):
        return {}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(CacheBase, "get_local_cache_path", staticmethod(lambda: tmp_path / "system"))
    with config.patch(autotune_local_cache=True):
        yield PersistentCache()


def test_cache_retries_only_invalid_choices(cache):
    a, b, c = map(Choice, ("a", "b", "c"))
    rounds = []

    def benchmark(choices):
        rounds.append(list(choices))
        if len(rounds) == 1:
            return {a: 0.02, b: -15622204.0, c: math.nan}
        return {x: (0.03 if x == b else 0.0) for x in choices}

    assert cache.lookup([a, b, c], "mm", "shape", benchmark) == {a: 0.02, b: 0.03}
    assert rounds == [[a, b, c], [b, c], [c]]
    bucket = cache.get_local_cache()["mm"]["shape"][torch.get_float32_matmul_precision()]
    assert bucket == {"a": 0.02, "b": 0.03}
    assert cache.lookup([a, b], "mm", "shape", None) == {a: 0.02, b: 0.03}


def test_missing_and_corrupt_cache_entries_are_misses(cache):
    keys = list(map(Choice, ["good", "negative", "zero", "nan", "inf", "minus_inf", "text", "bool"]))
    values = [0.5, -1.0, 0.0, math.nan, math.inf, -math.inf, "0.1", True]
    precision = torch.get_float32_matmul_precision()
    legacy = CacheBase.get_local_cache_path()
    legacy.write_text(json.dumps({"cache": {"mm": {"shape": {precision: {"good": 18909684.0}}}}}))
    assert cache.lookup(keys, "mm", "shape", None) == {}
    bucket = dict(zip((c.key for c in keys), values))
    cache.update_local_cache({"mm": {"shape": {precision: bucket}}})
    assert cache.lookup(keys, "mm", "shape", None) == {keys[0]: 0.5}
    assert cache.get_local_cache()["mm"]["shape"][precision] == {"good": 0.5}
    new = cache.lookup(keys, "mm", "shape", lambda choices: {c: 0.1 for c in choices})
    assert all(new[c] == 0.1 for c in keys)
    assert cache.get_local_cache_path() != legacy


def test_exhausted_choices_do_not_enter_cache(cache):
    a, b = map(Choice, ("a", "b"))
    rounds = []

    def all_failed(choices):
        rounds.append(list(choices))
        return {c: (math.inf if c == a else -math.inf) for c in choices}

    assert cache.lookup([a, b], "mm", "shape", all_failed) == {}
    # The existing +inf sentinel represents an unsupported or failed kernel.
    assert rounds == [[a, b], [b], [b]]
    assert cache.get_local_cache()["mm"]["shape"][torch.get_float32_matmul_precision()] == {}
    assert cache.lookup([b], "mm", "shape", lambda _: {b: 0.2}) == {b: 0.2}


def test_cache_disabled_performs_no_disk_write(cache):
    c = Choice("a")
    with config.patch(autotune_local_cache=False):
        assert cache.lookup([c], "mm", "shape", lambda _: {c: 0.2}) == {c: 0.2}
    assert not cache.get_local_cache_path().exists()


def test_concurrent_benchmarks_merge_cache_entries(cache):
    wait = threading.Barrier(2)

    def run(key):
        choice = Choice(key)

        def measure(choices):
            wait.wait(timeout=10)
            return {choice: 0.01}

        return cache.lookup([choice], "mm", "shape", measure)

    with ThreadPoolExecutor(max_workers=2) as threads:
        list(threads.map(run, ["a", "b"]))
    assert cache.get_local_cache()["mm"]["shape"][torch.get_float32_matmul_precision()] == {"a": 0.01, "b": 0.01}


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, 18909684.0])
def test_gpu_benchmark_retries_bad_numeric_results(bad):
    device = mock.Mock()
    device.device.return_value = nullcontext()
    with mock.patch.object(autotune_process, "get_interface_for_device", return_value=device), \
         mock.patch.object(autotune_process.benchmarker, "benchmark_gpu", side_effect=[bad, bad, 0.02]) as timer:
        assert autotune_process.GPUDeviceBenchmarkMixin().do_bench(lambda: None) == 0.02
    assert timer.call_count == device.synchronize.call_count == 3


def test_close_gemm_reconfirms_in_different_orders_before_caching(cache):
    a, b, c = map(Choice, ("a", "b", "c"))
    orders = []
    results = iter([
        {a: 0.0260, b: 0.0259, c: 0.050},
        {c: 0.051, b: 0.028, a: 0.021},
        {b: 0.027, c: 0.049, a: 0.022},
    ])

    def measure(choices):
        orders.append(list(choices))
        return next(results)

    expected = {a: 0.022, b: 0.027, c: 0.050}
    assert cache.lookup([a, b, c], "mm", "shape", measure) == expected
    assert orders == [[a, b, c], [c, b, a], [b, c, a]]
    assert cache.lookup([a, b, c], "mm", "shape", measure) == expected


def test_gemm_failed_in_confirmation_is_excluded(cache):
    a, b = map(Choice, ("a", "b"))
    results = iter([{a: 0.020, b: 0.0201}, {b: 0.021, a: math.inf}, {b: 0.022}])
    assert cache.lookup([a, b], "addmm", "shape", lambda _: next(results)) == {b: 0.021}
    assert cache.lookup([a, b], "addmm", "shape", None) == {b: 0.021}


def test_close_non_gemm_keeps_one_measurement(cache):
    a, b = map(Choice, ("a", "b"))
    measure = mock.Mock(return_value={a: 0.020, b: 0.0201})
    assert cache.lookup([a, b], "convolution", "shape", measure) == measure.return_value
    measure.assert_called_once_with([a, b])


@pytest.mark.parametrize("mode,expected", [("min", 0.018), ("all", [0.018, 0.025])])
def test_native_inductor_replaces_bad_pilot_and_sample_before_reduction(mode, expected):
    from torch._inductor.runtime.benchmarking import InductorBenchmarker

    # The negative pilot and the out-of-window final sample each get a new pair.
    readings = iter([-15622204.0, 0.021, 0.022, 18909684.0, 0.018, 0.025])
    ends = []
    buffer = mock.Mock()
    fn = mock.Mock()

    def event(enable_timing):
        assert enable_timing
        handle = mock.Mock()

        def elapsed(end):
            ends.append(end)
            return next(readings)

        handle.elapsed_time.side_effect = elapsed
        return handle

    bm = InductorBenchmarker()
    bm.L2_cache_size = 8
    with mock.patch.object(torch.cuda, "synchronize"), mock.patch.object(torch.cuda, "Event", side_effect=event), \
            mock.patch.object(torch, "empty", return_value=buffer):
        assert bm.benchmark_gpu(fn, estimation_iters=2, benchmark_iters=2, memory_warmup_iters=0,
                                return_mode=mode) == expected
    assert len(set(ends)) == 6 and fn.call_count == 7


def test_native_inductor_rejects_persistently_invalid_events():
    from torch._inductor.runtime.benchmarking import InductorBenchmarker

    bm = InductorBenchmarker()
    bm.L2_cache_size = 8

    def event(enable_timing):
        result = mock.Mock()
        result.elapsed_time.return_value = float("nan")
        return result

    fn = mock.Mock()
    with mock.patch.object(torch.cuda, "synchronize"), mock.patch.object(torch.cuda, "Event", side_effect=event), \
            mock.patch.object(torch, "empty"), pytest.raises(RuntimeError, match="Invalid benchmark event interval"):
        bm.benchmark_gpu(fn, estimation_iters=1, memory_warmup_iters=0)
    assert fn.call_count == 4


def test_ambiguous_tiles_use_grouped_measurements_and_restore_mode(cache):
    from torch._inductor.runtime.benchmarking import gemm_timing_group_size

    class GemmChoice(Choice):
        def info_dict(self):
            return {"backend": "Triton", "tile_shape": self.key}

    a, b = map(GemmChoice, ("64x64", "128x128"))
    sizes = []

    def measure(choices):
        size = gemm_timing_group_size.get()
        sizes.append(size)
        return {a: 0.028, b: 0.024} if size == 16 else {a: 0.020, b: 0.0202}

    with config.patch(autotune_in_subproc=False):
        assert cache.lookup([a, b], "mm", "shape", measure) == {a: 0.028, b: 0.024}
    assert sizes == [1, 1, 1, 16]
    assert gemm_timing_group_size.get() == 1
    assert cache.lookup([a, b], "mm", "shape", None) == {a: 0.028, b: 0.024}


def test_grouped_gpu_benchmark_returns_median_per_real_call():
    from torch._inductor.runtime.benchmarking import gemm_timing_group_size

    device = mock.Mock()
    device.device.return_value = nullcontext()
    fn = mock.Mock()
    def benchmark(group, return_mode):
        assert return_mode == "all"
        group()
        return [0.16, 0.32, 0.48]

    token = gemm_timing_group_size.set(16)
    try:
        with mock.patch.object(autotune_process, "get_interface_for_device", return_value=device), \
                mock.patch.object(autotune_process.benchmarker, "benchmark_gpu", side_effect=benchmark):
            assert autotune_process.GPUDeviceBenchmarkMixin().do_bench(fn) == 0.02
    finally:
        gemm_timing_group_size.reset(token)
    assert fn.call_count == 16
