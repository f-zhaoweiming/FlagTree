"""Event samples must be valid before a GEMM time can be summarized."""
from types import SimpleNamespace

import pytest

from triton import testing


def fake_bench(monkeypatch, values, *, rep=0, mode="all"):
    readings = iter(values)
    pairs = []
    calls = []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, end):
            pairs.append((self, end))
            return next(readings)

    device = SimpleNamespace(Event=Event, synchronize=lambda: None)
    driver = SimpleNamespace(get_device_interface=lambda: device, get_empty_cache_for_benchmark=lambda: None,
                             clear_cache=lambda _: None)
    monkeypatch.setattr(testing.runtime, "driver", SimpleNamespace(active=driver))

    def fn():
        calls.append(1)

    def bench():
        return testing.do_bench(fn, warmup=0, rep=rep, return_mode=mode)

    return bench, pairs, calls


@pytest.mark.parametrize("bad", [-15622204.0, 0.0, float("nan"), float("inf"), -float("inf"), 18909684.0])
def test_remeasure_bad_estimate_and_sample(monkeypatch, bad):
    bench, pairs, calls = fake_bench(monkeypatch, [bad, .25, bad, .125])
    assert bench() == [.125]
    assert len(pairs) == 4 and len({event for pair in pairs for event in pair}) == 8
    assert len(calls) == 1 + 2 * 5 + 1 + 2


def test_invalid_sample_replaced_before_median(monkeypatch):
    bench, pairs, _ = fake_bench(monkeypatch, [.25, .02, -1234., .04, .03], rep=.17, mode="median")
    assert bench() == .03 and len(pairs) == 5


@pytest.mark.parametrize("values", [[0.] * 3, [.25] + [float("nan")] * 3])
def test_persistent_bad_event_does_not_return_a_time(monkeypatch, values):
    bench, pairs, _ = fake_bench(monkeypatch, values)
    with pytest.raises(RuntimeError, match="Invalid benchmark event interval"):
        bench()
    assert len(pairs) == len(values)


def test_positive_value_within_host_window_is_valid():
    assert testing._possible_event_ms(15000., 15001.)
    assert not testing._possible_event_ms(18909684., 6.94)
