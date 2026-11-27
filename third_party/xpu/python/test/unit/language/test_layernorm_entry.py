"""Guarded replay must read current affine LayerNorm parameters on P800."""
import inspect
import sys

import pytest
import torch
from triton.backends import backends
from triton.runtime import driver


def test_layernorm_replay_reads_registered_parameters():
    if (sys.version_info[:2] != (3, 10) or not torch.__version__.startswith("2.9.")
            or not torch.cuda.is_available() or "xpu" not in backends
            or not isinstance(driver.active, backends["xpu"].driver)
            or driver.active.get_current_target().arch != 3):
        pytest.skip("requires the Torch 2.9/Python 3.10 P800 module entry")
    from triton.experimental import graph_fusion

    class AddLayerNorm(torch.nn.LayerNorm):
        def forward(self, x, bias):
            return torch.nn.functional.layer_norm(x + bias, self.normalized_shape, self.weight, self.bias, self.eps)

    mod = AddLayerNorm(1024, eps=1e-3, device="cuda").eval()
    backend = graph_fusion.autotune_resource_backend(tolerance={"atol": 3e-4, "rtol": 3e-4},
                                                     environment_key="layernorm-parameter-replay")
    public = torch.compile(mod, backend=backend, dynamic=False, fullgraph=True)
    x = torch.randn(64, 1024, device="cuda")
    bias = torch.randn(1024, device="cuda")

    with torch.no_grad():
        original = public(x, bias)
        saved = original.cpu()
        assert getattr(public.forward, "_flagtree_hot_entry", None) is backend
        assert inspect.getclosurevars(public.forward).nonlocals["cached"] is not None
        mod.weight.add_(2)
        after_mutation = public(x, bias)
        torch.testing.assert_close(after_mutation.cpu(), mod(x, bias).cpu())
        assert not torch.equal(after_mutation.cpu(), saved)
        mod.bias = torch.nn.Parameter(torch.full_like(mod.bias, 3.0))
        after_replacement = public(x, bias)
        torch.testing.assert_close(after_replacement.cpu(), mod(x, bias).cpu())
        assert not torch.equal(after_replacement, after_mutation)
        torch.testing.assert_close(original.cpu(), saved, atol=0, rtol=0)


def test_public_calibration_rejects_mutation_before_discovery():
    if (sys.version_info[:2] != (3, 10) or not torch.__version__.startswith("2.9.")
            or not torch.cuda.is_available() or "xpu" not in backends
            or not isinstance(driver.active, backends["xpu"].driver)
            or driver.active.get_current_target().arch != 3):
        pytest.skip("requires the Torch 2.9/Python 3.10 P800 module entry")
    from triton.experimental import graph_fusion

    class Mutating(torch.nn.Module):
        def forward(self, value):
            return value.add_(1.)

    backend = graph_fusion.autotune_resource_backend(tolerance={"atol": 3e-4, "rtol": 3e-4},
                                                     environment_key="reject-before-discovery")
    public = torch.compile(Mutating(), backend=backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        x = torch.zeros(8, device="cuda")
        with pytest.raises(Exception, match="mutating|mutations|unsupported"):
            public(x)
        torch.testing.assert_close(x.cpu(), torch.zeros(8), atol=0, rtol=0)


def test_public_trials_stay_on_the_calibrating_thread(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    if (sys.version_info[:2] != (3, 10) or not torch.__version__.startswith("2.9.")
            or not torch.cuda.is_available() or "xpu" not in backends
            or not isinstance(driver.active, backends["xpu"].driver)
            or driver.active.get_current_target().arch != 3):
        pytest.skip("requires the Torch 2.9/Python 3.10 P800 module entry")
    from triton.experimental.graph_fusion import _resource_autotune as auto
    from triton.experimental.graph_fusion import _resource_benchmark as bench

    class Add(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    calibrating = threading.Event()
    second_entered = threading.Event()
    entered_trials = []
    committed = []

    def calibrate(gm, inputs, *, public_timer, **options):
        def trial(*args):
            entered_trials.append(threading.get_ident())
            calibrating.set()
            assert second_entered.wait(10)
            return gm.forward(*args)

        def chosen(*args):
            committed.append(threading.get_ident())
            return gm.forward(*args)

        public_timer({'trial': trial}, inputs, seed=1)
        return chosen, {}

    def short_round(functions, inputs, *, before_call, **options):
        before_call('trial')
        functions['trial'](*inputs)
        return {'trial': [1.0]}

    monkeypatch.setattr(auto, 'calibrate_resources', calibrate)
    monkeypatch.setattr(bench, 'paired_end_to_end', short_round)
    backend = auto.autotune_resource_backend(tolerance={'atol': 3e-4, 'rtol': 3e-4}, environment_key='concurrent')
    module = torch.compile(Add(), backend=backend, fullgraph=True, dynamic=False)
    x = torch.randn(8, device='cuda')
    y = torch.randn_like(x)

    def call(second):
        with torch.no_grad():
            if second:
                assert calibrating.wait(10)
                second_entered.set()
            return threading.get_ident(), module(x, y).cpu()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call, False)
        second = pool.submit(call, True)
        first_id, first_value = first.result(timeout=30)
        second_id, second_value = second.result(timeout=30)
    assert entered_trials == [first_id]
    assert first_id in committed and second_id in committed
    for result in (first_value, second_value):
        torch.testing.assert_close(result, (x + y).cpu())


@pytest.mark.parametrize("arity", [2, 3, 4])
def test_warm_entry_checks_current_positional_tensors(monkeypatch, arity):
    if (sys.version_info[:2] != (3, 10) or not torch.__version__.startswith("2.9.")
            or not torch.cuda.is_available() or "xpu" not in backends
            or not isinstance(driver.active, backends["xpu"].driver)
            or driver.active.get_current_target().arch != 3):
        pytest.skip("requires the Torch 2.9/Python 3.10 P800 module entry")
    from triton.experimental.graph_fusion import _resource_autotune as auto

    class Two(torch.nn.Module):
        def forward(self, right, left):
            return left + right * 2

    class Three(torch.nn.Module):
        def forward(self, right, left, extra):
            return left + right * extra

    class Four(torch.nn.Module):
        def forward(self, right, left, extra, last):
            return left + right * extra + last * 3

    def calibrate(gm, inputs, **options):
        return gm.forward, {}

    monkeypatch.setattr(auto, 'calibrate_resources', calibrate)
    mod = {2: Two, 3: Three, 4: Four}[arity]().eval()
    backend = auto.autotune_resource_backend(tolerance={'atol': 3e-4, 'rtol': 3e-4}, environment_key='arity')
    public = torch.compile(mod, backend=backend, dynamic=False, fullgraph=True)
    names = list(inspect.signature(mod.forward).parameters)
    with torch.no_grad():
        for size, dtype in [(8, torch.float32), (8, torch.float32), (16, torch.float16)]:
            args = tuple(torch.randn(size, device="cuda", dtype=dtype) for _ in names)
            torch.testing.assert_close(public(*args), mod(*args))
            torch.testing.assert_close(public(**dict(zip(names, args))), mod(*args))
            assert inspect.getclosurevars(public.forward).nonlocals['cached'] is not None
    # Fallbacks must still observe current grad state and module hooks after warmup.
    args = tuple(torch.randn(8, device="cuda", requires_grad=True) for _ in names)
    with torch.enable_grad():
        assert public(*args).requires_grad
    with torch.no_grad():
        with mod.register_forward_pre_hook(lambda _, xs: (xs[0] + 5, *xs[1:])):
            torch.testing.assert_close(public(*args), mod(*args))
        torch.testing.assert_close(public(*args), mod(*args))
