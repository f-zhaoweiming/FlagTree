"""Validate guarded replay against a real Dynamo guard tree on the P800 runtime ABI."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import inspect
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 10)
    or not torch.__version__.startswith("2.9."),
    reason="The native P800 entry uses the CPython 3.10 and Torch 2.9 frame ABI",
)


class WeightedAdd(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(16))

    def forward(self, a, b):
        return (a + b) * self.weight


def test_warm_frame_preserves_dynamo_and_module_semantics():
    from triton.experimental.graph_fusion import _resource_autotune, _resource_codegen, _resource_entry

    model = WeightedAdd().eval()
    a, b = torch.randn(8, 16), torch.randn(16)
    shifted = torch.randn(9, 16)
    target = {"arch": 3, "warp_size": 1}
    with (torch.no_grad(), patch.object(_resource_codegen, "baidu_target", return_value=target),
          patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
          patch.object(_resource_autotune, "calibrate_resources",
                       side_effect=lambda gm, xs, **kwargs: (gm.forward, {})) as calibrations):
        backend = _resource_autotune.autotune_resource_backend(
            tolerance=dict(rtol=.003, atol=.003), environment_key="native-entry-regression")
        public = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        if getattr(public.forward, "_flagtree_hot_entry", None) is not backend:
            public = backend._flagtree_wrap_module(public)

        def binding():
            return inspect.getclosurevars(public.forward).nonlocals["native"]

        held = public(a, b)
        torch.testing.assert_close(held, model(a, b))
        original = held.clone()
        frame, hot = binding()
        assert isinstance(hot((a, b)), torch.Tensor)
        assert hot((shifted, b)) is _resource_entry._GUARD_MISS
        for i in (1, 2, 3):
            torch.testing.assert_close(public(a + i, b + i), model(a + i, b + i))
        torch.testing.assert_close(held, original)

        with torch.enable_grad():
            assert hot((a, b)) is _resource_entry._MISS
        with torch.autocast("cpu"):
            value = hot((a, b))
            assert value is _resource_entry._MISS or value is _resource_entry._GUARD_MISS
        with torch.overrides.TorchFunctionMode():
            assert hot((a, b)) is _resource_entry._MISS
        with model.register_forward_hook(lambda mod, inputs, out: out + 1):
            assert hot((a, b)) is _resource_entry._MISS
            torch.testing.assert_close(public(a, b), model(a, b))
        with torch.nn.modules.module.register_module_forward_pre_hook(lambda mod, inputs: None):
            assert hot((a, b)) is _resource_entry._MISS

        fast_dispatch = vars(public)['_call_impl']
        tag = fast_dispatch.__self__[3]
        previous_tag = tag[0]
        with patch.object(type(public), 'unrelated_entry_test', True, create=True):
            torch.testing.assert_close(public(a, b), model(a, b))
            assert tag[0] != previous_tag
            refreshed_tag = tag[0]
            torch.testing.assert_close(public(a, b), model(a, b))
            assert tag[0] == refreshed_tag
        hook_reference = model(a, b)
        with public.register_forward_hook(lambda wrapped, inputs, out: out + 3):
            torch.testing.assert_close(public(a, b), hook_reference + 3)
        previous_forward = public.forward
        try:
            public.forward = lambda x, y: x - y
            torch.testing.assert_close(public(a, b), a - b)
        finally:
            public.forward = previous_forward
        torch.testing.assert_close(public(a, b), model(a, b))

        manager = frame[0]
        original_check = manager.check
        checks = []

        def count_public_checks(locals_):
            if sys._getframe(1).f_code is public.forward.__code__:
                checks.append(1)
            return original_check(locals_)

        before = calibrations.call_count
        with patch.object(manager, "check", new=count_public_checks):
            torch.testing.assert_close(public(shifted, b), model(shifted, b))
        assert checks == [1]
        assert calibrations.call_count > before

        model.weight = torch.nn.Parameter(torch.full((16,), 2.5))
        torch.testing.assert_close(public(a=shifted, b=b), model(shifted, b))
        torch.testing.assert_close(public(shifted, b), model(shifted, b))

        def assert_refreshed(previous):
            assert previous((shifted, b)) is _resource_entry._REFRESH
            torch.testing.assert_close(public(shifted, b), model(shifted, b))
            assert isinstance(binding()[1]((shifted, b)), torch.Tensor)

        previous = binding()[1]
        model.tag_for_test = 1
        assert_refreshed(previous)
        previous = binding()[1]
        with patch.object(WeightedAdd, "unused_class_field", 1, create=True):
            assert_refreshed(previous)

        with patch.object(WeightedAdd, "forward", lambda self, a, b: a - b):
            value = hot((a, b))
            assert value is _resource_entry._MISS or value is _resource_entry._REFRESH
            torch.testing.assert_close(public(a, b), model(a, b))
        torch._dynamo.reset()
        value = hot((a, b))
        assert value is _resource_entry._MISS or value is _resource_entry._REFRESH
        torch.testing.assert_close(public(a, b), model(a, b))

        def concurrent_inference(i):
            with torch.no_grad():
                return torch.equal(public(a + i, b), model(a + i, b))

        with ThreadPoolExecutor(max_workers=3) as pool:
            assert all(pool.map(concurrent_inference, range(12)))


def test_packed_launch_reads_current_addresses_and_stream():
    from triton.experimental.graph_fusion import _resource_entry
    extension = _resource_entry._frame_extension()
    assert extension is not None
    calls = []
    stream = [7]
    x = torch.empty(8)
    sizes = (8, 8, 8)
    template = (1, 1, 1, None, 44, (), None, None, None, None, 8, None, None)
    launch = extension.bind_dense((lambda *args: calls.append(args), template, (9, 11, 12), sizes,
                                   torch.Tensor.data_ptr, torch.empty, ((8,),), lambda device: stream[0], 0))
    y = torch.empty_like(x)
    a = launch(x, y)
    stream[0] = 17
    b = launch(y, x)
    assert calls[0][3] == 7 and calls[1][3] == 17
    assert calls[0][9:] == ((x.data_ptr(), 8), 8, (y.data_ptr(), 8), (a.data_ptr(), 8))
    assert calls[1][9:] == ((y.data_ptr(), 8), 8, (x.data_ptr(), 8), (b.data_ptr(), 8))
    assert a.data_ptr() != b.data_ptr()
    assert launch(x[1:], y) is None and len(calls) == 2
    assert launch(x) is None


def test_layernorm_packed_launch_accepts_live_affine_or_aliased_bias():
    from triton.experimental.graph_fusion._p800_bound_launch import _pack_launch

    calls = []
    stream = [7]

    def allocate(shape, stride, dtype):
        return torch.empty_strided(shape, stride, dtype=dtype)

    launch = _pack_launch(lambda *args: calls.append(args), (12, 1, 1, None, 44, (), None, None, None),
                          (None,) * 5, (9, 10, 11, 12, 13), (16, 8, 8, 8, 16),
                          allocate, ((2, 8), (8, 1), torch.float32), lambda device: stream[0], 0)
    matrix = torch.empty(2, 8)
    bias, weight, shift = (torch.empty(8) for _ in range(3))
    for w, s in ((bias, bias), (weight, shift)):
        y = launch(matrix, bias, w, s)
        assert calls[-1][3] == stream[0]
        assert calls[-1][9:] == tuple((t.data_ptr(), t.numel()) for t in (matrix, bias, w, s, y))
        stream[0] += 1
    assert launch(matrix, bias[1:], weight, shift) is None
    assert len(calls) == 2


@pytest.mark.parametrize("metadata,sized", [({"tensor_args": []}, False), ({"tensor_args": [0]}, True), ({}, True)])
def test_layernorm_launch_obeys_compiled_pointer_metadata(metadata, sized):
    from triton import runtime
    from triton.backends.xpu import driver as xpu_driver
    from triton.experimental.graph_fusion._p800_reduce import _bind_layernorm_launch

    calls = []
    launcher = type("Launcher", (), {"__module__": xpu_driver.__name__, "supports_sized_pointers": True})()
    launcher.launch = lambda *args: calls.append(args)
    binary = SimpleNamespace(run=launcher, metadata=SimpleNamespace(is_sdnn=False, **metadata), function=44,
                             packed_metadata=())
    active = SimpleNamespace(get_current_device=lambda: None, get_current_stream=lambda _: 17)

    def allocate(shape, stride, dtype):
        return torch.empty_strided(shape, stride, dtype=dtype)

    with torch.no_grad(), patch.object(runtime, "driver", SimpleNamespace(active=active)), patch.object(
            torch._C._dynamo.guards, "_empty_strided_cuda", allocate):
        matrix, bias = torch.empty(2, 8), torch.empty(8)
        run = _bind_layernorm_launch(binary, SimpleNamespace(pre_run_hooks=[]), 12, (matrix, bias), False)
        out = run((matrix, bias))
    assert len(calls) == 1 and calls[0][3] == 17
    tensors = (matrix, bias, bias, bias, out)
    assert calls[0][9:] == tuple((t.data_ptr(), t.numel()) if sized else t.data_ptr() for t in tensors)


def test_native_dense_tracks_storage_offsets_and_preserves_accessor_errors():
    from triton.experimental.graph_fusion._p800_bound_launch import _pack_launch

    calls = []
    stream = [1]
    launch = _pack_launch(lambda *args: calls.append(args), (12, 1, 1, None, 44, (), None, None, None),
                          (None,) * 4, (9, 10, 11, 12), (8, 8, 4, 8),
                          torch.empty, ((8,),), lambda _: stream[0], 0)
    assert launch is not None
    x, bias = torch.empty(16), torch.nn.Parameter(torch.empty(8))
    with torch.no_grad():
        hold = launch(x, bias, bias)
        assert calls[-1][9:] == ((x.data_ptr(), 8), (bias.data_ptr(), 8), (bias.data_ptr(), 4),
                                 (hold.data_ptr(), 8))
        replacement = torch.empty(24)
        x.set_(replacement[4:])
        stream[0] = 9
        fresh = launch(x, bias, bias)
        assert calls[-1][3] == 9 and calls[-1][9] == (replacement[4:].data_ptr(), 8)
        assert fresh.data_ptr() != hold.data_ptr()
        assert launch(x[1:], bias, bias) is None
        with pytest.raises(RuntimeError):
            launch(x.to_sparse(), bias, bias)
        assert len(calls) == 2


def test_dense_preserves_an_overridden_accessor_on_aliased_inputs():
    from triton.experimental.graph_fusion._p800_bound_launch import _pack_launch

    original = torch.Tensor.data_ptr
    seen = []

    def pointer(tensor):
        seen.append(tensor)
        return original(tensor)

    x = torch.empty(8)
    with patch.object(torch.Tensor, 'data_ptr', pointer):
        launch = _pack_launch(lambda *args: None, (1, 1, 1, None, 44, (), None, None, None),
                              (None,) * 3, (9, 10, 11), (8, 8, 8),
                              torch.empty, ((8,),), lambda _: 0, 0)
        assert launch is not None
        out = launch(x, x)
    assert len(seen) == 3 and seen[0] is x and seen[1] is x and seen[2] is out


def test_single_output_projection_preserves_call_order_and_guards():
    from triton.experimental.graph_fusion import _resource_autotune, _resource_codegen, _resource_entry

    class Product(torch.nn.Module):
        def forward(self, a, b):
            return (a @ b).relu()
    a = torch.tensor([[1., 3.], [2., 0.]])
    b = torch.tensor([[2., 1.], [1., 4.]])
    model = Product()
    with (torch.no_grad(), patch.object(_resource_codegen, 'baidu_target', return_value={'arch': 3}),
          patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
          patch.object(_resource_autotune, 'calibrate_resources',
                       side_effect=lambda gm, xs, **kwargs: (gm.forward, {}))):
        backend = _resource_autotune.autotune_resource_backend(tolerance=dict(atol=0., rtol=0.),
                                                              environment_key='project-return')
        public = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        if getattr(public.forward, '_flagtree_hot_entry', None) is not backend:
            public = backend._flagtree_wrap_module(public)
        assert torch.equal(public(a, b), model(a, b))
        (manager, replay), hot = inspect.getclosurevars(public.forward).nonlocals['native']
        assert getattr(replay, '__name__', '') == 'single_output'
        assert torch.equal(public(b, a), model(b, a))
        assert hot((a[:1], b[:1])) is _resource_entry._GUARD_MISS
        with patch.object(manager, 'check', return_value=False):
            assert hot((a, b)) is _resource_entry._GUARD_MISS
