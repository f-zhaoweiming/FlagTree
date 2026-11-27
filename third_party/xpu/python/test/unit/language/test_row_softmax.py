"""KL3 fused row-softmax regressions for the bias lifetime and packet reduction."""
import pytest
import torch
from triton.backends import backends
from triton.runtime import driver


@pytest.mark.parametrize("route", ("inductor", "tle"))
def test_row_softmax_reuses_bias_within_one_launch(route):
    if not torch.cuda.is_available() or "xpu" not in backends:
        pytest.skip("P800 required")
    if not isinstance(driver.active, backends["xpu"].driver) or driver.active.get_current_target().arch != 3:
        pytest.skip("P800 required")
    from triton.experimental.graph_fusion import _p800_reduce, _resource_autotune

    class RowSoftmax(torch.nn.Module):
        def forward(self, x, bias):
            return torch.softmax(x + bias, -1)

    n = 2048
    lane = torch.arange(n, device="cuda")
    x = torch.full((n, n), -100000., device="cuda")
    x[lane, (103 * lane) % n] += 12.
    x[0, 3] = torch.nan
    x[1, 5] = torch.inf
    x[2].fill_(-torch.inf)
    bias = torch.zeros(n, device="cuda")
    if route == "tle":
        run, _ = _p800_reduce.compile(dict(kind="add_softmax", m=n, n=n, dtype="float32", eps=0.), (x, bias))

        def fused(x, b):
            return run([x, b])[0]
    else:
        def backend(graph, inputs):
            compiled, _ = _resource_autotune.compile_candidate(graph, inputs, _resource_autotune.Candidate("inductor"))
            return compiled

        fused = torch.compile(RowSoftmax(), backend=backend, fullgraph=True, dynamic=False)

    outputs = []
    for shift in (1, 7):
        # Change the same allocation between calls. A tile cached across
        # launches would reuse the first bias despite the unchanged pointer.
        bias.copy_(((lane * shift) % 19).float() * .25)
        expected = torch.softmax(x + bias, -1).cpu()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = fused(x, bias)
        stream.synchronize()
        outputs.append((actual, expected))
    assert outputs[0][0].data_ptr() != outputs[1][0].data_ptr()
    for actual, expected in outputs:
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("rows,n,dtype", [
    (32, 512, torch.float32), (64, 1024, torch.float32), (64, 1024, torch.float16),
    (512, 1024, torch.float32), (512, 1024, torch.float16), (64, 2048, torch.float16),
])
def test_small_row_softmax_caches_launch_layout(rows, n, dtype, monkeypatch):
    if not torch.cuda.is_available() or "xpu" not in backends:
        pytest.skip("P800 required")
    if not isinstance(driver.active, backends["xpu"].driver) or driver.active.get_current_target().arch != 3:
        pytest.skip("P800 required")
    from triton.experimental.graph_fusion import _p800_reduce
    from triton.tools.tensor_descriptor import TensorDescriptor

    a = torch.randn(rows, n, device='cuda', dtype=dtype)
    b = torch.randn(n, device='cuda', dtype=dtype)
    spec = dict(kind='add_softmax', m=rows, n=n, dtype=str(dtype).split('.')[-1], eps=0.)
    run, proof = _p800_reduce.compile(spec, (a, b))
    create = TensorDescriptor.from_tensor
    calls = []

    def counted(tensor, block_shape, *args, **kwargs):
        calls.append(tuple(block_shape))
        return create(tensor, block_shape, *args, **kwargs)

    monkeypatch.setattr(TensorDescriptor, 'from_tensor', staticmethod(counted))
    first = run([a, b])[0]
    assert proof['tiles']['route'] == 'tle'
    assert len(calls) == 3
    tol = .003 if dtype == torch.float16 else .0003
    outputs = [(first, torch.softmax(a + b, -1).cpu())]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Mutate the original allocation, then replace both input addresses.
        b.add_(1.5 * torch.randn_like(b))
        for x, bias in ((a, b), (torch.randn_like(a), torch.randn_like(b))):
            y = run([x, bias])[0]
            outputs.append((y, torch.softmax(x + bias, -1).cpu()))
    stream.synchronize()
    assert len(calls) == 3
    assert len({y.data_ptr() for y, _ in outputs}) == len(outputs)
    for y, expected in outputs:
        torch.testing.assert_close(y.cpu(), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize('dtype', (torch.float16, torch.float32))
def test_512_row_softmax_guards_and_reduction(dtype, monkeypatch):
    if not torch.cuda.is_available() or 'xpu' not in backends:
        pytest.skip('P800 required')
    if not isinstance(driver.active, backends['xpu'].driver) or driver.active.get_current_target().arch != 3:
        pytest.skip('P800 required')
    from torch.utils._python_dispatch import TorchDispatchMode
    from triton.experimental.graph_fusion import _p800_reduce as reduce

    bound = []
    submitted = []
    bind = reduce._bind_row_softmax

    def save(binary, *args):
        entry = binary.run.launch

        def observe(*params):
            submitted.append((entry, params))
            return entry(*params)

        with monkeypatch.context() as patch:
            patch.setattr(binary.run, 'launch', observe)
            launch = bind(binary, *args)
        bound.append(launch)
        return launch

    monkeypatch.setattr(reduce, '_bind_row_softmax', save)
    x = torch.randn(512, 1024, device='cuda', dtype=dtype) * 12
    bias = torch.randn_like(x[0]) * 7
    x[0] = -torch.inf
    x[1, 0] = torch.inf
    x[2, 1] = torch.nan
    # Each row's maximum is in a different lane of the 16-lane packet.
    rows = torch.arange(509, device='cuda')
    x[3:] = -1000
    x[3 + rows, (rows * 71) % 1024] = -973
    run, proof = reduce.compile(dict(kind='add_softmax', m=512, n=1024,
                                     dtype=str(dtype).split('.')[-1], eps=0.), (x, bias))
    y = run([x, bias])[0]
    assert proof['tiles']['xb'] == (8 if dtype == torch.float16 else 16)
    tol = .003 if dtype == torch.float16 else .0003
    torch.testing.assert_close(y.cpu(), (x + bias).softmax(-1).cpu(), rtol=tol,
                               atol=1e-6 if dtype == torch.float16 else 1e-7, equal_nan=True)
    launch, = bound
    assert launch is not None
    result = launch(x, bias)
    torch.testing.assert_close(result.cpu(), y.cpu(), rtol=0, atol=0, equal_nan=True)
    entry, params = submitted[-1]
    for index, tensor in ((9, x), (14, bias), (17, result)):
        assert params[index] == (tensor.data_ptr(), tensor.numel())
    for bad, error in [((0, 1), ValueError), ((-1, x.numel()), OverflowError),
                       ((x.data_ptr(), -1), ValueError), ((None, x.numel()), TypeError)]:
        with pytest.raises(error):
            entry(*params[:9], bad, *params[10:])
    assert launch(x[:256], bias) is None
    other_dtype = torch.float32 if dtype == torch.float16 else torch.float16
    assert launch(x.to(other_dtype), bias.to(other_dtype)) is None
    noncontiguous = torch.empty(512, 2048, dtype=dtype, device=x.device)[:, ::2]
    assert launch(noncontiguous, bias) is None
    unaligned = torch.empty(1025, dtype=dtype, device=x.device)[1:]
    assert launch(x, unaligned) is None

    class PassThrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with PassThrough():
        assert launch(x, bias) is None
