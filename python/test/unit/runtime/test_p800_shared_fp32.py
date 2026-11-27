"""Shared P800 F06 body, SDK precision boundaries and cold-order coverage."""
import pytest
import torch


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
def test_unrounded_reference_exposes_intermediate_rounding(dtype):
    from triton.experimental.graph_fusion._p800_shared_fp32 import _FP32Reference
    graph = torch.fx.symbolic_trace(lambda a, b, c: a @ b + c)
    unit = torch.finfo(dtype).eps
    a = torch.tensor([[1., unit / 4]], dtype=dtype)
    b = torch.ones((2, 1), dtype=dtype)
    c = torch.tensor([-1.], dtype=dtype)
    reference = _FP32Reference(graph, dtype)
    tol = dict(atol=0., rtol=0.)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        wide = reference(a, b, c)
    assert wide.dtype == torch.float32 and wide.item() == unit / 4
    reference._f06_assert_close(wide.to(dtype), wide, tol)
    with pytest.raises(AssertionError):
        reference._f06_assert_close(graph(a, b, c), wide, tol)
    with pytest.raises(AssertionError):
        reference._f06_assert_close(wide, wide, tol)


def test_bf16_midpoint_does_not_veto_fp32_and_final_store_still_matters():
    from triton.experimental.graph_fusion._p800_shared_fp32 import _verify_fp32_probes, _FP32Reference
    ref = torch.tensor([0x411f8000], dtype=torch.int32).view(torch.float32)
    near = torch.tensor([0x411f7fff], dtype=torch.int32).view(torch.float32)
    tol = dict(atol=.0003, rtol=.0003)
    probes = [((), ref)]
    _verify_fp32_probes(lambda: near, probes, tol)
    with pytest.raises(AssertionError):
        _verify_fp32_probes(lambda: ref + .01, probes, tol)
    with pytest.raises(AssertionError):
        _verify_fp32_probes(lambda: ref.bfloat16(), probes, tol)
    oracle = _FP32Reference(torch.nn.Identity(), torch.bfloat16)
    oracle._f06_assert_close(near.bfloat16(), ref, tol)
    with pytest.raises(AssertionError):
        oracle._f06_assert_close(near.bfloat16(), ref, dict(atol=0., rtol=0.))
    with pytest.raises(AssertionError):
        oracle._f06_assert_close(torch.tensor([9.875], dtype=torch.bfloat16), ref, tol)
    specials = torch.tensor([float('nan'), float('inf'), -float('inf'), 0., torch.finfo(torch.float32).max])
    oracle._f06_assert_close(specials.bfloat16(), specials, tol)
    with pytest.raises(AssertionError):
        oracle._f06_assert_close(-specials.bfloat16(), specials, tol)


def test_shared_template_keeps_explicit_casts_and_multiple_outputs_separate():
    from triton.experimental.graph_fusion._p800_shared_fp32 import template_graph

    def returned(x, b):
        y = x + b
        return y, y.relu()

    for fn, expected in ((lambda x, b: (x + b).relu(), True),
                         (lambda x, b: (x + b).half().relu(), False),
                         (returned, False)):
        assert template_graph(torch.fx.symbolic_trace(fn).graph) is expected


@pytest.fixture
def p800():
    from triton.runtime import driver
    if (not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver'
            or driver.active.get_current_target().arch != 3):
        pytest.skip('requires P800')


def test_bf16_first_shares_program_and_keeps_affine_parameters_live(p800):
    from triton.experimental import graph_fusion as gf

    class Norm(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1024, device='cuda', dtype=dtype))
            self.shift = torch.nn.Parameter(torch.zeros_like(self.weight))

        def forward(self, x, bias):
            return torch.nn.functional.layer_norm(x + bias, (1024,), self.weight, self.shift, .001)

    choices = []
    with torch.no_grad():
        for dtype in (torch.bfloat16, torch.float32, torch.float16):
            audit = []
            tolerance = dict(atol=.0003, rtol=.0003) if dtype == torch.float32 else dict(
                atol=.03 if dtype == torch.bfloat16 else .003, rtol=.003)
            backend = gf.autotune_resource_backend(tolerance=tolerance, environment_key='qa/shared-bf16-first', audit=audit)
            model = Norm(dtype)
            fn = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
            x = torch.randn(64, 1024, device='cuda').to(dtype)
            bias = torch.randn(1024, device='cuda').to(dtype)
            previous = fn(x, bias)
            saved = previous.clone()
            model.shift.add_(.5)
            out = fn(x, bias)
            want = torch.nn.functional.layer_norm(x.float() + bias.float(), (1024,), model.weight.float(),
                                                  model.shift.float(), .001).to(dtype)
            torch.testing.assert_close(out.cpu(), want.cpu(), **tolerance)
            assert out.dtype == dtype and out.data_ptr() != previous.data_ptr()
            assert torch.equal(previous, saved)
            d = audit[-1]['decision']
            choices.append((d['shared_fp32'], d.get('compute_route',d['selected'])))
            if dtype != torch.float32:
                assert d['conversions'] == 'kernel_load_store' and d['launches']==1
            if dtype != torch.bfloat16:
                assert d['shared_cache_hit']
        assert len(set(choices)) == 1


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_ingress_uses_same_encodings_as_sdk(p800, dtype):
    from triton.experimental.graph_fusion._p800_fp32_region import fp32_input, round_output

    with torch.no_grad():
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        x = bits.view(dtype).cuda()
        expected = x.float()
        widened = fp32_input(x)
        assert torch.equal(widened.view(torch.int32).cpu(), expected.view(torch.int32).cpu())
        assert torch.equal(round_output(widened, dtype).view(torch.int16).cpu(),
                           expected.to(dtype).view(torch.int16).cpu())


@pytest.mark.parametrize('m,n', [(64, 1024), (512, 1024), (2048, 2048)])
def test_fp32_product_sum_matches_sdk_chunk_order(p800, m, n):
    from triton.experimental.graph_fusion import _p800_reduce
    with torch.no_grad():
        g = torch.Generator().manual_seed(97)
        a = torch.randn(m, n, generator=g).bfloat16().float().cuda()
        b = torch.randn(n, generator=g).bfloat16().float().cuda()
        spec = dict(kind='mul_sum', m=m, n=n, dtype='float32', eps=0., keepdim=False, nparam=0)
        fn, _ = _p800_reduce.compile(spec, (a, b))
        result, = fn((a, b))
        assert torch.equal(result.cpu(), (a * b).sum(-1).cpu())


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_common_body_final_cast_at_both_sides_of_ulp_midpoint(p800, dtype):
    from triton.experimental.graph_fusion._p800_shared_fp32 import bind

    step = torch.finfo(dtype).eps
    half = torch.tensor(step / 2, dtype=dtype)
    smaller = torch.nextafter(half, torch.zeros_like(half)).item()
    larger = torch.nextafter(half, torch.full_like(half, float('inf'))).item()
    a = torch.tensor([1, 1, 1, 1 + step, 1 + step, 1 + step, -0.], dtype=dtype, device='cuda')
    b = torch.tensor([smaller, half, larger, smaller, half, larger, -0.], dtype=dtype, device='cuda')
    with torch.no_grad():
        result = bind(torch.add, dtype, torch.fx.Graph().placeholder('result'))(a, b)
        expected = (a.float() + b.float()).to(dtype)
        assert torch.equal(result.cpu().view(torch.int16), expected.cpu().view(torch.int16))
        assert expected.float().cpu().tolist()[:6] == [1, 1, 1 + step, 1 + step, 1 + 2*step, 1 + 2*step]


def test_existing_handles_share_the_same_correctness_fallback():
    from triton.experimental.graph_fusion._p800_shared_fp32 import _Program, bind

    output = torch.fx.Graph().placeholder('output')
    program = _Program(lambda x: x + 2, 'cluster')
    high = bind(program, torch.float32, output)
    low = bind(program, torch.bfloat16, output)
    x = torch.ones(2)
    assert torch.equal(low(x.bfloat16()).float(), high(x))
    program.use_reference(lambda x: x + 1, 'changed-input validation')
    assert torch.equal(high(x), x + 1)
    assert torch.equal(low(x.bfloat16()).float(), high(x))
    assert program.state['route'] == 'eager'


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_mixed_sdnn_store_rounds_ties_in_one_kernel(p800, dtype):
    from triton.experimental.graph_fusion import _p800_fp32_mm

    with torch.no_grad():
        shape = (64, 1024)
        a = torch.zeros((64, 64), dtype=dtype, device='cuda')
        b = torch.zeros(shape, dtype=dtype, device='cuda')
        step = 64 * torch.finfo(dtype).eps
        half = torch.tensor(step / 2, dtype=dtype)
        offsets = [half.nextafter(torch.zeros_like(half)).item(), half.item(),
                   half.nextafter(torch.full_like(half, float('inf'))).item()]
        columns = torch.arange(1024) % 6
        a[:, 0] = 1
        b[0].copy_(torch.tensor([64.] * 3 + [64. + step] * 3, dtype=dtype)[columns])
        bias = torch.tensor(offsets * 2, dtype=dtype)[columns].cuda()
        kernel, proof = _p800_fp32_mm.prepare({'M': 64, 'N': 1024, 'act': 0}, final_only=True)
        result, = kernel((a, b, bias))
        expected = (a.float() @ b.float() + bias.float()).to(dtype)
        assert torch.equal(result.cpu().view(torch.int16), expected.cpu().view(torch.int16))
        assert proof['resource_verified'] and len(proof['kernel_evidence']) == 1


def test_sdnn_row_sum_admission_checks_partial_rows():
    from triton.experimental.graph_fusion._resource_fx import unsafe_sdnn_sum

    module = torch.fx.symbolic_trace(lambda x, b: (x * b).sum(-1))
    for rows, dtype, unsafe in [(64, torch.float32, False), (65, torch.float32, True),
                                (64, torch.float16, True)]:
        inputs = (torch.empty(rows, 32, dtype=dtype), torch.empty(32, dtype=dtype))
        assert unsafe_sdnn_sum(module.graph.nodes, inputs) == unsafe


def test_mixed_gemm_preserves_tail_columns_rows_and_output_guards(p800):
    from triton.experimental.graph_fusion._p800_compiler import resource_scope
    from triton.experimental.graph_fusion._p800_fp32_mm import mm_bias

    m, n, k = 65, 73, 63
    a = torch.zeros((m, k), dtype=torch.float16)
    b = torch.zeros((k, n), dtype=torch.float16)
    a[:, 0] = torch.arange(m) % 7
    b[0] = torch.arange(n) % 11
    a[:, -1] = 0.5
    b[-1] = 2.
    bias = (torch.arange(n) % 5 - 3).half()
    expected = torch.relu(a.float() @ b.float() + bias.float()).half()
    a, b, bias = (t.cuda() for t in (a, b, bias))
    storage = torch.full((32 + 128 * 128,), -123., dtype=torch.float16, device='cuda')
    out = storage[32:32 + m*n].view(m, n)
    with resource_scope('sdnn'):
        mm_bias[(2, 2, 1)](a, b, bias, out, m, n, 64, 64, ACT=2, FINAL_ONLY=True,
                            K=k, num_stages=1, num_warps=4, is_sdnn=True)
    assert torch.equal(out.cpu(), expected)
    assert storage[:32].eq(-123.).all().item()
    assert storage[32 + m*n:].eq(-123.).all().item()


def test_boundary_timing_rechecks_finalists_after_slow_matrix_launch(monkeypatch):
    from types import SimpleNamespace
    from triton.experimental.graph_fusion import _p800_inline_fp32 as inline, _resource_benchmark as bench
    graph = torch.fx.symbolic_trace(lambda a, b, h: a @ b + h)
    inputs = (torch.ones(2, 2, dtype=torch.bfloat16),) * 2 + (torch.ones(2, dtype=torch.bfloat16),)

    def reference(a, b, h):
        return (a.float() @ b.float() + h.float()).bfloat16()

    def compile_candidate(core, values, *, split=False, sdk_store=False):
        count = 2 if split else 1
        return reference, dict(resource_verified=True, kernel_evidence=[{}] * count, kernel_count=count)

    rounds = []
    def time_public(functions, values, *, seed):
        rounds.append(tuple(functions))
        if 'fp32_inline' in functions:
            costs = dict(fp32_inline=37300, fp32_inline_split=153, sdk_eager=169)
        else:
            costs = dict(fp32_inline_split=145, sdk_eager=114)
        return {k: [costs[k]] * 4 for k in functions}

    monkeypatch.setattr(inline, 'compile_body', compile_candidate)
    monkeypatch.setattr(bench, 'validate', lambda *args: {})
    monkeypatch.setattr(bench, 'GraphMeasurement', lambda *args, **kwargs:
                        SimpleNamespace(fresh_data_passed=True, close=lambda: None))
    winner, records, timings = inline.select(graph, inputs, reference, dict(atol=0., rtol=0.),
                                              public_timer=time_public, shared=reference)
    assert winner is None
    assert [r['status'] for r in records] == ['passed', 'passed']
    assert rounds[-2:] == [('fp32_inline_split', 'sdk_eager')] * 2
    assert timings['sdk_eager'] == [169, 169, 114, 114]


@pytest.mark.parametrize('m,n', [(256, 2048), (512, 1009)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_sharded_product_sum_propagates_sdk_special_values(p800, m, n, dtype):
    from triton.experimental.graph_fusion import _p800_reduce
    a = torch.ones(m, n, device='cuda', dtype=dtype)
    b = torch.ones(n, device='cuda', dtype=dtype)
    spec = dict(kind='mul_sum', m=m, n=n, dtype=str(dtype).split('.')[-1],
                final_round_only=dtype != torch.float32, eps=0., keepdim=False, nparam=0)
    fn, _ = _p800_reduce.compile(spec, (a, b))
    shard_rows = (m + 11) // 12
    for row in (0, 1, shard_rows-1, shard_rows, m-1):
        for col in (0, 511, 512, n-1):
            for value in (float('inf'), -float('inf'), float('nan')):
                a[row, col] = value
                actual, = fn((a, b))
                expect = (a.float() * b.float()).sum(-1).to(dtype)
                torch.testing.assert_close(actual.cpu(), expect.cpu(), atol=0., rtol=0., equal_nan=True)
                a[row, col] = 1.


def test_grouped_matrix_masks_incomplete_eight_tile_group(p800):
    from triton.experimental.graph_fusion._p800_compiler import resource_scope
    from triton.experimental.graph_fusion._p800_native_mm import _aligned_bias_kernel
    m, padded_m, n, k = 1025, 2048, 512, 64
    a = torch.zeros(padded_m, k, dtype=torch.bfloat16)
    a[:, 0] = torch.arange(padded_m) % 7
    b = torch.zeros(k, n, dtype=torch.bfloat16)
    b[0] = torch.arange(n) % 11
    bias = (torch.arange(n) % 5 - 3).bfloat16()
    want = (a[:m].float() @ b.float() + bias.float()).bfloat16()
    a, b, bias = (t.cuda() for t in (a[:m], b, bias))
    storage = torch.full((padded_m * n + 64,), -123., dtype=torch.bfloat16, device='cuda')
    out = storage[32:32 + m * n].view(m, n)
    with resource_scope('sdnn'):
        _aligned_bias_kernel()[(32, 1, 1)](a, b, bias, out, n, k, 128, 256, M=m,
                                           num_warps=4, num_stages=3, is_sdnn=True)
    assert torch.equal(out.cpu(), want)
    assert storage[:32].eq(-123.).all().item()
    assert storage[32 + m * n:].eq(-123.).all().item()


@pytest.mark.parametrize('mode', [10, 11])
@pytest.mark.parametrize('m,n', [(5, 17), (65, 1009)])
def test_bf16_odd_activation_keeps_output_guard(p800, mode, m, n):
    from triton.experimental.graph_fusion._p800_vector_rows import _vector_rows
    from triton.experimental.graph_fusion._resource_benchmark import assert_close
    g = torch.Generator().manual_seed(231)
    a = torch.randn(m, n, generator=g).bfloat16().cuda()
    b = torch.randn(n, generator=g).bfloat16().cuda()
    a[0, -1] = float('inf')
    a[-1, -1] = -float('inf')
    storage = torch.full((64 + m * n,), -123., dtype=a.dtype, device=a.device)
    out = storage[32:-32].view(m, n)
    _vector_rows[(12,)](a, b, out, m, n, mode, 32, (m + 11) // 12, num_warps=4, num_stages=1)
    z = a.float() + b.float()
    expected = torch.nn.functional.silu(z) if mode == 10 else torch.nn.functional.gelu(z, approximate='tanh')
    assert_close(out, expected, dict(atol=.03, rtol=.003), output_dtype=a.dtype)
    assert storage[:32].eq(-123.).all().item()
    assert storage[-32:].eq(-123.).all().item()


def test_bound_launch_packet_uses_live_tensor_addresses_and_stream(p800):
    import triton
    import triton.language as tl
    from triton.experimental.graph_fusion._p800_bound_launch import bind

    @triton.jit
    def copy_tile(a, out, size: tl.constexpr):
        pos = tl.arange(0, size)
        tl.store(out + pos, tl.load(a + pos) + 1.)

    a = torch.arange(64, dtype=torch.float32, device='cuda')
    out = torch.empty_like(a)
    with torch.no_grad():
        binary = copy_tile[(1,)](a, out, 64, num_warps=4, num_stages=1)
        run = bind(copy_tile, binary, (1, 1, 1), (a,), (64,), (), 'cluster')
        assert run is not None
        packed = run.__closure__[run.__code__.co_freevars.index('packed_launch')].cell_contents
        assert packed is not None and 'f06.dense_packet.v1' in repr(packed.__self__[-1])
        first = run(a)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            b = a + 7
            second = run(b)
        stream.synchronize()
        assert torch.equal(first.cpu(), torch.arange(64) + 1)
        assert torch.equal(second.cpu(), torch.arange(64) + 8)
        assert first.data_ptr() != second.data_ptr()
        shifted = torch.empty(65, device='cuda')[1:]
        assert run(shifted) is None


def test_bound_launch_packet_handles_repeated_bias(p800):
    import triton
    import triton.language as tl
    from triton.experimental.graph_fusion._p800_reduce import _bind_layernorm_launch

    @triton.jit
    def add_four(a, b, c, d, out, size: tl.constexpr):
        pos = tl.arange(0, size)
        tl.store(out + pos, tl.load(a + pos) + tl.load(b + pos) + tl.load(c + pos) + tl.load(d + pos))

    a = torch.arange(32, device='cuda', dtype=torch.float32).reshape(1, 32)
    bias = torch.ones(32, device=a.device, dtype=a.dtype)
    out = torch.empty_like(a)
    with torch.no_grad():
        binary = add_four[(1,)](a, bias, bias, bias, out, 32, num_warps=4, num_stages=1)
        # The non-affine LayerNorm entry guards two model inputs, then repeats the
        # bias in three launcher slots. Torch's TensorGuards expects unique inputs.
        launch = _bind_layernorm_launch(binary, add_four, 1, (a, bias), False)
        assert launch is not None
        first = launch((a, bias))
        second = launch((a, bias + 2))
        assert torch.equal(first.cpu(), torch.arange(32).reshape(1, 32) + 3)
        assert torch.equal(second.cpu(), torch.arange(32).reshape(1, 32) + 9)
        assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize('m', [383, 384, 385])
def test_half_silu_shared_bias_at_partial_batch_boundary(p800, m):
    from triton.experimental.graph_fusion._p800_vector_rows import _vector_rows
    from triton.experimental.graph_fusion._resource_benchmark import assert_close

    n = 2048
    gen = torch.Generator().manual_seed(148)
    a = torch.randn(m, n, generator=gen).half().cuda()
    bias = torch.randn(n, generator=gen).half().cuda()
    a[-1, :3] = torch.tensor([float('inf'), -float('inf'), float('nan')], device=a.device)
    storage = torch.full((m * n + 64,), -123., dtype=a.dtype, device=a.device)
    out = storage[32:-32].view(m, n)
    _vector_rows[(12,)](a, bias, out, m, n, 2, 512, 0, False, True, num_warps=4, num_stages=1)
    expected = torch.nn.functional.silu(a.float() + bias.float())
    assert_close(out, expected, dict(atol=.003, rtol=.003), output_dtype=a.dtype)
    assert storage[:32].eq(-123.).all().item()
    assert storage[-32:].eq(-123.).all().item()


def test_affine_layernorm_partial_row_group_writes_last_row(p800):
    from triton.experimental.graph_fusion import _p800_reduce

    m, n = 1025, 2048
    gen = torch.Generator().manual_seed(157)
    a, bias, weight, shift = (torch.randn(size, generator=gen).cuda() for size in [(m, n), (n,), (n,), (n,)])
    spec = dict(kind='add_layernorm_affine', dtype='float32', m=m, n=n, eps=.001, keepdim=False, nparam=2)
    with torch.no_grad():
        run, proof = _p800_reduce.compile(spec, (a, bias, weight, shift))
        actual, = run((a, bias, weight, shift))
        expected = torch.nn.functional.layer_norm(a + bias, (n,), weight, shift, .001)
        assert proof['tiles']['rows'] == 96
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=.0003, rtol=.0003)
