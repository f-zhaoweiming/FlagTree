"""Regression cases for FP32 arithmetic between P800 precision boundaries."""
import pytest
import torch
import torch.nn.functional as F
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def rewritten(fn, inputs):
    from triton.experimental.graph_fusion._p800_fp32_region import rewrite
    gm = torch.fx.symbolic_trace(fn)
    FakeTensorProp(gm).propagate(*inputs)
    assert rewrite(gm.graph)
    gm.recompile()
    return gm


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_round_once_all_templates(dtype):
    torch.manual_seed(817)
    a, b, h = torch.randn(8, 4).to(dtype), torch.randn(4, 16).to(dtype), torch.randn(16).to(dtype)
    x, w, shift = torch.randn(8, 16).to(dtype), torch.randn(16).to(dtype), torch.randn(16).to(dtype)
    cases = [
        (lambda a,b,h: a @ b + h, (a,b,h)),
        (lambda a,b,h: (a @ b + h).relu(), (a,b,h)),
        (lambda a,b,h: F.gelu(a @ b + h), (a,b,h)),
        (lambda a,b,h: F.gelu(a @ b + h, approximate='tanh'), (a,b,h)),
        (lambda a,b,h: F.silu(a @ b + h), (a,b,h)),
        (lambda x,h: (x+h).softmax(-1), (x,h)), (lambda x,h: F.silu(x+h), (x,h)),
        (lambda x,h: F.gelu(x+h), (x,h)), (lambda x,h: F.gelu(x+h, approximate='tanh'), (x,h)),
        (lambda x,h: F.layer_norm(x+h, (16,)), (x,h)),
        (lambda x,h,w,s: F.layer_norm(x+h, (16,), w, s, .001), (x,h,w,shift)),
        (lambda x,h: (x*h).sum(-1), (x,h)), (lambda x,h: (x*h).sum(-1, keepdim=True), (x,h)),
    ]
    for fn, args in cases:
        gm = rewritten(fn, args)
        out = gm(*args)
        want = fn(*(t.float() for t in args)).to(dtype)
        assert torch.equal(out, want)
        assert sum(n.target is torch.ops.flagtree_p800.round_output.default for n in gm.graph.nodes) == 1


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_bias_cancellation_and_observable_intermediate(dtype):
    eps = 2.**(-12 if dtype == torch.float16 else -9)
    a = torch.tensor([[1., eps]], dtype=dtype)
    b = torch.ones(2, 1, dtype=dtype)
    h = torch.full((1,), -1., dtype=dtype)

    def chain(a,b,h):
        z = a @ b
        return z, z+h, z

    gm = rewritten(chain, (a,b,h))
    product, total, duplicate = gm(a,b,h)
    assert product.item() == 1 and duplicate is product
    assert total.dtype == dtype and total.item() == eps

    def explicit(a,b,h):
        z = (a @ b).to(dtype)
        return z+h

    assert rewritten(explicit, (a,b,h))(a,b,h).item() == 0


def test_fp32_input_guard():
    from triton.experimental.graph_fusion._p800_fp32_region import rewrite
    gm = torch.fx.symbolic_trace(lambda a,b: F.silu(a+b))
    FakeTensorProp(gm).propagate(torch.randn(4,8),torch.randn(8))
    assert not rewrite(gm.graph)


def test_single_result_shortcut_rejects_duplicate_outputs():
    from triton.experimental.graph_fusion._p800_fp32_region import inner_graph

    def shared(a, b):
        value = a + b
        return (value, value)

    gm = rewritten(shared, (torch.ones(4).half(), torch.ones(4).half()))
    assert inner_graph(gm) is None


def test_native_explicit_boundary_and_autocast():
    from triton.experimental.graph_fusion._p800_fp32_region import rewrite

    a = torch.tensor([[1., 2.**-9]], dtype=torch.bfloat16)
    b = torch.ones(2, 1, dtype=a.dtype)
    bias = torch.full((1,), -1., dtype=a.dtype)
    for explicit in (False, True):
        def model(a, b, bias):
            product = a @ b
            if explicit:
                product = product.bfloat16()
            return product + bias

        with torch.autocast('cpu', dtype=torch.bfloat16):
            gm = torch.fx.symbolic_trace(model)
            FakeTensorProp(gm).propagate(a, b, bias)
            assert rewrite(gm.graph, native=True)
            gm.recompile()
            result = gm(a, b, bias)
        assert result.dtype == a.dtype and result.item() == (0 if explicit else 2.**-9)


def test_bf16_sum_uses_fp32_product_and_sdk_reduction():
    # (1 + u)^2 - (1 + 2*u) exposes a premature BF16 product store.
    u = 2.**-7
    a = torch.tensor([[128 * (1 + u), -128]], dtype=torch.bfloat16)
    b = torch.tensor([1 + u, 1 + 2*u], dtype=a.dtype)
    gm = rewritten(lambda a, b: (a * b).sum(-1, keepdim=True), (a, b))
    assert sum(n.target == torch.ops.flagtree_p800.fp32_row_sum.default for n in gm.graph.nodes) == 1
    assert gm(a, b).item() == 128*u*u
