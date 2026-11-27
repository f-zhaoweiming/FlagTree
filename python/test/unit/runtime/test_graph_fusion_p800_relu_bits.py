"""Exercise the production BF16 bias/ReLU epilogue on a P800."""
import pytest


def test_bf16_bias_relu_retains_signed_zero():
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        pytest.skip("Requires PyTorch")
    from triton.experimental.graph_fusion import _p800_flat
    from triton.experimental.graph_fusion._resource_codegen import baidu_target

    if (baidu_target() or {}).get("arch") != 3:
        pytest.skip("Requires a P800")

    # The vendor addition maps some negative subnormals to -0; ReLU preserves
    # that representation. Tensor equality and tolerance checks cannot detect +0.
    with torch.no_grad():
        words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        x = words.view(torch.bfloat16).reshape(64, 1024).cuda()
        bias = torch.full((1024,), -0.0, dtype=x.dtype).to(x.device)
        spec = dict(kind="relu", m=64, n=1024, total=x.numel(), dtype="bfloat16")
        run, _ = _p800_flat.compile(spec, (x, bias))
        actual = run((x, bias))[0].cpu()
        expected = torch.relu(x + bias).cpu()
        assert torch.equal(actual.isnan(), expected.isnan())
        finite_or_inf = ~expected.isnan()
        assert torch.equal(actual.view(torch.int16)[finite_or_inf], expected.view(torch.int16)[finite_or_inf])
