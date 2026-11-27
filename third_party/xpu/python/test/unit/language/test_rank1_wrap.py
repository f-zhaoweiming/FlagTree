"""P800 device regressions: exact modulo-load bits and allocation guards."""
import json

import pytest
import triton
import triton.language as tl
libdevice = pytest.importorskip("triton.language.extra.xpu.libdevice")


@triton.jit
def wrapped_copy(src, dst, PERIOD: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = x < LENGTH
    value = tl.load(src + x % PERIOD, mask)
    tl.store(dst + x, value, mask)


@pytest.fixture(scope="module")
def p800():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("P800 device required")
    from triton.backends import backends
    from triton.runtime import driver
    xpu = backends.get("xpu")
    target = driver.active.get_current_target()
    # torch_xmlir may report the generic name "GPU" for a real P800. Identify
    # the XPU driver and its hardware model instead of that display string.
    if xpu is None or not isinstance(driver.active, xpu.driver) or target.arch != 3:
        pytest.skip("P800 device required")
    return torch


CASES = [(7, 19, 16), (17, 19, 16), (32, 19, 16), (7, 16, 16), (17, 1, 16), (2, 37, 16), (4099, 5125, 512),
         (4099, 5125, 1024)]


@pytest.mark.parametrize("period,length,block", CASES)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_wrapped_copy_bits_and_guards(p800, tmp_path, period, length, block, dtype):
    torch = p800
    dtype = getattr(torch, dtype)
    integer = torch.int32 if dtype == torch.float32 else torch.int16
    # Retain enough physical storage even for the old, incorrect unwrapped load.
    # Reading beyond the logical source must fail the assertion, not harm a GPU.
    grid = triton.cdiv(length, block) + 1
    host = torch.full((max(period, grid * block) + 64, ), -1000, dtype=dtype)
    special = {
        torch.float32: [0, -2147483648, 2139095040, -8388608, 2143289344, 1, -2147483647, 8388607],
        torch.float16: [0, -32768, 31744, -1024, 32256, 1, -32767, 1023],
        torch.bfloat16: [0, -32768, 32640, -128, 32704, 1, -32767, 127],
    }
    checks = []
    for kind in ("ordinary", "special", "one_hot"):
        if kind == "ordinary":
            host[:period] = (100 + torch.arange(period) % 31).to(dtype)
        elif kind == "special":
            bits = special[dtype]
            host[:period] = torch.tensor((bits * ((period + len(bits) - 1) // len(bits)))[:period],
                                         dtype=integer).view(dtype)
        else:
            host[:period] = 0
            host[0] = 1
        src = host.to("cuda")
        dst = torch.full((grid * block + 64, ), -777, dtype=dtype, device="cuda")
        kernel = wrapped_copy[(grid, )](src[:period], dst[:length], PERIOD=period, LENGTH=length, BLOCK=block,
                                        is_sdnn=True)
        torch.cuda.synchronize()
        actual = dst.cpu()
        expected = host[torch.arange(length) % period]
        bad = (actual[:length].view(integer) != expected.view(integer)).nonzero().flatten().tolist()
        source_ok = torch.equal(src.cpu().view(integer), host.view(integer))
        guard_ok = bool(torch.all(actual[length:] == -777))
        checks.append(dict(kind=kind, mismatch_indices=bad, source_unchanged=source_ok, guard_unchanged=guard_ok))
        (tmp_path / "result.json").write_text(
            json.dumps(
                dict(period=period, length=length, block=block, dtype=str(dtype), checks=checks,
                     is_sdnn=kernel.metadata.is_sdnn), indent=2))
        assert kernel.metadata.is_sdnn
        assert not bad and source_ok and guard_ok


@triton.jit
def _wrapped_full_tile(a, b, out, PERIOD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(a + i) + tl.load(b + i % PERIOD))


def test_unmasked_periodic_broadcast(p800):
    torch = p800
    period, block = 1024, 8192
    a = torch.zeros(2 * block, device='cuda')
    storage = torch.full_like(a, -42.)
    storage[:period].copy_(torch.arange(period, dtype=torch.float32))
    output = torch.empty_like(a)
    _wrapped_full_tile[(2,)](a, storage[:period], output, PERIOD=period, BLOCK=block, is_sdnn=True)
    torch.testing.assert_close(output.cpu(), torch.arange(period).float().repeat(16), atol=0, rtol=0)


@triton.jit
def _wrapped_bias_exact_gelu(a, bias, out, PERIOD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + i) + tl.load(bias + i % PERIOD)
    tl.store(out + i, (x * 0.5) * (1.0 + libdevice.erf(x * 0.7071067811865476)))


@pytest.mark.parametrize("period,block", [(2048, 1024), (513, 1024)])
def test_f06_exact_gelu_preserves_wrapped_bias(p800, period, block):
    from triton.experimental.graph_fusion._p800_compiler import resource_scope
    torch = p800
    length = 4 * block
    a = torch.linspace(-2, 2, length).cuda()
    bias_storage = torch.full((length,), 12345.)
    bias_storage[:period] = torch.linspace(-1, 1, period)
    bias_storage[1:4] = torch.tensor([float('nan'), float('inf'), -float('inf')])
    bias = bias_storage.cuda()
    out = torch.full((length + 8,), -777., device='cuda')
    with resource_scope('sdnn'):
        kernel = _wrapped_bias_exact_gelu[(length // block,)](a, bias[:period], out[:length], period, block)
    assert kernel.metadata.is_sdnn
    assert json.loads(kernel.metadata.native_exact_gelu_restored)
    assert json.loads(kernel.metadata.native_gelu_lut_refinement)
    actual = out.cpu()
    expected = torch.nn.functional.gelu(a + bias_storage[torch.arange(length) % period].cuda()).cpu()
    torch.testing.assert_close(actual[:length], expected, equal_nan=True, rtol=3e-4, atol=3e-4)
    assert torch.all(actual[length:] == -777.)
