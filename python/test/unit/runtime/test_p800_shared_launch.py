"""Regression cases for the static SDNN wrapper and final low-precision pack."""
from textwrap import dedent

import pytest
import torch


def test_single_sdnn_wrapper_recognizes_only_the_complete_generated_call():
    from triton.experimental.graph_fusion._p800_generated import _static_call

    source = dedent("""\
        class Runner:
            def call(self, args):
                a, b = args
                args.clear()
                assert_size_stride(a, (64,), (1,))
                assert_size_stride(b, (2, 64), (64, 1))
                with torch.cuda._DeviceGuard(0):
                    torch.cuda.set_device(0)
                    out = empty_strided_cuda((2, 64), (64, 1), torch.float32)
                    stream = get_raw_stream(0)
                    kernel.run(b, a, out, 128, stream=stream)
                    del a
                    del b
                return (out,)
    """)
    device, _, size, _, reorder, constants = _static_call(source, 'kernel')
    assert (device, size, constants) == (0, (2, 64), (128,))
    assert reorder(['bias', 'matrix']) == ('matrix', 'bias')
    for original, replacement in [('del a', 'a.zero_()'), ('(out,)', '(a,)'),
                                  ('out, 128', 'out, b.numel()'), ('(64, 1), torch.float32', '(1, 2), torch.float32')]:
        assert _static_call(source.replace(original, replacement), 'kernel') is None


@pytest.mark.parametrize('dtype,elements', [(torch.float16, 512), (torch.bfloat16, 512),
                                           (torch.bfloat16, 2048 * 2048)])
def test_accumulator_writeback_preserves_midpoints_and_signed_zero(dtype, elements):
    from triton.runtime import driver
    if (not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver'
            or driver.active.get_current_target().arch != 3):
        pytest.skip('requires P800')
    from triton.experimental.graph_fusion._p800_bf16_gelu import vector_finish

    with torch.no_grad():
        lo = torch.tensor([1., 1.+torch.finfo(dtype).eps, -1., -1.-torch.finfo(dtype).eps], dtype=dtype)
        hi = lo.nextafter(torch.full_like(lo, float('inf')))
        mid = .5*(lo.float()+hi.float())
        special = torch.tensor([0., -0., -float('inf'), float('inf'), float('nan'), 65520.])
        # P800 casts NaN payloads from their original FP32 words. Some BF16
        # payloads round across an exponent boundary and change classification.
        payloads = torch.tensor([0x7f800001, 0x7f807fff, 0x7f808000, 0x7fffffff,
                                 0xff800001, 0xff807fff, 0xff808000, 0xffffffff],
                                dtype=torch.int64).to(torch.int32).view(torch.float32)
        pattern = torch.cat((mid, mid.nextafter(lo.float()), mid.nextafter(hi.float()), special, payloads))
        source = pattern.repeat((elements + pattern.numel() - 1) // pattern.numel())[:elements].cuda()
        result = torch.empty_like(source, dtype=dtype)
        width = 512 if elements == 2048 * 2048 else 32
        vector_finish[(12,)](source, result, source.numel(), width, 0, num_warps=4, num_stages=1)
        actual, expected = result.cpu(), source.to(dtype).cpu()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize('shape,symbol', [((512, 1024), 'kl3_silu_balanced'),
                                          ((2048, 2048), 'kl3_silu_pipeline')])
@pytest.mark.parametrize('dtype', (torch.float32, torch.float16, torch.bfloat16))
def test_inductor_shared_silu_uses_its_selected_schedule(shape, symbol, dtype):
    from triton.runtime import driver
    if (not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver'
            or driver.active.get_current_target().arch != 3):
        pytest.skip('requires P800')
    from torch._inductor.utils import run_and_get_code

    def operation(x, bias):
        return torch.nn.functional.silu(x + bias)

    with torch.no_grad():
        torch._dynamo.reset()
        x = torch.randn(shape, device='cuda', dtype=dtype)
        bias = torch.randn(shape[1], device='cuda', dtype=dtype)
        compiled = torch.compile(operation, backend='inductor', fullgraph=True, dynamic=False)
        y, code = run_and_get_code(compiled, x, bias)
        if dtype == torch.float32:
            assert any(f'tle.raw.call({symbol},' in src for src in code)
        expected = operation(x.float(), bias.float()).to(dtype)
        torch.testing.assert_close(y.cpu(), expected.cpu(), atol=.03 if dtype == torch.bfloat16 else .003,
                                   rtol=.003, equal_nan=True)


def test_grouped_product_sdk_store_rounds_only_after_bias():
    from triton.runtime import driver
    if (not torch.cuda.is_available() or type(driver.active).__module__ != 'triton.backends.xpu.driver'
            or driver.active.get_current_target().arch != 3):
        pytest.skip('requires P800')
    from triton.experimental.graph_fusion._resource_autotune import candidate_plans, compile_candidate

    with torch.no_grad():
        gm = torch.fx.symbolic_trace(lambda a, b, bias: a @ b + bias)
        a = torch.zeros((2048, 64), dtype=torch.bfloat16, device='cuda')
        b = torch.zeros((64, 2048), dtype=a.dtype, device=a.device)
        bias = torch.full_like(b[0], -1.)
        a[:, 0], a[:, 1], b[:2] = 1., 2.**-9, 1.
        inputs = (a, b, bias)
        name = 'fp32_product_sdk_store'
        assert all(p.name != name for p in candidate_plans(gm, inputs, single_kernel=True))
        plan = next(p for p in candidate_plans(gm, inputs) if p.name == name)
        fn, proof = compile_candidate(gm, inputs, plan)
        value = fn(*inputs)
        expected = (a.float() @ b.float() + bias.float()).bfloat16()
        assert torch.equal(value, expected) and value[0, 0].item() == 2.**-9
        assert proof['external_calls'] == 1 and proof['kernel_count'] is None
        assert not proof['resource_verified'] and proof['regions'][0]['resource_verified']
