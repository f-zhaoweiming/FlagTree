"""Generic scalar-op legalizations; contains no matrix template."""
import re

COEFFICIENTS = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
P = 0.3275911


def lower_scalar_ops(mod, *, erf=True, clamp_max=False):
    from triton._C.libtriton import ir, passes
    ops = []
    mod.walk(ops.append)
    rewritten = []
    for op in ops:
        name = op.get_name()
        is_erf = name == 'math.erf' or (name == 'tt.extern_elementwise'
                                        and op.get_str_attr('symbol') in ('_ZN3xpu3erfEf', '__nv_erff'))
        is_max = name == 'arith.maximumf'
        if not ((erf and is_erf) or (clamp_max and is_max)):
            continue
        x = op.get_operand(0)
        ty = x.get_type()
        if not re.fullmatch(r'tensor<(?:[0-9]+x)+f32>', str(ty)):
            continue
        b = ir.builder(mod.context)
        b.set_insertion_point_after(op)

        def c(v):
            return b.create_splat(ty, b.get_fp32(v))

        if is_erf:
            # Approximate the erf primitive itself, not the tanh-GELU formula.
            # All arithmetic is fp32; CPU oracle and original device tolerances
            # are separate acceptance gates. Preserve +/-0, infinities and NaN.
            a = b.create_fabs(x)
            t = b.create_fdiv(c(1.0), b.create_fadd(c(1.0), b.create_fmul(c(P), a)))
            polynomial = c(COEFFICIENTS[-1])
            for coefficient in reversed(COEFFICIENTS[:-1]):
                polynomial = b.create_fadd(b.create_fmul(polynomial, t), c(coefficient))
            polynomial = b.create_fmul(polynomial, t)
            exponential = b.create_exp(b.create_fsub(c(0.0), b.create_fmul(a, a)))
            positive = b.create_fsub(c(1.0), b.create_fmul(polynomial, exponential))
            signed = b.create_select(b.create_fcmpOLT(x, c(0.0)), b.create_fsub(c(0.0), positive), positive)
            value = b.create_select(b.create_fcmpOEQ(a, c(0.0)), x, signed)
        else:
            # Equivalent maximum using the native clamp op, explicitly keeping
            # NaN propagation. Whether SDNN supports this form is tested below.
            value = b.create_clampf(x, op.get_operand(1), c(float('inf')), ir.PROPAGATE_NAN.ALL)
        op.get_result(0).replace_all_uses_with(value)
        rewritten.append(name)
    pm = ir.pass_manager(mod.context)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(mod, 'after_scalar_legalization')
    assert mod.verify()
    return rewritten


def cpu_oracle():
    import math
    import torch
    # Dense cancellation region, every BF16 value, and explicit special values.
    dense = torch.linspace(-10, 10, 1000001, dtype=torch.float32)
    words = (torch.arange(65536, dtype=torch.int64) << 16).to(torch.int32)
    inputs = torch.cat(
        (dense, words.view(torch.float32), torch.tensor([-0.0, 0.0, -float('inf'),
                                                         float('inf'),
                                                         float('nan')])))
    a = inputs.abs()
    t = 1.0 / (1.0 + P * a)
    poly = torch.full_like(t, COEFFICIENTS[-1])
    for coefficient in reversed(COEFFICIENTS[:-1]):
        poly = poly * t + coefficient
    positive = 1.0 - poly * t * torch.exp(-(a * a))
    actual = torch.where(inputs < 0.0, -positive, positive)
    actual = torch.where(a == 0.0, inputs, actual)
    ref = torch.erf(inputs.double()).float()
    finite = torch.isfinite(inputs)
    max_error = (actual[finite] - ref[finite]).abs().max().item()
    assert max_error < 6e-7, max_error
    assert torch.equal(torch.isnan(actual), torch.isnan(ref))
    assert torch.equal(actual[torch.isinf(inputs)], ref[torch.isinf(inputs)])
    zeros = inputs == 0
    assert torch.equal(actual[zeros].view(torch.int32), inputs[zeros].view(torch.int32))
    # Original exact-GELU formula on a dense finite input range, without changing
    # the approximate='none' contract or relaxing the original FP32 tolerance.
    x = dense * math.sqrt(2)
    erf_part = actual[:len(dense)]
    gelu = (0.5 * x) * (1.0 + erf_part)
    expected = torch.nn.functional.gelu(x, approximate='none')
    torch.testing.assert_close(gelu, expected, atol=3e-4, rtol=3e-4)
    return {
        'cases': len(inputs), 'max_abs_erf_error': max_error, 'exact_gelu_original_fp32_tolerance': 'passed',
        'special_values': 'passed', 'scope':
        'CPU float32 arithmetic only; native exp lookup and GPU execution unverified'
    }
