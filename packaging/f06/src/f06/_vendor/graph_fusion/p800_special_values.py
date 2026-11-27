"""Explicit, correctness-gated preparation for the F06 SDNN experiment.

The private lowering module is also used by diagnostics, including intentionally
ineligible configurations. Use this entry to prepare a candidate for comparison.
It does not replace resource calibration or change the default backend.
"""


def _admit(gm, inputs):
    import torch
    from ._patterns import default_registry, match_graph

    calls = tuple(n for n in gm.graph.nodes if n.op not in ('placeholder', 'get_attr', 'output'))
    selected, _ = match_graph(gm, default_registry())
    if len(selected) != 1 or selected[0][1].nodes != calls:
        raise ValueError('expected one complete F06 activation or multiply-sum graph')
    match = selected[0][1]
    placeholders = tuple(n for n in gm.graph.nodes if n.op == 'placeholder')
    if (tuple(match.inputs) != placeholders or len(inputs) != len(placeholders)
            or not all(isinstance(x, torch.Tensor) for x in inputs)):
        raise ValueError('expected the measured tensor argument order')
    if len({x.dtype for x in inputs}) != 1 or inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('mixed or unsupported input precision')
    if any(not x.is_contiguous() for x in inputs):
        raise ValueError('only contiguous input layouts were measured')

    if match.kind in ('mul_sum', 'mul_sum_keepdim'):
        if inputs[0].dtype != torch.float16:
            raise ValueError('FP32/BF16 SDNN Mul+Sum failed the extreme-value oracle')
        if (tuple(inputs[0].shape) not in ((16, 128), (32, 256))
                or tuple(inputs[1].shape) != (inputs[0].shape[1],)):
            raise ValueError('no admitted SDNN reduction for this shape')
    elif match.kind in ('matmul_bias_silu', 'matmul_bias_gelu_tanh'):
        a, b, bias = inputs
        if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or tuple(bias.shape) != (b.shape[1],)
                or (a.shape[0], b.shape[1], a.shape[1]) not in (
                    (16, 128, 64), (32, 256, 64), (64, 1024, 64),
                    (128, 1024, 128), (512, 1024, 1024), (2048, 2048, 2048))):
            raise ValueError('no measured F06 activation for this shape')
    else:
        raise ValueError('no experimental repair for ' + match.kind)
    return match.kind


def _special_oracle(kind, fn, inputs, tolerance):
    import torch
    from . import _resource_benchmark as bench

    cpu = [torch.zeros_like(x, device='cpu') for x in inputs]
    if kind.startswith('matmul_bias_'):
        bias = cpu[2]
        bias[:7] = bias.new_tensor([float('inf'), -float('inf'), float('nan'), -0., 0.,
                                     -torch.finfo(bias.dtype).max, torch.finfo(bias.dtype).max])
        # Bias is added to a zero GEMM (including its signed-zero behavior).
        value = torch.zeros_like(bias) + bias
        if kind == 'matmul_bias_silu':
            value = torch.nn.functional.silu(value)
        else:
            value = torch.nn.functional.gelu(value, approximate='tanh')
        want = value.expand(cpu[0].shape[0], -1)
        got = fn(*(value.to(x.device) for value, x in zip(cpu, inputs)))
        bench.assert_close(got, want, tolerance)
    else:
        first, second = cpu
        # A non-finite broadcast multiplier would contaminate unrelated rows.
        # Test one pair per call, including both operand orders.
        tiny = torch.finfo(first.dtype).tiny
        for left, right in ((float('inf'), 0.), (0., float('inf')),
                            (float('nan'), 0.), (0., float('nan')),
                            (float('inf'), tiny), (tiny, -float('inf'))):
            first.zero_()
            second.fill_(1)
            first[0, 0], second[0] = left, right
            want = (first * second).sum(-1, keepdim=kind.endswith('keepdim'))
            got = fn(*(value.to(x.device) for value, x in zip(cpu, inputs)))
            bench.assert_close(got, want, tolerance)


def prepare_special_candidate(gm, inputs, *, tolerance):
    """Return a validated, prepared native SDNN entry, or reject the candidate.

    Preparation is outside both graph capture and timed ordinary calls. Bind
    this fixed graph and these input metadata exactly as for calibrate_resources.
    The caller still compares against eager and native Inductor before selection.
    """
    import os
    import torch
    from . import _resource_autotune as auto, _resource_benchmark as bench
    from ._p800_special_values import experimental_scope
    from ._resource_codegen import baidu_target

    inputs = tuple(inputs)
    kind = _admit(gm, inputs)
    if (torch.is_grad_enabled() or any(x.requires_grad for x in inputs)
            or any(x.requires_grad for x in gm.parameters())):
        raise RuntimeError('requires no-grad inference')
    if (len({x.device for x in inputs}) != 1 or inputs[0].device.type != 'cuda'
            or baidu_target() is None or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError('prepare on the native P800 outside graph capture')
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            or os.environ.get('XMLIR_MATMUL_FAST_MODE') != '0' or os.environ.get('TRITONXPU_DTYPE_CONVERT') != '1'):
        raise RuntimeError('configure the original full-precision P800 benchmark math modes before preparation')
    low = inputs[0].dtype
    if low == torch.float32:
        limits = (3e-4, 3e-4)
    elif low == torch.float16 or (kind.startswith('matmul_bias_') and inputs[0].shape[0] > 64):
        limits = (3e-3, 3e-3)
    else:
        limits = (3e-2, 3e-3)
    if set(tolerance) != {'atol', 'rtol'} or not all(
            type(tolerance[k]) in (int, float) and 0 <= tolerance[k] <= cap
            for k, cap in zip(('atol', 'rtol'), limits)):
        raise ValueError('expected tolerances at least as strict as the F06 reference')
    plan = next(p for p in auto.candidate_plans(gm, inputs) if p.name == 'sdnn')
    with experimental_scope('optimized'):
        fn, proof = auto.compile_candidate(gm, inputs, plan)
    bench.validate(fn, gm, inputs, tolerance)
    _special_oracle(kind, fn, inputs, tolerance)
    return fn, proof
