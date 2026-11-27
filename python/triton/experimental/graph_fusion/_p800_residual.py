"""Dense residual LayerNorm for automatically extracted model regions."""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.experimental.tle as tle


@tle.raw.dialect('xpu3', file='_p800_layernorm.xpu')
def kl3_residual_layernorm(a, residual, weight, shift, out, first, rows, n, segment, eps, affine, dtype):
    ...


@triton.jit
def residual_layernorm(A, B, W, S, OUT, M: tl.constexpr, N: tl.constexpr,
                       EPS: tl.constexpr, AFFINE: tl.constexpr, ROWS: tl.constexpr):
    first = tl.program_id(0) * ROWS
    count = tl.minimum(ROWS, M - first)
    dtype = 2 if A.dtype.element_ty == tl.bfloat16 else A.dtype.element_ty == tl.float16
    # The admitted aligned vendor path consumes a complete row for N <= 768.
    # The 256-column small-batch segment order belongs to a different path
    # and changes the FP32 sum/square order even with the right divisor.
    segment = 768
    tle.raw.call(kl3_residual_layernorm, (A, B, W, S, OUT, first, count,
        tl.full((), N, tl.int32), tl.full((), segment, tl.int32),
        tl.full((), EPS, tl.float32), tl.full((), AFFINE, tl.int32),
        tl.full((), dtype, tl.int32)))


def _bind(binary, operands, shape):
    from ._p800_bound_launch import _current_device, _pack_launch
    from triton.runtime import driver
    from triton import knobs
    runner = binary.run
    if (type(runner).__module__ != 'triton.backends.xpu.driver'
            or binary.metadata.is_sdnn or getattr(binary.metadata, 'tensor_args', None) not in ([], ())):
        return None
    guards = torch._C._dynamo.guards
    device, dtype = operands[0].device.index, operands[0].dtype
    stride = tuple(operands[0].stride())
    programs = (math.prod(shape[:-1]) + 63) // 64
    current, stream = _current_device(driver.active), driver.active.get_current_stream
    guard = guards.TensorGuards(*operands, dynamic_dims_sizes=None, dynamic_dims_strides=None).check
    packed = _pack_launch(runner.launch,
        (programs, 1, 1, None, binary.function, binary.packed_metadata, None, None, None),
        (None,)*5, (9,10,11,12,13), (None,)*5, guards._empty_strided_cuda,
        (shape,stride,dtype), stream, device)
    if packed is None:
        return None
    def run(*values):
        if (torch.is_grad_enabled() or current()!=device or not guard(*values)
                or torch._C._len_torch_dispatch_stack() or torch._C._is_torch_function_mode_enabled()
                or knobs.runtime.debug or knobs.runtime.disable_launch_fast_path
                or knobs.compilation.instrumentation_mode
                or residual_layernorm.pre_run_hooks or residual_layernorm.debug
                or getattr(knobs.runtime.launch_enter_hook, 'calls', knobs.runtime.launch_enter_hook)
                or getattr(knobs.runtime.launch_exit_hook, 'calls', knobs.runtime.launch_exit_hook)):
            return None
        return packed(*values)
    return run


def prepare(kind, options, inputs):
    if kind not in ('add_layernorm', 'add_layernorm_affine') or len(inputs) not in (2,4):
        return None
    a, b = inputs[:2]
    if (a.device.type!='cuda' or a.ndim<2 or a.shape!=b.shape or not a.is_contiguous()
            or not b.is_contiguous() or a.dtype not in (torch.float32,torch.float16,torch.bfloat16)
            or b.dtype!=a.dtype or b.device!=a.device):
        return None
    n, m = a.shape[-1], math.prod(a.shape[:-1])
    if (type(n) is not int or type(m) is not int or not 0<n<=768 or n % 64 != 0 or not 0<m*n<=0x7fffffff
            or options.get('normalized_shape')!=(n,)):
        return None
    affine = len(inputs)==4
    if affine and any(x.shape!=(n,) or x.dtype!=a.dtype or x.device!=a.device or not x.is_contiguous() for x in inputs[2:]):
        return None
    eps = options['eps']
    if not math.isfinite(eps) or eps<0:
        return None
    proof=dict(route='residual_layernorm', resource='cluster', input_shape=tuple(a.shape), dtype=str(a.dtype),
               compute_dtype='float32', output_dtype=str(a.dtype), final_round_only=True,
               eps=eps, kernel_count=1, resource_verified=False)
    warm = None
    def run(*xs):
        nonlocal warm
        operands = xs if affine else (xs[0],xs[1],xs[1],xs[1])
        if warm is not None:
            out = warm(*operands)
            if out is not None:
                return out
        out = torch.empty_like(xs[0])
        binary = residual_layernorm[((m+63)//64,)](*operands,out,m,n,eps,affine,64,num_warps=4,num_stages=1)
        if not proof['resource_verified']:
            from ._resource_codegen import bind_evidence
            bind_evidence(proof,binary)
            warm = _bind(binary,operands,tuple(a.shape))
            proof['prepared_launch']=warm is not None
        return out
    def reference(*xs):
        return F.layer_norm(xs[0].float()+xs[1].float(),(n,),
                            xs[2].float() if affine else None,
                            xs[3].float() if affine else None,eps)
    return run,reference,proof
