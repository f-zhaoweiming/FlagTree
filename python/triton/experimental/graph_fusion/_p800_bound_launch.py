"""Warm launch for dense KL3 kernels with tensor pointers and fixed integer arguments."""


def _pack_launch(entry, header, arguments, slots, counts, allocator, layout, stream, device):
    import sys
    import torch
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 10)
            or not torch.__version__.startswith('2.9.')):
        return None
    from ._resource_entry import _frame_extension
    module = _frame_extension()
    bind_dense = getattr(module, 'bind_dense', None)
    if bind_dense is None:
        return None
    data_ptr = torch.Tensor.data_ptr
    config = (entry, header + arguments, slots, counts, data_ptr, allocator, layout, stream, device)
    native = getattr(module, 'bind_native_dense', None)
    if native is not None and data_ptr is getattr(getattr(torch._C, 'TensorBase', None), 'data_ptr', None):
        return native(config, torch.Tensor, torch.nn.Parameter)
    return bind_dense(config)


def _current_device(active):
    # Binding follows a successful real launch. On this driver current_device
    # calls the same Torch primitive after lazy initialization on every hit.
    import torch
    query = active.get_current_device
    get_device = getattr(torch._C, '_cuda_getDevice', None)
    bad_fork = getattr(torch.cuda, '_is_in_bad_fork', None)
    if (get_device is not None and bad_fork is not None
            and type(active).__module__ == 'triton.backends.xpu.driver' and torch.cuda.is_initialized()
            and query is getattr(type(active), '_get_current_xpu_device', None)):
        def current():
            return -1 if bad_fork() else get_device()
        return current
    return query


def bind(kernel, binary, grid, inputs, output_shape, scalar_args, resource, *, output_dtype=None, inplace_tail=None):
    import sys
    import torch
    from triton import knobs
    from triton.runtime import driver

    runner, metadata = binary.run, binary.metadata
    guards = torch._C._dynamo.guards
    dtype = inputs[0].dtype
    output_dtype = dtype if output_dtype is None else output_dtype
    pointer_types = {torch.float16: '*fp16', torch.float32: '*fp32', torch.bfloat16: '*bf16'}
    ptr_type = pointer_types.get(dtype)
    out_ptr_type = pointer_types.get(output_dtype)
    signature = tuple(t for t in getattr(runner, '_signature', {}).values() if t != 'constexpr')
    if (not ptr_type or type(runner).__module__ != 'triton.backends.xpu.driver'
            or metadata.xpu_arch != 3 or bool(metadata.is_sdnn) != (resource == 'sdnn')
            or metadata.name != kernel.fn.__name__ or getattr(metadata, 'tensor_args', None) not in ([], ())
            or signature != (ptr_type,) * len(inputs) + (out_ptr_type,) + ('i32',) * len(scalar_args)
            or not hasattr(guards, '_empty_strided_cuda') or not hasattr(guards, 'TensorGuards')
            or any(type(t) is not torch.Tensor or t.device != inputs[0].device or t.dtype != dtype for t in inputs)
            or inputs[0].device.type != 'cuda'):
        return None
    device = inputs[0].device.index
    layout_check = guards.TensorGuards(*inputs, dynamic_dims_sizes=None, dynamic_dims_strides=None).check
    current, stream = _current_device(driver.active), driver.active.get_current_stream
    launch, packed = runner.launch, binary.packed_metadata
    empty = guards._empty_strided_cuda
    out_stride = (1,) if len(output_shape) == 1 else (output_shape[1], 1)
    runtime = knobs.runtime
    jit = sys.modules[type(kernel).__module__]
    prism = getattr(jit, '_flagprism', None)
    distributed = getattr(jit, 'DistributedRtContext', None)
    launch_module = sys.modules[type(runner).__module__]
    packed_launch = None

    if inplace_tail is not None:
        tail_kernel, tail_binary, tail_grid, tail_args = inplace_tail
        tail = tail_binary.run
        tail_signature = tuple(t for t in tail._signature.values() if t != 'constexpr')
        if (type(tail) is not type(runner) or tail_binary.metadata.xpu_arch != 3
                or tail_binary.metadata.name != tail_kernel.fn.__name__
                or getattr(tail_binary.metadata, 'tensor_args', None) not in ([], ())
                or tail_signature != (out_ptr_type, out_ptr_type) + ('i32',) * len(tail_args)):
            return None
        tail_launch, tail_metadata = tail.launch, tail_binary.packed_metadata
    else:
        count = len(inputs) + 1
        packed_launch = _pack_launch(
            launch, (*grid, None, binary.function, packed, None, None, None),
            (None,) * count + tuple(scalar_args), tuple(range(9, 9 + count)), (None,) * count,
            empty, (output_shape, out_stride, output_dtype), stream, device)

    def run(*xs):
        enter, leave = runtime.launch_enter_hook, runtime.launch_exit_hook
        if (current() != device or torch.is_grad_enabled() or torch._C._len_torch_dispatch_stack()
                or torch._C._is_torch_function_mode_enabled() or kernel.pre_run_hooks or kernel.used_global_vals
                or kernel.debug or runtime.debug or getattr(runtime, 'disable_launch_fast_path', False)
                or knobs.compilation.instrumentation_mode or getattr(enter, 'calls', enter)
                or getattr(leave, 'calls', leave)
                or getattr(launch_module, '_PRINT_TRITON_FUNC', False)
                or (prism is not None and prism._registered_components())
                or (distributed is not None and distributed._init_count and distributed().is_lite_mode)
                or not layout_check(*xs)):
            return None
        if inplace_tail is not None and (tail_kernel.pre_run_hooks or tail_kernel.used_global_vals or tail_kernel.debug):
            return None
        if packed_launch is not None:
            return packed_launch(*xs)
        # Scaled kernels require a live numel and were declined at bind time.
        pointers = tuple(x.data_ptr() for x in xs)
        if any(p & 15 for p in pointers):
            return None
        out = empty(output_shape, out_stride, output_dtype)
        last = out.data_ptr()
        if last & 15:
            return None
        current_stream = stream(device)
        launch(grid[0], grid[1], grid[2], current_stream, binary.function, packed, None, None, None,
               *pointers, last, *scalar_args)
        if inplace_tail is not None:
            tail_launch(*tail_grid, current_stream, tail_binary.function, tail_metadata, None, None, None,
                        last, last, *tail_args)
        return out

    return run
