"""Specialize the verified single-launch Torch 2.9 SDNN inference wrapper."""
import ast
import inspect
import sys
import types
from operator import itemgetter


def _static_call(source, binding):
    """Accept only unpack, layout guards, one allocation/launch and a tuple return."""
    methods = [n for c in ast.parse(source).body if isinstance(c, ast.ClassDef) and c.name == 'Runner'
               for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'call']
    if len(methods) != 1:
        return None
    body = methods[0].body
    try:
        unpack = body[0].targets[0]
        names = [v.id for v in unpack.elts]
        if not (2 <= len(names) <= 3 and len(set(names)) == len(names)):
            return None
        block = body[-2]
        dev = ast.literal_eval(block.items[0].context_expr.args[0])
        allocation, get_stream, issue = block.body[1:4]
        out, stream = allocation.targets[0].id, get_stream.targets[0].id
        shape, stride = [ast.literal_eval(n) for n in allocation.value.args[:2]]
        if (type(dev) is not int or type(shape) is not tuple or len(shape) != 2
                or any(type(n) is not int or n <= 0 for n in shape) or stride != (shape[1], 1)):
            return None
        layouts = [(ast.literal_eval(n.value.args[1]), ast.literal_eval(n.value.args[2])) for n in body[2:-2]]
        if len(layouts) != len(names):
            return None
        args = issue.value.args
        tensor_names = [node.id for node in args[:len(names)+1]]
        if tensor_names[-1] != out or sorted(tensor_names[:-1]) != sorted(names):
            return None
        constants = tuple(ast.literal_eval(n) for n in args[len(names)+1:])
        if any(type(n) is not int for n in constants):
            return None
    except (AttributeError, ValueError, IndexError, TypeError):
        return None
    # Comparing the complete AST also excludes aliases, extra operations,
    # mutations, alternate returns, and symbolic dimensions.
    statements = [f"{', '.join(names)} = args", "args.clear()"]
    statements += [f"assert_size_stride({name}, {size!r}, {step!r})"
                   for name, (size, step) in zip(names, layouts)]
    arguments = ', '.join(tensor_names + [repr(n) for n in constants])
    statements += [f"with torch.cuda._DeviceGuard({dev}):", f"    torch.cuda.set_device({dev})",
                   f"    {out} = empty_strided_cuda({shape!r}, {stride!r}, torch.float32)",
                   f"    {stream} = get_raw_stream({dev})", f"    {binding}.run({arguments}, stream={stream})"]
    statements += [f"    del {name}" for name in names]
    statements += [f"return ({out},)"]
    expected = ast.parse('\n'.join(statements)).body
    if [ast.dump(n) for n in body] != [ast.dump(n) for n in expected]:
        return None
    return dev, layouts, shape, stride, itemgetter(*(names.index(n) for n in tensor_names[:-1])), constants


def _packet_for_launcher(launcher, binary, layouts, shape, stride, constants, empty, stream, dev, *, dtype=None):
    """Resolve the static Inductor launch using opaque argument markers, once."""
    from ._p800_bound_launch import _pack_launch
    import torch
    runner = binary.run
    scope = getattr(launcher, '__globals__', {})
    code = getattr(launcher, '__code__', None)
    from triton import knobs
    from triton.compiler.compiler import CompiledKernel
    if type(binary) is not CompiledKernel or 'launch_metadata' in vars(binary):
        return None
    names = {'runner', 'function', 'metadata', 'bin', 'launch_metadata', 'launch_enter_hook', 'launch_exit_hook'}
    for name in ('launch_enter_hook', 'launch_exit_hook'):
        hook = scope.get(name)
        if hook is not None and (type(hook) is not knobs.HookChain or hook.calls
                                 or hook is not getattr(knobs.runtime, name)):
            return None
    if (not isinstance(launcher, types.FunctionType) or code.co_filename != '<string>'
            or code.co_freevars or not set(code.co_names) <= names or 'runner' not in code.co_names
            or scope.get('bin') is not binary or scope.get('runner') is not runner
            or scope.get('function') != binary.function or scope.get('metadata') is not binary.packed_metadata
            or type(runner).__module__ != 'triton.backends.xpu.driver'
            or getattr(binary.metadata, 'global_scratch_size', 0) or getattr(binary.metadata, 'profile_scratch_size', 0)):
        return None
    markers = tuple(object() for _ in range(len(layouts)+1))
    saved = []
    def record(*args):
        saved.append(args)
    # The generated launcher constructs a LazyDict for an empty HookChain.
    # There is no consumer in this path; an installed hook uses the original launcher.
    no_metadata = types.SimpleNamespace(launch_metadata=lambda *args: None)
    probe = types.FunctionType(code, dict(scope, runner=record, bin=no_metadata))
    try:
        probe(*markers, *constants, stream=None)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return None
    if len(saved) != 1:
        return None
    args, = saved
    signature = tuple(runner._signature.values())
    if (len(args) != 9 + len(signature) or any(type(v) is not int or v <= 0 for v in args[:3])
            or args[3] is not None or args[4] != binary.function or args[5] is not binary.packed_metadata
            or args[6] is not None or args[7] is not scope.get('launch_enter_hook')
            or args[8] is not scope.get('launch_exit_hook')):
        return None
    runtime_args = [arg for typ, arg in zip(signature, args[9:]) if typ != 'constexpr']
    if any(not typ.startswith('*') and typ not in ('constexpr', 'i1', 'i32', 'i64') for typ in signature):
        return None
    seen = [arg for arg in runtime_args if type(arg) is not int]
    if len(seen) != len(markers) or any(arg is not marker for arg, marker in zip(seen, markers)):
        return None
    slots = tuple(9+i for i,v in enumerate(runtime_args) if type(v) is not int)
    words = tuple(None if type(v) is not int else v for v in runtime_args)
    # The normal SDNN tensor ABI passes numel beside each live pointer.
    import math
    counts = (tuple(math.prod(layout[0]) for layout in layouts) + (math.prod(shape),)
              if binary.metadata.is_sdnn else (None,) * (len(layouts) + 1))
    header = args[:6] + (None, None, None)
    fn = _pack_launch(runner.launch, header, words, slots, counts, empty,
                      (shape, stride, torch.float32 if dtype is None else dtype), stream, dev)
    state = getattr(fn, '__self__', None)
    if type(state) is not tuple or len(state) != 13:
        return None
    return fn, code, runner, args[4], args[5]


def prepare(compiled, proof):
    """The caller retains CompiledFxGraph profiling, metrics and bundler cleanup."""
    import torch
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton.runtime import driver
    from ._resource_codegen import _bound_module
    from ._p800_bound_launch import _current_device

    evidence = proof.get('kernel_evidence', ())
    bindings = {e.get('wrapper_binding') for e in evidence}
    if len(bindings) != 1 or not all(e.get('effective') == 'sdnn' for e in evidence):
        return None
    binding = bindings.pop()
    if not binding:
        return None
    plan = _static_call(compiled.source_code, binding)
    if plan is None:
        return None
    module = _bound_module(compiled)
    native = compiled.current_callable
    underlying = native
    if (getattr(native, '__module__', '') == 'torch._inductor.utils'
            and getattr(native, '__qualname__', '') == 'align_inputs_from_check_idxs.<locals>.run'):
        underlying = inspect.getclosurevars(native).nonlocals['model']
    if underlying != module.call:
        return None
    tuner = getattr(module, binding)
    normal = CachingAutotuner.run
    if getattr(tuner.run, '__func__', None) is not normal:
        return None
    dev, layouts, shape, stride, reorder, constants = plan
    current, stream = _current_device(driver.active), driver.active.get_current_stream
    from triton import knobs
    runtime = knobs.runtime
    guards = torch._C._dynamo.guards
    empty = guards._empty_strided_cuda
    state = None

    def allocator(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device=tuner.device_props.type)

    def run(inputs):
        nonlocal state
        enter, leave = runtime.launch_enter_hook, runtime.launch_exit_hook
        if (sys.gettrace() or sys.getprofile() or torch.jit.is_tracing() or runtime.debug
                or knobs.compilation.instrumentation_mode or getattr(enter, 'calls', enter)
                or getattr(leave, 'calls', leave) or compiled.current_callable is not native or getattr(module, binding) is not tuner
                or getattr(tuner.run, '__func__', None) is not normal or torch._C._len_torch_dispatch_stack()
                or torch._C._is_torch_function_mode_enabled() or current() != dev or tuner.triton_interpret
                or tuner.dump_launch_params or len(tuner.launchers) != 1):
            return compiled.current_callable(inputs)
        launcher = tuner.launchers[0]
        if (launcher.store_cubin or tuner.inductor_meta.get('coordinate_descent_tuning', False)
                and not getattr(launcher.config, 'found_by_coordesc', False)):
            return native(inputs)
        if state is None or state[0] is not launcher:
            chosen = getattr(launcher, '__globals__', {}).get('bin')
            if (chosen is None or not chosen.metadata.is_sdnn or chosen.metadata.xpu_arch != 3
                    or len(inputs) != len(layouts) or any(type(x) is not torch.Tensor or x.dtype != torch.float32
                    or x.device.type != 'cuda' or x.device.index != dev or (tuple(x.shape), tuple(x.stride())) != layout
                    for x, layout in zip(inputs, layouts))):
                return native(inputs)
            packet = _packet_for_launcher(launcher, chosen, reorder(layouts), shape, stride, constants, empty, stream, dev)
            state = launcher, guards.TensorGuards(*inputs, dynamic_dims_sizes=None, dynamic_dims_strides=None).check, packet, chosen, launcher.__globals__.get('launch_enter_hook'), launcher.__globals__.get('launch_exit_hook')
            proof['schedule'] = 'generated_static_sdnn'
        if not state[1](*inputs):
            return native(inputs)
        args = reorder(inputs)
        if state[2] is not None:
            packed, code, runner, function, metadata = state[2]
            scope = launcher.__globals__
            if (launcher.__code__ is code and scope.get('runner') is runner and scope.get('function') == function
                    and scope.get('metadata') is metadata and scope.get('bin') is state[3]
                    and scope.get('launch_enter_hook') is state[4] and scope.get('launch_exit_hook') is state[5]
                    and (state[4] is None or state[4] is enter) and (state[5] is None or state[5] is leave)
                    and not getattr(sys.modules[type(runner).__module__], '_PRINT_TRITON_FUNC', False)):
                if hasattr(triton, 'set_allocator'):
                    triton.set_allocator(allocator)
                output = packed(*args)
                if output is None:
                    return native(inputs)
                inputs.clear()
                proof['schedule'] = 'generated_static_sdnn_packet'
                proof['prepared_launch'] = True
                return (output,)
        if any(x.data_ptr() & 15 for x in inputs):
            return native(inputs)
        inputs.clear()
        output = empty(shape, stride, torch.float32)
        if hasattr(triton, 'set_allocator'):
            triton.set_allocator(allocator)
        # This is Inductor's selected launcher; it owns grid calculation,
        # constexpr filtering, launch hooks and scaled SDNN pointer sizes.
        launcher(*args, output, *constants, stream=stream(dev))
        proof['prepared_launch'] = True
        return (output,)

    return run
