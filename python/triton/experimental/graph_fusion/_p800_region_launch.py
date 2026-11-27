"""Prepare the verified static Cluster wrapper, including live output views."""
import ast
import math
from pathlib import Path
import sys


def _plan(source, binding):
    """Match the whole wrapper body; decline arbitrary operations or aliases."""
    try:
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'call']
        if len(functions) != 1:
            return None
        body = functions[0].body
        names = [v.id for v in body[0].targets[0].elts]
        if len(names) != 4 or len(set(names)) != 4:
            return None
        block = body[-2]
        dev = ast.literal_eval(block.items[0].context_expr.args[0])
        allocation, get_stream, issue = block.body[1:4]
        out, stream = allocation.targets[0].id, get_stream.targets[0].id
        shape, stride = [ast.literal_eval(n) for n in allocation.value.args[:2]]
        dtype = allocation.value.args[2].attr
        if (type(dev) is not int or dtype not in ('float16', 'bfloat16', 'float32')
                or type(shape) is not tuple or len(shape) != 3 or shape[0] not in (1, 2)
                or any(type(n) is not int or n <= 0 for n in shape)
                or stride != (shape[1]*shape[2], shape[2], 1)):
            return None
        layouts = [(ast.literal_eval(n.value.args[1]), ast.literal_eval(n.value.args[2])) for n in body[2:-2]]
        if len(layouts) != len(names):
            return None
        arguments = issue.value.args
        if [v.id for v in arguments[:5]] != names + [out]:
            return None
        constants = tuple(ast.literal_eval(n) for n in arguments[5:])
        if len(constants) != 3 or any(type(n) is not int or n <= 0 for n in constants):
            return None
        views = []
        for node in body[-1].value.elts:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'reinterpret_tensor':
                return None
            if len(node.args) != 4 or node.keywords or node.args[0].id != out:
                return None
            size, steps, offset = (ast.literal_eval(v) for v in node.args[1:])
            if (type(size) is not tuple or type(steps) is not tuple or len(size) != len(steps)
                    or type(offset) is not int or offset < 0 or not size
                    or any(type(v) is not int or v <= 0 for v in size + steps)
                    or offset + 1 + sum((n-1)*s for n,s in zip(size,steps)) > math.prod(shape)):
                return None
            views.append((size, steps, offset))
        if len(views) != shape[0]:
            return None
        statements = [f"{', '.join(names)} = args", 'args.clear()']
        statements += [f'assert_size_stride({name}, {size!r}, {step!r})' for name,(size,step) in zip(names,layouts)]
        statements += [f'with torch.cuda._DeviceGuard({dev}):', f'    torch.cuda.set_device({dev})',
                       f'    {out} = empty_strided_cuda({shape!r}, {stride!r}, torch.{dtype})',
                       f'    {stream} = get_raw_stream({dev})',
                       f"    {binding}.run({', '.join(names+[out]+[repr(v) for v in constants])}, stream={stream})"]
        statements += [f'    del {name}' for name in names]
        statements += ['return (' + ', '.join(f'reinterpret_tensor({out}, {size!r}, {step!r}, {offset})' for size,step,offset in views) + ',)']
        if [ast.dump(n) for n in body] != [ast.dump(n) for n in ast.parse('\n'.join(statements)).body]:
            return None
        return dev, layouts, shape, stride, dtype, constants, views
    except (AttributeError, ValueError, TypeError, IndexError, SyntaxError):
        return None


def prepare(module, args, proof):
    """Caller owns live argument guards; all unsupported runtime modes fall back."""
    import torch
    from triton import knobs
    from triton.runtime import driver
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from ._p800_bound_launch import _current_device
    from ._p800_generated import _packet_for_launcher

    evidence = proof.get('binary_evidence', ())
    if len(evidence) != 1 or evidence[0].get('effective') != 'cluster':
        return None
    binding = evidence[0].get('wrapper_binding')
    if not binding:
        return None
    plan = _plan(Path(module.__file__).read_text(), binding)
    if plan is None:
        return None
    dev, layouts, shape, stride, dtype_name, constants, views = plan
    dtype = getattr(torch, dtype_name)
    if (len(args) != 4 or any(type(x) not in (torch.Tensor, torch.nn.Parameter)
            or x.device.type != 'cuda' or x.device.index != dev or x.dtype != dtype
            or (tuple(x.shape), tuple(x.stride())) != layout for x,layout in zip(args,layouts))):
        return None
    tuner = getattr(module, binding)
    normal = CachingAutotuner.run
    if getattr(tuner.run, '__func__', None) is not normal or len(tuner.launchers) != 1:
        return None
    launcher = tuner.launchers[0]
    binary = launcher.__globals__.get('bin')
    if (binary is None or binary.metadata.is_sdnn or binary.metadata.xpu_arch != 3
            or getattr(binary.metadata, 'tensor_args', None) not in ([], ())):
        return None
    current, stream = _current_device(driver.active), driver.active.get_current_stream
    packet = _packet_for_launcher(launcher, binary, layouts, shape, stride, constants,
                                 torch._C._dynamo.guards._empty_strided_cuda, stream, dev, dtype=dtype)
    if packet is None:
        return None
    packed, code, runner, function, metadata = packet
    native, reinterpret = module.call, module.reinterpret_tensor
    runtime = knobs.runtime
    scope = launcher.__globals__
    enter, leave = scope.get('launch_enter_hook'), scope.get('launch_exit_hook')
    proof['prepared_region_launch'] = dict(resource='cluster',outputs=len(views),live_stream=True,
                                         live_pointers=True,abi='verified_dense_packet')

    def call(*values):
        if (sys.gettrace() or sys.getprofile() or torch.autograd._profiler_enabled()
                or runtime.debug or getattr(runtime, 'disable_launch_fast_path', False)
                or knobs.compilation.instrumentation_mode or current() != dev
                or getattr(runtime.launch_enter_hook, 'calls', runtime.launch_enter_hook)
                or getattr(runtime.launch_exit_hook, 'calls', runtime.launch_exit_hook)
                or (enter is not None and enter is not runtime.launch_enter_hook)
                or (leave is not None and leave is not runtime.launch_exit_hook)
                or module.call is not native or getattr(module,binding) is not tuner
                or getattr(tuner.run,'__func__',None) is not normal
                or tuner.triton_interpret or tuner.dump_launch_params or len(tuner.launchers) != 1
                or tuner.launchers[0] is not launcher or launcher.store_cubin
                or tuner.inductor_meta.get('coordinate_descent_tuning', False)
                or launcher.__code__ is not code or scope.get('runner') is not runner
                or scope.get('bin') is not binary or scope.get('function') != function
                or scope.get('metadata') is not metadata or scope.get('launch_enter_hook') is not enter
                or scope.get('launch_exit_hook') is not leave
                or getattr(sys.modules[type(runner).__module__], '_PRINT_TRITON_FUNC', False)):
            return module.call(list(values))
        output = packed(*values)
        if output is None:
            return native(list(values))
        return tuple(reinterpret(output, size, steps, offset) for size,steps,offset in views)

    return call
