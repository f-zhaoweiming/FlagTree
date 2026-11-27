"""Guarded warm replay for the Torch 2.9 P800 fullgraph module entry."""
import dis
import functools
import inspect
import sys
import types


_MISS = object()
_GUARD_MISS = object()
_REFRESH = object()


@functools.lru_cache(maxsize=1)
def _frame_extension():
    from pathlib import Path
    from subprocess import CalledProcessError
    try:
        from triton.runtime.build import compile_module_from_src
        source = Path(__file__).with_suffix('.c').read_text()
        # The XPU launcher builder uses C++ so exceptions from Inductor's
        # native tensor accessor cannot escape across a Python C frame.
        try:
            from triton.backends.xpu.driver import compile_module_from_src as compile_xpu_module
            return compile_xpu_module(source, '_p800_frame_guard', 3)
        except (OSError, ImportError, RuntimeError, CalledProcessError):
            pass
        return compile_module_from_src(source, '_p800_frame_guard')
    except (OSError, ImportError, RuntimeError, CalledProcessError):
        return None


def _before_graph():
    return None


def _after_graph(context):
    pass


# A warmed frame can only load arguments, call this backend, and reconstruct
# the public return value. Anything requiring Python side effects, a nested
# graph, globals from user code, or a different bytecode ABI stays with Dynamo.
_REPLAY_OPS = frozenset("""
    LOAD_FAST STORE_FAST DELETE_FAST LOAD_CONST LOAD_GLOBAL LOAD_ATTR CALL_FUNCTION
    DUP_TOP ROT_TWO ROT_THREE BUILD_LIST BUILD_TUPLE LIST_EXTEND
    BINARY_SUBSCR UNPACK_SEQUENCE POP_TOP RETURN_VALUE
""".split())


def _single_tensor_replay(code, bindings, graph):
    """Recognize Torch 2.9's positional graph call and result[0] reconstruction."""
    ops = list(dis.get_instructions(code))
    # Record hooks in this private replay already resolve to our two no-ops.
    prefix = ('LOAD_GLOBAL', 'LOAD_GLOBAL', 'LOAD_ATTR', 'DUP_TOP', 'STORE_FAST',
              'CALL_FUNCTION', 'STORE_FAST')
    suffix = ('LOAD_GLOBAL', 'LOAD_ATTR', 'DUP_TOP', 'STORE_FAST', 'LOAD_FAST',
              'CALL_FUNCTION', 'POP_TOP', 'CALL_FUNCTION', 'STORE_FAST', 'LOAD_FAST',
              'LOAD_CONST', 'BINARY_SUBSCR', 'BUILD_LIST', 'BUILD_LIST', 'ROT_TWO',
              'BUILD_LIST', 'LIST_EXTEND', 'POP_TOP', 'BUILD_LIST', 'ROT_TWO',
              'DELETE_FAST', 'RETURN_VALUE')
    n = code.co_argcount - 1
    if n not in (2, 3) or len(ops) != len(prefix) + n + len(suffix):
        return None
    start, args, end = ops[:len(prefix)], ops[len(prefix):len(prefix)+n], ops[-len(suffix):]
    if (tuple(i.opname for i in start) != prefix or tuple(i.opname for i in end) != suffix
            or bindings.get(start[0].argval) is not graph
            or getattr(bindings.get(start[1].argval), 'record_pregraph_bytecode_enter', None) is not _before_graph
            or getattr(bindings.get(start[1].argval), 'record_pregraph_bytecode_exit', None) is not _after_graph
            or start[2].argval != 'record_pregraph_bytecode_enter' or start[5].arg != 0
            or end[0].argval != start[1].argval or end[1].argval != 'record_pregraph_bytecode_exit'
            or end[4].argval != start[6].argval or end[5].arg != 1 or end[7].arg != n
            or end[8].argval != end[9].argval or end[20].argval != end[8].argval
            or type(end[10].argval) is not int or end[10].argval != 0
            or [end[j].arg for j in (12, 13, 15, 16, 18)] != [0, 0, 0, 2, 1]
            or any(i.opname != 'LOAD_FAST' for i in args)
            or tuple(i.argval for i in args) != code.co_varnames[1:n+1]):
        return None
    # The compiler's original formal arguments and temporaries must be disjoint.
    temps = {start[4].argval, start[6].argval, end[3].argval, end[8].argval}
    if len(temps) != 4 or not temps.isdisjoint(code.co_varnames[:n+1]):
        return None
    module = _frame_extension()
    bind = getattr(module, 'bind_single_output', None)
    return bind(graph, n) if bind is not None else None


def _guarded_frame(forward, backend, frame_locals):
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    from torch._dynamo import utils
    for entry in _debug_get_cache_entry_list(forward.__code__):
        code = entry.code
        if code is None or code.co_freevars or code.co_cellvars:
            continue
        bindings = {}
        graph = None
        receiver = forward.__code__.co_varnames[0]
        previous = None
        for op in dis.get_instructions(code):
            if op.opname not in _REPLAY_OPS:
                break
            if op.opname in ('STORE_FAST', 'DELETE_FAST') and op.argval == receiver:
                break
            if op.opname == 'LOAD_ATTR':
                if op.argval == '_parameters':
                    # Replay the live dictionary lookup in Dynamo's original bytecode.
                    if previous is None or previous.opname != 'LOAD_FAST' or previous.argval != receiver:
                        break
                elif op.argval not in ('record_pregraph_bytecode_enter', 'record_pregraph_bytecode_exit'):
                    break
            previous = op
            if op.opname != 'LOAD_GLOBAL':
                continue
            value = forward.__globals__.get(op.argval)
            if value is utils:
                bindings[op.argval] = types.SimpleNamespace(
                    record_pregraph_bytecode_enter=_before_graph, record_pregraph_bytecode_exit=_after_graph)
            elif op.argval.startswith('__compiled_fn_') and graph is None:
                graph = getattr(value, '_torchdynamo_orig_callable', None)
                if graph is None or getattr(graph, '_flagtree_backend', None) is not backend:
                    break
                bindings[op.argval] = graph
            else:
                break
        else:
            if graph is not None and entry.guard_manager.check(frame_locals):
                # Execute Dynamo's bytecode, including its argument order and
                # return reconstruction. Never guess the FX calling convention.
                replay = _single_tensor_replay(code, bindings, graph)
                return entry.guard_manager, replay or types.FunctionType(code, bindings, forward.__name__)
    return None


def attach_module_entry(optimized, backend, calibration):
    """Install only on a new, opted-in OptimizedModule; keep native miss handling."""
    import torch
    from torch._dynamo import eval_frame
    from torch._C._dynamo.eval_frame import get_eval_frame_callback
    from torch.nn.modules import module as nn_module
    from ._resource_codegen import baidu_target

    if not isinstance(optimized, eval_frame.OptimizedModule):
        return optimized
    if getattr(optimized.forward, '_flagtree_hot_entry', None) is backend:
        return optimized
    ctx = optimized.dynamo_ctx
    if (sys.version_info[:2] != (3, 10) or not torch.__version__.startswith('2.9.')
            or not ctx.fullgraph or ctx._dynamic is not False
            or any(getattr(ctx._hooks, name, None) is not None for name in
                   ('guard_filter_fn', 'guard_export_fn', 'guard_fail_fn'))
            or (baidu_target() or {}).get('arch') != 3):
        return optimized
    mod = optimized._orig_mod
    cls = type(mod)
    if (cls.__getattribute__ is not object.__getattribute__ or cls.__getattr__ is not torch.nn.Module.__getattr__
            or cls.__call__ is not torch.nn.Module.__call__ or cls._call_impl is not torch.nn.Module._call_impl):
        return optimized
    forward = cls.forward
    if not isinstance(forward, types.FunctionType):
        return optimized
    code = forward.__code__
    if (code.co_freevars or code.co_cellvars or code.co_kwonlyargcount or forward.__defaults__
            or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS | inspect.CO_GENERATOR | inspect.CO_COROUTINE)):
        return optimized

    normal = optimized.forward
    names = code.co_varnames[:code.co_argcount]
    nargs = len(names) - 1
    tensor_type = torch.Tensor
    cached = None
    module_call, call_impl = cls.__call__, cls._call_impl
    is_fx_tracing = torch.fx._symbolic_trace.is_fx_symbolic_tracing
    native_profiler = torch.autograd._profiler_enabled
    profiler = torch.autograd.profiler
    grad_enabled = torch.is_grad_enabled
    function_mode_enabled = torch._C._is_torch_function_mode_enabled
    dispatch_depth = torch._C._len_torch_dispatch_stack
    transforms_active = torch._C._are_functorch_transforms_active
    tracing_state = torch._C._get_tracing_state

    native = None

    def bind_frame(frame):
        from collections import OrderedDict
        if (frame is None or nargs not in (2, 3) or type(cls) is not type
                or type(mod) is not cls or cls.forward is not forward or forward.__code__ is not code
                or cls.__call__ is not module_call or cls._call_impl is not call_impl
                or cls.__getattribute__ is not object.__getattribute__ or cls.__getattr__ is not torch.nn.Module.__getattr__):
            return None
        state = mod.__dict__
        slots = ('_modules', '_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')
        # Only plain attribute storage can use CPython's class/dict version tags.
        # Check each live hook mapping's size on every hit, including in-place mutations.
        if (type(state) is not dict or 'forward' in state or '_call_impl' in state or
                mod._compiled_call_impl is not None or
                any(inspect.getattr_static(cls, key, None) is not None for key in (*slots, '_compiled_call_impl'))):
            return None
        mappings = tuple(state.get(key) for key in slots)
        if any(type(value) not in (dict, OrderedDict) or value for value in mappings):
            return None
        module = _frame_extension()
        if module is None:
            return None
        manager, replay = frame
        modes = (grad_enabled, native_profiler, function_mode_enabled, dispatch_depth, transforms_active,
                 tracing_state, is_fx_tracing)
        attempt = module.bind((mod, forward, code, state, mappings, modes, get_eval_frame_callback, profiler,
                               eval_frame, nn_module, tensor_type, names, manager, replay, _MISS, _GUARD_MISS, _REFRESH))
        if (attempt is not None and type(optimized) is eval_frame.OptimizedModule
                and '_call_impl' not in vars(optimized)
                and type(optimized)._call_impl is torch.nn.Module._call_impl
                and type(optimized).__getattribute__ is object.__getattribute__):
            # Module.__call__ keeps ownership of _compiled_call_impl. Preserve
            # its normal hook dispatcher for misses and any replacement forward.
            keys = ('forward', '_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks')
            values = tuple(vars(optimized).get(key) for key in keys)
            if values[0] is run and all(type(v) in (dict, OrderedDict) for v in values[1:]):
                cell = run.__closure__[run.__code__.co_freevars.index('native')]
                default_dispatch = optimized._call_impl
                def valid_dispatch():
                    wrapper_type = type(optimized)
                    return (wrapper_type is eval_frame.OptimizedModule
                            and wrapper_type._call_impl is default_dispatch.__func__
                            and wrapper_type.__getattribute__ is object.__getattribute__
                            and all(not inspect.isdatadescriptor(inspect.getattr_static(wrapper_type, k, None))
                                    for k in keys))
                def current_dispatch(*args, **kwargs):
                    method = inspect.getattr_static(type(optimized), '_call_impl')
                    bind_method = getattr(method, '__get__', None)
                    return (bind_method(optimized, type(optimized)) if bind_method else method)(*args, **kwargs)
                entry = (module.bind_dispatch(optimized, default_dispatch, current_dispatch, cell, keys, values,
                                              _MISS, _GUARD_MISS, _REFRESH, valid_dispatch)
                         if valid_dispatch() else None)
                if entry is not None:
                    object.__setattr__(optimized, '_call_impl', entry)
        return (frame, attempt) if attempt is not None else None

    @functools.wraps(normal)
    def run(*args, **kwargs):
        nonlocal cached, native
        checked = refresh = None
        fast = native
        if fast is not None and not kwargs:
            frame, attempt = fast
            result = attempt(args)
            if result is _GUARD_MISS:
                # The complete guard already declined this frame. A concurrent caller
                # may have installed a different snapshot, which still needs its own check.
                checked = frame
            elif result is _REFRESH:
                refresh = fast
            elif result is not _MISS:
                return result
        state = mod.__dict__
        stance = eval_frame._stance
        if (kwargs or len(args) != nargs or grad_enabled()
                or get_eval_frame_callback() is not None or stance.stance != 'default'
                or stance.backend is not None or stance.skip_guard_eval_unsafe or sys.getprofile() or sys.gettrace()
                or native_profiler() or profiler._is_profiler_enabled
                or function_mode_enabled() or dispatch_depth()
                or transforms_active() or tracing_state() or is_fx_tracing()
                or type(mod) is not cls or cls.forward is not forward or forward.__code__ is not code
                or 'forward' in state or '_call_impl' in state
                or cls.__call__ is not module_call or cls._call_impl is not call_impl
                or mod._compiled_call_impl is not None or mod._modules
                or mod._forward_hooks or mod._forward_pre_hooks or mod._backward_hooks or mod._backward_pre_hooks
                or nn_module._has_any_global_hook()):
            return normal(*args, **kwargs)
        # Use the original frame names and this call's tensors. The common
        # binary and matmul signatures avoid a generator, tuple expansion and zip.
        if nargs == 2:
            a, b = args
            if type(a) is not tensor_type or type(b) is not tensor_type:
                return normal(*args, **kwargs)
            frame_locals = {names[0]: mod, names[1]: a, names[2]: b}
        elif nargs == 3:
            a, b, c = args
            if type(a) is not tensor_type or type(b) is not tensor_type or type(c) is not tensor_type:
                return normal(*args, **kwargs)
            frame_locals = {names[0]: mod, names[1]: a, names[2]: b, names[3]: c}
        else:
            if any(type(arg) is not tensor_type for arg in args):
                return normal(*args, **kwargs)
            frame_locals = dict(zip(names, (mod, *args)))
        snapshot = cached
        if snapshot is not None and snapshot is not checked:
            manager, replay = snapshot
            # Dynamo clears cache_entry on code reset, eviction or weakref
            # invalidation. Use the complete guard tree, not diff guards.
            if manager.cache_entry is not None and manager.check(frame_locals):
                if refresh is not None and native is refresh and cached is snapshot:
                    # Tracing or framework setup can invalidate a class/dict version
                    # without changing its behaviour. All live checks just passed.
                    native = bind_frame(snapshot)
                return replay(mod, *args)
        # A public timing round calls the OptimizedModule repeatedly. Hooks
        # attached to the wrapper itself must only run for the user's call.
        if (optimized._forward_hooks or optimized._forward_pre_hooks or optimized._backward_hooks
                or optimized._backward_pre_hooks):
            return normal(*args, **kwargs)
        previous = getattr(calibration, 'pending', None)
        calibration.pending = pending = []
        try:
            result = normal(*args, **kwargs)
        finally:
            calibration.pending = previous
        frame = _guarded_frame(forward, backend, frame_locals)
        if frame is not None and len(pending) == 1:
            prepare, graph_inputs, lock = pending[0]
            # Publish the frame under its graph's preparation lock. A second
            # caller seeing this frame must wait for the public winner.
            with lock:
                cached = frame
                native = bind_frame(frame)
                try:
                    prepare(graph_inputs, optimized, args)
                except BaseException:
                    cached = native = None
                    raise
                if cached is frame:
                    native = bind_frame(frame)
            # Discovery ran eager. Return the committed plan's first output.
            return run(*args)

        cached = frame
        native = bind_frame(frame)
        if frame is None and optimized.forward is run:
            optimized.forward = normal
        for prepare, graph_inputs, _ in pending:
            prepare(graph_inputs)
        return result

    run._flagtree_hot_entry = backend
    optimized.forward = run
    return optimized
