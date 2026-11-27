"""Reuse Inductor's generated wrapper and launcher after initial preparation.

No device code or launch geometry is generated here. The private, plan-local
copy changes only the resolved kernel's ``run`` binding. Alignment repair,
device/stream selection, allocation, launch metadata and output code remain
Inductor's. Unknown runtime layouts keep the original compiled callable.
"""

import inspect
import types


def _prepare(compiled, kernel_name):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.utils import align_inputs_from_check_idxs

    wrapped = getattr(compiled, "current_callable", None)
    generated = wrapped
    alignment = None
    if isinstance(wrapped, types.FunctionType):
        if (wrapped.__module__ != "torch._inductor.utils"
                or wrapped.__qualname__ != "align_inputs_from_check_idxs.<locals>.run"):
            return None, "unknown_callable_wrapper"
        alignment = inspect.getclosurevars(wrapped).nonlocals
        if set(alignment) != {"model", "inputs_to_check", "mutated_input_idxs"}:
            return None, "unknown_alignment_wrapper"
        generated = alignment["model"]
    if not isinstance(generated, types.MethodType) or generated.__func__.__qualname__ != "Runner.call":
        return None, "unknown_generated_callable"
    fn = generated.__func__
    kernel = fn.__globals__.get(kernel_name)
    if type(kernel) is not CachingAutotuner or len(kernel.launchers) != 1:
        return None, "unresolved_kernel_launcher"
    launcher = kernel.launchers[0]
    if (kernel.triton_interpret or kernel.dump_launch_params or kernel.dump_launch_tensors
            or kernel.inductor_meta.get("coordinate_descent_tuning", False) or launcher.store_cubin):
        return None, "instrumented_or_tuning_launcher"
    if not isinstance(launcher, types.FunctionType):
        return None, "unknown_launcher"
    binary = launcher.__globals__.get("bin")
    metadata = getattr(binary, "metadata", None)
    # The generic runner also sets a scratch allocator. Bypass it only when
    # this compiled kernel is known not to need either scratch allocation.
    if (getattr(metadata, "global_scratch_size", None) != 0 or getattr(metadata, "profile_scratch_size", None) != 0):
        return None, "scratch_allocation_not_proven_absent"
    scope = dict(fn.__globals__)
    scope[kernel_name] = types.SimpleNamespace(run=launcher)
    cloned = types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    cloned.__kwdefaults__ = fn.__kwdefaults__
    direct = types.MethodType(cloned, generated.__self__)
    if alignment is not None:
        direct = align_inputs_from_check_idxs(direct, alignment["inputs_to_check"], alignment["mutated_input_idxs"])
    return (direct, wrapped, kernel, kernel.launchers, launcher), None


def prepared_inductor_call(compiled, kernel_name, info):
    import torch
    try:
        from torch._dynamo.utils import get_runtime_metrics_context
        from torch._inductor import debug
        from torch._inductor.runtime.autotune_cache import AutotuneCacheBundler
        from torch.utils._debug_mode import get_active_debug_mode
    except ImportError:
        # Torch 2.5 has a different, already thinner call stack. Its native
        # entry point remains valid; do not apply a 2.11-specific adapter.
        info["prepared_launch"] = "native"
        info["prepared_launch_skipped_reason"] = "runtime_launch_api_unavailable"
        return compiled

    state = None
    attempted = False
    info["prepared_launch"] = "pending_first_call"

    def call(values):
        nonlocal state, attempted
        if not attempted:
            # Complete lazy launch preparation and normal metrics/cache
            # finalization once; never execute an additional warm-up call.
            result = compiled(values)
            state, reason = _prepare(compiled, kernel_name)
            attempted = True
            info["prepared_launch"] = "enabled" if state is not None else "native"
            if reason is not None:
                info["prepared_launch_skipped_reason"] = reason
            return result
        if state is None:
            return compiled(values)
        direct, wrapped, kernel, launchers, launcher = state
        # Preserve dynamic instrumentation and pending compiler bookkeeping.
        # A changed callable or launcher also uses the original entry point.
        if (torch.autograd.profiler._is_profiler_enabled or debug.RECORD_GRAPH_EXECUTION
                or AutotuneCacheBundler._bundler is not None or get_runtime_metrics_context()._metrics
                or get_active_debug_mode() is not None or compiled.current_callable is not wrapped
                or kernel.launchers is not launchers or len(launchers) != 1 or launchers[0] is not launcher):
            return compiled(values)
        return direct(values)

    return call
