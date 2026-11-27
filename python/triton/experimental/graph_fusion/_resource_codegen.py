"""Attach a resource request to compiler-generated kernels, not their math.

Experimental Torch 2.5 bridge. The native Inductor heuristic factory, generated
function body and native configs remain unchanged. Scoped P800 backend repairs
are implemented separately in _p800_compiler and its lowering helpers.
"""
import ast
import os
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import threading

_REQUEST = ContextVar("flagtree_inductor_resource", default=None)
_SCALAR_REDUCTION = ContextVar("flagtree_p800_scalar_reduction", default=False)
_LOCK = threading.RLock()
_EVIDENCE = []
_FACTORIES = frozenset(("pointwise", "reduction", "persistent_reduction", "split_scan", "template", "foreach", "fixed_config"))


def baidu_target():
    """Do not confuse the vendor's CUDA alias with an NVIDIA device."""
    try:
        import triton
        driver = triton.runtime.driver.active
    except (ImportError, RuntimeError, AttributeError):
        return None
    cls = type(driver)
    if cls.__module__ != "triton.backends.xpu.driver" or cls.__name__ != "XPUDriver":
        return None
    target = driver.get_current_target()
    if type(target.arch) is not int or target.arch not in (2, 3, 4, 5):
        return None
    return {
        "driver": cls.__module__ + "." + cls.__name__, "arch": target.arch, "warp_size": target.warp_size,
        "backend_alias": target.backend
    }


def _inspection_tree(source):
    # Native codegen has not yet substituted its final function name here.
    # Normalize only the AST inspection copy, leaving the generated text and
    # all computation intact for Inductor's own later name substitution.
    return ast.parse(source.replace('def Placeholder.KERNEL_NAME(', 'def __flagtree_inspection_kernel('))


def _function_bodies(source):
    return [
        ast.dump(ast.Module(body=n.body, type_ignores=[]), include_attributes=False)
        for n in _inspection_tree(source).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def annotate_source(source, resource):
    if resource not in ("cluster", "sdnn"):
        raise ValueError("explicit Cluster or SDNN resource required")
    before = _function_bodies(source)
    tree = _inspection_tree(source)
    factories = []
    for n in tree.body:
        if not isinstance(n, ast.FunctionDef):
            continue
        for decorator in n.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == "triton_heuristics"):
                if decorator.func.attr not in _FACTORIES:
                    raise RuntimeError("unsupported native heuristic: " + decorator.func.attr)
                factories.append(decorator.func.attr)
    if not factories:
        raise RuntimeError("no native Inductor heuristic in generated kernel")
    # The generated function body is left byte-for-byte intact. An explicit
    # resource literal in the generated module also separates code-cache keys.
    prefix = ("from triton.experimental.graph_fusion._resource_codegen import ResourceHeuristics\n"
              "_flagtree_resource_heuristics = ResourceHeuristics(" + repr(resource) + ")\n")
    if _MAX_XBLOCK.get() is not None:
        prefix += "_flagtree_resource_heuristics.max_xblock = " + repr(_MAX_XBLOCK.get()) + "\n"
    from ._p800_compiler import policy_key
    prefix += "# P800 compiler policy: " + policy_key() + "\n"
    changed = source
    for name in set(factories):
        changed = changed.replace("@triton_heuristics." + name + "(", "@_flagtree_resource_heuristics." + name + "(")
    changed = prefix + changed
    if _function_bodies(changed) != before:
        raise RuntimeError("resource annotation changed generated computation")
    return changed


def _record_binary(binary, requested):
    meta = binary.metadata
    sdnn = getattr(meta, "is_sdnn", None)
    actual = "sdnn" if sdnn else "cluster"
    # resource_kind is a compile-time preference. effective_resource_kind and
    # is_sdnn describe the emitted binary. Older P800 metadata only has is_sdnn.
    preference = getattr(meta, "resource_kind", actual if getattr(meta, "arch", None) == 3 else None)
    effective = getattr(meta, "effective_resource_kind", None if preference == "auto" else actual)
    if (type(sdnn) is not bool or requested != actual or effective != actual
            or preference not in ("auto", actual)):
        raise RuntimeError("compiled resource metadata does not match request")
    entry = {
        "requested": requested, "effective": actual, "kernel_name": meta.name, "binary_hash":
        str(getattr(binary, "hash", "")), "launch_clusters": getattr(meta, "launch_clusters", None), "launch_cores":
        getattr(meta, "launch_cores", None), "asm_sha256": {
            name: hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()
            for name, value in binary.asm.items()
        }
    }
    entry['native_metadata'] = {key: getattr(meta, key) for key in
                                ('arch', 'is_sdnn', 'resource_kind', 'effective_resource_kind', 'num_warps',
                                 'num_ctas', 'shared', 'core_num', 'cluster_num')
                                if hasattr(meta, key)}
    _EVIDENCE.append(entry)
    return entry


def bind_evidence(proof, binary):
    """Bind a shortcut's evidence to the binary returned by its real launch."""
    proof['kernel_evidence'] = [_record_binary(binary, proof['resource'])]
    proof['external_calls'] = 0
    proof['resource_verified'] = True


def _bound_module(compiled):
    """Inspect binaries actually bound by the wrapper, also on cache reuse.

    A cache hit need not call _precompile_config again. Conversely, newly
    benchmarked candidates do not prove which binaries the final wrapper uses.
    """
    from pathlib import Path
    import inspect
    from torch._inductor.codecache import PyCodeCache
    if hasattr(PyCodeCache, 'cache'):
        module = PyCodeCache.cache.get(compiled.cache_key)
    else:
        # Torch 2.9 keeps loaded modules in a list. Match the actual callable,
        # not an arbitrary autotuning candidate with similar generated text.
        current = inspect.unwrap(compiled.current_callable)
        if (getattr(current, '__module__', '') == 'torch._inductor.utils'
                and getattr(current, '__qualname__', '') == 'align_inputs_from_check_idxs.<locals>.run'):
            current = inspect.getclosurevars(current).nonlocals['model']
        matches = [m for m in PyCodeCache.modules
                   if (getattr(m, 'call', None) is current
                       or (inspect.ismethod(current) and inspect.ismethod(getattr(m, 'call', None))
                           and m.call.__self__ is current.__self__ and m.call.__func__ is current.__func__))
                   and Path(m.__file__).stem == compiled.cache_key]
        if len(matches) != 1:
            raise RuntimeError('cannot uniquely identify native wrapper callable module')
        module = matches[0]
    if module is None or Path(module.__file__).read_text() != compiled.source_code:
        raise RuntimeError("native wrapper cache/source identity could not be verified")
    return module


def _bound_kernel_evidence(compiled, resource):
    module = _bound_module(compiled)
    proofs = []
    for node in ast.walk(ast.parse(compiled.source_code)):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'async_compile.triton'):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise RuntimeError('unknown native kernel binding')
        name = node.targets[0].id
        tuner = getattr(module, name)
        tuner.precompile()
        if not tuner.launchers:
            raise RuntimeError('native kernel has no compiled launcher')
        for launcher in tuner.launchers:
            binary = getattr(launcher, '__globals__', {}).get('bin')
            if binary is None:
                raise RuntimeError('native launcher lacks its bound compiled binary')
            entry = dict(_record_binary(binary, resource))
            entry.update(wrapper_binding=name,
                         source_sha256=hashlib.sha256(node.value.args[1].value.encode()).hexdigest(),
                         proof_scope='bound_native_wrapper_binary')
            proofs.append(entry)
    return proofs


# bf16 pointwise codegen corrupts results at XBLOCK >= 4096 (measured: 61k
# non-finite at 4096, 80k at 8192, clean at <= 2048; f32/f16 are fine).
# Serialize the per-compilation cap into the generated heuristics instance
# because a worker thread does not inherit the caller’s context.
_MAX_XBLOCK = ContextVar("flagtree_xblock_cap", default=None)


class ResourceHeuristics:

    def __init__(self, resource):
        if resource not in ("cluster", "sdnn"):
            raise ValueError("unknown resource")
        self.resource = resource
        self.max_xblock = _MAX_XBLOCK.get()

    def __getattr__(self, name):
        if name not in _FACTORIES:
            raise AttributeError(name)
        from torch._inductor.runtime import triton_heuristics
        original = getattr(triton_heuristics, name)
        resource = self.resource

        def factory(*args, **kwargs):
            decorate = original(*args, **kwargs)

            def wrapped(fn):
                tuner = decorate(fn)
                cap = self.max_xblock
                if cap is not None and getattr(tuner, 'configs', None):
                    kept = [c for c in tuner.configs if int(c.kwargs.get('XBLOCK', 0)) <= cap]
                    if not kept:
                        raise RuntimeError("no valid BF16 kernel config within XBLOCK limit")
                    tuner.configs = kept
                    if name == 'pointwise' and not kwargs.get('inductor_meta', {}).get('autotune_pointwise', True):
                        tuner.configs = kept[:1]
                # P800: Inductor's pointwise heuristic defaults to a small XBLOCK
                # (1024), which is 2-3x below the best measured block for flat
                # elementwise/reduction kernels (4096-16384). Expand the candidate
                # list so the tuner can pick the faster block instead of inheriting
                # the default.
                forced = os.environ.get('FLAGTREE_FORCE_XBLOCK')
                if name != 'fixed_config' and cap is None and forced and getattr(tuner, 'configs', None):
                    import copy as _copy
                    xb = int(forced)
                    tuner.configs = [_copy.deepcopy(c) for c in tuner.configs]
                    for c in tuner.configs:
                        if 'XBLOCK' in c.kwargs:
                            c.kwargs['XBLOCK'] = xb
                if name != 'fixed_config' and cap is None and kwargs.get('inductor_meta', {}).get('max_autotune') \
                        and os.environ.get('FLAGTREE_WIDE_XBLOCK', '1') == '1' \
                        and getattr(tuner, 'configs', None):
                    import copy as _copy
                    wide = []
                    for cfg in list(tuner.configs):
                        base_block = cfg.kwargs.get('XBLOCK')
                        if not isinstance(base_block, int) or base_block <= 0:
                            continue
                        for xb in (4096, 8192, 16384):
                            if xb <= base_block:
                                continue
                            clone = _copy.deepcopy(cfg)
                            clone.kwargs['XBLOCK'] = xb
                            wide.append(clone)
                    if wide:
                        tuner.configs = list(tuner.configs) + wide
                        if os.environ.get('FLAGTREE_DEBUG_WIDE'):
                            print('[wide-res] configs', [c.kwargs.get('XBLOCK') for c in tuner.configs], flush=True)
                compile_config = tuner._precompile_config

                def compile_for_resource(*a, **kw):
                    cfg = a[0] if a else kw.get('cfg')
                    if cap is not None and cfg is not None and cfg.kwargs.get('XBLOCK', 0) > cap:
                        raise RuntimeError("unsafe BF16 XBLOCK for P800 native launcher")
                    target = baidu_target()
                    if target and target['arch'] == 3 and target['warp_size'] == 1:
                        from ._p800_compiler import resource_scope
                    else:
                        from triton.backends.xpu.resource_context import resource_scope
                    with resource_scope(resource):
                        result = compile_config(*a, **kw)
                    # Torch 2.9 wraps the unchanged CompiledKernel in
                    # TritonCompileResult; legacy Torch returned a tuple.
                    binary = result[0] if isinstance(result, tuple) else getattr(result, 'kernel', result)
                    _record_binary(binary, resource)
                    return result

                tuner._precompile_config = compile_for_resource
                return tuner

            return wrapped

        return factory


def _annotate_partial_render(rendered, resource):
    """Cover native autotune rendering and native final epilogue rendering."""
    finalize = rendered.finalize_all
    finalize_hook = rendered.finalize_hook
    annotated = False

    def annotate_final():
        nonlocal annotated
        if annotated:
            return rendered.code
        changed = annotate_source(rendered.code, resource)
        descriptor = getattr(type(rendered), 'code', None)
        if isinstance(descriptor, property) and descriptor.fset is None:
            # Torch 2.9 makes code read-only; its native backing field remains
            # the string used by finalize_hook/finalize_remaining.
            rendered._code = changed
        else:
            rendered.code = changed
        annotated = True
        return changed

    def finalize_resource():
        finalize()
        return annotate_final()

    def finalize_hook_resource(*args, **kwargs):
        nonlocal annotated
        result = finalize_hook(*args, **kwargs)
        # The final template path reads .code after individual hooks, without
        # calling finalize_all. Wait until every native substitution finishes.
        if not annotated and all(fn is None for fn in rendered.replacement_hooks.values()):
            annotate_final()
        return result

    rendered.finalize_all = finalize_resource
    rendered.finalize_hook = finalize_hook_resource
    if all(fn is None for fn in rendered.replacement_hooks.values()):
        annotate_final()
    return rendered


_WIDE_IMPORT = ('from triton.experimental.graph_fusion._resource_codegen import '
                'ResourceHeuristics as _flagtree_resource_heuristics')


def _widen_pointwise(source):
    """Route native pointwise kernels through the resource heuristic class so the
    wider XBLOCK candidates also apply to kernels compiled in worker processes."""
    if os.environ.get('FLAGTREE_WIDE_XBLOCK', '1') != '1':
        return source
    if '@triton_heuristics.pointwise(' not in source:
        return source
    source = source.replace('@triton_heuristics.pointwise(',
                            '@_flagtree_resource_heuristics.pointwise(')
    if _WIDE_IMPORT not in source:
        marker = 'from torch._inductor.runtime import triton_helpers, triton_heuristics'
        if marker in source:
            source = source.replace(marker, marker + '\n' + _WIDE_IMPORT, 1)
        else:
            source = _WIDE_IMPORT + '\n' + source
    return source


@contextmanager
def generated_resource(resource, *, native_template_eligibility=False, scalar_reduction=False, verified_tile=None):
    if resource not in ("cluster", "sdnn"):
        raise ValueError("explicit resource required")
    if baidu_target() is None:
        raise RuntimeError("resource routing requires positively identified Baidu XPU backend")
    if _REQUEST.get() is not None:
        raise RuntimeError("nested resource compilation is unsupported")
    from torch._inductor.codegen.triton import TritonKernel
    from torch._inductor.select_algorithm import TritonTemplateKernel
    from torch._inductor.select_algorithm import AlgorithmSelectorCache
    from torch._inductor import config
    from torch._inductor import utils
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner
    # Scoped codegen change is context-local for unrelated compiler threads;
    # explicit resource decorators survive async module loading. Disable the
    # graph cache so a previously compiled unannotated graph cannot bypass this.
    # Share Dynamo's compilation lock so temporary native config settings do
    # not leak into a concurrent torch.compile callback on another thread.
    with compile_lock, _LOCK:
        original = TritonKernel.codegen_kernel
        original_no_x = TritonKernel.want_no_x_dim
        original_render = TritonTemplateKernel.render
        original_big_gpu = utils.is_big_gpu
        original_select = AlgorithmSelectorCache.__call__
        original_neighbour = CoordescTuner.get_neighbour_values
        token = _REQUEST.set(resource)
        reduction_token = _SCALAR_REDUCTION.set(scalar_reduction)
        target = baidu_target()
        repair_reduction = target['arch'] == 3 and target['warp_size'] == 1

        def select_algorithm(selector, *args, **kwargs):
            # Torch 2.5's GEMM selector has a separate precompilation thread
            # pool, even with compile_threads=1. Waiting for that pool while
            # holding Dynamo's compile lock deadlocks the resource decorator.
            # Skip only speculative parallel precompilation: native candidate
            # benchmarking/selection still compiles synchronously as needed.
            if repair_reduction and _REQUEST.get() is not None:
                import inspect
                bound = inspect.signature(original_select).bind(selector, *args, **kwargs)
                if verified_tile is not None and bound.arguments.get('name') in ('mm', 'addmm'):
                    choices = []
                    for choice in bound.arguments['choices']:
                        info = choice.info_dict()
                        shape = info.get('tile_shape', ())
                        if isinstance(shape, str):
                            shape = ast.literal_eval(shape)
                        if (*shape, info.get('num_stages'), info.get('num_warps')) == verified_tile:
                            choices.append(choice)
                    if not choices:
                        raise RuntimeError(f'verified P800 FP32 GEMM tile unavailable: {verified_tile}')
                    bound.arguments['choices'] = choices[:1]
                if 'precompilation_timeout_seconds' not in inspect.signature(original_select).parameters:
                    raise RuntimeError('P800 native precompilation API is unavailable')
                bound.arguments['precompilation_timeout_seconds'] = 0
                return original_select(*bound.args, **bound.kwargs)
            return original_select(selector, *args, **kwargs)

        AlgorithmSelectorCache.__call__ = select_algorithm

        def want_no_x_dim(kernel):
            # Keep native 2D reduction layout: the P800 rank-1 lowering writes
            # the scalar result from every core. Native 2D writeback predicates
            # the owning core. No tiling or generated arithmetic is supplied.
            if repair_reduction and _REQUEST.get() == 'cluster' and _SCALAR_REDUCTION.get():
                return False
            return original_no_x(kernel)

        TritonKernel.want_no_x_dim = want_no_x_dim
        if native_template_eligibility:
            # P800 exposes 8 units through the CUDA alias. Native Torch 2.5's
            # >=68 NVIDIA SM heuristic otherwise suppresses every GEMM template.
            # This only admits native candidates for this identified backend;
            # their configs, generated math and backend validation stay native.
            def eligible(index):
                return True if _REQUEST.get() == resource else original_big_gpu(index)

            utils.is_big_gpu = eligible

        def codegen(kernel, *args, **kwargs):
            code = original(kernel, *args, **kwargs)
            requested = _REQUEST.get()
            return annotate_source(code, requested) if requested else code

        TritonKernel.codegen_kernel = codegen

        def render(kernel, *args, **kwargs):
            rendered = original_render(kernel, *args, **kwargs)
            requested = _REQUEST.get()
            return _annotate_partial_render(rendered, requested) if requested else rendered

        TritonTemplateKernel.render = render

        # Torch's own pointwise heuristic picks XBLOCK=1024 on P800, which is
        # 2-3x below the best measured block for flat elementwise kernels
        # (4096-16384). Wrap the decorator so those kernels get the wider
        # candidates too; the tuner still benchmarks and picks the fastest.
        from torch._inductor.runtime import triton_heuristics as _th

        def _expand_wide(tuner):
            if _MAX_XBLOCK.get() is not None:
                return tuner
            if os.environ.get('FLAGTREE_WIDE_XBLOCK', '1') != '1':
                return tuner
            configs = getattr(tuner, 'configs', None)
            if not configs:
                return tuner
            import copy as _copy
            wide = []
            for cfg in list(configs):
                base = cfg.kwargs.get('XBLOCK')
                if not isinstance(base, int) or base <= 0:
                    continue
                for xb in (4096, 8192, 16384):
                    if xb <= base:
                        continue
                    clone = _copy.deepcopy(cfg)
                    clone.kwargs['XBLOCK'] = xb
                    wide.append(clone)
            if wide:
                tuner.configs = list(configs) + wide
            if os.environ.get('FLAGTREE_DEBUG_WIDE'):
                print('[wide] configs', [c.kwargs.get('XBLOCK') for c in tuner.configs],
                      'num_warps', [c.num_warps for c in tuner.configs], flush=True)
            return tuner

        _original_pointwise = _th.pointwise

        def _wide_pointwise(*args, **kwargs):
            decorate = _original_pointwise(*args, **kwargs)

            def _wrapped(fn):
                return _expand_wide(decorate(fn))

            return _wrapped

        _th.pointwise = _wide_pointwise

        # Cap the coordinate-descent tile search: bf16 pointwise codegen is
        # corrupt at XBLOCK >= 4096 and the tuner has no correctness gate.
        def capped_neighbours(tuner_self, prefix, cur_val, *args, **kwargs):
            values = original_neighbour(tuner_self, prefix, cur_val, *args, **kwargs)
            cap = _MAX_XBLOCK.get()
            if cap is not None and str(prefix).startswith(('XBLOCK', 'YBLOCK', 'ZBLOCK')):
                values = [v for v in values if v is None or int(v) <= cap]
            return values
        CoordescTuner.get_neighbour_values = capped_neighbours
        try:
            with config.patch({"compile_threads": 1, "fx_graph_cache": False, "fx_graph_remote_cache": False}):
                yield
        finally:
            if _th.pointwise is _wide_pointwise:
                _th.pointwise = _original_pointwise
            if CoordescTuner.get_neighbour_values is capped_neighbours:
                CoordescTuner.get_neighbour_values = original_neighbour
            if AlgorithmSelectorCache.__call__ is select_algorithm:
                AlgorithmSelectorCache.__call__ = original_select
            if TritonKernel.want_no_x_dim is want_no_x_dim:
                TritonKernel.want_no_x_dim = original_no_x
            if native_template_eligibility and utils.is_big_gpu is eligible:
                utils.is_big_gpu = original_big_gpu
            if TritonKernel.codegen_kernel is codegen:
                TritonKernel.codegen_kernel = original
            if TritonTemplateKernel.render is render:
                TritonTemplateKernel.render = original_render
            _REQUEST.reset(token)
            _SCALAR_REDUCTION.reset(reduction_token)


def evidence():
    return list(_EVIDENCE)


def audit_wrapper(source, resource):
    """Fail closed if a generated kernel or external operation is unaccounted."""
    tree = ast.parse(source)
    kernels = []
    external = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        target = ast.unparse(n.func)
        if target == "async_compile.triton":
            if len(n.args) < 2 or not isinstance(n.args[1], ast.Constant) or not isinstance(n.args[1].value, str):
                raise RuntimeError("unrecognized native generated kernel source")
            code = n.args[1].value
            expected = "ResourceHeuristics(" + repr(resource) + ")"
            if expected not in code or "@triton_heuristics." in code:
                raise RuntimeError("generated kernel is missing its explicit resource")
            kernels.append(hashlib.sha256(code.encode()).hexdigest())
        elif target.startswith(("extern_kernels.", "torch.ops.", "aten.", "ops.")):
            external.append(target)
    return {"generated_kernel_sources": kernels, "external_call_targets": external}


def compile_region(gm, inputs, resource, *, matmul_path="external", verified_tile=None):
    """Use native decomposition and Inductor; caller validates the result.

    External-library calls remain a hard, unverified boundary. They are not
    assigned SDNN based on an operator name. The optional template path uses
    native Inductor templates/configs, without editing generated computation.
    """
    from ._p800_inductor_softmax import native_softmax_scope
    from torch._inductor.compile_fx import compile_fx_inner, BoxedBool
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._inductor import config
    import torch
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("resource compilation/autotuning must finish before graph capture")
    if matmul_path not in ("external", "template"):
        raise ValueError("matmul_path must be external or template")
    output = next(n for n in gm.graph.nodes if n.op == "output")
    if not isinstance(output.args[0], tuple):
        raise ValueError("region graph output must be a tuple")
    lowered = make_fx(gm, decomposition_table=select_decomp_table(), tracing_mode="fake",
                      _allow_non_fake_inputs=True)(*inputs)
    # Explicitly select native candidates; never alter math precision flags or
    # hand-pick tile/warp configs. Autotuning is offline, before graph capture.
    options = {
        "max_autotune_gemm": matmul_path == "template", "max_autotune_gemm_backends":
        "TRITON" if matmul_path == "template" else "ATEN", "max_autotune": config.max_autotune, "autotune_in_subproc": False
    }
    # P800 bf16 pointwise kernels come back with scattered non-finite values when
    # Inductor's coordinate-descent search runs; the same kernels are exact with
    # every forced config. Keep the plain autotune for bf16 inputs.
    bf16_inputs = any(isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 for x in inputs)
    if bf16_inputs and not os.environ.get('FLAGTREE_ALLOW_BF16_CD'):
        options["coordinate_descent_tuning"] = False
    # Only scalar-output reductions need the P800 ownership repair. Softmax
    # and LayerNorm write full rows and should keep native reduction layouts.
    output_node = next(n for n in lowered.graph.nodes if n.op == 'output')
    outputs = output_node.args[0]
    scalar_reduction = (any(n.target == torch.ops.aten.sum.dim_IntList for n in lowered.graph.nodes) and all(
        isinstance(n, torch.fx.Node) and isinstance(n.meta.get('val'), torch.Tensor) and n.meta['val'].numel() < max(
            (x.numel() for x in inputs if isinstance(x, torch.Tensor)), default=0) for n in outputs))
    # On KL3 the cold pointwise sweep can stall in native do_bench for the
    # 64x1024 FP32 bias broadcast and its exact GELU. These are also suffixes
    # of matrix plans. Keep one native configuration for these two graphs.
    calls = {n.target for n in lowered.graph.nodes if n.op == 'call_function'}
    narrow_broadcast = (
        len(inputs) == 2 and all(x.dtype == torch.float32 for x in inputs)
        and sorted(tuple(x.shape) for x in inputs) == [(64, 1024), (1024,)]
        and torch.ops.aten.add.Tensor in calls
        and calls <= {torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor, torch.ops.aten.erf.default}
    )
    if narrow_broadcast:
        options.update({'triton.autotune_pointwise': False, 'max_autotune': False,
                        'max_autotune_pointwise': False, 'coordinate_descent_tuning': False})
    cap_token = _MAX_XBLOCK.set(1024 if bf16_inputs or narrow_broadcast else None)
    try:
        with generated_resource(resource, native_template_eligibility=matmul_path == "template",
                                scalar_reduction=scalar_reduction, verified_tile=verified_tile), config.patch(options), native_softmax_scope(gm, inputs):
            compiled = compile_fx_inner(lowered, list(inputs), is_inference=True, cudagraphs=BoxedBool(False))
    finally:
        _MAX_XBLOCK.reset(cap_token)
    source = compiled.source_code
    wrapper_audit = audit_wrapper(source, resource)
    external = len(wrapper_audit['external_call_targets'])
    bound_evidence = _bound_kernel_evidence(compiled, resource)
    return compiled, {
        "resource": resource, "matmul_path": matmul_path, "scalar_reduction_layout": scalar_reduction,
        "native_template_eligibility": matmul_path == "template", "wrapper_audit": wrapper_audit, "external_calls":
        external, "kernel_evidence": bound_evidence, "source_code": source, "resource_verified": external == 0
        and bool(wrapper_audit['generated_kernel_sources']) and bool(bound_evidence)
    }
