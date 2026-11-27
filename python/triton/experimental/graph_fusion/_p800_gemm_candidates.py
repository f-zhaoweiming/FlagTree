"""Bounded native FP32 GEMM candidates for identified P800 compilations.

Keep the original graph, dtypes and native Inductor templates. NVIDIA shared
memory estimates do not describe XPU SRAM; the XPU compiler still validates
every emitted candidate. Complete model validation and timing remain required.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache, partial

_ACTIVE = ContextVar('p800_large_gemm_candidates', default=False)
_ROW_BIAS = ContextVar('p800_matrix_row_bias', default=False)


def _precompile_sdnn_config(tuner, *args, **kwargs):
    from ._p800_model_schedule import _compile_buffer_option
    bounds = tuner.triton_meta['xpu_sdnn_tensor_bytes']
    with _compile_buffer_option(tuner.fn, 512, sdnn_tensor_bytes=bounds):
        result = type(tuner)._precompile_config(tuner, *args, **kwargs)
    binary = getattr(result, 'kernel', result)
    if tuple(getattr(binary.metadata, 'xpu_sdnn_tensor_bytes', ())) != bounds:
        raise RuntimeError('SDNN tensor bounds were not forwarded to the compiler')
    return result


def p800_sdnn_template(*args, **kwargs):
    """Forward static bounds through Torch's backend-option whitelist."""
    from torch._inductor.runtime.triton_heuristics import template
    decorate = template(*args, **kwargs)
    def apply(fn):
        tuner = decorate(fn)
        # Inductor sends this object back from compile workers. A local closure
        # is not picklable; a module-level function plus partial preserves it.
        tuner._precompile_config = partial(_precompile_sdnn_config, tuner)
        return tuner
    return apply


def _sdnn_template_source(text, spans, helper):
    # jit_lines is later spliced with dedent. Imports at column zero would
    # prevent that dedent and leave the original decorators indented.
    import hashlib
    from textwrap import dedent
    text = dedent(text).replace('triton_meta={',
        f"triton_meta={{'xpu_sdnn_tensor_bytes': {spans!r}, ", 1)
    bootstrap = ('import f06\nf06.activate()\n'
                 if helper.parent.parent.name == '_vendor' else '')
    identity = hashlib.sha256(helper.read_bytes()).hexdigest()[:16]
    return (bootstrap + '# P800 SDNN reuse policy: ' + identity + '\n'
            'from triton.experimental.graph_fusion._p800_gemm_candidates import p800_sdnn_template\n'
            + text.replace('@triton_heuristics.template(', '@p800_sdnn_template('))


def static_tensor_spans(signature, lookup):
    """Storage spans for conservative SDNN runtime overlap checks.

    Dynamic/negative layouts and mixed scalar signatures keep the original
    compilation. These are byte bounds, never an assertion that tensors cannot
    alias; the generated kernel still tests their ranges at runtime.
    """
    import sympy
    from torch._inductor.codegen.common import TensorArg
    spans = []
    for arg in signature:
        if not isinstance(arg, TensorArg):
            return ()
        node = lookup(arg.buffer)
        if node is None:
            return ()
        size, stride = node.get_size(), node.get_stride()
        offset = node.get_layout().offset
        if not all(isinstance(v, (int, sympy.Integer)) for v in (*size, *stride, offset)):
            return ()
        if offset < 0 or any(s <= 0 for s in size) or any(s < 0 for s in stride):
            return ()
        span = int(1 + offset + sum((n - 1) * s for n, s in zip(size, stride))) * arg.dtype.itemsize
        if not 0 < span <= 1 << 40:
            return ()
        spans.append(span)
    return tuple(spans)


def template_tensor_spans(kernel, signature, graph_lookup):
    # Autotuning uses a temporary buf_out that is not registered in V.graph.
    # Read its actual layout from the template, as Inductor's own codegen does.
    from torch._inductor.codegen.common import TensorArg
    nodes = [*kernel.input_nodes, kernel.output_node]
    spans = []
    for arg in signature:
        if not isinstance(arg, TensorArg):
            return ()
        candidates = [node for node in nodes if node.get_name() == arg.buffer]
        try:
            node = graph_lookup(arg.buffer)
            if node is not None:
                candidates.append(node)
        except RuntimeError as error:
            if 'Failed to find buffer matching name' not in str(error):
                raise
        sizes = [static_tensor_spans([arg], lambda _, node=node: node) for node in candidates]
        if not sizes or not all(sizes):
            return ()
        # Multiple views can share a base argument. Cover all of their spans.
        spans.append(max(size[0] for size in sizes))
    return tuple(spans)


def _native_row_bias_mm(inp, mat1, mat2, *, out, alpha=1, beta=1):
    """Native FP32 contraction and row add without materializing an M*N bias."""
    import torch
    assert alpha == beta == 1
    assert inp.ndim == 2 and inp.stride(0) == 0
    torch.mm(mat1, mat2, out=out)
    torch.add(out, inp[0], out=out)
    return out


@lru_cache(None)
def _native_row_bias_choice():
    from torch._inductor.select_algorithm import ExternKernelChoice
    return ExternKernelChoice(_native_row_bias_mm, name='p800_native_row_bias_mm')


def template_boundary_eligibility(node):
    """Only compose validated additions/casts with the native SDNN template.

    In particular, x * sigmoid(1.702*x) currently lowers to an incorrect EW
    table composition. Keep its original pointwise graph outside the template.
    Unknown operations are retained as separate kernels, not discarded.
    """
    allowed = {'load', 'store', 'index_expr', 'constant', 'add', 'to_dtype'}
    observed = set()
    for part in node.get_nodes():
        body = getattr(part, '_body', None)
        if body is None:
            return False, ['unknown_body']
        for op in body.get_nodes():
            if op.op in ('placeholder', 'output'):
                continue
            if op.op == 'call_module' and op.target == 'get_index':
                continue
            if op.op != 'call_method' or not isinstance(op.target, str):
                return False, ['unknown_operation']
            # LoopBody's index lookup is bookkeeping, not an arithmetic op.
            if op.target != 'get_index':
                observed.add(op.target)
    return observed <= allowed, sorted(observed)


def eligible_shape(lhs, rhs):
    import torch
    import sympy
    if lhs.get_dtype() != torch.float32 or rhs.get_dtype() != torch.float32:
        return None
    a, b = lhs.get_size(), rhs.get_size()
    if len(a) != 2 or len(b) != 2:
        return None
    if not all(isinstance(x, (int, sympy.Integer)) for x in (*a, *b)):
        return None
    m, k, k2, n = map(int, (*a, *b))
    if k != k2 or min(m, n, k) < 512:
        return None
    return m, n, k


def extended_configs(configs):
    """Add seven tiles; never mutate Inductor's shared heuristic instance."""
    if not configs:
        return list(configs)
    existing = {(c.block_m, c.block_n, c.block_k, c.num_stages, c.num_warps)
                for c in configs}
    result = list(configs)
    for m in (128, 256):
        for n in (128, 256):
            for k in (64, 128):
                if (m, n, k) == (128, 128, 64) or (m, n, k, 3, 8) in existing:
                    continue
                result.append(replace(configs[0], block_m=m, block_n=n,
                                      block_k=k, num_stages=3, num_warps=8))
    return result


def full_k_configs(configs, k):
    """Keep one SDNN dot across K, avoiding FP32 partial-sum boundaries.

    This is an alternative numerical contract, not a claim that all FP32
    GEMMs must be bitwise identical. The full model validator still decides.
    Native ATen remains available if the XPU resource check rejects a tile.
    """
    # Padded K and larger single-dot descriptors need separate device
    # qualification. In particular, do not turn an unverified K mask into
    # an autotuning launch merely because the SRAM estimate fits.
    if not configs or not isinstance(k, int) or k <= 0 or k > 1024 or k & (k - 1):
        return []
    extent = 1 << (k - 1).bit_length()
    tiles = [(m, n, 1, 8) for m, n in
             ((64, 64), (64, 128), (128, 128), (256, 128), (256, 256))]
    # A full-K tile can exhaust L1D before it exhausts UB. A smaller M and
    # wider N leave room for two L1 slots and reduce repeated A transfers.
    # Grouping is a separate candidate dimension: it trades A reuse against
    # B reuse and must be timed for the actual shape, not fixed globally.
    tiles += [(128, 128, 2, 8), (64, 256, 2, 8),
              (64, 256, 2, 32), (64, 256, 2, 64)]
    return [replace(configs[0], block_m=m, block_n=n, block_k=extent,
                    num_stages=stages, num_warps=8, group_m=group)
            for m, n, stages, group in tiles]


def _valid_epilogue_timing(scheduler, module, device, record):
    """Retry invalid measurements without retaining their kernel_perf entry."""
    import math
    from pathlib import Path
    from torch._inductor.codecache import cache_dir
    for attempt in range(3):
        ms, path = scheduler.benchmark_codegened_module(module, device)
        if math.isfinite(ms) and ms > 0:
            return ms, path
        # Positive infinity is Inductor's compile/resource-rejection sentinel,
        # not a measured duration. Leave that rejection outside the candidates.
        if ms == float('inf'):
            return ms, path
        record.setdefault('invalid_measurements', []).append(str(ms))
        source = Path(module.__file__).resolve()
        cached = source.with_suffix('.kernel_perf')
        if Path(path).resolve() == source and source.is_relative_to(Path(cache_dir()).resolve()):
            cached.unlink(missing_ok=True)
    return float('nan'), path


def benchmark_full_k_epilogue(scheduler, node1, node2, audit):
    """Measure legal fused tiles instead of pruning by their unfused time.

    A template can absorb transfers, reorders and launches from its epilogue.
    Its unfused time is therefore not a lower bound for the fused region, and
    the fastest unfused tile need not be the fastest fused tile. This function
    only runs in the identified P800 full-K scope after can_fuse's legal checks.
    It retains the normal unfused alternative and whole-model accuracy gate.
    """
    import math
    import torch
    from torch._inductor import config, ir
    from torch._inductor.codecache import PyCodeCache
    if not node1.is_template() or node2.is_template() or config.multi_kernel_hints:
        return None
    multi = node1.get_template_node()
    if not isinstance(multi, ir.MultiTemplateBuffer) or multi.get_dtype() != torch.float32:
        return None
    allowed, operations = template_boundary_eligibility(node2)
    if not allowed or not {'add', 'to_dtype'} & set(operations):
        return None
    timings = multi.choice_timings()
    valid = {c: float(t) for c, t in timings.items()
             if math.isfinite(float(t)) and float(t) > 0}
    choices = sorted(((c, t) for c, t in valid.items()
                      if isinstance(c, ir.TritonTemplateCallerBase)), key=lambda x: x[1])
    event = dict(event='full_k_epilogue_comparison', operations=operations,
                 output_size=[str(x) for x in multi.get_size()],
                 template_choices=len(choices), rejected_unfused_timings=len(timings)-len(valid),
                 candidates=[], selected=None)
    audit.append(event)
    if not choices:
        event['reason'] = 'no_valid_template_timing'
        return False
    ms1 = min(valid.values())
    pointwise = PyCodeCache.load(scheduler.generate_kernel_code_from_nodes(
        node2.get_nodes(), benchmark_kernel=True))
    ms2, _ = _valid_epilogue_timing(scheduler, pointwise, node2.get_device(), event)
    if not math.isfinite(ms2) or ms2 <= 0:
        event['reason'] = 'invalid_pointwise_timing'
        return False
    event.update(unfused_matrix_ms=ms1, pointwise_ms=ms2,
                 unfused_region_ms=ms1+ms2)
    nodes = [*node1.get_nodes(), *node2.get_nodes()]
    device = nodes[0].get_device()
    best, best_ms = None, float('inf')
    fused_timings = {}
    for choice, unfused_ms in choices:
        row = dict(choice=choice.name, description=choice.description, unfused_ms=unfused_ms,
                   excluded_by_old_bound=unfused_ms >= ms1+ms2)
        event['candidates'].append(row)
        # The domain is already bounded to the qualified full-K configurations.
        # Do not drop all but one tile based on a different, unfused program.
        try:
            with multi.swap_as_triton_caller(choice):
                source = scheduler.generate_kernel_code_from_nodes(nodes, benchmark_kernel=True)
                module = PyCodeCache.load(source)
                ms, path = _valid_epilogue_timing(scheduler, module, device, row)
            row['source_path'] = path
        except Exception as error:
            row['compile_or_benchmark_rejection'] = repr(error)
            continue
        if not math.isfinite(ms) or ms <= 0:
            row['invalid_fused_timing'] = str(ms)
            continue
        row['fused_ms'] = ms
        fused_timings[choice] = ms
        if ms < best_ms:
            best, best_ms = choice, ms
    if best is not None and best_ms < ms1+ms2:
        multi.finalize_as_triton_caller(best)
        multi._choice_timings[None] = fused_timings
        event.update(selected=best.name, fused_ms=best_ms, reason='measured_fused_region_faster')
        return True
    event.update(reason='no_faster_valid_fused_region', fused_ms=best_ms if best is not None else None)
    return False


@contextmanager
def p800_gemm_candidates(device, audit, *, full_k=False, native_bias=False):
    import torch
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import config
    from torch._inductor.lowering import lowerings
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.virtualized import V
    from torch._inductor.kernel import mm as registration  # noqa: F401
    from torch._inductor.select_algorithm import TritonTemplateKernel
    from ._p800_inductor_gelu import _on_p800
    from ._p800_matmul import native_template_eligibility
    if not _on_p800(device):
        raise ValueError('P800 matrix candidates require the identified XPU backend')
    from torch._inductor.template_heuristics.triton import BaseConfigHeuristic
    with compile_lock, native_template_eligibility(device):
        previous_choices = V.choices

        class Choices:
            def __getattr__(self, name):
                return getattr(previous_choices, name)

            def get_mm_configs(self, inputs, layout, templates, name, *args, **kwargs):
                if _ROW_BIAS.get() and templates == [registration.aten_bias_addmm]:
                    return []
                if _ROW_BIAS.get() and templates == [registration.aten_addmm]:
                    return [_native_row_bias_choice().bind(inputs.nodes(), layout, alpha=1, beta=1)]
                return previous_choices.get_mm_configs(inputs, layout, templates, name, *args, **kwargs)

        original_fuse = Scheduler.can_fuse
        original_speedup = Scheduler.speedup_by_fusion
        original_jit_lines = TritonTemplateKernel.jit_lines

        def jit_lines(kernel):
            text = original_jit_lines(kernel)
            if not full_k or kernel.use_jit:
                return text
            _, _, signature, _ = kernel.args.python_argdefs()
            spans = template_tensor_spans(kernel, signature, V.graph.get_buffer)
            if spans and 'triton_meta={' in text:
                from pathlib import Path
                kernel.triton_meta['xpu_sdnn_tensor_bytes'] = spans
                text = _sdnn_template_source(text, spans, Path(__file__).resolve())
            return text

        def can_fuse(scheduler, node1, node2):
            # The multi-output template shortcut returns before V.choices.
            # This is a correctness boundary, so check before that shortcut.
            boundaries = []
            if node1.is_template():
                boundaries.append(('epilogue', node2))
            if node2.is_template():
                boundaries.append(('prologue', node1))
            for boundary, pointwise in boundaries:
                allowed, operations = template_boundary_eligibility(pointwise)
                if not allowed:
                    audit.append(dict(event='separate_template_boundary', boundary=boundary,
                                      operations=operations,
                                      reason='unvalidated_sdnn_pointwise'))
                    return False
            return original_fuse(scheduler, node1, node2)

        def speedup(scheduler, node1, node2):
            result = benchmark_full_k_epilogue(scheduler, node1, node2, audit) if full_k else None
            return original_speedup(scheduler, node1, node2) if result is None else result

        original_configs = BaseConfigHeuristic.get_mm_configs
        original_shared = BaseConfigHeuristic._get_exceeding_shared_memory_checker
        ops = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)
        originals = {op: lowerings[op] for op in ops}

        def configs(self):
            if not _ACTIVE.get():
                return original_configs(self)
            return partial(self.preprocess_mm_configs,
                           configs=(full_k_configs(self.mm_configs, _ACTIVE.get()[2])
                                    if full_k else extended_configs(self.mm_configs)))

        def shared(self):
            return None if _ACTIVE.get() else original_shared(self)

        def wrap(op, original):
            def lower(*args, **kwargs):
                lhs, rhs = args[-2:]
                shape = eligible_shape(lhs, rhs)
                if op is torch.ops.aten.addmm.default and args[0].get_dtype() != torch.float32:
                    shape = None
                if shape is None:
                    return original(*args, **kwargs)
                if full_k and (shape[2] > 1024 or shape[2] & (shape[2] - 1)):
                    audit.append(dict(operator=str(op), m=shape[0], n=shape[1], k=shape[2],
                        event='retained_native', reason='unqualified_single_dot_k_extent'))
                    return original(*args, **kwargs)
                audit.append(dict(operator=str(op), m=shape[0], n=shape[1], k=shape[2],
                    dtype='float32', backends=['ATEN', 'TRITON'],
                    nvidia_shared_memory_estimate=False,
                    xpu_compiler_resource_checks=True, extra_tiles=6 if full_k else 7,
                    configuration_variants=9 if full_k else None,
                    accumulation='single_full_k_dot' if full_k else 'partitioned_k'))
                token = _ACTIVE.set(shape)
                row_bias = (native_bias and op is torch.ops.aten.addmm.default
                            and tuple(args[0].get_size()) == (shape[1],)
                            and all(type(kwargs.get(x, 1)) in (int, float)
                                    and kwargs.get(x, 1) == 1 for x in ('alpha', 'beta')))
                bias_token = _ROW_BIAS.set(row_bias)
                if row_bias:
                    audit.append(dict(event='native_row_bias_candidate', m=shape[0], n=shape[1], k=shape[2],
                                      materialized_bias=False, template_epilogue_eligible=True))
                try:
                    with config.patch({'max_autotune_gemm': True,
                                       'max_autotune_gemm_backends': 'ATEN,TRITON'}):
                        # The grouped schedule normally splits addmm before
                        # selection. Here keep its efficient native alternative
                        # alongside the template so safe epilogues can still fuse.
                        lower_addmm = registration.tuned_addmm if row_bias else original
                        return lower_addmm(*args, **kwargs)
                finally:
                    _ROW_BIAS.reset(bias_token)
                    _ACTIVE.reset(token)
            return lower
        try:
            Scheduler.can_fuse = can_fuse
            Scheduler.speedup_by_fusion = speedup
            TritonTemplateKernel.jit_lines = jit_lines
            BaseConfigHeuristic.get_mm_configs = configs
            BaseConfigHeuristic._get_exceeding_shared_memory_checker = shared
            for op, original in originals.items():
                lowerings[op] = wrap(op, original)
            with V.set_choices_handler(Choices()):
                yield
        finally:
            Scheduler.can_fuse = original_fuse
            Scheduler.speedup_by_fusion = original_speedup
            TritonTemplateKernel.jit_lines = original_jit_lines
            for op, original in originals.items():
                lowerings[op] = original
            BaseConfigHeuristic.get_mm_configs = original_configs
            BaseConfigHeuristic._get_exceeding_shared_memory_checker = original_shared
