"""Specialize the FP32 F06 arithmetic for low-precision global-memory operands."""
import torch
from torch.fx import Node


def sdk_store_candidate(core, inputs):
    return (len(inputs) == 3 and inputs[0].dtype == torch.bfloat16
            and inputs[0].ndim == inputs[1].ndim == 2
            and inputs[0].shape[1] == inputs[1].shape[0] == 64
            and inputs[0].shape[0] >= 1024 and inputs[1].shape[1] >= 512 and inputs[1].shape[1] % 256 == 0
            and sum(n.op in ('call_function', 'call_method') for n in core.graph.nodes) == 2)


def compile_body(core, inputs, *, split=False, sdk_store=False, pointer_gelu=False):
    from . import _p800_flat, _p800_fp32_mm, _p800_native_mm, _p800_reduce
    if not inputs or inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if any(t.dtype != inputs[0].dtype or not t.is_contiguous() for t in inputs):
        return None
    dtype = str(inputs[0].dtype).removeprefix('torch.')
    softmax = _p800_reduce.match(core, inputs) if len(inputs) == 2 else None
    if softmax is not None and softmax['kind'] == 'add_softmax':
        if split:
            return None
        kernel, proof = _p800_reduce.compile(dict(softmax, final_round_only=dtype != 'float32'), inputs)
    else:
        if inputs[0].dtype == torch.float32:
            return None
        layout = tuple(torch.empty_strided(t.shape, t.stride(), dtype=torch.float32, device=t.device) for t in inputs)
        mm = _p800_native_mm.match(core, layout)
        if mm is not None:
            m, n, k = mm['M'], mm['N'], mm['K']
            if min(m, n, k) < 1 or max(m*n, m*k, k*n) > 0x7fffffff:
                return None
            if sdk_store:
                if mm['act'] != 0 or dtype != 'bfloat16' or not sdk_store_candidate(core, inputs):
                    return None
                kernel, proof = _p800_fp32_mm.prepare_sdk_store(mm)
            elif split:
                kernel, proof = _p800_fp32_mm.prepare_split(mm)
            else:
                kernel, proof = _p800_fp32_mm.prepare(mm, final_only=True)
        else:
            if split:
                return None
            for module in (_p800_flat, _p800_reduce):
                spec = module.match(core, layout)
                if spec is not None:
                    kernel, proof = module.compile(dict(spec, dtype=dtype, final_round_only=True,
                                                      pointer_gelu=pointer_gelu), inputs)
                    break
            else:
                return None
    out = next(n for n in core.graph.nodes if n.op == 'output').args[0]
    def run(*xs):
        result = kernel(xs)
        if isinstance(out, tuple):
            return result
        if isinstance(out, Node):
            return result[0]
        return list(result)
    proof.update(compute_dtype='float32', input_conversion='on_load', output_dtype=dtype,
                 output_conversion='sdk_store' if sdk_store else 'on_store',
                 kernel_count=None if sdk_store else (2 if split else 1))
    return run, proof


def select(core, inputs, reference, tolerance, *, single_kernel=False, public_timer=None, shared=None):
    """Compare validated boundary-fused implementations; all math stays FP32."""
    import statistics
    from ._resource_benchmark import validate, GraphMeasurement, paired_end_to_end, assert_close
    from ._p800_fp32_region import _MATMUL
    compare = getattr(reference, '_f06_assert_close', assert_close)
    is_matrix = any(node.target in _MATMUL for node in core.graph.nodes)
    candidates = {}
    records = []
    schedules = ['fp32_inline']
    if not is_matrix:
        from ._resource_autotune import _add_gelu_mode
        ops = [n for n in core.graph.nodes if n.op in ('call_method', 'call_function')]
        if _add_gelu_mode(ops) == 'none':
            schedules.append('fp32_inline_pointer')
    if is_matrix and not single_kernel:
        schedules.append('fp32_inline_split')
        if sdk_store_candidate(core, inputs):
            schedules.append('fp32_product_sdk_store')
    for name in schedules:
        sdk_store = name == 'fp32_product_sdk_store'
        try:
            if name == 'fp32_inline_pointer':
                compiled = compile_body(core, inputs, pointer_gelu=True)
            else:
                compiled = compile_body(core, inputs, split=name != 'fp32_inline', sdk_store=sdk_store)
            if compiled is None:
                continue
            fn, proof = compiled
            checks = validate(fn, reference, inputs, tolerance)
            # The FP32/low boundary is sensitive at ties, especially in GEMM.
            for seed in (97, 231):
                gen=torch.Generator(device='cpu').manual_seed(seed)
                data = tuple(torch.randn(t.shape, generator=gen, device='cpu').to(device=t.device, dtype=t.dtype)
                             for t in inputs)
                compare(fn(*data), reference(*data), tolerance)
            if is_matrix:
                # At this magnitude all five activations preserve positive values.
                # A truncating BF16 DMA must fail even if ordinary inputs fit tolerance.
                a, b, bias = [torch.zeros_like(t) for t in inputs]
                dt = a.dtype
                step = 64 * torch.finfo(dt).eps
                half = torch.tensor(step/2, dtype=dt, device='cpu')
                offsets = [half.nextafter(torch.zeros_like(half)).item(), half.item(),
                           half.nextafter(torch.full_like(half, float('inf'))).item()]
                columns = torch.arange(bias.numel(), device='cpu') % 6
                b[0].copy_(torch.tensor([64.]*3 + [64.+step]*3, dtype=dt)[columns])
                bias.copy_(torch.tensor(offsets*2,dtype=dt)[columns])
                a[:,0] = 1.
                got, want = fn(a,b,bias), reference(a,b,bias)
                if isinstance(got,(list,tuple)):
                    got, want = got[0],want[0]
                if not torch.equal(got.cpu().view(torch.int16), want.to(dt).cpu().view(torch.int16)):
                    raise ValueError('final output does not preserve SDK rounding at both midpoint parities')
            graph = GraphMeasurement(fn, reference, inputs, tolerance, calls=8)
            checks['graph_fresh_data'] = graph.fresh_data_passed
            graph.close()
            if sdk_store:
                # Verify the product. The SDK cast is external, so its launch count
                # and physical resource come only from the runtime trace.
                product, = proof['regions']
                if not product['resource_verified'] or len(product['kernel_evidence']) != 1:
                    raise RuntimeError('missing verified SDNN product before the SDK cast')
            elif not proof.get('resource_verified') or len(proof.get('kernel_evidence', ())) != proof['kernel_count']:
                raise RuntimeError('incomplete physical resource evidence for the inlined program')
            candidates[name] = (fn, proof)
            records.append(dict(name=name,status='passed',proof=proof,checks=checks))
        except Exception as exc:
            records.append(dict(name=name,status='rejected',error=type(exc).__name__ + ': ' + str(exc)))
    if not candidates:
        return None, records, {}
    times = {}
    winner = next(iter(candidates))
    functions = {name: item[0] for name, item in candidates.items()}
    if shared is not None and not single_kernel:
        # The canonical FP32 plan has already passed the caller's boundary
        # checks. Measure the cost of its actual SDK conversions as well.
        functions['sdk_eager'] = shared
    if len(functions) > 1:
        timer = public_timer or paired_end_to_end
        for seed in (7103, 7203):
            samples = timer(functions, inputs, seed=seed)
            for name in functions:
                times.setdefault(name, []).append(statistics.median(samples[name]))
        # A millisecond-scale launch interleaved with microsecond arms can
        # change their ranking. Remeasure competitive arms without that launch.
        fastest = [min(t[i] for t in times.values()) for i in range(2)]
        finalists = {name: fn for name, fn in functions.items()
                     if any(a <= 4 * b for a, b in zip(times[name], fastest))}
        if 1 < len(finalists) < len(functions):
            isolated = {name: [] for name in finalists}
            for seed in (7303, 7403):
                samples = timer(finalists, inputs, seed=seed)
                for name in finalists:
                    isolated[name].append(statistics.median(samples[name]))
            for name, samples in isolated.items():
                times[name].extend(samples)
            functions = finalists
            winner = 'sdk_eager' if 'sdk_eager' in finalists else next(iter(finalists))
        for name in functions:
            if all(a <= b * .97 for a, b in zip(times[name][-2:], times[winner][-2:])):
                winner = name
    if winner == 'sdk_eager':
        return None, records, times
    fn, proof = candidates[winner]
    return (winner, fn, proof), records, times
