"""Correctness gates and common graph-replay timing for Inductor plan tuning."""
import math
import random
import statistics
import time


def _leaves(value):
    import torch
    from torch.utils._pytree import tree_flatten
    leaves, spec = tree_flatten(value)
    if not leaves or any(not isinstance(x, torch.Tensor) for x in leaves):
        raise TypeError('resource calibration requires tensor outputs')
    return leaves, spec


def _cpu(value):
    from torch.utils._pytree import tree_map
    return tree_map(lambda x: x.detach().cpu().clone(), value)


def assert_close(got, want, tolerance, *, output_dtype=None):
    import torch
    a, sa = _leaves(got)
    b, sb = _leaves(want)
    if sa != sb:
        raise AssertionError('output structure changed')
    for x, y in zip(a, b):
        if output_dtype is not None:
            if x.dtype != output_dtype or y.dtype != torch.float32 or x.shape != y.shape:
                raise AssertionError('expected the model output dtype and shape against an FP32 eager reference')
            if output_dtype in (torch.float16, torch.bfloat16):
                _assert_final_round(x, y, tolerance)
                continue
        x, y = x.detach().cpu(), y.detach().cpu()
        torch.testing.assert_close(x, y, equal_nan=True, **tolerance)
        assert torch.equal(torch.isnan(x), torch.isnan(y)), 'NaN propagation changed'
        assert torch.equal(torch.isinf(x), torch.isinf(y)), 'Inf propagation changed'


def _assert_final_round(output, reference, tolerance):
    """Check whether one final SDK cast can map an in-tolerance FP32 result to the observed value."""
    import torch
    actual = output.detach().cpu()
    ref = reference.detach().to(output.device)
    rounded = ref.to(output.dtype).cpu()
    mismatch = (actual != rounded) & ~(actual.isnan() & rounded.isnan())
    if not mismatch.any():
        return
    # Preserve the actual SDK's overflow and NaN conversion, including nonstandard payloads.
    target = ref.cpu()[mismatch]
    observed = actual[mismatch].float()
    if not (target.isfinite() & rounded[mismatch].isfinite() & observed.isfinite()).all():
        raise AssertionError('nonfinite output differs from the final FP32 eager conversion')
    # RNE is monotone on finite values. Project just the mismatches into the model dtype;
    # this admits adjacent bins at a true midpoint without granting an extra ULP everywhere.
    delta = tolerance['atol'] + tolerance['rtol'] * target.double().abs()
    low, high = target.double() - delta, target.double() + delta
    # Round the endpoints inward to FP32 so a zero or tiny tolerance cannot admit an
    # out-of-tolerance FP32 predecessor or successor.
    lower, upper = low.float(), high.float()
    lower = torch.where(lower.double() < low, torch.nextafter(lower, torch.full_like(lower, math.inf)), lower)
    upper = torch.where(upper.double() > high, torch.nextafter(upper, torch.full_like(upper, -math.inf)), upper)
    lower = lower.to(output.device).to(output.dtype).cpu().float()
    upper = upper.to(output.device).to(output.dtype).cpu().float()
    bad = (observed < lower) | (observed > upper)
    if bad.any():
        i = bad.nonzero()[0].item()
        raise AssertionError(f'output {observed[i].item()} is outside [{lower[i].item()}, {upper[i].item()}] '
                             f'after one final cast from the FP32 reference {target[i].item()}')


def _tensor_bytes(x):
    import torch
    return x.detach().cpu().contiguous().view(torch.uint8)


def _changed(inputs):
    import torch
    result = []
    for x in inputs:
        if not isinstance(x, torch.Tensor):
            result.append(x)
            continue
        # Nonuniform perturbation: adding a uniform constant would not detect
        # stale Softmax/LayerNorm output, owing to translation invariance.
        y = x.detach().cpu().clone()
        if y.is_floating_point():
            y.reshape(-1)[::7] = y.reshape(-1)[::7] * 0.7 + 0.03125
        result.append(y.to(device=x.device))
    return tuple(result)


def _validate(fn, reference, inputs, tolerance):
    import torch
    if set(tolerance) != {'atol', 'rtol'} or any(
            type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in tolerance.values()):
        raise ValueError('explicit finite nonnegative atol/rtol required')
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    attrs = list(reference.parameters()) + list(reference.buffers())
    originals = [_tensor_bytes(x).clone() for x in tensors + attrs]
    compare = getattr(reference, '_f06_assert_close', assert_close)
    checks = {}
    expected = _cpu(reference(*inputs))
    actual = fn(*inputs)
    compare(actual, expected, tolerance)
    checks['ordinary'] = True
    held = [actual] + [fn(*inputs) for _ in range(4)]
    for index in range(len(_leaves(actual)[0])):
        assert len({_leaves(y)[0][index].data_ptr() for y in held}) == len(held), 'output storage reused'
    changed = _changed(inputs)
    compare(fn(*changed), _cpu(reference(*changed)), tolerance)
    checks['changed'] = True
    for value in held:
        compare(value, expected, tolerance)
    checks['repeat'] = True
    floating = [i for i, x in enumerate(inputs)
                if isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel()]
    for index in floating:
        original = inputs[index].detach().cpu().contiguous()
        # Mixing NaN and infinities in one row can mask incorrect Inf handling
        # in a reduction. Exercise each operand, including broadcast operands.
        for label, marks in (('mixed', [math.nan, math.inf, -math.inf, 0., -0.]),
                             ('NaN', [math.nan]), ('+Inf', [math.inf]), ('-Inf', [-math.inf])):
            marked = original.clone()
            values = marked.new_tensor(marks[:marked.numel()])
            marked.view(-1)[:values.numel()] = values
            special = list(inputs)
            special[index] = marked.to(inputs[index].device)
            want = _cpu(reference(*special))
            try:
                compare(fn(*special), want, tolerance)
            except AssertionError as exc:
                raise AssertionError(f'special input {index} ({label}): {exc}') from exc
    # Eager implementations may lose precision on near-constant rows. A
    # numerically stable kernel must still respect this graph's eager contract.
    if any('layer_norm' in str(n.target) for n in reference.graph.nodes):
        conditioned = list(inputs)
        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor) and x.ndim == 2 and x.is_floating_point():
                conditioned[i] = (1000 + .1 * x.float()).to(x.dtype)
                break
        compare(fn(*conditioned), _cpu(reference(*conditioned)), tolerance)
    checks['special'] = True
    for original, tensor in zip(originals, tensors + attrs):
        assert torch.equal(original, _tensor_bytes(tensor)), 'input/parameter mutated'
    checks['immutable'] = True
    return checks


def validate(fn, reference, inputs, tolerance):
    """A failed candidate must not contaminate any later candidate's inputs."""
    import torch
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    tensors += list(reference.parameters()) + list(reference.buffers())
    originals = [x.detach().cpu().clone() for x in tensors]
    try:
        return _validate(fn, reference, inputs, tolerance)
    finally:
        with torch.no_grad():
            for tensor, original in zip(tensors, originals):
                tensor.copy_(original)


def _capture(fn, inputs, calls):
    import torch
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            output = fn(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def replay_us(graph, calls):
    import torch
    torch.cuda.synchronize()
    for repeats in (1, 4, 16):
        # Replace both event handles when their elapsed time is unusable.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        us = start.elapsed_time(end) * 1000 / (calls * repeats)
        if math.isfinite(us) and us > 0:
            return us
    raise InvalidReplayTiming('invalid replay elapsed time')


class InvalidReplayTiming(RuntimeError):
    """All event pairs failed to measure a completed replay."""


def _calibration_sample(graph, record):
    try:
        return graph.sample()
    except InvalidReplayTiming:
        # Keep the same validated capture and sample count. Execution and
        # correctness failures must still reject the candidate immediately.
        record['timing_retries'] += 1
        return graph.sample()


class GraphMeasurement:
    """One validated capture, reused for selection or independent final samples."""
    def __init__(self, fn, reference, inputs, tolerance, *, calls=None):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('benchmark capture cannot be nested')
        for _ in range(8):
            fn(*inputs)
        torch.cuda.synchronize()
        if calls is None:
            pilot, output = _capture(fn, inputs, 8)
            us = replay_us(pilot, 8)
            calls = max(1, min(4096, math.ceil(4000 / us)))
            del pilot, output
        self.calls = calls
        self.graph, self.output = _capture(fn, inputs, calls)
        self.inputs, self.fn = inputs, fn
        originals = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in inputs]
        compare = getattr(reference, '_f06_assert_close', assert_close)
        want = _cpu(reference(*inputs))
        compare(self.output, want, tolerance)
        try:
            changed = _changed(inputs)
            for x, y in zip(inputs, changed):
                if isinstance(x, torch.Tensor):
                    x.copy_(y)
            # Poison output so an empty/stale captured graph cannot pass even
            # for invariant expressions. Inputs/attributes may not alias output.
            output_leaves, _ = _leaves(self.output)
            input_ptrs = {x.data_ptr() for x in inputs if isinstance(x, torch.Tensor)}
            input_ptrs.update(x.data_ptr() for x in list(reference.parameters()) + list(reference.buffers()))
            for y in output_leaves:
                assert y.data_ptr() not in input_ptrs, 'captured output aliases input'
                y.fill_(float('nan'))
            self.graph.replay()
            torch.cuda.synchronize()
            compare(self.output, _cpu(reference(*inputs)), tolerance)
        finally:
            for x, y in zip(inputs, originals):
                if isinstance(x, torch.Tensor):
                    x.copy_(y)
        self.graph.replay()
        torch.cuda.synchronize()
        compare(self.output, want, tolerance)
        self.fresh_data_passed = True

    def sample(self):
        return replay_us(self.graph, self.calls)

    def close(self):
        import torch
        torch.cuda.synchronize()
        self.graph = self.output = None


def end_to_end(fn, inputs, count=31):
    import torch
    samples = []
    for _ in range(count):
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        fn(*inputs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1000)
    return samples


def paired_end_to_end(functions, inputs, *, count=128, seed=6103, warmup=24, before_call=None):
    """Ordinary-call wall latency; randomized forward/reverse pairs.

    Output release and completion synchronization are included; pre-sync and
    warmup are excluded. Even identical aliases are measured independently.
    """
    import torch
    if count < 2 or count % 2:
        raise ValueError('an even sample count >= 2 is required')
    for name, fn in functions.items():
        for _ in range(warmup):
            if before_call is not None:
                before_call(name)
            fn(*inputs)
    torch.cuda.synchronize()
    result = {name: [] for name in functions}
    names = list(functions)
    rng = random.Random(seed)
    for _ in range(count // 2):
        order = names[:]
        rng.shuffle(order)
        for direction in (order, list(reversed(order))):
            for name in direction:
                if before_call is not None:
                    before_call(name)
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                functions[name](*inputs)
                torch.cuda.synchronize()
                result[name].append((time.perf_counter_ns() - start) / 1000)
    return result


def evaluate(fn, reference, inputs, tolerance):
    checks = validate(fn, reference, inputs, tolerance)
    graph = GraphMeasurement(fn, reference, inputs, tolerance)
    try:
        samples = [graph.sample() for _ in range(7)]
        checks['graph_fresh_data'] = graph.fresh_data_passed
        return dict(status='passed', checks=checks, gpu_samples_us=samples,
                    gpu_us=statistics.median(samples), calls_per_graph=graph.calls)
    finally:
        graph.close()


class GraphGroup:
    """Independent captures of one callable, shared by aliases of that route."""
    def __init__(self, graphs):
        self.graphs = tuple(graphs)
        self.calls = self.graphs[0].calls
        assert all(g.calls == self.calls for g in self.graphs)

    def sample(self, capture_index):
        return self.graphs[capture_index].sample()

    def close(self):
        for graph in self.graphs:
            graph.close()


def measure_candidates(functions, reference, inputs, tolerance, *, retain=None, public_timer=None):
    """Common batch size, rotating capture order and three independent captures.

    P800 native operators can have capture-dependent latency. A separately
    captured alias is not a fair control. Keep the exact calibrated captures
    for final measurements, and share them when selection falls back to a
    baseline. Report capture dispersion instead of hiding it in one sample.
    """
    import traceback
    records, pilots, groups = {}, {}, {}
    for name, fn in functions.items():
        try:
            checks = validate(fn, reference, inputs, tolerance)
            record = dict(checks=checks, status='ready', timing_retries=0)
            pilot = GraphMeasurement(fn, reference, inputs, tolerance, calls=8)
            try:
                pilots[name] = statistics.median(_calibration_sample(pilot, record) for _ in range(3))
            finally:
                pilot.close()
            records[name] = record
        except Exception:
            records[name] = dict(status='rejected', error=traceback.format_exc())
    if not pilots:
        return records
    calls = max(8, min(2048, math.ceil(4000 / min(pilots.values()))))
    names = list(pilots)
    for capture_index in range(3):
        order = names[capture_index % len(names):] + names[:capture_index % len(names)]
        for name in order:
            if records[name]['status'] != 'ready':
                continue
            try:
                graph = GraphMeasurement(functions[name], reference, inputs, tolerance, calls=calls)
                groups.setdefault(name, []).append(graph)
            except Exception:
                records[name].update(status='rejected', error=traceback.format_exc())
    try:
        admitted = [name for name, captures in groups.items()
                    if len(captures) == 3 and records[name]['status'] == 'ready']
        by_capture = {name: [[], [], []] for name in admitted}
        ordinary_functions = {name: functions[name] for name in admitted}
        timer = public_timer or paired_end_to_end
        e2e_before = timer(ordinary_functions, inputs, seed=6103)
        # Warm every retained capture after validation/allocations, then
        # interleave candidates in both directions to limit time drift.
        def sample(name, index):
            try:
                return _calibration_sample(groups[name][index], records[name])
            except Exception:
                records[name].update(status='rejected', error=traceback.format_exc())
                return None

        for index in range(3):
            for name in admitted:
                if records[name]['status'] == 'ready':
                    sample(name, index)
        for order in [admitted, list(reversed(admitted))]:
            for sample_index in range(3):
                for offset in range(3):
                    index = (sample_index + offset) % 3
                    for name in order:
                        if records[name]['status'] == 'ready':
                            value = sample(name, index)
                            if value is not None:
                                by_capture[name][index].append(value)
        admitted = [name for name in admitted if records[name]['status'] == 'ready']
        e2e_after = timer({name: functions[name] for name in admitted}, inputs, seed=6203)
        for name in admitted:
            captures, samples = groups[name], by_capture[name]
            flat = [v for row in samples for v in row]
            records[name].update(status='passed', gpu_samples_us=flat,
                gpu_us=statistics.median(flat), calls_per_graph=calls,
                capture_samples_us=samples, capture_medians_us=[statistics.median(s) for s in samples])
            records[name].update(e2e_protocol='public_module_ab_ba_v1' if public_timer else 'randomized_ab_ba_v1',
                e2e_round_samples_us=[e2e_before[name], e2e_after[name]])
            if public_timer and 'vs_direct' in e2e_before:
                controls = (e2e_before['vs_direct'], e2e_after['vs_direct'])
                records[name]['vs_direct_round_samples_us'] = [round_[name] for round_ in controls]
                if name == 'eager':
                    records[name]['direct_eager_round_samples_us'] = [round_['direct_eager'] for round_ in controls]
            records[name]['checks']['graph_fresh_data'] = all(g.fresh_data_passed for g in captures)
            if retain is not None:
                retain[name] = GraphGroup(captures)
    finally:
        for name, captures in groups.items():
            if retain is None or name not in retain:
                for graph in captures:
                    graph.close()
    return records
