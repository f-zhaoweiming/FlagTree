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


def assert_close(got, want, tolerance):
    import torch
    a, sa = _leaves(got)
    b, sb = _leaves(want)
    if sa != sb:
        raise AssertionError('output structure changed')
    for x, y in zip(a, b):
        x, y = x.detach().cpu(), y.detach().cpu()
        torch.testing.assert_close(x, y, equal_nan=True, **tolerance)
        assert torch.equal(torch.isnan(x), torch.isnan(y)), 'NaN propagation changed'
        assert torch.equal(torch.isinf(x), torch.isinf(y)), 'Inf propagation changed'


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
    checks = {}
    expected = _cpu(reference(*inputs))
    actual = fn(*inputs)
    assert_close(actual, expected, tolerance)
    checks['ordinary'] = True
    held = [actual, fn(*inputs), fn(*inputs)]
    for index in range(len(_leaves(actual)[0])):
        assert len({_leaves(y)[0][index].data_ptr() for y in held}) == len(held), 'output storage reused'
    changed = _changed(inputs)
    assert_close(fn(*changed), _cpu(reference(*changed)), tolerance)
    checks['changed'] = True
    for value in held:
        assert_close(value, expected, tolerance)
    checks['repeat'] = True
    special = tuple(x.detach().clone() if isinstance(x, torch.Tensor) else x for x in inputs)
    first = next(x for x in special if isinstance(x, torch.Tensor) and x.is_floating_point())
    # Build on CPU: exclude all input preparation/transfer from GPU timing.
    marked = first.cpu()
    marked.reshape(-1)[:5] = torch.tensor([float('nan'), float('inf'), -float('inf'), 0., -0.], dtype=first.dtype)
    first.copy_(marked)
    assert_close(fn(*special), _cpu(reference(*special)), tolerance)
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
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); pilot.replay(); b.record(); b.synchronize()
            us = a.elapsed_time(b) * 1000 / 8
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError('invalid native GPU event timing')
            calls = max(1, min(4096, math.ceil(4000 / us)))
            del pilot, output
        self.calls = calls
        self.graph, self.output = _capture(fn, inputs, calls)
        self.inputs, self.fn = inputs, fn
        originals = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in inputs]
        want = _cpu(reference(*inputs))
        assert_close(self.output, want, tolerance)
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
            assert_close(self.output, _cpu(reference(*inputs)), tolerance)
        finally:
            for x, y in zip(inputs, originals):
                if isinstance(x, torch.Tensor):
                    x.copy_(y)
        self.graph.replay()
        torch.cuda.synchronize()
        assert_close(self.output, want, tolerance)
        self.fresh_data_passed = True

    def sample(self):
        import torch
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record(); self.graph.replay(); b.record(); b.synchronize()
        us = a.elapsed_time(b) * 1000 / self.calls
        if not math.isfinite(us) or us <= 0:
            raise RuntimeError('invalid replay elapsed time')
        return us

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


def paired_end_to_end(functions, inputs, *, count=128, seed=6103, warmup=24):
    """Ordinary-call wall latency; randomized forward/reverse pairs.

    Output release and completion synchronization are included; pre-sync and
    warmup are excluded. Even identical aliases are measured independently.
    """
    import torch
    if count < 2 or count % 2:
        raise ValueError('an even sample count >= 2 is required')
    for fn in functions.values():
        for _ in range(warmup):
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


def measure_candidates(functions, reference, inputs, tolerance, *, retain=None):
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
            pilot = GraphMeasurement(fn, reference, inputs, tolerance, calls=8)
            try:
                pilots[name] = statistics.median(pilot.sample() for _ in range(3))
            finally:
                pilot.close()
            records[name] = dict(checks=checks, status='ready')
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
        e2e_before = paired_end_to_end(ordinary_functions, inputs, seed=6103)
        # Warm every retained capture after validation/allocations, then
        # interleave candidates in both directions to limit time drift.
        for index in range(3):
            for name in admitted:
                groups[name][index].sample()
        for order in [admitted, list(reversed(admitted))]:
            for sample_index in range(3):
                for offset in range(3):
                    index = (sample_index + offset) % 3
                    for name in order:
                        by_capture[name][index].append(groups[name][index].sample())
        e2e_after = paired_end_to_end(ordinary_functions, inputs, seed=6203)
        for name in admitted:
            captures, samples = groups[name], by_capture[name]
            flat = [v for row in samples for v in row]
            records[name].update(status='passed', gpu_samples_us=flat,
                gpu_us=statistics.median(flat), calls_per_graph=calls,
                capture_samples_us=samples, capture_medians_us=[statistics.median(s) for s in samples])
            records[name].update(e2e_protocol='randomized_ab_ba_v1',
                e2e_round_samples_us=[e2e_before[name], e2e_after[name]])
            records[name]['checks']['graph_fresh_data'] = all(g.fresh_data_passed for g in captures)
            if retain is not None:
                retain[name] = GraphGroup(captures)
    finally:
        for name, captures in groups.items():
            if retain is None or name not in retain:
                for graph in captures:
                    graph.close()
    return records
