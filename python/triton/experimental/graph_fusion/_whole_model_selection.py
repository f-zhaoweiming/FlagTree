"""Select complete inference call paths using validated end-to-end samples.

The caller supplies independent model routes and the public output contract.
Preparation runs outside timing; the selected callable has no per-call router.
"""
from dataclasses import dataclass
import math
import statistics


@dataclass
class ModelRoute:
    call: object
    activate: object = lambda: None


def select_whole_model(routes, *, validation_inputs, validate, measure,
                       record, rounds=4, preferred=None, minimum_gain=0.01,
                       max_remeasurements=1, expanded_routes=None):
    """Validate all supplied examples, then compare complete-route means.

    ``measure(call)`` must include input preparation, output conversion and
    completion synchronization, and return positive finite raw seconds.
    Explicit calibration keeps candidate compilation outside steady timing.
    No candidate is accepted on a subgraph-only correctness result.
    Prepared pipelines may supply resource variants; these undergo the same
    complete-output gate. Optional expanded_routes receives the named callable
    map for subsequent diagnostics. The returned callable has no warm router.
    """
    if rounds < 2 or rounds % 2:
        raise ValueError('Use an even number of forward/reverse rounds')
    if not validation_inputs or not routes or not 0 <= minimum_gain < 1:
        raise ValueError('Provide routes, validation examples and a valid gain margin')
    if type(max_remeasurements) is not int or not 0 <= max_remeasurements <= 2:
        raise ValueError('Use zero to two bounded timing remeasurements')
    record.pop('route', None)
    record.pop('mean_seconds', None)
    record.pop('gain_check', None)
    record.pop('timing_attempts', None)
    record.pop('interface_audit', None)
    record.update(state='validating', scope='complete_model_call', rejected=[], candidates={}, rounds=rounds,
                  minimum_gain=minimum_gain, statistic='arithmetic_mean_of_raw_seconds', resource_variants={})
    valid = {}
    pending = list(routes.items())
    all_routes = dict(routes)
    for name, route in pending:
        # A device execution failure can poison the context. Propagate it,
        # rather than attempting another route on the same context.
        route.activate()
        for args, kwargs in validation_inputs:
            output = route.call(*args, **kwargs)
            try:
                validate(name, output, args, kwargs)
            except (AssertionError, ValueError) as error:
                record['rejected'].append(dict(route=name, stage='full_output_validation',
                                               error=str(error)))
                break
            finally:
                # Validation outputs must not remain live during calibration.
                # A retained full-model result can change allocation/cache
                # behavior compared with the eventual warm entry point.
                del output
        else:
            valid[name] = route
            record['candidates'][name] = dict(validated_examples=len(validation_inputs),
                                              raw_seconds=[], round_means=[])
        for factory_name in ('_f06_resource_variants', '_f06_buffer_variants', '_f06_gemm_variants'):
            variants = getattr(route.call, factory_name, None)
            if variants is None:
                continue
            # Even a precision-rejected baseline may have a valid resource
            # alternative. Each is validated on every complete public output.
            for kind, call in variants().items():
                candidate = name + '__' + kind
                if candidate in all_routes:
                    raise ValueError('Duplicate whole-model resource candidate: '+candidate)
                alternative = ModelRoute(call, route.activate)
                all_routes[candidate] = alternative
                pending.append((candidate, alternative))
                record['resource_variants'][candidate] = dict(parent=name, resource=kind,
                    dimension={'_f06_buffer_variants': 'buffer', '_f06_resource_variants': 'resource',
                               '_f06_gemm_variants': 'matrix'}[factory_name])
    if expanded_routes is not None:
        expanded_routes.update(all_routes)
    if not valid:
        record['state'] = 'no_valid_route'
        raise RuntimeError('No complete model route passed the output contract')
    record['state'] = 'measuring'
    names = list(valid)
    for attempt in range(max_remeasurements + 1):
        for number in range(rounds):
            order = names if number % 2 == 0 else names[::-1]
            for name in order:
                route = valid[name]
                route.activate()
                samples = list(measure(route.call))
                if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                    # A retry must never turn invalid timestamps into a winner.
                    record['state'] = 'invalid_timing'
                    raise RuntimeError('Invalid complete-model timing for ' + name)
                item = record['candidates'][name]
                item['raw_seconds'].extend(samples)
                item['round_means'].append(statistics.mean(samples))
        means = {name: statistics.mean(item['raw_seconds'])
                 for name, item in record['candidates'].items()}
        winner = min(means, key=means.get)
        comparison = preferred
        parent = record['resource_variants'].get(winner, {}).get('parent')
        if parent in valid and (comparison not in valid or means[parent] < means[comparison]):
            # A resource change must improve its complete baseline stably,
            # including when original-dtype eager failed the FP32 contract.
            comparison = parent
        if comparison in valid and means[winner] >= means[comparison] * (1 - minimum_gain):
            winner = comparison
        if comparison in valid and winner != comparison:
            chosen = record['candidates'][winner]['round_means']
            baseline = record['candidates'][comparison]['round_means']
            wins = sum(a < b * (1 - minimum_gain) for a, b in zip(chosen, baseline))
            both_orders = all(statistics.mean(chosen[offset::2]) <
                              statistics.mean(baseline[offset::2]) * (1 - minimum_gain)
                              for offset in (0, 1))
            chosen_spread = (max(chosen) - min(chosen)) / statistics.mean(chosen)
            baseline_spread = (max(baseline) - min(baseline)) / statistics.mean(baseline)
            stationary = max(chosen_spread, baseline_spread) <= 0.05
            consistent = wins >= math.ceil(0.75 * rounds) and both_orders
            stable = consistent and stationary
            record['gain_check'] = dict(candidate=winner, preferred=comparison,
                                        winning_rounds=wins, rounds=rounds,
                                        both_orders_faster=both_orders,
                                        candidate_round_spread=chosen_spread,
                                        preferred_round_spread=baseline_spread,
                                        stationary=stationary, stable=stable)
            if consistent and not stationary and attempt < max_remeasurements:
                # Keep the complete noisy block for audit, then measure every
                # valid route again. Do not cherry-pick rounds or pool them with
                # the fresh block. All original admission thresholds still apply.
                record.setdefault('timing_attempts', []).append(dict(
                    attempt=attempt, reason='nonstationary_comparison',
                    candidates=record['candidates'], mean_seconds=means,
                    gain_check=record['gain_check']))
                record['candidates'] = {
                    name: dict(validated_examples=item['validated_examples'],
                               raw_seconds=[], round_means=[])
                    for name, item in record['candidates'].items()}
                record.pop('gain_check', None)
                continue
            if not stable:
                winner = comparison
        break
    valid[winner].activate()
    record.update(state='selected', route=winner, mean_seconds=means)
    # Audit only at preparation time. The returned callable remains the chosen
    # direct entry; warm calls do not repeat validation, ranking or dispatch.
    record['interface_audit'] = {
        name: dict(status=('selected' if name == winner else
                           'slower_measured' if elapsed > means[winner] * (1 + minimum_gain)
                           else 'similar_time'),
                   mean_seconds=elapsed, relative_to_selected=elapsed / means[winner],
                   replacement=winner if elapsed > means[winner] * (1 + minimum_gain) else None,
                   scope='this validated complete model and calibration')
        for name, elapsed in means.items()
    }
    return valid[winner].call


def output_cast_like(output, exemplar):
    """Cast only public floating outputs, preserving the original output tree."""
    import torch
    import numpy as np
    from torch.utils._pytree import tree_flatten, tree_unflatten
    values, structure = tree_flatten(output)
    expected, target_structure = tree_flatten(exemplar)
    if structure != target_structure:
        raise ValueError('Native FP32 output structure differs from the model contract')
    result = []
    for value, target in zip(values, expected):
        if isinstance(target, torch.Tensor):
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError('Native FP32 output tensor shape changed')
            if value.device != target.device or value.layout != target.layout:
                raise ValueError('Native FP32 output device or layout changed')
            if target.is_floating_point():
                value = value.to(dtype=target.dtype)
            elif value.dtype != target.dtype:
                raise ValueError('Non-floating output dtype changed')
        elif isinstance(target, np.ndarray):
            if not isinstance(value, np.ndarray) or value.shape != target.shape:
                raise ValueError('Native FP32 output array shape changed')
            if np.issubdtype(target.dtype, np.floating):
                value = value.astype(target.dtype, copy=False)
            elif value.dtype != target.dtype:
                raise ValueError('Non-floating output array dtype changed')
        elif type(value) is not type(target):
            raise ValueError('Native FP32 output leaf type changed')
        result.append(value)
    return tree_unflatten(result, structure)
