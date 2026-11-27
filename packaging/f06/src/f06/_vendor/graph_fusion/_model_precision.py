"""One complete-output precision contract for graph and model route admission."""
from collections.abc import Mapping
import math

import torch
from torch.utils._pytree import tree_flatten

from ._resource_benchmark import assert_close


def _leaf_paths(spec, prefix='output'):
    """Label the existing tree without requiring a separate keyed flattener.

    Transformers ModelOutput registers ordinary PyTree flattening, but some
    versions do not register flatten_with_keys_fn. Labels must not reject an
    otherwise supported public output type or change its flattening contract.
    """
    if spec.is_leaf():
        return [prefix]
    keys = range(len(spec.children_specs))
    if (isinstance(spec.type, type) and issubclass(spec.type, Mapping)
            and isinstance(spec.context, (tuple, list))
            and len(spec.context) == len(spec.children_specs)):
        keys = spec.context
    return [path for key, child in zip(keys, spec.children_specs)
            for path in _leaf_paths(child, prefix+'['+repr(key)+']')]


def _eager_error_metrics(actual, reference, baseline, tolerance):
    """Compare each leaf to the same FP32 oracle, using no added error margin."""
    if not isinstance(baseline, torch.Tensor) or (
            baseline.shape, baseline.dtype, baseline.layout, baseline.device) != (
            actual.shape, actual.dtype, actual.layout, actual.device):
        raise AssertionError('eager baseline output contract differs')
    if reference.dtype != torch.float32:
        raise AssertionError('eager error budget requires the original FP32 reference')
    atol, rtol = tolerance['atol'], tolerance['rtol']
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol > 0 and rtol >= 0):
        raise ValueError('eager error normalization requires finite atol > 0 and rtol >= 0')
    values = [x.to_dense() if x.layout != torch.strided else x
              for x in (actual, reference, baseline)]
    value, ref, eager = [x.detach().to(device='cpu', dtype=torch.float64) for x in values]
    # Nonfinite values cannot turn into an infinite error budget. Preserve
    # NaN locations and both infinity signs exactly as in the FP32 reference.
    for name, x in (('candidate', value), ('eager baseline', eager)):
        if not all(torch.equal(test(x), test(ref))
                   for test in (torch.isnan, torch.isposinf, torch.isneginf)):
            raise AssertionError(name + ' nonfinite propagation differs from FP32')
    finite = ref.isfinite()
    scale = atol + rtol * ref[finite].abs()

    def metrics(x):
        error = (x[finite] - ref[finite]).abs()
        if not error.numel():
            return dict(rmse=0., max_abs=0., max_normalized_abs=0.)
        return dict(rmse=error.square().mean().sqrt().item(),
                    max_abs=error.max().item(),
                    max_normalized_abs=(error / scale).max().item())

    observed, allowed = metrics(value), metrics(eager)
    return dict(candidate_error_metrics=observed, eager_error_metrics=allowed,
                error_metric_regressions=[name for name in observed if observed[name] > allowed[name]],
                error_normalization=dict(atol=atol, rtol=rtol,
                                         formula='abs(error)/(atol+rtol*abs(original_fp32))'),
                error_budget_margin=0., comparison='per-output error no worse than eager against original FP32')


def compare_model_outputs(actual, reference, tolerance, *, output_template=None,
                          allow_boxed_sequence=False, records=None, eager_baseline=None):
    """Check every output leaf against FP32, allowing only the final dtype cast.

    A public output template additionally fixes structure, dtype, device and
    layout. AOT's outer list/tuple convention is allowed only for graph calls.
    Records describe all leaves, even when one fails; they are never timed.
    An explicit eager_baseline opts into a different contract: each output's
    RMSE, maximum absolute error and maximum normalized error against the
    original FP32 reference must all be no greater than that baseline's.
    This does not relax the default strict-FP32 gate or compare to eager alone.
    """
    if allow_boxed_sequence and isinstance(actual, (tuple, list)) and isinstance(reference, (tuple, list)):
        actual, reference = tuple(actual), tuple(reference)
    leaves, structure = tree_flatten(actual)
    expected, expected_structure = tree_flatten(reference)
    if structure != expected_structure:
        raise AssertionError('whole-model output structure changed')
    baselines = None
    if eager_baseline is not None:
        if allow_boxed_sequence and isinstance(eager_baseline, (tuple, list)):
            eager_baseline = tuple(eager_baseline)
        baselines, baseline_structure = tree_flatten(eager_baseline)
        if structure != baseline_structure:
            raise AssertionError('eager baseline output structure differs')
    templates = None
    if output_template is not None:
        templates, template_structure = tree_flatten(output_template)
        if structure != template_structure:
            raise AssertionError('public model output structure changed')
    failures = []
    for index, (path, value, want) in enumerate(zip(_leaf_paths(structure), leaves, expected)):
        row = dict(path=path, passed=False)
        try:
            if isinstance(want, torch.Tensor):
                if not isinstance(value, torch.Tensor):
                    raise AssertionError('tensor output replaced by a non-tensor')
                if value.shape != want.shape or value.layout != want.layout:
                    raise AssertionError('output shape or layout changed')
                if templates is not None:
                    template = templates[index]
                    if not isinstance(template, torch.Tensor) or (
                            value.shape, value.dtype, value.device, value.layout) != (
                            template.shape, template.dtype, template.device, template.layout):
                        raise AssertionError('public output dtype, shape, device or layout changed')
                row.update(shape=list(value.shape), actual_dtype=str(value.dtype),
                           reference_dtype=str(want.dtype), elements=value.numel())
                if value.is_floating_point():
                    # Sparse outputs are analyzed densely, preserving the public
                    # tensors and their layout. The same final-cast gate applies.
                    got = value.to_dense() if value.layout != torch.strided else value
                    ref = want.to_dense() if want.layout != torch.strided else want
                    if records is not None:
                        x, y = got.detach().float().cpu(), ref.detach().float().cpu()
                        finite = x.isfinite() & y.isfinite()
                        error = (x[finite]-y[finite]).abs()
                        bound = tolerance['atol']+tolerance['rtol']*y[finite].abs()
                        row.update(max_finite_absolute_error=error.max().item() if error.numel() else 0.0,
                                   elements_outside_uncast_tolerance=int((error>bound).sum().item()),
                                   comparison='FP32 reference with one final output cast')
                    if baselines is None:
                        assert_close(got, ref, tolerance, output_dtype=value.dtype)
                    else:
                        row.update(_eager_error_metrics(value, want, baselines[index], tolerance))
                        if row['error_metric_regressions']:
                            raise AssertionError('error exceeds eager budget: ' + ', '.join(
                                f'{name}={row["candidate_error_metrics"][name]:.12g} > '
                                f'{row["eager_error_metrics"][name]:.12g}'
                                for name in row['error_metric_regressions']))
                else:
                    torch.testing.assert_close(value.detach().cpu(), want.detach().cpu(), rtol=0, atol=0)
                    if baselines is not None:
                        torch.testing.assert_close(value, baselines[index], rtol=0, atol=0)
            elif type(value) is not type(want) or value != want:
                raise AssertionError('non-tensor output changed')
            elif baselines is not None and (type(value) is not type(baselines[index]) or value != baselines[index]):
                raise AssertionError('non-tensor eager output changed')
            row['passed'] = True
        except AssertionError as error:
            row['error'] = str(error)
            failures.append(row['path']+': '+str(error))
        if records is not None:
            records.append(row)
    if failures:
        raise AssertionError('Complete output validation failed: '+'; '.join(failures))
