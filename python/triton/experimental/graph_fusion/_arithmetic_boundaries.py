"""Keep native arithmetic boundaries when local source bindings change dtype.

These contracts constrain candidate construction; they are not numerical
acceptance tests or evidence of physical multiplier/accumulator precision.
"""
import time
import types

import torch
from torch import fx


class ArithmeticBoundaryError(ValueError):
    pass


def _description(value, *, keep_dtype):
    if isinstance(value, fx.Node):
        if 'example_value' not in value.meta and 'val' not in value.meta:
            raise ArithmeticBoundaryError('Missing value metadata: ' + value.name)
        return _description(value.meta.get('example_value', value.meta.get('val')),
                            keep_dtype=keep_dtype)
    if isinstance(value, torch.Tensor):
        return ('tensor', str(value.dtype) if keep_dtype else 'floating' if value.is_floating_point() else str(value.dtype),
                tuple(map(str, value.shape)), tuple(map(str, value.stride())),
                str(value.device), str(value.layout))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_description(v, keep_dtype=keep_dtype) for v in value))
    if isinstance(value, dict):
        return ('dict', tuple((str(k), _description(v, keep_dtype=keep_dtype)) for k, v in sorted(value.items())))
    return (type(value).__name__, str(value))


def arithmetic_contract(gm, *, reviewed_sdk):
    from ._arithmetic_search import graph_arithmetic
    nodes = {node.name: node for node in gm.graph.nodes}
    result = []
    for row in graph_arithmetic(gm, reviewed_sdk=reviewed_sdk):
        if row['arithmetic'] in ('none', 'conversion'):
            continue
        node = nodes[row['node']]
        result.append(dict(operator=(node.op, str(node.target)), arithmetic=row['arithmetic'],
            evidence=row['evidence'], node=node.name,
            native=_description((node.args, node.kwargs, node), keep_dtype=True),
            geometry=_description((node.args, node.kwargs, node), keep_dtype=False)))
    return result


def check_arithmetic_boundaries(gm, contracts, *, reviewed_sdk):
    """Match an observed graph; unknown/low arithmetic must retain its ABI.

    Check the original classification, not the candidate's classification: a
    half ReLU can appear to be verified FP32 after an unauthorized widening.
    The full operation sequence also rejects newly introduced dtype branches.
    """
    actual = arithmetic_contract(gm, reviewed_sdk=reviewed_sdk)
    differences = []
    for index, original in enumerate(contracts):
        if len(original) != len(actual):
            differences.append('compute operation count changed')
            continue
        mismatch = None
        for before, after in zip(original, actual):
            if before['operator'] != after['operator']:
                mismatch = 'compute operation changed: ' + before['node']
            elif before['arithmetic'] == 'fp32':
                if after['arithmetic'] != 'fp32' or before['geometry'] != after['geometry']:
                    mismatch = 'verified FP32 contract changed: ' + before['node']
            elif before['native'] != after['native']:
                mismatch = 'native arithmetic boundary changed: ' + before['node']
            if mismatch:
                break
        if mismatch is None:
            return dict(reference_graph=index,
                protected_nodes=sum(row['arithmetic'] != 'fp32' for row in original))
        differences.append(mismatch)
    raise ArithmeticBoundaryError('; '.join(dict.fromkeys(differences)) or 'No original arithmetic contract')


def verify_candidate_boundaries(model, *, prepare, validation_inputs, contracts, reviewed_sdk, record):
    """Cold check before executing a changed compute graph; no warm wrapper."""
    started = time.perf_counter()
    checks, errors = [], []
    completed = False

    def backend(gm, inputs):
        try:
            checks.append(check_arithmetic_boundaries(gm, contracts, reviewed_sdk=reviewed_sdk))
        except ArithmeticBoundaryError as exc:
            errors.append(str(exc))
            raise
        return gm.forward

    try:
        # M58: independent cold candidates must not consume one shared
        # forward's Dynamo recompile budget. Retain all normal guards.
        class CandidateCapture(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.model = original

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        forward = CandidateCapture.forward
        CandidateCapture.forward = types.FunctionType(
            forward.__code__.replace(), forward.__globals__, forward.__name__,
            forward.__defaults__, forward.__closure__)
        compiled = torch.compile(CandidateCapture(model).eval(), backend=backend, fullgraph=True, dynamic=False)
        for args, kwargs in validation_inputs:
            positional, keywords = prepare(*args, **kwargs)
            output = compiled(*positional, **keywords)
            del output
        if not checks:
            raise ArithmeticBoundaryError('Candidate did not produce an arithmetic graph')
        completed = True
    except Exception:
        if errors:
            raise ArithmeticBoundaryError(errors[-1]) from None
        raise
    finally:
        record.append(dict(passed=completed and bool(checks) and not errors, checks=checks, errors=errors,
                           seconds=time.perf_counter() - started))
