"""Cold precision-domain selection from reviewed P800 arithmetic contracts.

Storage dtype alone never authorizes promotion. The existing region contracts
cover add/mul/sigmoid/LayerNorm, not every operation supported by the SDK. An
unknown node keeps the graph out of the FP32-only domain. This is a search
decision, not an accuracy acceptance result or a promise of faster execution.
"""
import copy
import operator

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_leaves


def graph_arithmetic(gm, *, reviewed_sdk):
    from ._fp32_compute_regions import _kind
    from ._operator_precision import _minmax_call, _tensors
    records = []
    views = {'view', 'reshape', 'transpose', 'permute', 'flatten', 'unsqueeze',
             'squeeze', 'expand', 'contiguous', 'clone', 'detach', 't'}
    matrix = {F.linear, torch.mm, torch.bmm, torch.matmul, operator.matmul,
              torch.ops.aten.linear.default, torch.ops.aten.mm.default,
              torch.ops.aten.bmm.default, torch.ops.aten.matmul.default}
    for node in gm.graph.nodes:
        if not node.op.startswith('call_'):
            continue
        tensors = _tensors((node.args, node.kwargs, node))
        dtypes = {v.dtype for v in tensors if v.is_floating_point()}
        arithmetic, evidence = 'unknown', 'no_reviewed_arithmetic_contract'
        kind = _kind(node)
        minmax = _minmax_call(node) if reviewed_sdk else None
        if (node.op == 'call_method' and node.target in views or kind == 'view'
                or node.op == 'call_function' and node.target is operator.getitem):
            arithmetic, evidence = 'none', 'view_or_selection'
        elif kind == 'cast' or (node.op == 'call_function'
                and str(node.target) == 'flagtree_p800.final_cast.default'):
            arithmetic, evidence = 'conversion', 'explicit_dtype_conversion'
        elif (node.op == 'call_method' and node.target in ('size', 'stride', 'numel', 'dim')
              or node.op == 'call_function' and node.target is getattr and len(node.args) == 2
              and node.args[1] in ('shape', 'dtype', 'device', 'ndim', 'layout')):
            arithmetic, evidence = 'none', 'tensor_metadata'
        elif reviewed_sdk and kind == 'compute':
            arithmetic, evidence = 'fp32', 'reviewed_P800_region_contract_M06'
        elif minmax:
            target = minmax[0]
            # Scalar bounds hit the same comparison kernel with either bound
            # omitted. Its missing-bound sentinel depends on storage dtype.
            reviewed_comparison = target in (torch.ops.aten.relu.default, torch.ops.aten.clamp.default)
            if reviewed_comparison and dtypes == {torch.float32}:
                arithmetic, evidence = 'fp32', 'reviewed_native_float_minmax_M06'
            elif reviewed_comparison and dtypes == {torch.float16}:
                arithmetic, evidence = 'fp16', 'reviewed_native_half_minmax_M06'
            else:
                evidence = 'minmax_overload_or_dtype_unverified_M64'
        elif reviewed_sdk and dtypes == {torch.float16} and (
                node.op == 'call_function' and (node.target in matrix
                    or str(node.target) in ('_dynamo_workaround.linear', '_dynamo_workaround.linear.default'))
                or node.op == 'call_method' and node.target in ('mm', 'bmm', 'matmul')):
            # SDNN modes 5/37 identify the input/output configuration. They do
            # not establish multiplier or accumulator precision. Keep this
            # boundary until the actual arithmetic has independent evidence.
            arithmetic, evidence = 'unknown', 'matrix_io_mode_is_not_arithmetic_proof_M64'
        # A BF16 or FP64 tensor, a mutation or an unrecognized cast must not
        # inherit the FP16 contract. Even views can observe changed dtypes.
        if (dtypes - {torch.float16, torch.float32}
                or node.kwargs.get('inplace', False) or 'out' in node.kwargs
                or getattr(getattr(node.target, '_schema', None), 'is_mutable', False)):
            arithmetic, evidence = 'unknown', 'unsupported_dtype_or_effect'
        records.append(dict(node=node.name, operator=str(node.target),
                            storage_dtypes=sorted(map(str, dtypes)),
                            arithmetic=arithmetic, evidence=evidence))
    return records


def choose_domain(records):
    """Only proven arithmetic can authorize a whole-graph precision domain."""
    kinds = {row['arithmetic'] for row in records}
    if 'fp16' in kinds:
        return 'mixed'
    if 'unknown' in kinds or not records:
        return 'unresolved'
    return 'fp32_only'


def inspect_model_arithmetic(model, *, prepare, validation_inputs, record):
    """P800-only cold capture; other backends keep their existing entry."""
    args, kwargs = validation_inputs[0]
    prepared = prepare(*args, **kwargs)
    tensors = [x for x in tree_leaves(prepared) if isinstance(x, torch.Tensor)]
    if not any(x.device.type == 'cuda' for x in tensors):
        record.update(domain='not_p800', graphs=[])
        return []
    # Optional plugins must not become an import requirement on CUDA/CPU.
    try:
        from triton.backends.xpu.driver import XPUDriver
        from triton.runtime import driver
    except ImportError:
        record.update(domain='not_p800', graphs=[])
        return []
    if not isinstance(driver.active, XPUDriver):
        record.update(domain='not_p800', graphs=[])
        return []
    from ._p800_inductor_gelu import _on_p800
    if not all(x.device.type == 'cuda' and _on_p800(x.device) for x in tensors):
        record.update(domain='unresolved', reason='mixed_or_unverified_devices', graphs=[])
        return []
    from ._p800_arithmetic_runtime import reviewed_runtime
    sdk = reviewed_runtime()
    from ._model import enable_vendor_capture
    enable_vendor_capture()
    graphs, rows = [], []

    def capture(gm, inputs):
        graphs.append(copy.deepcopy(gm))
        rows.append(graph_arithmetic(gm, reviewed_sdk=sdk['matched']))
        return gm.forward

    compiled = torch.compile(model, backend=capture, fullgraph=True)
    for args, kwargs in validation_inputs:
        positional, keywords = prepare(*args, **kwargs)
        result = compiled(*positional, **keywords)
        del result
    del compiled
    all_rows = [row for graph in rows for row in graph]
    record.update(domain=choose_domain(all_rows), graphs=rows, runtime=sdk,
                  fp32_nodes=sum(r['arithmetic'] == 'fp32' for r in all_rows),
                  fp16_nodes=sum(r['arithmetic'] == 'fp16' for r in all_rows),
                  unknown_nodes=[r['node'] for r in all_rows if r['arithmetic'] == 'unknown'],
                  storage_dtype_is_not_arithmetic=True, full_output_validation_required=True)
    return graphs


def original_fp32_model(model, *, source_tensors, prepare, source_prepare, validation_inputs):
    """Bind unrounded sources without deriving a graph from a reference model."""
    from ._local_precision import _state
    state = _state(model)
    source_tensors = {} if source_tensors is None else source_tensors
    memo = {}
    for name, value in state.items():
        if not value.is_floating_point():
            memo[id(value)] = value
            continue
        source = source_tensors.get(name, value if value.dtype == torch.float32 else None)
        if (not isinstance(source, torch.Tensor) or source.dtype != torch.float32 or source.requires_grad
                or source.shape != value.shape or source.device != value.device):
            raise ValueError('FP32-only search requires frozen original FP32 state: ' + name)
        if not torch.equal(value, source.to(dtype=value.dtype)):
            raise ValueError('FP32 source does not reproduce the captured state: ' + name)
        if id(value) in memo:
            if not torch.equal(memo[id(value)], source):
                raise ValueError('Conflicting original sources for tied state: ' + name)
        else:
            memo[id(value)] = torch.nn.Parameter(source, requires_grad=False) if isinstance(value, torch.nn.Parameter) else source
    source_prepare = prepare if source_prepare is None else source_prepare
    for args, kwargs in validation_inputs:
        old, old_spec = tree_flatten(prepare(*args, **kwargs))
        new, new_spec = tree_flatten(source_prepare(*args, **kwargs))
        if old_spec != new_spec:
            raise ValueError('Original FP32 input structure differs')
        for low, source in zip(old, new):
            if isinstance(low, torch.Tensor):
                if (not isinstance(source, torch.Tensor) or source.shape != low.shape or source.device != low.device
                        or low.is_floating_point() and source.dtype != torch.float32
                        or not low.is_floating_point() and source.dtype != low.dtype
                        or not torch.equal(low, source.to(dtype=low.dtype))):
                    raise ValueError('FP32-only search requires original FP32 prepared inputs')
            elif type(source) is not type(low) or source != low:
                raise ValueError('Original FP32 non-tensor input differs')
    return copy.deepcopy(model, memo), source_prepare
