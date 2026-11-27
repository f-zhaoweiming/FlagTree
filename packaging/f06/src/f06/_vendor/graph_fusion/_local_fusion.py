"""Independent local-fusion candidates with complete-model admission.

Only reviewed, stateless QuickGELU leaves are eligible. The actual FX pattern
is verified; no model names, layer indices or calibrated shapes are used.
Surrounding operators execute through the model's original vendor calls.
"""
import copy
import hashlib
import operator
from pathlib import Path

import torch


def _quickgelu_types():
    try:
        from transformers.activations import QuickGELUActivation
    except ImportError:
        return ()
    return (QuickGELUActivation,)


def _p800_model(model):
    from ._p800_inductor_gelu import _on_p800
    return any(x.device.type == 'cuda' and _on_p800(x.device)
               for x in (*model.parameters(), *model.buffers()))


def _quickgelu_graph(module):
    if (module.training or module._parameters or module._buffers or module._modules
            or 'forward' in vars(module)
            or any(getattr(module, key) for key in
                   ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'))):
        return None
    graph = torch.fx.symbolic_trace(copy.deepcopy(module))
    nodes = list(graph.graph.nodes)
    if len(nodes) != 5:
        return None
    x, scale, sigmoid, product, output = nodes
    def mul(node):
        return node.op == 'call_function' and node.target in (operator.mul, torch.mul) and not node.kwargs
    if (x.op != 'placeholder' or not mul(scale) or len(scale.args) != 2
            or not any(scale.args[i] is x and type(scale.args[1-i]) in (int, float)
                       and scale.args[1-i] == 1.702 for i in (0, 1))
            or sigmoid.op != 'call_function' or sigmoid.target is not torch.sigmoid
            or sigmoid.args != (scale,) or sigmoid.kwargs or not mul(product)
            or product.args not in ((x, sigmoid), (sigmoid, x))
            or output.op != 'output' or output.args != (product,)):
        return None
    return graph


def _signature(x):
    return (tuple(x.shape), tuple(x.stride()), x.dtype, x.device,
            x.requires_grad, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
            x.is_conj(), x.is_neg())


def _supported_tensor(x):
    # Misses retain ordinary PyTorch semantics, including dispatch modes and
    # training. No cached tensor values, output buffers or stream are reused.
    return (type(x) is torch.Tensor and x.device.type == 'cuda'
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.layout == torch.strided and x.is_contiguous() and x.numel() > 0
            and x.storage_offset() == 0 and x.data_ptr() % 64 == 0
            and not x.is_conj() and not x.is_neg()
            and not x.requires_grad and not torch.is_grad_enabled() and _ambient_supported())


def _ambient_supported():
    return (not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu')
            and not torch._C._is_torch_function_mode_enabled()
            and not torch._C._len_torch_dispatch_stack()
            and not torch._C._are_functorch_transforms_active()
            and not torch._C._get_tracing_state()
            and not torch.fx._symbolic_trace.is_fx_symbolic_tracing())


def _matches(x, signature):
    # A signature was admitted by _supported_tensor during calibration. Its
    # equality already proves nonempty contiguous shape/stride, allowed dtype,
    # device and disabled grad. Avoid testing those same facts twice on a hit.
    return (type(x) is torch.Tensor and x.layout == torch.strided
            and _signature(x) == signature and x.storage_offset() == 0
            and x.data_ptr() % 64 == 0 and _ambient_supported())


def _compile_region(graph, example, buffer_bytes, record):
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner
    from ._p800_model_schedule import p800_lm_schedule
    from ._resource_codegen import _bound_module, _bound_kernel_evidence
    captured = []
    def inner(*args, **kwargs):
        result = compile_fx_inner(*args, **kwargs)
        captured.append(result)
        return result
    def backend(gm, inputs):
        with p800_lm_schedule(buffer_bytes=buffer_bytes):
            return compile_fx(gm, inputs, inner_compile=inner, config_patches={
                'triton.cudagraphs': False, 'compile_threads': 1,
                'max_autotune': False, 'triton.autotune_pointwise': False,
                'emulate_precision_casts': True, 'pattern_matcher': False,
                'coordinate_descent_tuning': False, 'shape_padding': False})
    with config.patch({'emulate_precision_casts': True}):
        compiled = torch.compile(graph, backend=backend, fullgraph=True, dynamic=False)
        expected = compiled(example)
    if len(captured) != 1:
        raise RuntimeError('Local fusion must compile exactly one inference graph')
    module = _bound_module(captured[0])
    kernels = [(k, v) for k, v in vars(module).items() if k.startswith('triton_') and hasattr(v, 'run')]
    if len(kernels) != 1:
        raise RuntimeError('Local QuickGELU must bind exactly one generated kernel')
    # This restricted graph has one unaliased input, one fresh tensor output,
    # no parameters, mutations, saved tensors or backward. Calling its exact
    # generated wrapper preserves allocation/stream/launch semantics (N09).
    def call(x):
        return module.call([x])[0]
    observed = call(example)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0, equal_nan=True)
    if observed.data_ptr() in (example.data_ptr(), expected.data_ptr()):
        raise RuntimeError('Local fusion output must be newly allocated')
    record.update(state='compiled', signature=repr(_signature(example)),
        buffer_bytes=buffer_bytes, graph_code=graph.code, wrapper_path=module.__file__,
        wrapper_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        binding=kernels[0][0], call_boundary='guarded_generated_wrapper',
        precision='preserve_operator_boundaries', kernel_count=1,
        buffer_choice=getattr(kernels[0][1], 'p800_buffer_choice', None),
        binary_evidence=_bound_kernel_evidence(captured[0], 'cluster'))
    return call


class _LocalPipeline:
    def __init__(self, model, prepare, region_paths, graphs, buffer_bytes, record):
        tensors = (*model.parameters(), *model.buffers())
        self.model = copy.deepcopy(model, {id(x): x for x in tensors})
        self.prepare = prepare
        self.record = record
        self.prepared = False
        self._f06_scope = 'complete_model_with_local_fusion'
        # Identical graph/signature regions share compiled code, never data.
        self.bindings = {}
        self._f06_audit = [record]
        seen = set()
        for path in region_paths:
            module = self.model.get_submodule(path)
            if id(module) in seen:
                continue
            seen.add(id(module))
            module.forward = self._region_forward(module.forward, graphs[path], buffer_bytes, record)

    def _region_forward(self, original, graph, buffer_bytes, record):
        graph_code = graph.code
        last = None
        def forward(x):
            nonlocal last
            # Read the immutable pair once for concurrent calls. The shared
            # signature dictionary is only needed on cold misses.
            hit = last
            if hit is not None and _matches(x, hit[0]):
                return hit[1](x)
            if not _supported_tensor(x):
                return original(x)
            signature = _signature(x)
            key = (graph_code, signature)
            bound = self.bindings.get(key)
            if bound is None:
                # A new signature after calibration keeps the native op.
                if self.prepared:
                    return original(x)
                proof = {}
                bound = _compile_region(graph, x, buffer_bytes, proof)
                self.bindings[key] = bound
                record['compiled_regions'].append(proof)
            last = (signature, bound)
            return bound(x)
        return forward

    def __call__(self, *args, **kwargs):
        prepared = self.prepare(*args, **kwargs)
        if (not isinstance(prepared, tuple) or len(prepared) != 2
                or not isinstance(prepared[0], tuple) or not isinstance(prepared[1], dict)):
            raise TypeError('prepare must return (positional_args_tuple, keyword_args_dict)')
        return self.model(*prepared[0], **prepared[1])

    def _f06_prepare(self, validation_inputs):
        for args, kwargs in validation_inputs:
            output = self(*args, **kwargs)
            del output
        self.prepared = True
        if not self.bindings:
            raise RuntimeError('No eligible local QuickGELU signature executed')


def local_fusion_candidate_routes(model, *, prepare, validation_inputs, record, buffer_sizes=(512, 1024)):
    """Propose private inference models; the selector validates full outputs."""
    from ._whole_model_selection import ModelRoute
    # Avoid optional imports or device initialization for unrelated models.
    leaves = [(p, m) for p, m in model.named_modules(remove_duplicate=False)
              if type(m).__name__ == 'QuickGELUActivation']
    if not leaves or not _p800_model(model):
        record.update(state='no_supported_regions', implementations={})
        return {}
    types = _quickgelu_types()
    graphs = {}
    for path, module in leaves:
        if type(module) not in types:
            continue
        graph = _quickgelu_graph(module)
        if graph is not None:
            graphs[path] = graph
    record.update(state='proposed' if graphs else 'no_supported_regions',
        region_paths=list(graphs), implementation='Inductor generated pure QuickGELU',
        outside_regions='original model operators', implementations={})
    routes = {}
    for size in buffer_sizes if graphs else ():
        if size not in (512, 1024):
            raise ValueError('Unsupported local QuickGELU buffer size')
        name = 'local_quickgelu_lm'+str(size)
        audit = dict(route='inductor_local_quickgelu', region_paths=list(graphs),
            buffer_bytes=size, compiled_regions=[], capture_source='original_dtype',
            complete_output_validation_required=True)
        call = _LocalPipeline(model, prepare, list(graphs), graphs, size, audit)
        call._f06_prepare(validation_inputs)
        routes[name] = ModelRoute(call)
        record['implementations'][name] = audit
    return routes
