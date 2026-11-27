"""An explicit preprocessing boundary for complete tensor-model compilation."""


class _CompiledPipeline:
    def __init__(self, prepare, compiled, source_precision=None):
        self.prepare = prepare
        self.compiled = compiled
        self._f06_audit = compiled._f06_audit
        self._f06_capture = compiled._f06_capture
        self._f06_scope = 'complete_tensor_forward'
        self._f06_source_precision = source_precision
        self._f06_resource_factory = None
        self._f06_resource_candidates = None
        self._f06_resource_allowlist = None
        self._f06_buffer_factory = None
        self._f06_buffer_candidates = None
        self._f06_gemm_factory = None
        self._f06_gemm_candidates = None

    def _f06_gemm_variants(self):
        if self._f06_gemm_factory is None:
            return {}
        if self._f06_gemm_candidates is None:
            self._f06_gemm_candidates = self._f06_gemm_factory()
        return self._f06_gemm_candidates

    def _f06_buffer_variants(self):
        if self._f06_buffer_factory is None:
            return {}
        if self._f06_buffer_candidates is None:
            self._f06_buffer_candidates = self._f06_buffer_factory()
        return self._f06_buffer_candidates

    def _f06_resource_variants(self):
        """Cold-only expansion, after the baseline graph has been prepared."""
        if self._f06_resource_factory is None:
            return {}
        if self._f06_resource_candidates is None:
            self._f06_resource_candidates = self._f06_resource_factory()
        if self._f06_resource_allowlist is not None:
            return {kind: call for kind, call in self._f06_resource_candidates.items()
                    if kind in self._f06_resource_allowlist}
        return self._f06_resource_candidates

    def __call__(self, *args, **kwargs):
        prepared = self.prepare(*args, **kwargs)
        if (not isinstance(prepared, tuple) or len(prepared) != 2
                or not isinstance(prepared[0], tuple) or not isinstance(prepared[1], dict)):
            raise TypeError('prepare must return (positional_args_tuple, keyword_args_dict)')
        if self._f06_source_precision == 'float32':
            import torch
            from torch.utils._pytree import tree_leaves
            if any(isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float32
                   for x in tree_leaves(prepared)):
                raise ValueError('FP32 source capture requires original FP32 prepared inputs; widening rounded inputs is not supported')
            if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
                raise ValueError('FP32 source capture requires autocast to be disabled')
        return self.compiled(*prepared[0], **prepared[1])


def compile_pipeline(tensor_model, *, prepare, output_template=None, source_precision=None, **kwargs):
    """Compile one complete tensor forward, keeping preprocessing in each call.

    ``prepare(*inputs, **kwargs)`` returns ``(args, kwargs)`` for an inference
    module that includes every tensor operation up to the public outputs,
    including pooling, projection, normalization and output ordering. Text and
    image preprocessing stays outside Dynamo and is never cached by this API.

    Graph breaks raise instead of silently compiling separate module entries.
    Guards can compile new complete graphs for new shapes. Vendor operators
    may remain opaque nodes in each graph; one graph need not be one kernel.

    With ``source_precision='float32'`` callers must supply the original FP32
    model and preprocessing, before any low-precision conversion. This API
    never widens a rounded checkpoint. ``output_template`` supplies only the
    public output structure and dtypes; final casts belong to the captured
    graph, after the complete model forward. Output sizes may change normally.

    With the default resource_policy='auto', select_model_routes expands a
    prepared P800 pipeline into independent Cluster and SDNN alternatives.
    They preserve the prepared arithmetic graph and require the caller's full
    original-FP32 output validator and complete-model latency measurement.
    Calling this pipeline alone retains its baseline resource schedule.
    """
    from ._model import compile_model
    if not callable(prepare):
        raise TypeError('prepare must be callable')
    if kwargs.pop('fullgraph', True) is not True:
        raise ValueError('A complete tensor pipeline requires fullgraph=True')
    if source_precision not in (None, 'float32'):
        raise ValueError('source_precision must be None or float32')
    if kwargs.get('precision_policy') == 'source_fp32_mixed' and source_precision != 'float32':
        raise ValueError('Source mixed precision requires source_precision=float32 and the original model')
    if source_precision == 'float32':
        import torch
        if any(x.is_floating_point() and x.dtype != torch.float32
               for x in (*tensor_model.parameters(), *tensor_model.buffers())):
            raise ValueError('FP32 source capture requires original FP32 parameters and buffers')
    if output_template is not None:
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten
        leaves, expected_spec = tree_flatten(output_template)
        contracts = tuple((x.dtype, x.ndim, x.layout, x.device.type) if isinstance(x, torch.Tensor)
                          else None for x in leaves)
        # M06: keep final narrowing separate from FP32 values still needed by
        # another public output. The current XPU mixed-output lowering fails if
        # Inductor merges these stores into the same pointwise kernel.
        vendor_final_cast = None
        if any(c is not None and c[3] == 'cuda' and c[0] in (torch.float16, torch.bfloat16)
               for c in contracts):
            from ._p800_inductor_gelu import _on_p800
            if any(isinstance(x, torch.Tensor) and x.device.type == 'cuda' and _on_p800(x.device)
                   for x in leaves):
                from ._p800_model_layernorm import _final_cast
                vendor_final_cast = _final_cast

        class FinalOutputCast(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                values, spec = tree_flatten(self.model(*args, **kwargs))
                if spec != expected_spec:
                    raise ValueError('Public output structure changed')
                result = []
                for value, contract in zip(values, contracts):
                    if contract is not None:
                        dtype, ndim, layout, device = contract
                        if (not isinstance(value, torch.Tensor) or value.ndim != ndim
                                or value.layout != layout or value.device.type != device):
                            raise ValueError('Public output tensor contract changed')
                        if value.is_floating_point():
                            if vendor_final_cast is not None and value.dtype != dtype:
                                value = vendor_final_cast(value, dtype)
                            else:
                                value = value.to(dtype=dtype)
                        elif value.dtype != dtype:
                            raise ValueError('Non-floating output dtype changed')
                    result.append(value)
                return tree_unflatten(result, spec)

        tensor_model = FinalOutputCast(tensor_model).eval()
    result = _CompiledPipeline(prepare, compile_model(tensor_model, fullgraph=True, **kwargs), source_precision)
    if kwargs.get('resource_policy','auto') == 'auto':
        def resource_variants():
            import torch
            from ._p800_inductor_gelu import _on_p800
            if not any(x.device.type == 'cuda' and _on_p800(x.device)
                       for x in (*tensor_model.parameters(), *tensor_model.buffers())):
                return {}
            # A precision-boundary rejection can leave the original vendor
            # graph as the baseline. It has no generated kernel to retarget;
            # do not relabel another rejected compile as a resource candidate.
            if not result._f06_audit or any(r.get('route') == 'original' for r in result._f06_audit):
                for record in result._f06_audit:
                    record['resource_enumeration'] = 'skipped_original_vendor_fallback'
                return {}
            options = dict(kwargs)
            options.pop('audit', None)
            options.pop('resource_policy', None)
            if options.get('precision_policy','fp32_reference') == 'fp32_reference':
                choices = {r.get('route') for r in result._f06_audit}
                # Mirror the prepared arithmetic graph. Resource alternatives
                # must not perform an inner tensor-only timing elimination.
                if len(choices) != 1 or None in choices or 'original' in choices:
                    return {}
                options['graph_choice'] = choices.pop()
            return {kind:compile_pipeline(tensor_model, prepare=prepare,
                        source_precision=source_precision, resource_policy=kind, **options)
                    for kind in ('cluster','sdnn')}
        result._f06_resource_factory = resource_variants
    if (kwargs.get('precision_policy') in ('preserve_operator', 'fuse_fp32_internals')
            and kwargs.get('resource_policy', 'auto') != 'sdnn'
            and 'pointwise_buffer_bytes' not in (kwargs.get('precision_options') or {})):
        def buffer_variants():
            # The historical field includes homogeneous FP32 kernels as well.
            # Smaller packets may enable dependency-safe multi-buffer DMA.
            if not any(r.get('fp16_buffer_kernels') for r in result._f06_audit):
                return {}
            options = dict(kwargs)
            options.pop('audit', None)
            variants = {}
            for size in (512, 1024):
                precision = dict(options.get('precision_options') or {}, pointwise_buffer_bytes=size)
                call = compile_pipeline(tensor_model, prepare=prepare, source_precision=source_precision,
                    **dict(options, precision_options=precision))
                # The resource dimension is expanded independently by the
                # parent. These are complete-model buffer candidates only.
                call._f06_resource_factory = None
                variants['lm'+str(size)] = call
            return variants
        result._f06_buffer_factory = buffer_variants
    precision = kwargs.get('precision_options') or {}
    if (kwargs.get('precision_policy') in ('preserve_operator', 'fuse_fp32_internals')
            and kwargs.get('resource_policy', 'auto') == 'auto'
            and precision.get('linear_schedule') in ('inductor', 'grouped')
            and precision.get('pointwise_policy', 'generated') == 'generated'
            and precision.get('matmul_policy', 'native') == 'native'):
        def gemm_variants():
            import torch
            from ._p800_inductor_gelu import _on_p800
            if not any(x.device.type == 'cuda' and _on_p800(x.device)
                       for x in (*tensor_model.parameters(), *tensor_model.buffers())):
                return {}
            if not any(any(x.get('input_dtype') == 'torch.float32' for x in r.get('exposed_linears', []))
                       for r in result._f06_audit):
                return {}
            variants = {}
            for policy in ('tuned', 'full_k'):
                options = dict(kwargs)
                options.pop('audit', None)
                options['precision_options'] = dict(precision, matmul_policy=policy)
                candidate = compile_pipeline(tensor_model, prepare=prepare,
                    source_precision=source_precision, **options)
                # Preserve the complete output gate and avoid a recursive
                # cross product of resource, packet and matrix alternatives.
                candidate._f06_resource_factory = None
                candidate._f06_buffer_factory = None
                candidate._f06_gemm_factory = None
                variants['gemm_'+policy] = candidate
            return variants
        result._f06_gemm_factory = gemm_variants
    return result
