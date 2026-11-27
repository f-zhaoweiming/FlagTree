"""Build bounded mixed-precision candidates from an original FP32 graph.

The caller supplies audited module scopes. Original inputs and parameters stay
in the graph; only explicitly selected native compute edges are narrowed.
Complete-model FP32 validation, outside this pass, remains mandatory.
"""
import re
import torch
import torch.nn.functional as F
from torch import fx


def module_path(node):
    stack=list(node.meta.get('nn_module_stack',{}).values())
    if not stack:return ''
    # Dynamo uses ``fn`` for top-level Sequential/module wrappers and ``self``
    # for a directly captured forward. Both refer to the captured root.
    path=re.sub(r"^L\['(?:self|fn)'\]\.?",'',str(stack[-1][0]))
    path=re.sub(r"_modules\['([^']+)'\]",r'\1',path)
    return re.sub(r'\[(\d+)\]',r'.\1',path).strip('.')


def matches(path, prefixes):
    return any(('.'+prefix+'.') in ('.'+path+'.') for prefix in prefixes)


def _kind(node):
    if node.op!='call_function' or node.kwargs.get('out') is not None:return None
    if node.target in (F.linear,torch.ops.aten.linear.default) or str(node.target) in (
            '_dynamo_workaround.linear','_dynamo_workaround.linear.default'):
        return 'linear'
    if node.target in (F.scaled_dot_product_attention,torch.ops.aten.scaled_dot_product_attention.default):
        return 'attention'
    return None


def source_mixed_candidate(gm, example_inputs, *, fp16_modules, fp32_modules=(),
                           fp16_ops=('linear','attention')):
    from ._model_inductor import _value,refresh_graph_metadata
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    allowed=tuple(fp16_modules);protected=tuple(fp32_modules);ops=set(fp16_ops)
    if (not ops or ops-{'linear','attention'} or
            any(not isinstance(p,str) or not p or p.startswith('.') or p.endswith('.')
                for p in (*allowed,*protected))):
        raise ValueError('Explicit module scopes and reviewed compute kinds are required')
    # Output-only casts in the public dtype wrapper are allowed. Ingress values
    # must be original FP32; this API never restores an already rounded source.
    if any(isinstance(x,torch.Tensor) and x.is_floating_point() and x.dtype!=torch.float32
           for x in example_inputs):
        raise ValueError('Source mixed precision requires original FP32 inputs and parameters')
    graph=gm.graph;nodes=list(graph.nodes)
    placeholders=[n for n in nodes if n.op=='placeholder']
    if len(placeholders)!=len(example_inputs):raise ValueError('Example/placeholder count mismatch')
    positions={n:i for i,n in enumerate(placeholders)}
    examples=list(example_inputs);packed={};records=[]
    mode=detect_fake_mode(example_inputs) or FakeTensorMode(allow_non_fake_inputs=True)
    last_placeholder=placeholders[-1]
    def parameter_copy(node):
        nonlocal last_placeholder
        if node not in positions:return None
        index=positions[node];value=example_inputs[index]
        if (not isinstance(value,torch.nn.Parameter) or value.requires_grad or
                any(_kind(u)!='linear' or node not in u.args[1:3] for u in node.users)):
            return None
        if index not in packed:
            with graph.inserting_after(last_placeholder):
                new=graph.placeholder('source_half_parameter_'+str(index))
            with mode:
                fake=torch.empty_like(mode.from_tensor(value),dtype=torch.float16)
            new.meta['example_value']=fake
            examples.append(fake);packed[index]=new;last_placeholder=new
        return packed[index]
    for node in nodes:
        kind=_kind(node);path=module_path(node)
        if kind not in ops or not matches(path,allowed) or matches(path,protected):continue
        value=_value(node)
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32:continue
        original_args=node.args
        # Keywords/aliases that do not match the reviewed native ABI stay FP32.
        if kind=='linear':
            if node.kwargs or len(node.args) not in (2,3):continue
            x,w=node.args[:2];b=node.args[2] if len(node.args)==3 else None
            pw=parameter_copy(w)
            pb=parameter_copy(b) if b is not None else None
            if pw is None or (b is not None and pb is None):continue
            with graph.inserting_before(node):hx=graph.call_method('half',(x,))
            node.args=(hx,pw,pb)
        else:
            if len(node.args)<3:continue
            q,k,v=node.args[:3]
            if any(not isinstance(_value(x),torch.Tensor) or _value(x).dtype!=torch.float32 for x in (q,k,v)):continue
            if len(node.args)>4 or set(node.kwargs)-{'attn_mask','dropout_p','is_causal','scale','enable_gqa'}:continue
            mask=node.kwargs.get('attn_mask',node.args[3] if len(node.args)>3 else None)
            mv=_value(mask)
            if isinstance(mv,torch.Tensor) and mv.dtype not in (torch.bool,torch.float32):continue
            if node.kwargs.get('dropout_p',0.)!=0.:continue
            with graph.inserting_before(node):
                low=[graph.call_method('half',(x,)) for x in (q,k,v)]
            # A floating mask stays FP32: narrowing its sentinel can change
            # causal semantics. Native SDPA accepts an FP32 mask with half QKV.
            node.args=(*low,*node.args[3:])
        original_users=list(node.users)
        with graph.inserting_after(node):wide=graph.call_method('float',(node,))
        for user in original_users:user.replace_input_with(node,wide)
        records.append(dict(kind='source_fp32_native_half_'+kind,node=node.name,module=path,
            source='original_fp32',source_dtype='torch.float32',native_dtype='torch.float16',
            output_dtype='torch.float32',protected_scopes=list(protected),
            state='requires_complete_original_fp32_validation'))
    # A rejected candidate may have tentatively allocated an unused parameter.
    # Erase only these known new unused placeholders; preserve all graph effects.
    unused=[(i,n) for i,n in packed.items() if not n.users]
    for i,n in unused:graph.erase_node(n);del packed[i]
    examples=list(example_inputs)+[n.meta['example_value'] for n in packed.values()]
    # Retained FP32 vendor Linear needs the same opaque fake ABI as half Linear.
    # Its real execution still lowers directly to the vendor entry.
    from ._operator_precision import _linear, _lower_reviewed
    from torch._inductor.lowering import make_fallback
    target=_linear._opoverload
    make_fallback(target,warn=False);_lower_reviewed(target,_linear)
    for node in nodes:
        if _kind(node)=='linear' and not node.kwargs:
            node.target=target
            if len(node.args)==2:node.args=(*node.args,None)
    graph.lint();gm.recompile();refresh_graph_metadata(gm,examples)
    return gm,examples,tuple(packed),records


class SourceParameterPack:
    """Retain FP32 masters and versioned half copies, never cached activations."""
    def __init__(self,fn,input_count,indices):
        from ._model_inductor import _ParameterPack
        self.input_count=input_count;self.indices=tuple(indices)
        self.pack=_ParameterPack(fn,range(input_count,input_count+len(self.indices)),
                                 transform=lambda x:x.half())

    def __call__(self,*args):
        if len(args)!=self.input_count:raise ValueError('Source graph argument count changed')
        return self.pack(*args,*(args[i] for i in self.indices))
