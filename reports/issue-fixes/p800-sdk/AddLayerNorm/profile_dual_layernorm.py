"""KL3 schedule ablations. Diagnostic images with partial math never enter public routing."""
import ctypes
import copy
import faulthandler
import hashlib
import importlib.util
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

COPY_LOOP = """
    const int core = core_id(), lane = core % group, col = lane * slice;
    __shared__ __simd__ uint16_t w[COLS], b[COLS];
    __simd__ uint16_t left[slice], right[slice];
    if (core == 0) GM2SM_ASYNC(gamma, w, sizeof(w));
    if (core == 1) GM2SM_ASYNC(beta, b, sizeof(b));
    mfence_sm_gm(); sync_cluster();
    const int rows_per_cluster = core_num() / group;
    for (int base=cluster_id()*rows_per_cluster; base<rows; base+=cluster_num()*rows_per_cluster) {
        const int row=base+core/group, offset=row*COLS+col;
        if (row < rows) {
            GM2LM_ASYNC(x+offset,left,sizeof(left)); GM2LM_ASYNC(y+offset,right,sizeof(right));
            mfence_lm_gm();
            LM2GM(left,residual+offset,sizeof(left));
        }
        EXTRA_SYNC
        if (row < rows) LM2GM(right,output+offset,sizeof(right));
    }
}
"""

REPEAT = """
int repeat_ln(int count, void *kernel, int clusters, void *x, void *y, void *w, void *b,
              void *out, void *res, int rows, float eps) {
    for (int i=0; i<count; ++i) {
        int rc=launch_dual_ln(kernel,clusters,x,y,w,b,out,res,rows,eps);
        if(rc) return rc;
    }
    return 0;
}
"""

def replace(text, needle, value):
    if text.count(needle) != 1:
        raise RuntimeError('Legacy source differs at '+needle[:60])
    return text.replace(needle,value)


def all_sources(base, current):
    head = base[:base.index('    #pragma clang fp contract(off)')]
    local = base.replace('if (group > 1)', 'if (false)')
    point = replace(local, 'float mean = total / COLS;', 'float mean = 0.f;')
    point = replace(point, 'float inv = 1.f / __builtin_sqrtf(ss / COLS + eps);', 'float inv = 1.f;')
    return {'empty':head+'    asm volatile("");\n}\n',
            'dma':head+COPY_LOOP.replace('EXTRA_SYNC',''),
            'dma_barriers':head+COPY_LOOP.replace('EXTRA_SYNC','mfence(); sync_cluster(); mfence(); sync_cluster();'),
            'point':point,'local_stats':local,'old':base,'new':current,'four_core':replace(current, 'COMBINED_MOMENTS ? 1024', 'COMBINED_MOMENTS ? 512'),
            'async_only':replace(base,'LM2GM(left, residual','LM2GM_ASYNC(left, residual')}


def build_variant(path,name,source):
    # Use the original checked wrapper for every device-control comparison.
    root=Path(__file__).parent
    folder=path/('control-'+name)
    folder.mkdir()
    (folder/'dual_layernorm.py').write_bytes((root/'baseline.py').read_bytes())
    (folder/'launch.c').write_text((root/'baseline-launch.c').read_text()+REPEAT)
    (folder/'dual_layernorm.xpu').write_text(source)
    spec=importlib.util.spec_from_file_location('control_'+name,folder/'dual_layernorm.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel=mod.DualLayerNorm(folder/'build',2048,False)
    repeat=ctypes.CDLL(str(folder/'build/launch.so')).repeat_ln
    repeat.argtypes=[ctypes.c_int]+kernel.launch.argtypes
    repeat.restype=ctypes.c_int
    return kernel,repeat


def median_fields(samples):
    return {key:statistics.median(v[key] for v in samples) for key in samples[0]}


@torch.inference_mode()
def run(path,report,save):
    import dual_layernorm
    root=Path(__file__).parent
    faulthandler.dump_traceback_later(90,exit=True)
    rows=cols=2048
    x,y=(torch.randn(rows,cols,device='cpu').half().cuda() for _ in range(2))
    w,b=(torch.randn(cols,device='cpu').half().cuda() for _ in range(2))
    out,res=torch.empty_like(x),torch.empty_like(x)
    args=(x,y,w,b,out,res)
    ptrs=tuple(t.data_ptr() for t in args)
    rr=x+y
    def eager_add(): return x+y
    def eager_norm(): return F.layer_norm(rr,(cols,),w,b,1e-5)
    def eager():
        r=x+y
        return F.layer_norm(r,(cols,),w,b,1e-5),r
    from torch.utils._python_dispatch import TorchDispatchMode
    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            value=func(*args,**(kwargs or {}))
            def meta(x):
                return {'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride())} if isinstance(x,torch.Tensor) else str(x)
            report['eager_dispatch'].append(dict(op=str(func),inputs=[meta(a) for a in args],
                        outputs=[meta(i) for i in (value if isinstance(value,tuple) else (value,))]))
            return value
    report['eager_dispatch']=[]
    with Capture(): eager()
    raw={}; repeats={}; compiled={}; old=None
    report['binaries']={}
    for name,source in all_sources((root/'baseline.xpu').read_text(),(root/'dual_layernorm.xpu').read_text()).items():
        report['stage']='build-'+name;save()
        k,rep=build_variant(path,name,source)
        if name=='old':old=k
        params=(ctypes.byref(k.descriptor),12,*ptrs,rows,1e-5)
        def once(k=k,params=params):
            if k.launch(*params):raise RuntimeError('launch')
        def batch(count,rep=rep,params=params):
            if rep(count,*params):raise RuntimeError('batch launch')
        raw[name]=once;repeats[name]=batch;compiled[name]=k
        report['binaries'][name]=dict(k.binary_info,source_sha256=hashlib.sha256(source.encode()).hexdigest())
    actual=dual_layernorm.DualLayerNorm(path/'public-new',cols,False)
    calls=dict(eager=eager,eager_add=eager_add,eager_norm=eager_norm,old_public=lambda:old(*args),
               new_public=lambda:actual(*args),old_raw=raw['old'],new_raw=raw['new'])
    raw['dma'](); torch.cuda.synchronize()
    assert torch.equal(out,y) and torch.equal(res,x)
    raw['dma_barriers'](); torch.cuda.synchronize()
    assert torch.equal(out,y) and torch.equal(res,x)
    report['stage']='wall';save()
    sync=torch.cuda.synchronize
    rng=random.Random(61126)
    for _ in range(6):
        for fun in calls.values(): fun();sync()
    data={k:[] for k in calls}
    for _ in range(70):
        order=list(calls);rng.shuffle(order)
        for name in order:
            sync()
            a=time.perf_counter_ns();result=calls[name]();btime=time.perf_counter_ns();sync();c=time.perf_counter_ns()
            data[name].append(dict(submit=(btime-a)/1000,after_return=(c-btime)/1000,total=(c-a)/1000))
            del result
    report['wall_median_us']={k:median_fields(v) for k,v in data.items()};report['wall_samples_us']=data
    # Cross the public guard and the actual device image at fixed geometry
    # (both kernels launch 12 clusters at this shape). Each copy keeps its
    # original descriptor owner alive until all synchronized calls finish.
    old_guard_new_image=copy.copy(old)
    old_guard_new_image.descriptor=type(old.descriptor).from_buffer_copy(actual.descriptor)
    new_guard_old_image=copy.copy(actual)
    capsule=ctypes.pythonapi.PyCapsule_New
    new_guard_old_image._capsule=capsule(ctypes.addressof(old.descriptor),b'sdk04.cluster',None)
    new_guard_old_image.rows_per_cluster=16
    crossed={'eager':eager,'old_guard_old_image':calls['old_public'],'new_guard_new_image':calls['new_public'],
             'old_guard_new_image':lambda:old_guard_new_image(*args),
             'new_guard_old_image':lambda:new_guard_old_image(*args)}
    for guard,kernel in [(new_guard_old_image,old),(old_guard_new_image,actual)]:
        guard(*args);sync()
        saved=(out.clone(),res.clone())
        kernel(*args);sync()
        assert torch.equal(saved[0],out) and torch.equal(saved[1],res)
    # Use the same no-extra-pre-sync and post-call synchronize as the shape benchmark.
    paired={k:[] for k in crossed}
    for fn in crossed.values():fn();sync()
    for _ in range(10):
        order=list(crossed);rng.shuffle(order)
        for offset in range(len(order)):
            for name in order[offset:]+order[:offset]:
                a=time.perf_counter_ns();value=crossed[name]();btime=time.perf_counter_ns()
                sync();c=time.perf_counter_ns()
                paired[name].append(dict(submit=(btime-a)/1000,after_return=(c-btime)/1000,total=(c-a)/1000))
                del value
    report['crossed_wall_median_us']={k:median_fields(v) for k,v in paired.items()}
    report['crossed_wall_samples_us']=paired
    report['stage']='events';save()
    # Repeated native launches from C exclude Python dispatch gaps. Device
    # intervals include command processor dispatch, as does the null image.
    batches={k:[] for k in repeats}
    invalid=[]
    for trial in range(18):
        order=list(repeats);rng.shuffle(order)
        for name in order:
            count=32
            for attempt in range(3):
                begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                sync();begin.record();repeats[name](count);end.record();end.synchronize()
                us=begin.elapsed_time(end)*1000/count
                if math.isfinite(us) and 0 < us < 1000:
                    batches[name].append(us);break
                invalid.append([trial,name,count,us if math.isfinite(us) else str(us)]);count*=2
            else:raise RuntimeError('repeated invalid XPU event')
    report['event_us']=batches;report['event_median_us']={k:statistics.median(v) for k,v in batches.items()}
    # Prefill the same default stream so Python has time to enqueue both events
    # and all eager kernels before the start event reaches the device.
    device_calls = dict(eager=eager, add=eager_add, norm=eager_norm, old=raw['old'], new=raw['new'])
    prefills={}
    for depth in (16,32):
        values={name:[] for name in device_calls}
        for trial in range(24):
            order=list(values);rng.shuffle(order)
            for name in order:
                for attempt in range(3):
                    begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                    repeats['old'](depth);begin.record();value=device_calls[name]();end.record();end.synchronize()
                    us=begin.elapsed_time(end)*1000
                    if math.isfinite(us) and 0<us<1000:
                        values[name].append(us);break
                    invalid.append(['prefill',depth,trial,name,str(us)])
                else: raise RuntimeError('invalid prefilled interval')
                del value
        prefills[str(depth)]=dict(samples=values,median_us={k:statistics.median(v) for k,v in values.items()})
    report['prefilled_single']=prefills
    report['invalid_events']=invalid
    # Validate complete optimized math before allowing performance conclusions.
    raw['new']();sync()
    a,z=out.clone(),res.clone()
    raw['old']();sync()
    report['new_vs_old']={'norm_max_abs':(a.float()-out.float()).abs().max().item(),
                          'residual_diff':int((z.cpu()!=res.cpu()).sum()),
                          'finite':bool(torch.isfinite(a.cpu()).all())}
    for name in ('four_core','async_only'):
        raw[name]();sync()
        saved=(out.clone(),res.clone())
        raw['old']();sync()
        report['new_vs_old'][name]=dict(norm_max_abs=(saved[0].float()-out.float()).abs().max().item(),
                                        residual_diff=int((saved[1].cpu()!=res.cpu()).sum()),
                                        finite=bool(torch.isfinite(saved[0].cpu()).all()))
    faulthandler.cancel_dump_traceback_later()
    save()
