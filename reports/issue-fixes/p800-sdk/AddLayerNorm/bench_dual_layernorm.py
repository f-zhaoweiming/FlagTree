"""Paired complete-output benchmark. Invoked by the guarded P800 interface runner."""
import ctypes
import faulthandler
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from dual_layernorm import DualLayerNorm


def check_contract(kernel):
    # Test live storage and alias guards outside all performance windows.
    x = torch.zeros((2,2048),device='cuda',dtype=torch.float16)
    y = torch.ones_like(x)
    w = torch.ones_like(x[0])
    b = torch.zeros_like(w)
    out,r = torch.empty_like(x),torch.empty_like(x)
    original = [x,y,w,b,out,r]
    rejected = []
    for label,index,bad in [('shape',1,y[:1]),('dtype',2,w.float()),('alias_x',4,x),
                            ('alias_out',5,out),('strided',0,x.T.contiguous().T),
                            ('unaligned',0,torch.empty(4097,device=x.device,dtype=x.dtype)[1:].view_as(x))]:
        args = list(original); args[index]=bad
        try: kernel(*args)
        except ValueError: rejected.append(label)
        else: raise AssertionError('accepted '+label)
    with torch.cuda.stream(torch.cuda.Stream()):
        try: kernel(*original)
        except ValueError: rejected.append('other_stream')
        else: raise AssertionError('accepted alternate stream')
    for m in (1,33,385):
        a = torch.randn((m,2048),device='cpu').half().cuda()
        h = torch.randn_like(a,device='cpu').cuda()
        target,tmp = torch.empty_like(a),torch.empty_like(a)
        for shift in (0.,1.):
            b.fill_(shift)
            v,z=kernel(a,h,w,b,target,tmp)
            full=a.float()+h.float()
            ref=F.layer_norm(full,(2048,),w.float(),b.float(),1e-5)
            assert torch.equal(z,full.half()) and torch.allclose(v.float(),ref,rtol=.001,atol=.0005)
    return dict(rejected=rejected,tail_rows=[1,33,385],parameter_updates=True)


@torch.inference_mode()
def run(path, report, save, metrics, trace=False):
    report['method'] = ('Original same-device FP32 eager reference; three per-output errors <= FP16 eager. '
                        'Checked public candidate call, supplied FP16 output buffers. Two-step writes residual '
                        'with add.out and copies layer_norm to supplied output. Eager returns newly allocated '
                        'outputs. Warm single synchronized calls, balanced randomized order; compilation excluded.')
    report['sdk'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                     if 'xmlir' in (d.metadata['Name'] or '').lower()}
    report['binaries'] = {}
    phase = ctypes.CDLL(None).f06_phase if trace else None
    if trace:
        phase.argtypes = [ctypes.c_char_p]
    for n in (64, 1024, 2048):
        kernels = {}
        for rounded in (False, True):
            name = 'rounded' if rounded else 'wide'
            kernel = DualLayerNorm(path/f'{name}-{n}', n, rounded)
            kernels[name] = kernel
            report['binaries'][f'{name}-{n}'] = kernel.binary_info
        if n == 2048 and not trace:
            report['guards']=check_contract(kernels['wide'])
        for m in {64: (16,), 1024: (64,577), 2048: (2048,)}[n]:
            for pattern in ('random','high_bias'):
                tag = f'{m}x{n}-{pattern}'
                def step(label):
                    report['stage'] = f'{tag}/{label}'
                    save()
                    faulthandler.dump_traceback_later(60, exit=True)
                step('create_inputs')
                gen = torch.Generator().manual_seed(20261126+m+n)
                def original(*shape):
                    return torch.randn(shape, generator=gen)
                x0, y0 = original(m,n), original(m,n)
                if pattern == 'high_bias':
                    x0 += 64
                    y0 += 16
                originals = [x0, y0, original(n)*.05+1, original(n)*.05]
                hashes = [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in originals]
                x0,y0,w0,b0 = (t.cuda() for t in originals)
                x,y,w,b = (t.half() for t in (x0,y0,w0,b0))
                step('reference')
                z0 = x0+y0
                refs = (F.layer_norm(z0,(n,),w0,b0,1e-5), z0)
                guard = 32
                storage = [torch.full((m*n+2*guard,), 19., dtype=x.dtype).to(x.device) for _ in range(2)]
                out,res = (t[guard:-guard].view(m,n) for t in storage)
                def eager():
                    r = x+y
                    return F.layer_norm(r,(n,),w,b,1e-5), r
                def two_step():
                    torch.add(x,y,out=res)
                    out.copy_(F.layer_norm(res,(n,),w,b,1e-5))
                    return out,res
                calls = {'eager':eager, 'two_step':two_step}
                for name, kernel in kernels.items():
                    calls[name] = lambda kernel=kernel: kernel(x,y,w,b,out,res)
                baseline = eager()
                step('reference_metrics')
                baseline_metrics = [metrics(v,ref) for v,ref in zip(baseline,refs)]
                baseline_cpu = [v.cpu() for v in baseline]
                case = dict(shape=[m,n], pattern=pattern, fp32_input_sha256=hashes, routes={})
                for name,fn in calls.items():
                    step(name+'-initialize')
                    out.fill_(float('nan')); res.fill_(float('nan'))
                    if trace:
                        directory = path / (tag+'-'+name)
                        directory.mkdir()
                        os.environ['F06_MACHINE_DIR'] = str(directory)
                        torch.cuda.synchronize()
                        phase((tag+'-'+name).encode())
                    try:
                        step(name+'-launch')
                        pair = fn()
                        step(name+'-sync')
                        torch.cuda.synchronize()
                    finally:
                        if trace: phase(b'')
                    checks = {}
                    step(name+'-metrics')
                    for idx, field in enumerate(('normalized','residual')):
                        cur = metrics(pair[idx],refs[idx])
                        base = baseline_metrics[idx]
                        checks[field] = dict(actual=cur, fp16_eager=base,
                            differs_from_eager=int((pair[idx].cpu()!=baseline_cpu[idx]).sum()),
                            passed=cur['nonfinite']==base['nonfinite']==0 and
                                   all(cur[k]<=base[k] for k in ('rmse','max_abs','max_normalized')))
                    assert all(bool((s[:guard].cpu()==19).all()) and bool((s[-guard:].cpu()==19).all())
                               for s in storage)
                    case['routes'][name] = dict(outputs=checks, passed=all(v['passed'] for v in checks.values()))
                    if name in ('eager','two_step'):
                        assert all(v['differs_from_eager']==0 and v['passed'] for v in checks.values())
                if not trace:
                    step('timing')
                    for _ in range(6):
                        for fn in calls.values(): fn()
                    torch.cuda.synchronize()
                    samples = {k:[] for k in calls}
                    rng = random.Random(20261126+m+n)
                    for block in range(10):
                        order = list(calls)
                        rng.shuffle(order)
                        for shift in range(len(order)):
                            for name in order[shift:]+order[:shift]:
                                start = time.perf_counter_ns()
                                pair = calls[name]()
                                torch.cuda.synchronize()
                                elapsed = (time.perf_counter_ns()-start)/1000
                                del pair
                                samples[name].append(elapsed)
                    for name, values in samples.items():
                        t = statistics.median(values)
                        case['routes'][name].update(wall_us=values, median_us=t,
                            speedup_vs_eager=statistics.median(samples['eager'])/t,
                            speedup_vs_two_step=statistics.median(samples['two_step'])/t)
                report['cases'].append(case)
                save()
                faulthandler.cancel_dump_traceback_later()
