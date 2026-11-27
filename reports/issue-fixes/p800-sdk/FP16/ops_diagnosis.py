"""Compare public operator implementations on identical inputs without changing the wheel."""
import ctypes as C
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F
import kunlun_ops as ops


def run(root, report, save, metrics, check):
    sdk = Path(sys.prefix) / 'lib/python3.10/site-packages/torch_xmlir'
    code = root / 'gelu.cc'
    code.write_text('''
#include <xpu/xdnn.h>
namespace api = baidu::xpu::api;
extern "C" void *new_ctx() {
    auto *ctx = api::create_context();
    if (ctx) ctx->set_stream(nullptr, false);
    return ctx;
}
extern "C" int run_gelu(api::Context *ctx, const float16 *x, float16 *out, int64_t n, bool approximate) {
    return api::gelu<float16>(ctx, x, out, n, approximate, nullptr, nullptr);
}
extern "C" void free_ctx(api::Context *ctx) { api::destroy_context(ctx); }
''')
    cmd = ['c++','-shared','-O2','-fPIC','-std=c++17',str(code),'-o',str(root/'gelu.so'),
           '-I'+str(sdk/'xhpc/xdnn/include'),'-I'+str(sdk/'include'),'-I'+str(sdk/'xre/include'),
           '-I/usr/local/xpu/include','-L'+str(sdk),'-lxpuapi',
           '-L'+str(Path(sys.prefix)/'xcudart/lib'),'-l:libxpurt.so.2','-Wl,-rpath,'+str(sdk),'-Wl,-z,defs']
    build = subprocess.run(cmd,capture_output=True,text=True)
    lib = None
    if build.returncode:
        report['bridge_error'] = build.stderr[-1400:]
    else:
        lib = C.CDLL(str(root/'gelu.so'))
        lib.new_ctx.restype = C.c_void_p
        lib.run_gelu.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_bool]
        lib.free_ctx.argtypes = [C.c_void_p]
        ctx = lib.new_ctx()
        assert ctx
    def normalized(x,w,b):
        return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)
    for m,n in [(16,64),(64,1024),(577,1024),(2048,2048)]:
        for pattern in ['random','high_bias']:
            gen = torch.Generator().manual_seed(20261126+m+n)
            def original(*shape):
                return torch.randn(shape,generator=gen)
            x, y = original(m,n), original(m,n)
            if pattern == 'high_bias':
                x += 64; y += 16
            x,y=x.cuda(),y.cuda()
            w,b=(original(n)*.05+1).cuda(),(original(n)*.05).cuda()
            a,h,wh,bh=(t.half() for t in (x,y,w,b))
            wf,bf=wh.float(),bh.float()
            out,res=torch.empty_like(a),torch.empty_like(a)
            if pattern=='random':
                exact=F.gelu(a);tanh=F.gelu(a,approximate='tanh')
                ref=F.gelu(x)
                checks={'public':lambda:ops.gelu(a,out),'aten_out':lambda:torch.ops.aten.gelu.out(a,approximate='none',out=out)}
                def public_fp32():
                    tmp = torch.empty_like(x)
                    check(ops.gelu(a.float(),tmp))
                    out.copy_(tmp)
                checks['public_fp32'] = public_fp32
                if lib:
                    for mode in (False,True):
                        checks['xdnn_'+str(mode)]=lambda mode=mode:check(lib.run_gelu(ctx,a.data_ptr(),out.data_ptr(),a.numel(),mode))
                original_result=None
                for name,fn in checks.items():
                    row=dict(api='gelu',route=name,shape=[m,n])
                    try:
                        out.fill_(float('nan'));fn();torch.cuda.synchronize()
                        if name=='public':original_result=out.clone()
                        row.update(exact_diff=int((out!=exact).sum()),tanh_diff=int((out!=tanh).sum()),
                                   legacy_diff=int((out!=original_result).sum()),actual=metrics(out,ref),eager=metrics(exact,ref))
                    except Exception as exc:row['error']=repr(exc)[:700]
                    report['cases'].append(row);save()
            refsum=x+y; refout=normalized(refsum,w,b)
            eagerres=a+h; eagerout=normalized(eagerres,wh,bh)
            for name in ['all_half','wide_affine','fp32_fused','rounded_residual']:
                row=dict(api='add_layernorm',shape=[m,n],pattern=pattern,route=name)
                try:
                    out.fill_(float('nan'));res.fill_(float('nan'))
                    if name=='all_half':check(ops.add_layernorm(a,h,wh,out,1e-5,False,bh,res,None))
                    if name=='wide_affine':check(ops.add_layernorm(a,h,wf,out,1e-5,False,bf,res,None))
                    if name=='fp32_fused':
                        zo,ro=out.float(),res.float()
                        check(ops.add_layernorm(a.float(),h.float(),wf,zo,1e-5,False,bf,ro,None))
                        out.copy_(zo);res.copy_(ro)
                    if name=='rounded_residual':
                        torch.add(a,h,out=res);out.copy_(normalized(res,wh,bh))
                    torch.cuda.synchronize()
                    for field,v,e,refval in [('out',out,eagerout,refout),('residual',res,eagerres,refsum)]:
                        om,em=metrics(v,refval),metrics(e,refval)
                        row[field]=dict(actual=om,eager=em,diff=int((v!=e).sum()),
                            passed=not (om['nonfinite'] or em['nonfinite']) and all(om[k]<=em[k] for k in ('rmse','max_abs','max_normalized')))
                except Exception as exc:row['error']=repr(exc)[:700]
                report['cases'].append(row);save()
    if lib:
        import math
        bits=torch.arange(65536,dtype=torch.int32).short().view(torch.float16)
        host=bits[bits.isfinite()]
        a=host.cuda();out=torch.empty_like(a)
        ref=(0.5*host.double()*torch.erfc(-host.double()/math.sqrt(2))).half()
        exact=F.gelu(a).cpu();tanh=F.gelu(a,approximate='tanh').cpu()
        domain={}
        for name in ['public','xdnn_exact','xdnn_tanh','public_fp32','aten_out']:
            if name=='public':check(ops.gelu(a,out))
            elif name=='public_fp32':
                tmp=torch.empty_like(a,dtype=torch.float32)
                check(ops.gelu(a.float(),tmp));out.copy_(tmp)
            elif name=='aten_out':torch.ops.aten.gelu.out(a,approximate='none',out=out)
            else:check(lib.run_gelu(ctx,a.data_ptr(),out.data_ptr(),a.numel(),name=='xdnn_tanh'))
            v=out.cpu()
            domain[name]={'exact_eager_diff':int((v!=exact).sum()),'tanh_eager_diff':int((v!=tanh).sum()),
                          'cpu_correctly_rounded_diff':int((v!=ref).sum()),
                          'nonfinite':int((~v.isfinite()).sum()),'elements':v.numel()}
        report['finite_binary16_domain']=domain;save()
        lib.free_ctx(ctx)
