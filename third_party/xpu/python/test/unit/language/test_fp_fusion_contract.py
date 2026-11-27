"""Disabling contraction preserves two roundings; explicit FMA is independent."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _contract(a,b,c,out,N:tl.constexpr,EXPLICIT:tl.constexpr):
    i=tl.program_id(0)*4096+tl.arange(0,4096)
    x=tl.load(a+i,i<N,0);y=tl.load(b+i,i<N,0);z=tl.load(c+i,i<N,0)
    if EXPLICIT:value=tl.fma(x,y,z)
    else:value=x*y+z
    tl.store(out+i,value,i<N)


@pytest.mark.parametrize('explicit,enabled',[(False,False),(False,True),(True,False),(True,True)])
def test_contraction_respects_requested_rounding(explicit,enabled):
    a=torch.full((4096,),1+2**-23,device='cuda')
    b=torch.full_like(a,1-2**-23);c=torch.full_like(a,-1);out=torch.empty_like(a)
    _contract[(1,)](a,b,c,out,a.numel(),explicit,num_warps=8,enable_fp_fusion=enabled)
    actual=out.cpu()
    if explicit:assert bool((actual==-2**-46).all())
    elif not enabled:assert bool((actual==0).all())
    else:
        # Contraction is permitted, not required, for an ordinary expression.
        assert bool(((actual==0)|(actual==-2**-46)).all())


@triton.jit
def _tree(a,out,ROWS:tl.constexpr):
    row=tl.arange(0,ROWS)[:,None]
    col=tl.arange(0,16)[None,:]
    value=tl.load(a+row*16+col)
    total=tl.sum(value,1)
    tl.store(out+tl.arange(0,ROWS),total)


def test_requested_scalar_tree_preserves_8_4_2_1_order():
    # Cancellation deliberately distinguishes this tree from a serial fold.
    x=torch.tensor([1e8,1.,-1e8,3.,2.,4.,7.,-8.,5.,-2.,9.,-4.,1.,2.,3.,4.]).repeat(64,1)
    reference=x.clone()
    for step in (8,4,2,1):
        reference[:,:16-step]=reference[:,:16-step]+reference[:,step:]
    expected=reference[:,0].clone()
    a=x.cuda();out=torch.empty(64,device='cuda')
    _tree[(1,)](a,out,64,num_warps=8,xpu_reduction_tree=True,
                isCloseUnrollControl=True,isCloseVectorization=True,enable_fp_fusion=False)
    assert torch.equal(out.cpu(),expected)


@pytest.mark.parametrize('n',[768,1024])
@pytest.mark.parametrize('high_bias',[False,True])
def test_cache_only_change_preserves_fixed_row_algorithm(n,high_bias):
    from test_row_schedule import normalized_residual
    rows=65;count=128*n
    a=((torch.arange(count)%61-30)/16+(64 if high_bias else 0)).half().cuda()
    b=((torch.arange(count)%17-8)/32).half().cuda()
    w=((torch.arange(n)%13+3)/16).half().cuda()
    shift=((torch.arange(n)%7-3)/16).half().cuda()
    outputs=[];layouts=[]
    for reuse in (False,True):
        residual=torch.full((count+128,),-73,dtype=torch.float16,device='cuda')
        out=torch.full_like(residual,-73)
        kernel=normalized_residual[(2,)](a,b,w,shift,residual[64:],out[64:],rows,n,
            triton.next_power_of_2(n),True,num_warps=8,num_stages=1,
            # 2048 is an upper bound: stack-budget retuning chose 128 for the
            # uncached kernel but 256 for the cached one. Pin a jointly legal
            # size and verify the resulting geometry, not just the request.
            buffer_size_limit=128,xpu_row_shared=True,
            xpu_reduction_reuse=reuse,xpu_reduction_half_cache=reuse,enable_fp_fusion=False)
        assert kernel.metadata.buffer_size_limit==128
        layouts.append(kernel.asm['ttxir'].splitlines()[0])
        if reuse and n==1024:assert "xpu.fast_producer_reuse" in kernel.asm["ttxir"]
        r,y=residual.cpu(),out.cpu()
        assert bool((y[:64]==-73).all() and (y[64+rows*n:]==-73).all())
        assert torch.equal(r[64:64+rows*n],(a[:rows*n].float()+b[:rows*n].float()).half().cpu())
        outputs.append((r,y))
    assert layouts[0]==layouts[1]
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
