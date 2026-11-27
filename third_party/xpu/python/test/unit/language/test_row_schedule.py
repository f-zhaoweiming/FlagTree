"""Empty rows and multi-output wait scheduling preserve every visible value."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def normalized_residual(a, b, w, bias, residual, out, rows,
                        N: tl.constexpr, R: tl.constexpr, DUAL: tl.constexpr):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)[:, None]
    col = tl.arange(0, R)[None, :]
    mask = (row < rows) & (col < N)
    x = tl.load(a + row * N + col, mask, other=0).to(tl.float32)
    y = tl.load(b + row * N + col, mask, other=0).to(tl.float32)
    value = (x + y).to(a.dtype.element_ty).to(tl.float32)
    mean = tl.sum(tl.where(mask, value, 0.), 1)[:, None] / N
    delta = value - mean
    var = tl.sum(tl.where(mask, delta * delta, 0.), 1)[:, None] / N
    weight = tl.load(w + col, col < N, other=0).to(tl.float32)
    shift = tl.load(bias + col, col < N, other=0).to(tl.float32)
    result = delta * tl.rsqrt(var + 1.e-5) * weight + shift
    if DUAL:
        tl.store(residual + row * N + col, value, mask)
    tl.store(out + row * N + col, result, mask)


@pytest.mark.parametrize('dtype,n,rows,high_bias', [
    (torch.float16,1024,0,False), (torch.float16,1024,1,False),
    (torch.float16,1024,63,False), (torch.float16,1024,64,False),
    (torch.float16,1024,65,False), (torch.float16,1024,65,True),
    (torch.float16,1009,65,False), (torch.float16,128,127,False),
    (torch.float32,1024,65,True), (torch.bfloat16,1024,65,True),
])
@pytest.mark.parametrize('dual',[False,True])
def test_row_schedule_full_outputs_and_sentinels(dtype,n,rows,high_bias,dual):
    count=128*n
    a=((torch.arange(count)%61-30)/16+(64 if high_bias else 0)).to(dtype).cuda()
    b=((torch.arange(count)%17-8)/32).to(dtype).cuda()
    w=((torch.arange(n)%13+3)/16).to(dtype).cuda()
    shift=((torch.arange(n)%7-3)/16).to(dtype).cuda()
    expected_residual=(a[:rows*n].float()+b[:rows*n].float()).to(dtype).cpu()
    outputs=[]
    for mode in (0,2):
        residual=torch.full((count+128,),-73,dtype=dtype,device='cuda')
        out=torch.full_like(residual,-73)
        normalized_residual[(2,)](a,b,w,shift,residual[64:],out[64:],rows,
            n,triton.next_power_of_2(n),dual,num_warps=8,num_stages=1,
            is_sdnn=False,launch_clusters=12,launch_cores=64,
            buffer_size_limit=2048,xpu_row_shared=True,
            xpu_reduction_reuse=True,xpu_reduction_half_cache=True,
            xpu_dma_pipeline=mode,enable_fp_fusion=False)
        torch.cuda.synchronize()
        r,o=residual.cpu(),out.cpu()
        assert bool(torch.isfinite(o).all())
        assert bool((o[:64]==-73).all() and (o[64+rows*n:]==-73).all())
        if dual:
            assert torch.equal(r[64:64+rows*n],expected_residual)
            assert bool((r[:64]==-73).all() and (r[64+rows*n:]==-73).all())
        else:
            assert bool((r==-73).all())
        outputs.append((r,o))
    assert all(torch.equal(x,y) for x,y in zip(outputs[0],outputs[1]))
