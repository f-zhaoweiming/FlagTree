"""Vector/shared column loads preserve tails, row guards and dtype boundaries."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def staged_column_affine(a, w, b, out, rows, N: tl.constexpr, GAP: tl.constexpr):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)[:, None]
    lane = tl.arange(0, 32)[None, :]
    offset = (lane // 16) * 32 + lane % 16 if GAP else lane
    for block in range(N // 64):
        col = block * 64 + offset
        value = tl.load(a + row * N + col, row < rows, other=0).to(tl.float32)
        weight = tl.load(w + col).to(tl.float32)
        bias = tl.load(b + col).to(tl.float32)
        tl.store(out + row * N + col, value * weight + bias, row < rows)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('gap', [False, True])
def test_dense_local_parameter_staging_vectorizes_unknown_global_stride(dtype, gap):
    rows, n = 65, 128
    a = ((torch.arange(rows*n) % 61 - 30) / 8).to(dtype).reshape(rows, n)
    w = ((torch.arange(n) % 9 - 4) / 8).to(dtype)
    b = ((torch.arange(n) % 13 - 6) / 8).to(dtype)
    storage = torch.full((rows*n + 128,), -73, dtype=dtype, device='cuda')
    k = staged_column_affine[(2,)](a.cuda(), w.cuda(), b.cuda(), storage[64:], rows,
        n, gap, num_warps=8, num_stages=1, is_sdnn=False,
        buffer_size_limit=2048, enable_fp_fusion=False, isCloseUnrollControl=True)
    columns = torch.arange(n) % 64
    valid = columns % 32 < 16 if gap else columns < 32
    expected = torch.full((rows, n), -73, dtype=dtype)
    expected[:, valid] = (a[:, valid].float()*w[valid].float()+b[valid].float()).to(dtype)
    actual = storage.cpu()
    assert torch.equal(actual[64:-64].reshape(rows, n), expected)
    assert bool((actual[:64] == -73).all() and (actual[-64:] == -73).all())
    assert 'lm_vector_broadcast' in k.asm['ttxir']


@triton.jit
def row_affine(a, gamma, beta, output, rows, stride: tl.constexpr,
               N: tl.constexpr, X: tl.constexpr, R: tl.constexpr,
               HOLES: tl.constexpr):
    row = tl.program_id(0) * X + tl.arange(0, X)[:, None]
    col = tl.arange(0, R)[None, :]
    mask = (row < rows) & (col < N)
    if HOLES:
        mask = mask & (col % 7 != 3)
    value = tl.load(a + row * stride + col, mask, other=0).to(tl.float32)
    weight = tl.load(gamma + col, col < N, other=0).to(tl.float32)
    bias = tl.load(beta + col, col < N, other=0).to(tl.float32)
    result = value * weight + bias
    tl.store(output + row * N + col, result, mask)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize('n,stride_extra,holes', [(128,0,False),(1024,0,False),
                                                (1024,3,True),(1009,3,False)])
def test_row_broadcast_interfaces(dtype, n, stride_extra, holes):
    # Exact binary fractions distinguish addressing/order errors from tolerance.
    rows=65;stride=n+stride_extra
    a=((torch.arange(rows*stride)%61-30)/8).to(dtype).reshape(rows,stride)
    gamma=((torch.arange(n)%9-4)/8).to(dtype)
    beta=((torch.arange(n)%13-6)/8).to(dtype)
    expected=(a[:,:n].float()*gamma.float()+beta.float()).to(dtype)
    if holes:expected[:,torch.arange(n)%7==3]=-73
    inputs=[x.cuda() for x in (a,gamma,beta)]
    for shared in (False,True):
        storage=torch.full((rows*n+128,),-73,dtype=dtype,device='cuda')
        kernel=row_affine[(triton.cdiv(rows,64),)](*inputs,storage[64:],rows,stride,
            n,64,triton.next_power_of_2(n),holes,num_warps=8,num_stages=1,
            is_sdnn=False,buffer_size_limit=2048,xpu_row_shared=shared,
            enable_fp_fusion=False)
        actual=storage.cpu()
        assert torch.equal(actual[64:64+rows*n].reshape(rows,n),expected)
        assert bool((actual[:64]==-73).all() and (actual[-64:]==-73).all())
        # BF16 and masked columns must retain the existing conservative path.
        if dtype==torch.bfloat16 or n!=triton.next_power_of_2(n):
            assert 'xpu.shared_column' not in kernel.asm['ttxir']


@triton.jit
def uniform_row_copy(a, output, rows, B: tl.constexpr, WRAP: tl.constexpr):
    row=tl.program_id(0)*16+tl.arange(0,16)[:,None]
    col=tl.arange(0,B)[None,:]
    if WRAP:
        # All lanes valid; deliberately cross int8's signed discontinuity.
        index=(col+120).to(tl.int8).to(tl.int32)+128
    else:
        index=col
    value=tl.load(a+row*256+index,row<rows,other=-23)
    tl.store(output+row*B+col,value,row<rows)


@pytest.mark.parametrize('wrap', [False, True])
@pytest.mark.parametrize('rows', [0,1,15,16,17])
def test_uniform_row_dma_does_not_erase_integer_wrap_or_false_rows(wrap, rows):
    data=(torch.arange(32*256)%61-30).half().reshape(32,256)
    idx=torch.arange(128)
    if wrap:idx=(idx+120).to(torch.int8).int()+128
    output=torch.full((32*128+128,),-73,dtype=torch.float16,device='cuda')
    uniform_row_copy[(2,)](data.cuda(),output[64:],rows,128,wrap,
        num_warps=8,num_stages=1,is_sdnn=False)
    expected=torch.full((32,128),-73,dtype=torch.float16)
    expected[:rows]=data[:rows,idx]
    actual=output.cpu()
    assert torch.equal(actual[64:-64].reshape(32,128),expected)
    assert bool((actual[:64]==-73).all() and (actual[-64:]==-73).all())


@triton.jit
def row_norm(a, b, gamma, beta, output, rows, N: tl.constexpr,
             X: tl.constexpr, R: tl.constexpr):
    row=tl.program_id(0)*X+tl.arange(0,X)[:,None]
    col=tl.arange(0,R)[None,:]
    mask=(row<rows)&(col<N)
    x=tl.load(a+row*N+col,mask,other=0).to(tl.float32)
    y=tl.load(b+col,col<N,other=0).to(tl.float32)
    value=(x+y).to(a.dtype.element_ty).to(tl.float32)
    value=tl.where(mask,value,0.)
    mean=tl.sum(value,1)[:,None]/N
    delta=value-mean
    variance=tl.sum(tl.where(mask,delta*delta,0.),1)[:,None]/N
    weight=tl.load(gamma+col,col<N,other=0).to(tl.float32)
    bias=tl.load(beta+col,col<N,other=0).to(tl.float32)
    result=delta*tl.rsqrt(variance+1.e-5)*weight+bias
    tl.store(output+row*N+col,result,mask)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.bfloat16])
@pytest.mark.parametrize('n',[128,768,1009,1024])
def test_tiled_reduction_initializer_preserves_operation_order(dtype,n):
    rows=65
    tensors=[((torch.arange(rows*n)%61-30)/16).to(dtype).cuda(),
             ((torch.arange(n)%17-8)/16).to(dtype).cuda(),
             ((torch.arange(n)%13+3)/16).to(dtype).cuda(),
             ((torch.arange(n)%7-3)/16).to(dtype).cuda()]
    outputs=[];metadata=[]
    for shared,reuse,half_cache in ((False,False,False),(True,False,False),
                                   (True,True,False),(True,True,True)):
        storage=torch.full((rows*n+128,),-73,dtype=dtype,device='cuda')
        k=row_norm[(triton.cdiv(rows,16),)](*tensors,storage[64:],rows,n,16,
            triton.next_power_of_2(n),num_warps=8,num_stages=1,is_sdnn=False,
            buffer_size_limit=512,xpu_row_shared=shared,xpu_reduction_reuse=reuse,
            xpu_reduction_half_cache=half_cache,
            enable_fp_fusion=False)
        outputs.append(storage.cpu());metadata.append(k.metadata.buffer_size_limit)
    assert all(m==metadata[0] for m in metadata), 'fixed layout required for operation-order control'
    for actual in outputs[1:]:
        assert torch.equal(outputs[0],actual)
        assert bool(torch.isfinite(actual).all())
        assert bool((actual[:64]==-73).all() and (actual[-64:]==-73).all())


@triton.jit
def row_other(a, output, rows, N: tl.constexpr, VARIABLE: tl.constexpr):
    row=tl.program_id(0)*64+tl.arange(0,64)[:,None]
    col=tl.arange(0,N)[None,:]
    other=(row+col)%17-8 if VARIABLE else -23
    value=tl.load(a+row*N+col,row<rows,other=other)
    # Expose false-row load results, including a nonuniform other tensor.
    tl.store(output+row*N+col,value)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.bfloat16])
@pytest.mark.parametrize('rows',[0,17,64])
@pytest.mark.parametrize('variable',[False,True])
def test_row_other_is_preserved_when_full_valid_rows_skip_prefill(dtype,rows,variable):
    n=128
    data=(torch.arange(64*n)%61-30).to(dtype).reshape(64,n)
    other=((torch.arange(64)[:,None]+torch.arange(n)[None,:])%17-8).to(dtype)
    if not variable:other.fill_(-23)
    expected=other.clone();expected[:rows]=data[:rows]
    storage=torch.full((64*n+128,),-73,dtype=dtype,device='cuda')
    row_other[(1,)](data.cuda(),storage[64:],rows,n,variable,
        num_warps=8,num_stages=1,is_sdnn=False,buffer_size_limit=2048,
        xpu_row_shared=True)
    actual=storage.cpu()
    assert torch.equal(actual[64:-64].reshape(64,n),expected)
    assert bool((actual[:64]==-73).all() and (actual[-64:]==-73).all())
