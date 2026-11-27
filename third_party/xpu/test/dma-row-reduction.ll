target datalayout = "e-p:32:32-p1:64:64-p2:32:32-i64:64-v512:512"
target triple = "xpu3-baidu-none-gnu"
@weights = internal addrspace(2) global [64 x float] zeroinitializer, align 64

define void @row(ptr addrspace(1) %input, ptr addrspace(1) %out,
                 ptr addrspace(1) %residual, i32 %rows, i32 %rounded) {
entry:
  br label %header
header:
  %r = phi i32 [0, %entry], [%next, %latch]
  %a = alloca [64 x float], align 64
  %b = alloca [64 x float], align 64
  %index = mul i32 %r, 64
  %wide = zext i32 %index to i64
  %src = getelementptr float, ptr addrspace(1) %input, i64 %wide
  %dst = getelementptr float, ptr addrspace(1) %out, i64 %wide
  %res = getelementptr float, ptr addrspace(1) %residual, i64 %wide
  %valid = icmp ult i32 %r, %rows
  br i1 %valid, label %read, label %fill
read:
  call void @llvm.xpu.gm2lm_v3(ptr %a, ptr addrspace(1) %src, i32 0, i32 256)
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %ready
fill:
  store float 0.0, ptr %a, align 64
  br label %ready
ready:
  br label %tiles
tiles:
  %i = phi i32 [0, %ready], [%inc, %tiles]
  %offset = shl nuw nsw i32 %i, 4
  %p = getelementptr float, ptr %a, i32 %offset
  %q = getelementptr float, ptr %b, i32 %offset
  %w = getelementptr [64 x float], ptr addrspace(2) @weights, i32 0, i32 %offset
  %x = load <16 x float>, ptr %p, align 64
  %weight = load <16 x float>, ptr addrspace(2) %w, align 64
  %value = fadd <16 x float> %x, %weight
  store <16 x float> %value, ptr %p, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  store <16 x float> %value, ptr %q, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %inc = add nuw nsw i32 %i, 1
  %moretiles = icmp ult i32 %inc, 4
  br i1 %moretiles, label %tiles, label %done
done:
  br i1 %valid, label %write, label %latch
write:
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %dst, ptr %a, i32 0, i32 256)
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %res, ptr %b, i32 0, i32 256)
  br label %latch
latch:
  call void @llvm.xpu.mfence.v2(i32 1)
  %next = add i32 %r, 1
  %more = icmp ult i32 %next, %rounded
  br i1 %more, label %header, label %exit
exit:
  ret void
}
declare void @llvm.xpu.gm2lm_v3(ptr, ptr addrspace(1), i32, i32)
declare void @llvm.xpu.lm2gm_v3(ptr addrspace(1), ptr, i32, i32)
declare void @llvm.xpu.mfence.v2(i32)
