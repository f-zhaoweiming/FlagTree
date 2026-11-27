target datalayout = "e-p:32:32-p1:64:64-i64:64-v512:512-n32-S32"
target triple = "xpu3-baidu-none-gnu"
define void @row(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i1 %valid) {
entry:
  %tmp = alloca [16 x i32], align 64
  %hi = getelementptr i8, ptr %tmp, i32 32
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %join]
  %acc = phi float [0.0, %entry], [%sum, %join]
  %off = mul nuw nsw i64 %i, 64
  br i1 %valid, label %read, label %join
read:
  %pa = getelementptr i8, ptr addrspace(1) %a, i64 %off
  %pah = getelementptr i8, ptr addrspace(1) %pa, i64 32
  %pb = getelementptr i8, ptr addrspace(1) %b, i64 %off
  %pbh = getelementptr i8, ptr addrspace(1) %pb, i64 32
  call void @llvm.xpu.gm2lm_v3(ptr %tmp, ptr addrspace(1) %pa, i32 0, i32 32)
  call void @llvm.xpu.gm2lm_v3(ptr %hi, ptr addrspace(1) %pah, i32 0, i32 32)
  call void @llvm.xpu.mfence.v2(i32 1)
  %v = load float, ptr %tmp, align 4
  call void @llvm.xpu.gm2lm_v3(ptr %tmp, ptr addrspace(1) %pb, i32 0, i32 32)
  call void @llvm.xpu.gm2lm_v3(ptr %hi, ptr addrspace(1) %pbh, i32 0, i32 32)
  call void @llvm.xpu.mfence.v2(i32 1)
  %w = load float, ptr %tmp, align 4
  %s = fadd float %v, %w
  br label %join
join:
  %x = phi float [0.0, %loop], [%s, %read]
  %sum = fadd float %acc, %x
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 16
  br i1 %done, label %exit, label %loop
exit:
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}
declare void @llvm.xpu.gm2lm_v3(ptr, ptr addrspace(1), i32, i32)
declare void @llvm.xpu.mfence.v2(i32)
