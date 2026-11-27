; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-f64:64:64-v512:512-a:0:32-n32-S32"
target triple = "xpu3-baidu-none-gnu"

; Function Attrs: nounwind
define dso_local amdgpu_gfx_whole_wave void @dense_bias_add(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6) local_unnamed_addr #0 {
  %8 = tail call i32 @llvm.xpu.load_param(i32 0)
  %9 = tail call i32 @llvm.xpu.load_param(i32 1)
  %10 = mul i32 %5, %4
  %11 = mul i32 %10, %6
  %12 = tail call i32 @llvm.xpu.load_param(i32 2)
  %13 = tail call i32 @llvm.xpu.core_id()
  %14 = shl i32 %13, 8
  %15 = icmp slt i32 %8, %11
  br i1 %15, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %7, %73
  %16 = phi i32 [ %74, %73 ], [ %8, %7 ]
  %17 = sdiv i32 %16, %6
  %18 = sdiv i32 %17, %5
  %19 = alloca [256 x float], align 4
  %20 = alloca [256 x float], align 4
  %21 = alloca [256 x float], align 4
  %22 = shl i32 %18, 14
  %23 = add i32 %22, %14
  %24 = icmp slt i32 %23, 1000000
  br i1 %24, label %25, label %73

25:                                               ; preds = %.lr.ph
  %26 = sub i32 1000000, %23
  %27 = sext i32 %23 to i64
  %28 = getelementptr float, ptr addrspace(1) %0, i64 %27
  %29 = call i32 @llvm.smax.i32(i32 %26, i32 0)
  %30 = call i32 @llvm.umin.i32(i32 %29, i32 256)
  %31 = shl nuw nsw i32 %30, 2
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %21, ptr addrspace(1) %28, i32 0, i32 %31)
  %32 = getelementptr float, ptr addrspace(1) %1, i64 %27
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %20, ptr addrspace(1) %32, i32 0, i32 %31)
  %33 = srem i32 %23, 1024
  %34 = sext i32 %33 to i64
  %35 = getelementptr float, ptr addrspace(1) %2, i64 %34
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %19, ptr addrspace(1) %35, i32 0, i32 %31)
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %36

36:                                               ; preds = %25, %36
  %37 = phi i32 [ 0, %25 ], [ %71, %36 ]
  %38 = shl nuw nsw i32 %37, 2
  %39 = getelementptr <16 x float>, ptr %21, i32 %38
  %40 = load <16 x float>, ptr %39, align 64
  %41 = getelementptr i8, ptr %39, i32 64
  %42 = load <16 x float>, ptr %41, align 64
  %43 = getelementptr i8, ptr %39, i32 128
  %44 = load <16 x float>, ptr %43, align 64
  %45 = getelementptr i8, ptr %39, i32 192
  %46 = load <16 x float>, ptr %45, align 64
  %47 = getelementptr <16 x float>, ptr %20, i32 %38
  %48 = load <16 x float>, ptr %47, align 64
  %49 = getelementptr i8, ptr %47, i32 64
  %50 = load <16 x float>, ptr %49, align 64
  %51 = getelementptr i8, ptr %47, i32 128
  %52 = load <16 x float>, ptr %51, align 64
  %53 = getelementptr i8, ptr %47, i32 192
  %54 = load <16 x float>, ptr %53, align 64
  %55 = getelementptr <16 x float>, ptr %19, i32 %38
  %56 = load <16 x float>, ptr %55, align 64
  %57 = getelementptr i8, ptr %55, i32 64
  %58 = load <16 x float>, ptr %57, align 64
  %59 = getelementptr i8, ptr %55, i32 128
  %60 = load <16 x float>, ptr %59, align 64
  %61 = getelementptr i8, ptr %55, i32 192
  %62 = load <16 x float>, ptr %61, align 64
  %63 = fadd <16 x float> %48, %56
  %64 = fadd <16 x float> %50, %58
  %65 = fadd <16 x float> %52, %60
  %66 = fadd <16 x float> %54, %62
  %67 = fadd <16 x float> %40, %63
  %68 = fadd <16 x float> %42, %64
  %69 = fadd <16 x float> %44, %65
  %70 = fadd <16 x float> %46, %66
  store <16 x float> %67, ptr %55, align 64
  store <16 x float> %68, ptr %57, align 64
  store <16 x float> %69, ptr %59, align 64
  store <16 x float> %70, ptr %61, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %71 = add nuw nsw i32 %37, 1
  %exitcond.not = icmp eq i32 %71, 4
  br i1 %exitcond.not, label %72, label %36

72:                                               ; preds = %36
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %28, ptr nonnull %19, i32 0, i32 %31)
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %73

73:                                               ; preds = %72, %.lr.ph
  %74 = add i32 %16, %9
  %75 = icmp slt i32 %74, %11
  br i1 %75, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %73, %7
  ret void
}

; Function Attrs: nofree nosync nounwind memory(none)
; Unknown intrinsic
declare i32 @llvm.xpu.load_param(i32) #1

; Function Attrs: nofree nosync nounwind memory(none)
; Unknown intrinsic
declare i32 @llvm.xpu.core_id() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.gm2lm_v3(ptr, ptr addrspace(1), i32, i32) #3

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.mfence.v2(i32) #4

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.lm2gm_v3(ptr addrspace(1), ptr, i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #5

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="xpu3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind memory(argmem: readwrite) }
attributes #4 = { nounwind }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }


