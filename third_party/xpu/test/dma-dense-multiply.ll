; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-f64:64:64-v512:512-a:0:32-n32-S32"
target triple = "xpu3-baidu-none-gnu"

; Function Attrs: nounwind
define dso_local amdgpu_gfx_whole_wave void @dense_mul(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, i32 %4, i32 %5) local_unnamed_addr #0 {
  %7 = tail call i32 @llvm.xpu.load_param(i32 0)
  %8 = tail call i32 @llvm.xpu.load_param(i32 1)
  %9 = mul i32 %4, %3
  %10 = mul i32 %9, %5
  %11 = tail call i32 @llvm.xpu.load_param(i32 2)
  %12 = tail call i32 @llvm.xpu.core_id()
  %13 = shl i32 %12, 9
  %14 = icmp slt i32 %7, %10
  br i1 %14, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %6, %80
  %15 = phi i32 [ %81, %80 ], [ %7, %6 ]
  %16 = sdiv i32 %15, %5
  %17 = sdiv i32 %16, %4
  %18 = alloca [512 x half], align 2
  %19 = alloca [512 x half], align 2
  %20 = shl i32 %17, 15
  %21 = add i32 %20, %13
  %22 = icmp slt i32 %21, %2
  br i1 %22, label %23, label %80

23:                                               ; preds = %.lr.ph
  %24 = sub i32 %2, %21
  %25 = sext i32 %21 to i64
  %26 = getelementptr half, ptr addrspace(1) %0, i64 %25
  %27 = call i32 @llvm.smax.i32(i32 %24, i32 0)
  %28 = call i32 @llvm.umin.i32(i32 %27, i32 512)
  %29 = shl nuw nsw i32 %28, 1
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %19, ptr addrspace(1) %26, i32 0, i32 %29)
  call void @llvm.xpu.mfence.v2(i32 1)
  %30 = getelementptr half, ptr addrspace(1) %1, i64 %25
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %18, ptr addrspace(1) %30, i32 0, i32 %29)
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %31

31:                                               ; preds = %23, %31
  %32 = phi i32 [ 0, %23 ], [ %78, %31 ]
  %33 = shl nuw nsw i32 %32, 2
  %34 = getelementptr <32 x half>, ptr %19, i32 %33
  %35 = load <32 x half>, ptr %34, align 64
  %36 = getelementptr i8, ptr %34, i32 64
  %37 = load <32 x half>, ptr %36, align 64
  %38 = getelementptr i8, ptr %34, i32 128
  %39 = load <32 x half>, ptr %38, align 64
  %40 = getelementptr i8, ptr %34, i32 192
  %41 = load <32 x half>, ptr %40, align 64
  %42 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %35) #4
  %43 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %35) #4
  %44 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %37) #4
  %45 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %37) #4
  %46 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %39) #4
  %47 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %39) #4
  %48 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %41) #4
  %49 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %41) #4
  %50 = getelementptr <32 x half>, ptr %18, i32 %33
  %51 = load <32 x half>, ptr %50, align 64
  %52 = getelementptr i8, ptr %50, i32 64
  %53 = load <32 x half>, ptr %52, align 64
  %54 = getelementptr i8, ptr %50, i32 128
  %55 = load <32 x half>, ptr %54, align 64
  %56 = getelementptr i8, ptr %50, i32 192
  %57 = load <32 x half>, ptr %56, align 64
  %58 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %51) #4
  %59 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %51) #4
  %60 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %53) #4
  %61 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %53) #4
  %62 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %55) #4
  %63 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %55) #4
  %64 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %57) #4
  %65 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %57) #4
  %66 = fmul <16 x float> %42, %58
  %67 = fmul <16 x float> %43, %59
  %68 = fmul <16 x float> %44, %60
  %69 = fmul <16 x float> %45, %61
  %70 = fmul <16 x float> %46, %62
  %71 = fmul <16 x float> %47, %63
  %72 = fmul <16 x float> %48, %64
  %73 = fmul <16 x float> %49, %65
  %74 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %66, <16 x float> %67) #4
  %75 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %68, <16 x float> %69) #4
  %76 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %70, <16 x float> %71) #4
  %77 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %72, <16 x float> %73) #4
  store <32 x half> %74, ptr %50, align 64
  store <32 x half> %75, ptr %52, align 64
  store <32 x half> %76, ptr %54, align 64
  store <32 x half> %77, ptr %56, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %78 = add nuw nsw i32 %32, 1
  %exitcond.not = icmp eq i32 %78, 4
  br i1 %exitcond.not, label %79, label %31

79:                                               ; preds = %31
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %26, ptr nonnull %18, i32 0, i32 %29)
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %80

80:                                               ; preds = %79, %.lr.ph
  %81 = add i32 %15, %8
  %82 = icmp slt i32 %81, %10
  br i1 %82, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %80, %6
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

