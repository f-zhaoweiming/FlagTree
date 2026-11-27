; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-f64:64:64-v512:512-a:0:32-n32-S32"
target triple = "xpu3-baidu-none-gnu"

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare !dbg !8 void @llvm.xpu.dsmadd(ptr, ptr, ptr, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !11 void @llvm.xpu.ewcoeff_cfg_scalar(float, float, float, i32) #1

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare !dbg !12 void @llvm.xpu.mm(ptr, ptr, ptr, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !13 void @llvm.xpu.mm_cfg_int(i32, i32, i32, i32) #1

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare !dbg !14 void @llvm.xpu.shuffle_batch.v2(ptr, ptr, i32, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !15 void @llvm.xpu.ds_cfg(i32, i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !16 void @llvm.xpu.dma_run.v3(i32) #1

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare !dbg !17 void @llvm.xpu.dma_cfg(i32, i32, i32, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !18 void @llvm.xpu.xwait(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !19 void @llvm.xpu.xsignal(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !21 void @llvm.xpu.csr_set_sync_group(i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !22 void @llvm.xpu.unlock(i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !23 void @llvm.xpu.ew_cfg(i32, i32, i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !24 void @llvm.xpu.lock.v2(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !25 void @llvm.xpu.uni_sram_cfg(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare !dbg !26 void @llvm.xpu.xsignal_mode_cfg(i32) #1

; Function Attrs: nounwind
define dso_local amdgpu_gfx_whole_wave void @triton_tem__9(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7) local_unnamed_addr #2 !dbg !27 {
  tail call void @llvm.xpu.xsignal_mode_cfg(i32 1)
  %9 = tail call i32 @llvm.xpu.core_id(), !dbg !28
  %10 = icmp eq i32 %9, 0, !dbg !29
  br i1 %10, label %11, label %.critedge

11:                                               ; preds = %8
  tail call void @llvm.xpu.uni_sram_cfg(i32 1, i32 1)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 1, i32 0, i32 0, i32 16)
  tail call void @llvm.xpu.unlock(i32 2)
  tail call void @llvm.xpu.csr_set_sync_group(i32 -1)
  tail call void @llvm.xpu.xsignal(i32 5, i32 0)
  tail call void @llvm.xpu.xsignal(i32 5, i32 0)
  tail call void @llvm.xpu.xsignal(i32 3, i32 0)
  tail call void @llvm.xpu.xsignal(i32 9, i32 0)
  tail call void @llvm.xpu.xsignal(i32 9, i32 0)
  tail call void @llvm.xpu.xsignal(i32 7, i32 0)
  tail call void @llvm.xpu.xsignal(i32 12, i32 0)
  tail call void @llvm.xpu.xsignal(i32 14, i32 0)
  tail call void @llvm.xpu.xsignal(i32 16, i32 0)
  br label %12

.critedge:                                        ; preds = %8
  tail call void @llvm.xpu.csr_set_sync_group(i32 -1)
  br label %12

12:                                               ; preds = %.critedge, %11
  %13 = tail call i32 @llvm.xpu.load_param(i32 0)
  %14 = tail call i32 @llvm.xpu.load_param(i32 1)
  %15 = mul i32 %6, %5
  %16 = mul i32 %15, %7
  %17 = add i32 %16, -1
  %18 = add i32 %17, %14
  %19 = sdiv i32 %18, %14
  %20 = mul i32 %19, %13
  %21 = add i32 %20, %19
  %22 = tail call i32 @llvm.smin.i32(i32 %21, i32 %16)
  %23 = sext i32 %22 to i64
  %24 = icmp slt i32 %20, %22
  br i1 %24, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %12
  %25 = sext i32 %20 to i64
  %26 = icmp eq i32 %9, 1
  %27 = icmp eq i32 %9, 2
  br label %28

28:                                               ; preds = %.lr.ph, %.critedge2.thread19
  %29 = phi i64 [ %25, %.lr.ph ], [ %143, %.critedge2.thread19 ]
  %30 = phi i32 [ 0, %.lr.ph ], [ %53, %.critedge2.thread19 ]
  %31 = trunc nsw i64 %29 to i32
  %32 = sdiv i32 %31, %7
  %33 = sdiv i32 %32, %6
  %34 = sdiv i32 %33, 64, !dbg !30
  %35 = shl nsw i32 %34, 3, !dbg !31
  %36 = sub nsw i32 91, %35, !dbg !32
  %37 = tail call i32 @llvm.smin.i32(i32 %36, i32 8), !dbg !33
  %38 = srem i32 %33, %37, !dbg !34
  %39 = add nsw i32 %35, %38, !dbg !35
  %.fr = freeze i32 %39, !dbg !36
  %40 = srem i32 %33, 64, !dbg !37
  %41 = sdiv i32 %40, %37, !dbg !38
  %42 = icmp sgt i32 %.fr, -1, !dbg !39
  tail call void @llvm.assume(i1 %42), !dbg !40
  %43 = icmp sgt i32 %41, -1, !dbg !41
  tail call void @llvm.assume(i1 %43), !dbg !42
  %44 = shl i32 %.fr, 6, !dbg !43
  %45 = zext nneg i32 %.fr to i64, !dbg !36
  %46 = shl nuw nsw i32 %41, 7, !dbg !44
  %47 = zext nneg i32 %41 to i64, !dbg !45
  %48 = shl nuw nsw i64 %47, 7, !dbg !45
  %49 = shl nuw nsw i64 %45, 16, !dbg !46
  %50 = srem i32 %30, 2, !dbg !47
  %51 = shl nsw i32 %50, 18, !dbg !47
  %52 = inttoptr i32 %51 to ptr, !dbg !47
  %53 = add nuw i32 %30, 1, !dbg !47
  br i1 %26, label %.thread, label %66, !dbg !28

.thread:                                          ; preds = %28
  %54 = icmp samesign ugt i32 %.fr, 90, !dbg !48
  %55 = urem i64 %49, 5908480, !dbg !48
  %56 = select i1 %54, i64 %55, i64 %49, !dbg !48
  %57 = lshr exact i64 %56, 10, !dbg !48
  %58 = tail call i64 @llvm.usub.sat.i64(i64 %57, i64 5706), !dbg !48
  %59 = trunc i64 %58 to i32, !dbg !28
  %60 = sub i32 64, %59, !dbg !28
  %61 = getelementptr float, ptr addrspace(1) %0, i64 %49, !dbg !28
  tail call void @llvm.xpu.xwait(i32 3, i32 0), !dbg !28
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4), !dbg !28
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0), !dbg !28
  %62 = ptrtoint ptr addrspace(1) %61 to i64, !dbg !28
  %63 = lshr i64 %62, 32, !dbg !28
  %64 = trunc nuw i64 %63 to i32, !dbg !28
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %64, i32 0, i32 2), !dbg !28
  %65 = trunc i64 %62 to i32, !dbg !28
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %65, i32 4096, i32 3), !dbg !28
  tail call void @llvm.xpu.dma_cfg(i32 4096, i32 4096, i32 %60, i32 4), !dbg !28
  tail call void @llvm.xpu.dma_run.v3(i32 0), !dbg !28
  tail call void @llvm.xpu.unlock(i32 4), !dbg !28
  tail call void @llvm.xpu.xsignal(i32 2, i32 0), !dbg !28
  br label %.thread6, !dbg !47

66:                                               ; preds = %28
  br i1 %27, label %67, label %69, !dbg !47

67:                                               ; preds = %66
  tail call void @llvm.xpu.xwait(i32 2, i32 0), !dbg !47
  tail call void @llvm.xpu.xwait(i32 5, i32 0), !dbg !47
  tail call void @llvm.xpu.lock.v2(i32 0, i32 0), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 2, i32 0, i32 0), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 1), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 14), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 64, i32 0, i32 13), !dbg !47
  tail call void @llvm.xpu.shuffle_batch.v2(ptr %52, ptr addrspacecast (ptr addrspace(2) null to ptr), i32 1024, i32 1), !dbg !47
  tail call void @llvm.xpu.unlock(i32 0), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 3, i32 0), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 4, i32 0), !dbg !47
  br label %.thread6, !dbg !47

.thread6:                                         ; preds = %.thread, %67
  %68 = shl nsw i32 %50, 19, !dbg !47
  br label %.thread10, !dbg !49

69:                                               ; preds = %66
  %70 = shl nsw i32 %50, 19, !dbg !47
  switch i32 %9, label %.thread10 [
    i32 6, label %71
    i32 7, label %86
  ], !dbg !49

71:                                               ; preds = %69
  %72 = shl nuw nsw i64 %47, 17, !dbg !46
  %73 = icmp samesign ugt i32 %41, 7, !dbg !50
  %74 = and i64 %72, 281474975662080, !dbg !50
  %75 = select i1 %73, i64 %74, i64 0, !dbg !50
  %76 = sub nsw i64 %72, %75, !dbg !50
  %77 = ashr exact i64 %76, 10, !dbg !50
  %78 = tail call i64 @llvm.smin.i64(i64 %77, i64 896), !dbg !50
  %reass.sub = sub nsw i64 %78, %77, !dbg !50
  %79 = trunc nsw i64 %reass.sub to i32, !dbg !49
  %80 = add nsw i32 %79, 128, !dbg !49
  %81 = getelementptr float, ptr addrspace(1) %1, i64 %72, !dbg !49
  tail call void @llvm.xpu.xwait(i32 7, i32 0), !dbg !49
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4), !dbg !49
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0), !dbg !49
  %82 = ptrtoint ptr addrspace(1) %81 to i64, !dbg !49
  %83 = lshr i64 %82, 32, !dbg !49
  %84 = trunc nuw i64 %83 to i32, !dbg !49
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %84, i32 0, i32 2), !dbg !49
  %85 = trunc i64 %82 to i32, !dbg !49
  tail call void @llvm.xpu.dma_cfg(i32 262144, i32 %85, i32 4096, i32 3), !dbg !49
  tail call void @llvm.xpu.dma_cfg(i32 4096, i32 4096, i32 %80, i32 4), !dbg !49
  tail call void @llvm.xpu.dma_run.v3(i32 0), !dbg !49
  tail call void @llvm.xpu.unlock(i32 4), !dbg !49
  tail call void @llvm.xpu.xsignal(i32 6, i32 0), !dbg !49
  br label %.thread10, !dbg !49

86:                                               ; preds = %69
  %87 = inttoptr i32 %70 to ptr, !dbg !47
  tail call void @llvm.xpu.xwait(i32 6, i32 0), !dbg !47
  tail call void @llvm.xpu.xwait(i32 9, i32 0), !dbg !47
  tail call void @llvm.xpu.lock.v2(i32 0, i32 0), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 2, i32 0, i32 0), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 1), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 14), !dbg !47
  tail call void @llvm.xpu.ds_cfg(i32 128, i32 0, i32 13), !dbg !47
  tail call void @llvm.xpu.shuffle_batch.v2(ptr %87, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 262144 to ptr addrspace(2)) to ptr), i32 1024, i32 2), !dbg !47
  tail call void @llvm.xpu.unlock(i32 0), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 7, i32 0), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 8, i32 0), !dbg !47
  br label %.thread10, !dbg !47

.thread10:                                        ; preds = %69, %71, %.thread6, %86
  %88 = phi i32 [ %70, %69 ], [ %70, %86 ], [ %70, %71 ], [ %68, %.thread6 ]
  %89 = phi i1 [ false, %69 ], [ false, %86 ], [ true, %71 ], [ false, %.thread6 ]
  %90 = add nuw nsw i64 %49, %48, !dbg !51
  %91 = sext i32 %44 to i64, !dbg !52
  %92 = tail call i64 @llvm.smin.i64(i64 %91, i64 5706), !dbg !52
  %93 = add nsw i64 %92, 64, !dbg !52
  %94 = tail call i64 @llvm.smax.i64(i64 %93, i64 %91), !dbg !52
  %95 = sub nsw i64 %94, %91, !dbg !52
  %96 = zext nneg i32 %46 to i64, !dbg !52
  %97 = tail call i64 @llvm.umin.i64(i64 %96, i64 896), !dbg !52
  %98 = add nuw nsw i64 %97, 128, !dbg !52
  %99 = tail call i64 @llvm.usub.sat.i64(i64 %98, i64 %96), !dbg !52
  %100 = tail call i64 @llvm.smin.i64(i64 %95, i64 64), !dbg !52
  %101 = tail call i64 @llvm.umin.i64(i64 %99, i64 128), !dbg !52
  %102 = trunc i64 %100 to i32, !dbg !52
  %103 = trunc nuw nsw i64 %101 to i32, !dbg !52
  br i1 %26, label %104, label %111, !dbg !52

104:                                              ; preds = %.thread10
  %105 = getelementptr float, ptr addrspace(1) %2, i64 %90, !dbg !52
  tail call void @llvm.xpu.xwait(i32 12, i32 0), !dbg !52
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4), !dbg !52
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0), !dbg !52
  %106 = ptrtoint ptr addrspace(1) %105 to i64, !dbg !52
  %107 = lshr i64 %106, 32, !dbg !52
  %108 = trunc nuw i64 %107 to i32, !dbg !52
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %108, i32 0, i32 2), !dbg !52
  %109 = shl nuw nsw i32 %103, 2, !dbg !52
  %110 = trunc i64 %106 to i32, !dbg !52
  tail call void @llvm.xpu.dma_cfg(i32 786432, i32 %110, i32 %109, i32 3), !dbg !52
  tail call void @llvm.xpu.dma_cfg(i32 512, i32 4096, i32 %102, i32 4), !dbg !52
  tail call void @llvm.xpu.dma_run.v3(i32 0), !dbg !52
  tail call void @llvm.xpu.unlock(i32 4), !dbg !52
  tail call void @llvm.xpu.xsignal(i32 10, i32 0), !dbg !52
  br label %111, !dbg !52

111:                                              ; preds = %104, %.thread10
  br i1 %89, label %.critedge2.thread, label %118, !dbg !53

.critedge2.thread:                                ; preds = %111
  %112 = getelementptr float, ptr addrspace(1) %3, i64 %48, !dbg !53
  tail call void @llvm.xpu.xwait(i32 14, i32 0), !dbg !53
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4), !dbg !53
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0), !dbg !53
  %113 = ptrtoint ptr addrspace(1) %112 to i64, !dbg !53
  %114 = lshr i64 %113, 32, !dbg !53
  %115 = trunc nuw i64 %114 to i32, !dbg !53
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %115, i32 0, i32 2), !dbg !53
  %116 = shl nuw nsw i32 %103, 2, !dbg !53
  %117 = trunc i64 %113 to i32, !dbg !53
  tail call void @llvm.xpu.dma_cfg(i32 819200, i32 %117, i32 %116, i32 3), !dbg !53
  tail call void @llvm.xpu.dma_cfg(i32 512, i32 512, i32 1, i32 4), !dbg !53
  tail call void @llvm.xpu.dma_run.v3(i32 0), !dbg !53
  tail call void @llvm.xpu.unlock(i32 4), !dbg !53
  tail call void @llvm.xpu.xsignal(i32 13, i32 0), !dbg !53
  br label %.critedge2.thread19, !dbg !29

118:                                              ; preds = %111
  switch i32 %9, label %.critedge2 [
    i32 4, label %119
    i32 5, label %130
  ], !dbg !47

119:                                              ; preds = %118
  tail call void @llvm.xpu.xwait(i32 4, i32 0), !dbg !47
  tail call void @llvm.xpu.xwait(i32 8, i32 0), !dbg !47
  tail call void @llvm.xpu.xwait(i32 16, i32 0), !dbg !47
  br label %120, !dbg !47

120:                                              ; preds = %119, %120
  %121 = phi i1 [ true, %119 ], [ false, %120 ]
  %122 = phi i32 [ 0, %119 ], [ 1, %120 ]
  %123 = shl nuw nsw i32 %122, 18, !dbg !47
  %124 = shl nuw nsw i32 %122, 8, !dbg !47
  %125 = or disjoint i32 %123, %88, !dbg !47
  %126 = or disjoint i32 %124, 819712, !dbg !47
  tail call void @llvm.xpu.lock.v2(i32 0, i32 1), !dbg !47
  tail call void @llvm.xpu.mm_cfg_int(i32 64, i32 1024, i32 64, i32 0), !dbg !47
  tail call void @llvm.xpu.mm_cfg_int(i32 64, i32 4, i32 1, i32 2), !dbg !47
  %127 = inttoptr i32 %126 to ptr, !dbg !47
  %128 = inttoptr i32 %125 to ptr, !dbg !47
  tail call void @llvm.xpu.mm(ptr nonnull %127, ptr %52, ptr %128, i32 8), !dbg !47
  tail call void @llvm.xpu.unlock(i32 1), !dbg !47
  br i1 %121, label %120, label %129, !dbg !47

129:                                              ; preds = %120
  tail call void @llvm.xpu.xsignal(i32 5, i32 0), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 9, i32 0), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 15, i32 0), !dbg !47
  br label %.critedge2.thread19, !dbg !47

130:                                              ; preds = %118
  tail call void @llvm.xpu.xwait(i32 13, i32 0), !dbg !47
  tail call void @llvm.xpu.xwait(i32 15, i32 0), !dbg !47
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2), !dbg !47
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 1), !dbg !47
  tail call void @llvm.xpu.ewcoeff_cfg_scalar(float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, i32 0), !dbg !47
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 127, i32 0, i32 2), !dbg !47
  tail call void @llvm.xpu.ew_cfg(i32 1, i32 0, i32 0, i32 3), !dbg !47
  br label %131, !dbg !47

131:                                              ; preds = %130, %131
  %132 = phi i32 [ 0, %130 ], [ %134, %131 ]
  %.idx = shl nuw nsw i32 %132, 9, !dbg !47
  %133 = getelementptr i8, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 819712 to ptr addrspace(2)) to ptr), i32 %.idx, !dbg !47
  tail call void @llvm.xpu.dsmadd(ptr %133, ptr %133, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 819200 to ptr addrspace(2)) to ptr), i32 0), !dbg !47
  %134 = add nuw nsw i32 %132, 1, !dbg !47
  %exitcond.not = icmp eq i32 %134, 64, !dbg !47
  br i1 %exitcond.not, label %135, label %131, !dbg !47

135:                                              ; preds = %131
  tail call void @llvm.xpu.unlock(i32 2), !dbg !47
  tail call void @llvm.xpu.xsignal(i32 14, i32 0), !dbg !47
  tail call void @llvm.xpu.xwait(i32 10, i32 0), !dbg !54
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2), !dbg !54
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 1), !dbg !54
  tail call void @llvm.xpu.ewcoeff_cfg_scalar(float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, i32 0), !dbg !54
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 2047, i32 0, i32 2), !dbg !54
  tail call void @llvm.xpu.ew_cfg(i32 4, i32 0, i32 0, i32 3), !dbg !54
  tail call void @llvm.xpu.ew_cfg(i32 4, i32 4, i32 4, i32 19), !dbg !54
  tail call void @llvm.xpu.dsmadd(ptr addrspacecast (ptr addrspace(2) inttoptr (i32 786432 to ptr addrspace(2)) to ptr), ptr addrspacecast (ptr addrspace(2) inttoptr (i32 786432 to ptr addrspace(2)) to ptr), ptr addrspacecast (ptr addrspace(2) inttoptr (i32 819712 to ptr addrspace(2)) to ptr), i32 0), !dbg !54
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 19), !dbg !54
  tail call void @llvm.xpu.unlock(i32 2), !dbg !54
  tail call void @llvm.xpu.xsignal(i32 11, i32 0), !dbg !54
  tail call void @llvm.xpu.xsignal(i32 16, i32 0), !dbg !54
  br label %.critedge2.thread19, !dbg !54

.critedge2:                                       ; preds = %118
  br i1 %10, label %136, label %.critedge2.thread19, !dbg !29

136:                                              ; preds = %.critedge2
  %137 = getelementptr float, ptr addrspace(1) %4, i64 %90, !dbg !29
  tail call void @llvm.xpu.xwait(i32 11, i32 0), !dbg !29
  tail call void @llvm.xpu.lock.v2(i32 0, i32 5), !dbg !29
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0), !dbg !29
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 1, i32 0, i32 1), !dbg !29
  %138 = ptrtoint ptr addrspace(1) %137 to i64, !dbg !29
  %139 = lshr i64 %138, 32, !dbg !29
  %140 = trunc nuw i64 %139 to i32, !dbg !29
  tail call void @llvm.xpu.dma_cfg(i32 %140, i32 0, i32 0, i32 2), !dbg !29
  %141 = shl nuw nsw i32 %103, 2, !dbg !29
  %142 = trunc i64 %138 to i32, !dbg !29
  tail call void @llvm.xpu.dma_cfg(i32 %142, i32 786432, i32 %141, i32 3), !dbg !29
  tail call void @llvm.xpu.dma_cfg(i32 4096, i32 512, i32 %102, i32 4), !dbg !29
  tail call void @llvm.xpu.dma_run.v3(i32 0), !dbg !29
  tail call void @llvm.xpu.unlock(i32 5), !dbg !29
  tail call void @llvm.xpu.xsignal(i32 12, i32 0), !dbg !29
  br label %.critedge2.thread19, !dbg !29

.critedge2.thread19:                              ; preds = %135, %129, %.critedge2.thread, %136, %.critedge2
  %143 = add nsw i64 %29, 1
  %144 = icmp slt i64 %143, %23
  br i1 %144, label %28, label %._crit_edge

._crit_edge:                                      ; preds = %.critedge2.thread19, %12
  tail call void @llvm.xpu.xsignal(i32 0, i32 0), !dbg !55
  br i1 %10, label %145, label %146, !dbg !55

145:                                              ; preds = %._crit_edge
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.xwait(i32 0, i32 0), !dbg !55
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2), !dbg !55
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 16), !dbg !55
  tail call void @llvm.xpu.unlock(i32 2), !dbg !55
  tail call void @llvm.xpu.xsignal_mode_cfg(i32 0), !dbg !55
  br label %146, !dbg !55

146:                                              ; preds = %145, %._crit_edge
  ret void, !dbg !55
}

; Function Attrs: nofree nosync nounwind memory(none)
; Unknown intrinsic
declare i32 @llvm.xpu.core_id() #3

; Function Attrs: nofree nosync nounwind memory(none)
; Unknown intrinsic
declare i32 @llvm.xpu.load_param(i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #6

attributes #0 = { nounwind memory(argmem: readwrite) "target-features"="+cdnn3" }
attributes #1 = { nounwind "target-features"="+cdnn3" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="xpu3" "target-features"="+cdnn3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nosync nounwind memory(none) "target-features"="+cdnn3" }
attributes #4 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) "target-features"="+cdnn3" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) "target-features"="+cdnn3" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) "target-features"="+cdnn3" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!xpu.host.triple = !{!5}
!llvm.ident = !{!6}
!xpu.annotations = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "csq7zxcaiufmalt26s5syk42tmfaohoznojvob46ys3nrhwkyglk.py", directory: "/data2/zhaoweiming/f06-all-models-20261119/validation/sdnn-stage-sweep-1795380978221109720/torchinductor_cache_dir/sq")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"x86_64-baidu-linux-gnu"}
!6 = !{!"KUNLUNXIN XPU C/C++ Compiler Version 4.0.0, based on clang version 22.1.8 (ssh://zhangzhuangzhuang01@icode.baidu.com:8235/baidu/xpu/llvm 2579dbaec939a3e7e5a0f46f43a0a00255c03f1f)"}
!7 = !{ptr @triton_tem__9, !"kernel", i32 1}
!8 = !DISubprogram(name: "llvm.xpu.dsmadd", linkageName: "llvm.xpu.dsmadd", scope: !1, file: !1, line: 85, type: !9, scopeLine: 85, spFlags: DISPFlagOptimized)
!9 = !DISubroutineType(cc: DW_CC_normal, types: !10)
!10 = !{}
!11 = !DISubprogram(name: "llvm.xpu.ewcoeff_cfg_scalar", linkageName: "llvm.xpu.ewcoeff_cfg_scalar", scope: !1, file: !1, line: 85, type: !9, scopeLine: 85, spFlags: DISPFlagOptimized)
!12 = !DISubprogram(name: "llvm.xpu.mm", linkageName: "llvm.xpu.mm", scope: !1, file: !1, line: 85, type: !9, scopeLine: 85, spFlags: DISPFlagOptimized)
!13 = !DISubprogram(name: "llvm.xpu.mm_cfg_int", linkageName: "llvm.xpu.mm_cfg_int", scope: !1, file: !1, line: 85, type: !9, scopeLine: 85, spFlags: DISPFlagOptimized)
!14 = !DISubprogram(name: "llvm.xpu.shuffle_batch.v2", linkageName: "llvm.xpu.shuffle_batch.v2", scope: !1, file: !1, line: 85, type: !9, scopeLine: 85, spFlags: DISPFlagOptimized)
!15 = !DISubprogram(name: "llvm.xpu.ds_cfg", linkageName: "llvm.xpu.ds_cfg", scope: !1, file: !1, line: 85, type: !9, scopeLine: 85, spFlags: DISPFlagOptimized)
!16 = !DISubprogram(name: "llvm.xpu.dma_run.v3", linkageName: "llvm.xpu.dma_run.v3", scope: !1, file: !1, line: 77, type: !9, scopeLine: 77, spFlags: DISPFlagOptimized)
!17 = !DISubprogram(name: "llvm.xpu.dma_cfg", linkageName: "llvm.xpu.dma_cfg", scope: !1, file: !1, line: 77, type: !9, scopeLine: 77, spFlags: DISPFlagOptimized)
!18 = !DISubprogram(name: "llvm.xpu.xwait", linkageName: "llvm.xpu.xwait", scope: !1, file: !1, line: 77, type: !9, scopeLine: 77, spFlags: DISPFlagOptimized)
!19 = !DISubprogram(name: "llvm.xpu.xsignal", linkageName: "llvm.xpu.xsignal", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!20 = !DIFile(filename: "<unknown>", directory: "")
!21 = !DISubprogram(name: "llvm.xpu.csr_set_sync_group", linkageName: "llvm.xpu.csr_set_sync_group", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!22 = !DISubprogram(name: "llvm.xpu.unlock", linkageName: "llvm.xpu.unlock", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!23 = !DISubprogram(name: "llvm.xpu.ew_cfg", linkageName: "llvm.xpu.ew_cfg", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!24 = !DISubprogram(name: "llvm.xpu.lock.v2", linkageName: "llvm.xpu.lock.v2", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!25 = !DISubprogram(name: "llvm.xpu.uni_sram_cfg", linkageName: "llvm.xpu.uni_sram_cfg", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!26 = !DISubprogram(name: "llvm.xpu.xsignal_mode_cfg", linkageName: "llvm.xpu.xsignal_mode_cfg", scope: !20, file: !20, type: !9, spFlags: DISPFlagOptimized)
!27 = distinct !DISubprogram(name: "triton_tem__9", linkageName: "triton_tem__9", scope: !1, file: !1, line: 18, type: !9, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!28 = !DILocation(line: 77, column: 20, scope: !27)
!29 = !DILocation(line: 101, column: 80, scope: !27)
!30 = !DILocation(line: 49, column: 22, scope: !27)
!31 = !DILocation(line: 50, column: 41, scope: !27)
!32 = !DILocation(line: 50, column: 30, scope: !27)
!33 = !DILocation(line: 50, column: 50, scope: !27)
!34 = !DILocation(line: 51, column: 40, scope: !27)
!35 = !DILocation(line: 51, column: 34, scope: !27)
!36 = !DILocation(line: 91, column: 15, scope: !27)
!37 = !DILocation(line: 52, column: 19, scope: !27)
!38 = !DILocation(line: 52, column: 30, scope: !27)
!39 = !DILocation(line: 53, column: 23, scope: !27)
!40 = !DILocation(line: 53, column: 14, scope: !27)
!41 = !DILocation(line: 54, column: 23, scope: !27)
!42 = !DILocation(line: 54, column: 14, scope: !27)
!43 = !DILocation(line: 56, column: 17, scope: !27)
!44 = !DILocation(line: 57, column: 17, scope: !27)
!45 = !DILocation(line: 92, column: 15, scope: !27)
!46 = !DILocation(line: 0, scope: !27)
!47 = !DILocation(line: 85, column: 25, scope: !27)
!48 = !DILocation(line: 77, column: 25, scope: !27)
!49 = !DILocation(line: 82, column: 20, scope: !27)
!50 = !DILocation(line: 82, column: 25, scope: !27)
!51 = !DILocation(line: 97, column: 30, scope: !27)
!52 = !DILocation(line: 97, column: 79, scope: !27)
!53 = !DILocation(line: 98, column: 66, scope: !27)
!54 = !DILocation(line: 100, column: 18, scope: !27)
!55 = !DILocation(line: 18, scope: !27)
