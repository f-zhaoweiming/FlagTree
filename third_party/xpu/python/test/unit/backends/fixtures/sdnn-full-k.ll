; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-f64:64:64-v512:512-a:0:32-n32-S32"
target triple = "xpu3-baidu-none-gnu"

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.dsmadd(ptr, ptr, ptr, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.ewcoeff_cfg_scalar(float, float, float, i32) #1

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.mm(ptr, ptr, ptr, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.mm_cfg_int(i32, i32, i32, i32) #1

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.shuffle_batch.v2(ptr, ptr, i32, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.ds_cfg(i32, i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.dma_run.v3(i32) #1

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.dma_cfg(i32, i32, i32, i32) #0

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.xwait(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.xsignal(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.csr_set_sync_group(i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.unlock(i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.ew_cfg(i32, i32, i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.lock.v2(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.uni_sram_cfg(i32, i32) #1

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.xsignal_mode_cfg(i32) #1

; Function Attrs: nounwind
define dso_local amdgpu_gfx_whole_wave void @full_k_gemm(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7) local_unnamed_addr #2 {
  tail call void @llvm.xpu.xsignal_mode_cfg(i32 1)
  %9 = tail call i32 @llvm.xpu.core_id()
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %.critedge

11:                                               ; preds = %8
  tail call void @llvm.xpu.uni_sram_cfg(i32 1, i32 1)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 1, i32 0, i32 0, i32 16)
  tail call void @llvm.xpu.unlock(i32 2)
  tail call void @llvm.xpu.csr_set_sync_group(i32 -1)
  tail call void @llvm.xpu.xsignal(i32 5, i32 0)
  tail call void @llvm.xpu.xsignal(i32 3, i32 0)
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
  %26 = icmp eq i32 %9, 6
  br label %27

27:                                               ; preds = %.lr.ph, %.critedge2.thread14
  %28 = phi i64 [ %25, %.lr.ph ], [ %121, %.critedge2.thread14 ]
  %29 = trunc nsw i64 %28 to i32
  %30 = sdiv i32 %29, %7
  %31 = sdiv i32 %30, %6
  %32 = sdiv i32 %31, 64
  %33 = shl nsw i32 %32, 3
  %34 = sub nsw i32 46, %33
  %35 = tail call i32 @llvm.smin.i32(i32 %34, i32 8)
  %36 = srem i32 %31, %35
  %37 = add nsw i32 %33, %36
  %.fr = freeze i32 %37
  %38 = srem i32 %31, 64
  %39 = sdiv i32 %38, %35
  %40 = icmp sgt i32 %.fr, -1
  tail call void @llvm.assume(i1 %40)
  %41 = icmp sgt i32 %39, -1
  tail call void @llvm.assume(i1 %41)
  %42 = shl i32 %.fr, 7
  %43 = zext nneg i32 %.fr to i64
  %44 = shl nuw nsw i32 %39, 7
  %45 = zext nneg i32 %39 to i64
  %46 = shl nuw nsw i64 %45, 7
  %47 = shl nuw nsw i64 %43, 17
  br i1 %26, label %.thread, label %60

.thread:                                          ; preds = %27
  %48 = icmp samesign ugt i32 %.fr, 45
  %49 = urem i64 %47, 5908480
  %50 = select i1 %48, i64 %49, i64 %47
  %51 = lshr exact i64 %50, 10
  %52 = tail call i64 @llvm.usub.sat.i64(i64 %51, i64 5642)
  %53 = trunc i64 %52 to i32
  %54 = sub i32 128, %53
  %55 = getelementptr float, ptr addrspace(1) %0, i64 %47
  tail call void @llvm.xpu.xwait(i32 3, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4)
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0)
  %56 = ptrtoint ptr addrspace(1) %55 to i64
  %57 = lshr i64 %56, 32
  %58 = trunc nuw i64 %57 to i32
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %58, i32 0, i32 2)
  %59 = trunc i64 %56 to i32
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %59, i32 4096, i32 3)
  tail call void @llvm.xpu.dma_cfg(i32 4096, i32 4096, i32 %54, i32 4)
  tail call void @llvm.xpu.dma_run.v3(i32 0)
  tail call void @llvm.xpu.unlock(i32 4)
  tail call void @llvm.xpu.xsignal(i32 2, i32 0)
  br label %.thread8

60:                                               ; preds = %27
  switch i32 %9, label %.thread8 [
    i32 2, label %61
    i32 1, label %62
    i32 7, label %77
  ]

61:                                               ; preds = %60
  tail call void @llvm.xpu.xwait(i32 2, i32 0)
  tail call void @llvm.xpu.xwait(i32 5, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 0)
  tail call void @llvm.xpu.ds_cfg(i32 2, i32 0, i32 0)
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 1)
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 14)
  tail call void @llvm.xpu.ds_cfg(i32 128, i32 0, i32 13)
  tail call void @llvm.xpu.shuffle_batch.v2(ptr null, ptr addrspacecast (ptr addrspace(2) null to ptr), i32 1024, i32 1)
  tail call void @llvm.xpu.unlock(i32 0)
  tail call void @llvm.xpu.xsignal(i32 3, i32 0)
  tail call void @llvm.xpu.xsignal(i32 4, i32 0)
  br label %.thread8

62:                                               ; preds = %60
  %63 = shl nuw nsw i64 %45, 17
  %64 = icmp samesign ugt i32 %39, 7
  %65 = and i64 %63, 281474975662080
  %66 = select i1 %64, i64 %65, i64 0
  %67 = sub nsw i64 %63, %66
  %68 = ashr exact i64 %67, 10
  %69 = tail call i64 @llvm.smin.i64(i64 %68, i64 896)
  %reass.sub = sub nsw i64 %69, %68
  %70 = trunc nsw i64 %reass.sub to i32
  %71 = add nsw i32 %70, 128
  %72 = getelementptr float, ptr addrspace(1) %1, i64 %63
  tail call void @llvm.xpu.xwait(i32 7, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4)
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0)
  %73 = ptrtoint ptr addrspace(1) %72 to i64
  %74 = lshr i64 %73, 32
  %75 = trunc nuw i64 %74 to i32
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %75, i32 0, i32 2)
  %76 = trunc i64 %73 to i32
  tail call void @llvm.xpu.dma_cfg(i32 524288, i32 %76, i32 4096, i32 3)
  tail call void @llvm.xpu.dma_cfg(i32 4096, i32 4096, i32 %71, i32 4)
  tail call void @llvm.xpu.dma_run.v3(i32 0)
  tail call void @llvm.xpu.unlock(i32 4)
  tail call void @llvm.xpu.xsignal(i32 6, i32 0)
  br label %.thread8

77:                                               ; preds = %60
  tail call void @llvm.xpu.xwait(i32 6, i32 0)
  tail call void @llvm.xpu.xwait(i32 9, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 0)
  tail call void @llvm.xpu.ds_cfg(i32 2, i32 0, i32 0)
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 1)
  tail call void @llvm.xpu.ds_cfg(i32 1024, i32 0, i32 14)
  tail call void @llvm.xpu.ds_cfg(i32 128, i32 0, i32 13)
  tail call void @llvm.xpu.shuffle_batch.v2(ptr null, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 524288 to ptr addrspace(2)) to ptr), i32 1024, i32 2)
  tail call void @llvm.xpu.unlock(i32 0)
  tail call void @llvm.xpu.xsignal(i32 7, i32 0)
  tail call void @llvm.xpu.xsignal(i32 8, i32 0)
  br label %.thread8

.thread8:                                         ; preds = %60, %61, %.thread, %62, %77
  %78 = phi i1 [ false, %60 ], [ false, %77 ], [ true, %62 ], [ false, %.thread ], [ false, %61 ]
  %79 = add nuw nsw i64 %47, %46
  %80 = sext i32 %42 to i64
  %81 = tail call i64 @llvm.smin.i64(i64 %80, i64 5642)
  %82 = add nsw i64 %81, 128
  %83 = tail call i64 @llvm.smax.i64(i64 %82, i64 %80)
  %84 = sub nsw i64 %83, %80
  %85 = zext nneg i32 %44 to i64
  %86 = tail call i64 @llvm.umin.i64(i64 %85, i64 896)
  %87 = add nuw nsw i64 %86, 128
  %88 = tail call i64 @llvm.usub.sat.i64(i64 %87, i64 %85)
  %89 = tail call i64 @llvm.smin.i64(i64 %84, i64 128)
  %90 = tail call i64 @llvm.umin.i64(i64 %88, i64 128)
  %91 = trunc i64 %89 to i32
  %92 = trunc nuw nsw i64 %90 to i32
  br i1 %26, label %93, label %100

93:                                               ; preds = %.thread8
  %94 = getelementptr float, ptr addrspace(1) %2, i64 %79
  tail call void @llvm.xpu.xwait(i32 12, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4)
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0)
  %95 = ptrtoint ptr addrspace(1) %94 to i64
  %96 = lshr i64 %95, 32
  %97 = trunc nuw i64 %96 to i32
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %97, i32 0, i32 2)
  %98 = shl nuw nsw i32 %92, 2
  %99 = trunc i64 %95 to i32
  tail call void @llvm.xpu.dma_cfg(i32 1048576, i32 %99, i32 %98, i32 3)
  tail call void @llvm.xpu.dma_cfg(i32 512, i32 4096, i32 %91, i32 4)
  tail call void @llvm.xpu.dma_run.v3(i32 0)
  tail call void @llvm.xpu.unlock(i32 4)
  tail call void @llvm.xpu.xsignal(i32 10, i32 0)
  br label %100

100:                                              ; preds = %93, %.thread8
  br i1 %78, label %.critedge2.thread, label %107

.critedge2.thread:                                ; preds = %100
  %101 = getelementptr float, ptr addrspace(1) %3, i64 %46
  tail call void @llvm.xpu.xwait(i32 14, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 4)
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0)
  %102 = ptrtoint ptr addrspace(1) %101 to i64
  %103 = lshr i64 %102, 32
  %104 = trunc nuw i64 %103 to i32
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 %104, i32 0, i32 2)
  %105 = shl nuw nsw i32 %92, 2
  %106 = trunc i64 %102 to i32
  tail call void @llvm.xpu.dma_cfg(i32 1114112, i32 %106, i32 %105, i32 3)
  tail call void @llvm.xpu.dma_cfg(i32 512, i32 512, i32 1, i32 4)
  tail call void @llvm.xpu.dma_run.v3(i32 0)
  tail call void @llvm.xpu.unlock(i32 4)
  tail call void @llvm.xpu.xsignal(i32 13, i32 0)
  br label %.critedge2.thread14

107:                                              ; preds = %100
  switch i32 %9, label %.critedge2 [
    i32 4, label %.critedge16
    i32 5, label %108
  ]

.critedge16:                                      ; preds = %107
  tail call void @llvm.xpu.xwait(i32 4, i32 0)
  tail call void @llvm.xpu.xwait(i32 8, i32 0)
  tail call void @llvm.xpu.xwait(i32 16, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 1)
  tail call void @llvm.xpu.mm_cfg_int(i32 64, i32 1024, i32 128, i32 0)
  tail call void @llvm.xpu.mm_cfg_int(i32 64, i32 4, i32 1, i32 2)
  tail call void @llvm.xpu.mm(ptr nonnull inttoptr (i32 1114624 to ptr), ptr null, ptr null, i32 8)
  tail call void @llvm.xpu.unlock(i32 1)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 1)
  tail call void @llvm.xpu.mm_cfg_int(i32 64, i32 1024, i32 128, i32 0)
  tail call void @llvm.xpu.mm_cfg_int(i32 64, i32 4, i32 1, i32 2)
  tail call void @llvm.xpu.mm(ptr nonnull inttoptr (i32 1114880 to ptr), ptr null, ptr nonnull inttoptr (i32 262144 to ptr), i32 8)
  tail call void @llvm.xpu.unlock(i32 1)
  tail call void @llvm.xpu.xsignal(i32 5, i32 0)
  tail call void @llvm.xpu.xsignal(i32 9, i32 0)
  tail call void @llvm.xpu.xsignal(i32 15, i32 0)
  br label %.critedge2.thread14

108:                                              ; preds = %107
  tail call void @llvm.xpu.xwait(i32 13, i32 0)
  tail call void @llvm.xpu.xwait(i32 15, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 1)
  tail call void @llvm.xpu.ewcoeff_cfg_scalar(float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, i32 0)
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 127, i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 1, i32 0, i32 0, i32 3)
  br label %109

109:                                              ; preds = %108, %109
  %110 = phi i32 [ 0, %108 ], [ %112, %109 ]
  %.idx = shl nuw nsw i32 %110, 9
  %111 = getelementptr i8, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 1114624 to ptr addrspace(2)) to ptr), i32 %.idx
  tail call void @llvm.xpu.dsmadd(ptr %111, ptr %111, ptr addrspacecast (ptr addrspace(2) inttoptr (i32 1114112 to ptr addrspace(2)) to ptr), i32 0)
  %112 = add nuw nsw i32 %110, 1
  %exitcond.not = icmp eq i32 %112, 128
  br i1 %exitcond.not, label %113, label %109

113:                                              ; preds = %109
  tail call void @llvm.xpu.unlock(i32 2)
  tail call void @llvm.xpu.xsignal(i32 14, i32 0)
  tail call void @llvm.xpu.xwait(i32 10, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 1)
  tail call void @llvm.xpu.ewcoeff_cfg_scalar(float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, i32 0)
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 4095, i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 4, i32 0, i32 0, i32 3)
  tail call void @llvm.xpu.ew_cfg(i32 4, i32 4, i32 4, i32 19)
  tail call void @llvm.xpu.dsmadd(ptr addrspacecast (ptr addrspace(2) inttoptr (i32 1048576 to ptr addrspace(2)) to ptr), ptr addrspacecast (ptr addrspace(2) inttoptr (i32 1048576 to ptr addrspace(2)) to ptr), ptr addrspacecast (ptr addrspace(2) inttoptr (i32 1114624 to ptr addrspace(2)) to ptr), i32 0)
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 19)
  tail call void @llvm.xpu.unlock(i32 2)
  tail call void @llvm.xpu.xsignal(i32 11, i32 0)
  tail call void @llvm.xpu.xsignal(i32 16, i32 0)
  br label %.critedge2.thread14

.critedge2:                                       ; preds = %107
  br i1 %10, label %114, label %.critedge2.thread14

114:                                              ; preds = %.critedge2
  %115 = getelementptr float, ptr addrspace(1) %4, i64 %79
  tail call void @llvm.xpu.xwait(i32 11, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 5)
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.xpu.dma_cfg(i32 0, i32 1, i32 0, i32 1)
  %116 = ptrtoint ptr addrspace(1) %115 to i64
  %117 = lshr i64 %116, 32
  %118 = trunc nuw i64 %117 to i32
  tail call void @llvm.xpu.dma_cfg(i32 %118, i32 0, i32 0, i32 2)
  %119 = shl nuw nsw i32 %92, 2
  %120 = trunc i64 %116 to i32
  tail call void @llvm.xpu.dma_cfg(i32 %120, i32 1048576, i32 %119, i32 3)
  tail call void @llvm.xpu.dma_cfg(i32 4096, i32 512, i32 %91, i32 4)
  tail call void @llvm.xpu.dma_run.v3(i32 0)
  tail call void @llvm.xpu.unlock(i32 5)
  tail call void @llvm.xpu.xsignal(i32 12, i32 0)
  br label %.critedge2.thread14

.critedge2.thread14:                              ; preds = %113, %.critedge16, %.critedge2.thread, %114, %.critedge2
  %121 = add nsw i64 %28, 1
  %122 = icmp slt i64 %121, %23
  br i1 %122, label %27, label %._crit_edge

._crit_edge:                                      ; preds = %.critedge2.thread14, %12
  tail call void @llvm.xpu.xsignal(i32 0, i32 0)
  br i1 %10, label %123, label %124

123:                                              ; preds = %._crit_edge
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.xwait(i32 0, i32 0)
  tail call void @llvm.xpu.lock.v2(i32 0, i32 2)
  tail call void @llvm.xpu.ew_cfg(i32 0, i32 0, i32 0, i32 16)
  tail call void @llvm.xpu.unlock(i32 2)
  tail call void @llvm.xpu.xsignal_mode_cfg(i32 0)
  br label %124

124:                                              ; preds = %123, %._crit_edge
  ret void
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


