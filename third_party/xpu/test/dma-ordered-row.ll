; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-f64:64:64-v512:512-a:0:32-n32-S32"
target triple = "xpu3-baidu-none-gnu"

; Function Attrs: nounwind
define dso_local amdgpu_gfx_whole_wave void @ordered(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7) local_unnamed_addr #0 {
  %9 = tail call i32 @llvm.xpu.load_param(i32 0)
  %10 = tail call i32 @llvm.xpu.load_param(i32 1)
  %11 = mul i32 %6, %5
  %12 = mul i32 %11, %7
  %13 = tail call i32 @llvm.xpu.core_id()
  %14 = getelementptr i8, ptr addrspace(1) %4, i64 11816960
  %15 = icmp slt i32 %9, %12
  br i1 %15, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %8
  %16 = icmp slt i32 %13, 64
  br label %17

17:                                               ; preds = %.lr.ph, %463
  %18 = phi i32 [ %9, %.lr.ph ], [ %464, %463 ]
  %19 = sdiv i32 %18, %7
  %20 = sdiv i32 %19, %6
  %21 = alloca [32 x half], align 2
  %22 = alloca [32 x half], align 2
  %23 = alloca [32 x half], align 2
  %24 = alloca [32 x half], align 2
  %25 = alloca [32 x half], align 2
  %26 = getelementptr inbounds nuw i8, ptr %25, i32 32
  %27 = shl i32 %20, 6
  %28 = add i32 %27, %13
  %29 = shl i32 %28, 10
  %30 = sext i32 %29 to i64
  %31 = getelementptr half, ptr addrspace(1) %0, i64 %30
  %32 = icmp sgt i32 %28, 5769
  %33 = srem i32 %28, 5770
  %34 = shl nsw i32 %33, 10
  %35 = sext i32 %34 to i64
  %36 = getelementptr half, ptr addrspace(1) %1, i64 %35
  br label %37

37:                                               ; preds = %17, %.critedge6
  %38 = phi <16 x float> [ zeroinitializer, %17 ], [ %215, %.critedge6 ]
  %39 = phi <16 x float> [ zeroinitializer, %17 ], [ %216, %.critedge6 ]
  %40 = phi <16 x float> [ zeroinitializer, %17 ], [ %211, %.critedge6 ]
  %41 = phi <16 x float> [ zeroinitializer, %17 ], [ %212, %.critedge6 ]
  %42 = phi i32 [ 0, %17 ], [ %217, %.critedge6 ]
  %43 = shl nuw nsw i32 %42, 6
  %44 = or disjoint i32 %43, 32
  %45 = zext nneg i32 %43 to i64
  %46 = zext nneg i32 %44 to i64
  br i1 %32, label %47, label %.critedge10

47:                                               ; preds = %37
  store i16 0, ptr %25, align 2
  %48 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %48, align 2
  %49 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %49, align 2
  %50 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %50, align 2
  %51 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %51, align 2
  %52 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %52, align 2
  %53 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %53, align 2
  %54 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %54, align 2
  %55 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %55, align 2
  %56 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %56, align 2
  %57 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %57, align 2
  %58 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %58, align 2
  %59 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %59, align 2
  %60 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %60, align 2
  %61 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %61, align 2
  %62 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %62, align 2
  %63 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %63, align 2
  %64 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %64, align 2
  %65 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %65, align 2
  %66 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %66, align 2
  %67 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %67, align 2
  %68 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %68, align 2
  %69 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %69, align 2
  %70 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %70, align 2
  %71 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %71, align 2
  %72 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %72, align 2
  %73 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %73, align 2
  %74 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %74, align 2
  %75 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %75, align 2
  %76 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %76, align 2
  %77 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %77, align 2
  %78 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %78, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %.critedge

.critedge10:                                      ; preds = %37
  %79 = getelementptr half, ptr addrspace(1) %31, i64 %46
  %80 = getelementptr half, ptr addrspace(1) %31, i64 %45
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %80, i32 0, i32 32)
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %26, ptr addrspace(1) %79, i32 0, i32 32)
  br label %.critedge

.critedge:                                        ; preds = %47, %.critedge10
  call void @llvm.xpu.mfence.v2(i32 1)
  %81 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %82 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %81) #3
  %83 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %81) #3
  br i1 %32, label %84, label %.critedge12

84:                                               ; preds = %.critedge
  store i16 0, ptr %25, align 2
  %85 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %85, align 2
  %86 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %86, align 2
  %87 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %87, align 2
  %88 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %88, align 2
  %89 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %89, align 2
  %90 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %90, align 2
  %91 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %91, align 2
  %92 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %92, align 2
  %93 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %93, align 2
  %94 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %94, align 2
  %95 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %95, align 2
  %96 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %96, align 2
  %97 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %97, align 2
  %98 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %98, align 2
  %99 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %99, align 2
  %100 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %100, align 2
  %101 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %101, align 2
  %102 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %102, align 2
  %103 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %103, align 2
  %104 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %104, align 2
  %105 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %105, align 2
  %106 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %106, align 2
  %107 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %107, align 2
  %108 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %108, align 2
  %109 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %109, align 2
  %110 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %110, align 2
  %111 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %111, align 2
  %112 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %112, align 2
  %113 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %113, align 2
  %114 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %114, align 2
  %115 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %115, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %.critedge2

.critedge12:                                      ; preds = %.critedge
  %116 = getelementptr half, ptr addrspace(1) %36, i64 %46
  %117 = getelementptr half, ptr addrspace(1) %36, i64 %45
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %117, i32 0, i32 32)
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %26, ptr addrspace(1) %116, i32 0, i32 32)
  br label %.critedge2

.critedge2:                                       ; preds = %84, %.critedge12
  call void @llvm.xpu.mfence.v2(i32 1)
  %118 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %119 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %118) #3
  %120 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %118) #3
  %121 = fadd <16 x float> %82, %119
  %122 = fadd <16 x float> %83, %120
  %123 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %121, <16 x float> %122) #3
  %124 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %123) #3
  %125 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %123) #3
  %126 = or disjoint i32 %43, 16
  %127 = or disjoint i32 %43, 48
  %128 = zext nneg i32 %126 to i64
  %129 = zext nneg i32 %127 to i64
  br i1 %32, label %130, label %.critedge14

130:                                              ; preds = %.critedge2
  store i16 0, ptr %25, align 2
  %131 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %131, align 2
  %132 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %132, align 2
  %133 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %133, align 2
  %134 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %134, align 2
  %135 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %135, align 2
  %136 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %136, align 2
  %137 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %137, align 2
  %138 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %138, align 2
  %139 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %139, align 2
  %140 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %140, align 2
  %141 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %141, align 2
  %142 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %142, align 2
  %143 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %143, align 2
  %144 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %144, align 2
  %145 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %145, align 2
  %146 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %146, align 2
  %147 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %147, align 2
  %148 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %148, align 2
  %149 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %149, align 2
  %150 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %150, align 2
  %151 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %151, align 2
  %152 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %152, align 2
  %153 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %153, align 2
  %154 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %154, align 2
  %155 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %155, align 2
  %156 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %156, align 2
  %157 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %157, align 2
  %158 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %158, align 2
  %159 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %159, align 2
  %160 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %160, align 2
  %161 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %161, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %.critedge4

.critedge14:                                      ; preds = %.critedge2
  %162 = getelementptr half, ptr addrspace(1) %31, i64 %129
  %163 = getelementptr half, ptr addrspace(1) %31, i64 %128
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %163, i32 0, i32 32)
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %26, ptr addrspace(1) %162, i32 0, i32 32)
  br label %.critedge4

.critedge4:                                       ; preds = %130, %.critedge14
  call void @llvm.xpu.mfence.v2(i32 1)
  %164 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %165 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %164) #3
  %166 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %164) #3
  br i1 %32, label %167, label %.critedge16

167:                                              ; preds = %.critedge4
  store i16 0, ptr %25, align 2
  %168 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %168, align 2
  %169 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %169, align 2
  %170 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %170, align 2
  %171 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %171, align 2
  %172 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %172, align 2
  %173 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %173, align 2
  %174 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %174, align 2
  %175 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %175, align 2
  %176 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %176, align 2
  %177 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %177, align 2
  %178 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %178, align 2
  %179 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %179, align 2
  %180 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %180, align 2
  %181 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %181, align 2
  %182 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %182, align 2
  %183 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %183, align 2
  %184 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %184, align 2
  %185 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %185, align 2
  %186 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %186, align 2
  %187 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %187, align 2
  %188 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %188, align 2
  %189 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %189, align 2
  %190 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %190, align 2
  %191 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %191, align 2
  %192 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %192, align 2
  %193 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %193, align 2
  %194 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %194, align 2
  %195 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %195, align 2
  %196 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %196, align 2
  %197 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %197, align 2
  %198 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %198, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %.critedge6

.critedge16:                                      ; preds = %.critedge4
  %199 = getelementptr half, ptr addrspace(1) %36, i64 %129
  %200 = getelementptr half, ptr addrspace(1) %36, i64 %128
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %200, i32 0, i32 32)
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %26, ptr addrspace(1) %199, i32 0, i32 32)
  br label %.critedge6

.critedge6:                                       ; preds = %167, %.critedge16
  call void @llvm.xpu.mfence.v2(i32 1)
  %201 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %202 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %201) #3
  %203 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %201) #3
  %204 = fadd <16 x float> %165, %202
  %205 = fadd <16 x float> %166, %203
  %206 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %204, <16 x float> %205) #3
  %207 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %206) #3
  %208 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %206) #3
  %209 = fadd <16 x float> %40, %124
  %210 = fadd <16 x float> %41, %125
  %211 = fadd <16 x float> %209, %207
  %212 = fadd <16 x float> %210, %208
  %213 = call <16 x float> asm "vmac.f.mz.rn $0{mr1}, $1, $2", "=v,v,v,0"(<16 x float> %124, <16 x float> %124, <16 x float> %38) #3
  %214 = call <16 x float> asm "vmac.f.mz.rn $0{mr1}, $1, $2", "=v,v,v,0"(<16 x float> %125, <16 x float> %125, <16 x float> %39) #3
  %215 = call <16 x float> asm "vmac.f.mz.rn $0{mr1}, $1, $2", "=v,v,v,0"(<16 x float> %207, <16 x float> %207, <16 x float> %213) #3
  %216 = call <16 x float> asm "vmac.f.mz.rn $0{mr1}, $1, $2", "=v,v,v,0"(<16 x float> %208, <16 x float> %208, <16 x float> %214) #3
  %217 = add nuw nsw i32 %42, 1
  %exitcond.not = icmp eq i32 %217, 16
  br i1 %exitcond.not, label %218, label %37

218:                                              ; preds = %.critedge6
  %219 = call i32 @llvm.xpu.load_param(i32 2)
  %220 = fadd <16 x float> %211, %212
  %221 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 256, <16 x float> %220) #3
  %222 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %220, <16 x float> %221) #3
  %223 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 128, <16 x float> %222) #3
  %224 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %222, <16 x float> %223) #3
  %225 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 64, <16 x float> %224) #3
  %226 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %224, <16 x float> %225) #3
  %227 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 32, <16 x float> %226) #3
  %228 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %226, <16 x float> %227) #3
  %229 = extractelement <16 x float> %228, i64 0
  %230 = fmul float %229, 0x3F50000000000000
  %231 = fadd <16 x float> %215, %216
  %232 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 256, <16 x float> %231) #3
  %233 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %231, <16 x float> %232) #3
  %234 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 128, <16 x float> %233) #3
  %235 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %233, <16 x float> %234) #3
  %236 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 64, <16 x float> %235) #3
  %237 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %235, <16 x float> %236) #3
  %238 = call <16 x float> asm "vsrl.p $0, $1, $2", "=v,r,v"(i32 32, <16 x float> %237) #3
  %239 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,v,v"(<16 x float> %237, <16 x float> %238) #3
  %240 = extractelement <16 x float> %239, i64 0
  %241 = fmul float %240, 0x3F50000000000000
  %242 = fmul float %230, %230
  %243 = fsub float %241, %242
  %244 = fadd float %243, 0x3EE4F8B580000000
  %245 = call float @llvm.sqrt.f32(float %244)
  %246 = fdiv float 1.000000e+00, %245
  %247 = getelementptr half, ptr addrspace(1) %4, i64 %30
  %248 = getelementptr half, ptr addrspace(1) %14, i64 %30
  %249 = fneg float %230
  br label %250

250:                                              ; preds = %218, %461
  %251 = phi i32 [ 0, %218 ], [ %462, %461 ]
  %252 = shl nuw nsw i32 %251, 6
  %253 = zext nneg i32 %252 to i64
  br i1 %32, label %254, label %.critedge18

254:                                              ; preds = %250
  store i16 0, ptr %25, align 2
  %255 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %255, align 2
  %256 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %256, align 2
  %257 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %257, align 2
  %258 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %258, align 2
  %259 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %259, align 2
  %260 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %260, align 2
  %261 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %261, align 2
  %262 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %262, align 2
  %263 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %263, align 2
  %264 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %264, align 2
  %265 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %265, align 2
  %266 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %266, align 2
  %267 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %267, align 2
  %268 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %268, align 2
  %269 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %269, align 2
  %270 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %270, align 2
  %271 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %271, align 2
  %272 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %272, align 2
  %273 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %273, align 2
  %274 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %274, align 2
  %275 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %275, align 2
  %276 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %276, align 2
  %277 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %277, align 2
  %278 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %278, align 2
  %279 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %279, align 2
  %280 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %280, align 2
  %281 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %281, align 2
  %282 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %282, align 2
  %283 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %283, align 2
  %284 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %284, align 2
  %285 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %285, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %287

.critedge18:                                      ; preds = %250
  %286 = getelementptr half, ptr addrspace(1) %31, i64 %253
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %286, i32 0, i32 64)
  br label %287

287:                                              ; preds = %254, %.critedge18
  call void @llvm.xpu.mfence.v2(i32 1)
  %288 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %289 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %288) #3
  %290 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %288) #3
  br i1 %32, label %291, label %.critedge20

291:                                              ; preds = %287
  store i16 0, ptr %25, align 2
  %292 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %292, align 2
  %293 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %293, align 2
  %294 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %294, align 2
  %295 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %295, align 2
  %296 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %296, align 2
  %297 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %297, align 2
  %298 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %298, align 2
  %299 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %299, align 2
  %300 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %300, align 2
  %301 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %301, align 2
  %302 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %302, align 2
  %303 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %303, align 2
  %304 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %304, align 2
  %305 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %305, align 2
  %306 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %306, align 2
  %307 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %307, align 2
  %308 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %308, align 2
  %309 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %309, align 2
  %310 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %310, align 2
  %311 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %311, align 2
  %312 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %312, align 2
  %313 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %313, align 2
  %314 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %314, align 2
  %315 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %315, align 2
  %316 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %316, align 2
  %317 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %317, align 2
  %318 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %318, align 2
  %319 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %319, align 2
  %320 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %320, align 2
  %321 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %321, align 2
  %322 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %322, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %324

.critedge20:                                      ; preds = %287
  %323 = getelementptr half, ptr addrspace(1) %36, i64 %253
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %323, i32 0, i32 64)
  br label %324

324:                                              ; preds = %291, %.critedge20
  call void @llvm.xpu.mfence.v2(i32 1)
  %325 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %326 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %325) #3
  %327 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %325) #3
  %328 = fadd <16 x float> %289, %326
  %329 = fadd <16 x float> %290, %327
  %330 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %328, <16 x float> %329) #3
  %331 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %330) #3
  %332 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %330) #3
  %333 = getelementptr half, ptr addrspace(1) %2, i64 %253
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %24, ptr addrspace(1) %333, i32 0, i32 64)
  %334 = getelementptr half, ptr addrspace(1) %3, i64 %253
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %23, ptr addrspace(1) %334, i32 0, i32 64)
  %335 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %249, <16 x float> %331) #3
  %336 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %249, <16 x float> %332) #3
  %337 = call <16 x float> asm "vmul.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %246, <16 x float> %335) #3
  %338 = call <16 x float> asm "vmul.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %246, <16 x float> %336) #3
  call void @llvm.xpu.mfence.v2(i32 1)
  %339 = load <32 x half>, ptr %24, align 64
  %340 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %339) #3
  %341 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %339) #3
  %342 = load <32 x half>, ptr %23, align 64
  %343 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %342) #3
  %344 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %342) #3
  %345 = call <16 x float> asm "vmac.f.mz.rn $0{mr1}, $1, $2", "=v,v,v,0"(<16 x float> %337, <16 x float> %340, <16 x float> %343) #3
  %346 = call <16 x float> asm "vmac.f.mz.rn $0{mr1}, $1, $2", "=v,v,v,0"(<16 x float> %338, <16 x float> %341, <16 x float> %344) #3
  store <32 x half> %330, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  br i1 %32, label %349, label %347

347:                                              ; preds = %324
  %348 = getelementptr half, ptr addrspace(1) %247, i64 %253
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %348, ptr nonnull %25, i32 0, i32 64)
  br label %349

349:                                              ; preds = %347, %324
  call void @llvm.xpu.mfence.v2(i32 1)
  %350 = getelementptr half, ptr addrspace(1) %248, i64 %253
  %351 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %345, <16 x float> %346) #3
  br i1 %16, label %352, label %355

352:                                              ; preds = %349
  store <32 x half> %351, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  br i1 %32, label %354, label %353

353:                                              ; preds = %352
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %350, ptr nonnull %25, i32 0, i32 64)
  br label %354

354:                                              ; preds = %353, %352
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %355

355:                                              ; preds = %354, %349
  %356 = or disjoint i32 %252, 32
  %357 = zext nneg i32 %356 to i64
  br i1 %32, label %358, label %.critedge22

358:                                              ; preds = %355
  store i16 0, ptr %25, align 2
  %359 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %359, align 2
  %360 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %360, align 2
  %361 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %361, align 2
  %362 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %362, align 2
  %363 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %363, align 2
  %364 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %364, align 2
  %365 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %365, align 2
  %366 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %366, align 2
  %367 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %367, align 2
  %368 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %368, align 2
  %369 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %369, align 2
  %370 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %370, align 2
  %371 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %371, align 2
  %372 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %372, align 2
  %373 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %373, align 2
  %374 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %374, align 2
  %375 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %375, align 2
  %376 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %376, align 2
  %377 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %377, align 2
  %378 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %378, align 2
  %379 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %379, align 2
  %380 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %380, align 2
  %381 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %381, align 2
  %382 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %382, align 2
  %383 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %383, align 2
  %384 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %384, align 2
  %385 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %385, align 2
  %386 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %386, align 2
  %387 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %387, align 2
  %388 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %388, align 2
  %389 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %389, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %391

.critedge22:                                      ; preds = %355
  %390 = getelementptr half, ptr addrspace(1) %31, i64 %357
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %390, i32 0, i32 64)
  br label %391

391:                                              ; preds = %358, %.critedge22
  call void @llvm.xpu.mfence.v2(i32 1)
  %392 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %393 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %392) #3
  %394 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %392) #3
  br i1 %32, label %395, label %.critedge24

395:                                              ; preds = %391
  store i16 0, ptr %25, align 2
  %396 = getelementptr inbounds i16, ptr %25, i32 1
  store i16 0, ptr %396, align 2
  %397 = getelementptr inbounds i16, ptr %25, i32 2
  store i16 0, ptr %397, align 2
  %398 = getelementptr inbounds i16, ptr %25, i32 3
  store i16 0, ptr %398, align 2
  %399 = getelementptr inbounds i16, ptr %25, i32 4
  store i16 0, ptr %399, align 2
  %400 = getelementptr inbounds i16, ptr %25, i32 5
  store i16 0, ptr %400, align 2
  %401 = getelementptr inbounds i16, ptr %25, i32 6
  store i16 0, ptr %401, align 2
  %402 = getelementptr inbounds i16, ptr %25, i32 7
  store i16 0, ptr %402, align 2
  %403 = getelementptr inbounds i16, ptr %25, i32 8
  store i16 0, ptr %403, align 2
  %404 = getelementptr inbounds i16, ptr %25, i32 9
  store i16 0, ptr %404, align 2
  %405 = getelementptr inbounds i16, ptr %25, i32 10
  store i16 0, ptr %405, align 2
  %406 = getelementptr inbounds i16, ptr %25, i32 11
  store i16 0, ptr %406, align 2
  %407 = getelementptr inbounds i16, ptr %25, i32 12
  store i16 0, ptr %407, align 2
  %408 = getelementptr inbounds i16, ptr %25, i32 13
  store i16 0, ptr %408, align 2
  %409 = getelementptr inbounds i16, ptr %25, i32 14
  store i16 0, ptr %409, align 2
  %410 = getelementptr inbounds i16, ptr %25, i32 15
  store i16 0, ptr %410, align 2
  %411 = getelementptr inbounds i16, ptr %25, i32 16
  store i16 0, ptr %411, align 2
  %412 = getelementptr inbounds i16, ptr %25, i32 17
  store i16 0, ptr %412, align 2
  %413 = getelementptr inbounds i16, ptr %25, i32 18
  store i16 0, ptr %413, align 2
  %414 = getelementptr inbounds i16, ptr %25, i32 19
  store i16 0, ptr %414, align 2
  %415 = getelementptr inbounds i16, ptr %25, i32 20
  store i16 0, ptr %415, align 2
  %416 = getelementptr inbounds i16, ptr %25, i32 21
  store i16 0, ptr %416, align 2
  %417 = getelementptr inbounds i16, ptr %25, i32 22
  store i16 0, ptr %417, align 2
  %418 = getelementptr inbounds i16, ptr %25, i32 23
  store i16 0, ptr %418, align 2
  %419 = getelementptr inbounds i16, ptr %25, i32 24
  store i16 0, ptr %419, align 2
  %420 = getelementptr inbounds i16, ptr %25, i32 25
  store i16 0, ptr %420, align 2
  %421 = getelementptr inbounds i16, ptr %25, i32 26
  store i16 0, ptr %421, align 2
  %422 = getelementptr inbounds i16, ptr %25, i32 27
  store i16 0, ptr %422, align 2
  %423 = getelementptr inbounds i16, ptr %25, i32 28
  store i16 0, ptr %423, align 2
  %424 = getelementptr inbounds i16, ptr %25, i32 29
  store i16 0, ptr %424, align 2
  %425 = getelementptr inbounds i16, ptr %25, i32 30
  store i16 0, ptr %425, align 2
  %426 = getelementptr inbounds i16, ptr %25, i32 31
  store i16 0, ptr %426, align 2
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %428

.critedge24:                                      ; preds = %391
  %427 = getelementptr half, ptr addrspace(1) %36, i64 %357
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %25, ptr addrspace(1) %427, i32 0, i32 64)
  br label %428

428:                                              ; preds = %395, %.critedge24
  call void @llvm.xpu.mfence.v2(i32 1)
  %429 = load <32 x half>, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  %430 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %429) #3
  %431 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %429) #3
  %432 = fadd <16 x float> %393, %430
  %433 = fadd <16 x float> %394, %431
  %434 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %432, <16 x float> %433) #3
  %435 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %434) #3
  %436 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %434) #3
  %437 = getelementptr half, ptr addrspace(1) %2, i64 %357
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %22, ptr addrspace(1) %437, i32 0, i32 64)
  %438 = getelementptr half, ptr addrspace(1) %3, i64 %357
  call void @llvm.xpu.gm2lm_v3(ptr nonnull %21, ptr addrspace(1) %438, i32 0, i32 64)
  %439 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %249, <16 x float> %435) #3
  %440 = call <16 x float> asm "vadd.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %249, <16 x float> %436) #3
  %441 = call <16 x float> asm "vmul.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %246, <16 x float> %439) #3
  %442 = call <16 x float> asm "vmul.f.mz.rn $0{mr1}, $1, $2", "=v,r,v"(float %246, <16 x float> %440) #3
  call void @llvm.xpu.mfence.v2(i32 1)
  %443 = load <32 x half>, ptr %22, align 64
  %444 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %443) #3
  %445 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %443) #3
  %446 = fmul <16 x float> %441, %444
  %447 = fmul <16 x float> %442, %445
  %448 = load <32 x half>, ptr %21, align 64
  %449 = call <16 x float> asm "vfp162float_l.rn $0, $1", "=&v,v"(<32 x half> %448) #3
  %450 = call <16 x float> asm "vfp162float_h.rn $0, $1", "=&v,v"(<32 x half> %448) #3
  %451 = fadd <16 x float> %446, %449
  %452 = fadd <16 x float> %447, %450
  store <32 x half> %434, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  br i1 %32, label %455, label %453

453:                                              ; preds = %428
  %454 = getelementptr half, ptr addrspace(1) %247, i64 %357
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %454, ptr nonnull %25, i32 0, i32 64)
  br label %455

455:                                              ; preds = %453, %428
  call void @llvm.xpu.mfence.v2(i32 1)
  %456 = getelementptr half, ptr addrspace(1) %248, i64 %357
  %457 = call <32 x half> asm "vfloat2fp16_l.rn $0, $1\0Avfloat2fp16_h.rn $0, $2", "=&v,v,v"(<16 x float> %451, <16 x float> %452) #3
  br i1 %16, label %458, label %461

458:                                              ; preds = %455
  store <32 x half> %457, ptr %25, align 64
  call void @llvm.xpu.mfence.v2(i32 1)
  br i1 %32, label %460, label %459

459:                                              ; preds = %458
  call void @llvm.xpu.lm2gm_v3(ptr addrspace(1) %456, ptr nonnull %25, i32 0, i32 64)
  br label %460

460:                                              ; preds = %459, %458
  call void @llvm.xpu.mfence.v2(i32 1)
  br label %461

461:                                              ; preds = %460, %455
  %462 = add nuw nsw i32 %251, 1
  %exitcond29.not = icmp eq i32 %462, 16
  br i1 %exitcond29.not, label %463, label %250

463:                                              ; preds = %461
  %464 = add i32 %18, %10
  %465 = icmp slt i32 %464, %12
  br i1 %465, label %17, label %._crit_edge

._crit_edge:                                      ; preds = %463, %8
  ret void
}

; Function Attrs: nofree nosync nounwind memory(none)
; Unknown intrinsic
declare i32 @llvm.xpu.load_param(i32) #1

; Function Attrs: nofree nosync nounwind memory(none)
; Unknown intrinsic
declare i32 @llvm.xpu.core_id() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #2

; Function Attrs: nounwind
; Unknown intrinsic
declare void @llvm.xpu.mfence.v2(i32) #3

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.gm2lm_v3(ptr, ptr addrspace(1), i32, i32) #4

; Function Attrs: nounwind memory(argmem: readwrite)
; Unknown intrinsic
declare void @llvm.xpu.lm2gm_v3(ptr addrspace(1), ptr, i32, i32) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="xpu3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }
attributes #4 = { nounwind memory(argmem: readwrite) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }


