#map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @triton_(%arg0: memref<*xbf16, 1> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16, 1> {tt.divisibility = 16 : i32}, %arg2: memref<*xbf16, 1> {tt.divisibility = 16 : i32}, %arg3: memref<*xbf16, 1> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32767_i32 = arith.constant 32767 : i32
    %c-65536_i32 = arith.constant -65536 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c2139095040_i32 = arith.constant 2139095040 : i32
    %c32768 = arith.constant 32768 : index
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c16384 = arith.constant 16384 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4194304_i32 = arith.constant 4194304 : i32
    %cst = arith.constant 1.000000e+00 : f32
    "sdnn.init"() {signal = [array<i32: 14, 5, 3, 9, 7, 19, 17>], signal_num = array<i64: 1, 255, 255, 255, 255, 1, 1>} : () -> ()
    %0 = sdnn.get_sdnn_id : i32
    %1 = sdnn.get_num_sdnn : i32
    %2 = arith.muli %arg4, %arg5 : i32
    %3 = arith.muli %2, %arg6 : i32
    %4 = arith.subi %1, %c1_i32 : i32
    %5 = arith.addi %3, %4 : i32
    %6 = arith.divsi %5, %1 : i32
    %7 = arith.muli %0, %6 : i32
    %8 = arith.addi %7, %6 : i32
    %9 = arith.minsi %8, %3 : i32
    %10 = arith.index_cast %7 : i32 to index
    %11 = arith.index_cast %9 : i32 to index
    %12 = sdnn.counter {bufferNum = 255 : i32} : memref<1xi32, 6>
    %13 = sdnn.counter {bufferNum = 255 : i32} : memref<1xi32, 6>
    %14 = sdnn.counter {bufferNum = 255 : i32} : memref<1xi32, 6>
    %15 = sdnn.counter {bufferNum = 255 : i32} : memref<1xi32, 6>
    scf.for %arg7 = %10 to %11 step %c1 {
      %16 = arith.index_cast %arg7 : index to i32
      %17 = arith.divsi %16, %arg6 : i32
      %18 = arith.divsi %17, %arg5 : i32
      %19 = arith.divsi %18, %c256_i32 : i32
      %20 = arith.muli %19, %c8_i32 : i32
      %21 = arith.subi %c2_i32, %20 : i32
      %22 = arith.minsi %21, %c8_i32 : i32
      %23 = arith.remsi %18, %22 : i32
      %24 = arith.addi %20, %23 : i32
      %25 = arith.remsi %18, %c256_i32 : i32
      %26 = arith.divsi %25, %22 : i32
      %27 = arith.muli %24, %c32_i32 : i32
      %28 = arith.index_cast %24 : i32 to index
      %29 = arith.muli %26, %c32_i32 : i32
      %30 = arith.index_cast %26 : i32 to index
      %31 = arith.muli %30, %c32 : index
      %32 = arith.muli %28, %c2048 : index
      %alloca = memref.alloca() : memref<1xi32>
      memref.store %c0_i32, %alloca[%c0] : memref<1xi32>
      %alloc = memref.alloc() {bufferNum = 1 : index, bufferSize = 4096 : index, init_signal = 14 : i64, sram = "uni_sram"} : memref<32x32xf32, #map>
      sdnn.core_wait(4, 14) {core_name = "mma"}
      %33:2 = scf.for %arg8 = %c0_i32 to %c64_i32 step %c16_i32 iter_args(%arg9 = %32, %arg10 = %c0) -> (index, index)  : i32 {
        %48 = arith.addi %arg10, %31 : index
        %49 = arith.remsi %48, %c1024 : index
        %50 = arith.addi %49, %c32 : index
        %51 = arith.minsi %50, %c1024 : index
        %52 = arith.subi %51, %49 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%48], sizes: [16, %52], strides: [1024, 1] : memref<*xbf16, 1> to memref<16x?xbf16, strided<[1024, 1], offset: ?>, 1>
        %cast_7 = memref.cast %reinterpret_cast_6 : memref<16x?xbf16, strided<[1024, 1], offset: ?>, 1> to memref<?x?xbf16, strided<[?, ?], offset: ?>, 1>
        %53 = arith.cmpi sge, %arg9, %c4096 : index
        %54 = arith.divsi %arg9, %c4096 : index
        %55 = arith.muli %54, %c4096 : index
        %56 = arith.subi %arg9, %55 : index
        %57 = arith.select %53, %56, %arg9 : index
        %58 = arith.divsi %57, %c64 : index
        %59 = arith.addi %58, %c32 : index
        %60 = arith.minsi %59, %c64 : index
        %61 = arith.subi %60, %58 : index
        %reinterpret_cast_8 = memref.reinterpret_cast %arg0 to offset: [%arg9], sizes: [%61, 16], strides: [64, 1] : memref<*xbf16, 1> to memref<?x16xbf16, strided<[64, 1], offset: ?>, 1>
        %cast_9 = memref.cast %reinterpret_cast_8 : memref<?x16xbf16, strided<[64, 1], offset: ?>, 1> to memref<?x?xbf16, strided<[?, ?], offset: ?>, 1>
        %alloc_10 = memref.alloc() {bufferNum = 255 : index, bufferSize = 2048 : index, init_signal = 5 : i64, processed, sram = "l1d"} : memref<32x16xf32, #map1, 3>
        %62 = sdnn.get_fifo_buffer %alloc_10, %15, {bufferNum = 255 : i32, bufferSize = 2048 : i32} : memref<32x16xf32, #map1, 3> memref<1xi32, 6> to memref<32x16xf32, #map1, 3>
        %alloc_11 = memref.alloc() {bufferNum = 255 : index, bufferSize = 4096 : index, init_signal = 3 : i64, processed, sram = "uni_sram", trans = false} : memref<32x16xf32, #map>
        %63 = sdnn.get_fifo_buffer %alloc_11, %14, {bufferNum = 255 : i32, bufferSize = 4096 : i32} : memref<32x16xf32, #map> memref<1xi32, 6> to memref<32x16xf32, #map>
        %dim = memref.dim %reinterpret_cast_8, %c0 : memref<?x16xbf16, strided<[64, 1], offset: ?>, 1>
        %subview_12 = memref.subview %63[0, 0] [%dim, 16] [1, 1] : memref<32x16xf32, #map> to memref<?x16xf32, strided<[32, 1]>>
        %cast_13 = memref.cast %subview_12 : memref<?x16xf32, strided<[32, 1]>> to memref<?x?xf32, strided<[32, 1]>>
        sdnn.dma in %cast_9, %cast_13 {core_id = 1 : i32, signal = array<i32: 2>, wait = array<i32: 3>} : memref<?x?xbf16, strided<[?, ?], offset: ?>, 1> memref<?x?xf32, strided<[32, 1]>> ->
        sdnn.ds %63, %62 mask_dims[%c32, %c16] {sdnn.layout = "CR_RowMajor", signal = array<i32: 3, 4>, signal_cached = array<i32: 4>, wait = array<i32: 2, 5>, wait_cached = array<i32: 5>} : memref<32x16xf32, #map> memref<32x16xf32, #map1, 3> ->
        %alloc_14 = memref.alloc() {bufferNum = 255 : index, bufferSize = 2048 : index, init_signal = 9 : i64, processed, sram = "l1w"} : memref<16x32xf32, #map, 4>
        %64 = sdnn.get_fifo_buffer %alloc_14, %13, {bufferNum = 255 : i32, bufferSize = 2048 : i32} : memref<16x32xf32, #map, 4> memref<1xi32, 6> to memref<16x32xf32, #map, 4>
        %alloc_15 = memref.alloc() {bufferNum = 255 : index, bufferSize = 2048 : index, init_signal = 7 : i64, processed, sram = "uni_sram", trans = false} : memref<16x32xf32, #map>
        %65 = sdnn.get_fifo_buffer %alloc_15, %12, {bufferNum = 255 : i32, bufferSize = 2048 : i32} : memref<16x32xf32, #map> memref<1xi32, 6> to memref<16x32xf32, #map>
        %dim_16 = memref.dim %reinterpret_cast_6, %c1 : memref<16x?xbf16, strided<[1024, 1], offset: ?>, 1>
        %subview_17 = memref.subview %65[0, 0] [16, %dim_16] [1, 1] : memref<16x32xf32, #map> to memref<16x?xf32, strided<[32, 1]>>
        %cast_18 = memref.cast %subview_17 : memref<16x?xf32, strided<[32, 1]>> to memref<?x?xf32, strided<[32, 1]>>
        sdnn.dma in %cast_7, %cast_18 {core_id = 6 : i32, signal = array<i32: 6>, wait = array<i32: 7>} : memref<?x?xbf16, strided<[?, ?], offset: ?>, 1> memref<?x?xf32, strided<[32, 1]>> ->
        sdnn.ds %65, %64 mask_dims[%c16, %c32] {sdnn.layout = "CR_RowMajor", signal = array<i32: 7, 8>, signal_cached = array<i32: 8>, wait = array<i32: 6, 9>, wait_cached = array<i32: 9>} : memref<16x32xf32, #map> memref<16x32xf32, #map, 4> ->
        %66 = memref.load %alloca[%c0] : memref<1xi32>
        sdnn.mma %62, %64, %alloc acc %66 mask_dims[%c32, %c32, %c16] {indexing_maps = [#map2, #map3, #map4], rcOut = false, sdnn.layout = "CR_RowMajor", signal = array<i32: 5, 9>, wait = array<i32: 4, 8>} : (memref<32x16xf32, #map1, 3>, memref<16x32xf32, #map, 4>, memref<32x32xf32, #map>, i32, index, index, index) -> ()
        %67 = arith.addi %66, %c1_i32 : i32
        memref.store %67, %alloca[%c0] : memref<1xi32>
        %68 = arith.addi %arg9, %c16 : index
        %69 = arith.addi %arg10, %c16384 : index
        scf.yield %68, %69 : index, index
      } {mac_acc = true, sdnn.arg_layout.3 = "CR_RowMajor", sdnn.layout.2 = "CR_RowMajor"}
      sdnn.core_arrive(4, 10) {core_name = "mma"}
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 32], strides: [32, 1] : memref<*xbf16, 1> to memref<1x32xbf16, strided<[32, 1], offset: ?>, 1>
      %34 = arith.index_cast %27 : i32 to index
      %35 = arith.addi %34, %c32 : index
      %36 = arith.minsi %35, %c64 : index
      %37 = arith.maxsi %36, %34 : index
      %38 = arith.subi %37, %34 : index
      %39 = arith.index_cast %29 : i32 to index
      %40 = arith.addi %39, %c32 : index
      %41 = arith.minsi %40, %c1024 : index
      %42 = arith.maxsi %41, %39 : index
      %43 = arith.subi %42, %39 : index
      %44 = arith.minsi %43, %c32 : index
      %alloc_0 = memref.alloc() {bufferNum = 1 : index, bufferSize = 128 : index, init_signal = 19 : i64, sram = "uni_sram", trans = false} : memref<1x32xf32, #map>
      %subview = memref.subview %reinterpret_cast[0, 0] [1, %44] [1, 1] : memref<1x32xbf16, strided<[32, 1], offset: ?>, 1> to memref<1x?xbf16, strided<[32, 1], offset: ?>, 1>
      %alloc_1 = memref.alloc() {bufferNum = 1 : index, bufferSize = 4096 : index, init_signal = 17 : i64, sram = "uni_sram"} : memref<32x32xf32, #map>
      sdnn.core_specialize ins(%alloc, %c2147483647_i32, %c4194304_i32, %c2139095040_i32, %c32767_i32, %c16_i32, %c1_i32, %c-65536_i32 : memref<32x32xf32, #map>, i32, i32, i32, i32, i32, i32, i32) outs(%alloc_1 : memref<32x32xf32, #map>) attributes {little_core, sdnn.layout = "CR_RowMajor", signal = array<i32: 11, 15>, wait = array<i32: 10, 17>} {
      ^bb0(%arg8: memref<32x32xf32, #map>, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<32x32xf32, #map>):
        %c0_6 = arith.constant 0 : index
        %c32_7 = arith.constant 32 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg17 = %c0_6 to %c32_7 step %c1_8 {
          scf.for %arg18 = %c0_6 to %c32_7 step %c1_8 {
            %48 = memref.load %arg8[%arg17, %arg18] : memref<32x32xf32, #map>
            %49 = arith.bitcast %48 : f32 to i32
            %50 = arith.andi %49, %arg9 : i32
            %51 = arith.ori %49, %arg10 : i32
            %52 = arith.cmpi ugt, %50, %arg11 : i32
            %53 = arith.select %52, %51, %49 : i32
            %54 = arith.addi %49, %arg12 : i32
            %55 = arith.shrui %49, %arg13 : i32
            %56 = arith.andi %55, %arg14 : i32
            %57 = arith.addi %54, %56 : i32
            %58 = arith.andi %57, %arg15 : i32
            %59 = arith.cmpi uge, %50, %arg11 : i32
            %60 = arith.select %59, %53, %58 : i32
            %61 = arith.bitcast %60 : i32 to f32
            memref.store %61, %arg16[%arg17, %arg18] : memref<32x32xf32, #map>
          }
        }
        sdnn.core_specialize_return
      }
      %subview_2 = memref.subview %alloc_0[0, 0] [1, %44] [1, 1] : memref<1x32xf32, #map> to memref<1x?xf32, strided<[32, 1]>>
      sdnn.dma in %subview, %subview_2 {core_id = 1 : i32, signal = array<i32: 18>, wait = array<i32: 19>} : memref<1x?xbf16, strided<[32, 1], offset: ?>, 1> memref<1x?xf32, strided<[32, 1]>> ->
      sdnn.ew linear %alloc_1 src0 %alloc_1 src1 %alloc_0 a %cst b %cst {sdnn.layout = "CR_RowMajor", signal = array<i32: 16, 19>, wait = array<i32: 15, 18>} : (memref<32x32xf32, #map>, memref<32x32xf32, #map>, memref<1x32xf32, #map>, f32, f32) -> ()
      sdnn.core_specialize ins(%alloc_1, %c2147483647_i32, %c4194304_i32, %c2139095040_i32, %c32767_i32, %c16_i32, %c1_i32, %c-65536_i32 : memref<32x32xf32, #map>, i32, i32, i32, i32, i32, i32, i32) outs(%alloc : memref<32x32xf32, #map>) attributes {little_core, sdnn.layout = "CR_RowMajor", signal = array<i32: 12, 17>, wait = array<i32: 11, 16>} {
      ^bb0(%arg8: memref<32x32xf32, #map>, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<32x32xf32, #map>):
        %c0_6 = arith.constant 0 : index
        %c32_7 = arith.constant 32 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg17 = %c0_6 to %c32_7 step %c1_8 {
          scf.for %arg18 = %c0_6 to %c32_7 step %c1_8 {
            %48 = memref.load %arg8[%arg17, %arg18] : memref<32x32xf32, #map>
            %49 = arith.bitcast %48 : f32 to i32
            %50 = arith.andi %49, %arg9 : i32
            %51 = arith.ori %49, %arg10 : i32
            %52 = arith.cmpi ugt, %50, %arg11 : i32
            %53 = arith.select %52, %51, %49 : i32
            %54 = arith.addi %49, %arg12 : i32
            %55 = arith.shrui %49, %arg13 : i32
            %56 = arith.andi %55, %arg14 : i32
            %57 = arith.addi %54, %56 : i32
            %58 = arith.andi %57, %arg15 : i32
            %59 = arith.cmpi uge, %50, %arg11 : i32
            %60 = arith.select %59, %53, %58 : i32
            %61 = arith.bitcast %60 : i32 to f32
            memref.store %61, %arg16[%arg17, %arg18] : memref<32x32xf32, #map>
          }
        }
        sdnn.core_specialize_return
      }
      sdnn.ew linear %alloc src0 %alloc a %cst {activation = 1 : i32, sdnn.layout = "CR_RowMajor", signal = array<i32: 13>, wait = array<i32: 12>} : (memref<32x32xf32, #map>, memref<32x32xf32, #map>, f32) -> ()
      %45 = arith.muli %28, %c32768 : index
      %46 = arith.addi %45, %31 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%46], sizes: [32, 32], strides: [1024, 1] : memref<*xbf16, 1> to memref<32x32xbf16, strided<[1024, 1], offset: ?>, 1>
      %47 = arith.minsi %38, %c32 : index
      %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%47, %44] [1, 1] : memref<32x32xbf16, strided<[1024, 1], offset: ?>, 1> to memref<?x?xbf16, strided<[1024, 1], offset: ?>, 1>
      %cast = memref.cast %subview_4 : memref<?x?xbf16, strided<[1024, 1], offset: ?>, 1> to memref<?x?xbf16, strided<[?, 1], offset: ?>, 1>
      %subview_5 = memref.subview %alloc[0, 0] [%47, %44] [1, 1] : memref<32x32xf32, #map> to memref<?x?xf32, strided<[32, 1]>>
      sdnn.dma out %subview_5, %cast {signal = array<i32: 14>, wait = array<i32: 13>} : memref<?x?xf32, strided<[32, 1]>> memref<?x?xbf16, strided<[?, 1], offset: ?>, 1> ->
    } {loop_grid}
    "sdnn.end"() : () -> ()
    return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "any()",
      disable_threading: false,
      verify_each: false
    }
  }
#-}
