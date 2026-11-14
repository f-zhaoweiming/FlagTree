// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="min-branch-ops=1 max-partitions=2 force=true" | FileCheck %s
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="min-branch-ops=1 max-partitions=2 force=true" -tritongpu-allocate-warp-groups -allocate-shared-memory-nv="compute-capability=89 ptx-version=86" -convert-triton-gpu-to-llvm="compute-capability=89 ptx-version=86" -convert-nv-gpu-to-llvm -convert-warp-specialize-to-llvm | FileCheck %s --check-prefix=LLVM

// Predicates loaded from memory cannot be rematerialized. Both the capture and
// the worker result must use byte-sized shared storage: native shared-memory
// lowering does not support a sub-byte memdesc element type.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #tile}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // CHECK-LABEL: tt.func @boolean_capture_and_result
  // CHECK: %[[MASK:.*]] = tt.load
  // CHECK: %[[BYTES:.*]] = arith.extui %[[MASK]] {{.*}}: tensor<128xi1, {{.*}}> to tensor<128xi8, {{.*}}>
  // CHECK: %[[CAPTURE:.*]] = ttg.local_alloc %[[BYTES]] {{.*}}: (tensor<128xi8, {{.*}}>) -> !ttg.memdesc<128xi8,
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: %[[LOADED:.*]] = ttg.local_load {{.*}} -> tensor<128xi8,
  // CHECK: arith.trunci %[[LOADED]] : tensor<128xi8, {{.*}}> to tensor<128xi1,
  // CHECK: %[[RIGHT:.*]] = arith.ori
  // CHECK: %[[RIGHTBYTES:.*]] = arith.extui %[[RIGHT]] : tensor<128xi1, {{.*}}> to tensor<128xi8,
  // CHECK: ttg.local_store %[[RIGHTBYTES]]
  // CHECK: ttg.warp_return
  // CHECK: %[[JOINBYTES:.*]] = ttg.local_load %[[CAPTURE]]
  // CHECK: arith.trunci %[[JOINBYTES]] {{.*}}: tensor<128xi8, {{.*}}> to tensor<128xi1,
  // CHECK: arith.xori
  // CHECK: tt.return
  // LLVM-LABEL: llvm.func @boolean_capture_and_result
  // LLVM-NOT: ttg.warp_specialize
  // LLVM: llvm.zext {{.*}} : i1 to i8
  // LLVM: llvm.trunc {{.*}} : i8 to i1
  // LLVM-NOT: ttg.warp_specialize
  // LLVM: llvm.return
  tt.func @boolean_capture_and_result(%mask_ptr: !tt.ptr<i1>, %a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %out: !tt.ptr<i1>) {
    %zero = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %mp = tt.splat %mask_ptr : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>, #blocked>
    %mask = tt.load %mp : tensor<128x!tt.ptr<i1>, #blocked>
    %ap = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %a = tt.load %ap : tensor<128x!tt.ptr<f32>, #blocked>
    %positive = arith.cmpf ogt, %a, %zero : tensor<128xf32, #blocked>
    %left = arith.andi %positive, %mask : tensor<128xi1, #blocked>
    %bp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %b = tt.load %bp : tensor<128x!tt.ptr<f32>, #blocked>
    %negative = arith.cmpf olt, %b, %zero : tensor<128xf32, #blocked>
    %right = arith.ori %negative, %mask : tensor<128xi1, #blocked>
    %joined = arith.xori %left, %right : tensor<128xi1, #blocked>
    %op = tt.splat %out : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>, #blocked>
    tt.store %op, %joined : tensor<128x!tt.ptr<i1>, #blocked>
    tt.return
  }

  // A slice layout with no register bits still supports the same transfer path.
  // CHECK-LABEL: tt.func @slice_result
  // CHECK: %[[SLICEOUT:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<4xf32,
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: ttg.local_store {{.*}} : tensor<4xf32, #ttg.slice<
  // CHECK: ttg.local_load %[[SLICEOUT]] {{.*}} -> tensor<4xf32, #ttg.slice<
  // CHECK: arith.addf
  // LLVM-LABEL: llvm.func @slice_result
  // LLVM-NOT: ttg.warp_specialize
  // LLVM: nvvm.barrier
  // LLVM-NOT: ttg.warp_specialize
  // LLVM: llvm.return
  tt.func @slice_result(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %ap = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %a = tt.load %ap : tensor<4x32x!tt.ptr<f32>, #tile>
    %squared = arith.mulf %a, %a : tensor<4x32xf32, #tile>
    %left = "tt.reduce"(%squared) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %bp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %b = tt.load %bp : tensor<4x32x!tt.ptr<f32>, #tile>
    %exponent = math.exp %b : tensor<4x32xf32, #tile>
    %right = "tt.reduce"(%exponent) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    %op = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #row>
    tt.store %op, %joined : tensor<4x!tt.ptr<f32>, #row>
    tt.return
  }
}
