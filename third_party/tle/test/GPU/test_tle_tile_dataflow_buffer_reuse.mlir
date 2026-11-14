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

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="force=true max-partitions=3" | FileCheck %s

#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #tile}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // The default group uses its original register value. Only the worker reads
  // the staged copy, so its result can overwrite that copy before the join.
  // CHECK-LABEL: tt.func @private_capture
  // CHECK: %[[BUF:[a-zA-Z0-9_]+]] = ttg.local_alloc %arg0
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttg.warp_specialize(%[[BUF]])
  // CHECK: partition0(%[[ARG:[a-zA-Z0-9_]+]]:
  // CHECK: ttg.local_load %[[ARG]]
  // CHECK: math.sqrt
  // CHECK: ttg.local_store {{.*}}, %[[ARG]]
  // CHECK: ttg.warp_return
  // CHECK: ttg.local_load %[[BUF]]
  // CHECK: ttg.local_dealloc %[[BUF]]
  // CHECK-NOT: ttg.local_dealloc
  // CHECK: tt.return
  tt.func @private_capture(%input: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %l0 = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %left = math.exp %l0 : tensor<4x32xf32, #tile>
    %r0 = math.absf %input : tensor<4x32xf32, #tile>
    %right = math.sqrt %r0 : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // Independent workers can each reuse their own staged tensor. The common
  // explicit capture signature does not imply that both workers read it.
  // CHECK-LABEL: tt.func @private_captures_multiple_workers
  // CHECK: ttg.local_alloc %arg1
  // CHECK: ttg.local_alloc %arg2
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK: ttg.local_store
  // CHECK: partition1
  // CHECK: ttg.local_load
  // CHECK: ttg.local_store
  // CHECK: ttg.local_load
  // CHECK: ttg.local_load
  // CHECK: ttg.local_dealloc
  // CHECK: ttg.local_dealloc
  // CHECK-NOT: ttg.local_dealloc
  // CHECK: tt.return
  tt.func @private_captures_multiple_workers(%a: tensor<4x32xf32, #tile>, %b: tensor<4x32xf32, #tile>, %c: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %a0 = arith.mulf %a, %a : tensor<4x32xf32, #tile>
    %a1 = math.exp %a0 : tensor<4x32xf32, #tile>
    %b0 = arith.mulf %b, %b : tensor<4x32xf32, #tile>
    %b1 = math.exp %b0 : tensor<4x32xf32, #tile>
    %c0 = arith.mulf %c, %c : tensor<4x32xf32, #tile>
    %c1 = math.exp %c0 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a1, %b1 : tensor<4x32xf32, #tile>
    %abc = arith.addf %ab, %c1 : tensor<4x32xf32, #tile>
    tt.return %abc : tensor<4x32xf32, #tile>
  }

  // No cross-worker barrier follows tensor entry loads. A shared capture must
  // remain separate from both output buffers to prevent a read/write race.
  // CHECK-LABEL: tt.func @shared_capture_multiple_workers
  // CHECK: %[[IN:[a-zA-Z0-9_]+]] = ttg.local_alloc %arg0
  // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = ttg.local_alloc {{.*}}: ()
  // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = ttg.local_alloc {{.*}}: ()
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttg.warp_specialize(%[[IN]], %[[OUT0]], %[[OUT1]])
  // CHECK: partition0
  // CHECK: partition1
  // CHECK: ttg.local_load %[[OUT0]]
  // CHECK: ttg.local_load %[[OUT1]]
  // CHECK: tt.return
  tt.func @shared_capture_multiple_workers(%input: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %a0 = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %a1 = math.exp %a0 : tensor<4x32xf32, #tile>
    %b0 = math.absf %input : tensor<4x32xf32, #tile>
    %b1 = math.sqrt %b0 : tensor<4x32xf32, #tile>
    %c0 = arith.addf %input, %input : tensor<4x32xf32, #tile>
    %c1 = math.exp %c0 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a1, %b1 : tensor<4x32xf32, #tile>
    %abc = arith.addf %ab, %c1 : tensor<4x32xf32, #tile>
    tt.return %abc : tensor<4x32xf32, #tile>
  }

  // A reduced result has a different shape/layout and keeps separate storage.
  // CHECK-LABEL: tt.func @different_result_shape
  // CHECK: ttg.local_alloc %arg0
  // CHECK: ttg.local_alloc {{.*}}: () -> !ttg.memdesc<4xf32,
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @different_result_shape(%input: tensor<4x32xf32, #tile>) -> tensor<4xf32, #row> {
    %a = math.exp %input : tensor<4x32xf32, #tile>
    %left = "tt.reduce"(%a) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %b = math.absf %input : tensor<4x32xf32, #tile>
    %right = "tt.reduce"(%b) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    tt.return %joined : tensor<4xf32, #row>
  }
}
