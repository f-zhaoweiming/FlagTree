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

// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=89 ptx-version=86' -test-print-membar | FileCheck %s

#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#other = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#matrix = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#matrix_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {

  // Even identical, nonreplicated result layouts cannot prove thread-private
  // physical accesses: local_load may lower to a collective ldmatrix. Retain
  // WAR, WAW, and RAW synchronization in persistent worker bodies.
  // CHECK-LABEL: tt.func @private_worker_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: {{.*}}arith.addf
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: {{.*}}ttg.local_load
  tt.func @private_worker_war(%value: tensor<128xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<128xf32, #full>) -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<128xf32, #shared, #smem, mutable>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<128xf32, #full>
      %loaded = ttg.local_load %arg : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #full>
      %twice = arith.addf %loaded, %loaded : tensor<128xf32, #full>
      ttg.local_store %twice, %arg : tensor<128xf32, #full> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      ttg.local_store %zero, %arg : tensor<128xf32, #full> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %again = ttg.local_load %arg : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #full>
      ttg.warp_return
    } : (!ttg.memdesc<128xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // The same physical-access limitation applies to the default group.
  // CHECK-LABEL: tt.func @private_default_war
  // CHECK: default
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @private_default_war(%value: tensor<128xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<128xf32, #full>) -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      %loaded = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #full>
      ttg.local_store %loaded, %buffer : tensor<128xf32, #full> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<128xf32, #shared, #smem, mutable>) num_warps(4) {
      ttg.warp_return
    } : (!ttg.memdesc<128xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // This layout reproduces the two-wave GPU racecheck regression: a 4x512
  // tile has exactly 16 logical values per thread, but ldmatrix cooperatively
  // reads addresses that the following scalar store assigns to other lanes.
  // CHECK-LABEL: tt.func @collective_load_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: {{.*}}arith.addf
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @collective_load_war(%value: tensor<4x512xf32, #matrix>) {
    %buffer = ttg.local_alloc %value : (tensor<4x512xf32, #matrix>) -> !ttg.memdesc<4x512xf32, #matrix_shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<4x512xf32, #matrix_shared, #smem, mutable>) num_warps(4) {
      %loaded = ttg.local_load %arg : !ttg.memdesc<4x512xf32, #matrix_shared, #smem, mutable> -> tensor<4x512xf32, #matrix>
      %twice = arith.addf %loaded, %loaded : tensor<4x512xf32, #matrix>
      ttg.local_store %twice, %arg : tensor<4x512xf32, #matrix> -> !ttg.memdesc<4x512xf32, #matrix_shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<4x512xf32, #matrix_shared, #smem, mutable>) -> ()
    tt.return
  }

  // Different layouts can assign an address to different load/store threads.
  // CHECK-LABEL: tt.func @different_layout_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @different_layout_war(%value: tensor<256xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<256xf32, #full>) -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<256xf32, #shared, #smem, mutable>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<256xf32, #other>
      %loaded = ttg.local_load %arg : !ttg.memdesc<256xf32, #shared, #smem, mutable> -> tensor<256xf32, #full>
      ttg.local_store %zero, %arg : tensor<256xf32, #other> -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<256xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // Shape-clipped warp counts would hide the two replicated physical warps.
  // CHECK-LABEL: tt.func @replicated_warps_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @replicated_warps_war(%value: tensor<64xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<64xf32, #full>) -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<64xf32, #shared, #smem, mutable>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<64xf32, #full>
      %loaded = ttg.local_load %arg : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #full>
      ttg.local_store %zero, %arg : tensor<64xf32, #full> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<64xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // Shape-clipped lane counts would also hide replication within each warp.
  // CHECK-LABEL: tt.func @replicated_lanes_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @replicated_lanes_war(%value: tensor<16xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<16xf32, #full>) -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<16xf32, #shared, #smem, mutable>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<16xf32, #full>
      %loaded = ttg.local_load %arg : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #full>
      ttg.local_store %zero, %arg : tensor<16xf32, #full> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<16xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // Equivalent alias views still require synchronization: SSA is not identical.
  // CHECK-LABEL: tt.func @aliased_descriptor_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @aliased_descriptor_war(%value: tensor<128xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<128xf32, #full>) -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer) attributes {tle.tile_dataflow.logical_waves = 2 : i32}
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<128xf32, #shared, #smem, mutable>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<128xf32, #full>
      %view = ttg.memdesc_subslice %arg[0] : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %arg : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #full>
      ttg.local_store %zero, %view : tensor<128xf32, #full> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // Ordinary, unmerged specialization retains its existing WAR behavior.
  // CHECK-LABEL: tt.func @unmerged_war
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  tt.func @unmerged_war(%value: tensor<128xf32, #full>) {
    %buffer = ttg.local_alloc %value : (tensor<128xf32, #full>) -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer)
    default {
      ttg.warp_yield
    }
    partition0(%arg: !ttg.memdesc<128xf32, #shared, #smem, mutable>) num_warps(4) {
      %zero = arith.constant dense<0.0> : tensor<128xf32, #full>
      %loaded = ttg.local_load %arg : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #full>
      ttg.local_store %zero, %arg : tensor<128xf32, #full> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

