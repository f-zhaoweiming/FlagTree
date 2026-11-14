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

// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=89 ptx-version=86' -test-print-membar | FileCheck %s --check-prefix=MEMBAR
// RUN: triton-opt %s -triton-tle-allocate-named-barriers --allocate-shared-memory-nv='compute-capability=89 ptx-version=86' --convert-triton-gpu-to-llvm='compute-capability=89 ptx-version=86' -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  // A proven whole-group wait closes RAW/WAW/WAR dependencies. A new WAR
  // dependency between the two waits still requires a local barrier.
  // MEMBAR-LABEL: tt.func @full_group_wait
  // MEMBAR: ttg.local_store
  // MEMBAR: ttng.wait_barrier_named
  // MEMBAR-NEXT: {{.*}}ttg.local_load
  // MEMBAR-NEXT: ttg.local_barrier
  // MEMBAR-NEXT: ttg.local_store
  // MEMBAR-NEXT: ttng.wait_barrier_named
  // MEMBAR-NEXT: {{.*}}ttg.local_load
  // MEMBAR-NEXT: tt.return
  // LLVM-LABEL: llvm.func @full_group_wait
  // LLVM-NOT: llvm.nvvm.barrier.cta.sync.aligned.count
  // LLVM: llvm.nvvm.barrier.cta.sync.count
  // LLVM: nvvm.barrier0
  // LLVM: llvm.nvvm.barrier.cta.sync.count
  // LLVM-NOT: nvvm.barrier0
  // LLVM-NOT: llvm.nvvm.barrier.cta.sync.aligned.count
  // LLVM: llvm.return
  tt.func @full_group_wait(%value: tensor<128xf32, #blocked>) {
    %id = arith.constant 16 : i32
    %threads = arith.constant 128 : i32
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.local_store %value, %buffer : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttng.wait_barrier_named %id, %threads {tle.tile_dataflow.full_group_sync} : i32, i32
    %loaded = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    ttg.local_store %loaded, %buffer : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttng.wait_barrier_named %id, %threads {tle.tile_dataflow.full_group_sync} : i32, i32
    %result = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }

  // An ordinary named wait has no proof that its participants include every
  // thread in the current group, so it must retain conservative analysis.
  // MEMBAR-LABEL: tt.func @ordinary_wait
  // MEMBAR: ttng.wait_barrier_named
  // MEMBAR-NEXT: ttg.local_barrier
  // MEMBAR-NEXT: {{.*}}ttg.local_load
  // LLVM-LABEL: llvm.func @ordinary_wait
  // LLVM: llvm.nvvm.barrier.cta.sync.aligned.count
  // LLVM: nvvm.barrier0
  // LLVM: llvm.return
  tt.func @ordinary_wait(%value: tensor<128xf32, #blocked>) {
    %id = arith.constant 16 : i32
    %threads = arith.constant 64 : i32
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.local_store %value, %buffer : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttng.wait_barrier_named %id, %threads : i32, i32
    %result = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }

  // Arrive is nonblocking and cannot close hazards, even with a stray marker.
  // MEMBAR-LABEL: tt.func @arrive_is_not_a_rendezvous
  // MEMBAR: ttng.arrive_barrier_named
  // MEMBAR-NEXT: ttg.local_barrier
  // MEMBAR-NEXT: {{.*}}ttg.local_load
  // LLVM-LABEL: llvm.func @arrive_is_not_a_rendezvous
  // LLVM: llvm.nvvm.barrier.cta.arrive.aligned.count
  // LLVM: nvvm.barrier0
  // LLVM: llvm.return
  tt.func @arrive_is_not_a_rendezvous(%value: tensor<128xf32, #blocked>) {
    %id = arith.constant 16 : i32
    %threads = arith.constant 128 : i32
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.local_store %value, %buffer : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttng.arrive_barrier_named %id, %threads {tle.tile_dataflow.full_group_sync} : i32, i32
    %result = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}
