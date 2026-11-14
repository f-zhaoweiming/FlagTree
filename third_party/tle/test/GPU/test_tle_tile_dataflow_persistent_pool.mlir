// Copyright 2025-     FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="force=true max-partitions=4" -o %t
// RUN: FileCheck %s --check-prefix=POOL < %t
// RUN: triton-opt %t --allocate-shared-memory-nv='compute-capability=89 ptx-version=86' -test-print-membar | FileCheck %s --check-prefix=MEMBAR
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="force=true max-partitions=4 coalesce-waves=false" | FileCheck %s --check-prefix=SEPARATE

#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // Three dependent waves use two transport allocations, alternating their
  // roles after the full-group barriers. The removed allocation must also
  // disappear from every explicit capture and worker argument list.
  // POOL-LABEL: tt.func @three_waves_reuse
  // POOL: %[[FIRST:[a-zA-Z0-9_]+]] = ttg.local_alloc %arg1
  // POOL: %[[SECOND:[a-zA-Z0-9_]+]] = ttg.local_alloc {
  // POOL-NOT: ttg.local_alloc
  // POOL: ttg.warp_specialize(%[[FIRST]], %[[SECOND]])
  // POOL-SAME: tle.tile_dataflow.logical_waves = 3 : i32
  // POOL-SAME: tle.tile_dataflow.pooled_buffers = 1 : i32
  // POOL: default {
  // POOL: ttg.local_load %[[FIRST]]
  // POOL: ttg.local_store {{.*}}, %[[FIRST]]
  // POOL: partition0(%[[W0:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[W1:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>) num_warps(4)
  // POOL: ttg.local_load %[[W0]]
  // POOL: ttg.local_store {{.*}}, %[[W0]]
  // POOL: ttng.wait_barrier_named
  // POOL: ttg.local_load %[[W1]]
  // POOL: ttg.local_store {{.*}}, %[[W1]]
  // POOL: ttng.wait_barrier_named
  // POOL: ttg.local_load %[[W0]]
  // POOL: ttg.local_store {{.*}}, %[[W0]]
  // POOL: ttg.warp_return
  // POOL: ttg.local_dealloc %[[SECOND]]
  // POOL: ttg.local_load %[[FIRST]]
  // POOL: ttg.local_dealloc %[[FIRST]]
  // POOL-NOT: ttg.local_dealloc
  // POOL: tt.return
  // SEPARATE-LABEL: tt.func @three_waves_reuse
  // SEPARATE-COUNT-3: ttg.local_alloc
  // SEPARATE-NOT: ttg.local_alloc
  // SEPARATE: tt.return
  // MEMBAR-LABEL: tt.func @three_waves_reuse
  // MEMBAR: default {
  // MEMBAR: ttng.wait_barrier_named
  // MEMBAR-NEXT: %{{[a-zA-Z0-9_]+}} = ttg.local_load %[[D_BUFFER:[a-zA-Z0-9_]+]]
  // MEMBAR: ttg.local_barrier
  // MEMBAR: ttg.local_store {{.*}}, %[[D_BUFFER]]
  // MEMBAR: partition0(%[[MW0:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[MW1:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>) num_warps(4)
  // MEMBAR: ttg.local_load %[[MW0]]
  // MEMBAR: ttg.local_barrier
  // MEMBAR: ttg.local_store {{.*}}, %[[MW0]]
  // MEMBAR: ttng.wait_barrier_named
  // MEMBAR-NEXT: %{{[a-zA-Z0-9_]+}} = ttg.local_load %[[MW1]]
  // MEMBAR: ttg.local_barrier
  // MEMBAR: ttg.local_store {{.*}}, %[[MW1]]
  // MEMBAR: ttng.wait_barrier_named
  // MEMBAR-NEXT: %{{[a-zA-Z0-9_]+}} = ttg.local_load %[[MW0]]
  // MEMBAR: ttg.local_barrier
  // MEMBAR: ttg.local_store {{.*}}, %[[MW0]]
  tt.func @three_waves_reuse(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %x0 = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %a = math.exp %x0 : tensor<4x32xf32, #tile>
    %y0 = arith.mulf %y, %y : tensor<4x32xf32, #tile>
    %b = math.sqrt %y0 : tensor<4x32xf32, #tile>
    %first = arith.addf %a, %b : tensor<4x32xf32, #tile>
    %first0 = arith.mulf %first, %first : tensor<4x32xf32, #tile>
    %c = math.exp %first0 : tensor<4x32xf32, #tile>
    %first1 = arith.negf %first : tensor<4x32xf32, #tile>
    %d = math.sin %first1 : tensor<4x32xf32, #tile>
    %second = arith.addf %c, %d : tensor<4x32xf32, #tile>
    %second0 = arith.mulf %second, %second : tensor<4x32xf32, #tile>
    %e = math.exp %second0 : tensor<4x32xf32, #tile>
    %second1 = arith.negf %second : tensor<4x32xf32, #tile>
    %f = math.sin %second1 : tensor<4x32xf32, #tile>
    %third = arith.addf %e, %f : tensor<4x32xf32, #tile>
    tt.return %third : tensor<4x32xf32, #tile>
  }

  // Cross-phase reuse may recycle earlier private results, but a broadcast
  // still read by another worker in this phase must not become an output.
  // Eight unmerged allocations become six distinct descriptors. The last
  // two worker arguments hold the two broadcast phases and are never stored
  // by either worker; every store below targets a private result descriptor.
  // POOL-LABEL: tt.func @three_way_three_waves
  // POOL-COUNT-6: ttg.local_alloc
  // POOL-NOT: ttg.local_alloc
  // POOL: ttg.warp_specialize
  // POOL-SAME: tle.tile_dataflow.logical_waves = 3 : i32
  // POOL-SAME: tle.tile_dataflow.pooled_buffers = 2 : i32
  // POOL: partition0(%[[P0A:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P0B:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P0C:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P0D:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P0E:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P0F:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>) num_warps(4)
  // POOL: ttg.local_store {{.*}}, %[[P0A]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_load %[[P0F]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_store {{.*}}, %[[P0C]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_load %[[P0E]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_store {{.*}}, %[[P0A]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.warp_return
  // POOL: partition1(%[[P1A:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P1B:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P1C:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P1D:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P1E:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>, %[[P1F:[a-zA-Z0-9_]+]]: !ttg.memdesc<{{[^>]+}}>) num_warps(4)
  // POOL: ttg.local_store {{.*}}, %[[P1B]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_load %[[P1F]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_store {{.*}}, %[[P1D]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_load %[[P1E]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.local_store {{.*}}, %[[P1B]]
  // POOL-NOT: ttg.local_store
  // POOL: ttg.warp_return
  // SEPARATE-LABEL: tt.func @three_way_three_waves
  // SEPARATE-COUNT-8: ttg.local_alloc
  // SEPARATE-NOT: ttg.local_alloc
  // SEPARATE: tt.return
  tt.func @three_way_three_waves(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>, %z: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %w0b0a = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %w0b0 = math.exp %w0b0a : tensor<4x32xf32, #tile>
    %w0b1a = arith.mulf %y, %y : tensor<4x32xf32, #tile>
    %w0b1 = math.sqrt %w0b1a : tensor<4x32xf32, #tile>
    %w0b2a = arith.mulf %z, %z : tensor<4x32xf32, #tile>
    %w0b2 = math.sin %w0b2a : tensor<4x32xf32, #tile>
    %joined0b1 = arith.addf %w0b0, %w0b1 : tensor<4x32xf32, #tile>
    %joined0b2 = arith.addf %joined0b1, %w0b2 : tensor<4x32xf32, #tile>
    %w1b0a = arith.mulf %joined0b2, %joined0b2 : tensor<4x32xf32, #tile>
    %w1b0 = math.exp %w1b0a : tensor<4x32xf32, #tile>
    %w1b1a = arith.mulf %joined0b2, %joined0b2 : tensor<4x32xf32, #tile>
    %w1b1 = math.sqrt %w1b1a : tensor<4x32xf32, #tile>
    %w1b2a = arith.mulf %joined0b2, %joined0b2 : tensor<4x32xf32, #tile>
    %w1b2 = math.sin %w1b2a : tensor<4x32xf32, #tile>
    %joined1b1 = arith.addf %w1b0, %w1b1 : tensor<4x32xf32, #tile>
    %joined1b2 = arith.addf %joined1b1, %w1b2 : tensor<4x32xf32, #tile>
    %w2b0a = arith.mulf %joined1b2, %joined1b2 : tensor<4x32xf32, #tile>
    %w2b0 = math.exp %w2b0a : tensor<4x32xf32, #tile>
    %w2b1a = arith.mulf %joined1b2, %joined1b2 : tensor<4x32xf32, #tile>
    %w2b1 = math.sqrt %w2b1a : tensor<4x32xf32, #tile>
    %w2b2a = arith.mulf %joined1b2, %joined1b2 : tensor<4x32xf32, #tile>
    %w2b2 = math.sin %w2b2a : tensor<4x32xf32, #tile>
    %joined2b1 = arith.addf %w2b0, %w2b1 : tensor<4x32xf32, #tile>
    %joined2b2 = arith.addf %joined2b1, %w2b2 : tensor<4x32xf32, #tile>
    tt.return %joined2b2 : tensor<4x32xf32, #tile>
  }
}
