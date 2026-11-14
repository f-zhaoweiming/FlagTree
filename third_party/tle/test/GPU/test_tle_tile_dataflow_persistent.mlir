// Copyright 2025-     FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="force=true max-partitions=4" -o %t
// RUN: FileCheck %s --check-prefix=MERGED < %t
// RUN: triton-opt %t -triton-tle-tile-dataflow-schedule="force=true max-partitions=4" -o %t.again
// RUN: diff %t %t.again
// RUN: triton-opt %t -triton-tle-allocate-named-barriers | FileCheck %s --check-prefix=ALLOCATED
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="force=true max-partitions=4 coalesce-waves=false" | FileCheck %s --check-prefix=SEPARATE --implicit-check-not=logical_waves --implicit-check-not=wait_barrier_named

#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // Each group publishes its first result, meets at one cross-group barrier,
  // then reconstructs the cheap join. Its own result remains in registers.
  // MERGED-LABEL: tt.func @two_dependent_waves
  // MERGED: ttg.warp_specialize
  // MERGED-SAME: tle.tile_dataflow.logical_waves = 2 : i32
  // MERGED-SAME: tle.tile_dataflow.wave_costs = [array<f64:
  // MERGED: default {
  // MERGED: math.exp
  // MERGED: ttg.local_store
  // MERGED: arith.constant 256 : i32
  // MERGED: ttng.wait_barrier_named
  // MERGED-SAME: tle.tile_dataflow.full_group_sync
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: ttg.warp_yield
  // MERGED: partition0
  // MERGED: %[[OWN:[a-zA-Z0-9_]+]] = math.sqrt
  // MERGED: ttg.local_store %[[OWN]],
  // MERGED: arith.constant 256 : i32
  // MERGED: ttng.wait_barrier_named
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: %[[OTHER:[a-zA-Z0-9_]+]] = ttg.local_load
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: arith.addf %[[OTHER]], %[[OWN]]
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: math.sin
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: ttg.warp_return
  // MERGED-NOT: ttg.warp_specialize
  // MERGED: tt.return
  // SEPARATE-LABEL: tt.func @two_dependent_waves
  // SEPARATE-COUNT-2: ttg.warp_specialize
  // SEPARATE-NOT: ttg.warp_specialize
  // SEPARATE: tt.return
  // ALLOCATED-LABEL: tt.func @two_dependent_waves
  // ALLOCATED: default {
  // ALLOCATED: %[[DEFAULT_ID:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
  // ALLOCATED: ttng.wait_barrier_named %[[DEFAULT_ID]],
  // ALLOCATED: partition0
  // ALLOCATED: %[[WORKER_ID:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
  // ALLOCATED: ttng.wait_barrier_named %[[WORKER_ID]],
  tt.func @two_dependent_waves(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
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
    tt.return %second : tensor<4x32xf32, #tile>
  }

  // The already merged first pair must remain eligible for another merge.
  // Exactly two handoff barriers appear in each persistent group.
  // MERGED-LABEL: tt.func @three_dependent_waves
  // MERGED: ttg.warp_specialize
  // MERGED-SAME: tle.tile_dataflow.logical_waves = 3 : i32
  // MERGED: default {
  // MERGED-COUNT-2: ttng.wait_barrier_named
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: ttg.warp_yield
  // MERGED: partition0
  // MERGED-COUNT-2: ttng.wait_barrier_named
  // MERGED-NOT: ttng.wait_barrier_named
  // MERGED: ttg.warp_return
  // MERGED-NOT: ttg.warp_specialize
  // MERGED: tt.return
  // SEPARATE-LABEL: tt.func @three_dependent_waves
  // SEPARATE-COUNT-3: ttg.warp_specialize
  // SEPARATE-NOT: ttg.warp_specialize
  // SEPARATE: tt.return
  // ALLOCATED-LABEL: tt.func @three_dependent_waves
  // ALLOCATED: default {
  // ALLOCATED: %[[FIRST_ID:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
  // ALLOCATED: ttng.wait_barrier_named %[[FIRST_ID]],
  // ALLOCATED: %[[SECOND_ID:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
  // ALLOCATED: ttng.wait_barrier_named %[[SECOND_ID]],
  tt.func @three_dependent_waves(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
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

  // Changing worker-group cardinality requires a separate dispatch.
  // MERGED-LABEL: tt.func @different_widths
  // MERGED: ttg.warp_specialize
  // MERGED-NOT: tle.tile_dataflow.logical_waves
  // MERGED: partition1
  // MERGED: ttg.warp_specialize
  // MERGED-NOT: tle.tile_dataflow.logical_waves
  // MERGED: partition0
  // MERGED-NOT: partition1
  // MERGED-NOT: ttg.warp_specialize
  // MERGED: tt.return
  tt.func @different_widths(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>, %z: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %x0 = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %a = math.exp %x0 : tensor<4x32xf32, #tile>
    %y0 = arith.mulf %y, %y : tensor<4x32xf32, #tile>
    %b = math.sqrt %y0 : tensor<4x32xf32, #tile>
    %z0 = arith.negf %z : tensor<4x32xf32, #tile>
    %c = math.sin %z0 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a, %b : tensor<4x32xf32, #tile>
    %first = arith.addf %ab, %c : tensor<4x32xf32, #tile>
    %first0 = arith.mulf %first, %first : tensor<4x32xf32, #tile>
    %d = math.exp %first0 : tensor<4x32xf32, #tile>
    %first1 = arith.negf %first : tensor<4x32xf32, #tile>
    %e = math.sin %first1 : tensor<4x32xf32, #tile>
    %second = arith.addf %d, %e : tensor<4x32xf32, #tile>
    tt.return %second : tensor<4x32xf32, #tile>
  }

  // A global write between dependent waves cannot move into a worker or
  // across the persistent handoff. Keep both dispatches and the write order.
  // MERGED-LABEL: tt.func @side_effect_boundary
  // MERGED: ttg.warp_specialize
  // MERGED-NOT: tle.tile_dataflow.logical_waves
  // MERGED: tt.store
  // MERGED: ttg.warp_specialize
  // MERGED-NOT: tle.tile_dataflow.logical_waves
  // MERGED-NOT: ttg.warp_specialize
  // MERGED: tt.return
  tt.func @side_effect_boundary(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>, %output: tensor<4x32x!tt.ptr<f32>, #tile>) -> tensor<4x32xf32, #tile> {
    %x0 = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %a = math.exp %x0 : tensor<4x32xf32, #tile>
    %y0 = arith.mulf %y, %y : tensor<4x32xf32, #tile>
    %b = math.sqrt %y0 : tensor<4x32xf32, #tile>
    %first = arith.addf %a, %b : tensor<4x32xf32, #tile>
    tt.store %output, %first : tensor<4x32x!tt.ptr<f32>, #tile>
    %first0 = arith.mulf %first, %first : tensor<4x32xf32, #tile>
    %c = math.exp %first0 : tensor<4x32xf32, #tile>
    %first1 = arith.negf %first : tensor<4x32xf32, #tile>
    %d = math.sin %first1 : tensor<4x32xf32, #tile>
    %second = arith.addf %c, %d : tensor<4x32xf32, #tile>
    tt.return %second : tensor<4x32xf32, #tile>
  }
}
