// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false sync-cost=3 shared-bandwidth=2048 min-speedup=1" | FileCheck %s

// Synthetic historical cost records isolate metadata compatibility from the
// arithmetic in old waves. Reprice both records using the current pass
// parameters. The old five-component format has no storage-reuse barrier:
// 32 - 0.5 * (32 - 16) + 2048 / 2048 + 3 = 28. The new six-component format
// adds one group barrier, costing 3 / 3 = 1. The following join adds one unit
// to each outside_cost. Treating either format as unknown would reject the
// new candidate, and retaining a stale barrier cost would fail these checks.
#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK: \22decision\22:\22parallel\22
// CHECK-SAME: \22function\22:\22historical_five_components\22
// CHECK-SAME: \22outside_cost\22:29
// CHECK-SAME: \22decision\22:\22parallel\22
// CHECK-SAME: \22function\22:\22reuse_six_components\22
// CHECK-SAME: \22outside_cost\22:30
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // CHECK-LABEL: tt.func @historical_five_components
  // CHECK: ttg.warp_specialize
  // CHECK: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @historical_five_components(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    ttg.warp_specialize() attributes {tle.tile_dataflow.costs = array<f64: 32.0, 16.0, 0.0, 2048.0, 2.0>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    %x0 = math.exp %x : tensor<4x32xf32, #tile>
    %x1 = math.exp %x0 : tensor<4x32xf32, #tile>
    %y0 = math.exp %y : tensor<4x32xf32, #tile>
    %y1 = math.exp %y0 : tensor<4x32xf32, #tile>
    %joined = arith.addf %x1, %y1 : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // CHECK-LABEL: tt.func @reuse_six_components
  // CHECK: ttg.warp_specialize
  // CHECK: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @reuse_six_components(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    ttg.warp_specialize() attributes {tle.tile_dataflow.costs = array<f64: 32.0, 16.0, 0.0, 2048.0, 2.0, 1.0>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    %x0 = math.exp %x : tensor<4x32xf32, #tile>
    %x1 = math.exp %x0 : tensor<4x32xf32, #tile>
    %y0 = math.exp %y : tensor<4x32xf32, #tile>
    %y1 = math.exp %y0 : tensor<4x32xf32, #tile>
    %joined = arith.addf %x1, %y1 : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }
}
