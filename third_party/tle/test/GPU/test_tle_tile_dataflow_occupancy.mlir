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

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="max-shared-bytes=131072 sync-cost=0 shared-bandwidth=1000000000 min-speedup=1" | FileCheck %s

// This test isolates one function because worker groups elsewhere in a module
// also contribute to its launch occupancy. With negligible transfer cost and
// zero synchronization cost, resource occupancy still rejects the legal fork.
// The generous storage cap separates this decision from capture legality.
// The worker reuses its private 32 KiB input buffer for the result. Count this
// buffer once, plus the 20-byte dispatch/capture allocation, instead of charging
// a separate 32 KiB output buffer.
#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: \22baseline_active_ctas\22:3
// CHECK-SAME: \22occupancy_cost\22:{{[1-9][0-9.]*}}
// CHECK-SAME: \22parallel_active_ctas\22:1
// CHECK-SAME: \22parallel_shared_bytes\22:32788
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // CHECK-LABEL: tt.func @shared_footprint_occupancy
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @shared_footprint_occupancy(%input: tensor<4x2048xf32, #tile>) -> tensor<4x2048xf32, #tile> {
    %a = math.exp %input : tensor<4x2048xf32, #tile>
    %left = arith.negf %a : tensor<4x2048xf32, #tile>
    %b = math.sin %input : tensor<4x2048xf32, #tile>
    %right = arith.negf %b : tensor<4x2048xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x2048xf32, #tile>
    tt.return %joined : tensor<4x2048xf32, #tile>
  }
}
