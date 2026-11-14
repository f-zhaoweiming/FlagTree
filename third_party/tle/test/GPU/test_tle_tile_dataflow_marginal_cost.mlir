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

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false sync-cost=0.75 shared-bandwidth=2048 min-speedup=1.1" | FileCheck %s --check-prefix=MARGINAL
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false sync-cost=4.5 shared-bandwidth=2048 min-speedup=1" | FileCheck %s --check-prefix=RESERVE

// Reserve the same worker capacity before and after scheduling to isolate the
// marginal-work and communication-reserve decisions from occupancy changes.
// The two branches each cost 16, their transfer costs 1, and their independent
// work overlaps by 8. Worker input/output storage reuse adds one WAR barrier,
// priced at one third of the dispatch coefficient: 0.75 + 0.25 = 1 in the first
// run and 4.5 + 1.5 = 6 in the second. With total synchronization cost 1,
// the marginal saving is 6:
// enough for a 10% wave margin even though the long serial tail dilutes the
// whole-function speedup below 1.1. With synchronization cost 6, the saving is
// still positive (1), but below the communication uncertainty budget (3.5).
// min-speedup=1 must reject this fragile candidate.
#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// MARGINAL: \22communication_reserve\22:1
// MARGINAL-SAME: \22decision\22:\22parallel\22
// MARGINAL-SAME: \22marginal_saving\22:6
// MARGINAL-SAME: \22marginal_speedup\22:1.23
// MARGINAL-SAME: \22required_saving\22:2.6
// MARGINAL-SAME: \22reuse_barriers\22:1
// RESERVE: \22communication_reserve\22:3.5
// RESERVE-SAME: \22decision\22:\22serial\22
// RESERVE-SAME: \22marginal_saving\22:1
// RESERVE-SAME: \22marginal_speedup\22:1.03
// RESERVE-SAME: \22required_saving\22:3.5
// RESERVE-SAME: \22reuse_barriers\22:1
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  tt.func private @reserve_workers() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }

  // MARGINAL-LABEL: tt.func @wave_with_serial_tail
  // MARGINAL: ttg.warp_specialize
  // MARGINAL: ttg.local_load
  // MARGINAL-NOT: ttg.warp_specialize
  // MARGINAL: tt.return
  // RESERVE-LABEL: tt.func @wave_with_serial_tail
  // RESERVE-NOT: ttg.warp_specialize
  // RESERVE: tt.return
  tt.func @wave_with_serial_tail(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %x0 = math.exp %x : tensor<4x32xf32, #tile>
    %x1 = math.exp %x0 : tensor<4x32xf32, #tile>
    %y0 = math.exp %y : tensor<4x32xf32, #tile>
    %y1 = math.exp %y0 : tensor<4x32xf32, #tile>
    %joined = arith.addf %x1, %y1 : tensor<4x32xf32, #tile>
    %t0 = math.exp %joined : tensor<4x32xf32, #tile>
    %t1 = math.exp %t0 : tensor<4x32xf32, #tile>
    %t2 = math.exp %t1 : tensor<4x32xf32, #tile>
    %t3 = math.exp %t2 : tensor<4x32xf32, #tile>
    %t4 = math.exp %t3 : tensor<4x32xf32, #tile>
    %t5 = math.exp %t4 : tensor<4x32xf32, #tile>
    %t6 = math.exp %t5 : tensor<4x32xf32, #tile>
    %t7 = math.exp %t6 : tensor<4x32xf32, #tile>
    %t8 = math.exp %t7 : tensor<4x32xf32, #tile>
    %t9 = math.exp %t8 : tensor<4x32xf32, #tile>
    %t10 = math.exp %t9 : tensor<4x32xf32, #tile>
    %t11 = math.exp %t10 : tensor<4x32xf32, #tile>
    %t12 = math.exp %t11 : tensor<4x32xf32, #tile>
    %t13 = math.exp %t12 : tensor<4x32xf32, #tile>
    %t14 = math.exp %t13 : tensor<4x32xf32, #tile>
    %t15 = math.exp %t14 : tensor<4x32xf32, #tile>
    tt.return %t15 : tensor<4x32xf32, #tile>
  }
}
