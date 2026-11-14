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

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false force=true max-partitions=4" -o %t
// RUN: FileCheck %s < %t
// RUN: triton-opt %t -triton-tle-tile-dataflow-schedule="coalesce-waves=false force=true max-partitions=4" -o %t.again
// RUN: diff %t %t.again
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false min-speedup=1000000 max-partitions=4" | FileCheck %s --check-prefix=SERIAL
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false sync-cost=0 overlap=1 shared-bandwidth=1000000000 min-speedup=1 max-partitions=4" | FileCheck %s --check-prefix=AUTO
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="coalesce-waves=false force=true max-partitions=2" | FileCheck %s --check-prefix=LATE

// A binary expression tree must expose its full independent frontier. The
// scheduling unit is a DAG wave, not a pair of operands of one operation.
#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #tile}>

// SERIAL: module attributes
// SERIAL-SAME: tle.tile_dataflow.cost_model
// SERIAL-NOT: ttg.warp_specialize
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {

  // CHECK-LABEL: tt.func @three_way_tree
  // CHECK: ttg.warp_specialize
  // CHECK: default {
  // CHECK: math.exp
  // CHECK: partition0
  // CHECK: math.exp
  // CHECK: partition1
  // CHECK: math.exp
  // CHECK-NOT: partition2
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @three_way_tree(%a: tensor<4x32xf32, #tile>, %b: tensor<4x32xf32, #tile>, %c: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
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

  // CHECK-LABEL: tt.func @four_way_tree
  // CHECK: ttg.warp_specialize
  // CHECK: default {
  // CHECK: math.exp
  // CHECK: partition0
  // CHECK: math.exp
  // CHECK: partition1
  // CHECK: math.exp
  // CHECK: partition2
  // CHECK: math.exp
  // CHECK-NOT: partition3
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: arith.addf
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @four_way_tree(%a: tensor<4x32xf32, #tile>, %b: tensor<4x32xf32, #tile>, %c: tensor<4x32xf32, #tile>, %d: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %a0 = arith.mulf %a, %a : tensor<4x32xf32, #tile>
    %a1 = math.exp %a0 : tensor<4x32xf32, #tile>
    %b0 = arith.mulf %b, %b : tensor<4x32xf32, #tile>
    %b1 = math.exp %b0 : tensor<4x32xf32, #tile>
    %c0 = arith.mulf %c, %c : tensor<4x32xf32, #tile>
    %c1 = math.exp %c0 : tensor<4x32xf32, #tile>
    %d0 = arith.mulf %d, %d : tensor<4x32xf32, #tile>
    %d1 = math.exp %d0 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a1, %b1 : tensor<4x32xf32, #tile>
    %cd = arith.addf %c1, %d1 : tensor<4x32xf32, #tile>
    %all = arith.addf %ab, %cd : tensor<4x32xf32, #tile>
    tt.return %all : tensor<4x32xf32, #tile>
  }

  // No single operation consumes all four branches. A global frontier still
  // schedules them together and returns both independently joined outputs.
  // CHECK-LABEL: tt.func @disconnected_outputs
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: partition1
  // CHECK: partition2
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return {{.*}}, {{.*}} :
  tt.func @disconnected_outputs(%a: tensor<4x32xf32, #tile>, %b: tensor<4x32xf32, #tile>, %c: tensor<4x32xf32, #tile>, %d: tensor<4x32xf32, #tile>) -> (tensor<4x32xf32, #tile>, tensor<4x32xf32, #tile>) {
    %a0 = arith.mulf %a, %a : tensor<4x32xf32, #tile>
    %a1 = math.exp %a0 : tensor<4x32xf32, #tile>
    %b0 = arith.mulf %b, %b : tensor<4x32xf32, #tile>
    %b1 = math.exp %b0 : tensor<4x32xf32, #tile>
    %c0 = arith.mulf %c, %c : tensor<4x32xf32, #tile>
    %c1 = math.exp %c0 : tensor<4x32xf32, #tile>
    %d0 = arith.mulf %d, %d : tensor<4x32xf32, #tile>
    %d1 = math.exp %d0 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a1, %b1 : tensor<4x32xf32, #tile>
    %cd = arith.addf %c1, %d1 : tensor<4x32xf32, #tile>
    tt.return %ab, %cd : tensor<4x32xf32, #tile>, tensor<4x32xf32, #tile>
  }

  // CHECK-LABEL: tt.func @three_dependent_waves
  // AUTO-LABEL: tt.func @three_dependent_waves
  // AUTO: ttg.warp_specialize
  // AUTO: ttg.local_load
  // AUTO: arith.addf
  // AUTO: ttg.warp_specialize
  // AUTO: ttg.local_load
  // AUTO: arith.addf
  // AUTO: ttg.warp_specialize
  // AUTO: ttg.local_load
  // AUTO: arith.addf
  // AUTO-NOT: ttg.warp_specialize
  // AUTO: tt.return
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
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

  // An input capture shared by workers and a result with multiple consumers
  // must remain available to all consumers after the complete wave joins.
  // CHECK-LABEL: tt.func @common_input_and_fanout
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: partition1
  // CHECK: partition2
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return {{.*}}, {{.*}} :
  tt.func @common_input_and_fanout(%x: tensor<4x32xf32, #tile>) -> (tensor<4x32xf32, #tile>, tensor<4x32xf32, #tile>) {
    %x2 = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %a = math.exp %x2 : tensor<4x32xf32, #tile>
    %nx = arith.negf %x : tensor<4x32xf32, #tile>
    %b = math.sin %nx : tensor<4x32xf32, #tile>
    %ax = math.absf %x : tensor<4x32xf32, #tile>
    %c = math.sqrt %ax : tensor<4x32xf32, #tile>
    %one = arith.constant dense<1.0> : tensor<4x32xf32, #tile>
    %x1 = arith.addf %x, %one : tensor<4x32xf32, #tile>
    %d = math.cos %x1 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a, %b : tensor<4x32xf32, #tile>
    %cd = arith.addf %c, %d : tensor<4x32xf32, #tile>
    %all = arith.addf %ab, %cd : tensor<4x32xf32, #tile>
    tt.return %all, %a : tensor<4x32xf32, #tile>, tensor<4x32xf32, #tile>
  }

  // Distinct pointer arguments may alias at runtime. Neither forcing nor a
  // wider wave is allowed to reorder these RAW/WAR/WAW dependencies.
  // CHECK-LABEL: tt.func @aliasing_three_reads
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @aliasing_three_reads(%ptr: !tt.ptr<f32>, %alias: !tt.ptr<f32>) -> tensor<4x32xf32, #tile> {
    %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %q = tt.splat %alias : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %one = arith.constant dense<1.0> : tensor<4x32xf32, #tile>
    %two = arith.constant dense<2.0> : tensor<4x32xf32, #tile>
    %x = tt.load %p : tensor<4x32x!tt.ptr<f32>, #tile>
    %x2 = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %a = math.exp %x2 : tensor<4x32xf32, #tile>
    tt.store %q, %one : tensor<4x32x!tt.ptr<f32>, #tile>
    %y = tt.load %p : tensor<4x32x!tt.ptr<f32>, #tile>
    %y2 = arith.mulf %y, %y : tensor<4x32xf32, #tile>
    %b = math.sqrt %y2 : tensor<4x32xf32, #tile>
    tt.store %q, %two : tensor<4x32x!tt.ptr<f32>, #tile>
    %z = tt.load %p : tensor<4x32x!tt.ptr<f32>, #tile>
    %z2 = arith.mulf %z, %z : tensor<4x32xf32, #tile>
    %c = math.sin %z2 : tensor<4x32xf32, #tile>
    %ab = arith.addf %a, %b : tensor<4x32xf32, #tile>
    %all = arith.addf %ab, %c : tensor<4x32xf32, #tile>
    tt.return %all : tensor<4x32xf32, #tile>
  }

  // The later branches have distinct constants defined before the first wave.
  // Backward collection must keep those early definitions as captures: adding
  // them to the later branches invents a cross-branch path through the earlier
  // warp-specialize boundary. Cheap constants may be rematerialized by workers.
  // The terminal reduction also ensures this is the final executable wave.
  // LATE-LABEL: tt.func @zz_later_wave_uses_earlier_constants
  // LATE: %[[LEFTC:.*]] = arith.constant dense<2.000000e+00>
  // LATE: arith.constant dense<3.000000e+00>
  // LATE: ttg.warp_specialize
  // LATE: default {
  // LATE: math.exp
  // LATE: partition0
  // LATE: math.sqrt
  // LATE: ttg.local_load
  // LATE: %[[FIRST:.*]] = arith.addf
  // LATE: ttg.warp_specialize
  // LATE: default {
  // LATE: arith.mulf %[[FIRST]], %[[LEFTC]]
  // LATE: math.exp
  // LATE: partition0
  // LATE: arith.constant dense<3.000000e+00>
  // LATE: arith.addf
  // LATE: math.sqrt
  // LATE: ttg.local_load
  // LATE: arith.addf
  // LATE: tt.reduce
  // LATE-NOT: ttg.warp_specialize
  // LATE: tt.return
  tt.func @zz_later_wave_uses_earlier_constants(%x: tensor<4x32xf32, #tile>, %y: tensor<4x32xf32, #tile>) -> tensor<4xf32, #row> {
    %left_constant = arith.constant dense<2.0> : tensor<4x32xf32, #tile>
    %right_constant = arith.constant dense<3.0> : tensor<4x32xf32, #tile>
    %x2 = arith.mulf %x, %x : tensor<4x32xf32, #tile>
    %nx2 = arith.negf %x2 : tensor<4x32xf32, #tile>
    %a0 = math.exp %nx2 : tensor<4x32xf32, #tile>
    %a = math.absf %a0 : tensor<4x32xf32, #tile>
    %y2 = arith.mulf %y, %y : tensor<4x32xf32, #tile>
    %ay2 = math.absf %y2 : tensor<4x32xf32, #tile>
    %b0 = math.sqrt %ay2 : tensor<4x32xf32, #tile>
    %b = math.absf %b0 : tensor<4x32xf32, #tile>
    %first = arith.addf %a, %b : tensor<4x32xf32, #tile>
    %left_scaled = arith.mulf %first, %left_constant : tensor<4x32xf32, #tile>
    %left = math.exp %left_scaled : tensor<4x32xf32, #tile>
    %right_shifted = arith.addf %first, %right_constant : tensor<4x32xf32, #tile>
    %right = math.sqrt %right_shifted : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    %reduced = "tt.reduce"(%joined) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    tt.return %reduced : tensor<4xf32, #row>
  }
}

