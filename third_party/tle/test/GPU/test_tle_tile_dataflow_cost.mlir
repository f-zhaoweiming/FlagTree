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

// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="sync-cost=1000000000" | FileCheck %s --check-prefixes=SERIAL,LEGALITY
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="shared-bandwidth=0.000001" | FileCheck %s --check-prefixes=SERIAL,LEGALITY
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="min-speedup=1000000" | FileCheck %s --check-prefixes=SERIAL,LEGALITY
// RUN: sed 's/cuda:89/cuda:100/g' %s | triton-opt -triton-tle-tile-dataflow-schedule | FileCheck %s --check-prefixes=SERIAL,LEGALITY
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="force=true sync-cost=1000000000 shared-bandwidth=0.000001 min-speedup=1000000" | FileCheck %s --check-prefixes=FORCED,LEGALITY
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="min-speedup=1000000" | FileCheck %s --check-prefix=ORIENTATION
// RUN: triton-opt %s -triton-tle-tile-dataflow-schedule="sync-cost=0 overlap=1 shared-bandwidth=1000000000 min-speedup=1" | FileCheck %s --check-prefix=LOW-COST
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="min-branch-ops=0" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="max-partitions=0" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="max-shared-bytes=0" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="min-speedup=0.9" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="min-speedup=nan" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="min-speedup=inf" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="sync-cost=-1" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="sync-cost=nan" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="shared-bandwidth=0" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="shared-bandwidth=inf" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="overlap=-0.1" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="overlap=1.01" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-opt %s -triton-tle-tile-dataflow-schedule="force=true overlap=nan" 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: error: invalid tile dataflow limits or cost-model parameters

// Increasing each independent cost or required benefit can reject a legal
// schedule. An architecture without a resource model also stays serial. Force
// bypasses profitability, but never the memory ordering constraint in the
// second function. No operator-name pattern is used.
#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #tile}>

// SERIAL: module attributes
// SERIAL-SAME: tle.tile_dataflow.cost_model =
// FORCED: module attributes
// FORCED-SAME: tle.tile_dataflow.cost_model =
// ORIENTATION: \22function\22:\22prefer_low_transfer_worker\22
// ORIENTATION-SAME: \22transfer_bytes\22:32
// ORIENTATION-SAME: \22worker_operand\22:0
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:89"} {
  // SERIAL-LABEL: tt.func @cost_independent_branches
  // SERIAL-NOT: ttg.warp_specialize
  // SERIAL: tt.return
  // FORCED-LABEL: tt.func @cost_independent_branches
  // FORCED: ttg.warp_specialize
  // FORCED: partition0
  // FORCED: ttg.local_store
  // FORCED: ttg.local_load
  // FORCED: tt.return
  tt.func @cost_independent_branches(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>) -> tensor<4xf32, #row> {
    %ap = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<4x256x!tt.ptr<f32>, #tile>
    %a = tt.load %ap : tensor<4x256x!tt.ptr<f32>, #tile>
    %a2 = arith.mulf %a, %a : tensor<4x256xf32, #tile>
    %aneg = arith.negf %a2 : tensor<4x256xf32, #tile>
    %aexp = math.exp %aneg : tensor<4x256xf32, #tile>
    %left = "tt.reduce"(%aexp) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x256xf32, #tile>) -> tensor<4xf32, #row>
    %bp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<4x256x!tt.ptr<f32>, #tile>
    %b = tt.load %bp : tensor<4x256x!tt.ptr<f32>, #tile>
    %b2 = arith.mulf %b, %b : tensor<4x256xf32, #tile>
    %bneg = arith.negf %b2 : tensor<4x256xf32, #tile>
    %bexp = math.exp %bneg : tensor<4x256xf32, #tile>
    %right = "tt.reduce"(%bexp) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x256xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    tt.return %joined : tensor<4xf32, #row>
  }

  // LEGALITY-LABEL: tt.func @force_preserves_alias_hazards
  // LEGALITY-NOT: ttg.warp_specialize
  // LEGALITY: tt.return
  tt.func @force_preserves_alias_hazards(%ptr: !tt.ptr<f32>, %alias: !tt.ptr<f32>, %value: tensor<4x256xf32, #tile>) -> tensor<4x256xf32, #tile> {
    %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x256x!tt.ptr<f32>, #tile>
    %a = tt.splat %alias : !tt.ptr<f32> -> tensor<4x256x!tt.ptr<f32>, #tile>
    %before = tt.load %p : tensor<4x256x!tt.ptr<f32>, #tile>
    %left = math.exp %before : tensor<4x256xf32, #tile>
    tt.store %a, %value : tensor<4x256x!tt.ptr<f32>, #tile>
    %after = tt.load %p : tensor<4x256x!tt.ptr<f32>, #tile>
    %right = math.exp %after : tensor<4x256xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x256xf32, #tile>
    tt.return %joined : tensor<4x256xf32, #tile>
  }

  // Operand 1 depends on an existing distributed tensor, while operand 0 can
  // load its own input from a uniform pointer. Choose operand 0 for the worker:
  // its result transfers 32 bytes, versus 8224 with the extra tensor capture.
  // LOW-COST-LABEL: tt.func @prefer_low_transfer_worker
  // LOW-COST: ttg.local_alloc {{.*}}: () -> !ttg.memdesc<4xf32,
  // LOW-COST-NEXT: {{.*}}ttg.warp_specialize
  // LOW-COST: default {
  // LOW-COST-NOT: tt.load
  // LOW-COST: math.exp
  // LOW-COST: partition0
  // LOW-COST: tt.load
  // LOW-COST: tt.reduce
  // LOW-COST: ttg.local_store
  // LOW-COST: ttg.local_load
  // LOW-COST: tt.return
  tt.func @prefer_low_transfer_worker(%ptr: !tt.ptr<f32>, %input: tensor<4x256xf32, #tile>) -> tensor<4xf32, #row> {
    %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x256x!tt.ptr<f32>, #tile>
    %loaded = tt.load %p : tensor<4x256x!tt.ptr<f32>, #tile>
    %square_left = arith.mulf %loaded, %loaded : tensor<4x256xf32, #tile>
    %neg_left = arith.negf %square_left : tensor<4x256xf32, #tile>
    %exp_left = math.exp %neg_left : tensor<4x256xf32, #tile>
    %left = "tt.reduce"(%exp_left) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x256xf32, #tile>) -> tensor<4xf32, #row>
    %square_right = arith.mulf %input, %input : tensor<4x256xf32, #tile>
    %neg_right = arith.negf %square_right : tensor<4x256xf32, #tile>
    %exp_right = math.exp %neg_right : tensor<4x256xf32, #tile>
    %right = "tt.reduce"(%exp_right) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x256xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    tt.return %joined : tensor<4xf32, #row>
  }

}
