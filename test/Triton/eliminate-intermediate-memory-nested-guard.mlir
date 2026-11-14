// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' --verify-each | FileCheck %s --check-prefix=GUARD
// RUN: triton-opt %s -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true max-transcendental-depth=0' --verify-each | FileCheck %s --check-prefix=FORCE

// The forwarding and the transcendental math are in different SSA scopes.
// Depth accounting is entry-block based, so the guard must conservatively
// reject this plan rather than under-count the join. The body is unchanged.
// GUARD: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}nested region with transcendental operations
// GUARD-LABEL: tt.func @nested_transcendental
// GUARD: tt.load
// GUARD: tt.store
// GUARD: tt.load
// GUARD: scf.if
// GUARD: math.sin
// GUARD: tt.store
// GUARD: tt.return

// A zero threshold disables only the profitability guard: forwarding still
// happens, the private store disappears, and the select is retained because
// the nested region is outside the mask-select proof.
// FORCE: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// FORCE-LABEL: tt.func @nested_transcendental
// FORCE: %[[VALUE:.*]] = tt.load
// FORCE-NOT: tt.store
// FORCE-NOT: tt.load
// FORCE: %[[SELECT:.*]] = arith.select
// FORCE: scf.if
// FORCE: math.sin
// FORCE: tt.store
// FORCE: tt.return
module {
  tt.func @nested_transcendental(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c256 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c256 : i32
    %base_tile = tt.splat %base : i32 -> tensor<256xi32>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %offsets = arith.addi %base_tile, %range : tensor<256xi32>
    %n_tile = tt.splat %n : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %offsets, %n_tile : tensor<256xi32>
    %in_base = tt.splat %input : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %in_ptr = tt.addptr %in_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %tmp_base = tt.splat %private : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %other = arith.constant dense<7.0> : tensor<256xf32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %condition = arith.cmpi sgt, %n, %c256 : i32
    %result = scf.if %condition -> tensor<256xf32> {
      %sine = math.sin %consumed : tensor<256xf32>
      scf.yield %sine : tensor<256xf32>
    } else {
      scf.yield %consumed : tensor<256xf32>
    }
    tt.store %out_ptr, %result, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
