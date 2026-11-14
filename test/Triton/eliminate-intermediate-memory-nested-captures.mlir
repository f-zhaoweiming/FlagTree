// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -split-input-file -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' --verify-each | FileCheck %s --check-prefix=GUARD
// RUN: triton-opt %s -split-input-file -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true max-transcendental-depth=0' --verify-each | FileCheck %s --check-prefix=FORCE

// Forwarding crosses a region capture, then a top-level sine chain. The region
// itself has no transcendental op, so its result must retain the forwarded
// depth and influence; otherwise 5+5 sine stages would be under-counted.
// GUARD: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}profitability:
// GUARD-LABEL: tt.func @capture_bridge
// GUARD: tt.load
// GUARD: tt.store
// GUARD: tt.load
// GUARD: scf.if
// GUARD: math.sin
// GUARD: tt.store
// GUARD: tt.return
module {
  tt.func @capture_bridge(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %p0 = math.sin %value : tensor<256xf32>
    %p1 = math.sin %p0 : tensor<256xf32>
    %p2 = math.sin %p1 : tensor<256xf32>
    %p3 = math.sin %p2 : tensor<256xf32>
    %p4 = math.sin %p3 : tensor<256xf32>
    tt.store %tmp_ptr, %p4, %mask : tensor<256x!tt.ptr<f32>>
    %other = arith.constant dense<7.0> : tensor<256xf32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %condition = arith.cmpi sgt, %n, %c256 : i32
    %bridged = scf.if %condition -> tensor<256xf32> {
      scf.yield %consumed : tensor<256xf32>
    } else {
      scf.yield %consumed : tensor<256xf32>
    }
    %c0 = math.sin %bridged : tensor<256xf32>
    %c1 = math.sin %c0 : tensor<256xf32>
    %c2 = math.sin %c1 : tensor<256xf32>
    %c3 = math.sin %c2 : tensor<256xf32>
    %c4 = math.sin %c3 : tensor<256xf32>
    tt.store %out_ptr, %c4, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The region produces a sine chain, then store/load forwards it into another
// top-level sine chain. The region result has an unknown producer depth and must
// make the forwarding plan conservatively unprofitable.
// GUARD: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}profitability:
// GUARD-LABEL: tt.func @region_producer
// GUARD: scf.if
// GUARD: math.sin
// GUARD: tt.store
// GUARD: tt.load
// GUARD: math.sin
// GUARD: tt.store
// GUARD: tt.return
module {
  tt.func @region_producer(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %condition = arith.cmpi sgt, %n, %c256 : i32
    %produced = scf.if %condition -> tensor<256xf32> {
      %p0 = math.sin %value : tensor<256xf32>
      %p1 = math.sin %p0 : tensor<256xf32>
      %p2 = math.sin %p1 : tensor<256xf32>
      %p3 = math.sin %p2 : tensor<256xf32>
      %p4 = math.sin %p3 : tensor<256xf32>
      scf.yield %p4 : tensor<256xf32>
    } else {
      scf.yield %value : tensor<256xf32>
    }
    tt.store %tmp_ptr, %produced, %mask : tensor<256x!tt.ptr<f32>>
    %other = arith.constant dense<7.0> : tensor<256xf32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %c0 = math.sin %consumed : tensor<256xf32>
    %c1 = math.sin %c0 : tensor<256xf32>
    %c2 = math.sin %c1 : tensor<256xf32>
    %c3 = math.sin %c2 : tensor<256xf32>
    %c4 = math.sin %c3 : tensor<256xf32>
    tt.store %out_ptr, %c4, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// FORCE: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// FORCE-LABEL: tt.func @capture_bridge
// FORCE: scf.if
// FORCE: math.sin
// FORCE: tt.store
// FORCE: tt.return
// FORCE: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// FORCE-LABEL: tt.func @region_producer
// FORCE: scf.if
// FORCE: math.sin
// FORCE: tt.store
// FORCE: tt.return

// -----

// Unknown producer taint must survive a second pure region capture before the
// store/load, otherwise the final sine chain is under-counted.
// GUARD: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}profitability:
// GUARD-LABEL: tt.func @transit_capture
// GUARD: scf.if
// GUARD: scf.if
// GUARD: tt.store
// GUARD: tt.load
// GUARD: math.sin
// GUARD: tt.store
// GUARD: tt.return
module {
  tt.func @transit_capture(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %condition = arith.cmpi sgt, %n, %c256 : i32
    %produced = scf.if %condition -> tensor<256xf32> {
      %p0 = math.sin %value : tensor<256xf32>
      %p1 = math.sin %p0 : tensor<256xf32>
      %p2 = math.sin %p1 : tensor<256xf32>
      %p3 = math.sin %p2 : tensor<256xf32>
      %p4 = math.sin %p3 : tensor<256xf32>
      scf.yield %p4 : tensor<256xf32>
    } else {
      scf.yield %value : tensor<256xf32>
    }
    %bridged = scf.if %condition -> tensor<256xf32> {
      scf.yield %produced : tensor<256xf32>
    } else {
      scf.yield %produced : tensor<256xf32>
    }
    tt.store %tmp_ptr, %bridged, %mask : tensor<256x!tt.ptr<f32>>
    %other = arith.constant dense<7.0> : tensor<256xf32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %c0 = math.sin %consumed : tensor<256xf32>
    %c1 = math.sin %c0 : tensor<256xf32>
    %c2 = math.sin %c1 : tensor<256xf32>
    %c3 = math.sin %c2 : tensor<256xf32>
    %c4 = math.sin %c3 : tensor<256xf32>
    tt.store %out_ptr, %c4, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// FORCE: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// FORCE-LABEL: tt.func @transit_capture
// FORCE: scf.if
// FORCE: scf.if
// FORCE: math.sin
// FORCE: tt.store
// FORCE: tt.return
