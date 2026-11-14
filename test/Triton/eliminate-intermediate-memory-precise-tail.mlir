// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -split-input-file -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' --verify-each | FileCheck %s

// The new proof is mask-only: precise operations remain scheduling fences.
// All accepted negative cases must actually forward the private load. A whole
// pass rejection must not masquerade as successful select-retention coverage.

// Changing inactive sqrt/div inputs must not erase the fp16 store rounding.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:1,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_fp16_rounding
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK: %[[ROUND:.*]] = arith.truncf %[[VALUE]] : tensor<256xf32> to tensor<256xf16>
// CHECK-NOT: tt.store
// CHECK-NOT: tt.load
// CHECK-NOT: arith.select
// CHECK: %[[EXT:.*]] = arith.extf %[[ROUND]] : tensor<256xf16> to tensor<256xf32>
// CHECK: %[[SQRT:.*]] = tt.precise_sqrt %[[EXT]] : tensor<256xf32>
// CHECK: %[[DIV:.*]] = tt.precise_divf %[[SQRT]], %[[EXT]] : tensor<256xf32>
// CHECK-NOT: arith.select
// CHECK: tt.store {{.*}}, %[[DIV]],
// CHECK: tt.return
module {
  tt.func @precise_fp16_rounding(%input: !tt.ptr<f32>, %private: !tt.ptr<f16>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf16>
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
    %tmp_base = tt.splat %private : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    %rounded = arith.truncf %value : tensor<256xf32> to tensor<256xf16>
    tt.store %tmp_ptr, %rounded, %mask : tensor<256x!tt.ptr<f16>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f16>>
    %extended = arith.extf %consumed : tensor<256xf16> to tensor<256xf32>
    %root = tt.precise_sqrt %extended : tensor<256xf32>
    %quotient = tt.precise_divf %root, %extended : tensor<256xf32>
    tt.store %out_ptr, %quotient, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Every use is safe, including a divisor that becomes zero in discarded lanes.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:1,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_divisor_fanout
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK-NOT: arith.select
// CHECK: %[[DIV:.*]] = tt.precise_divf {{.*}}, %[[VALUE]] : tensor<256xf32>
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[VALUE]] : tensor<256xf32>
// CHECK: %[[SUM:.*]] = arith.addf %[[DIV]], %[[ROOT]]
// CHECK-NOT: arith.select
// CHECK: tt.store {{.*}}, %[[SUM]],
// CHECK: tt.return
module {
  tt.func @precise_divisor_fanout(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %quotient = tt.precise_divf %one, %consumed : tensor<256xf32>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %result = arith.addf %quotient, %root : tensor<256xf32>
    tt.store %out_ptr, %result, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Even a pure extern is not a proven lane-local builtin; its library can be overridden.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_then_libdevice_expf
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: %[[EXTERNAL:.*]] = tt.extern_elementwise %[[ROOT]] {{.*}}pure = true, symbol = "__nv_expf"
// CHECK: tt.store {{.*}}, %[[EXTERNAL]],
// CHECK: tt.return
module {
  tt.func @precise_then_libdevice_expf(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %external = tt.extern_elementwise %root {libname = "", libpath = "", pure = true, symbol = "__nv_expf"} : (tensor<256xf32>) -> tensor<256xf32>
    tt.store %out_ptr, %external, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Even a pure extern is not a proven lane-local builtin; its library can be overridden.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_then_unknown_extern
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: %[[EXTERNAL:.*]] = tt.extern_elementwise %[[ROOT]] {{.*}}pure = true, symbol = "custom_lane_function"
// CHECK: tt.store {{.*}}, %[[EXTERNAL]],
// CHECK: tt.return
module {
  tt.func @precise_then_unknown_extern(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %external = tt.extern_elementwise %root {libname = "", libpath = "", pure = true, symbol = "custom_lane_function"} : (tensor<256xf32>) -> tensor<256xf32>
    tt.store %out_ptr, %external, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Pure Elementwise assembly can exchange two logical lanes.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_then_packed_asm
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: %[[PACKED:.*]] = tt.elementwise_inline_asm {{.*}}packed_element = 2 : i32, pure = true} %[[ROOT]]
// CHECK: tt.store {{.*}}, %[[PACKED]],
// CHECK: tt.return
module {
  tt.func @precise_then_packed_asm(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %packed = tt.elementwise_inline_asm "mov.b32 $0, $3; mov.b32 $1, $2;" {constraints = "=f,=f,f,f", packed_element = 2 : i32, pure = true} %root : tensor<256xf32> -> tensor<256xf32>
    tt.store %out_ptr, %packed, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A successful same-mask store use cannot hide a second cross-lane observer.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_mixed_reduction
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: tt.store {{.*}}, %[[ROOT]],
// CHECK: %[[SUM:.*]] = "tt.reduce"(%[[ROOT]])
// CHECK: %[[TOTAL:.*]] = tt.splat %[[SUM]]
// CHECK: tt.store {{.*}}, %[[TOTAL]],
// CHECK: tt.return
module {
  tt.func @precise_mixed_reduction(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    tt.store %out_ptr, %root, %mask : tensor<256x!tt.ptr<f32>>
    %sum = "tt.reduce" (%root) ({
    ^bb0(%left: f32, %right: f32):
      %add = arith.addf %left, %right : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %total = tt.splat %sum : f32 -> tensor<256xf32>
    %aux_base = tt.splat %aux : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %aux_ptr = tt.addptr %aux_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %aux_ptr, %total, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A computed select predicate remains outside the conservative proof.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_then_predicate
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: %[[PRED:.*]] = arith.cmpf ogt, %[[ROOT]],
// CHECK: %[[CHOICE:.*]] = arith.select %[[PRED]], %[[ROOT]],
// CHECK: tt.store {{.*}}, %[[CHOICE]],
// CHECK: tt.return
module {
  tt.func @precise_then_predicate(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %predicate = arith.cmpf ogt, %root, %one : tensor<256xf32>
    %choice = arith.select %predicate, %root, %one : tensor<256xi1>, tensor<256xf32>
    tt.store %out_ptr, %choice, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The extra root is readonly so the pass accepts it, but addresses cannot lose the select.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_then_readonly_address
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: %[[INDEX:.*]] = arith.fptosi %[[ROOT]]
// CHECK: %[[PTR:.*]] = tt.addptr {{.*}}, %[[INDEX]]
// CHECK: %[[GATHER:.*]] = tt.load %[[PTR]],
// CHECK: tt.store {{.*}}, %[[GATHER]],
// CHECK: tt.return
module {
  tt.func @precise_then_readonly_address(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %index = arith.fptosi %root : tensor<256xf32> to tensor<256xi32>
    %aux_base = tt.splat %aux : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %aux_ptr = tt.addptr %aux_base, %index : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %gathered = tt.load %aux_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %out_ptr, %gathered, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// An unmasked terminal store observes the load other after precise arithmetic.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,{{.*}}\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @precise_then_unmasked_store
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[SELECT]] : tensor<256xf32>
// CHECK: tt.store {{.*}}, %[[ROOT]] : tensor<256x!tt.ptr<f32>>
// CHECK: tt.return
module {
  tt.func @precise_then_unmasked_store(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    tt.store %out_ptr, %root : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Unknown effects are rejected before forwarding, not a local select negative.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}unresolved call, control flow, atomic or unknown side effect
// CHECK-LABEL: tt.func @impure_extern_rejects_elimination
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK: tt.store {{.*}}, %[[VALUE]],
// CHECK: %[[CONSUMED:.*]] = tt.load
// CHECK: %[[ROOT:.*]] = tt.precise_sqrt %[[CONSUMED]]
// CHECK: %[[EXTERNAL:.*]] = tt.extern_elementwise %[[ROOT]] {{.*}}pure = false
// CHECK: tt.store {{.*}}, %[[EXTERNAL]],
// CHECK: tt.return
module {
  tt.func @impure_extern_rejects_elimination(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %one = arith.constant dense<1.0> : tensor<256xf32>
    %other = arith.constant dense<7.0> : tensor<256xf32>
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
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<f32>>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %root = tt.precise_sqrt %consumed : tensor<256xf32>
    %external = tt.extern_elementwise %root {libname = "", libpath = "", pure = false, symbol = "__nv_expf"} : (tensor<256xf32>) -> tensor<256xf32>
    tt.store %out_ptr, %external, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
