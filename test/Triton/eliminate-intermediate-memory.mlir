// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -split-input-file -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' | FileCheck %s

// The forwarded value is the already rounded fp16 store operand. Invalid
// lanes cannot reach the same-mask store, so their distinct `other` is dead.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:1,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @private_rounding
// CHECK-NOT: tt.splat %arg1
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK: %[[ROUNDED:.*]] = arith.truncf %[[VALUE]] : tensor<256xf32> to tensor<256xf16>
// CHECK-NOT: tt.store
// CHECK-NOT: tt.load
// CHECK-NOT: arith.select
// CHECK: %[[EXT:.*]] = arith.extf %[[ROUNDED]]
// CHECK: tt.store {{.*}}, %[[EXT]],
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
// CHECK: tt.return
module {
  tt.func @private_rounding(%input: !tt.ptr<f32>, %private: !tt.ptr<f16>, %out: !tt.ptr<f32>, %n: i32) {
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
    %tmp_base = tt.splat %private : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    %rounded = arith.truncf %value : tensor<256xf32> to tensor<256xf16>
    tt.store %tmp_ptr, %rounded, %mask : tensor<256x!tt.ptr<f16>>
    %other = arith.constant dense<7.0> : tensor<256xf16>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f16>>
    %extended = arith.extf %consumed : tensor<256xf16> to tensor<256xf32>
    tt.store %out_ptr, %extended, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// An ordinary intermediate remains externally observable after the kernel returns.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:1,\22eliminated_stores\22:0,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @retained_intermediate
// CHECK: %[[RETAINED:.*]] = tt.splat %arg4
// CHECK: %[[PTR:.*]] = tt.addptr %[[RETAINED]],
// CHECK: %[[R:.*]] = arith.truncf
// CHECK: tt.store %[[PTR]], %[[R]],
// CHECK-NOT: tt.load
// CHECK-NOT: arith.select
// CHECK: %[[EXT:.*]] = arith.extf %[[R]]
// CHECK: tt.store {{.*}}, %[[EXT]],
// CHECK: tt.return
module {
  tt.func @retained_intermediate(%input: !tt.ptr<f32>, %private: !tt.ptr<f16>, %out: !tt.ptr<f32>, %n: i32, %retained: !tt.ptr<f16>) {
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
    %tmp_base = tt.splat %retained : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    %rounded = arith.truncf %value : tensor<256xf32> to tensor<256xf16>
    tt.store %tmp_ptr, %rounded, %mask : tensor<256x!tt.ptr<f16>>
    %other = arith.constant dense<7.0> : tensor<256xf16>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f16>>
    %extended = arith.extf %consumed : tensor<256xf16> to tensor<256xf32>
    tt.store %out_ptr, %extended, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}consumer address differs from producer CTA address
// CHECK-LABEL: tt.func @cross_cta_neighbor
// CHECK: tt.store
// CHECK: tt.addptr
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return
module {
  tt.func @cross_cta_neighbor(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %shift = arith.constant dense<256> : tensor<256xi32>
    %neighbor_ptr = tt.addptr %tmp_ptr, %shift : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %consumed = tt.load %neighbor_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %out_ptr, %consumed, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}producer and consumer masks or element types differ
// CHECK-LABEL: tt.func @different_mask
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return
module {
  tt.func @different_mask(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %one = arith.constant dense<1> : tensor<256xi32>
    %other_limit = arith.addi %n_tile, %one : tensor<256xi32>
    %other_mask = arith.cmpi slt, %offsets, %other_limit : tensor<256xi32>
    %consumed = tt.load %tmp_ptr, %other_mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %out_ptr, %consumed, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,
// CHECK-LABEL: tt.func @atomic_effect
// CHECK: tt.atomic_rmw
// CHECK: tt.return
module {
  tt.func @atomic_effect(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %consumed = tt.atomic_rmw fadd, relaxed, gpu, %tmp_ptr, %value, %mask : (tensor<256x!tt.ptr<f32>>, tensor<256xf32>, tensor<256xi1>) -> tensor<256xf32>
    tt.store %out_ptr, %consumed, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}pointer escapes through an unsupported operation
// CHECK-LABEL: tt.func @private_pointer_escape
// CHECK: tt.ptr_to_int
// CHECK: tt.return
module {
  tt.func @private_pointer_escape(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %address = tt.ptr_to_int %private : !tt.ptr<f32> -> i64
    %address_f32 = arith.uitofp %address : i64 to f32
    %consumed = tt.splat %address_f32 : f32 -> tensor<256xf32>
    tt.store %out_ptr, %consumed, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}unsupported load or nested memory access
// CHECK-LABEL: tt.func @region_memory
// CHECK: scf.if
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return
module {
  tt.func @region_memory(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %condition = arith.cmpi sgt, %n, %c256 : i32
    scf.if %condition {
      %consumed = tt.load %tmp_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
      tt.store %out_ptr, %consumed, %mask : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,{{.*}}private intermediate is read before it is written
// CHECK-LABEL: tt.func @private_read_before_write
// CHECK: tt.load
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return
module {
  tt.func @private_read_before_write(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %consumed = tt.load %tmp_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    tt.store %out_ptr, %consumed, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Every branch ends in a same-mask store; both uses can share the raw value.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:1,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @same_mask_fanout
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK-NOT: arith.select
// CHECK: %[[DOUBLE:.*]] = arith.addf %[[VALUE]], %[[VALUE]]
// CHECK: %[[NEGATIVE:.*]] = arith.negf %[[VALUE]]
// CHECK: tt.store {{.*}}, %[[DOUBLE]],
// CHECK: tt.store {{.*}}, %[[NEGATIVE]],
// CHECK: tt.return
module {
  tt.func @same_mask_fanout(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
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
    %aux_base = tt.splat %aux : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %aux_ptr = tt.addptr %aux_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %doubled = arith.addf %consumed, %consumed : tensor<256xf32>
    %negated = arith.negf %consumed : tensor<256xf32>
    tt.store %out_ptr, %doubled, %mask : tensor<256x!tt.ptr<f32>>
    tt.store %aux_ptr, %negated, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Producer/load masks match, but the final store exposes the consumer other.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @wider_output_mask
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[WIDER:.*]] = arith.cmpi
// CHECK: tt.store {{.*}}, %[[SELECT]], %[[WIDER]]
// CHECK: tt.return
module {
  tt.func @wider_output_mask(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    %one = arith.constant dense<1> : tensor<256xi32>
    %wider_limit = arith.addi %n_tile, %one : tensor<256xi32>
    %wider_mask = arith.cmpi slt, %offsets, %wider_limit : tensor<256xi32>
    tt.store %out_ptr, %consumed, %wider_mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// An unmasked output also makes the inactive consumer lanes observable.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @unmasked_output
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: tt.store {{.*}}, %[[SELECT]] :
// CHECK: tt.return
module {
  tt.func @unmasked_output(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
    tt.store %out_ptr, %consumed : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A safe store branch must not hide a second branch that reduces inactive lanes.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @mixed_fanout_reduction
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: arith.addf %[[SELECT]], %[[SELECT]]
// CHECK: tt.store
// CHECK: "tt.reduce"(%[[SELECT]])
// CHECK: tt.store
// CHECK: tt.return
module {
  tt.func @mixed_fanout_reduction(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %aux: !tt.ptr<f32>) {
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
    %aux_base = tt.splat %aux : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %aux_ptr = tt.addptr %aux_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %doubled = arith.addf %consumed, %consumed : tensor<256xf32>
    tt.store %out_ptr, %doubled, %mask : tensor<256x!tt.ptr<f32>>
    %sum = "tt.reduce" (%consumed) ({
    ^bb0(%left: f32, %right: f32):
      %add = arith.addf %left, %right : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %total = tt.splat %sum : f32 -> tensor<256xf32>
    tt.store %aux_ptr, %total, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Pure nested computations are fusible, but are outside the select-elision proof.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @pure_nested_control_flow
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: scf.if
// CHECK: arith.addf %[[SELECT]], %[[SELECT]]
// CHECK: scf.yield %[[SELECT]]
// CHECK: tt.store
// CHECK: tt.return
module {
  tt.func @pure_nested_control_flow(%input: !tt.ptr<f32>, %private: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
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
      %doubled = arith.addf %consumed, %consumed : tensor<256xf32>
      scf.yield %doubled : tensor<256xf32>
    } else {
      scf.yield %consumed : tensor<256xf32>
    }
    tt.store %out_ptr, %result, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The load other=1 prevents inactive-lane division by zero, even for Pure remainder ops.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @inactive_divsi_divisor
// CHECK: %[[ONE:.*]] = arith.constant dense<1>
// CHECK: %[[SELECT:.*]] = arith.select {{.*}}, %[[ONE]]
// CHECK: %[[RESULT:.*]] = arith.divsi {{.*}}, %[[SELECT]]
// CHECK: tt.store {{.*}}, %[[RESULT]],
// CHECK: tt.return
module {
  tt.func @inactive_divsi_divisor(%input: !tt.ptr<i32>, %private: !tt.ptr<i32>, %out: !tt.ptr<i32>, %n: i32) {
    %c256 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c256 : i32
    %base_tile = tt.splat %base : i32 -> tensor<256xi32>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %offsets = arith.addi %base_tile, %range : tensor<256xi32>
    %n_tile = tt.splat %n : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %offsets, %n_tile : tensor<256xi32>
    %in_base = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %in_ptr = tt.addptr %in_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %out_base = tt.splat %out : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %tmp_base = tt.splat %private : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %zero = arith.constant dense<0> : tensor<256xi32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<i32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<i32>>
    %other = arith.constant dense<1> : tensor<256xi32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<i32>>
    %numerator = arith.constant dense<17> : tensor<256xi32>
    %result = arith.divsi %numerator, %consumed : tensor<256xi32>
    tt.store %out_ptr, %result, %mask : tensor<256x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// The load other=1 prevents inactive-lane division by zero, even for Pure remainder ops.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @inactive_remsi_divisor
// CHECK: %[[ONE:.*]] = arith.constant dense<1>
// CHECK: %[[SELECT:.*]] = arith.select {{.*}}, %[[ONE]]
// CHECK: %[[RESULT:.*]] = arith.remsi {{.*}}, %[[SELECT]]
// CHECK: tt.store {{.*}}, %[[RESULT]],
// CHECK: tt.return
module {
  tt.func @inactive_remsi_divisor(%input: !tt.ptr<i32>, %private: !tt.ptr<i32>, %out: !tt.ptr<i32>, %n: i32) {
    %c256 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c256 : i32
    %base_tile = tt.splat %base : i32 -> tensor<256xi32>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %offsets = arith.addi %base_tile, %range : tensor<256xi32>
    %n_tile = tt.splat %n : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %offsets, %n_tile : tensor<256xi32>
    %in_base = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %in_ptr = tt.addptr %in_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %out_base = tt.splat %out : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %tmp_base = tt.splat %private : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %zero = arith.constant dense<0> : tensor<256xi32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<i32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<i32>>
    %other = arith.constant dense<1> : tensor<256xi32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<i32>>
    %numerator = arith.constant dense<17> : tensor<256xi32>
    %result = arith.remsi %numerator, %consumed : tensor<256xi32>
    tt.store %out_ptr, %result, %mask : tensor<256x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// The load other=1 prevents inactive-lane division by zero, even for Pure remainder ops.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:0,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @inactive_remui_divisor
// CHECK: %[[ONE:.*]] = arith.constant dense<1>
// CHECK: %[[SELECT:.*]] = arith.select {{.*}}, %[[ONE]]
// CHECK: %[[RESULT:.*]] = arith.remui {{.*}}, %[[SELECT]]
// CHECK: tt.store {{.*}}, %[[RESULT]],
// CHECK: tt.return
module {
  tt.func @inactive_remui_divisor(%input: !tt.ptr<i32>, %private: !tt.ptr<i32>, %out: !tt.ptr<i32>, %n: i32) {
    %c256 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c256 : i32
    %base_tile = tt.splat %base : i32 -> tensor<256xi32>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %offsets = arith.addi %base_tile, %range : tensor<256xi32>
    %n_tile = tt.splat %n : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %offsets, %n_tile : tensor<256xi32>
    %in_base = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %in_ptr = tt.addptr %in_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %out_base = tt.splat %out : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %tmp_base = tt.splat %private : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %zero = arith.constant dense<0> : tensor<256xi32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<i32>>
    tt.store %tmp_ptr, %value, %mask : tensor<256x!tt.ptr<i32>>
    %other = arith.constant dense<1> : tensor<256xi32>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<i32>>
    %numerator = arith.constant dense<17> : tensor<256xi32>
    %result = arith.remui %numerator, %consumed : tensor<256xi32>
    tt.store %out_ptr, %result, %mask : tensor<256x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// bf16 storage rounding must survive forwarding just like fp16.
// CHECK: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,\22eliminated_mask_selects\22:1,\22eliminated_stores\22:1,{{.*}}\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @bf16_rounding
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK: %[[ROUND:.*]] = arith.truncf %[[VALUE]] : tensor<256xf32> to tensor<256xbf16>
// CHECK-NOT: tt.store
// CHECK-NOT: tt.load
// CHECK-NOT: arith.select
// CHECK: %[[EXT:.*]] = arith.extf %[[ROUND]] : tensor<256xbf16> to tensor<256xf32>
// CHECK: tt.store {{.*}}, %[[EXT]],
// CHECK: tt.return
module {
  tt.func @bf16_rounding(%input: !tt.ptr<f32>, %private: !tt.ptr<bf16>, %out: !tt.ptr<f32>, %n: i32) {
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
    %tmp_base = tt.splat %private : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %zero = arith.constant dense<0.0> : tensor<256xf32>
    %value = tt.load %in_ptr, %mask, %zero : tensor<256x!tt.ptr<f32>>
    %rounded = arith.truncf %value : tensor<256xf32> to tensor<256xbf16>
    tt.store %tmp_ptr, %rounded, %mask : tensor<256x!tt.ptr<bf16>>
    %other = arith.constant dense<7.0> : tensor<256xbf16>
    %consumed = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<bf16>>
    %extended = arith.extf %consumed : tensor<256xbf16> to tensor<256xf32>
    tt.store %out_ptr, %extended, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
