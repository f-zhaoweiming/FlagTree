// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' --verify-each | FileCheck %s --check-prefix=GUARD
// RUN: triton-opt %s -triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true max-transcendental-depth=0' --verify-each | FileCheck %s --check-prefix=FORCE

// A 4 + 5 transcendental chain retains its memory boundary by default.
// The override only changes profitability; fp16 rounding is still required.
// GUARD: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:false,
// GUARD-SAME: \22max_forwarded_transcendental_depth\22:9
// GUARD-SAME: \22private_arguments\22:[]
// GUARD-SAME: profitability:
// GUARD-LABEL: tt.func @private_rounding
// GUARD: tt.load
// GUARD: arith.truncf
// GUARD: tt.store
// GUARD: tt.load
// GUARD: arith.extf
// GUARD: tt.store
// GUARD: tt.return

// FORCE: module attributes {triton.intermediate_memory_elimination = "{\22accepted\22:true,
// FORCE-SAME: \22max_forwarded_transcendental_depth\22:9
// FORCE-SAME: \22private_arguments\22:[1]
// FORCE-LABEL: tt.func @private_rounding
// FORCE: tt.load
// FORCE-NOT: tt.load
// FORCE-NOT: tt.store
// FORCE: arith.truncf
// FORCE: arith.extf
// FORCE-NOT: tt.load
// FORCE-NOT: arith.select
// FORCE: tt.store
// FORCE-NEXT: tt.return

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
    %other = arith.constant dense<7.0> : tensor<256xf16>
    %math_0_0 = math.sin %value : tensor<256xf32>
    %math_0_1 = math.sin %math_0_0 : tensor<256xf32>
    %math_0_2 = math.sin %math_0_1 : tensor<256xf32>
    %math_0_3 = math.sin %math_0_2 : tensor<256xf32>
    %rounded0 = arith.truncf %math_0_3 : tensor<256xf32> to tensor<256xf16>
    tt.store %tmp_ptr, %rounded0, %mask : tensor<256x!tt.ptr<f16>>
    %read0 = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f16>>
    %extended0 = arith.extf %read0 : tensor<256xf16> to tensor<256xf32>
    %math_1_0 = math.sin %extended0 : tensor<256xf32>
    %math_1_1 = math.sin %math_1_0 : tensor<256xf32>
    %math_1_2 = math.sin %math_1_1 : tensor<256xf32>
    %math_1_3 = math.sin %math_1_2 : tensor<256xf32>
    %math_1_4 = math.sin %math_1_3 : tensor<256xf32>
    tt.store %out_ptr, %math_1_4, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
