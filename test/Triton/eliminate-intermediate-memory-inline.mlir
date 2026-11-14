// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s --inline --canonicalize --cse --symbol-dce --triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' --verify-each | FileCheck %s

// CHECK: triton.intermediate_memory_elimination = "{\22accepted\22:true,{{.*}}\22eliminated_stores\22:1,\22forwarded_loads\22:1,
// CHECK-LABEL: tt.func @composed
// CHECK-NOT: tt.splat %arg1
// CHECK-NOT: tt.call
// CHECK: %[[VALUE:.*]] = tt.load
// CHECK: %[[ROUNDED:.*]] = arith.truncf %[[VALUE]]
// CHECK-NOT: tt.store
// CHECK-NOT: tt.load
// CHECK: %[[EXTENDED:.*]] = arith.extf %[[ROUNDED]]
// CHECK: tt.store {{.*}}, %[[EXTENDED]]
// CHECK-NOT: tt.call
// CHECK-NOT: tt.store
// CHECK: tt.return
module {
  tt.func private @produce(%input: tensor<256x!tt.ptr<f32>>, %temporary: tensor<256x!tt.ptr<f16>>) {
    %value = tt.load %input : tensor<256x!tt.ptr<f32>>
    %rounded = arith.truncf %value : tensor<256xf32> to tensor<256xf16>
    tt.store %temporary, %rounded : tensor<256x!tt.ptr<f16>>
    tt.return
  }
  tt.func private @consume(%temporary: tensor<256x!tt.ptr<f16>>, %output: tensor<256x!tt.ptr<f32>>) {
    %value = tt.load %temporary : tensor<256x!tt.ptr<f16>>
    %extended = arith.extf %value : tensor<256xf16> to tensor<256xf32>
    tt.store %output, %extended : tensor<256x!tt.ptr<f32>>
    tt.return
  }
  tt.func @composed(%input: !tt.ptr<f32>, %temporary: !tt.ptr<f16>, %output: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c256 : i32
    %tile = tt.splat %base : i32 -> tensor<256xi32>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %offsets = arith.addi %tile, %range : tensor<256xi32>
    %in_base = tt.splat %input : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %in_ptr = tt.addptr %in_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %tmp_base = tt.splat %temporary : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %tmp_ptr = tt.addptr %tmp_base, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %out_base = tt.splat %output : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.call @produce(%in_ptr, %tmp_ptr) : (tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f16>>) -> ()
    tt.call @consume(%tmp_ptr, %out_ptr) : (tensor<256x!tt.ptr<f16>>, tensor<256x!tt.ptr<f32>>) -> ()
    tt.return
  }
}
