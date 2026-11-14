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

// RUN: triton-opt %s -allow-unregistered-dialect -triton-tle-tile-dataflow-schedule="min-branch-ops=1 max-partitions=2 force=true" -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=NOFUSE < %t
// RUN: FileCheck %s --check-prefix=ALLOCS < %t
// RUN: triton-opt %t -allow-unregistered-dialect -triton-tle-tile-dataflow-schedule="min-branch-ops=1 max-partitions=2 force=true" -o %t.again
// RUN: diff %t %t.again

// The pass schedules tile DAG branches, independent of named operator patterns.
// These tests require executable warp-specialized regions and shared-memory
// communication, rather than annotations that only describe a proposed plan.

#tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #tile}>
#column = #ttg.slice<{dim = 0, parent = #tile}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {

  // CHECK-LABEL: tt.func @independent_load_reduce_branches
  // CHECK: ttg.local_alloc
  // CHECK: ttg.warp_specialize
  // CHECK: default {
  // CHECK: tt.load
  // CHECK: arith.mulf
  // CHECK: tt.reduce
  // CHECK: ttg.warp_yield
  // CHECK: partition0
  // CHECK: tt.load
  // CHECK: math.exp
  // CHECK: tt.reduce
  // CHECK: ttg.local_store
  // CHECK: ttg.warp_return
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @independent_load_reduce_branches(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>) -> tensor<4xf32, #row> {
    %ap = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %a = tt.load %ap : tensor<4x32x!tt.ptr<f32>, #tile>
    %squared = arith.mulf %a, %a : tensor<4x32xf32, #tile>
    %left = "tt.reduce"(%squared) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %bp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %b = tt.load %bp : tensor<4x32x!tt.ptr<f32>, #tile>
    %exponent = math.exp %b : tensor<4x32xf32, #tile>
    %right = "tt.reduce"(%exponent) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    tt.return %joined : tensor<4xf32, #row>
  }

  // A distributed tensor cannot be passed as a uniform worker capture. Stage
  // the shared DAG input explicitly and load it using the worker layout.
  // ALLOCS-LABEL: tt.func @shared_tensor_input
  // ALLOCS: ttg.local_alloc %arg0
  // ALLOCS-NOT: ttg.local_alloc
  // ALLOCS: tt.return
  // CHECK-LABEL: tt.func @shared_tensor_input
  // CHECK: ttg.local_alloc %arg0
  // CHECK: ttg.warp_specialize
  // CHECK: default {
  // CHECK: arith.mulf
  // CHECK: ttg.warp_yield
  // CHECK: partition0
  // CHECK: ttg.local_load
  // CHECK: math.exp
  // CHECK: ttg.local_store
  // CHECK: ttg.warp_return
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @shared_tensor_input(%input: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %left = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %right = math.exp %input : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // Common integer indexing, masks and constants must be recomputed in the
  // worker. Only its reduction result needs shared storage; transferring a
  // constant or a broadcast mask wastes storage and can use an awkward layout.
  // ALLOCS-LABEL: tt.func @recompute_common_indices_and_mask
  // ALLOCS: ttg.local_alloc
  // ALLOCS-NOT: ttg.local_alloc
  // ALLOCS: tt.return
  // CHECK-LABEL: tt.func @recompute_common_indices_and_mask
  // CHECK: ttg.local_alloc
  // CHECK: ttg.warp_specialize
  // CHECK: default {
  // CHECK: tt.load
  // CHECK: arith.mulf
  // CHECK: tt.reduce
  // CHECK: ttg.warp_yield
  // CHECK: partition0
  // CHECK-NOT: ttg.local_load
  // CHECK: tt.make_range
  // CHECK: tt.expand_dims
  // CHECK: tt.broadcast
  // CHECK: tt.splat
  // CHECK: arith.addi
  // CHECK: tt.splat
  // CHECK: arith.cmpi slt
  // CHECK: arith.constant dense<0.000000e+00>
  // CHECK: tt.splat
  // CHECK: tt.addptr
  // CHECK: tt.load
  // CHECK: math.exp
  // CHECK: tt.reduce
  // CHECK: ttg.local_store
  // CHECK: ttg.warp_return
  // CHECK: ttg.local_load
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @recompute_common_indices_and_mask(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %start: i32, %limit: i32) -> tensor<4xf32, #row> {
    %columns = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #column>
    %expanded = tt.expand_dims %columns {axis = 0 : i32} : tensor<32xi32, #column> -> tensor<1x32xi32, #tile>
    %offsets = tt.broadcast %expanded : tensor<1x32xi32, #tile> -> tensor<4x32xi32, #tile>
    %base = tt.splat %start : i32 -> tensor<4x32xi32, #tile>
    %indices = arith.addi %base, %offsets : tensor<4x32xi32, #tile>
    %limits = tt.splat %limit : i32 -> tensor<4x32xi32, #tile>
    %mask = arith.cmpi slt, %indices, %limits : tensor<4x32xi32, #tile>
    %zero = arith.constant dense<0.0> : tensor<4x32xf32, #tile>
    %ap = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %a_addresses = tt.addptr %ap, %indices : tensor<4x32x!tt.ptr<f32>, #tile>, tensor<4x32xi32, #tile>
    %a = tt.load %a_addresses, %mask, %zero : tensor<4x32x!tt.ptr<f32>, #tile>
    %squared = arith.mulf %a, %a : tensor<4x32xf32, #tile>
    %left = "tt.reduce"(%squared) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %bp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #tile>
    %b_addresses = tt.addptr %bp, %indices : tensor<4x32x!tt.ptr<f32>, #tile>, tensor<4x32xi32, #tile>
    %b = tt.load %b_addresses, %mask, %zero : tensor<4x32x!tt.ptr<f32>, #tile>
    %exponent = math.exp %b : tensor<4x32xf32, #tile>
    %right = "tt.reduce"(%exponent) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    tt.return %joined : tensor<4xf32, #row>
  }

  // A producer and its consumer are not independent branches.
  // NOFUSE-LABEL: tt.func @dependent_operands
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @dependent_operands
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: arith.mulf
  // CHECK: math.exp
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @dependent_operands(%input: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %left = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %right = math.exp %left : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // The first load is before a write and the second after it: both WAR and RAW
  // edges must survive even though there is no tensor SSA edge through store.
  // NOFUSE-LABEL: tt.func @read_write_read
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @read_write_read
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK-NEXT: tt.store
  // CHECK-NEXT: tt.load
  // CHECK: arith.addf
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @read_write_read(%ptr: tensor<4x32x!tt.ptr<f32>, #tile>, %value: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %before = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    tt.store %ptr, %value : tensor<4x32x!tt.ptr<f32>, #tile>
    %after = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // Different function arguments have no noalias contract. Their memory
  // dependencies must be treated conservatively, just like identical pointers.
  // NOFUSE-LABEL: tt.func @different_arguments_may_alias
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @different_arguments_may_alias
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK-NEXT: tt.store
  // CHECK-NEXT: tt.load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @different_arguments_may_alias(%a: tensor<4x32x!tt.ptr<f32>, #tile>, %b: tensor<4x32x!tt.ptr<f32>, #tile>, %value: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %before = tt.load %a : tensor<4x32x!tt.ptr<f32>, #tile>
    tt.store %b, %value : tensor<4x32x!tt.ptr<f32>, #tile>
    %after = tt.load %a : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // Pointer arithmetic must not hide the store's alias with a prior load.
  // NOFUSE-LABEL: tt.func @pointer_view_alias
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @pointer_view_alias
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @pointer_view_alias(%ptr: tensor<4x32x!tt.ptr<f32>, #tile>, %value: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %zero = arith.constant dense<0> : tensor<4x32xi32, #tile>
    %alias = tt.addptr %ptr, %zero : tensor<4x32x!tt.ptr<f32>, #tile>, tensor<4x32xi32, #tile>
    %before = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    tt.store %alias, %value : tensor<4x32x!tt.ptr<f32>, #tile>
    %after = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // tt.store uses GlobalMemory effects even for address-space-3 pointers.
  // Backing allocation identity, not the effect resource label, reveals alias.
  // NOFUSE-LABEL: tt.func @local_pointer_memdesc_alias
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @local_pointer_memdesc_alias
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: ttg.local_load
  // CHECK: tt.store
  // CHECK: ttg.local_load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @local_pointer_memdesc_alias(%mem: !ttg.memdesc<4x32xf32, #shared, #smem, mutable>, %value: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %ptr = "tle.local_pointers"(%mem) : (!ttg.memdesc<4x32xf32, #shared, #smem, mutable>) -> tensor<4x32x!tt.ptr<f32, 3>, #tile>
    %before = ttg.local_load %mem : !ttg.memdesc<4x32xf32, #shared, #smem, mutable> -> tensor<4x32xf32, #tile>
    tt.store %ptr, %value : tensor<4x32x!tt.ptr<f32, 3>, #tile>
    %after = ttg.local_load %mem : !ttg.memdesc<4x32xf32, #shared, #smem, mutable> -> tensor<4x32xf32, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // NOFUSE-LABEL: tt.func @unknown_effect_fence
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @unknown_effect_fence
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK-NEXT: "unknown.effect"
  // CHECK-NEXT: tt.load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @unknown_effect_fence(%ptr: tensor<4x32x!tt.ptr<f32>, #tile>) -> tensor<4x32xf32, #tile> {
    %before = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    "unknown.effect"() : () -> ()
    %after = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // A CTA-wide barrier cannot be moved into just one warp partition.
  // NOFUSE-LABEL: tt.func @cta_barrier_fence
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @cta_barrier_fence
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: tt.load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @cta_barrier_fence(%ptr: tensor<4x32x!tt.ptr<f32>, #tile>) -> tensor<4x32xf32, #tile> {
    %before = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    gpu.barrier
    %after = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // Volatile reads have ordering semantics beyond their SSA result.
  // NOFUSE-LABEL: tt.func @volatile_load_fence
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @volatile_load_fence
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.load {{.*}}isVolatile = true
  // CHECK: tt.load
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @volatile_load_fence(%ptr: tensor<4x32x!tt.ptr<f32>, #tile>) -> tensor<4x32xf32, #tile> {
    %before = tt.load %ptr {isVolatile = true} : tensor<4x32x!tt.ptr<f32>, #tile>
    %after = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // A pending TLE async-load annotation is a boundary even before its normal
  // lowering pass has materialized asynchronous copy and synchronization ops.
  // NOFUSE-LABEL: tt.func @annotated_async_load_fence
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @annotated_async_load_fence
  // CHECK: tt.load {{.*}}tt.load.async = true
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @annotated_async_load_fence(%ptr: tensor<4x32x!tt.ptr<f32>, #tile>) -> tensor<4x32xf32, #tile> {
    %before = tt.load %ptr {tt.load.async = true} : tensor<4x32x!tt.ptr<f32>, #tile>
    %after = tt.load %ptr : tensor<4x32x!tt.ptr<f32>, #tile>
    %joined = arith.addf %before, %after : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // A nominally pure reduction can contain a physical thread query. Moving
  // that combiner to a different warp changes its result, so inspect the body.
  // NOFUSE-LABEL: tt.func @reducer_thread_identity
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @reducer_thread_identity
  // CHECK: tt.reduce
  // CHECK: gpu.thread_id x
  // CHECK: tt.reduce
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @reducer_thread_identity(%input: tensor<4x32xf32, #tile>, %other: tensor<4x32xf32, #tile>) -> tensor<4xf32, #row> {
    %left = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %tid = gpu.thread_id x
      %tid_i32 = arith.index_cast %tid : index to i32
      %tid_f32 = arith.sitofp %tid_i32 : i32 to f32
      %sum = arith.addf %x, %y : f32
      %result = arith.addf %sum, %tid_f32 : f32
      tt.reduce.return %result : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %right = "tt.reduce"(%other) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %sum = arith.addf %x, %y : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x32xf32, #tile>) -> tensor<4xf32, #row>
    %joined = arith.addf %left, %right : tensor<4xf32, #row>
    tt.return %joined : tensor<4xf32, #row>
  }

  // An i32 scalar is not automatically uniform. Explicit worker captures use
  // one value for the group and cannot transport per-thread register values.
  // NOFUSE-LABEL: tt.func @nonuniform_scalar_capture
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @nonuniform_scalar_capture
  // CHECK: nvvm.read.ptx.sreg.tid.x
  // CHECK: arith.sitofp
  // CHECK: tt.splat
  // CHECK: math.exp
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @nonuniform_scalar_capture(%input: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %tid = "nvvm.read.ptx.sreg.tid.x"() : () -> i32
    %left = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %value = arith.sitofp %tid : i32 to f32
    %tile_value = tt.splat %value : f32 -> tensor<4x32xf32, #tile>
    %right = math.exp %tile_value : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // A device-function scalar argument can originate from per-thread state in
  // its caller. It has no kernel-entry uniformity guarantee.
  // NOFUSE-LABEL: tt.func private @device_function_scalar_argument
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func private @device_function_scalar_argument
  // CHECK: tt.splat
  // CHECK: math.exp
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func private @device_function_scalar_argument(%input: tensor<4x32xf32, #tile>, %value: f32) -> tensor<4x32xf32, #tile> {
    %left = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %tile_value = tt.splat %value : f32 -> tensor<4x32xf32, #tile>
    %right = math.exp %tile_value : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // Physical named-barrier IDs/thread counts are user contracts. Even a fork
  // after the barrier could reserve colliding IDs or change its participants.
  // NOFUSE-LABEL: tt.func @physical_named_barrier
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @physical_named_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: arith.mulf
  // CHECK: math.exp
  // CHECK: arith.addf
  // CHECK: tt.return
  tt.func @physical_named_barrier(%input: tensor<4x32xf32, #tile>) -> tensor<4x32xf32, #tile> {
    %id = arith.constant 1 : i32
    %threads = arith.constant 128 : i32
    ttng.arrive_barrier_named %id, %threads : i32, i32
    %left = arith.mulf %input, %input : tensor<4x32xf32, #tile>
    %right = math.exp %input : tensor<4x32xf32, #tile>
    %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
    tt.return %joined : tensor<4x32xf32, #tile>
  }

  // No inter-iteration schedule is inferred from a per-block DAG. Preserve
  // loop-carried state and keep the first version's loop scope untouched.
  // NOFUSE-LABEL: tt.func @loop_carried_boundary
  // NOFUSE-NOT: ttg.warp_specialize
  // NOFUSE: tt.return
  // CHECK-LABEL: tt.func @loop_carried_boundary
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: scf.for
  // CHECK: arith.mulf
  // CHECK: math.exp
  // CHECK: arith.addf
  // CHECK: scf.yield
  // CHECK-NOT: ttg.warp_specialize
  // CHECK: tt.return
  tt.func @loop_carried_boundary(%input: tensor<4x32xf32, #tile>, %n: index) -> tensor<4x32xf32, #tile> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result = scf.for %i = %zero to %n step %one iter_args(%carry = %input) -> (tensor<4x32xf32, #tile>) {
      %left = arith.mulf %carry, %carry : tensor<4x32xf32, #tile>
      %right = math.exp %carry : tensor<4x32xf32, #tile>
      %joined = arith.addf %left, %right : tensor<4x32xf32, #tile>
      scf.yield %joined : tensor<4x32xf32, #tile>
    }
    tt.return %result : tensor<4x32xf32, #tile>
  }
}
