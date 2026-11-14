/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonEliminateIntermediateMemory
    : Pass<"triton-eliminate-intermediate-memory", "mlir::ModuleOp"> {
  let summary = "eliminate proven CTA-local intermediate global-memory accesses";
  let description = [{
    A standalone TTIR memory optimization, run after inlining and CSE. Forward
    same-address loads from their preceding stores, preserving the stored dtype
    and observable masked-load other values. Remove stores only for explicitly
    private argument buffers after proving that no uses of those pointers remain.
    Non-private stores stay observable. Rejection leaves the body unchanged and
    records the reason in the triton.intermediate_memory_elimination attribute.
    A configurable profitability guard avoids extending long transcendental
    dependency chains across forwarded loads, which can regress code generation.
    It bounds both dependency depth and static depth times tile elements, a
    conservative proxy for unrolled transcendental code expansion.

    The caller must explicitly guarantee disjoint storage for distinct pointer
    arguments and unobservable final contents for private arguments. Mutable
    accesses must share a contiguous program_id(0) * BLOCK + arange(0, BLOCK)
    mapping in a one-dimensional launch with one CTA per program and no cluster.
    A one-dimensional grid may contain many independent programs; inter-CTA
    communication and incompatible per-program tile ownership are unsupported.
    The entry function must have a single block. Pure nested regions may remain,
    but are outside the mask-select proof; the profitability guard conservatively
    rejects forwarding into a nested region containing transcendental math or
    forwarding a region-produced value with unknown transcendental depth.
    The pass does not combine launches, retile operators, specialize launch-grid
    bounds, allocate host buffers, or invoke a graph compiler. It is not enabled
    in the default compilation pipeline.
  }];
  let options = [
    ListOption<"privateArguments", "private-arguments", "unsigned",
               "Entry argument indices of private intermediate buffers">,
    Option<"assumeDisjointArguments", "assume-disjoint-arguments", "bool", "false",
           "Caller guarantees distinct pointer arguments have disjoint storage">,
    Option<"numCTAs", "num-ctas", "int64_t", "1",
           "CTAs per program; must be one">,
    Option<"clusterSize", "cluster-size", "int64_t", "1",
           "Product of launch cluster dimensions; must be one">,
    Option<"maxTranscendentalDepth", "max-transcendental-depth", "unsigned", "8",
           "Maximum transcendental dependency depth extended by forwarding; zero disables the profitability guard">,
    Option<"maxTranscendentalWork", "max-transcendental-work", "unsigned", "4096",
           "Maximum forwarding depth times tile element count; zero disables this work subguard">
  ];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorPointer : Pass</*cli-arg*/"triton-rewrite-tensor-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores with tensor pointers into legacy load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy
    semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute
    the pointer/mask/other for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopUnroll : Pass</*cli-arg*/"triton-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopInvariantCodeMotion : Pass</*cli-arg*/"triton-licm", /*Op*/"mlir::ModuleOp"> {
  let summary = "MLIR's LICM plus hoist load ops out of loops with masks.";
  let description = [{
    This pass uses MLIR's LICM pass as base. Additionally, it hoists load ops
    out of loops that consists of pure/read-only ops. For scf.for loops, it
    generates a trip-count check. For scf.while loops, it clones the condition
    from the before body.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.
  }];
}

#endif
