/*
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

#include "tle/dialect/include/Analysis/TileDataflowGraph.h"

#include "tle/dialect/include/Analysis/TleMemoryEffectAnalysis.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir::triton::tle {
namespace {

namespace ttg = mlir::triton::gpu;

struct MemoryAccess {
  Value location;
  bool writes;
};

static bool isExplicitBoundary(Operation *op);

static bool isPureReduction(Operation *op) {
  if (!isa<triton::ReduceOp>(op))
    return false;
  // ReduceOp itself has the Pure trait. Check its body as well rather than
  // assuming that arbitrary user-written combiner IR obeys that promise.
  WalkResult result = op->walk([&](Operation *nested) {
    if (nested == op || isa<triton::ReduceReturnOp>(nested))
      return WalkResult::advance();
    if (nested->getNumRegions() || isExplicitBoundary(nested) ||
        !isMemoryEffectFree(nested) || !isSpeculatable(nested))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

static bool hasAsyncToken(Operation *op) {
  auto isToken = [](Type type) { return isa<ttg::AsyncTokenType>(type); };
  return llvm::any_of(op->getOperandTypes(), isToken) ||
         llvm::any_of(op->getResultTypes(), isToken);
}

static bool isExplicitBoundary(Operation *op) {
  StringRef name = op->getName().getStringRef();
  // These operations either have ordering semantics beyond ordinary memory
  // effects, or depend on the physical executing thread/warp. In particular,
  // inline assembly marked pure can still inspect a special register.
  return name.contains("barrier") || name.contains("fence") ||
         name.contains("async") || name.contains("atomic") ||
         name.contains("wait") || name.contains("commit") ||
         name.contains("arrive") || name.starts_with("ttng.") ||
         name.starts_with("nvws.") || name.starts_with("tle.pipe") ||
         name.starts_with("tle.wgmma") || name.starts_with("tle.tma") ||
         name.starts_with("nvvm.") || name.starts_with("llvm.") ||
         name.starts_with("gpu.") ||
         isa<triton::ElementwiseInlineAsmOp>(op) || hasAsyncToken(op);
}

static SmallVector<MemoryAccess> getMemoryAccesses(Operation *op) {
  SmallVector<MemoryAccess> accesses;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
    // Immutable allocations intentionally advertise no effects before physical
    // allocation. They still create storage (and initialize it when src is
    // present), so include their physical memory dependence explicitly.
    accesses.push_back({alloc.getResult(), true});
    return accesses;
  }
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return {{Value(), true}};
  for (const auto &effect : *effects) {
    bool writes = isa<MemoryEffects::Write, MemoryEffects::Allocate,
                      MemoryEffects::Free>(effect.getEffect());
    if (writes || isa<MemoryEffects::Read>(effect.getEffect()))
      accesses.push_back({effect.getValue(), writes});
    else
      accesses.push_back({Value(), true});
  }
  return accesses;
}

static PointerAddressClass getAddressClass(Value value) {
  if (!value)
    return PointerAddressClass::Unknown;
  if (auto desc = dyn_cast<ttg::MemDescType>(value.getType())) {
    if (isa<ttg::SharedMemorySpaceAttr>(desc.getMemorySpace()))
      return PointerAddressClass::Shared;
    return PointerAddressClass::Unknown;
  }
  return classifyPointerAddress(value);
}

static Value getAllocationRoot(Value value) {
  if (!value)
    return Value();
  // This existing helper normalizes local pointers and their offset/alias
  // views. Do not infer the memory space from an effect resource: tt.load and
  // tt.store label even shared-address-space pointers as GlobalMemory.
  if (auto root = getSharedPointerMemDescRoot(value))
    value = *root;
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (Operation *def = value.getDefiningOp()) {
    if (!visited.insert(def).second)
      return Value();
    if (isa<ttg::LocalAllocOp>(def))
      return value;
    if (auto local = dyn_cast<LocalPointersOp>(def)) {
      value = local.getSrc();
      continue;
    }
    if (isa<ttg::ConvertLayoutOp>(def) ||
        (isa<ttg::MemDescType>(value.getType()) &&
         def->hasTrait<OpTrait::MemDescViewTrait>() &&
         def->getNumRegions() == 0)) {
      if (def->getNumOperands() == 0)
        return Value();
      value = def->getOperand(0);
      continue;
    }
    return Value();
  }
  return Value();
}

static bool mayAlias(Value lhs, Value rhs) {
  if (!lhs || !rhs || lhs == rhs)
    return true;
  PointerAddressClass lhsClass = getAddressClass(lhs);
  PointerAddressClass rhsClass = getAddressClass(rhs);
  if (lhsClass != PointerAddressClass::Unknown &&
      rhsClass != PointerAddressClass::Unknown && lhsClass != rhsClass)
    return false;
  Value lhsAlloc = getAllocationRoot(lhs);
  Value rhsAlloc = getAllocationRoot(rhs);
  if (lhsAlloc && rhsAlloc && lhsAlloc != rhsAlloc)
    return false;
  // Distinct block arguments, pointer arithmetic, casts, and unresolved views
  // are deliberately not a proof of non-aliasing.
  return true;
}

static bool hasMemoryHazard(ArrayRef<MemoryAccess> before,
                            ArrayRef<MemoryAccess> after) {
  for (const MemoryAccess &lhs : before)
    for (const MemoryAccess &rhs : after)
      if ((lhs.writes || rhs.writes) && mayAlias(lhs.location, rhs.location))
        return true;
  return false;
}

} // namespace

bool isSchedulingBoundary(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>() ||
      op->getNumSuccessors() != 0 || isa<CallOpInterface>(op) ||
      isExplicitBoundary(op))
    return true;
  if (op->getNumRegions())
    return !isPureReduction(op);
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return load.getIsVolatile() || op->hasAttr("tt.load.async");
  if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
    return static_cast<bool>(load.getToken());
  if (isa<ttg::LocalAllocOp, ttg::LocalDeallocOp>(op))
    return false;
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return true;
  if (effects->empty())
    return !isSpeculatable(op);
  return llvm::any_of(*effects, [](const auto &effect) {
    return !effect.getValue() ||
           !isa<MemoryEffects::Read, MemoryEffects::Write,
                MemoryEffects::Allocate, MemoryEffects::Free>(
               effect.getEffect());
  });
}

bool isMovableTileOp(Operation *op) {
  if (isSchedulingBoundary(op))
    return false;
  if (isa<triton::LoadOp, ttg::LocalLoadOp>(op))
    return true;
  // Allocations are unique storage identities, even when an immutable
  // allocation advertises no effects. They remain outside moved branches.
  if (isa<ttg::LocalAllocOp, ttg::LocalDeallocOp>(op))
    return false;
  if (op->getNumRegions())
    return isPureReduction(op);
  return isMemoryEffectFree(op) && isSpeculatable(op);
}

TileDataflowGraph::TileDataflowGraph(Block &block) {
  for (Operation &op : block) {
    indices[&op] = operations.size();
    operations.push_back(&op);
  }
  predecessors.resize(operations.size());
  successors.resize(operations.size());

  SmallVector<SmallVector<MemoryAccess>> accesses;
  SmallVector<bool> boundaries;
  for (Operation *op : operations) {
    accesses.push_back(getMemoryAccesses(op));
    boundaries.push_back(isSchedulingBoundary(op));
    for (Value operand : mlir::getNestedOperands(op)) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      if (Operation *ancestor = block.findAncestorOpInBlock(*def))
        addEdge(ancestor, op);
    }
  }
  // RAW, WAR and WAW all constrain scheduling. Read/read is independent even
  // for aliases. Boundary edges also order scalar operations, which would not
  // otherwise appear in the memory-effects relation.
  for (unsigned after = 0; after < operations.size(); ++after)
    for (unsigned before = 0; before < after; ++before)
      if (boundaries[before] || boundaries[after] ||
          hasMemoryHazard(accesses[before], accesses[after]))
        addEdge(operations[before], operations[after]);
}

void TileDataflowGraph::addEdge(Operation *from, Operation *to) {
  auto fromIt = indices.find(from);
  auto toIt = indices.find(to);
  if (fromIt == indices.end() || toIt == indices.end() ||
      fromIt->second >= toIt->second)
    return;
  auto &users = successors[fromIt->second];
  if (llvm::is_contained(users, to))
    return;
  users.push_back(to);
  predecessors[toIt->second].push_back(from);
}

ArrayRef<Operation *>
TileDataflowGraph::getPredecessors(Operation *op) const {
  auto found = indices.find(op);
  if (found == indices.end())
    return {};
  return predecessors[found->second];
}

ArrayRef<Operation *> TileDataflowGraph::getSuccessors(Operation *op) const {
  auto found = indices.find(op);
  if (found == indices.end())
    return {};
  return successors[found->second];
}

bool TileDataflowGraph::hasPath(Operation *from, Operation *to) const {
  auto fromIt = indices.find(from);
  auto toIt = indices.find(to);
  if (fromIt == indices.end() || toIt == indices.end() ||
      fromIt->second > toIt->second)
    return false;
  if (from == to)
    return true;

  auto [cached, inserted] =
      reachableFrom.try_emplace(fromIt->second, operations.size());
  llvm::BitVector &reachable = cached->second;
  if (inserted) {
    // Global antichain search asks many ordering questions with the same
    // source. Traverse its descendants once instead of repeating a DFS for
    // every candidate pair. Sources never queried need no bit-vector storage.
    SmallVector<unsigned> worklist{fromIt->second};
    reachable.set(fromIt->second);
    while (!worklist.empty()) {
      unsigned current = worklist.pop_back_val();
      for (Operation *next : successors[current]) {
        unsigned index = indices.lookup(next);
        if (!reachable.test(index)) {
          reachable.set(index);
          worklist.push_back(index);
        }
      }
    }
  }
  return reachable.test(toIt->second);
}

} // namespace mlir::triton::tle
