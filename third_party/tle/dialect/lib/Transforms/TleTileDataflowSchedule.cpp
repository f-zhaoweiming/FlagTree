/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
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
#include "tle/dialect/include/Analysis/TileDataflowResources.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include <optional>
#include <set>
#include <vector>

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TRITONTLETILEDATAFLOWSCHEDULE
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
using OpSet = llvm::SmallPtrSet<Operation *, 32>;

// A wave of independent branch subgraphs selected from the whole block DAG.
struct Branch {
  Value root;
  SmallVector<Operation *> ops;
  SmallVector<Operation *> worker;
};
struct Fork {
  Operation *join; // Insertion cut: earliest consumer outside the wave.
  SmallVector<Branch> branches;
  unsigned defaultBranch;
};

static bool canStage(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getRank() > 0 && tensor.hasStaticShape() &&
         tensor.getEncoding() && tensor.getElementType().isIntOrFloat() &&
         (isa<IntegerType>(tensor.getElementType()) ||
          tensor.getElementType().getIntOrFloatBitWidth() >= 8);
}

static void collectBranchImpl(Value value, Block *block, Operation *boundary,
                              OpSet &ops) {
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != block || !isMovableTileOp(op) ||
      (boundary && (op == boundary || op->isBeforeInBlock(boundary))) ||
      op->hasAttr("tle.tile_dataflow.transport") || !ops.insert(op).second)
    return;
  for (Value operand : getNestedOperands(op))
    collectBranchImpl(operand, block, boundary, ops);
}

static void collectBranch(Value value, Block *block, OpSet &ops) {
  Operation *root = value.getDefiningOp();
  if (!root || root->getBlock() != block)
    return;
  // Old waves order the whole block. Pulling a constant from before an old
  // wave into a later branch creates a false cross-branch ordering conflict.
  // Keep all earlier work as captures; workers may still recompute cheap pure
  // captures through the existing recomputation path.
  Operation *boundary = root->getPrevNode();
  while (boundary && !isSchedulingBoundary(boundary))
    boundary = boundary->getPrevNode();
  collectBranchImpl(value, block, boundary, ops);
}

static void pruneEscaping(OpSet &ops, Value root) {
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> remove;
    for (Operation *op : ops)
      for (Value result : op->getResults())
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user);
              return !ops.contains(ancestor) &&
                     result != root;
            })) {
          remove.push_back(op);
          break;
        }
    for (Operation *op : remove)
      changed |= ops.erase(op);
  } while (changed);
}

static unsigned tileWork(const OpSet &ops) {
  return llvm::count_if(ops, [](Operation *op) {
    if (isa<arith::ConstantOp, triton::SplatOp, triton::MakeRangeOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::AddPtrOp>(op))
      return false;
    return llvm::any_of(op->getResultTypes(), [](Type type) {
      return isa<RankedTensorType>(type);
    });
  });
}

static llvm::SetVector<Value> captures(ArrayRef<Operation *> ops) {
  OpSet inside(ops.begin(), ops.end());
  llvm::SetVector<Value> result;
  for (Operation *op : ops)
    for (Value operand : getNestedOperands(op))
      if (!inside.contains(operand.getDefiningOp()))
        result.insert(operand);
  return result;
}

static bool isCheapToRecompute(Operation *op) {
  if (!isMovableTileOp(op) || !isMemoryEffectFree(op) || op->getNumRegions())
    return false;
  if (isa<arith::ConstantOp, triton::SplatOp, triton::MakeRangeOp,
          triton::ExpandDimsOp, triton::BroadcastOp>(op))
    return true;
  auto isInteger = [](Type type) {
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    return type.isIntOrIndex();
  };
  return op->getName().getDialectNamespace() == "arith" &&
         llvm::all_of(op->getOperandTypes(), isInteger) &&
         llvm::all_of(op->getResultTypes(), isInteger);
}

static void collectRecomputed(Value value, Block *block, const OpSet &inside,
                              OpSet &recomputed) {
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != block || inside.contains(op) ||
      !isCheapToRecompute(op) || !recomputed.insert(op).second)
    return;
  for (Value operand : getNestedOperands(op))
    collectRecomputed(operand, block, inside, recomputed);
}

static uint64_t tileBytes(Type type) {
  auto tensor = cast<RankedTensorType>(type);
  return tensor.getNumElements() *
         llvm::divideCeil(tensor.getElementType().getIntOrFloatBitWidth(), 8u);
}

static bool isUniformCapture(Value value, Block *entry, OpSet &visited) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner() == entry;
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != entry || isSchedulingBoundary(op) ||
      op->getNumRegions() || !isMemoryEffectFree(op) || !isSpeculatable(op))
    return false;
  if (!visited.insert(op).second)
    return true;
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    return !isa<RankedTensorType>(operand.getType()) &&
           isUniformCapture(operand, entry, visited);
  });
}

// Build an antichain of branch outputs anywhere in the block, including outputs
// consumed by different joins/stores. The insertion cut is the earliest outside
// consumer. Both upward and downward motion obey *all* graph edges.
static std::optional<Fork> makeFork(ArrayRef<Value> roots,
                                    TileDataflowGraph &graph,
                                    unsigned minBranchOps) {
  SmallVector<OpSet, 4> sets(roots.size());
  Block *block = roots.front().getDefiningOp()->getBlock();
  for (auto [i, root] : llvm::enumerate(roots)) {
    collectBranch(root, block, sets[i]);
    for (unsigned j = 0; j < i; ++j)
      if (graph.hasPath(root.getDefiningOp(), roots[j].getDefiningOp()) ||
          graph.hasPath(roots[j].getDefiningOp(), root.getDefiningOp()))
        return std::nullopt;
  }
  OpSet common;
  for (unsigned i = 0; i < sets.size(); ++i)
    for (Operation *op : sets[i])
      for (unsigned j = i + 1; j < sets.size(); ++j)
        if (sets[j].contains(op))
          common.insert(op);
  OpSet selected;
  for (auto [i, ops] : llvm::enumerate(sets)) {
    for (Operation *op : common)
      ops.erase(op);
    pruneEscaping(ops, roots[i]);
    if (!ops.contains(roots[i].getDefiningOp()) || tileWork(ops) < minBranchOps)
      return std::nullopt;
    selected.insert(ops.begin(), ops.end());
  }
  Operation *anchor = nullptr;
  for (Value root : roots)
    for (Operation *user : root.getUsers()) {
      Operation *outside = block->findAncestorOpInBlock(*user);
      if (!outside)
        return std::nullopt;
      if (!selected.contains(outside) &&
          (!anchor || outside->isBeforeInBlock(anchor)))
        anchor = outside;
    }
  if (!anchor)
    return std::nullopt;
  for (auto [i, ops] : llvm::enumerate(sets))
    for (Operation *op : ops) {
      for (unsigned j = i + 1; j < sets.size(); ++j)
        for (Operation *other : sets[j])
          if (graph.hasPath(op, other) || graph.hasPath(other, op))
            return std::nullopt;
      for (Operation *pred : graph.getPredecessors(op))
        if (!selected.contains(pred) && !pred->isBeforeInBlock(anchor))
          return std::nullopt;
      for (Operation *succ : graph.getSuccessors(op))
        if (!selected.contains(succ) && succ != anchor &&
            succ->isBeforeInBlock(anchor))
          return std::nullopt;
    }
  Fork fork{anchor, {}, 0};
  for (auto [i, root] : llvm::enumerate(roots)) {
    Branch branch{root, {}, {}};
    for (Operation *op : graph.getOperations())
      if (sets[i].contains(op))
        branch.ops.push_back(op);
    OpSet recomputed;
    for (Value value : captures(branch.ops))
      collectRecomputed(value, block, sets[i], recomputed);
    for (Operation *op : graph.getOperations())
      if (sets[i].contains(op) || recomputed.contains(op))
        branch.worker.push_back(op);
    fork.branches.push_back(std::move(branch));
  }
  return fork;
}

static SmallVector<Value> reusableCaptures(Fork &fork);

static bool legalCaptures(Fork &fork, uint64_t maxSharedBytes) {
  llvm::SetVector<Value> staged;
  auto reused = reusableCaptures(fork);
  uint64_t bytes = 0;
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    if (i == fork.defaultBranch)
      continue;
    if (!reused[i])
      bytes += tileBytes(branch.root.getType());
    for (Value value : captures(branch.worker)) {
      if (isa<RankedTensorType>(value.getType())) {
        if (!canStage(value.getType()))
          return false;
        if (staged.insert(value))
          bytes += tileBytes(value.getType());
      } else if (!value.getType().isIntOrFloat() &&
                 !isa<triton::PointerType, ttg::MemDescType>(value.getType())) {
        return false;
      } else if (!isa<ttg::MemDescType>(value.getType())) {
        OpSet visited;
        if (!isUniformCapture(value, fork.join->getBlock(), visited))
          return false;
      }
    }
  }
  return bytes <= maxSharedBytes;
}

static SmallVector<Fork> findForks(TileDataflowGraph &graph,
                                   unsigned minBranchOps,
                                   uint64_t maxSharedBytes,
                                   unsigned maxPartitions) {
  struct Root { Value value; unsigned work; };
  SmallVector<Root> roots;
  for (Operation *op : graph.getOperations()) {
    if (!isMovableTileOp(op) || op->hasAttr("tle.tile_dataflow.transport"))
      continue;
    for (Value value : op->getResults()) {
      if (!canStage(value.getType()) || value.use_empty())
        continue;
      OpSet ancestors;
      collectBranch(value, op->getBlock(), ancestors);
      unsigned work = tileWork(ancestors);
      if (work >= minBranchOps)
        roots.push_back({value, work});
    }
  }
  // Internal single-chain values cannot expose useful parallelism at their
  // immediate consumers. Retain roots at genuine independent-input joins and
  // graph outputs; their backward slices still include every intervening node.
  // This avoids quadratic candidate cloning for long elementwise chains.
  llvm::DenseSet<Value> useful;
  for (const Root &root : roots)
    useful.insert(root.value);
  llvm::erase_if(roots, [&](const Root &root) {
    for (Operation *user : root.value.getUsers()) {
      Operation *consumer = root.value.getDefiningOp()->getBlock()->findAncestorOpInBlock(*user);
      if (!consumer || isSchedulingBoundary(consumer) ||
          isa<triton::StoreOp, ttg::LocalStoreOp>(consumer))
        return false;
      for (Value other : getNestedOperands(consumer))
        if (other != root.value && useful.contains(other) &&
            !graph.hasPath(root.value.getDefiningOp(), other.getDefiningOp()) &&
            !graph.hasPath(other.getDefiningOp(), root.value.getDefiningOp()))
          return false;
    }
    return true;
  });
  // Prefer completed, substantial chains. Keep search polynomial: each seed
  // greedily grows an antichain and exposes all its widths to the cost model.
  // The selected wave is the best candidate across the entire block, not the
  // first join encountered. This is a bounded heuristic, not optimal partitioning.
  llvm::stable_sort(roots, [](const Root &a, const Root &b) {
    return a.work > b.work;
  });
  SmallVector<Fork> candidates;
  std::set<std::vector<unsigned>> seen;
  // Preserve every legal pair before greedy growth. Otherwise a cheap partial
  // chain can consume a seed first and hide a later, balanced fork, especially
  // after earlier waves have introduced new graph boundaries.
  for (unsigned a = 0; a < roots.size(); ++a) {
    for (unsigned b = a + 1; b < roots.size(); ++b) {
      auto fork = makeFork({roots[a].value, roots[b].value}, graph, minBranchOps);
      if (!fork)
        continue;
      bool legal = false;
      for (unsigned d = 0; d < 2; ++d) {
        fork->defaultBranch = d;
        if (legalCaptures(*fork, maxSharedBytes)) {
          candidates.push_back(*fork);
          legal = true;
        }
      }
      if (legal)
        seen.insert({a, b});
    }
  }
  for (unsigned seed = 0; seed < roots.size(); ++seed) {
    SmallVector<Value> group{roots[seed].value};
    std::vector<unsigned> indices{seed};
    for (unsigned next = 0; next < roots.size(); ++next) {
      if (next == seed || group.size() >= maxPartitions)
        continue;
      SmallVector<Value> trial(group);
      trial.push_back(roots[next].value);
      auto fork = makeFork(trial, graph, minBranchOps);
      if (!fork)
        continue;
      // A newly added branch can become the default even if previous choices
      // could not capture its values; do not commit to a direction prematurely.
      bool legal = false;
      for (unsigned d = 0; d < trial.size(); ++d) {
        fork->defaultBranch = d;
        legal |= legalCaptures(*fork, maxSharedBytes);
      }
      if (!legal)
        continue;
      group = std::move(trial);
      indices.push_back(next);
      auto key = indices;
      llvm::sort(key);
      if (!seen.insert(key).second)
        continue;
      for (unsigned d = 0; d < group.size(); ++d) {
        fork->defaultBranch = d;
        if (legalCaptures(*fork, maxSharedBytes))
          candidates.push_back(*fork);
      }
    }
  }
  // Expand composite branches at internal joins. Greedy growth alone can get
  // stuck at {A, B, C+D}; expansion exposes {A, B, C, D}, including when the
  // original expression is a binary tree or has disconnected outputs.
  llvm::DenseMap<Value, unsigned> rootIds;
  for (auto [i, root] : llvm::enumerate(roots))
    rootIds[root.value] = i;
  std::set<std::vector<unsigned>> expanded;
  unsigned expansions = 0;
  for (unsigned index = 0; index < candidates.size() && expansions < 256; ++index) {
    Fork original = candidates[index];
    std::vector<unsigned> originalKey;
    for (Branch &branch : original.branches)
      originalKey.push_back(rootIds.lookup(branch.root));
    llvm::sort(originalKey);
    if (!expanded.insert(originalKey).second || original.branches.size() >= maxPartitions)
      continue;
    for (unsigned branchIndex = 0; branchIndex < original.branches.size(); ++branchIndex) {
      for (Operation *op : llvm::reverse(original.branches[branchIndex].ops)) {
        SmallVector<Value> inputs;
        for (Value value : op->getOperands())
          if (rootIds.contains(value) && !llvm::is_contained(inputs, value))
            inputs.push_back(value);
        if (inputs.size() < 2 || original.branches.size() - 1 + inputs.size() > maxPartitions)
          continue;
        SmallVector<Value> trial;
        for (auto [i, branch] : llvm::enumerate(original.branches))
          if (i != branchIndex)
            trial.push_back(branch.root);
        llvm::append_range(trial, inputs);
        std::vector<unsigned> key;
        for (Value value : trial)
          key.push_back(rootIds.lookup(value));
        llvm::sort(key);
        if (seen.find(key) != seen.end())
          continue;
        auto fork = makeFork(trial, graph, minBranchOps);
        if (!fork)
          continue;
        seen.insert(key);
        ++expansions;
        for (unsigned d = 0; d < trial.size(); ++d) {
          fork->defaultBranch = d;
          if (legalCaptures(*fork, maxSharedBytes))
            candidates.push_back(*fork);
        }
        if (expansions >= 256)
          return candidates;
      }
    }
  }
  return candidates;
}

static RankedTensorType storageType(RankedTensorType tensor) {
  // Shared-memory lowering addresses bytes. In particular, an i1 memdesc
  // would imply packed storage unsupported by the existing local-load/store
  // lowering. Keep that contract explicit by staging predicates as bytes.
  if (auto integer = dyn_cast<IntegerType>(tensor.getElementType());
      integer && integer.getWidth() < 8)
    return RankedTensorType::get(tensor.getShape(),
                                 IntegerType::get(tensor.getContext(), 8),
                                 tensor.getEncoding());
  return tensor;
}

static ttg::MemDescType sharedType(RankedTensorType tensor) {
  tensor = storageType(tensor);
  return ttg::MemDescType::get(
      tensor.getShape(), tensor.getElementType(), getSharedEncoding(tensor),
      ttg::SharedMemorySpaceAttr::get(tensor.getContext()), /*mutable=*/true);
}

// The materializer and both storage/cost analyses must make the same reuse
// decision. A shared capture read by multiple workers cannot be overwritten
// without introducing another cross-worker synchronization protocol.
static SmallVector<Value> reusableCaptures(Fork &fork) {
  llvm::DenseMap<Value, unsigned> readers;
  for (auto [i, branch] : llvm::enumerate(fork.branches))
    if (i != fork.defaultBranch)
      for (Value value : captures(branch.worker))
        ++readers[value];
  SmallVector<Value> reused(fork.branches.size());
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    if (i == fork.defaultBranch)
      continue;
    auto output = sharedType(cast<RankedTensorType>(branch.root.getType()));
    for (Value value : captures(branch.worker)) {
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      if (tensor && canStage(tensor) && readers.lookup(value) == 1 &&
          sharedType(tensor) == output) {
        reused[i] = value;
        break;
      }
    }
  }
  return reused;
}

static Value toSharedStorage(OpBuilder &builder, Location loc, Value value) {
  auto tensor = cast<RankedTensorType>(value.getType());
  auto storage = storageType(tensor);
  if (storage == tensor)
    return value;
  return arith::ExtUIOp::create(builder, loc, storage, value);
}

static Value loadShared(OpBuilder &builder, Location loc,
                        RankedTensorType tensor, Value memory) {
  auto storage = storageType(tensor);
  Value value = ttg::LocalLoadOp::create(builder, loc, storage, memory);
  if (storage != tensor)
    value = arith::TruncIOp::create(builder, loc, tensor, value);
  return value;
}

static bool hasUnmanagedThreadSemantics(triton::FuncOp function) {
  return function.walk([](Operation *op) {
    StringRef name = op->getName().getStringRef();
    // Explicit physical barriers and inline assembly may use IDs reserved by
    // warp-specialize or assume the original CTA size. Their contracts cannot
    // be recovered from ordinary memory effects. gpu.barrier is safe because
    // the existing lowering rewrites it to a warp-group barrier.
    if (isa<triton::ElementwiseInlineAsmOp>(op) ||
        (name.contains("barrier_named") &&
         !op->hasAttr("tle.tile_dataflow.full_group_sync")) || name.starts_with("nvvm.") ||
        name.starts_with("llvm.") ||
        (name.starts_with("gpu.") && name != "gpu.barrier"))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }).wasInterrupted();
}

static void markTransport(Operation *op) {
  op->setAttr("tle.tile_dataflow.transport", UnitAttr::get(op->getContext()));
}

static ttg::WarpSpecializeOp materialize(Fork &fork, double work, ArrayRef<double> components) {
  OpBuilder builder(fork.join);
  Location loc = fork.join->getLoc();
  // Every worker has the same explicit-capture signature. Deduplicate staged
  // inputs across workers. A result may reuse a private staged input buffer.
  llvm::SetVector<Value> external;
  auto reused = reusableCaptures(fork);
  for (auto [i, branch] : llvm::enumerate(fork.branches))
    if (i != fork.defaultBranch)
      for (Value value : captures(branch.worker))
        external.insert(value);
  SmallVector<Value> workerCaptures, staged, outputs(fork.branches.size());
  for (Value value : external) {
    if (auto tensor = dyn_cast<RankedTensorType>(value.getType())) {
      Value stored = toSharedStorage(builder, loc, value);
      if (stored != value)
        markTransport(stored.getDefiningOp());
      Value memory = ttg::LocalAllocOp::create(builder, loc, sharedType(tensor), stored);
      markTransport(memory.getDefiningOp());
      workerCaptures.push_back(memory);
      staged.push_back(memory);
    } else {
      workerCaptures.push_back(value);
    }
  }
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    if (i == fork.defaultBranch)
      continue;
    auto type = cast<RankedTensorType>(branch.root.getType());
    // A private staged tensor dies after this worker's entry loads. Reuse its
    // mutable storage for the result after all branch computations finish.
    // Other workers must not read it: the warp-specialize entry barrier occurs
    // before their tensor loads and cannot protect a cross-worker overwrite.
    // The default region retains the captured SSA tensor in registers, and
    // the existing all-group join protects the final result load below.
    if (reused[i]) {
      outputs[i] = workerCaptures[llvm::find(external, reused[i]) - external.begin()];
      continue;
    }
    Value output = ttg::LocalAllocOp::create(builder, loc, sharedType(type));
    markTransport(output.getDefiningOp());
    outputs[i] = output;
    workerCaptures.push_back(output);
    staged.push_back(output);
  }
  int32_t warps = ttg::lookupNumWarps(fork.join);
  SmallVector<int32_t> partitionWarps(fork.branches.size() - 1, warps);
  Branch &main = fork.branches[fork.defaultBranch];
  auto ws = ttg::WarpSpecializeOp::create(
      builder, loc, TypeRange{main.root.getType()}, partitionWarps,
      partitionWarps.size());
  ws->setOperands(workerCaptures);
  ws->setAttr("tle.tile_dataflow", builder.getUnitAttr());
  // Intrinsic wave cost excludes whole-kernel occupancy. Later waves re-evaluate
  // the shared whole-kernel resource envelope, so costs are not charged twice.
  if (!components.empty()) {
    ws->setAttr("tle.tile_dataflow.work", builder.getF64FloatAttr(work));
    ws->setAttr("tle.tile_dataflow.costs", builder.getDenseF64ArrayAttr(components));
  }
  Block &defaultBlock = ws.getDefaultRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&defaultBlock);
  IRMapping mainMapping;
  for (Operation *op : main.ops)
    builder.clone(*op, mainMapping);
  ttg::WarpYieldOp::create(builder, loc, ValueRange{mainMapping.lookup(main.root)});

  unsigned partition = 0;
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    if (i == fork.defaultBranch)
      continue;
    Region &region = *ws.getPartitionRegions()[partition++];
    Block &worker = region.emplaceBlock();
    for (Value value : workerCaptures)
      worker.addArgument(value.getType(), loc);
    builder.setInsertionPointToEnd(&worker);
    IRMapping mapping;
    auto needed = captures(branch.worker);
    for (auto [index, value] : llvm::enumerate(external)) {
      if (!needed.contains(value))
        continue;
      Value local = worker.getArgument(index);
      if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
        local = loadShared(builder, loc, tensor, local);
      mapping.map(value, local);
    }
    for (Operation *op : branch.worker)
      builder.clone(*op, mapping);
    unsigned outputIndex = llvm::find(workerCaptures, outputs[i]) - workerCaptures.begin();
    ttg::LocalStoreOp::create(builder, loc,
        toSharedStorage(builder, loc, mapping.lookup(branch.root)),
        worker.getArgument(outputIndex));
    ttg::WarpReturnOp::create(builder, loc);
  }
  builder.setInsertionPointAfter(ws);
  SmallVector<Value> results(fork.branches.size());
  results[fork.defaultBranch] = ws.getResult(0);
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    if (i == fork.defaultBranch)
      continue;
    results[i] = loadShared(builder, loc,
        cast<RankedTensorType>(branch.root.getType()), outputs[i]);
    markTransport(results[i].getDefiningOp());
    if (auto trunc = results[i].getDefiningOp<arith::TruncIOp>())
      markTransport(trunc.getIn().getDefiningOp());
  }
  for (Value memory : staged)
    markTransport(ttg::LocalDeallocOp::create(builder, loc, memory));
  for (auto [i, branch] : llvm::enumerate(fork.branches))
    branch.root.replaceAllUsesWith(results[i]);
  for (Branch &branch : fork.branches)
    for (Operation *op : llvm::reverse(branch.ops))
      op->erase();
  return ws;
}

// These are comparable weighted instruction units, not GPU clock cycles. The
// model deliberately credits only partial overlap: both groups share issue,
// memory and special-function resources on the same SM.
static Value peelView(Value value) {
  while (auto *op = value.getDefiningOp()) {
    if (!isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
             ttg::ConvertLayoutOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

static std::optional<int64_t> constantInteger(Value value) {
  auto constant = peelView(value).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
    return attr.getInt();
  if (auto attr = dyn_cast<DenseIntElementsAttr>(constant.getValue());
      attr && attr.isSplat())
    return attr.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}

// Static tail masks reduce useful work without assuming anything about dynamic
// launch dimensions. Unknown masks conservatively retain full activity.
static double activeFraction(Value mask) {
  mask = peelView(mask);
  if (auto c = constantInteger(mask))
    return *c ? 1.0 : 0.0;
  if (auto both = mask.getDefiningOp<arith::AndIOp>())
    return std::min(activeFraction(both.getLhs()), activeFraction(both.getRhs()));
  auto compare = mask.getDefiningOp<arith::CmpIOp>();
  if (!compare)
    return 1.0;
  auto range = peelView(compare.getLhs()).getDefiningOp<triton::MakeRangeOp>();
  auto bound = constantInteger(compare.getRhs());
  if (!range || !bound || range.getEnd() <= range.getStart())
    return 1.0;
  if ((compare.getPredicate() == arith::CmpIPredicate::ult ||
       compare.getPredicate() == arith::CmpIPredicate::ule) &&
      (*bound < 0 || range.getStart() < 0))
    return 1.0;
  int64_t end = *bound;
  switch (compare.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    if (end == INT64_MAX)
      return 1.0;
    ++end;
    break;
  default:
    return 1.0;
  }
  return std::clamp((double(end) - double(range.getStart())) /
                        double(range.getEnd() - range.getStart()),
                    0.0, 1.0);
}

static double elementsPerThread(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getEncoding() ? ttg::getTotalElemsPerThread(tensor)
                                        : 1.0;
}

static std::optional<double> operationWork(Operation *op,
                                           double synchronization = 128.0,
                                           double bandwidth = 128.0,
                                           double overlap = 0.5) {
  if (op->hasAttr("tle.tile_dataflow.transport"))
    return 0.0;
  if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(op)) {
    if (auto components = ws->getAttrOfType<DenseF64ArrayAttr>("tle.tile_dataflow.costs");
        components && (components.size() >= 5 && components.size() <= 7)) {
      auto c = components.asArrayRef();
      double reuseBarriers = c.size() >= 6 ? c[5] : 0.0;
      return c[0] - overlap * (c[0] - c[1]) + c[2] + c[3] / bandwidth +
             synchronization * (1.0 + 0.25 * (c[4] - 2) +
                                reuseBarriers / 3.0 + (c.size() == 7 ? c[6] : 0.0));
    }
    return std::nullopt;
  }
  if (isa<arith::ConstantOp, triton::SplatOp, triton::MakeRangeOp,
          triton::ExpandDimsOp, triton::BroadcastOp, triton::AddPtrOp,
          triton::GetProgramIdOp, triton::ReturnOp>(op))
    return 0.0;
  if (auto reduction = dyn_cast<triton::ReduceOp>(op)) {
    ReduceOpHelper helper(reduction);
    double combine = 0;
    for (Operation &nested : reduction.getCombineOp().front().without_terminator()) {
      auto cost = operationWork(&nested);
      if (!cost)
        return std::nullopt;
      combine += *cost;
    }
    double input = elementsPerThread(reduction.getOperand(0).getType());
    double output = elementsPerThread(reduction->getResult(0).getType());
    return std::max(0.0, input - output) * combine +
           output * (4.0 * llvm::Log2_64_Ceil(
                               helper.getIntraWarpSizeWithUniqueData()) +
                     (helper.isWarpSynchronous() ? 0.0 : 32.0));
  }
  if (isa<ttg::LocalDeallocOp>(op))
    return 0.0;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    return alloc.getSrc() ? 2.0 * elementsPerThread(alloc.getSrc().getType()) : 0.0;
  if (isa<ttg::ConvertLayoutOp>(op)) {
    unsigned scratch = triton::defaultAllocationAnalysisScratchSizeFn(op);
    return elementsPerThread(op->getResult(0).getType()) +
           (scratch ? 32.0 + 2.0 * scratch / 128.0 : 0.0);
  }
  if (op->getNumRegions())
    return std::nullopt;
  double elements = 1;
  for (Type type : op->getResultTypes())
    elements = std::max(elements, elementsPerThread(type));
  if (isa<triton::LoadOp, ttg::LocalLoadOp>(op))
    return 2.0 * elements;
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return 2.0 * elementsPerThread(store.getValue().getType());
  if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
    return 2.0 * elementsPerThread(store.getSrc().getType());
  StringRef name = op->getName().getStringRef();
  if (name.starts_with("math."))
    return elements * (name == "math.absf" ? 1.0 : 8.0);
  if (name.starts_with("arith.")) {
    if (name == "arith.divf") {
      auto denominator = peelView(op->getOperand(1)).getDefiningOp<arith::ConstantOp>();
      return elements * (denominator ? 1.0 : 12.0);
    }
    // Indexing and predicate expressions commonly fold or share instructions.
    Type result = op->getNumResults() ? op->getResult(0).getType() : Type{};
    if (auto tensor = dyn_cast_or_null<RankedTensorType>(result))
      result = tensor.getElementType();
    return elements * (result && result.isIntOrIndex() ? 0.25 : 1.0);
  }
  // Unknown instructions/layout conversions need a target-specific model.
  return std::nullopt;
}

static std::optional<double> branchWork(ArrayRef<Operation *> ops) {
  double work = 0, activity = 1;
  for (Operation *op : ops) {
    auto cost = operationWork(op);
    if (!cost)
      return std::nullopt;
    work += *cost;
    if (auto load = dyn_cast<triton::LoadOp>(op); load && load.getMask())
      activity = std::min(activity, activeFraction(load.getMask()));
  }
  return work * activity;
}

// A persistent group keeps its own previous result in SSA. Cheap join DAGs are
// rematerialized by each reader after exchanging the other groups' results.
// This changes three dispatch phases into one counted, cross-group barrier.
// Only pass-owned transport and pure straight-line join work may be crossed.
static bool cheapJoin(Operation *op) {
  if (!isMemoryEffectFree(op) || op->getNumRegions())
    return false;
  StringRef name = op->getName().getStringRef();
  return name.starts_with("arith.") || name == "math.absf" ||
         isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp>(op);
}

static unsigned logicalWaves(ttg::WarpSpecializeOp ws) {
  auto attr = ws->getAttrOfType<IntegerAttr>("tle.tile_dataflow.logical_waves");
  return attr ? attr.getInt() : 1;
}

static unsigned persistentWARPhases(ttg::WarpSpecializeOp ws) {
  SmallVector<unsigned> phaseCosts(logicalWaves(ws));
  SmallVector<Region *> regions{&ws.getDefaultRegion()};
  llvm::append_range(regions, ws.getPartitionRegions());
  for (Region *region : regions) {
    unsigned phase = 0, barriers = 0;
    SmallVector<ttg::LocalLoadOp> reads;
    for (Operation &op : region->front()) {
      if (op.hasAttr("tle.tile_dataflow.full_group_sync")) {
        phaseCosts[phase] = std::max(phaseCosts[phase], barriers);
        ++phase;
        barriers = 0;
        reads.clear();
      } else if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
        reads.push_back(load);
      } else if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
        bool hazard = llvm::any_of(reads, [&](ttg::LocalLoadOp load) {
          // A local load can lower to cooperative ldmatrix instructions.
          // Identical SSA element ownership does not imply that each lane
          // issued the read for the address it will subsequently overwrite.
          return load.getSrc() == store.getDst();
        });
        if (hazard) {
          ++barriers;
          reads.clear();
        }
      }
    }
    phaseCosts[phase] = std::max(phaseCosts[phase], barriers);
  }
  unsigned total = 0;
  for (unsigned count : phaseCosts)
    total += count;
  return total;
}

static void phaseSync(OpBuilder &builder, Location loc, unsigned threads) {
  Value id = arith::ConstantIntOp::create(builder, loc, 16, 32);
  Value count = arith::ConstantIntOp::create(builder, loc, threads, 32);
  auto sync = ttng::NamedBarrierWaitOp::create(builder, loc, id, count);
  sync->setAttr("tle.tile_dataflow.full_group_sync", builder.getUnitAttr());
}

// Call only after the old first/second regions have been erased: they otherwise look like escaping
// users of each buffer. This helper does not run Membar; the normal lowering
// pipeline must run its existing memory hazard analysis afterwards.
struct PersistentPoolResult {
  unsigned reusedAllocations = 0;
  unsigned unusedAllocations = 0;
  // Conservative count, useful if the caller charges new local WAR barriers.
  unsigned samePhaseReusePairs = 0;
};

struct PersistentAccess {
  int phase;
  unsigned group;
  unsigned order;
};

struct PersistentBuffer {
  ttg::LocalAllocOp alloc;
  SmallVector<PersistentAccess> accesses;
  SmallVector<ttg::LocalDeallocOp> deallocs;
  int firstPhase = 0;
  int lastPhase = 0;
  Operation *lastOuterAccess = nullptr;
};

static PersistentPoolResult poolPersistentTransport(ttg::WarpSpecializeOp ws) {
  PersistentPoolResult result;
  if (logicalWaves(ws) < 2)
    return result;

  // A phase boundary is only a blocking rendezvous that covers every group.
  // Match its constant ID/count sequence in all regions before using it as a
  // proof of happens-before. Ordinary named barriers are not phase boundaries.
  auto constantInteger = [](Value value) -> std::optional<int64_t> {
    auto op = value.getDefiningOp<arith::ConstantOp>();
    if (!op)
      return std::nullopt;
    auto attr = dyn_cast<IntegerAttr>(op.getValue());
    return attr ? std::optional<int64_t>(attr.getInt()) : std::nullopt;
  };
  SmallVector<Region *> regions{&ws.getDefaultRegion()};
  llvm::append_range(regions, ws.getPartitionRegions());
  unsigned threads = 32 * (ttg::lookupNumWarps(ws) +
                           ws.getTotalPartitionWarps());
  SmallVector<std::pair<int64_t, int64_t>> rendezvous;
  llvm::DenseMap<Operation *, PersistentAccess> positions;
  int lastPhase = 0;
  for (auto [group, region] : llvm::enumerate(regions)) {
    if (!llvm::hasSingleElement(*region))
      return result;
    SmallVector<std::pair<int64_t, int64_t>> localRendezvous;
    int phase = 0;
    unsigned order = 0;
    for (Operation &op : region->front()) {
      positions[&op] = {phase, static_cast<unsigned>(group), order++};
      if (!op.hasAttr("tle.tile_dataflow.full_group_sync"))
        continue;
      auto barrier = dyn_cast<ttng::NamedBarrierWaitOp>(op);
      if (!barrier)
        return result;
      auto id = constantInteger(barrier.getBar());
      auto count = constantInteger(barrier.getNumThreads());
      if (!id || !count || *count != threads)
        return result;
      localRendezvous.emplace_back(*id, *count);
      ++phase;
    }
    if (group == 0) {
      rendezvous = localRendezvous;
      lastPhase = phase;
    } else if (localRendezvous != rendezvous) {
      return result;
    }
  }
  if (rendezvous.empty())
    return result;

  Block *outer = ws->getBlock();
  llvm::DenseMap<Operation *, unsigned> outerOrder;
  unsigned ordinal = 0;
  for (Operation &op : *outer)
    outerOrder[&op] = ordinal++;
  unsigned wsOrder = outerOrder.lookup(ws);

  // Consider only pass-owned, mutable allocations which dominate this region.
  // Checking every use below rejects aliases/views, nested control flow,
  // async consumers, another WS region, and escaping memdescs.
  SmallVector<PersistentBuffer> buffers;
  for (Operation &op : *outer) {
    if (&op == ws.getOperation())
      break;
    auto alloc = dyn_cast<ttg::LocalAllocOp>(op);
    if (!alloc || !op.hasAttr("tle.tile_dataflow.transport") ||
        !alloc.isSharedMemoryAlloc() || !alloc.getType().getMutableMemory())
      continue;
    PersistentBuffer buffer;
    buffer.alloc = alloc;
    buffer.lastOuterAccess = ws.getOperation();
    if (alloc.getSrc())
      buffer.accesses.push_back({-1, 0, outerOrder.lookup(alloc)});

    SmallVector<Value> descriptors{alloc.getResult()};
    bool captured = false;
    for (auto [index, value] : llvm::enumerate(ws.getExplicitCaptures())) {
      if (value != alloc.getResult())
        continue;
      captured = true;
      for (Region *region : ws.getPartitionRegions())
        descriptors.push_back(region->front().getArgument(index));
    }
    // Do not extend the scope to unrelated pass-owned temporary allocations.
    if (!captured)
      continue;
    bool valid = true;
    for (Value descriptor : descriptors) {
      for (OpOperand &use : descriptor.getUses()) {
        Operation *user = use.getOwner();
        if (user == ws.getOperation() && descriptor == alloc.getResult())
          continue;
        if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user)) {
          if (descriptor != alloc.getResult() || user->getBlock() != outer ||
              outerOrder.lookup(user) <= wsOrder ||
              !user->hasAttr("tle.tile_dataflow.transport")) {
            valid = false;
            break;
          }
          buffer.deallocs.push_back(dealloc);
          continue;
        }
        bool access = false;
        if (auto load = dyn_cast<ttg::LocalLoadOp>(user))
          access = load.getSrc() == descriptor;
        if (auto store = dyn_cast<ttg::LocalStoreOp>(user))
          access = store.getDst() == descriptor;
        if (!access) {
          valid = false;
          break;
        }
        auto position = positions.find(user);
        if (position != positions.end()) {
          buffer.accesses.push_back(position->second);
        } else if (user->getBlock() == outer &&
                   user->hasAttr("tle.tile_dataflow.transport")) {
          // Before dispatch and after its final join only the default group
          // accesses transport. These two intervals are separate phases.
          unsigned order = outerOrder.lookup(user);
          if (order > outerOrder.lookup(buffer.lastOuterAccess))
            buffer.lastOuterAccess = user;
          buffer.accesses.push_back(
              {order < wsOrder ? -1 : lastPhase + 1, 0, order});
        } else {
          valid = false;
          break;
        }
      }
      if (!valid)
        break;
    }
    if (!valid)
      continue;
    if (!buffer.accesses.empty()) {
      buffer.firstPhase = buffer.lastPhase = buffer.accesses.front().phase;
      for (const auto &access : buffer.accesses) {
        buffer.firstPhase = std::min(buffer.firstPhase, access.phase);
        buffer.lastPhase = std::max(buffer.lastPhase, access.phase);
      }
    }
    buffers.push_back(std::move(buffer));
  }

  // Strict cross-phase ordering permits different groups. At an equal phase
  // boundary all accesses on BOTH sides must belong to one group, and their
  // operation orders must be strictly disjoint. Membar then owns any needed
  // local WAR/WAW barrier. A reader in a second group always forbids reuse.
  auto ordered = [](const PersistentBuffer &a, const PersistentBuffer &b,
                    bool &samePhase) {
    if (a.lastPhase < b.firstPhase)
      return true;
    if (a.lastPhase != b.firstPhase)
      return false;
    int phase = a.lastPhase;
    std::optional<unsigned> group;
    unsigned lastA = 0;
    std::optional<unsigned> firstB;
    for (const auto &access : a.accesses) {
      if (access.phase != phase)
        continue;
      if (group && *group != access.group)
        return false;
      group = access.group;
      lastA = std::max(lastA, access.order);
    }
    for (const auto &access : b.accesses) {
      if (access.phase != phase)
        continue;
      if (!group || *group != access.group)
        return false;
      firstB = firstB ? std::min(*firstB, access.order) : access.order;
    }
    if (!firstB || lastA >= *firstB)
      return false;
    samePhase = true;
    return true;
  };

  struct Pool {
    unsigned owner;
    SmallVector<unsigned> members;
  };
  SmallVector<Pool> pools;
  for (unsigned i = 0; i < buffers.size(); ++i) {
    auto &buffer = buffers[i];
    if (buffer.accesses.empty()) {
      // A removed reconstructed input can remain in the old capture list.
      // Drop its unused arguments before deleting the no-op allocation.
      SmallVector<unsigned> indices;
      for (auto [index, value] : llvm::enumerate(ws.getExplicitCaptures()))
        if (value == buffer.alloc.getResult())
          indices.push_back(index);
      for (unsigned index : llvm::reverse(indices)) {
        for (Region *region : ws.getPartitionRegions()) {
          assert(region->front().getArgument(index).use_empty());
          region->front().eraseArgument(index);
        }
        ws->eraseOperand(index);
      }
      for (auto dealloc : buffer.deallocs)
        dealloc.erase();
      buffer.alloc.erase();
      ++result.unusedAllocations;
      continue;
    }

    Pool *chosen = nullptr;
    // Keep every live initialized allocation intact: moving its initializer
    // into a store changes transport accounting. Uninitialized allocations
    // may reuse an earlier initialized or uninitialized owner.
    if (!buffer.alloc.getSrc()) {
      for (auto &pool : pools) {
        auto owner = buffers[pool.owner].alloc;
        if (owner.getType() != buffer.alloc.getType() ||
            owner.getAlignmentOrDefault() !=
                buffer.alloc.getAlignmentOrDefault())
          continue;
        bool disjoint = true;
        for (unsigned member : pool.members) {
          bool samePhase = false;
          if (!ordered(buffers[member], buffer, samePhase) &&
              !ordered(buffer, buffers[member], samePhase)) {
            disjoint = false;
            break;
          }
        }
        if (disjoint) {
          chosen = &pool;
          break;
        }
      }
    }
    if (!chosen) {
      pools.push_back({i, {i}});
      continue;
    }
    Value owner = buffers[chosen->owner].alloc.getResult();
    buffer.alloc.getResult().replaceAllUsesWith(owner);
    buffer.alloc.erase();
    chosen->members.push_back(i);
    ++result.reusedAllocations;
  }

  // Canonicalize aliases inside isolated worker regions too. Keeping duplicate
  // capture operands is legal, but all uses must visibly resolve to one block
  // argument so the existing allocation/Membar analyses see one storage root.
  llvm::DenseMap<Value, unsigned> firstCapture;
  SmallVector<unsigned> duplicateCaptures;
  for (auto [index, value] : llvm::enumerate(ws.getExplicitCaptures())) {
    auto [it, inserted] = firstCapture.try_emplace(value, index);
    if (inserted || !isa<ttg::MemDescType>(value.getType()))
      continue;
    for (Region *region : ws.getPartitionRegions())
      region->front().getArgument(index).replaceAllUsesWith(
          region->front().getArgument(it->second));
    duplicateCaptures.push_back(index);
  }
  for (unsigned index : llvm::reverse(duplicateCaptures)) {
    for (Region *region : ws.getPartitionRegions())
      region->front().eraseArgument(index);
    ws->eraseOperand(index);
  }

  for (const auto &pool : pools) {
    if (pool.members.size() < 2)
      continue;
    SmallVector<unsigned> chronological(pool.members);
    llvm::sort(chronological, [&](unsigned a, unsigned b) {
      bool samePhase = false;
      return ordered(buffers[a], buffers[b], samePhase);
    });
    for (unsigned i = 1; i < chronological.size(); ++i) {
      bool samePhase = false;
      ordered(buffers[chronological[i - 1]], buffers[chronological[i]], samePhase);
      result.samePhaseReusePairs += samePhase;
    }
    // Existing deallocations were all after WS. Keep the latest and extend
    // it past any member with an implicit end; remove all earlier ends.
    Operation *lastAccess = ws.getOperation();
    for (unsigned member : pool.members)
      if (outerOrder.lookup(buffers[member].lastOuterAccess) >
          outerOrder.lookup(lastAccess))
        lastAccess = buffers[member].lastOuterAccess;
    ttg::LocalDeallocOp lastDealloc;
    for (unsigned member : pool.members)
      for (auto dealloc : buffers[member].deallocs)
        if (!lastDealloc || outerOrder.lookup(dealloc) >
                                outerOrder.lookup(lastDealloc))
          lastDealloc = dealloc;
    for (unsigned member : pool.members)
      for (auto dealloc : buffers[member].deallocs)
        if (dealloc != lastDealloc)
          dealloc.erase();
    if (lastDealloc &&
        outerOrder.lookup(lastDealloc) < outerOrder.lookup(lastAccess))
      lastDealloc->moveAfter(lastAccess);
    // When there was no explicit end, retain the allocator's implicit lifetime
    // rule. There is nothing to merge and inserting an early end is unsafe.
  }
  return result;
}

static bool mergeWaves(ttg::WarpSpecializeOp first,
                       ttg::WarpSpecializeOp second) {
  if (!first->hasAttr("tle.tile_dataflow") ||
      !second->hasAttr("tle.tile_dataflow") ||
      first.getPartitionNumWarps() != second.getPartitionNumWarps() ||
      first.getNumResults() != 1 || second.getNumResults() != 1)
    return false;
  auto ca = first->getAttrOfType<DenseF64ArrayAttr>("tle.tile_dataflow.costs");
  auto cb = second->getAttrOfType<DenseF64ArrayAttr>("tle.tile_dataflow.costs");
  if (!ca || !cb || ca.size() < 6 || cb.size() < 6)
    return false;
  SmallVector<Operation *> middle;
  OpSet between;
  for (Operation *op = first->getNextNode(); op != second; op = op->getNextNode()) {
    if (!op)
      return false;
    if (!cheapJoin(op) &&
        !(op->hasAttr("tle.tile_dataflow.transport") &&
          isa<ttg::LocalAllocOp, ttg::LocalLoadOp, ttg::LocalDeallocOp>(op)))
      return false;
    middle.push_back(op);
    between.insert(op);
  }
  // Scalar/tensor values cannot escape the moved default region. Memdescs are
  // hoisted and remain valid for the final output loads/deallocations.
  SmallVector<Operation *> checked{first.getOperation()};
  llvm::append_range(checked, middle);
  for (Operation *op : checked) {
    for (Value result : op->getResults()) {
      if (isa<ttg::MemDescType>(result.getType()))
        continue;
      for (Operation *user : result.getUsers()) {
        Operation *top = first->getBlock()->findAncestorOpInBlock(*user);
        if (top != second && !between.contains(top))
          return false;
      }
    }
  }
  llvm::SetVector<Value> operands;
  for (Value value : first.getExplicitCaptures())
    operands.insert(value);
  for (Value value : second.getExplicitCaptures())
    operands.insert(value);
  llvm::DenseMap<Value, Value> reconstructed; // shared buffer -> original tensor
  OpSet rematerialized;
  bool needsDefault = false;
  std::function<bool(Value)> available = [&](Value value) {
    if (value == first.getResult(0)) {
      needsDefault = true;
      return canStage(value.getType());
    }
    Operation *op = value.getDefiningOp();
    if (op && between.contains(op)) {
      if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
        return operands.contains(load.getSrc()) &&
               !between.contains(load.getSrc().getDefiningOp());
      if (!cheapJoin(op))
        return false;
      if (!rematerialized.insert(op).second)
        return true;
      return llvm::all_of(op->getOperands(), available);
    }
    if (op && isa<arith::ConstantOp>(op))
      return true;
    if (!isa<RankedTensorType, ttg::MemDescType>(value.getType())) {
      operands.insert(value);
      return true;
    }
    return false;
  };
  for (Value capture : second.getExplicitCaptures()) {
    auto alloc = capture.getDefiningOp<ttg::LocalAllocOp>();
    if (!alloc || !between.contains(alloc) || !alloc.getSrc())
      continue;
    Operation *source = alloc.getSrc().getDefiningOp();
    if (!source || (!between.contains(source) && source != first)) {
      // An independent capture may be staged before the persistent dispatch.
      continue;
    }
    if (!available(alloc.getSrc()))
      return false;
    reconstructed[capture] = alloc.getSrc();
  }
  if (reconstructed.empty())
    return false;
  // Every use of a reconstructed capture inside a worker must be a direct
  // tensor load or output store. No views/escaping descriptors are rewritten.
  for (auto [part, region] : llvm::enumerate(second.getPartitionRegions())) {
    Block &worker = region->front();
    for (auto [index, capture] : llvm::enumerate(second.getExplicitCaptures())) {
      if (!reconstructed.count(capture))
        continue;
      for (Operation *user : worker.getArgument(index).getUsers())
        if (!isa<ttg::LocalLoadOp, ttg::LocalStoreOp>(user))
          return false;
    }
  }
  double recompute = 0;
  for (Operation *op : rematerialized) {
    auto work = operationWork(op);
    if (!work)
      return false;
    recompute += *work;
  }
  // Bound duplication independently of kernel size. A long arithmetic tail
  // belongs in its own scheduled phase, rather than being copied into workers.
  if (recompute > 64)
    return false;

  OpBuilder builder(first);
  Location loc = first.getLoc();
  // Reuse a removed next-phase input for the previous default result. A
  // worker may later overwrite this private input; existing WAR analysis owns
  // its group-local load/store barrier.
  Value defaultBuffer;
  if (needsDefault)
    for (Value capture : second.getExplicitCaptures())
      if (reconstructed.count(capture) && capture.getType() ==
          sharedType(cast<RankedTensorType>(first.getResult(0).getType()))) {
        // A previously private input becomes a broadcast when several workers
        // rebuild the join. Its owner must not overwrite another reader.
        bool written = false;
        unsigned index = llvm::find(second.getExplicitCaptures(), capture) -
                         second.getExplicitCaptures().begin();
        for (Region *region : second.getPartitionRegions())
          for (Operation *user : region->front().getArgument(index).getUsers())
            written |= isa<ttg::LocalStoreOp>(user);
        if (first.getPartitionNumWarps().size() > 1 && written)
          continue;
        defaultBuffer = capture;
        break;
      }
  SmallVector<ttg::LocalDeallocOp> deallocs;
  for (Operation *op : middle) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (reconstructed.count(alloc.getResult()))
        alloc.getSrcMutable().clear();
      // Only independent sources or removed reconstructed sources are hoisted.
      if (alloc.getSrc() && between.contains(alloc.getSrc().getDefiningOp()))
        llvm_unreachable("unstaged join capture");
      alloc->moveBefore(first);
    } else if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(op)) {
      deallocs.push_back(dealloc);
    }
  }
  builder.setInsertionPoint(first);
  if (needsDefault && !defaultBuffer) {
    defaultBuffer = ttg::LocalAllocOp::create(builder, loc,
        sharedType(cast<RankedTensorType>(first.getResult(0).getType())));
    markTransport(defaultBuffer.getDefiningOp());
    operands.insert(defaultBuffer);
  }
  SmallVector<int32_t> warps(first.getPartitionNumWarps());
  auto merged = ttg::WarpSpecializeOp::create(builder, loc,
      second.getResultTypes(), warps, warps.size());
  merged->setOperands(operands.getArrayRef());
  merged->setAttr("tle.tile_dataflow", builder.getUnitAttr());
  merged->setAttr("tle.tile_dataflow.logical_waves",
      builder.getI32IntegerAttr(logicalWaves(first) + logicalWaves(second)));
  SmallVector<Attribute> waveCosts;
  for (auto ws : {first, second}) {
    if (auto costs = ws->getAttrOfType<ArrayAttr>("tle.tile_dataflow.wave_costs"))
      llvm::append_range(waveCosts, costs);
    else
      waveCosts.push_back(ws->getAttr("tle.tile_dataflow.costs"));
  }
  merged->setAttr("tle.tile_dataflow.wave_costs", builder.getArrayAttr(waveCosts));
  unsigned threads = 32 * (ttg::lookupNumWarps(first) + first.getTotalPartitionWarps());
  Block &main = merged.getDefaultRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&main);
  IRMapping mainMap;
  for (Operation &op : first.getDefaultRegion().front().without_terminator())
    builder.clone(op, mainMap);
  Value firstRoot = mainMap.lookupOrDefault(first.getDefaultRegion().front().getTerminator()->getOperand(0));
  mainMap.map(first.getResult(0), firstRoot);
  if (needsDefault)
    ttg::LocalStoreOp::create(builder, loc, toSharedStorage(builder, loc, firstRoot), defaultBuffer);
  phaseSync(builder, loc, threads);
  for (Operation *op : middle)
    if (!isa<ttg::LocalAllocOp, ttg::LocalDeallocOp>(op))
      builder.clone(*op, mainMap);
  for (Operation &op : second.getDefaultRegion().front().without_terminator())
    builder.clone(op, mainMap);
  Value finalRoot = mainMap.lookupOrDefault(second.getDefaultRegion().front().getTerminator()->getOperand(0));
  ttg::WarpYieldOp::create(builder, loc, ValueRange{finalRoot});

  for (unsigned p = 0; p < warps.size(); ++p) {
    Block &worker = merged.getPartitionRegions()[p]->emplaceBlock();
    IRMapping mapping;
    for (Value value : operands)
      mapping.map(value, worker.addArgument(value.getType(), loc));
    Block &oldFirst = first.getPartitionRegions()[p]->front();
    for (auto [index, value] : llvm::enumerate(first.getExplicitCaptures()))
      mapping.map(oldFirst.getArgument(index), mapping.lookup(value));
    builder.setInsertionPointToEnd(&worker);
    llvm::DenseMap<Value, Value> retained;
    for (Operation &op : oldFirst.without_terminator()) {
      builder.clone(op, mapping);
      if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
        auto argument = dyn_cast<BlockArgument>(store.getDst());
        if (argument && argument.getOwner() == &oldFirst)
          retained[first.getExplicitCaptures()[argument.getArgNumber()]] = mapping.lookupOrDefault(store.getSrc());
      }
    }
    phaseSync(builder, loc, threads);
    std::function<Value(Value)> rebuild = [&](Value value) -> Value {
      if (mapping.contains(value))
        return mapping.lookup(value);
      if (value == first.getResult(0)) {
        Value local = loadShared(builder, loc, cast<RankedTensorType>(value.getType()), mapping.lookup(defaultBuffer));
        mapping.map(value, local);
        return local;
      }
      Operation *op = value.getDefiningOp();
      if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
        Value local = retained.lookup(load.getSrc());
        if (!local)
          local = ttg::LocalLoadOp::create(builder, loc, load.getType(), mapping.lookup(load.getSrc()));
        mapping.map(value, local);
        return local;
      }
      for (Value input : op->getOperands())
        rebuild(input);
      builder.clone(*op, mapping);
      return mapping.lookup(value);
    };
    Block &oldSecond = second.getPartitionRegions()[p]->front();
    for (auto [index, value] : llvm::enumerate(second.getExplicitCaptures()))
      mapping.map(oldSecond.getArgument(index), mapping.lookup(value));
    for (Operation &op : oldSecond.without_terminator()) {
      if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
        auto arg = dyn_cast<BlockArgument>(load.getSrc());
        if (arg && arg.getOwner() == &oldSecond) {
          Value original = reconstructed.lookup(second.getExplicitCaptures()[arg.getArgNumber()]);
          if (original) {
            mapping.map(load.getResult(), rebuild(original));
            continue;
          }
        }
      }
      builder.clone(op, mapping);
    }
    ttg::WarpReturnOp::create(builder, loc);
  }
  // Composite cost retains original per-wave components for diagnostics, but
  // prices actual communication and join duplication in the persistent region.
  auto a = ca.asArrayRef(), b = cb.asArrayRef();
  SmallVector<double> costs(7);
  for (unsigned i = 0; i < 4; ++i)
    costs[i] = a[i] + b[i];
  costs[2] += recompute * warps.size();
  for (Operation *op : middle)
    if (!isa<ttg::LocalAllocOp, ttg::LocalDeallocOp>(op))
      costs[2] += operationWork(op).value_or(0);
  // Count actual loads/stores, including broadcasts to other workers. The
  // last worker result is consumed outside this region and still pays a read.
  costs[3] = 0;
  merged.walk([&](Operation *op) {
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
      costs[3] += tileBytes(load.getType());
    if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
      costs[3] += tileBytes(store.getSrc().getType());
  });
  for (Value value : operands)
    if (auto alloc = value.getDefiningOp<ttg::LocalAllocOp>(); alloc && alloc.getSrc())
      costs[3] += tileBytes(alloc.getSrc().getType());
  // Only stores of the last phase escape through the old second-wave outputs.
  for (Region *region : second.getPartitionRegions())
    for (auto store : region->front().getOps<ttg::LocalStoreOp>())
      costs[3] += tileBytes(store.getSrc().getType());
  costs[4] = a[4];
  costs[5] = a[5] + b[5];
  // A merged boundary pays one phase instead of the second wave's three.
  costs[6] = (a.size() == 7 ? a[6] : 0) + (b.size() == 7 ? b[6] : 0) + 1.0 / 3;
  merged->setAttr("tle.tile_dataflow.costs", builder.getDenseF64ArrayAttr(costs));
  merged->setAttr("tle.tile_dataflow.work", builder.getF64FloatAttr(*operationWork(merged)));
  second.getResult(0).replaceAllUsesWith(merged.getResult(0));
  for (auto dealloc : deallocs)
    dealloc->moveAfter(merged);
  if (needsDefault && !reconstructed.count(defaultBuffer)) {
    builder.setInsertionPointAfter(merged);
    markTransport(ttg::LocalDeallocOp::create(builder, loc, defaultBuffer));
  }
  second.erase();
  for (Operation *op : llvm::reverse(middle))
    if (!isa<ttg::LocalAllocOp, ttg::LocalDeallocOp>(op))
      op->erase();
  first.erase();
  auto pooled = poolPersistentTransport(merged);
  if (pooled.reusedAllocations || pooled.unusedAllocations) {
    merged->setAttr("tle.tile_dataflow.pooled_buffers",
        builder.getI32IntegerAttr(pooled.reusedAllocations));
  }
  costs[5] = persistentWARPhases(merged);
  merged->setAttr("tle.tile_dataflow.costs", builder.getDenseF64ArrayAttr(costs));
  merged->setAttr("tle.tile_dataflow.work", builder.getF64FloatAttr(*operationWork(merged)));
  return true;
}

static unsigned coalesceWavesIn(triton::FuncOp function) {
  unsigned count = 0;
  bool changed;
  do {
    changed = false;
    ttg::WarpSpecializeOp previous;
    for (auto ws : llvm::make_early_inc_range(function.getBody().front().getOps<ttg::WarpSpecializeOp>())) {
      if (previous && mergeWaves(previous, ws)) {
        ++count;
        changed = true;
        break;
      }
      previous = ws;
    }
  } while (changed);
  return count;
}

// Late extraction of an earlier indexing wave must not extend uninitialized
// communication buffers across that dispatch. Allocation follows first use;
// initialized captures stay where their producer write was scheduled.
static void sinkTransportAllocations(triton::FuncOp function) {
  Block &block = function.getBody().front();
  SmallVector<ttg::LocalAllocOp> allocations(block.getOps<ttg::LocalAllocOp>());
  for (auto alloc : allocations) {
    if (alloc.getSrc() || !alloc->hasAttr("tle.tile_dataflow.transport"))
      continue;
    Operation *first = nullptr;
    for (Operation *user : alloc.getResult().getUsers()) {
      Operation *top = block.findAncestorOpInBlock(*user);
      if (!top) {
        first = nullptr;
        break;
      }
      if (!first || top->isBeforeInBlock(first))
        first = top;
    }
    if (first && alloc->getNextNode() != first)
      alloc->moveBefore(first);
  }
}

static unsigned totalWarps(triton::FuncOp function) {
  unsigned base = ttg::lookupNumWarps(function), additional = 0;
  function->getParentOfType<ModuleOp>().walk([&](ttg::WarpSpecializeOp ws) {
    unsigned workers = 0;
    for (int32_t warps : ws.getPartitionNumWarps())
      workers += warps;
    additional = std::max(additional, workers);
  });
  // Match AllocateWarpGroups: worker storage is reused across regions and
  // module kernels, with the maximum worker count rounded to four warps.
  return base + llvm::alignTo(additional, 4u);
}

struct ScheduleCost {
  double serial = 0, parallel = 0, sync = 0, shared = 0, occupancy = 0;
  double recompute = 0, outside = 0, intrinsic = 0;
  double marginalSaving = 0, requiredSaving = 0, communicationReserve = 0;
  double marginalSpeedup = 0;
  SmallVector<double> branchCosts, components;
  uint64_t baseShared = 0, parallelShared = 0, transferBytes = 0;
  unsigned baseRegisters = 0, parallelRegisters = 0;
  unsigned baseCTAs = 0, parallelCTAs = 0;
  unsigned reuseBarriers = 0;
  unsigned coalesced = 0;
  bool profitable = false;
  std::string reason = "unknown_target";
};

static ScheduleCost estimateCost(ModuleOp module, triton::FuncOp function,
                                 Fork &fork, double synchronization,
                                 double bandwidth, double overlap,
                                 double margin, bool coalesce) {
  ScheduleCost cost;
  auto device = getTileDeviceResources(module);
  if (!device)
    return cost;
  OpSet selected;
  double sum = 0, longest = 0;
  llvm::SetVector<Value> staged;
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    auto work = branchWork(branch.ops);
    if (!work) {
      cost.reason = "unmodeled_branch_operation";
      return cost;
    }
    cost.branchCosts.push_back(*work);
    sum += *work;
    longest = std::max(longest, *work);
    selected.insert(branch.ops.begin(), branch.ops.end());
    if (i == fork.defaultBranch)
      continue;
    auto worker = branchWork(branch.worker);
    if (!worker) {
      cost.reason = "unmodeled_worker_operation";
      return cost;
    }
    cost.recompute += std::max(0.0, *worker - *work);
    cost.transferBytes += 2 * tileBytes(branch.root.getType());
    for (Value value : captures(branch.worker))
      if (isa<RankedTensorType>(value.getType())) {
        cost.transferBytes += tileBytes(value.getType()); // Each worker read.
        if (staged.insert(value))
          cost.transferBytes += tileBytes(value.getType()); // One producer write.
      }
  }
  for (Operation &op : function.getBody().front()) {
    if (selected.contains(&op))
      continue;
    auto work = operationWork(&op, synchronization, bandwidth, overlap);
    if (!work) {
      cost.reason = "unmodeled_surrounding_operation";
      return cost;
    }
    cost.outside += *work;
  }
  cost.serial = sum + cost.outside;
  // The common dispatch/join is paid once; extra workers add dispatch/capture
  // bookkeeping. The same coefficient reduces to the previous two-way model.
  // Reusing a worker's input storage for its output introduces a WAR hazard.
  // Existing membar analysis inserts a warp-group barrier before the write.
  // Workers run concurrently, so charge one such barrier on the critical path
  // when any worker reuses storage. The dispatch coefficient already covers
  // three synchronization phases; use one third for this additional phase.
  cost.reuseBarriers = llvm::any_of(reusableCaptures(fork),
                                  [](Value value) { return bool(value); });
  cost.sync = synchronization * (1.0 + 0.25 * (fork.branches.size() - 2) +
                                 cost.reuseBarriers / 3.0);
  cost.shared = cost.transferBytes / bandwidth;
  cost.intrinsic = sum - overlap * (sum - longest) + cost.sync +
                   cost.shared + cost.recompute;
  cost.components = {sum, longest, cost.recompute, double(cost.transferBytes),
                     double(fork.branches.size()), double(cost.reuseBarriers)};
  ModuleAllocation baseline(module);
  cost.baseShared = baseline.getSharedMemorySize(function);
  cost.baseRegisters = estimateTileRegistersPerThread(function);
  IRMapping mapping;
  OwningOpRef<ModuleOp> trial(cast<ModuleOp>(module->clone(mapping)));
  auto trialFunction = cast<triton::FuncOp>(mapping.lookup(function.getOperation()));
  Fork candidate{mapping.lookup(fork.join), {}, fork.defaultBranch};
  for (Branch &branch : fork.branches) {
    Branch clone{mapping.lookup(branch.root), {}, {}};
    for (Operation *op : branch.ops)
      clone.ops.push_back(mapping.lookup(op));
    for (Operation *op : branch.worker)
      clone.worker.push_back(mapping.lookup(op));
    candidate.branches.push_back(std::move(clone));
  }
  materialize(candidate, cost.intrinsic, cost.components);
  if (coalesce && (cost.coalesced = coalesceWavesIn(trialFunction))) {
    double trialWork = 0, trialBytes = 0, oldBytes = 0;
    double trialSync = 0, oldSync = 0;
    auto syncUnits = [](ArrayRef<double> c) {
      return 1.0 + 0.25 * (c[4] - 2) + (c.size() >= 6 ? c[5] / 3 : 0) +
             (c.size() == 7 ? c[6] : 0);
    };
    for (Operation &op : function.getBody().front())
      if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(op))
        if (auto c = ws->getAttrOfType<DenseF64ArrayAttr>("tle.tile_dataflow.costs")) {
          oldBytes += c.asArrayRef()[3];
          oldSync += syncUnits(c.asArrayRef());
        }
    for (Operation &op : trialFunction.getBody().front()) {
      trialWork += operationWork(&op, synchronization, bandwidth, overlap).value_or(0);
      if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(op))
        if (auto c = ws->getAttrOfType<DenseF64ArrayAttr>("tle.tile_dataflow.costs")) {
          trialBytes += c.asArrayRef()[3];
          trialSync += syncUnits(c.asArrayRef());
        }
    }
    cost.intrinsic = trialWork - cost.outside;
    cost.transferBytes = std::max(0.0, trialBytes - oldBytes);
    cost.shared = cost.transferBytes / bandwidth;
    cost.sync = std::max(0.0, (trialSync - oldSync) * synchronization);
    cost.recompute = std::max(0.0, cost.intrinsic -
        (sum - overlap * (sum - longest)) - cost.sync - cost.shared);
  }
  if (coalesce)
    sinkTransportAllocations(trialFunction);
  ModuleAllocation allocation(*trial);
  cost.parallelShared = allocation.getSharedMemorySize(trialFunction);
  cost.parallelRegisters = estimateTileRegistersPerThread(trialFunction);
  unsigned baseWarps = totalWarps(function);
  unsigned parallelWarps = totalWarps(trialFunction);
  cost.baseCTAs = estimateTileActiveCTAs(*device, baseWarps,
                                       cost.baseRegisters, cost.baseShared);
  cost.parallelCTAs = estimateTileActiveCTAs(
      *device, parallelWarps, cost.parallelRegisters, cost.parallelShared);
  if (!cost.baseCTAs || !cost.parallelCTAs) {
    cost.reason = "resource_limit";
    return cost;
  }
  double parallel = cost.intrinsic + cost.outside;
  double warpPenalty = std::max(1.0, double(cost.baseCTAs * baseWarps) /
                                        (cost.parallelCTAs * parallelWarps));
  double ctaPenalty = std::max(0.0, double(cost.baseCTAs) / cost.parallelCTAs - 1);
  cost.occupancy = parallel * (warpPenalty - 1) + cost.outside * ctaPenalty;
  cost.parallel = parallel + cost.occupancy;
  // Require headroom on the work actually changed by this candidate. Applying
  // the margin to unrelated work makes a useful wave fail merely because a
  // long serial tail or an earlier wave is present. Any resource regression
  // still charges its full whole-function occupancy cost against the saving.
  cost.marginalSaving = sum - cost.intrinsic - cost.occupancy;
  double marginalParallel = cost.intrinsic + cost.occupancy;
  cost.marginalSpeedup = marginalParallel > 0 ? sum / marginalParallel : 0;
  // These are coarse instruction weights rather than measured clock cycles.
  // Reserve the non-overlapped fraction of communication cost for estimation
  // error and contention, even when min-speedup=1. Use the larger headroom
  // requirement so the user margin and communication reserve are not added
  // twice. This is an acceptance budget, not extra executed work: old waves'
  // intrinsic cost components and the whole-function cost stay unchanged.
  cost.communicationReserve = (cost.sync + cost.shared) * (1.0 - overlap);
  cost.requiredSaving =
      std::max((margin - 1.0) * cost.intrinsic, cost.communicationReserve);
  cost.profitable = cost.marginalSaving >= cost.requiredSaving;
  cost.reason = cost.profitable ? "predicted_benefit" : "cost_exceeds_benefit";
  return cost;
}

static llvm::json::Object costRecord(const ScheduleCost &cost, Fork &fork,
                                     triton::FuncOp function, bool force,
                                     unsigned wave) {
  llvm::json::Array operands, branchCosts;
  for (auto [i, branch] : llvm::enumerate(fork.branches)) {
    if (i == fork.defaultBranch)
      continue;
    int64_t operand = -1;
    for (auto [index, value] : llvm::enumerate(fork.join->getOperands()))
      if (value == branch.root)
        operand = index;
    operands.push_back(operand);
  }
  for (double work : cost.branchCosts)
    branchCosts.push_back(work);
  int64_t operand = *operands.front().getAsInteger();
  return llvm::json::Object{
      {"function", function.getName().str()},
      {"join", fork.join->getName().getStringRef().str()},
      {"wave", wave}, {"branches", fork.branches.size()},
      {"default_branch", fork.defaultBranch},
      {"worker_operand", operand}, {"worker_operands", std::move(operands)},
      {"branch_costs", std::move(branchCosts)},
      {"decision", force ? "forced" : cost.profitable ? "parallel" : "serial"},
      {"reason", force ? "forced_legal_schedule" : cost.reason},
      {"serial_cost", cost.serial}, {"parallel_cost", cost.parallel},
      {"sync_cost", cost.sync}, {"shared_cost", cost.shared},
      {"reuse_barriers", cost.reuseBarriers},
      {"coalesced_boundaries", cost.coalesced},
      {"occupancy_cost", cost.occupancy}, {"recompute_cost", cost.recompute},
      {"outside_cost", cost.outside}, {"intrinsic_wave_cost", cost.intrinsic},
      {"marginal_saving", cost.marginalSaving},
      {"required_saving", cost.requiredSaving},
      {"communication_reserve", cost.communicationReserve},
      {"marginal_speedup", cost.marginalSpeedup},
      {"transfer_bytes", cost.transferBytes},
      {"baseline_shared_bytes", cost.baseShared},
      {"parallel_shared_bytes", cost.parallelShared},
      {"baseline_registers", cost.baseRegisters},
      {"parallel_registers", cost.parallelRegisters},
      {"baseline_active_ctas", cost.baseCTAs},
      {"parallel_active_ctas", cost.parallelCTAs},
      {"predicted_speedup", cost.parallel > 0 ? cost.serial / cost.parallel : 0}};
}

// Include the allocator's power-of-two padding partitions in every function
// when a candidate increases the module-wide maximum worker-warp allocation.
static bool fitsDispatchStates(triton::FuncOp function, unsigned branches) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  unsigned extra = (branches - 1) * ttg::lookupNumWarps(function);
  unsigned maximum = extra;
  module.walk([&](ttg::WarpSpecializeOp ws) {
    maximum = std::max(maximum, ws.getTotalPartitionWarps());
  });
  maximum = llvm::alignTo(maximum, 4u);
  for (auto entry : module.getOps<triton::FuncOp>()) {
    if (ttg::lookupNumWarps(entry) + maximum > 32)
      return false;
    unsigned states = entry == function ? branches - 1 + llvm::popcount(maximum - extra) : 0;
    entry.walk([&](ttg::WarpSpecializeOp ws) {
      states += ws.getPartitionNumWarps().size() +
                llvm::popcount(maximum - ws.getTotalPartitionWarps());
    });
    if (states > 255)
      return false;
  }
  return true;
}

class TileDataflowSchedulePass
    : public impl::TritonTleTileDataflowScheduleBase<TileDataflowSchedulePass> {
public:
  using TritonTleTileDataflowScheduleBase::TritonTleTileDataflowScheduleBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    TritonTleTileDataflowScheduleBase::getDependentDialects(registry);
    registry.insert<ttng::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    if (minBranchOps < 1 || maxPartitions < 1 || maxSharedBytes < 1 ||
        !std::isfinite(minSpeedup) || minSpeedup < 1 ||
        !std::isfinite(syncCost) || syncCost < 0 ||
        !std::isfinite(sharedBandwidth) || sharedBandwidth <= 0 ||
        !std::isfinite(overlap) || overlap < 0 || overlap > 1) {
      getOperation().emitError("invalid tile dataflow limits or cost-model parameters");
      signalPassFailure();
      return;
    }
    if (maxPartitions < 2)
      return;
    // NVIDIA's existing warp-specialize lowering supplies branch execution,
    // local barriers and the all-group join. Other targets remain unchanged.
    auto target = getOperation()->getAttrOfType<StringAttr>("ttg.target");
    auto ctas = getOperation()->getAttrOfType<IntegerAttr>("ttg.num-ctas");
    auto warps = getOperation()->getAttrOfType<IntegerAttr>("ttg.num-warps");
    if (!target || !target.getValue().starts_with("cuda:") ||
        (ctas && ctas.getInt() != 1) || !warps || warps.getInt() < 4 ||
        warps.getInt() > 16 || !llvm::isPowerOf2_64(warps.getInt()) ||
        getOperation()->hasAttr("ttg.maxnreg"))
      return;
    llvm::json::Array decisions;
    if (auto previous = getOperation()->getAttrOfType<StringAttr>("tle.tile_dataflow.cost_model")) {
      auto parsed = llvm::json::parse(previous.getValue());
      if (parsed) {
        if (auto array = parsed->getAsArray())
          for (auto &value : *array)
            if (auto object = value.getAsObject()) {
              auto decision = object->getString("decision");
              if (decision && (*decision == "parallel" || *decision == "forced"))
                decisions.push_back(value);
            }
      } else {
        llvm::consumeError(parsed.takeError());
      }
    }
    bool newDecisions = false;
    unsigned capacity = std::min<unsigned>(maxPartitions, 32 / warps.getInt());
    for (auto function : getOperation().getOps<triton::FuncOp>()) {
      if (!function.isPublic() || function.getBody().getBlocks().size() != 1 ||
          hasUnmanagedThreadSemantics(function))
        continue;
      Block &block = function.getBody().front();
      unsigned wave = 0;
      function.walk([&](ttg::WarpSpecializeOp ws) {
        wave += ws->hasAttr("tle.tile_dataflow") ? logicalWaves(ws) : 0;
      });
      // Select globally across the current DAG, then rebuild dependencies and
      // re-price every candidate against the new resource envelope. Existing
      // waves stay graph boundaries but have an intrinsic work estimate.
      while (true) {
        TileDataflowGraph graph(block);
        auto forks = findForks(graph, minBranchOps, maxSharedBytes, capacity);
        if (forks.empty())
          break;
        std::optional<unsigned> best;
        ScheduleCost chosen;
        unsigned chosenWork = 0;
        for (auto [index, fork] : llvm::enumerate(forks)) {
          if (!fitsDispatchStates(function, fork.branches.size()))
            continue;
          unsigned work = 0;
          for (Branch &branch : fork.branches)
            work += branch.ops.size();
          if (forceSchedule && best &&
              (fork.branches.size() < forks[*best].branches.size() ||
               (fork.branches.size() == forks[*best].branches.size() &&
                work <= chosenWork)))
            continue;
          auto cost = estimateCost(getOperation(), function, fork, syncCost,
                                   sharedBandwidth, overlap, minSpeedup, coalesceWaves);
          bool better = !best;
          if (best && !forceSchedule) {
            if (cost.profitable != chosen.profitable)
              better = cost.profitable;
            else if (cost.parallel > 0)
              better = chosen.parallel == 0 ||
                  cost.serial - cost.parallel > chosen.serial - chosen.parallel;
          }
          if (forceSchedule)
            better = true;
          if (better) {
            best = index;
            chosen = cost;
            chosenWork = work;
          }
        }
        if (!best)
          break;
        decisions.push_back(costRecord(chosen, forks[*best], function,
                                       forceSchedule, wave));
        newDecisions = true;
        if (!forceSchedule && !chosen.profitable)
          break;
        materialize(forks[*best], chosen.intrinsic, chosen.components);
        if (coalesceWaves)
          coalesceWavesIn(function);
        if (coalesceWaves)
          sinkTransportAllocations(function);
        ++wave;
      }
    }
    if (newDecisions) {
      llvm::stable_sort(decisions, [](const llvm::json::Value &a, const llvm::json::Value &b) {
        auto *lhs = a.getAsObject();
        auto *rhs = b.getAsObject();
        auto lf = lhs->getString("function").value_or("");
        auto rf = rhs->getString("function").value_or("");
        return lf != rf ? lf < rf : lhs->getInteger("wave").value_or(0) < rhs->getInteger("wave").value_or(0);
      });
      getOperation()->setAttr("tle.tile_dataflow.cost_model", StringAttr::get(
          &getContext(), llvm::formatv("{0}", llvm::json::Value(std::move(decisions))).str()));
    }
  }
};
} // namespace
} // namespace mlir::triton::tle
