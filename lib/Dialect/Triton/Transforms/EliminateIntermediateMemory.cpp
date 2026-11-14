// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONELIMINATEINTERMEDIATEMEMORY
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
namespace tt = mlir::triton;

// The caller explicitly proves distinct argument storages do not alias.
// Here we prove every mutable storage is partitioned between CTAs and accesses
// use exactly the same lane addresses. A CTA barrier cannot replace a kernel
// boundary for a producer in a different CTA.
static BlockArgument pointerRoot(Value pointer) {
  for (unsigned depth = 0; pointer && depth < 32; ++depth) {
    if (auto arg = dyn_cast<BlockArgument>(pointer))
      return isa<tt::PointerType>(arg.getType()) ? arg : BlockArgument{};
    if (auto add = pointer.getDefiningOp<tt::AddPtrOp>())
      pointer = add.getPtr();
    else if (auto splat = pointer.getDefiningOp<tt::SplatOp>())
      pointer = splat.getSrc();
    else
      return {};
  }
  return {};
}

static Value stripSplatAndExtend(Value value) {
  for (unsigned depth = 0; value && depth < 16; ++depth) {
    if (auto splat = value.getDefiningOp<tt::SplatOp>())
      value = splat.getSrc();
    else if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
      value = ext.getIn();
    else if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
      value = ext.getIn();
    else
      break;
  }
  return value;
}

static bool isIntegerConstant(Value value, int64_t expected) {
  auto constant = stripSplatAndExtend(value).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getInt() == expected;
  if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue()))
    return dense.isSplat() &&
           dense.getSplatValue<APInt>().getSExtValue() == expected;
  return false;
}

static bool isProgramId(Value value) {
  auto pid = stripSplatAndExtend(value).getDefiningOp<tt::GetProgramIdOp>();
  return pid && static_cast<unsigned>(pid.getAxis()) == 0;
}

static bool isTileBase(Value value, int64_t extent) {
  auto mul = stripSplatAndExtend(value).getDefiningOp<arith::MulIOp>();
  return mul && ((isProgramId(mul.getLhs()) &&
                  isIntegerConstant(mul.getRhs(), extent)) ||
                 (isProgramId(mul.getRhs()) &&
                  isIntegerConstant(mul.getLhs(), extent)));
}

static int64_t independentTileExtent(Value pointer) {
  auto type = dyn_cast<RankedTensorType>(pointer.getType());
  auto addPtr = pointer.getDefiningOp<tt::AddPtrOp>();
  if (!type || type.getRank() != 1 || !addPtr)
    return 0;
  // Do not accept an additional offset on the base pointer. Such an offset
  // could wrap, or make two apparently independent argument ranges overlap.
  auto base = addPtr.getPtr().getDefiningOp<tt::SplatOp>();
  if (!base || !isa<BlockArgument>(base.getSrc()))
    return 0;
  auto add =
      stripSplatAndExtend(addPtr.getOffset()).getDefiningOp<arith::AddIOp>();
  if (!add)
    return 0;
  auto match = [&](Value rangeValue, Value baseValue) -> int64_t {
    auto range =
        stripSplatAndExtend(rangeValue).getDefiningOp<tt::MakeRangeOp>();
    int64_t extent = type.getDimSize(0);
    if (!range || range.getStart() != 0 || range.getEnd() != extent ||
        extent <= 0 || !isTileBase(baseValue, extent))
      return 0;
    return extent;
  };
  if (auto extent = match(add.getLhs(), add.getRhs()))
    return extent;
  return match(add.getRhs(), add.getLhs());
}

// Changing masked-off values is safe only if no lane can make them observable.
// Restrict propagation to known scalar arithmetic/math operations; the generic
// Elementwise trait also appears on packed inline assembly, which can mix
// lanes.
static bool isSafeSameLaneComputation(Operation *op, ArrayRef<int64_t> shape) {
  StringRef dialect = op->getName().getDialectNamespace();
  if ((dialect != "arith" && dialect != "math") || op->getNumRegions() ||
      op->getNumResults() == 0 || !op->hasTrait<OpTrait::Elementwise>() ||
      !op->hasTrait<OpTrait::AlwaysSpeculatableImplTrait>() ||
      !isMemoryEffectFree(op) || !isSpeculatable(op))
    return false;

  // Do not rely on conditional speculation using the original select's value:
  // that proof could become false after replacement. Also reject all integer
  // division/remainder/shift operations explicitly. In this LLVM version the
  // remainder ops carry Pure, despite division by zero being undefined for the
  // whole tensor if even one masked-off lane has a zero divisor.
  if (isa<arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
          arith::CeilDivUIOp, arith::FloorDivSIOp, arith::RemSIOp,
          arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(op))
    return false;

  auto sameShape = [&](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getShape() == shape &&
           isa<IntegerType, FloatType>(type.getElementType());
  };
  return llvm::all_of(op->getOperands(), sameShape) &&
         llvm::all_of(op->getResults(), sameShape);
}

// These built-in RN operations evaluate each f32 lane independently, including
// zero divisors and negative square roots, without exposing inactive lanes.
static bool isSafeMaskPropagationOperation(Operation *op,
                                           ArrayRef<int64_t> shape) {
  if (isSafeSameLaneComputation(op, shape))
    return true;
  if (!isa<tt::PreciseSqrtOp, tt::PreciseDivFOp>(op) || op->getNumRegions() ||
      op->getNumResults() != 1 || !op->hasTrait<OpTrait::Elementwise>() ||
      !op->hasTrait<OpTrait::AlwaysSpeculatableImplTrait>() ||
      !isMemoryEffectFree(op) || !isSpeculatable(op))
    return false;

  // Match the supported language/intrinsic contract, not the broader float
  // types allowed by the dialect's operation definitions. External calls are
  // deliberately excluded: pure + Elementwise does not prove a substitutable
  // library implementation cannot exchange values between physical lanes.
  auto sameF32Shape = [&](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getShape() == shape && type.getElementType().isF32();
  };
  return llvm::all_of(op->getOperands(), sameF32Shape) &&
         llvm::all_of(op->getResults(), sameF32Shape);
}

static bool maskedOffLanesAreUnobservable(arith::SelectOp select) {
  auto type = dyn_cast<RankedTensorType>(select.getType());
  auto maskType = dyn_cast<RankedTensorType>(select.getCondition().getType());
  if (!type || !maskType || type.getShape() != maskType.getShape())
    return false;

  Value mask = select.getCondition();
  SmallVector<Value> worklist{select.getResult()};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getBlock() != select->getBlock())
        return false;
      if (auto store = dyn_cast<tt::StoreOp>(user)) {
        // Matching a Value is insufficient: it could also be used as a mask.
        if (&use != &store.getValueMutable() || store.getMask() != mask)
          return false;
        continue;
      }
      // A computed predicate is outside this proof, even for an elementwise
      // select. Memory addresses, guards, reductions and lane rearrangements
      // are likewise rejected by the lane-local propagation below.
      if (auto otherSelect = dyn_cast<arith::SelectOp>(user)) {
        if (&use == &otherSelect.getConditionMutable())
          return false;
      }
      if (!isSafeMaskPropagationOperation(user, type.getShape()))
        return false;
      worklist.append(user->result_begin(), user->result_end());
    }
  }
  return true;
}

struct Access {
  Operation *op;
  BlockArgument root;
  Value pointer;
};

// Count dependency depth, not the number of operations in the module: unrelated
// math and independent fanout branches must not consume each other's budget.
// This is a conservative profitability heuristic, not a legality restriction or
// a target latency model. In particular, sin/cos can expand into large library
// implementations. Joining long chains across a load changes backend scheduling
// and can lose more than the saved traffic, even with fewer registers/no
// spills.
static bool isTranscendental(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Cases("math.sin", "math.cos", "math.tan", true)
      .Cases("math.asin", "math.acos", "math.atan", "math.atan2", true)
      .Cases("math.sinh", "math.cosh", "math.tanh", true)
      .Cases("math.asinh", "math.acosh", "math.atanh", true)
      .Cases("math.exp", "math.exp2", "math.expm1", true)
      .Cases("math.log", "math.log2", "math.log10", "math.log1p", true)
      .Cases("math.erf", "math.erfc", "math.powf", "math.fpowi", "math.cbrt",
             true)
      .Default(false);
}

static bool regionsContainTranscendental(Operation *op) {
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        if (isTranscendental(&nested) || regionsContainTranscendental(&nested))
          return true;
  return false;
}

static bool
regionsUseUnknownTranscendental(Operation *op,
                                const DenseMap<Value, bool> &unknown) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (found || nested == op)
      return;
    for (Value operand : nested->getOperands())
      if (unknown.lookup(operand)) {
        found = true;
        return;
      }
  });
  return found;
}

static unsigned
regionsInfluencedMaxDepth(Operation *op,
                          const DenseMap<Value, bool> &influenced,
                          const DenseMap<Value, unsigned> &depth, bool &found) {
  unsigned maxDepth = 0;
  found = false;
  op->walk([&](Operation *nested) {
    if (nested == op)
      return;
    for (Value operand : nested->getOperands())
      if (influenced.lookup(operand)) {
        found = true;
        maxDepth = std::max(maxDepth, depth.lookup(operand));
      }
  });
  return maxDepth;
}

static uint64_t forwardedTranscendentalWork(unsigned depth, int64_t tile) {
  if (depth == 0 || tile <= 0)
    return 0;
  uint64_t extent = static_cast<uint64_t>(tile);
  const uint64_t max = std::numeric_limits<uint64_t>::max();
  // Saturate instead of wrapping for implausibly large static tiles.
  return static_cast<uint64_t>(depth) > max / extent
             ? max
             : static_cast<uint64_t>(depth) * extent;
}

static unsigned forwardedTranscendentalDepth(
    Block &block, ArrayRef<std::pair<tt::LoadOp, tt::StoreOp>> forwards,
    bool &nestedTranscendentalRisk) {
  DenseMap<Value, Value> replacements;
  for (auto [load, store] : forwards)
    replacements[load.getResult()] = store.getValue();

  DenseMap<Value, unsigned> originalDepth, plannedDepth;
  // Track whether a value depends on a forwarded load. Depth alone is not
  // enough: forwarding a cheap producer into a nested sin chain starts at
  // depth zero but still joins two computations.
  DenseMap<Value, bool> influenced;
  // A region that contains transcendental math has a producer depth the
  // entry-block model cannot measure. If such a value later feeds a forwarded
  // load, the joined chain is conservatively rejected.
  DenseMap<Value, bool> unknownTranscendentalDepth;
  unsigned maxJoinedDepth = 0;
  nestedTranscendentalRisk = false;
  // Process the entry block as straight-line SSA for the forwarding plan,
  // including chains through several intermediates, without mutating IR.
  for (Operation &op : block) {
    unsigned original = 0, planned = 0;
    bool opInfluenced = false;
    bool opUnknownTranscendental = false;
    bool hasRegions = !op.getRegions().empty();
    if (auto load = dyn_cast<tt::LoadOp>(op)) {
      if (auto it = replacements.find(load.getResult());
          it != replacements.end()) {
        planned = plannedDepth.lookup(it->second);
        opInfluenced = true;
        opUnknownTranscendental = unknownTranscendentalDepth.lookup(it->second);
        if (opUnknownTranscendental)
          nestedTranscendentalRisk = true;
      }
    } else if (isMemoryEffectFree(&op)) {
      for (Value operand : op.getOperands()) {
        original = std::max(original, originalDepth.lookup(operand));
        planned = std::max(planned, plannedDepth.lookup(operand));
        opInfluenced |= influenced.lookup(operand);
        opUnknownTranscendental |= unknownTranscendentalDepth.lookup(operand);
      }
      if (hasRegions) {
        bool nestedInfluenced = false;
        unsigned nestedDepth = regionsInfluencedMaxDepth(
            &op, influenced, plannedDepth, nestedInfluenced);
        opInfluenced |= nestedInfluenced;
        planned = std::max(planned, nestedDepth);
        opUnknownTranscendental |=
            regionsUseUnknownTranscendental(&op, unknownTranscendentalDepth);
        if (regionsContainTranscendental(&op)) {
          // A forwarded value entering such a region is outside the depth
          // model; a region result entering a forwarded store is rejected when
          // the store value is consumed below.
          if (opInfluenced)
            nestedTranscendentalRisk = true;
          opUnknownTranscendental = true;
        }
      }
      if (isTranscendental(&op)) {
        ++original;
        ++planned;
        // Only charge a chain extended by forwarding. A long producer followed
        // by cheap arithmetic, or unrelated long math, does not trigger this.
        if (planned > original)
          maxJoinedDepth = std::max(maxJoinedDepth, planned);
      }
    }
    for (Value result : op.getResults()) {
      originalDepth[result] = original;
      plannedDepth[result] = planned;
      influenced[result] = opInfluenced;
      unknownTranscendentalDepth[result] = opUnknownTranscendental;
    }
  }
  return maxJoinedDepth;
}

class TritonEliminateIntermediateMemory
    : public impl::TritonEliminateIntermediateMemoryBase<
          TritonEliminateIntermediateMemory> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp original = getOperation();
    // Rejection must preserve the original program, including failures found
    // after forwarding and dead-code cleanup. Only an accepted complete plan
    // publishes its body; a rejection publishes only its diagnostic
    // certificate.
    OwningOpRef<ModuleOp> candidate(cast<ModuleOp>(original->clone()));
    ModuleOp module = *candidate;
    unsigned maxForwardedTranscendentalDepth = 0;
    uint64_t maxForwardedTranscendentalWork = 0;
    auto report = [&](bool accepted, StringRef reason, unsigned loads = 0,
                      unsigned stores = 0, int64_t extent = 0,
                      unsigned maskSelects = 0) {
      llvm::json::Array privateArgs;
      if (accepted)
        for (unsigned index : privateArguments)
          privateArgs.push_back(index);
      llvm::json::Object object{
          {"accepted", accepted},
          {"reason", reason},
          {"forwarded_loads", loads},
          {"eliminated_stores", stores},
          {"eliminated_mask_selects", maskSelects},
          {"private_arguments", std::move(privateArgs)},
          {"max_forwarded_transcendental_depth",
           maxForwardedTranscendentalDepth},
          {"max_forwarded_transcendental_work", maxForwardedTranscendentalWork},
          {"max_tile_elements", extent}};
      if (accepted)
        original.getBodyRegion().takeBody(module.getBodyRegion());
      original->setAttr(
          "triton.intermediate_memory_elimination",
          StringAttr::get(
              &getContext(),
              llvm::formatv("{0}", llvm::json::Value(std::move(object)))
                  .str()));
    };

    if (!assumeDisjointArguments) {
      report(false, "requires an explicit disjoint argument storage contract");
      return;
    }
    auto ctaAttribute = module->getAttr("ttg.num-ctas");
    auto moduleCTAs = dyn_cast_or_null<IntegerAttr>(ctaAttribute);
    if (numCTAs != 1 || clusterSize != 1 ||
        (ctaAttribute && (!moduleCTAs || moduleCTAs.getInt() != 1))) {
      report(false, "intermediate memory elimination requires one CTA and a "
                    "unit cluster");
      return;
    }

    SmallVector<tt::FuncOp> entries;
    for (auto fn : module.getOps<tt::FuncOp>())
      if (fn.isPublic())
        entries.push_back(fn);
    if (entries.size() != 1 ||
        entries.front().getBody().getBlocks().size() != 1) {
      report(false, "requires a single-block entry function");
      return;
    }
    auto fn = entries.front();
    Block &block = fn.getBody().front();
    llvm::DenseSet<unsigned> privateArgs;
    for (unsigned index : privateArguments) {
      if (index >= block.getNumArguments() ||
          !isa<tt::PointerType>(block.getArgument(index).getType()) ||
          !privateArgs.insert(index).second) {
        report(false, "invalid private argument indices");
        return;
      }
    }

    SmallVector<Access> accesses;
    DenseMap<Value, Value> writtenPointers;
    int64_t maxExtent = 0;
    StringRef rejection;
    fn.walk([&](Operation *op) {
      if (!rejection.empty() || op == fn.getOperation())
        return;
      if (auto load = dyn_cast<tt::LoadOp>(op)) {
        auto root = pointerRoot(load.getPtr());
        if (!root || root.getOwner() != &block || op->getBlock() != &block ||
            load.getIsVolatile() || !load.getBoundaryCheck().empty()) {
          rejection = "unsupported load or nested memory access";
          return;
        }
        accesses.push_back({op, root, load.getPtr()});
      } else if (auto store = dyn_cast<tt::StoreOp>(op)) {
        auto root = pointerRoot(store.getPtr());
        int64_t extent = independentTileExtent(store.getPtr());
        if (!root || root.getOwner() != &block || op->getBlock() != &block ||
            !extent || !store.getBoundaryCheck().empty()) {
          rejection = "store is not a proven independent contiguous CTA tile";
          return;
        }
        auto [it, inserted] = writtenPointers.try_emplace(root, store.getPtr());
        if (!inserted && !privateArgs.contains(root.getArgNumber())) {
          rejection =
              "non-private mutable argument must have exactly one store";
          return;
        }
        if (!inserted && it->second != store.getPtr()) {
          rejection = "multiple address mappings for one mutable argument";
          return;
        }
        maxExtent = std::max(maxExtent, extent);
        accesses.push_back({op, root, store.getPtr()});
      } else if (!isa<tt::ReturnOp>(op) && !isMemoryEffectFree(op)) {
        rejection =
            "unresolved call, control flow, atomic or unknown side effect";
      }
      // A pointer used as a value (including integer casts) is observable, and
      // cannot be replaced with an unallocated private tensor's null pointer.
      for (Value operand : op->getOperands()) {
        if (!isa<tt::PointerType>(getElementTypeOrSelf(operand.getType())))
          continue;
        if (!isa<tt::SplatOp, tt::AddPtrOp, tt::LoadOp, tt::StoreOp>(op) ||
            (isa<tt::StoreOp>(op) &&
             operand == cast<tt::StoreOp>(op).getValue()))
          rejection = "pointer escapes through an unsupported operation";
      }
    });
    if (!rejection.empty()) {
      report(false, rejection);
      return;
    }
    if (writtenPointers.empty()) {
      report(false, "entry function has no supported mutable tensor output");
      return;
    }

    // Validate the entire plan before rewriting anything. If any dependency
    // would require another CTA, leave the original body intact.
    DenseMap<Value, tt::StoreOp> latestStores;
    SmallVector<std::pair<tt::LoadOp, tt::StoreOp>> forwards;
    SmallVector<tt::StoreOp> removals;
    for (const Access &access : accesses) {
      auto written = writtenPointers.find(access.root);
      if (written != writtenPointers.end() &&
          written->second != access.pointer) {
        report(false, "consumer address differs from producer CTA address");
        return;
      }
      if (auto store = dyn_cast<tt::StoreOp>(access.op)) {
        latestStores[access.root] = store;
        if (privateArgs.contains(access.root.getArgNumber()))
          removals.push_back(store);
        continue;
      }
      auto load = cast<tt::LoadOp>(access.op);
      auto previous = latestStores.find(access.root);
      if (previous == latestStores.end()) {
        if (privateArgs.contains(access.root.getArgNumber())) {
          report(false, "private intermediate is read before it is written");
          return;
        }
        if (written != writtenPointers.end()) {
          report(false,
                 "non-private mutable argument is loaded before its store");
          return;
        }
        continue;
      }
      auto store = previous->second;
      if (load.getType() != store.getValue().getType() ||
          (store.getMask() && store.getMask() != load.getMask())) {
        report(false, "producer and consumer masks or element types differ");
        return;
      }
      forwards.emplace_back(load, store);
    }

    bool nestedTranscendentalRisk = false;
    maxForwardedTranscendentalDepth =
        forwardedTranscendentalDepth(block, forwards, nestedTranscendentalRisk);
    maxForwardedTranscendentalWork =
        forwardedTranscendentalWork(maxForwardedTranscendentalDepth, maxExtent);
    if (maxTranscendentalDepth != 0 && nestedTranscendentalRisk) {
      report(false,
             "profitability: forwarding enters a nested region with "
             "transcendental operations",
             0, 0, maxExtent);
      return;
    }
    if (maxTranscendentalDepth != 0 &&
        maxForwardedTranscendentalDepth > maxTranscendentalDepth) {
      // Preserve the whole body and the allocation contract. Partially skipping
      // a private buffer must never certify that its allocation can be removed.
      report(false,
             "profitability: forwarding extends a transcendental chain "
             "beyond max-transcendental-depth",
             0, 0, maxExtent);
      return;
    }
    if (maxTranscendentalDepth != 0 && maxTranscendentalWork != 0 &&
        maxForwardedTranscendentalWork > maxTranscendentalWork) {
      // Static chain depth times tile elements is a conservative proxy for the
      // unrolled transcendental code expansion; it is deliberately not a
      // hardware latency model. Depth 0 keeps the existing complete override.
      report(false,
             "profitability: forwarding extends static transcendental work "
             "beyond max-transcendental-work",
             0, 0, maxExtent);
      return;
    }

    // Use the TTIR store operand, which already includes the conversion to
    // the intermediate storage dtype. Forwarding the original fp32 expression
    // would silently remove fp16/bf16 rounding at the storage boundary.
    SmallVector<arith::SelectOp> forwardedSelects;
    for (auto [load, store] : forwards) {
      Value value = store.getValue();
      if (load.getMask() && load.getOther()) {
        OpBuilder builder(load);
        auto select = arith::SelectOp::create(
            builder, load.getLoc(), load.getMask(), value, load.getOther());
        forwardedSelects.push_back(select);
        value = select.getResult();
      }
      load.replaceAllUsesWith(value);
      load.erase();
    }
    for (auto store : removals)
      store.erase();

    // Inspect the complete rewritten use DAG, so a proof can span several
    // inlined helpers. All paths must preserve the lane and end at a store
    // whose unchanged predicate discards the values that we may alter.
    unsigned eliminatedMaskSelects = 0;
    for (arith::SelectOp select : forwardedSelects) {
      if (!maskedOffLanesAreUnobservable(select))
        continue;
      select.replaceAllUsesWith(select.getTrueValue());
      select.erase();
      ++eliminatedMaskSelects;
    }

    // Drop dead pointer address chains and prove that private pointers have
    // no remaining uses before reporting successful elimination to the caller.
    bool changed;
    do {
      changed = false;
      for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block))) {
        if (!op.hasTrait<OpTrait::IsTerminator>() && isOpTriviallyDead(&op)) {
          op.erase();
          changed = true;
        }
      }
    } while (changed);
    for (unsigned index : privateArguments) {
      if (!block.getArgument(index).use_empty()) {
        report(false, "private intermediate still has observable uses");
        return;
      }
    }
    report(true, "same-CTA data dependencies proven", forwards.size(),
           removals.size(), maxExtent, eliminatedMaskSelects);
  }
};
} // namespace
} // namespace mlir::triton
