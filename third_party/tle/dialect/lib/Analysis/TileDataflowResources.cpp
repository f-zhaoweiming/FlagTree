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

#include "tle/dialect/include/Analysis/TileDataflowResources.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>

namespace mlir::triton::tle {
namespace {

namespace ttg = mlir::triton::gpu;
using LiveSet = llvm::DenseSet<Value>;

static unsigned saturate(uint64_t value) {
  return std::min<uint64_t>(value, std::numeric_limits<unsigned>::max());
}

class RegisterEstimator {
public:
  unsigned estimate(triton::FuncOp function) {
    // Scalar kernel arguments, affine address arithmetic and specialization's
    // state machine need registers even when no distributed payload is live.
    constexpr unsigned scalarAndControlReserve = 12;
    return saturate(uint64_t(estimateRegion(function.getBody(), {})) +
                    scalarAndControlReserve);
  }

private:
  Value payloadRoot(Value value) {
    auto found = roots.find(value);
    if (found != roots.end())
      return found->second;
    Value root = value;
    if (Operation *op = value.getDefiningOp()) {
      // These views share their source registers. Follow uses through them so
      // ignoring the view does not prematurely end its source's lifetime.
      StringRef name = op->getName().getStringRef();
      if (op->getNumOperands() == 1 &&
          (name == "tt.expand_dims" || name == "tt.broadcast" ||
           name == "tt.reshape" || name == "tt.trans" || name == "tt.splat"))
        root = payloadRoot(op->getOperand(0));
    }
    roots[value] = root;
    return root;
  }

  bool isRecomputedIndex(Value value) {
    value = payloadRoot(value);
    if (!isa<RankedTensorType>(value.getType()))
      return true; // Scalar ingredients are covered by the fixed reserve.
    auto found = indexExpressions.find(value);
    if (found != indexExpressions.end())
      return found->second;
    // Install the conservative answer before descending through operands.
    indexExpressions[value] = false;
    Operation *op = value.getDefiningOp();
    if (!op)
      return false;
    bool result = isa<arith::ConstantOp, triton::MakeRangeOp>(op);
    auto tensor = cast<RankedTensorType>(value.getType());
    if (!result && tensor.getElementType().isIntOrIndex() &&
        op->getName().getDialectNamespace() == "arith" &&
        !op->getNumRegions() && isMemoryEffectFree(op) && isSpeculatable(op))
      result = llvm::all_of(op->getOperands(), [&](Value operand) {
        auto type = dyn_cast<RankedTensorType>(operand.getType());
        return (!type || type.getElementType().isIntOrIndex()) &&
               isRecomputedIndex(operand);
      });
    if (auto add = dyn_cast<triton::AddPtrOp>(op))
      result = isRecomputedIndex(add.getPtr()) &&
               isRecomputedIndex(add.getOffset());
    indexExpressions[value] = result;
    return result;
  }

  unsigned payload(Value value) {
    value = payloadRoot(value);
    auto found = payloads.find(value);
    if (found != payloads.end())
      return found->second;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || !tensor.hasStaticShape() || isRecomputedIndex(value))
      return payloads[value] = 0;
    Type element = tensor.getElementType();
    if (!element.isIntOrFloat() && !isa<triton::PointerType>(element))
      return payloads[value] = 0;
    unsigned bits = isa<triton::PointerType>(element)
                        ? 64
                        : element.getIntOrFloatBitWidth();
    uint64_t elements = tensor.getEncoding()
                            ? ttg::getTotalElemsPerThread(tensor)
                            : tensor.getNumElements();
    return payloads[value] =
               saturate(elements * llvm::divideCeil(bits, 32u));
  }

  unsigned liveRegisters(const LiveSet &live) {
    uint64_t result = 0;
    for (Value value : live)
      result += payload(value);
    return saturate(result);
  }

  unsigned estimateRegion(Region &region, const LiveSet &keepAlive) {
    unsigned peak = liveRegisters(keepAlive);
    for (Block &block : region)
      peak = std::max(peak, estimateBlock(block, keepAlive));
    return peak;
  }

  unsigned estimateBlock(Block &block, const LiveSet &keepAlive) {
    SmallVector<Operation *> operations;
    for (Operation &op : block)
      operations.push_back(&op);
    llvm::DenseMap<Value, unsigned> lastUse;
    for (auto [index, op] : llvm::enumerate(operations))
      for (Value operand : getNestedOperands(op)) {
        Value root = payloadRoot(operand);
        if (payload(root))
          lastUse[root] = index;
      }
    // Unknown cross-block control flow conservatively keeps outgoing values
    // live to the end of this block; block arguments account for incoming data.
    for (Operation *op : operations)
      for (Value value : op->getResults())
        if (llvm::any_of(value.getUsers(), [&](Operation *user) {
              return !block.findAncestorOpInBlock(*user);
            }))
          lastUse[payloadRoot(value)] = operations.size();

    LiveSet live(keepAlive);
    for (const auto &entry : lastUse) {
      Value value = entry.first;
      if (value.getParentBlock() != &block || isa<BlockArgument>(value))
        live.insert(value);
    }
    unsigned peak = liveRegisters(live);
    for (auto [index, op] : llvm::enumerate(operations)) {
      // An instruction may overwrite dead source registers. Counting sources
      // and results together would overestimate every elementwise chain.
      for (Value operand : getNestedOperands(op)) {
        Value root = payloadRoot(operand);
        if (!keepAlive.contains(root) && lastUse.lookup(root) == index)
          live.erase(root);
      }
      if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        peak = std::max(peak, estimateRegion(ws.getDefaultRegion(), live));
        // Worker warps have separate register files; they do not also carry
        // values that only the default group needs after the fork.
        for (Region *region : ws.getPartitionRegions())
          peak = std::max(peak, estimateRegion(*region, {}));
      } else {
        for (Region &region : op->getRegions())
          peak = std::max(peak, estimateRegion(region, live));
      }
      for (Value result : op->getResults()) {
        Value root = payloadRoot(result);
        if (payload(root) && lastUse.contains(root) &&
            lastUse.lookup(root) > index)
          live.insert(root);
      }
      peak = std::max(peak, liveRegisters(live));
    }
    return peak;
  }

  llvm::DenseMap<Value, Value> roots;
  llvm::DenseMap<Value, bool> indexExpressions;
  llvm::DenseMap<Value, unsigned> payloads;
};

} // namespace

std::optional<TileDeviceResources> getTileDeviceResources(ModuleOp module) {
  auto target = module->getAttrOfType<StringAttr>("ttg.target");
  if (!target)
    return std::nullopt;
  StringRef name = target.getValue();
  unsigned capability;
  if (!name.consume_front("cuda:") || name.getAsInteger(10, capability))
    return std::nullopt;
  // Architectural limits, including the 1 KiB per-CTA shared-memory reserve:
  // https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
  // https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
  // https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
  switch (capability) {
  case 80:
    return TileDeviceResources{65536, 164 * 1024, 64, 32};
  case 86:
    return TileDeviceResources{65536, 100 * 1024, 48, 16};
  case 89:
    return TileDeviceResources{65536, 100 * 1024, 48, 24};
  case 90:
    return TileDeviceResources{65536, 228 * 1024, 64, 32};
  default:
    return std::nullopt;
  }
}

unsigned estimateTileRegistersPerThread(triton::FuncOp function) {
  return RegisterEstimator().estimate(function);
}

unsigned estimateTileActiveCTAs(const TileDeviceResources &resources,
                                unsigned warps, unsigned registers,
                                uint64_t sharedBytes) {
  if (!warps || warps > 32 || registers > 255 ||
      sharedBytes > resources.sharedBytesPerSM ||
      resources.sharedReservedBytes > resources.sharedBytesPerSM - sharedBytes)
    return 0;
  uint64_t registersPerWarp =
      llvm::alignTo(uint64_t(std::max(registers, 1u)) * 32, uint64_t(256));
  uint64_t registersPerCTA = registersPerWarp * warps;
  uint64_t sharedPerCTA =
      llvm::alignTo(sharedBytes + resources.sharedReservedBytes, uint64_t(256));
  uint64_t active = std::min<uint64_t>(resources.maxCTAsPerSM,
                                      resources.maxWarpsPerSM / warps);
  active = std::min(active, resources.registersPerSM / registersPerCTA);
  if (sharedPerCTA)
    active = std::min(active, resources.sharedBytesPerSM / sharedPerCTA);
  return static_cast<unsigned>(active);
}

} // namespace mlir::triton::tle
