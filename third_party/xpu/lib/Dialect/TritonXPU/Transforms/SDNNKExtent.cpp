// Keep the physical K packing of both SDNN operands consistent with MMA.
// Load masks describe which source values need DMA; they do not change the
// logical reduction extent of a Triton dot. In particular, padded other values
// still belong to the input tensor and can be nonzero or nonfinite.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
namespace {
bool named(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}
bool canonicalMap(AffineMap map, int first, int second) {
  return map.getNumDims() == 3 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 2 &&
         map.getResult(0) == getAffineDimExpr(first, map.getContext()) &&
         map.getResult(1) == getAffineDimExpr(second, map.getContext());
}
constexpr StringLiteral kTag = "flagtree.sdnn.logical_k.";

void collectK(Location loc, llvm::SmallSet<int64_t, 4> &extents) {
  if (auto name = dyn_cast<NameLoc>(loc)) {
    StringRef text = name.getName().getValue();
    int64_t k;
    if (text.consume_front(kTag) && !text.getAsInteger(10, k) && k > 0)
      extents.insert(k);
    collectK(name.getChildLoc(), extents);
  } else if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations())
      collectK(child, extents);
  } else if (auto call = dyn_cast<CallSiteLoc>(loc)) {
    collectK(call.getCallee(), extents);
    collectK(call.getCaller(), extents);
  }
}

struct CaptureSDNNKExtent
    : PassWrapper<CaptureSDNNKExtent, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CaptureSDNNKExtent)
  StringRef getArgument() const final {
    return "tritonxpu-capture-sdnn-k-extent";
  }
  void runOnOperation() final {
    getOperation().walk([&](Operation *dot) {
      if (!named(dot, "tt.dot") || dot->getNumOperands() != 3)
        return;
      auto a = dyn_cast<RankedTensorType>(dot->getOperand(0).getType());
      auto b = dyn_cast<RankedTensorType>(dot->getOperand(1).getType());
      if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
          !a.hasStaticShape() || !b.hasStaticShape() ||
          !a.getElementType().isF32() || !b.getElementType().isF32() ||
          a.getDimSize(1) != b.getDimSize(0) || a.getDimSize(1) <= 0)
        return;
      auto tag = StringAttr::get(dot->getContext(),
                                 kTag.str() + std::to_string(a.getDimSize(1)));
      dot->setLoc(NameLoc::get(tag, dot->getLoc()));
    });
  }
};

// Only undo a compiler-generated prefix view of a DS result. The original
// tt.dot extent is retained in its location, so an intentional smaller dot
// cannot accidentally expand to a larger backing allocation.
Value fullOperand(Value operand, unsigned kDim, int64_t logicalK) {
  auto view = operand.getDefiningOp<tensor::ExtractSliceOp>();
  if (!view)
    return operand;
  auto sourceTy = dyn_cast<RankedTensorType>(view.getSource().getType());
  auto resultTy = dyn_cast<RankedTensorType>(operand.getType());
  auto ds = view.getSource().getDefiningOp();
  if (!named(ds, "sdnn.ds") || ds->getNumOperands() != 4 || !sourceTy ||
      !resultTy || sourceTy.getRank() != 2 || resultTy.getRank() != 2 ||
      !sourceTy.hasStaticShape() || logicalK <= 0 ||
      sourceTy.getDimSize(kDim) > logicalK || !view.getSource().hasOneUse() ||
      !operand.hasOneUse())
    return {};
  auto offsets = view.getMixedOffsets();
  auto strides = view.getMixedStrides();
  auto sizes = view.getMixedSizes();
  for (unsigned d = 0; d < 2; ++d) {
    if (getConstantIntValue(offsets[d]) != 0 ||
        getConstantIntValue(strides[d]) != 1)
      return {};
    if (d != kDim && getConstantIntValue(sizes[d]) != sourceTy.getDimSize(d))
      return {};
  }
  // Independent input masks produce independent prefix lengths; MMA can
  // combine them with max. Match each slice to its own DS mask, not to that
  // combined MMA length.
  Value maskedK = ds->getOperand(2 + kDim);
  if (auto dynamicK = dyn_cast<Value>(sizes[kDim])) {
    if (dynamicK != maskedK)
      return {};
  } else if (getConstantIntValue(sizes[kDim]) !=
             getConstantIntValue(OpFoldResult(maskedK))) {
    return {};
  }
  return view.getSource();
}

struct PreserveSDNNKExtent
    : PassWrapper<PreserveSDNNKExtent, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreserveSDNNKExtent)
  StringRef getArgument() const final {
    return "tritonxpu-preserve-sdnn-k-extent";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }
  void runOnOperation() final {
    getOperation().walk([&](Operation *mma) {
      if (!named(mma, "sdnn.mma") || mma->getNumOperands() != 7)
        return;
      auto maps = mma->getAttrOfType<ArrayAttr>("indexing_maps");
      if (!maps || maps.size() != 3)
        return;
      auto aMap = dyn_cast<AffineMapAttr>(maps[0]);
      auto bMap = dyn_cast<AffineMapAttr>(maps[1]);
      auto cMap = dyn_cast<AffineMapAttr>(maps[2]);
      if (!aMap || !bMap || !cMap || !canonicalMap(aMap.getValue(), 0, 2) ||
          !canonicalMap(bMap.getValue(), 2, 1) ||
          !canonicalMap(cMap.getValue(), 0, 1))
        return;
      llvm::SmallSet<int64_t, 4> extents;
      collectK(mma->getLoc(), extents);
      int64_t logicalK = extents.size() == 1 ? *extents.begin() : 0;
      Value aValue = fullOperand(mma->getOperand(0), 1, logicalK);
      Value bValue = fullOperand(mma->getOperand(1), 0, logicalK);
      if (!aValue || !bValue)
        return;
      Operation *a = aValue.getDefiningOp();
      Operation *b = bValue.getDefiningOp();
      if (!named(a, "sdnn.ds") || !named(b, "sdnn.ds") ||
          a->getNumOperands() != 4 || b->getNumOperands() != 4)
        return;
      auto aTy = dyn_cast<RankedTensorType>(a->getOperand(0).getType());
      auto bTy = dyn_cast<RankedTensorType>(b->getOperand(0).getType());
      if (!aTy || !bTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
          !aTy.hasStaticShape() || !bTy.hasStaticShape() ||
          aTy.getDimSize(1) != bTy.getDimSize(0) ||
          !aTy.getElementType().isF32() || !bTy.getElementType().isF32())
        return;
      int64_t k = aTy.getDimSize(1);
      if (k <= 0)
        return;
      mma->setOperand(0, aValue);
      mma->setOperand(1, bValue);
      // Insert each constant at its consumer; a DS can dominate the MMA from
      // an outer scope and must not use a value defined inside the MMA block.
      for (auto [op, index] :
           {std::pair<Operation *, unsigned>{a, 3}, {b, 2}, {mma, 6}}) {
        OpBuilder builder(op);
        Value extent = builder.create<arith::ConstantIndexOp>(op->getLoc(), k);
        op->setOperand(index, extent);
      }
    });
  }
};
} // namespace
namespace mlir::triton::xpu {
std::unique_ptr<Pass> createCaptureSDNNKExtentPass() {
  return std::make_unique<CaptureSDNNKExtent>();
}
std::unique_ptr<Pass> createPreserveSDNNKExtentPass() {
  return std::make_unique<PreserveSDNNKExtent>();
}
} // namespace mlir::triton::xpu
