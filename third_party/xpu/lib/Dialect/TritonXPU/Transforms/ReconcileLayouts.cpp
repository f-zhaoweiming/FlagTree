//===----------------------------------------------------------------------===//
// TritonXPUReconcileLayouts
//
// The TLE path fixes the encoding of a loop-carried accumulator in CoreTiling
// (TLECoreTiling.cpp:198-210: a slice along dim 1 must use cpg=[g] so that it
// matches getOutput1D). The plain path has no equivalent, so the iter_arg keeps
// the type converter's default encoding while the reduction result that updates
// it is slice-encoded, and the arith verifier rejects the pair:
//
//   'arith.subf' op requires the same encoding for all operands and results
//
// (historically surfaced as a bogus OutOfResources("uni_sram")).
//
// This pass runs late in make_ttxir, after the passes that re-create such ops
// (normalize / vectorize / tile-analysis / cf-to-scf / loop-grid), and inserts
// triton_xpu.convert_layout so that values which must share an encoding do.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPURECONCILELAYOUTS
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

static bool sameShapeAndElement(Type a, Type b) {
  auto ra = dyn_cast<RankedTensorType>(a);
  auto rb = dyn_cast<RankedTensorType>(b);
  if (!ra || !rb || ra.getRank() == 0)
    return false;
  return ra.getShape() == rb.getShape() &&
         ra.getElementType() == rb.getElementType();
}

static bool requiresSameEncoding(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::CmpFOp, arith::CmpIOp,
             arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
             arith::OrIOp, arith::SelectOp>(op);
}

static LogicalResult reconcile(ModuleOp mod) {
  for (unsigned iter = 0; iter < 8; ++iter) {
    bool changed = false;
    // (1) scf.yield: the yielded value must carry the iter_arg's encoding.
    SmallVector<scf::YieldOp> yields;
    mod.walk([&](scf::YieldOp y) { yields.push_back(y); });
    for (scf::YieldOp y : yields) {
      auto forOp = dyn_cast<scf::ForOp>(y->getParentOp());
      if (!forOp)
        continue;
      auto iterArgs = forOp.getRegionIterArgs();
      for (unsigned i = 0; i < y.getNumOperands(); ++i) {
        if (i >= iterArgs.size())
          break;
        Type want = iterArgs[i].getType();
        Value v = y.getOperand(i);
        if (v.getType() == want || !sameShapeAndElement(want, v.getType()))
          continue;
        OpBuilder builder(y);
        Value cv =
            builder.create<triton::xpu::ConvertLayoutOp>(y.getLoc(), want, v);
        y->setOperand(i, cv);
        changed = true;
      }
    }
    // (2) elementwise ops whose operands must share the result encoding.
    SmallVector<Operation *> ops;
    mod.walk([&](Operation *op) {
      if (requiresSameEncoding(op))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      if (op->getNumResults() != 1)
        continue;
      auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!resTy || !resTy.getEncoding())
        continue;
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Type opTy = op->getOperand(i).getType();
        auto opTensorTy = dyn_cast<RankedTensorType>(opTy);
        if (!opTensorTy || !opTensorTy.getEncoding())
          continue;
        if (opTensorTy.getEncoding() == resTy.getEncoding())
          continue;
        if (!sameShapeAndElement(resTy, opTy))
          continue;
        OpBuilder builder(op);
        Value cv = builder.create<triton::xpu::ConvertLayoutOp>(
            op->getLoc(), resTy, op->getOperand(i));
        op->setOperand(i, cv);
        changed = true;
      }
    }
    if (!changed)
      break;
  }
  return success();
}

} // namespace

struct TritonXPUReconcileLayouts
    : public impl::TritonXPUReconcileLayoutsBase<TritonXPUReconcileLayouts> {
  using impl::TritonXPUReconcileLayoutsBase<
      TritonXPUReconcileLayouts>::TritonXPUReconcileLayoutsBase;

  void runOnOperation() override {
    if (failed(reconcile(getOperation())))
      signalPassFailure();
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
