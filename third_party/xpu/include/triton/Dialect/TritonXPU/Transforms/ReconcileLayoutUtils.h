//===----------------------------------------------------------------------===//
// Shared cluster-encoding reconciliation for the plain (non-TLE) path.
//
// CoreTiling fixes the encoding of loop-carried accumulators for the TLE path
// (see TLECoreTiling.cpp:198-210). The plain path has no equivalent, so a
// loop-carried value can keep the converter's default encoding while the
// reduction result feeding it is slice-encoded; the arith verifier then rejects
// the pair ("requires the same encoding for all operands and results").
//
// Insert triton_xpu.convert_layout where a value must match an operand/result
// or an scf.for iter_arg. Idempotent: returns the number of inserted ops.
//===----------------------------------------------------------------------===//
#ifndef TRITON_DIALECT_TRITONXPU_TRANSFORMS_RECONCILELAYOUTUTILS_H_
#define TRITON_DIALECT_TRITONXPU_TRANSFORMS_RECONCILELAYOUTUTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "triton/Dialect/TritonXPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace xpu {

namespace reconcile_detail {
inline bool sameShapeAndElement(Type a, Type b) {
  auto ra = dyn_cast<RankedTensorType>(a);
  auto rb = dyn_cast<RankedTensorType>(b);
  if (!ra || !rb || ra.getRank() == 0)
    return false;
  return ra.getShape() == rb.getShape() &&
         ra.getElementType() == rb.getElementType();
}

inline bool requiresSameEncoding(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::CmpFOp, arith::CmpIOp,
             arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
             arith::OrIOp, arith::SelectOp>(op);
}
} // namespace reconcile_detail

inline int64_t reconcileClusterLayouts(ModuleOp mod) {
  int64_t total = 0;
  for (unsigned iter = 0; iter < 8; ++iter) {
    bool changed = false;
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
        if (v.getType() == want ||
            !reconcile_detail::sameShapeAndElement(want, v.getType()))
          continue;
        OpBuilder builder(y);
        Value cv =
            builder.create<triton::xpu::ConvertLayoutOp>(y.getLoc(), want, v);
        y->setOperand(i, cv);
        changed = true;
        ++total;
      }
    }
    SmallVector<Operation *> ops;
    mod.walk([&](Operation *op) {
      if (reconcile_detail::requiresSameEncoding(op))
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
        if (!reconcile_detail::sameShapeAndElement(resTy, opTy))
          continue;
        OpBuilder builder(op);
        Value cv = builder.create<triton::xpu::ConvertLayoutOp>(
            op->getLoc(), resTy, op->getOperand(i));
        op->setOperand(i, cv);
        changed = true;
        ++total;
      }
    }
    if (!changed)
      break;
  }
  return total;
}

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif
