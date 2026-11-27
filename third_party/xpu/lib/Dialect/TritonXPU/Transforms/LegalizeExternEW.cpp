//===----------------------------------------------------------------------===//
// Legalize unsupported tt.extern_elementwise ops for the non-SDNN XPU path.
//
// libdevice.fast_expf and libdevice.fast_dividef are not present in
// libdevice-xpu3.bc.  In the non-SDNN path these ops survive as unresolved
// external LLVM function calls and cause linker errors.  This pass rewrites
// them to native MLIR ops that are fully supported by TritonXPUToLLVM:
//
//   tt.extern_elementwise(%x)    {symbol="libdevice.fast_expf"}
//     -> math.exp(%x)
//       -> KL3 FP32 packet: llvm.exp.v16f32 -> grouped exp.f.rn
//   The instruction computes e^x, so the input needs no log2(e) scaling.
//
//   tt.extern_elementwise(%a,%b) {symbol="libdevice.fast_dividef"}
//   tt.extern_elementwise(%a,%b) {symbol="__triton_houyi_fast_fdividef"}
//     -> arith.divf %a, %b
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-legalize-extern-ew"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULEGALIZEEXTERNEW
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// fast_expf(x) = e^x
//
// KL3 FP32 packets reach exp.f.rn through the grouped LLVM pass.
// The hardware instruction exp.f.rn computes the natural exponential e^x
// directly (not 2^x), so no log2(e) pre-scaling is required.
//
// tt.extern_elementwise(%x) {symbol="libdevice.fast_expf"}
//   -> math.exp(%x)
struct FastExpfToMathExpPattern
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSymbol() != "libdevice.fast_expf")
      return failure();
    auto srcs = op.getSrcs();
    if (srcs.size() != 1)
      return failure();

    rewriter.replaceOpWithNewOp<math::ExpOp>(op, srcs[0]);
    return success();
  }
};

// tt.extern_elementwise(%a, %b) {symbol="libdevice.fast_dividef"}
// tt.extern_elementwise(%a, %b) {symbol="__triton_houyi_fast_fdividef"}
//   -> arith.divf %a, %b
struct FastDividefToArithDivfPattern
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    StringRef symbol = op.getSymbol();
    if (symbol != "libdevice.fast_dividef" &&
        symbol != "__triton_houyi_fast_fdividef")
      return failure();
    auto srcs = op.getSrcs();
    if (srcs.size() != 2)
      return failure();
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, srcs[0], srcs[1]);
    return success();
  }
};

// exp2(x) -> exp(x * ln2)
//
// The XPU backend maps math::ExpOp onto the ISA instruction exp.f.rn, which
// computes e^x (see the note above). math::Exp2Op has no dedicated lowering of
// its own, so it also ends up on exp.f.rn and returns e^x instead of 2^x --
// tl.exp2 is wrong on this backend. Rewriting exp2(x) as exp(x * ln2) restores
// 2^x semantics while still using the hardware instruction.
struct Exp2ToExpScaledPattern : public OpRewritePattern<math::Exp2Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::Exp2Op op,
                                PatternRewriter &rewriter) const override {
    Value x = op.getOperand();
    auto rt = dyn_cast<RankedTensorType>(x.getType());
    if (!rt || !rt.getElementType().isF32())
      return failure();
    auto loc = op.getLoc();
    Attribute ln2Attr = DenseElementsAttr::get(
        rt, rewriter.getF32FloatAttr(0.6931471805599453f));
    auto ln2 = rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(ln2Attr));
    auto scaled = rewriter.create<arith::MulFOp>(loc, rt, x, ln2);
    rewriter.replaceOpWithNewOp<math::ExpOp>(op, scaled);
    return success();
  }
};

// The cluster dtype legalization widens bf16 registers to f32. Preserve the
// observable rounding of an explicit f32 -> bf16 -> f32 before that pass.
struct PreserveBF16RoundTrip : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp ext,
                                PatternRewriter &b) const override {
    auto trunc = ext.getIn().getDefiningOp<arith::TruncFOp>();
    auto type = dyn_cast<RankedTensorType>(ext.getType());
    if (!trunc || !type || !type.getElementType().isF32() ||
        trunc.getIn().getType() != type ||
        !cast<RankedTensorType>(trunc.getType()).getElementType().isBF16() ||
        trunc.getRoundingmode().has_value())
      return failure();
    Location loc = ext.getLoc();
    auto i32 = type.clone(b.getI32Type());
    auto constant = [&](uint32_t value) -> Value {
      return b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(
          i32, b.getI32IntegerAttr(value)));
    };
    Value bits = b.create<triton::BitcastOp>(loc, i32, trunc.getIn());
    Value hi = b.create<arith::ShRUIOp>(loc, bits, constant(16));
    Value tie = b.create<arith::AndIOp>(loc, hi, constant(1));
    Value bias = b.create<arith::AddIOp>(loc, constant(0x7fff), tie);
    Value rounded = b.create<arith::AddIOp>(loc, bits, bias);
    Value abs = b.create<arith::AndIOp>(loc, bits, constant(0x7fffffff));
    Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, abs,
                                       constant(0x7f800000));
    Value quiet = b.create<arith::OrIOp>(loc, bits, constant(0x00400000));
    Value selected = b.create<arith::SelectOp>(loc, nan, quiet, rounded);
    Value result = b.create<arith::AndIOp>(loc, selected, constant(0xffff0000));
    b.replaceOpWithNewOp<triton::BitcastOp>(ext, type, result);
    return success();
  }
};

} // namespace

struct TritonXPULegalizeExternEW
    : public impl::TritonXPULegalizeExternEWBase<TritonXPULegalizeExternEW> {

  using impl::TritonXPULegalizeExternEWBase<
      TritonXPULegalizeExternEW>::TritonXPULegalizeExternEWBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // The KL3 scalar tanhf divides (e^x - e^-x) by (e^x + e^-x).
    // Bound its argument where FP32 tanh already rounds exactly to +/-1,
    // before an overflow can turn the quotient into Inf/Inf. Ordered
    // comparisons leave NaNs untouched for the original library call.
    getOperation().walk([](triton::ExternElementwiseOp op) {
      if (op.getSymbol() != "_ZN3xpu5tanhfEf" || op.getSrcs().size() != 1)
        return;
      Value x = op.getSrcs().front();
      auto type = dyn_cast<RankedTensorType>(x.getType());
      if (!type || !type.getElementType().isF32())
        return;
      OpBuilder b(op);
      Location loc = op.getLoc();
      auto constant = [&](float value) -> Value {
        return b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(
            type, b.getF32FloatAttr(value)));
      };
      Value lo = constant(-16.0f), hi = constant(16.0f);
      Value below = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, x, lo);
      Value above = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, x, hi);
      Value bounded = b.create<arith::SelectOp>(loc, below, lo, x);
      op->setOperand(0, b.create<arith::SelectOp>(loc, above, hi, bounded));
    });
    RewritePatternSet patterns(context);
    patterns.add<FastExpfToMathExpPattern, FastDividefToArithDivfPattern,
                 Exp2ToExpScaledPattern, PreserveBF16RoundTrip>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
