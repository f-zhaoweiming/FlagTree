#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Dialect/LLVMXPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mfence-check"

using namespace mlir;

namespace mlir {

namespace LLVM {
namespace XPU {

#define GEN_PASS_DEF_LLVMXPUMFENCECHECK
#include "triton/Dialect/LLVMXPU/Transforms/Passes.h.inc"

struct LLVMXPUMfenceCheckPass
    : public impl::LLVMXPUMfenceCheckBase<LLVMXPUMfenceCheckPass> {
  using impl::LLVMXPUMfenceCheckBase<
      LLVMXPUMfenceCheckPass>::LLVMXPUMfenceCheckBase;
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    OpBuilder builder(&getContext());
    insertMixedLocalReadFences(func, builder);

    // Find Next User Of This Buffer And Insert Mfence LM For Removing Conflict
    // Example: %31 And %29 Use Buffer %15, %31 Should Wait %29 Complete
    /*
      %28 = "triton_xpu.gm2lm_mask"(%27, %15)
      %29 = "triton_xpu.load"(%28)
      %30 = tt.addptr %arg3, %26
      %31 = "triton_xpu.gm2lm_mask"(%30, %15)
      %32 = "triton_xpu.load"(%31)
    */

    LLVM_DEBUG(llvm::dbgs() << "//---------- LLVMXPUMfenceCheckPass\n");

    for (auto &block : func.getBlocks()) {
      DenseSet<Value> pendingLoads;
      DenseSet<Value> pendingPtrs;
      // Track load pointers and later check for dependent users.
      for (auto &op : llvm::make_early_inc_range(block)) {

        for (auto opp : op.getOperands()) {
          if (pendingLoads.contains(opp)) {
            auto prevLoad = opp.getDefiningOp<LLVM::LoadOp>();
            if (pendingPtrs.contains(prevLoad.getAddr()))
              pendingPtrs.erase(prevLoad.getAddr());
          }
        }

        if (auto mfenceOp = dyn_cast<LLVM::XPU::MfenceOp>(&op)) {
          auto fenceValue = mfenceOp.getNum();
          pendingLoads.clear();
          pendingPtrs.clear();
          LLVM_DEBUG(llvm::dbgs() << "Clear Record\n" << op << "\n");
        } else if (auto gm2lmOp = dyn_cast<LLVM::XPU::GM2LMOp_v3>(&op)) {
          Value destBuffer = gm2lmOp.getDst();
          if (pendingPtrs.contains(destBuffer)) {
            builder.setInsertionPoint(&op);
            createMfenceOpForLM(builder, op.getLoc());
            pendingPtrs.erase(destBuffer);
            LLVM_DEBUG(llvm::dbgs() << "Insert Mfence For\n"
                                    << gm2lmOp << "\n");
          }
        } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op)) {
          Value ptr = loadOp.getAddr();
          pendingLoads.insert(loadOp);
          pendingPtrs.insert(ptr);
          LLVM_DEBUG(llvm::dbgs() << "Record Load\n"
                                  << loadOp << " | " << ptr << "\n");
        }
        // else if (auto phiOp = dyn_cast<LLVM::PHINode>(&op)) {
        // }
      }
    }
  }

private:
  // Scalar and SIMD accesses use different LM pipes. LLVM dependencies alone
  // do not order a scalar store before a vector load (or the reverse).
  // Keep independent allocations separate; an unrecognized pointer may alias
  // every allocation. Track writes through control-flow joins and backedges.
  struct LocalWrites {
    DenseMap<Value, unsigned> allocations;
    unsigned unknown = 0;

    void merge(const LocalWrites &other) {
      unknown |= other.unknown;
      for (auto entry : other.allocations)
        allocations[entry.first] |= entry.second;
    }
    bool operator==(const LocalWrites &other) const {
      if (unknown != other.unknown ||
          allocations.size() != other.allocations.size())
        return false;
      for (auto entry : allocations)
        if (other.allocations.lookup(entry.first) != entry.second)
          return false;
      return true;
    }
    unsigned aliases(Value allocation) const {
      unsigned result = unknown;
      if (allocation)
        return result | allocations.lookup(allocation);
      for (auto entry : allocations)
        result |= entry.second;
      return result;
    }
    void forget(Value allocation, unsigned bits) {
      unknown &= ~bits;
      for (auto it = allocations.begin(); it != allocations.end();) {
        auto current = it++;
        if (!allocation || current->first == allocation) {
          current->second &= ~bits;
          if (!current->second)
            allocations.erase(current);
        }
      }
    }
  };

  static bool isLocalPointer(Value pointer) {
    auto type = dyn_cast<LLVM::LLVMPointerType>(pointer.getType());
    return type && type.getAddressSpace() == 0;
  }

  static Value localAllocation(Value pointer) {
    while (Operation *definition = pointer.getDefiningOp()) {
      if (isa<LLVM::AllocaOp>(definition))
        return pointer;
      if (auto gep = dyn_cast<LLVM::GEPOp>(definition)) {
        pointer = gep.getBase();
      } else if (isa<LLVM::BitcastOp>(definition)) {
        pointer = definition->getOperand(0);
      } else {
        break;
      }
    }
    return {};
  }

  static bool fencesLocalMemory(Operation &operation) {
    auto fence = dyn_cast<LLVM::XPU::MfenceOp>(operation);
    if (!fence)
      return false;
    auto constant = fence.getNum().getDefiningOp<LLVM::ConstantOp>();
    if (!constant)
      return false;
    auto mask = dyn_cast<IntegerAttr>(constant.getValue());
    return mask && (mask.getInt() & 1);
  }

  void transferLocalWrites(Block &block, LocalWrites &state,
                           OpBuilder *inserter = nullptr) const {
    for (Operation &operation : llvm::make_early_inc_range(block)) {
      if (fencesLocalMemory(operation)) {
        state = LocalWrites{};
      } else if (auto store = dyn_cast<LLVM::StoreOp>(operation)) {
        if (!isLocalPointer(store.getAddr()))
          continue;
        unsigned kind = isa<VectorType>(store.getValue().getType()) ? 2 : 1;
        if (Value allocation = localAllocation(store.getAddr()))
          state.allocations[allocation] |= kind;
        else
          state.unknown |= kind;
      } else if (auto load = dyn_cast<LLVM::LoadOp>(operation)) {
        if (!isLocalPointer(load.getAddr()))
          continue;
        Value allocation = localAllocation(load.getAddr());
        unsigned otherPipe = isa<VectorType>(load.getType()) ? 1 : 2;
        if (inserter && (state.aliases(allocation) & otherPipe)) {
          inserter->setInsertionPoint(load);
          createMfenceOpForLM(*inserter, load.getLoc());
          state = LocalWrites{};
        } else {
          // A fixed kill set keeps the analysis monotone even when adding a
          // fence would also complete writes to unrelated allocations.
          state.forget(allocation, otherPipe);
        }
      }
    }
  }

  void insertMixedLocalReadFences(LLVM::LLVMFuncOp func, OpBuilder &builder) const {
    DenseMap<Block *, LocalWrites> incoming, outgoing;
    bool changed;
    do {
      changed = false;
      for (Block &block : func.getBlocks()) {
        LocalWrites input;
        for (Block *predecessor : block.getPredecessors())
          input.merge(outgoing[predecessor]);
        LocalWrites output = input;
        transferLocalWrites(block, output);
        if (!(input == incoming[&block]) || !(output == outgoing[&block])) {
          incoming[&block] = std::move(input);
          outgoing[&block] = std::move(output);
          changed = true;
        }
      }
    } while (changed);
    for (Block &block : func.getBlocks()) {
      LocalWrites state = incoming[&block];
      transferLocalWrites(block, state, &builder);
    }
  }

  void createMfenceOpForLM(OpBuilder &builder, mlir::Location loc) const {
    // The magic number 1 of MfenceOp means mfencing on LM (bit0=LM).
    // Mask bits are the xpu2 mfence template's (xtdk_sys_xpu2.h):
    // 1=mfence_lm, 2=mfence_sm, 4=mfence_gm, 3=lm_sm, 5=lm_gm, 7=all.
    // Order LM accesses, including mixed scalar/SIMD and the LM side of GM2LM.
    auto i32ty = builder.getIntegerType(32);
    auto one = builder.create<LLVM::ConstantOp>(loc, i32ty,
                                                IntegerAttr::get(i32ty, 1));
    builder.create<mlir::LLVM::XPU::MfenceOp>(loc, one);
  }
};

} // namespace XPU
} // namespace LLVM
} // namespace mlir
