#include "RowSoftmax.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {
// The hardware DMA initializes a private per-core tile. Reject any later
// writer, escape, or observable scalar access, including through derived GEPs.
bool isReadOnlyTile(AllocaInst *tile, CallInst *copy, const DominatorTree &dom) {
  SmallVector<Value *> work{tile};
  SmallPtrSet<Value *, 32> seen;
  while (!work.empty()) {
    Value *ptr = work.pop_back_val();
    if (!seen.insert(ptr).second)
      continue;
    for (User *user : ptr->users()) {
      if (user == copy)
        continue;
      if (auto *gep = dyn_cast<GetElementPtrInst>(user)) {
        work.push_back(gep);
        continue;
      }
      auto *load = dyn_cast<LoadInst>(user);
      if (!load || !load->isSimple() || !dom.dominates(copy, load))
        return false;
    }
  }
  return true;
}

// Native Inductor can assemble one vector from 16 adjacent scalar loads of
// this tile. Read the packet from the same bytes once, in original lane order.
void foldBiasPacket(InsertElementInst *last, AllocaInst *tile) {
  auto *type = dyn_cast<FixedVectorType>(last->getType());
  if (!type || type->getNumElements() != 16 || !type->getElementType()->isFloatTy())
    return;
  const DataLayout &dl = last->getModule()->getDataLayout();
  Value *chain = last;
  int64_t lastOffset = 0;
  for (int lane = 15; lane >= 0; --lane) {
    auto *insert = dyn_cast<InsertElementInst>(chain);
    if (!insert)
      return;
    auto *index = dyn_cast<ConstantInt>(insert->getOperand(2));
    auto *load = dyn_cast<LoadInst>(insert->getOperand(1));
    if (!index || !index->equalsInt(lane) || !load || !load->isSimple())
      return;
    int64_t offset;
    if (GetPointerBaseWithConstantOffset(load->getPointerOperand(), offset, dl) != tile)
      return;
    if (lane == 15)
      lastOffset = offset;
    if (offset != lastOffset - 4 * (15 - lane))
      return;
    if (lane == 0) {
      // This operation owns the alloca. Align it before selecting a KL3
      // packet load; never strengthen alignment on external GM pointers.
      if (offset < 0 || offset % 64)
        return;
      tile->setAlignment(std::max(tile->getAlign(), Align(64)));
      IRBuilder<> builder(last);
      auto *packet = builder.CreateAlignedLoad(type, load->getPointerOperand(), Align(64));
      last->replaceAllUsesWith(packet);
      RecursivelyDeleteTriviallyDeadInstructions(last);
      return;
    }
    chain = insert->getOperand(0);
  }
}
} // namespace

void mlir::triton::xpu::reuseRowSoftmaxBias(Function &kernel) {
  // Both graph entries pass matrix, bias and a fresh output in this order.
  // Tensor descriptors may interleave sizes/strides between the GM pointers.
  Argument *bias = nullptr;
  unsigned gmPointers = 0;
  for (Argument &arg : kernel.args()) {
    if (arg.getType()->isPointerTy() && arg.getType()->getPointerAddressSpace() == 1 &&
        ++gmPointers == 2) {
      bias = &arg;
      break;
    }
  }
  if (!bias)
    return;
  DominatorTree dom(kernel);
  LoopInfo loops(dom);
  for (Instruction &inst : instructions(kernel)) {
    auto *dma = dyn_cast<CallInst>(&inst);
    if (!dma || !dma->getCalledFunction() ||
        dma->getCalledFunction()->getName() != "llvm.xpu.gm2lm_v3" ||
        dma->arg_size() != 4 || getUnderlyingObject(dma->getArgOperand(1)) != bias)
      continue;
    auto *tile = dyn_cast<AllocaInst>(dma->getArgOperand(0));
    auto *loop = loops.getLoopFor(dma->getParent());
    if (!tile || !loop || !isa<ConstantInt>(tile->getArraySize()) ||
        !isReadOnlyTile(tile, dma, dom))
      continue;
    while (loop->getParentLoop())
      loop = loop->getParentLoop();
    BasicBlock *preheader = loop->getLoopPreheader();
    if (!preheader || tile->getParent() != loop->getHeader() ||
        !loop->isLoopInvariant(dma->getArgOperand(1)) ||
        !loop->isLoopInvariant(dma->getArgOperand(2)) ||
        !loop->isLoopInvariant(dma->getArgOperand(3)))
      continue;
    // The preheader is reached only for active clusters. Require the first
    // iteration to reach the copy unconditionally, including nested row loops.
    BasicBlock *bb = loop->getHeader();
    SmallPtrSet<BasicBlock *, 4> visited;
    while (bb && bb != dma->getParent() && visited.insert(bb).second)
      bb = bb->getSingleSuccessor();
    if (bb != dma->getParent())
      continue;

    auto *fence = kernel.getParent()->getFunction("llvm.xpu.mfence.v2");
    if (!fence)
      continue;
    tile->moveBefore(preheader->getTerminator()->getIterator());
    dma->moveBefore(preheader->getTerminator()->getIterator());
    // Finish the preload before entering the loop. Its original fences stay
    // in place: a fence may also order transfers of the matrix or output.
    IRBuilder<> builder(preheader->getTerminator());
    builder.CreateCall(fence, {builder.getInt32(1)});

    SmallVector<WeakTrackingVH> packets;
    for (Instruction &i : instructions(kernel))
      if (auto *insert = dyn_cast<InsertElementInst>(&i))
        if (auto *lane = dyn_cast<ConstantInt>(insert->getOperand(2));
            lane && lane->equalsInt(15))
          packets.push_back(insert);
    for (Value *value : packets)
      if (auto *packet = dyn_cast_or_null<InsertElementInst>(value))
        foldBiasPacket(packet, tile);
    return;
  }
}
