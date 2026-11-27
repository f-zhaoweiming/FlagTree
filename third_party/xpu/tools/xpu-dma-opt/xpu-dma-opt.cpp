// Structural scheduling of KL3 local-memory DMA. No arithmetic replacement.
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <map>
#include <functional>
#include <climits>
#include <set>
using namespace llvm;

namespace {
struct Stats { unsigned loads=0, stores=0, pipelines=0, emptyRows=0; } stats;
static bool named(Instruction *I, StringRef N) {
  auto *C=dyn_cast<CallInst>(I);
  return C && C->getCalledFunction() && C->getCalledFunction()->getName()==N;
}
static bool fence(Instruction *I) {
  if (!named(I,"llvm.xpu.mfence.v2")) return false;
  auto *K=dyn_cast<ConstantInt>(cast<CallInst>(I)->getArgOperand(0));
  return K && K->equalsInt(1);
}
static bool pure(Instruction *I) {
  if (I->isTerminator() || isa<PHINode,AllocaInst>(I)) return false;
  if (auto *C=dyn_cast<CallInst>(I)) {
    if (auto *A=dyn_cast<InlineAsm>(C->getCalledOperand()))
      return !A->hasSideEffects() && !StringRef(A->getConstraintString()).contains("~{memory}");
  }
  return !I->mayReadOrWriteMemory() && !I->mayHaveSideEffects();
}
static AllocaInst *local(Value *P) {
  return dyn_cast<AllocaInst>(getUnderlyingObject(P));
}
static bool groupLoads(Function &F) {
  bool Changed=false;
  for (BasicBlock &BB:F) {
    SmallVector<Instruction *> Remove;
    for (Instruction &I:BB) {
      if (!named(&I,"llvm.xpu.gm2lm_v3")) continue;
      auto *First=cast<CallInst>(&I); auto *Wait=I.getNextNode();
      if (!Wait || !fence(Wait)) continue;
      Instruction *Next=Wait->getNextNode();
      while (Next && pure(Next)) Next=Next->getNextNode();
      if (!Next || !named(Next,"llvm.xpu.gm2lm_v3")) continue;
      auto *Second=cast<CallInst>(Next);
      auto *A=local(First->getArgOperand(0));
      auto *B=local(Second->getArgOperand(0));
      if (!A || !B || A==B) continue;
      Next=Next->getNextNode();
      while (Next && pure(Next)) Next=Next->getNextNode();
      if (!Next || !fence(Next)) continue;
      Remove.push_back(Wait);
    }
    for (auto *I:Remove) { I->eraseFromParent(); ++stats.loads; Changed=true; }
  }
  return Changed;
}

// The whole set of local loads/stores must access disjoint slices on successive
// iterations. A wait remains on the unique exit before any DMA consumes stores.
static bool hoistStores(Function &F, FunctionAnalysisManager &AM) {
  auto &LI=AM.getResult<LoopAnalysis>(F);
  auto &SE=AM.getResult<ScalarEvolutionAnalysis>(F);
  const DataLayout &DL=F.getParent()->getDataLayout();
  bool Changed=false;
  for (Loop *L:LI.getLoopsInPreorder()) {
    if (L->getNumBlocks()!=1 || !L->getUniqueExitBlock()) continue;
    auto *BB=L->getHeader(), *Exit=L->getUniqueExitBlock();
    if (Exit->getSinglePredecessor()!=BB) continue;
    SmallVector<Instruction *> Waits;
    Instruction *LastMemory=nullptr;
    bool OK=true,SawStore=false;
    struct Span { int64_t lo=INT64_MAX,hi=INT64_MIN,step=0; };
    struct Access { AllocaInst *base; int64_t lo,hi; };
    std::map<AllocaInst *,Span> Spans;
    SmallVector<Access> Accesses;
    for (Instruction &I:*BB) {
      if (fence(&I)) { Waits.push_back(&I); continue; }
      Value *Ptr=nullptr; Type *Ty=nullptr;
      if (auto *R=dyn_cast<LoadInst>(&I)) {
        if (R->isVolatile() || R->isAtomic()) { OK=false; break; }
        Ptr=R->getPointerOperand();Ty=R->getType();
        // LM waits do not order SM. A readonly SM column load neither aliases
        // an LM store nor needs an LM iteration-span proof. Writes/calls with
        // unknown SM effects are still rejected below.
        if (Ptr->getType()->getPointerAddressSpace()==2) {
          LastMemory=&I;
          continue;
        }
        // An intermediate wait may order a same-iteration store->load chain.
        // Keep that chain rather than treating disjoint iterations as enough.
        if (!Waits.empty()) { OK=false; break; }
      } else if (auto *W=dyn_cast<StoreInst>(&I)) {
        if (W->isVolatile() || W->isAtomic()) { OK=false; break; }
        Ptr=W->getPointerOperand();Ty=W->getValueOperand()->getType();SawStore=true;
      } else if (!pure(&I) && !isa<PHINode,BranchInst>(I)) { OK=false; break; }
      if (!Ptr) continue;
      LastMemory=&I;
      auto *A=local(Ptr); auto *AR=dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Ptr));
      if (!A || !AR || AR->getLoop()!=L || !AR->isAffine()) { OK=false; break; }
      auto *Step=dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
      auto *Start=dyn_cast<SCEVConstant>(SE.getMinusSCEV(AR->getStart(),SE.getSCEV(A)));
      if (!Step || !Start || DL.getTypeStoreSize(Ty).isScalable()) { OK=false; break; }
      int64_t S=Step->getAPInt().getSExtValue(),O=Start->getAPInt().getSExtValue();
      if (S<=0 || O<0) { OK=false; break; }
      uint64_t Bytes=DL.getTypeStoreSize(Ty).getFixedValue();
      if (Bytes>uint64_t(INT64_MAX-O)) { OK=false; break; }
      int64_t End=O+int64_t(Bytes);
      // After an intermediate wait, only independent stores may proceed.
      // Reject RAW/WAW overlap within this iteration as well as across it.
      if (!Waits.empty())
        for (const auto &A:Accesses)
          if (A.base==local(Ptr) && O<A.hi && A.lo<End) OK=false;
      if (!OK) break;
      Accesses.push_back({A,O,End});
      Span &P=Spans[A]; if (P.step && P.step!=S) { OK=false; break; }
      P.step=S;P.lo=std::min(P.lo,O);P.hi=std::max(P.hi,End);
    }
    for (auto &[A,P]:Spans) if (P.hi-P.lo>P.step) OK=false;
    if (OK && SawStore && !Waits.empty() && !Spans.empty() &&
        LastMemory && LastMemory->comesBefore(Waits.back())) {
      auto *Last=Waits.pop_back_val();
      for (auto *Wait:Waits) Wait->eraseFromParent();
      Last->moveBefore(&*Exit->getFirstInsertionPt());++stats.stores;Changed=true;
    }
  }
  return Changed;
}

// Only flat dense packets with a common element index are eligible. The DMA
// length is clamp(N-index,0,packetElements)*elementBytes. This proves the full
// allocation range used by the runtime alias guard, without model/shape names.
struct Packet { Value *bound=nullptr,*index=nullptr; uint64_t bytes=0,elem=0,period=0; };
static Value *stripIntCast(Value *V) {
  while (auto *C=dyn_cast<CastInst>(V)) {
    if (!C->isIntegerCast()) break; V=C->getOperand(0);
  }
  return V;
}
static Packet packet(CallInst *DMA,const DataLayout &DL) {
  Packet P; Value *Len=DMA->getArgOperand(3);uint64_t Elem=1;
  if (auto *B=dyn_cast<BinaryOperator>(Len)) {
    auto *K=dyn_cast<ConstantInt>(B->getOperand(1));
    if (B->getOpcode()==Instruction::Shl && K && K->getZExtValue()<4) {
      Elem=1ull<<K->getZExtValue();Len=B->getOperand(0);
    }
  }
  auto *Min=dyn_cast<CallInst>(Len);
  if (!Min || Min->getIntrinsicID()!=Intrinsic::umin) return P;
  auto *Cap=dyn_cast<ConstantInt>(Min->getArgOperand(1));
  auto *Max=dyn_cast<CallInst>(Min->getArgOperand(0));
  if (!Cap || !Max || Max->getIntrinsicID()!=Intrinsic::smax) return P;
  auto *Zero=dyn_cast<ConstantInt>(Max->getArgOperand(1));
  auto *Sub=dyn_cast<BinaryOperator>(Max->getArgOperand(0));
  if (!Zero || !Zero->isZero() || !Sub || Sub->getOpcode()!=Instruction::Sub) return P;
  auto *GP=dyn_cast<GetElementPtrInst>(DMA->getArgOperand(1));
  if (!GP || GP->getNumIndices()!=1 || !isa<Argument>(GP->getPointerOperand()) ||
      DL.getTypeAllocSize(GP->getSourceElementType())!=Elem) return P;
  Value *Index=stripIntCast(GP->idx_begin()->get());
  uint64_t Period=0;
  if (Index!=Sub->getOperand(1)) {
    auto *Rem=dyn_cast<BinaryOperator>(Index);
    if(!Rem || (Rem->getOpcode()!=Instruction::SRem && Rem->getOpcode()!=Instruction::URem) ||
       Rem->getOperand(0)!=Sub->getOperand(1))return P;
    auto *C=dyn_cast<ConstantInt>(Rem->getOperand(1));
    if(!C || C->getSExtValue()<=0 || C->getZExtValue()<Cap->getZExtValue())return P;
    Period=C->getZExtValue();
  }
  P={Sub->getOperand(0),Sub->getOperand(1),Cap->getZExtValue()*Elem,Elem,Period};return P;
}

static Value *cloneCone(Value *V, Loop *L, ValueToValueMapTy &Map, IRBuilder<> &B) {
  if (auto I=Map.find(V);I!=Map.end()) return I->second;
  auto *I=dyn_cast<Instruction>(V);
  if (!I || !L->contains(I)) return V;
  if (!pure(I)) return nullptr;
  SmallVector<Value *> Args;
  for (Value *O:I->operands()) { auto *C=cloneCone(O,L,Map,B);if(!C)return nullptr;Args.push_back(C); }
  auto *C=I->clone();for(unsigned J=0;J<Args.size();++J)C->setOperand(J,Args[J]);
  // Lookahead arithmetic is speculative on the final iteration. Do not
  // propagate flags which make an otherwise unused address poison.
  if(auto *G=dyn_cast<GetElementPtrInst>(C))G->setNoWrapFlags(GEPNoWrapFlags::none());
  if(auto *O=dyn_cast<BinaryOperator>(C)) {
    if(isa<OverflowingBinaryOperator>(O)){O->setHasNoUnsignedWrap(false);O->setHasNoSignedWrap(false);}
    if(isa<PossiblyExactOperator>(O))O->setIsExact(false);
  }
  B.Insert(C);Map[V]=C;return C;
}
static bool clonable(Value *V, Loop *L, PHINode *IV, SmallPtrSetImpl<Value *> &Seen) {
  if (V==IV || !Seen.insert(V).second) return true;
  auto *I=dyn_cast<Instruction>(V);if (!I || !L->contains(I)) return true;
  if (!pure(I)) return false;
  for(Value *O:I->operands())if(!clonable(O,L,IV,Seen))return false;
  return true;
}

// Admit only known local scratch operations; unknown calls remain barriers.
// Their ordering and fences are unchanged by the outer DMA pipeline.
static AllocaInst *scratchCall(CallInst *C) {
  if(!C->getCalledFunction())return nullptr;
  StringRef N=C->getCalledFunction()->getName();
  if(N=="llvm.lifetime.start.p0" || N=="llvm.lifetime.end.p0" ||
     N=="llvm.xpu.vload_mask16_mr1")return local(C->getArgOperand(0));
  if(N=="llvm.xpu.vstore_mask16_mr1")return local(C->getArgOperand(1));
  return nullptr;
}
// Reuse a fully consumed flat input for vector output, preserving an invariant
// broadcast buffer. The merged read/write slices must be disjoint across the
// single-block vector loop, and all reads must precede every output write.
static bool reusableOutput(Loop *Outer, AllocaInst *From, AllocaInst *To,
                           LoopInfo &LI, ScalarEvolution &SE,
                           SmallVectorImpl<StoreInst *> &Stores) {
  BasicBlock *BB=nullptr;
  for(auto *B:Outer->blocks())for(Instruction &I:*B) {
    if(auto *W=dyn_cast<StoreInst>(&I);W && local(W->getPointerOperand())==From) {
      if(W->isVolatile() || W->isAtomic() || !W->getValueOperand()->getType()->isVectorTy())return false;
      Value *Chain=W->getPointerOperand();
      while(auto *G=dyn_cast<GetElementPtrInst>(Chain))Chain=G->getPointerOperand();
      if(Chain!=From)return false;
      if(BB && BB!=B)return false;BB=B;Stores.push_back(W);
    }
    if(auto *W=dyn_cast<StoreInst>(&I);W && local(W->getPointerOperand())==To)return false;
  }
  if(!BB || Stores.empty())return false;
  Loop *L=LI.getLoopFor(BB);
  if(!L || L==Outer || L->getNumBlocks()!=1)return false;
  int64_t Lo=INT64_MAX,Hi=INT64_MIN,Stride=0;bool Writing=false;
  const auto &DL=BB->getModule()->getDataLayout();
  for(auto *B:Outer->blocks())for(Instruction &I:*B) {
    Value *Ptr=nullptr;Type *Ty=nullptr;
    if(auto *R=dyn_cast<LoadInst>(&I)) {
      Ptr=R->getPointerOperand();Ty=R->getType();
      if(local(Ptr)!=From && local(Ptr)!=To)continue;
      if(B!=BB || Writing || R->isVolatile() || R->isAtomic())return false;
    } else if(auto *W=dyn_cast<StoreInst>(&I)) {
      if(local(W->getPointerOperand())!=From)continue;
      Ptr=W->getPointerOperand();Ty=W->getValueOperand()->getType();Writing=true;
    } else continue;
    auto *AR=dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Ptr));
    if(!AR || AR->getLoop()!=L || !AR->isAffine() || DL.getTypeStoreSize(Ty).isScalable())return false;
    auto *Step=dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    auto *Start=dyn_cast<SCEVConstant>(SE.getMinusSCEV(AR->getStart(),SE.getSCEV(local(Ptr))));
    if(!Step || !Start)return false;
    int64_t S=Step->getAPInt().getSExtValue(),O=Start->getAPInt().getSExtValue();
    if(S<=0 || O<0 || (Stride && S!=Stride))return false;
    Stride=S;Lo=std::min(Lo,O);Hi=std::max(Hi,O+(int64_t)DL.getTypeStoreSize(Ty).getFixedValue());
  }
  return Hi-Lo<=Stride;
}
static bool pipelineLoop(Loop *L, Function &F, LoopInfo &LI, ScalarEvolution &SE) {
  auto *Header=L->getHeader(),*Latch=L->getLoopLatch();
  auto *Pre=L->getLoopPredecessor();
  auto *Exit=L->getUniqueExitBlock();
  if (!Latch || !Pre || Header==Latch || !Exit) return false;
  PHINode *IV=nullptr;
  for (PHINode &P:Header->phis()) {if(IV)return false;IV=&P;}
  if (!IV || !IV->getType()->isIntegerTy(32))return false;
  auto *Next=dyn_cast<BinaryOperator>(IV->getIncomingValueForBlock(Latch));
  auto *Branch=dyn_cast<BranchInst>(Latch->getTerminator());
  if (!Next || Next->getOpcode()!=Instruction::Add || !Branch || !Branch->isConditional() ||
      Branch->getSuccessor(0)!=Header || Next->getOperand(0)!=IV)return false;
  SmallVector<CallInst *> Loads;CallInst *Output=nullptr;
  for (auto *BB:L->blocks()) for (Instruction &I:*BB) {
    if (named(&I,"llvm.xpu.gm2lm_v3")) Loads.push_back(cast<CallInst>(&I));
    else if(named(&I,"llvm.xpu.lm2gm_v3")) {if(Output)return false;Output=cast<CallInst>(&I);}
    else if(auto *C=dyn_cast<CallInst>(&I)) {
      // XTDK marks scalar exp memory(none) but omits willreturn; public
      // LLVM consequently considers it side-effecting. Keep it in place,
      // accepting only the exact arithmetic intrinsic, never an arbitrary call.
      bool Exp=named(C,"llvm.xpu.exp2f") && C->arg_size()==1 &&
          C->getType()->isFloatTy() && C->getArgOperand(0)->getType()->isFloatTy() &&
          C->doesNotAccessMemory();
      if(!fence(C) && !pure(C) && !scratchCall(C) && !Exp)return false;
    }
  }
  if (Loads.empty() || Loads.size()>3 || !Output)return false;
  BasicBlock *InputBB=Loads.front()->getParent();
  const DataLayout &DL=F.getParent()->getDataLayout();Packet P=packet(Loads[0],DL);
  if (!P.bound || P.period || P.bytes>2048 || P.bytes<64 || (P.elem!=2 && P.elem!=4))return false;
  // Two independent input transfers benefit from output overlap. A single
  // input has less transfer work and uses the cheaper two-slot schedule.
  auto *StoreWait=Output->getPrevNode(), *OutputWait=Output->getNextNode();
  unsigned Slots = Loads.size()==2 && P.bytes<=1024 &&
      StoreWait && fence(StoreWait) && OutputWait && fence(OutputWait) ? 3 : 2;
  if(!isa<Argument,ConstantInt>(P.bound))return false;
  auto *OutGP=dyn_cast<GetElementPtrInst>(Output->getArgOperand(0));
  if (!OutGP || OutGP->getNumIndices()!=1 || !isa<Argument>(OutGP->getPointerOperand()) ||
      stripIntCast(OutGP->idx_begin()->get())!=P.index ||
      DL.getTypeAllocSize(OutGP->getSourceElementType())!=P.elem ||
      Output->getArgOperand(3)!=Loads[0]->getArgOperand(3))return false;
  SmallVector<AllocaInst *> Buffers;
  SmallVector<Packet> Packets;
  SmallPtrSet<Value *,32> Seen;
  if(!clonable(Branch->getCondition(),L,IV,Seen))return false;
  for(auto *D:Loads) {
    auto Q=packet(D,DL);auto *A=dyn_cast<AllocaInst>(D->getArgOperand(0));
    if(D->getParent()!=InputBB || Q.bound!=P.bound || Q.index!=P.index ||
       Q.bytes!=P.bytes || Q.elem!=P.elem || !A || !A->isStaticAlloca() && A->getParent()!=Header ||
       !A->getAllocatedType()->isArrayTy() ||
       DL.getTypeAllocSize(A->getAllocatedType())!=P.bytes ||
       llvm::is_contained(Buffers,A))return false;
    for(unsigned J:{1u,3u})if(!clonable(D->getArgOperand(J),L,IV,Seen))return false;
    Buffers.push_back(A);Packets.push_back(Q);
  }
  bool HasBroadcast=llvm::any_of(Packets,[](Packet Q){return Q.period!=0;});
  // Keep read-only broadcast inputs in their existing single buffer. Prefetch
  // the large flat inputs only; this avoids doubling an invariant bias/output
  // scratch buffer and leaves room for sigmoid's private arithmetic scratch.
  if(HasBroadcast)Slots=2;
  if(!llvm::is_contained(Buffers,local(Output->getArgOperand(1))))return false;
  SmallPtrSet<AllocaInst *,32> Scratch;
  auto Account=[&](Value *Ptr) {
    auto *A=local(Ptr);
    if(!A)return false;
    if(llvm::is_contained(Buffers,A))return true;
    if(!A->isStaticAlloca() || !A->getArraySize()->getType()->isIntegerTy() ||
       !isa<ConstantInt>(A->getArraySize()))return false;
    Scratch.insert(A);return true;
  };
  // All function-local buffers in this loop are accounted for. No unknown
  // memory operations may consume a buffer before its DMA completion fence.
  Instruction *Wait=Loads.back()->getNextNode();while(Wait && pure(Wait))Wait=Wait->getNextNode();
  if(!Wait || !fence(Wait))return false;
  for(Instruction *I=Loads.front();I!=Wait;I=I->getNextNode())
    if(!pure(I) && !llvm::is_contained(Loads,dyn_cast<CallInst>(I)))return false;
  for(auto *BB:L->blocks())for(Instruction &I:*BB) {
    if(auto *A=dyn_cast<AllocaInst>(&I);A && !llvm::is_contained(Buffers,A))return false;
    if(auto *C=dyn_cast<CallInst>(&I))if(auto *A=scratchCall(C)) {
      if(llvm::is_contained(Buffers,A) || !Account(A))return false;
    }
    if(auto *R=dyn_cast<LoadInst>(&I);R && (!Account(R->getPointerOperand()) || R->isVolatile() || R->isAtomic()))return false;
    if(auto *W=dyn_cast<StoreInst>(&I);W && (!Account(W->getPointerOperand()) || W->isVolatile() || W->isAtomic()))return false;
  }
  uint64_t ScratchBytes=0;
  for(auto *A:Scratch) {
    if(DL.getTypeAllocSize(A->getAllocatedType()).isScalable())return false;
    uint64_t Count=cast<ConstantInt>(A->getArraySize())->getZExtValue();
    uint64_t Size=DL.getTypeAllocSize(A->getAllocatedType()).getFixedValue();
    if(Count>7168 || Size>7168 || Count*Size>7168)return false;
    ScratchBytes+=alignTo(Count*Size,64);
  }
  uint64_t DmaBytes=0;
  for(Packet Q:Packets)DmaBytes+=Q.bytes*(Q.period?1:Slots);
  if(DmaBytes+ScratchBytes>7168)return false;
  auto *OutBranch=dyn_cast<BranchInst>(Output->getParent()->getTerminator());
  if (!OutBranch || !OutBranch->isUnconditional() || OutBranch->getSuccessor(0)!=Latch || InputBB==Header) return false;
  // The block-end store wait completes both current vector stores and the
  // lookahead input. With three slots the next compute and prefetch cannot
  // overwrite the preceding asynchronous output slot.
  if (Slots==3 && (!StoreWait || !fence(StoreWait) || !OutputWait || !fence(OutputWait) ||
      OutputWait->getNextNode()!=OutBranch)) return false;
  for (BasicBlock *Pred:predecessors(Latch))
    if (Pred!=Header && Pred!=Output->getParent())return false;
  SmallVector<StoreInst *> ReuseStores;
  auto *OldOutput=local(Output->getArgOperand(1));
  bool OutputIsBias=false;
  for(unsigned J=0;J<Buffers.size();++J)
    if(Buffers[J]==OldOutput)OutputIsBias=Packets[J].period!=0;
  if(HasBroadcast && OutputIsBias &&
     !reusableOutput(L,OldOutput,Buffers[0],LI,SE,ReuseStores))return false;
  // Bias buffers that are not the output must be immutable inside the loop.
  if(HasBroadcast)for(auto *BB:L->blocks())for(Instruction &I:*BB)
    if(auto *W=dyn_cast<StoreInst>(&I))for(unsigned J=0;J<Buffers.size();++J)
      if(Packets[J].period && local(W->getPointerOperand())==Buffers[J] &&
         Buffers[J]!=OldOutput)return false;
  for(auto *W:ReuseStores) {
    IRBuilder<> WB(W);
    std::function<Value *(Value *)> Replace=[&](Value *Ptr)->Value * {
      if(Ptr==OldOutput)return Buffers[0];
      auto *G=cast<GetElementPtrInst>(Ptr);
      auto *C=G->clone();C->setOperand(0,Replace(G->getPointerOperand()));
      WB.Insert(C);return C;
    };
    W->setOperand(1,Replace(W->getPointerOperand()));
  }
  if(!ReuseStores.empty())Output->setArgOperand(1,Buffers[0]);
  IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
  auto *I64=Entry.getInt64Ty();Value *N=Entry.CreateZExt(P.bound,I64);
  Value *Total=Entry.CreateMul(N,Entry.getInt64(P.elem));
  Value *OB=Entry.CreatePtrToInt(OutGP->getPointerOperand(),I64),*OE=Entry.CreateAdd(OB,Total);
  Value *Safe=Entry.CreateAnd(Entry.CreateICmpSGE(P.bound,ConstantInt::get(P.bound->getType(),0)),Entry.CreateICmpUGE(OE,OB));
  for(unsigned J=0;J<Loads.size();++J) {
    auto *D=Loads[J];
    Value *InputTotal=Packets[J].period?Entry.getInt64(Packets[J].period*P.elem):Total;
    auto *G=cast<GetElementPtrInst>(D->getArgOperand(1));Value *IB=Entry.CreatePtrToInt(G->getPointerOperand(),I64),*IE=Entry.CreateAdd(IB,InputTotal);
    Value *NoOverlap=Entry.CreateOr(Entry.CreateICmpUGE(IB,OE),Entry.CreateICmpUGE(OB,IE));
    Value *Same=Entry.CreateICmpEQ(IB,OB);
    Safe=Entry.CreateAnd(Safe,Entry.CreateAnd(Entry.CreateICmpUGE(IE,IB),(Packets[J].period?NoOverlap:Entry.CreateOr(Same,NoOverlap))));
  }
  IRBuilder<> HB(&*Header->getFirstInsertionPt());
  auto *Slot=HB.CreatePHI(Slots==3?HB.getInt32Ty():HB.getInt1Ty(),2,"dma.slot");
  Slot->addIncoming(ConstantInt::get(Slot->getType(),0),Pre);
  auto *Ready=HB.CreatePHI(HB.getInt1Ty(),2,"dma.ready");Ready->addIncoming(HB.getFalse(),Pre);
  IRBuilder<> LB(Latch->getTerminator());
  if(Slots==3) {
    Value *Following=LB.CreateAdd(Slot,LB.getInt32(1));
    Slot->addIncoming(LB.CreateSelect(LB.CreateICmpEQ(Following,LB.getInt32(3)),LB.getInt32(0),Following),Latch);
  } else Slot->addIncoming(LB.CreateNot(Slot),Latch);
  ValueToValueMapTy Opposite;
  for(unsigned J=0;J<Buffers.size();++J) {
    auto *A=Buffers[J];
    if(Packets[J].period)continue;
    auto *Array=cast<ArrayType>(A->getAllocatedType());
    auto *NewTy=ArrayType::get(Array->getElementType(),Array->getNumElements()*Slots);
    auto *Storage=Entry.CreateAlloca(NewTy,nullptr,Slots==3?"dma.triple":"dma.pingpong");Storage->setAlignment(Align(64));
    IRBuilder<> AB(A);Value *Index,*Other;
    if(Slots==3) {
      Index=AB.CreateMul(Slot,AB.getInt32(Array->getNumElements()));
      Value *NextSlot=AB.CreateSelect(AB.CreateICmpEQ(Slot,AB.getInt32(2)),AB.getInt32(0),AB.CreateAdd(Slot,AB.getInt32(1)));
      Other=AB.CreateMul(NextSlot,AB.getInt32(Array->getNumElements()));
    } else {
      Index=AB.CreateSelect(Slot,AB.getInt32(Array->getNumElements()),AB.getInt32(0));
      Other=AB.CreateSelect(Slot,AB.getInt32(0),AB.getInt32(Array->getNumElements()));
    }
    Value *Current=AB.CreateGEP(NewTy,Storage,{AB.getInt32(0),Index});
    Value *Future=AB.CreateGEP(NewTy,Storage,{AB.getInt32(0),Other});
    A->replaceAllUsesWith(Current);Opposite[Current]=Future;A->eraseFromParent();
  }
  // Hoist only pure address arithmetic so all original input operands
  // dominate the conditional first/current-packet load block.
  SmallVector<Instruction *> Addresses;
  for (Instruction *I=Loads.front()->getNextNode();I!=Wait;I=I->getNextNode())
    if (pure(I)) Addresses.push_back(I);
  for (auto *I:Addresses) I->moveBefore(Loads.front());
  // A previous iteration's store wait has completed the prefetched packet.
  // When Ready is false, keep the original input DMA completion wait.
  IRBuilder<> PB(Wait->getNextNode());ValueToValueMapTy Map;Map[IV]=IV;
  Value *NextIV=cloneCone(Next,L,Map,PB);assert(NextIV);
  Value *HasNext=cloneCone(Branch->getCondition(),L,Map,PB);assert(HasNext);
  Map.clear();Map[IV]=NextIV;
  SmallVector<Value *> Sources,Sizes,Destinations;
  for(unsigned J=0;J<Loads.size();++J) {
    auto *D=Loads[J];
    if(Packets[J].period){
      Sources.push_back(cloneCone(D->getArgOperand(1),L,Map,PB));
      Sizes.push_back(nullptr);Destinations.push_back(nullptr);continue;
    }
    Value *Src=cloneCone(D->getArgOperand(1),L,Map,PB);
    Value *Size=cloneCone(D->getArgOperand(3),L,Map,PB);
    assert(Src && Size);Sources.push_back(Src);Sizes.push_back(Size);
    Destinations.push_back(Opposite.lookup(D->getArgOperand(0)));
  }
  Value *Go=PB.CreateAnd(Safe,HasNext);
  for(Value *Size:Sizes)if(Size)Go=PB.CreateAnd(Go,PB.CreateICmpUGT(Size,ConstantInt::get(Size->getType(),0)));
  // Protect in-place tensors across ALL 64 cores, including unusual grids.
  // Carry the actual prefetch decision so a rejected lookahead reloads its
  // packet normally on the next iteration.
  for(unsigned J=0;J<Loads.size();++J) {
    if(Packets[J].period) {
      // Reuse bias only when the exact next address is unchanged, and the
      // runtime disjointness guard proves that outputs cannot modify it.
      Go=PB.CreateAnd(Go,PB.CreateICmpEQ(Sources[J],Loads[J]->getArgOperand(1)));
      continue;
    }
    Value *Delta=PB.CreateSub(PB.CreatePtrToInt(Sources[J],I64),PB.CreatePtrToInt(Loads[J]->getArgOperand(1),I64));
    Go=PB.CreateAnd(Go,PB.CreateICmpUGE(Delta,PB.getInt64(P.bytes*64)));
    Go=PB.CreateAnd(Go,PB.CreateICmpUGE(PB.CreatePtrToInt(Sources[J],I64),PB.CreatePtrToInt(Loads[J]->getArgOperand(1),I64)));
  }
  auto *NextReady=PHINode::Create(HB.getInt1Ty(),2,"dma.next.ready",&*Latch->begin());
  for(BasicBlock *Pred:predecessors(Latch))NextReady->addIncoming(Pred==Output->getParent()?Go:HB.getFalse(),Pred);
  Ready->addIncoming(NextReady,Latch);
  auto *PrefetchTerm=SplitBlockAndInsertIfThen(Go,&*PB.GetInsertPoint(),false);
  IRBuilder<> PF(PrefetchTerm);
  for(unsigned J=0;J<Loads.size();++J)if(!Packets[J].period)PF.CreateCall(Loads[J]->getFunctionType(),Loads[J]->getCalledOperand(),{Destinations[J],Sources[J],Loads[J]->getArgOperand(2),Sizes[J]});
  IRBuilder<> IB(Loads.front());Value *Need=IB.CreateNot(Ready);
  auto *ThenTerm=SplitBlockAndInsertIfThen(Need,Loads.front(),false);
  for(auto *D:Loads)D->moveBefore(ThenTerm);
  if(Slots==3) {
    Wait->moveBefore(ThenTerm);
    // Without a proven lookahead, the next ordinary GM load must not race
    // this output (partially overlapping arguments or a repeated grid index).
    // A prefetched successor is independent and may overlap the output.
    Instruction *ExitWait=OutputWait->clone();
    ExitWait->insertBefore(&*Exit->getFirstInsertionPt());
    IRBuilder<> OB(OutputWait);
    auto *DrainTerm=SplitBlockAndInsertIfThen(OB.CreateNot(Go),OutputWait,false);
    OutputWait->moveBefore(DrainTerm);
  }
  ++stats.pipelines;return true;
}

#include "SDNNReuse.h"
#include "EmptyRow.h"
#include "LoopReadCache.h"
#include "OrderedRowReuse.h"

class XPUDmaPipelinePass : public PassInfoMixin<XPUDmaPipelinePass> {
  bool Pipeline;
public:
  explicit XPUDmaPipelinePass(bool P):Pipeline(P){}
  PreservedAnalyses run(Function &F,FunctionAnalysisManager &AM) {
    if (!F.isDeclaration() && !SDNNBounds.empty())
      return sdnnReuse(F, AM) ? PreservedAnalyses::none() : PreservedAnalyses::all();
    if(F.isDeclaration() || !F.getParent()->getTargetTriple().getArchName().starts_with("xpu3") ||
       F.getFnAttribute("target-features").getValueAsString().contains("cdnn"))return PreservedAnalyses::all();
    if(OrderedRowRequested) {
      bool Changed=orderedRowReuse(F,AM,Pipeline);
      return Changed?PreservedAnalyses::none():PreservedAnalyses::all();
    }
    bool Changed=cacheLoopReads(F,AM);if(Changed)AM.invalidate(F,PreservedAnalyses::none());
    Changed|=groupLoads(F);if(Changed)AM.invalidate(F,PreservedAnalyses::none());
    Changed|=hoistStores(F,AM);AM.invalidate(F,PreservedAnalyses::none());
    Changed|=skipEmptyRows(F,AM);AM.invalidate(F,PreservedAnalyses::none());
    if(Pipeline) {
      auto &LI=AM.getResult<LoopAnalysis>(F);
      auto &SE=AM.getResult<ScalarEvolutionAnalysis>(F);
      for(Loop *L:LI)if(pipelineLoop(L,F,LI,SE)){Changed=true;break;}
    }
    return Changed?PreservedAnalyses::none():PreservedAnalyses::all();
  }
};
}
int main(int argc,char **argv) {
  bool Pipeline=argc>1 && StringRef(argv[1])=="--pipeline";
  for(int I=1;I<argc;++I) {
    StringRef A(argv[I]);
    if(A.consume_front("--sdnn-bounds=")) {
      SmallVector<StringRef> Parts;A.split(Parts,',');
      for(StringRef P:Parts){uint64_t N;if(P.getAsInteger(10,N)||!N||N>(1ULL<<40))return 2;SDNNBounds.push_back(N);}
    }
    if(A=="--sdnn-dma-only")SDNNOnlyDMA=true;
    if(A=="--sdnn-weights-only")SDNNOnlyWeights=true;
    if(A=="--ordered-row")OrderedRowRequested=true;
  }
  auto Input=MemoryBuffer::getSTDIN();if(!Input){errs()<<"stdin error\n";return 2;}
  // Public LLVM's IRReader auto-upgrade verifies cc124 as AMDGPU before
  // analyses can run. Normalize only kernel definitions while parsing, and
  // restore the XTDK convention before emitting the module.
  std::string ParseText=(*Input)->getBuffer().str();
  std::set<std::string> KernelNames;
  size_t Pos=0;
  while ((Pos=ParseText.find("amdgpu_gfx_whole_wave",Pos))!=std::string::npos) {
    size_t Start=ParseText.rfind('\n',Pos);Start=Start==std::string::npos?0:Start+1;
    if (ParseText.compare(Start,7,"define ")!=0) {Pos+=21;continue;}
    size_t At=ParseText.find('@',Pos),End=ParseText.find('(',At);
    if(At==std::string::npos || End==std::string::npos)return 2;
    KernelNames.insert(ParseText.substr(At+1,End-At-1));
    ParseText.replace(Pos,21,"ccc");Pos+=3;
  }
  LLVMContext C;SMDiagnostic Error;
  auto M=parseIR(MemoryBufferRef(ParseText,"xpu-dma-input"),Error,C);
  if(!M){Error.print(argv[0],errs());return 2;}
  LoopAnalysisManager LAM;FunctionAnalysisManager FAM;CGSCCAnalysisManager CGAM;ModuleAnalysisManager MAM;
  PassBuilder PB;PB.registerModuleAnalyses(MAM);PB.registerCGSCCAnalyses(CGAM);PB.registerFunctionAnalyses(FAM);PB.registerLoopAnalyses(LAM);PB.crossRegisterProxies(LAM,FAM,CGAM,MAM);
  FunctionPassManager PM;PM.addPass(XPUDmaPipelinePass(Pipeline));
  for(Function &F:*M)if(!F.isDeclaration())PM.run(F,FAM);
  // XTDK cc124 is an XPU kernel, while public LLVM names it AMDGPU. Verify
  // the transformed SSA/CFG using ordinary C convention, then restore it.
  SmallVector<std::pair<Function *,CallingConv::ID>> CC;
  for(Function &F:*M){CC.emplace_back(&F,F.getCallingConv());F.setCallingConv(CallingConv::C);}
  bool Bad=verifyModule(*M,&errs());for(auto [F,ID]:CC)F->setCallingConv(ID);
  if(Bad)return 3;
  for(const auto &Name:KernelNames)if(auto *F=M->getFunction(Name))F->setCallingConv(CallingConv::AMDGPU_Gfx_WholeWave);
  errs()<<"{\"grouped_load_waits\":"<<stats.loads<<",\"hoisted_store_waits\":"<<stats.stores<<",\"pipelined_loops\":"<<stats.pipelines<<",\"skipped_empty_rows\":"<<stats.emptyRows<<",\"sdnn_cached_dma\":"<<SDNNPackets<<",\"sdnn_cached_ds\":"<<SDNNShuffles<<",\"loop_cached_reads\":"<<LoopCachedReads<<",\"ordered_resident_rows\":"<<OrderedRows<<",\"ordered_compact_rounds\":"<<OrderedRoundPacks<<"}\n";
  if(stats.loads+stats.stores+stats.pipelines+stats.emptyRows+SDNNPackets+SDNNShuffles+LoopCachedReads+OrderedRows)M->print(outs(),nullptr);else outs()<<(*Input)->getBuffer();
}
