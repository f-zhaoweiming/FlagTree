// Coalesce complete, readonly, uniformly guarded loop input spans into LM.
// Keep the original reduction, rounding, local packing and output ordering.

static unsigned LoopCachedReads = 0;
static bool cacheLoopReads(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  const DataLayout &DL = F.getParent()->getDataLayout();
  uint64_t Used = 0;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *A = dyn_cast<AllocaInst>(&I)) {
        auto Size = A->getAllocationSize(DL);
        if (!Size || Size->isScalable()) return false;
        Used += alignTo(Size->getFixedValue(), 64);
      }
  if (Used >= 7168) return false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->getSubLoops().empty() || !L->getLoopPreheader() || !L->getLoopLatch()) continue;
    auto *Trips = dyn_cast<SCEVConstant>(SE.getBackedgeTakenCount(L));
    if (!Trips || Trips->getAPInt().getActiveBits() > 10) continue;
    uint64_t Count = Trips->getAPInt().getZExtValue() + 1;
    if (Count < 2) continue;
    struct Read { CallInst *call; const SCEVAddRecExpr *ar; int64_t off; unsigned bytes; };
    struct Group { const SCEV *base; int64_t step; SmallVector<Read> reads; AllocaInst *cache; Value *start; };
    SmallVector<Group> Groups;
    Value *Guard = nullptr;
    bool GuardOnTrue = true;
    bool Valid = true;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (named(&I, "llvm.xpu.gm2lm_v3")) {
          auto *C = cast<CallInst>(&I);
          auto *Bytes = dyn_cast<ConstantInt>(C->getArgOperand(3));
          auto *Zero = dyn_cast<ConstantInt>(C->getArgOperand(2));
          auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(C->getArgOperand(1)));
          if (!local(C->getArgOperand(0)) || !Bytes || !Zero || !Zero->isZero() ||
              (Bytes->getZExtValue()!=32 && Bytes->getZExtValue()!=64) ||
              !AR || AR->getLoop()!=L || !AR->isAffine()) { Valid=false; break; }
          auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
          if (!Step || Step->getAPInt().getActiveBits()>12 || Step->getAPInt().isNegative()) { Valid=false; break; }
          int64_t S = Step->getAPInt().getSExtValue();
          if (!S || S*Count>4096) { Valid=false; break; }
          // Every transfer must have the same positive, loop-invariant guard.
          // A varying column/tail mask cannot justify a whole-span preload.
          BasicBlock *Pred = BB->getSinglePredecessor();
          auto *Br = Pred ? dyn_cast<BranchInst>(Pred->getTerminator()) : nullptr;
          if (!Br || !Br->isConditional() ||
              (Br->getSuccessor(0)!=BB && Br->getSuccessor(1)!=BB) ||
              !DT.dominates(Pred,L->getLoopLatch()) ||
              !L->isLoopInvariant(Br->getCondition()) ||
              (Guard && (Guard!=Br->getCondition() || GuardOnTrue!=(Br->getSuccessor(0)==BB)))) { Valid=false; break; }
          Guard=Br->getCondition();GuardOnTrue=Br->getSuccessor(0)==BB;
          Group *G=nullptr; int64_t Off=0;
          for (auto &Candidate:Groups) {
            auto *D=dyn_cast<SCEVConstant>(SE.getMinusSCEV(AR->getStart(),Candidate.base));
            if (D && D->getAPInt().isSignedIntN(32) && Candidate.step==S) {
              G=&Candidate;Off=D->getAPInt().getSExtValue();break;
            }
          }
          if (!G) { Groups.push_back({AR->getStart(),S,{},{},{}});G=&Groups.back(); }
          G->reads.push_back({C,AR,Off,unsigned(Bytes->getZExtValue())});
        } else if (fence(&I) || pure(&I) || isa<PHINode,BranchInst>(I)) {
          continue;
        } else if (auto *R=dyn_cast<LoadInst>(&I)) {
          if (!local(R->getPointerOperand()) || R->isAtomic() || R->isVolatile()) Valid=false;
        } else if (auto *W=dyn_cast<StoreInst>(&I)) {
          if (!local(W->getPointerOperand()) || W->isAtomic() || W->isVolatile()) Valid=false;
        } else { Valid=false; }
        if (!Valid) break;
      }
      if (!Valid) break;
    }
    if (!Valid || !Guard || Groups.empty() || Groups.size()>2) continue;
    uint64_t Budget=Used;
    for (auto &G:Groups) {
      llvm::sort(G.reads,[](const Read &A,const Read &B){return A.off<B.off;});
      int64_t Min=G.reads.front().off,End=Min;
      for (auto &R:G.reads) {
        if (R.off!=End) { Valid=false;break; } // no holes or overlaps
        End+=R.bytes;
      }
      if (End-Min!=G.step) Valid=false;
      G.base=SE.getAddExpr(G.base,SE.getConstant(SE.getEffectiveSCEVType(G.base->getType()),Min));
      for (auto &R:G.reads) R.off-=Min;
      Budget+=alignTo(G.step*Count,64);
    }
    if (!Valid || Budget>7168) continue;
    auto *Pre=L->getLoopPreheader()->getTerminator();
    if (auto *I=dyn_cast<Instruction>(Guard))
      if (!DT.dominates(I,Pre)) continue;
    SCEVExpander Exp(SE,"dma.row");
    if (llvm::any_of(Groups,[&](const Group &G){return !Exp.isSafeToExpandAt(G.base,Pre);})) continue;
    IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
    for (auto &G:Groups) {
      G.cache=Entry.CreateAlloca(ArrayType::get(Entry.getInt8Ty(),G.step*Count),nullptr,"dma.row.cache");
      G.cache->setAlignment(Align(64));
      G.start=Exp.expandCodeFor(G.base,G.base->getType(),Pre);
    }
    // Compute scalar byte offsets while the original loop analyses are valid.
    SmallVector<std::pair<Read *,Value *>> Offsets;
    for (auto &G:Groups)
      for (auto &R:G.reads) {
        auto *Offset=SE.getMinusSCEV(R.ar,G.base);
        Value *V=Exp.expandCodeFor(Offset,Offset->getType(),R.call);
        Offsets.emplace_back(&R,V);
      }
    if (!GuardOnTrue) { IRBuilder<> B(Pre);Guard=B.CreateNot(Guard,"dma.row.valid"); }
    auto *Then=SplitBlockAndInsertIfThen(Guard,Pre,false);
    IRBuilder<> Init(Then);
    for (auto &G:Groups) {
      auto *C=G.reads.front().call;
      Init.CreateCall(C->getCalledFunction(),{G.cache,G.start,Init.getInt32(0),Init.getInt32(G.step*Count)});
    }
    Init.CreateCall(F.getParent()->getFunction("llvm.xpu.mfence.v2"),{Init.getInt32(1)});
    unsigned J=0;
    for (auto &G:Groups)
      for (auto &R:G.reads) {
        IRBuilder<> B(R.call);
        Value *P=B.CreateGEP(B.getInt8Ty(),G.cache,B.CreateTruncOrBitCast(Offsets[J++].second,B.getInt32Ty()));
        auto *Ty=FixedVectorType::get(B.getInt32Ty(),R.bytes/4);
        auto *V=B.CreateAlignedLoad(Ty,P,Align(4),"dma.row.local");
        B.CreateAlignedStore(V,R.call->getArgOperand(0),Align(4));
        R.call->eraseFromParent();++LoopCachedReads;
      }
    return true; // CFG changed: rebuild analyses before considering another loop.
  }
  return false;
}
