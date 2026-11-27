// Ordered two-pass row kernels. Enabled only by an Inductor template whose
// inputs are readonly and whose output is freshly allocated and disjoint.
// Keep the arithmetic graph; change only where the rounded residual lives.
static bool rowReuseReject(unsigned Line) {
  if (std::getenv("F06_DMA_DIAGNOSTIC")) errs()<<"ordered row reject "<<Line<<"\n";
  return false;
}
static bool OrderedRowRequested = false;
static unsigned OrderedRows = 0;
static unsigned OrderedRoundPacks = 0;

// Ping-pong only A: B also holds the normalized output, so doubling both
// inputs would exceed the reviewed local-memory budget. The row-latch fence
// completes both the outgoing residual and the next A packet before reuse.
static bool prefetchOrderedA(Function &F, BasicBlock *Header, Instruction *Exit,
    BranchInst *InputGuard, CallInst *LoadA, CallInst *InputWait, Value *Source,
    Value *Valid, Value *CacheA, unsigned Bytes, uint64_t Used) {
  if(Used+Bytes>7168 || Header->phis().empty())return false;
  auto *IV=dyn_cast<PHINode>(&Header->front());
  auto *Latch=Exit->getParent();auto *Back=dyn_cast<BranchInst>(Latch->getTerminator());
  if(!IV || !IV->getType()->isIntegerTy(32) || IV->getNumIncomingValues()!=2 ||
     std::next(Header->phis().begin())!=Header->phis().end() || !Back || !Back->isConditional())return false;
  int LI=IV->getBasicBlockIndex(Latch);if(LI<0)return false;
  auto *Pre=IV->getIncomingBlock(1-LI);Value *Next=IV->getIncomingValue(LI);
  if(Back->getSuccessor(0)!=Header && Back->getSuccessor(1)!=Header)return false;
  auto *Point=InputWait->getNextNode();DominatorTree DT(F);
  std::map<Value *,bool> Dep;
  SmallPtrSet<Value *,32> Visiting;
  bool Safe=true;
  std::function<bool(Value *)> Depends=[&](Value *V)->bool{
    if(V==IV)return true;
    if(auto It=Dep.find(V);It!=Dep.end())return It->second;
    auto *I=dyn_cast<Instruction>(V);if(!I)return false;
    if(!Visiting.insert(V).second){Safe=false;return false;}
    bool D=false;for(Value *O:I->operands())D|=Depends(O);
    Visiting.erase(V);
    if(D && !pure(I))Safe=false;
    if(!D && !DT.dominates(I,Point))Safe=false;
    Dep[V]=D;return D;
  };
  for(Value *V:{Next,Back->getCondition(),Source,Valid})Depends(V);
  if(!Safe)return false;
  // An invalid row bypasses all work directly from InputGuard to the latch.
  // Every other predecessor must be dominated by the completed input DMA.
  for(BasicBlock *Pred:predecessors(Latch))
    if(Pred!=InputGuard->getParent() && !DT.dominates(InputWait,Pred->getTerminator()))return false;
  IRBuilder<> HB(&*Header->getFirstInsertionPt());
  auto *Slot=HB.CreatePHI(HB.getInt1Ty(),2,"row.slot");Slot->addIncoming(HB.getFalse(),Pre);
  auto *Ready=HB.CreatePHI(HB.getInt1Ty(),2,"row.ready");Ready->addIncoming(HB.getFalse(),Pre);
  IRBuilder<> EB(&*F.getEntryBlock().getFirstInsertionPt());
  auto *Storage=EB.CreateAlloca(ArrayType::get(EB.getInt8Ty(),2*Bytes),nullptr,"row.pingpong");Storage->setAlignment(Align(64));
  Value *Current=HB.CreateGEP(HB.getInt8Ty(),Storage,HB.CreateSelect(Slot,HB.getInt32(Bytes),HB.getInt32(0)));
  Value *Future=HB.CreateGEP(HB.getInt8Ty(),Storage,HB.CreateSelect(Slot,HB.getInt32(0),HB.getInt32(Bytes)));
  CacheA->replaceAllUsesWith(Current);cast<Instruction>(CacheA)->eraseFromParent();
  IRBuilder<> B(Point);
  std::function<Value *(Value *,ValueToValueMapTy &)> Clone=[&](Value *V,ValueToValueMapTy &Map)->Value *{
    if(auto It=Map.find(V);It!=Map.end())return It->second;
    if(!Dep[V])return V;
    auto *I=cast<Instruction>(V);auto *C=I->clone();
    for(unsigned J=0;J<I->getNumOperands();++J)C->setOperand(J,Clone(I->getOperand(J),Map));
    if(auto *G=dyn_cast<GetElementPtrInst>(C))G->setNoWrapFlags(GEPNoWrapFlags::none());
    if(auto *O=dyn_cast<BinaryOperator>(C)){
      if(isa<OverflowingBinaryOperator>(O)){O->setHasNoUnsignedWrap(false);O->setHasNoSignedWrap(false);}
      if(isa<PossiblyExactOperator>(O))O->setIsExact(false);
    }
    B.Insert(C);Map[V]=C;return C;
  };
  ValueToValueMapTy Now;Now[IV]=IV;
  Value *NextIV=Clone(Next,Now);Value *More=Clone(Back->getCondition(),Now);
  if(Back->getSuccessor(0)!=Header)More=B.CreateNot(More);
  ValueToValueMapTy Ahead;Ahead[IV]=NextIV;
  Value *NextSource=Clone(Source,Ahead),*NextValid=Clone(Valid,Ahead);
  Value *Go=B.CreateAnd(B.CreateAnd(More,NextValid),B.CreateICmpSGT(NextIV,IV));
  auto *NextReady=PHINode::Create(B.getInt1Ty(),2,"row.next.ready",&*Latch->begin());
  for(BasicBlock *Pred:predecessors(Latch))NextReady->addIncoming(Pred==InputGuard->getParent()?B.getFalse():Go,Pred);
  Ready->addIncoming(NextReady,Latch);
  IRBuilder<> LB(Latch->getTerminator());Slot->addIncoming(LB.CreateNot(Slot),Latch);
  auto *Prefetch=SplitBlockAndInsertIfThen(Go,Point,false);IRBuilder<> PF(Prefetch);
  PF.CreateCall(LoadA->getFunctionType(),LoadA->getCalledOperand(),{Future,NextSource,PF.getInt32(0),PF.getInt32(Bytes)});
  IRBuilder<> First(LoadA);auto *Need=SplitBlockAndInsertIfThen(First.CreateNot(Ready),LoadA,false);
  LoadA->moveBefore(Need);
  ++stats.pipelines;return true;
}

static bool rowAsm(CallInst *C, StringRef Prefix) {
  auto *A = C ? dyn_cast<InlineAsm>(C->getCalledOperand()) : nullptr;
  return A && !A->hasSideEffects() && StringRef(A->getAsmString()).starts_with(Prefix);
}
static LoadInst *rowPacketLoad(CallInst *C) {
  auto *Base = local(C->getArgOperand(0));
  Instruction *I = C->getNextNode();
  for (unsigned K=0; I && K<256; ++K) {
    if (auto *R=dyn_cast<LoadInst>(I))
      if (local(R->getPointerOperand())==Base &&
          R->getType()==FixedVectorType::get(Type::getHalfTy(C->getContext()),32)) return R;
    if (auto *B=dyn_cast<BranchInst>(I)) {
      if (B->isConditional()) return nullptr;
      I=&*B->getSuccessor(0)->begin();
    } else I=I->getNextNode();
  }
  return nullptr;
}
static LoadInst *rowHalfSource(Value *V, StringRef Instruction) {
  auto *C=dyn_cast<CallInst>(V);
  return rowAsm(C,Instruction) && C->arg_size()==1 ? dyn_cast<LoadInst>(C->getArgOperand(0)) : nullptr;
}
static std::pair<LoadInst *,LoadInst *> rowRoundedInputs(CallInst *C) {
  if (!rowAsm(C,"vfloat2fp16_l.rn") || C->arg_size()!=2) return {};
  if(cast<InlineAsm>(C->getCalledOperand())->getAsmString()!=
     "vfloat2fp16_l.rn $0, $1\nvfloat2fp16_h.rn $0, $2")return {};
  auto *L=dyn_cast<BinaryOperator>(C->getArgOperand(0));
  auto *H=dyn_cast<BinaryOperator>(C->getArgOperand(1));
  if (!L || !H || L->getOpcode()!=Instruction::FAdd || H->getOpcode()!=Instruction::FAdd) return {};
  auto *A=rowHalfSource(L->getOperand(0),"vfp162float_l.rn");
  auto *B=rowHalfSource(L->getOperand(1),"vfp162float_l.rn");
  if (!A || !B || rowHalfSource(H->getOperand(0),"vfp162float_h.rn")!=A ||
      rowHalfSource(H->getOperand(1),"vfp162float_h.rn")!=B) return {};
  return {A,B};
}
static bool orderedRowReuse(Function &F, FunctionAnalysisManager &AM, bool Prefetch) {
  if (!OrderedRowRequested || F.arg_size()!=8) return rowReuseReject(__LINE__);
  for (unsigned J=0;J<5;++J)
    if (!F.getArg(J)->getType()->isPointerTy() || F.getArg(J)->getType()->getPointerAddressSpace()!=1) return rowReuseReject(__LINE__);
  // Existing shared allocations or calls with unknown effects require their
  // own allocation/lifetime analysis. Do not reserve over somebody else's SM.
  for (auto &G:F.getParent()->globals()) if (G.getAddressSpace()==2) return rowReuseReject(__LINE__);
  auto &LI=AM.getResult<LoopAnalysis>(F);
  auto &SE=AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT=AM.getResult<DominatorTreeAnalysis>(F);
  struct Read { CallInst *dma; LoadInst *load; const SCEVAddRecExpr *ar; unsigned arg,bytes; int64_t off; Value *offset=nullptr; };
  struct Write { CallInst *dma; StoreInst *store; const SCEVAddRecExpr *ar; int64_t off; unsigned group; Value *offset=nullptr; };
  struct Phase { Loop *loop; unsigned count; SmallVector<Read> reads; SmallVector<CallInst *> stores,rounded; };
  SmallVector<Phase> phases;
  for (Loop *L:LI.getLoopsInPreorder()) {
    if (!L->getSubLoops().empty() || !L->getLoopPreheader() || !L->getUniqueExitBlock()) continue;
    auto *Trips=dyn_cast<SCEVConstant>(SE.getBackedgeTakenCount(L));
    if (!Trips || Trips->getAPInt().getActiveBits()>5) continue;
    unsigned Count=Trips->getAPInt().getZExtValue()+1;
    if (Count<1 || Count>16) continue;
    Phase P{L,Count,{},{},{}}; bool OK=true;
    for (BasicBlock *BB:L->blocks()) for (Instruction &I:*BB) {
      if (named(&I,"llvm.xpu.gm2lm_v3")) {
        auto *C=cast<CallInst>(&I);auto *Arg=dyn_cast<Argument>(getUnderlyingObject(C->getArgOperand(1)));
        auto *AR=dyn_cast<SCEVAddRecExpr>(SE.getSCEV(C->getArgOperand(1)));
        auto *Bytes=dyn_cast<ConstantInt>(C->getArgOperand(3));auto *Zero=dyn_cast<ConstantInt>(C->getArgOperand(2));
        auto *Load=rowPacketLoad(C);
        auto *Step=AR?dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE)):nullptr;
        if (!Arg || Arg->getArgNo()>3 || !AR || AR->getLoop()!=L || !Step || Step->getAPInt()!=128 ||
            !Bytes || (Bytes->getZExtValue()!=32 && Bytes->getZExtValue()!=64) || !Zero || !Zero->isZero() || !Load || !L->contains(Load)) {OK=false;continue;}
        P.reads.push_back({C,Load,AR,Arg->getArgNo(),unsigned(Bytes->getZExtValue()),0});
      } else if (named(&I,"llvm.xpu.lm2gm_v3")) P.stores.push_back(cast<CallInst>(&I));
      else if (auto *C=dyn_cast<CallInst>(&I)) {
        if (rowRoundedInputs(C).first) P.rounded.push_back(C);
        if (!fence(C) && !pure(C)) OK=false;
      } else if (auto *R=dyn_cast<LoadInst>(&I)) {
        if (!local(R->getPointerOperand()) || R->isVolatile() || R->isAtomic()) OK=false;
      } else if (auto *W=dyn_cast<StoreInst>(&I)) {
        if (!local(W->getPointerOperand()) || W->isVolatile() || W->isAtomic()) OK=false;
      }
    }
    if (OK && P.reads.size()==8 && P.rounded.size()==2) phases.push_back(P);
  }
  if (phases.size()!=2) return rowReuseReject(__LINE__);
  if (!phases[0].stores.empty()) std::swap(phases[0],phases[1]);
  auto &Reduce=phases[0];auto &Affine=phases[1];
  if (!Reduce.stores.empty() || (Affine.stores.size()!=2 && Affine.stores.size()!=4) ||
      Reduce.count!=Affine.count || !DT.dominates(Reduce.loop->getUniqueExitBlock(),Affine.loop->getLoopPreheader())) return rowReuseReject(__LINE__);
  unsigned Bytes=Reduce.count*128;
  const SCEV *Bases[4]={}; Value *Guard=nullptr; bool True=false;
  for (auto &P:phases) for (auto &R:P.reads) {
    if (&P==&Reduce && (R.arg>1 || R.bytes!=32)) return rowReuseReject(__LINE__);
    if (&P==&Affine && R.bytes!=64) return rowReuseReject(__LINE__);
    if (!Bases[R.arg]) Bases[R.arg]=R.ar->getStart();
    auto *D=dyn_cast<SCEVConstant>(SE.getMinusSCEV(R.ar->getStart(),Bases[R.arg]));
    if (!D || !D->getAPInt().isSignedIntN(32)) return rowReuseReject(__LINE__);
    R.off=D->getAPInt().getSExtValue();
    if (&P==&Reduce) {
      auto *Pred=R.dma->getParent()->getSinglePredecessor();
      auto *Br=Pred?dyn_cast<BranchInst>(Pred->getTerminator()):nullptr;
      if (!Br || !Br->isConditional() || !Reduce.loop->isLoopInvariant(Br->getCondition()) ||
          (Guard && (Guard!=Br->getCondition() || True!=(Br->getSuccessor(0)==R.dma->getParent())))) return rowReuseReject(__LINE__);
      Guard=Br->getCondition();True=Br->getSuccessor(0)==R.dma->getParent();
    }
  }
  if (!Guard) return rowReuseReject(__LINE__);
  auto sameGuard = [&](CallInst *C) {
    auto *Pred=C->getParent()->getSinglePredecessor();
    auto *Br=Pred?dyn_cast<BranchInst>(Pred->getTerminator()):nullptr;
    return Br && Br->isConditional() && Br->getCondition()==Guard &&
           (Br->getSuccessor(0)==C->getParent())==True;
  };
  for (auto &R:Affine.reads) if (R.arg<2 && !sameGuard(R.dma)) return rowReuseReject(__LINE__);
  for (auto *C:Affine.stores) if (!sameGuard(C)) return rowReuseReject(__LINE__);
  for (unsigned J=0;J<4;++J) {
    int64_t Min=INT64_MAX;
    for (auto &P:phases) for (auto &R:P.reads) if(R.arg==J)Min=std::min(Min,R.off);
    Bases[J]=SE.getAddExpr(Bases[J],SE.getConstant(SE.getEffectiveSCEVType(Bases[J]->getType()),Min));
    for (auto &P:phases) {
      SmallVector<std::pair<int64_t,unsigned>> spans;
      for(auto &R:P.reads)if(R.arg==J){R.off-=Min;spans.emplace_back(R.off,R.bytes);}
      if (&P==&Reduce && J>1)continue;
      llvm::sort(spans);int64_t End=0;for(auto [O,S]:spans){if(O!=End)return rowReuseReject(__LINE__);End+=S;}
      if(End!=128)return rowReuseReject(__LINE__);
    }
  }
  for(unsigned J=2;J<4;++J)if(Bases[J]!=SE.getSCEV(F.getArg(J)))return rowReuseReject(__LINE__);
  // Bind each half-vector conversion to the proven A/B packet, not to names.
  struct Rounded {CallInst *value;LoadInst *a;SmallVector<Read *> reads;};
  SmallVector<Rounded> Saved,Reused;
  for(auto &P:phases)for(auto *C:P.rounded){
    auto [A,B]=rowRoundedInputs(C);unsigned IA=99,IB=99;SmallVector<Read *> Reads;
    for(auto &R:P.reads){if(R.load==A){IA=R.arg;Reads.push_back(&R);}if(R.load==B)IB=R.arg;}
    if(IA!=0 || IB!=1)return rowReuseReject(__LINE__);
    llvm::sort(Reads,[](Read *A,Read *B){return A->off<B->off;});
    if((&P==&Reduce && (Reads.size()!=2 || Reads[1]->off-Reads[0]->off!=64)) ||
       (&P==&Affine && Reads.size()!=1))return rowReuseReject(__LINE__);
    (&P==&Reduce?Saved:Reused).push_back({C,A,Reads});
  }
  // Every gapped packet is consumed only by the two explicit FP16 halves.
  // Read aligned contiguous packets and select l/h in the conversion itself;
  // no vector shuffle or scalar LM packing is necessary.
  std::map<LoadInst *,SmallVector<Read *>> ReductionLoads;
  for(auto &R:Reduce.reads)ReductionLoads[R.load].push_back(&R);
  for(auto &[Load,Reads]:ReductionLoads){
    llvm::sort(Reads,[](Read *A,Read *B){return A->off<B->off;});
    if(Reads.size()!=2 || Reads[1]->off-Reads[0]->off!=64 ||
       (Reads[0]->off!=0 && Reads[0]->off!=32))return rowReuseReject(__LINE__);
    for(auto *U:Load->users()){
      auto *C=dyn_cast<CallInst>(U);
      if(!C || !(rowAsm(C,"vfp162float_l.rn") || rowAsm(C,"vfp162float_h.rn")))return rowReuseReject(__LINE__);
    }
  }
  llvm::sort(Saved,[](const Rounded &A,const Rounded &B){return A.reads[0]->off<B.reads[0]->off;});
  if(Saved[0].reads[0]->off!=0 || Saved[1].reads[0]->off!=32 ||
     !DT.dominates(Saved[0].value,Saved[1].value))return rowReuseReject(__LINE__);
  // Both packet layouts contain the same rounded lanes. We can form only
  // contiguous packets if every scalar-half consumer can wait until both
  // additions are available. Never move a dependent accumulation or load.
  bool CompactRound=true;
  SmallVector<CallInst *> Unpack[2];
  for(unsigned J=0;J<2;++J)for(User *U:Saved[J].value->users()){
    auto *C=dyn_cast<CallInst>(U);
    if(!C || !(rowAsm(C,"vfp162float_l.rn") || rowAsm(C,"vfp162float_h.rn")) ||
       C->arg_size()!=1){CompactRound=false;continue;}
    Unpack[J].push_back(C);
    for(User *V:C->users()){
      auto *I=dyn_cast<Instruction>(V);
      if(!I || isa<PHINode>(I) || !DT.dominates(Saved[1].value,I))CompactRound=false;
    }
  }
  SmallVector<const SCEV *> OutBases;SmallVector<Write> Writes;
  for(auto *C:Affine.stores){
    auto *AR=dyn_cast<SCEVAddRecExpr>(SE.getSCEV(C->getArgOperand(0)));
    auto *Step=AR?dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE)):nullptr;
    auto *Size=dyn_cast<ConstantInt>(C->getArgOperand(3));auto *Zero=dyn_cast<ConstantInt>(C->getArgOperand(2));
    if(getUnderlyingObject(C->getArgOperand(0))!=F.getArg(4) || !AR || AR->getLoop()!=Affine.loop ||
       !Step || Step->getAPInt()!=128 || !Size || !Size->equalsInt(64) || !Zero || !Zero->isZero())return rowReuseReject(__LINE__);
    StoreInst *S=nullptr;
    for(BasicBlock *BB:Affine.loop->blocks())for(Instruction &I:*BB)if(auto *W=dyn_cast<StoreInst>(&I))
      if(W->getPointerOperand()==C->getArgOperand(1) && W->getValueOperand()->getType()==FixedVectorType::get(Type::getHalfTy(F.getContext()),32) && DT.dominates(W,C))
        if(!S || DT.dominates(S,W))S=W;
    if(!S)return rowReuseReject(__LINE__);
    unsigned G=0;int64_t Off=0;
    for(;G<OutBases.size();++G){auto *D=dyn_cast<SCEVConstant>(SE.getMinusSCEV(AR->getStart(),OutBases[G]));
      if(D && D->getAPInt().isSignedIntN(32) && std::abs(D->getAPInt().getSExtValue())<128){Off=D->getAPInt().getSExtValue();break;}}
    if(G==OutBases.size())OutBases.push_back(AR->getStart());
    Writes.push_back({C,S,AR,Off,G});
  }
  if(OutBases.size()!=Affine.stores.size()/2)return rowReuseReject(__LINE__);
  for(unsigned G=0;G<OutBases.size();++G){
    int64_t Min=INT64_MAX;for(auto &W:Writes)if(W.group==G)Min=std::min(Min,W.off);
    OutBases[G]=SE.getAddExpr(OutBases[G],SE.getConstant(SE.getEffectiveSCEVType(OutBases[G]->getType()),Min));
    SmallVector<int64_t> O;for(auto &W:Writes)if(W.group==G){W.off-=Min;O.push_back(W.off);}
    llvm::sort(O);if(O!=SmallVector<int64_t>({0,64}))return rowReuseReject(__LINE__);
  }
  // Only the last output is normalized. A dual output's first row is the
  // rounded residual; sort by the actual GM base rather than block order.
  if(OutBases.size()==2){auto *D=dyn_cast<SCEVConstant>(SE.getMinusSCEV(OutBases[1],OutBases[0]));
    if(!D || !D->getAPInt().isSignedIntN(63) ||
       std::abs(D->getAPInt().getSExtValue())<Bytes)return rowReuseReject(__LINE__);
    if(D->getAPInt().isNegative()){std::swap(OutBases[0],OutBases[1]);for(auto &W:Writes)W.group=1-W.group;}}
  SCEVExpander Exp(SE,"row.reuse");auto *Pre=Reduce.loop->getLoopPreheader()->getTerminator();
  auto *Exit=&*Affine.loop->getUniqueExitBlock()->getFirstInsertionPt();
  for(auto *Base:Bases)if(!Exp.isSafeToExpandAt(Base,Pre))return rowReuseReject(__LINE__);
  for(auto *Base:OutBases)if(!Exp.isSafeToExpandAt(Base,Exit))return rowReuseReject(__LINE__);
  // All plans checked before mutating the IR. The maximum live row storage is
  // two FP16 rows + the original <= 1 KiB staging, bounded below 7 KiB.
  uint64_t Used=2*Bytes;for(BasicBlock &BB:F)for(Instruction &I:BB)if(auto *A=dyn_cast<AllocaInst>(&I)){
    auto S=A->getAllocationSize(F.getParent()->getDataLayout());if(!S || S->isScalable())return rowReuseReject(__LINE__);Used+=alignTo(S->getFixedValue(),64);}
  if(Used>7168)return rowReuseReject(__LINE__);
  // A masked tail row has no visible result, but running its scalar zero-fill
  // and reduction still delays the whole cluster. Prove that both phases form
  // the entire row body and that no result is carried across grid iterations.
  // Parameter-copy collectives are inserted outside this loop below.
  Loop *Rows=Reduce.loop->getParentLoop();
  BasicBlock *RowLatch=Rows?Rows->getLoopLatch():nullptr;
  bool SkipTail=Rows && Rows==Affine.loop->getParentLoop() &&
      Rows->getHeader()==Pre->getParent() &&
      RowLatch==Affine.loop->getUniqueExitBlock() && RowLatch->phis().empty() &&
      Rows->getSubLoops().size()==2;
  SmallPtrSet<BasicBlock *,32> Body;
  if(SkipTail){
    for(BasicBlock *BB:Rows->blocks())
      if(BB!=Rows->getHeader() && BB!=RowLatch)Body.insert(BB);
    for(BasicBlock *BB:Body)for(Instruction &I:*BB){
      if(auto *C=dyn_cast<CallInst>(&I)){
        if(!rowPure(C) && !fence(C) && !rowDMA(C))SkipTail=false;
      }else if(auto *L=dyn_cast<LoadInst>(&I)){
        if(!local(L->getPointerOperand()) || L->isAtomic() || L->isVolatile())SkipTail=false;
      }else if(auto *S=dyn_cast<StoreInst>(&I)){
        if(!local(S->getPointerOperand()) || S->isAtomic() || S->isVolatile())SkipTail=false;
      }
      for(User *U:I.users()){
        auto *Use=dyn_cast<Instruction>(U);
        if(!Use || !Body.contains(Use->getParent()))SkipTail=false;
      }
    }
    for(Instruction &I:*RowLatch){
      if(!rowPure(&I) && !fence(&I) && !isa<BranchInst>(I))SkipTail=false;
      for(Value *V:I.operands())if(auto *D=dyn_cast<Instruction>(V))
        if(D->getParent()!=RowLatch && !DT.dominates(D,Pre))SkipTail=false;
    }
  }
  IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
  Value *Cache[4];for(unsigned J=0;J<2;++J){auto *A=Entry.CreateAlloca(ArrayType::get(Entry.getInt8Ty(),Bytes),nullptr,"row.resident");A->setAlignment(Align(64));Cache[J]=A;}
  auto *SM=new GlobalVariable(*F.getParent(),ArrayType::get(Entry.getInt8Ty(),0),false,GlobalValue::InternalLinkage,UndefValue::get(ArrayType::get(Entry.getInt8Ty(),0)),"global_smem",nullptr,GlobalValue::NotThreadLocal,2);
  SM->setAlignment(Align(8));
  for(unsigned J=2;J<4;++J)Cache[J]=Entry.CreateGEP(Entry.getInt8Ty(),SM,Entry.getInt32(256*1024-(J-1)*Bytes));
  Value *Start[2];for(unsigned J=0;J<2;++J)Start[J]=Exp.expandCodeFor(Bases[J],Bases[J]->getType(),Pre);
  for(auto &P:phases)for(auto &R:P.reads)R.offset=Exp.expandCodeFor(SE.getMinusSCEV(R.ar,Bases[R.arg]),Entry.getInt64Ty(),P.loop->getHeader()->getTerminator());
  for(auto &W:Writes)W.offset=Exp.expandCodeFor(SE.getMinusSCEV(W.ar,OutBases[W.group]),Entry.getInt64Ty(),W.store);
  SmallVector<Value *> OutStarts;for(auto *Base:OutBases)OutStarts.push_back(Exp.expandCodeFor(Base,Base->getType(),Exit));
  auto *Fence=F.getParent()->getFunction("llvm.xpu.mfence.v2");
  auto GM2LM=Reduce.reads.front().dma->getCalledFunction();
  IRBuilder<> PB(Pre);Value *Valid=True?Guard:PB.CreateNot(Guard);
  auto *Then=SplitBlockAndInsertIfThen(Valid,Pre,false);IRBuilder<> Init(Then);
  auto *InputGuard=cast<BranchInst>(Then->getParent()->getSinglePredecessor()->getTerminator());
  CallInst *Inputs[2];
  for(unsigned J=0;J<2;++J)Inputs[J]=Init.CreateCall(GM2LM,{Cache[J],Start[J],Init.getInt32(0),Init.getInt32(Bytes)});
  auto *InputWait=Init.CreateCall(Fence,{Init.getInt32(1)});
  // Partition the two readonly parameter rows between all physical cores.
  // Both barriers are unconditional, even for cores with no row work.
  auto *Top=F.getEntryBlock().getTerminator();IRBuilder<> SB(Top);
  auto Core=F.getParent()->getOrInsertFunction("llvm.xpu.core_id",SB.getInt32Ty());
  auto Barrier=F.getParent()->getOrInsertFunction("llvm.xpu.csr_set_sync_group",SB.getVoidTy(),SB.getInt32Ty());
  auto GM2SM=F.getParent()->getOrInsertFunction("llvm.xpu.gm2sm_v3",SB.getVoidTy(),PointerType::get(F.getContext(),2),PointerType::get(F.getContext(),1),SB.getInt32Ty(),SB.getInt32Ty());
  Value *CoreId=SB.CreateCall(Core);Value *Begin=SB.CreateMul(CoreId,SB.getInt32(Bytes/64));
  SB.CreateCall(Fence,{SB.getInt32(7)});SB.CreateCall(Barrier,{SB.getInt32(65535)});
  auto *ST=SplitBlockAndInsertIfThen(SB.CreateICmpULT(CoreId,SB.getInt32(64)),Top,false);IRBuilder<> Copy(ST);
  for(unsigned J=2;J<4;++J)Copy.CreateCall(GM2SM,{Copy.CreateGEP(Copy.getInt8Ty(),Cache[J],Begin),Copy.CreateGEP(Copy.getInt8Ty(),F.getArg(J),Copy.CreateZExt(Begin,Copy.getInt64Ty())),Copy.getInt32(0),Copy.getInt32(Bytes/64)});
  IRBuilder<> After(Top);After.CreateCall(Fence,{After.getInt32(7)});After.CreateCall(Barrier,{After.getInt32(65535)});
  // First pass: preserve both independent accumulator lanes, while selecting
  // their columns with the existing l/h conversion opcodes.
  for(auto &[Load,Reads]:ReductionLoads){
    SmallVector<CallInst *> Users;for(auto *U:Load->users())Users.push_back(cast<CallInst>(U));
    for(auto *C:Users){IRBuilder<> B(C);unsigned Packet=rowAsm(C,"vfp162float_l.rn")?0:1;
      auto *R=Reads[Packet];bool High=R->off%64==32;
      Value *Offset=B.CreateSub(B.CreateTrunc(R->offset,B.getInt32Ty()),B.getInt32(High?32:0));
      Value *Ptr=B.CreateGEP(B.getInt8Ty(),Cache[R->arg],Offset);
      auto *V=B.CreateAlignedLoad(Load->getType(),Ptr,Align(64));
      auto *Old=cast<InlineAsm>(C->getCalledOperand());
      C->setCalledOperand(InlineAsm::get(C->getFunctionType(),High?"vfp162float_h.rn $0, $1":"vfp162float_l.rn $0, $1",Old->getConstraintString(),false));
      C->setArgOperand(0,V);
    }
  }
  for(auto &R:Reduce.reads)R.dma->eraseFromParent();
  // Form contiguous FP16 vectors once. When proven safe, both reduction
  // lanes and the residual cache consume these same rounded packets.
  {
    IRBuilder<> B(Saved[1].value->getNextNode());auto *C=Saved[0].value;
    Value *Packets[2];
    for(unsigned Packet=0;Packet<2;++Packet){
      auto *V=B.CreateCall(C->getFunctionType(),C->getCalledOperand(),
                          {Saved[0].value->getArgOperand(Packet),Saved[1].value->getArgOperand(Packet)});
      V->setAttributes(C->getAttributes());
      Packets[Packet]=V;
      Value *Ptr=B.CreateGEP(B.getInt8Ty(),Cache[0],B.CreateTrunc(Saved[0].reads[Packet]->offset,B.getInt32Ty()));
      B.CreateAlignedStore(V,Ptr,Align(64));
    }
    if(CompactRound){
      for(unsigned J=0;J<2;++J)for(auto *U:Unpack[J]){
        unsigned Packet=rowAsm(U,"vfp162float_l.rn")?0:1;
        auto *Old=cast<InlineAsm>(U->getCalledOperand());
        auto *Asm=InlineAsm::get(U->getFunctionType(),J?"vfp162float_h.rn $0, $1":"vfp162float_l.rn $0, $1",Old->getConstraintString(),false);
        auto *V=B.CreateCall(U->getFunctionType(),Asm,{Packets[Packet]});V->setAttributes(U->getAttributes());
        U->replaceAllUsesWith(V);U->eraseFromParent();
      }
      for(auto &S:Saved)S.value->eraseFromParent();
      ++OrderedRoundPacks;
    }
    IRBuilder<> Done(&*Reduce.loop->getUniqueExitBlock()->getFirstInsertionPt());
    Done.CreateCall(Fence,{Done.getInt32(1)});
  }
  // Second pass: consume the saved rounded residual directly. Parameters live
  // in SM. No local staging copy and no repeated A+B or FP16 boundary.
  for(auto &S:Reused)S.value->replaceAllUsesWith(S.a);
  for(auto &R:Affine.reads){IRBuilder<> B(R.load);Value *Ptr=B.CreateGEP(B.getInt8Ty(),Cache[R.arg],B.CreateTrunc(R.offset,B.getInt32Ty()));R.load->setOperand(0,Ptr);R.dma->eraseFromParent();}
  for(auto &W:Writes){IRBuilder<> B(W.store);unsigned Slot=OutBases.size()==1?1:W.group;
    W.store->setOperand(1,B.CreateGEP(B.getInt8Ty(),Cache[Slot],B.CreateTrunc(W.offset,B.getInt32Ty())));W.dma->eraseFromParent();}
  // Every iteration reads its own 128B slice before writing it; slices do not
  // overlap. Residual stays in A and normalized output in B. Drain once before
  // the whole-row DMA and once before the next grid iteration can refill LM.
  SmallVector<Instruction *> Remove;
  for(auto &P:phases)for(BasicBlock *BB:P.loop->blocks())for(Instruction &I:*BB)if(fence(&I))Remove.push_back(&I);
  for(auto *I:Remove)I->eraseFromParent();
  IRBuilder<> End(Exit);End.CreateCall(Fence,{End.getInt32(1)});
  auto *WT=SplitBlockAndInsertIfThen(Valid,Exit,false);IRBuilder<> WB(WT);
  auto LM2GM=F.getParent()->getFunction("llvm.xpu.lm2gm_v3");
  for(unsigned J=0;J<OutBases.size();++J)WB.CreateCall(LM2GM,{OutStarts[J],Cache[OutBases.size()==1?1:J],WB.getInt32(0),WB.getInt32(Bytes)});
  IRBuilder<> Drain(Exit);Drain.CreateCall(Fence,{Drain.getInt32(1)});
  if(SkipTail){
    // The false edge bypasses all row work, but retains the grid latch drain.
    // Active rows keep exactly the same arithmetic and completed input DMA.
    auto *Continue=InputGuard->getSuccessor(1);
    Continue->removePredecessor(InputGuard->getParent());
    InputGuard->setSuccessor(1,Exit->getParent());
    for(BasicBlock *BB:Body)
      if(auto *B=dyn_cast<BranchInst>(BB->getTerminator());B && B->isConditional() && B->getCondition()==Guard)
        B->setCondition(ConstantInt::getBool(F.getContext(),True));
    ++stats.emptyRows;
  }
  if(SkipTail && Prefetch)prefetchOrderedA(F,Rows->getHeader(),Exit,InputGuard,Inputs[0],InputWait,Start[0],Valid,Cache[0],Bytes,Used);
  ++OrderedRows;return true;
}
