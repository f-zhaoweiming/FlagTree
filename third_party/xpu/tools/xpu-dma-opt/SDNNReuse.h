// Reuse consecutive identical SDNN input and weight packets. Token waits/signals remain
// unconditional: other coprocessors must observe exactly the original protocol.
// Bounds are supplied by the caller's static tensor layouts. A runtime overlap
// guard disables reuse when this input aliases any other tensor argument.
static unsigned SDNNPackets = 0, SDNNShuffles = 0;
static SmallVector<uint64_t> SDNNBounds;
static bool SDNNOnlyDMA = false;
static bool SDNNOnlyWeights = false;
static ScalarEvolution *SDNNSE = nullptr;

static std::optional<uint64_t> ci(Value *V) {
  if (auto *C = dyn_cast<ConstantInt>(V)) return C->getZExtValue();
  return std::nullopt;
}
static std::optional<uint64_t> pointerMin(Value *V) {
  if (isa<ConstantPointerNull>(V)) return 0;
  if (auto *C = dyn_cast<ConstantExpr>(V)) {
    if (C->getOpcode() == Instruction::IntToPtr) return ci(C->getOperand(0));
    if (C->isCast()) return pointerMin(C->getOperand(0));
  }
  if (auto *P = dyn_cast<IntToPtrInst>(V)) {
    // The matrix writer may traverse N slices with a bounded integer offset.
    // Use the actual unsigned range, not its syntactic constant operand.
    if (!SDNNSE) return {};
    auto Range = SDNNSE->getUnsignedRange(SDNNSE->getSCEV(P->getOperand(0)));
    if (Range.getUnsignedMax().uge(1ULL << 31)) return {};
    return Range.getUnsignedMin().getZExtValue();
  }
  if (auto *G = dyn_cast<GetElementPtrInst>(V)) {
    // Generated row add offsets are unsigned shifts of a nonnegative IV.
    if (G->getNumIndices() != 1 || !G->getSourceElementType()->isIntegerTy(8)) return {};
    auto *Shift = dyn_cast<BinaryOperator>(G->idx_begin()->get());
    if (!Shift || Shift->getOpcode() != Instruction::Shl || !Shift->hasNoUnsignedWrap()) return {};
    // GEP sign-extends a 32-bit index. A merely unsigned offset could wrap
    // backwards into the cached weight buffer.
    if (!SDNNSE || SDNNSE->getUnsignedRange(SDNNSE->getSCEV(Shift)).getUnsignedMax().uge(1ULL<<31)) return {};
    return pointerMin(G->getPointerOperand());
  }
  return {};
}
static std::optional<uint64_t> upperBound(Value *V) {
  if (auto C = ci(V)) return C;
  if (auto *C = dyn_cast<CastInst>(V)) {
    if (C->getOpcode() == Instruction::Trunc || C->getOpcode() == Instruction::ZExt)
      return upperBound(C->getOperand(0));
  }
  if (auto *C = dyn_cast<CallInst>(V)) {
    if (C->getCalledFunction() &&
        (C->getCalledFunction()->getName().starts_with("llvm.umin.") ||
         C->getCalledFunction()->getName().starts_with("llvm.smin."))) {
      auto A = upperBound(C->getArgOperand(0)), B = upperBound(C->getArgOperand(1));
      if (A && B) return std::min(*A, *B);
      return A ? A : B;
    }
  }
  if (auto *B = dyn_cast<BinaryOperator>(V)) {
    if (B->getOpcode() == Instruction::Shl) {
      auto A = upperBound(B->getOperand(0)), S = ci(B->getOperand(1));
      if (A && S && *S < 24 && *A < (1ULL << 32)) return *A << *S;
    }
  }
  return {};
}
struct SDNNPacket {
  CallInst *lock = nullptr, *unlock = nullptr, *run = nullptr;
  std::map<unsigned, CallInst *> cfg;
  Argument *arg = nullptr;
  uint64_t base = 0, bytes = 0;
  SmallVector<Value *> keys;
};
static bool sdnnReuse(Function &F, FunctionAnalysisManager &AM) {
  if (SDNNBounds.empty() || !F.getParent()->getTargetTriple().getArchName().starts_with("xpu3") ||
      !F.getFnAttribute("target-features").getValueAsString().contains("cdnn")) return false;
  unsigned NumPointers = 0;
  for (Argument &A : F.args()) if (A.getType()->isPointerTy()) {
    if (A.getType()->getPointerAddressSpace() != 1 || A.getArgNo() >= SDNNBounds.size() || !SDNNBounds[A.getArgNo()]) return false;
    ++NumPointers;
  }
  if (NumPointers != SDNNBounds.size()) return false;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  SDNNSE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  SmallVector<SDNNPacket> Packets;
  SmallVector<CallInst *> Shuffles, Writes;
  SmallVector<Argument *> GlobalWrites;
  for (BasicBlock &BB : F) for (Instruction &I : BB) {
    if (isa<StoreInst, AtomicRMWInst, AtomicCmpXchgInst>(I)) return false;
    auto *Call = dyn_cast<CallInst>(&I);
    if (!Call) continue;
    if (!Call->getCalledFunction()) return false;
    StringRef N = Call->getCalledFunction()->getName();
    if (N == "llvm.xpu.uni_sram_cfg" && LI.getLoopFor(&BB)) return false;
    if (!N.starts_with("llvm.xpu.") && Call->mayWriteToMemory() && N != "llvm.assume") return false;
    if (N == "llvm.xpu.shuffle_batch.v2") Shuffles.push_back(Call);
    else if (N == "llvm.xpu.mm" || N == "llvm.xpu.dsmadd") {
      if (N == "llvm.xpu.mm" && ci(Call->getArgOperand(3)) != 8) return false;
      Writes.push_back(Call);
    }
    else if (N.starts_with("llvm.xpu.") &&
             N != "llvm.xpu.dma_run.v3" && N != "llvm.xpu.dma_cfg" &&
             N != "llvm.xpu.mm_cfg_int" && N != "llvm.xpu.ds_cfg" &&
             N != "llvm.xpu.ew_cfg" && N != "llvm.xpu.ewcoeff_cfg_scalar" &&
             N != "llvm.xpu.xsignal" && N != "llvm.xpu.xwait" &&
             N != "llvm.xpu.lock.v2" && N != "llvm.xpu.unlock" &&
             N != "llvm.xpu.core_id" && N != "llvm.xpu.load_param" &&
             N != "llvm.xpu.csr_set_sync_group" && N != "llvm.xpu.uni_sram_cfg" &&
             N != "llvm.xpu.xsignal_mode_cfg") return false;
    if (N != "llvm.xpu.dma_run.v3") continue;
    if (ci(Call->getArgOperand(0)) != 0) return false;
    SDNNPacket P; P.run = Call;
    P.unlock = dyn_cast_or_null<CallInst>(I.getNextNode());
    if (!P.unlock || !named(P.unlock, "llvm.xpu.unlock")) return false;
    Instruction *C = I.getPrevNode();
    for (; C && !named(C, "llvm.xpu.lock.v2"); C = C->getPrevNode()) {
      if (named(C, "llvm.xpu.dma_cfg")) {
        auto *Cfg = cast<CallInst>(C); auto ID = ci(Cfg->getArgOperand(3));
        if (!ID || *ID > 4 || P.cfg.count(*ID)) return false;
        P.cfg[*ID] = Cfg;
      } else if (!pure(C)) return false;
    }
    if (!C || !P.cfg.count(0) || !P.cfg.count(2) || !P.cfg.count(3) || !P.cfg.count(4)) return false;
    P.lock = cast<CallInst>(C);
    if (!LI.getLoopFor(&BB) || !named(C->getPrevNode(), "llvm.xpu.xwait")) return false;
    if (P.cfg.count(1)) {
      auto *PTI=dyn_cast<PtrToIntInst>(stripIntCast(P.cfg[3]->getArgOperand(0)));
      auto *A=PTI?dyn_cast<Argument>(getUnderlyingObject(PTI->getPointerOperand())):nullptr;
      if (!A) return false;
      GlobalWrites.push_back(A);
      continue; // global output, never cache
    }
    for (unsigned J = 0; J < 3; ++J) if (ci(P.cfg[0]->getArgOperand(J)) != 0) return false;
    auto Dst = ci(P.cfg[3]->getArgOperand(0));
    if (!Dst) return false; // multi-buffer layouts need separate proof
    P.base = *Dst;
    auto Stride = ci(P.cfg[4]->getArgOperand(0));
    auto Rows = upperBound(P.cfg[4]->getArgOperand(2));
    auto Width = upperBound(P.cfg[3]->getArgOperand(2));
    if (Stride && Rows && Width && *Rows <= (1ULL<<20) && *Stride <= (1ULL<<20))
      P.bytes = *Stride * (*Rows ? *Rows - 1 : 0) + *Width;
    // Source is one global GEP, reconstructed from its low/high words.
    Value *Low = stripIntCast(P.cfg[3]->getArgOperand(1));
    auto *PTI = dyn_cast<PtrToIntInst>(Low);
    if (!PTI) return false;
    P.arg = dyn_cast<Argument>(getUnderlyingObject(PTI->getPointerOperand()));
    if (!P.arg) return false;
    for (auto [ID, Cfg] : P.cfg) for (unsigned J = 0; J < 3; ++J)
      if (!isa<Constant>(Cfg->getArgOperand(J))) P.keys.push_back(Cfg->getArgOperand(J));
    Packets.push_back(P);
  }
  // One writer per L1 space proves the packet remains valid. Build every
  // plan before changing the CFG so analyses still describe the original IR.
  SmallVector<CallInst *> WeightShuffles, DataShuffles;
  for (auto *S : Shuffles) {
    if (ci(S->getArgOperand(3)) == 2) WeightShuffles.push_back(S);
    else if (ci(S->getArgOperand(3)) == 1) DataShuffles.push_back(S);
    else return false;
  }
  struct Plan { SDNNPacket *packet; CallInst *lock; CallInst *unlock;
                Value *destination; uint64_t slotStride; };
  SmallVector<Plan> Plans;
  auto Qualify = [&](CallInst *DS) -> bool {
  // A rotating L1 destination needs independent validity and keys per slot.
  // Recognize only the backend's two-slot byte-offset form. Unknown offsets
  // invalidate both slots at runtime; they never become a cache hit.
  Value *Destination = nullptr;
  uint64_t SlotStride = 0;
  if (!isa<Constant>(DS->getArgOperand(0)) || !pointerMin(DS->getArgOperand(0))) {
    auto *Pointer = dyn_cast<IntToPtrInst>(DS->getArgOperand(0));
    auto *Shift = Pointer ? dyn_cast<BinaryOperator>(Pointer->getOperand(0)) : nullptr;
    if (!Shift || Shift->getOpcode() != Instruction::Shl) return false;
    auto Bits = ci(Shift->getOperand(1));
    auto *Index = dyn_cast<BinaryOperator>(Shift->getOperand(0));
    if (!Bits || *Bits >= 31 || !Index) return false;
    bool TwoSlots = (Index->getOpcode() == Instruction::SRem && ci(Index->getOperand(1)) == 2) ||
                    (Index->getOpcode() == Instruction::URem && ci(Index->getOperand(1)) == 2) ||
                    (Index->getOpcode() == Instruction::And && ci(Index->getOperand(1)) == 1);
    if (!TwoSlots) return false;
    Destination = DS->getArgOperand(0);
    SlotStride = 1ULL << *Bits;
  }
  auto Source = pointerMin(DS->getArgOperand(1));
  if (!Source) return false;
  SDNNPacket *P = nullptr;
  for (auto &Candidate : Packets) if (Candidate.base == *Source) {
    if (P) return false; P = &Candidate;
  }
  if (!P || P->keys.empty()) return false;
  if (llvm::is_contained(GlobalWrites,P->arg)) return false;
  // DMA has a clamped edge row count. The static DS shape supplies its upper
  // bound only for the matched producer, not for unrelated buffers.
  uint64_t Rows = 0, Cols = 0;
  CallInst *DSLock = nullptr, *DSUnlock = dyn_cast_or_null<CallInst>(DS->getNextNode());
  for (Instruction *I = DS->getPrevNode(); I; I = I->getPrevNode()) {
    if (named(I, "llvm.xpu.lock.v2")) { DSLock = cast<CallInst>(I); break; }
    if (!named(I, "llvm.xpu.ds_cfg")) return false;
    auto *C = cast<CallInst>(I); auto ID = ci(C->getArgOperand(2));
    auto Size = ci(C->getArgOperand(0)); if (!Size || !ID) return false;
    if (*ID == 0 && (*Size != 2 || ci(C->getArgOperand(1)) != 0)) return false;
    if (*ID == 1) Cols = *Size;
    if (*ID == 13) Rows = *Size;
  }
  if (!DSLock || !DSUnlock || !named(DSUnlock, "llvm.xpu.unlock") ||
      !Rows || !Cols || Rows > 4096 || Cols > 4096) return false;
  P->bytes = Rows * Cols * 4;
  if (Destination && SlotStride < P->bytes) return false;
  // Qualify only contiguous FP32 packets with the same row pitch as DS.
  if (ci(P->cfg[3]->getArgOperand(2)) != Cols*4 ||
      ci(P->cfg[4]->getArgOperand(0)) != Cols*4 ||
      ci(P->cfg[4]->getArgOperand(1)) != Cols*4 ||
      ci(DS->getArgOperand(2)) != Cols) return false;
  for (auto &Other : Packets) if (&Other != P) {
    if (!Other.bytes) {
      // A likewise clamped matrix packet can be bounded by its DS descriptor.
      for (auto *S : Shuffles) if (pointerMin(S->getArgOperand(1)) == Other.base) {
        uint64_t R = 0, C = 0;
        for (Instruction *I = S->getPrevNode(); I && named(I,"llvm.xpu.ds_cfg"); I=I->getPrevNode()) {
          auto *D=cast<CallInst>(I);auto ID=ci(D->getArgOperand(2)),V=ci(D->getArgOperand(0));
          if (ID && V) { if (*ID==13) R=*V; if (*ID==1) C=*V; }
        }
        if (R && C && R<=4096 && C<=4096) Other.bytes=R*C*4;
      }
    }
    if (!Other.bytes || !(Other.base+Other.bytes <= P->base || Other.base >= P->base+P->bytes)) return false;
  }
  for (auto *W : Writes) {
    auto Base = pointerMin(W->getArgOperand(0));
    if (!Base || *Base < P->base + P->bytes) return false;
  }
  if (LI.getLoopFor(P->lock->getParent()) != LI.getLoopFor(DS->getParent())) return false;
  // Clone only pure scalar address/length expressions to each coprocessor.
  auto CanClone = [&](auto &&Self, Value *V, Instruction *At) -> bool {
    if (isa<Constant,Argument>(V)) return true;
    auto *I=dyn_cast<Instruction>(V); if (!I) return false;
    if (DT.dominates(I,At)) return true;
    if (!pure(I) || isa<PHINode>(I)) return false;
    for (Value *Op:I->operands()) if (!Self(Self,Op,At)) return false;
    return true;
  };
  for (auto *K:P->keys) if (!CanClone(CanClone,K,P->lock) || !CanClone(CanClone,K,DSLock)) return false;
  Plans.push_back({P, DSLock, DSUnlock, Destination, SlotStride});
  return true;
  };
  if (WeightShuffles.size() == 1) Qualify(WeightShuffles[0]);
  if (!SDNNOnlyWeights && DataShuffles.size() == 1) Qualify(DataShuffles[0]);
  if (Plans.empty()) return false;
  for (auto &Plan : Plans) {
    SDNNPacket *P = Plan.packet;
    CallInst *DSLock = Plan.lock, *DSUnlock = Plan.unlock;
  IRBuilder<> Entry(&*F.getEntryBlock().getFirstInsertionPt());
  Value *Safe = Entry.getTrue();
  Value *Start = Entry.CreatePtrToInt(P->arg,Entry.getInt64Ty());
  Value *End = Entry.CreateAdd(Start,Entry.getInt64(SDNNBounds[P->arg->getArgNo()]));
  Safe = Entry.CreateAnd(Safe,Entry.CreateICmpUGE(End,Start));
  for (Argument &A:F.args()) if (A.getType()->isPointerTy() && &A != P->arg) {
    Value *Other=Entry.CreatePtrToInt(&A,Entry.getInt64Ty());
    Value *OtherEnd=Entry.CreateAdd(Other,Entry.getInt64(SDNNBounds[A.getArgNo()]));
    Safe=Entry.CreateAnd(Safe,Entry.CreateAnd(Entry.CreateICmpUGE(OtherEnd,Other),
      Entry.CreateOr(Entry.CreateICmpULE(End,Other),Entry.CreateICmpULE(OtherEnd,Start))));
  }
  auto Guard = [&](CallInst *Lock,CallInst *Unlock, Value *Destination, uint64_t SlotStride) {
    // Keys and validity are private to this issuing coprocessor.
    unsigned Slots = Destination ? 2 : 1;
    SmallVector<AllocaInst*> Valid;
    SmallVector<SmallVector<AllocaInst*>> Saved;
    for (unsigned Slot = 0; Slot < Slots; ++Slot) {
      auto *V=Entry.CreateAlloca(Entry.getInt1Ty(),nullptr,"sdnn.reuse.valid");
      Entry.CreateStore(Entry.getFalse(),V); Valid.push_back(V);
      SmallVector<AllocaInst*> Keys;
      for (Value *K:P->keys) {auto *A=Entry.CreateAlloca(K->getType(),nullptr,"sdnn.reuse.key");Entry.CreateStore(Constant::getNullValue(K->getType()),A);Keys.push_back(A);}
      Saved.push_back(Keys);
    }
    IRBuilder<> B(Lock);DenseMap<Value*,Value*> Clones;
    std::function<Value*(Value*)> Clone=[&](Value *V)->Value* {
      if (isa<Constant,Argument>(V)) return V;
      auto *I=cast<Instruction>(V);if(DT.dominates(I,Lock))return V;
      if(Clones.count(V))return Clones[V];
      auto *New=I->clone();for(unsigned J=0;J<New->getNumOperands();++J)New->setOperand(J,Clone(I->getOperand(J)));
      B.Insert(New,"sdnn.key");return Clones[V]=New;
    };
    SmallVector<Value*> CurrentKeys;
    for (Value *K:P->keys) CurrentKeys.push_back(Clone(K));
    SmallVector<Value*> Matches;
    Value *Known=B.getFalse();
    for (unsigned Slot=0;Slot<Slots;++Slot) {
      Value *Match=Destination ? B.CreateICmpEQ(B.CreatePtrToInt(Destination,B.getInt64Ty()),B.getInt64(Slot*SlotStride)) : B.getTrue();
      Matches.push_back(Match);Known=B.CreateOr(Known,Match);
    }
    Value *Same=B.getFalse();
    for (unsigned Slot=0;Slot<Slots;++Slot) {
      Value *Hit=B.CreateAnd(Matches[Slot],B.CreateLoad(B.getInt1Ty(),Valid[Slot]));
      for(unsigned J=0;J<CurrentKeys.size();++J) {
        Value *K=CurrentKeys[J], *Old=B.CreateLoad(K->getType(),Saved[Slot][J]);
        Hit=B.CreateAnd(Hit,B.CreateICmpEQ(K,Old));
        B.CreateStore(B.CreateSelect(Matches[Slot],K,Old),Saved[Slot][J]);
      }
      Same=B.CreateOr(Same,Hit);
      Value *WasValid=B.CreateLoad(B.getInt1Ty(),Valid[Slot]);
      B.CreateStore(B.CreateAnd(Known,B.CreateOr(WasValid,Matches[Slot])),Valid[Slot]);
    }
    Same=B.CreateAnd(Safe,Same);
    auto *Before=Lock->getParent();auto *Work=Before->splitBasicBlock(Lock,"sdnn.refill");
    auto *After=Work->splitBasicBlock(Unlock->getNextNode(),"sdnn.ready");
    Before->getTerminator()->eraseFromParent();IRBuilder<>(Before).CreateCondBr(Same,After,Work);
  };
  // Perform DS first: it clones producer expressions using the original CFG.
  if (!SDNNOnlyDMA) {Guard(DSLock,DSUnlock,Plan.destination,Plan.slotStride);++SDNNShuffles;DT.recalculate(F);}
  Guard(P->lock,P->unlock,nullptr,0);++SDNNPackets;
  // The next packet can share scalar ancestors with this one. Refresh
  // dominance after splitting the two issuing-coprocessor blocks.
  DT.recalculate(F);
  }
  return true;
}
