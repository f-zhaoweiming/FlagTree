// Skip an unobservable masked iteration, not an arbitrary masked load.
// This is a CFG/effect proof; it does not depend on shape or kernel names.
static bool rowPure(Instruction *I) {
  // Public LLVM does not know the effects of these private XPU intrinsics.
  // These exact ABIs read only registers/immutable launch parameters. In
  // particular vvor's convergent marker is not a cross-core collective.
  if (auto *C = dyn_cast<CallInst>(I); C && C->getCalledFunction() &&
      C->getCalledFunction()->isDeclaration()) {
    if (named(C, "llvm.xpu.sqrtf") && C->arg_size() == 1 &&
        C->getType()->isFloatTy() &&
        C->getArgOperand(0)->getType()->isFloatTy()) return true;
    if (named(C, "llvm.xpu.load_param") && C->arg_size() == 1 &&
        C->getType()->isIntegerTy(32) &&
        C->getArgOperand(0)->getType()->isIntegerTy(32)) return true;
    if (named(C, "llvm.xpu.vvor_f_mh_rn") && C->arg_size() == 4) {
      auto *V = FixedVectorType::get(Type::getFloatTy(I->getContext()), 16);
      auto *M = FixedVectorType::get(Type::getInt1Ty(I->getContext()), 32);
      if (C->getType() == V && C->getArgOperand(0)->getType() == M &&
          C->getArgOperand(1)->getType() == V &&
          C->getArgOperand(2)->getType() == V &&
          C->getArgOperand(3)->getType() == V) return true;
    }
  }
  if (auto *C = dyn_cast<CallBase>(I); C && C->isConvergent())
    return false;
  return pure(I);
}

static bool rowDMA(Instruction *I) {
  return named(I, "llvm.xpu.gm2lm_v3") ||
         named(I, "llvm.xpu.lm2gm_v3");
}

// Scratch must be fresh in this iteration and must not escape through a
// pointer-valued store, an unknown call, a cast to GM, or a later iteration.
static bool rowScratch(AllocaInst *A, Loop *L) {
  if (!A || A->getParent() != L->getHeader()) return false;
  SmallVector<Value *> Work{A};
  SmallPtrSet<Value *, 32> Seen;
  while (!Work.empty()) {
    Value *P = Work.pop_back_val();
    if (!Seen.insert(P).second) continue;
    for (User *U : P->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || !L->contains(I)) return false;
      if (isa<GetElementPtrInst, BitCastInst>(I)) {
        Work.push_back(I);
      } else if (auto *R = dyn_cast<LoadInst>(I)) {
        if (R->getPointerOperand() != P || R->isAtomic() || R->isVolatile())
          return false;
      } else if (auto *W = dyn_cast<StoreInst>(I)) {
        if (W->getPointerOperand() != P || W->getValueOperand() == P ||
            W->isAtomic() || W->isVolatile()) return false;
      } else if (auto *C = dyn_cast<CallInst>(I); C && rowDMA(C)) {
        unsigned LocalArg = named(C, "llvm.xpu.gm2lm_v3") ? 0 : 1;
        if (C->getArgOperand(LocalArg) != P) return false;
        for (unsigned J = 0; J < C->arg_size(); ++J)
          if (J != LocalArg && C->getArgOperand(J) == P) return false;
      } else {
        return false;
      }
    }
  }
  return true;
}

static bool skipEmptyRows(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  // Recompute analyses after each CFG edit. Typical row kernels have one
  // outer grid loop; other loops remain unchanged for this invocation.
  for (Loop *L : LI) {
    BasicBlock *Header = L->getHeader(), *Latch = L->getLoopLatch();
    auto *Guard = dyn_cast<BranchInst>(Header->getTerminator());
    if (!Latch || Latch == Header || !Latch->phis().empty() || !Guard ||
        !Guard->isConditional() || !isa<ICmpInst>(Guard->getCondition()))
      continue;
    auto *Backedge = dyn_cast<BranchInst>(Latch->getTerminator());
    if (!Backedge || !Backedge->isConditional() ||
        (Backedge->getSuccessor(0) != Header &&
         Backedge->getSuccessor(1) != Header)) continue;

    bool OK = true, HasDrain = false, HasOutput = false;
    // Discarding a potentially nonterminating loop would change behavior,
    // even when its memory effects are private to the inactive iteration.
    SmallVector<Loop *> Nested(L->getSubLoops().begin(), L->getSubLoops().end());
    while (!Nested.empty()) {
      Loop *Inner = Nested.pop_back_val();
      if (!SE.getSmallConstantTripCount(Inner)) OK = false;
      Nested.append(Inner->getSubLoops().begin(), Inner->getSubLoops().end());
    }
    if (!OK) continue;
    for (Instruction &I : *Latch) {
      if (fence(&I)) HasDrain = true;
      else if (!rowPure(&I) && !isa<BranchInst>(I)) { OK = false; break; }
      for (Value *V : I.operands())
        if (auto *Def = dyn_cast<Instruction>(V))
          if (Def->getParent() != Latch && !DT.dominates(Def, Guard)) OK = false;
    }
    if (!OK || !HasDrain) continue;

    SmallPtrSet<AllocaInst *, 8> Scratch;
    // Reject collective operations even when unreachable on the inactive
    // branch: active cores may otherwise wait forever for a skipped core.
    for (BasicBlock *BB : L->blocks()) for (Instruction &I : *BB) {
      if (auto *C = dyn_cast<CallInst>(&I)) {
        if (!rowPure(C) && !fence(C) && !rowDMA(C)) OK = false;
        HasOutput |= named(C, "llvm.xpu.lm2gm_v3");
      }
      if (auto *A = dyn_cast<AllocaInst>(&I)) {
        if (!rowScratch(A, L)) OK = false;
        Scratch.insert(A);
      }
      // An accumulator or result carried to the next iteration/loop exit
      // must not silently lose the inactive row's contribution.
      if (BB != Header && BB != Latch)
        for (User *U : I.users()) {
          auto *Use = dyn_cast<Instruction>(U);
          if (!Use || !L->contains(Use) || Use->getParent() == Header ||
              Use->getParent() == Latch) OK = false;
        }
    }
    if (!OK || !HasOutput || Scratch.empty()) continue;

    Value *Predicate = Guard->getCondition();
    BasicBlock *Inactive = Guard->getSuccessor(1);
    if (Inactive == Latch) continue;
    SmallVector<BasicBlock *> Work{Inactive};
    SmallPtrSet<BasicBlock *, 32> Seen;
    bool ReachesLatch = false;
    while (OK && !Work.empty()) {
      BasicBlock *BB = Work.pop_back_val();
      if (BB == Latch) { ReachesLatch = true; continue; }
      if (BB == Header || !L->contains(BB)) { OK = false; break; }
      if (!Seen.insert(BB).second) continue;
      for (Instruction &I : *BB) {
        if (auto *R = dyn_cast<LoadInst>(&I)) {
          if (R->isAtomic() || R->isVolatile() ||
              (R->getPointerAddressSpace() != 2 &&
               !Scratch.contains(local(R->getPointerOperand())))) OK = false;
        } else if (auto *W = dyn_cast<StoreInst>(&I)) {
          if (W->isAtomic() || W->isVolatile() ||
              !Scratch.contains(local(W->getPointerOperand()))) OK = false;
        } else if (!rowPure(&I) && !fence(&I) &&
                   !isa<PHINode, BranchInst, AllocaInst>(I)) {
          OK = false;
        }
      }
      auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
      if (!Br) { OK = false; break; }
      if (Br->isConditional() && Br->getCondition() == Predicate)
        Work.push_back(Br->getSuccessor(1));
      else
        for (BasicBlock *Next : successors(BB)) Work.push_back(Next);
    }
    if (!OK || !ReachesLatch) continue;
    // The latch's original LM drain and progress arithmetic stay in place.
    Inactive->removePredecessor(Header);
    Guard->setSuccessor(1, Latch);
    removeUnreachableBlocks(F);
    ++stats.emptyRows;
    return true;
  }
  return false;
}
