# KL3 DMA scheduling

`xpu-dma-opt` runs a structural LLVM function pass after XPU LLVM lowering and
before the LLVM 19 adapter/code generator. It does not replace arithmetic or
vendor libraries. Both Inductor-generated and directly compiled Triton kernels
use this compiler stage. The executable hash and `xpu_dma_pipeline` option are
part of the code-generation cache identity.

Modes: `0` disables this stage, `1` applies dependency-proven wait scheduling,
and `2` also enables eligible cross-iteration prefetch (default). Scheduled IR
and pass statistics are saved as `.dma.llir` and `.dma.json` in the kernel cache.

The pass:

- Groups adjacent GM-to-LM transfers only for distinct underlying allocations,
  with no intervening memory access; one common completion wait remains.
- Moves a vector-loop store wait to its unique exit only when ScalarEvolution
  proves that consecutive iterations access disjoint LM slices. Readonly SM
  column loads do not block this LM proof. Multiple waits can be merged when
  intervening stores are independent; a store-to-load dependency, overlapping
  write, volatile/atomic access, or unknown memory effect keeps the waits.
- Skips a wholly inactive row iteration only after a CFG/effect proof: its
  scratch is fresh and does not escape, inner loops have known finite trips,
  the inactive path has no visible output, and the row has no collective.
  The existing latch drain and progress arithmetic remain. Partial masks,
  live-out accumulators, unknown calls and cross-core synchronization are not
  bypassed. The count is recorded as `skipped_empty_rows` (transformed loops,
  not the dynamic number of skipped rows).
- Recognizes flat dense packets with a common bounded index and one output.
  Single-input packets use two buffers. Two-input packets up to 1024 bytes use
  three buffers to overlap output transfer with the successor's compute. Other
  eligible packets use two buffers, subject to a 7168-byte allocation budget
  including known local arithmetic scratch.
- Supports periodic read-only bias operands when their global allocation is
  disjoint from the output and the next packet uses exactly the same bias
  address. Keeps bias in a single buffer. If memory-inplace lowering had used
  that buffer for output, proves read-before-write and disjoint vector-loop
  slices before reusing a consumed flat input buffer instead. Unknown stores,
  non-affine slices, and mutable/overlapping bias retain ordinary loads.
- Accounts for local lifetime and masked-vector scratch intrinsics, and the
  exact memory-free scalar exp intrinsic, without moving their arithmetic or
  internal fences. This does not replace the vendor sigmoid implementation.
- Checks global overlap, address wraparound, loop bounds, packet lengths and
  progress across all 64 cores before prefetch. The ready state carries the
  actual prefetch decision. Without a proven lookahead, the output wait stays
  before the next ordinary input read. Every exit drains outstanding outputs.

Unknown memory effects, volatile/atomic operations, incompatible indices, or
unaccounted local buffers prevent pipelining. SDNN and non-KL3 modules are not
transformed. These conditions deliberately do not cover every reduction or
library call.

Public LLVM 22 interprets XTDK kernel convention 124 as an AMDGPU convention.
The reader normalizes kernel definitions while parsing/verifying and restores
the XTDK convention on output. Runtime code generation still uses the installed
XPU toolchain.

The backend CMake build compiles and packages the tool. A standalone build is
also supported with `LLVM_DIR` pointing to the matching LLVM development tree.
Run `test/test_dma_pipeline.py` with `XPU_DMA_OPT` set to the built executable.
Device verification must additionally exercise tails, exact and partial aliases,
repeated grid indices, and complete model output contracts.

## Ordered two-pass row residency

`xpu_ordered_row_reuse=True` opts an ordered Add/LayerNorm template into
`--ordered-row`. The caller must provide readonly inputs and a fresh, disjoint
output. It is not a general alias-analysis shortcut. The integration requires
64 cores, explicit contraction boundaries, and FP16 rows of 128–1024 elements
in multiples of 64. The pass additionally proves the actual two-loop layout,
complete packet coverage, common row guards, exact RN conversions and disjoint
output slices, with total known LM allocations bounded by 7168 bytes.

The two inputs are loaded once into separate LM rows. The first phase keeps
the original accumulation order and saves the rounded residual in the consumed
input row. The second phase reuses it; gamma and beta are cached in a reserved
SM window for that launch only. All cores partition the parameter transfer and
participate in both unconditional barriers, including cores without valid rows.
The residual and normalized output have separate live buffers. Whole-row output
transfers occur after the vector stores complete and drain before LM refill.

Contiguous vector loads use the existing low/high FP16 conversion instructions
to preserve lane order without scalar packet assembly. This does not change
rounding mode, reassociate the reduction, or persist parameters between launches.
Existing shared allocations and unknown effects reject this transformation.
If LLVM simplification produces an unproved layout, the backend retries the
ordinary scheduling pass on the original IR and records `ordered_row_fallback`.
Successful transforms record `ordered_resident_rows`; the integration also
checks `ordered_row_requested` to distinguish fallback from dropped options.

The pass can additionally bypass an invalid row's reduction and affine loops.
This requires a common row predicate, no live values leaving those loops,
no unknown effects or collectives in the bypassed region, and a safe grid
latch. Parameter collectives remain unconditional outside the row loop, and
the latch still drains outstanding transfers. Only branches governed by that
same predicate are folded in the valid-row body. The pass records
`skipped_empty_rows` separately; failure of this extra proof keeps normal
masked-row execution without disabling proven row residency.

`test_ordered_row_reuse.py` checks the LLVM proof boundaries;
`test_ordered_row_driver.py` checks safe retry and resource-contract enforcement.
