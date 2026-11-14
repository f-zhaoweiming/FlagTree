# Persistent multiwave DAG scheduling: validation

The persistent scheduler improves the execution of the complete parallel plan.
Across the same six heavy two-/three-wave inputs, merging adjacent waves lowers
forced-parallel latency by a geometric mean of **4.964%** compared with
`tile_dataflow_coalesce=False` on the same compiler. All six cases improve.
The forced-parallel speedup against each run's serial baseline rises from
**1.033903x to 1.090525x**. The previous v2 compiler measured 1.034042x on this
population.

This does not solve automatic scheduling for every size. With the explicitly
relaxed `min_speedup=1.0`, automatic scheduling now keeps every logical wave
parallel, but the two 4096x512 cases remain **10.580% and 7.167% slower** than
serial. The default 1.1 threshold retains only the first parallel wave and has
no regression on this heavy population. Those two policies are reported
separately; the default partial schedule is not evidence that complete
multiwave parallelism always wins.

## Version and protocol

The tested library SHA-256 is
`20c4fa6fa938f15f0c4b0ccfe1c8b7525e6b6ef779889c4d373f9a72671e9055`,
built with `triton-opt` from base commit
`f4463cbb9d6a75523663e298af7c6bc7975f8a5a` plus the source hashes in the
[validation manifest](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_validation_checks.json).
The library also includes the cross-kernel fusion pass, which is disabled in
the existing DAG benchmarks. The matching backend and Python sources were
frozen, uploaded through staging, and checked against the final remote hashes.

Measurements used an RTX 4090 (sm89), CPU16 affinity, 50 warmup launches,
six interleaved permutations of serial/forced/automatic, and 30 ms CUDA-graph
replay measurements per round. Latency is the median of the six round medians
for hot reused inputs; compilation and Python launch overhead are excluded.
The two other coordinated GPU tasks released the device before this run.
GPU ownership snapshots show only this benchmark during measurement and no
compute process after completion. These are snapshots, not a continuous
hardware trace.

The original 40 DAG cases, 12 two-branch cases, and the heavy six-case inputs
and measurement protocol match v2. Each result below contains all three
variants and six timing rounds with correct outputs. No aggregate combines
these different populations.

## Changes exercised

Adjacent, pass-owned waves with equal partition widths can use one persistent
`ttg.warp_specialize`. Workers retain their preceding result in SSA, the default
group publishes any needed root value, and each group reconstructs a bounded,
cheap join after one counted full-group rendezvous per logical boundary.
The existing named-barrier allocator and warp-specialization lowering provide
the hardware protocol. The marked counted wait uses the non-`aligned` form.

A phase-aware pool reuses compatible shared memdescs, removes duplicate capture
arguments, and extends deallocation to the last use. It excludes escaping
descriptors, views and unsupported users. Shared broadcasts cannot be
overwritten by one worker while another still reads them. The seven-component
cost record includes the actual remaining transport, join recomputation,
group-local WAR barriers and persistent boundary waits. It grants no occupancy
improvement bonus.

These changes reduce dispatch and synchronization in this benchmark. They do
**not** reduce its total modeled compute-transport traffic: two-wave traffic
remains 49,152 bytes and three-wave traffic remains 81,920 bytes per tile. The
default-root publication replaces the removed joined-tensor transfer. Retaining
a worker's own SSA value avoids an additional reload when reconstructing the
join, but is not a net byte reduction relative to the original separate plan
for these binary joins. The final forced kernel's measured shared allocation
is 16,432 bytes, versus 16,408 bytes without coalescing; this comparison also
does not show a shared-footprint reduction.

## Same benchmark populations

Speedups are serial latency divided by variant latency. “Slower >1%” compares
unrounded latencies, not rounded speedups.

| Population | Cases | v2 automatic | v3 automatic | v2 forced | v3 forced | v3 auto slower >1% | v3 forced slower >1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original DAG, threshold 1.1 | 40 | 1.018496x | 1.017963x | 0.818044x | 0.838373x | 0 | 28 |
| Original two-branch suite | 12 | 1.013672x | 1.011064x | 0.847782x | 0.845209x | 1 | 12 |
| Heavy waves, threshold 1.0 | 6 | 1.094402x | 1.069475x | 1.034042x | 1.090525x | 2 | 1 |
| Heavy waves, threshold 1.1 | 6 | 1.093570x | 1.092730x | 1.034027x | 1.091182x | 0 | 1 |
| Heavy waves, threshold 1.0, no coalescing | 6 | — | 1.093853x | — | 1.033903x | 0 | 2 |

The 40-case automatic suite still selects 11 parallel and 29 serial cases.
All 29 serial cases have the exact serial PTX and diagnostic-stripped TTGIR.
Its ratio of summed medians is 1.003325x; the geometric mean remains about
1.8%, because most inputs retain serial execution and many selected gains are
small. The full-parallel heavy improvement should not be presented as an
improvement of the entire 40-case automatic population.

The 12-case suite selects two parallel and ten serial cases. The single
automatic measurement over 1% slower is a 128x128 balanced shared-input case
whose TTGIR, PTX and reported resources are identical to serial. Its round
medians change from approximately 0.919 us to 0.985 us during measurement:
serial has three rounds at each level, automatic two at the lower level and
four at the higher level. The resulting medians are 0.952145 and 0.985305 us.
The observed 3.483% difference is retained in the report; it is not evidence
of a new generated-code regression. No replacement timing run was selected.

Raw results:
[40 DAG cases](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_rtx4090_cpu16.json),
[12 baseline cases](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_rtx4090_baseline_cpu16.json),
[heavy threshold 1.0](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_rtx4090_heavy_cpu16.json),
[heavy threshold 1.1](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_rtx4090_heavy_default_cpu16.json),
[heavy no-coalescing](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_rtx4090_heavy_separate_cpu16.json).

## Complete parallel-plan ablation

Both forced variants execute every nonlinear wave in parallel. The following
comparison changes coalescing on the same compiler and retains all six cases.

| Rows x columns | Logical waves | Separate forced us | Persistent forced us | Latency reduction | Persistent speedup vs serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 x 512 | 2 | 25.720692 | 25.042583 | 2.636% | 1.156768x |
| 15 x 512 | 3 | 28.026747 | 26.707963 | 4.705% | 1.172867x |
| 128 x 512 | 2 | 24.697704 | 24.128009 | 2.307% | 1.105895x |
| 128 x 512 | 3 | 26.904638 | 25.778009 | 4.187% | 1.120727x |
| 4096 x 512 | 2 | 95.992690 | 87.612348 | 8.730% | 1.013760x |
| 4096 x 512 | 3 | 104.444836 | 97.080854 | 7.051% | 0.986663x |

The geometric mean of separate/persistent latency is 1.052234x, or 4.964%
lower latency. Normalizing each run by its own serial baseline gives a
1.054765x speedup ratio. The large three-wave case improves substantially but
still takes 1.352% longer than serial; this residual is not hidden by fallback.

The generated TTGIR was checked independently of diagnostic JSON. Without
coalescing, the heavy forced kernels have two or three physical **compute**
regions. With coalescing they have one compute region containing two or three
logical waves. Force also schedules an index-only region in both variants:
all-region counts are therefore 3→2 and 4→2, while logical counts remain 3
and 4 including indexing. Launch size remains eight warps. Each group has
exactly one or two named waits inside the merged compute region, respectively.
The [174-file IR check](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_ir_checks.json)
and [IR/PTX archive](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_ir.tar.gz)
preserve this distinction.

## Remaining automatic-model limitation

At threshold 1.0, all six automatic heavy cases use one persistent compute
region with all two or three logical waves. The shorter 15- and 128-row cases
improve, but the large cases are:

| Shape | Waves | Serial us | Automatic all-wave us | Forced all-wave us | Automatic speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 x 512 | 2 | 88.817867 | 98.214853 | 87.612348 | 0.904322x |
| 4096 x 512 | 3 | 95.786069 | 102.651146 | 97.080854 | 0.933122x |

Each later-wave decision has modeled marginal saving 178.667 and required
saving 170.667, only eight weighted units of headroom. The model charges zero
occupancy penalty. At the default threshold 1.1 the larger profitability
requirement rejects these tails; all six default automatic kernels contain
only one parallel logical wave, with geometric speedup 1.092730x.

Static inspection does not establish the hardware cause of the large
automatic/forced gap. In the examined two-wave 4096x512 case, both have eight
warps, 44 registers/thread, 16,432 shared bytes, five resource-limited active
CTAs/SM and no spills. Their main tensor layouts, floating-point instruction
sequences and main synchronization counts agree. The forced index region
changes the default branch's global-load prologue from four vector loads to
16 scalar loads and changes code placement and entry timing. Issue contention,
instruction-cache behavior or SASS scheduling need profiling before attribution.
An occupancy decrease alone is not supported by the same-compiler ablation.

The default branch also has a different PTX instruction mix and about 25%
more arithmetic instructions in the long first wave than the worker, while
the heuristic assigns equal branch costs. Instruction count is not elapsed
time, but this is a concrete calibration target. The current model has neither
an exact launch-grid throughput model nor a calibrated account of resource
contention between persistent branches. The full-parallel mechanism improved;
its profitability prediction remains incomplete.

## Correctness and synchronization

- 12 dataflow native tests, one cross-kernel native test and two existing core
  warp-specialization tests pass. Full TLE: 92/99 pass. The seven failing names
  match the previously recorded alignment/DSL-region/WGMMA/remote-operation
  failures; there is no pristine-base comparison, and the full suite is not
  clean.
- All 85 existing GPU tests pass. The depth-six automatic multiwave assertions
  remain unchanged and pass; correctness and guard checks were not relaxed.
  The same 85 pass under each of racecheck, memcheck and synccheck; all summaries
  report zero errors, and racecheck reports zero warnings.
- The unified cross-kernel build passes 33 ordinary CPU/GPU tests. All 20 GPU
  cases pass each sanitizer with zero errors and zero racecheck warnings.
  Its separate [12-case benchmark](../../python/benchmarks/tle/results/kernel_call_fusion_rtx4090_cpu16.json)
  is not included in any DAG aggregate above.

An intermediate attempt to remove WAR barriers from identical nonreplicated
SSA load/store layouts was rejected by validation. The two-wave, 257-column
test reported 23,808 warnings, including a read by thread 224 and a write by
thread 225 to the same shared address. The load lowered to collective
`ldmatrix`; SSA value ownership was insufficient to prove physical access
ownership. The same test with the preceding compiler retaining WAR barriers
had zero warnings. The final compiler restores these barriers, and its cost
model counts them. The native regression includes the observed 4x512 layout.
The [A/B evidence](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_logs/war_regression/manifest.json)
retains the failed intermediate implementation and its hashes.

The first cross-kernel run also encountered two stale test assertions looking
for an `accepted` metadata key. They were corrected to the existing
`decision` field, and all final ordinary and sanitizer checks used the corrected
test SHA `92b2b8c407495ff9a7712556572e6186da79ecdf6017686fca7f39af27428eba`.
This changed the test only, not the compiler or runtime.

The [combined runner](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_logs/run-flagtree-combined-validation.sh),
[logs](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_logs),
[cross-kernel logs](../../python/benchmarks/tle/results/kernel_call_fusion_logs),
[derived analysis](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_analysis.json)
and [validation manifest](../../python/benchmarks/tle/results/tile_dataflow_dag_v3_validation_checks.json)
record commands, source/library hashes, resources, samples and the unsuccessful
intermediate attempts. The historical
[v2 report](TILE_DATAFLOW_DAG_OPTIMIZATION.md) remains unchanged.
