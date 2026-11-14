# Tile dataflow DAG scheduling: validation record

Report date: 2026-10-18. This report covers the whole-block DAG scheduler with
multiple simultaneous branches and successive fork/join waves. A unified GPU
correctness run passed all 71 tests in 11.69 seconds with the process bound to
CPU16. Unified racecheck, memcheck and synccheck also passed all 71 tests on the
same production compiler library. The final CPU16-bound 40-case DAG benchmark
completed all three variants and measured a 1.018749x geometric-mean speedup in
automatic mode versus 0.829732x in forced mode. The final CPU16-bound 12-case
two-branch benchmark measured 1.014564x automatically. An earlier forced host-side timing
failure is retained as historical evidence, separate from the final complete
DAG run. CPU16-bound follow-up measurements also cover
heavy successive waves: automatic geometric means are 1.053989x with a reduced
1.0 profitability threshold and 1.055178x with the default 1.1 threshold, using
different selected schedules. A host heap failure has also been reproduced in
an independent LLVM-container control outside Triton and this pass. Its precise
cause remains unresolved, so host reliability beyond the recorded validation
conditions is not established. The
[2026-10-17 report](TILE_DATAFLOW_VALIDATION.md) records the previous
two-branch implementation and remains historical evidence.

## Implementation under validation

The pass builds one dependency graph per kernel entry block. Nodes are IR
operations, including scalar support for tile operations. Edges preserve SSA
uses, conservative RAW/WAR/WAW memory hazards, and scheduling boundaries.
Distinct external pointer arguments may alias; distinct local allocations or
proven different address spaces can establish independence.

Candidate outputs are retained at independent-input joins, stores, returns and
other boundaries. Internal single-chain endpoints are pruned to reduce search
cost; their work remains in the backward branch slices. Candidate branches may
feed different joins or disconnected outputs. Common work stays outside the
fork, and intermediate values that escape a branch prevent that work from
moving. Every moved operation must preserve its external predecessor and
successor constraints around the earliest outside consumer.

Backward branch collection stops at the nearest preceding scheduling boundary.
Earlier values remain captures rather than being moved across an existing wave.
Workers may still rematerialize eligible pure constants and index expressions.
This distinction preserves true memory ordering while avoiding false
cross-branch paths caused by pulling distinct early constants into later waves.

The search retains every legal pair of candidate endpoints, greedily grows
independent branch sets and expands internal join frontiers, with at most 256 successful frontier expansions per graph snapshot.
It compares legal default/worker assignments across the block and rebuilds the
graph after each accepted wave. This is a heuristic search, not exhaustive
partitioning or a proof of the fastest possible schedule.

Each wave becomes `ttg.warp_specialize` with one default group and one or more
workers. Every group keeps the original warp count and tensor layouts. Tensor
captures use shared storage deduplicated across workers; predicates use bytes.
Worker results use separate buffers, are loaded after the join, and are then
deallocated. Existing allocation, barrier analysis and lowering provide the
storage and synchronization.

The cost model compares branch work against synchronization, shared transfers,
recomputation and whole-function occupancy estimates. A common capture read by
`k` workers incurs one write plus `k` reads. Existing waves retain cost
components that can be evaluated with current model coefficients; unknown
forced-wave work is not treated as free. Allocation and register estimates are
recomputed for each candidate. Worker resources are reused between waves, and
all surrounding work is penalized when a wider wave reduces resident CTAs.

See [the design and options](TILE_DATAFLOW.md) for the model equations and API.

## Scope and limits

- Public, single-block NVIDIA kernels with one CTA per program and 4, 8 or 16
  original warps. There is no cross-kernel launch fusion or runtime task queue.
- Up to four simultaneous branches by default, including the default branch.
  `tile_dataflow_max_partitions` controls this limit; hardware also caps it at
  `floor(32 / original_num_warps)`. Branches receive equal warp counts.
- Successive waves reuse the module-wide maximum worker allocation. Their warp
  counts are not summed. Dispatch states, including allocator padding, must fit
  the lowering's 255-state limit per function.
- Loops, conditionals, existing warp-specialization regions, asynchronous
  operations and unknown effects remain scheduling boundaries. The pass does
  not schedule loop-carried dependencies. Arbitrary existing WS regions without
  modeled cost components keep automatic candidates conservative.
- Named physical barriers, inline assembly, low-level thread operations and
  explicit `ttg.maxnreg` contracts are excluded. Scalar captures must be uniform;
  staged tensors must have static shapes and supported numeric element types.
- Automatic resource profiles cover SM80/86/89/90. Register liveness and
  residency are estimates, not ptxas register counts or achieved occupancy.
  Grid size, cache state and runtime contention are not precisely modeled.
- `wave` in diagnostics is selection order, which can differ from execution
  order. A worker operand index of `-1` means the worker output is consumed by
  another operation rather than the insertion operation itself.

## Native compiler checks

The following observations come from local native-build logs. They do not
substitute for GPU execution of the final compiler artifact.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Scheduler/compiler build | Compiled and linked `triton-opt`; final boundary-scoped extension linked | `multiway-dag-endpoint-build.log`, `multiway-dag-boundary-build.log` |
| Five tile-dataflow MLIR regressions | 5 passed, including the earlier-constant/later-wave regression | `multiway-dag-boundary-lit.log` |
| Existing core warp-specialization regressions | 2 passed | `boundary-core-ws.log` |
| Full TLE suite | 84 passed, 7 failed | `boundary-full-tle.log` |

The local logs are under `/tmp/flagtree-tile-build/`. The full-suite run failed
in `test-alignment.mlir`, `test_tle_dslregion_inline_extract_ptr.mlir`,
`test_tle_lower_wgmma.mlir`, `test_tle_reduce_data_duplication_remote_if.mlir`,
`test_tle_remote_atomic_splat_ptr_lowering.mlir`,
`test_tle_remote_call_arg_store_lowering.mlir`, and
`test_tle_wgmma_pipeline_accumulator_chain.mlir`. These are the same seven
failing test names recorded in the historical 83/90 run; the additional DAG
regression passes. Their RUN lines do not invoke this scheduling pass, but no
clean pristine-base comparison was performed. The full suite is not clean.

The focused regressions exercise three/four-way join trees, disconnected
outputs, common input and fanout, successive dependent waves, alias hazards,
automatic and forced decisions, idempotence, predicates, shared memory and
occupancy. The byte-staging regression also exercises native LLVM lowering.

## GPU correctness and environment

The latest unified run of `test_tile_dataflow_schedule.py` and
`test_tile_dataflow_dag.py` passed **71/71 tests in 11.69 seconds**, with the
process bound to CPU16. Its remote log is
`results/dag-gpu-71-cpu16.log`. Earlier validation passed 70/70 tests in 17.27
seconds and separately passed
`test_tile_dataflow_four_groups_at_cta_warp_limit`: four independent branches,
eight original warps per group and 32 total warps. CPU16-bound unified sanitizer
runs subsequently passed all 71 tests under racecheck in 37.59 seconds (zero
hazards, errors or warnings), memcheck in 6.11 seconds (zero errors) and
synccheck in 3.05 seconds (zero errors).
Their remote records are `results/dag-racecheck-71-cpu16.log`,
`results/dag-racecheck-71-cpu16-tests.log`,
`results/dag-memcheck-71-cpu16.log`,
`results/dag-memcheck-71-cpu16-tests.log`,
`results/dag-synccheck-71-cpu16.log` and
`results/dag-synccheck-71-cpu16-tests.log`. Earlier racecheck, memcheck and
synccheck each passed the 70-test suite and the separate boundary case; those
records remain historical evidence. All four unified runs use the same
production compiler library, with SHA256:

```text
7cc709846314910c37bbc79943379e1bd5386468b8420d213438600d76633bd7
```

The environment matches the historical run: RTX 4090 (SM89), driver 615.71.09,
ptxas 13.4.92, Python 3.12.15 and PyTorch 2.11.0+cu130. The native compiler uses
LLVM `f6ded0be897e2878612dd903f7e8bb85448269e5`, GCC 13, a Release build and
Python 3.12.14 headers. The local provenance record
`/tmp/flagtree-dag-provenance.json` identifies base commit
`f4463cbb9d6a75523663e298af7c6bc7975f8a5a` with working-tree changes and source
hashes. The [final validation manifest](../../python/benchmarks/tle/results/tile_dataflow_dag_validation_checks.json)
records final source and artifact hashes, test results, benchmark aggregates and
reproduction commands. The production library and compiler sources match the
initial provenance. The final DAG test additionally covers the 32-warp boundary;
the benchmark additionally records CPU affinity and checkpoints completed cases.
Benchmark timestamps are preserved in the raw artifacts linked below. GPU test
logs record durations but do not contain individual run timestamps.

| Item | Status |
| --- | --- |
| Final source and artifact manifest | Recorded in `tile_dataflow_dag_validation_checks.json`; includes benchmark timestamps, source hashes, log hashes and reproduction commands |
| Compiler-library SHA256 and runtime environment | Recorded above |
| Existing and multiway/multiwave GPU correctness suites | Unified 71/71 passed in 11.69 seconds, bound to CPU16 |
| Unified racecheck | 71/71 passed in 37.59 seconds, CPU16; zero hazards, errors or warnings |
| Unified memcheck | 71/71 passed in 6.11 seconds, CPU16; zero errors |
| Unified synccheck | 71/71 passed in 3.05 seconds, CPU16; zero errors |
| Final three-variant DAG benchmark | CPU16-bound 40/40 cases passed numerical checks and all six timing rounds for all three variants; no failed variants |
| Final two-branch baseline benchmark | CPU16-bound 12/12 cases passed numerical checks and all six timing rounds for all three variants |
| Automatically retained serial TTGIR/PTX equivalence | Exact for all 29 final DAG and 10 final two-branch serial cases, plus all three default-threshold heavy three-wave cases and the fanout retry |
| Heavy multiwave performance | Two six-case CPU16-bound runs completed; reduced-threshold large-grid regressions retained below |
| Original failed fanout variant | Separate retry and full CPU16-bound rerun completed all three variants; original failed artifact remains historical evidence |
| Repeated-compilation host memory problem | Unresolved; CPU-affinity probes do not establish a root cause or a fix |

The GPU tests check numerical references and output guards, inspect generated
wave widths, verify that sequential waves reuse the maximum warp count, compare
worker register-pressure estimates, and exercise aliased global-memory writes.
The added boundary test checks numerical results against the baseline and Torch,
input immutability, output guards, four real nonlinear branches, eight warps in
each worker partition and 32 physical launch warps. The GPU sanitizer results do
not resolve the separate compiler/host memory problem described below.

### Initial failures and fixes

The first combined run reported **56 passed and 14 failed**. It exposed both
structural test assumptions and a scheduler candidate-discovery defect:

- Raw substring counting confused diagnostic text with actual WS operations;
  forced scheduling can also create legitimate small indexing/output waves.
  The tests now inspect actual regions and verify the required nonlinear waves
  separately, while still checking every wave's branch structure and launch
  resources.
- Later branch collection pulled distinct constants from before an already
  emitted WS. Boundary edges then created a path from one branch's constant,
  through the old WS, to another branch, incorrectly suppressing a later legal
  fork. Branch collection now stops at the nearest boundary and keeps earlier
  values as captures, with pure recomputation available to workers.

The precise MLIR regression `zz_later_wave_uses_earlier_constants` requires the
later fork to remain schedulable. The final 71-test run includes the repaired
multiwave cases; the first failure is retained as investigation history rather
than counted as successful validation. Its local log is
`/tmp/flagtree-dag-gpu-first.log`.

## Measured performance of this revision

The raw artifacts retain every shape, input-sharing mode, chain depth, selected
wave width, timing round, model decision, compiled resource measurement and
per-variant failure:

| Artifact | Cases | Recorded UTC timestamp | SHA256 |
| --- | ---: | --- | --- |
| [Final DAG results, CPU16](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_cpu16.json) | 40 | 2026-10-17 22:23:13.250128 | `9534ddc2efa66d5b6cbb5c543bcb8d790fce9a001251cc437b05c1dd6904a837` |
| [Final two-branch results, CPU16](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_baseline_cpu16.json) | 12 | 2026-10-17 22:25:58.154640 | `8cde34d5546b57b6a6b8fff789a0a5a018018a5102f2419fb2a383564f376cee` |
| [Original DAG results, historical](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090.json) | 40 | 2026-10-17 21:33:25.448548 | `305f9ff86bb9d134577a4a78da8e097e524e90cca43a30e25b7feb0b6ddd0f07` |
| [Original two-branch results, historical](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_baseline.json) | 12 | 2026-10-17 21:37:10.555102 | `265425fe97673bbddd8674bc3e4c74353161c292fa9861230fbdbcf02ef00f06` |
| [Heavy waves, threshold 1.0, CPU16](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_heavy_cpu16.json) | 6 | 2026-10-17 22:05:54.620927 | `a91dd58948ecbf9d38f7d8dac87bfdbe38ec46443e0bfab91dc6cdc50a85e268` |
| [Heavy waves, default threshold 1.1, CPU16](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_heavy_default_cpu16.json) | 6 | 2026-10-17 22:08:06.813803 | `45bc391af38046f5932346af208599f3a50d26848ee7e60c29eff24ee8918dda` |
| [Fanout retry, CPU16](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_fanout_retry_cpu16.json) | 1 | 2026-10-17 22:09:57.362179 | `ec686fff8a1ac06c158b68658bd9d7a07f000b269a3384c5bf779495f330df54` |

All recorded runs use FP32, four original warps, 50 warmup launches per variant and six
timing rounds covering all permutations of baseline/forced/automatic order.
Each round uses 30 ms of CUDA-graph replay with reused input buffers. Per-case
latency is the median of the round medians. These are kernel replay timings;
compilation, Python launch overhead and cold-cache end-to-end behavior are not
included. Speedup is baseline latency divided by variant latency. Geometric
means use the complete CPU16-bound runs for the final table: all 40 DAG cases
and 12 two-branch cases passed numerical checks and all six rounds in all three
variants. No partial rounds or earlier-run timings are substituted into this
table. The final DAG JSON records `cpu_affinity: [16]`; the older two-branch
benchmark script does not emit that field, so its CPU16 affinity comes from the
recorded `taskset -c 16` run invocation rather than the JSON environment object.

The DAG suite uses shapes 5x96, 96x384, 1025x768 and 6144x1536. Three/four-way
trees, disconnected outputs and fanout each include independent and shared
inputs, with three nonlinear iterations per branch. The two/three-wave cases
use one nonlinear iteration per wave. Automatic mode uses the default 1.1
minimum predicted speedup and at most four groups. The final two-branch row
also uses this compiler revision and CPU16 affinity.

| Topology | Automatic valid / cases | Forced valid / cases | Automatic geometric mean | Forced geometric mean | Automatic parallel cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-way join tree | 8 / 8 | 8 / 8 | 1.025454x | 0.906822x | 2 |
| Four-way join tree | 8 / 8 | 8 / 8 | 1.021159x | 0.889462x | 3 |
| Disconnected outputs | 8 / 8 | 8 / 8 | 1.023106x | 0.845081x | 3 |
| Fanout | 8 / 8 | 8 / 8 | 1.024067x | 0.853282x | 3 |
| Two successive waves | 4 / 4 | 4 / 4 | 1.000022x | 0.693758x | 0 |
| Three successive waves | 4 / 4 | 4 / 4 | 1.000334x | 0.659003x | 0 |
| All DAG cases, CPU16 | 40 / 40 | 40 / 40 | **1.018749x** | **0.829732x** | 11 |
| Two-branch baseline, CPU16 | 12 / 12 | 12 / 12 | **1.014564x** | **0.851634x** | 2 |

The 11 automatically parallel DAG cases all improved over baseline, with a
1.069356x geometric mean and a range of 1.017317x to 1.133467x. Their measured
speedups are listed completely below; a dash means that automatic mode retained
serial execution for that shape/topology combination.

| Shape / inputs | Three-way | Four-way | Disconnected | Fanout |
| --- | ---: | ---: | ---: | ---: |
| 96x384 / independent | 1.076735x | 1.102204x | 1.099019x | 1.105647x |
| 96x384 / shared | — | 1.017317x | 1.018229x | 1.022109x |
| 1025x768 / independent | 1.133467x | 1.054113x | 1.071997x | 1.069215x |

All 29 automatically serial DAG cases have exact baseline TTGIR after removing
only cost diagnostics, and exact baseline PTX. Their timing geometric mean is
1.000186x, with a range of 0.999735x to 1.001191x. The final two-branch run
selected parallel execution only for independent inputs at 64x1024 and
4096x1024, measuring 1.089140x and 1.092697x; the other ten
cases are also code-equivalent to baseline. Timing variation for identical
code is not a compiler optimization.

### CPU16-bound heavy-wave follow-up

The two heavy-wave runs use shapes 15x512, 128x512 and 4096x512, with 80
nonlinear iterations in the first wave and six in each later wave. Each run
contains both two-wave and three-wave kernels, and all six cases completed
numerical checks and all timing rounds for baseline, forced and automatic
variants. These follow-up processes and the fanout retry were explicitly bound
to CPU16, as recorded by `environment.cpu_affinity: [16]`. They use the same
compiler library and measurement protocol as above. Their aggregates remain
separate from the 40-case DAG suite and from each other because the
profitability threshold and selected schedules differ.

| Shape | Topology | Automatic widths, threshold 1.0 | Automatic speedup, 1.0 | Forced speedup, 1.0 run | Automatic widths, default 1.1 | Automatic speedup, 1.1 | Forced speedup, 1.1 run |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 15x512 | Two waves | [2, 2] | 1.129166x | 1.123777x | [2] | 1.138477x | 1.123775x |
| 15x512 | Three waves | [2, 2, 2] | 1.121364x | 1.113231x | Serial | 1.000290x | 1.112989x |
| 128x512 | Two waves | [2, 2] | 1.079249x | 1.077703x | [2] | 1.088718x | 1.078015x |
| 128x512 | Three waves | [2, 2, 2] | 1.072099x | 1.064959x | Serial | 0.999182x | 1.065723x |
| 4096x512 | Two waves | [2, 2] | 0.972525x | 0.925831x | [2] | 1.114631x | 0.925990x |
| 4096x512 | Three waves | [2, 2, 2] | 0.962177x | 0.916010x | Serial | 0.999570x | 0.915623x |
| Geometric mean | Six cases | — | **1.053989x** | **1.033368x** | — | **1.055178x** | **1.033460x** |

At threshold 1.0, automatic mode schedules every nonlinear wave and rejects the
cheap auxiliary index wave that forced mode also schedules. Small and medium
grids improve by 1.072099x to 1.129166x. At 4096 rows, however, two-wave latency
rises from 88.755 to 91.262 us and three-wave latency from 95.708 to 99.470 us:
2.83% and 3.93% longer than baseline. Lowering the threshold demonstrates
automatic successive-wave execution, not a universal performance win.

At the default threshold 1.1, all three two-wave cases schedule only their heavy
first wave and improve by 1.088718x to 1.138477x. The model estimates 1.163588x
for that first wave and only 1.006391x for the second. For three-wave kernels,
the additional surrounding work lowers the first-wave estimate to 1.070290x;
all three remain serial with exact baseline TTGIR/PTX. This conservative choice
misses the small/medium-grid gains seen in the threshold-1.0 run, while avoiding
the large-grid regression. No model coefficients were changed for this probe.

Every fully scheduled heavy kernel launches eight warps for either two or three
waves, confirming that physical groups are reused. Automatic mode uses 16420 B
shared memory and 42 or 43 actual registers per thread, with five resident
CTAs/SM; the baseline uses 64 B, 72 registers and seven CTAs/SM. Default-threshold
two-wave kernels use only 8224 B shared memory because only one wave is moved.
They report 40 or 44 registers and six or five resident CTAs/SM. All follow-up
variants report zero spills, and their driver occupancy queries succeeded.
These measurements expose the tradeoff between independent branch overlap,
inter-wave transfers and whole-kernel resource use; they do not isolate each
factor through profiling.

### Resource effects and regressions

Driver occupancy queries succeeded for every recorded variant. All variants
report zero spills. Driver-reported active CTAs/warps are resource-limited
residency, not profiled achieved occupancy. The following examples connect
measured regressions to generated scheduling and resource changes; they are
explanations supported by these records, not a profiler-based causal breakdown.

| Case | Baseline / automatic / forced latency (us) | Relevant evidence |
| --- | --- | --- |
| Three-way, 96x384, independent | 1.944 / 1.805 / 2.113 | Automatic keeps one three-way wave and 320 B shared memory. Forced adds a small index wave and uses 8264 B. Both launch 12 warps and allow four CTAs/SM. Extra synchronization and transfers erase the main wave's gain. |
| Fanout, 1025x768, shared | 5.654 / 5.654 / 12.321 | Forced speedup is 0.458907x. Registers rise from 56 to 106, shared memory from 64 to 16704 B, resident CTAs fall from nine to one and active warps from 36 to 12. The forced wave records 49248 B of transfers. Automatic rejects it. |
| Three waves, 6144x1536 | 20.567 / 20.543 / 32.029 | Forced speedup is 0.642149x. Shared memory rises from 64 to 65560 B, resident CTAs fall from seven to one and active warps from 28 to eight. Both actual scheduled waves record 65536 B of transfers; the legal forced schedule does not necessarily parallelize every source-level wave. |
| Disconnected outputs, 5x96, independent | 1.174 / 1.173 / 1.760 | Forced creates widths [4, 2, 2], including cheap index/output work. Small branch work cannot repay the added synchronization. Automatic retains serial code. |
| Two-branch baseline, 4096x1024, independent | 8.786 / 8.041 / 8.968 | Automatic uses only 192 B shared memory, lowers actual registers from 42 to 40 and raises resident active warps from 40 to 48. Forced adds a second wave, uses 16436 B and reports 48 registers. |

The model is useful for rejection but is not an exact hardware predictor. For
6144x1536 four-way independent inputs it estimates 332 baseline registers,
while the compiled kernel reports 79. This produces a conservative
`resource_limit` decision; forced execution measures 0.998934x in that case. The
model also chooses a lower-transfer orientation for 96x384 shared four-way and
fanout cases, yet forced orientation is slightly faster in these measurements.
The parameters were not retuned from these results.

### Retained failure and outstanding work

In the original, superseded 40-case run, fanout at 6144x1536 with shared input
had no legal forced wave and
produced a serial kernel. Its numerical check passed and two timing rounds
completed. During the third round, the host raised:

```text
SystemError: .../triton/runtime/jit.py:-1: unknown opcode 0
```

The raw artifact retains `correct: true` for that variant's numerical check,
`status: failed`, `median_us: null`, its partial rounds and the failure stage
`timing_round_2`. The whole case is marked `correct: false` to flag incomplete
validation. No forced latency was synthesized from partial rounds. Baseline and
automatic completed all rounds; the automatic serial kernel is code-equivalent
to baseline. The original run's historical aggregate was 1.019462x for 40
automatic results and 0.827290x for 39 valid forced results. Neither value nor
any partial timing is included in the final CPU16-bound table.

The original two-branch artifact is also retained separately: it measured
1.016140x automatically and 0.850622x forced across 12 cases. Those historical
timings are not mixed into the final CPU16-bound baseline row.

A separate retry bound to CPU16 completed all three variants and six timing
rounds for this same case. Baseline, forced and automatic latencies were 25.443,
25.419 and 25.440 us, respectively. All remained serial with identical recorded
resources; automatic TTGIR/PTX matched baseline. The retry is preserved in its
own raw artifact. It does **not** overwrite the original failure or change its
historical 39-result forced aggregate. The subsequent full CPU16-bound run
completed all 40 cases in all three variants and is the source of the final
performance table. These successful reruns do not establish that the underlying
host problem has been fixed.

The repeated-compilation investigation now includes a heap failure reproduced
outside the new pass, in an independent LLVM-container control. The precise
root cause remains unresolved. The observed affinity probes are:

- Eight automatic compilations completed on CPU0, and eight completed on CPU16.
- Two production automatic-compilation attempts on CPU12 failed in their first
  iteration through a hang or a double-free report. Eight baseline compilations
  on CPU12 completed.
- A layout-ASan probe on CPU12 failed inside AddressSanitizer with an internal
  `alloc_type != dealloc_type` assertion, reporting equal operands (`1 == 1`).
  This is not a successful sanitizer run and does not identify a source-level
  defect in this scheduling pass.
- An independently written
  [container control](../../python/benchmarks/tle/results/tile_dataflow_dag_logs/host-container-check.cpp)
  uses only LLVM `SmallVector`, `MapVector` and LLVM Support, without Triton,
  this pass, Python or Torch. On remote CPU12 it reported
  `munmap_chunk(): invalid pointer` and exited with status 134. The same binary
  completed 37,031,936 checked container iterations in 20 seconds on remote
  CPU16. The control also completed 2,299,904 iterations in a one-second local
  run.
- Alternating three-second runs of that container binary on CPU12, CPU16 and
  CPU0 were repeated twice. CPU12 failed with SIGSEGV/status 139 in one round
  and a container-size invariant failure/status 2 in the other; all CPU16 and
  CPU0 runs passed.
- A [libc-only allocation control](../../python/benchmarks/tle/results/tile_dataflow_dag_logs/host-allocation-check.c)
  using `malloc`/`realloc` passed 20-second runs on both cores: 500,231,168
  iterations on CPU12 and 248,797,184 on CPU16. A
  [Python standard-library control](../../python/benchmarks/tle/results/tile_dataflow_dag_logs/host-python-check.py)
  also passed 25-second runs on both cores.

The failure therefore reproduces in an independent LLVM-container path outside
the new pass and correlates with core selection, while the libc and Python
controls did not reproduce it. These controls do not establish whether all
observed failures share one cause, or distinguish an LLVM, system or hardware
root cause. CPU16 pinning is a recorded validation condition, not a diagnosed
fix. The completed CPU16 GPU and performance measurements are reported with
this limitation, without a root-cause conclusion. Existing compiler and
sanitizer logs are preserved in
[the validation log directory](../../python/benchmarks/tle/results/tile_dataflow_dag_logs/).
Passing GPU racecheck, memcheck and synccheck does not establish host/compiler
memory safety. The successful heavy-wave and fanout measurements remain valid
for their recorded environment while this host reliability limitation remains
open.

## Reproduction

Use matching FlagTree Python sources and the compiled native extension. Run the
MLIR tests through the configured build's `llvm-lit` environment:

```bash
llvm-lit third_party/tle/test/GPU/test_tle_tile_dataflow*.mlir
llvm-lit test/Conversion/warp_specialize_to_llvm.mlir test/TritonGPU/automatic-warp-specialization.mlir
```

Run both GPU suites on the final artifact, then rerun them under each sanitizer:

```bash
python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py python/test/unit/language/test_tile_dataflow_dag.py -q --tb=native
compute-sanitizer --tool racecheck --error-exitcode 1 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py python/test/unit/language/test_tile_dataflow_dag.py -q --tb=native
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py python/test/unit/language/test_tile_dataflow_dag.py -q --tb=native
compute-sanitizer --tool synccheck --error-exitcode 1 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py python/test/unit/language/test_tile_dataflow_dag.py -q --tb=native
```

The latest unified correctness run used CPU16 affinity:

```bash
taskset -c 16 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py python/test/unit/language/test_tile_dataflow_dag.py -q --tb=native
```

The DAG benchmark compares baseline, forced and automatic variants of the same
kernel. It records raw CUDA-graph replay timing rounds, model decisions, selected
wave widths, compiled resources and per-variant failures. Compilation and Python
launch overhead are excluded from the timing:

```bash
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --input-mode both --rounds 6 --resources --dump-ir-dir dag-ir --output dag-cpu16.json
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --suite baseline --rounds 6 --resources --output baseline-cpu16.json
python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --suite holdout --rounds 6 --resources --output holdout.json
```

The DAG suite includes three/four-way trees, disconnected outputs, fanout, and
two/three sequential waves. `--shape`, `--dtype`, `--depth`,
`--first-wave-depth`, `--later-wave-depth`, `--num-warps` and
`--max-partitions` support additional cases. The heavy-wave follow-ups and fanout
retry use CPU16 affinity and separate outputs. The two heavy runs explicitly
identify their different profitability thresholds:

```bash
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --topology waves2 --topology waves3 --shape 15 512 --shape 128 512 --shape 4096 512 --first-wave-depth 80 --later-wave-depth 6 --min-speedup 1.0 --rounds 6 --resources --output heavy-cpu16.json
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --topology waves2 --topology waves3 --shape 15 512 --shape 128 512 --shape 4096 512 --first-wave-depth 80 --later-wave-depth 6 --min-speedup 1.1 --rounds 6 --resources --output heavy-default-cpu16.json
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --topology fanout --shape 6144 1536 --input-mode shared --rounds 6 --resources --output fanout-retry-cpu16.json
```

When automatic mode retains a fully serial kernel, the benchmark compares TTGIR
after removing only the diagnostic string and compares exact PTX. Driver
occupancy queries under `--resources` report resource-limited residency, not
profiled achieved occupancy.
