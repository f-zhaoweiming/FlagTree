# Tile dataflow DAG scheduling: multiwave optimization

Report date: 2026-10-18. This record separates the original whole-block DAG
scheduler (v1), a shared-buffer-only ablation, and the final scheduling-policy
revision. The [v1 validation record](TILE_DATAFLOW_DAG_VALIDATION.md), its raw
measurements, and its historical failures remain unchanged. The final revision
passes all 77 GPU tests and the three sanitizer suites. Its same 40-case DAG
suite measures **1.018496x** automatically, versus v1's **1.018749x**. Overall
gain remains approximately **1.85%**: this revision has not significantly
expanded the set of cases that benefit. On the separate six-case heavy suite
at threshold 1.0, the automatic geometric mean improves from **1.053989x to
1.094402x** by retaining a useful first fork and rejecting marginal later
forks; both previously regressing cases now beat their within-run baselines.
At the default threshold 1.1, the same six cases improve from **1.055178x to
1.093570x**, with one parallel fork selected in every case.
The shared-buffer-only ablation below is recorded separately from final
revision results.

## Why the original overall gain was 1.87%

The original CPU16-bound 40-case DAG suite measured an automatic geometric-mean
speedup of **1.018749x**, using each case's own serial baseline. Automatic mode
selected parallel execution for 11 cases, whose geometric mean was **1.069356x**
and whose individual speedups were **1.017317–1.133467x**. The other 29 cases
retained the baseline's exact PTX and diagnostic-stripped TTGIR. Consequently,
most entries contribute approximately 1.0 to the overall geometric mean.

| Shape | Cases | Parallel cases | Automatic geometric mean | Forced geometric mean |
| --- | ---: | ---: | ---: | ---: |
| 5 x 96 | 10 | 0 | 1.000267x | 0.722466x |
| 96 x 384 | 10 | 7 | 1.043223x | 0.940697x |
| 1025 x 768 | 10 | 4 | 1.031983x | 0.758817x |
| 6144 x 1536 | 10 | 0 | 1.000238x | 0.919066x |
| All | 40 | 11 | 1.018749x | 0.829732x |

The eight `waves2`/`waves3` cases in this default suite all remained serial.
Very short kernels cannot amortize dispatch, joins and shared transfers. The
large-input cases do not automatically benefit from more branch workers:
existing CTAs already provide parallelism, and each specialized CTA consumes
more warps and may reduce resident CTAs. Forcing every eligible fork produced a
**0.829732x** geometric mean, demonstrating that increasing the number of
parallel decisions alone is not an improvement.

The geometric mean weights every case equally in log space; it is not a
prediction of application wall-clock savings. Summing the 40 per-case median
latencies gives **823.462650 us serial versus 820.609990 us automatic**, or
**1.003476x** for the hypothetical workload that invokes each case once.
The longest cases receive much greater weight in that sum and were retained
serial. This ratio of sums also is not a measured application run: it excludes
application-specific frequencies, data movement and host overhead.

Raw input:
[`tile_dataflow_dag_rtx4090_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_cpu16.json).

## Why later parallel waves regressed

The heavier diagnostic suite keeps the same `waves2`/`waves3` kernels, with 80
iterations in the first fork and six in each subsequent fork. It uses the
shapes 15 x 512, 128 x 512 and 4096 x 512. At the deliberately relaxed
`tile_dataflow_min_speedup=1.0`, v1 selected all two or three computation waves.
It sped up small grids but regressed on both 4096-row cases.

Three observations explain why merely selecting more waves was fragile:

1. A later wave has modeled serial work 2144 and intrinsic parallel work 1992.
   Its overlap credit is 536, while synchronization plus shared transfer costs
   already consume 384. The remaining saving is only 152 weighted instruction
   units, or approximately **0.59–0.64% of the modeled whole kernel**. A small
   communication or scheduling underestimate reverses that decision.
2. The register estimator predicts the first specialized kernel at 76 registers
   per thread and three resident CTAs, versus approximately 40–44 registers and
   five or six CTAs observed after backend lowering. Adding a later wave may
   reduce the estimate to 44 registers and five CTAs, so the incremental
   occupancy penalty becomes zero. This predicted improvement is not a reliable
   reflection of the actual before/after resource difference.
3. All three row counts receive the same cost decisions. The pass does not have
   a general launch-grid model, so fixed overlap credit cannot distinguish
   single-CTA latency from throughput under many concurrent CTAs. The 4096-row
   case makes the communication and scheduling cost particularly visible.

The original policy also misses useful prefixes. Adding a third, still-serial
wave changes the first-wave prediction from 1.163588x to 1.070290x through the
whole-kernel outside-work penalty, causing the default 1.1 threshold to reject
the first fork. This combines two different questions: whether the candidate
wave has sufficient local gain and whether its resources damage the remainder
of the kernel. The final policy revision evaluates these explicitly.

## Shared-buffer-only ablation

This ablation enables safe input/output buffer reuse without changing v1's
profitability policy. It preserves the original heavy-suite parameters and
the relaxed 1.0 threshold. All six cases completed all three variants with
correct numerical results, six interleaved timing rounds, 50 warmup launches,
30 ms CUDA graph repetitions, and host affinity CPU16 on the same RTX 4090.
These are benchmark correctness checks, not a substitute for final sanitizer
validation.

| Topology | Rows x columns | v1 automatic us | Buffer reuse automatic us | v1 speedup | Buffer reuse speedup | Automatic computation waves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| waves2 | 15 x 512 | 25.591746 | 25.672408 | 1.129166x | 1.128234x | 2 |
| waves3 | 15 x 512 | 27.851996 | 27.930228 | 1.121364x | 1.120812x | 3 |
| waves2 | 128 x 512 | 24.650187 | 24.597262 | 1.079249x | 1.083742x | 2 |
| waves3 | 128 x 512 | 26.869294 | 26.918470 | 1.072099x | 1.072415x | 3 |
| waves2 | 4096 x 512 | 91.261937 | 90.753649 | 0.972525x | 0.978433x | 2 |
| waves3 | 4096 x 512 | 99.469902 | 99.841568 | 0.962177x | 0.959987x | 3 |
| Geometric mean | Six cases | — | — | 1.053989x | 1.055204x | — |

Each speedup uses that run's own serial baseline. Direct latency differences
between the two runs should not be interpreted as isolated causal effects or
as confidence intervals.

Buffer reuse reduces actual automatic-mode shared allocation from **16420 to
8224 bytes** in every case. However, actual registers remain 42–43 per thread
and residency remains five CTAs / 40 warps per SM. The modeled later-wave
transfer volume remains **32768 bytes** and synchronization remains necessary.
The two large-grid cases still take **2.20% and 4.17% more time than their
respective serial baselines** (speedups 0.978433x and 0.959987x). Therefore the
allocation reduction alone does not solve multiwave regression; the model must
also reject marginal communication-heavy parallel decisions.

Forced mode still emits an additional small fork and uses 16408 shared bytes,
versus 16420 previously. Its six-case geometric mean is 1.034273x, compared
with 1.033368x previously. The ablation does not establish that storage capacity
is the sole or dominant cause of the observed slowdown, nor does it prove an
exact hardware cost attribution without counters.

Original raw input:
[`tile_dataflow_dag_rtx4090_heavy_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_rtx4090_heavy_cpu16.json).
The ablation raw file is
[`tile_dataflow_dag_v2_transport_heavy_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_transport_heavy_cpu16.json)
(SHA-256 `6f92a81d0b0e9feb84150056c59d6d7ea8bcb5c727298838a4157cf46d1aa129`).

## Final validation protocol

The final comparison must keep these populations separate:

| Population | Cases | Profitability threshold | Purpose |
| --- | ---: | ---: | --- |
| Original multiway DAG suite | 40 | 1.1 | Same broad topologies and shapes as v1 |
| Original two-branch baseline suite | 12 | Default 1.1 | Existing scheduling compatibility |
| Heavy successive-wave suite | 6 | 1.0 | Previously regressing relaxed-policy cases |
| Heavy successive-wave suite | 6 | 1.1 | Default-policy prefix and later-wave decisions |

The last two populations use identical tensors and computation depths but
different policies; their results must not be pooled into one geometric mean.
Likewise, the broad 40-case suite and the 12-case baseline suite remain separate.
For each population, report all cases, all three execution variants, selected
wave widths, actual shared/register/occupancy resources, geometric mean,
regression count and worst regression. Small measurement differences in
unchanged serial code should be identified as such.

Final runs use CPU16 because the earlier validation documented host failures
correlated with CPU12, including an independent LLVM-container reproducer
outside Triton and the pass. The root cause remains unresolved. This report
does not reclassify, delete or replace those historical failures. Compiler and
source hashes, exact run commands, GPU correctness, sanitizer results and
native regression outcomes belong with the final revision results below.

## Final revision results

### Separate population summary

All final cases completed all three variants, including all six timing rounds,
with correct outputs. Speedups are relative to each run's own serial baseline.
No combined geometric mean is formed across these populations.

| Population | Cases | v1 automatic geo | Final automatic geo | Final forced geo | Automatic cases >1% slower | Forced cases >1% slower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original DAG, threshold 1.1 | 40 | 1.018749x | 1.018496x | 0.818044x | 0 | 27 |
| Original two-branch baseline | 12 | 1.014564x | 1.013672x | 0.847782x | 0 | 12 |
| Heavy waves, threshold 1.0 | 6 | 1.053989x | 1.094402x | 1.034042x | 0 | 2 |
| Heavy waves, threshold 1.1 | 6 | 1.055178x | 1.093570x | 1.034027x | 0 | 2 |

Here “>1% slower” means `candidate_us / serial_us > 1.01`, rather than a
rounded speedup below one. Worst measured automatic latency increases are
0.1873% and 0.0881% in the 40- and 12-case suites, both on unchanged serial
code. Neither heavy suite has an automatic regression. The respective worst
forced latency increases are 117.8816%, 34.8462%, 9.3416% and 9.3400%.

### Policy and resource changes

The materializer can reuse a staged input buffer for a worker's output when
the shared-memory type matches and exactly one worker reads that capture.
Multi-reader captures are excluded because overwriting them would require a
different inter-worker synchronization protocol. Existing allocation and
barrier insertion handle the selected layout and its reuse hazards. Legality,
materialization and cost estimation use the same reuse decision; the model
includes the extra reuse barrier when required.

Acceptance now examines the work changed by the candidate:

```text
marginal_saving = branch_serial_work - intrinsic_parallel_work - occupancy_cost
communication_reserve = (sync_cost + shared_cost) * (1 - overlap)
required_saving = max((min_speedup - 1) * intrinsic_parallel_work,
                     communication_reserve)
accept = marginal_saving >= required_saving
```

This avoids applying the profitability margin to unrelated serial work or
already accepted waves. The full whole-function occupancy penalty is still
charged against the candidate's saving. Communication reserve provides an
error and contention allowance even at `min_speedup=1.0`; it is an acceptance
budget, not an instruction cost added again to previously accepted waves.
Diagnostics record `marginal_saving`, `marginal_speedup`, `communication_reserve`,
`required_saving` and `reuse_barriers` so these decisions are reviewable.
Register estimation and missing general launch-grid knowledge remain
limitations; this revision does not claim to have solved either completely.

### Same 40-case DAG suite

All 40 cases and all three variants completed successfully. Measurement settings
match v1: 50 warmups, six interleaved timing orders, 30 ms graph repetitions,
the default 1.1 threshold, and CPU16 affinity on the RTX 4090.

| Shape | Cases | v1 automatic | Final automatic | v1 forced | Final forced |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 x 96 | 10 | 1.000267x | 1.000244x | 0.722466x | 0.716245x |
| 96 x 384 | 10 | 1.043223x | 1.043187x | 0.940697x | 0.939270x |
| 1025 x 768 | 10 | 1.031983x | 1.031286x | 0.758817x | 0.757086x |
| 6144 x 1536 | 10 | 1.000238x | 0.999980x | 0.919066x | 0.879242x |
| All | 40 | 1.018749x | 1.018496x | 0.829732x | 0.818044x |

The automatic selection count remains **11 parallel / 29 serial**. Selected
cases have a geometric mean of **1.068658x** and a range of
**1.017697–1.133760x**. All 29 serial cases have the exact baseline PTX and
diagnostic-stripped TTGIR within the final run. No automatic case regressed by
1% or more; the minimum measured speedup is 0.998130x, a serial case with
unchanged code. The ratio of summed medians is **1.003345x**. These measurements
show that the automatic broad-suite result is essentially unchanged, not an
overall performance improvement over v1.

Forced performance worsens. Raw decisions establish a concrete scheduling
change at 6144 x 1536:

| Topology | v1 forced widths | Final forced widths | Registers/thread, v1 to final | Shared bytes | Resident CTAs/SM | v1 forced speedup | Final forced speedup |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| waves2 | [2, 2] | [2, 2, 2] | 65 to 80 | 65560 | 1 | 0.667685x | 0.562491x |
| waves3 | [2, 2] | [2, 2, 2, 2] | 66 to 104 | 65560 | 1 | 0.642149x | 0.488278x |

Buffer reuse reduces a candidate's required storage and permits later waves
under the unchanged per-candidate shared-memory limit. Forced mode ignores
profitability and emits these additional waves. Each new later-wave decision
records 131072 transfer bytes, one reuse barrier and **negative** marginal
saving (-874.67 weighted units). The final allocation and residency are unchanged
in these two cases, but there are more transfers, joins and registers. Thus the
forced comparison includes different schedules; it is not an isolated timing
comparison of the same schedule with one storage optimization. This explains
the important new regression mechanism, without attributing every smaller
cross-run timing difference to it. Automatic mode rejects these waves.

Raw final input:
[`tile_dataflow_dag_v2_rtx4090_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_rtx4090_cpu16.json).

### Same 12-case two-branch baseline suite

All 12 cases completed all variants with correct outputs. Automatic mode
selects the same two parallel cases as v1 and retains exact baseline PTX and
diagnostic-stripped TTGIR for the other ten cases.

| Metric | v1 | Final |
| --- | ---: | ---: |
| Automatic geometric mean | 1.014564x | 1.013672x |
| Forced geometric mean | 0.851634x | 0.847782x |
| Automatic, independent 64 x 1024 | 1.089140x | 1.088389x |
| Automatic, independent 4096 x 1024 | 1.092697x | 1.083445x |

The final ratio of summed medians is 1.028754x in automatic mode. Its minimum
speedup is 0.999120x, again a serial case with unchanged code; no automatic case
adds more than 1% latency. Forced mode adds more than 1% latency to all twelve
cases, with speedups from 0.741586x to 0.975400x. The final baseline result is
compatibility evidence, not a demonstrated improvement over v1.

Raw input:
[`tile_dataflow_dag_v2_rtx4090_baseline_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_rtx4090_baseline_cpu16.json).

### Same six heavy cases, relaxed threshold 1.0

All six cases and all variants completed correctly. The first/later computation
depths remain 80/6, shapes and measurement settings match the original heavy
suite, and the final automatic schedule contains **one two-way fork in every
case**. Thus the improvement is a scheduling choice: profitable prefixes are
retained, and expensive later forks remain serial. It does not mean that an
unchanged fully parallel multiwave schedule became faster.

| Topology | Rows x columns | v1 automatic waves | Final automatic waves | v1 automatic us | Final automatic us | v1 speedup | Final speedup | Final forced speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| waves2 | 15 x 512 | 2 | 1 | 25.591746 | 25.444774 | 1.129166x | 1.138600x | 1.123679x |
| waves3 | 15 x 512 | 3 | 1 | 27.851996 | 27.655046 | 1.121364x | 1.131851x | 1.114309x |
| waves2 | 128 x 512 | 2 | 1 | 24.650187 | 24.501838 | 1.079249x | 1.089927x | 1.078880x |
| waves3 | 128 x 512 | 3 | 1 | 26.869294 | 26.611344 | 1.072099x | 1.085161x | 1.070829x |
| waves2 | 4096 x 512 | 2 | 1 | 91.261937 | 79.411710 | 0.972525x | 1.117852x | 0.924001x |
| waves3 | 4096 x 512 | 3 | 1 | 99.469902 | 94.884592 | 0.962177x | 1.008380x | 0.914565x |
| Geometric mean | Six cases | — | — | — | — | 1.053989x | 1.094402x | 1.034042x |

Each speedup uses its own run's baseline. All six automatic cases now exceed
1.0x, with a ratio of summed medians of 1.078262x. The two large-grid cases go
from 0.972525x/0.962177x to 1.117852x/1.008380x. The smaller improvement for
`waves3` is material: its single-fork final kernel uses 79–80 registers per
thread and three CTAs/SM, versus 40–44 registers and five or six CTAs/SM for
`waves2`. The policy does not claim globally optimal scheduling or accurate
backend register prediction. Forced mode still regresses on both 4096-row
cases and has a 0.980485x ratio of summed medians despite its above-one geometric
mean.

The diagnostics explain the rejected later forks. Including reuse synchronization
raises their intrinsic cost from 1992 to 2034.67 weighted units. Their marginal
saving is only **109.33**, below the communication reserve of **213.33**, so
they are rejected even with a 1.0 user threshold. The model does not disable
multiple waves unconditionally: the new
`test_tile_dataflow_automatic_multiple_profitable_waves` cases with later depth
12 still automatically emit two and three actual computation waves. Those
cases check output correctness, wave structure and guards, and pass in the
77-test ordinary and sanitizer runs. They are correctness/structure coverage,
not an additional performance population pooled into the six-case result.

Raw input:
[`tile_dataflow_dag_v2_rtx4090_heavy_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_rtx4090_heavy_cpu16.json).

### Same six heavy cases, default threshold 1.1

This population preserves the default threshold and matches the original
default-heavy run. All cases and variants complete correctly. The final
automatic choices are again one two-way fork per case. V1 already selected
that first fork for `waves2`, but retained entirely serial code for `waves3`.
The marginal-work policy now admits the useful first fork in `waves3` while
rejecting the later communication-heavy candidates.

| Topology | Rows x columns | v1 automatic waves | Final automatic waves | v1 automatic us | Final automatic us | v1 speedup | Final speedup | Final forced speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| waves2 | 15 x 512 | 1 | 1 | 25.382992 | 25.442722 | 1.138477x | 1.138508x | 1.123663x |
| waves3 | 15 x 512 | 0 | 1 | 31.216777 | 27.658906 | 1.000290x | 1.131817x | 1.114224x |
| waves2 | 128 x 512 | 1 | 1 | 24.470690 | 24.497132 | 1.088718x | 1.089946x | 1.078860x |
| waves3 | 128 x 512 | 0 | 1 | 28.813581 | 26.606455 | 0.999182x | 1.085137x | 1.070703x |
| waves2 | 4096 x 512 | 1 | 1 | 79.791754 | 79.730947 | 1.114631x | 1.113210x | 0.924117x |
| waves3 | 4096 x 512 | 0 | 1 | 95.703803 | 94.898676 | 0.999570x | 1.008092x | 0.914578x |
| Geometric mean | Six cases | — | — | — | — | 1.055178x | 1.093570x | 1.034027x |

Automatic speedups range from 1.008092x to 1.138508x, with a ratio of summed
medians of 1.076866x. Forced mode still regresses on both 4096-row cases and
has a 0.980511x ratio of summed medians. The accepted schedules match the
relaxed-threshold population for these six inputs, but the results remain
separate repeated measurements under distinct policy settings.

Raw input:
[`tile_dataflow_dag_v2_rtx4090_heavy_default_cpu16.json`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_rtx4090_heavy_default_cpu16.json).

### Correctness and native regressions

All final GPU validation uses the compiler library with SHA-256
`29c5bc0ae0ccfe11201f913b4428092123e588fe4b31974f345a71ebc7ffb279`
and CPU16 affinity.

| Check | Result |
| --- | --- |
| GPU correctness | 77/77 passed in 49.80 s |
| Compute Sanitizer racecheck | 77/77 passed in 59.45 s; zero hazards/errors/warnings |
| Compute Sanitizer memcheck | 77/77 passed in 9.09 s; zero errors |
| Compute Sanitizer synccheck | 77/77 passed in 3.61 s; zero errors |
| Tile-dataflow MLIR regressions | 8 passed |
| Existing core warp-specialization regressions | 2 passed |
| Full TLE suite | 87/94 passed; seven failures retained |

The full-suite failing test names match the prior recorded run:
`test-alignment`, `test_tle_dslregion_inline_extract_ptr`,
`test_tle_lower_wgmma`, `test_tle_reduce_data_duplication_remote_if`,
`test_tle_remote_atomic_splat_ptr_lowering`,
`test_tle_remote_call_arg_store_lowering`, and
`test_tle_wgmma_pipeline_accumulator_chain`. There is no pristine-base clean
comparison, so this is not evidence that the complete repository is green or
proof of the cause of those failures.

Final logs are retained in
[`tile_dataflow_dag_v2_logs`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_logs).

### Reproduction and provenance

The
[`validation manifest`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_validation_checks.json)
records source and library hashes, raw-file hashes, agreement of case keys and
measurement settings with each original population, separate metrics, test
outcomes, and the prior manifest's hash. It preserves the distinction between
the transport ablation and final revision. The exact staged host commands are
saved in
[`run-flagtree-dag-v2-validation.sh`](../../python/benchmarks/tle/results/tile_dataflow_dag_v2_logs/run-flagtree-dag-v2-validation.sh).

With the matching compiler/Python package active on the validation host, the
four benchmark commands from the repository root are:

```bash
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --input-mode both --rounds 6 --resources --output /tmp/dag-final.json
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --suite baseline --rounds 6 --resources --output /tmp/baseline-final.json
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --topology waves2 --topology waves3 --shape 15 512 --shape 128 512 --shape 4096 512 --first-wave-depth 80 --later-wave-depth 6 --min-speedup 1.0 --rounds 6 --resources --output /tmp/heavy-relaxed-final.json
taskset -c 16 python python/benchmarks/tle/benchmark_tile_dataflow_dag.py --topology waves2 --topology waves3 --shape 15 512 --shape 128 512 --shape 4096 512 --first-wave-depth 80 --later-wave-depth 6 --min-speedup 1.1 --rounds 6 --resources --output /tmp/heavy-default-final.json
```

Use the staged script for the ordinary 77-test run and the three sanitizer
commands. The CPU affinity is part of this host's recorded conditions, not a
general requirement that every machine has or uses CPU16. Benchmark scripts
and all raw JSON files are retained in the repository; rebuilding or rerunning
still requires a compatible LLVM/compiler toolchain, Python/CUDA environment
and NVIDIA GPU.

### Limits of the result

The change improves which prefixes execute in parallel on the measured heavy
DAGs and reduces reusable-buffer capacity. It does not expand the 40-case
suite's profitable coverage, guarantee that every legal parallel wave is fast,
or establish performance for the later-depth-12 correctness cases. Occupancy
and instruction weights remain estimates, the launch grid is not generally
known, and no hardware-counter attribution was collected. The original
single-entry-block scope, conservative memory-dependence rules and existing
allocation/synchronization pipeline remain in effect; cross-loop or
cross-conditional scheduling is not introduced here.
