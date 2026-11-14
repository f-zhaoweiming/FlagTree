# Historical tile dataflow validation: 2026-10-17

This report records the previous two-branch scheduler. Its GPU measurements and
compiler hash apply to that revision, not the newer whole-DAG, multiway and
multiwave scheduler. See [the current DAG validation record](TILE_DATAFLOW_DAG_VALIDATION.md)
for the new implementation's status and reproduction steps.

Validated on 2026-10-17 using the working tree based on
`f4463cbb9d6a75523663e298af7c6bc7975f8a5a`, including the uncommitted pass changes.
LLVM: `f6ded0be897e2878612dd903f7e8bb85448269e5`.
The compiler was built with GCC 13, Release, Python 3.12.14 headers. GPU runs used
Python 3.12.15, PyTorch 2.11.0+cu130, RTX 4090 (SM89), driver 615.71.09 and
ptxas 13.4.92. Matching Python sources and compiler library ran in the authorized
isolated task directory; the existing installation was not replaced.

Compiler-library SHA256:
`212a75b8cedba89a84ef14d77612e493c4446e14cf177e22ac9e9ca1fd8aa0d8`.
The result JSON files include source hashes and raw timing rounds.

## Why the first scheduler regressed

The earlier scheduler selected every legal fork. Its original 12-case result
was **0.898× geometric mean speedup**, with a worst latency increase of about
35%. The [original measurements](../../python/benchmarks/tle/results/tile_dataflow_rtx4090_fp32.json)
are retained. The updated benchmark's forced mode reproduces the same policy.

Inspection of generated code and measured resources supports three causes:

- **Synchronization and dispatch dominate small tiles.** The scheduled kernels
  launch 8 warps instead of 4 and execute the warp-specialization dispatch,
  branch rendezvous and join. Static PTX inspection of the original independent
  128-column kernel showed 104 versus 142 instructions and 1 versus 10 barrier
  instruction sites. These are static sites, not dynamic barrier counts or a
  measured breakdown of elapsed cycles.
- **Shared inputs add communication.** At 1024 columns the shared-input kernel
  grows from 64 to 16,576 shared bytes. The model counts 32,800 transferred bytes
  per fork, including both writes and reads of the staged input and result.
  The independent-input alternative uses only 192 shared bytes.
- **Resource occupancy and useful default-group work change.** CUDA driver's
  occupancy query reports resource-limited maxima of 12 → 5 resident CTAs and
  48 → 40 resident warps per SM
  for the shared-input 4096×1024 case. For independent inputs it reports
  10 → 6 CTAs and 40 → 48 warps. Worker groups also consume resources while the
  default group executes work outside the fork. These API values are not
  profiled achieved occupancy. Tile independence alone cannot
  predict whether this tradeoff helps.

All successfully compiled measured kernels have **zero spills**. Register
spilling is therefore not an explanation for these measured regressions.
These observations do not isolate exact time attributable to each cost.

## Implemented decision model

The pass still builds the SSA/memory-dependence graph; there is no operator-chain
pattern table or shape lookup table. For each legal two-branch candidate it
compares both choices of worker branch and estimates:

```text
serial   = left + right + surrounding_work
parallel = serial - 0.5 * min(left, right)
           + synchronization + staged_shared_traffic + recomputation
           + occupancy_penalty
```

Work uses layout-derived elements per thread, operation categories, reduction
communication and statically understood tail masks. These are weighted
instruction units, not predicted GPU cycles. Default synchronization cost is
128 units and staged shared bandwidth is 128 bytes/unit. The default decision
requires `serial / parallel >= 1.1`.

The existing `ModuleAllocation` analysis evaluates both the original IR and a
temporary clone containing the candidate. A liveness-based register proxy and
SM80/86/89/90 resource profiles estimate resident CTAs. Decreased resident warps
and reduced CTA residency for surrounding serial work add penalties; apparent
occupancy gains are not credited. Unknown target/work estimates retain serial
execution. Legality, aliasing and thread-semantics checks still apply in forced
mode. Rejected candidates leave instructions unchanged.

The register estimate is deliberately a proxy: for the independent 1024-column
case it estimates 76 baseline registers/thread, while ptxas actually uses 42.
Backend scalarization and instruction scheduling explain why IR payload
liveness must not be presented as final register allocation. Driver occupancy
is recorded separately in the benchmark and is not fed back into compilation.
Only SM89 performance was measured here.

Decisions and individual costs are available as JSON in
`compiled.metadata.tile_dataflow_cost_model`. Enable with:

```python
kernel[grid](..., enable_tile_dataflow=True)
# Profitability override for experiments; dependency legality remains enforced:
kernel[grid](..., enable_tile_dataflow=True, tile_dataflow_force=True)
# Optional acceptance margin:
kernel[grid](..., enable_tile_dataflow=True, tile_dataflow_min_speedup=1.1)
```

The option was disabled by default. This historical revision had a two-branch,
straight-line scope and treated functions already containing warp-specialization
conservatively. See [the design and options](TILE_DATAFLOW.md) for the current
implementation.

## Correctness and synchronization

- All **48 GPU tests pass**, with no skips. Coverage includes FP32/FP16/BF16,
  tails, padded strides, shared inputs, nondefault streams, aliased arguments,
  4/8/16 default warps, byte-staged predicates, automatic decisions, forced
  profitability overrides, and automatic reversal of worker/default branches.
- The reverse-worker test verifies the result, scratch copy, input immutability
  and output guards; the chosen worker transfers 32 bytes instead of staging
  the other branch's 16 KiB input.
- All four dataflow MLIR files pass. Cost regressions cover synchronization,
  shared bandwidth, required speedup, resource occupancy, unknown targets,
  invalid parameters, lower-traffic orientation, and legality under forcing.
- Both existing core warp-specialization tests pass.
- Compute Sanitizer reruns the final 48 tests under each tool: racecheck reports
  **0 hazards, 0 errors, 0 warnings**; memcheck and synccheck each report
  **0 errors**.
- In all **33** automatically rejected benchmark cases, TTGIR is identical after
  removing only the diagnostic string attribute, and **PTX is byte-identical**
  to baseline. This checks that rejection does not leave speculative rewrites.

The full TLE suite is **83/90 passed**, with the same seven unrelated failures
as before. These pipelines do not invoke the new scheduling pass:

- `Analysis/test-alignment.mlir`: expected analysis remarks missing.
- `GPU/test_tle_dslregion_inline_extract_ptr.mlir`: missing `arg_dialect`.
- `GPU/test_tle_reduce_data_duplication_remote_if.mlir`: invalid operand segments.
- `GPU/test_tle_remote_atomic_splat_ptr_lowering.mlir`: invalid operand segments.
- `GPU/test_tle_remote_call_arg_store_lowering.mlir`: invalid operand segments.
- `GPU/test_tle_lower_wgmma.mlir`: FileCheck mismatch.
- `GPU/test_tle_wgmma_pipeline_accumulator_chain.mlir`: FileCheck mismatch.

The [check results](../../python/benchmarks/tle/results/tile_dataflow_cost_validation_checks.json)
retain the GPU test/sanitizer output and native-suite summaries.

## Performance methodology

All variants use the same kernel, FP32 inputs, 4-row tiles, 4 requested default
warps, disabled FP fusion, and fixed seeded inputs. Forced/accepted schedules
use 8 total warps. Each case checks against PyTorch and the serial result before
timing. Measurements use 50 warmup launches and six rounds of 30 ms CUDA-graph
replay per variant, rotating through all six variant orders. Tables report the
median of per-round median GPU times with hot inputs; compilation and Python
launch overhead are excluded. GPU clocks were not locked.

The original 12 cases informed the model coefficients. After that, coefficients
were held fixed for the 24 additional cases: six different shapes, independent
or shared inputs, balanced nonlinear reductions or imbalanced nonlinear
full-tile branches. This is a synthetic scheduling study, not a whole-model
speedup claim.

## Original 12 cases

| Shape | Inputs | Baseline µs | Forced µs | Automatic µs | Auto choice | Forced speedup | Auto speedup |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 4×128 | Independent | 0.919 | 1.065 | 0.919 | serial | 0.862× | 1.000× |
| 4×128 | Shared | 0.897 | 1.069 | 0.897 | serial | 0.839× | 1.000× |
| 64×1024 | Independent | 1.712 | 1.572 | 1.572 | parallel | 1.089× | 1.089× |
| 64×1024 | Shared | 1.470 | 1.593 | 1.470 | serial | 0.923× | 1.000× |
| 128×128 | Independent | 0.932 | 1.082 | 0.932 | serial | 0.861× | 1.000× |
| 128×128 | Shared | 0.914 | 1.083 | 0.914 | serial | 0.844× | 0.999× |
| 4096×128 | Independent | 1.856 | 2.229 | 1.856 | serial | 0.833× | 1.000× |
| 4096×128 | Shared | 1.604 | 2.163 | 1.604 | serial | 0.741× | 1.000× |
| 4096×1024 | Independent | 8.922 | 8.003 | 8.008 | parallel | 1.115× | 1.114× |
| 4096×1024 | Shared | 5.604 | 6.047 | 5.606 | serial | 0.927× | 1.000× |
| 1027×513 | Independent | 2.323 | 2.262 | 2.322 | serial | 1.027× | 1.000× |
| 1027×513 | Shared | 1.855 | 2.291 | 1.857 | serial | 0.810× | 0.999× |

Geometric mean: **0.899× forced → 1.016× automatic**.
Automatic scheduling keeps the two profitable large independent forks, with
speedups of about 1.089× and 1.114× versus baseline, and rejects the other ten.
The ten serial kernels differ in timing by at most about 0.12% and have identical
PTX. The conservative policy also rejects a small 2.7% forced gain on 1027×513.
Raw data: [original-shape three-way results](../../python/benchmarks/tle/results/tile_dataflow_cost_rtx4090_baseline.json).

## Additional 24 cases, fixed model

| Shape | Inputs | Branches | Auto choice | Forced speedup | Auto speedup |
| --- | --- | --- | --- | ---: | ---: |
| 3×65 | Independent | balanced | serial | 0.995× | 1.001× |
| 3×65 | Independent | imbalanced | serial | 1.008× | 1.001× |
| 3×65 | Shared | balanced | serial | 0.944× | 1.001× |
| 3×65 | Shared | imbalanced | serial | 0.912× | 1.001× |
| 12×257 | Independent | balanced | serial | 0.973× | 1.000× |
| 12×257 | Independent | imbalanced | serial | 1.065× | 1.000× |
| 12×257 | Shared | balanced | serial | 0.884× | 1.000× |
| 12×257 | Shared | imbalanced | serial | 0.975× | 1.000× |
| 97×769 | Independent | balanced | serial | 0.951× | 1.000× |
| 97×769 | Independent | imbalanced | serial | 1.677× | 1.000× |
| 97×769 | Shared | balanced | serial | 1.034× | 1.000× |
| 97×769 | Shared | imbalanced | serial | 1.527× | 1.001× |
| 257×255 | Independent | balanced | serial | 1.166× | 1.001× |
| 257×255 | Independent | imbalanced | serial | 0.947× | 0.998× |
| 257×255 | Shared | balanced | serial | 0.977× | 1.002× |
| 257×255 | Shared | imbalanced | serial | 0.915× | 0.997× |
| 1536×1009 | Independent | balanced | parallel | 1.032× | 1.031× |
| 1536×1009 | Independent | imbalanced | serial | 0.838× | 0.996× |
| 1536×1009 | Shared | balanced | serial | 0.915× | 1.000× |
| 1536×1009 | Shared | imbalanced | serial | 0.952× | 1.001× |
| 8192×511 | Independent | balanced | serial | 1.018× | 1.000× |
| 8192×511 | Independent | imbalanced | serial | 1.102× | 1.002× |
| 8192×511 | Shared | balanced | serial | 0.810× | 1.001× |
| 8192×511 | Shared | imbalanced | serial | 1.073× | 1.002× |

All 24 cases pass correctness in all three variants. Geometric mean:
**1.014× forced**, **1.001× automatic**.
The automatic mode accepts one balanced independent fork (about 3.1% faster)
and retains baseline PTX for the other 23; their largest timing variation is
about 0.38%. No accepted schedule regresses in this measured set.

The model is conservative and misses real opportunities: the imbalanced
97×769 cases gain about **1.68× independent / 1.53× shared** when forced, but are
rejected. Backend instruction scheduling and actual per-kernel latency are not
fully represented by the static overlap/resource estimate. Consequently the
forced mode's mean is higher on this extra set, while also retaining cases as
slow as about 0.81×. Automatic mode should not be described as choosing the
fastest schedule for every graph. No parameter was retuned using these results.
Raw data: [additional-case results](../../python/benchmarks/tle/results/tile_dataflow_cost_rtx4090_holdout.json).

### Assembler failure retained in the record

The first additional-set run encountered one `ptxas` SIGSEGV during forced
compilation of the shared-input, imbalanced 1536×1009 case. The original failed
PTX and the successful retry are byte-identical (SHA256
`31a5f5155b7735fb88833ca3e923e773731d7b7f4e7f4094606e475489d997d0`). The original
assembler command succeeded on three consecutive retries, and the full
24-case rerun passed without compiler or model changes. This looks transient;
its underlying cause is unresolved, and no deterministic pass defect was
reproduced. The benchmark now preserves successful variants and explicit
per-variant failures instead of losing the whole case's evidence.

The [failed first attempt](../../python/benchmarks/tle/results/tile_dataflow_cost_rtx4090_holdout_first_attempt.json)
and [three assembler retries](../../python/benchmarks/tle/results/tile_dataflow_cost_ptxas_retries.json)
are retained separately from the final successful run.

## Reproduction

```bash
python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py -q --tb=native
compute-sanitizer --tool racecheck --error-exitcode 1 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py -q --tb=native
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py -q --tb=native
compute-sanitizer --tool synccheck --error-exitcode 1 python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py -q --tb=native
python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --suite baseline --rounds 6 --resources --output baseline.json
python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --suite holdout --rounds 6 --resources --output holdout.json
```
