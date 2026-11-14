# Tile dataflow branch scheduling

`triton-tle-tile-dataflow-schedule` builds a dependency graph of tile-level
TritonGPU IR and schedules independent branches onto separate warp groups.
It does not dispatch on named operator combinations. Enable it per kernel:

```python
import json

compiled = kernel[grid](
    ..., num_warps=4,
    enable_tile_dataflow=True,
    tile_dataflow_min_speedup=1.1,
    tile_dataflow_coalesce=True,
)
decisions = json.loads(compiled.metadata.tile_dataflow_cost_model or "[]")
```

The option defaults to `False`. When enabled, the pass first checks dependency
and capture legality, then estimates whether parallel execution pays for its
synchronization, communication and occupancy costs. A legal graph may remain
serial. The default `tile_dataflow_min_speedup=1.1` requires the candidate's saving, after
whole-function occupancy penalties, to cover at least 10% of its intrinsic wave
cost. A communication uncertainty budget may require more headroom. This is a
heuristic acceptance threshold, not a predicted whole-kernel speedup guarantee.

For debugging and comparative measurements, `tile_dataflow_force=True` together
with `enable_tile_dataflow=True` bypasses profitability. It still respects graph
dependencies, capture requirements and the introduced-buffer limit.
`tile_dataflow_coalesce=True` is the default when scheduling is enabled. Set
`tile_dataflow_coalesce=False` to retain separate fork/join dispatches for an
ablation comparison; this keeps multiwave DAG scheduling and its profitability
checks enabled.

## Python kernel-call fusion

Cross-call fusion is implemented in core Triton and does not require TLE.
See [experimental kernel-call fusion](../../docs/python-api/kernel-call-fusion.rst)
for capture, replay, address-dependency proofs, supported cases and standalone
validation. It does not enable this tile-dataflow branch scheduler by default.

## Graph and schedule

`TileDataflowGraph` has one node per operation in a block, including scalar
address/shape calculations supporting tile operations. Edges represent:

- SSA definitions and uses, including operands captured by reductions;
- memory read-after-write, write-after-read and write-after-write hazards;
- control-flow, synchronization, asynchronous and unknown-effect boundaries.

Read/read accesses can run concurrently. Different pointer arguments are not
assumed disjoint. The analysis reuses TLE pointer provenance helpers to prove
different address spaces or different shared-memory allocations disjoint, and
otherwise conservatively assumes aliasing.

The scheduler considers tensor outputs throughout the block DAG, including
outputs consumed by different joins or stores. It follows their ancestors,
leaves common work outside the fork, and excludes operations whose intermediate
results escape the branches. Backward slices stop at the nearest prior scheduling
boundary. Values defined earlier remain captures, preventing constants before an
existing wave from creating a false conflict between later branches. The chosen outputs form an antichain: no branch
has a dependency path to another. The insertion point is their earliest outside
consumer. Both earlier and later movement must preserve every external
predecessor and successor edge, including memory hazards and boundaries. Each
branch must contain at least two tile work operations by default.

Candidate search preserves every legal pair of retained endpoints, greedily
grows independent sets from substantial chains, and expands internal joins to
expose wider frontiers. For example, it can
expand `{A, B, C+D}` into `{A, B, C, D}`. Internal-frontier expansion has a budget
of 256 successful expansions per graph snapshot. Automatic mode evaluates legal
choices of the default branch and selects the candidate with the largest
predicted saving across the block. Force mode prefers more branches, then more
selected operations. This is a global candidate search with a bounded heuristic,
not an optimal DAG partitioner or an exhaustive search of all schedules.

The selected branches become one default region and one or more worker regions
of `ttg.warp_specialize`. Each uses the original kernel's warp count and tensor
layouts; the pass does not divide the original warps among branches. Uniform
scalars and pointers are explicit captures. Cheap index/mask expressions and
constants are recomputed in workers. Remaining tensor captures are stored in
shared memory before the fork. Captures are deduplicated across workers and every
worker receives the same capture signature, loading only the values it needs.
Predicates use byte storage with explicit extension/truncation.

Each worker stores its result in shared memory. Its output can reuse a staged
input buffer when that capture is read by only this worker and its shared
storage type exactly matches the output: shape, element storage and shared
encoding must agree. This includes predicate byte storage. A capture read by
several workers cannot be reused, because one worker could overwrite it before
another finishes reading. Each worker reuses at most one matching capture;
otherwise it receives a separate output buffer. Legality, materialization and
cost estimation use the same reuse decision.

Reuse overwrites only the private staged copy, not the original input tensor or
global memory. The default region retains its original SSA values in registers.
Existing memory-barrier analysis handles the worker's read-before-write hazard
and supplies the required warp-group synchronization. The default branch returns
its result with `ttg.warp_yield`; after all workers join, the default group loads
their results and replaces every original output use. Shared buffers are
deallocated once after these loads. The graph is rebuilt after every accepted
wave, and all remaining candidates are re-evaluated. Successive waves can
therefore be scheduled automatically within the same kernel.

### Persistent adjacent waves

After materializing a candidate, the pass tries to combine adjacent, pass-owned
waves with identical worker counts and per-worker warp counts into one
`ttg.warp_specialize`. Workers stay in their regions across the logical waves.
The intervening code must consist of pass-owned shared transport and pure,
straight-line arithmetic, splats, broadcasts or dimension expansion. Moved
tensor and scalar results must not escape the combined region. The join work
duplicated in workers is capped at 64 weighted instruction units; arbitrary
joins, reductions, layout conversions and control flow between waves are not
rematerialized.

At an eligible boundary, each worker keeps its own preceding result in SSA.
The default group publishes its result if workers need it, and all participating
groups execute one counted named-barrier wait. Each worker then reads the other
results it needs and reconstructs the cheap join locally. This avoids reloading
its own previous result and removes the default group's write of the joined
tensor followed by a worker reload. Results still needed by other groups retain
their shared stores and loads. The default group also computes its join for the
next default branch. Subsequent compatible waves can extend the same persistent
region.

There is one logical cross-group rendezvous per merged boundary, in addition to
the outer warp-specialization entry and exit protocol. Each group emits its own
wait with the same virtual ID and the actual participating thread count;
padding warps are excluded. Existing named-barrier allocation maps the virtual
ID to an available hardware resource. The generated full-group marker allows
Membar to close pending shared dependencies, and lowering uses the counted
non-`aligned` barrier form because groups arrive through different control
paths. Ordinary named waits and nonblocking arrives retain their existing
behavior.

A buffer used to publish the default result becomes a broadcast when multiple
workers reconstruct the join. It may reuse a matching next-wave input only when
no worker will overwrite that broadcast during the phase; otherwise the pass
allocates separate storage. With a single worker, a private input can still
become its output with any required group-local WAR barrier. A local barrier
alone cannot protect a broadcast from a different worker's early overwrite.

### Shared-buffer phases

The persistent region also pools compatible pass-owned shared allocations.
Its phase boundaries are the blocking full-group waits: every region must have
the same constant barrier-ID and thread-count sequence. Allocations with
nonoverlapping phases can share a memdesc across groups. Within one phase,
reuse requires every access on both sides to belong to the same group, with
strictly ordered, nonoverlapping operation ranges. Existing Membar then inserts
any needed group-local WAR/WAW barrier. Multiple readers in different groups
prevent same-phase reuse.

Pooling requires matching shared types and alignment, and accepts only direct
local loads/stores and known deallocations. Views, descriptor escapes, nested
control flow, asynchronous users and another warp-specialization region reject
pooling for that allocation. Initialized captures keep their producer writes;
uninitialized buffers may reuse earlier storage. Duplicate capture operands and
worker arguments are removed, unused reconstructed inputs are dropped, and the
remaining deallocation covers the last use of every pooled member. Uninitialized
transport allocations are placed before their first use so they do not remain
live across an unrelated earlier dispatch.

Membar retains group-local WAR barriers even when a load and a later store
have identical, nonreplicated tensor layouts and use the exact same memdesc.
The layout describes ownership of the resulting SSA values, which does not
prove ownership of every physical read: NVIDIA lowering may implement the load
with a collective `ldmatrix` instruction before scalar stores. A two-wave
4x512 float32 case reported a read by thread 224 followed by a write by thread
225 to the same shared address when this barrier was omitted. Restoring the
barrier removed the racecheck reports in the same case. Rank-one tensors and
plain shared layouts are not blanket exceptions because matrix-load lowering
is selected from the layout conversion, not solely from tensor rank. The
cross-group named rendezvous and existing RAW/WAW handling remain in effect.

Existing warp-group allocation determines launch dimensions; existing shared
allocation, memory-barrier analysis and warp-specialization lowering provide
storage and synchronization. Worker resources are reused between waves: launch
warps equal the original warps plus the module-wide maximum worker-warp count,
rounded up to a multiple of four. They are not summed across waves. Named-barrier
allocation runs after this pass so it sees the new groups. No runtime task queue
or host launch per graph node is introduced.

## Profitability model

Costs use weighted instruction units, not GPU clock cycles or microseconds.
There are no operator-combination or benchmark-shape lookup tables. The model
uses each tensor's layout-dependent elements per thread, operation categories,
reduction communication and recognized static tail masks. It includes worker
recomputation separately. Unknown branch or surrounding operations cause the
automatic candidate to remain serial.

For `n` branch costs `W[i]`, surrounding work `O`, worker recomputation `E`,
synchronization cost `S`, and shared communication cost `C`, the intrinsic
estimates are:

```text
B      = sum(W)
serial = B + O
wave   = B - overlap * (B - max(W)) + S + C + E
S      = sync-cost * (1 + 0.25 * (n - 2)
                      + reuse_barriers / 3 + boundary_sync_units)
```

By default, only half of the work outside the longest branch is credited as
overlap: all groups share the SM's instruction, memory and special-function
resources. `sync-cost` remains the aggregate two-branch dispatch/join cost. The
existing lowering has three common synchronization phases: release waiting
workers, protect capture loading, and join completed branches. Additional worker
bookkeeping scales this aggregate. If any worker reuses an input buffer for its
output, `reuse_barriers=1`; otherwise it is zero. One extra warp-group WAR barrier
is charged on the critical path at `sync-cost / 3`. Concurrent workers do not
each add another serialized barrier to this estimate. The division is a
conservative cost approximation, not a claim that hardware barriers have equal
latency.

For an ordinary wave, `boundary_sync_units=0`. A persistent region pays its
outer dispatch/join once and adds `1/3` per merged logical boundary for its
single counted rendezvous. Its reuse-barrier component is re-estimated from
the resulting loads and stores after pooling. The model tracks WAR hazards per
group and phase, takes the maximum group count within each phase, and sums
those phase costs. Identical SSA tensor layouts do not exempt a hazard. Existing Membar
decides the barriers actually emitted; this estimate is not a complete
simulation of lowering.

`C` counts modeled shared-memory traffic. A capture used by `k` workers costs
`(k+1) * bytes`: one producer write and `k` reads. Each worker result costs two
transfers. Buffer allocation counts a common capture only once, and predicate
storage uses bytes. Input/output buffer reuse reduces the allocated footprint
and capture metadata, but does not remove these tensor reads and writes; the
traffic charge remains.

Ordinary emitted waves store six cost components: branch-work sum, longest
branch, recomputation, transfer bytes, branch count and the extra reuse-barrier
count. Persistent regions add a seventh component, `boundary_sync_units`.
Their branch-work and longest-branch terms sum the logical waves; recomputation
includes the intervening join and worker copies. Shared traffic is recounted
from the resulting local loads/stores, initialized captures and final output
reads, so eliminating a round trip reduces its charge. Pooling without removing
loads or stores reduces allocation, not transfer bytes. Historical
five-component records remain supported and imply no reuse barrier; six-component
records imply no persistent boundary charge.
Subsequent decisions include their intrinsic work in `O`, using the current
overlap, bandwidth and synchronization coefficients, including `sync-cost / 3`
for a recorded reuse barrier.
Their transport operations are already included in this cost. Whole-kernel
resource penalties are re-evaluated separately. A forced wave with unknown work
has no modeled components, so a later automatic decision cannot treat it as
free work.

The existing `ModuleAllocation` analysis runs on the original module and a
temporary clone containing the proposed schedule, including eligible adjacent
wave merges, phase pooling and allocation placement. It includes existing shared
buffers, scratch allocations and warp-specialization communication storage.
Rejecting the clone leaves the original instructions unchanged.

A forward liveness sweep estimates live tensor registers, allowing dead inputs
to supply result registers. Register views share their source lifetime;
constants and recomputable indexing do not add full tensor payloads. The model
adds a 12-register scalar/control reserve. It takes the maximum pressure across
the default group, worker groups and the surrounding function, while preserving
default-group values needed after the fork. This estimate is not ptxas register
allocation: scalarization, instruction scheduling, register packing and spills
can change the actual result.

Supported automatic resource profiles are CUDA SM80, SM86, SM89 and SM90.
Resident-CTA estimates consider registers, shared memory, warp and CTA limits,
256-register allocation units per warp, 256-byte shared allocation units and a
1 KiB shared reserve per CTA. They assume maximum shared-memory carveout and
single-CTA programs. The architecture limits come from NVIDIA's
[Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[Ada](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) and
[Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) guides.
Unknown targets remain serial in automatic mode.

The parallel estimate is penalized when resident warps decrease. All surrounding
work, including earlier waves, receives an additional penalty when fewer CTAs
can reside on an SM: added worker warps occupy resources even while idle during
serial work or narrower waves. Apparent occupancy improvements receive no
speedup credit: the register estimate is a proxy, and the unknown launch grid
may not fill additional resident CTAs. For a merged candidate, `wave` is the incremental
intrinsic cost of the transformed function relative to its previous surrounding
work; synchronization and communication diagnostics likewise compare the old
and trial schedules. With the nonnegative full resource-loss penalty `P`,
acceptance uses a robust marginal saving:

```text
parallel              = wave + O + P
marginal_saving        = B - wave - P
communication_reserve = (S + C) * (1 - overlap)
required_saving       = max((min-speedup - 1) * wave, communication_reserve)
accept                = marginal_saving >= required_saving
```

The full occupancy loss is deducted, but
unrelated serial work and earlier waves do not inflate the required percentage
saving. This prevents a long
serial tail from rejecting an otherwise worthwhile wave solely by diluting its
whole-function ratio. The communication reserve protects against accepting a
tiny positive saving that coarse synchronization or transfer estimates could
reverse, including when `min-speedup=1`. It uses the existing non-overlap
fraction as a heuristic uncertainty budget. The model takes the larger of this
reserve and the requested margin, rather than adding both. Neither acceptance
budget is counted as executed work or stored in an old wave's intrinsic cost.

This remains a local, greedy estimate after each accepted wave. It does not
optimize the final schedule jointly, roll back earlier choices or perform
profile-guided tuning. The pass does not receive the actual launch grid, so CTA
resource estimates alone cannot distinguish an underfilled launch from a
saturated one. Cache state, runtime carveout and instruction issue contention
are also not predicted precisely. A lower shared footprint or more predicted
resident warps does not by itself establish a throughput gain. Measure
representative inputs before relying on an automatic decision.

The module string attribute `tle.tile_dataflow.cost_model` contains a JSON array,
also exposed as `compiled.metadata.tile_dataflow_cost_model`. Each record names
the function, insertion operation (`join`), wave, branch count, default branch,
worker operands, decision and reason. A worker operand index is `-1` when that
output is not an operand of the insertion operation. `wave` is the selection
ordinal within the function, not its final execution order: a later-selected
wave can precede an earlier-selected wave in the IR. Records also report
estimated serial/parallel work, synchronization, shared transfers, recomputation,
surrounding work, occupancy adjustments, registers, shared bytes and resident CTAs.
`reuse_barriers` reports the candidate's initial private input/output reuse
estimate; a persistent region's cost components instead use its resulting
per-phase WAR estimate. The acceptance
fields `marginal_saving`, `required_saving` and `communication_reserve` expose the
comparison above. `marginal_speedup` is `B / (wave + P)`, while the existing
`predicted_speedup` remains the whole-function ratio `serial / parallel`.
Neither ratio alone determines acceptance: the margin applies to intrinsic
wave cost and the occupancy adjustment is applied separately. These are
estimated model ratios, not measured GPU speedups.
`coalesced_boundaries` reports how many boundaries the candidate combines.
In TTGIR, `tle.tile_dataflow.logical_waves` records the number of logical waves
inside a persistent region, `tle.tile_dataflow.wave_costs` preserves their
original components, and `tle.tile_dataflow.pooled_buffers` records allocations
reused by its latest pooling step. A physical WS count alone no longer measures
the number of scheduled logical waves.
Accepted history is preserved on repeated pass runs; records are sorted by
function and selection ordinal. An empty string means that no eligible candidate
produced a record. Compare
these estimates with actual compiled-kernel resources and timing separately.

## Current scope

This version schedules straight-line regions in public, single-block CUDA
kernel entry functions with one CTA per program and 4, 8 or 16 default warps.
The default allows up to four simultaneous branches, including the default
group. The hardware cap is `floor(32 / original_num_warps)` branches, and the
module-wide launch must fit 32 warps. For example, 16 original warps allow at
most two branches. Lowering also limits each function to 255 worker dispatch
states across all waves, including allocator padding; candidates check this
budget for every function affected by the module-wide allocation.

It does not extract branches across or inside loops, conditionals or existing
warp-specialization regions. These regions separate scheduling intervals in the
block DAG. Compatible pass-owned waves can subsequently be combined by the
restricted transformation described above; arbitrary existing regions are not
merged. Dependent branches, volatile loads, atomics, asynchronous operations
and unknown side effects remain ordered. Pure reductions are supported.

The pass skips kernels with user-provided named barriers, inline assembly,
low-level thread operations, or a `ttg.maxnreg` setting. These may impose physical
thread/barrier/register contracts that ordinary dependence analysis cannot
recover. Captured scalar values must be proven uniform; tensor captures must
have static shapes and integer or floating-point elements.

Existing warp-specialization regions emitted by this pass have reusable cost
components, allowing further automatic waves. Arbitrary pre-existing regions
without these components remain unmodeled and cause automatic candidates in
that function to stay serial. Force mode can still schedule successive legal
forks and combine eligible waves. Persistent merging currently requires one
default result per wave, equal partition widths, a reconstructible staged
capture and a bounded cheap join. It does not migrate arbitrary work between
workers or search all possible persistent schedules. Loop-carried dependency
scheduling and unequal warp allocation per branch remain outside this pass.
Cross-kernel capture and address validation are handled by the upstream
Python frontend and TTIR fusion pass described above; this scheduler continues
to operate on the resulting single-kernel graph.

The command-line pass exposes the following options. Python launch options
expose `force` as `tile_dataflow_force` and `min-speedup` as
`tile_dataflow_min_speedup`, `max-partitions` as `tile_dataflow_max_partitions`,
and `coalesce-waves` as `tile_dataflow_coalesce`.

| Pass option | Default | Meaning |
| --- | ---: | --- |
| `min-branch-ops` | 2 | Minimum tensor work operations in each branch |
| `max-partitions` | 4 | Maximum simultaneous branches, including default; 1 disables scheduling |
| `max-shared-bytes` | 32768 | Limit on introduced tensor buffers per fork |
| `force` | false | Bypass profitability while retaining legality checks |
| `coalesce-waves` | true | Combine compatible adjacent waves and pool their shared transport; disable for separate-dispatch ablation |
| `min-speedup` | 1.1 | Requires saving of at least `(value - 1) * intrinsic_wave_cost`, after occupancy loss; communication reserve also applies |
| `sync-cost` | 128 | Aggregate two-branch dispatch/join cost; extra workers scale it and storage reuse adds one third |
| `shared-bandwidth` | 128 | Shared bytes transferred per model unit |
| `overlap` | 0.5 | Credited overlap fraction outside the longest branch; its complement sets the communication uncertainty budget |

The introduced-buffer limit is separate from the estimated whole-kernel
allocation. Final compiler and runtime resource validation still applies,
especially when profitability is forced.

## Validation

The MLIR regression tests cover graph-driven fork/join construction, multiple
branches and waves, disconnected outputs, shared inputs, idempotence, resource
decisions, SSA dependencies, pointer aliasing, memory hazards, synchronization,
unknown effects, volatile accesses, loop boundaries and nonuniform captures.
They also check private input/output reuse, preservation of captures shared by
workers, marginal acceptance with a serial tail, rejection of fragile savings,
and repricing historical cost records. Persistent tests cover logical wave
preservation, named-barrier lowering, intermediate SSA reuse, phase pooling,
broadcast overwrite protection, preservation of WAR barriers for cooperative
shared loads (including matrix loads), and resource accounting:

```bash
llvm-lit third_party/tle/test/GPU/test_tle_tile_dataflow*.mlir
```

GPU tests compare serial, forced and automatic modes against PyTorch references
and inspect generated TTGIR. The benchmark records all three modes, their model
decisions and actual compiled resources:

```bash
python -m pytest python/test/unit/language/test_tile_dataflow_schedule.py -q
python -m pytest python/test/unit/language/test_tile_dataflow_dag.py -q
python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --help
python python/benchmarks/tle/benchmark_tile_dataflow_schedule.py --suite holdout --resources --output holdout.json
```

The [multiwave optimization report](TILE_DATAFLOW_DAG_OPTIMIZATION.md) records
the previous policy's comparison against the same RTX 4090 benchmark groups.
The [persistent scheduling report](TILE_DATAFLOW_PERSISTENT_VALIDATION.md)
records adjacent-wave coalescing, the complete-parallel-plan ablation and
the remaining large-input profitability failures on the same groups.
The [previous DAG validation report](TILE_DATAFLOW_DAG_VALIDATION.md) retains
the original correctness, sanitizer results, speedups and suite failures.
The holdout suite includes different tile sizes and an imbalanced kernel;
`--prefix-depth` also varies surrounding serial work. When automatic mode retains
serial execution, the benchmark checks TTGIR equality after removing only the
cost diagnostic and exact PTX equality. `--resources` additionally attempts to
query resource-limited occupancy from the CUDA driver for the compiled kernel. A performance claim should include the GPU,
shapes, dtype, compiler revision and timing method; small graphs may lose to
branch-launch and shared-memory overhead.

## Related implementation

The approach follows TileLang's separation of tile/buffer dependence analysis
from scheduling decisions. Useful references are its
[pipeline planning](https://github.com/tile-ai/tilelang/blob/main/src/transform/pipeline_planning.cc)
and
[producer/consumer warp specialization](https://github.com/tile-ai/tilelang/blob/main/src/cuda/transform/producer_consumer_ws.cc).
FlagTree uses its own MLIR dependency graph and existing warp-specialization,
allocation and synchronization infrastructure.
