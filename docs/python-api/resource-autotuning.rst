P800 resource and partition autotuning
=====================================

``graph_fusion.autotune_resource_backend`` is an opt-in, inference-only
``torch.compile`` backend. It compares native Inductor and eager with P800 specialization candidates.
A specialized region binds its resource evidence to the binary returned by its
actual Triton launch. Native Inductor regions retain their wrapper-based evidence.
On torch_xmlir images that mock torch.compile, set
``XMLIR_ENABLE_MOCK_TORCH_COMPILE=false`` before importing Torch.

.. code-block:: python

   import torch
   from triton.experimental import graph_fusion

   backend = graph_fusion.autotune_resource_backend(
       tolerance={"atol": 3e-4, "rtol": 3e-4},
       environment_key="P800-device-id-driver-SDK-build",
   )
   compiled = torch.compile(gm, backend=backend, fullgraph=True, dynamic=False)
   with torch.no_grad():
       compiled(*inputs)  # finish calibration before any GPU graph capture
       result = compiled(*inputs)

The P800 Torch 2.9 SDK image with
``third_party/xpu/patches/pytorch-2.9-inductor-bias-gelu.patch`` installs the
F06 precision policy at the native Inductor compiler entry and in the pre-grad
pass for larger graphs.
For FP16 and BF16 inference, inputs to a fused arithmetic region are promoted to FP32.
MatMul, bias addition, activations, row reductions and normalization use FP32
intermediates. Each observable result is converted to its model-declared dtype
at the boundary. A returned intermediate and an internal use of that same value
see the declared output and the FP32 intermediate respectively.
Explicit dtype conversions in the model remain precision boundaries.

.. code-block:: python

   compiled = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
   with torch.no_grad():
       result = compiled(matrix, bias)

On the P800 Torch 2.9 image, apply
``third_party/xpu/patches/pytorch-2.9-inductor-valid-autotune-timings.patch``
with the matching FlagTree ``triton.testing.do_bench``. Both the pilot and the
individual device-event samples are checked against a synchronized host timing
window. Invalid readings trigger fresh work with new events. Candidates that
exhaust timing retries are excluded from ranking and the persistent timing
cache. The patched cache uses a separate versioned file. On a cold GEMM
benchmark, if the two leading candidates are within 5 percent, Inductor
benchmarks viable candidates in two further orders and caches the median
of the three measurements. If different Triton tile shapes in an in-process
GEMM still measure within 5 percent, it remeasures every surviving candidate
with groups of 16 real calls and uses the median per-call cost. A complete
timing-cache hit reuses these costs.

For the dense single-result F06 templates, the public backend prepares one FP32
body for FP32, FP16 and BF16 calls with the same promoted tensor layouts.
Its bounded process-local cache keys the graph, device, layout, compiler and
math settings, calibration tolerances, split options and caller environment.
The cache holds code and selection evidence. Model inputs and affine parameters
are read live. For the supported static shapes, FP16/BF16 specializations
load the model dtype, compute FP32 intermediates, and narrow on the final store.
The mixed GEMM candidate masks partial output tiles and the last K tile.
Pointwise candidates can use a partial final block. Row reductions use the
actual column count for normalization and mask or pad inactive lanes during
the reduction. The P800 selector omits the SDNN sum candidate on an odd
number of rows after a kernel fault reproduced with 65 rows. Eligible cluster
and native candidates still undergo numerical validation and end-to-end
timing to select a route for each shape. For Add + Softmax, partial row
batches can use the FP32 or FP16 vector-row path as well as BF16. Short rows
with partial final vectors can stage the broadcast bias once per cluster and
retain the logits in private memory through the max and sum reductions.
The public entry and the patched native Inductor entry both validate these
device functions against the canonical FP32 graph on real inputs. Native
Inductor compiles that graph for its fallback and prepares the specialization
on the first ordinary invocation. Each
ordinary call remains guarded by Dynamo. Explicit model precision boundaries,
training, Inductor CUDA-graph modes and unsupported graphs keep the ordinary compiler
path.

Calibration checks the FP32 body and its rounded outputs before accepting a
route. Each low-precision graph also checks its current inputs and tolerance.
The audit records the canonical FP32 program key, its selected implementation,
the boundary candidates and the physical launch evidence. In a mixed-input
record, ``compute_route`` names the FP32 control. The selected boundary
candidate's launch evidence identifies the resources that executed the call.
When the canonical
body fails a later validation, its shared handle switches to FP32 eager.
The specialized low-precision kernels undergo independent validation.
Explicit conversions and observable intermediate results
retain their graph semantics. Direct eager on the original model provides
the end-to-end latency baseline for each dtype.

Calibration requires no-grad inference and pure supported FX operators.
Lifted module Parameters are detached for calibration under no-grad; replay
continues to read the live arguments. Other autograd inputs are rejected. Native Dynamo guards
bound each prepared callable. Unsupported graphs are rejected rather than
assigned unverified resources. CPU and non-P800 backends retain native Inductor.
``calibrate_resources`` is the explicit offline preparation API for a fixed FX
graph and exact input metadata; its returned callable does not itself add
Dynamo guards. Use the backend factory for guarded application execution.

Whole-graph SDNN and cluster candidates are admitted according to actual SDK
support. For matrix graphs, native dot/MMA remains on SDNN. Matrix graphs also
consider cutting immediately after MatMul or after Bias, followed by a cluster
suffix. Pure normalization/reduction candidates keep their complete graph on
one resource. At every input size, single-launch and multi-launch resource plans
compete on measured end-to-end latency. Passing ``single_kernel=True`` explicitly
limits resource candidates to a verified single launch. An omitted value,
``None``, or ``False`` allows multiple launches.

The shared body uses the ordinary FP32 template candidates, including native
MatMul, SDNN, cluster and multi-region plans. Low-precision row and pointwise
candidates convert in the kernel. For K=64 MatMul, calibration also tries
a two-kernel schedule with an FP32 intermediate. When both schedules pass,
the split must win by at least three percent in both paired timing rounds
to replace the single launch. The formal selector also measures the validated
shared FP32 body with real SDK input and output conversions when multiple
launches are allowed. This route can replace the boundary-fused schedule if it
wins both end-to-end rounds by three percent. When an arm is over four times
the fastest in both initial rounds, the selector reruns the remaining competitive
arms head-to-head if at least two remain. The audit retains both initial and
isolated medians. A validated SDK conversion route in that final comparison
remains the incumbent until a challenger wins both rounds by three percent.
Packed FP16/BF16 activation and
matrix epilogues accumulate in FP32 and narrow on the output store.
For the 2048 by 2048 BF16 MatMul + Bias, the selector also measures the
grouped FP32 product followed by the SDK cast on its contiguous output.
The input conversions and bias remain in the product kernel. The audit
records the external cast separately from the verified SDNN kernel.
Wide BF16 matrix products with complete column tiles and a partial row group
also compare the SDK output conversion. Large, contiguous whole 512-element
tiles can overlap the next read with final rounding in the cluster epilogue.
Mixed-precision LayerNorm narrows directly from the normalized FP32 lanes
before the final local-to-global copy. For eligible partial batches of 2048
columns, the four 512-column moment sums are accumulated in column order.
For short odd-width BF16 SiLU and tanh GELU rows, the mixed boundary can stage
a broadcast bias once per cluster, compute in FP32, and copy only the valid
output columns.
For the non-affine 2048 by 2048 BF16 template, each core retains one packed
row in local memory. It widens operands in logical column order for FP32
moments and combines the final 512 columns, middle 768 columns and first
768 columns in that order. After computing the row statistics, it repeats
the broadcast addition in FP32 registers and packs the normalized lanes
on the final store.
The FP32 Mul+Sum cluster kernel follows the SDK reduction
order: the last 32 elements seed each 512-element chunk, then the remaining
vectors accumulate in address order. The 1024-column schedule retains the
vendor's cross-core compensation for non-finite inputs.

For the P800 Add + Softmax template, the matrix is dense and the reduction
axis is the last dimension. The broadcast bias may have shape ``(N,)`` or
``(1, N)``. The descriptor launcher checks the actual input layout for every
prepared shape. A masked row kernel covers incomplete tiles and a streaming
kernel uses bounded local memory for wider rows. A supported BF16 vector
schedule widens the input lanes, computes row statistics in FP32, and packs
the final probabilities with round-to-nearest-even. A cached launch reads
new input and bias addresses, allocates a fresh output, and uses the active
stream on each invocation. The patched Inductor entry can select the same
prepared row kernel for FP32, FP16 and BF16. Function and tensor-method forms
accept either addition order. A statically empty result preserves the model's
output shape and dtype with zero device launches. The F06 backend times the
full call when choosing among nonempty implementations.

On P800, the FP32 Add + SiLU cluster kernel for 512 by 1024 partitions
contiguous input slices across cores and shares the bias within a cluster.
For 2048 by 2048 it prefetches the next slice into a second local buffer.
Both compute the exponential and division after adding the bias. For a
2048 by 64 times 64 by 2048 FP32 product, the native SDNN candidate
groups eight output row tiles and writes the bias in the matrix epilogue.
The same grouped product can feed SiLU and tanh-GELU tails. For mixed
inputs the grouped tile widens its operands and bias before computing.
The cluster activation kernels can unpack FP16 or BF16 into FP32 lanes
and narrow the final result as they write it. The packed BF16 store
rounds the original FP32 bits with the P800 SDK tie-to-even rule,
including its handling of NaN payloads.
Calibrated dense launchers can bind the static SDK argument layout on
CPython 3.10 and Torch 2.9. A KL3 launcher can publish its integer-scalar
and pointer ABI. A compatible dense entry binds the fixed grid, scalar words
and element counts, then fills tensor addresses on the call stack and reads
the current stream for every submission. Repeated tensor arguments retain
their own ABI slots. Hooked launches use the ordinary launcher.

For a static, positional Dynamo frame that calls the selected graph and
returns its first output, the guarded entry can issue the call directly.
It still runs Dynamo's full guard tree and the module, hook and mode checks.
Frames that read live module parameters use the normal bytecode replay.

The Inductor GEMM baseline for a shared graph uses the SDK GEMM backend;
the explicitly measured native MatMul and SDNN candidates supply fused
alternatives. Internal runtime graph replay is disabled on ordinary calls.

Calibration uses ordinary, repeated, changed-input, special-value and immutable
input/parameter checks. Tolerances must be explicitly supplied and are never
relaxed. Calibration injects a single NaN, positive infinity and negative
infinity separately into each nonempty floating operand, in addition to the
mixed probe. GPU graph capture additionally poisons outputs and changes inputs
nonuniformly; a stale or empty graph cannot pass using Softmax translation
invariance. An invalid candidate cannot poison subsequent input values.
Only verified bound resource binaries qualify as SDNN/cluster candidates;
external calls are not attributed to either resource. A single-kernel candidate
must have exactly one launch in its generated wrapper.

Resource candidates on Torch 2.9 use an instance-local inference entry that
omits the unused ``record_function`` scope. It retains the native current
callable (including alignment, checks, allocations and launchers) and native
metrics/cache finalization, including on failure. Active profilers, debug
execution recording, grad-enabled calls, modified native entries and unknown
Torch versions retain the original entry. No global classes or baselines are
patched.

For a static FP32 wrapper containing one verified SDNN allocation and launch,
the inference entry can reuse Inductor's selected launcher. Its guards check
the layout, alignment, current device and launcher identity. It allocates a
fresh output on each call and submits on the current stream. Other wrapper
layouts retain the native callable.

Selection first requires ordinary-call end-to-end acceptance. Two independent
rounds of 128 randomized forward/reverse samples are collected for each
validated callable. Native Inductor is the fallback only if it beats eager by
at least 3% in both rounds; otherwise eager is the fallback. Resource plans
must beat this stable fallback by at least 3% in both rounds. This is a
performance noise margin, not a numerical tolerance change. Missing, short,
nonfinite or old blocked-timing records cannot pass acceptance. Isolated
offline records admit numerically safe candidates only; newly compiled
callables always receive fresh paired end-to-end acceptance. Final reporting
samples are independent and never fed back into selection.

Among admitted plans, selection minimizes the mean of the two round medians.
On the guarded P800 module entry, each sample calls the same warmed
``OptimizedModule`` with its original arguments and waits for device completion.
The timed interval includes Dynamo guards, output reconstruction and allocation.
The eager candidate uses this entry too. Each round first measures the public
routes in their original order, then collects a separate randomized comparison
that includes the original eager module. Among otherwise eligible routes, those
beating direct eager by 3% in both rounds take priority. When none does, the
fastest eligible public route is used. The audit records the direct control
medians and qualifying routes under ``e2e_acceptance.direct_eager``.
Candidate switching is thread-local and occurs before
the timer starts. The audit protocol for these rounds is ``public_module_ab_ba_v1``.
Standalone FX calibration measures its own callable entry under ``randomized_ab_ba_v1``.
Device graph replay remains available in ``calibration_gpu_us`` for analysis.

On CPython 3.10 and Torch 2.9 the static P800 module entry can bind a native
guarded replay. An ordinary hit checks module state, hooks, execution modes
and Dynamo's full guard tree against the current inputs before invoking the
captured bytecode. A changed module or class version triggers revalidation.
The Python entry handles unsupported frames and extension build failures.

For K=64 MatMul + Bias, the selected FP32 body includes its bias and optional
activation. The mixed-input SDNN candidate reads both matrices and bias in
their stored dtype, then accumulates and adds bias in FP32. It either writes
the final result in the same launch or passes FP32 values to a cluster
epilogue. Calibration checks final rounding near both even and odd midpoints.

FP32 SDNN Mul+Sum can reuse the executable prepared by its first JIT launch.
The plan checks the P800 kernel signature and resource before binding the
launcher. Each call allocates a fresh output and reads the current stream.
Instrumentation and incompatible inputs use the JIT path. This optimization
also applies during candidate measurement and does not bypass numerical
admission.

For short FP16 bias + GELU(tanh) plans with at most 65,536 output elements,
the flat kernel tuner times each validated tile using five device-event samples
of a replay containing 32 launches. The tile cache records the clock used.
If device graph timing is unavailable, the current calibration can use the
launch loop and a later cold process can retry device timing.

The resource proof checks the emitted binary's ``is_sdnn`` flag against the
planned resource. For ``resource_kind="auto"`` it also requires matching
``effective_resource_kind``. Legacy P800 binaries exposing only ``is_sdnn``
retain their existing verification path.

Offline records must match graph, shape, dtype, layout, attributes,
precision flags, implementation version and caller-supplied environment key.
Admitted offline candidates are revalidated and recalibrated after recompilation.
All admitted candidates share a call count per graph and use three independent
captures. Calibration interleaves candidates in forward/reverse order after
warming every capture. The benchmark retains these exact prepared callables
and captures for independent final samples. A fallback and its baseline share
the same capture objects; separate captures of the same callable can have
different latency on this SDK. Capture dispersion is retained in the records.
This prevents a new native tuning choice or recapture from silently invalidating
the selection timings, but does not guarantee that independent samples will
preserve the ranking of nearly equal candidates. Audit records distinguish
resource fusion, resource splitting, eager fallback and Inductor fallback.

The isolated P800 test harness retains failed native compiler processes and
uses separate calibration and final AB/BA measurement samples. Native compiler
crashes cannot be caught by Python exceptions in the online convenience API;
use isolated offline calibration when exploring unsupported SDK configurations.

Comparing P800 performance changes
----------------------------------

For each F06 optimization, run direct vendor eager, the public F06 module and
its selected plan on the same inputs in the same warmed process. Record the
ordered executable images passed to the KL3 runtime at launch, their entry
offsets, execution resources and cluster/core counts. Retain the transferred
bytes and their SHA-256 digests. Disassemble both sides with the matching
KL3 target and check the instruction words against the original bytes.
Check that the public entry launches the plan selected in that process.

Time ordinary calls with tracing disabled. Report paired host submission,
synchronization wait and end-to-end times, and time device execution separately.
If the public and eager launch sequences match, use a fixed-plan host comparison
to locate the overhead. If they differ, inspect the relevant instructions,
kernel boundaries, memory traffic and launch geometry. Static instruction
counts alone do not measure time or prove that a register spilled.

Experimental P800 SDNN special-value lowering
---------------------------------------------

The F06 SDNN experiment is opt-in. The normal resource backend does not import
the experimental module. To reproduce one already-admitted candidate in an
isolated P800 process, use the explicitly imported preparation entry::

    import torch
    from triton.experimental.graph_fusion.p800_special_values import prepare_special_candidate

    # gm is the fixed FX graph and xs its device-resident inputs.
    with torch.no_grad():
        fn, proof = prepare_special_candidate(gm, xs, tolerance=original_tolerance)

Prepare in the original benchmark environment: TF32 and FP16/BF16 reduced
precision reductions disabled, XMLIR_MATMUL_FAST_MODE=0 and
TRITONXPU_DTYPE_CONVERT=1. The entry rejects different math modes rather than
changing process-wide flags. Use inference inputs and frozen model parameters.

This checks the F06 shape, dtype, finite values and special probes before
returning a native Inductor callable. Coordinate-descent compilation retains
the experimental identity even when it happens later. There is no extra
validation or wrapper on warm ordinary calls. Compare the prepared candidate
with eager and max-autotune-no-cudagraphs before using it. The private raw
lowering scope is for diagnostic comparisons and bypasses this admission.

For the recognized tanh-GELU epilogue, the experiment bounds the tanh argument
and classifies the original activation input once. For SiLU it bounds the
sigmoid argument, preserves the original input in the final multiply and
checks the native operand order and output alias. This allows positive infinity
to reach the final output. For SDNN Mul+Sum it computes the product once. It inspects operand bits to
return a quiet NaN for a NaN operand or infinity times signed zero, and signed
infinity for other infinite products. Two finite operands keep the native
product. The original sum and its output rounding are unchanged. BF16 still follows the existing FP32
computation and rounding boundaries. FP32/BF16 reduction mismatches involving
extreme tiny operands, and signed-zero differences for some activations, are
known limitations. The checked entry rejects FP32/BF16 SDNN Mul+Sum, and
only admits the two tested small FP16 reduction shapes. These rewrites do not establish universal IEEE conformance
or change any default placement or selection policy.

The private ``experimental_scope("parallel")`` diagnostic mode partitions the
single-product repair for native 1x128 and 1x256 SRAM tiles across eight SDNN
control cores. Each core owns complete 64-byte cache lines. Arithmetic, BF16
rounding, reduction order, and producer/consumer event dependencies are
preserved. An input readiness fence and two collective barriers coordinate
the disjoint writes. Unsupported layouts are rejected by this experiment.
This mode does not widen the checked entry's precision admission or change
default kernel selection. Diagnostic timings for a candidate that fails any
precision or special-value check do not qualify it for deployment.
