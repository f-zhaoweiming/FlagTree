P800 resource and partition autotuning
=====================================

``graph_fusion.autotune_resource_backend`` is an opt-in, inference-only
``torch.compile`` backend. It reuses native Inductor decompositions, templates,
heuristics, configuration search, generated arithmetic and launchers. It does
not provide arithmetic kernels or manually selected tiles.

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

The current calibration contract requires tensor arguments and attributes with
``requires_grad=False`` and pure supported FX operators. Native Dynamo guards
bound each prepared callable. Unsupported graphs are rejected rather than
assigned unverified resources. CPU and non-P800 backends retain native Inductor.
``calibrate_resources`` is the explicit offline preparation API for a fixed FX
graph and exact input metadata; its returned callable does not itself add
Dynamo guards. Use the backend factory for guarded application execution.

Whole-graph SDNN and cluster candidates are admitted according to actual SDK
support. For matrix graphs, native dot/MMA remains on SDNN. Large graphs also
consider cutting immediately after MatMul or after Bias, followed by a cluster
suffix. Pure normalization/reduction candidates keep their complete graph on
one resource. Small matrices (M <= 64, N <= 1024, K <= 128) require one actual
native launch; nonmatrix graphs with rows <= 64 and columns <= 1024 use the same
single-kernel constraint. ``single_kernel`` can explicitly override this policy.
This is an admission bound, not a measured universal hardware crossover.

The opt-in ``split_reductions=True`` experiment additionally considers a
two-region Mul -> Sum partition for large graphs. Both the producer and the
consumer can use SDNN or cluster, yielding four resource combinations. The
complete product is materialized in its original dtype before reduction;
Inductor selects each region's configurations without forced column slicing.
Defaults and the small-matrix single-kernel constraint remain unchanged.
Historical operator and transfer ratios can prioritize offline experiments,
but cannot replace full-plan precision, special-value and timing acceptance.
In particular, a standalone SDNN reduction advantage does not establish a
speedup for a fused Mul+Sum kernel, and repeated-FMA measurements are not
standalone multiplication measurements.

Eager and native ``max-autotune-no-cudagraphs`` Inductor are measured fallback
candidates. The Inductor mode is obtained from the installed Torch mode table.
Both native Inductor and resource candidates use the same scoped XPU target /
exp / rsqrt ABI bridge and disable the NVIDIA-only static launcher and shared
memory pruning. The native baseline admits native GEMM candidates on the
identified P800 despite its CUDA-alias SM-count heuristic. It does not receive
FUSION resource assignments or activation/rounding legalization passes. Resource
candidates retain explicit low-precision boundaries and benchmark complete
Inductor epilogues. Internal runtime graph replay is disabled on all routes.

Calibration uses ordinary, repeated, changed-input, special-value and immutable
input/parameter checks. Tolerances must be explicitly supplied and are never
relaxed. GPU graph capture additionally poisons outputs and changes inputs
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

Among these accepted plans and the fallback, selection minimizes measured
complete-plan GPU graph replay time. This already
includes inter-kernel dependencies and intermediate writes/reads; no separate
copy or memory penalty is added. Compilation, calibration, validation, warmup
and capture precede timing. End-to-end ordinary-call latency is measured
separately on prepared callables with device-resident inputs, including Python
dispatch, output allocation and completion synchronization. Input construction,
host/device transfers, compilation and the outer Dynamo guards are outside
this steady-state measurement. Offline records must match graph, shape, dtype, layout, attributes,
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

Experimental P800 SDNN special-value lowering
---------------------------------------------

The FUSION SDNN experiment is opt-in. The normal resource backend does not import
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

This checks the FUSION shape, dtype, finite values and special probes before
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
