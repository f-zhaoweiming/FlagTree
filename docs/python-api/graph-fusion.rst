Graph Fusion
============

Scope
-----

The graph-fusion package provides structural graph capture, pattern matching,
stage inlining, and guarded static inference. Eligible small NVIDIA matrix
compositions enable native Inductor automatic GEMM/epilogue fusion. This package
does not supply an arithmetic kernel or a combination-specific Triton template.

Public API
----------

By default, ``compile(graph_module, example_inputs=None, **kwargs)`` forwards the exact graph
object to the existing native ``torch.compile``/Inductor wrapper. It accepts
``example_inputs`` for cross-call compatibility but does not execute or inspect
them. ``fullgraph``, ``dynamic``, ``mode``, ``options``, and ``disable`` are
forwarded unchanged; removed kernel/tuning/implementation keywords raise
``TypeError``.

The structural ``FusionTemplate``/``Registry``/``composition_candidates`` API
and ``cross_call_candidate`` stage inlining remain. ``default_registry`` is a
zero-argument factory returning a ``Registry``. Patterns never execute tensor
inputs. Stage inlining keeps live parameter references and rejects mutation and
unsupported multi-output stages.

Guarded static inference
------------------------

``static_inference=True`` opts into an eagerly prepared, fixed-shape inference
plan. Supply a tuple of example tensors. A single supported operator-only FX
template, flat tensor arguments, tensor attributes, and one tensor output are
required. Gradients, mutation and module calls are unsupported. Dynamic shapes
and custom public mode/options are rejected. The small-matrix policy below
applies scoped internal compiler options; other graphs use default options.
Inputs and tensor attributes that require gradients are supported inside
``torch.no_grad()`` or inference mode, without changing their flags. Enabling
gradient recording later routes such calls to the original callable, preserving
autograd. ``eval()`` alone does not disable gradient recording. Optional graph
parameters are supported when every positional argument is explicitly supplied;
omitted arguments and keyword calls retain the original callable's behavior.

.. code-block:: python

   plan = graph_fusion.compile(
       gm, (x, weight), fullgraph=True,
       static_inference=True, avoid_bias_materialization=True,
   )
   result = plan(x, weight)

The plan lifts live tensor attributes into explicit arguments. Steady-state calls validate
tensor metadata, aliases, precision settings, transform modes, hooks and the
forward binding before calling the compiled graph directly. Attribute values
are fetched on every call, so replacing a bias with a tensor of the same
metadata does not leave a stale value. Guard misses return to the original
``torch.compile`` callable. Unsupported initial contracts also keep that native
path; compiler errors propagate. This path depends on private PyTorch APIs and
requires version-specific validation.

Small matrix single-kernel execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``static_inference=True``, NVIDIA SM80+ dense contiguous matrix multiplication
plus bias, optionally followed by ReLU, exact/tanh GELU or SiLU, enables compiler
fusion when ``1 <= M <= 64``, ``1 <= N <= 1024`` and ``1 <= K <= 128``.
Supported dtypes are FP32, FP16 and BF16. Bias may be a scalar, vector, row,
column or matrix, supplied as an argument or a live tensor attribute. Matrix
operands may also be live attributes. TF32 must be disabled; FP16/BF16 additionally
require reduced-precision reduction to be disabled. Other cases continue through
the existing native plan.

P800 (the XPU driver with architecture 3 and warp size 1) also enters native
Inductor GEMM-template selection for these bounded graphs. During that
compilation only, a context-local, device-specific override replaces Inductor's
NVIDIA SM-count eligibility heuristic. Native template selection, configurations,
arithmetic remain compiler-generated. The default Inductor baseline
does not use this override. Graph-cache reuse is disabled for this scoped P800
compilation so a previously generated external ``addmm`` wrapper cannot bypass
the policy. The P800 compiler repairs below apply only to this opt-in static
path and explicit P800 resource compilation.

Inductor selects its built-in GEMM template and tile by autotuning, then
generates the epilogue from the unchanged FX/ATen operations. The scoped policy
uses ``max_autotune=True``, ``max_autotune_gemm_backends="TRITON"``,
``epilogue_fusion=True`` and ``benchmark_epilogue_fusion=False``. CUDA Graph
replay remains disabled. This policy increases compilation cost; it does not
change the report's default Inductor baseline.

``pattern_matcher=False`` prevents rewriting materialized ``mm + add`` into
``addmm``, which changes low-precision rounding. ``emulate_precision_casts=True``
retains the output-dtype casts after both matmul and addition while generating
one kernel. On P800 it is enabled during native ``make_fx`` tracing as well as
lowering, so materialized low-precision pointwise boundaries are annotated
before decomposition; internal FP32 LayerNorm arithmetic is not rounded. There is no hand-authored fused matrix kernel or launch tile in
this package. ``use_static_cuda_launcher=False`` keeps Inductor's Triton
launcher, including hooks added or removed after compilation. Current streams,
launch hooks, normal guards, gradient fallback
and live attributes are preserved. Floating-point accumulation order may
differ from native library GEMM; bitwise equivalence is not promised.

``plan_info["kind"] == "inductor_small_matmul_single_kernel"`` identifies a
generated wrapper with one Triton declaration and one launch, and no external
GEMM call. The wrapper is audited after compilation; an unknown or multi-kernel
wrapper keeps the ordinary native-plan label and a skip reason. Metadata records
the compiler options and generated kernel name. ``generated_sources`` contains
Inductor's Python wrapper and generated Triton code. Device tests also profile
actual kernel counts. Performance includes the common guards and final output
allocation.

The P800 path additionally verifies the binary bound by the generated wrapper
and records its actual SDNN or cluster resource in ``plan_info["p800_binary"]``.
An external/multiple-kernel wrapper or unverified binary raises a compilation
error instead of silently returning to external ``addmm`` for an admitted
small-matrix candidate. On Torch 2.5, execution retains its native launcher
entry point. The newer NVIDIA launcher adaptation below is not applied there.

After the first normal call finishes lazy preparation, the small-matrix plan
reuses Inductor's selected launcher through a private copy of its generated
Python wrapper. No arithmetic, tile, grid or device code is rewritten. Native
input-alignment repair, output allocation, device guards, current streams and
Triton launch hooks remain in place. Misaligned external CUDA storage can
require additional alignment-copy kernels; the single-compute-kernel claim
does not exclude these copies.

The steady-state path avoids repeated generic autotuner dispatch and compiler
bookkeeping. Profiling, graph/debug recording, pending runtime metrics or cache
bundling, and a changed native callable/launcher use the original Inductor
entry point. Unknown private-runtime layouts and kernels needing scratch or
launch dumping retain the native path. ``plan_info["prepared_launch"]`` reports
whether preparation succeeded. Generated size/stride assertions are disabled
for this path because every public call already performs the corresponding
TensorGuards checks; alignment checks are retained.

Large NVIDIA matrix activation epilogues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dense NVIDIA SM80+ MatMul+Bias+ReLU, exact/tanh GELU or SiLU graphs outside the
small-matrix bounds use a scoped autotuning policy. The dtype, contiguous-layout,
bias-broadcast and disabled-TF32/reduced-reduction requirements above still apply.
``max_autotune=True`` and ``coordinate_descent_tuning=True`` search native GEMM
and pointwise configurations. Both ``ATEN`` and ``TRITON`` GEMM candidates remain
available; ``benchmark_epilogue_fusion=True`` compares fused and split execution.
This can increase compilation time and does not promise a single kernel or a
global optimum. Native CUDA Graphs remain disabled.

``pattern_matcher=False`` and ``emulate_precision_casts=True`` retain the original
MatMul and Add dtype boundaries during tracing and lowering. The policy does not
change precision tolerances or replace exact GELU with an approximation. The plan
keeps its ordinary guards, output allocation and live attributes, and records
the options in ``plan_info["large_matmul_compiler_options"]``. Small matrices,
bias-only graphs, non-matrix templates and P800 retain their existing policies.

With ``static_inference=True, avoid_bias_materialization=True``, an exact
two-dimensional MatMul+Bias pattern with contiguous, same-dtype CUDA tensors
can use a native operator execution plan. It invokes the original ``matmul``
or ``mm``, then ``add(..., out=new_matmul_output)`` while preserving add operand
order. Only the freshly allocated output is reused; inputs and bias remain
unchanged and subsequent calls return fresh outputs. This avoids full broadcast
bias materialization in the observed P800 addmm path and a separate add output
allocation. It does not introduce a compute kernel, fused addmm, activation
approximation or lower-precision arithmetic. Ineligible patterns retain native
Inductor compilation.

``plan_info``, ``generated_sources``, ``fast_calls``, ``fallback_calls`` and
``last_fallback_reason`` expose the selected path and guard behavior. For the
native MatMul+Bias execution plan, source descriptions are operator sequences,
not generated kernel code. Benchmarking must include guards and allocation.

Low-precision materialization boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static compiler plans preserve eager-visible FP16/BF16 output rounding across
operator boundaries on every backend. The scoped native
``emulate_precision_casts`` option covers both ``make_fx`` tracing and
Inductor lowering, so vector reductions and normalization receive the same
semantic protection as matrix plans. For example, Mul+Sum rounds the product
before reduction, and Add+LayerNorm rounds the sum before normalization.
Internal decomposition arithmetic is not marked as a separate eager output.
These conversions can remain in registers within an automatically fused
kernel; no intermediate global-memory tensor or handwritten kernel is added.
The native option participates in Inductor's cache identity and is restored
after compilation, including failures. ``plan_info["precision_policy"]``
records this contract. It does not promise bitwise equivalence of reductions
with different accumulation orders.

NVIDIA GELU(tanh) expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static NVIDIA plans for ``MatMul+Bias+GELU(tanh)`` and ``Add+GELU(tanh)``
use Inductor's native ``libdevice.tanh`` expression inside the fused kernel.
FP16/BF16 computations retain Inductor's precision casts and eager-visible
rounding boundaries. The default path uses the native compiler cache policy.

The branchless tanh helper is retained for experiment reproduction, but is no
longer connected to compiler plans. Runtime tanh selection, separate device
linking and device LTO remain standalone experiments in ``test-results``.

P800 compiler repairs
~~~~~~~~~~~~~~~~~~~~~

On the positively identified P800 backend, admitted small matrices request a
native SDNN kernel and other accepted static templates request native cluster
code generation. The bound binary's actual resource is checked; a requested
resource alone is not execution evidence. ``plan_info["p800_resource"]`` retains
wrapper and binary provenance. CPU and NVIDIA compilation do not use these
P800 passes.

Scalar-output cluster reductions (Mul+Sum and keepdim) retain Inductor's
two-dimensional layout. Softmax and LayerNorm keep their native layout. The deployed
P800 rank-one reduction lowering emits scalar writeback from every core,
whereas its two-dimensional lowering predicates the owning core. The repair
changes the native layout decision without supplying reduction arithmetic,
tiles or a Triton fusion template.

A late TTIR pass preserves FP32-to-BF16 round-to-nearest-even boundaries,
including infinities and NaN classification. SDNN additionally receives missing
tanh/sigmoid lookup mappings, an FP32 approximation of the ``erf`` primitive
used by exact-form GELU, and row-wise legalization of native tensor division.
The GELU graph remains ``approximate="none"``; it is not replaced by the tanh
GELU formula. These backend approximations are checked against the original
report tolerances, rather than claimed to be bitwise identical to eager.

Compiler changes are serialized, context-scoped and restored on exceptions.
Generated-module policy fingerprints and isolated Triton caches prevent reuse
of an unpatched kernel. Resource decorators reapply the passes during native
lazy precompilation. All of this occurs before steady-state timing; the public
callable still performs its normal guards and output allocation.

Data movement
-------------

Retained compiler work includes intermediate-memory elimination, tile-dataflow
graph/resource analysis and scheduling, graph fusion tests, and XPU launch
configuration/resource-limit metadata. These features preserve original
operator targets, edges, options, aliases, and special-value behavior.

No universal performance or bitwise-eager guarantee is made. The one-kernel
claim applies only to admitted small-matrix plans.
