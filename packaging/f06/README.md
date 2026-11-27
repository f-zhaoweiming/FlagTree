# F06 P800 inference package

This package contains the existing F06 fusion kernels, precision checks,
resource selection and warm-call cache, plus opt-in whole-graph Inductor scheduling.
It requires the matching P800 Torch/FlagTree/TLE environment; vendor drivers,
LLVM binaries and the vendor SDK are not bundled.

```bash
python -m pip install --no-deps f06_p800-0.3.11+v51e.inductor12-py3-none-any.whl
python -m f06
```

```python
import torch
import f06

class AddSoftmax(torch.nn.Module):
    def forward(self, x, bias):
        return torch.softmax(x + bias, dim=-1)

with torch.no_grad():
    x = torch.randn(64, 1024, device="cuda")
    bias = torch.randn(1024, device="cuda")
    optimized = f06.compile(AddSoftmax().eval(), (x, bias))
    output = optimized(x, bias)  # First call validates and calibrates.
    torch.cuda.synchronize()
    print(optimized._f06_audit)
```

Call `f06.activate()` before importing a different F06 implementation or using
existing Torch integration hooks. Activation selects bundled sources under the
canonical `triton.experimental.graph_fusion` import path for this process.
It does not edit installed Triton or Torch files. Mixing implementations in one
process raises an error. All generated imports resolve to the same package.

FP16/BF16 templates use the established FP32 internal arithmetic contract and
retain the model output dtype. Timing comparisons use same-dtype native eager.
This numerical contract must be evaluated against real model task metrics.

For a whole inference model, use the whole-graph Inductor entry:

```python
model = model.eval().requires_grad_(False)
optimized = f06.compile_model(model)
with torch.no_grad():
    output = optimized(**inputs)
print(optimized._f06_audit)
```

When an application has a tokenizer or image processor, provide its tensor
boundary explicitly. The tensor module must include pooling, projection,
normalization, output ordering and every other tensor operation needed for its
public outputs. Compile that module once; discovering child modules through
`__call__` hooks misses parents invoked directly through `.forward()`.

```python
def prepare(raw_inputs):
    tensors = processor(raw_inputs)  # Runs on every call, including timing.
    return (), tensors              # Positional arguments, keyword arguments.

optimized = f06.compile_pipeline(complete_tensor_model.eval(), prepare=prepare)
with torch.no_grad():
    output = optimized(raw_inputs)
```

The default `model_candidate_routes` search captures the supplied model with
its original operator and tensor dtypes (`capture_source='original_dtype'`).
It does not cast the input model to FP32 or force low-precision Linear outputs
back to FP32. Preserve a separate, unrounded FP32 model and original inputs for
the complete-output validator; that reference is not used to construct the
candidate graph. Captured dtype-dependent branches and explicit conversions
remain in the graph.

The current domain compares native versus Inductor Linear lowering, generated
versus native pointwise implementations, reviewed FP32-internal fusion regions,
and eligible Cluster/SDNN schedules. A matrix or low-precision arithmetic node
does not become FP32 arithmetic merely because its surrounding region is fused.
Unknown precision boundaries reject compilation explicitly; an uncompiled FX
fallback is not reported as a compiled candidate. This is a bounded domain, not
an exhaustive mixed-precision search. If no candidate passes the original-FP32
output contract, report no admissible route rather than replacing the graph
with the FP32 reference or the uncaptured eager model.

Local candidates also reuse the original nine-family template registry on the
captured original-dtype FX graph. Each detected family is proposed separately;
an all-family plan and a plan combined with detected QuickGELU regions are
included when applicable, with 512/1024-byte buffer alternatives. This is a
bounded set of plans, not exhaustive subset enumeration. Opaque vendor Linear
calls are not split into artificial matmul/bias nodes just to force a match.

An intermediate used outside a region becomes an additional region output.
For example, Add + LayerNorm can return both the rounded residual sum and the
normalized result. Contiguous pure matches preserve the surrounding operation
order; intervening computation or mutation is not moved across a region.
Native matrix and ReLU boundaries and graph-visible dtype rounding are retained.
Generated region wrappers receive live tensor arguments and retain context,
layout and alias guards; no inputs, outputs or preprocessing values are cached.
Recognized regions without a supported runtime specialization are not presented
as compiled candidates. Every complete candidate still goes through the same
full-output validator and end-to-end selector, with no whole-model eager entry.

Optional `source_tensors`, `source_prepare`, and `precision_score` arguments
enable bounded local precision refinement. Sources must be frozen, original
FP32 tensors whose cast exactly reproduces the captured state. Their model
forward is never used to construct a candidate. Search starts with zero
promoted regions and checks bitwise agreement with the original application,
then expands precision in discovered branches and refines the selected regions.
Residual blocks stay together, including their Q/K/V projections. Adjacent
promoted regions retain FP32 intermediates; native regions restore their
observed input dtype at the boundary. Public outputs keep their original dtype.
Tied state, unobserved regions, input mutations, and floating output aliases
currently reject local refinement rather than silently changing semantics.
Native screening proposes regions only: all compiled variants still require
complete-output validation and whole-model timing. This finite search neither
proves that a failed native profile cannot pass after compilation nor covers
every possible subregion partition.

For an explicitly requested legacy experiment, to capture an unrounded FP32 graph and keep a lower-precision public output,
retain the original FP32 model and processor inputs before creating the eager
comparison model. Pass `source_precision="float32"` and `output_template` to
`compile_pipeline`; the template supplies output structure and dtypes only.
The entry rejects low-precision parameters, prepared floating inputs, and
active autocast. Widening a rounded checkpoint cannot recover its precision.
Final output conversions are graph nodes; P800 uses the vendor conversion
when narrowing, so live FP32 intermediates remain available to other outputs.
The old FP32-rooted automatic search is accessible only with
`capture_source='legacy_fp32'`; it is not the default candidate source.

The P800 FP32 schedule candidate retains short last-axis softmax (2–128 values)
and explicit noncontiguous copies as vendor graph nodes. Large pointwise work
uses a 32768-element tile, with separate 512- and 2048-byte buffer candidates
for FP32 pointwise kernels. Pointwise event autotuning is disabled after a
reproduced calibration wait; complete-graph host selection remains enabled.
These are measured alternatives: the original and
ordinary Inductor candidates stay eligible and whole-model validation and
end-to-end timing determine the actual route. It is not a promise that every
model or every input shape gets faster, and one complete graph can launch
multiple kernels.

`compile_pipeline` requires `fullgraph=True`. A graph break raises; it cannot
silently become a collection of independently compiled modules. Shape guards
may compile additional complete graphs. It does not cache preprocessing,
inputs or outputs. The application defines and validates the tensor boundary;
the API cannot infer preprocessing or public outputs from arbitrary adapters.
Keep the uncaptured native application callable for complete-model comparison.

Graph selection does not include an enclosing application's tokenization,
graph guards, pooling outside the compiled region, or final output conversion outside the compiled region.
The `original` graph candidate is a captured FX graph, not necessarily the
original uncaptured model call. Applications with multiple graph regions should
also use `f06.select_model_routes` on complete prepared call paths.

This explicit selection API accepts `ModelRoute(call, activate)` entries from
`triton.experimental.graph_fusion._whole_model_selection`, validation examples,
a complete-output validator, and a measurement callback returning raw seconds.
The callback must include the whole adapter, final conversion and completion
synchronization. Selection uses arithmetic means across alternating route
orders, rejects invalid timing, and returns the chosen callable directly.
Activation and compilation are outside the measured steady-state region.
Pass the routes returned by the candidate factory, without adding the uncaptured
eager model or setting it as `preferred`. Keep native eager and the original
FP32 model outside selection as accuracy and timing controls. Vendor operator
implementations remain available within generated graphs; this does not make
the complete eager application a candidate. A generated route may still be
slower than the external eager control, and that slowdown must be reported.
If every candidate fails validation, selection reports `no_valid_route` and
does not substitute eager.

The general selector still supports a supplied `preferred` route. Resource
alternatives also retain their generated parent as a stability comparison.
Such an alternative needs the requested mean gain in at least three of four
rounds, in both measurement orders, with no more than 5% spread between round
means for either route. An unstable comparison retains the valid comparison
route. These checks run only during calibration; no router is added to hot calls.
Numerical or output-contract rejection should raise `AssertionError` or
`ValueError` from the validator. Execution failures propagate immediately;
another route must not be tried after a device execution fault. Invalid timing
clears the winner and is never cached.

On P800, a prepared `compile_pipeline(..., resource_policy='auto')` also supplies
Cluster and SDNN resource alternatives to `select_model_routes`. These mirror
the prepared arithmetic graph; they do not replace vendor matrix kernels or
relax the original-FP32 output contract. SDNN alternatives currently cover large
contiguous FP16/FP32 pointwise kernels with explicit extent masks, including EW
arithmetic and DMA format conversion. Broadcasts, gathers and reductions retain
their existing implementation. The audit records each accepted/excluded SDNN
proposal and the actual compiled resource is checked before execution.

Each resource alternative is validated and timed as a complete application
call, without inner tensor-only resource selection. A change must stably improve
its valid parent route, even when native low-precision eager failed the accuracy
gate. Vendor-only fallback graphs are not relabelled as resource candidates.
Pass `expanded_routes={}` to the selector to retain the expanded named callables
for diagnostics; the returned winner still has no per-call selection wrapper.
Calling a pipeline directly retains its baseline schedule. Use the selector and
a validator built from original FP32 weights and inputs to enable resource
comparison safely.

`model_candidate_routes(..., pruning='calibrated')` optionally shortlists the
pointwise implementation policies before complete-model selection. The default
`pruning='none'` retains the full configured domain for comparison. Calibration
uses actual precision/Linear-rewritten graph regions, their boundary conversions,
shapes, dtypes and strides, with the same guarded compilation entry for every
policy. Regions cannot cross an outside dependency and rejoin through a residual
edge. Each resource retains its two cheapest policies, policies within 3% of the
fastest, and the generated implementation; unknown domains retain all policies.
Precision partitions and Linear schedules are preserved. Resource shortlists
are kept separately, with the required auto parents retained for the existing
complete-model gain check.

If all retained policies in a pruned precision/Linear/resource family fail the
complete original-FP32 output contract, the omitted policies in that family are
reopened. This validation and its compilation cost are part of cold search.
Every returned candidate still passes the normal complete-model selection gate;
there is no per-inference pruning or output cache. Local costs are a heuristic,
so an exhaustive comparison remains necessary to establish fast-path coverage
for a given workload. The `record['pruning']` audit contains regional costs,
resource shortlists, precision recovery and calibration times.

A native FP32 candidate should preserve the original FP32 model before applying
FP16 to the comparison model. Run the original adapter entirely in FP32 and use
`output_cast_like` at its public output boundary. Converting already rounded
FP16 weights back to FP32 does not recover the original checkpoint precision.
Validate the complete public output contract; passing pooled embeddings does
not justify dropping separately exposed token outputs. FP32 native selection
must be reported as a native route, not as a fusion speedup.
The original-dtype native route may also be supplied, but it must pass the same
complete-model FP32 reference validation before it is eligible. It retains its
original rounding semantics and must not be labeled as all-FP32 execution.

`compile_model` captures with Dynamo and compiles the complete graph through
Inductor. Vendor opaque Linear is exposed as `aten.linear` before AOTAutograd,
so Inductor can schedule its GEMM, bias and neighboring operators. Residual
LayerNorm retains live residual outputs, including broadcast inputs. QuickGELU
keeps `x * sigmoid(1.702 * x)`. FP16/BF16 intermediates inside these patterns use
FP32, while their externally visible outputs keep the original dtype. Inductor
owns allocation, code generation and the final fusion/splitting decisions.

Embedding and advanced indexing keep the original vendor operators because
this XPU toolchain cannot link the generated bounds assertion. Their bounds
checks are not disabled. Noncontiguous concatenation also retains the vendor
path after a transpose/concatenation code-generation error was reproduced. This fallback registration only affects the current process.
The capture substitution does not modify installed Torch/Triton or vendor files.

An optional whole-graph FP32 candidate additionally removes narrowing between
regions. The final narrowing uses the vendor conversion, keeping a live FP32
intermediate separate from a low-precision output when the XPU backend cannot
lower a mixed-output kernel. Its parameter conversions are reused only while parameter identity,
storage and PyTorch version counters match. Ordinary in-place parameter updates
invalidate them; input activations and outputs are never cached. Direct writes
that bypass PyTorch version tracking (for example through an external pointer
or `parameter.data.copy_`) are outside this cache contract.

P800 matrix shape-padding calibration is disabled because its separate GPU-graph
benchmark stalled during a cold model compile. Complete-graph selection still
uses synchronized host timing.

The first real invocation validates complete candidate outputs against FP32
eager, then chooses among original FX, ordinary Inductor, and F06-exposed
Inductor using randomized synchronized host timing. Invalid timings cannot be
cached. Warm calls directly use the selected complete-graph callable. The audit
records the actual route, rejected candidates and timings; an exposed pattern
is not by itself evidence of a single generated kernel. Compilation failures
and failed accuracy checks are recorded, and no candidate is selected unless
its outputs pass. GPU execution faults should be treated as a device-health
failure before continuing measurements.

`select_model_routes` also records `interface_audit` after calibration. Each
validated route is marked `selected`, `slower_measured`, or `similar_time`, with
its measured time and the selected replacement for a slower route. These labels
apply only to that complete model, validated inputs and calibration. They are
not a global blacklist for other shapes or dtypes. The returned callable invokes
the selected route directly; reading the audit adds no warm-call dispatch.

The P800 local Add+LayerNorm search retains the original schedules and compares
producer reuse, shared column loading, and tree reduction independently. Actual
buffer sizes after compiler resource adjustment are recorded. Tree reduction
can change floating-point accumulation order, so it receives the same complete
output validation as every other candidate. No resource limit or precision
threshold is raised to admit a candidate. The matching FlagTree compiler is a
separate prerequisite; installing this Python wheel alone does not install the
XPU lowering changes.

Whole-model input shapes are static: Dynamo recompiles when their guards
change. Training, autograd, arbitrary dynamic shapes, and full downstream
task-quality evaluation are outside this inference entry's validated scope.

Build from the source checkout:

```bash
python packaging/f06/prepare.py
python -m build --no-isolation packaging/f06
```

`manifest.json` records every bundled source hash. `f06-doctor` verifies these
files without running an accelerator workload. Passing that check does not
certify device health or performance.
