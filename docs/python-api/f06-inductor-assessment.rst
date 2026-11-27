F06 Inductor Assessment
=======================

Status
------

The graph-only path uses the existing native ``torch.compile``/Inductor wrapper.
The former F06 low-precision, SIMT/Tensor matmul, activation approximation,
and tuning providers remain removed. Guarded static inference now enables native
Inductor single-kernel fusion for bounded NVIDIA small matrices, described in
:doc:`graph-fusion`. Scoped compiler configuration preserves matmul/add dtype
rounding and generates the selected activation variant from the graph. The
temporary project-owned small-matrix arithmetic kernel is no longer active.

The native wrapper is lazy and does not promise bitwise equality to eager. It
forwards the original graph and its options; this project does not force
custom precision policies, cudagraphs, or an alternate compiler backend.

Retained data movement
----------------------

Intermediate-memory elimination, tile-dataflow graph/resources/scheduling,
pure pattern/capture/inlining, and XPU launch-config/resource metadata remain.
They optimize graph boundaries, copies, allocations, and materialization only.

XPU compiler cleanup is limited to the branch-introduced opt-in vendor
optimizer and the nested ``np_silu`` helper/``tt.silu`` table entry. Baseline
``llvm19_toolchain`` downgrade/stamp/link/llc/launch code and launch_config
metadata remain unchanged.

Validation scope
----------------

CPU contracts and device acceptance are separate. The small-matrix GPU suite
checks report shapes, tails, broadcast biases, nonfinite values, stream/hooks,
live parameters, gradients and one actual kernel launch per admitted call.
Performance claims require the matching report baseline and full host timing;
they do not apply to arbitrary FX graphs or untested devices.
