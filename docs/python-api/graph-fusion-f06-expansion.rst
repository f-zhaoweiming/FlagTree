Graph Fusion F06 Migration
==========================

Removed
-------

The F06 expansion's custom JIT arithmetic, low-precision and matmul variants,
SiLU LUT/approximate activation helper, handwritten-kernel replacement runtime,
tuning/launch selection, and FlagGems provider extension are removed from the
active package. Historical reports and archives remain local evidence and are
not active test inputs.

Retained
--------

Structural pattern matching, graph capture, cross-call stage inlining, and pure
data-movement compiler passes remain. Retained graph paths invoke original
PyTorch/Inductor operations and preserve operator targets, edges, options,
aliases, mutation behavior, NaN/special semantics, and original rounding points.

Migration
---------

- Use default ``compile`` for graph modules; it forwards the exact graph to the existing
  native ``torch.compile``/Inductor wrapper without executing example inputs.
- Use the structural template/registry and cross-call APIs for graph work.
- Do not import removed kernel, low-precision, reuse-provider, backend, or
  tuning modules.
- Old algorithm expectations in tests/docs are replaced by graph-only tests and
  migration/status documentation.

The optional guarded static inference entry is described in :doc:`graph-fusion`.
It now enables native Inductor automatic single-kernel fusion for bounded
NVIDIA MatMul+Bias compositions, including the original ReLU/GELU/SiLU activation.
The temporary project-owned small-matrix kernel has been removed from the active
package; compiler templates, tile selection and epilogue code generation are
owned by Inductor.
Other shapes and backends retain native compilation or the native MatMul+Bias
output reuse plan. The former backend-specific providers and tuning interfaces
remain removed. CPU contracts, device numerical acceptance and performance
measurements are reported separately; no universal speedup is claimed.
