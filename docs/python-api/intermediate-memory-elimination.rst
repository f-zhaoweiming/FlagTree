Intermediate global-memory elimination
======================================

``triton-eliminate-intermediate-memory`` is a standalone core TTIR pass that
removes proven intermediate global-memory traffic inside an already composed
kernel. It is not part of the default compiler pipeline and has no TLE or
graph-fusion dependency. Enable it explicitly with ``triton-opt`` or the Python
pass manager.

A typical pipeline runs it after inlining, canonicalization and CSE, before GPU
layout assignment:

.. code-block:: bash

    triton-opt input.mlir --inline --canonicalize --cse \
      --triton-eliminate-intermediate-memory='private-arguments=1 assume-disjoint-arguments=true' \
      --canonicalize --verify-each

The equivalent Python pass-manager integration is:

.. code-block:: python

    import json
    from triton._C.libtriton import ir, passes

    # module is an existing TTIR module; argument 1 is a private temporary.
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_eliminate_intermediate_memory(
        pm, private_arguments=[1], assume_disjoint_arguments=True,
        num_ctas=1, cluster_size=1, max_transcendental_depth=8,
        max_transcendental_work=4096,
    )
    passes.common.add_canonicalizer(pm)
    pm.run(module, "eliminate_intermediate_memory")
    report = json.loads(module.get_operation().get_str_attr(
        "triton.intermediate_memory_elimination"))

The C++ entry point is ``createTritonEliminateIntermediateMemory``. The
``triton-opt`` registration and CLI option are generated from the pass
definition; the Python wrapper is a thin binding to the same factory. No extra
backend registration is required.

Caller contract
---------------

``assume_disjoint_arguments=True`` is an explicit promise that distinct pointer
arguments refer to disjoint storage, including all accessed offsets. The pass
does not inspect host allocations and cannot validate this promise. Without it,
the pass preserves the body and reports rejection. Multiple references to one
buffer must use the same entry argument.

``private_arguments`` lists zero-based entry-function argument indices whose
final buffer contents are unobservable outside the kernel. It defaults to empty:
loads may still be forwarded, but externally visible stores are retained. The
pass reports the private indices it actually eliminated; a rejected plan reports
an empty certificate and the caller must retain every temporary allocation.

The entry function must have one block. The launch must be one-dimensional with
grid Y and Z equal to one. ``num_ctas`` and the product of cluster dimensions
must both be one per program; a one-dimensional grid may still contain many
independent CTA programs. Inter-CTA communication and incompatible per-program
tile ownership are unsupported. Every mutable access must use the same
contiguous ``program_id(0) * BLOCK + arange(0, BLOCK)`` mapping for its buffer.

Volatile or nested memory accesses, atomics, unknown side effects, unresolved
calls, pointer escapes, incompatible masks or element types, and unsupported
address mappings are rejected conservatively. Operations with nested regions
are accepted only when they are memory-effect-free; they are outside the
mask-select cleanup proof. When the profitability guard is enabled, forwarding
into a nested region that contains transcendental math is rejected
conservatively.

Defaults
--------

``private-arguments`` is empty, ``assume-disjoint-arguments`` is false,
``num-ctas`` and ``cluster-size`` are one,
``max-transcendental-depth`` is eight, and ``max-transcendental-work`` is 4096.
A default invocation therefore rejects unless the caller supplies the disjoint
contract and, optionally, private indices. The Python wrapper uses the same
defaults.

Report and failure semantics
----------------------------

The module receives a JSON attribute ``triton.intermediate_memory_elimination``
with ``accepted``, ``reason``, ``forwarded_loads``, ``eliminated_stores``,
``eliminated_mask_selects``, ``private_arguments``, ``max_tile_elements`` and
``max_forwarded_transcendental_depth``. Rejection is not a compiler error: the
body is preserved and the report explains why the plan was skipped. The report
describes this pass invocation, not runtime validation or a speedup estimate.

The pass rewrites a clone and publishes the new body only after all checks
succeed. It does not remove a host allocation, function argument or kernel
launch. A caller may drop temporary storage only after checking a successful
report and verifying that the private pointer has no remaining uses in the
final compiled path.

Semantics
---------

Forwarding uses the stored operand, including the conversion to the storage
dtype, so fp16/bf16 rounding at the intermediate boundary is preserved. A
masked load's ``other`` value is retained through a select whenever inactive
lanes can reach a reduction, an unmasked output, unsafe integer arithmetic or
any other observable use. Selects are removed only when every use path is
lane-local and terminates at a store protected by the same mask. Bounds masks
remain independent of launch-grid specialization.

Profitability guard
-------------------

Forwarding can join long transcendental dependency chains across a memory
boundary and change backend scheduling. Two limits are applied to the complete
forwarding plan:

- ``max-transcendental-depth`` (CLI/Python: ``max-transcendental-depth`` /
  ``max_transcendental_depth``) bounds the dependency depth. It defaults to 8.
- ``max-transcendental-work`` (CLI/Python: ``max-transcendental-work`` /
  ``max_transcendental_work``) bounds ``max depth * max tile elements``, a
  static proxy for the unrolled transcendental code expansion. It defaults to
  4096. The multiplication saturates instead of wrapping on implausible static
  tiles.

Independent branches are combined with a maximum, not summed; unrelated math
and a long producer followed only by cheap arithmetic do not trigger the guard.
A plan is rejected when either limit is exceeded. The defaults reject the
observed depth-8, tile-1024 regressions while accepting depth-8/tile-256 and
depth-4/tile-1024; the work proxy is conservative and can reject some profitable
large-tile configurations (for example depth-4/tile-4096), which is intentional
for a default-off safety limit.

``max-transcendental-depth=0`` disables the whole profitability guard, retaining
the previous override semantics. ``max-transcendental-work=0`` disables only the
work subguard while the depth limit remains active. Neither zero bypasses the
disjoint-storage contract, mask, dtype-rounding or any legality check. Setting
either limit is a policy choice, not a hardware latency model or a universal
speed guarantee.

The guard analyzes entry-block SSA chains. Nested transcendental math is not
depth-accounted; forwarding into it, or forwarding a region-produced value whose
depth cannot be measured, is rejected instead while the guard is enabled.

Validation
----------

The ``test/Triton/eliminate-intermediate-memory*.mlir`` tests cover forwarding,
observable and private stores, incompatible mappings and masks, fp16/bf16
rounding, reductions and inactive lanes, precise arithmetic, nested pure
regions and profitability boundaries. The Python binding and caller contract
tests need no GPU:

.. code-block:: bash

    python -m pytest -q python/test/unit/runtime/test_intermediate_memory_elimination.py

Validation snapshot and default policy
-------------------------------------

The v2 work guard was evaluated on a full 42-fixture synthetic training matrix
and a pre-registered 66-fixture holdout (the original 46-case set plus 20
coverage cases registered before any GPU measurement). Both used real
``enable_fp_fusion=True``, ptxas ``--fmad=true``, 128 threads, and three
independent processes with 15 randomized paired A/B/C rounds.

- The four previously observed depth-8/tile-1024 regressions are now rejected
  and their C binaries equal A.
- Training: 50 stable gains (32 B-clean) and no pair met the stable-regression
  rule; 19 activated pairs were inconclusive, with 9 accepted no-ops and 48
  guard rejections. B was noisy in 36/126 pairs.
- Holdout: 78 stable gains (58 B-clean), no pair met the stable-regression rule,
  and 27 activated pairs were inconclusive; 6 accepted no-ops and 87 guard
  rejections. B was noisy in 41/198 pairs.
- Activated C cases that are only inconclusive can still be consistently slower
  by roughly 0.3--1.8%. The strict 2% rule therefore does not establish
  non-inferiority, and no universal no-regression claim is made.
- The work guard is conservative: it rejected 8 gains that were stable in the
  earlier matrix. Two other pairs became inconclusive with unchanged binaries;
  that variation is not attributable to the new guard. The guard intentionally
  trades some benefit for protection.

The pass remains default-OFF. The exact default invocation B has no
disjoint/private contract and is a no-op on all 108 fixtures; a meaningful
contract C is assessed above, but the evidence does not justify enabling it by
default. A post-hoc confirmatory followup (not pooled with the frozen matrix)
reran four suspicious fixtures in three extra processes; one B-clean case at
n=8,388,613 had two runs about 3.75--3.80% slower (still inconclusive under the
predeclared "all process medians <0.98" rule), reinforcing the default-off
decision. The holdout reuses the synthetic generator with unseen shapes, depths
and topology combinations; it is not a real-application corpus. Results cover
one warm-graph RTX 4070 Laptop (SM89) and do not establish cold-cache,
model-level, or cross-GPU safety.
