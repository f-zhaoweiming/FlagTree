Operator graph fusion with Inductor
===================================

``triton.experimental.fusion.compile`` connects PyTorch operator graphs to
Inductor through the public ``torch.compile`` API. The former
``triton-fuse-kernel-calls`` pass and launch-capture implementation have been
removed. Fusion decisions now happen before each operator has a fixed Triton
tile or CTA mapping.

The former memory-elimination transform is available independently as
:doc:`intermediate-memory-elimination`, with its own explicit caller contract
and pass-manager entrypoint. This Inductor adapter does not invoke that pass.

.. code-block:: python

    import torch
    from triton.experimental import fusion

    def model(x, bias):
        shifted = torch.sin(x) + bias
        scale = torch.rsqrt(shifted.square().mean(dim=-1, keepdim=True) + 0.01)
        return shifted * scale, shifted

    compiled = fusion.compile(model, dynamic=True)
    x = torch.randn(17, 257, device="cuda")
    bias = torch.randn(257, device="cuda")
    with torch.no_grad():
        actual, intermediate = compiled(x, bias)

Functions, ``torch.nn.Module`` instances and ``torch.fx.GraphModule`` instances
are accepted. Decorator syntax, including ``@fusion.compile(dynamic=True)``,
is also supported. Compilation is lazy; creating the callable does not run
sample inputs. The returned object is PyTorch's compiled callable directly.

Architecture and ownership
--------------------------

For the F06 capability audit, reuse boundaries and implementation sequence, see
:doc:`f06-inductor-assessment`.

Dynamo captures tensor operators and maintains input guards. Inductor chooses
fusion regions, iteration schedules, tiling and layouts, using its resource
heuristics and configured autotuning. It can emit multiple kernels or call
external implementations. NVIDIA Triton kernels are compiled by the installed
``triton`` package; install FlagTree in that environment to use this compiler.
Inductor's CPU backend uses CPU code generation, so a successful CPU run does
not validate FlagTree CUDA execution.

This integration does not implement another cost model or impose a shared
``grid`` or ``BLOCK`` on source operators. There are no prewritten fusion
patterns in this entrypoint. Supported operators, devices and compiler-version
compatibility follow the installed PyTorch/Inductor and Triton combination.
The separate :doc:`graph-fusion` module remains an independent three-template
prototype; it is not used by this path.

The adapter uses the public `torch.compile API
<https://docs.pytorch.org/docs/stable/generated/torch.compile>`_. PyTorch's
`compiler overview <https://docs.pytorch.org/docs/stable/torch.compiler.html>`_
describes the graph capture and backend boundary.

Configuration and semantics
---------------------------

``compile(model, *, fullgraph=True, dynamic=None, mode=None, options=None,
disable=False)`` always selects the ``inductor`` backend. Other arguments are
passed to PyTorch without globally changing compiler configuration.

* ``fullgraph=True`` makes graph-capture failures visible. It requires one
  captured graph, not one generated kernel. Set ``fullgraph=False`` explicitly
  to permit PyTorch's native graph-break behavior.
* ``dynamic=True`` requests dynamic-shape compilation. ``None`` uses PyTorch's
  automatic policy. Shape/stride guards and recompilation belong to PyTorch;
  new inputs are not tied to original tensor objects or storage addresses.
* ``mode`` and ``options`` control Inductor. They are not Triton launch kwargs.
  Availability and mutual-exclusion rules follow the installed PyTorch.
* ``disable=True`` provides an explicit eager baseline. Training and autograd
  use the native compiled execution path; this adapter is not inference-only.

Numerics follow Inductor's precision and optimization policy. Fusion can change
rounding and reduction order, and default execution is not promised to be
bitwise identical to eager kernels. Where supported, an application can request
``options={"emulate_precision_casts": True}`` to retain intermediate
low-precision casts. That option does not by itself promise identical reduction
order or elementary-function results. Check application-specific tolerances.

Compiler and device execution errors propagate. The adapter does not catch an
execution error and retry the model, which could duplicate mutations. Native
PyTorch configuration, including any user-configured error suppression, still
applies. Successful compilation or fewer kernels is not a speedup guarantee.

Migration from kernel-call capture
----------------------------------

Replace the captured sequence with its tensor-operator computation and pass
that callable to ``fusion.compile``. ``fusion.session()`` now raises a migration
error before entering a capture context. ``FusionPlan``, ``PrivateTensor`` and
the private compiler options named ``kernel_call_fusion*`` are removed.
Existing hand-written Triton kernels cannot be automatically retiled merely by
capturing their launches. Normal Triton JIT/autotuning remains available.

The CUDA launcher rejects metadata from old kernel-call-fusion binaries before
compiling a launcher or accessing arguments. Recompile through the new entry
point; a cached bounds-mask certificate from the removed pass must not silently
lose its launch-grid constraint. Old benchmark result directories remain
historical evidence and do not describe this integration.

Validation
----------

.. code-block:: bash

    python -m pytest -q python/test/unit/runtime/test_fusion_inductor.py
    python python/benchmarks/benchmark_operator_fusion.py \
        --device cuda --output /tmp/operator-fusion-results

The suite exercises real Inductor code generation for a branched graph with
broadcast and reduction domains, functions/modules/FX inputs, changing shapes
and strides, autograd, live module state, precision options, capture failures,
and removal of the old pass and options. A separate CUDA test checks emitted
Triton source and numerical results on the installed FlagTree. It is skipped
when no CUDA device is available.

The benchmark retains generated source, runtime/compiler provenance, first-call
cost and alternating eager/compiled synchronized wall-time samples. It includes
host dispatch and allocation and does not label those samples GPU-only time.
CPU measurements and CUDA measurements must be reported separately.
