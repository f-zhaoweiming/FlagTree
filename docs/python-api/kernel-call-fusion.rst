:orphan:

Kernel-call fusion has been removed
===================================

The post-inline ``triton-fuse-kernel-calls`` pass, its compiler options and
``fusion.session`` capture implementation have been removed. Use
:doc:`operator-fusion` to compile tensor-operator graphs through Inductor.

The new entrypoint is ``triton.experimental.fusion.compile(model)``. It does
not accept captured Triton launches or their ``grid``/``BLOCK`` parameters.
Express the computation as tensor operators so the graph compiler can choose
its own fusion regions and execution mapping.

The memory-elimination transform is now available separately as
:doc:`intermediate-memory-elimination`. Its standalone TTIR interface does not
restore the removed capture API or compiler options.

Historical measurements remain in
``python/benchmarks/results/kernel_call_fusion_optimization`` and related
result directories. They measure the removed implementation and must not be
used as validation or performance claims for the Inductor integration.
