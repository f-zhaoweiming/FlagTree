"""Operator-graph fusion through PyTorch Inductor and the installed Triton.

Use ``compile(model)`` for a PyTorch callable, module, or FX GraphModule.
Dynamo captures operators before they have kernel launch grids; Inductor owns
fusion, tiling, layout selection, resource heuristics and code generation.
CUDA Triton kernels use the installed Triton/FlagTree compiler. CPU execution
uses Inductor's CPU backend, not FlagTree GPU code generation.

This module deliberately returns PyTorch's compiled callable directly. Input
guards, recompilation, autograd, graph breaks and compiler errors retain their
native behavior. ``fullgraph=True`` requires one captured graph, not one kernel
or a guaranteed speedup. Numerical behavior follows the selected Inductor
options and is not promised to be bitwise identical to eager execution.
"""

from functools import partial

__all__ = ["compile"]


def compile(model=None, *, fullgraph=True, dynamic=None, mode=None, options=None, disable=False):
    """Compile an operator graph with Inductor; compilation is lazy.

    The arguments have the same meaning as ``torch.compile`` except that the
    backend is always Inductor and ``fullgraph`` defaults to True, making
    unsupported capture explicit. Set ``fullgraph=False`` to allow native
    graph breaks, or ``disable=True`` for an eager baseline. ``dynamic=True``
    requests dynamic-shape compilation; None uses PyTorch's automatic policy.

    ``options`` are Inductor options, not Triton launch options. For example,
    ``options={"emulate_precision_casts": True}`` requests preservation of
    low-precision intermediate casts on PyTorch versions supporting that option.
    Backend option availability and numerical contracts depend on PyTorch.

    This supports both ``fusion.compile(model)`` and ``@fusion.compile(...)``.
    It never executes sample inputs or silently retries a failed compiled call.
    """
    if model is None:
        return partial(compile, fullgraph=fullgraph, dynamic=dynamic, mode=mode,
                       options=dict(options) if options is not None else None, disable=disable)

    # Keep importing the experimental namespace independent of PyTorch.
    import torch

    return torch.compile(model, backend="inductor", fullgraph=fullgraph, dynamic=dynamic,
                         mode=mode, options=dict(options) if options is not None else None,
                         disable=disable)


def session(*args, **kwargs):
    """Reject the removed kernel-capture API before entering its context."""
    raise RuntimeError("fusion.session and the kernel-call fusion pass were removed. "
                       "Express the computation as PyTorch operators and use "
                       "fusion.compile(model); existing Triton kernels cannot be "
                       "retiled by capturing their launches.")
