"""Graph fusion, P800 resource autotuning and native Inductor options."""

from ._cross_call import CrossCallCandidate, cross_call_candidate
from ._patterns import (
    FusionTemplate,
    Registry,
    composition_candidates,
    default_registry,
)
from ._runtime import compile
from ._resource_runtime import resource_backend
from ._resource_autotune import autotune_resource_backend, calibrate_resources


def compile_model(model, **kwargs):
    from ._model import compile_model as prepare
    return prepare(model, **kwargs)


def compile_pipeline(tensor_model, **kwargs):
    from ._model_pipeline import compile_pipeline as prepare
    return prepare(tensor_model, **kwargs)


def select_model_routes(routes, **kwargs):
    """Select complete prepared call paths, including the caller's adapter."""
    from ._whole_model_selection import select_whole_model
    return select_whole_model(routes, **kwargs)


def model_candidate_routes(model, **kwargs):
    """Discover bounded precision partitions and full-model implementations."""
    from ._model_candidates import model_candidate_routes as prepare
    return prepare(model, **kwargs)


def p800_inductor_options():
    """Enable P800 FP32 region arithmetic and the native inference templates."""
    from torch._inductor import config
    from ._p800_fp32_region import rewrite_native
    from ._p800_inductor_gelu import rewrite
    previous = config.pre_grad_custom_pass
    def composed(graph):
        if previous is not None and previous is not rewrite:
            previous(graph)
        rewrite_native(graph)
        rewrite(graph)

    return {"pre_grad_custom_pass": composed}


__all__ = [
    "CrossCallCandidate",
    "FusionTemplate",
    "Registry",
    "compile",
    "compile_model",
    "compile_pipeline",
    "select_model_routes",
    "model_candidate_routes",
    "p800_inductor_options",
    "composition_candidates",
    "cross_call_candidate",
    "default_registry",
    "resource_backend",
    "autotune_resource_backend",
    "calibrate_resources",
]
