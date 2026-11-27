"""Graph fusion and guarded static inference public exports.

Static inference enables native compiler fusion for bounded small matrices.
Project-specific arithmetic kernels and public precision-policy interfaces
remain absent.
"""

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

__all__ = [
    "CrossCallCandidate",
    "FusionTemplate",
    "Registry",
    "compile",
    "composition_candidates",
    "cross_call_candidate",
    "default_registry",
    "resource_backend",
    "autotune_resource_backend",
    "calibrate_resources",
]
