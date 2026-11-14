"""Compile-time XPU resource selection; never changes a computation expression."""
from contextlib import contextmanager
from contextvars import ContextVar

_RESOURCE = ContextVar("flagtree_xpu_compiler_resource", default="auto")
KINDS = frozenset(("auto", "sdnn", "cluster"))


def validate_resource(kind):
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("XPU resource_kind must be auto, sdnn or cluster")
    return kind


def current_resource():
    return _RESOURCE.get()


@contextmanager
def resource_scope(kind):
    token = _RESOURCE.set(validate_resource(kind))
    try:
        yield
    finally:
        _RESOURCE.reset(token)


def effective_sdnn(requested, legacy_sdnn, detected_sdnn, is_tle=False):
    validate_resource(requested)
    if requested == "cluster" and (legacy_sdnn or detected_sdnn):
        raise ValueError("Cluster resource cannot compile an SDNN-required kernel")
    if requested == "sdnn" and is_tle:
        raise ValueError("SDNN resource cannot compile a Cluster TLE kernel")
    return bool(legacy_sdnn or detected_sdnn or requested == "sdnn")
