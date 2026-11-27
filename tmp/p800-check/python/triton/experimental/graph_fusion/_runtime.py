"""Graph runtime with opt-in guarded static inference.

Default behavior forwards the exact graph to native torch.compile/Inductor.
Opt-in static inference enables native Inductor GEMM/epilogue fusion for
eligible small NVIDIA graphs. No project arithmetic kernel is supplied.
Precision settings and live tensor metadata remain guarded.
"""

_ALLOWED_KWARGS = frozenset({"fullgraph", "dynamic", "mode", "options", "disable"})


def compile(graph_module, example_inputs=None, *, static_inference=False, avoid_bias_materialization=False, **kwargs):
    unknown = sorted(set(kwargs) - _ALLOWED_KWARGS)
    if unknown:
        raise TypeError("unexpected graph-fusion compile keyword(s): " + ", ".join(unknown))
    if type(static_inference) is not bool or type(avoid_bias_materialization) is not bool:
        raise TypeError("static_inference and avoid_bias_materialization must be bools")
    if avoid_bias_materialization and not static_inference:
        raise ValueError("avoid_bias_materialization requires static_inference=True")
    from triton.experimental import fusion as _fusion
    fallback = _fusion.compile(graph_module, **kwargs)
    if not static_inference or kwargs.get("disable", False):
        return fallback
    if kwargs.get("dynamic") is True or kwargs.get("mode") is not None or kwargs.get("options"):
        raise ValueError("static inference currently requires static shapes and default Inductor options")
    from ._inference import compile_static, UnsupportedStaticInference
    try:
        return compile_static(graph_module, example_inputs, avoid_bias_materialization=avoid_bias_materialization,
                              fallback=fallback)
    except UnsupportedStaticInference as exc:
        # Unsupported contracts retain native torch.compile behavior; native compiler
        # failures deliberately propagate instead of becoming hidden eager fallbacks.
        fallback._f06_static_rejection = str(exc)
        return fallback


__all__ = ["compile"]
