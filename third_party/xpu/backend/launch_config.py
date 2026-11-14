"""Physical launch-topology configuration for the XPU backend.

The XPU runtime launch ABI takes one ``(nclusters, ncores)`` pair.  For a
cluster kernel the pair is (clusters, cores per cluster); for an SDNN kernel
the same positions mean (SDNN units, control cores per unit).  The compiler's
existing ``cluster_num``/``core_num`` options are tiling parameters consumed by
the XPU passes, so they must not be reused for this physical launch setting.

This module deliberately has no Triton or device imports: validation,
resolution, and the generated-launcher tests can run on a host without
``libtriton`` or a P800.
"""

# -1 means "use the historical per-architecture/per-mode default".
DEFAULT_LAUNCH = -1


def get_xpu_spec(xpu_arch, is_sdnn=False):
    """Return the default launch pair for an XPU architecture.

    ``is_sdnn=False``: (num_clusters, num_cores)
    ``is_sdnn=True``:  (num_sdnns, num_cores)
    """
    if xpu_arch == 2:
        return (8, 8) if is_sdnn else (8, 64)
    elif xpu_arch == 3:
        return (12, 8) if is_sdnn else (12, 64)
    elif xpu_arch == 4:
        return (6, 8) if is_sdnn else (12, 64)
    else:
        raise RuntimeError(f"Unknown XPU architecture: {xpu_arch}")


def _checked_int(name, value):
    # bool is an int subclass, but a boolean launch size is a user error.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or -1, got {value!r}")
    return value


def resolve_launch_config(xpu_arch, is_sdnn, launch_clusters=DEFAULT_LAUNCH,
                          launch_cores=DEFAULT_LAUNCH):
    """Validate explicit launch knobs and return the effective (units, cores).

    Defaults are unchanged: -1 selects ``get_xpu_spec(xpu_arch, is_sdnn)``.

    The upper bounds below are conservative safety caps, not a claim about the
    physical maximum: the installed public XPU headers expose the default
    launch sizes but not a maximum-core attribute.  Allowing values from 1 up
    to the existing per-mode default permits candidate down-scaling settings;
    hardware validity and output correctness remain unverified until tested on
    the target device. Zero and larger-than-default values are rejected rather
    than guessed to be functional.
    """
    default_clusters, default_cores = get_xpu_spec(xpu_arch, is_sdnn)
    mode = "SDNN" if is_sdnn else "cluster"

    launch_clusters = _checked_int("launch_clusters", launch_clusters)
    launch_cores = _checked_int("launch_cores", launch_cores)

    if launch_clusters == DEFAULT_LAUNCH:
        clusters = default_clusters
    elif 1 <= launch_clusters <= default_clusters:
        clusters = launch_clusters
    else:
        raise ValueError(
            f"launch_clusters={launch_clusters} is invalid for {mode} mode on "
            f"xpu{xpu_arch}; expected -1 (default {default_clusters}) or an "
            f"integer in [1, {default_clusters}]")

    if launch_cores == DEFAULT_LAUNCH:
        cores = default_cores
    elif 1 <= launch_cores <= default_cores:
        if xpu_arch == 3 and is_sdnn and launch_cores != 8:
            raise ValueError(
                f"launch_cores={launch_cores} is not enabled for SDNN on xpu3 "
                f"(P800): the current backend/runtime support has only validated "
                f"the default 8 control cores. An experimental launch_cores=4 "
                f"compiled but hung until a 180s timeout; smaller counts are not "
                f"yet validated. Use -1 (default 8) or launch_cores=8. "
                f"launch_clusters remains configurable in [1, {default_clusters}].")
        cores = launch_cores
    else:
        raise ValueError(
            f"launch_cores={launch_cores} is invalid for {mode} mode on "
            f"xpu{xpu_arch}; expected -1 (default {default_cores}) or an "
            f"integer in [1, {default_cores}]")

    return clusters, cores
