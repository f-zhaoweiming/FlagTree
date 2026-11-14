# XPU per-kernel launch topology

The XPU launch ABI takes one `(ncluster, ncores)` pair. On XPU3/P800 the
historical defaults are:

| kernel mode | `is_sdnn` | pair | meaning |
|---|---:|---:|---|
| cluster | false | `(12, 64)` | 12 clusters, 64 cores per cluster |
| SDNN | true | `(12, 8)` | 12 SDNN units, 8 control cores per unit |

An SDNN kernel is selected by the compiler pipeline (`metadata.is_sdnn`,
normally because the source contains `tl.dot`). The runtime loader also uses
that flag to choose `KT_SDCDNN` versus `KT_CLUSTER`; the launch pair is not a
device-wide property.

## Per-kernel knobs

Pass new optional keywords to a `@triton.jit` launch, or include them in
`triton.compile(..., options={...})`:

```python
# Cluster example: pair compiled tiling with the requested launch core count.
kernel[grid](x, y, n, core_num=32, launch_clusters=6, launch_cores=32)
```

- `launch_clusters`: `ncluster` argument (cluster count, or SDNN-unit count
  for an SDNN kernel).
- `launch_cores`: `ncores` argument.
- Omit either field, or set it to `-1`, to keep the historical default.

The two values are validated before compilation/cache lookup. The current
conservative bounds are 1..default for the active arch/mode. The installed
public headers do not expose a maximum-core attribute, so larger values are
rejected rather than guessed to be functional. This cap is a safety bound, not
a claim about the physical maximum, and passing it does not establish
hardware correctness.

## Known xpu3 SDNN limitation

An experimental `launch_cores=4` SDNN launch compiled (`is_sdnn=True`,
compile `core_num=64`) but hung until the 180 s timeout; `launch_cores=2` was
manually aborted after ~25 s and is not a confirmed timeout. The current
backend therefore rejects explicit xpu3 SDNN `launch_cores` other than the
validated `8`; `-1` resolves to 8. `launch_clusters` remains configurable in
1..12, and ordinary cluster kernels keep their existing core range. This is a
software-support restriction, not a claim about the physical hardware limit.

## Compile-time coupling

`core_num` remains a separate compile-time tiling parameter. The cluster
lowering passes use it for work partitioning, so a runtime-only downscale can
mismatch the compiled participant count. The backend warns in that case.
Cluster benchmark candidates therefore set `core_num` to the same value as
`launch_cores`; SDNN retains its historical compile `core_num=64` while
launching 1..8 control cores. Every candidate must pass the benchmark's
pre- and post-timing device correctness checks before its timing is reported.

## Important distinction

`XPUOptions.core_num` controls compile-time tiling; changing it alone does not
change the physical launch pair. `cluster_num` is also an existing compiler
option, but this snapshot does not consume it in the lowering passes.
`launch_clusters`/`launch_cores` select the generated launcher's
`xpuLaunchKernel` arguments. For Cluster experiments, pair `core_num` with
`launch_cores` as above and validate the output. For SDNN experiments, retain
the existing compiler settings while varying the launch pair.

Both fields are part of `XPUOptions.hash()`, and their resolved values are
embedded in the generated C launcher. Consequently the compiled-kernel cache
and the launcher `.so` cache both distinguish different settings. Effective
values are recorded in the kernel metadata JSON.

No environment-variable defaults are provided: per-kernel keywords avoid
process-wide state and keep each JIT cache key explicit.
