"""XPU grid values must participate in caching without retaining callbacks."""
import gc
import weakref

import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture(autouse=True)
def require_xpu():
    from triton.compiler.compiler import make_backend
    from triton.runtime import driver
    get_target = getattr(driver.active, "get_current_target_inside", driver.active.get_current_target)
    target = get_target()
    if make_backend(target).binary_ext != "xpubin":
        pytest.skip("XPU grid is a compilation option")
    torch.set_num_threads(1)


def tensors(n=128):
    host = torch.arange(n, dtype=torch.float32).pin_memory()
    x = host.cuda(non_blocking=True)
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    return host, x, y


def check(y, expected):
    torch.cuda.synchronize()
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


def copy_kernel():
    @triton.jit
    def kernel(x, y, n, BLOCK: tl.constexpr = 32):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x + i, i < n, other=0)
        tl.store(y + i, value + 1, i < n)
    return kernel


def kernels(kernel):
    return kernel.device_caches[torch.cuda.current_device()][0]


@pytest.mark.parametrize("as_list", [False, True])
def test_fresh_callbacks_reuse_kernel_and_are_not_retained(as_list):
    expected, x, y = tensors()
    kernel = copy_kernel()
    refs = []
    for _ in range(5):
        grid = lambda meta: [triton.cdiv(meta["n"], meta["BLOCK"])] if as_list else (
            triton.cdiv(meta["n"], meta["BLOCK"]),)
        refs.append(weakref.ref(grid))
        kernel[grid](x, y, n=x.numel())
        check(y, expected + 1)
    del grid
    gc.collect()
    assert len(kernels(kernel)) == 1
    assert not kernel._bound_launchers
    assert all(ref() is None for ref in refs)
    assert all("<function" not in key for key in kernels(kernel))


def test_changed_callable_result_invalidates_compilation_grid():
    expected, x, y = tensors()
    kernel = copy_kernel()
    shape = {"blocks": 4}
    grid = lambda meta: (shape["blocks"],)
    for blocks in (4, 5, 4):
        shape["blocks"] = blocks
        y.fill_(-1)
        compiled = kernel[grid](x, y, x.numel())
        check(y, expected + 1)
        assert tuple(compiled.metadata.grid) == (blocks,)
    assert len(kernels(kernel)) == 2


def test_static_and_callable_grid_share_cached_compilation():
    expected, x, y = tensors()
    kernel = copy_kernel()
    kernel[(4,)](x, y, x.numel())
    kernel[lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)](x, y, x.numel())
    check(y, expected + 1)
    assert len(kernels(kernel)) == 1


def test_warmup_binds_grid_defaults_without_launching():
    expected, x, y = tensors()
    kernel = copy_kernel()
    y.fill_(-1)
    kernel.warmup(x, y, n=x.numel(), grid=lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),))
    check(y, torch.full_like(expected, -1))
    kernel[(4,)](x, y, x.numel())
    check(y, expected + 1)
    assert len(kernels(kernel)) == 1


def test_autotune_grid_uses_each_config_and_reuses_winner():
    expected, x, y = tensors(320)
    kernel = copy_kernel()
    tuned = triton.autotune(configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})],
                            key=["n"])(kernel)
    for _ in range(3):
        tuned[lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)](x, y, x.numel())
        check(y, expected + 1)
    assert len(kernels(kernel)) == 2
