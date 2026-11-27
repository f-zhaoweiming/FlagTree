"""An in-kernel descriptor supplies a GM base and explicit strides to TLE DMA."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _copy_tile(source, destination):
    src = tl.make_tensor_descriptor(source, shape=[16, 144], strides=[144, 1], block_shape=[1, 128])
    dst = tl.make_tensor_descriptor(destination, shape=[16, 160], strides=[160, 1], block_shape=[1, 128])
    row = tl.program_id(0)
    data = tle.gpu.alloc([1, 128], dtype=tl.bfloat16, scope=tle.gpu.lmem)
    tle.gpu.copy(src, data, [1, 128], [row + 1, 8])
    tle.gpu.copy(data, dst, [1, 128], [row, 16])


def test_pitched_in_kernel_descriptor():
    if not torch.cuda.is_available() or type(triton.runtime.driver.active).__module__ != 'triton.backends.xpu.driver':
        pytest.skip("requires XPU")
    if triton.runtime.driver.active.get_current_target().arch != 3:
        pytest.skip("requires P800")
    src = torch.randn((16, 144), dtype=torch.bfloat16).cuda()
    out = torch.full((16, 160), -17, dtype=torch.bfloat16, device="cuda")
    expected = out.cpu()
    expected[:15, 16:144] = src.cpu()[1:, 8:136]
    _copy_tile[(15,)](src, out, num_warps=4, num_stages=1)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
