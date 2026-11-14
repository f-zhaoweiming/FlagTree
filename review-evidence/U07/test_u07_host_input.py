from pathlib import Path
import json
import torch
from test_sdnn_silu_lut import _silu_kernel

def test_sdnn_silu_lut_with_host_reference():
    x = torch.linspace(-3, 3, 65536, device='cpu', dtype=torch.float32)
    src = x.cuda()
    out = torch.full_like(x, float('nan')).cuda()
    _silu_kernel[(8,)](src, out, BLOCK=8192, num_warps=8, num_stages=1, is_sdnn=True)
    torch.cuda.synchronize()
    diff = (out.cpu() - torch.nn.functional.silu(x)).abs()
    stats = {'elements':diff.numel(), 'max_abs':diff.max().item(), 'threshold':3e-4,
             'outside':int((~torch.isfinite(diff) | (diff>=3e-4)).sum().item())}
    Path('/out/numeric.json').write_text(json.dumps(stats)+'\n')
    assert stats['outside'] == 0
