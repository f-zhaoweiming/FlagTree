"""Single-card checks of the public, output-buffer kunlun_ops FP16 interfaces."""
import argparse
import json

import torch
from torch.nn import functional as F
import kunlun_ops
import xpu_flash_ops


@torch.inference_mode()
def verify_gelu():
    bits = torch.arange(65536, dtype=torch.int32).short().view(torch.float16)
    values = bits.cuda()
    expected = F.gelu(values, approximate='none')
    output = torch.empty_like(values)
    status = kunlun_ops.gelu(values, output)
    a, b = output.cpu(), expected.cpu()
    valid = ~b.isnan()
    gelu = dict(status=status, inputs=values.numel(), finite_inputs=int(bits.isfinite().sum()),
                different_non_nan_bits=int((a.view(torch.int16)[valid] != b.view(torch.int16)[valid]).sum()),
                different_nan_mask=int((a.isnan() != b.isnan()).sum()))
    view = values[:512].reshape(16, 32).T
    ref = F.gelu(view, approximate='none')
    alias = view.clone(memory_format=torch.preserve_format)
    rc = kunlun_ops.gelu(alias, alias)
    gelu['strided_inplace_equal'] = rc == 0 and torch.equal(alias, ref)
    gelu['passed'] = bool(status == 0 and gelu['different_non_nan_bits'] == 0 and
                          gelu['different_nan_mask'] == 0 and gelu['strided_inplace_equal'])

    return gelu


@torch.inference_mode()
def verify_add_layernorm():
    # All operands here are actual public FP16 inputs. Construct the native
    # FP32 oracle independently, with the same half input values.
    gen = torch.Generator().manual_seed(20261126)
    def half(shape):
        return torch.randn(shape, generator=gen).half().cuda()
    checks = []
    for use_bias, want_residual, strided, alias_output, eps in (
        (False, False, False, False, 1e-5),
        (True, False, False, True, .001),
        (False, True, True, False, .002),
        (True, True, False, False, 1e-5),
    ):
        m, n = 5, 64
        x = half((n,m)).T if strided else half((m,n))
        y, w = half((m,n)), half((n,))
        b = half((n,)) if use_bias else None
        out = x if alias_output else torch.empty_like(x)
        res = torch.empty_like(x) if want_residual else None
        out32 = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        res32 = torch.empty_like(out32) if want_residual else None
        raw = xpu_flash_ops.add_layernorm(x.float().contiguous(), w.float(), out32, y.float(),
                                          eps, False, b.float() if use_bias else None, res32, None)
        case = dict(bias=use_bias, residual=want_residual, strided=strided, output_alias=alias_output, eps=eps)
        try:
            rc = kunlun_ops.add_layernorm(x, y, w, out, eps, False, b, res, None)
            case['passed'] = bool(raw == rc == 0 and torch.equal(out, out32.half()) and
                                  (not want_residual or torch.equal(res, res32.half())))
        except RuntimeError as exc:
            case.update(passed=False, error=str(exc))
        checks.append(case)
    return checks


def verify(operator='all'):
    result = {}
    passed = True
    if operator in ('all', 'gelu'):
        result['gelu'] = verify_gelu()
        passed = result['gelu']['passed']
    if operator in ('all', 'add_layernorm'):
        result['add_layernorm'] = verify_add_layernorm()
        passed = passed and all(c['passed'] for c in result['add_layernorm'])
    result['passed'] = passed
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', choices=('all', 'gelu', 'add_layernorm'), default='all')
    result = verify(parser.parse_args().operator)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
