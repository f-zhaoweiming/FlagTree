"""Prepare a private, ABI-checked kunlun_ops profile in the P800 audit container."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

WHEEL = 'kunlun_ops-0.1.227+2b100f96-cp310-cp310-linux_x86_64.whl'
SHA256 = '7c17a6a8014366ba1f2cfaa06d8cfe838f4333f5d97c53eefe28bf83750f5f19'
URL = 'https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/' + WHEEL.replace('+', '%2B')
ROOT = Path('/data2/zhaoweiming/p800-interface-benchmark-20261126')
API = ('gelu', 'relu', 'silu', 'matmul', 'matmul_v2', 'bmm', 'add_layernorm')


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for data in iter(lambda: src.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def probe(site):
    import torch
    import torch_xmlir
    result = {'torch': str(torch.__version__), 'xmlir': str(Path(torch_xmlir.__file__).resolve())}
    try:
        import kunlun_ops
        for name in ('kunlun_ops', 'xpu_kunlun_ops', 'xpu_flash_ops'):
            if not Path(sys.modules[name].__file__).resolve().is_relative_to(site):
                raise RuntimeError(f'{name} resolved outside the staged wheel')
        result['apis'] = [name for name in API if callable(getattr(kunlun_ops, name, None))]
        if len(result['apis']) != len(API):
            raise RuntimeError('missing benchmark API')
    except (ImportError, OSError, RuntimeError) as exc:
        result['import_error'] = str(exc)
    paths = {Path(l.split()[-1]) for l in Path('/proc/self/maps').read_text().splitlines()
             if l.split()[-1].endswith('/libxpu_flash_attention.so')}
    if len(paths) != 1:
        raise RuntimeError(f'expected one loaded XFA provider, found {paths}')
    provider = paths.pop()
    lib = ctypes.CDLL(str(provider))
    symbols = set()
    for elf in sorted(site.rglob('*.so')):
        table = subprocess.check_output(['nm', '-D', '--undefined-only', str(elf)], text=True)
        symbols.update(t[1] for line in table.splitlines()
                       if len(t := line.split()) == 2 and t[0] == 'U' and '5baidu3xpu3xfa' in t[1])
    if not symbols:
        raise RuntimeError('no required XFA symbols found in the staged wheel')
    missing = sorted(sym for sym in symbols if not hasattr(lib, sym))
    result.update(provider=str(provider), provider_sha256=fingerprint(provider),
                  required=len(symbols), missing=missing)
    result['ok'] = not missing and 'import_error' not in result
    return result


def repair_fp16_wrappers(site):
    """Patch the two FP16 entry points in the pinned, privately extracted wheel."""
    activation = site / 'kunlun_ops/_activation.py'
    norm = site / 'kunlun_ops/_xflashinfer_norm.py'
    replacements = [
        (activation, '    return xpu_kunlun_ops.gelu(x, y)', '''    if (x.is_cuda and y.device == x.device and x.dtype == y.dtype == torch.float16
            and x.shape == y.shape):
        with torch.no_grad():
            torch.ops.aten.gelu.out(x, approximate="none", out=y)
        return 0
    return xpu_kunlun_ops.gelu(x, y)'''),
        (norm, '    return xpu_flash_ops.add_layernorm(\n', '''    if (x.is_cuda and x.ndim == 2 and x.shape == y.shape == output.shape
            and weight.shape == (x.shape[-1],) and x.device == y.device == weight.device == output.device
            and x.dtype == y.dtype == weight.dtype == output.dtype == torch.float16
            and (bias is None or (bias.shape == weight.shape and bias.device == x.device and bias.dtype == x.dtype))
            and (residual_output is None or (residual_output.shape == x.shape and residual_output.device == x.device
                                             and residual_output.dtype == x.dtype))
            and output_max is None):
        # The native half kernel requires float affine parameters. The FP32
        # fusion also keeps the residual sum wide through normalization.
        with torch.no_grad():
            out32 = torch.empty(output.shape, device=output.device, dtype=torch.float32)
            residual32 = None if residual_output is None else torch.empty(
                residual_output.shape, device=residual_output.device, dtype=torch.float32)
            status = xpu_flash_ops.add_layernorm(
                x.float().contiguous(), weight.float().contiguous(), out32,
                y.float().contiguous(), eps, False,
                None if bias is None else bias.float().contiguous(), residual32, None)
            if status == 0:
                output.copy_(out32)
                if residual_output is not None:
                    residual_output.copy_(residual32)
            return status
    return xpu_flash_ops.add_layernorm(
'''),
    ]
    applied = {}
    for module, before, after in replacements:
        source = module.read_text()
        if source.count(before) != 1:
            raise ValueError(f'unexpected source in {module.name}')
        module.write_text(source.replace(before, after))
        applied[str(module.relative_to(site))] = fingerprint(module)
    return applied


def stage(wheel, python):
    if not Path('/.dockerenv').exists() or not ROOT.resolve().is_relative_to('/data2/zhaoweiming'):
        raise RuntimeError('run in the personal Docker')
    ROOT.mkdir(parents=True, exist_ok=True)
    # Each attempt gets its own directory. The manifest is replaced only after the import passes.
    work = Path(tempfile.mkdtemp(prefix='ops-candidate-', dir=ROOT))
    if wheel is None:
        archive = ROOT.parent / 'sdk-release-20260806' / WHEEL
        wheel = archive if archive.exists() else work / WHEEL
        if not wheel.exists():
            urllib.request.urlretrieve(URL, wheel)
    if wheel.name != WHEEL or fingerprint(wheel) != SHA256:
        raise ValueError('unexpected wheel name or SHA-256')
    site = work / 'site'
    with zipfile.ZipFile(wheel) as archive:
        if any(not (site / m).resolve().is_relative_to(site) for m in archive.namelist()):
            raise ValueError('wheel member outside target')
        archive.extractall(site)
    wrappers = repair_fp16_wrappers(site)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(work),
               CUDA_VISIBLE_DEVICES='', XPU_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join(filter(None, [str(site), os.environ.get('PYTHONPATH') or ''])))
    if '__file__' in globals():
        program = [str(Path(__file__).resolve())]
    else:
        # The existing staging runner ships this file as python -c.
        program = ['-c', sys.orig_argv[sys.orig_argv.index('-c') + 1]]
    child = subprocess.run([str(python), *program, '--probe', str(site)],
                           env=env, cwd=work, capture_output=True, text=True, timeout=120)
    lines = [s[len('ABI_RESULT '):] for s in child.stdout.splitlines() if s.startswith('ABI_RESULT ')]
    result = json.loads(lines[-1]) if lines else {'ok': False, 'error': (child.stderr or child.stdout)[-1800:]}
    result.update(wheel=wheel.name, wheel_sha256=SHA256, site=str(site), python=str(python),
                  child_exit=child.returncode, wrapper_sha256=wrappers)
    (work / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['ok'] and child.returncode == 0:
        manifest = work / 'manifest.tmp'
        manifest.write_text(json.dumps(result, indent=2) + '\n')
        manifest.replace(ROOT / 'ops-profile.json')
    print(json.dumps(result))
    return bool(result['ok'] and child.returncode == 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--probe', type=Path)
    args = parser.parse_args()
    if args.probe:
        report = probe(args.probe.resolve())
        print('ABI_RESULT ' + json.dumps(report))
        sys.exit(0 if report['ok'] else 1)
    sys.exit(0 if stage(args.wheel, args.python.absolute()) else 1)
