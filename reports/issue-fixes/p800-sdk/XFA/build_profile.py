"""Build a private XPyTorch 2.9 profile from the verified vendor release."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from prepare_ops import fingerprint

SDK_SHA = '51c2261bb34d5e0b378c3697c85ffd08e245c430209f042575c98950b8b9fd36'


def build(installer, parent):
    if not Path('/.dockerenv').exists() or not any(parent.resolve().is_relative_to(root)
            for root in ('/data1/zhaoweiming', '/data2/zhaoweiming')):
        raise ValueError('select a personal directory inside Docker')
    if fingerprint(installer) != SDK_SHA:
        raise ValueError('unexpected vendor installer SHA-256')
    parent.mkdir(exist_ok=True, parents=True)
    work = Path(tempfile.mkdtemp(prefix='xpytorch29-', dir=parent))
    kit, envdir = work / 'release', work / 'venv'
    env = dict(os.environ, HOME=str(work), TMPDIR=str(work), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run(['bash', str(installer), '--noexec', '--target', str(kit)], cwd=work, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(envdir)], env=env, check=True)
    subprocess.run(['dpkg-deb', '-x', str(kit / 'xcudart_package.deb'), str(envdir)], env=env, check=True)
    python = envdir / 'bin/python'
    site = envdir / 'lib/python3.10/site-packages'
    libs = [envdir / 'xcudart/lib', site / 'torch_xmlir', site / 'lib']
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, libs)) + ':' + env.get('LD_LIBRARY_PATH','')
    env['CONDA_PREFIX'] = str(envdir)
    env['VIRTUAL_ENV'] = str(envdir)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir',
                    '--no-build-isolation', '--no-compile', '--target', str(site),
                    *map(str, sorted(kit.glob('*.whl'))), str(kit / 'torch_plugin-0.1.0.tar.gz')],
                   cwd=work, env=env, check=True)
    launcher = work / 'python'
    ld = shlex.quote(os.pathsep.join(map(str, libs)))
    launcher.write_text('#!/bin/sh\n' + 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1\n' +
                        f'export CONDA_PREFIX={shlex.quote(str(envdir))} VIRTUAL_ENV={shlex.quote(str(envdir))}\n' +
                        f'export LD_LIBRARY_PATH={ld}${{LD_LIBRARY_PATH:+:\"$LD_LIBRARY_PATH\"}}\n' +
                        f'exec {shlex.quote(str(python))} "$@"\n')
    launcher.chmod(0o700)
    return launcher


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('installer', type=Path)
    p.add_argument('parent', type=Path)
    args = p.parse_args()
    print(build(args.installer.resolve(), args.parent.resolve()))
