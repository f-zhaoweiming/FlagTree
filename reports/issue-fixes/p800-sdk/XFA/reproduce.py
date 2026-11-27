"""Probe the ELF imports of a vendor wheel in a fresh process without installing it."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from prepare_ops import fingerprint, SHA256

parser = argparse.ArgumentParser()
parser.add_argument('wheel', type=Path)
parser.add_argument('scratch', type=Path)
parser.add_argument('--python', type=Path, default=Path(sys.executable))
args = parser.parse_args()
known = {SHA256, '064bec6b29e833df0c10fc167ca3e966e303fb2ea7cd456da8b9aee91cb6121d'}
if fingerprint(args.wheel) not in known:
    parser.error('use one of the two published wheels from the issue')
with tempfile.TemporaryDirectory(dir=args.scratch) as directory:
    site = Path(directory).resolve()
    with zipfile.ZipFile(args.wheel) as archive:
        if any(not (site / n).resolve().is_relative_to(site) for n in archive.namelist()):
            parser.error('wheel member outside scratch directory')
        archive.extractall(site)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(site), os.environ.get('PYTHONPATH')])),
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(site), HOME=str(site))
    result = subprocess.run([str(args.python), str(Path(__file__).with_name('prepare_ops.py').resolve()),
                             '--probe', str(site)], cwd=site, env=env)
sys.exit(result.returncode)
