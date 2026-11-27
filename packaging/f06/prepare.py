"""Snapshot only F06 technical sources. No vendor binaries or private records."""
import hashlib
import json
from pathlib import Path
import shutil
import re

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
src=REPO/'python/triton/experimental'
dest=HERE/'src/f06/_vendor'
(dest/'graph_fusion').mkdir(parents=True,exist_ok=True)
(dest/'__init__.py').write_text('"""Bundled sources selected explicitly by f06.activate()."""\n')
for path in (src/'graph_fusion').iterdir():
    if path.suffix in ('.py','.c','.xpu'):
        shutil.copyfile(path,dest/'graph_fusion'/path.name)
shutil.copyfile(src/'fusion.py',dest/'fusion.py')
shutil.copyfile(REPO/'LICENSE',HERE/'LICENSE')
files={str(p.relative_to(HERE/'src/f06')):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(dest.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
version=re.search(r"__version__ = '([^']+)'",(HERE/'src/f06/__init__.py').read_text())[1]
record={'version':version,'source':'F06 sources with whole-graph Inductor scheduling and validated host-latency selection',
        'requirements':['P800 KL3, validated XPU-enabled Torch 2.9/Python 3.10 runtime',
                        'FlagTree XPU backend with TLE, XPU toolchain and matching runtime',
                        'Existing v51e LLVM/codegen fixes are external prerequisites; this wheel does not replace vendor binaries'],
        'files':files}
(HERE/'src/f06/manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print('Prepared',len(files),'bundled source files')
