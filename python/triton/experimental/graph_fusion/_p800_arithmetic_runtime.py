"""Bind the reviewed arithmetic rules to the inspected vendor runtime.

This cold check is deliberately separate from dispatch. Different SDK binaries
require a new audit; matching storage dtypes or header names is insufficient.
"""
import hashlib
from functools import lru_cache
from pathlib import Path


_REVIEWED = {
    'libxdnn_pytorch.so': 'cd8d9ba247f3485e591d71097ab069d590359f5ae5a0250298322b301552571c',
    'libxpu_dnn.so': '6219be06e43bced32c8ea14e047bf7922a29f65c07530d32d43a78909b904bfe',
    'libxpu_blas.so': '8b77a853c0225d538bdf387dee3b8e84fa68320b95bd641a0fa03491baba15e6',
    'libxpuapi.so': 'b2933b3313eb671d391b3f60429f9ca191cfc65d74732c9860ad2b55fe0eb918',
}


@lru_cache(maxsize=16)
def _digest(path, identity):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reviewed_runtime():
    paths = {name: set() for name in _REVIEWED}
    try:
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and Path(fields[5]).name in paths:
                paths[Path(fields[5]).name].add(fields[5])
        rows = {}
        for name, expected in _REVIEWED.items():
            if len(paths[name]) != 1:
                return dict(matched=False, reason='missing_or_ambiguous_loaded_library', library=name)
            path = Path(next(iter(paths[name])))
            st = path.stat()
            identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            digest = _digest(str(path), identity)
            rows[name] = digest
            if digest != expected:
                return dict(matched=False, reason='unreviewed_library_binary', libraries=rows)
        return dict(matched=True, libraries=rows, contract='P800_M06_M60_reviewed_operations_only')
    except OSError as exc:
        return dict(matched=False, reason='runtime_identity_unavailable', error=type(exc).__name__)
