"""Supply native SDNN lookup functions, without GEMM code."""
import hashlib
import json
from pathlib import Path


def make_ewtable(mod, metadata, opt):
    import numpy as np
    import torch
    from triton._C.libtriton import xpu
    from triton.runtime.cache import get_cache_manager
    info_map = xpu.get_lut_info(mod)
    metadata['ewtable'] = ''
    if not info_map:
        return mod
    functions = {
        'math.exp': np.exp,
        'math.exp2': np.exp2,
        'math.log2': np.log2,
        'tt.gelu': lambda x: torch.nn.functional.gelu(torch.tensor(x)).item(),
        'tt.tanh': np.tanh,
        # Native sigmoid instructions implement 0.5 * tanh(x/2) + 0.5 and
        # therefore consume a tanh table, not a logistic table.
        'tt.sigmoid': np.tanh,
    }
    records = [(name, i.dtype, i.mode, i.size, i.min, i.interval) for name, i in info_map.items()]
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps((digest, records)).encode()).hexdigest()
    cache = get_cache_manager(key)
    name = 'activation_ewtable.dat'
    cached = cache.get_file(name)
    if cached is None:
        ks, bs = b'', b''
        for name, dtype, mode, size, lower, interval in records:
            fn = functions[name]
            table = [0.0] * (size * 2)
            if mode == 0:
                values = [fn(lower + i * interval) for i in range(size + 1)]
                for i in range(size):
                    table[i] = (values[i + 1] - values[i]) / interval
                    table[i + size] = values[i] - table[i] * (lower + i * interval)
            else:
                # Preserve the native backend's current interpolation layout.
                # Its exp-specific refinement branch is unreachable after fn
                # becomes callable; do not silently alter it in this candidate.
                values = [fn(lower + i * interval) for i in range(2 * (size + 1))]
                for i in range(size):
                    table[i], table[i + size] = values[2 * i], values[2 * i + 1]
            data = np.array(table, dtype={0: np.float32, 1: np.float16}[dtype]).tobytes()
            mid = len(data) // 2
            ks += data[:mid]
            bs += data[mid:]
        cached = cache.put(ks + bs, 'activation_ewtable.dat', binary=True)
    metadata['ewtable'] = cached
    metadata['activation_lut_diagnostic'] = json.dumps(records)
    return mod
