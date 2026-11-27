import importlib.util
import pathlib
import sys

import torch

path = pathlib.Path('/tmp/p800-match-test/python/triton/experimental/graph_fusion/_p800_reduce.py')
spec = importlib.util.spec_from_file_location('_p800_reduce_poc3', path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def gm_mul_sum():
    g = torch.fx.Graph()
    a = g.placeholder('a')
    b = g.placeholder('b')
    p = g.call_function(torch.mul, (a, b))
    s = g.call_function(torch.sum, (p, -1))
    g.output(s)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def gm_layernorm():
    import torch.nn.functional as F
    g = torch.fx.Graph()
    a = g.placeholder('a')
    b = g.placeholder('b')
    z = g.call_function(torch.add, (a, b))
    out = g.call_function(F.layer_norm, (z, (256,)))
    g.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), g)


m, n = 8, 256
a = torch.randn(m, n, device='cuda')
b = torch.randn(n, device='cuda')

spec = mod.match(gm_mul_sum(), (a, b))
print('mul_sum match:', spec)
assert spec is not None and spec['kind'] == 'mul_sum'
compiled, proof = mod.compile(spec, (a, b))
out = compiled((a, b))[0]
ref = (a * b).sum(-1)
err = (out.float() - ref.float()).abs().max().item()
print('mul_sum max_err:', err, 'tiles:', proof['tiles'])
assert err < 1e-4, err

spec = mod.match(gm_layernorm(), (a, b))
print('layernorm match:', spec)
assert spec is not None and spec['kind'] == 'add_layernorm'
compiled, proof = mod.compile(spec, (a, b))
out = compiled((a, b))[0]
ref = torch.nn.functional.layer_norm(a + b, (256,))
err = (out.float() - ref.float()).abs().max().item()
print('layernorm max_err:', err, 'tiles:', proof['tiles'])
assert err < 1e-4, err
print('reduce POC OK')
