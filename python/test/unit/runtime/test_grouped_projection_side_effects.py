"""Projection motion must retain input mutations and unrelated graph effects."""
import pytest
import torch
import torch.nn.functional as F
from torch import fx
from triton.experimental.graph_fusion._p800_model_gemm import (
    GroupedProjectionPack,
    grouped_projection_candidate,
)


@fx.wrap
def opaque_write(x):
    x.add_(1.)
    return x


class MutatingModule(torch.nn.Module):

    def forward(self, x):
        x.add_(1.)
        return x


class ProjectionModel(torch.nn.Module):

    def __init__(self, mode, bias):
        super().__init__()
        self.mode, self.bias = mode, bias
        self.mutator = MutatingModule()

    def forward(self, x, w1, w2, w3, b1, b2, b3):
        if self.mode == 'mask':
            mask = torch.ones_like(x[:, :, 0])
            mask.masked_fill_(mask > 0, 0)
        q = torch.ops.aten.linear.default(x, w1, b1 if self.bias else None).view(2, 7, 3, 4).transpose(1, 2)
        if self.mode == 'out':
            torch.add(x, 1., out=x)
        elif self.mode == 'alias_out':
            alias = x.view_as(x)
            torch.mul(x, 2., out=alias)
        elif self.mode == 'method':
            x.add_(1.)
        elif self.mode == 'schema':
            torch.ops.aten.add_.Tensor(x, 1.)
        elif self.mode == 'functional_inplace':
            F.relu(x, True)
        elif self.mode == 'opaque':
            opaque_write(x)
        elif self.mode == 'module':
            self.mutator(x)
        elif self.mode == 'pure':
            extra = torch.ops.aten.sin.default(x)
        k = torch.ops.aten.linear.default(x, w2, b2 if self.bias else None).view(2, 7, 3, 4).transpose(1, 2)
        v = torch.ops.aten.linear.default(x, w3, b3 if self.bias else None).view(2, 7, 3, 4).transpose(1, 2)
        result = q + k + v
        if self.mode == 'mask':
            result = result + mask[:, None, :, None]
        if self.mode == 'pure':
            return result, x, extra
        return result, x


def _trace(model, args):

    class Tracer(fx.Tracer):

        def is_leaf_module(self, module, name):
            return isinstance(module, MutatingModule) or super().is_leaf_module(module, name)

    graph = fx.GraphModule(model, Tracer().trace(model))

    class Values(fx.Interpreter):

        def run_node(self, node):
            value = super().run_node(node)
            if isinstance(value, torch.Tensor):
                node.meta['example_value'] = value
            return value

    Values(graph).run(args[0].clone(), *args[1:])
    return graph


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize(
    'mode', ['out', 'alias_out', 'method', 'schema', 'functional_inplace', 'opaque', 'module', 'plain', 'pure', 'mask'])
def test_grouped_projection_motion_respects_effects(device, bias, mode):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('P800/CUDA execution required')
    x = (torch.arange(224).reshape(2, 7, 16) % 7 - 3).float().to(device)
    params = [
        torch.nn.Parameter((torch.arange(192).reshape(12, 16) % 5 - 2).float().to(device) * (i + 1),
                           requires_grad=False) for i in range(3)
    ]
    params += [torch.nn.Parameter(torch.arange(12).float().to(device) - i, requires_grad=False) for i in range(3)]
    model = ProjectionModel(mode, bias)
    graph = _trace(model, (x, *params))
    before = graph.code
    candidate = grouped_projection_candidate(graph, (x, *params))
    unsafe = mode in ('out', 'alias_out', 'method', 'schema', 'functional_inplace', 'opaque', 'module')
    assert (candidate is None) == unsafe
    assert graph.code == before
    if candidate is None:
        call = graph.forward
    else:
        body, examples, groups, records = candidate
        assert len(records) == 1
        if mode == 'mask':
            assert any(n.op == 'call_method' and n.target == 'masked_fill_' for n in body.graph.nodes)
        call = GroupedProjectionPack(body.forward, groups)
    with torch.no_grad():
        actual = call(x.clone(), *params)
        expected = model(x.clone(), *params)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
