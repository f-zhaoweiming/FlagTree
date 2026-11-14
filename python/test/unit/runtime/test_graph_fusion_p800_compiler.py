"""CPU contracts for the integrated P800 codegen policy."""
from contextvars import Context
import types
import unittest
from unittest import mock

from test_graph_fusion_static import InferenceCPUTestBase, torch, _load_module


@unittest.skipIf(torch is None, "torch is required")
class P800CompilerTests(InferenceCPUTestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.codegen = _load_module('_resource_codegen')

    def test_native_tracing_marks_only_materialized_low_precision_boundary(self):
        from torch._inductor import config

        def graph(x, bias, weight):
            return (torch.nn.functional.layer_norm(x + bias, (128, ), weight, None, 1e-3), )

        args = (torch.randn(2, 128, dtype=torch.bfloat16), torch.randn(128, dtype=torch.bfloat16),
                torch.randn(128, dtype=torch.bfloat16))
        with config.patch({'emulate_precision_casts': False}):
            traced = self.inference._trace_compiler_graph(graph, args, emulate_precision=True)
            self.assertFalse(config.emulate_precision_casts)
            plain = self.inference._trace_compiler_graph(graph, args, emulate_precision=False)
        add = next(n for n in traced.graph.nodes if n.target == torch.ops.aten.add.Tensor)
        self.assertTrue(add.meta.get('low_precision_pointwise_barrier'))
        plain_add = next(n for n in plain.graph.nodes if n.target == torch.ops.aten.add.Tensor)
        self.assertFalse(plain_add.meta.get('low_precision_pointwise_barrier', False))
        internal = [n for n in traced.graph.nodes if n.target == torch.ops.aten.mul.Tensor]
        self.assertTrue(internal)
        self.assertTrue(all(not n.meta.get('low_precision_pointwise_barrier', False) for n in internal))

    def test_schedule_is_scoped_and_restored_on_exception(self):
        from torch._inductor.codegen.triton import TritonKernel
        original = mock.Mock(return_value=True)
        with mock.patch.object(self.codegen, 'baidu_target', return_value={'arch': 3, 'warp_size': 1}), \
             mock.patch.object(TritonKernel, 'want_no_x_dim', original):
            with self.assertRaisesRegex(LookupError, 'compile failed'):
                with self.codegen.generated_resource('cluster', scalar_reduction=True):
                    self.assertFalse(TritonKernel.want_no_x_dim(object()))
                    self.assertTrue(Context().run(TritonKernel.want_no_x_dim, object()))
                    raise LookupError('compile failed')
            self.assertIs(TritonKernel.want_no_x_dim, original)
            with self.codegen.generated_resource('cluster'):
                self.assertTrue(TritonKernel.want_no_x_dim(object()))
            with self.codegen.generated_resource('sdnn'):
                self.assertTrue(TritonKernel.want_no_x_dim(object()))
        self.assertIsNone(self.codegen._REQUEST.get())

    def test_schedule_does_not_change_other_architectures(self):
        from torch._inductor.codegen.triton import TritonKernel
        with mock.patch.object(self.codegen, 'baidu_target', return_value={'arch': 4, 'warp_size': 1}), \
             mock.patch.object(TritonKernel, 'want_no_x_dim', return_value=True):
            with self.codegen.generated_resource('cluster', scalar_reduction=True):
                self.assertTrue(TritonKernel.want_no_x_dim(object()))
        with mock.patch.object(self.codegen, 'baidu_target', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'positively identified'):
                with self.codegen.generated_resource('cluster'):
                    pass

    def test_parallel_precompile_disabled_without_changing_selection(self):
        from torch._inductor.select_algorithm import AlgorithmSelectorCache
        calls = []

        def select(selector, choices, precompilation_timeout_seconds=3600):
            calls.append((choices, precompilation_timeout_seconds))
            return choices

        choices = object()
        with mock.patch.object(self.codegen, 'baidu_target', return_value={'arch': 3, 'warp_size': 1}), \
             mock.patch.object(AlgorithmSelectorCache, '__call__', select):
            with self.codegen.generated_resource('sdnn'):
                self.assertIs(AlgorithmSelectorCache.__call__(None, choices, 100), choices)
                self.assertIs(Context().run(AlgorithmSelectorCache.__call__, None, choices), choices)
            self.assertIs(AlgorithmSelectorCache.__call__, select)
        self.assertEqual(calls, [(choices, 0), (choices, 3600)])

    def test_annotation_preserves_native_arithmetic_and_versions_policy(self):
        source = '@triton_heuristics.reduction(size_hints=[32, 1024])\ndef native(x):\n    y = x * x\n    return y\n'
        changed = self.codegen.annotate_source(source, 'cluster')
        self.assertEqual(self.codegen._function_bodies(source), self.codegen._function_bodies(changed))
        self.assertIn('# P800 compiler policy: ', changed)
        self.assertIn("ResourceHeuristics('cluster')", changed)

    def test_legacy_resource_requires_boolean_binary_metadata(self):
        meta = types.SimpleNamespace(arch=3, is_sdnn=False, name='native')
        binary = types.SimpleNamespace(metadata=meta, hash='native-hash', asm={'xpubin': b'ELF'})
        self.assertEqual(self.codegen._record_binary(binary, 'cluster')['effective'], 'cluster')
        meta.resource_kind = 'sdnn'
        with self.assertRaises(RuntimeError):
            self.codegen._record_binary(binary, 'cluster')
        del meta.resource_kind
        meta.is_sdnn = 0
        with self.assertRaises(RuntimeError):
            self.codegen._record_binary(binary, 'cluster')
