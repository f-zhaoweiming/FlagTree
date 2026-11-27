"""CPU tests for graph-only compile forwarding and public exports."""

import unittest
from unittest import mock

import triton.experimental.graph_fusion as graph_fusion
from triton.experimental.graph_fusion import compile as graph_compile


class _ExplodingInputs:

    def __iter__(self):
        raise AssertionError("example_inputs must not be iterated")

    def __len__(self):
        raise AssertionError("example_inputs must not be inspected")


class GraphOnlyRuntimeTests(unittest.TestCase):

    def test_exact_forwarding_ignores_example_inputs(self):
        graph_module = object()
        result = object()
        inputs = _ExplodingInputs()
        with mock.patch("triton.experimental.fusion.compile", return_value=result) as compile_mock:
            got = graph_compile(graph_module, inputs, fullgraph=False, dynamic=True, mode=None, options=None,
                                disable=False)
        self.assertIs(got, result)
        compile_mock.assert_called_once_with(graph_module, fullgraph=False, dynamic=True, mode=None, options=None,
                                             disable=False)

    def test_original_graph_identity_and_laziness(self):
        graph_module = object()
        with mock.patch("triton.experimental.fusion.compile", return_value="compiled") as compile_mock:
            self.assertEqual(compile_mock.call_count, 0)
            got = graph_compile(graph_module, None)
        self.assertEqual(got, "compiled")
        compile_mock.assert_called_once_with(graph_module)

    def test_deleted_algorithm_kwargs_are_rejected(self):
        graph_module = object()
        for key in ("kernel_spec", "tuning_plan", "implementation", "precision_policy"):
            with self.subTest(key=key):
                with mock.patch("triton.experimental.fusion.compile") as compile_mock:
                    with self.assertRaises(TypeError):
                        graph_compile(graph_module, None, **{key: object()})
                    compile_mock.assert_not_called()

    def test_compile_is_public_and_no_algorithm_exports(self):
        self.assertTrue(hasattr(graph_fusion, "compile"))
        self.assertFalse(hasattr(graph_fusion, "runtimecompile"))
        for name in ("FusionTemplate", "Registry", "composition_candidates", "cross_call_candidate",
                     "default_registry"):
            self.assertTrue(hasattr(graph_fusion, name), name)
        for name in ("CompiledGraph", "TuningPlan", "finite_configs", "select_configs"):
            self.assertFalse(hasattr(graph_fusion, name), name)
        for name in ("_kernels", "_kernels_f06", "_xpu_lowp", "_xpu_matmul", "_reuse", "_backend", "_tuning"):
            self.assertFalse(hasattr(graph_fusion, name), name)

    def test_unsupported_static_contract_retains_native_callable(self):

        def native(*args):
            return args

        with mock.patch("triton.experimental.fusion.compile", return_value=native):
            result = graph_compile(object(), (), static_inference=True)
        self.assertIs(result, native)
        self.assertIn("not_fx_graph_module", result._f06_static_rejection)

    def test_output_reuse_requires_explicit_static_inference(self):
        with mock.patch("triton.experimental.fusion.compile") as native:
            with self.assertRaises(ValueError):
                graph_compile(object(), (), avoid_bias_materialization=True)
            native.assert_not_called()

    def test_disabled_static_mode_keeps_native_disable_behavior(self):
        with mock.patch("triton.experimental.fusion.compile", return_value="native") as native:
            result = graph_compile(object(), _ExplodingInputs(), static_inference=True, disable=True)
        self.assertEqual(result, "native")
        native.assert_called_once_with(mock.ANY, disable=True)


if __name__ == "__main__":
    unittest.main()
