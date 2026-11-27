"""CPU safety contracts for the legacy P800 native activation pass.

Numerical hardware acceptance is separate: these checks ensure that an
unsupported compiler or unrelated expression cannot acquire a placeholder.
"""
from contextvars import Context
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

_SOURCE = (Path(__file__).resolve().parents[3] / "triton" / "experimental" / "graph_fusion" /
           "_p800_native_activation.py")
_SPEC = importlib.util.spec_from_file_location("_p800_activation_contracts", _SOURCE)
activation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(activation)


def _unsupported_pipeline(mod, metadata, opt):
    return mod


def _missing_table_pipeline(mod, metadata, opt):
    xpu.passes.ttsdnnir.add_tritonsdnn_fuse_relu_activation_pass(pm)  # noqa: F821
    return mod


class NativeActivationSafetyTests(unittest.TestCase):

    def test_scope_is_context_local_and_restored_after_failure(self):
        with self.assertRaisesRegex(LookupError, "compile failed"):
            with activation.activation_scope():
                self.assertTrue(activation._ENABLED.get())
                self.assertFalse(Context().run(activation._ENABLED.get))
                with self.assertRaisesRegex(RuntimeError, "nested"):
                    with activation.activation_scope():
                        pass
                raise LookupError("compile failed")
        self.assertFalse(activation._ENABLED.get())

    def test_inactive_pass_does_not_touch_native_module(self):
        module, metadata = object(), {}
        with mock.patch.object(activation, "serialize") as serialize:
            self.assertIs(activation.prepare(module, metadata), module)
        serialize.assert_not_called()
        self.assertEqual(metadata, {"native_exact_gelu": "[]", "native_relu": "[]"})

    def test_unknown_or_incomplete_native_pipeline_is_rejected(self):
        for pipeline in (_unsupported_pipeline, _missing_table_pipeline):
            with self.subTest(pipeline=pipeline.__name__):
                with self.assertRaisesRegex(AssertionError, "unknown native SDNN"):
                    activation.install_stage(pipeline)

    def test_real_exp_cannot_be_reinterpreted_as_gelu_placeholder(self):
        module = object()
        text = "%actual = math.exp %x : tensor<32x32xf32>"
        with mock.patch.object(activation, "serialize", return_value=text), \
             mock.patch.object(activation, "parse") as parse:
            result, rewrites = activation.exact_gelu_to_placeholder(module, ".")
        self.assertIs(result, module)
        self.assertEqual(rewrites, [])
        parse.assert_not_called()

    def test_unrelated_comparison_does_not_become_relu(self):
        module = object()
        text = """%cst = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
%cmp = arith.cmpf ogt, %cst, %x : tensor<32x32xf32>
%out = arith.select %cmp, %cst, %x : tensor<32x32xi1>, tensor<32x32xf32>
"""
        with mock.patch.object(activation, "serialize", return_value=text), \
             mock.patch.object(activation, "parse") as parse:
            result, rewrites = activation.relu(module, ".", "native_relu_placeholder")
        self.assertIs(result, module)
        self.assertEqual(rewrites, [])
        parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
