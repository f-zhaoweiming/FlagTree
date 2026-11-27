"""CPU tests for the opt-in static inference plan.

The tests load ``_patterns.py`` and ``_inference.py`` through a synthetic
package so they also work when this directory is not imported as part of a
package.  A mocked ``compile_fx_inner`` keeps the suite CPU-only and avoids
running Inductor code generation; the public CUDA gate for native mm+add(out=)
reuse is verified separately, and an internal test hook exercises the same
mm/add plan on CPU.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - parent remote environment has torch
    torch = None

_HERE = pathlib.Path(__file__).resolve().parents[3] / "triton" / "experimental" / "graph_fusion"
_PACKAGE = "_f06_static_inference_cpu_tests"
_LOADED = {}


def _load_module(name):
    full_name = f"{_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules[_PACKAGE], name, module)
    return module


def _load_under_test():
    if "inference" not in _LOADED:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_HERE)]
        sys.modules[_PACKAGE] = package
        _LOADED["patterns"] = _load_module("_patterns")
        _LOADED["inference"] = _load_module("_inference")
    return _LOADED


def _bit_equal(left, right):
    """Exact representation equality, including NaN payloads and signed zero."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype == torch.float32:
        int_dtype = torch.int32
    elif left.dtype == torch.float64:
        int_dtype = torch.int64
    elif left.dtype == torch.float16:
        int_dtype = torch.int16
    elif left.dtype == torch.bfloat16:
        int_dtype = torch.int16
    else:
        return torch.equal(left, right)
    return torch.equal(
        left.detach().contiguous().view(int_dtype),
        right.detach().contiguous().view(int_dtype),
    )


class _Backstop:

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        return "fallback"


class _FakeCompiled:

    def __init__(self, aten_gm, source_code):
        self.aten_gm = aten_gm
        self.source_code = source_code
        self.calls = 0
        self.last_inputs = None

    def __call__(self, values):
        self.calls += 1
        self.last_inputs = list(values)
        return self.aten_gm(*values)


@unittest.skipIf(torch is None, "torch is required for static inference tests")
class InferenceCPUTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        loaded = _load_under_test()
        cls.patterns = loaded["patterns"]
        cls.inference = loaded["inference"]

    def _graph_module(self, graph, attrs=None):
        root = torch.nn.Module()
        for name, value in (attrs or {}).items():
            setattr(root, name, value)
        return torch.fx.GraphModule(root, graph)

    def _sample_inputs(self, rows=3, inner=5, width=4):
        return torch.randn(rows, inner), torch.randn(inner, width)

    def _make_symbolic_mm_bias(self, *, bias_first=False, matmul=False):

        class Mod(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(4), requires_grad=False)

            def forward(self, x, w):
                product = torch.matmul(x, w) if matmul else torch.mm(x, w)
                if bias_first:
                    return torch.add(self.bias, product)
                return torch.add(product, self.bias)

        return torch.fx.symbolic_trace(Mod())

    def _make_mm_bias_graph(
        self,
        *,
        bias_first=False,
        matmul=False,
        alpha=None,
        out_keyword=False,
        bias=None,
        extra_attr=False,
        output_tuple=False,
    ):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        product = graph.call_function(torch.matmul if matmul else torch.mm, (x, w))
        bias_node = graph.get_attr("bias")
        kwargs = {}
        if alpha is not None:
            kwargs["alpha"] = alpha
        if out_keyword:
            kwargs["out"] = None
        operands = (bias_node, product) if bias_first else (product, bias_node)
        addition = graph.call_function(torch.add, operands, kwargs)
        if extra_attr:
            graph.get_attr("extra")
        if output_tuple:
            graph.output((addition, addition))
        else:
            graph.output(addition)

        attrs = {"bias": torch.randn(4) if bias is None else bias}
        if extra_attr:
            attrs["extra"] = torch.randn(4)
        return self._graph_module(graph, attrs)

    def _install_mock_compiler(self, source_code="mock_inductor_source"):
        capture = {"compiled": [], "compile_inputs": [], "aten_gm": []}

        def fake_compile(aten_gm, flat):
            capture["aten_gm"].append(aten_gm)
            capture["compile_inputs"].append(tuple(flat))
            compiled = _FakeCompiled(aten_gm, source_code)
            capture["compiled"].append(compiled)
            return compiled

        patcher = mock.patch.object(self.inference, "_call_compile_fx_inner", side_effect=fake_compile)
        patcher.start()
        self.addCleanup(patcher.stop)
        return capture

    def _assert_fallback_call(self, plan, fallback, compiled, *args, **kwargs):
        compiled_before = compiled.calls
        fallback_before = len(fallback.calls)
        result = plan(*args, **kwargs)
        self.assertEqual(result, "fallback")
        self.assertEqual(compiled.calls, compiled_before)
        self.assertEqual(len(fallback.calls), fallback_before + 1)
        return fallback.calls[-1]


class SmallMatmulSelectionTests(InferenceCPUTestBase):

    def _matrix_graph(self):

        class Mod(torch.nn.Module):

            def forward(self, bias, w, x):
                return torch.relu(torch.add(input=bias, other=torch.mm(input=x, mat2=w)))

        return torch.fx.symbolic_trace(Mod())

    def test_operand_identity_and_bias_broadcasts(self):
        gm = self._matrix_graph()
        for shape, strides in (((), (0, 0)), ((1, ), (0, 0)), ((4, ), (0, 1)), ((1, 4), (0, 1)), ((3, 1), (1, 0)),
                               ((3, 4), (4, 1))):
            with self.subTest(shape=shape):
                args = (torch.randn(shape), torch.randn(5, 4), torch.randn(3, 5))
                spec, reason = self.inference._find_small_matmul(self.inference._analyze(gm, args))
                self.assertIsNone(reason)
                self.assertEqual(spec.indices, (2, 1, 0))
                self.assertEqual(spec.shape, (3, 4, 5))
                self.assertEqual(spec.bias_strides, strides)
                self.assertTrue(spec.bias_first)
                self.assertEqual(spec.activation, "relu")

    def test_shape_layout_dtype_and_precision_rejection(self):
        from dataclasses import replace
        gm = self._matrix_graph()
        for m, n, k in ((0, 4, 5), (65, 4, 5), (3, 1025, 5), (3, 4, 129), (3, 4, 0)):
            args = (torch.randn(n), torch.randn(k, n), torch.randn(m, k))
            spec, reason = self.inference._find_small_matmul(self.inference._analyze(gm, args))
            self.assertIsNone(spec)
            self.assertEqual(reason, "outside_small_matmul_bounds")
        args = (torch.randn(4), torch.randn(5, 4), torch.randn(3, 5))
        analysis = self.inference._analyze(gm, args)
        state = list(analysis.precision_state)
        state[1] = True
        self.assertEqual(
            self.inference._find_small_matmul(replace(analysis, precision_state=tuple(state)))[1],
            "small_matmul_requires_tf32_disabled")
        half = tuple(t.half() for t in args)
        analysis = self.inference._analyze(gm, half)
        state = list(analysis.precision_state)
        state[3] = (True, True)
        self.assertEqual(
            self.inference._find_small_matmul(replace(analysis, precision_state=tuple(state)))[1],
            "small_matmul_requires_full_precision_reduction")
        for operands, expected in (
            (tuple(t.double() for t in args), "small_matmul_dtype"),
            ((args[0], args[1].T.contiguous().T, args[2]), "small_matmul_requires_dense_contiguous"),
            ((torch.randn(3), args[1], args[2]), "small_matmul_bias_broadcast"),
        ):
            self.assertEqual(self.inference._find_small_matmul(self.inference._analyze(gm, operands))[1], expected)

    def test_fused_plan_uses_live_attrs_and_existing_guard_fallback(self):
        gm = self._make_symbolic_mm_bias()
        args = self._sample_inputs()
        fallback = _Backstop()
        source = "kernel = async_compile.triton('kernel', '')\ndef call(args):\n    return kernel.run(*args)\n"
        capture = self._install_mock_compiler(source)
        from torch._inductor import config
        options = self.inference._small_matmul_options()
        before = {k: config.get_config_copy()[k] for k in options}
        with mock.patch.object(self.inference, "_small_matmul_device_supported", return_value=True):
            plan = self.inference.compile_static(gm, args, fallback=fallback, avoid_bias_materialization=True)
        self.assertEqual(plan.kind, "inductor_small_matmul_single_kernel")
        self.assertEqual(plan.plan_info["kernel_provider"], "native_inductor_codegen")
        self.assertEqual(plan.plan_info["kernel_count"], 1)
        self.assertEqual(plan.plan_info["small_matmul_compiler_options"], options)
        self.assertEqual(before, {k: config.get_config_copy()[k] for k in before})
        self.assertEqual(len(capture["compiled"]), 1)
        compiled = capture["compiled"][0]
        held = plan(*args)
        old = held.clone()
        gm.bias = torch.nn.Parameter(torch.randn(4), requires_grad=False)
        torch.testing.assert_close(plan(*args), gm(*args))
        self.assertIs(compiled.last_inputs[-1], gm.bias)
        torch.testing.assert_close(held, old)
        self.assertEqual(plan(torch.randn(2, 5), args[1]), "fallback")
        self.assertEqual(plan.fallback_calls, 1)
        self.assertEqual(compiled.calls, 2)

    def test_cpu_keeps_native_compiler_and_kernel_errors_propagate(self):
        gm = self._matrix_graph()
        args = (torch.randn(4), torch.randn(5, 4), torch.randn(3, 5))
        self._install_mock_compiler()
        plan = self.inference.compile_static(gm, args)
        self.assertEqual(plan.kind, "inductor_static_inference")
        self.assertEqual(plan.plan_info["small_matmul_skipped_reason"], "small_matmul_requires_nvidia_sm80_or_p800")
        with mock.patch.object(self.inference, "_small_matmul_device_supported", return_value=True), \
             mock.patch.object(self.inference, "_call_compile_fx_inner", side_effect=RuntimeError("kernel compiler failed")):
            with self.assertRaisesRegex(RuntimeError, "kernel compiler failed"):
                self.inference.compile_static(gm, args)

    def test_compiler_single_kernel_audit_does_not_trust_a_requested_count(self):
        source = "kernel = async_compile.triton('kernel', '')\ndef call(args):\n    kernel.run(*args)\n"
        self.assertEqual(self.inference._single_inductor_kernel([source]), "kernel")
        for sources in ([], [source, source], ["mock unknown source"], [source + "    kernel.run(*args)\n"
                                                                        ], [source + "    extern_kernels.mm(*args)\n"],
                        [source + "other = async_compile.triton('other', '')\n"
                         ], [source + "other = async_compile.multi_kernel([])\n"]):
            self.assertIsNone(self.inference._single_inductor_kernel(sources))
        gm = self._matrix_graph()
        args = (torch.randn(4), torch.randn(5, 4), torch.randn(3, 5))
        self._install_mock_compiler("unknown compiler wrapper")
        with mock.patch.object(self.inference, "_small_matmul_device_supported", return_value=True):
            plan = self.inference.compile_static(gm, args)
        self.assertEqual(plan.kind, "inductor_static_inference")
        self.assertNotIn("kernel_count", plan.plan_info)
        self.assertEqual(plan.plan_info["small_matmul_skipped_reason"], "compiler_single_kernel_not_verified")


class LargeMatmulSelectionTests(InferenceCPUTestBase):

    def _graph(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.randn(4))

            def forward(self, x, w):
                return torch.relu(x @ w + self.bias)

        return torch.fx.symbolic_trace(Mod())

    def test_scoped_policy_live_attributes_and_guard_fallback(self):
        from torch._inductor import config
        gm = self._graph()
        args = (torch.randn(65, 5), torch.randn(5, 4))
        fallback = _Backstop()
        capture = self._install_mock_compiler()
        before = config.get_config_copy()
        with mock.patch.object(self.inference, "_large_matmul_device_supported", return_value=True):
            plan = self.inference.compile_static(gm, args, fallback=fallback)
        self.assertEqual(config.get_config_copy(), before)
        self.assertIn("large_matmul_compiler_options", plan.plan_info)
        # A generic/multiple-kernel compiler result is valid for the large path.
        self.assertEqual(plan.kind, "inductor_static_inference")
        self.assertNotIn("kernel_count", plan.plan_info)
        held = plan(*args)
        old = held.clone()
        gm.bias = torch.randn(4)
        torch.testing.assert_close(plan(*args), gm(*args))
        torch.testing.assert_close(held, old)
        self.assertIs(capture["compiled"][0].last_inputs[-1], gm.bias)
        self.assertEqual(plan(torch.randn(64, 5), args[1]), "fallback")
        self.assertEqual(plan.fallback_calls, 1)

    def test_cpu_bias_only_and_small_policies_are_unchanged(self):
        self._install_mock_compiler()
        gm = self._graph()
        args = (torch.randn(65, 5), torch.randn(5, 4))
        plan = self.inference.compile_static(gm, args)
        self.assertNotIn("large_matmul_compiler_options", plan.plan_info)
        with mock.patch.object(self.inference, "_large_matmul_device_supported", return_value=True):
            for graph, inputs in ((gm, (args[0][:64], args[1])),
                                  (self._make_symbolic_mm_bias(), args)):
                plan = self.inference.compile_static(graph, inputs)
                self.assertNotIn("large_matmul_compiler_options", plan.plan_info)

    def test_precision_gate_and_compile_failure_restore_options(self):
        from torch._inductor import config
        from dataclasses import replace
        gm = self._graph()
        args = (torch.randn(65, 5), torch.randn(5, 4))
        analysis = self.inference._analyze(gm, args)
        state = list(analysis.precision_state)
        state[1] = True
        spec, _ = self.inference._find_matmul_epilogue(replace(analysis, precision_state=tuple(state)), small=False)
        self.assertIsNone(spec)
        before = config.get_config_copy()
        with mock.patch.object(self.inference, "_large_matmul_device_supported", return_value=True), \
             mock.patch.object(self.inference, "_call_compile_fx_inner", side_effect=RuntimeError("compile failed")):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                self.inference.compile_static(gm, args)
        self.assertEqual(config.get_config_copy(), before)


class GraphLiftingAndPlanTests(InferenceCPUTestBase):

    def test_compiler_plan_lifts_attrs_and_fetches_live_values(self):
        gm = self._make_symbolic_mm_bias()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()

        plan = self.inference.compile_static(gm, (x, w), fallback=None)

        self.assertEqual(plan.kind, "inductor_static_inference")
        self.assertEqual(plan.generated_sources, ["mock_inductor_source"])
        self.assertEqual(len(capture["aten_gm"]), 1)
        self.assertTrue(any(node.op == "get_attr" for node in gm.graph.nodes))

        aten_gm = capture["aten_gm"][0]
        aten_placeholders = [node for node in aten_gm.graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(aten_placeholders), 3)  # x, w, lifted bias
        self.assertFalse(any(node.op == "get_attr" for node in aten_gm.graph.nodes))

        output = plan(x, w)
        expected = torch.mm(x, w) + gm.bias
        self.assertTrue(_bit_equal(output, expected))
        self.assertEqual(capture["compiled"][0].calls, 1)
        self.assertIs(capture["compiled"][0].last_inputs[-1], gm.bias)

        with torch.no_grad():
            gm.bias.add_(0.5)
        output_after_mutation = plan(x, w)
        expected_after_mutation = torch.mm(x, w) + gm.bias
        self.assertTrue(_bit_equal(output_after_mutation, expected_after_mutation))
        self.assertFalse(_bit_equal(output, output_after_mutation))
        self.assertIs(capture["compiled"][0].last_inputs[-1], gm.bias)

    def test_attr_replacement_with_same_metadata_is_live(self):
        gm = self._make_symbolic_mm_bias()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        plan = self.inference.compile_static(gm, (x, w), fallback=None)

        replacement = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4), requires_grad=False)
        gm.bias = replacement
        output = plan(x, w)
        expected = torch.mm(x, w) + replacement
        self.assertTrue(_bit_equal(output, expected))
        self.assertIs(capture["compiled"][0].last_inputs[-1], replacement)

    def test_lift_preserves_kwargs_and_uses_one_element_tuple_output(self):
        tensor_x, tensor_w = self._sample_inputs()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        product = graph.call_function(torch.mm, (x, w))
        bias = graph.get_attr("bias")
        addition = graph.call_function(torch.add, (product, bias), {"alpha": 1.0})
        graph.output(addition)
        gm = self._graph_module(graph, {"bias": torch.randn(4)})

        lifted = self.inference._lift_graph_module(gm, (x, w), ("bias", ))
        lifted_nodes = list(lifted.graph.nodes)
        self.assertFalse(any(node.op == "get_attr" for node in lifted_nodes))
        self.assertEqual(sum(node.op == "placeholder" for node in lifted_nodes), 3)
        lifted_add = next(node for node in lifted_nodes if node.op == "call_function" and node.target is torch.add)
        self.assertEqual(lifted_add.kwargs, {"alpha": 1.0})
        output_node = lifted_nodes[-1]
        self.assertEqual(output_node.op, "output")
        self.assertEqual(len(output_node.args), 1)
        self.assertIsInstance(output_node.args[0], tuple)
        self.assertEqual(len(output_node.args[0]), 1)

        # The original module still owns its get_attr and is not rewritten.
        self.assertTrue(any(node.op == "get_attr" for node in gm.graph.nodes))

        result = lifted(tensor_x, tensor_w, gm.bias)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertTrue(_bit_equal(result[0], torch.mm(tensor_x, tensor_w) + gm.bias))

    def test_native_compile_call_contract(self):
        from torch._inductor.utils import BoxedBool

        captured = {}

        def fake_compile(aten_gm, flat_inputs, **kwargs):
            captured["gm"] = aten_gm
            captured["inputs"] = flat_inputs
            captured["kwargs"] = kwargs
            return lambda values: (torch.zeros(1), )

        with mock.patch(
                "torch._inductor.compile_fx.compile_fx_inner",
                side_effect=fake_compile,
        ):
            result = self.inference._call_compile_fx_inner(object(), (1, 2, 3))
        self.assertTrue(callable(result))
        self.assertEqual(captured["inputs"], [1, 2, 3])
        self.assertIs(captured["kwargs"]["is_inference"], True)
        cudagraphs = captured["kwargs"]["cudagraphs"]
        self.assertIsInstance(cudagraphs, BoxedBool)
        self.assertIs(cudagraphs.value, False)

    def test_other_template_uses_compiler_path_with_multiple_attrs(self):

        class LayerNormAdd(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.randn(4), requires_grad=False)

            def forward(self, x, w):
                return torch.nn.functional.layer_norm(x + w, (4, ), self.weight, self.bias, 1e-5)

        gm = torch.fx.symbolic_trace(LayerNormAdd())
        x = torch.randn(3, 4)
        w = torch.randn(3, 4)
        capture = self._install_mock_compiler(source_code="layernorm_source")
        plan = self.inference.compile_static(gm, (x, w), fallback=None)

        self.assertEqual(plan.kind, "inductor_static_inference")
        self.assertEqual(plan.plan_info["template"], "add_layernorm")
        self.assertEqual(plan.plan_info["attr_targets"], ("weight", "bias"))
        self.assertEqual(plan.plan_info["traced_placeholders"], 4)
        self.assertEqual(plan.generated_sources, ["layernorm_source"])
        aten_gm = capture["aten_gm"][0]
        self.assertFalse(any(node.op == "get_attr" for node in aten_gm.graph.nodes))

        output = plan(x, w)
        expected = torch.nn.functional.layer_norm(x + w, (4, ), gm.weight, gm.bias, 1e-5)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_plan_info_and_source_provenance(self):
        gm = self._make_symbolic_mm_bias()
        x, w = self._sample_inputs()
        self._install_mock_compiler(source_code="fake_inductor_source")
        plan = self.inference.compile_static(gm, (x, w), fallback=None)

        info = plan.plan_info
        self.assertIsInstance(info, dict)
        self.assertEqual(info["kind"], "inductor_static_inference")
        self.assertEqual(info["template"], "matmul_bias")
        self.assertEqual(info["match_kind"], "matmul_bias")
        self.assertEqual(info["attr_targets"], ("bias", ))
        self.assertEqual(info["input_count"], 2)
        self.assertEqual(info["output_count"], 1)
        self.assertTrue(info["inference_only"])
        self.assertFalse(info["fallback_supplied"])
        self.assertEqual(info["inputs"][0]["shape"], [3, 5])
        self.assertEqual(info["inputs"][1]["shape"], [5, 4])
        self.assertEqual(info["attrs"][0]["shape"], [4])
        self.assertEqual(info["generated_source_count"], 1)
        self.assertIn("float32_matmul_precision", info["precision_state"])
        self.assertEqual(plan.generated_sources, ["fake_inductor_source"])
        self.assertIn("inductor_static_inference", repr(plan))


class GuardFallbackTests(InferenceCPUTestBase):

    def test_explicit_optional_argument_uses_plan_without_rewriting_graph(self):

        class Mod(torch.nn.Module):

            def forward(self, x, w, bias=None):
                return torch.matmul(x, w) + bias

        gm = torch.fx.symbolic_trace(Mod())
        x, w = self._sample_inputs()
        bias = torch.randn(4)
        original_code, original_graph = gm.code, str(gm.graph)
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w, bias), fallback=fallback)
        self.assertTrue(_bit_equal(plan(x, w, bias), gm(x, w, bias)))
        self.assertEqual(plan.fast_calls, 1)
        self.assertEqual((gm.code, str(gm.graph)), (original_code, original_graph))
        compiled = capture["compiled"][0]
        self._assert_fallback_call(plan, fallback, compiled, x, w)
        self._assert_fallback_call(plan, fallback, compiled, x, w, None)
        self._assert_fallback_call(plan, fallback, compiled, x, w, bias=bias)

    def test_no_grad_parameter_plan_preserves_flags_and_gradient_fallback(self):

        class Mod(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4))
                self.bias = torch.nn.Parameter(torch.randn(4))

            def forward(self, x, y):
                return torch.nn.functional.layer_norm(x + y, (4, ), self.weight, self.bias, 1e-3)

        gm = torch.fx.symbolic_trace(Mod())
        x, y = torch.randn(3, 4), torch.randn(4)
        self._install_mock_compiler()
        with torch.no_grad():
            plan = self.inference.compile_static(gm, (x, y), fallback=gm)
            torch.testing.assert_close(plan(x, y), gm(x, y))
            self.assertEqual(plan.fast_calls, 1)
        self.assertTrue(gm.weight.requires_grad and gm.bias.requires_grad)
        self.assertTrue(all(s["requires_grad"] for s in plan.plan_info["attrs"]))
        output = plan(x, y)
        self.assertTrue(output.requires_grad)
        output.sum().backward()
        self.assertEqual(plan.fallback_calls, 1)
        self.assertIsNotNone(gm.weight.grad)
        torch.testing.assert_close(gm.bias.grad, torch.full_like(gm.bias, 3))
        with torch.no_grad():
            gm.bias.add_(0.5)
            torch.testing.assert_close(plan(x, y), gm(x, y))
            self.assertEqual(plan.fast_calls, 2)

    def test_no_grad_input_plan_never_bypasses_enabled_autograd(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        x.requires_grad_(True)
        self._install_mock_compiler()
        with torch.no_grad():
            plan = self.inference.compile_static(gm, (x, w), fallback=gm)
            torch.testing.assert_close(plan(x, w), gm(x, w))
        self.assertTrue(x.requires_grad)
        plan(x, w).sum().backward()
        self.assertEqual(plan.fallback_calls, 1)
        self.assertIsNotNone(x.grad)

    def test_native_precision_queries_preserve_every_public_state_guard(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        controls = (
            (torch.backends.cuda.matmul, "allow_tf32"),
            (torch.backends.cudnn, "allow_tf32"),
            (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
            (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"),
            (torch.backends.cudnn, "deterministic"),
        )
        for obj, attr in controls:
            original = getattr(obj, attr)
            if attr.endswith("reduced_precision_reduction") and hasattr(obj, attr + "_split_k"):
                original = (original, getattr(obj, attr + "_split_k"))
            try:
                # PyTorch disallows (True, False). Start at (False, True)
                # so each supported flag can change independently.
                if isinstance(original, tuple):
                    setattr(obj, attr, (False, True))
                    variants = [(True, True), (False, False)]
                else:
                    variants = [not original]
                plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
                compiled = capture["compiled"][-1]
                for value in variants:
                    setattr(obj, attr, value)
                    self.assertEqual(self.inference._precision_state(), self.inference._python_precision_state())
                    self._assert_fallback_call(plan, fallback, compiled, x, w)
            finally:
                setattr(obj, attr, original)
        old_mode = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][-1]
        try:
            for mode, warn in ((not old_mode, old_warn), (old_mode, not old_warn)):
                torch.use_deterministic_algorithms(mode, warn_only=warn)
                self.assertEqual(self.inference._precision_state(), self.inference._python_precision_state())
                self._assert_fallback_call(plan, fallback, compiled, x, w)
        finally:
            torch.use_deterministic_algorithms(old_mode, warn_only=old_warn)

    def test_shape_stride_storage_offset_and_dtype_guards(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]

        self._assert_fallback_call(plan, fallback, compiled, torch.randn(2, 5), w)
        strided = torch.randn(3, 10)[:, ::2]
        self.assertEqual(tuple(strided.shape), (3, 5))
        self.assertNotEqual(tuple(strided.stride()), tuple(x.stride()))
        self._assert_fallback_call(plan, fallback, compiled, strided, w)
        self._assert_fallback_call(plan, fallback, compiled, x.double(), w)

        # Same shape and stride, different storage offset, requires its own
        # compile-time signature.
        shifted_source = torch.randn(3, 10)
        shifted_gm = self._make_mm_bias_graph()
        shifted_plan = self.inference.compile_static(shifted_gm, (shifted_source[:, :5], w), fallback=fallback)
        shifted_compiled = capture["compiled"][-1]
        before = shifted_compiled.calls
        result = shifted_plan(shifted_source[:, 5:10], w)
        self.assertEqual(result, "fallback")
        self.assertEqual(shifted_compiled.calls, before)

    def test_requires_grad_conjugate_and_negative_view_guards(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]

        grad_input = x.clone().requires_grad_(True)
        self._assert_fallback_call(plan, fallback, compiled, grad_input, w)

        if hasattr(torch, "_neg_view"):
            negative = torch._neg_view(x)
            self.assertTrue(negative.is_neg())
            self._assert_fallback_call(plan, fallback, compiled, negative, w)

        replacement_grad_bias = torch.nn.Parameter(gm.bias.detach().clone(), requires_grad=True)
        gm.bias = replacement_grad_bias
        self._assert_fallback_call(plan, fallback, compiled, x, w)

        complex_gm = self._make_mm_bias_graph(bias=torch.randn(4, dtype=torch.complex64))
        complex_x = torch.randn(3, 5, dtype=torch.complex64)
        complex_w = torch.randn(5, 4, dtype=torch.complex64)
        complex_plan = self.inference.compile_static(complex_gm, (complex_x, complex_w), fallback=fallback)
        complex_compiled = capture["compiled"][-1]
        conjugate_view = torch.conj(complex_x)
        self.assertTrue(conjugate_view.is_conj())
        self._assert_fallback_call(complex_plan, fallback, complex_compiled, conjugate_view, complex_w)

    def test_exact_tensor_type_guard(self):

        class TensorSubclass(torch.Tensor):
            pass

        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        subclass_x = x.as_subclass(TensorSubclass)
        self.assertNotEqual(type(subclass_x), torch.Tensor)
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference.compile_static(gm, (subclass_x, w))

        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]
        self._assert_fallback_call(plan, fallback, compiled, subclass_x, w)

    def test_alias_arity_and_live_attr_guards(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs(rows=4, inner=4, width=4)
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]

        self._assert_fallback_call(plan, fallback, compiled, x, x)
        self._assert_fallback_call(plan, fallback, compiled, x)
        self._assert_fallback_call(plan, fallback, compiled, x, w, x)

        gm.bias = x[0]
        self._assert_fallback_call(plan, fallback, compiled, x, w)

        # GraphModule normally refuses to register a non-tensor buffer; bypass
        # the registration check to simulate a live attribute that mutated
        # outside the contract.
        object.__setattr__(gm, "bias", object())
        self._assert_fallback_call(plan, fallback, compiled, x, w)

        alias_gm = self._make_mm_bias_graph(bias=torch.randn(4))
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference.compile_static(alias_gm, (x, x), fallback=fallback)

    def test_precision_autocast_and_transform_mode_guards(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]

        original_precision = torch.get_float32_matmul_precision()
        changed_precision = ("high" if original_precision != "high" else "highest")
        try:
            torch.set_float32_matmul_precision(changed_precision)
            self._assert_fallback_call(plan, fallback, compiled, x, w)
        finally:
            torch.set_float32_matmul_precision(original_precision)

        with torch.autocast(device_type="cpu"):
            self._assert_fallback_call(plan, fallback, compiled, x, w)

        from torch.utils._python_dispatch import TorchDispatchMode

        class PassthroughMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with PassthroughMode():
            self._assert_fallback_call(plan, fallback, compiled, x, w)

        function_mode_type = getattr(torch.overrides, "TorchFunctionMode", None)
        if function_mode_type is not None:

            class PassthroughFunctionMode(function_mode_type):

                def __torch_function__(self, func, types_, args=(), kwargs=None):
                    return func(*args, **(kwargs or {}))

            with PassthroughFunctionMode():
                self._assert_fallback_call(plan, fallback, compiled, x, w)

    def test_forward_hooks_and_forward_code_guards(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]

        handle = gm.register_forward_hook(lambda module, inputs, output: None)
        try:
            self._assert_fallback_call(plan, fallback, compiled, x, w)
        finally:
            handle.remove()

        global_handle = torch.nn.modules.module.register_module_forward_hook(lambda module, inputs, output: None)
        try:
            self._assert_fallback_call(plan, fallback, compiled, x, w)
        finally:
            global_handle.remove()

        old_forward = gm.forward
        try:
            gm.forward = lambda x, w: x
            self._assert_fallback_call(plan, fallback, compiled, x, w)
        finally:
            gm.forward = old_forward

    def test_kwargs_fallback_and_missing_fallback(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()
        fallback = _Backstop()
        plan = self.inference.compile_static(gm, (x, w), fallback=fallback)
        compiled = capture["compiled"][0]

        call = self._assert_fallback_call(plan, fallback, compiled, x, w, training=False)
        self.assertEqual(call[1], {"training": False})

        no_fallback_plan = self.inference.compile_static(gm, (x, w), fallback=None)
        no_fallback_compiled = capture["compiled"][-1]
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            no_fallback_plan(x, w, training=False)
        self.assertEqual(no_fallback_compiled.calls, 0)
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            no_fallback_plan(torch.randn(2, 5), w)
        self.assertEqual(no_fallback_compiled.calls, 0)

    def test_compiler_errors_propagate(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        with mock.patch.object(
                self.inference,
                "_call_compile_fx_inner",
                side_effect=RuntimeError("inductor boom"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                self.inference.compile_static(gm, (x, w))
        self.assertNotIsInstance(raised.exception, self.inference.UnsupportedStaticInference)
        self.assertIn("inductor boom", str(raised.exception))


class InitialRejectionTests(InferenceCPUTestBase):

    def test_unsupported_initial_graphs_raise_specific_error(self):
        capture = self._install_mock_compiler()
        x, w = self._sample_inputs()

        no_template = torch.fx.Graph()
        node = no_template.placeholder("x")
        neg = no_template.call_function(torch.neg, (node, ))
        no_template.output(neg)
        cases = [
            (
                "no_template",
                self._graph_module(no_template, {}),
                (x, ),
            ),
        ]

        call_module_graph = torch.fx.Graph()
        cm_input = call_module_graph.placeholder("x")
        cm = call_module_graph.call_module("sub", (cm_input, ))
        call_module_graph.output(cm)
        call_module_root = torch.nn.Module()
        call_module_root.sub = torch.nn.Linear(5, 5)
        cases.append((
            "call_module",
            self._graph_module(call_module_graph, {
                "sub": call_module_root.sub,
            }),
            (x, ),
        ))

        mutation_graph = torch.fx.Graph()
        mx = mutation_graph.placeholder("x")
        mw = mutation_graph.placeholder("w")
        mm = mutation_graph.call_function(torch.mm, (mx, mw))
        relu = mutation_graph.call_method("relu_", (mm, ))
        mutation_graph.output(relu)
        cases.append((
            "mutation",
            self._graph_module(mutation_graph, {}),
            (x, w),
        ))

        non_tensor_attr_graph = torch.fx.Graph()
        nx = non_tensor_attr_graph.placeholder("x")
        nw = non_tensor_attr_graph.placeholder("w")
        nmm = non_tensor_attr_graph.call_function(torch.mm, (nx, nw))
        nattr = non_tensor_attr_graph.get_attr("scale")
        nadd = non_tensor_attr_graph.call_function(torch.add, (nmm, nattr))
        non_tensor_attr_graph.output(nadd)
        cases.append((
            "non_tensor_get_attr",
            self._graph_module(non_tensor_attr_graph, {"scale": 2}),
            (x, w),
        ))

        uncovered_graph = torch.fx.Graph()
        ux = uncovered_graph.placeholder("x")
        uw = uncovered_graph.placeholder("w")
        umm = uncovered_graph.call_function(torch.mm, (ux, uw))
        ubias = uncovered_graph.get_attr("bias")
        uadd = uncovered_graph.call_function(torch.add, (umm, ubias))
        uneg = uncovered_graph.call_function(torch.neg, (uadd, ))
        uncovered_graph.output(uneg)
        cases.append((
            "uncovered_computation",
            self._graph_module(uncovered_graph, {"bias": torch.randn(4)}),
            (x, w),
        ))

        two_regions_graph = torch.fx.Graph()
        tx = two_regions_graph.placeholder("x")
        tw = two_regions_graph.placeholder("w")
        tmm1 = two_regions_graph.call_function(torch.mm, (tx, tw))
        tb1 = two_regions_graph.get_attr("bias1")
        tadd1 = two_regions_graph.call_function(torch.add, (tmm1, tb1))
        tmm2 = two_regions_graph.call_function(torch.mm, (tx, tw))
        tb2 = two_regions_graph.get_attr("bias2")
        tadd2 = two_regions_graph.call_function(torch.add, (tmm2, tb2))
        tcombine = two_regions_graph.call_function(torch.add, (tadd1, tadd2))
        two_regions_graph.output(tcombine)
        cases.append((
            "multiple_templates",
            self._graph_module(two_regions_graph, {
                "bias1": torch.randn(4),
                "bias2": torch.randn(4),
            }),
            (x, w),
        ))

        tuple_output_graph = torch.fx.Graph()
        qx = tuple_output_graph.placeholder("x")
        qw = tuple_output_graph.placeholder("w")
        qmm = tuple_output_graph.call_function(torch.mm, (qx, qw))
        qbias = tuple_output_graph.get_attr("bias")
        qadd = tuple_output_graph.call_function(torch.add, (qmm, qbias))
        tuple_output_graph.output((qadd, qadd))
        cases.append((
            "tuple_output",
            self._graph_module(tuple_output_graph, {"bias": torch.randn(4)}),
            (x, w),
        ))

        exact_gm = self._make_mm_bias_graph()
        cases.append((
            "non_tensor_example_input",
            exact_gm,
            (x, 3),
        ))
        cases.append((
            "arity_mismatch",
            exact_gm,
            (x, ),
        ))
        square_x = torch.randn(4, 4)
        cases.append((
            "aliased_example_inputs",
            exact_gm,
            (square_x, square_x),
        ))

        for name, gm, inputs in cases:
            with self.subTest(case=name):
                with self.assertRaises(self.inference.UnsupportedStaticInference):
                    self.inference.compile_static(gm, inputs)
        self.assertEqual(capture["compiled"], [])

        with self.assertRaises(self.inference.UnsupportedStaticInference) as raised:
            self.inference.compile_static(cases[1][1], cases[1][2])
        self.assertEqual(raised.exception.reason, "call_module_not_allowed")

    def test_forward_hook_at_compile_time_is_rejected(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        handle = gm.register_forward_hook(lambda module, inputs, output: None)
        try:
            with self.assertRaises(self.inference.UnsupportedStaticInference) as raised:
                self.inference.compile_static(gm, (x, w))
        finally:
            handle.remove()
        self.assertEqual(raised.exception.reason, "forward_hooks_active")


class NativeMmAddReuseTests(InferenceCPUTestBase):

    def test_bitwise_ordinary_nan_inf_signed_zero_and_tiny(self):
        special_x = torch.tensor(
            [
                [float("nan"), float("inf"), -0.0, 1e-45],
                [float("-inf"), 0.0, 1.0, -1.0],
                [2.0, -3.0, 1e-30, -1e-30],
            ],
            dtype=torch.float32,
        )
        special_w = torch.tensor(
            [
                [1.0, -0.0, float("inf"), 1e-15],
                [0.0, 2.0, -1.0, float("nan")],
                [3.0, -2.0, 1e-15, float("inf")],
                [1e-30, 1.0, -1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        special_bias = torch.tensor(
            [float("nan"), -0.0, 1e-45, float("inf")],
            dtype=torch.float32,
        )
        gm = self._make_mm_bias_graph(bias=special_bias.clone())
        plan = self.inference._compile_native_mm_add(gm, (special_x, special_w), allow_non_cuda=True)
        self.assertEqual(plan.kind, "native_mm_add_out_reuse")
        self.assertTrue(all("python_operator:" in item for item in plan.generated_sources))
        self.assertFalse(any("triton" in item.lower() for item in plan.generated_sources))

        x_before = special_x.clone()
        w_before = special_w.clone()
        bias_before = gm.bias.clone()
        first = plan(special_x, special_w)
        second = plan(special_x, special_w)

        reference = torch.mm(special_x, special_w)
        torch.add(reference, special_bias, out=reference)
        self.assertTrue(_bit_equal(first, reference))
        self.assertTrue(_bit_equal(second, reference))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(_bit_equal(first, reference))  # first output not reused

        for output in (first, second):
            self.assertFalse(torch._C._is_alias_of(output, special_x))
            self.assertFalse(torch._C._is_alias_of(output, special_w))
            self.assertFalse(torch._C._is_alias_of(output, gm.bias))
        self.assertTrue(_bit_equal(special_x, x_before))
        self.assertTrue(_bit_equal(special_w, w_before))
        self.assertTrue(_bit_equal(gm.bias, bias_before))

        # Ordinary values follow the same bitwise contract.
        ordinary_x = torch.randn(3, 4)
        ordinary_w = torch.randn(4, 4)
        ordinary_bias = torch.randn(4)
        ordinary_gm = self._make_mm_bias_graph(bias=ordinary_bias.clone())
        ordinary_plan = self.inference._compile_native_mm_add(ordinary_gm, (ordinary_x, ordinary_w),
                                                              allow_non_cuda=True)
        ordinary = ordinary_plan(ordinary_x, ordinary_w)
        ordinary_ref = torch.mm(ordinary_x, ordinary_w)
        torch.add(ordinary_ref, ordinary_bias, out=ordinary_ref)
        self.assertTrue(_bit_equal(ordinary, ordinary_ref))

        # The live attribute is fetched every call, not captured at build time.
        replacement = torch.tensor(
            [float("inf"), 0.0, -0.0, float("-inf")],
            dtype=torch.float32,
        )
        gm.bias = replacement
        after_replacement = plan(special_x, special_w)
        replacement_ref = torch.mm(special_x, special_w)
        torch.add(replacement_ref, replacement, out=replacement_ref)
        self.assertTrue(_bit_equal(after_replacement, replacement_ref))

    def test_bias_first_operand_order_is_found_structurally(self):
        gm = self._make_mm_bias_graph(bias_first=True)
        x, w = self._sample_inputs()
        plan = self.inference._compile_native_mm_add(gm, (x, w), allow_non_cuda=True)
        output = plan(x, w)
        reference = torch.mm(x, w)
        torch.add(reference, gm.bias, out=reference)
        self.assertTrue(_bit_equal(output, reference))
        self.assertEqual(plan.plan_info["x_placeholder"], "x")
        self.assertEqual(plan.plan_info["w_placeholder"], "w")
        self.assertEqual(plan.plan_info["bias_attr_target"], "bias")

    def test_ineligible_patterns_are_rejected(self):
        x, w = self._sample_inputs(rows=3, inner=3, width=3)

        matmul_gm = self._make_mm_bias_graph(matmul=True)
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(matmul_gm, (x, w), allow_non_cuda=True)

        noncontiguous_x = torch.randn(3, 6)[:, ::2]
        self.assertFalse(noncontiguous_x.is_contiguous())
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(
                self._make_mm_bias_graph(bias=torch.randn(3)),
                (noncontiguous_x, w),
                allow_non_cuda=True,
            )

        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(
                self._make_mm_bias_graph(bias=torch.randn(2)),
                (x, w),
                allow_non_cuda=True,
            )

        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(
                self._make_mm_bias_graph(alpha=0.5),
                (x, w),
                allow_non_cuda=True,
            )

        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(
                self._make_mm_bias_graph(out_keyword=True),
                (x, w),
                allow_non_cuda=True,
            )

        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(
                self._make_mm_bias_graph(extra_attr=True),
                (x, w),
                allow_non_cuda=True,
            )

        extra_user_graph = torch.fx.Graph()
        ex = extra_user_graph.placeholder("x")
        ew = extra_user_graph.placeholder("w")
        emm = extra_user_graph.call_function(torch.mm, (ex, ew))
        ebias = extra_user_graph.get_attr("bias")
        eadd = extra_user_graph.call_function(torch.add, (emm, ebias))
        extra_user_graph.call_function(torch.neg, (emm, ))
        extra_user_graph.output(eadd)
        extra_user_gm = self._graph_module(extra_user_graph, {"bias": torch.randn(3)})
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(extra_user_gm, (x, w), allow_non_cuda=True)

    def test_native_reuse_guards_route_to_fallback(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs(rows=4, inner=4, width=4)
        fallback = _Backstop()
        plan = self.inference._compile_native_mm_add(gm, (x, w), fallback=fallback, allow_non_cuda=True)

        self.assertEqual(plan(x, x), "fallback")
        self.assertEqual(plan(x, w, training=False), "fallback")
        self.assertEqual(plan(torch.randn(3, 4), w), "fallback")

        gm.bias = x[0]
        self.assertEqual(plan(x, w), "fallback")
        self.assertEqual(len(fallback.calls), 4)

        no_fallback_plan = self.inference._compile_native_mm_add(self._make_mm_bias_graph(), (x, w),
                                                                 allow_non_cuda=True)
        with self.assertRaises(self.inference.UnsupportedStaticInference):
            no_fallback_plan(x, x)

        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(self._make_mm_bias_graph(), (x, x), allow_non_cuda=True)

    def test_production_cuda_gate_and_internal_test_hook(self):
        gm = self._make_mm_bias_graph()
        x, w = self._sample_inputs()
        capture = self._install_mock_compiler()

        compiler_plan = self.inference.compile_static(
            gm,
            (x, w),
            avoid_bias_materialization=True,
            fallback=_Backstop(),
        )
        self.assertEqual(compiler_plan.kind, "inductor_static_inference")
        self.assertEqual(
            compiler_plan.plan_info["bias_reuse_skipped_reason"],
            "requires_cuda_device",
        )
        self.assertEqual(capture["compiled"][0].calls, 0)

        native_plan = self.inference._compile_native_mm_add(gm, (x, w), allow_non_cuda=True)
        self.assertEqual(native_plan.kind, "native_mm_add_out_reuse")
        self.assertEqual(
            native_plan.plan_info["source_kind"],
            "native_python_operator_sequence",
        )
        self.assertEqual(native_plan.plan_info["operator_sequence"], (
            "torch.mm(x, w)",
            "torch.add(result, bias, out=result)",
        ))
        self.assertTrue(all("python_operator:" in item for item in native_plan.generated_sources))

        with self.assertRaises(self.inference.UnsupportedStaticInference):
            self.inference._compile_native_mm_add(gm, (x, w))


if __name__ == "__main__":
    unittest.main()
