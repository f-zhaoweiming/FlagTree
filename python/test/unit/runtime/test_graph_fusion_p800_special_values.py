"""Compiler-boundary invariants; device precision/performance gates are separate."""
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[3] / 'triton/experimental/graph_fusion/_p800_special_values.py'
spec = importlib.util.spec_from_file_location('p800_special_values_under_test', PATH)
special = importlib.util.module_from_spec(spec)
spec.loader.exec_module(special)
TY = 'tensor<32x128xf32>'


def gelu():
    return '\n'.join([
        f'%one = arith.constant dense<1.000000e+00> : {TY}',
        f'%half = arith.constant dense<5.000000e-01> : {TY}',
        f'%scale = arith.constant dense<0.797884583> : {TY}',
        f'%cubic = arith.constant dense<4.471500e-02> : {TY}',
        f'%x = arith.extf %low : tensor<32x128xbf16> to {TY}',
        f'%a = arith.mulf %x, %half : {TY}',
        f'%b = arith.mulf %x, %x : {TY}',
        f'%c = arith.mulf %b, %x : {TY}',
        f'%d = arith.mulf %c, %cubic : {TY}',
        f'%e = arith.addf %x, %d : {TY}',
        f'%f = arith.mulf %e, %scale : {TY}',
        f'%t = tt.extern_elementwise %f {{pure = true, symbol = "_ZN3xpu5tanhfEf"}} : ({TY}) -> {TY}',
        f'%p = arith.addf %t, %one : {TY}',
        f'%y = arith.mulf %a, %p : {TY}',
        f'%low_y = arith.truncf %y : {TY} to tensor<32x128xbf16>',
    ])


def silu():
    return '\n'.join([
        f'%one = arith.constant dense<1.000000e+00> : {TY}',
        f'%zero = arith.constant dense<0.000000e+00> : {TY}',
        f'%n = arith.subf %zero, %x : {TY}',
        f'%e = math.exp %n : {TY}',
        f'%d = arith.addf %e, %one : {TY}',
        f'%s = arith.divf %one, %d : {TY}',
        f'%y = arith.mulf %x, %s : {TY}',
    ])


class RewriteBoundaries(unittest.TestCase):
    def test_vector_pointwise_repair_requires_explicit_opt_in_and_keeps_cast(self):
        source = (f'%product = arith.mulf %a, %b : {TY}\n'
                  f'%low = arith.truncf %product : {TY} to tensor<32x128xbf16>')
        self.assertIsNone(special.single_reduction_multiply(source)[1])
        changed, record = special.single_reduction_multiply(source, allow_pointwise=True)
        self.assertEqual(record['products'], 1)
        self.assertEqual(changed.count('arith.mulf '), 1)
        self.assertIn(source.splitlines()[-1], changed)
        self.assertIsNone(special.single_reduction_multiply(
            source + f'\n%other = arith.mulf %c, %d : {TY}', allow_pointwise=True)[1])
    def test_bf16_boundaries_survive_gelu_rewrite(self):
        before = gelu()
        changed, record = special.activation_text(before)
        self.assertEqual(record['kind'], 'gelu_tanh')
        changed, repairs = special.gelu_final_specials(changed, record)
        self.assertEqual(len(repairs), 1)
        for line in before.splitlines():
            if 'arith.extf' in line or 'arith.truncf' in line:
                self.assertIn(line, changed)
        self.assertIn(f'raw = arith.mulf %a, %p : {TY}', changed)
        self.assertIn('arith.cmpi ne, %f06_gelu_special_bits, %f06_gelu_special_inf', changed)

    def test_standalone_tanh_output_is_not_rewritten(self):
        source = gelu() + '\n tt.store %other, %t : tensor<32x128x!tt.ptr<f32>>'
        self.assertIsNone(special.activation_text(source)[1])

    def test_other_polynomial_keeps_default(self):
        source = gelu().replace('4.471500e-02', '1.000000e-01')
        self.assertIsNone(special.activation_text(source)[1])

    def test_silu_keeps_original_final_input(self):
        changed, record = special.activation_text(silu())
        self.assertEqual(record['kind'], 'silu')
        self.assertIn(f'%y = arith.mulf %x, %s : {TY}', changed)
        self.assertIn('%n = arith.subf %zero, %f06_fast_sigmoid_clip', changed)

    def test_unrelated_multiply_cannot_lose_repairs(self):
        source = silu() + f'\n%unrelated = arith.mulf %unknown0, %unknown1 : {TY}'
        self.assertIsNone(special.activation_text(source)[1])

    def test_native_silu_reuses_dead_first_input(self):
        source = (f'%factor = sdnn.ew minimum %scratch src0 %scratch c %limit lut_func sigmoid '
                  f'{{sdnn.layout = "CR_RowMajor"}} : ({TY}, {TY}, f32) -> {TY}\n'
                  f'%y = sdnn.ew mul %x src0 %x src1 %factor {{sdnn.layout = "CR_RowMajor"}} '
                  f': ({TY}, {TY}, {TY}) -> {TY}\n')
        changed, record = special.order_output_multiply(source, 'silu', before_bufferization=True)
        self.assertTrue(record['swapped'])
        self.assertIn('sdnn.ew mul %factor src0 %factor src1 %x', changed)

    def test_scheduled_second_input_alias_is_rejected(self):
        ty = 'memref<32x128xf32>'
        source = (f'sdnn.ew lut %factor src0 %scratch lut_func sigmoid {{}} : ({ty}, {ty}) -> ()\n'
                  f'sdnn.ew mul %x src0 %factor src1 %x {{}} : ({ty}, {ty}, {ty}) -> ()\n')
        with self.assertRaisesRegex(AssertionError, 'aliases second'):
            special.order_output_multiply(source, 'silu')

    def test_single_product_and_unchanged_reduction(self):
        reduce = ('%z = "tt.reduce"(%product) <{axis = 1 : i32}> ({\n'
                  '^bb0(%l: f32, %r: f32):\n'
                  '  %v = arith.addf %l, %r : f32\n  tt.reduce.return %v : f32\n})')
        before = f'%product = arith.mulf %raw, %s : {TY}\n' + reduce
        changed, record = special.single_reduction_multiply(before)
        self.assertEqual(record['products'], 1)
        self.assertEqual(changed.count('arith.mulf '), 1)
        self.assertIn(f'arith.mulf %raw, %s : {TY}', changed)
        self.assertTrue(changed.endswith(reduce))
        for other in (before.replace('arith.addf', 'arith.maximumf'), before + '\n' + before):
            self.assertEqual(special.single_reduction_multiply(other), (other, None))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'scope test requires the offline Torch environment')
class ScopeRestoration(unittest.TestCase):
    def test_default_identity_and_exception_restoration(self):
        from triton.experimental.graph_fusion import _p800_compiler as compiler
        from triton.experimental.graph_fusion import _p800_native_activation as activation
        from triton.experimental.graph_fusion import _resource_codegen as codegen
        original = (compiler.policy_key, activation.prepare, activation.install_stage,
                    activation.preserve_multiply_specials,codegen.annotate_source)
        key = compiler.policy_key()
        with special.experimental_scope('on'):
            self.assertEqual(compiler.policy_key(), key)
            self.assertIs(activation.prepare, original[1])
        with self.assertRaisesRegex(ValueError, 'deliberate'):
            with special.experimental_scope():
                self.assertNotEqual(compiler.policy_key(), key)
                with self.assertRaisesRegex(RuntimeError, 'nested'):
                    with special.experimental_scope():
                        pass
                raise ValueError('deliberate')
        self.assertEqual(original, (compiler.policy_key, activation.prepare, activation.install_stage,
                                    activation.preserve_multiply_specials,codegen.annotate_source))
        self.assertEqual(compiler.policy_key(), key)

    def test_deferred_compilation_retains_policy(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from triton.experimental.graph_fusion import _resource_codegen as codegen
        def native_factory(*args,**kwargs):
            return lambda fn: SimpleNamespace(_precompile_config=lambda: special._SCOPED.get())
        for name in ('optimized', 'parallel', 'vector_multiply'):
            with self.subTest(mode=name):
                with patch.object(codegen.ResourceHeuristics,'__getattr__',lambda self,name:native_factory):
                    h=special.SpecialValueHeuristics(codegen.ResourceHeuristics('sdnn'),name,special._policy_identity(name))
                    tuner=h.persistent_reduction()(lambda:None)
                self.assertIsNone(special._SCOPED.get())
                self.assertEqual(tuner._precompile_config(),name)
                self.assertIsNone(special._SCOPED.get())

    def test_jit_body_key_contains_policy(self):
        from triton.experimental.graph_fusion import _resource_codegen as codegen
        source='''import triton
@triton_heuristics.pointwise()
@triton.jit
def kernel(x):
    value = x
'''
        with special.experimental_scope():
            changed=codegen.annotate_source(source,'sdnn')
            pending=codegen.annotate_source(source.replace('def kernel(', 'def Placeholder.KERNEL_NAME('),'sdnn')
        self.assertIn('SpecialValueHeuristics(',changed)
        self.assertIn('def kernel(x):\n    # P800 experimental special-value policy:',changed)
        self.assertIn('def Placeholder.KERNEL_NAME(x):\n    # P800 experimental special-value policy:',pending)
        wrapper="kernel = async_compile.triton('kernel', "+repr(changed)+", device_str='cuda')"
        self.assertEqual(len(codegen.audit_wrapper(wrapper,'sdnn')['generated_kernel_sources']),1)
        with self.assertRaisesRegex(RuntimeError,'explicit resource'):
            codegen.audit_wrapper(wrapper,'cluster')


if __name__ == '__main__':
    unittest.main()
