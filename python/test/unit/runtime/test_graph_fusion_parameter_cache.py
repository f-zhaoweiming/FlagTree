"""Frozen inference tensors are mutable without a PyTorch version counter."""
import unittest

import torch

from triton.experimental.graph_fusion._model_inductor import _ParameterPack
from triton.experimental.graph_fusion._p800_model_gemm import GroupedProjectionPack, pack_column_major


class ParameterCacheTests(unittest.TestCase):
    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    @staticmethod
    def parameter(device, inference, value=1.):
        with torch.inference_mode(inference):
            return torch.nn.Parameter(torch.full((8, 16), value, device=device), requires_grad=False)

    def test_versionless_parameter_refresh_and_versioned_reuse(self):
        for device in self.devices():
            for transform in [lambda x: x.float(), lambda x: x.half(), pack_column_major]:
                with self.subTest(device=device, transform=transform):
                    regular = self.parameter(device, False)
                    inference = self.parameter(device, True)
                    counts = [0, 0]

                    def convert(value):
                        counts[value is inference] += 1
                        return transform(value)

                    pack = _ParameterPack(lambda *args: args, [0, 1], transform=convert)
                    first = pack(regular, inference)
                    second = pack(regular, inference)
                    self.assertEqual(counts, [1, 2])
                    self.assertIs(first[0], second[0])
                    with torch.inference_mode():
                        inference.add_(2)
                    actual = pack(regular, inference)
                    self.assertEqual(counts, [1, 3])
                    torch.testing.assert_close(actual[1], transform(inference), atol=0, rtol=0)
                    self.assertIs(actual[0], first[0])
                    regular.add_(3)
                    actual = pack(regular, inference)
                    self.assertEqual(counts, [2, 4])
                    torch.testing.assert_close(actual[0], transform(regular), atol=0, rtol=0)

    def test_group_refreshes_if_any_source_has_no_version(self):
        for device in self.devices():
            for flags in [(False, False), (True, False), (False, True), (True, True)]:
                with self.subTest(device=device, inference=flags):
                    values = [self.parameter(device, flag, i + 1.) for i, flag in enumerate(flags)]
                    pack = GroupedProjectionPack(lambda *args: args[0], [(0, 1)])
                    first = pack(*values)
                    second = pack(*values)
                    self.assertEqual(first is second, not any(flags))
                    for index in range(2):
                        with torch.inference_mode():
                            values[index].add_(3)
                        torch.testing.assert_close(pack(*values), torch.cat(values), atol=0, rtol=0)
                    values[0] = self.parameter(device, flags[0], 9.)
                    torch.testing.assert_close(pack(*values), torch.cat(values), atol=0, rtol=0)

    def test_non_frozen_parameters_still_rejected(self):
        for device in self.devices():
            for value in [torch.ones(8, 16, device=device),
                          torch.nn.Parameter(torch.ones(8, 16, device=device))]:
                with self.subTest(device=device, value_type=type(value)):
                    for pack in [_ParameterPack(lambda x: x, [0]),
                                 GroupedProjectionPack(lambda x: x, [(0,)])]:
                        with self.assertRaisesRegex(RuntimeError, 'frozen Parameters'):
                            pack(value)


if __name__ == '__main__':
    unittest.main()
