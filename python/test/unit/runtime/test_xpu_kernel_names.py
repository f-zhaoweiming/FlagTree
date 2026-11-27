"""Long descriptive entry names must compile and execute on XPU."""
import importlib.util

import pytest
import torch
import triton
from triton.runtime import driver


def test_long_entry_names_preserve_execution_and_identity(tmp_path):
    if not torch.cuda.is_available() or type(driver.active).__module__ != "triton.backends.xpu.driver":
        pytest.skip("XPU backend required")
    # Equal prefixes must not collapse distinct kernels into one cache entry.
    prefix = "kernel_" + "descriptive_" * 24
    cases = [("short_kernel", 1), (prefix + "first", 3), (prefix + "second", 7)]
    x = torch.arange(128, dtype=torch.float32, device="cuda")
    names = []
    for index, (name, increment) in enumerate(cases):
        path = tmp_path / f"kernel_{index}.py"
        path.write_text(
            "import triton\nimport triton.language as tl\n@triton.jit\n"
            f"def {name}(X, Y):\n"
            "    i = tl.arange(0, 128)\n"
            f"    tl.store(Y + i, tl.load(X + i) + {increment})\n"
        )
        spec = importlib.util.spec_from_file_location(f"xpu_names_{index}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = getattr(module, name)
        for offset in (0, 11):
            actual_input = x + offset
            output = torch.empty_like(x)
            compiled = kernel[(1,)](actual_input, output, num_warps=4)
            torch.testing.assert_close(output, actual_input + increment, rtol=0, atol=0)
        names.append(compiled.metadata.name)
        assert len(compiled.metadata.name.encode("utf-8")) <= 128
        if len(name) > 128:
            assert compiled.metadata.original_name == name
    assert len(set(names)) == len(cases)
    assert names[0] == cases[0][0]
