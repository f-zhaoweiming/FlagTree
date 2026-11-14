"""Standalone TTIR pass tests; no device or launch-capture runtime is needed."""

import json
from pathlib import Path

import pytest

from triton._C.libtriton import ir, passes


@pytest.fixture
def module(tmp_path):
    source = (Path(__file__).resolve().parents[4] / "test/Triton/eliminate-intermediate-memory.mlir")
    path = tmp_path / "intermediate.mlir"
    path.write_text(source.read_text().split("// -----", 1)[0])
    context = ir.context()
    ir.load_dialects(context)
    result = ir.parse_mlir_module(str(path), context)
    result.context = context
    return result


def run_pass(module, private_arguments=(), **options):
    pm = ir.pass_manager(module.context)
    passes.ttir.add_eliminate_intermediate_memory(pm, list(private_arguments), **options)
    pm.run(module, "eliminate_intermediate_memory")
    return json.loads(module.get_operation().get_str_attr("triton.intermediate_memory_elimination"))


def body_text(module):
    # Printed location aliases precede the module header in the Python binding.
    return "\n".join(line for line in str(module).splitlines() if not line.startswith("module "))


@pytest.mark.parametrize("private_arguments, stores", [([1], 1), ([], 2)])
def test_forwarding_preserves_rounding_and_visible_stores(module, private_arguments, stores):
    report = run_pass(module, private_arguments, assume_disjoint_arguments=True)
    assert report["accepted"]
    assert report["forwarded_loads"] == 1
    assert report["eliminated_stores"] == 2 - stores
    assert report["private_arguments"] == private_arguments
    body = str(module)
    assert body.count("tt.load") == 1
    assert body.count("tt.store") == stores
    assert "arith.truncf" in body and "arith.extf" in body
    assert "arith.cmpi slt" in body  # No launch-grid-dependent mask folding.
    assert "arith.select" not in body


@pytest.mark.parametrize("private_arguments, options, reason", [
    ([1], {}, "disjoint argument storage contract"),
    ([1], {"num_ctas": 2, "assume_disjoint_arguments": True}, "one CTA and a unit cluster"),
    ([1], {"cluster_size": 2, "assume_disjoint_arguments": True}, "one CTA and a unit cluster"),
    ([1, 1], {"assume_disjoint_arguments": True}, "invalid private argument indices"),
    ([4], {"assume_disjoint_arguments": True}, "invalid private argument indices"),
    ([3], {"assume_disjoint_arguments": True}, "invalid private argument indices"),
    ([0, 1], {"assume_disjoint_arguments": True}, "read before it is written"),
])
def test_rejection_keeps_entire_body(module, private_arguments, options, reason):
    before = body_text(module)
    report = run_pass(module, private_arguments, **options)
    assert not report["accepted"]
    assert reason in report["reason"]
    assert report["private_arguments"] == []
    assert report["eliminated_stores"] == report["forwarded_loads"] == 0
    # Only the top-level report attribute may change, including on rejection.
    assert body_text(module) == before


def test_module_cta_count_cannot_be_overridden(module):
    module.set_attr("ttg.num-ctas", ir.builder(module.context).get_int32_attr(2))
    before = body_text(module)
    report = run_pass(module, [1], assume_disjoint_arguments=True, num_ctas=1)
    assert not report["accepted"]
    assert "one CTA and a unit cluster" in report["reason"]
    assert body_text(module) == before


def math_chain_module(tmp_path, stages, *, fanout=1, unrelated=0, block=256):
    """Real rounded storage boundaries, including multiple stores to one buffer."""
    source = Path(__file__).resolve().parents[4] / "test/Triton/eliminate-intermediate-memory.mlir"
    text = source.read_text().split("// -----", 1)[0]
    start, end = text.index("    %rounded ="), text.index("    tt.return")
    lines = ["    %other = arith.constant dense<7.0> : tensor<256xf16>"]
    value = "%value"
    for stage, count in enumerate(stages):
        for step in range(count):
            result = f"%math_{stage}_{step}"
            lines.append(f"    {result} = math.sin {value} : tensor<256xf32>")
            value = result
        if stage + 1 < len(stages):
            lines += [
                f"    %rounded{stage} = arith.truncf {value} : tensor<256xf32> to tensor<256xf16>",
                f"    tt.store %tmp_ptr, %rounded{stage}, %mask : tensor<256x!tt.ptr<f16>>",
                f"    %read{stage} = tt.load %tmp_ptr, %mask, %other : tensor<256x!tt.ptr<f16>>",
                f"    %extended{stage} = arith.extf %read{stage} : tensor<256xf16> to tensor<256xf32>",
            ]
            value = f"%extended{stage}"
    if fanout > 1:
        incoming = value
        for branch in range(fanout):
            result = f"%branch{branch}"
            lines.append(f"    {result} = math.cos {incoming} : tensor<256xf32>")
            if branch == 0:
                value = result
            else:
                lines.append(f"    %sum{branch} = arith.addf {value}, {result} : tensor<256xf32>")
                value = f"%sum{branch}"
    other = "%value"
    for step in range(unrelated):
        lines.append(f"    %unrelated{step} = math.sin {other} : tensor<256xf32>")
        other = f"%unrelated{step}"
    if unrelated:
        lines.append(f"    %merged = arith.addf {value}, {other} : tensor<256xf32>")
        value = "%merged"
    lines.append(f"    tt.store %out_ptr, {value}, %mask : tensor<256x!tt.ptr<f32>>")
    generated = text[:start] + "\n".join(lines) + "\n" + text[end:]
    if block != 256:
        generated = generated.replace("256", str(block))
    path = tmp_path / "math-chain.mlir"
    path.write_text(generated)
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


@pytest.mark.parametrize("stages, limit, depth, accepted", [
    ([4, 4], 8, 8, True),
    ([4, 5], 8, 9, False),
    ([16, 16], 8, 32, False),
    ([16, 16], 0, 32, True),
    ([4, 5], 9, 9, True),
    ([3, 3, 3], 8, 9, False),
    ([16, 0], 8, 0, True),
    ([0, 16], 8, 0, True),
])
def test_transcendental_forwarding_profitability(tmp_path, stages, limit, depth, accepted):
    module = math_chain_module(tmp_path, stages)
    before = body_text(module)
    report = run_pass(module, [1], assume_disjoint_arguments=True, max_transcendental_depth=limit)
    assert report["accepted"] == accepted
    assert report["max_forwarded_transcendental_depth"] == depth
    if accepted:
        assert report["private_arguments"] == [1]
        assert report["forwarded_loads"] == len(stages) - 1
        assert str(module).count("tt.store") == 1
        assert str(module).count("tt.load") == 1
        assert "arith.truncf" in str(module) and "arith.extf" in str(module)
    else:
        assert report["reason"].startswith("profitability:")
        assert report["private_arguments"] == []
        assert report["forwarded_loads"] == report["eliminated_stores"] == 0
        assert body_text(module) == before


@pytest.mark.parametrize("fanout, unrelated, depth", [(12, 0, 5), (1, 32, 0)])
def test_profitability_counts_dependencies_not_total_math(tmp_path, fanout, unrelated, depth):
    module = math_chain_module(tmp_path, [4, 0], fanout=fanout, unrelated=unrelated)
    report = run_pass(module, [1], assume_disjoint_arguments=True)
    assert report["accepted"]
    assert report["max_forwarded_transcendental_depth"] == depth
    assert report["max_forwarded_transcendental_work"] == depth * 256


def test_profitability_default_also_protects_observable_intermediates(tmp_path):
    module = math_chain_module(tmp_path, [16, 16])
    before = body_text(module)
    report = run_pass(module, assume_disjoint_arguments=True)
    assert not report["accepted"]
    assert report["max_forwarded_transcendental_depth"] == 32
    assert body_text(module) == before


def test_default_guard_boundary_and_zero_override(tmp_path):
    # Omitted max_transcendental_depth must use the documented default of 8.
    rejected = math_chain_module(tmp_path, [4, 5])
    before = body_text(rejected)
    report = run_pass(rejected, [1], assume_disjoint_arguments=True)
    assert not report["accepted"]
    assert report["reason"].startswith("profitability:")
    assert report["max_forwarded_transcendental_depth"] == 9
    assert body_text(rejected) == before

    accepted = math_chain_module(tmp_path, [4, 4])
    report = run_pass(accepted, [1], assume_disjoint_arguments=True)
    assert report["accepted"]
    assert report["max_forwarded_transcendental_depth"] == 8
    assert report["max_forwarded_transcendental_work"] == 8 * 256


def test_work_guard_large_tile_and_override(tmp_path):
    below = math_chain_module(tmp_path, [2, 2], block=512)
    report = run_pass(below, [1], assume_disjoint_arguments=True)
    assert report["accepted"]
    assert report["max_forwarded_transcendental_depth"] == 4
    assert report["max_forwarded_transcendental_work"] == 4 * 512

    exact = math_chain_module(tmp_path, [2, 2], block=1024)
    report = run_pass(exact, [1], assume_disjoint_arguments=True)
    assert report["accepted"]
    assert report["max_forwarded_transcendental_depth"] == 4
    assert report["max_forwarded_transcendental_work"] == 4 * 1024

    rejected = math_chain_module(tmp_path, [4, 4], block=1024)
    before = body_text(rejected)
    report = run_pass(rejected, [1], assume_disjoint_arguments=True)
    assert not report["accepted"]
    assert "max-transcendental-work" in report["reason"]
    assert report["max_forwarded_transcendental_depth"] == 8
    assert report["max_forwarded_transcendental_work"] == 8 * 1024
    assert body_text(rejected) == before

    override = math_chain_module(tmp_path, [4, 4], block=1024)
    report = run_pass(override, [1], assume_disjoint_arguments=True, max_transcendental_work=0)
    assert report["accepted"]
    assert report["max_forwarded_transcendental_work"] == 8 * 1024

    # work=0 disables only the work subguard; depth 9 must still reject.
    depth_reject = math_chain_module(tmp_path, [4, 5], block=256)
    report = run_pass(depth_reject, [1], assume_disjoint_arguments=True, max_transcendental_work=0)
    assert not report["accepted"]
    assert "max-transcendental-depth" in report["reason"]


def test_zero_depth_disables_profitability_only(tmp_path):
    legal = math_chain_module(tmp_path, [4, 4], block=1024)
    report = run_pass(legal, [1], assume_disjoint_arguments=True, max_transcendental_depth=0)
    assert report["accepted"]
    assert report["max_forwarded_transcendental_work"] == 8 * 1024

    # The same override must not overcome the disjoint contract.
    illegal = math_chain_module(tmp_path, [4, 4], block=1024)
    before = body_text(illegal)
    report = run_pass(illegal, [1], max_transcendental_depth=0)
    assert not report["accepted"]
    assert "disjoint argument storage contract" in report["reason"]
    assert body_text(illegal) == before


def test_zero_depth_does_not_bypass_alias_contract(module):
    before = body_text(module)
    report = run_pass(module, [1], max_transcendental_depth=0)
    assert not report["accepted"]
    assert "disjoint argument storage contract" in report["reason"]
    assert report["private_arguments"] == []
    assert body_text(module) == before
