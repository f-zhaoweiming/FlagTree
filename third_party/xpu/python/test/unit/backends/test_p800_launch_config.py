"""Host-only tests for explicit XPU launch topology options.

No libtriton, XPU driver, or P800 is needed.  Minimal module stubs let the
backend modules be imported so generated launcher source and option hashing
are tested, not just the pure validation helper.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import types
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[6]
BACKEND_DIR = ROOT / "third_party" / "xpu" / "backend"


def _module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    """Install only what the backend modules touch at import time."""
    triton = _module("triton")
    runtime = _module("triton.runtime")
    build = _module("triton.runtime.build")
    cache = _module("triton.runtime.cache")
    backends = _module("triton.backends")
    backend_compiler = _module("triton.backends.compiler")
    backend_driver = _module("triton.backends.driver")
    llvm19 = _module("triton.backends.llvm19_toolchain")
    c_mod = _module("triton._C")
    libtriton = _module("triton._C.libtriton")

    build._build = lambda *a, **k: None
    cache.get_cache_manager = lambda *a, **k: None
    backend_compiler.BaseBackend = object
    backend_compiler.GPUTarget = object
    backend_driver.GPUDriver = object
    llvm19.get_llvm19_bin_dir = lambda *a, **k: "/nonexistent-llvm19"
    llvm19.install_intrinsic_attr_table = lambda *a, **k: None
    llvm19.llc_flags = lambda *a, **k: []
    llvm19.llir_to_asm = lambda *a, **k: ""
    llvm19.llir_to_object = lambda *a, **k: b""
    llvm19.codegen_key = lambda *a, **k: "test"
    llvm19.LLC_ALWAYS_FLAGS = ()
    for name in ("ir", "passes", "xpu", "llvm"):
        setattr(libtriton, name, types.SimpleNamespace())

    triton.runtime = runtime
    triton.backends = backends
    triton._C = c_mod
    runtime.build = build
    runtime.cache = cache
    backends.compiler = backend_compiler
    backends.driver = backend_driver
    backends.llvm19_toolchain = llvm19
    c_mod.libtriton = libtriton

    # Make `triton.backends.xpu` a real package rooted at the snapshot backend.
    xpu_pkg = _module("triton.backends.xpu")
    xpu_pkg.__path__ = [str(BACKEND_DIR)]
    backends.xpu = xpu_pkg


_AFFECTED_MODULES = (
    "triton",
    "triton.runtime",
    "triton.runtime.build",
    "triton.runtime.cache",
    "triton.backends",
    "triton.backends.compiler",
    "triton.backends.driver",
    "triton.backends.llvm19_toolchain",
    "triton._C",
    "triton._C.libtriton",
    "triton.backends.xpu",
    "triton.backends.xpu.driver",
    "triton.backends.xpu.compiler",
    "triton.backends.xpu.launch_config",
    "launch_config",
)
_MISSING = object()
_PATH_BEFORE = list(sys.path)
_MODULE_BEFORE = {name: sys.modules.get(name, _MISSING) for name in _AFFECTED_MODULES}
_TRITON_KEYS_BEFORE = {name for name in sys.modules
                       if name == "triton" or name.startswith("triton.")}


def _restore_import_state():
    sys.path[:] = _PATH_BEFORE
    for name, previous in _MODULE_BEFORE.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


try:
    _install_stubs()
    sys.path.insert(0, str(BACKEND_DIR))
    lc = importlib.import_module("launch_config")
    driver = importlib.import_module("triton.backends.xpu.driver")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiler = importlib.import_module("triton.backends.xpu.compiler")
finally:
    # Keep the loaded module objects for the tests, but never leave the fake
    # triton package or the extra sys.path entry visible to other test modules.
    _restore_import_state()


def fake_metadata(arch=3, is_sdnn=False, clusters=-1, cores=-1):
    return SimpleNamespace(
        xpu_arch=arch, is_sdnn=is_sdnn, ewtable="", tensor_args=[],
        name="k", launch_clusters=clusters, launch_cores=cores)


def launcher_src(**kwargs):
    return driver.make_launcher({}, {"x": "*fp32"}, {"ids_of_const_exprs": ()},
                                fake_metadata(**kwargs))


def launcher_key(src):
    return hashlib.sha256((src + driver.dirname).encode("utf-8")).hexdigest()


class ImportIsolationTests(unittest.TestCase):
    def test_backend_stub_import_does_not_leak(self):
        self.assertEqual(sys.path, _PATH_BEFORE)
        for name, before in _MODULE_BEFORE.items():
            if before is _MISSING:
                self.assertNotIn(name, sys.modules, name)
            else:
                self.assertIs(sys.modules.get(name), before, name)
        after = {name for name in sys.modules
                 if name == "triton" or name.startswith("triton.")}
        self.assertEqual(after, _TRITON_KEYS_BEFORE)


class LaunchSpecTests(unittest.TestCase):
    def test_defaults_are_unchanged(self):
        self.assertEqual(lc.get_xpu_spec(2, False), (8, 64))
        self.assertEqual(lc.get_xpu_spec(3, False), (12, 64))
        self.assertEqual(lc.get_xpu_spec(3, True), (12, 8))
        self.assertEqual(lc.get_xpu_spec(4, False), (12, 64))
        self.assertEqual(lc.get_xpu_spec(4, True), (6, 8))
        self.assertEqual(driver.get_xpu_spec(3), (12, 64))
        self.assertEqual(lc.resolve_launch_config(3, False), (12, 64))
        self.assertEqual(lc.resolve_launch_config(3, True), (12, 8))

    def test_explicit_downscaling_and_errors(self):
        self.assertEqual(lc.resolve_launch_config(3, False, 6, 32), (6, 32))
        self.assertEqual(lc.resolve_launch_config(3, True, 6, 8), (6, 8))
        with self.assertRaisesRegex(ValueError, "only validated"):
            lc.resolve_launch_config(3, True, 12, 4)
        with self.assertRaisesRegex(ValueError, "only validated"):
            lc.resolve_launch_config(3, True, 12, 2)
        bad = [
            (3, False, 0, 64), (3, False, 13, 64), (3, False, 12, 0),
            (3, False, 12, 65), (3, True, 12, 9), (3, True, 13, 8),
            (3, False, True, 64), (3, False, 2.0, 64),
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                lc.resolve_launch_config(*args)


class OptionsAndMetadataTests(unittest.TestCase):
    def test_options_hash_separates_launch_settings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = compiler.XPUOptions(arch=3)
            other = compiler.XPUOptions(arch=3, launch_clusters=6, launch_cores=32)
            sdnn = compiler.XPUOptions(arch=3, is_sdnn=True)
        self.assertEqual(base.launch_clusters, -1)  # sentinel survives reparse
        self.assertNotEqual(base.hash(), other.hash())
        self.assertEqual(compiler._record_effective_launch({"is_sdnn": False}, base),
                         None)
        meta = {"is_sdnn": True}
        compiler._record_effective_launch(meta, sdnn)
        self.assertEqual((meta["launch_clusters"], meta["launch_cores"]), (12, 8))

    def test_cluster_runtime_only_mismatch_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compiler.XPUOptions(arch=3, launch_clusters=12, launch_cores=32)
        self.assertTrue(any("core_num" in str(w.message) for w in caught))

    def test_invalid_options_fail_at_parse(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                compiler.XPUOptions(arch=3, launch_cores=65)
            with self.assertRaises(ValueError):
                compiler.XPUOptions(arch=3, is_sdnn=True, launch_cores=9)
            with self.assertRaisesRegex(ValueError, "only validated"):
                compiler.XPUOptions(arch=3, is_sdnn=True, launch_cores=4)
            with self.assertRaises(ValueError):
                compiler.XPUOptions(arch=3, launch_clusters=13)


class LauncherGenerationTests(unittest.TestCase):
    def test_defaults_embed_legacy_pair(self):
        cluster = launcher_src()
        sdnn = launcher_src(is_sdnn=True)
        self.assertIn("int nclusters = 12;", cluster)
        self.assertIn("int ncores = 64;", cluster)
        self.assertIn("int nclusters = 12;", sdnn)
        self.assertIn("int ncores = 8;", sdnn)
        self.assertNotIn("= -1;", cluster)
        self.assertNotEqual(launcher_key(cluster), launcher_key(sdnn))

    def test_explicit_pair_reaches_c_and_cache_key(self):
        src = launcher_src(clusters=6, cores=32)
        self.assertIn("int nclusters = 6;", src)
        self.assertIn("int ncores = 32;", src)
        self.assertNotEqual(launcher_key(src), launcher_key(launcher_src()))
        self.assertNotEqual(launcher_key(src),
                            launcher_key(launcher_src(clusters=6, cores=64)))

    def test_invalid_pair_rejected_before_source_generation(self):
        with self.assertRaises(ValueError):
            launcher_src(is_sdnn=True, cores=9)
        with self.assertRaises(ValueError):
            launcher_src(clusters=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
