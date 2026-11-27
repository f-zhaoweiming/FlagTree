"""Explicitly select packaged F06 code for this Python process."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading

__version__ = '0.3.81+v51e.inductor64'
_LOCK = threading.RLock()
_ROOT = Path(__file__).resolve().parent


def manifest():
    return json.loads((_ROOT / 'manifest.json').read_text())


def verify():
    """Check all bundled computation sources without importing Torch or a device."""
    expected = manifest()['files']
    bad = [name for name, digest in expected.items()
           if not (_ROOT / name).is_file()
           or hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() != digest]
    if bad:
        raise RuntimeError('F06 package integrity check failed: ' + ', '.join(bad))
    return len(expected)


def activate():
    """Load the packaged implementation; leave installed Triton files untouched.

    Existing Torch hooks and generated modules use the canonical Triton import
    path. Selecting it before first import also avoids duplicate custom-op
    registrations. Mixing two F06 versions in one process is rejected.
    """
    with _LOCK:
        package_dir = _ROOT / '_vendor'
        for name, module in tuple(sys.modules.items()):
            if name == 'triton.experimental.graph_fusion' or name.startswith('triton.experimental.graph_fusion.'):
                path = getattr(module, '__file__', None)
                if path is None or package_dir not in Path(path).resolve().parents:
                    raise RuntimeError('Another F06 implementation is already loaded; import f06 and call activate() in a fresh process before torch.compile.')
        verify()
        experimental = importlib.import_module('triton.experimental')
        location = str(package_dir)
        if location not in experimental.__path__:
            experimental.__path__.insert(0, location)
        module = importlib.import_module('triton.experimental.graph_fusion')
        if package_dir not in Path(module.__file__).resolve().parents:
            raise RuntimeError('F06 package selection did not take effect')
        return module


def compile(model, example_inputs=None, *, tolerance=None, environment_key=None,
            single_kernel=None, audit=None, **kwargs):
    """Prepare the original F06 backend, with cold calibration on first call.

    Unsupported models and compiler failures propagate. This API never silently
    converts an unsuccessful compile into a claimed fusion success.
    """
    import torch
    core = activate()
    if tolerance is None:
        values = list(example_inputs or ())
        if hasattr(model, 'parameters'):
            values += list(model.parameters())
        dtypes = {v.dtype for v in values if isinstance(v, torch.Tensor) and v.is_floating_point()}
        if len(dtypes) != 1:
            raise ValueError('Provide example_inputs with one floating dtype, or an explicit tolerance')
        dtype, = dtypes
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('F06 supports FP32, FP16 and BF16 inference')
        tolerance = dict(atol=.0003, rtol=.0003) if dtype == torch.float32 else dict(
            atol=.03 if dtype == torch.bfloat16 else .003, rtol=.003)
    records = [] if audit is None else audit
    backend = core.autotune_resource_backend(
        tolerance=tolerance, environment_key=environment_key or 'f06-p800/' + __version__,
        single_kernel=single_kernel, audit=records)
    result = core.compile(model, example_inputs, backend=backend, **kwargs)
    result._f06_audit = records
    return result


def doctor():
    result = {'version': __version__, 'package': str(_ROOT), 'verified_files': verify(),
              'python': sys.version.split()[0], 'tested_python': '3.10',
              'tested_torch': '2.9.0+cu129', 'device_execution_checked': False}
    for name in ('torch', 'triton', 'flagtree'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    result['requirements'] = manifest()['requirements']
    return result


def compile_model(model, **kwargs):
    """Automatically fuse eligible regions and retain other original operators."""
    return activate().compile_model(model, **kwargs)


def compile_pipeline(tensor_model, **kwargs):
    """Compile a complete tensor forward with explicit, uncached preprocessing."""
    return activate().compile_pipeline(tensor_model, **kwargs)


def select_model_routes(routes, **kwargs):
    """Validate and time complete application call paths before choosing one."""
    return activate().select_model_routes(routes, **kwargs)


def model_candidate_routes(model, **kwargs):
    return activate().model_candidate_routes(model, **kwargs)


def __getattr__(name):
    if name in ('autotune_resource_backend', 'resource_backend', 'calibrate_resources',
                'p800_inductor_options', 'default_registry'):
        return getattr(activate(), name)
    raise AttributeError(name)


__all__ = ['compile', 'compile_model', 'compile_pipeline', 'select_model_routes', 'model_candidate_routes', 'activate', 'verify', 'doctor', 'manifest', '__version__',
           'autotune_resource_backend', 'p800_inductor_options']
