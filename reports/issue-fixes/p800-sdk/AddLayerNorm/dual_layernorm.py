"""Direct FP16 dual-output KL3 kernel for contiguous two-dimensional inference."""
import collections
import ctypes as C
import hashlib
import importlib.util
import functools
import math
import sysconfig
import tempfile
import os
from pathlib import Path
import subprocess

import torch


class Descriptor(C.Structure):
    # KL3 Cluster descriptor from FlagTree's XPU driver.
    _fields_ = [('type_place', C.c_uint32), ('code', C.c_uint64), ('size', C.c_uint32),
                ('pc', C.c_uint32), ('params', C.c_uint32), ('hash', C.c_uint64),
                ('name', C.c_char_p), ('private', C.c_void_p), ('printf_offset', C.c_uint64)]


@functools.lru_cache(maxsize=1)
def _checked_entry(source, runtime, build_root):
    """Build against the active Torch ABI once. The entry reads live tensor metadata on every call."""
    path = Path(tempfile.mkdtemp(prefix='checked-launch-', dir=build_root))
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run(['cc', '-fPIC', '-O2', '-c', source/'launch.c', '-o', path/'launch.o'], check=True)
    target = path / ('_sdk04_checked' + sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['c++', '-std=c++17', '-shared', '-fPIC', '-O2',
                    '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
                    '-I'+sysconfig.get_path('include'), '-I'+str(torch_root/'include'),
                    '-I'+str(torch_root/'include/torch/csrc/api/include'), source/'checked_launch.cpp',
                    path/'launch.o', runtime, '-L'+str(torch_root/'lib'), '-ltorch_python',
                    '-ltorch_cpu','-lc10','-o',target], check=True, timeout=120)
    spec = importlib.util.spec_from_file_location('_sdk04_checked', target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.launch


class DualLayerNorm:
    def __init__(self, build_dir, columns, round_residual):
        if columns not in (64, 1024, 2048) or type(round_residual) is not bool:
            raise ValueError("unsupported width or residual rounding policy")
        self.columns = columns
        tile = 1024 if columns == 2048 and not round_residual else min(columns, 512)
        self.rows_per_cluster = 64 // (columns // tile)
        source = Path(__file__).resolve().parent
        backend = Path('/opt/hsp/triton/backends/xpu')
        clang = backend/'llvm19/bin/clang++'
        converter = backend/'xpu3/bin/xpu3-elfconv-triton'
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = ':'.join((str(backend/'llvm19/__lib'), str(backend/'xpu3/__lib'),
                                           env.get('LD_LIBRARY_PATH', '')))
        # The caller supplies a fresh, private build directory for each specialization.
        path = Path(build_dir)
        path.mkdir()
        def run(args, **kwargs):
            return subprocess.run(args, cwd=path, env=env, check=True, timeout=60, **kwargs)
        run([clang, '-x', 'xpu', '--xpu-arch=xpu3', '--xpu-static-lib', '-O2',
             f'-DCOLS={columns}', f'-DROUND_RESIDUAL={int(round_residual)}',
             '-c', source/'dual_layernorm.xpu', '-o', 'kernel.o'])
        run([converter, 'kernel.o', 'kernel.bin', converter.parent])
        blob = (path/'kernel.bin').read_bytes()
        if not blob or len(blob) % 4 or C.sizeof(Descriptor) != 64:
            raise RuntimeError("unexpected KL3 binary or runtime descriptor")
        words = '\n'.join(' '.join(f'0x{b:02x}' for b in blob[i:i+4]) for i in range(0,len(blob),4))
        asm = run([backend/'llvm19/bin/llvm-mc', '--triple=xpu3-baidu-none-gnu', '--mcpu=xpu3',
                   '--mattr=+cdnn3', '--disassemble'], input=words+'\n', capture_output=True, text=True)
        (path/'kernel.asm').write_text(asm.stdout)
        self.binary_info = dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                                unknown_words=asm.stderr.count('invalid instruction encoding'),
                                mnemonics=dict(collections.Counter(line.strip().split()[0]
                                    for line in asm.stdout.splitlines() if line.strip() and not line.strip().startswith('.'))))
        self.buffer = C.create_string_buffer(blob, len(blob))
        checksum = sum(b if b < 128 else b-256 for b in blob) & 0xffffffff
        name = f'dual_ln_{columns}_{int(round_residual)}'.encode()
        self.descriptor = Descriptor(0, C.addressof(self.buffer), len(blob), 0, 0, checksum, name, None, 0)
        # Link the same runtime that the current Torch process actually loaded.
        libraries = {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libxpurt.so' in line and line.split()[-1].startswith('/')}
        if len(libraries) != 1:
            raise RuntimeError(f'expected one active libxpurt, found {libraries}')
        runtime = libraries.pop()
        run(['cc', '-shared', '-fPIC', '-O2', source/'launch.c', runtime,
             '-Wl,-z,defs', '-o', 'launch.so'])
        self.launch = C.CDLL(str(path/'launch.so')).launch_dual_ln
        self.launch.argtypes = [C.POINTER(Descriptor), C.c_int] + [C.c_void_p]*6 + [C.c_int, C.c_float]
        self.launch.restype = C.c_int
        self._checked = _checked_entry(source, runtime, path.parent)
        capsule = C.pythonapi.PyCapsule_New
        capsule.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
        capsule.restype = C.py_object
        self._capsule = capsule(C.addressof(self.descriptor), b'sdk04.cluster', None)

    def __call__(self, x, y, weight, bias, output, residual, eps=1e-5):
        device = torch.cuda.current_device()
        stream = torch.cuda.current_stream(device).cuda_stream
        value = self._checked(self._capsule, self.columns, self.rows_per_cluster, device, stream, eps, x, y, weight, bias, output, residual)
        if value is not NotImplemented:
            return value
        # Preserve the original checks and Python metadata semantics for Tensor subclasses.
        tensors = (x, y, weight, bias, output, residual)
        if (x.ndim != 2 or x.shape[1] != self.columns or x.shape != y.shape
                or x.shape != output.shape or x.shape != residual.shape
                or weight.shape != (self.columns,) or weight.shape != bias.shape
                or not (0 < x.shape[0] < 2**31 // self.columns - 64)
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError("unsupported shape or epsilon")
        if (not x.is_cuda or torch.cuda.current_device() != x.device.index
                or torch.cuda.current_stream(x.device).cuda_stream != 0
                or any(t.device != x.device or t.dtype != torch.float16 or not t.is_contiguous()
                       or t.requires_grad or t.is_neg() or t.is_conj() for t in tensors)):
            raise ValueError("expected contiguous FP16 inference tensors on the active default stream")
        pointers = [t.data_ptr() for t in tensors]
        sizes = [t.numel()*2 for t in tensors]
        if any(p % 64 for p in pointers):
            raise ValueError("KL3 buffers require 64-byte alignment")
        for i in (4, 5):
            for j in range(i):
                if pointers[i] < pointers[j]+sizes[j] and pointers[j] < pointers[i]+sizes[i]:
                    raise ValueError("output overlaps another buffer")
        groups_per_core_block = self.rows_per_cluster
        clusters = min(12, (x.shape[0]+groups_per_core_block-1)//groups_per_core_block)
        rc = self.launch(C.byref(self.descriptor), clusters, *pointers, x.shape[0], eps)
        if rc:
            raise RuntimeError(f'KL3 launch failed with status {rc}')
        return output, residual
