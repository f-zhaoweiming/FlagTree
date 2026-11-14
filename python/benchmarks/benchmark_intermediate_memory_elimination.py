"""Isolated A/B benchmark for triton-eliminate-intermediate-memory.

prepare: compile identical TTIR with/without the pass to PTX (no GPU needed).
run: assemble both variants identically and measure on CUDA using PyTorch and
     the CUDA driver API; no remote Triton installation is required.
analyze: summarize paired measurements across independent process sessions.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time


WORKLOADS = ("pointwise", "chain4", "fanout", "compute", "reduction", "retained", "control")
SIZES = (1037, 1048613, 16777229)
BLOCKS = (256, 1024)
DTYPES = ("fp16", "fp32")


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_source():
    # This function is imported only by the preparation command.
    import triton
    import triton.language as tl

    @triton.jit
    def memory_kernel(X, T0, T1, T2, Y, N, BLOCK: tl.constexpr, KIND: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask, other=0).to(tl.float32)
        v = x * 0.5 + 0.125
        if KIND == "control":
            y = v * v + 0.25
        elif KIND == "chain4":
            for _ in tl.static_range(4):
                v = v * 0.75 + 0.125
                tl.store(T0 + offsets, v, mask)
                v = tl.load(T0 + offsets, mask, other=0).to(tl.float32)
                v = v * v
            y = v
        elif KIND == "fanout":
            tl.store(T0 + offsets, v, mask)
            a = tl.load(T0 + offsets, mask, other=0).to(tl.float32)
            tl.store(T1 + offsets, a * a + 0.125, mask)
            b = tl.load(T0 + offsets, mask, other=0).to(tl.float32)
            tl.store(T2 + offsets, b * 0.25 - 0.125, mask)
            a = tl.load(T1 + offsets, mask, other=0).to(tl.float32)
            b = tl.load(T2 + offsets, mask, other=0).to(tl.float32)
            y = a + b
        elif KIND == "compute":
            for _ in tl.static_range(16):
                v = tl.sin(v) * 0.5 + 0.25
            tl.store(T0 + offsets, v, mask)
            v = tl.load(T0 + offsets, mask, other=0).to(tl.float32)
            for _ in tl.static_range(16):
                v = tl.sin(v) * 0.5 + 0.25
            y = v
        elif KIND == "reduction":
            tl.store(T0 + offsets, v, mask)
            v = tl.load(T0 + offsets, mask, other=0).to(tl.float32)
            y = tl.sum(v, 0) / BLOCK + tl.full((BLOCK,), 0, tl.float32)
        else:
            tl.store(T0 + offsets, v, mask)
            v = tl.load(T0 + offsets, mask, other=0).to(tl.float32)
            y = v * v + 0.25
        tl.store(Y + offsets, y, mask)

    return memory_kernel


def prepare(args):
    import triton
    from triton._C import libtriton
    from triton._C.libtriton import ir, passes
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler.compiler import ASTSource

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    fn = kernel_source()
    target = GPUTarget("cuda", args.arch, 32)
    backend = CUDABackend(target)
    options = backend.parse_options({"ptx_version": 80, "num_warps": 4, "enable_fp_fusion": False})
    manifest = {"arch": args.arch, "num_warps": 4, "sizes": list(SIZES), "variants": [],
                "protocol": "same-kernel-pass-ab-v1", "harness_sha256": sha(__file__),
                "libtriton_sha256": sha(libtriton.__file__), "triton_file": triton.__file__,
                "pass_sha256": sha(Path(__file__).resolve().parents[2] /
                                   "lib/Dialect/Triton/Transforms/EliminateIntermediateMemory.cpp"),
                "options": {"enable_fp_fusion": False, "ptx_version": 80,
                            "max_transcendental_depth": args.max_transcendental_depth},
                "notes": "Same pre-pass IR, same compiler and downstream pipeline; only candidate adds the pass."}
    for kind in WORKLOADS:
        for dtype in DTYPES:
            for block in BLOCKS:
                key = f"{kind}-{dtype}-b{block}"
                signature = {name: "*" + dtype for name in ("X", "T0", "T1", "T2", "Y")}
                signature["N"] = "i32"
                source = ASTSource(fn, signature, {"BLOCK": block, "KIND": kind})
                ctx = ir.context()
                ir.load_dialects(ctx)
                backend.load_dialects(ctx)
                initial = source.make_ir(target, options, backend.get_codegen_implementation(options),
                                         backend.get_module_map(), ctx)
                initial.context = ctx
                initial = backend.make_ttir(initial, {}, options, args.arch)
                input_path = root / (key + ".input.ttir")
                input_path.write_text(str(initial))
                private = [] if kind in ("retained", "control") else ([1, 2, 3] if kind == "fanout" else [1])
                entry = {"id": key, "kind": kind, "dtype": dtype, "block": block,
                         "private_arguments": private, "requested_private_arguments": private,
                         "input_sha256": sha(input_path), "code": {}}
                for label in ("off", "on"):
                    module = ir.parse_mlir_module(str(input_path), ctx)
                    module.context = ctx
                    pm = ir.pass_manager(ctx)
                    if label == "on":
                        passes.ttir.add_eliminate_intermediate_memory(
                            pm, private, assume_disjoint_arguments=True, num_ctas=1, cluster_size=1,
                            max_transcendental_depth=args.max_transcendental_depth)
                    passes.common.add_canonicalizer(pm)
                    passes.common.add_cse(pm)
                    begin = time.perf_counter()
                    pm.run(module, "benchmark_intermediate_memory_" + label)
                    transform_ms = (time.perf_counter() - begin) * 1000
                    report = None
                    if label == "on":
                        report = json.loads(module.get_operation().get_str_attr("triton.intermediate_memory_elimination"))
                        if not report["accepted"] and not report["reason"].startswith("profitability:"):
                            raise RuntimeError((key, report))
                        if report["accepted"] and kind != "control" and report["forwarded_loads"] == 0:
                            raise RuntimeError((key, "no loads forwarded"))
                        # A profitability rejection retains the original body and
                        # its temporary allocations. Never treat requested private
                        # arguments as a certificate that their accesses disappeared.
                        entry["private_arguments"] = report["private_arguments"]
                    ttir = str(module)
                    (root / (key + "." + label + ".ttir")).write_text(ttir)
                    metadata = {"num_warps": options.num_warps, "num_ctas": options.num_ctas,
                                "cluster_dims": options.cluster_dims}
                    module = backend.make_ttgir(module, metadata, options, args.arch)
                    llir = backend.make_llir(module, metadata, options, args.arch)
                    ptx = backend.make_ptx(llir, metadata, options, args.arch)
                    if metadata["global_scratch_size"] or metadata["profile_scratch_size"]:
                        raise RuntimeError("This harness does not allocate compiler scratch")
                    ptx_path = root / (key + "." + label + ".ptx")
                    ptx_path.write_text(ptx)
                    if label == "on" and not report["accepted"]:
                        if ptx != (root / entry["code"]["off"]["ptx"]).read_text():
                            raise RuntimeError((key, "rejected plan changed generated PTX"))
                    params = re.search(r"\.entry \w+\((.*?)\)\n", ptx, re.S).group(1)
                    if params.count(".param") != 8:
                        raise RuntimeError((key, "unexpected CUDA kernel ABI", params))
                    entry["code"][label] = {"ptx": ptx_path.name, "ptx_sha256": sha(ptx_path),
                                             "metadata": metadata, "report": report,
                                             "ttir_loads": ttir.count("tt.load"),
                                             "ttir_stores": ttir.count("tt.store"),
                                             "transform_ms_diagnostic": transform_ms,
                                             "ptx_global_loads": len(re.findall(r"\bld\.global", ptx)),
                                             "ptx_global_stores": len(re.findall(r"\bst\.global", ptx))}
                manifest["variants"].append(entry)
                save(root / "manifest.json", manifest)
                print(key, {label: (entry["code"][label]["ttir_loads"], entry["code"][label]["ttir_stores"])
                            for label in ("off", "on")}, flush=True)


class Driver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        vp, ui = ctypes.c_void_p, ctypes.c_uint
        self.lib.cuModuleLoad.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        self.lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, vp]
        self.lib.cuLaunchKernel.argtypes = [vp, ui, ui, ui, ui, ui, ui, ui, vp,
                                          ctypes.POINTER(vp), ctypes.POINTER(vp)]
        self.modules = []

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"CUDA driver error {status}")

    def load(self, path, name):
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuModuleLoad(ctypes.byref(module), os.fsencode(path)))
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode()))
        self.modules.append(module)
        attrs = {}
        for key, attr in (("registers", 4), ("local_bytes", 3), ("static_shared_bytes", 1)):
            value = ctypes.c_int()
            self.check(self.lib.cuFuncGetAttribute(ctypes.byref(value), attr, function))
            attrs[key] = value.value
        return function, attrs

    def launch(self, function, pointers, n, block, shared, stream):
        values = [ctypes.c_uint64(p) for p in pointers] + [ctypes.c_int32(n), ctypes.c_uint64(0), ctypes.c_uint64(0)]
        params = (ctypes.c_void_p * len(values))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values])
        self.check(self.lib.cuLaunchKernel(function, (n + block - 1) // block, 1, 1, 128, 1, 1,
                                           shared, stream, params, None))


def environment(torch):
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
               "--format=csv"]
    return {"time": time.time(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "gpu": subprocess.run(command, capture_output=True, text=True, check=True).stdout,
            "compute_processes": subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                                                 capture_output=True, text=True, check=True).stdout}


def reference(torch, x, kind, block):
    v = x.float() * 0.5 + 0.125
    rounded = lambda z: z.to(x.dtype).float()
    if kind == "chain4":
        for _ in range(4):
            v = rounded(v * 0.75 + 0.125)
            v = v * v
        y = v
    elif kind == "fanout":
        v = rounded(v)
        y = rounded(v * v + 0.125) + rounded(v * 0.25 - 0.125)
    elif kind == "compute":
        for _ in range(16):
            v = torch.sin(v) * 0.5 + 0.25
        v = rounded(v)
        for _ in range(16):
            v = torch.sin(v) * 0.5 + 0.25
        y = v
    elif kind == "reduction":
        v = rounded(v)
        v = torch.nn.functional.pad(v, (0, (-x.numel()) % block))
        y = (v.reshape(-1, block).sum(1) / block).repeat_interleave(block)[:x.numel()]
    else:
        if kind != "control":
            v = rounded(v)
        y = v * v + 0.25
    return y.to(x.dtype)


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    if tuple(torch.cuda.get_device_capability()) != (manifest["arch"] // 10, manifest["arch"] % 10):
        raise RuntimeError("GPU architecture differs from compiled target")
    if sha(__file__) != manifest["harness_sha256"]:
        raise RuntimeError("Harness differs from preparation snapshot")
    torch.manual_seed(1029 + args.session)
    torch.cuda.init()
    # CUDA 13/PyTorch may defer creating the primary context until allocation.
    context_anchor = torch.empty(1, device="cuda")
    driver = Driver()
    assembled = {}
    for entry in manifest["variants"]:
        for label, code in entry["code"].items():
            path = root / code["ptx"]
            assert sha(path) == code["ptx_sha256"]
            cubin = path.with_suffix(".cubin")
            if not cubin.exists():
                result = subprocess.run([args.ptxas, str(path), "-o", str(cubin), "-arch=sm_" + str(manifest["arch"]),
                                         "--fmad=false", "-v"], capture_output=True, text=True)
                path.with_suffix(".ptxas.log").write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stderr)
            function, attrs = driver.load(cubin, code["metadata"]["name"])
            assembled[(entry["id"], label)] = (function, attrs, sha(cubin))
    report = {"session": args.session, "rounds": args.rounds, "environment_before": environment(torch),
              "manifest_sha256": sha(root / "manifest.json"), "timing": "CUDA events around CUDA graph replay; same buffers reused; allocations excluded",
              "case_order_seed": 1061 + args.session, "ptxas": subprocess.check_output([args.ptxas, "--version"], text=True), "cases": []}
    cases = [(entry, n) for entry in manifest["variants"] for n in manifest["sizes"]]
    rng = random.Random(1061 + args.session)
    rng.shuffle(cases)
    if args.limit:
        cases = cases[:args.limit]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(call):
        start.record(stream)
        call()
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end) * 1000

    for entry, n in cases:
        dtype = torch.float16 if entry["dtype"] == "fp16" else torch.float32
        tensors = [torch.rand(n, device="cuda", dtype=dtype) - 0.5]
        tensors += [torch.empty(n, device="cuda", dtype=dtype) for _ in range(4)]
        pointers = [t.data_ptr() for t in tensors]
        # Real disjoint allocations, reused for both variants; no allocator timing.
        assert len(set(pointers)) == 5
        expected = reference(torch, tensors[0], entry["kind"], entry["block"])
        stream.wait_stream(torch.cuda.current_stream())
        launches, resources = {}, {}
        for label in ("off", "on"):
            function, attrs, digest = assembled[(entry["id"], label)]
            shared = entry["code"][label]["metadata"]["shared"]
            launches[label] = lambda f=function, s=shared: driver.launch(f, pointers, n, entry["block"], s, stream.cuda_stream)
            resources[label] = dict(attrs, dynamic_shared_bytes=shared, cubin_sha256=digest)
        with torch.cuda.stream(stream):
            baseline = None
            for label in ("off", "on"):
                for tensor in tensors[1:]:
                    tensor.fill_(float("nan"))
                launches[label]()
                torch.cuda.synchronize()
                # Arithmetic reductions may change rounding across frameworks.
                torch.testing.assert_close(tensors[4], expected, rtol=2e-3 if dtype == torch.float16 else 3e-5,
                                           atol=2e-4 if dtype == torch.float16 else 2e-6)
                if label == "off":
                    baseline = tensors[4].clone()
                else:
                    torch.testing.assert_close(tensors[4], baseline, rtol=0, atol=0)
                    for index in entry["private_arguments"]:
                        assert torch.isnan(tensors[index]).all(), "Candidate accessed private storage"
                if entry["kind"] == "retained":
                    torch.testing.assert_close(tensors[1], (tensors[0].float() * 0.5 + 0.125).to(dtype), rtol=0, atol=0)
            del expected, baseline
            # Warm both variants and calibrate graph length independently of A/B ordering.
            for _ in range(10):
                launches["off"]()
                launches["on"]()
            torch.cuda.synchronize()
            calibration = {label: statistics.median(timed(launches[label]) for _ in range(3)) for label in ("off", "on")}
            repeats = max(4, min(256, math.ceil(2000 / max(calibration.values()))))
            graphs = {}
            for label in ("off", "on"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(repeats):
                        launches[label]()
                graphs[label] = graph
            for _ in range(3):
                graphs["off"].replay()
                graphs["on"].replay()
            torch.cuda.synchronize()
            samples = []
            for round_index in range(args.rounds):
                order = ["off", "on"] if (round_index + args.session) % 2 == 0 else ["on", "off"]
                row = {"order": order}
                for label in order:
                    row[label] = timed(graphs[label].replay) / repeats
                row["speedup"] = row["off"] / row["on"]
                samples.append(row)
        item = {"id": entry["id"] + f"-n{n}", "variant_id": entry["id"], "kind": entry["kind"], "n": n,
                "dtype": entry["dtype"], "block": entry["block"], "correct": True, "bitwise_ab": True,
                "repeats": repeats, "resources": resources, "samples_us": samples,
                "median_speedup": statistics.median(row["speedup"] for row in samples),
                "gpu_after": environment(torch)["gpu"]}
        report["cases"].append(item)
        save(root / f"session-{args.session}.json", report)
        print(json.dumps({"session": args.session, "case": item["id"], "speedup": round(item["median_speedup"], 4)}), flush=True)
        del graphs, launches, tensors
    report["environment_after"] = environment(torch)
    save(root / f"session-{args.session}.json", report)


def bootstrap_ci(values, rng):
    # Resample paired ratios, not independent numerator/denominator timings.
    draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(3000))
    return [draws[74], draws[2924]]


def analyze(args):
    root = args.output
    sessions = [json.loads(p.read_text()) for p in sorted(root.glob("session-*.json"))]
    if not sessions:
        raise RuntimeError("No measurements")
    rows = {}
    for session in sessions:
        for item in session["cases"]:
            rows.setdefault(item["id"], []).append(item)
    rng = random.Random(1019)
    summary = {"sessions": len(sessions), "cases": [],
               "stable_gain_definition": "Every session median > 1.02 and pooled paired-bootstrap 95% CI lower bound > 1.02",
               "stable_regression_definition": "Every session median < 0.98 and pooled paired-bootstrap 95% CI upper bound < 0.98",
               "scope": "Warm repeated-buffer CUDA graph execution on one GPU; no launch fusion or host allocation benefit included."}
    for key, items in sorted(rows.items()):
        if len(items) != len(sessions):
            raise RuntimeError((key, "missing session"))
        ratios = [sample["speedup"] for item in items for sample in item["samples_us"]]
        medians = [item["median_speedup"] for item in items]
        ci = bootstrap_ci(ratios, rng)
        status = ("stable_gain" if min(medians) > 1.02 and ci[0] > 1.02 else
                  "stable_regression" if max(medians) < 0.98 and ci[1] < 0.98 else "not_stable_beyond_2pct")
        summary["cases"].append({"id": key, "kind": items[0]["kind"], "n": items[0]["n"],
                                 "dtype": items[0]["dtype"], "block": items[0]["block"], "status": status,
                                 "median_speedup": statistics.median(ratios), "ci95": ci, "session_medians": medians,
                                 "off_us": statistics.median(s["off"] for x in items for s in x["samples_us"]),
                                 "on_us": statistics.median(s["on"] for x in items for s in x["samples_us"]),
                                 "on_faster_fraction": sum(r > 1 for r in ratios) / len(ratios),
                                 "resources": items[0]["resources"]})
    summary["counts"] = {status: sum(x["status"] == status for x in summary["cases"])
                         for status in ("stable_gain", "stable_regression", "not_stable_beyond_2pct")}
    save(root / "summary.json", summary)
    print(json.dumps(summary["counts"], indent=2))
    for kind in WORKLOADS:
        rows = [x for x in summary["cases"] if x["kind"] == kind]
        print(kind, min(x["median_speedup"] for x in rows), max(x["median_speedup"] for x in rows),
              {status: sum(x["status"] == status for x in rows) for status in summary["counts"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", type=int, default=89)
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-transcendental-depth", type=int, default=8)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("at least five rounds required")
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
