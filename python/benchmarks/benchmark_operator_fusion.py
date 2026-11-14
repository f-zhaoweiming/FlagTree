"""Measure the Inductor integration with generated-code and runtime provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1:
        parser.error("rounds and iterations must be positive")

    import torch
    import triton
    from triton.experimental import fusion
    from torch._inductor.utils import run_and_get_code

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA validation requires a working CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(61)

    def branched(x, bias):
        shifted = torch.sin(x) + bias
        row = shifted.square().mean(dim=-1, keepdim=True)
        return shifted * torch.rsqrt(row + 0.01), row, shifted

    def softmax(x, bias):
        return torch.softmax(x + bias, dim=-1)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    report = {"torch_version": torch.__version__, "torch_file": torch.__file__,
              "triton_version": triton.__version__, "triton_file": triton.__file__,
              "fusion_file": fusion.__file__,
              "fusion_sha256": hashlib.sha256(Path(fusion.__file__).read_bytes()).hexdigest(),
              "device": args.device, "cuda_available": torch.cuda.is_available(),
              "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
              "timing": "synchronized wall time per call, including host dispatch and allocation",
              "numerics": "Inductor defaults; float32 tolerance rtol=atol=2e-5",
              "cases": []}
    with torch.no_grad():
        for name, model in (("branched_reduction", branched), ("broadcast_softmax", softmax)):
            for rows, columns, transposed in ((17, 257, False), (129, 1025, True)):
                x = torch.randn((columns, rows) if transposed else (rows, columns), device=args.device)
                if transposed:
                    x = x.t()
                bias = torch.randn(columns, device=args.device)
                torch.compiler.reset()
                compiled = fusion.compile(model, dynamic=True)
                sync()
                start = time.perf_counter()
                actual, sources = run_and_get_code(compiled, x, bias)
                sync()
                first_seconds = time.perf_counter() - start
                torch.testing.assert_close(actual, model(x, bias), rtol=2e-5, atol=2e-5)
                if not sources:
                    raise RuntimeError("No generated code observed; refusing to count eager execution as compiled")
                stem = f"{name}-{rows}x{columns}"
                paths = []
                for index, source in enumerate(sources):
                    path = args.output / f"{stem}-{index}.py"
                    path.write_text(source)
                    paths.append(path.name)
                for _ in range(5):
                    model(x, bias)
                    compiled(x, bias)
                samples = {"eager": [], "inductor": []}
                for round_index in range(args.rounds):
                    order = (("eager", model), ("inductor", compiled))
                    for label, fn in order if round_index % 2 == 0 else reversed(order):
                        sync()
                        begin = time.perf_counter()
                        for _ in range(args.iterations):
                            fn(x, bias)
                        sync()
                        samples[label].append((time.perf_counter() - begin) * 1e6 / args.iterations)
                record = {"name": name, "shape": list(x.shape), "stride": list(x.stride()),
                          "first_call_seconds": first_seconds, "generated_sources": paths,
                          "wall_us": samples,
                          "eager_over_compiled": statistics.median(samples["eager"]) /
                                                 statistics.median(samples["inductor"])}
                report["cases"].append(record)
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"case": stem, "correct": True, "eager_over_compiled":
                                  record["eager_over_compiled"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
