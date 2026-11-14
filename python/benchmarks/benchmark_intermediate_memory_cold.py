"""Validate the same pass A/B binaries after a 256 MiB L2 eviction sweep."""

import argparse
import json
from pathlib import Path
import random
import statistics

from benchmark_intermediate_memory_elimination import Driver, environment, reference, save, sha


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    torch.manual_seed(731 + args.session)
    torch.cuda.init()
    eviction = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    driver = Driver()
    report = {"session": args.session, "rounds": 15, "eviction_bytes": eviction.numel(),
              "timing": "External CUDA event nodes inside a graph, after eviction and around one target kernel only",
              "harness_sha256": sha(__file__), "main_harness_sha256": sha(Path(__file__).with_name("benchmark_intermediate_memory_elimination.py")),
              "manifest_sha256": sha(root / "manifest.json"), "environment_before": environment(torch), "cases": []}
    variants = list(manifest["variants"])
    random.Random(833 + args.session).shuffle(variants)
    for entry in variants:
        n = max(manifest["sizes"])
        dtype = torch.float16 if entry["dtype"] == "fp16" else torch.float32
        tensors = [torch.rand(n, device="cuda", dtype=dtype) - 0.5]
        tensors += [torch.empty(n, device="cuda", dtype=dtype) for _ in range(4)]
        pointers = [t.data_ptr() for t in tensors]
        expected = reference(torch, tensors[0], entry["kind"], entry["block"])
        stream.wait_stream(torch.cuda.current_stream())
        graphs, events, resources = {}, {}, {}
        with torch.cuda.stream(stream):
            for label in ("off", "on"):
                code = entry["code"][label]
                cubin = (root / code["ptx"]).with_suffix(".cubin")
                fn, attrs = driver.load(cubin, code["metadata"]["name"])
                resources[label] = dict(attrs, cubin_sha256=sha(cubin))
                launch = lambda f=fn: driver.launch(f, pointers, n, entry["block"], code["metadata"]["shared"], stream.cuda_stream)
                launch()
                torch.cuda.synchronize()
                torch.testing.assert_close(tensors[4], expected, rtol=2e-3 if dtype == torch.float16 else 3e-5,
                                           atol=2e-4 if dtype == torch.float16 else 2e-6)
                if label == "off":
                    baseline = tensors[4].clone()
                else:
                    torch.testing.assert_close(tensors[4], baseline, rtol=0, atol=0)
                # Warm the eviction kernel before capture. Its duration is excluded.
                eviction.add_(1)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    eviction.add_(1)
                    start.record(stream)
                    launch()
                    end.record(stream)
                graphs[label], events[label] = graph, (start, end)
            del expected, baseline
            for _ in range(4):
                graphs["off"].replay()
                graphs["on"].replay()
            torch.cuda.synchronize()
            samples = []
            for round_index in range(15):
                order = ["off", "on"] if (round_index + args.session) % 2 == 0 else ["on", "off"]
                row = {"order": order}
                for label in order:
                    graphs[label].replay()
                    start, end = events[label]
                    end.synchronize()
                    row[label] = start.elapsed_time(end) * 1000
                row["speedup"] = row["off"] / row["on"]
                samples.append(row)
        item = {"id": entry["id"] + f"-n{n}", "variant_id": entry["id"], "kind": entry["kind"], "n": n,
                "dtype": entry["dtype"], "block": entry["block"], "correct": True, "bitwise_ab": True,
                "resources": resources, "samples_us": samples,
                "median_speedup": statistics.median(row["speedup"] for row in samples)}
        report["cases"].append(item)
        save(root / f"cold-{args.session}.json", report)
        print(json.dumps({"session": args.session, "case": item["id"], "speedup": round(item["median_speedup"], 4)}), flush=True)
        del tensors, graphs, events
    report["environment_after"] = environment(torch)
    save(root / f"cold-{args.session}.json", report)


if __name__ == "__main__":
    main()
