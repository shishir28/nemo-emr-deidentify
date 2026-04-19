"""
Performance benchmark — measures throughput and latency of the de-identification pipeline.

Usage:
    python3 scripts/benchmark.py
    python3 scripts/benchmark.py --n 200
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.redactor import Redactor


def load_notes(n: int) -> list[str]:
    path = Path("data/synthetic/notes.jsonl")
    notes = []
    with path.open() as f:
        for line in f:
            notes.append(json.loads(line)["text"])
            if len(notes) >= n:
                break
    while len(notes) < n:
        notes.extend(notes)
    return notes[:n]


def run_benchmark(n: int):
    print(f"Loading model ...")
    redactor = Redactor()

    print(f"Loading {n} notes ...")
    notes = load_notes(n)

    # Warmup
    for note in notes[:5]:
        redactor.deidentify(note)

    print(f"Benchmarking {n} notes ...\n")
    latencies = []
    t_total = time.monotonic()

    for note in notes:
        t0 = time.monotonic()
        redactor.deidentify(note)
        latencies.append((time.monotonic() - t0) * 1000)

    total_s = time.monotonic() - t_total
    throughput = n / total_s

    print(f"{'Notes processed':<25} {n}")
    print(f"{'Total time':<25} {total_s:.2f}s")
    print(f"{'Throughput':<25} {throughput:.1f} notes/sec")
    print(f"{'Latency p50':<25} {statistics.median(latencies):.1f}ms")
    print(f"{'Latency p95':<25} {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    print(f"{'Latency p99':<25} {sorted(latencies)[int(len(latencies)*0.99)]:.1f}ms")
    print(f"{'Latency max':<25} {max(latencies):.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.n)
