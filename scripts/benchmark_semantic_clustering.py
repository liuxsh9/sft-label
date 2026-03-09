#!/usr/bin/env python3
"""CPU benchmark helper for semantic clustering throughput and memory.

Usage:
  python3 scripts/benchmark_semantic_clustering.py --samples 5000 --out /tmp/sc-bench
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from pathlib import Path

from sft_label.config import PipelineConfig
from sft_label.semantic_clustering import run_semantic_clustering


def _make_sample(i: int, turns: int = 24):
    convs = [
        {"from": "system", "value": "You are an assistant."},
        {"from": "human", "value": f"Task {i}: build and explain a utility."},
        {"from": "gpt", "value": "I will provide a robust implementation."},
    ]
    for t in range(max(turns // 2, 1)):
        if t % 3 == 0:
            q = f"请给出第{t}步优化建议，并包含错误处理。"
        else:
            q = f"Step {t}: how do we improve latency and reliability?"
        convs.append({"from": "human", "value": q})
        convs.append({"from": "gpt", "value": "Use batching, retries, and explicit validation."})
    return {"id": f"bench_{i}", "conversations": convs, "value": {"value_score": random.uniform(4.5, 8.5)}}


def _estimate_rss_mb() -> float:
    # macOS ru_maxrss is bytes, Linux is KB. This normalization handles both.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10**9:
        return rss / (1024 * 1024)
    return rss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples")
    ap.add_argument("--turns", type=int, default=24, help="Approx turns per sample")
    ap.add_argument("--out", type=str, default="benchmark_out", help="Output directory")
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--stride", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / "benchmark_input.json"

    samples = [_make_sample(i, turns=args.turns) for i in range(args.samples)]
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

    cfg = PipelineConfig(
        semantic_window_size=args.window_size,
        semantic_window_stride=args.stride,
        semantic_embedding_provider="local",
        semantic_embedding_dim=384,
        semantic_embedding_batch_size=256,
        semantic_embedding_max_workers=8,
        semantic_semhash_bits=256,
        semantic_semhash_bands=8,
        semantic_hamming_radius=64,
        semantic_ann_top_k=32,
        semantic_ann_sim_threshold=0.82,
    )

    t0 = time.time()
    stats = run_semantic_clustering(
        input_path=input_file,
        output_dir=out_dir,
        config=cfg,
        resume=False,
    )
    elapsed = max(time.time() - t0, 1e-9)
    rss_mb = _estimate_rss_mb()

    windows = stats.get("total_windows", 0)
    samples_per_sec = stats.get("total_samples", 0) / elapsed
    windows_per_sec = windows / elapsed

    report = {
        "samples": args.samples,
        "turns_per_sample": args.turns,
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(samples_per_sec, 3),
        "windows_per_second": round(windows_per_sec, 3),
        "max_rss_mb": round(rss_mb, 3),
        "stats": stats,
    }
    with open(out_dir / "benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Back-of-envelope scaling assumption for 8M.
    est_seconds_8m = 8_000_000 / max(samples_per_sec, 1e-9)
    print("Semantic clustering benchmark complete")
    print(f"Samples: {args.samples}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Samples/s: {samples_per_sec:.2f}")
    print(f"Windows/s: {windows_per_sec:.2f}")
    print(f"Max RSS: {rss_mb:.2f} MB")
    print(
        "Projected end-to-end time for 8M samples (same hardware/config): "
        f"{est_seconds_8m/3600:.2f} hours"
    )
    print(f"Report: {out_dir / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
