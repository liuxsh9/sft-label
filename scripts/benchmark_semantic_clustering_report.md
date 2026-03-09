# Semantic Clustering CPU Benchmark Notes

## Goal

Measure CPU-only throughput and memory usage for trajectory semantic clustering:
- embedding stage (local lightweight backend),
- SemHash + ANN clustering stage,
- representative selection and artifact writing.

## Command

```bash
python3 scripts/benchmark_semantic_clustering.py \
  --samples 2000 \
  --turns 24 \
  --out benchmark_out
```

## What to Record

- `elapsed_seconds`
- `samples_per_second`
- `windows_per_second`
- `max_rss_mb`
- cluster statistics from `semantic_cluster_stats.json`

## 8M Scaling Assumptions

- Throughput scales near-linearly with sample count under similar average turn length.
- Embedding and clustering parameters remain unchanged.
- CPU core count and memory are stable (`~44 cores`, `~96 GB`).
- Dataset language mix remains similar (zh/en mixed traffic).

Projected wall-clock estimate:

```text
estimated_seconds = 8_000_000 / measured_samples_per_second
```

## Caveats

- This benchmark is full-pipeline, not clustering-only.
- Real datasets with longer trajectories (>50 turns) increase window count and cost.
- API embedding mode has different latency and cost profile; benchmark separately if enabled.
