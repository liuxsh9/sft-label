# sft-label

SFT Capability Taxonomy & Auto-Labeling Pipeline.

Multi-pass automated labeling pipeline for SFT code-generation training data:
- **Pass 1 (Tag Labeling)**: 9-category taxonomy (224 tags) via LLM calls
- **Pass 2 (Value Scoring)**: Complexity, quality, reasoning, and rarity scoring for data selection
- **Pass 2.5 (Conversation Aggregation)**: Conversation-level metrics from multi-turn slices (no LLM)
- **Pass 3 (Filtering & Selection)**: Multi-condition sample and conversation filtering
- **Pass 4 (Trajectory Semantic Clustering)**: Pinned-prefix windowing + SemHash + ANN clustering + SNR representative selection

## Install

```bash
pip install -e .
# or with uv
uv sync
```

## Quick Start

```bash
# Set LLM endpoint
export LITELLM_BASE="http://localhost:4000/v1"
export LITELLM_KEY="your-key"

# Recommended: interactive launcher (grouped by workflow)
sft-label start

# Preview generated command without executing
sft-label start --dry-run

# Pass 1: Tag labeling
sft-label run --input data.json

# Pass 2: Value scoring (standalone, on pre-labeled data)
sft-label score --input labeled.json --tag-stats stats.json

# Pass 1 + Pass 2 in one go
sft-label run --input data.json --score

# Compact mode: reduced prompt size (~32% smaller, for size-limited endpoints)
sft-label run --input data.json --score --prompt-mode compact

# Pass 1 + Pass 2 + Pass 4 in one go
sft-label run --input data.json --score --semantic-cluster

# Standalone Pass 4
sft-label semantic-cluster --input run_dir/
sft-label export-semantic --input run_dir/ --output representatives.jsonl
```

## Usage

### CLI

```bash
# Interactive launcher (recommended when unsure about flags)
sft-label start
```

The interactive launcher groups commands by workflow:
- Pipeline: Pass 1/2/4 combinations, standalone scoring, standalone semantic clustering
- Data curation: filtering
- Maintenance: stats recompute, dashboard regeneration, taxonomy validation
- Export: semantic rows, review CSV/TSV

It also supports advanced tuning via optional raw flags input, so all existing CLI flags remain available.

```bash
# Run labeling pipeline (Pass 1)
sft-label run --input data.json

# Run on a directory
sft-label run --input data_dir/ --output results/

# Resume interrupted run
sft-label run --input data_dir/ --resume data_dir-labeled-20250101_120000/

# Continuous mode: Pass 1 + Pass 2 value scoring
sft-label run --input data.json --score

# Continuous mode: Pass 1 + Pass 2 + trajectory semantic clustering
sft-label run --input data.json --score --semantic-cluster

# Compact prompt mode (reduced payload size for size-limited endpoints)
sft-label run --input data.json --score --prompt-mode compact

# Continuous mode with external rarity stats
sft-label run --input data.json --score --tag-stats global_stats.json

# Standalone value scoring (Pass 2)
sft-label score --input labeled.json
sft-label score --input labeled.json --tag-stats global_stats.json
sft-label score --input results_dir/

# Standalone trajectory semantic clustering (Pass 4)
sft-label semantic-cluster --input run_dir/
sft-label semantic-cluster --input run_dir/ --output semantic_out/
sft-label semantic-cluster --input run_dir/ --resume

# Tune clustering parameters (CPU-first local embedding backend)
sft-label semantic-cluster --input run_dir/ \
  --semantic-embedding-provider local \
  --semantic-window-size 50 \
  --semantic-window-stride 30 \
  --semantic-semhash-bits 256 \
  --semantic-semhash-bands 8 \
  --semantic-hamming-radius 64 \
  --semantic-ann-top-k 32 \
  --semantic-ann-sim-threshold 0.82

# API embedding fallback (OpenAI-compatible /v1/embeddings)
sft-label semantic-cluster --input run_dir/ --semantic-embedding-provider api

# Export representative windows (default: representatives only)
sft-label export-semantic --input run_dir/ --output representatives.jsonl

# Export all windows (including non-representatives)
sft-label export-semantic --input run_dir/ --output all_windows.jsonl --include-all

# Filter high-value samples from scored data
sft-label filter --input scored.json --value-min 6.0
sft-label filter --input scored.json --value-min 6 --difficulty advanced,expert
sft-label filter --input scored.json --selection-min 7.0 --exclude-inherited
sft-label filter --input run_dir/ --value-min 7 --format training
sft-label filter --input scored.json --value-min 6 --thinking-mode slow

# Filter by conversation-level metrics (multi-turn)
sft-label filter --input scored.json --conv-value-min 7
sft-label filter --input run_dir/ --conv-value-min 6 --conv-selection-min 5
sft-label filter --input scored.json --turn-value-min 5 --turn-count-min 3

# Validate taxonomy
sft-label validate

# Export to review CSV
sft-label export-review --input labeled.json --output review.csv
```

### Library

```python
import asyncio
from sft_label import run, PipelineConfig

config = PipelineConfig(
    labeling_model="gpt-4o-mini",
    scoring_model="gpt-4o-mini",
    concurrency=50,
    litellm_base="http://localhost:4000/v1",
    litellm_key="your-key",
    prompt_mode="compact",  # or "full" (default)
)

stats = asyncio.run(run("data.json", config=config))
print(f"Labeled {stats['success']}/{stats['total_samples']} samples")
```

```python
# Standalone value scoring
from sft_label.scoring import run_scoring

config = PipelineConfig(
    scoring_model="gpt-4o-mini",
    scoring_concurrency=50,
)

stats = asyncio.run(run_scoring(
    input_path="labeled.json",
    tag_stats_path="stats.json",
    config=config,
))
```

## Architecture

```
Input (ShareGPT JSON / Pangu JSONL)
  │
  ├─ Pass 1: Tag Labeling
  │   ├─ Format detection + normalization (preprocessing.py)
  │   ├─ Multi-turn slicing: each reply → one sample
  │   ├─ Call 1 (LLM): Intent, Language, Domain, Task, Difficulty
  │   ├─ Call 2 (LLM): Concept, Agentic, Constraint, Context
  │   ├─ Validation: tag pool check, cross-dimension consistency
  │   ├─ Arbitration (optional): re-run low-confidence dimensions
  │   └─ Output: labeled.json + stats.json + dashboard.html
  │
  ├─ Pass 2: Value Scoring (scoring.py)
  │   ├─ Rarity computation: tag IDF + combo rarity from tag distributions
  │   ├─ COT-preserving truncation: head + middle fragments + tail
  │   ├─ LLM scoring: complexity, quality, reasoning (1 call per sample)
  │   ├─ Weighted aggregation: value_score = Σ(weight × dimension)
  │   ├─ Selection score: intra-class rank + absolute quality + global rarity
  │   └─ Output: scored.json + stats_value.json + dashboard_value.html
  │
  ├─ Pass 2.5: Conversation Aggregation (conversation.py)
  │   ├─ Group multi-turn slices by source conversation
  │   ├─ Position-weighted averaging (later turns weighted higher)
  │   ├─ Quality floor + negative flag penalties
  │   └─ Output: conversation_scores.json (conv_value, conv_selection)
  │
  ├─ Pass 3: Filtering & Selection (tools/filter_value.py)
  │   ├─ Sample-level: value_min, selection_min, difficulty, thinking_mode
  │   ├─ Conversation-level: conv_value_min, conv_selection_min, peak_complexity_min
  │   ├─ Turn-level: turn_value_min, turn_count_min/max, keep_first_last
  │   ├─ Tag filtering: include_tags / exclude_tags (dim:tag format)
  │   ├─ Source control: exclude_inherited, verify_source
  │   └─ Output formats: scored (preserves metadata) or training (stripped)
  │
  └─ Pass 4: Trajectory Semantic Clustering (semantic_clustering.py)
      ├─ Long trajectory segmentation (>50 turns) with pinned task-definition prefix
      ├─ Bilingual role-aware rendering for embedding text
      ├─ Lightweight embedding backend (local CPU hash model) or API fallback
      ├─ Deterministic SemHash (seeded random hyperplanes)
      ├─ Coarse candidate retrieval (banded SemHash + hamming radius)
      ├─ ANN refinement (cosine threshold + top-k)
      ├─ Union-find cluster assembly
      └─ Representative selection by SNR (action/observation), tie-break by value_score
```

## Taxonomy (Pass 1)

9 orthogonal categories, 224 tags:

| Category   | Tags | Select |
|-----------|------|--------|
| Intent     | 6    | single |
| Difficulty | 4    | single |
| Context    | 10   | single |
| Language   | 75   | multi  |
| Domain     | 38   | multi  |
| Task       | 22   | multi  |
| Concept    | 26   | multi  |
| Agentic    | 23   | multi  |
| Constraint | 20   | multi  |

## Value Scoring (Pass 2)

Each sample receives multi-dimensional scores (1-10):

| Dimension   | Sub-scores | Weight |
|------------|-----------|--------|
| Complexity  | instruction, reasoning, implementation | 0.25 |
| Quality     | correctness, code_quality, explanation, completeness | 0.40 |
| Reasoning   | clarity, consistency, self_correction | 0.20 |
| Rarity      | tag IDF, combo rarity (computed, no LLM) | 0.15 |

Additional outputs per sample:
- `selection_score`: Weighted fusion of intra-class percentile rank (0.55), absolute quality (0.20), and global rarity (0.25)
- `flags`: Qualitative markers (e.g., `has-bug`, `excellent-explanation`, `clean-code`)
- `thinking_mode`: Auto-detected `slow` (explicit COT) or `fast` (inline reasoning)
- `confidence`: Model confidence in its assessment (0-1)

### Output Files (Per File)

| File | Contents |
|------|----------|
| `scored.json` | Labeled samples with `value` field added |
| `conversation_scores.json` | Conversation-level aggregated metrics (multi-turn) |
| `stats_value.json` | Aggregate statistics, score distributions, cross-analysis |
| `dashboard_value.html` | Self-contained interactive dashboard |
| `monitor_value.jsonl` | Per-sample LLM call metadata |
| `failed_value.jsonl` | Samples that failed scoring |

## Trajectory Semantic Clustering (Pass 4)

### What It Does

- Segments long trajectories into logically complete sliding windows.
- Preserves a pinned task-definition prefix for windows from long trajectories.
- Computes semantic fingerprints and clusters windows with SemHash + ANN.
- Selects one representative per cluster using:
  `snr = action_tokens / max(observation_tokens, 1)`.

### Output Files (Per Run)

| File | Contents |
|------|----------|
| `trajectory_windows.jsonl` | Windowed trajectory records with source linkage + turn ranges |
| `trajectory_embeddings.jsonl` | Normalized embedding vectors per window |
| `trajectory_semhash.jsonl` | SemHash bits + band values per window |
| `trajectory_cluster_membership.jsonl` | Cluster membership + SNR + representative flag |
| `trajectory_clusters.json` | Cluster-to-window map |
| `trajectory_representatives.jsonl` | Representative windows only |
| `semantic_cluster_stats.json` | Cluster diagnostics and throughput metrics |
| `semantic_cluster_manifest.json` | Versioned state manifest for compatibility/resume |

### Key Configs (PipelineConfig)

- `semantic_long_turn_threshold` (default `50`)
- `semantic_window_size` (default `50`)
- `semantic_window_stride` (default `30`)
- `semantic_pinned_prefix_max_turns` (default `3`)
- `semantic_embedding_provider` (`local` or `api`, default `local`)
- `semantic_embedding_model`
- `semantic_embedding_dim` (default `384`)
- `semantic_semhash_bits` (default `256`)
- `semantic_semhash_bands` (default `8`)
- `semantic_hamming_radius` (default `64`)
- `semantic_ann_top_k` (default `32`)
- `semantic_ann_sim_threshold` (default `0.82`)

### CPU Benchmark

```bash
python3 scripts/benchmark_semantic_clustering.py --samples 5000 --out /tmp/sc-bench
```

See [benchmark_semantic_clustering_report.md](scripts/benchmark_semantic_clustering_report.md) for projection assumptions.

### Dashboard Sections

- Value Overview Cards (mean scores, top 10%, token usage)
- Score Distributions (histograms for all dimensions)
- Sub-score Breakdown (per-dimension detail)
- Value x Tag Cross-Analysis (quality by difficulty, value by domain, etc.)
- Thinking Mode Analysis (slow vs fast comparison)
- Flag Analysis (frequency and value impact)
- Coverage Impact Analysis (tag retention at different thresholds)
- File Ranking Table (global dashboard, sortable)

## Filtering (Pass 3)

Multi-condition sample selection with AND logic between criteria, OR within tag lists:

| Criterion | Flag | Example |
|-----------|------|---------|
| Value score | `--value-min` | `--value-min 6.0` |
| Selection score | `--selection-min` | `--selection-min 7.0` |
| Difficulty | `--difficulty` | `--difficulty advanced,expert` |
| Thinking mode | `--thinking-mode` | `--thinking-mode slow` |
| Include tags | `--include-tags` | `--include-tags domain:security,task:debugging` |
| Exclude tags | `--exclude-tags` | `--exclude-tags concept:basic-io` |
| Exclude inherited | `--exclude-inherited` | Drops sparse-sampled inherited labels |
| Source verification | `--verify-source` | `--verify-source original.json` |
| Output format | `--format` | `scored` (default) or `training` (stripped) |
| Preserve structure | `--preserve-structure` | Directory mode: mirror folder structure + per-file format (skip empty files) |

Conversation-level criteria (multi-turn):

| Criterion | Flag | Example |
|-----------|------|---------|
| Conv value | `--conv-value-min` | `--conv-value-min 7` |
| Conv selection | `--conv-selection-min` | `--conv-selection-min 5` |
| Peak complexity | `--peak-complexity-min` | `--peak-complexity-min 6` |
| Turn count | `--turn-count-min/max` | `--turn-count-min 3 --turn-count-max 20` |
| Turn-level pruning | `--turn-value-min` | `--turn-value-min 5` (prune low-value turns) |

## Incremental Update Workflow

### Recompute Statistics

After manually editing labels, merging runs, or changing datasets, recompute stats without re-running the LLM pipeline:

```bash
# Single file
uv run sft-label recompute-stats --input run_dir/labeled.json
uv run sft-label recompute-stats --input run_dir/scored.json --pass 2

# Entire run directory (recomputes per-file + summary stats)
uv run sft-label recompute-stats --input run_dir/
uv run sft-label recompute-stats --input run_dir/ --pass 1
uv run sft-label recompute-stats --input run_dir/ --workers 8

# Custom output directory
uv run sft-label recompute-stats --input run_dir/ --output /path/to/output/
```

Recomputed stats are marked with `"recomputed": true`. LLM token usage fields will be zero (not preserved in pipeline output).

`--workers` controls directory-mode file-level parallelism (default: `8`).

### Refresh Rarity (Offline)

Recompute Pass 2 rarity/value fields without calling LLM:

```bash
uv run sft-label refresh-rarity --input run_dir/
uv run sft-label refresh-rarity --input run_dir/ --tag-stats global_stats.json --workers 8
```

### Regenerate Dashboards

Re-generate HTML dashboards from existing stats and data files:

```bash
# Regenerate all dashboards in a run directory
uv run sft-label regenerate-dashboard --input run_dir/

# Pass 1 only, and open in browser
uv run sft-label regenerate-dashboard --input run_dir/ --pass 1 --open

# Pass 2 only
uv run sft-label regenerate-dashboard --input run_dir/ --pass 2

# Batch directory parallelism
uv run sft-label regenerate-dashboard --input run_dir/ --workers 8
```

Requires stats files to exist — run `recompute-stats` first if they are missing.

### Cross-Dataset Rarity with `--tag-stats`

Use a historical or global stats.json as the rarity baseline when scoring new data:

```bash
# Score new data using global tag distributions for rarity
uv run sft-label score --input new_labeled.json --tag-stats global_stats.json

# Continuous mode with external rarity baseline
uv run sft-label run --input data.json --score --tag-stats /path/to/reference_stats.json
```

### Where to Find stats.json

| Mode | Pass 1 | Pass 2 |
|------|--------|--------|
| Single file | `<run_dir>/stats.json` | `<run_dir>/stats_value.json` |
| Directory | `<run_dir>/<subdir>/stats.json` | `<run_dir>/<subdir>/stats_value.json` |
| Directory summary | `<run_dir>/summary_stats.json` | `<run_dir>/summary_stats_value.json` |

### Typical Workflows

1. **Edit labels → refresh stats/dashboards:**
   ```bash
   # Edit labeled.json manually, then:
   uv run sft-label recompute-stats --input run_dir/ --pass 1
   uv run sft-label regenerate-dashboard --input run_dir/ --pass 1
   ```

2. **Merge multiple runs → build combined reference:**
   ```bash
   # Copy scored files into a single directory, then:
   uv run sft-label recompute-stats --input merged_dir/
   # Use the merged stats as rarity baseline for new scoring:
   uv run sft-label score --input new_data.json --tag-stats merged_dir/summary_stats.json
   ```

3. **Lost dashboards → regenerate:**
   ```bash
   uv run sft-label regenerate-dashboard --input run_dir/ --open
   ```

## Development

```bash
uv sync --extra dev
uv run pytest                          # all tests
uv run pytest tests/test_e2e_mock.py   # e2e tests (mocked LLM)
uv run sft-label validate              # validate taxonomy
```

## Environment Variables

- `LITELLM_BASE` — LLM proxy base URL (default: `http://localhost:4000/v1`)
- `LITELLM_KEY` — API key for the LLM proxy

## License

[Apache License 2.0](LICENSE)
