# sft-label

SFT Capability Taxonomy & Auto-Labeling Pipeline.

Two-pass automated labeling pipeline for SFT code-generation training data:
- **Pass 1 (Tag Labeling)**: 9-category taxonomy (224 tags) via LLM calls
- **Pass 2 (Value Scoring)**: Complexity, quality, reasoning, and rarity scoring for data selection
- **Pass 2.5 (Conversation Aggregation)**: Conversation-level metrics from multi-turn slices (no LLM)
- **Pass 3 (Filtering & Selection)**: Multi-condition sample and conversation filtering

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

# Pass 1: Tag labeling
sft-label run --input data.json

# Pass 2: Value scoring (standalone, on pre-labeled data)
sft-label score --input labeled.json --tag-stats stats.json

# Pass 1 + Pass 2 in one go
sft-label run --input data.json --score
```

## Usage

### CLI

```bash
# Run labeling pipeline (Pass 1)
sft-label run --input data.json

# Run on a directory
sft-label run --input data_dir/ --output results/

# Resume interrupted run
sft-label run --input data_dir/ --resume data_dir-labeled-20250101_120000/

# Continuous mode: Pass 1 + Pass 2 value scoring
sft-label run --input data.json --score

# Continuous mode with external rarity stats
sft-label run --input data.json --score --tag-stats global_stats.json

# Standalone value scoring (Pass 2)
sft-label score --input labeled.json
sft-label score --input labeled.json --tag-stats global_stats.json
sft-label score --input results_dir/

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
  └─ Pass 3: Filtering & Selection (tools/filter_value.py)
      ├─ Sample-level: value_min, selection_min, difficulty, thinking_mode
      ├─ Conversation-level: conv_value_min, conv_selection_min, peak_complexity_min
      ├─ Turn-level: turn_value_min, turn_count_min/max, keep_first_last
      ├─ Tag filtering: include_tags / exclude_tags (dim:tag format)
      ├─ Source control: exclude_inherited, verify_source
      └─ Output formats: scored (preserves metadata) or training (stripped)
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

Conversation-level criteria (multi-turn):

| Criterion | Flag | Example |
|-----------|------|---------|
| Conv value | `--conv-value-min` | `--conv-value-min 7` |
| Conv selection | `--conv-selection-min` | `--conv-selection-min 5` |
| Peak complexity | `--peak-complexity-min` | `--peak-complexity-min 6` |
| Turn count | `--turn-count-min/max` | `--turn-count-min 3 --turn-count-max 20` |
| Turn-level pruning | `--turn-value-min` | `--turn-value-min 5` (prune low-value turns) |

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
