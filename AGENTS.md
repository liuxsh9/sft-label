# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Install
uv sync --extra dev

# Install with HF dataset tools (for scripts/download_hf_dataset.py, scripts/sample_to_sft.py)
uv sync --extra dev --extra data

# Run tests
uv run pytest
uv run pytest tests/test_preprocessing.py::TestSliceMultiturn::test_multi_turn  # single test
uv run pytest tests/test_scoring.py  # scoring tests only

# Validate taxonomy (should report 0 errors)
uv run sft-label validate

# E2E test with real open-source data (requires LITELLM_BASE and LITELLM_KEY env vars)
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10  # quick smoke
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score              # full 1083 samples

# Run labeling pipeline (requires LITELLM_BASE and LITELLM_KEY env vars)
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json

# Run value scoring (Pass 2) on pre-labeled data
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label score --input labeled.json --tag-stats stats.json

# Continuous mode: Pass 1 + Pass 2
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score

# Compact prompt mode (reduced payload ~32% smaller, for size-limited endpoints)
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score --prompt-mode compact

# Continuous mode with external rarity stats for Pass 2
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score --tag-stats global_stats.json

# Filter high-value samples from scored data
uv run sft-label filter --input scored.json --threshold 6.0
uv run sft-label filter --input scored.json --value-min 6 --difficulty advanced,expert
uv run sft-label filter --input run_dir/ --value-min 7 --format training
uv run sft-label filter --input scored.json --value-min 6 --exclude-inherited --thinking-mode slow

# Filter by conversation-level metrics (multi-turn)
uv run sft-label filter --input scored.json --conv-value-min 7
uv run sft-label filter --input run_dir/ --conv-value-min 6 --conv-selection-min 5
uv run sft-label filter --input scored.json --conv-value-min 7 --peak-complexity-min 6 --format training

# Recompute stats offline (no LLM, after editing/merging outputs)
uv run sft-label recompute-stats --input run_dir/
uv run sft-label recompute-stats --input scored.json --pass 2

# Regenerate dashboards from existing stats
uv run sft-label regenerate-dashboard --input run_dir/ --open
uv run sft-label regenerate-dashboard --input run_dir/ --pass 1
```

## Architecture

This is a standalone extraction of the labeling subsystem from `build-user-query`. It labels SFT code-generation training data using a 9-category taxonomy (221 tags) via LLM calls, then optionally scores each sample's training value.

**Pass 1: Tag Labeling** (in `pipeline.py`):
1. **Preprocessing** (`preprocessing.py`) — detect format (ShareGPT vs Pangu), normalize, slice multi-turn conversations into per-reply samples, extract structural signals (languages, code blocks, tool usage)
2. **Call 1** (LLM) — labels 5 dimensions: Intent, Language, Domain, Task, Difficulty
3. **Call 2** (LLM, depends on Call 1 result) — labels 4 dimensions: Concept, Agentic, Constraint, Context
4. **Validation** — check tags against `TAG_POOLS` in `prompts.py`, run `CONSISTENCY_RULES` from `config.py`
5. **Arbitration** (optional) — re-run low-confidence dimensions at higher temperature

**Pass 2: Value Scoring** (in `scoring.py`):
1. **Rarity computation** — tag IDF + combo rarity from Pass 1's `stats.json` tag distributions, no LLM needed
2. **COT-preserving truncation** (`preprocessing.py`) — detects slow/fast thinking mode, preserves COT with head + middle fragments + tail sampling
3. **LLM scoring** — 1 call per sample using prompt from `prompts_value.py`: complexity (1-10), quality (1-10), reasoning (1-10), flags
4. **Validation** — `validate_score_response()` checks all fields, converts types, tracks unknown flags
5. **Aggregation** — weighted composite `value_score`, aggregate statistics, dashboard generation

**Pass 2.5: Conversation Aggregation** (in `conversation.py`):
- Post-scoring, no LLM — groups multi-turn slices by `source_id`, computes conversation-level metrics
- Position-weighted averaging (later turns weighted 1→2×), inherited slices get 0.7× confidence
- Quality floor penalty (min quality < 3 → 0.5×, < 5 → 0.8×) and negative flag penalty (0.95^count)
- `conv_value` = weighted_avg(value_scores) × penalty, clamped [1,10]
- `conv_selection` = 0.85×intra_class_rank + 0.15×conv_rarity (per-tag percentile with Bayesian shrinkage, same as sample-level but on conversations)
- Outputs `conversation_scores.json` alongside scored data

**Pass 3: Filtering & Selection** (in `tools/filter_value.py`):
- Multi-condition filtering: `value_min`, `selection_min`, `include_tags`/`exclude_tags`, `difficulty`, `thinking_mode`, `exclude_inherited`, `verify_source`
- Conversation-level filtering: `conv_value_min`, `conv_selection_min`, `peak_complexity_min` — applies to multi-turn slices via conversation_scores.json lookup
- Criteria use AND logic between different types, OR logic within tag lists
- Output formats: `scored` (preserves labels/scores) or `training` (training-ready, strips metadata)
- Training output for Pangu-original data uses `to_pangu_pseudo_multiturn()` in `preprocessing.py` to reconstruct the pseudo-multi-turn format with `[unused*]` tokens
- `FilterConfig` dataclass, `matches_filter()` for per-sample checks, `filter_samples()` for batch

**Key design decisions:**
- Two LLM calls per sample in Pass 1 (not one) because Call 2 needs Call 1 results as context
- `TAG_POOLS` in `prompts.py` define the valid tag set — the LLM's output is validated against these pools; out-of-pool tags become "unmapped"
- Single-select dimensions: intent, difficulty, context. Multi-select: everything else
- Directory mode uses watermark-based loading (`DIR_PIPELINE_WATERMARK`) to keep concurrency saturated across files
- Sparse sampling: for multi-turn slices from the same conversation, only a subset gets LLM-labeled; the rest inherit labels from their nearest labeled slice. Inherited samples are excluded from tag distributions and confidence stats to avoid inflating counts.
- Multi-turn slices get `thinking_mode="fast"` and `cot_text` removed (since COT is stripped during slicing), preventing misleading scoring in Pass 2
- Pass 2 uses COT-preserving truncation (unlike Pass 1 which strips COT) because COT quality is a key scoring dimension
- Rarity is computed from tag IDF (not LLM), normalized to 1-10 via percentile mapping
- Value score = 0.25×complexity + 0.40×quality + 0.20×reasoning + 0.15×rarity (configurable)
- Quality floor penalty: quality.overall < 4 → value_score *= 0.7
- Selection score = 0.85×intra_class_rank + 0.15×rarity (per-tag percentile with Bayesian shrinkage, structurally different from value_score)
- Pangu pseudo-multiturn reconstruction (`to_pangu_pseudo_multiturn`) uses `raw_pangu_data` saved during normalization for roundtrip fidelity; falls back to algorithmic reconstruction for sliced/modified samples
- `source_file` metadata is added to each sample during labeling to enable downstream source verification in filtering

**Taxonomy data** is embedded as package data in `src/sft_label/taxonomy/`. Load via `_resources.py` (uses `importlib.resources`), not file paths.

**Dual interface:**
- CLI: `sft-label run|validate|score|filter|export-review` (entry point in `cli.py`)
- Library: `from sft_label import run, PipelineConfig`

**Config layering:** Module-level constants in `config.py` serve as defaults. `PipelineConfig` dataclass allows runtime overrides. Environment variables `LITELLM_BASE` and `LITELLM_KEY` configure the LLM endpoint.

**Python version:** Requires >=3.9. Use `from __future__ import annotations` in any file using `X | Y` type union syntax (e.g., `float | None`).

## Key Files

| File | Purpose |
|------|---------|
| `src/sft_label/pipeline.py` | Pass 1 labeling pipeline, `async_llm_call()` |
| `src/sft_label/scoring.py` | Pass 2 value scoring pipeline |
| `src/sft_label/conversation.py` | Pass 2.5 conversation-level aggregation (no LLM) |
| `src/sft_label/preprocessing.py` | Format detection, slicing, truncation (both Pass 1 and Pass 2) |
| `src/sft_label/prompts.py` | Pass 1 system prompts, TAG_POOLS |
| `src/sft_label/prompts_value.py` | Pass 2 scoring prompts, few-shot examples |
| `src/sft_label/config.py` | All configuration constants + PipelineConfig |
| `src/sft_label/cli.py` | CLI entry point (run, validate, score, filter, recompute-stats, regenerate-dashboard, export-review) |
| `src/sft_label/tools/filter_value.py` | Multi-condition sample filtering + training format output |
| `src/sft_label/tools/recompute.py` | Offline stats recomputation + dashboard regeneration |
| `src/sft_label/tools/visualize_labels.py` | Pass 1 label dashboard generation |
| `src/sft_label/tools/visualize_value.py` | Pass 2 score dashboard generation |

## Test Data

**Unit test fixtures** (small, in-repo, for `uv run pytest`):
- `tests/fixtures/smoke_test.json` (5 ShareGPT samples) — used by `test_e2e_mock.py`
- `tests/fixtures/smoke_test_value.json` (5 samples) — used by `test_scoring.py`, `test_e2e_mock.py`
- `tests/fixtures/pangu_test_samples.jsonl` (12 Pangu samples) — used by `test_preprocessing.py`

**E2E test data** (`tests/fixtures/e2e_folder_test/`, Pangu JSONL format, real open-source data):
- This is the primary e2e test dataset. All files are JSONL (one JSON per line), matching production input format.
- **E2E testing should always use this folder**, either as directory input or individual files:
  ```bash
  # Full directory e2e (all 2300 samples)
  uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score
  # Quick smoke (10 samples from each file)
  uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10
  # Single file e2e
  uv run sft-label run --input tests/fixtures/e2e_folder_test/code/magicoder_oss_instruct.jsonl --score
  ```
- Structure:
  ```
  e2e_folder_test/
    code/
      mot_code_part1.jsonl              (500 single-turn, MoT competitive programming, COT)
      mot_code_part2.jsonl              (500 single-turn, MoT competitive programming, COT)
      magicoder_oss_instruct.jsonl      (500 single-turn, OSS-grounded code gen, 20 languages)
      commitpackft_multilang.jsonl      (500 single-turn, real git commits, 20 languages)
    multi_turn/
      nemotron_swe_repair.jsonl         (100 single-turn SWE repair, COT)
      code_feedback_multiturn.jsonl     (100 multi-turn iterative coding, avg 6.5 turns)
      coderforge_swe_trajectories.jsonl (100 multi-turn agent trajectories, tool calls, avg 125 turns)
  ```
- Sources:
  - `mot_code_*`: [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) code subset
  - `nemotron_swe_repair`: [nvidia/Nemotron-Cascade-SFT-Stage-2](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2) swe_repair subset
  - `magicoder_oss_instruct`: [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) — code gen grounded in real OSS code
  - `commitpackft_multilang`: [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) — real git commits across 20 languages
  - `code_feedback_multiturn`: [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) — iterative coding with execution feedback
  - `coderforge_swe_trajectories`: [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) SWE_Rebench — OpenHands agent trajectories with tool calls
- Raw datasets stored at: `/Volumes/MOVESPEED/datasets/open-source-sft/`

**Dataset download scripts** (in `scripts/`):
- `scripts/download_hf_dataset.py` — Download HuggingFace datasets (`--subset` to filter, `--convert` for ShareGPT JSON)
- `scripts/sample_to_sft.py` — Sample from local data files (parquet, JSONL, JSON) and convert to ShareGPT or Pangu format. Handles multi-turn conversations, `tool_calls` serialization, and `<think>` → `[unused16/17]` COT conversion.

## Origin

Extracted from `/Users/lxs/code/build-user-query/labeling/`. All imports were changed from `labeling.xxx` / `from prompts import` to `sft_label.xxx`. The `sys.path.insert` hacks in the source were removed in favor of proper package imports.
