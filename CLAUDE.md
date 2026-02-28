# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
uv sync --extra dev

# Run tests
uv run pytest
uv run pytest tests/test_preprocessing.py::TestSliceMultiturn::test_multi_turn  # single test
uv run pytest tests/test_scoring.py  # scoring tests only

# Validate taxonomy (should report 0 errors)
uv run sft-label validate

# Run labeling pipeline (requires LITELLM_BASE and LITELLM_KEY env vars)
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --model gpt-4o-mini

# Run value scoring (Pass 2) on pre-labeled data
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label score --input labeled.json --tag-stats stats.json

# Continuous mode: Pass 1 + Pass 2
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score
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

**Key design decisions:**
- Two LLM calls per sample in Pass 1 (not one) because Call 2 needs Call 1 results as context
- `TAG_POOLS` in `prompts.py` define the valid tag set — the LLM's output is validated against these pools; out-of-pool tags become "unmapped"
- Single-select dimensions: intent, difficulty, context. Multi-select: everything else
- Directory mode uses watermark-based loading (`DIR_PIPELINE_WATERMARK`) to keep concurrency saturated across files
- Sparse sampling: for multi-turn slices from the same conversation, only a subset gets LLM-labeled; the rest inherit labels from their nearest labeled slice
- Pass 2 uses COT-preserving truncation (unlike Pass 1 which strips COT) because COT quality is a key scoring dimension
- Rarity is computed from tag IDF (not LLM), normalized to 1-10 via percentile mapping
- Value score = 0.25×complexity + 0.35×quality + 0.15×reasoning + 0.25×rarity (configurable)

**Taxonomy data** is embedded as package data in `src/sft_label/taxonomy/`. Load via `_resources.py` (uses `importlib.resources`), not file paths.

**Dual interface:**
- CLI: `sft-label run|validate|score|export-review` (entry point in `cli.py`)
- Library: `from sft_label import run, PipelineConfig`

**Config layering:** Module-level constants in `config.py` serve as defaults. `PipelineConfig` dataclass allows runtime overrides. Environment variables `LITELLM_BASE` and `LITELLM_KEY` configure the LLM endpoint.

## Key Files

| File | Purpose |
|------|---------|
| `src/sft_label/pipeline.py` | Pass 1 labeling pipeline, `async_llm_call()` |
| `src/sft_label/scoring.py` | Pass 2 value scoring pipeline |
| `src/sft_label/preprocessing.py` | Format detection, slicing, truncation (both Pass 1 and Pass 2) |
| `src/sft_label/prompts.py` | Pass 1 system prompts, TAG_POOLS |
| `src/sft_label/prompts_value.py` | Pass 2 scoring prompts, few-shot examples |
| `src/sft_label/config.py` | All configuration constants + PipelineConfig |
| `src/sft_label/cli.py` | CLI entry point (run, validate, score, export-review) |
| `src/sft_label/tools/visualize_value.py` | Pass 2 dashboard generation |

## Origin

Extracted from `/Users/lxs/code/build-user-query/labeling/`. All imports were changed from `labeling.xxx` / `from prompts import` to `sft_label.xxx`. The `sys.path.insert` hacks in the source were removed in favor of proper package imports.
