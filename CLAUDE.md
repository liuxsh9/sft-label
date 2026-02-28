# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
uv sync --extra dev

# Run tests
uv run pytest
uv run pytest tests/test_preprocessing.py::TestSliceMultiturn::test_multi_turn  # single test

# Validate taxonomy (should report 0 errors)
uv run sft-label validate

# Run labeling pipeline (requires LITELLM_BASE and LITELLM_KEY env vars)
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --model gpt-4o-mini
```

## Architecture

This is a standalone extraction of the labeling subsystem from `build-user-query`. It labels SFT code-generation training data using a 9-category taxonomy (221 tags) via LLM calls.

**Pipeline flow** (in `pipeline.py`):
1. **Preprocessing** (`preprocessing.py`) — detect format (ShareGPT vs Pangu), normalize, slice multi-turn conversations into per-reply samples, extract structural signals (languages, code blocks, tool usage)
2. **Call 1** (LLM) — labels 5 dimensions: Intent, Language, Domain, Task, Difficulty
3. **Call 2** (LLM, depends on Call 1 result) — labels 4 dimensions: Concept, Agentic, Constraint, Context
4. **Validation** — check tags against `TAG_POOLS` in `prompts.py`, run `CONSISTENCY_RULES` from `config.py`
5. **Arbitration** (optional) — re-run low-confidence dimensions at higher temperature

**Key design decisions:**
- Two LLM calls per sample (not one) because Call 2 needs Call 1 results as context
- `TAG_POOLS` in `prompts.py` define the valid tag set — the LLM's output is validated against these pools; out-of-pool tags become "unmapped"
- Single-select dimensions: intent, difficulty, context. Multi-select: everything else
- Directory mode uses watermark-based loading (`DIR_PIPELINE_WATERMARK`) to keep concurrency saturated across files
- Sparse sampling: for multi-turn slices from the same conversation, only a subset gets LLM-labeled; the rest inherit labels from their nearest labeled slice

**Taxonomy data** is embedded as package data in `src/sft_label/taxonomy/`. Load via `_resources.py` (uses `importlib.resources`), not file paths.

**Dual interface:**
- CLI: `sft-label run|validate|export-review` (entry point in `cli.py`)
- Library: `from sft_label import run, PipelineConfig`

**Config layering:** Module-level constants in `config.py` serve as defaults. `PipelineConfig` dataclass allows runtime overrides. Environment variables `LITELLM_BASE` and `LITELLM_KEY` configure the LLM endpoint.

## Origin

Extracted from `/Users/lxs/code/build-user-query/labeling/`. All imports were changed from `labeling.xxx` / `from prompts import` to `sft_label.xxx`. The `sys.path.insert` hacks in the source were removed in favor of proper package imports.
