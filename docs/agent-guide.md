# Agent Guide for `sft-label`

This is the **canonical agent-facing repository guide** for `sft-label`.

- Update this file first when commands, architecture, fixtures, or workflows change.
- Keep `/Users/lxs/.codex/worktrees/e39f/sft-label/AGENTS.md` and `/Users/lxs/.codex/worktrees/e39f/sft-label/CLAUDE.md` as thin wrappers that point here.
- Use `/Users/lxs/.codex/worktrees/e39f/sft-label/README.md` and `/Users/lxs/.codex/worktrees/e39f/sft-label/docs/guides/` for user-facing walkthroughs; this file is optimized for coding agents operating in the repo.

## Repository snapshot

`sft-label` is a dataset-curation pipeline for code-generation SFT data. It can:

1. normalize ShareGPT/Pangu-style conversations,
2. label assistant replies with a 9-dimension taxonomy,
3. score training value,
4. aggregate multi-turn conversations,
5. cluster long trajectories semantically,
6. filter/export high-signal subsets, and
7. generate/publish HTML dashboards.

The default entry point for humans is:

```bash
uv run sft-label start
```

The programmatic/library entry point is:

```python
from sft_label import run, PipelineConfig
```

## Canonical commands

### Install

```bash
uv sync --extra dev
uv sync --extra dev --extra data
```

### Development checks

```bash
uv run pytest
uv run sft-label validate
```

Focused examples:

```bash
uv run pytest tests/test_preprocessing.py::TestSliceMultiturn::test_multi_turn
uv run pytest tests/test_scoring.py
```

### Recommended interactive path

```bash
uv run sft-label start
uv run sft-label start --dry-run
uv run sft-label start --lang en
uv run sft-label start --lang zh
```

### Core pipeline commands

```bash
# Pass 1 only
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json

# Pass 1 + Pass 2
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score

# Pass 1 + Pass 2 with compact prompt payloads
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score --prompt-mode compact

# Pass 2 only on existing labels
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label score --input labeled.json

# Refresh rarity/value fields offline (no LLM)
uv run sft-label refresh-rarity --input scored.json
```

### Smoke / e2e fixture commands

```bash
# Quick repo smoke test with bundled fixtures
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10

# Full bundled e2e fixture set (current snapshot: 1083 JSONL rows total)
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score
```

### Dashboard commands

```bash
# Rebuild dashboards from existing outputs
uv run sft-label regenerate-dashboard --input <run_dir>

# Initialize a local dashboard service
uv run sft-label dashboard-service init --web-root ~/sft-label-dashboard --service-type builtin

# Start the service
uv run sft-label dashboard-service start

# Publish an existing run
uv run sft-label dashboard-service register-run --run-dir <run_dir>
```

### Semantic clustering commands

```bash
uv run sft-label semantic-cluster --input <run_dir_or_json>
uv run sft-label export-semantic --input <run_dir_or_json>
```

### Filtering / export / maintenance

```bash
# Filter scored data
uv run sft-label filter --input scored.json --threshold 6.0
uv run sft-label filter --input scored.json --value-min 6 --difficulty advanced,expert
uv run sft-label filter --input <run_dir> --value-min 7 --format training
uv run sft-label filter --input scored.json --value-min 6 --exclude-inherited --thinking-mode slow

# Conversation-level filtering
uv run sft-label filter --input scored.json --conv-value-min 7
uv run sft-label filter --input <run_dir> --conv-value-min 6 --conv-selection-min 5
uv run sft-label filter --input scored.json --conv-value-min 7 --peak-complexity-min 6 --format training

# Recompute/rebuild offline
uv run sft-label recompute-stats --input <run_dir>
uv run sft-label recompute-stats --input scored.json --pass 2
uv run sft-label regenerate-dashboard --input <run_dir> --pass 1

# Review/export/debug helpers
uv run sft-label export-review --input <run_dir_or_json>
uv run sft-label analyze-unmapped --input <run_dir_or_json>
uv run sft-label optimize-layout --input <historical_run_dir>
```

## Current architecture

### 1. Entry modes

- **Interactive launcher**: `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/launcher.py` and the `start` CLI flow guide users through common workflows, dashboard publishing, and maintenance.
- **Standard runs**: file or directory input produces run directories with labeled/scored outputs plus dashboard artifacts.
- **Mirrored inline JSONL runs**: inline labels/scores are written back into mirrored dataset trees, with summary artifacts stored under `meta_label_data/`.

### 2. Pass 1: labeling

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/pipeline.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/preprocessing.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/prompts.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/labels.py`

Flow:

1. detect/normalize input format,
2. slice multi-turn conversations into per-reply samples,
3. extract structural signals (languages, code blocks, tool usage),
4. call the LLM for **Intent / Language / Domain / Task / Difficulty**,
5. call the LLM again for **Concept / Agentic / Constraint / Context** using Pass 1 call-1 output as context,
6. validate tags against `TAG_POOLS`,
7. apply consistency rules,
8. optionally arbitrate low-confidence dimensions.

Key design notes:

- Pass 1 intentionally uses **two LLM calls per sample** because the second call depends on the first call's output.
- `TAG_POOLS` in `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/prompts.py` define the valid label space.
- Single-select dimensions: `intent`, `difficulty`, `context`.
- Multi-select dimensions: the remaining dimensions.
- Directory mode uses a watermark-based loader to keep concurrency saturated across files.

### 3. Pass 2: value scoring

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/scoring.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/prompts_value.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/score_confidence.py`

Flow:

1. compute rarity from Pass 1 stats,
2. apply COT-preserving truncation,
3. run one LLM scoring call per sample,
4. validate/normalize the response,
5. aggregate `value_score` and summary stats,
6. emit score dashboards/artifacts.

Key design notes:

- Multi-turn slices are marked `thinking_mode="fast"` and have `cot_text` removed during slicing to avoid misleading Pass 2 evaluation.
- Pass 2 preserves COT fragments where available because reasoning quality is part of the score.
- Default weighted score:
  - `0.25 * complexity`
  - `0.40 * quality`
  - `0.20 * reasoning`
  - `0.15 * rarity`
- Quality floor penalty applies when `quality.overall < 4`.

### 4. Pass 2.5: conversation aggregation

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/conversation.py`

This stage groups slices by `source_id` and computes conversation-level metrics without additional LLM calls.

Important details:

- later turns are position-weighted more heavily,
- inherited slices get reduced confidence weight,
- negative flags and low-quality turns penalize `conv_value`,
- `conv_selection` combines intra-class rank and conversation rarity.

### 5. Pass 3: semantic clustering

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/semantic_clustering.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/semantic_artifacts.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/export_semantic_clusters.py`

This stage clusters long trajectories/windows to reduce redundancy and export representative windows for downstream review or training.

### 6. Pass 4: filtering and export

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/filter_value.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/export_review.py`

Filtering supports sample-level and conversation-level thresholds, inclusion/exclusion tags, difficulty, thinking mode, source verification, and training-format export.

### 7. Inline and migration helpers

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/inline_pass1.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/inline_scoring.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/inline_rows.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/inline_labels.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/inline_migration.py`

These modules support mirrored inline JSONL workflows where labels/scores live alongside dataset rows rather than only in standalone run outputs.

### 8. Dashboards and static publishing

Primary code:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/visualize_labels.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/visualize_value.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/dashboard_service.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/dashboard_aggregation.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/dashboard_explorer.py`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/dashboard_template.py`

Current dashboard output layout (standard runs):

```text
<run_dir>/
  dashboards/
    dashboard_labeling.html
    dashboard_labeling.data/
    dashboard_scoring.html
    dashboard_scoring.data/
    _dashboard_static/v1/
```

The static service can host multiple runs and produce stable per-run URLs.

## Key files

| File | Purpose |
|---|---|
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/cli.py` | All CLI command wiring and parser definitions |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/launcher.py` | Interactive `start` workflow |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/pipeline.py` | Pass 1 labeling pipeline |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/scoring.py` | Pass 2 value scoring |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/conversation.py` | Conversation-level aggregation |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/preprocessing.py` | Format detection, slicing, truncation, reconstruction helpers |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/prompts.py` | Pass 1 prompts and tag pools |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/prompts_value.py` | Pass 2 scoring prompt and examples |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/config.py` | Runtime defaults and `PipelineConfig` |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/dashboard_service.py` | Shared dashboard hosting/publishing |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/semantic_clustering.py` | Trajectory semantic clustering |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/semantic_artifacts.py` | Clustering artifact management |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/filter_value.py` | Filtering and training-format export |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/recompute.py` | Offline stats recomputation and dashboard regeneration |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/visualize_labels.py` | Pass 1 dashboard generation |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/visualize_value.py` | Pass 2 dashboard generation |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/analyze_unmapped.py` | Unmapped-tag inspection |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/tools/export_review.py` | Review CSV export |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/_resources.py` | Package-data loading via `importlib.resources` |
| `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/taxonomy/taxonomy.yaml` | Embedded taxonomy source |

## Test and fixture data

### Small unit fixtures

- `/Users/lxs/.codex/worktrees/e39f/sft-label/tests/fixtures/smoke_test.json`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/tests/fixtures/smoke_test_value.json`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/tests/fixtures/pangu_test_samples.jsonl`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/tests/fixtures/multiturn_regressions.json`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/tests/fixtures/nemotron_selection_regression.json`
- `/Users/lxs/.codex/worktrees/e39f/sft-label/tests/fixtures/smoke_semantic_clustering.json`

### Bundled e2e folder fixture

Current repository snapshot:

```text
tests/fixtures/e2e_folder_test/
  code/
    commitpackft_multilang.jsonl        200 rows
    magicoder_oss_instruct.jsonl        200 rows
    mot_code_part1.jsonl                200 rows
    mot_code_part2.jsonl                200 rows
  multi_turn/
    code_feedback_multiturn.jsonl        75 rows
    coderforge_swe_trajectories.jsonl     8 rows
    nemotron_agentless_file_lookup.jsonl 100 rows
    nemotron_swe_repair.jsonl            100 rows
```

Total rows in the checked-in JSONL fixture set: **1083**.

Use this folder for repo-local smoke/e2e runs unless you explicitly need a larger external dataset.

### Local developer note

Historically, larger raw open-source datasets have been stored outside the repo at:

```text
/Volumes/MOVESPEED/datasets/open-source-sft/
```

Treat that as an environment-specific note, not a guaranteed path.

## Implementation notes and invariants

- Taxonomy package data lives under `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/taxonomy/` and should be loaded via `_resources.py`, not raw file-system assumptions.
- Runtime defaults live in `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/config.py`; CLI flags selectively override those defaults.
- `LITELLM_BASE` and `LITELLM_KEY` configure the LLM endpoint.
- Python requirement is `>=3.9`; any file using `X | Y` unions should include:

```python
from __future__ import annotations
```

## Refresh checklist for future updates

When the repo changes, refresh this guide against:

1. `/Users/lxs/.codex/worktrees/e39f/sft-label/pyproject.toml`
2. `uv run sft-label --help`
3. `/Users/lxs/.codex/worktrees/e39f/sft-label/README.md`
4. current `src/sft_label/` module layout
5. current `tests/fixtures/` contents and counts

Keep wrappers thin. If `AGENTS.md` or `CLAUDE.md` grows large again, move project facts back here.
