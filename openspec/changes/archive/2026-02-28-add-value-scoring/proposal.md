## Why

The existing 9-dimension tag labeling pipeline (Pass 1) answers "what is this data?" — it classifies samples by intent, language, domain, etc. But it doesn't answer "how valuable is this data for SFT training?" We need a second scoring pass that evaluates each sample's actual training value: how complex is the task, how good is the response, how rare is this data point. This enables data selection — picking the highest-value subset from 750K+ samples for more efficient, higher-quality model training. Research (DEITA, s1, IFD) consistently shows that 5-10% carefully selected data matches or exceeds full-dataset training performance.

## What Changes

- Add a **value scoring pipeline** (Pass 2) that runs after tag labeling, producing per-sample scores for complexity (1-10), quality (1-10), reasoning quality (1-10), and rarity (computed from tag distributions)
- Add **COT-preserving smart truncation** — unlike Pass 1 which strips `<think>` blocks, Pass 2 preserves and evaluates chain-of-thought content using a head + middle-fragments + tail sampling strategy
- Add **rarity computation** from existing tag distributions (IDF-based, no LLM needed), with configurable per-dimension weights
- Add **`sft-label score` CLI subcommand** for standalone Pass 2 execution on pre-labeled data
- Add **`--score` flag** to `sft-label run` for continuous Pass 1 → Pass 2 execution
- Add **value dashboards** (per-file and global) with cross-analysis views: Value×Tag heatmaps, file ranking tables, coverage impact analysis, and data selection threshold simulation
- Add **stats_value.json** output with score distributions, per-tag value means, thinking mode breakdowns, and selection threshold precomputation

## Capabilities

### New Capabilities
- `value-scoring`: Core scoring logic — rarity computation (tag IDF + combo rarity), COT-preserving truncation, LLM-based quality/complexity/reasoning assessment, final weighted value score aggregation
- `value-dashboard`: Per-file and global dashboard generation with cross-analysis views (Value×Tag, file ranking, coverage impact, selection simulator)
- `value-prompts`: LLM prompt design for the scoring call — system prompt with scoring rubrics, anchoring examples, slow/fast thinking adaptation, and structured JSON output

### Modified Capabilities
_(none — Pass 2 is additive; Pass 1 pipeline behavior is unchanged)_

## Impact

- **New files**: `scoring.py` (core logic), `prompts_value.py` (prompts), `tools/visualize_value.py` (dashboards), `tests/test_scoring.py`, `tests/fixtures/smoke_test_value.json`
- **Modified files**: `config.py` (new PipelineConfig fields), `cli.py` (new `score` subcommand + `--score` flag), `pipeline.py` (continuous mode integration), `preprocessing.py` (COT-preserving truncation variant)
- **Dependencies**: No new external dependencies; uses same LLM API infrastructure as Pass 1
- **Backwards compatibility**: Fully additive. Existing `sft-label run` and `sft-label validate` behavior unchanged. Pass 1 output format unchanged.
