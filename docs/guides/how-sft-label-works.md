# How sft-label works

`sft-label` is organized as a staged curation pipeline.

## Pass 1: capability labeling

Goal: describe what each assistant reply is doing.

Main steps:

1. **Preprocessing**
   - detect input format
   - normalize conversations
   - slice multi-turn conversations into per-reply samples
   - extract structural signals such as languages, code blocks, and tools
2. **LLM call 1**
   - labels intent, language, domain, task, difficulty
3. **LLM call 2**
   - labels concept, agentic, constraint, context
4. **Optional extension labeling**
   - runs only when `--label-extension` specs are provided
   - uses the core labels + conversation context
   - writes `label_extensions` payloads per turn
   - surface diagnostics before the first extension call (trigger presence, prompt/schema warnings) and after the run (match counts, validation warnings, low-confidence fields, unmapped rows) so you can confirm the spec behavior before scaling
   - keep each extension focused on a single domain/trigger when you need multiple schemas; that keeps the dashboards, exports, and diagnostics clear for each intent
5. **Validation**
   - checks labels against the taxonomy pools
   - applies consistency rules
6. **Optional arbitration**
   - retries low-confidence dimensions with different settings

Output highlights:

- `labeled.json` or mirrored inline labels in `data_label`
- `stats_labeling.json`
- Pass 1 dashboards
- Optional extension aggregates in `stats_labeling.json` (see [Pass 1 extension labeling](pass1-extension-labeling.md))

## Pass 2: value scoring

Goal: estimate whether the sample is worth keeping for training.

Main steps:

1. **Rarity computation**
   - derived from Pass 1 statistics
   - no LLM needed
2. **CoT-preserving truncation**
   - keeps enough reasoning context for scoring
3. **LLM scoring**
   - complexity
   - quality
   - reasoning
   - coarse hard-filter flags (`has-bug`, `incomplete`)
4. **Aggregation**
   - computes composite `value_score`
   - computes `selection_score`

Output highlights:

- `scored.json`
- `stats_scoring.json`
- `conversation_scores.json` for multi-turn data
- Pass 2 dashboards

## Pass 2.5: conversation aggregation

Goal: treat a multi-turn conversation as a single selection unit.

This stage works without additional LLM calls. It groups turn-level slices by conversation/source ID and computes conversation-level metrics such as:

- `conv_value`
- `conv_selection`
- rarity-confidence-derived confidence fields
- peak complexity and other trajectory features

These metrics are especially useful for filtering multi-turn data.

## Pass 3: filtering

Goal: export subsets that are more useful for review or training.

Examples:

- high-value single-turn samples
- multi-turn conversations above a conversation value threshold
- training-ready output with metadata stripped
- source-verified subsets

See [Common workflows](common-workflows.md) for examples.

## Dashboard generation

Dashboards are built from existing run artifacts.

- the HTML file is now a lightweight shell
- detailed payloads live next to it in `dashboard_*.data/`
- runtime assets live under `_dashboard_static/v1/` for local runs, or under a shared static service when published

This keeps large runs shareable and avoids embedding giant payloads directly into the HTML.

## Adaptive runtime (optional)

When `--adaptive-runtime` is enabled (default), `sft-label` can treat
`--concurrency` and `--rps-limit` as caps and adjust effective pressure based on
observed endpoint health. During instability, it may lower effective throughput,
pause briefly, and probe for recovery, then gradually ramp back up.

When `--recovery-sweep` is enabled (default), `sft-label` can retry infra-failed
samples once at the end of Pass 1 and Pass 2 to improve completion rate.

See [Adaptive LLM runtime](adaptive-llm-runtime.md) for details.

## Input modes

### Standard file/directory mode

Use this when you want conventional artifacts such as `labeled.json`, `scored.json`, and `dashboards/`.

### Mirrored inline JSONL mode

Use this when your source of truth is a Pangu-style JSONL tree and you want labels embedded back into the original rows under `extra_info.unique_info.data_label`.

This mode writes derived metadata into `meta_label_data/` so stats and dashboards can be rebuilt later without relabeling everything.

## Why the interactive launcher exists

The CLI surface is wide because the project supports:

- multiple input styles
- full pipeline vs partial pipeline runs
- offline maintenance workflows
- dashboard hosting workflows

`uv run sft-label start` reduces this to a guided workflow and generates a normal CLI command that you can reuse later.
