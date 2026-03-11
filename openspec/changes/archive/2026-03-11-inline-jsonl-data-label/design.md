## Context

The current pipeline treats normalized samples as the primary persisted unit. `normalize_and_slice()` expands one input row into one or more temporary samples, Pass 1 writes `labels` at the sample root, Pass 2 writes `value`, and downstream tools rebuild their view from `labeled.json[l]` and `scored.json[l]`. That model is convenient for pipeline internals but it breaks the invariants users care about: one input JSONL row no longer corresponds to one output row, multi-turn annotations are spread across derived files, and human curation has to happen on transformed artifacts rather than on the dataset the user actually owns.

This change is cross-cutting because it touches persistence, run layout, CLI workflows, chunked processing, scoring, recompute, filtering, dashboards, and migration semantics. The most important constraints are:

- Pangu JSONL outputs MUST preserve the original row structure and line count.
- New annotation fields MUST live under `extra_info.unique_info` without overwriting unrelated content.
- Multi-turn samples MUST preserve per-turn results and conversation-level aggregates.
- Chunked processing and resume behavior MUST remain memory-bounded and safe for large JSONL files.
- Offline maintenance commands MUST derive results from embedded annotations, not from transient materialized sample files.

## Goals / Non-Goals

**Goals:**
- Make each labeled output JSONL line correspond to exactly one source JSONL line.
- Establish `extra_info.unique_info.data_id` and `extra_info.unique_info.data_label` as the single persisted source of truth.
- Preserve the current Pass 1/Pass 2 semantics by flattening rows into transient slices internally, then merging results back into the row.
- Support explicit inline dataset workflows for incremental labeling, full refresh, migration plus fill-in, and offline recomputation.
- Mirror the input directory tree in the labeled output while isolating logs, checkpoints, flattened caches, and stats under `meta_label_data/`.
- Keep single-turn and multi-turn datasets interoperable for scoring, filtering, and dashboard generation.

**Non-Goals:**
- Redesign the taxonomy, prompts, scoring rubrics, or arbitration behavior.
- Remove all transient intermediate representations; temporary flattened caches remain allowed as rebuildable process artifacts.
- Solve every legacy layout compatibility issue in one step; legacy `labeled.json[l]` / `scored.json[l]` support may remain as read-only compatibility during migration.
- Change the user’s raw conversation content outside the allowed additions under `extra_info.unique_info`.

## Decisions

### 1. Make row objects the persisted unit and slices the execution unit

The implementation will introduce a row-centered internal model:

- `RawRow`: the original parsed JSONL row plus file path, row number, and computed `data_id`
- `SliceSample`: the transient single-turn or multi-turn assistant-reply view used by Pass 1, Pass 2, conversation aggregation, and dashboards
- `EmbeddedDataLabel`: the persisted annotation payload merged back into the raw row

Pass 1 and Pass 2 will continue to operate on slices because prompt construction, sparse sampling, and conversation aggregation are already slice-oriented. The change is that slices are no longer the persisted source of truth. Instead, every pipeline phase must be able to:

1. flatten a row into slices
2. execute on slices
3. merge slice results back into a single row-local `data_label`

Alternative considered: keep `labeled.json[l]` / `scored.json[l]` as the canonical output and add an export step that rehydrates inline labels. Rejected because it creates two authoritative data models and leaves downstream curation on derived artifacts.

### 2. Use a stable embedded schema under `extra_info.unique_info`

The persisted contract will be:

- `extra_info.unique_info.data_id`
- `extra_info.unique_info.data_label`

`data_label` will contain:

- `meta`: schema version, label version, stage timestamps, refresh mode, and source format
- `turns`: one entry per assistant reply, containing `turn_index`, slice-level Pass 1 labels, Pass 2 values, inheritance metadata, and optional monitoring summaries
- `conversation`: the conversation-level aggregate used by multi-turn filtering and dashboards

Refreshing labels will replace the entire `data_label` object for the targeted stages while preserving sibling fields under `extra_info` and `unique_info`. This keeps the persisted contract compact, versioned, and easy to copy during migration.

Alternative considered: spread metadata across multiple sibling fields such as `label_version`, `score_version`, and `conversation_value`. Rejected because partial refresh semantics become fragile and migration becomes harder to reason about.

### 3. Compute `data_id` from canonicalized `meta_prompt` + `data`

`data_id` will be derived from a deterministic SHA-256 hash of canonical JSON built from the sample’s `meta_prompt` and `data` content only. The canonicalization rules are:

- UTF-8 serialization
- sorted object keys
- list order preserved
- no inclusion of `extra_info`, prior labels, process metadata, or source file path

This gives stable matching across copied, merged, or relocated datasets while changing automatically if the actual training content changes.

Alternatives considered:

- use the row’s existing `id`: rejected because many datasets omit it or make it unstable across exports
- hash the entire row: rejected because annotations and unrelated metadata would change the identifier
- use `(source_file, line_number)`: rejected because migration across dataset revisions would fail

### 4. Separate mirrored dataset outputs from process artifacts

The run layout will be formalized as:

- `<run_root>/<input_name>/...` mirrored dataset tree with inline-labeled JSONL files
- `<run_root>/meta_label_data/...` for stats, monitors, failures, checkpoints, migration indexes, and flattened caches
- dashboards at `<run_root>/dashboard_*.html`

All modules will use a shared layout helper instead of hardcoding file names and directories. This is necessary because the current code scatters output paths across `pipeline.py`, `scoring.py`, `recompute.py`, and dashboard generators.

Alternative considered: write process artifacts beside each mirrored JSONL file. Rejected because it pollutes the curated dataset tree and makes downstream file combination harder.

### 5. Define run modes explicitly

Inline persistence requires clear semantics for how existing annotations are treated. The CLI and launcher will expose four modes:

- `incremental`: skip rows whose embedded annotation already satisfies the requested stage and label version; process only missing or partial rows
- `refresh`: recompute the targeted stages and replace the full `data_label`
- `migrate`: copy `data_label` from a reference dataset by `data_id`, then run incremental labeling on unmatched or incomplete rows
- `recompute`: rebuild rarity, statistics, conversation aggregates, and dashboards from embedded annotations without LLM calls

Alternative considered: infer behavior implicitly from output directory state. Rejected because the difference between “skip”, “replace”, and “copy then fill” is user-visible and high impact.

### 6. Preserve multi-turn fidelity with both per-turn and conversation-level state

For multi-turn rows, `normalize_and_slice()` already produces one slice per assistant reply. The embedded schema will preserve that granularity in `data_label.turns[*]`, keyed by `turn_index`, while also storing a `data_label.conversation` aggregate.

This allows:

- exact reconstruction of turn-level label/value state
- shared statistics across single-turn and multi-turn datasets
- conversation-level filtering without re-running aggregation

Conversation keys used for run-local aggregation will incorporate output-relative file identity plus row `data_id`, not `data_id` alone, so duplicate rows in different files remain distinct within a run.

Alternative considered: store only the merged conversation summary for multi-turn rows. Rejected because it loses per-reply provenance and blocks turn-level filtering or manual review.

## Risks / Trade-offs

- [Large refactor surface] → Sequence implementation behind a shared row/slice adapter layer so downstream tools can switch one by one.
- [Schema drift during rollout] → Version `data_label.meta.schema_version` and keep translation helpers in one module.
- [Chunked rewrite corruption] → Use temp files plus atomic replace for mirrored JSONL writes and keep checkpoints under `meta_label_data/`.
- [Migration false positives or false negatives] → Restrict `data_id` to canonicalized training content and add fixture tests for identical vs changed rows.
- [Performance regressions from repeated flattening] → Allow rebuildable flattened caches under `meta_label_data/cache/` for scoring, recompute, and dashboard paths.
- [Legacy tool breakage] → Keep read compatibility for existing `labeled.json[l]` / `scored.json[l]` during transition and cut consumers over incrementally.

## Migration Plan

1. Introduce the row-centered schema helpers and mirrored run layout without removing legacy readers.
2. Move Pass 1 persistence to inline `data_label` for JSONL runs and keep temporary flatten caches for downstream compatibility.
3. Move Pass 2, recompute, rarity refresh, filter, and dashboards to embedded annotations.
4. Add migration mode and launcher support once `data_id` and inline readers are stable.
5. Deprecate legacy `labeled.json[l]` / `scored.json[l]` as primary outputs after all maintenance commands read inline datasets directly.

Rollback remains possible during rollout because legacy readers can continue to consume temporary flattened caches produced under `meta_label_data/`.

## Open Questions

- Whether non-JSONL inputs should be normalized into the mirrored inline layout in this same change or remain on legacy materialized outputs for now.
- Whether `label_version` should be a manually bumped constant, a prompt fingerprint, or a combined schema-plus-prompt digest.
- How much legacy aliasing should remain once inline datasets become the default, especially for review/export tools that currently assume sample-root fields.
