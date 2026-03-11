## Why

The current pipeline materializes normalized `labeled.json[l]` and `scored.json[l]` datasets as the primary outputs, which breaks the original JSONL row structure and makes downstream curation harder. Users need labeled datasets to remain line-aligned with the source files so they can merge, down-sample, delete bad rows, and train directly on the annotated files without extra conversion steps.

## What Changes

- Replace sample-root `labels` / `value` persistence with row-embedded `extra_info.unique_info.data_label` persistence for JSONL outputs, while preserving unrelated `extra_info` and `unique_info` content.
- Add stable per-row `extra_info.unique_info.data_id` generation derived from `meta_prompt` and `data` so labels can be migrated, refreshed, and incrementally completed across dataset revisions.
- Introduce explicit run modes for inline datasets: incremental labeling, full refresh, migration plus fill-in, and offline recomputation.
- Redesign output layout so labeled runs mirror the input directory tree and line counts, while process artifacts, logs, caches, and checkpoints live under `meta_label_data/`.
- Rework scoring, rarity refresh, recompute, filtering, and dashboard generation to use inline `data_label` as the source of truth, with temporary flattened caches allowed only as rebuildable process artifacts.
- Preserve multi-turn detail by storing per-assistant-turn results and conversation-level aggregates inside `data_label`, enabling shared statistics and dashboards without losing turn-level provenance.

## Capabilities

### New Capabilities
- `inline-data-labeling`: Persist Pass 1 and Pass 2 results inside each JSONL row, mirror input structure in labeled outputs, and preserve single-turn and multi-turn fidelity.
- `label-migration-ops`: Support inline dataset workflows for incremental labeling, full refresh, label migration via `data_id`, and offline recomputation from embedded metadata.

### Modified Capabilities
- `value-scoring`: Change Pass 2 input/output behavior to read from and write to embedded `data_label` records instead of standalone scored sample files.
- `value-dashboard`: Regenerate statistics and dashboards from inline-labeled datasets and run metadata instead of `labeled.json[l]` / `scored.json[l]` as the primary source.

## Impact

- Affected code: `preprocessing.py`, `pipeline.py`, `scoring.py`, `conversation.py`, `cli.py`, `launcher.py`, `tools/recompute.py`, `tools/filter_value.py`, dashboard generators, export/review helpers, and related tests.
- Affected data contracts: JSONL row schema for Pangu-format outputs, run directory layout, maintenance command inputs, and filter/export paths.
- Affected operational behavior: resume/checkpoint logic, chunked processing, migration/index building, and multi-turn flattening caches under `meta_label_data/`.
