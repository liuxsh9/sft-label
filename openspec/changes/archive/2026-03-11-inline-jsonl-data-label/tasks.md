## 1. Schema And Layout Foundations

- [x] 1.1 Add shared helpers for canonical `data_id` generation and embedded `data_label` schema read/write under `extra_info.unique_info`
- [x] 1.2 Add a shared run-layout helper that resolves mirrored dataset paths, `meta_label_data/` artifact paths, and run-root dashboard paths
- [x] 1.3 Add focused unit tests and fixtures for inline row persistence, sibling-field preservation, and deterministic `data_id` behavior

## 2. Pass 1 Inline Persistence

- [x] 2.1 Refactor JSONL ingestion to build row-centered objects and transient slice samples without losing source row alignment
- [x] 2.2 Update Pass 1 single-file, chunked, and directory writers to rewrite mirrored JSONL outputs with embedded `data_label` instead of `labeled.json[l]` as the primary output
- [x] 2.3 Preserve sparse-sampling inheritance, per-turn metadata, and row-local monitoring summaries inside embedded annotations

## 3. Pass 2 Inline Scoring

- [x] 3.1 Add inline scoring readers that flatten embedded Pass 1 turn labels back into transient scoring samples
- [x] 3.2 Rewrite Pass 2 persistence to store per-turn `value` results and conversation aggregates back into `data_label`
- [x] 3.3 Move Pass 2 stats, monitors, failure logs, and temporary flattened caches under `meta_label_data/` while keeping dashboards at the run root

## 4. Maintenance And Derived Views

- [x] 4.1 Refactor recompute and refresh-rarity commands to read embedded annotations as the source of truth and to write rebuilt artifacts under `meta_label_data/`
- [x] 4.2 Refactor dashboard generation to consume embedded annotations or rebuildable flattened caches from mirrored dataset trees
- [x] 4.3 Refactor filter, export-review, and related tooling to operate on inline-labeled rows while preserving single-turn and multi-turn semantics

## 5. Migration And CLI Modes

- [x] 5.1 Add CLI and launcher support for `incremental`, `refresh`, `migrate`, and `recompute` modes with clear user-facing prompts
- [x] 5.2 Implement migration indexing and copy-forward behavior based on `data_id`, followed by incremental fill-in for unmatched or incomplete rows
- [x] 5.3 Update resume, checkpoint, and run-summary behavior to work with mirrored inline outputs and mode-aware processing

## 6. Verification

- [x] 6.1 Add unit tests for row-to-slice flattening, slice-to-row merge, multi-turn embedding, refresh replacement, and migration provenance metadata
- [x] 6.2 Add integration tests for Pass 1, Pass 2, recompute, refresh-rarity, and filter on inline mirrored JSONL fixtures
- [x] 6.3 Add directory-level and e2e tests for mirrored structure invariants, line-count preservation, migration flow, and dashboard regeneration
