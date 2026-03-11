## ADDED Requirements

### Requirement: Stable row identity generation
The system SHALL compute `extra_info.unique_info.data_id` from a deterministic canonical representation of the row’s `meta_prompt` and `data` content. Rows with identical canonicalized training content SHALL receive the same `data_id`; rows with changed training content SHALL receive a different `data_id`.

#### Scenario: Equivalent content yields the same identity
- **WHEN** two rows have the same `meta_prompt` and `data` content but differ in file path, row number, or unrelated metadata
- **THEN** the system SHALL compute the same `data_id` for both rows

#### Scenario: Content change yields a different identity
- **WHEN** a row’s `meta_prompt` or `data` content changes
- **THEN** the system SHALL compute a different `data_id` from the previous version of that row

### Requirement: Explicit inline run modes
The system SHALL expose inline dataset modes for `incremental`, `refresh`, `migrate`, and `recompute`, and each mode SHALL have distinct behavior for how existing embedded annotations are treated.

#### Scenario: Incremental mode skips completed rows
- **WHEN** the system runs in incremental mode and a row already contains embedded annotations that satisfy the requested stage and active label version
- **THEN** the row SHALL be skipped for that stage

#### Scenario: Refresh mode replaces embedded annotations
- **WHEN** the system runs in refresh mode
- **THEN** it SHALL recompute the targeted stages and replace the row’s embedded `data_label` payload for those stages

#### Scenario: Recompute mode performs no LLM calls
- **WHEN** the system runs in recompute mode
- **THEN** it SHALL rebuild rarity, statistics, conversation aggregates, and dashboards from embedded annotations without invoking the labeling or scoring LLMs

### Requirement: Annotation migration with fill-in
The system SHALL support migration from one inline-labeled dataset to another by indexing source rows on `data_id`, copying the matching embedded annotations into the target dataset, and optionally running incremental labeling on target rows that remain unmatched or incomplete.

#### Scenario: Matching row copies embedded annotations
- **WHEN** migrate mode finds a source row with the same `data_id` as a target row
- **THEN** the target row SHALL receive a copy of the source row’s `data_label`

#### Scenario: Unmatched row falls back to incremental labeling
- **WHEN** migrate mode does not find a matching source row for a target row
- **THEN** the system SHALL leave that row eligible for incremental labeling rather than failing the migration

### Requirement: Offline maintenance uses embedded annotations as source of truth
The system SHALL compute rarity refresh, stats recomputation, dashboard regeneration, and inline filtering from embedded `data_label` data rather than requiring standalone `labeled.json[l]` or `scored.json[l]` files as the primary input contract.

#### Scenario: Recompute from mirrored inline dataset
- **WHEN** a maintenance command targets a mirrored inline-labeled dataset tree
- **THEN** the command SHALL read the embedded annotations from the JSONL rows and write rebuilt process artifacts under `meta_label_data/`

#### Scenario: Temporary flattened caches remain rebuildable
- **WHEN** a maintenance command generates a temporary flattened cache for performance
- **THEN** that cache SHALL be written under `meta_label_data/` and SHALL be treated as rebuildable process state rather than as the canonical data source
