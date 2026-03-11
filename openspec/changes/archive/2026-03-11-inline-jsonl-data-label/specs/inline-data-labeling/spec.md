## ADDED Requirements

### Requirement: Embedded row annotation contract
The system SHALL persist JSONL row annotations under `extra_info.unique_info` using two fields: `data_id` and `data_label`. The system SHALL create missing `extra_info` and `unique_info` containers when absent and SHALL preserve all unrelated existing fields under those containers.

#### Scenario: Create annotation containers without overwriting siblings
- **WHEN** a source row lacks `extra_info` or `extra_info.unique_info`
- **THEN** the system SHALL create the missing parent objects and write `data_id` and `data_label` without removing any other top-level row fields

#### Scenario: Refresh only replaces the embedded annotation payload
- **WHEN** the system runs in refresh mode for a row that already contains `extra_info.unique_info.data_label`
- **THEN** it SHALL replace the full `data_label` object for that row while preserving sibling fields such as other `extra_info` keys and other `unique_info` keys

### Requirement: Row-aligned mirrored dataset output
The system SHALL write labeled datasets as mirrored JSONL files that preserve the source file structure, row order, and row count. For each input JSONL file, the corresponding output JSONL file SHALL contain the same number of lines in the same order, with annotations added only through the embedded fields defined by this capability.

#### Scenario: Mirror directory input structure
- **WHEN** the input path is a directory tree of JSONL files
- **THEN** the run output SHALL contain a mirrored dataset subtree with the same relative file layout and a corresponding annotated JSONL file for each input JSONL file

#### Scenario: Preserve line alignment in chunked processing
- **WHEN** a large JSONL file is processed in chunked mode
- **THEN** the output file SHALL still contain exactly one output row for each input row in the original order

### Requirement: Embedded multi-turn annotation fidelity
The system SHALL preserve multi-turn annotation detail inside each row’s `data_label`. `data_label.turns` SHALL contain one entry per assistant reply with its `turn_index` and slice-level results. `data_label.conversation` SHALL contain the conversation-level aggregate derived from those turns.

#### Scenario: Single-turn row stores one turn result
- **WHEN** the input row contains a single assistant reply
- **THEN** `data_label.turns` SHALL contain exactly one turn record and `data_label.conversation.turn_count` SHALL equal `1`

#### Scenario: Multi-turn row stores per-reply results and aggregate
- **WHEN** the input row contains multiple assistant replies
- **THEN** `data_label.turns` SHALL contain one record for each assistant reply in turn order and `data_label.conversation` SHALL contain the aggregated conversation metrics computed from those turn records

### Requirement: Embedded annotation metadata
The system SHALL store annotation management metadata inside `data_label.meta`, including at minimum a schema version, label version, and stage timestamps needed to distinguish incremental labeling, refreshed labels, migrated labels, and recomputed artifacts.

#### Scenario: Newly labeled row records annotation metadata
- **WHEN** a row is labeled for the first time
- **THEN** `data_label.meta` SHALL include the active schema version, label version, and a timestamp for the completed stage

#### Scenario: Migrated row records provenance
- **WHEN** a row’s annotations are copied from a matching source dataset
- **THEN** `data_label.meta` SHALL record that the row was populated via migration before any subsequent fill-in or refresh

### Requirement: Run artifact isolation
The system SHALL place process artifacts under a dedicated `meta_label_data/` directory in the run root. Process artifacts include checkpoints, logs, monitors, failures, migration indexes, flattened caches, and per-stage stats files. Dashboard HTML files SHALL remain at the run root.

#### Scenario: Labeled run layout
- **WHEN** a labeling run completes for an input dataset named `llm_sft_data_v1`
- **THEN** the run root SHALL contain a mirrored dataset subtree for `llm_sft_data_v1`, a `meta_label_data/` subtree for process artifacts, and dashboard HTML files at the top level of the run root
