## MODIFIED Requirements

### Requirement: Rarity computation from tag distributions
The system SHALL compute a rarity score (1-10) for each scored turn using tag IDF values from a Pass 1 tag distribution stats file. The rarity score SHALL combine weighted per-dimension tag IDF with cross-dimension combo rarity and SHALL normalize raw rarity values to a 1-10 scale according to the configured mode. In inline dataset mode, the auto-discovered baseline SHALL come from the run’s embedded-label process artifacts rather than from standalone `labeled.json` outputs.

#### Scenario: Rarity from auto-discovered inline stats
- **WHEN** scoring an inline-labeled dataset and the run contains Pass 1 stats under `meta_label_data/`
- **THEN** the system SHALL read `tag_distributions` from that stats file and compute rarity for each scored turn

#### Scenario: Rarity from explicit stats file
- **WHEN** `--tag-stats` is provided
- **THEN** the system SHALL use the specified stats file’s `tag_distributions` for rarity computation

#### Scenario: No stats available
- **WHEN** no stats file is found or specified
- **THEN** the system SHALL fall back to the configured local-baseline behavior and SHALL emit a warning when rarity cannot be computed from an external distribution

#### Scenario: Stats reference metadata
- **WHEN** rarity is computed for a scored turn
- **THEN** the embedded rarity record SHALL contain `stats_ref.source`, `stats_ref.total_samples`, and `stats_ref.timestamp`

### Requirement: Standalone score CLI subcommand
The system SHALL provide `sft-label score` for Pass 2 scoring on inline-labeled datasets. It SHALL accept a labeled JSONL file or mirrored labeled dataset directory as input, read Pass 1 labels from embedded `data_label` turn records, and write Pass 2 results back into the same embedded `data_label` structure.

#### Scenario: Score single mirrored file
- **WHEN** `sft-label score --input <mirrored-file-or-run-dir>` is invoked on an inline-labeled dataset
- **THEN** the system SHALL read embedded turn labels, compute Pass 2 values, rewrite the mirrored JSONL output with updated embedded `data_label` values, and write Pass 2 process artifacts under `meta_label_data/`

#### Scenario: Score with external stats
- **WHEN** `sft-label score --input <dataset> --tag-stats <global-stats>` is provided
- **THEN** the system SHALL use the specified stats file for rarity computation instead of auto-discovery

### Requirement: Continuous Pass 1 + Pass 2 execution
The system SHALL support `sft-label run --score` for inline-labeled datasets. Pass 1 SHALL produce mirrored JSONL outputs with embedded Pass 1 annotations, and Pass 2 SHALL then read those embedded annotations and update the same mirrored JSONL outputs with Pass 2 values.

#### Scenario: Continuous inline mode
- **WHEN** `sft-label run --input <dataset> --score` is executed
- **THEN** the system SHALL run Pass 1 labeling first, then run Pass 2 scoring against the mirrored inline-labeled dataset produced by Pass 1, using the run’s Pass 1 stats as the default rarity baseline

### Requirement: Output file structure
Pass 2 SHALL persist results inside each row’s embedded `data_label` rather than emitting standalone `scored.json` as the primary output. Per-run process artifacts such as Pass 2 stats, monitors, failure logs, and flattened caches SHALL live under `meta_label_data/`. Dashboard HTML files SHALL live at the run root.

#### Scenario: Per-row output structure
- **WHEN** a row is successfully scored
- **THEN** the output row SHALL retain its embedded Pass 1 labels and SHALL add Pass 2 values under the matching `data_label.turns[*]` entries plus any derived conversation aggregate updates under `data_label.conversation`

#### Scenario: Directory mode output
- **WHEN** scoring a mirrored dataset directory
- **THEN** each mirrored JSONL file SHALL be rewritten in place within the run output tree, while run-level Pass 2 stats and process artifacts SHALL be written under `meta_label_data/` and dashboards SHALL be regenerated at the run root
