## ADDED Requirements

### Requirement: Rarity computation from tag distributions
The system SHALL compute a rarity score (1-10) for each scored turn using tag IDF values from a Pass 1 tag distribution stats file. The rarity score SHALL combine weighted per-dimension tag IDF with cross-dimension combo rarity and SHALL normalize raw rarity values to a 1-10 scale according to the configured mode. In inline dataset mode, the auto-discovered baseline SHALL come from the run’s embedded-label process artifacts rather than from standalone `labeled.json` outputs.

#### Scenario: Rarity from auto-discovered inline stats
- **WHEN** scoring an inline-labeled dataset and the run contains Pass 1 stats under `meta_label_data/`
- **THEN** the system SHALL read `tag_distributions` from that stats file and compute rarity for each scored turn

#### Scenario: Rarity from explicit stats file
- **WHEN** `--tag-stats global_stats.json` is provided
- **THEN** the system SHALL use the specified file's `tag_distributions` for rarity computation

#### Scenario: No stats available
- **WHEN** no stats file is found or specified
- **THEN** the system SHALL fall back to the configured local-baseline behavior and SHALL emit a warning when rarity cannot be computed from an external distribution

#### Scenario: Stats reference metadata
- **WHEN** rarity is computed for a scored turn
- **THEN** the embedded rarity record SHALL contain `stats_ref.source`, `stats_ref.total_samples`, and `stats_ref.timestamp`

### Requirement: COT-preserving smart truncation
The system SHALL provide a truncation mode that preserves chain-of-thought content for quality evaluation. The total budget (default 20K chars) SHALL be allocated as: instruction 15%, COT 45%, response 35%, meta 5%. For COT content exceeding its budget, the system SHALL keep head (30% of COT budget) + 2-3 evenly-spaced middle fragments (10% each) + tail (30% of COT budget), with position markers between gaps.

#### Scenario: Slow-thinking sample truncation
- **WHEN** a sample contains `<think>...</think>` blocks and total content exceeds budget
- **THEN** COT content SHALL be truncated using head + middle fragments + tail strategy with markers like `[... N chars omitted, fragment at X% ...]`

#### Scenario: Fast-thinking sample truncation
- **WHEN** a sample has no COT markers
- **THEN** COT budget (45%) SHALL merge into response budget (total 80%), and response SHALL use head 30% + middle fragments + tail 30%

#### Scenario: Short content no truncation
- **WHEN** total conversation content fits within budget
- **THEN** no truncation SHALL occur and full content SHALL be passed to the LLM

### Requirement: Thinking mode auto-detection
The system SHALL auto-detect whether a sample uses slow-thinking or fast-thinking by checking for `<think>`, `<thinking>`, or `[unused16]...[unused17]` markers in the raw conversation data before any stripping.

#### Scenario: Slow thinking detected
- **WHEN** conversation contains `<think>` or `[unused16]` markers
- **THEN** `thinking_mode` SHALL be set to `"slow"` and COT content SHALL be preserved for truncation

#### Scenario: Fast thinking default
- **WHEN** no COT markers are found
- **THEN** `thinking_mode` SHALL be set to `"fast"`

### Requirement: LLM-based value scoring
The system SHALL make one LLM call per sample to evaluate complexity (1-10), quality (1-10), and reasoning quality (1-10), each with sub-scores. The call SHALL receive the truncated conversation, Pass 1 tags as context, and meta information (thinking_mode, original lengths).

#### Scenario: Successful scoring
- **WHEN** the LLM returns valid JSON with all required fields
- **THEN** the system SHALL parse complexity (instruction/reasoning/implementation/overall), quality (correctness/code_quality/explanation/completeness/overall), reasoning (adapts to thinking_mode), flags array, and confidence score

#### Scenario: LLM call failure with retry
- **WHEN** the LLM call fails or returns invalid JSON
- **THEN** the system SHALL retry up to `SAMPLE_MAX_RETRIES` times, then mark the sample as failed

### Requirement: Value score aggregation
The system SHALL compute a composite `value_score` per sample as a weighted sum of complexity.overall, quality.overall, reasoning.overall, and rarity.score (normalized). Default weights: complexity=0.25, quality=0.35, reasoning=0.15, rarity=0.25. All weights SHALL be configurable via PipelineConfig.

#### Scenario: All dimensions available
- **WHEN** LLM scoring and rarity computation both succeed
- **THEN** `value_score` SHALL be the weighted mean of all four dimension scores

#### Scenario: Rarity unavailable
- **WHEN** rarity is null (no stats file)
- **THEN** `value_score` SHALL be computed from the three LLM-scored dimensions only, with weights renormalized

### Requirement: Standalone score CLI subcommand
The system SHALL provide `sft-label score` for Pass 2 scoring on inline-labeled datasets. It SHALL accept a labeled JSONL file or mirrored labeled dataset directory as input, read Pass 1 labels from embedded `data_label` turn records, and write Pass 2 results back into the same embedded `data_label` structure.

#### Scenario: Score single mirrored file
- **WHEN** `sft-label score --input <mirrored-file-or-run-dir>` is invoked on an inline-labeled dataset
- **THEN** the system SHALL read embedded turn labels, compute Pass 2 values, rewrite the mirrored JSONL output with updated embedded `data_label` values, and write Pass 2 process artifacts under `meta_label_data/`

#### Scenario: Score with external stats
- **WHEN** `--tag-stats` is provided while scoring an inline-labeled dataset
- **THEN** the system SHALL use `global_stats.json` for rarity computation instead of auto-discovery

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

### Requirement: Scoring configuration
PipelineConfig SHALL be extended with scoring-specific fields: value scoring weights (complexity/quality/reasoning/rarity), rarity dimension weights, rarity combo alpha, value truncation budget and ratios. Module-level defaults SHALL be defined in config.py.

#### Scenario: Custom weights
- **WHEN** `PipelineConfig(value_weights={"complexity": 0.3, "quality": 0.4, "reasoning": 0.1, "rarity": 0.2})`
- **THEN** the composite value_score SHALL use the custom weights

#### Scenario: Default configuration
- **WHEN** no scoring config overrides are provided
- **THEN** defaults SHALL be used: complexity=0.25, quality=0.35, reasoning=0.15, rarity=0.25
