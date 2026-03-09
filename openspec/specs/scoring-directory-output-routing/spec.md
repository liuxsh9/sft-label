# scoring-directory-output-routing Specification

## Purpose
TBD - created by archiving change fix-pass2-directory-output-routing. Update Purpose after archive.
## Requirements
### Requirement: Directory scoring SHALL honor explicit output root
When `run_scoring` is invoked on a directory with `output_dir` provided, the system SHALL write all per-file artifacts and run-level artifacts under that output root.

#### Scenario: Output root does not exist
- **WHEN** directory scoring starts with a non-existent `output_dir`
- **THEN** the system SHALL create the directory before writing summary artifacts

#### Scenario: Per-file output is mirrored under output root
- **WHEN** labeled files are discovered in nested folders under the input root
- **THEN** each file's scored artifacts SHALL be written to the mirrored relative folder under `output_dir`

#### Scenario: Global artifacts stay in output root
- **WHEN** directory scoring completes
- **THEN** `summary_stats_scoring.json` and global dashboard artifacts SHALL be written in `output_dir`

### Requirement: Global conversation aggregation SHALL include nested scored outputs
Directory-mode global conversation aggregation SHALL discover scored outputs recursively under the run output root.

#### Scenario: Nested scored outputs are present
- **WHEN** scored artifacts exist deeper than one folder level under `output_dir`
- **THEN** the system SHALL include those samples in global `conversation_scores.json`

#### Scenario: Mixed scored JSON and JSONL artifacts
- **WHEN** both `scored.json` and `scored.jsonl` are present for the same folder
- **THEN** aggregation SHALL use a deterministic preference and avoid duplicate ingestion

