# pass2-retry-consistency Specification

## Purpose
TBD - created by archiving change align-pass2-retry-semantics. Update Purpose after archive.
## Requirements
### Requirement: Pass 2 SHALL interpret sample_max_retries as additional retries
Pass 2 sample scoring SHALL attempt the initial call plus `sample_max_retries` additional retries on retryable failures.

#### Scenario: One configured retry
- **WHEN** `sample_max_retries` is set to `1`
- **THEN** Pass 2 SHALL attempt up to 2 total LLM calls per sample before final failure

#### Scenario: Zero configured retries
- **WHEN** `sample_max_retries` is set to `0`
- **THEN** Pass 2 SHALL attempt exactly 1 LLM call per sample

#### Scenario: Non-retryable failure
- **WHEN** a non-retryable error is returned
- **THEN** Pass 2 SHALL stop early without consuming remaining retry budget

### Requirement: Selection smoothing prior SHALL be runtime-configurable
Selection ranking computations SHALL use `PipelineConfig.selection_smoothing_prior` when provided.

#### Scenario: Custom smoothing prior in sample selection
- **WHEN** a custom `selection_smoothing_prior` is set in config
- **THEN** `compute_selection_scores` SHALL use that value instead of module-level constant

#### Scenario: Custom smoothing prior in chunked selection
- **WHEN** a custom `selection_smoothing_prior` is set in config
- **THEN** `compute_selection_scores_from_summaries` SHALL use that value instead of module-level constant

