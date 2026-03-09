## ADDED Requirements

### Requirement: Hard gates SHALL apply deterministic missing-field policy
When a hard gate criterion is configured, filter behavior for missing required fields SHALL follow an explicit policy.

#### Scenario: Strict policy drops missing correctness
- **WHEN** `correctness_min` is set and `value.quality.correctness` is missing and policy is `fail`
- **THEN** the sample SHALL be dropped

#### Scenario: Strict policy drops missing thinking mode
- **WHEN** `thinking_mode` filter is set and both value and metadata thinking mode are missing and policy is `fail`
- **THEN** the sample SHALL be dropped

#### Scenario: Permissive policy keeps missing fields
- **WHEN** a hard gate field is missing and policy is `ignore`
- **THEN** the missing field SHALL not cause a drop by itself

### Requirement: Turn-level gates SHALL honor missing-field policy
Turn-level pruning criteria SHALL apply the same missing-field policy semantics.

#### Scenario: Strict policy for turn quality gate
- **WHEN** `turn_quality_min` is set and a slice lacks `value.quality.overall` and policy is `fail`
- **THEN** the slice SHALL be treated as failing the turn gate

#### Scenario: Strict policy for turn value gate
- **WHEN** `turn_value_min` is set and a slice lacks `value.value_score` and policy is `fail`
- **THEN** the slice SHALL be treated as failing the turn gate

### Requirement: Filter summary SHALL report missing-field drops
Filter summary output SHALL include counts of drops due to missing required gate fields.

#### Scenario: Missing-field drop reporting
- **WHEN** strict policy is active and samples are dropped due to missing fields
- **THEN** summary output SHALL include per-criterion missing drop counts
