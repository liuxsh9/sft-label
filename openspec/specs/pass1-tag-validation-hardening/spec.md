# pass1-tag-validation-hardening Specification

## Purpose
TBD - created by archiving change harden-pass1-tag-validation. Update Purpose after archive.
## Requirements
### Requirement: Single-select tag validation SHALL be type-safe
Pass 1 validation SHALL handle malformed single-select dimension values without runtime exceptions.

#### Scenario: List returned for single-select dimension
- **WHEN** model output sets `intent` or `difficulty` or `context` to a list value
- **THEN** validation SHALL not throw and SHALL record a validation issue

#### Scenario: Object returned for single-select dimension
- **WHEN** model output sets a single-select dimension to an object value
- **THEN** validation SHALL not throw and SHALL sanitize the dimension to empty value

#### Scenario: Valid string remains unchanged
- **WHEN** model output sets a valid string in-pool single-select value
- **THEN** validation SHALL preserve existing behavior and keep that value

### Requirement: Malformed single-select values SHALL remain observable
Validation output SHALL preserve diagnostics for malformed single-select values.

#### Scenario: Malformed value appears in issues
- **WHEN** a malformed single-select value is encountered
- **THEN** validation issues SHALL include the affected dimension and invalid type detail

#### Scenario: Cleaned payload stays JSON-safe
- **WHEN** malformed values are processed
- **THEN** cleaned labels and unmapped entries SHALL remain JSON-serializable and safe for downstream stats

