# conversation-identity-grouping Specification

## Purpose
TBD - created by archiving change harden-conversation-id-grouping. Update Purpose after archive.
## Requirements
### Requirement: Conversation aggregation SHALL use collision-safe identity
Conversation-level aggregation SHALL group slices using a canonical conversation identity key.

#### Scenario: Canonical key with source file
- **WHEN** a slice has both `metadata.source_file` and `metadata.source_id`
- **THEN** aggregation SHALL group by `source_file::source_id`

#### Scenario: Legacy fallback key
- **WHEN** `metadata.source_file` is absent and `metadata.source_id` is present
- **THEN** aggregation SHALL group by `source_id`

#### Scenario: Cross-file source ID collision
- **WHEN** two files contain conversations with the same `source_id`
- **THEN** aggregation SHALL produce separate conversation records for each file

### Requirement: Conversation-level filtering SHALL resolve canonical IDs
Filtering with conversation-level thresholds SHALL use the same canonical identity as aggregation.

#### Scenario: Canonical conversation record lookup
- **WHEN** `conversation_scores.json` contains canonical conversation IDs
- **THEN** filter matching SHALL resolve slices using canonical key construction from sample metadata

#### Scenario: Legacy conversation score compatibility
- **WHEN** canonical lookup misses and a legacy `source_id` record exists
- **THEN** filter matching SHALL fallback to legacy `source_id` lookup

