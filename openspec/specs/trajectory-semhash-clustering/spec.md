## ADDED Requirements

### Requirement: Long trajectory windowing with pinned prefix
The system SHALL segment trajectories longer than 50 turns into multiple logically complete windows using sliding windows. Each window SHALL include a pinned task-definition prefix and a non-overlapping logical body range metadata (`window_index`, `turn_start`, `turn_end`, `total_windows`, `source_id`).

#### Scenario: Segmenting a long trajectory
- **WHEN** a trajectory contains more than 50 turns
- **THEN** the system SHALL emit multiple windows using configured `window_size` and `stride`, and SHALL include pinned-prefix context in every emitted window

#### Scenario: Keeping short trajectory unchanged
- **WHEN** a trajectory contains 50 turns or fewer
- **THEN** the system SHALL emit a single window with no additional sliding segmentation

### Requirement: Bilingual lightweight semantic embeddings
The system SHALL produce bilingual semantic embeddings for each window body (Chinese and English) using a lightweight embedding provider suitable for CPU-first deployment. Embedding generation SHALL support provider abstraction with local model and API fallback modes under a unified output schema.

#### Scenario: Local CPU embedding path
- **WHEN** local embedding mode is enabled
- **THEN** the system SHALL compute embeddings locally and persist normalized vectors for downstream SemHash and ANN stages

#### Scenario: API fallback embedding path
- **WHEN** local embedding mode is unavailable or explicitly disabled
- **THEN** the system SHALL call the configured embedding API and persist vectors in the same normalized format as local mode

### Requirement: Deterministic SemHash generation
The system SHALL compute deterministic semantic hash signatures for each embedding using fixed-seed random hyperplanes. The hash configuration (bit width, seed, projection version) SHALL be persisted in run metadata to ensure reproducibility.

#### Scenario: Reproducible hash outputs
- **WHEN** the same window content is processed with the same embedding output and hash configuration
- **THEN** the system SHALL produce identical SemHash bits across runs

### Requirement: SemHash plus ANN clustering pipeline
The system SHALL perform clustering using a two-stage retrieval process: coarse SemHash candidate selection followed by ANN-based similarity refinement. Final cluster assignment SHALL be derived from deterministic graph merge logic over refined neighbor links.

#### Scenario: Cluster formation from approximate neighbors
- **WHEN** windows share SemHash neighborhoods and pass ANN similarity thresholds
- **THEN** the system SHALL assign them to the same cluster ID

#### Scenario: Isolation of unrelated windows
- **WHEN** windows do not pass SemHash/ANN thresholds
- **THEN** the system SHALL keep them in separate clusters

### Requirement: Representative selection by trajectory SNR
For each cluster, the system SHALL select one representative window by maximizing trajectory signal-to-noise ratio `snr = action_tokens / max(observation_tokens, 1)` computed on non-duplicated window body content. If SNR ties occur, the system SHALL use existing `value.value_score` as secondary tie-break; if still tied, the system SHALL apply deterministic ordering by source identifiers.

#### Scenario: Selecting representative by SNR
- **WHEN** a cluster contains multiple candidate windows
- **THEN** the window with highest SNR SHALL be marked as representative

#### Scenario: Deterministic tie-break behavior
- **WHEN** two or more windows have identical SNR
- **THEN** the system SHALL choose the one with higher `value.value_score`, then deterministic source ordering if needed

### Requirement: Full-batch outputs and incremental-ready state
The system SHALL produce full-batch clustering artifacts including window metadata, semhash bits, cluster IDs, representative flags, and run statistics. It SHALL also persist index/state manifests required for future incremental updates without changing output contracts.

#### Scenario: Full-batch artifact generation
- **WHEN** full-batch semantic clustering is executed
- **THEN** the system SHALL output reproducible clustering artifacts and a manifest describing all index/hash parameters used

#### Scenario: State reuse readiness
- **WHEN** a prior run manifest exists
- **THEN** the system SHALL allow a future incremental mode to reuse persisted state without requiring schema migration
