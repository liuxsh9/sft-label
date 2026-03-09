## 1. Data Model and Configuration

- [x] 1.1 Add semantic clustering config fields (window size/stride, pinned-prefix rules, embedding provider, semhash bits, ANN thresholds, output paths) in `config.py` and `PipelineConfig`
- [x] 1.2 Define trajectory window and clustering artifact schemas (window metadata, hash fields, cluster ID, representative flag, run manifest)
- [x] 1.3 Add validation and defaulting logic for clustering config in CLI/runtime entrypoints

## 2. Trajectory Windowing with Pinned Prefix

- [x] 2.1 Implement pinned-prefix extraction rules for task-definition context from normalized conversations
- [x] 2.2 Implement sliding-window segmentation for trajectories over 50 turns with pair-aligned boundaries
- [x] 2.3 Emit deterministic window metadata (`source_id`, `window_index`, `turn_start`, `turn_end`, `total_windows`) and unit tests

## 3. Embedding Provider Layer (CPU-first, bilingual)

- [x] 3.1 Create embedding provider interface with local CPU backend and API fallback backend
- [x] 3.2 Implement bilingual text rendering for embeddings (role-aware formatting for zh/en conversations)
- [x] 3.3 Add batching/concurrency controls and persistent vector output format for large-batch runs

## 4. SemHash and Candidate Retrieval

- [x] 4.1 Implement deterministic SemHash projection with fixed-seed hyperplanes and versioned parameters
- [x] 4.2 Implement SemHash band indexing and coarse candidate generation by Hamming neighborhoods
- [x] 4.3 Persist SemHash artifacts and diagnostics (bit distribution, bucket size distribution, reproducibility checks)

## 5. ANN Refinement and Cluster Assembly

- [x] 5.1 Implement ANN refinement stage over SemHash candidates for cosine-space approximate neighbors
- [x] 5.2 Implement graph merge logic (connected components / union-find) for final cluster assignment
- [x] 5.3 Add cluster quality metrics (size distribution, singleton ratio, threshold hit rates) and failure handling

## 6. Representative Selection by SNR

- [x] 6.1 Implement action/observation token accounting on non-duplicated window body content
- [x] 6.2 Implement representative selection logic (SNR primary, `value.value_score` secondary, deterministic tertiary tie-break)
- [x] 6.3 Export representative markers and per-cluster summary records for downstream filtering

## 7. CLI and Pipeline Integration

- [x] 7.1 Add full-batch semantic clustering CLI command(s) and argument wiring
- [x] 7.2 Integrate with existing preprocessing/conversation pipeline without changing current Pass1/Pass2 behavior
- [x] 7.3 Add report/export tooling hooks for cluster outputs and representative-only dataset generation

## 8. Incremental-Ready State Persistence

- [x] 8.1 Define and write index-state manifest (hash seed/version, ANN params, thresholds, checkpoints)
- [x] 8.2 Implement reusable load/save state helpers for future incremental updates
- [x] 8.3 Add compatibility checks to prevent mixing incompatible manifests across runs

## 9. Testing and Benchmarking

- [x] 9.1 Add unit tests for segmentation, pinned prefix extraction, semhash determinism, and SNR ranking
- [x] 9.2 Add integration tests for end-to-end full-batch clustering on smoke fixtures (including bilingual samples)
- [x] 9.3 Add CPU benchmark script/report for throughput and memory (embedding stage vs clustering stage) with documented 8M scaling assumptions
