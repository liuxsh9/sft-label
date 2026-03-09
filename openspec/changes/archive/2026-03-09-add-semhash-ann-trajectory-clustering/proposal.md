## Why

The current pipeline can label and score SFT samples, but it lacks a scalable way to group multi-turn trajectories by semantic similarity and quickly select high-value representatives. With growing data volume (target 8M trajectories) and CPU-only infrastructure, we need a low-cost pipeline that first captures trajectory semantics via lightweight embeddings and semantic hashing, then performs near-neighbor clustering and representative selection aligned with SFT value.

## What Changes

- Add a trajectory preprocessing stage for long conversations: when a trajectory exceeds 50 turns, split it into logically complete windows using sliding windows with a pinned task-definition prefix.
- Add bilingual (Chinese/English) semantic vectorization for trajectory windows using lightweight embedding models suitable for CPU-first operation and optional API fallback.
- Add SemHash generation and ANN-assisted candidate retrieval to support fast large-scale clustering (full-batch first, incremental-ready design).
- Add cluster assembly and representative-sample selection based on trajectory signal-to-noise ratio (action/observation token ratio), with deterministic tie-breaking.
- Add offline artifacts and metrics for cluster quality, throughput, and reproducibility (window metadata, hash bits, cluster IDs, representative flags, run stats).
- Add CLI entry points for full-batch semantic clustering now, with incremental update mode designed as the next operational step.

## Capabilities

### New Capabilities
- `trajectory-semhash-clustering`: Segment long trajectories with pinned-prefix windows, compute bilingual semantic fingerprints, perform ANN-assisted clustering at large scale, and select representative trajectories by SNR for downstream SFT curation.

### Modified Capabilities
- `value-scoring`: Add compatibility requirements so representative selection can consume existing per-sample value outputs (for tie-breaking and downstream filtering interoperability) without changing current scoring behavior.

## Impact

- Affected code: preprocessing and conversation-level processing, new semantic clustering modules, CLI wiring, and data export/reporting utilities.
- Affected outputs: new trajectory window and clustering artifacts (for example JSONL/Parquet records with source trajectory linkage, window ranges, semhash, cluster membership, and representative flags).
- Dependencies: ANN and embedding runtime components suitable for CPU-first deployment, with configurable model provider fallback.
- Systems: batch data curation workflow, dataset quality analysis, and future incremental refresh workflow.
