## Context

The repository already supports multi-turn normalization/slicing, value scoring, and conversation-level aggregation. However, there is no trajectory-level semantic indexing layer for large-scale deduplication and representative mining.

Constraints from this change:
- Compute environment is CPU-only (`~44 cores`, `~96 GB RAM`).
- Data is bilingual (`zh` + `en`) and must remain semantically comparable across languages.
- Priority is fast value capture at large scale (8M trajectories), accepting approximate clustering quality.
- Operational rollout must start with full-batch processing, but architecture must support future incremental updates.

## Goals / Non-Goals

**Goals:**
- Segment long trajectories (>50 turns) into logically complete windows while preserving global task context via pinned prefix.
- Produce lightweight bilingual semantic vectors and deterministic SemHash signatures for each window.
- Use SemHash + ANN retrieval to cluster ~8M windows efficiently on CPU-oriented infrastructure.
- Select one representative per cluster using SFT-oriented signal-to-noise ratio (action/observation).
- Persist outputs and index state so incremental update mode can be added without redesign.

**Non-Goals:**
- Replacing current Pass 1/Pass 2 labeling-scoring logic.
- Exact nearest-neighbor guarantees across the full corpus.
- Real-time online clustering in this phase.
- GPU-only optimization paths.

## Decisions

### 1. Long trajectory segmentation uses pinned-prefix sliding windows

**Decision:** For trajectories with more than 50 turns, construct windows with:
- `pinned_prefix`: task-definition context (system + earliest task-defining user turns)
- `window_body`: rolling turn range
- default `window_size=50` turns, `stride=30` turns (configurable)

Window boundaries SHALL align to complete interaction pairs to keep each window logically complete.

**Rationale:** This keeps global intent visible while reducing long-context drift and preserving local coherence for embedding and clustering.

**Alternative considered:** Hard truncation to last N turns. Rejected because it drops intent-defining context and harms semantic consistency.

### 2. Lightweight bilingual embedding abstraction with CPU-first default

**Decision:** Add an embedding provider interface with:
- local CPU-first model path (small multilingual model)
- optional API fallback path (same normalized output contract)

Vectors are L2-normalized and persisted in compact numeric format for downstream search.

**Rationale:** CPU-only environment requires low-latency small models. Provider abstraction avoids locking to one runtime and enables fallback when local throughput is insufficient.

**Alternative considered:** Large embedding models only via API. Rejected due to cost/latency risk and weaker offline reproducibility.

### 3. Two-stage semantic retrieval: SemHash coarse buckets + ANN refinement

**Decision:** Build deterministic `256-bit` SemHash per window from normalized embeddings using fixed random hyperplanes.

Clustering search path:
1. Coarse candidate generation from SemHash bands/Hamming neighborhoods.
2. ANN refinement for candidate ranking (cosine-space approximate neighbors).
3. Union-find / connected-components merge to produce cluster IDs.

**Rationale:** Pure global ANN over 8M on CPU is expensive. SemHash first drastically reduces candidate sets; ANN refinement recovers useful recall at acceptable cost.

**Alternative considered:** Single global ANN index only. Rejected for CPU memory/build-time pressure and weaker explainability/debuggability.

### 4. Representative selection is SNR-first, value-compatible

**Decision:** Per cluster, pick representative by:
1. Highest `snr = action_tokens / max(observation_tokens, 1)`
2. Tie-break with higher existing `value_score` when available
3. Final deterministic tie-break by `(source_id, window_index)`

`action_tokens` are assistant-target tokens; `observation_tokens` are context-provider tokens (user/tool). SNR is computed on non-duplicated window body (excluding repeated pinned prefix).

**Rationale:** For SFT curation, windows with stronger actionable signal are preferred. Existing value score is reused without redefining scoring semantics.

**Alternative considered:** Value-score-only selection. Rejected because it may over-prefer verbose but low-action windows.

### 5. Full-batch first, incremental-ready state model

**Decision:** Persist:
- segmentation metadata
- semhash bits and band keys
- cluster assignments
- representative flags
- index state manifest (hyperplane seed/version, band params, thresholds, checkpoints)

Full-batch CLI builds all artifacts from scratch. Incremental mode (future step) appends new windows and re-evaluates only impacted clusters using persisted state.

**Rationale:** Meets immediate full-batch objective while minimizing future rework for incremental rollout.

**Alternative considered:** Full rebuild only forever. Rejected because recurring 8M-scale recompute is operationally costly.

## Risks / Trade-offs

- `[Embedding throughput bottleneck on CPU]` -> Mitigation: batching + parallel workers + quantized local model path + API fallback flag.
- `[Approximate clustering can merge unrelated windows]` -> Mitigation: conservative similarity thresholds, Hamming radius cap, and cluster-size diagnostics.
- `[Pinned prefix extraction may be inconsistent across datasets]` -> Mitigation: explicit prefix rules + fallback heuristic + per-sample trace fields for auditing.
- `[Bilingual semantic mismatch in edge domains]` -> Mitigation: multilingual calibration sample set and language-aware quality metrics in run stats.
- `[Large-index memory pressure]` -> Mitigation: compact vector storage, chunked processing, and staged index build with checkpoints.

## Migration Plan

1. Add full-batch semantic clustering command and output schema behind a new CLI entry.
2. Run on sampled corpus for threshold calibration (recall/precision proxy + cluster size distribution).
3. Run full 8M batch, produce representative set, and validate downstream sampling quality.
4. Freeze index/state output contract (`manifest + artifact versions`).
5. Add incremental ingestion command that reuses persisted state and updates impacted clusters only.

Rollback strategy:
- keep current labeling/scoring paths unchanged;
- disable semantic clustering CLI path by configuration;
- retain previous selection pipeline as fallback.

## Open Questions

- Final default embedding model for local CPU path (quality vs throughput trade-off under bilingual traffic).
- Precise SLO wording for "minute-level": clustering-only stage vs end-to-end (including embedding).
- ANN backend selection for the first implementation (dependency footprint vs operational simplicity).
