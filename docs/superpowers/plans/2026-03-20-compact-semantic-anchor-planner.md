# Compact Semantic Anchor Planner Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace position-led sparse planning for long multi-turn conversations with a deterministic semantic anchor planner that preserves compact prompt limits, improves turn selection accuracy, and reduces unnecessary Pass 1 / Pass 2 LLM work.

**Architecture:** Add a planner layer in preprocessing that assigns segment and anchor metadata using deterministic signals, keep Pass 1 compact prompts unchanged, teach Pass 1 sparse selection and Pass 2 selective scoring to consume planner outputs, and add observability plus fallback paths so production can safely roll back to the current sparse policy.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, existing preprocessing/pipeline/scoring/conversation modules, existing compact prompt mode, existing inline + directory workflows.

---

## Scope

This plan implements the v1 design only:

1. deterministic semantic planner,
2. segment-local anchor-led sparse labeling,
3. planner-fed selective scoring,
4. compact-safe budget gates and observability,
5. tests, docs, and rollout support.

Explicitly deferred:

- any always-on new LLM scout prompt,
- changes to Pass 1 taxonomy prompts,
- changes to Pass 2 scoring prompts,
- aggregation formula rewrites,
- extension-driven Pass 2 prompt changes.

---

## Files

**Create**
- `src/sft_label/multiturn_signals.py` (if needed to avoid preprocessing/scoring coupling)

**Modify**
- `src/sft_label/config.py`
- `src/sft_label/preprocessing.py`
- `src/sft_label/pipeline.py`
- `src/sft_label/inline_pass1.py`
- `src/sft_label/inline_scoring.py`
- `src/sft_label/scoring.py`
- `src/sft_label/conversation.py`
- `docs/guides/how-sft-label-works.md`
- `docs/guides/common-workflows.md`
- `tests/test_preprocessing.py`
- `tests/test_pipeline_estimation.py`
- `tests/test_inline_pass1.py`
- `tests/test_inline_scoring.py`
- `tests/test_scoring.py`
- `tests/test_scoring_estimation.py`
- `tests/test_conversation.py`

**Optional if needed**
- `tests/test_e2e_mock.py`

---

## Chunk 1: Planner contract and deterministic segmentation

### Task 1: Add planner config knobs with conservative defaults

**Files:**
- Modify: `src/sft_label/config.py`
- Test: `tests/test_preprocessing.py`

- [ ] Add config fields for planner enablement, fallback thresholds, segment size floor, boundary threshold, anchor cap, and planner policy version.
- [ ] Keep defaults conservative so existing behavior is recoverable.
- [ ] Add tests proving new config defaults deserialize cleanly through `PipelineConfig`.
- [ ] Run targeted config/preprocessing tests and confirm failures or missing coverage before implementation.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py -q`

### Task 2: Add failing segmentation tests before implementation

**Files:**
- Modify: `tests/test_preprocessing.py`

- [ ] Add a failing test showing semantic drift creates separate segments even when turn positions would otherwise inherit.
- [ ] Add a failing test showing stable local regions remain in one segment.
- [ ] Add a failing test showing planner fallback occurs when boundary churn exceeds the configured threshold.
- [ ] Add a failing test showing planner metadata is added without changing sample structure.
- [ ] Add a failing test proving segmentation compares only local request/trajectory/final-response views, not whole-prefix growth.
- [ ] Run the targeted preprocessing tests and verify they fail for expected reasons.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py -q -k 'segment or planner or sparse'`

### Task 3: Implement the planner in preprocessing

**Files:**
- Modify: `src/sft_label/preprocessing.py`

- [ ] Add deterministic planner helpers that score boundaries using existing local signals instead of introducing LLM calls.
- [ ] If signal extraction would create a preprocessing↔scoring dependency cycle, extract shared local-view helpers into `src/sft_label/multiturn_signals.py`.
- [ ] Add metadata enrichment for `segment_id`, `boundary_score`, `anchor_priority`, `anchor_distance`, `planner_policy`, and `planner_confidence`.
- [ ] Keep the output sample schema backward-compatible apart from additive metadata.
- [ ] Add conservative merge-back and fallback behavior for low-confidence over-fragmentation.
- [ ] Preserve the current inheritance direction invariant: prefer forward inheritance within the same segment, then backward fallback only if needed.
- [ ] Re-run preprocessing tests until they pass.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py -q`

### Task 4: Review chunk 1

**Files:**
- No additional production files expected

- [ ] Dispatch one reviewer focused on planner correctness and one reviewer focused on backward compatibility.
- [ ] Fix any Important/Critical issues before continuing.

---

## Chunk 2: Pass 1 sparse planning integration

### Task 5: Add failing sparse-plan integration tests

**Files:**
- Modify: `tests/test_preprocessing.py`
- Modify: `tests/test_inline_pass1.py`
- Modify: `tests/test_pipeline_estimation.py`

- [ ] Add a failing test showing sparse labeling uses planner anchors instead of only positional schedule when planner metadata is present.
- [ ] Add a failing test showing inheritance never crosses segment boundaries.
- [ ] Add a failing test showing inheritance still prefers the next labeled slice inside the same segment.
- [ ] Add a failing test showing inline mode preserves planner-driven inheritance behavior.
- [ ] Add a failing workload-estimation test proving planner-driven labeled/inherited counts flow into the estimator.
- [ ] Run the targeted tests and verify they fail before production changes.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py tests/test_inline_pass1.py tests/test_pipeline_estimation.py -q`

### Task 6: Implement planner-aware sparse planning

**Files:**
- Modify: `src/sft_label/preprocessing.py`
- Modify: `src/sft_label/pipeline.py`
- Modify: `src/sft_label/inline_pass1.py`

- [ ] Add a planner-aware sparse planner that still returns the existing `label_indices` / `inherit_map` contract.
- [ ] Restrict inheritance source selection to the same planner segment or inherit group.
- [ ] Preserve the current positional sparse planner as a fallback path when planner metadata is unavailable or invalid.
- [ ] Ensure workload estimation uses the same planner path as real execution.
- [ ] Re-run targeted Pass 1 integration tests until they pass.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py tests/test_inline_pass1.py tests/test_pipeline_estimation.py -q`

### Task 7: Add Pass 1 observability

**Files:**
- Modify: `src/sft_label/pipeline.py`
- Modify: `src/sft_label/conversation.py`

- [ ] Add planner-related stats fields such as fallback count, segment count summary, anchor coverage, and planner confidence summary.
- [ ] Add hard-required rollout metrics: byte-budget distributions, hard-cap hit rate, cross-segment inheritance ratio, and planner metadata-only shadow deltas.
- [ ] Keep stats additive and backward-compatible.
- [ ] Add or update tests for any new stats fields only where behavior is contractually important.

Suggested commands:
- `uv run pytest tests/test_pipeline_estimation.py tests/test_conversation.py -q`

### Task 8: Review chunk 2

**Files:**
- No additional production files expected

- [ ] Dispatch reviewers focused on inline-mode merge correctness and sparse planner behavior.
- [ ] Resolve Important/Critical issues before continuing.

---

## Chunk 3: Pass 2 selective scoring integration

### Task 9: Add failing scoring tests

**Files:**
- Modify: `tests/test_scoring.py`
- Modify: `tests/test_inline_scoring.py`
- Modify: `tests/test_scoring_estimation.py`

- [ ] Add a failing test showing planner anchors are always selected for real scoring when selective scoring is enabled.
- [ ] Add a failing test showing inherited mid-turns inside stable segments still use conservative estimates.
- [ ] Add a failing test showing boundary-adjacent turns are scored even if they are not final turns.
- [ ] Add a failing test showing planner fallback conversations still use the current selective scoring behavior.
- [ ] Add a failing test showing inline scoring / later recompute can recover planner behavior via persisted metadata or deterministic recomputation.
- [ ] Add a failing scoring-estimation test proving planner-aware scoring decisions change estimated call counts in the expected direction.
- [ ] Run the targeted scoring tests and verify they fail before code changes.

Suggested commands:
- `uv run pytest tests/test_scoring.py tests/test_inline_scoring.py tests/test_scoring_estimation.py -q -k 'selective_scoring or inherited_mid_turn or planner or estimate'`

### Task 10: Implement planner-fed selective scoring

**Files:**
- Modify: `src/sft_label/scoring.py`
- Modify: `src/sft_label/inline_scoring.py`

- [ ] Extend selective scoring decisions to prefer planner anchors and boundary turns when planner metadata is present.
- [ ] Keep the current interval-based anchor logic as a fallback.
- [ ] Reuse the current conservative estimate path for non-anchor inherited turns.
- [ ] Make inline scoring / recompute preserve or reconstruct planner metadata needed for planner-fed selective scoring.
- [ ] Ensure compact prompt budget behavior remains unchanged for scoring requests.
- [ ] Re-run targeted scoring tests until they pass.

Suggested commands:
- `uv run pytest tests/test_scoring.py tests/test_inline_scoring.py tests/test_scoring_estimation.py -q -k 'selective_scoring or planner or inherited_mid_turn or estimate'`

### Task 11: Review chunk 3

**Files:**
- No additional production files expected

- [ ] Dispatch reviewers focused on scoring correctness and compact prompt safety.
- [ ] Resolve Important/Critical issues before continuing.

---

## Chunk 4: Compact guards, docs, and rollout support

### Task 12: Add request budget gates and tests

**Files:**
- Modify: `src/sft_label/pipeline.py`
- Modify: `src/sft_label/scoring.py`
- Modify: `tests/test_scoring.py`
- Modify: `tests/test_preprocessing.py`

- [ ] Add soft/hard character-count budget checks for Pass 1 and Pass 2 request assembly.
- [ ] Add UTF-8 byte-count hard-gate checks in addition to coarse character-count checks.
- [ ] Add tests proving planner metadata is not serialized into Pass 1 `<preprocessed_signals>` or Pass 2 prompt bodies by default.
- [ ] Add worst-case assembly tests for Pass1 Call2 and Pass2 scoring under non-ASCII/CJK-heavy payloads.
- [ ] Add tests proving oversized assembled requests trigger truncation / downgrade / fallback instead of silently sending oversized prompts.
- [ ] Keep default behavior non-disruptive for current compact runs.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py tests/test_scoring.py -q -k 'compact or budget or truncation'`

### Task 13: Document workflows and operator guidance

**Files:**
- Modify: `docs/guides/how-sft-label-works.md`
- Modify: `docs/guides/common-workflows.md`

- [ ] Document the planner stage and its role in compact-first long multi-turn labeling.
- [ ] Document fallback behavior, selective scoring expectations, and recommended rollout order.
- [ ] Add operator notes for compact payload budgets and planner stats to watch.

### Task 14: Add rollout-oriented regression coverage

**Files:**
- Modify: `tests/test_e2e_mock.py` (if needed)
- Modify: `tests/test_conversation.py`
- Modify: `tests/test_pipeline_estimation.py`

- [ ] Add at least one representative long multi-turn regression covering planner → Pass 1 → Pass 2 → aggregation.
- [ ] Add assertions for planner fallback and compact-safe execution in a near-production path.
- [ ] Add offline A/B comparison fixtures for legacy sparse vs planner sparse, including anchor-count inflation and fallback-rate checks.
- [ ] Add a small calibration harness or regression fixture set for boundary thresholds, anchor caps, and planner confidence defaults.
- [ ] Keep fixture size small enough for repo test speed.

Suggested commands:
- `uv run pytest tests/test_e2e_mock.py tests/test_conversation.py tests/test_pipeline_estimation.py -q`

### Task 15: Review chunk 4

**Files:**
- No additional production files expected

- [ ] Dispatch reviewers focused on docs clarity, rollout safety, and compact guard correctness.
- [ ] Resolve Important/Critical issues before final verification.

---

## Chunk 5: Full verification, subagent review plan, and handoff

### Task 16: Run the focused verification suite

**Files:**
- No new production files expected

- [ ] Run the preprocessing, inline merge, scoring, conversation, and workload-estimation suites.
- [ ] Record any flaky or high-cost tests and stabilize them before sign-off.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py tests/test_inline_pass1.py tests/test_pipeline_estimation.py -q`
- `uv run pytest tests/test_scoring.py tests/test_conversation.py -q`
- `uv run pytest tests/test_e2e_mock.py -q`

### Task 17: Run repo-wide verification appropriate to the touched surface

**Files:**
- No new production files expected

- [ ] Run the broader regression suite covering affected modules.
- [ ] Run inline scoring and scoring estimation suites explicitly.
- [ ] Re-run compact/long-multiturn targeted tests if any failures are fixed during this phase.

Suggested commands:
- `uv run pytest tests/test_preprocessing.py tests/test_inline_pass1.py tests/test_pipeline_estimation.py tests/test_inline_scoring.py tests/test_scoring.py tests/test_scoring_estimation.py tests/test_conversation.py tests/test_e2e_mock.py -q`

### Task 18: Execute the final subagent review plan

**Files:**
- No additional production files expected

- [ ] Dispatch **Reviewer A (algorithm)** with scope limited to planner segmentation, anchor selection, and inheritance safety.
- [ ] Dispatch **Reviewer B (compact/prompt)** with scope limited to compact payload safety and prompt-budget invariants.
- [ ] Dispatch **Reviewer C (production rollout)** with scope limited to fallback behavior, observability, and rollout safety.
- [ ] Triage feedback into Critical / Important / Minor.
- [ ] Fix Critical and Important issues before declaring the work ready.

### Task 19: Prepare implementation handoff summary

**Files:**
- No new production files expected

- [ ] Summarize final architecture decisions, deferred items, and rollout recommendations in the final change summary.
- [ ] Include exact flags/configs recommended for first canary runs.
- [ ] Confirm the design doc and this plan still match the shipped implementation.

---

## Todo List (Execution Order)

- [ ] Ship planner in metadata-only shadow mode first
- [ ] Ship deterministic planner config + metadata
- [ ] Replace sparse planner with planner-aware contract-compatible planner
- [ ] Restrict inheritance to segment-local neighborhoods
- [ ] Preserve forward-first inheritance inside segments
- [ ] Persist or recompute planner metadata for inline scoring / recompute
- [ ] Feed planner anchors into selective scoring
- [ ] Add byte-based compact request budget gates
- [ ] Add planner observability and fallback counters
- [ ] Add rollout kill-switches and shadow-vs-live comparison stats
- [ ] Add offline calibration / A-B fixture checks
- [ ] Update long-multiturn docs
- [ ] Run focused + broader verification
- [ ] Complete 3-angle subagent review

---

## Test Plan

### Unit tests

- planner boundary detection
- planner local-view-only boundary detection
- planner fallback behavior
- metadata enrichment
- segment-local inheritance
- forward-first inheritance inside segments
- planner-aware anchor selection
- planner-aware selective scoring decisions
- compact byte/char budget gate behavior
- planner metadata exclusion from prompt bodies

### Integration tests

- directory-mode workload estimation with planner
- inline-mode merge / inheritance correctness
- inline scoring / recompute planner compatibility
- Pass 1 planner → sparse labeling path
- Pass 2 planner-fed selective scoring path
- conversation aggregation still behaves with inherited downweighting and additive planner stats

### Regression / end-to-end tests

- long multi-turn compact-mode representative fixture
- fallback-to-legacy sparse path
- selective scoring on planner anchors
- worst-case CJK / non-ASCII compact prompt assembly
- legacy-vs-planner A/B fixture comparison
- final stats and dashboards remain readable

### Review plan

- Reviewer A: algorithm / segmentation / inheritance
- Reviewer B: compact / prompt / payload safety
- Reviewer C: rollout / observability / fallback

---

## Suggested First Canary Configuration

- Phase 0: planner enabled in metadata-only shadow mode, planner selection disabled, selective scoring legacy, extensions disabled
- Phase 1: planner selection enabled for Pass 1 only, selective scoring still legacy
- Phase 2: planner-fed selective scoring enabled
- At every phase: compact prompt mode enabled, planner fallback enabled, explicit kill-switches documented, adaptive runtime and recovery sweep left on

---

Plan complete and saved to `docs/superpowers/plans/2026-03-20-compact-semantic-anchor-planner.md`. Ready to execute?
