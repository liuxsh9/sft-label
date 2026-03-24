# Reliability Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the confirmed P0/P1 reliability blockers in Pass 1, Pass 2, postprocess, and dashboard publish so large runs fail closed, resume correctly, and avoid the worst current OOM/half-publish paths without changing successful output semantics.

**Architecture:** Deliver the work in three batches. Batch 1 fixes correctness and safety guardrails around resume/finalize/publish. Batch 2 hardens large-run execution shape in Pass 1 and deferred postprocess, then tightens Pass 2 large-run memory/rewrite hotspots after Batch 1 stabilizes `scoring.py`. Batch 3 is an integrated review/test sweep. Keep business logic intact; only reliability, atomicity, gating, and execution-shape internals may change.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, existing file-based run layout, current CLI/launcher/dashboard-service modules.

---

## Scope

This plan covers the confirmed reliability issues from the approved spec at:
`/Users/lxs/.codex/worktrees/0e2a/sft-label/docs/superpowers/specs/2026-03-24-reliability-hardening-design.md`

Included:

1. Pass 2 resume/checkpoint/finalize correctness.
2. Launcher smart-resume correctness and auto-publish guardrails.
3. Atomic publish and publish-layer fail-closed checks.
4. Pass 1 chunked durability / stats growth / FD budgeting hardening.
5. Pass 2 large-run memory and global rewrite hotspot reduction.
6. Deferred postprocess / dashboard generation hardening.
7. Regression and fault-window tests.

Excluded:

- label taxonomy or prompt changes,
- score/rarity/selection semantic changes,
- dashboard UX redesign,
- broad refactors unrelated to the above.

---

## File Ownership Map

### Batch 1 (parallel-safe)

**Worker A — Pass 2 resume/finalize correctness**
- Modify: `src/sft_label/scoring.py`
- Create: `tests/test_scoring_resume_reliability.py`
- Optional light touch: `tests/test_scoring.py` only if a tiny hook is unavoidable; prefer the new test file.

**Worker B — Launcher/CLI safety guards**
- Modify: `src/sft_label/launcher.py`
- Modify: `src/sft_label/cli.py`
- Modify: `tests/test_launcher.py`
- Modify: `tests/test_cli_progress.py`

**Worker C — Atomic publish and publish locking**
- Modify: `src/sft_label/dashboard_service.py`
- Modify: `tests/test_dashboard_service.py`

### Batch 2 (parallel-safe except where noted)

**Worker D — Pass 1 reliability hardening**
- Modify: `src/sft_label/pipeline.py`
- Modify: `src/sft_label/http_limits.py`
- Modify: `src/sft_label/config.py` (only if a safe knob is needed)
- Create: `tests/test_pass1_chunked_reliability.py`
- Modify: `tests/test_inline_pass1.py`
- Modify: `tests/test_pipeline_estimation.py`
- Modify: `tests/test_http_limits.py`

**Worker E — Deferred postprocess / dashboard generation hardening**
- Modify: `src/sft_label/tools/recompute.py`
- Modify: `src/sft_label/conversation.py`
- Modify: `src/sft_label/tools/visualize_labels.py`
- Modify: `src/sft_label/tools/visualize_value.py`
- Modify: `src/sft_label/tools/dashboard_scopes.py`
- Modify: `src/sft_label/tools/dashboard_explorer.py`
- Modify: `tests/test_recompute.py`
- Modify: `tests/test_conversation.py`
- Modify: `tests/test_visualize_labels.py`
- Modify: `tests/test_visualize_value.py`
- Modify: `tests/test_dashboard_explorer.py`
- Modify: `tests/test_dashboard_scopes.py`

**Worker F — Pass 2 large-run memory/rewrite hotspot reduction**
- Starts only after Worker A lands.
- Prefer new helper files to minimize further `scoring.py` churn:
  - Create: `src/sft_label/scoring_large_run.py` (or similarly scoped helper)
- Modify: `src/sft_label/scoring.py` (narrow integration seam only)
- Create: `tests/test_scoring_large_run.py`
- Modify: `tests/test_scoring_estimation.py`
- Modify: `tests/test_scoring.py` only if strictly needed.

### Batch 3 (review/test sweep)

**Worker G — Targeted end-to-end reliability coverage**
- Create: `tests/test_reliability_e2e.py`
- May request tiny existing test seams from the owning Batch 1/2 worker if required; otherwise stays test-only.

**Reviewer/Test workers**
- Read-only review of all modified files.
- Verification commands only; no new feature work outside Worker G's test slice.

---

## Parallel Execution Map

### Batch 1: run in parallel
- Worker A
- Worker B
- Worker C

### Batch 2A: run in parallel after Batch 1 merges cleanly
- Worker D
- Worker E

### Batch 2B: run after Worker A is merged
- Worker F

### Batch 3: run in parallel after code is complete
- Worker G: targeted resume/publish e2e tests
- Reviewer 1: resume/finalize/checkpoint
- Reviewer 2: launcher/publish safety
- Reviewer 3: large-run execution shape / OOM regressions
- Reviewer 4: normal successful-path regression risk
- Test worker: final targeted suites + combined regression sweep

---

## Invariants to Preserve

1. A clean successful run must produce the same labels, scores, and dashboard semantics as before.
2. `selection_score`, `intra_class_rank`, and existing scored fields must still be materialized by Pass 2 completion for completed runs.
3. Pass 1 inline mirrored dataset-visible files must keep existing visible paths/semantics.
4. Labeling-only publish remains legal when scoring state is not applicable.
5. Publish must fail closed rather than publish incomplete or stale state.

---

## Chunk 1: Batch 1 correctness and safety guardrails

### Task 1: Fix Pass 2 resume/finalize/recovery correctness

**Files:**
- Modify: `src/sft_label/scoring.py`
- Create: `tests/test_scoring_resume_reliability.py`

- [ ] **Step 1: Add failing tests for checkpoint-only finalize**
  Add a test that creates only `.scored.jsonl.checkpoint` / `.monitor_value.jsonl.checkpoint` style resume artifacts, runs the chunked fast-path finalize, and asserts final `scored.jsonl` + `monitor_value.jsonl` exist afterward.

- [ ] **Step 2: Add failing tests for recovery/final failed-file consistency**
  Add a test where recovery sweep repairs a previously failed sample and assert the final `failed_value.jsonl` / `score_failures.jsonl` no longer contain the recovered sample.

- [ ] **Step 3: Add a stale-artifact precedence test**
  Cover the order between working, checkpoint, and final artifacts so finalize always leaves one authoritative final set.

- [ ] **Step 4: Run the new tests and confirm they fail before implementation**
  Run: `uv run pytest tests/test_scoring_resume_reliability.py -q`
  Expected: failing assertions around missing final files or stale failed artifacts.

- [ ] **Step 5: Implement the minimal finalize-order fix in `scoring.py`**
  Repair the checkpoint-only fast path and recovery/finalize ordering without changing scoring semantics.

- [ ] **Step 6: Re-run targeted tests until green**
  Run: `uv run pytest tests/test_scoring_resume_reliability.py -q`
  Expected: PASS.

- [ ] **Step 7: Run adjacent regression coverage**
  Run: `uv run pytest tests/test_scoring.py -q -k 'resume or recovery_sweep or deferred or selection'`
  Expected: PASS.

- [ ] **Step 8: Commit**
  Run:
  ```bash
  git add src/sft_label/scoring.py tests/test_scoring_resume_reliability.py
  git commit -m "fix: harden pass2 resume finalize and recovery outputs"
  ```

### Task 2: Fix launcher smart-resume and auto-publish safety guards

**Files:**
- Modify: `src/sft_label/launcher.py`
- Modify: `src/sft_label/cli.py`
- Modify: `tests/test_launcher.py`
- Modify: `tests/test_cli_progress.py`

- [ ] **Step 1: Add failing smart-resume detection tests**
  Cover hidden `.next` / `.checkpoint` Pass 2 artifacts and assert launcher routes to resume scoring instead of fresh scoring.

- [ ] **Step 2: Add failing tests for `run --resume ... --score` behavior**
  Assert the launcher/CLI path does not silently trigger a fresh Pass 2 when safe resume is expected.

- [ ] **Step 3: Add failing auto-publish guard tests**
  Add tests for deferred/pending/failed Pass 2 postprocess states causing auto-publish to skip/fail closed with a clear message.

- [ ] **Step 4: Run the targeted tests and confirm they fail first**
  Run:
  - `uv run pytest tests/test_launcher.py -q -k 'smart_resume or resume'`
  - `uv run pytest tests/test_cli_progress.py -q -k 'publish or dashboard_service or register_run'`

- [ ] **Step 5: Implement launcher/CLI guardrails**
  Update smart resume detection, resume routing, and auto-publish gating without changing healthy-path UX.

- [ ] **Step 6: Re-run targeted tests until green**
  Run the same commands as Step 4.

- [ ] **Step 7: Commit**
  Run:
  ```bash
  git add src/sft_label/launcher.py src/sft_label/cli.py tests/test_launcher.py tests/test_cli_progress.py
  git commit -m "fix: harden launcher resume and auto-publish safety"
  ```

### Task 3: Make publish atomic, locked, and publish-layer fail-closed

**Files:**
- Modify: `src/sft_label/dashboard_service.py`
- Modify: `tests/test_dashboard_service.py`

- [ ] **Step 1: Add failing tests for temp-dir publish swap**
  Add tests that simulate copy failure and assert the previous published directory remains intact.

- [ ] **Step 2: Add failing tests for publish eligibility checks**
  Assert `publish_run_dashboards()` rejects deferred/failed scoring publish, while still allowing labeling-only publish when Pass 2 is not applicable.

- [ ] **Step 3: Add failing tests for single-writer publish locking**
  Add a test seam for lock acquisition and assert concurrent/conflicting publish fails closed instead of mutating shared state.

- [ ] **Step 4: Run the targeted tests and confirm they fail first**
  Run: `uv run pytest tests/test_dashboard_service.py -q`

- [ ] **Step 5: Implement temp publish dir + atomic swap + atomic registry save + publish lock**
  Keep existing URL and run-id semantics intact.

- [ ] **Step 6: Re-run targeted tests until green**
  Run: `uv run pytest tests/test_dashboard_service.py -q`

- [ ] **Step 7: Commit**
  Run:
  ```bash
  git add src/sft_label/dashboard_service.py tests/test_dashboard_service.py
  git commit -m "fix: make dashboard publish atomic and fail closed"
  ```

### Task 4: Review Batch 1 before moving on

**Files:**
- No new code expected

- [ ] Request one reviewer subagent for Worker A diff.
- [ ] Request one reviewer subagent for Worker B diff.
- [ ] Request one reviewer subagent for Worker C diff.
- [ ] Fix any critical/important issues before Batch 2 starts.
- [ ] Run combined Batch 1 verification:
  `uv run pytest tests/test_scoring_resume_reliability.py tests/test_launcher.py tests/test_cli_progress.py tests/test_dashboard_service.py -q`
- [ ] Commit follow-up fixes if needed.

---

## Chunk 2: Batch 2 large-run hardening

### Task 5: Harden Pass 1 chunked durability, online stats, and FD budgeting

**Files:**
- Modify: `src/sft_label/pipeline.py`
- Modify: `src/sft_label/http_limits.py`
- Modify: `src/sft_label/config.py` (only if needed)
- Create: `tests/test_pass1_chunked_reliability.py`
- Modify: `tests/test_inline_pass1.py`
- Modify: `tests/test_pipeline_estimation.py`
- Modify: `tests/test_http_limits.py`

- [ ] **Step 1: Add failing tests for chunked Pass 1 intermediate artifact semantics**
  Cover interruption-safe sidecar behavior for non-inline chunked outputs.

- [ ] **Step 2: Add failing tests for inline-mode semantics preservation**
  Assert mirrored dataset-visible files remain on existing paths and are not redefined as hidden working artifacts.

- [ ] **Step 3: Add failing tests for stats accumulator online summaries**
  Protect against reintroducing grow-only list behavior where a running numeric summary is sufficient.

- [ ] **Step 4: Add failing tests or assertions for FD budgeting assumptions**
  At minimum, cover the cap computation path so chunked multi-file output handles are reflected in budgeting logic.

- [ ] **Step 5: Run the targeted tests and confirm failure**
  Run:
  - `uv run pytest tests/test_pass1_chunked_reliability.py -q`
  - `uv run pytest tests/test_inline_pass1.py -q -k 'chunk or inline or durability'`
  - `uv run pytest tests/test_pipeline_estimation.py tests/test_http_limits.py -q`

- [ ] **Step 6: Implement minimal non-inline working/final separation and online stats summaries**
  Do not change inline mirrored visible file semantics.

- [ ] **Step 7: Re-run targeted tests until green**
  Run the same commands as Step 5.

- [ ] **Step 8: Commit**
  Run:
  ```bash
  git add src/sft_label/pipeline.py src/sft_label/http_limits.py src/sft_label/config.py tests/test_pass1_chunked_reliability.py tests/test_inline_pass1.py tests/test_pipeline_estimation.py tests/test_http_limits.py
  git commit -m "fix: harden pass1 chunked durability and stats memory"
  ```

### Task 6: Harden deferred postprocess and dashboard generation for large runs

**Files:**
- Modify: `src/sft_label/tools/recompute.py`
- Modify: `src/sft_label/conversation.py`
- Modify: `src/sft_label/tools/visualize_labels.py`
- Modify: `src/sft_label/tools/visualize_value.py`
- Modify: `src/sft_label/tools/dashboard_scopes.py`
- Modify: `src/sft_label/tools/dashboard_explorer.py`
- Modify: `tests/test_recompute.py`
- Modify: `tests/test_conversation.py`
- Modify: `tests/test_visualize_labels.py`
- Modify: `tests/test_visualize_value.py`
- Modify: `tests/test_dashboard_explorer.py`
- Modify: `tests/test_dashboard_scopes.py`

- [ ] **Step 1: Add failing tests for `complete-postprocess` batch retention**
  Assert the implementation no longer needs to retain all per-file conversation batches before global merge.

- [ ] **Step 2: Add failing tests for atomic postprocess/status writes**
  Cover rewritten stats/postprocess/conversation output paths.

- [ ] **Step 3: Add failing tests for heavy-run explorer/dashboard gating**
  Assert heavy-run mode defers/disables explorer generation instead of building an unbounded preview bundle.

- [ ] **Step 4: Run targeted tests and confirm failure**
  Run:
  - `uv run pytest tests/test_recompute.py -q -k 'complete_postprocess or regenerate_dashboard'`
  - `uv run pytest tests/test_conversation.py tests/test_visualize_labels.py tests/test_visualize_value.py tests/test_dashboard_explorer.py tests/test_dashboard_scopes.py -q`

- [ ] **Step 5: Implement incremental merge / atomic writes / heavy-run explorer gating**
  Preserve dashboard semantics; do not introduce sampled semantics changes.

- [ ] **Step 6: Re-run targeted tests until green**
  Run the same commands as Step 4.

- [ ] **Step 7: Commit**
  Run:
  ```bash
  git add src/sft_label/tools/recompute.py src/sft_label/conversation.py src/sft_label/tools/visualize_labels.py src/sft_label/tools/visualize_value.py src/sft_label/tools/dashboard_scopes.py src/sft_label/tools/dashboard_explorer.py tests/test_recompute.py tests/test_conversation.py tests/test_visualize_labels.py tests/test_visualize_value.py tests/test_dashboard_explorer.py tests/test_dashboard_scopes.py
  git commit -m "fix: harden deferred postprocess and dashboard generation"
  ```

### Task 7: Reduce Pass 2 large-run memory and global rewrite hotspots

**Files:**
- Create: `src/sft_label/scoring_large_run.py`
- Modify: `src/sft_label/scoring.py`
- Create: `tests/test_scoring_large_run.py`
- Modify: `tests/test_scoring_estimation.py`
- Modify: `tests/test_scoring.py` only if strictly needed

- [ ] **Step 1: Add failing tests for the current heavy tail path**
  Cover directory-global selection rewrite and heavy-run deferred behavior so future changes prove the heavy work is gated or streamed.

- [ ] **Step 2: Add failing structural tests protecting field materialization semantics**
  Assert `selection_score` / `intra_class_rank` still exist by the end of Pass 2 for completed runs.

- [ ] **Step 3: Run the new tests and confirm failure**
  Run:
  - `uv run pytest tests/test_scoring_large_run.py -q`
  - `uv run pytest tests/test_scoring_estimation.py -q`
  - `uv run pytest tests/test_scoring.py -q -k 'selection or deferred or directory'`

- [ ] **Step 4: Move large-run execution logic into a helper module**
  Keep `scoring.py` integration narrow and avoid broad semantic churn.

- [ ] **Step 5: Gate or stream the heavy global rewrite path**
  Defer/gate only optional heavy work; do not defer core scored field materialization.

- [ ] **Step 6: Re-run targeted tests until green**
  Run the same commands as Step 3.

- [ ] **Step 7: Commit**
  Run:
  ```bash
  git add src/sft_label/scoring_large_run.py src/sft_label/scoring.py tests/test_scoring_large_run.py tests/test_scoring_estimation.py tests/test_scoring.py
  git commit -m "fix: reduce pass2 large-run memory and rewrite hotspots"
  ```

### Task 8: Review Batch 2 before final sweep

**Files:**
- No new code expected

- [ ] Request one reviewer subagent for Worker D diff.
- [ ] Request one reviewer subagent for Worker E diff.
- [ ] Request one reviewer subagent for Worker F diff.
- [ ] Fix any critical/important issues.
- [ ] Run combined Batch 2 verification for Pass 1/postprocess paths:
  `uv run pytest tests/test_pass1_chunked_reliability.py tests/test_inline_pass1.py tests/test_pipeline_estimation.py tests/test_http_limits.py tests/test_recompute.py tests/test_conversation.py tests/test_visualize_labels.py tests/test_visualize_value.py tests/test_dashboard_explorer.py tests/test_dashboard_scopes.py -q`
- [ ] Run combined Batch 2 verification for Pass 2 large-run regressions:
  `uv run pytest tests/test_scoring_large_run.py tests/test_scoring_estimation.py tests/test_scoring.py -q -k 'selection or deferred or directory or estimate'`
- [ ] Commit follow-up fixes if needed.

---

## Chunk 3: Batch 3 integrated review and verification

### Task 9: Add targeted end-to-end resume/publish reliability tests

**Files:**
- Create: `tests/test_reliability_e2e.py`

- [ ] **Step 1: Write end-to-end tests for interrupted resume and publish gating**
  Cover at least:
  - smart resume seeing hidden Pass 2 artifacts,
  - deferred scoring run refusing publish,
  - labeling-only run still allowed to publish,
  - republish failure leaving old public state intact via test seams.

- [ ] **Step 2: Run the new e2e tests and confirm failure first**
  Run: `uv run pytest tests/test_reliability_e2e.py -q`

- [ ] **Step 3: Implement only the missing seams required by prior batches**
  Prefer test-only changes in `tests/test_reliability_e2e.py`. If Step 2 exposes a missing seam, route the fix back to the owning Batch 1/2 task; do not introduce new broad production changes in Batch 3.

- [ ] **Step 4: Re-run the e2e tests until green**
  Run: `uv run pytest tests/test_reliability_e2e.py -q`

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add tests/test_reliability_e2e.py
  git commit -m "test: add reliability end-to-end guards"
  ```

### Task 10: Run the final regression sweep

**Files:**
- No new production files expected

- [ ] Run focused resume/publish/postprocess suites:
  ```bash
  uv run pytest \
    tests/test_scoring_resume_reliability.py \
    tests/test_launcher.py \
    tests/test_cli_progress.py \
    tests/test_dashboard_service.py \
    tests/test_pass1_chunked_reliability.py \
    tests/test_scoring_large_run.py \
    tests/test_scoring_estimation.py \
    tests/test_recompute.py \
    tests/test_visualize_value.py \
    tests/test_dashboard_explorer.py \
    tests/test_dashboard_scopes.py tests/test_reliability_e2e.py -q
  ```

- [ ] Run existing adjacent regression suites:
  ```bash
  uv run pytest \
    tests/test_scoring.py \
    tests/test_inline_pass1.py \
    tests/test_pipeline_estimation.py \
    tests/test_pass1_adaptive_runtime.py \
    tests/test_cli_adaptive_runtime.py \
    tests/test_http_limits.py \
    tests/test_conversation.py \
    tests/test_visualize_labels.py -q
  ```

- [ ] If runtime is acceptable, run one repo smoke check:
  `uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10`

### Task 11: Multi-reviewer final review

**Files:**
- Entire final diff

- [ ] Reviewer 1: inspect resume/checkpoint/finalize correctness only.
- [ ] Reviewer 2: inspect launcher/publish fail-closed behavior only.
- [ ] Reviewer 3: inspect large-run memory/deferred/postprocess behavior only.
- [ ] Reviewer 4: inspect regression risk to normal successful path only.
- [ ] Fix any remaining critical or important findings.

### Task 12: Final delivery summary

**Files:**
- No new code expected unless review finds issues.

- [ ] Produce a short operator note describing:
  - how smart resume now behaves,
  - when auto-publish is intentionally blocked,
  - when `complete-postprocess` is required,
  - which large-run paths were hardened without changing semantics.
- [ ] Produce a short engineering note listing any intentionally deferred follow-ups.
- [ ] Commit final documentation/test-only follow-up if needed.

---

## Reviewer / Test Subagent Prompts

Use these after each batch lands.

### Reviewer prompt: resume/finalize
"Review only resume/checkpoint/finalize correctness. Ignore style and unrelated code. Confirm whether final visible artifacts can still become missing, stale, or contradictory after interruption/recovery."

### Reviewer prompt: launcher/publish safety
"Review only launcher/publish fail-closed behavior. Confirm whether any path can still publish or route into fresh work when state is incomplete, deferred, stale, or inconsistent."

### Reviewer prompt: large-run execution shape
"Review only large-run memory/I/O behavior. Confirm whether hidden O(N) retained structures or full-file rewrites still remain in the repaired paths."

### Test worker prompt
"Run only the prescribed verification commands. Report failures with exact test names and the smallest relevant traceback excerpt. Do not modify code."

---

## Exit Criteria

This plan is complete only when all of the following are true:

1. Hidden Pass 2 working artifacts are recognized by smart resume.
2. Pass 2 fast-path finalize cannot drop final scored outputs.
3. Recovery sweep cannot leave stale failed-artifact views.
4. Auto-publish and manual publish both fail closed for incomplete scoring runs.
5. Publish uses atomic swap semantics plus single-writer protection.
6. Pass 1 chunked reliability is improved without changing mirrored inline semantics.
7. The worst confirmed Batch 2 large-run memory/rewrite hazards are gated, streamed, or made incremental.
8. All new reliability tests and targeted regressions pass.
