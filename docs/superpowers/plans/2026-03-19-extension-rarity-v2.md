# Extension Rarity V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel, opt-in Extension Rarity V2 pipeline that computes per-spec extension rarity, exposes preview/blended V2 fields, and preserves all existing core rarity/value/selection semantics by default.

**Architecture:** Keep the current core rarity/value/selection pipeline untouched as the canonical legacy path. Add a parallel extension-rarity computation path keyed by `spec_id + spec_version + spec_hash`, persist reproducible rarity baselines inside `extension_stats`, and surface additive V2 sample/conversation fields plus preview-only dashboard/export views. Explicitly isolate `label_extensions` from the current Pass 2 prompt before enabling any deterministic extension-rarity blending.

**Tech Stack:** Python, existing scoring/recompute/dashboard pipeline, pytest, markdown docs.

---

## File map and responsibilities

### Core scoring / rarity
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/scoring.py`
  - Add extension rarity computation helpers
  - Add V2 sample scoring fields
  - Keep legacy rarity/value/selection untouched
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/conversation.py`
  - Add conversation-level extension-V2 rarity / selection paths with non-conflicting field names
  - Preserve legacy conversation fields
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/config.py`
  - Add extension rarity mode/config defaults

### Extension baseline / stats
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/label_extensions_stats.py`
  - Persist reproducible rarity baseline metadata per spec
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/pipeline.py`
  - Ensure pass1 stats include extension rarity baseline structures
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/tools/recompute.py`
  - Recompute V2 rarity/value/selection offline
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/inline_pass1.py`
  - Define extension-rarity V2 invalidation/recompute semantics for inline mode without disturbing legacy Pass 2 fields

### Pass 2 prompt isolation
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/prompts_value.py`
  - Prevent `label_extensions` from entering current Pass 2 prompt payload
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/inline_scoring.py`
  - Preserve extension payloads in artifacts but keep scoring prompt inputs extension-blind

### CLI / launcher / export / dashboard
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/cli.py`
  - Add CLI flags for extension rarity mode
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/launcher.py`
  - Add interactive extension-rarity mode guidance
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/tools/dashboard.js`
  - Display core rarity vs extension rarity vs V2 fields clearly
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/tools/export_review.py`
  - Add explicit opt-in extension-rarity export columns (if this module is the current export-review entry)

### Tests
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_scoring.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_conversation.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_pipeline_label_extensions_integration.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_launcher.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_cli_progress.py`
- Add if needed: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_extension_rarity.py`
- Add if needed: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_refresh_rarity_extension_v2.py`

### Docs
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/README.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/README.zh-CN.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/guides/pass1-extension-labeling.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/guides/adaptive-llm-runtime.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/pass1-extension-labeling-intro.md`

---

## Chunk 1: Lock the compatibility boundary first

### Task 1: Isolate extension labels from the current Pass 2 prompt

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/prompts_value.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/inline_scoring.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_scoring.py`

- [ ] **Step 1: Write a failing test proving current Pass 2 prompt serialization excludes `label_extensions`**
- [ ] **Step 2: Run the targeted scoring test to verify failure**
- [ ] **Step 3: Implement the minimal prompt-filtering change so current Pass 2 remains extension-blind**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

### Task 2: Add config surface for Extension Rarity V2 without changing defaults

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/config.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/cli.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/launcher.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_cli_progress.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_launcher.py`

- [ ] **Step 1: Write failing CLI/launcher tests for `extension_rarity_mode=off|preview|bonus_only` with default `off`, plus an explicit off-mode artifact/schema-stability check**
- [ ] **Step 2: Run the targeted tests and verify failure**
- [ ] **Step 3: Add config defaults and flag plumbing without changing existing run behavior**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

---

## Chunk 2: Build reproducible per-spec extension rarity baselines

### Task 3: Extend `extension_stats` with rarity-ready baseline metadata

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/label_extensions_stats.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/pipeline.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_pipeline_label_extensions_integration.py`

- [ ] **Step 1: Write a failing integration test asserting `extension_stats.specs.<id>.baselines.<spec_hash>` now includes `success/failed/invalid/skipped`, `baseline_total`, `field_value_distributions`, and `field_presence_counts` while preserving existing summary fields**
- [ ] **Step 2: Run the targeted integration test and verify failure**
- [ ] **Step 3: Implement minimal stat aggregation changes while preserving existing dashboard-compatible fields**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

### Task 4: Add standalone extension rarity helpers in scoring

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/scoring.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_extension_rarity.py`

- [ ] **Step 1: Write failing unit tests for per-field extension IDF (with non-negative clamp), confidence shrinkage, multi-enum aggregation, per-spec normalization, and per-spec rarity computation**
- [ ] **Step 2: Run those tests to verify failure**
- [ ] **Step 3: Implement minimal pure helpers for extension rarity, keeping them isolated from legacy rarity code paths**
- [ ] **Step 4: Re-run the new tests and confirm pass**
- [ ] **Step 5: Commit**

---

## Chunk 3: Add additive sample-level V2 score fields

### Task 5: Compute `rarity_core`, `rarity_extension`, and `rarity_v2` without touching legacy fields

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/scoring.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_scoring.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_extension_rarity.py`

- [ ] **Step 1: Write failing tests showing `preview` computes extension rarity fields but leaves legacy `rarity.score` unchanged**
- [ ] **Step 2: Add failing tests showing `bonus_only` computes bounded `rarity_v2` bonus with no negative penalty**
- [ ] **Step 3: Run the targeted tests and verify failure**
- [ ] **Step 4: Implement V2 sample rarity fields with a single aggregate gate, capped bonus-only blending, and explicit local-baseline downgrade behavior**
- [ ] **Step 5: Re-run targeted tests and confirm pass**
- [ ] **Step 6: Commit**

### Task 6: Compute `value_score_v2` and `selection_score_v2`

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/scoring.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_scoring.py`

- [ ] **Step 1: Write failing tests showing legacy `value_score` / `selection_score` remain unchanged while V2 fields appear only in `bonus_only` mode**
- [ ] **Step 2: Run the targeted scoring tests and verify failure**
- [ ] **Step 3: Implement `value_score_v2` and `selection_score_v2` using legacy intra-class grouping and `rarity_v2` only in the final composition**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

---

## Chunk 4: Add conversation and recompute support

### Task 7: Add `conv_rarity_extension_v2` / `conv_selection_extension_v2`

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/conversation.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_conversation.py`

- [ ] **Step 1: Write failing tests for conversation extension-V2 rarity/selection fields with non-conflicting names that leave legacy conversation fields unchanged**
- [ ] **Step 2: Run the targeted conversation tests and verify failure**
- [ ] **Step 3: Implement additive conversation extension-V2 fields sourced from sample-level `rarity_v2`**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

### Task 8: Extend offline refresh/recompute for V2 fields

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/tools/recompute.py`
- Test: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_refresh_rarity_extension_v2.py`

- [ ] **Step 1: Write failing recompute tests for `preview` and `bonus_only` modes using persisted extension baseline metadata, including the rule that `bonus_only` cannot apply a non-zero bonus from local-only baselines**
- [ ] **Step 2: Run the targeted recompute tests and verify failure**
- [ ] **Step 3: Implement V2 refresh logic without overwriting legacy fields**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

---

### Task 8b: Define inline V2 invalidation and recompute semantics

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/inline_pass1.py`
- Modify any inline scoring/rebuild helper touched by V2 field persistence
- Test: inline pass1 / inline scoring regression suite

- [ ] **Step 1: Write a failing inline regression test showing extension-only/spec-hash changes invalidate or recompute V2-only fields while legacy Pass 2 fields remain untouched**
- [ ] **Step 2: Run the targeted inline regression and verify failure**
- [ ] **Step 3: Implement minimal inline invalidation/recompute semantics for extension-rarity V2 only**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

---

## Chunk 5: Surface V2 safely in dashboard / export / docs

### Task 9: Add dashboard views for core rarity vs extension rarity vs V2

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/tools/dashboard.js`
- Modify any dashboard payload builder touched by rarity display
- Test existing dashboard-related suites if present

- [ ] **Step 1: Add a failing fixture/test or at minimum a deterministic payload expectation for distinct labels: `Core rarity`, `Extension rarity (preview)`, `Rarity V2`, plus explicit gated-out / diagnostic-only states**
- [ ] **Step 2: Implement minimal UI changes with provenance display and low-support warnings**
- [ ] **Step 3: Run dashboard tests (or payload checks) and confirm pass**
- [ ] **Step 4: Commit**

### Task 10: Add explicit export-review support for extension rarity columns

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/tools/export_review.py`
- Test the corresponding export-review test module

- [ ] **Step 1: Write failing tests for explicit opt-in extension-rarity export columns using preview-safe names such as `extension_rarity_preview_score`**
- [ ] **Step 2: Run the targeted export tests and verify failure**
- [ ] **Step 3: Implement additive export columns without changing existing `--include-extensions` defaults**
- [ ] **Step 4: Re-run targeted tests and confirm pass**
- [ ] **Step 5: Commit**

### Task 11: Update user-facing docs and guidance

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/README.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/README.zh-CN.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/guides/pass1-extension-labeling.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/guides/adaptive-llm-runtime.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/pass1-extension-labeling-intro.md`

- [ ] **Step 1: Add docs explaining that legacy rarity remains canonical, extension rarity is preview/opt-in, and adaptive runtime changes pacing only**
- [ ] **Step 2: Add docs for `off|preview|bonus_only`, mark `bonus_only` as experimental, and add the non-comparability warning for tiny/local/mixed-hash baselines**
- [ ] **Step 3: Self-review wording so users cannot mistake preview extension rarity for legacy ranking**
- [ ] **Step 4: Commit**

---

## Chunk 6: Final verification

### Task 12: Full regression sweep

**Files:**
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_scoring.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_conversation.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_pipeline_label_extensions_integration.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_launcher.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_cli_progress.py`
- Verify: repo-wide `pytest`

- [ ] **Step 1: Run the focused extension-rarity, scoring, conversation, launcher, export, and inline invalidation suites**
- [ ] **Step 2: Fix any fallout conservatively**
- [ ] **Step 3: Run full `uv run pytest -q`**
- [ ] **Step 4: Summarize exactly what changed, what stayed backward-compatible, which fields are V2-only, and what conditions downgrade bonus mode to diagnostic preview behavior**

