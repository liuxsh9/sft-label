# Compact Semantic Anchor Planner Design

## Background

`sft-label` currently handles multi-turn conversations by:

1. normalizing and slicing each assistant reply into a prefix-aligned sample,
2. truncating oversized conversations to fit prompt budgets,
3. applying a front-dense / back-sparse labeling schedule,
4. inheriting labels for unlabeled slices, and
5. aggregating turn-level outputs into conversation-level metrics.

This architecture is robust and production-tested, but it has two important weaknesses for long conversations:

- the sparse planner is still primarily **position-based**,
- late-stage semantic drift is caught only by local pairwise heuristics.

At the same time, production runs are now strongly biased toward `compact` prompt mode, where API payload size is a hard operational constraint. Any new design must preserve the existing compact envelope rather than adding new full-conversation LLM passes.

---

## Design Goal

Improve long multi-turn labeling and scoring quality **without introducing an always-on new LLM stage**, by replacing position-led sparse planning with a **compact-first semantic anchor planner** that:

1. segments multi-turn conversations using deterministic local signals,
2. selects anchor turns near semantic boundaries and high-risk regions,
3. restricts inheritance to segment-local neighborhoods,
4. feeds planner outputs into existing Pass 1 and Pass 2 compact flows,
5. preserves current prompt contracts and payload budgets by default.

---

## Non-Goals

1. Do **not** replace the current Pass 1 two-call taxonomy prompt design.
2. Do **not** add a full-conversation LLM “global scout”, “router”, or “judge” pass in v1.
3. Do **not** redesign conversation aggregation formulas in v1.
4. Do **not** change the meaning of current Pass 1 labels or Pass 2 score fields.
5. Do **not** require historical runs to be migrated before they remain readable.

---

## Hard Constraints

### 1. Compact-first operation

The design must remain safe under the current compact envelopes:

- Pass 1 compact conversation budget: `COMPACT_CONVERSATION_CHARS = 8000`
- Pass 2 compact truncation budget: `COMPACT_VALUE_TRUNCATION_BUDGET = 14000`

Defined in:

- `/Users/lxs/.codex/worktrees/2aed/sft-label/src/sft_label/config.py`

### 2. No new always-on LLM stage

The planner must be deterministic / local by default. Any future optional LLM helper must be:

- opt-in,
- small-input only,
- independently budget-gated,
- disabled in the default rollout.

### 3. Backward-compatible sample contract

The existing pipeline shape should remain intact:

- `normalize_and_slice()` still returns per-slice samples,
- sparse planning still resolves to `label_indices` and `inherit_map`,
- inline mode and directory mode still reuse the same contracts.

### 4. Planner metadata must not enter prompt bodies in v1

In v1, planner outputs may affect:

- sparse selection,
- inheritance neighborhoods,
- selective scoring decisions,
- stats / dashboards / debug artifacts.

Planner outputs must **not** be serialized into:

- Pass 1 `<preprocessed_signals>`,
- Pass 1 `conversation_json`,
- Pass 2 scoring `<meta>`,
- Pass 2 prompt body.

This is a hard invariant for compact safety and prompt backward compatibility.

---

## Proposed Architecture

## High-Level Flow

```text
normalize_and_slice
    ↓
semantic anchor planner (deterministic)
    ↓
pass1 sparse plan (anchor-led, segment-local inheritance)
    ↓
existing pass1 label_one compact prompts
    ↓
pass2 selective scoring (planner-fed anchors)
    ↓
existing conversation aggregation
```

## Stage A: Semantic Anchor Planner

Add a new deterministic planning layer after slicing and before sparse selection.

### Inputs

The planner uses only existing or trivially derivable local signals from:

- `/Users/lxs/.codex/worktrees/2aed/sft-label/src/sft_label/preprocessing.py`
- `/Users/lxs/.codex/worktrees/2aed/sft-label/src/sft_label/scoring.py`

Signal families:

- `turn_index`, `total_turns`, `source_id`, `conversation_uid`
- current request / trajectory / final response segmentation
- tool-turn counts and role patterns
- code fence languages
- code block counts
- file-scope indicators
- request / response / window char-ngram similarities
- keyword group matches

### Outputs

The planner writes metadata only, without changing sample shape:

- `segment_id`
- `segment_turn_start`
- `segment_turn_end`
- `boundary_score`
- `anchor_priority`
- `anchor_reason`
- `anchor_distance`
- `inherit_group_id`
- `planner_policy`
- `planner_confidence`

### Segmentation Policy

Use a weighted boundary score built from deterministic signals.

#### Strong boundary triggers

Any of these can independently justify a boundary:

- code language changes,
- tool trajectory pattern changes,
- role pattern changes across request→trajectory→response,
- file-scope appearance / disappearance,
- code-heavy ↔ text-heavy transition,
- request similarity or response similarity falling below threshold,
- keyword group shift across adjacent slices.

### Local-view comparison invariant

Because current multi-turn samples are **prefix-aligned slices**, planner comparisons must be built from a local view, not the entire prefix payload.

Allowed comparison view:

- current request,
- current trajectory,
- final response,
- local role / tool / code / file-scope summaries derived from that view.

Disallowed comparison view:

- raw whole-prefix text similarity between adjacent slices.

Without this guard, normal prefix growth would be misread as semantic boundary drift.

#### Weak boundary triggers

The planner may accumulate evidence across:

- moderate similarity decline,
- turn-structure drift,
- repeated local changes across two adjacent slices,
- growing anchor distance within a long segment.

### Segment Safety Rules

- segments are contiguous in turn order,
- minimum segment size prevents over-fragmentation,
- planner may merge low-confidence tiny segments back into neighbors,
- if planner confidence is too low, the conversation falls back to the current sparse planner.

### Inheritance direction invariant

Segment-local inheritance must preserve the current semantic rule:

- prefer inheriting from the **next labeled slice** within the same segment,
- only fall back backward when no valid forward anchor exists.

This keeps inheritance aligned with the current prefix-completeness logic.

---

## Stage B: Anchor Selection

Replace the current position-first sparse schedule with a segment-first anchor schedule.

### Required anchors

Every conversation or segment must include:

- first anchor of the segment,
- last anchor of the segment,
- final conversation turn,
- high-boundary-score turns.

### Optional anchors

Add one mid-segment anchor when:

- the segment is long,
- planner confidence is low,
- risk indicators are high,
- tool-use density or file-scope activity is high.

### Inheritance rule change

Inheritance remains supported, but only within the same `inherit_group_id` / segment.

This reduces the current risk of inheriting labels across semantic phase changes.

---

## Stage C: Pass 1 Integration

The Pass 1 taxonomy prompt contract remains unchanged.

### What changes

- only the sparse planning inputs change,
- anchor-selected turns are labeled with existing `label_one()` logic,
- unlabeled turns inherit within segment-local neighborhoods.

### What does not change

- Call 1 / Call 2 prompts,
- compact prompt mode behavior,
- validation / consistency checks,
- arbitration flow,
- inline merge contract.

### Inline persistence requirement

If Pass 2, recompute, or later inline-only workflows need planner-driven anchors or segment data, v1 must support one of these explicitly:

1. persist planner metadata in inline turn records / conversation records, or
2. deterministically recompute planner outputs during inline scoring / recompute.

The design must not rely on in-memory planner state surviving across separate inline pipeline stages.

This keeps v1 focused on “better sample selection”, not “new label semantics”.

---

## Stage D: Pass 2 Integration

Pass 2 should consume planner signals before any larger redesign.

### Recommended v1 behavior

Use planner anchors to drive selective scoring:

- score all anchor turns,
- score final turns,
- score boundary-adjacent turns,
- estimate inherited mid-turns conservatively using the existing selective scoring machinery.

### Why

The current code already supports:

- anchor-based scoring decisions,
- conservative estimates for inherited middle turns.

Planner-fed anchors improve where real scoring effort is spent without changing the prompt contract.

---

## Stage E: Conversation Aggregation

Do not rewrite aggregation in v1.

Current aggregation already supports:

- inherited-turn downweighting,
- coverage confidence,
- rarity shrinkage,
- trajectory-aware features,
- v2 conversation shadow metrics.

### Optional additive fields

Conversation records may add planner-derived diagnostics such as:

- `segment_count`
- `anchor_coverage`
- `boundary_coverage`
- `planner_uncertainty`
- `cross_segment_inheritance_ratio`

These should be additive fields, not replacements.

---

## Compact / Payload Design

## Default rule

The planner is local-only, so v1 introduces **zero new always-on request payload**.

### Main request budgets

Recommended pre-send gates:

- soft cap: `22000 UTF-8 bytes`
- hard cap: `24000 UTF-8 bytes`

Applied to:

- Pass 1 requests,
- Pass 2 requests.

### Budget accounting model

Budget checks must happen **after full message assembly**, not only after conversation truncation.

Track separately:

- character count,
- UTF-8 byte count,
- token count when provider tokenization is available.

The hard gate for request safety should be byte-based; character count remains a coarse pre-check only.

### Request-type budget tables

#### Pass 1 Call 1

- assembled request hard cap: `24000 UTF-8 bytes`
- components to measure independently:
  - fixed prompt body
  - `conversation_json`
  - `<preprocessed_signals>`

#### Pass 1 Call 2

- assembled request hard cap: `24000 UTF-8 bytes`
- components to measure independently:
  - fixed prompt body
  - `conversation_json`
  - `<preprocessed_signals>`
  - `call1_result`

#### Pass 2 scoring

- assembled request hard cap: `24000 UTF-8 bytes`
- components to measure independently:
  - fixed prompt body
  - truncated scoring evidence
  - labels / meta wrapper

### Optional future micro-step budgets

If a future optional extension or micro-judge is introduced:

- soft cap: `10000 UTF-8 bytes`
- hard cap: `12000 UTF-8 bytes`

And it must never receive full `conversation_json`.

### Allowed future micro-step inputs

Only:

- last user turn,
- final assistant response,
- very short trajectory summary,
- minimal planner signals.

---

## Failure / Fallback Behavior

If the planner detects low-confidence segmentation, any of the following should trigger fallback to the current sparse policy:

- too many short segments,
- inconsistent or oscillating boundaries,
- anchor explosion beyond configured cap,
- missing required metadata,
- planner confidence below threshold.

### Operator kill-switches

Production rollout must support explicit operational switches independent of planner confidence:

- `planner_enabled = false` → force legacy sparse planner,
- `planner_metadata_only = true` → compute planner stats but do not alter selection,
- `planner_pass2_enabled = false` → force legacy selective scoring behavior.

This ensures the new system degrades to the old stable behavior rather than creating correlated labeling errors.

---

## Rollout Strategy

### Phase 0: Shadow metadata-only rollout

- default flags remain OFF,
- planner computes metadata and stats only,
- no effect on Pass 1 sparse selection,
- no effect on Pass 2 selective scoring,
- must collect planner fallback rate, segment/anchor distributions, byte-budget distributions, and estimated call deltas.

### Phase 1: Pass 1 planner enablement

- enable planner-driven sparse selection,
- keep Pass 2 on legacy selective scoring,
- validate inheritance quality and anchor distributions before changing scoring.

### Phase 2: Pass 2 planner-fed selective scoring

- enable planner-fed selective scoring only after Pass 1 planner behavior is accepted,
- keep prompt contracts unchanged,
- verify score coverage / call-count changes separately from Pass 1 changes.

### Phase 3: Additive diagnostics and dashboard surfacing

Make planner diagnostics first-class rather than optional:

- segment counts,
- anchor counts,
- planner fallback rate,
- cross-segment inheritance ratio,
- request byte-budget distributions,
- hard-cap hit rate.

---

## Risks

### 1. Correlated error risk

Wrong segmentation can create segment-wide inheritance errors.

Mitigation:

- conservative boundaries,
- fallback to current sparse planner,
- additional anchors in low-confidence segments.

### 2. Objective mismatch risk

Segments represent phases, while current labels still target the final turn of a slice.

Mitigation:

- planner influences only anchor choice,
- planner does not directly assign labels,
- Pass 1 prompt semantics remain unchanged.

### 3. Planner over-fragmentation risk

Too many segments would erase cost savings.

Mitigation:

- minimum segment size,
- per-conversation anchor cap,
- merge-back rules for low-confidence micro-segments.

### 4. Inline persistence risk

Planner-fed Pass 2 behavior may disappear in inline re-score / recompute paths if planner state is not persisted or recomputed.

Mitigation:

- persist planner metadata explicitly, or
- make planner deterministic and re-runnable in inline scoring / recompute code paths,
- test both same-run and later-run inline workflows.

### 5. Dependency layering risk

Planner logic must not create circular imports between preprocessing and scoring.

Mitigation:

- extract reusable local-view signal helpers into a lower-level module if needed,
- keep planner implementation below both Pass 1 and Pass 2 orchestration layers.

---

## Success Criteria

The design is successful when all of the following hold on a representative long-multi-turn evaluation set:

1. Pass 1 labeled slice count decreases versus current compact baseline **or** remains flat with better human-evaluated boundary accuracy.
2. Pass 2 scored slice count decreases when selective scoring is enabled.
3. Compact payload envelope remains within current production caps.
4. Human review finds fewer obvious cross-phase inheritance mistakes.
5. High-value conversation retention does not regress.

---

## Recommended v1 Decision

Implement the planner as a **deterministic metadata + sparse planning upgrade** only.

Do **not** add a new always-on LLM scout stage in v1.

This is the smallest change that:

- materially improves anchor quality,
- preserves compact guarantees,
- reuses the current Pass 1 / Pass 2 prompt contracts,
- and keeps rollback simple.
