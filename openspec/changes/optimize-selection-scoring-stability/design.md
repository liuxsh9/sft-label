## Context

Pass 2 currently produces useful value/selection ordering on real Nemotron SWE data, but manual review shows three systematic biases:
- opener and exploration slices keep too much value despite containing mostly tool actions;
- final-summary / finish-style responses dominate the highest selection buckets;
- selection is overwhelmingly driven by intra-class rank, while diversity/rarity has weak practical effect.

Operational constraints are strict:
- Prompt payload size MUST NOT increase, because larger requests risk production firewall rejection.
- Existing output schema should remain stable to avoid downstream dashboard/filter breakage.
- Improvements must be regression-testable on real audited slices, not prompt-only intuition.

## Goals / Non-Goals

**Goals:**
- Reduce value/selection for tool-call-only or low-information trajectory slices using local post-processing features.
- Prevent summary polish alone from pushing a sample into the top selection band unless the sample includes concrete technical evidence.
- Make selection more diversity-aware without replacing current score semantics or output fields.
- Improve repo-repair domain coverage using preprocessing/post-LLM heuristics rather than longer prompts.
- Add a stable regression pack from audited Nemotron SWE samples to protect ordering behavior.

**Non-Goals:**
- Rewriting Pass 2 from scratch.
- Adding more few-shot examples or increasing prompt length.
- Changing dashboard schema or filter vocabulary in this change.
- Requiring model-side chain-of-thought or multi-call scoring.

## Decisions

### 1. Add stage-aware local features outside the prompt

**Decision:** Infer a lightweight `trajectory_stage` from existing sample-local signals such as turn position, whether the assistant output is predominantly tool calls, whether the output contains verification evidence, and whether the output is a final-summary style message.

Use this stage only in post-LLM aggregation/selection heuristics, not as a new long prompt section.

**Rationale:** The audited failures are mostly structural and can be identified from existing content without spending prompt budget.

**Alternative considered:** Ask the scoring LLM to classify the stage explicitly. Rejected because it increases prompt size and model variance.

### 2. Penalize tool-action slices through evidence-aware post-processing

**Decision:** Add deterministic penalties when the final assistant message is predominantly tool calls or file-navigation actions and lacks technical conclusion/evidence. Keep penalties configurable and bounded so true implementation/verification steps are not over-suppressed.

**Rationale:** Manual review shows many low-selection-worthy slices are mechanically identifiable even when Pass 2 gives them mid-range value.

**Alternative considered:** Add harsher prompt wording about tool-only slices. Rejected as primary approach because prompt edits alone are less stable and may still increase payload if done carelessly.

### 3. Add summary-evidence guard instead of banning final summaries

**Decision:** Keep high scores for final summaries only when they include concrete evidence signals (file/function references, root cause, test/verification commands, before/after behavior, or explicit fix description). Summary-heavy slices without evidence receive a selection haircut, not an automatic failure.

**Rationale:** Many best SWE samples are indeed final summaries; the problem is unsupported polish, not summaries themselves.

**Alternative considered:** Penalize all finish/final-review messages. Rejected because it would suppress genuinely valuable end-of-trajectory training samples.

### 4. Rebalance selection by tuning post-rank composition, not prompt output fields

**Decision:** Adjust selection aggregation to blend current value, intra-class rank, rarity/diversity bonus, and stage/evidence modifiers. Keep the existing output fields (`value_score`, `selection_score`, `intra_class_rank`, `rarity`) so downstream consumers remain compatible.

**Rationale:** Current correlation shows intra-class rank nearly fully explains selection delta. Rebalancing should happen in deterministic code where behavior is auditable.

**Alternative considered:** Add new LLM-scored dimensions for diversity or representativeness. Rejected due to extra calls/size and weaker reproducibility.

### 5. Keep prompt changes budget-neutral and replace text instead of appending

**Decision:** Any prompt edits must be token-budget-neutral or smaller. New guardrails must replace existing wording, compress examples, or tighten phrasing in place. No new few-shot examples, no new long meta blocks, and no larger user payloads.

**Rationale:** Production safety explicitly forbids payload growth.

**Alternative considered:** Adding one more few-shot example for trajectory stages. Rejected due to payload risk.

### 6. Use audited regression fixtures as the acceptance gate

**Decision:** Add a curated regression set from the audited Nemotron SWE run covering:
- opener/incomplete lowest-value cases,
- mid-trajectory tool-action cases,
- high-value common bug-fix finals,
- high-selection summary-heavy finals,
- high-selection rare-domain finals.

Assertions should check relative ordering and bucket movement, not exact floats.

**Rationale:** The problem is ranking behavior, so tests must target ordering stability on real trajectories.

## Risks / Trade-offs

- `[Too much penalty on tool-heavy slices]` → Mitigation: use bounded penalties plus whitelisting for implementation/verification evidence.
- `[Summary-evidence heuristics miss valid high-quality summaries]` → Mitigation: require only one of several evidence patterns and validate on audited fixtures.
- `[Selection rebalance changes downstream threshold hit rates]` → Mitigation: compare old/new threshold distributions in regression stats before rollout.
- `[Domain backfill adds noisy labels]` → Mitigation: keep heuristic additions conservative and only fill obviously inferable repo-maintenance domains.
- `[Prompt compression harms model calibration]` → Mitigation: require prompt-length non-increase and run score-regression smoke tests before shipping.

## Migration Plan

1. Add stage/evidence feature extraction and bounded scoring modifiers behind config flags with safe defaults.
2. Add regression fixture pack plus offline comparison tooling for old/new ranking distributions.
3. Implement budget-neutral prompt tightening only after deterministic heuristics land and are measured.
4. Run sampled e2e scoring on the Nemotron SWE fixture and compare:
   - opener/tool-only slices move down,
   - rare/high-quality finals stay high,
   - summary-only finals do not gain disproportionate headroom.
5. Enable by default once ordering/regression checks pass.

Rollback strategy:
- disable new penalties/bonuses via config;
- keep old selection aggregation path available for one release window;
- no data migration needed because output schema is unchanged.

## Open Questions

- Whether domain backfill should create a new repo-maintenance-like taxonomy tag now or stay heuristic-only within current taxonomy.
- Whether summary-evidence detection should rely purely on regex/structure or include a tiny classifier built from local features.
- Whether conversation-level signals should later feed back into sample-level selection after this first stabilization pass.
