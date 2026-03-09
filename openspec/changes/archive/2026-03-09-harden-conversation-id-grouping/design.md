## Context

Pass 2.5 aggregates multi-turn slices into conversation records and outputs `conversation_scores.json`. The current grouping key is `source_id`, which is not globally unique across datasets or files. This can silently merge unrelated trajectories.

## Goals / Non-Goals

**Goals:**
- Guarantee stable, collision-safe conversation grouping for multi-file datasets.
- Preserve compatibility for legacy runs that only have `source_id`.
- Keep filtering semantics consistent with aggregation identity.

**Non-Goals:**
- No changes to value scoring formulas.
- No migration of historical artifacts beyond compatibility fallback.

## Decisions

1. Define canonical key as `source_file::source_id` when both values exist; fallback to `source_id` otherwise.
   - Rationale: minimal schema change while eliminating common collisions.
   - Alternative considered: hash full conversation text. Rejected due to cost and instability after preprocessing.

2. Emit `conversation_id` as canonical key in `conversation_scores.json`.
   - Rationale: downstream consumers can use a single unambiguous key.

3. Update filter conversation lookup and matching logic to build the same canonical key from sample metadata.
   - Rationale: conversation-level criteria must match the same identity used in aggregation.

4. Add compatibility fallback in filter matching: if canonical key not found, try legacy `source_id`.
   - Rationale: supports previously generated `conversation_scores.json` files.

## Risks / Trade-offs

- [Risk] Existing tools that assume `conversation_id == source_id` may break.
  - Mitigation: keep fallback matching and document key format in stats outputs.

- [Risk] Missing or malformed `source_file` metadata reduces uniqueness.
  - Mitigation: fallback remains valid; add monitoring counters for fallback usage.

- [Risk] Mixed legacy and new artifacts in the same run may be confusing.
  - Mitigation: prefer canonical key and only fallback when absent.
