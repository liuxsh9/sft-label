## Context

`matches_filter` and turn-level pruning currently fail open when key fields are missing, which is safe for exploratory filtering but unsafe for quality-focused curation. The project goal here is high-precision sample selection.

## Goals / Non-Goals

**Goals:**
- Make hard gate criteria deterministic when fields are missing.
- Default to strict behavior for quality-focused filtering.
- Keep a documented opt-out path for legacy permissive workflows.

**Non-Goals:**
- No changes to value scoring generation itself.
- No new ranking metrics.

## Decisions

1. Add `missing_gate_policy` to `FilterConfig` with values `fail` and `ignore`; default `fail`.
   - Rationale: explicit and auditable policy for missing fields.

2. Apply policy consistently to:
   - `correctness_min` against `value.quality.correctness`
   - `thinking_mode` against `value.thinking_mode` or `metadata.thinking_mode`
   - turn-level gates against per-slice `value_score` and `quality.overall`

3. Add CLI flag to control policy (`--missing-gate-policy`).
   - Rationale: parity between library and CLI workflows.

4. Extend summary output with per-criterion missing-drop counters.
   - Rationale: makes strict behavior observable and debuggable.

## Risks / Trade-offs

- [Risk] Strict defaults may reduce retained sample count for older runs missing fields.
  - Mitigation: provide `ignore` policy switch and clear summary counts.

- [Risk] Slightly more branching complexity in filter logic.
  - Mitigation: centralize policy checks in helper functions and test matrix.

- [Risk] Users may misinterpret lower retention as model quality regression.
  - Mitigation: include explicit missing-field drop reporting in summary.
