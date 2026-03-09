## Context

`validate_tags` currently assumes single-select values are string-like and hashable. When the model emits malformed structures (for example list), membership checks can throw runtime errors and fail the sample path instead of returning a clean validation issue.

## Goals / Non-Goals

**Goals:**
- Ensure single-select validation is robust for arbitrary JSON types.
- Preserve existing alias resolution behavior for valid strings.
- Emit actionable validation issues for malformed values.

**Non-Goals:**
- No taxonomy changes.
- No changes to multi-select validation semantics.

## Decisions

1. Add explicit normalization guard for single-select dimensions.
   - Rationale: avoid unhashable membership checks and unexpected exceptions.

2. Accept only non-empty strings for alias/pool resolution; all other types become invalid and map to empty value.
   - Rationale: deterministic behavior and simple invariants.

3. Record malformed type in validation issues (for observability) while keeping `cleaned` output JSON-safe.
   - Rationale: helps debugging model output quality without crashing pipeline.

4. Keep downstream monitor and stats behavior unchanged except reduced sample failures.
   - Rationale: minimize side effects.

## Risks / Trade-offs

- [Risk] Some malformed outputs that previously failed hard will now continue with empty labels.
  - Mitigation: surface clear validation issues and monitor counts.

- [Risk] Potential masking of severe model drift.
  - Mitigation: keep issue counts visible in stats and dashboards.

- [Risk] Edge cases around string coercion.
  - Mitigation: do not coerce arbitrary types to strings; only accept native string values.
