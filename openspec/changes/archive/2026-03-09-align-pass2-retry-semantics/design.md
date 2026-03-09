## Context

Pass 1 and Pass 2 both expose `sample_max_retries`, but they interpret it differently. This creates hidden reliability gaps and confusing operator experience. A second issue is unused configuration (`selection_smoothing_prior`) in selection ranking.

## Goals / Non-Goals

**Goals:**
- Normalize retry semantics across Pass 1 and Pass 2.
- Ensure scoring workload estimates remain directionally accurate.
- Make selection smoothing prior truly runtime-configurable.

**Non-Goals:**
- No changes to LLM prompt content.
- No changes to value score weights.

## Decisions

1. Use loop bounds equivalent to Pass 1 semantics in Pass 2 (`for attempt in range(max_retries + 1)`).
   - Rationale: one shared mental model for retries.

2. Keep monitor fields (`attempts`, `llm_calls`) aligned with actual calls.
   - Rationale: reliability diagnostics depend on precise accounting.

3. Update workload estimate helper to map `sample_max_retries` to expected total attempts using aligned semantics.
   - Rationale: prevent underestimation when retries are enabled.

4. Replace hardcoded `SELECTION_SMOOTHING_PRIOR` usage with config-sourced value in both full-sample and summary-based selection calculations.
   - Rationale: make runtime tuning effective and reproducible.

## Risks / Trade-offs

- [Risk] Higher call volume after retry alignment may increase latency/cost.
  - Mitigation: keep rps/concurrency guards and expose clear estimates.

- [Risk] Existing dashboards may shift due to smoothing prior now taking effect.
  - Mitigation: include config values in output stats metadata for traceability.

- [Risk] Tests that assumed old attempt counts will fail.
  - Mitigation: update tests to explicit semantic expectation.
