## Why

Pass 2 retry semantics are currently inconsistent with Pass 1: `sample_max_retries=1` yields one attempt in Pass 2 but two attempts in Pass 1. This lowers scoring success rate and makes configuration meaning ambiguous. Additionally, selection smoothing prior is configurable in `PipelineConfig` but not consistently applied.

## What Changes

- Align Pass 2 retry semantics with Pass 1 (`attempts = sample_max_retries + 1`).
- Keep retry backoff and monitor accounting consistent with updated semantics.
- Update scoring workload estimation to reflect aligned retry semantics.
- Wire `PipelineConfig.selection_smoothing_prior` into selection score computations.
- Add regression tests for retry counts and configurable smoothing prior behavior.

## Capabilities

### New Capabilities
- `pass2-retry-consistency`: Pass 2 retries and selection smoothing configuration behave consistently with documented config semantics.

### Modified Capabilities
- None.

## Impact

- Affected code: `src/sft_label/scoring.py` (retry loop, estimation, selection smoothing usage).
- Affected tests: scoring retry behavior and selection score config overrides.
- Operational impact: improved scoring robustness and predictable tuning behavior.
