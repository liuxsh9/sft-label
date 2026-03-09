## Why

For high-quality SFT selection, current filtering behavior is too permissive: when fields like `quality.correctness` or `thinking_mode` are missing, samples can pass hard gates unintentionally. This introduces low-confidence data into curated outputs.

## What Changes

- Define strict missing-field behavior for hard gate criteria (`correctness_min`, `thinking_mode`, turn-level quality/value gates).
- Add an explicit policy switch for missing gate fields (`fail` vs `ignore`) with strict mode as default.
- Extend filter summary with missing-field drop counts by criterion.
- Preserve compatibility by allowing users to opt into permissive behavior when needed.
- Add regression tests for strict and permissive policies.

## Capabilities

### New Capabilities
- `strict-filter-gates`: Filter gates treat missing required fields deterministically, preventing accidental retention.

### Modified Capabilities
- None.

## Impact

- Affected code: `src/sft_label/tools/filter_value.py`, CLI wiring in `src/sft_label/cli.py`.
- Affected tests: `tests/test_filter.py` for missing-field and policy behavior.
- Data quality impact: improves precision of high-value SFT subsets.
