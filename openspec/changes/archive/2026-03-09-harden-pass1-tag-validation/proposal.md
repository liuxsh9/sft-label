## Why

Pass 1 tag validation can crash on malformed single-select fields (for example, list/dict outputs from the model), causing avoidable sample failures. For large-scale labeling, validation should degrade gracefully and keep the pipeline moving.

## What Changes

- Harden single-select validation to be type-safe before alias/pool checks.
- Convert malformed single-select values into validation issues and sanitized fallback values instead of raising runtime exceptions.
- Preserve unmapped diagnostics without introducing non-serializable values.
- Add regression tests for malformed single-select payloads (`intent`, `difficulty`, `context`).
- Keep existing behavior for valid string outputs unchanged.

## Capabilities

### New Capabilities
- `pass1-tag-validation-hardening`: Pass 1 validation tolerates malformed single-select outputs without pipeline crashes.

### Modified Capabilities
- None.

## Impact

- Affected code: `src/sft_label/pipeline.py` (`validate_tags` and related normalization paths).
- Affected tests: Pass 1 validation and mock e2e malformed-response cases.
- Reliability impact: lower avoidable failure rates in long-running labeling jobs.
