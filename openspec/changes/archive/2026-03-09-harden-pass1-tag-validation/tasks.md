## 1. Validation Hardening

- [x] 1.1 Add guarded normalization for single-select values in `validate_tags`
- [x] 1.2 Ensure alias and pool checks run only for valid strings
- [x] 1.3 Convert malformed values into issues and sanitized empty values

## 2. Diagnostics and Monitoring

- [x] 2.1 Keep malformed single-select details visible in validation issues
- [x] 2.2 Ensure `unmapped` and cleaned outputs remain JSON-safe
- [x] 2.3 Confirm monitor/stats aggregation works with hardened outputs

## 3. Verification

- [x] 3.1 Add unit tests for list/dict/bool malformed single-select values
- [x] 3.2 Add regression test ensuring no `TypeError` is raised in malformed cases
- [x] 3.3 Add e2e-mock coverage for malformed call output degradation path
