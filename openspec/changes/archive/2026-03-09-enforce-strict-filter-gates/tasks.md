## 1. Config and CLI

- [x] 1.1 Add `missing_gate_policy` to `FilterConfig` with default `fail`
- [x] 1.2 Add CLI flag wiring for `--missing-gate-policy`
- [x] 1.3 Validate accepted policy values in CLI and library entry points

## 2. Filter Logic

- [x] 2.1 Apply policy-aware checks for `correctness_min` and `thinking_mode`
- [x] 2.2 Apply policy-aware checks in `_passes_turn_criteria`
- [x] 2.3 Add and populate missing-field drop counters in summary output

## 3. Verification

- [x] 3.1 Add tests for strict policy dropping missing correctness/thinking mode
- [x] 3.2 Add tests for permissive policy preserving current behavior
- [x] 3.3 Add tests for turn-level missing-field handling and summary counters
