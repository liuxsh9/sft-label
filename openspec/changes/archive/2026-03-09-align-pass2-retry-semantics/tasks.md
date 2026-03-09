## 1. Retry Semantics Alignment

- [x] 1.1 Update Pass 2 retry loop bounds to initial attempt + configured retries
- [x] 1.2 Keep monitor accounting (`attempts`, `llm_calls`) consistent with new loop
- [x] 1.3 Update scoring workload estimation logic to reflect aligned semantics

## 2. Config Wiring

- [x] 2.1 Wire `selection_smoothing_prior` config into `compute_selection_scores`
- [x] 2.2 Wire `selection_smoothing_prior` config into `compute_selection_scores_from_summaries`
- [x] 2.3 Ensure summary metadata records effective smoothing prior for traceability

## 3. Verification

- [x] 3.1 Add regression test: `sample_max_retries=1` yields up to 2 attempts in Pass 2
- [x] 3.2 Add regression test for non-retryable early exit behavior
- [x] 3.3 Add regression test that custom smoothing prior changes percentile shrinkage path
