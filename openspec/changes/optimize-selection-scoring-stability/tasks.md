## 1. Regression Fixtures and Measurement Baseline

- [x] 1.1 Add an audited Nemotron SWE regression fixture pack covering opener, tool-action exploration, common final bug-fix, rare-domain final, and summary-heavy final slices
- [x] 1.2 Add an offline comparison utility that reports old/new value-selection ordering deltas, threshold distributions, and top/bottom bucket shifts on the regression pack
- [x] 1.3 Add tests that assert relative ordering for audited slices instead of exact floating-point equality

## 2. Stage and Evidence Feature Extraction

- [x] 2.1 Implement deterministic `trajectory_stage` inference from turn position, response structure, tool-call density, and final-summary markers
- [x] 2.2 Implement low-information tool-action detection for view/grep/ls/setup-heavy slices with bounded heuristics
- [x] 2.3 Implement summary-evidence detection using concrete technical signals such as file paths, symbols, root-cause phrases, fix descriptions, and verification commands/results

## 3. Pass 2 Aggregation Rebalance

- [x] 3.1 Add configurable post-LLM penalties/bonuses for opener, exploration, implementation, verification, and final-summary slices
- [x] 3.2 Rebalance `selection_score` composition so rarity/diversity and stage/evidence modifiers have measurable effect beyond intra-class rank
- [x] 3.3 Keep the persisted scoring output schema unchanged and add compatibility tests for dashboards/filtering consumers

## 4. Domain Backfill and Common-Case Stabilization

- [x] 4.1 Add conservative deterministic domain backfill for clearly inferable repo-repair slices using existing taxonomy labels only
- [x] 4.2 Add tests showing ambiguous cases remain unlabeled while obvious compiler/ML/cloud/API cases are backfilled correctly
- [x] 4.3 Re-run audited regression pack and confirm common Python bug-fix finals remain high value but receive less selection bonus than rarer equally strong slices

## 5. Prompt-Safe Pass 2 Tightening

- [x] 5.1 Refactor Pass 2 prompt wording for tool-only and unsupported-summary guardrails without increasing prompt token count
- [x] 5.2 Add a prompt budget regression test that fails if the revised prompt payload exceeds the current compact production baseline
- [x] 5.3 Run smoke scoring on the Nemotron SWE sample set and verify no payload-growth regressions or firewall-risking message expansion

## 6. E2E Validation and Rollout Safety

- [x] 6.1 Run full e2e scoring on the audited Nemotron SWE sample file and compare value/selection distributions before and after the change
- [x] 6.2 Verify targeted outcomes: opener/tool-action slices move down, evidence-backed rare finals stay in the top band, summary-only finals lose excess headroom
- [x] 6.3 Add config gating and rollback toggles so the previous selection path can be restored during one release window if ranking regressions appear
