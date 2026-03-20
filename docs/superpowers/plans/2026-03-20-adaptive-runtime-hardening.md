# Adaptive Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the adaptive LLM runtime more faithful to the design goal: reduce pressure when the API becomes unstable, then recover safely and automatically after sustained health, with tests that prove the behavior.

**Architecture:** Harden the core state machine in `src/sft_label/llm_runtime.py`, then prove the behavior through a mix of primitive-level tests and pass1/pass2 integration tests. Keep the public runtime shape intact and prefer small behavioral fixes over broad refactors.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, existing runtime/pipeline/scoring modules.

---

## Scope

Planned fixes in this iteration:

1. Add a minimum-observation guard so one early failure does not immediately force `open`/`degraded` under default config.
2. Make adaptive RPS retargeting stricter by clipping token/burst state when the target RPS is reduced.
3. Allow repeated abnormal-response windows to trigger `open`, not only `degraded`, using an explicit abnormal-open threshold.
4. Prevent `probing` from getting stuck on non-infra terminal outcomes; define that path to fall back to `degraded`, not `healthy` or `open`.
5. Add regression tests that prove:
   - runtime primitive behavior for guarded degrade/open and recovery,
   - real admission control drops concurrency after instability,
   - concurrency can rise again after a healthy window,
   - pass2/runtime observe path covers abnormal responses.

Explicitly deferred unless they fall out naturally from the above:

- redesigning recovery to be time-based instead of request-count-based,
- adding new CLI / launcher knobs for the new adaptive thresholds,
- wiring a background probe loop.

---

## Files

**Modify**
- `src/sft_label/llm_runtime.py`
- `src/sft_label/config.py`
- `src/sft_label/pipeline.py`
- `src/sft_label/scoring.py`
- `tests/test_llm_runtime.py`
- `tests/test_pass1_adaptive_runtime.py`
- `tests/test_scoring.py`
- `tests/test_cli_adaptive_runtime.py`

---

## Task 1: Lock down the desired runtime semantics with failing tests

**Files:**
- Modify: `tests/test_llm_runtime.py`
- Modify: `tests/test_pass1_adaptive_runtime.py`
- Modify: `tests/test_scoring.py`

- [ ] Add a failing test showing a single early timeout/429 does **not** immediately open the circuit when the configured minimum sample count has not been reached.
- [ ] Add a failing test showing a single early timeout/429 does **not** immediately degrade the runtime either; `state`, `effective_concurrency`, and `effective_rps` stay at healthy defaults until the minimum-observation floor is reached.
- [ ] Add boundary tests for the new observation floors at `floor - 1` and `floor`.
- [ ] Add a failing test showing abnormal responses can contribute to an `open` transition once the observation floor is satisfied.
- [ ] Add a failing test showing lowering target RPS clips token availability so post-degrade bursts are constrained.
- [ ] Add a failing test showing `probing` transitions to `degraded` on non-infra terminal outcomes rather than remaining stuck.
- [ ] Add a failing concurrency test that measures admission before degrade, after degrade, and after recovery, and also asserts `effective_concurrency` / `effective_rps` rise back to base values.
- [ ] Add a failing pass1 test proving adaptive mode bypasses the legacy semaphore path and is controlled by runtime admission instead.
- [ ] Add a failing pass2 test proving repeated abnormal responses can actually move the runtime state, not just be observed.
- [ ] Add a failing config regression test proving the new adaptive thresholds flow from `PipelineConfig` into pass1/pass2 runtime construction without adding new CLI flags.
- [ ] Run the targeted tests and verify they fail for the expected reasons before touching production code.

Suggested commands:
- `uv run --extra dev pytest tests/test_llm_runtime.py -q`
- `uv run --extra dev pytest tests/test_pass1_adaptive_runtime.py -q`
- `uv run --extra dev pytest tests/test_scoring.py -q -k 'adaptive_runtime or malformed_response or recovery_sweep'`
- `uv run --extra dev pytest tests/test_cli_adaptive_runtime.py -q`

---

## Task 2: Implement the minimal runtime fixes

**Files:**
- Modify: `src/sft_label/llm_runtime.py`

- [ ] Add configurable minimum-observation thresholds for degrade/open decisions, with sensible defaults that preserve the current design but avoid one-sample overreaction.
- [ ] Add the supporting `PipelineConfig` fields and wire them through `pipeline._build_adaptive_runtime()` and `scoring._instantiate_runtime()` without expanding the CLI/launcher surface.
- [ ] Extend the state machine so severe abnormal-response windows may trigger `open`, using an explicit abnormal-open threshold.
- [ ] Update adaptive RPS retargeting so reducing the target also clips burst/token state.
- [ ] Ensure `probing` does not stall forever on terminal non-infra outcomes and instead falls back to `degraded`.
- [ ] Keep the change local to runtime internals unless a test seam requires a narrow surface change.
- [ ] Re-run the focused runtime tests until they pass.

Suggested commands:
- `uv run --extra dev pytest tests/test_llm_runtime.py -q`

---

## Task 3: Prove pass1/pass2 integration still behaves correctly

**Files:**
- Modify: `tests/test_pass1_adaptive_runtime.py`
- Modify: `tests/test_scoring.py`

- [ ] Confirm pass1 adaptive mode bypasses the legacy semaphore path and now exhibits runtime-controlled admission drop/recovery under the new thresholds.
- [ ] Confirm pass2 abnormal-response classification still reaches runtime observe, can influence runtime state under repeated abnormal windows, and remains recovery-sweep compatible.
- [ ] Add/adjust only the minimal pass1/pass2 integration code needed if the tests reveal a missing seam.
- [ ] Re-run the targeted integration tests until they pass.

Suggested commands:
- `uv run --extra dev pytest tests/test_pass1_adaptive_runtime.py -q`
- `uv run --extra dev pytest tests/test_scoring.py::test_pass2_score_one_malformed_response_counts_as_abnormal_infra tests/test_scoring.py::test_pass2_recovery_sweep_retries_only_infra_failures_and_emits_runtime_summary -q`

---

## Task 4: Full verification and review

**Files:**
- No new production files expected

- [ ] Run the combined targeted verification suite.
- [ ] Review the diff against the intended scope and confirm no unrelated behavior changed.
- [ ] Dispatch multiple reviewer subagents for final review of runtime logic, pass1 integration, and pass2 integration.
- [ ] Summarize what was fixed, what remains deferred, and what should be next if we want to go from “partially meets design” to “fully robust”.

Suggested commands:
- `uv run --extra dev pytest tests/test_llm_runtime.py tests/test_pass1_adaptive_runtime.py tests/test_cli_adaptive_runtime.py -q`
- `uv run --extra dev pytest tests/test_scoring.py -q -k 'adaptive_runtime or malformed_response or recovery_sweep'`
- `git diff --stat`
