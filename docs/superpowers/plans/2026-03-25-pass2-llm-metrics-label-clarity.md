# Pass 2 LLM Metrics Label Clarity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify Pass 2 and global LLM log labels without changing any counting or scoring behavior.

**Architecture:** Add focused regression tests around the affected strings, then update the CLI/scoring presentation layer only. Keep all computation paths untouched.

**Tech Stack:** Python, pytest, existing CLI/progress/summary helpers.

---

## Task 1: Lock in the desired wording with tests

**Files:**
- Modify: `tests/test_cli_progress.py`
- Modify: `tests/test_scoring_estimation.py`

- [ ] **Step 1: Write failing tests**
  - Add a tracker test asserting the global progress line is labeled `LLM (P1+P2)`.
  - Add a scoring summary test asserting the Pass 2 summary prints `Selective-estimated samples:` and `LLM calls (Pass 2 actual):`.

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:
`uv run pytest tests/test_cli_progress.py tests/test_scoring_estimation.py -q`

Expected:
FAIL on the new wording assertions.

## Task 2: Implement the copy-only output changes

**Files:**
- Modify: `src/sft_label/cli.py`
- Modify: `src/sft_label/scoring.py`

- [ ] **Step 1: Update global progress task label**
  - Change the task description from `LLM` to `LLM (P1+P2)`.

- [ ] **Step 2: Update Pass 2 plan wording**
  - Rename the plan field to `pass2_llm~...`.

- [ ] **Step 3: Update Pass 2 summary wording**
  - Rename the summary labels for selective-estimated samples and Pass 2 actual LLM calls.

## Task 3: Verify

**Files:**
- Verify only

- [ ] **Step 1: Run targeted tests**

Run:
`uv run pytest tests/test_cli_progress.py tests/test_scoring_estimation.py -q`

Expected:
PASS
