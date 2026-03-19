# Start Launcher Runtime and Dashboard Flow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the interactive launcher and direct CLI defaults consistent around concurrency=200, support friendlier custom runtime input, and move final execution confirmation after dashboard decisions with a richer execution overview.

**Architecture:** Keep the parser/runtime override model intact and implement the behavior mostly in `src/sft_label/launcher.py` and `src/sft_label/cli.py`. Split dashboard flow into prompt-only collection plus post-confirm application so the final confirmation is truly final, then update tests and docs around the new defaults and ordering.

**Tech Stack:** Python, argparse, pytest

---

## Chunk 1: Tests first for launcher/runtime behavior

### Task 1: Add failing tests for new runtime defaults and custom input helpers

**Files:**
- Modify: `tests/test_launcher.py`
- Test: `tests/test_launcher.py`

- [ ] Add failing tests asserting score/smart-resume plans now default to concurrency 200.
- [ ] Add failing tests for launcher helper behavior covering custom concurrency and custom RPS input in legacy/non-TTY flows.
- [ ] Run targeted launcher tests and confirm failure is due to old defaults/behavior.

### Task 2: Add failing tests for start/dashboard flow changes

**Files:**
- Modify: `tests/test_launcher.py`
- Test: `tests/test_launcher.py`

- [ ] Add failing tests asserting auto-publish prompt defaults to yes and that final execution confirmation occurs after dashboard decisions.
- [ ] Add failing tests for richer launch overview text (dashboard plan/runtime summary) at a minimally stable assertion level.
- [ ] Run the targeted `cmd_start` tests and confirm failure is due to the old ordering/default prompt behavior.

## Chunk 2: Implement runtime/default changes

### Task 3: Update global/runtime defaults and launcher custom numeric handling

**Files:**
- Modify: `src/sft_label/config.py`
- Modify: `src/sft_label/launcher.py`
- Test: `tests/test_launcher.py`

- [ ] Change global default concurrency to 200.
- [ ] Centralize launcher default concurrency values so run, score, and smart-resume are consistent.
- [ ] Implement custom numeric entry for concurrency and RPS max limit in both switch-panel and legacy flows with validation/re-prompting.
- [ ] Re-run targeted launcher tests and get them green.

## Chunk 3: Implement start/dashboard flow refactor

### Task 4: Split dashboard collection from side effects and move final confirmation

**Files:**
- Modify: `src/sft_label/cli.py`
- Test: `tests/test_launcher.py`
- Test: `tests/test_cli_start.py`

- [ ] Refactor dashboard setup into prompt-only collection plus post-confirm application.
- [ ] Move the final execute confirmation to after dashboard choices for eligible workflows.
- [ ] Expand launch summary into an execution overview that includes runtime and dashboard plan details while keeping dry-run behavior safe.
- [ ] Re-run targeted `cmd_start` and CLI parser/runtime tests and get them green.

## Chunk 4: Documentation and verification

### Task 5: Update docs and run verification

**Files:**
- Modify: `docs/guides/interactive-launcher.md`
- Modify: `docs/guides/getting-started.md`
- Modify: `docs/guides/output-files-and-dashboards.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

- [ ] Update docs to describe concurrency 200 defaults, custom runtime entry, auto-publish default yes, and the new final confirmation flow.
- [ ] Run targeted test suites, then a broader regression selection.
- [ ] Review changed files for consistency and summarize any follow-up risks.
