# Silent Progress Heartbeats Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add visible progress indicators for long silent phases and reuse precomputed workload estimates to avoid redundant scans in `run --score` flows.

**Architecture:** Introduce a small CLI-facing heartbeat utility that wraps long-running synchronous phases with a start message and lightweight animated heartbeat. Thread precomputed estimate objects from `cmd_run()` into Pass 1 and Pass 2 so the same directory/file scan results can be reused instead of recomputed.

**Tech Stack:** Python 3.9+, pytest, existing CLI/pipeline/scoring modules.

---

## Chunk 1: Tests for heartbeat and estimate reuse

### Task 1: Heartbeat utility contract

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/tests/test_cli_progress.py`
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/cli.py`

- [ ] **Step 1: Write failing tests for heartbeat frame cycling and lifecycle output**
- [ ] **Step 2: Run targeted pytest to verify failure**
- [ ] **Step 3: Implement minimal heartbeat helper / wrapper in `cli.py`**
- [ ] **Step 4: Run targeted pytest to verify pass**

### Task 2: Estimate reuse contract

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/tests/test_cli_progress.py`
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/cli.py`
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/pipeline.py`
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/scoring.py`

- [ ] **Step 1: Write failing tests that `cmd_run()` passes precomputed estimates into Pass 1 / Pass 2**
- [ ] **Step 2: Run targeted pytest to verify failure**
- [ ] **Step 3: Implement parameter plumbing and fallback behavior**
- [ ] **Step 4: Run targeted pytest to verify pass**

## Chunk 2: Silent-phase wrapping

### Task 3: Wrap long synchronous phases with stage messages + heartbeat

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/cli.py`
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/pipeline.py`
- Modify: `/Users/lxs/.codex/worktrees/17ca/sft-label/src/sft_label/scoring.py`

- [ ] **Step 1: Add wrappers for workload estimation, dashboard generation, conversation aggregation, and auto-publish**
- [ ] **Step 2: Ensure wrappers preserve existing output and fail/return semantics**
- [ ] **Step 3: Run targeted pytest and any nearby regression tests**

## Chunk 3: Verification

### Task 4: Validate behavior and scan impact

**Files:**
- Test: `/Users/lxs/.codex/worktrees/17ca/sft-label/tests/test_cli_progress.py`

- [ ] **Step 1: Run targeted pytest suite for CLI progress coverage**
- [ ] **Step 2: Summarize which duplicate scans were removed and which scans still remain by design**
