# UI Extension Example and Start Selection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a more professional Web-UI analysis extension example and make `start` support selecting one or more extension specs from built-in/custom directories, with strong cost warnings and better guidance.

**Architecture:** Keep the current extension runtime unchanged. Improve only example specs, launcher UX, and docs. The launcher should support both direct path entry and directory-based multi-select, while clearly warning that each enabled extension adds extra extension-labeling calls and is not suitable for full-dataset domain-specific runs.

**Tech Stack:** Python, pytest, launcher prompts, YAML example specs, markdown docs.

---

## Chunk 1: Example spec and docs

### Task 1: Add a professional Web UI analysis extension example

**Files:**
- Create: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/examples/extensions/ui_web_analysis_v1.yaml`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/guides/pass1-extension-labeling.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/README.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/README.zh-CN.md`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/docs/guides/interactive-launcher.md`

- [ ] **Step 1: Add the new Web-only UI analysis example spec**
- [ ] **Step 2: Update docs to define UI SFT data, explain analysis-oriented UI labels, and explain why mobile should be a separate extension**
- [ ] **Step 3: Add explicit cost/risk guidance against using domain-personalized extensions on full datasets**
- [ ] **Step 4: Self-review wording/examples for consistency**

## Chunk 2: Start UX and tests

### Task 2: Support directory-based extension selection in `start`

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/src/sft_label/launcher.py`
- Modify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_launcher.py`

- [ ] **Step 1: Add failing launcher tests for choosing extension specs from a directory and for cost-warning UX**
- [ ] **Step 2: Run launcher tests and confirm failure**
- [ ] **Step 3: Implement minimal directory-scanning / multi-select flow while preserving manual path entry**
- [ ] **Step 4: Re-run launcher tests and confirm pass**

## Chunk 3: Regression verification

### Task 3: Run focused and full verification

**Files:**
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_launcher.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_cli_progress.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_label_extensions_schema.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_export_review.py`
- Verify: `/Users/lxs/.codex/worktrees/a71d/sft-label/tests/test_visualize_labels.py`

- [ ] **Step 1: Run focused extension/launcher regressions**
- [ ] **Step 2: Fix any fallout conservatively**
- [ ] **Step 3: Run full test suite and lint**
- [ ] **Step 4: Summarize user-facing changes and remaining cautions**
