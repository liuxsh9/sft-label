# Agent Doc Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate agent-facing repo instructions into one shared guide and replace duplicated wrapper files with thin pointers.

**Architecture:** Store current-project facts in a single canonical markdown file under `docs/`, then keep `AGENTS.md` and `CLAUDE.md` as tool-specific wrappers. Refresh content from the actual CLI, README, source tree, and fixture layout instead of copying stale text.

**Tech Stack:** Markdown, repository documentation, CLI inspection, git diff review.

---

## Chunk 1: Gather and codify the current repo state

### Task 1: Build the canonical shared guide

**Files:**
- Create: `/Users/lxs/.codex/worktrees/e39f/sft-label/docs/agent-guide.md`
- Reference: `/Users/lxs/.codex/worktrees/e39f/sft-label/README.md`
- Reference: `/Users/lxs/.codex/worktrees/e39f/sft-label/src/sft_label/cli.py`
- Reference: `/Users/lxs/.codex/worktrees/e39f/sft-label/pyproject.toml`

- [ ] **Step 1: Gather current command and architecture information**

Inspect the CLI help, README, source layout, and fixture directories.

- [ ] **Step 2: Write the shared guide**

Include commands, architecture, key files, fixture data, invariants, and a refresh checklist.

- [ ] **Step 3: Review the guide against the current repo snapshot**

Check that command names and fixture counts match the live repository.

## Chunk 2: Replace duplicated wrappers

### Task 2: Convert AGENTS.md and CLAUDE.md into thin entry points

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e39f/sft-label/AGENTS.md`
- Modify: `/Users/lxs/.codex/worktrees/e39f/sft-label/CLAUDE.md`

- [ ] **Step 1: Replace duplicated content with concise wrappers**

Point both files at `/Users/lxs/.codex/worktrees/e39f/sft-label/docs/agent-guide.md`.

- [ ] **Step 2: Keep only tool-specific framing in each wrapper**

Do not duplicate project facts that now belong in the shared guide.

## Chunk 3: Verify consistency

### Task 3: Review the resulting diffs

**Files:**
- No additional code changes expected

- [ ] **Step 1: Inspect the final diff**

Run: `git diff -- docs/agent-guide.md AGENTS.md CLAUDE.md docs/superpowers/specs/2026-03-17-agent-doc-refresh-design.md docs/superpowers/plans/2026-03-17-agent-doc-refresh.md`

Expected: one new canonical guide, two thin wrappers, and supporting design/plan docs.

- [ ] **Step 2: Sanity-check wrapper wording**

Ensure future maintainers are directed to update the shared guide first.
