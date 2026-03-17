# Agent Doc Refresh Design

## Background

The repository had both `/Users/lxs/.codex/worktrees/e39f/sft-label/AGENTS.md` and `/Users/lxs/.codex/worktrees/e39f/sft-label/CLAUDE.md` as near-duplicate full guides. They had already started to drift from the current repository state, especially around CLI commands, dashboard workflow, fixture contents, and newer modules.

## Goal

Create a single shared source of truth for agent-facing repository instructions, then convert the tool-specific wrapper files into thin entry points that reference the shared guide.

## Approach

1. Add `/Users/lxs/.codex/worktrees/e39f/sft-label/docs/agent-guide.md` as the canonical repository guide for agents.
2. Refresh commands, architecture, key files, and fixture details from the current repo state.
3. Replace `/Users/lxs/.codex/worktrees/e39f/sft-label/AGENTS.md` and `/Users/lxs/.codex/worktrees/e39f/sft-label/CLAUDE.md` with concise wrappers that point to the shared guide.
4. Add a short maintenance checklist so future updates happen in one place.

## Non-goals

- Rewriting the human-facing README set.
- Changing runtime behavior or CLI semantics.
- Introducing agent-specific logic beyond thin wrapper notes.

## Deliverables

- Canonical shared guide in `docs/agent-guide.md`
- Thin `AGENTS.md` wrapper
- Thin `CLAUDE.md` wrapper
