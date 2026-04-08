# CLAUDE.md

This file is a thin wrapper for Claude Code.

Canonical repository guidance lives in:

- `/Users/lxs/.codex/worktrees/e39f/sft-label/docs/agent-guide.md`

Use that shared guide for:

- commands,
- architecture,
- key files,
- fixture data,
- maintenance/update instructions.

Keep project facts in the shared guide rather than duplicating them here.

## Quick references

- CLI complete parameter reference & non-interactive usage: `docs/guides/cli-reference.md`
  - AI agents / scripts should always use direct CLI (`run`, `score`, etc.), never `start`
  - All interactive launcher defaults are documented with their CLI flag equivalents
