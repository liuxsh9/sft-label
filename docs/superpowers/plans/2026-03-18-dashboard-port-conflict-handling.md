# Dashboard Port Conflict Handling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect dashboard service port conflicts early, show the occupying process clearly, and let interactive users choose a new port instead of failing hard.

**Architecture:** Add a small port-conflict inspection layer in the dashboard service module so service start/restart can raise structured errors before launching `http.server`/pm2. Handle those errors in CLI interactive flows by prompting for a replacement port, persisting the updated service config, and retrying without losing the user's session.

**Tech Stack:** Python, argparse CLI, pytest, existing dashboard service config store.

---

### Task 1: Add failing tests for conflict detection and interactive retry

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/tests/test_dashboard_service.py`
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/tests/test_launcher.py`
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/tests/test_cli_start.py` (if needed)

- [ ] **Step 1: Write failing dashboard-service tests** for:
  - service start raising a structured conflict when another process owns the port,
  - retry path updating service config/public URL for simple host:port URLs,
  - keeping custom public URLs stable except for direct host:port matches.
- [ ] **Step 2: Run focused tests to verify they fail**
- [ ] **Step 3: Write failing CLI/launcher tests** for interactive port re-selection on `dashboard-service start/restart` and auto-publish bootstrap start flow.
- [ ] **Step 4: Run focused tests to verify they fail**

### Task 2: Implement service-layer conflict detection and config update helpers

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/src/sft_label/dashboard_service.py`

- [ ] **Step 1: Add minimal structured conflict model/helper** that reports host, port, pid, command, and whether the conflict belongs to the current service.
- [ ] **Step 2: Add preflight conflict detection before builtin and pm2 starts/restarts**.
- [ ] **Step 3: Add helper(s) to persist port updates and cautiously rewrite simple share/public URLs**.
- [ ] **Step 4: Run service tests and make them pass**.

### Task 3: Implement interactive retry flow and user-facing messaging

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/src/sft_label/cli.py`
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/src/sft_label/launcher.py` (only if helper wiring needs it)

- [ ] **Step 1: Add CLI helper to catch port-conflict errors, print occupant details, and prompt for a new port in interactive flows**.
- [ ] **Step 2: Reuse the helper from `cmd_dashboard_service` and start-mode dashboard bootstrap flow**.
- [ ] **Step 3: Preserve non-interactive behavior as a clear error**.
- [ ] **Step 4: Run focused CLI/launcher tests and make them pass**.

### Task 4: Sync docs and verify

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/README.md`
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/README.zh-CN.md`
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/docs/guides/output-files-and-dashboards.md`
- Modify: `/Users/lxs/.codex/worktrees/44fc/sft-label/docs/agent-guide.md` (if agent-facing behavior changes need documentation)

- [ ] **Step 1: Document the new conflict handling behavior and recommended workflow**.
- [ ] **Step 2: Run focused tests plus any doc-related lint/consistency checks available**.
- [ ] **Step 3: Run final targeted pytest selection covering dashboard service + start flow**.
