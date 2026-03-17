# Top 10 Multi-turn Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the prioritized multi-turn/trajectory labeling, scoring, export, and test improvements while preserving correctness and making the system more production-ready for rising multi-turn data share.

**Architecture:** Land foundational identity/metadata changes first, then export/modeling improvements, then scoring/aggregation scaling features, then diagnostics and fixtures. Use isolated git worktrees per task branch, TDD within each task, and merge into a dedicated integration worktree in dependency order.

**Tech Stack:** Python 3.12, pytest, uv, git worktrees, Codex subagents.

---

## Dependency graph

- **T1 conversation_uid** → prerequisite for T6, T7, T10 and recommended for any conversation-keyed logic.
- **T2 turn metadata semantics** → prerequisite for T5, T9, T10.
- **T4 system turn preservation** → informs T3 export fidelity; can be developed in parallel but merge before or together with T3.
- **T3 Pangu same-request trajectory export** → depends on export shape decisions from T4.
- **T6 trajectory objects** → depends on T1; prerequisite for T7, T8, and recommended for T9/T10.
- **T7 streaming aggregation** → depends on T1 and T6.
- **T8 trajectory quality features** → depends on T6 and should merge after T7.
- **T9 thinking-mode conversation summary** → depends on T2 and should merge after T6 to align with new conversation model.
- **T5 selective Pass 2 scoring** → depends on T2; recommended to merge after T6 so trajectory-aware selection can use new object semantics.
- **T10 fixtures/regressions** → spans all behavior; stage in phases and finalize after T1/T2/T3/T4/T6/T9.

## Parallelization waves

### Wave A — foundation
- T1 conversation_uid
- T2 turn metadata semantics
- T4 system turn preservation
- T10 fixture scaffolding / accounting baselines

### Wave B — export + model semantics
- T3 Pangu export fidelity (after T4 decisions available)
- T6 trajectory objects (after T1)

### Wave C — aggregation + scoring
- T7 streaming aggregation (after T1+T6)
- T9 thinking-mode summary (after T2, ideally after T6)
- T5 selective Pass 2 scoring (after T2, ideally after T6)

### Wave D — quality features + final regressions
- T8 trajectory quality features (after T6+T7)
- T10 finalize fixtures/regressions across merged behavior

## Integration order in main worktree
1. T1
2. T2
3. T4
4. T3
5. T6
6. T7
7. T9
8. T5
9. T8
10. T10

## Per-task worktree map
- T1: `/Users/lxs/.codex/worktrees/t101/sft-label`
- T2: `/Users/lxs/.codex/worktrees/t102/sft-label`
- T3: `/Users/lxs/.codex/worktrees/t103/sft-label`
- T4: `/Users/lxs/.codex/worktrees/t104/sft-label`
- T5: `/Users/lxs/.codex/worktrees/t105/sft-label`
- T6: `/Users/lxs/.codex/worktrees/t106/sft-label`
- T7: `/Users/lxs/.codex/worktrees/t107/sft-label`
- T8: `/Users/lxs/.codex/worktrees/t108/sft-label`
- T9: `/Users/lxs/.codex/worktrees/t109/sft-label`
- T10: `/Users/lxs/.codex/worktrees/t110/sft-label`
- Integration: `/Users/lxs/.codex/worktrees/t10main/sft-label`
