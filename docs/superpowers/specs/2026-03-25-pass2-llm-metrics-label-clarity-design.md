# Pass 2 LLM Metrics Label Clarity Design

## Goal

Reduce operator confusion in Pass 2 logs by making it obvious which numbers represent:

1. Pass 2 planned LLM calls,
2. Pass 2 actual LLM calls,
3. selectively estimated samples with no LLM call, and
4. global Pass 1 + Pass 2 LLM progress.

## Scope

This change is intentionally copy-only. It must not alter:

- workload estimation formulas,
- LLM progress accounting,
- scoring behavior, or
- retry behavior.

## Recommended Approach

Keep the existing metrics and rename only the user-facing labels.

### Pass 2 plan line

Change the `Plan | ... llm~N` wording so it explicitly says this is the Pass 2 planned LLM count.

### Pass 2 summary

Rename:

- `Estimated:` → `Selective-estimated samples:`
- `LLM calls:` → `LLM calls (Pass 2 actual):`

This preserves the values while clarifying the counting basis.

### Global progress tracker

Rename the Rich task label from `LLM` to `LLM (P1+P2)` so it is clearly not Pass 2 only.

## Non-Goals

- No new counters.
- No changes to summary math.
- No changes to tests unrelated to wording.
