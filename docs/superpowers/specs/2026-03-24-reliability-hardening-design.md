# Reliability Hardening Design for Large-Scale Pass1/Pass2/Dashboard Runs

## Background

`sft-label` already contains several reliability-oriented mechanisms:

1. Pass 1 and Pass 2 both support resumable workflows.
2. Large JSONL inputs can be processed in chunked mode.
3. Adaptive runtime controls concurrency and request rate under upstream instability.
4. Dashboard generation and dashboard-service publishing are separated from core labeling/scoring logic.
5. Pass 2 already has a deferred postprocess mode for large runs.

Those mechanisms are useful, but the current system still behaves more like a collection of partially recoverable stages than a single production-safe long-running pipeline.

The review found several concrete failure modes that matter at 600k+ scale:

- Pass 2 resume can miss hidden working artifacts and re-run work.
- Some Pass 2 resume / recovery paths can leave inconsistent or missing output files.
- Pass 1 chunked output is not atomic and only resumes at whole-file granularity.
- Some “deferred” or “streaming” paths still perform full-memory aggregation or full-file rewrites at the tail.
- Auto-publish can publish incomplete runs when postprocess is still deferred.
- Dashboard publish is not atomic and can expose half-published content.

The user explicitly wants the **minimum repair set** needed to improve operational reliability without changing functional behavior.

---

## Design Goal

Harden `sft-label` so that the existing Pass 1 → Pass 2 → postprocess → dashboard publish workflow is materially safer for large long-running runs, by fixing the known P0/P1 reliability problems while preserving current labeling, scoring, and dashboard semantics.

Success means:

1. interrupted runs do not silently lose completed work,
2. resume logic detects real in-progress state correctly,
3. large-run protection paths no longer publish incomplete results as if complete,
4. the most dangerous O(N) memory / rewrite hotspots are reduced or gated,
5. publishing becomes atomic enough to avoid serving broken intermediate output,
6. all of the above are protected by regression tests.

---

## Non-Goals

1. Do **not** change Pass 1 label meaning, prompt content, or taxonomy logic.
2. Do **not** change Pass 2 scoring semantics, rarity formulas, or selection formulas.
3. Do **not** redesign dashboard UX or dashboard content schema beyond status / safety metadata needed for reliability.
4. Do **not** introduce a brand new global orchestration system or cross-stage database.
5. Do **not** perform broad cleanup refactors unrelated to the confirmed reliability issues.
6. Do **not** degrade the normal successful path for small/medium runs except where a safety guard is required to block an unsafe path.

---

## Hard Constraints

### 1. Safety-first, behavior-preserving changes only

This work may add:

- stricter resume detection,
- guardrails that refuse unsafe publish / resume flows,
- atomic write / swap behavior,
- more conservative execution for large-run postprocessing,
- streaming / incremental internal implementations.

This work must **not** change the output semantics of a successful healthy run.

### 2. No feature regressions in the normal successful path

If a run completes cleanly without interruption and without large-run defer conditions:

- labels must remain unchanged,
- scores must remain unchanged,
- dashboard payload semantics must remain unchanged,
- publish URLs and primary dashboard selection behavior must remain unchanged.

### 3. Large-run safety may block unsafe actions

Because the user chose **safety priority**, it is acceptable to:

- refuse auto-publish when postprocess is deferred or incomplete,
- refuse or redirect resume when artifacts are inconsistent,
- downgrade heavy postprocess behavior to a safer path.

Blocking an unsafe path is explicitly preferred over silently doing the wrong thing.

### 4. Publish safety must be enforced at the publish primitive, not only in launcher UX

Both of the following must be true:

1. launcher/CLI should warn early and avoid routing the user into unsafe publish paths;
2. the publish primitive itself must fail closed when the run is not publishable.

This includes:

- auto-publish from `start` / `run` flows,
- manual `dashboard-service register-run`,
- any direct use of `publish_run_dashboards()`.

### 5. Parallelizable implementation with explicit batch boundaries

The repair plan must be decomposed so multiple subagents can implement non-overlapping slices in parallel **within a batch**.

Not every workstream must be parallel with every other workstream. The design explicitly allows:

- **Batch 1** parallelism across correctness/guardrail/publish slices,
- **Batch 2** parallelism across Pass 1 hardening and postprocess/dashboard hardening,
- a **scoring large-run rewrite slice** that runs only after Batch 1 stabilizes scoring finalization seams.

This preserves safe parallelism without forcing conflicting edits in the same file.

---

## Problem Map

The confirmed issues fall into six clusters.

### Cluster A — Pass 2 resume / checkpoint correctness

Examples:

- smart resume misses hidden `*.next` / `*.checkpoint` artifacts,
- checkpoint-only fast path can discard the only complete scored output,
- recovery sweep can repair results but still leave stale `failed_value.jsonl` or `score_failures.jsonl`.

### Cluster B — Launcher / workflow safety

Examples:

- `smart resume` misclassifies incomplete Pass 2 state,
- `run --resume ... --score` can re-run Pass 2 unexpectedly,
- auto-publish can run even when heavy postprocess is still deferred.

### Cluster C — Publish atomicity and half-publish exposure

Examples:

- existing published directory is deleted before replacement content is fully copied,
- registry save is not atomic,
- repeated or concurrent publish can expose broken public state.

### Cluster D — Pass 1 reliability gaps

Examples:

- chunked Pass 1 writes directly to final filenames,
- resume granularity is still at whole-file level,
- chunked stats bookkeeping still grows with file size,
- FD budgeting does not fully reflect multi-file chunked output handles.

### Cluster E — Pass 2 large-run memory and rewrite hotspots

Examples:

- chunked scoring still retains full `raw_rarities` and `score_summaries`,
- directory mode still loads many `.json` inputs fully into memory,
- deferred postprocess still leaves directory-global selection rewrite as a heavy tail stage.

### Cluster F — Deferred postprocess / dashboard generation scalability

Examples:

- `complete-postprocess` retains all per-file conversation batches before global merge,
- `conversation_scores.json` paths are still full-memory oriented,
- explorer assets scale with all samples and can dominate dashboard generation / publish cost.

---

## Proposed Architecture

The design keeps the current file-based architecture, but introduces a tighter reliability contract around artifacts and stage completion.

## 1. Stage artifact contract

Each stage continues to write files into the existing run layout, but with stricter meaning.

### Pass 1

- **working artifacts** are explicitly intermediate and must not be mistaken for complete output,
- **final artifacts** must only become visible through atomic replace/swap where feasible,
- **checkpoint state** must distinguish “file flushed” from “file fully successful” when that distinction matters to workflow routing.

### Pass 2

- hidden working files (`*.next`, `*.checkpoint`) are first-class resume evidence,
- finalize must always leave exactly one authoritative final scored artifact set,
- recovery sweep must update final failed-artifact views consistently with recovered output.

### Postprocess / dashboard

A run may be in one of these sub-stage states:

- `pending`
- `deferred`
- `completed`
- `failed`
- `disabled`

`disabled` keeps its current meaning: the artifact is intentionally not expected in that flow and must **not** be treated as an error by publish eligibility rules.

This is **not** a new central database-backed state machine. It is a stronger contract over the existing file-based artifacts and summary metadata.

---

## 2. Safety guardrail model

Add a small number of explicit guardrails.

### Guardrail A — Resume detection must prefer in-progress truth over visible final files

Launcher smart resume should detect all of:

- visible final `scored*.json/jsonl`,
- hidden Pass 2 working files (`.scored.jsonl.next`, `.monitor_value.jsonl.next`, etc.),
- hidden checkpoint files,
- deferred postprocess status in summary stats when present.

If hidden in-progress Pass 2 artifacts exist, the launcher must route to **resume scoring**, not fresh scoring.

### Guardrail B — Auto-publish must refuse incomplete Pass 2 publishing

If summary metadata indicates that:

- conversation aggregation is `deferred`, `pending`, or `failed`, or
- scoring dashboard generation is `deferred`, `pending`, or `failed`,

then auto-publish must not publish the run as a fully complete run.

Allowed behavior:

- print a clear explanation,
- direct the user to `complete-postprocess`,
- skip publish safely.

If the scoring dashboard state is `disabled`, that remains a legitimate intentional state and must not be treated as a failure by itself.

### Guardrail C — Publish must fail closed when the run is not publishable

The publish layer itself must verify:

1. the dashboard bundle is internally complete enough to publish, and
2. publish eligibility derived from summary/postprocess state permits publication.

This check applies to both launcher auto-publish and manual `register-run`.

Eligibility must be evaluated against the **actual dashboard set being published**, not by unconditionally requiring Pass 2 state. Concretely:

- if a run is Pass 1-only and only labeling dashboards are being published, absence of Pass 2 summary/postprocess state is **not-applicable** and must not block publish;
- if scoring dashboards are present or requested, Pass 2 postprocess state must be complete or intentionally `disabled` where that state is valid.

If validation fails, publish should abort without mutating current public state.

### Guardrail D — Concurrent publish must be serialized or rejected

The publish layer must use a single-writer critical section for:

- run-id allocation,
- publish directory swap,
- registry mutation.

Recommended design:

- use a lock file rooted under the dashboard service config/web root,
- acquire it for the full publish critical section,
- fail closed (clear error) if lock acquisition times out or cannot be established safely.

The goal is not “best effort concurrent publish”; the goal is **safe publish with no lost updates**.

---

## 3. Workstream design

The implementation should be split into explicit batches so subagents can work safely without overlapping write ownership.

### Batch 1 — P0 correctness and safety guardrails (parallel)

These three workstreams are intentionally non-overlapping and may run in parallel.

#### Workstream 1 — Pass 2 resume and recovery correctness

Scope:

- `src/sft_label/scoring.py` sections that own working/checkpoint/finalize/recovery ordering,
- new focused tests in a dedicated file such as `tests/test_scoring_resume_reliability.py`.

Responsibilities:

1. Fix checkpoint-only resume finalize so the authoritative final `scored.jsonl` always survives.
2. Ensure recovery sweep cannot leave final failed-artifact files stale.
3. Normalize finalize order so recovered artifacts and final visible artifacts agree.
4. Add fault-injection style tests for resume / checkpoint / recovery windows.

Safety invariant:

> After any successful Pass 2 finalize path, `scored.jsonl`, `monitor_value.jsonl`, `failed_value.jsonl`, and `score_failures.jsonl` must describe the same final logical run state.

#### Workstream 2 — Launcher/workflow safety guards

Scope:

- `src/sft_label/launcher.py`
- `src/sft_label/cli.py`
- dedicated launcher/CLI tests

Responsibilities:

1. Extend smart resume detection to include hidden Pass 2 working/checkpoint artifacts.
2. Fix `run --resume ... --score` workflow so it does not silently re-run Pass 2 when safe resume is expected.
3. Prevent auto-publish for deferred/incomplete/failed Pass 2 postprocess states.
4. Ensure messages tell the user exactly what action is needed next.

Safety invariant:

> Launcher must never automatically route a user into a fresh scoring/publish path when the run is actually in an incomplete resumable state.

#### Workstream 3 — Atomic publish and registry safety

Scope:

- `src/sft_label/dashboard_service.py`
- dedicated publish tests

Responsibilities:

1. Replace `delete old dir -> copy new dir in place` with `build temp publish dir -> validate -> atomic swap`.
2. Make dashboard service registry save atomic.
3. Add single-writer locking around publish critical sections.
4. Keep existing URL semantics and run-id reuse behavior.
5. Add tests for republish interruption resistance and publish lock behavior.

Safety invariant:

> A failed or concurrent-conflicting publish must leave the previously published public directory and registry state intact.

### Batch 2 — P1 large-run execution hardening

These workstreams may run in parallel **except where noted**.

#### Workstream 4 — Pass 1 reliability hardening

Scope:

- `src/sft_label/pipeline.py`
- `src/sft_label/http_limits.py`
- `src/sft_label/config.py` only where necessary for safe knobs
- Pass 1/pipeline tests

Responsibilities:

1. Improve chunked Pass 1 sidecar output durability and intermediate-file semantics.
2. Reduce grow-only per-sample stats structures where possible by switching to online summaries.
3. Tighten FD budgeting assumptions for chunked multi-file runs.
4. Preserve existing inline mirrored dataset behavior for dataset-visible files.

Resolved design choice:

- for **non-inline Pass 1 sidecar outputs**, adopt explicit working/final separation where low-risk;
- for **inline mirrored dataset files**, preserve current visible dataset write contract and avoid changing downstream-visible filenames;
- use checkpoint/status hardening to distinguish in-progress vs complete state for inline mode instead of changing mirrored dataset semantics.

Safety invariant:

> Pass 1 interruption must not leave misleading final-looking sidecar artifacts that the workflow mistakes for completed durable outputs, while mirrored dataset semantics remain unchanged.

#### Workstream 5 — Pass 2 large-run memory and rewrite hotspot reduction

Scope:

- new helper modules and/or dedicated internal helpers for streaming/global selection execution,
- narrowly scoped integration hooks in `src/sft_label/scoring.py` after Batch 1 stabilizes scoring finalization seams,
- dedicated large-run scoring tests.

Dependency note:

- this workstream starts **after Batch 1 / Workstream 1** lands, to avoid overlapping edits in `scoring.py`.

Responsibilities:

1. Reduce all-at-once structures in chunked scoring where practical.
2. Rework directory-global selection rewrite to avoid the current most dangerous full-memory/full-rewrite pattern.
3. Make large-run deferred mode actually defer the heaviest optional work.
4. Preserve selection semantics and field availability.

Resolved design choice:

- `selection_score`, `intra_class_rank`, and existing scored fields must still be materialized by the end of Pass 2 for completed runs;
- do **not** defer these fields to postprocess;
- first iteration may defer/gate only heavy **dashboard/explorer/postprocess-adjacent** work, not core scored output semantics.

Safety invariant:

> Large-run “safe mode” must avoid hidden O(N) tail stages that defeat the point of chunked/deferred execution, without changing when core scored fields become available.

#### Workstream 6 — Deferred postprocess and dashboard generation scalability

Scope:

- `src/sft_label/tools/recompute.py`
- `src/sft_label/conversation.py`
- `src/sft_label/tools/visualize_labels.py`
- `src/sft_label/tools/visualize_value.py`
- `src/sft_label/tools/dashboard_scopes.py`
- `src/sft_label/tools/dashboard_explorer.py`
- dedicated recompute/dashboard tests

Responsibilities:

1. Make `complete-postprocess` avoid retaining all conversation batches at once.
2. Add atomic writes where postprocess status or conversation outputs are rewritten.
3. Add large-run safety behavior for explorer asset generation and/or dashboard generation gating.
4. Preserve dashboard semantics while making generation safer.

Resolved design choice:

- in heavy-run mode, prefer **defer/disable explorer asset generation** rather than sampling or altering dashboard semantics;
- dashboards remain publishable only when their required non-explorer artifacts are complete and truthful;
- no sampling-based semantic change is allowed in this minimal repair set.

Safety invariant:

> Deferred postprocess must be a genuinely safer fallback path for large runs, not another large-memory terminal stage.

### Batch 3 — Reliability review sweep

This batch is intentionally review/test heavy.

Responsibilities:

1. Integrated regression sweep across repaired resume/publish/postprocess flows.
2. Focused fault-injection coverage for interruption windows.
3. Documentation updates for large-run operator guidance.

---

## 4. Reliability semantics by stage

### Pass 1 semantics

Pass 1 should continue to support:

- single file,
- directory mode,
- chunked JSONL mode,
- inline mirrored output mode.

Design change:

- chunked sidecar outputs should write to explicit working paths and only expose final sidecar paths after safe completion where feasible;
- mirrored dataset-visible files in inline mode must keep their current visibility and path semantics;
- checkpoint/status metadata must clearly distinguish in-progress from complete state so workflow routing does not confuse partial inline work with durable completion.

### Pass 2 semantics

Pass 2 remains chunked for JSONL and can still use deferred postprocess.

Design change:

- hidden working/checkpoint files are part of the supported state model,
- fast-path finalize must materialize the final authoritative artifact set,
- recovery sweep must feed into final artifact publication deterministically,
- core scored output fields must still be present at Pass 2 completion and must not migrate into deferred-only semantics.

### Postprocess semantics

`summary_stats_scoring.json.postprocess` becomes the primary truth for whether postprocess-dependent publish is allowed.

The state meanings are:

- `completed`: artifact is complete and publish-eligible,
- `pending`: artifact is expected in this flow but not yet done,
- `deferred`: artifact intentionally postponed and not publish-eligible,
- `failed`: artifact attempted but failed and not publish-eligible,
- `disabled`: artifact intentionally not expected and does not block publish by itself.

This is not a user-facing functional change; it is a reliability rule.

### Publish semantics

Publishing remains static-copy based, but with three stronger guarantees:

1. publish does not expose half-copied output,
2. publish is blocked when the scoring dashboard state is known incomplete,
3. only one publish mutates a given service state at a time.

Labeling-only or intentionally dashboard-disabled flows remain legitimate when their state is explicitly `disabled` rather than incomplete.

---

## 5. Testing strategy

Every workstream must ship tests. The test strategy should prioritize failure windows and state consistency over happy-path breadth.

### Required test categories

#### A. Resume / checkpoint tests

Add tests for:

- hidden `.next` / `.checkpoint` artifact detection in launcher smart resume,
- checkpoint-only Pass 2 resume finalize,
- recovery sweep followed by finalize,
- stale vs final artifact precedence.

#### B. Publish safety tests

Add tests for:

- publish temp dir + atomic swap behavior,
- republish preserving existing public state on failure,
- publish-layer blocking of deferred/failed runs,
- publish locking / concurrent publish failure-closed behavior.

#### C. Large-run execution-shape tests

Add tests that assert structural behavior such as:

- no forbidden `json.load` on paths meant to stream,
- deferred mode skips heavy publish/dashboard/explorer actions,
- postprocess avoids retaining all per-file batches in representative cases.

#### D. Artifact consistency tests

For each repaired flow, assert that:

- stats metadata,
- final json/jsonl outputs,
- failed artifact outputs,
- publish eligibility state

all agree after completion.

---

## 6. Rollout strategy

Implement in three reviewable batches.

### Batch 1 — P0 correctness and guardrails

Includes:

- Pass 2 resume/recovery fixes,
- smart resume detection fixes,
- auto-publish guardrails,
- publish-layer eligibility checks,
- atomic publish + publish locking.

This batch should land first because it eliminates the most dangerous silent-wrong-state outcomes.

### Batch 2 — P1 large-run execution hardening

Includes:

- Pass 1 stats / intermediate artifact hardening,
- Pass 2 global rewrite / large-memory hotspot reduction,
- complete-postprocess memory improvements,
- heavy-run explorer/dashboard gating.

### Batch 3 — reliability review sweep

Includes:

- integrated regression suite,
- targeted end-to-end resume/publish tests,
- documentation updates for large-run operational guidance.

---

## 7. Operational guidance after this design lands

After the repairs, the recommended operator model should be:

1. run Pass 1 / Pass 2 as before,
2. if summary indicates deferred postprocess, run `complete-postprocess`,
3. only publish when postprocess status is complete or intentionally disabled,
4. rely on smart resume for interrupted runs because hidden working state is now recognized.

This preserves the existing workflow shape while making the unsafe shortcuts unavailable.

---

## Recommended Implementation Approach

Use **parallel, non-overlapping subagent workstreams by batch** with reviewers and targeted test workers.

Recommended order:

1. Batch 1: Workstream 1 + 2 + 3 in parallel,
2. Batch 2: Workstream 4 + 6 in parallel, then Workstream 5 after Batch 1 scoring seams are stable,
3. Batch 3: one integrated reviewer/test sweep.

This matches the user’s priority:

- fix correctness first,
- then fix stability and scale,
- without changing core functionality.
