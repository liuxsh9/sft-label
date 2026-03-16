# Dashboard Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert dashboard generation from giant single-file HTML into lightweight per-run HTML shells backed by run-local data directories and shared static assets, while preserving drill-down behavior and adding regression coverage for large directory runs.

**Architecture:** Split dashboard output into three layers: shared runtime assets (`dashboard.js` / `dashboard.css`), run-local bootstrap HTML, and run-local data bundles (`manifest.json`, scope detail JSON, explorer shards). Refactor Python export code to emit summary/detail payloads explicitly instead of serializing the full in-memory dashboard structure into the HTML template.

**Tech Stack:** Python 3.9+, pytest, JSON/HTML static asset generation, existing dashboard runtime in `src/sft_label/tools/`.

---

## Chunk 1: Lock in the new output contract with tests

### Task 1: Add failing tests for lightweight bootstrap HTML and sidecar data layout

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_dashboard_template.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_scoring.py`

- [ ] **Step 1: Write failing template/bootstrap tests**

Add assertions covering:
- HTML includes bootstrap config instead of `const DATA = ...`
- HTML references shared static assets
- HTML no longer embeds full scope payloads

- [ ] **Step 2: Run the focused template tests to verify they fail**

Run: `uv run pytest tests/test_dashboard_template.py -q`

Expected: FAIL because current template still inlines the full payload.

- [ ] **Step 3: Write failing generation tests for `.data/` output**

Add assertions covering:
- `generate_dashboard()` creates `<name>.data/manifest.json`
- scope detail JSON exists under `.data/scopes/`
- explorer shards live under `.data/explorer/`
- HTML contains static asset URL/bootstrap metadata rather than the full data blob

- [ ] **Step 4: Run focused generation tests to verify they fail**

Run: `uv run pytest tests/test_visualize_labels.py -q`

Expected: FAIL because generation currently emits monolithic HTML and `.assets/` only.

- [ ] **Step 5: Extend scoring-side tests with the same contract**

Add assertions covering scoring dashboard HTML/data layout and that scoring dashboards still render as valid HTML.

- [ ] **Step 6: Run focused scoring tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -q -k dashboard`

Expected: FAIL because scoring dashboards still use the old inline payload format.

## Chunk 2: Introduce shared asset/bootstrap rendering and explicit export payload builders

### Task 2: Refactor dashboard rendering to emit shared-asset bootstrap HTML

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/dashboard_template.py`
- Create: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/dashboard_runtime.js`
- Create: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/dashboard_runtime.css`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/artifacts.py`

- [ ] **Step 1: Extract the in-template JS/CSS runtime into shared asset source files**

Move the existing script/style bodies into dedicated asset files or Python-managed embedded resources so they can be written once and referenced by URL.

- [ ] **Step 2: Add helpers for dashboard bootstrap metadata and static asset paths**

Introduce helpers for:
- data directory naming (`dashboard_labeling.data`, `dashboard_scoring.data`)
- optional `static_base_url`
- relative bootstrap config

- [ ] **Step 3: Update `render_dashboard_html()` to accept bootstrap config instead of full payload**

The generated HTML should contain:
- page shell markup
- a serialized bootstrap config object
- `<link>` / `<script>` tags pointing at shared runtime assets

- [ ] **Step 4: Run the focused template tests to make them pass**

Run: `uv run pytest tests/test_dashboard_template.py -q`

Expected: PASS.

### Task 3: Add explicit manifest/scope-detail export builders and remove known payload bloat

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/dashboard_aggregation.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/dashboard_scopes.py`
- Add tests to: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_visualize_labels.py`

- [ ] **Step 1: Add a lightweight manifest builder**

Emit only scope tree metadata, scope summaries, dashboard title/subtitle, and explorer availability flags.

- [ ] **Step 2: Add a scope-detail builder for pass1/pass2/conversation payloads**

Ensure detail payloads keep `modes` but avoid duplicating top-level sample-mode content.

- [ ] **Step 3: Remove `conf_matrix` from exported dashboard payloads**

Keep any internal computation if needed, but do not write it into manifest/detail JSON.

- [ ] **Step 4: Apply payload trimming rules**

Include:
- limiting `unmapped_details.examples`
- rounding long floats
- keeping explorer metadata out of manifest/detail except for file references

- [ ] **Step 5: Run the focused labeling tests to verify builder behavior**

Run: `uv run pytest tests/test_visualize_labels.py -q`

Expected: PASS for the updated dashboard contract tests.

## Chunk 3: Emit `.data/` bundles and re-home explorer assets

### Task 4: Update labeling dashboard generation to write manifest, scopes, and explorer bundles

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/dashboard_explorer.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_visualize_labels.py`

- [ ] **Step 1: Change labeling generation to write `.data/manifest.json` and `scopes/*.json`**

Bootstrap HTML should now point to the data directory instead of embedding scope payloads.

- [ ] **Step 2: Move explorer output under `.data/explorer/`**

Preserve preview/detail shard behavior, but attach metadata via manifest/detail references rather than inline payload.

- [ ] **Step 3: Preserve existing per-run dashboard location conventions**

`dashboards/dashboard_labeling.html` should remain the entry point.

- [ ] **Step 4: Run focused labeling tests to verify the new output layout**

Run: `uv run pytest tests/test_visualize_labels.py -q`

Expected: PASS.

### Task 5: Update scoring dashboard generation to the same layout

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/visualize_value.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_scoring.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_recompute.py`

- [ ] **Step 1: Change scoring generation to write bootstrap HTML plus `.data/` bundle**

Match the labeling layout and preserve scoring-specific data/detail outputs.

- [ ] **Step 2: Update recompute/regenerate tests for the new sidecar directories**

Ensure regenerated dashboards still land under `meta_label_data/dashboards/` with the new `.data/` layout.

- [ ] **Step 3: Run focused scoring/recompute tests to verify the change**

Run: `uv run pytest tests/test_scoring.py -q -k dashboard tests/test_recompute.py -q -k dashboard`

Expected: PASS.

## Chunk 4: Add runtime asset management and end-to-end regression coverage

### Task 6: Add shared asset publication helpers and verify multiple-run compatibility

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/tools/visualize_value.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/src/sft_label/cli.py` (if a static path/config hook is required)
- Add tests to: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_e2e_mock.py`

- [ ] **Step 1: Add a configurable shared static asset base URL**

Default to a sensible relative/shared path and allow explicit override for server deployment.

- [ ] **Step 2: Ensure bootstrap HTML writes stable URLs for shared runtime assets**

Both labeling and scoring dashboards should point at the same versioned runtime asset locations.

- [ ] **Step 3: Add a regression test simulating multiple runs sharing the same runtime asset URL**

Validate that two generated dashboards keep separate `.data/` directories while using identical static asset references.

- [ ] **Step 4: Run the focused e2e mock tests**

Run: `uv run pytest tests/test_e2e_mock.py -q -k dashboard`

Expected: PASS.

### Task 7: Add directory-input e2e and payload-size regression checks

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_scoring.py`
- Optionally add: `/Users/lxs/.codex/worktrees/a9e4/sft-label/tests/test_dashboard_payloads.py`

- [ ] **Step 1: Add assertions that generated HTML no longer contains a giant inlined `DATA` blob**

Check for absence of the old inline structure and presence of fetch/bootstrap markers.

- [ ] **Step 2: Add payload regression assertions**

Cover:
- no exported `conf_matrix`
- manifest exists and is smaller than full old payload shape for fixtures
- explorer detail exists only in sidecar files

- [ ] **Step 3: Run the focused regression tests**

Run: `uv run pytest tests/test_visualize_labels.py tests/test_scoring.py -q -k "dashboard or payload"`

Expected: PASS.

## Chunk 5: Final verification

### Task 8: Run the targeted verification suite and summarize remaining gaps

**Files:**
- No additional code changes unless failures require them

- [ ] **Step 1: Run the dashboard-focused suite**

Run: `uv run pytest tests/test_dashboard_template.py tests/test_visualize_labels.py tests/test_scoring.py tests/test_recompute.py tests/test_e2e_mock.py -q -k dashboard`

Expected: PASS.

- [ ] **Step 2: Run a broader smoke test for affected modules**

Run: `uv run pytest tests/test_dashboard_template.py tests/test_visualize_labels.py tests/test_scoring.py tests/test_recompute.py tests/test_e2e_mock.py tests/test_dashboard_explorer.py -q`

Expected: PASS.

- [ ] **Step 3: Commit implementation changes**

```bash
git add src/sft_label/artifacts.py \
  src/sft_label/tools/dashboard_template.py \
  src/sft_label/tools/dashboard_aggregation.py \
  src/sft_label/tools/dashboard_explorer.py \
  src/sft_label/tools/visualize_labels.py \
  src/sft_label/tools/visualize_value.py \
  src/sft_label/cli.py \
  tests/test_dashboard_template.py \
  tests/test_visualize_labels.py \
  tests/test_scoring.py \
  tests/test_recompute.py \
  tests/test_e2e_mock.py

git commit -m "feat: slim dashboard outputs"
```
