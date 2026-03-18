# Dashboard Extension UX Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Pass 1 extension labels easier to notice and easier to use for drill-down analysis in the dashboard.

**Architecture:** Keep the existing extension stats contract and dashboard rendering path, but improve presentation in `dashboard.js` and lightly enrich payload shaping where needed. The first iteration focuses on three behaviors only: move the extension section earlier, add extension summary cards, and make extension field values clickable so they prefill and run the Sample Explorer.

**Tech Stack:** Python, existing dashboard aggregation helpers, vanilla JS dashboard renderer, pytest.

---

## Chunk 1: Tests and payload expectations

### Task 1: Add failing dashboard UX tests

**Files:**
- Modify: `tests/test_visualize_labels.py`
- Inspect: `src/sft_label/tools/dashboard_aggregation.py`
- Inspect: `src/sft_label/tools/dashboard.js`

- [ ] **Step 1: Add a failing test for extension summary payload shape**

Add assertions that `compute_viz_data(...)` exposes enough extension metadata for summary cards, including per-extension `total`, `matched`, `status_counts`, and field counts.

- [ ] **Step 2: Add a failing HTML test for extension summary/drill-down affordances**

Render a dashboard HTML fixture and assert the HTML contains:
- an extension summary label/card marker,
- `data-explorer-patch` hooks for extension field values,
- extension section appearing before Sample Explorer.

- [ ] **Step 3: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_visualize_labels.py -q -k 'extension'`
Expected: FAIL because the new summary/drill-down markers are not rendered yet.

## Chunk 2: Dashboard rendering changes

### Task 2: Implement extension summary + clickable drill-down

**Files:**
- Modify: `src/sft_label/tools/dashboard.js`
- Modify: `src/sft_label/tools/dashboard_aggregation.py` (only if payload shaping is needed)
- Verify: `tests/test_visualize_labels.py`

- [ ] **Step 1: Add summary-card rendering for each extension**

In `renderExtensions(...)`, show a compact summary row or cards with:
- total
- matched rate
- status breakdown
- field count

- [ ] **Step 2: Make extension field values clickable**

For each extension field distribution row, render the value as a button with `data-explorer-patch` so clicking it injects a query into Sample Explorer. Use a conservative first version:
- patch `tagQuery` with `ext:<spec_id>:<field>:<value>` for visibility
- patch `textQuery` with the raw value so the explorer query visibly changes
- if a more specific dimension query is available, include it without breaking current explorer behavior

- [ ] **Step 3: Move extension section earlier in the Pass 1 layout**

Render Extension Labels before Pool Coverage / Sample Explorer so extension results are visible without scrolling through the entire core taxonomy section.

- [ ] **Step 4: Run focused tests and iterate until green**

Run: `uv run pytest tests/test_visualize_labels.py -q -k 'extension'`
Expected: PASS.

## Chunk 3: Smoke verification

### Task 3: Verify the real dashboard output path

**Files:**
- Verify only: `/private/tmp/sft-pass1-ext-vnkZfX/run_mock_ui_ext/meta_label_data/files/fixture_subset_ui/dashboards/dashboard_labeling.html`

- [ ] **Step 1: Regenerate the dashboard if needed**

Run the existing mock validation script or regenerate-dashboard path so the dashboard reflects the new renderer.

- [ ] **Step 2: Open the dashboard and confirm UX improvements**

Check that:
- extension section is visible higher on the page,
- extension summary is readable at a glance,
- clicking extension values updates/runs the explorer.

- [ ] **Step 3: Run targeted verification commands**

Run:
- `uv run pytest tests/test_visualize_labels.py -q`
- `uv run ruff check src tests`

Expected: PASS.
