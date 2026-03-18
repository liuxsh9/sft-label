# Pass1 Extension Labeling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a schema-driven, multi-extension Pass1 labeling layer that lets users load multiple `prompt + schema` extension configs, persist extension labels alongside core Pass1 outputs, and surface them in stats, dashboards, and interactive launcher flows without breaking current core taxonomy or Pass2 behavior.

**Architecture:** Keep the existing core 9-dimension Pass1 pipeline unchanged, then add a parallel extension layer driven by external spec files. Extension outputs live in `label_extensions`, have their own validation/stats/dashboard rendering path, and are explicitly excluded from core `labels`, core rarity/scoring, and default review/export surfaces unless opt-in.

**Tech Stack:** Python 3.9+, pytest, existing `sft_label` pipeline/inline/dashboard modules, YAML/JSON config loading, static HTML dashboard runtime, interactive launcher in `src/sft_label/launcher.py`.

---

## File map and ownership boundaries

### New modules
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions.py`
  - Runtime orchestration helpers for loading specs, matching triggers, building prompts, validating LLM responses, and executing enabled extensions for one sample.
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_schema.py`
  - Spec schema validation, field-type normalization, config parsing, hash/version helpers.
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_stats.py`
  - Extension-specific aggregate statistics for sample-level and conversation-level views.

### Core runtime integration
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/pipeline.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/config.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/cli.py`

### Inline persistence / maintenance
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_pass1.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_scoring.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/recompute.py`

### Dashboard / stats / explorer
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_aggregation.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_scopes.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard.js`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_explorer.py`

### Launcher / docs / templates
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/launcher.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/README.md`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/README.zh-CN.md`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/docs/guides/how-sft-label-works.md`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/docs/guides/pass1-extension-labeling.md`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/fixtures/extensions/ui_fine_labels.yaml`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/fixtures/extensions/mobile_fine_labels.yaml`

### Test suites
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_cli_progress.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_pass1_adaptive_runtime.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_recompute.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_scoring.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_e2e_mock.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_launcher.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_export_review.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_explorer.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_scopes.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_pass1.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_scoring.py`

---

## Chunk 1: Lock the extension contract with tests and fixture specs

### Task 1: Add failing tests for extension spec parsing, hashing, and trigger matching

**Files:**
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/fixtures/extensions/ui_fine_labels.yaml`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/fixtures/extensions/mobile_fine_labels.yaml`

- [ ] **Step 1: Write fixture spec files for two extension profiles**

Create one UI-focused spec and one mobile-focused spec, each with:
- distinct `id`
- `spec_version`
- `prompt`
- `trigger`
- `schema`
- dashboard metadata

- [ ] **Step 2: Write failing tests for valid spec parsing and normalization**

Cover:
- YAML loading
- required field presence
- `enum` vs `multi_enum`
- duplicate option rejection
- duplicate extension id rejection when loading multiple specs

- [ ] **Step 3: Write failing tests for `spec_hash` stability**

Cover:
- same file content => same hash
- prompt/schema edits => different hash
- cosmetic ordering normalization where intended

- [ ] **Step 4: Write failing tests for trigger matching against core labels**

Cover:
- `domain_any_of`
- `domain_all_of`
- `language_any_of`
- `intent_any_of`
- `task_any_of`
- `context_any_of`
- `difficulty_any_of`
- unmatched sample => skipped
- multiple matched extensions in one sample

- [ ] **Step 5: Run the focused schema tests to verify they fail**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py -q`

Expected: FAIL because the extension schema/runtime helpers do not exist yet.

### Task 2: Add failing runtime tests for extension execution semantics

**Files:**
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py`

- [ ] **Step 1: Write failing tests for one-sample multi-extension execution**

Cover:
- two extension specs enabled
- one sample matches one extension
- another sample matches both
- outputs stored under `label_extensions.<id>`

- [ ] **Step 2: Write failing tests for failure isolation**

Cover:
- core labels remain usable when extension LLM call fails
- extension `status` becomes `failed` or `invalid`
- sample is not marked as partial core labels

- [ ] **Step 3: Write failing tests for extension output validation**

Cover:
- bad enum value => invalid
- scalar returned for `multi_enum` => normalized or rejected according to plan
- per-field confidence retention

- [ ] **Step 4: Run the focused runtime tests to verify they fail**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py -q`

Expected: FAIL because `label_extensions` runtime integration does not exist yet.

---

## Chunk 2: Build schema/runtime helpers and integrate them into Pass1

### Task 3: Implement extension spec loading, validation, and hashing helpers

**Files:**
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_schema.py`
- Test: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py`

- [ ] **Step 1: Write the minimal implementation for extension spec dataclasses and loaders**

Implement:
- YAML/JSON file loading
- normalized in-memory spec objects
- field type validation
- `spec_hash` generation

- [ ] **Step 2: Run the schema tests to verify partial progress**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py -q`

Expected: Some tests still fail if trigger/runtime pieces are not implemented yet.

- [ ] **Step 3: Implement trigger matching helpers**

Support the trigger keys defined in the design doc and return deterministic `matched` / `skipped` results.

- [ ] **Step 4: Run the schema tests again to verify they pass**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the schema helper work**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_schema.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/fixtures/extensions/ui_fine_labels.yaml \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/fixtures/extensions/mobile_fine_labels.yaml

git commit -m "feat: add extension spec schema helpers"
```

### Task 4: Implement extension prompt/runtime execution helpers

**Files:**
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions.py`
- Test: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py`

- [ ] **Step 1: Write failing subtests for prompt-builder shape if still missing**

Add assertions for:
- conversation inclusion
- core labels inclusion
- per-extension schema rendering
- spec prompt inclusion

- [ ] **Step 2: Implement prompt builder and response validator**

Implement:
- request message builder
- enum/multi_enum response validation
- confidence normalization
- `status=success/skipped/failed/invalid`

- [ ] **Step 3: Implement multi-extension per-sample execution entrypoint**

Given enabled specs + merged core labels, return deterministic `label_extensions` payload for one sample.

- [ ] **Step 4: Run the runtime tests to verify they pass**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the runtime helper work**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py

git commit -m "feat: add extension labeling runtime helpers"
```

### Task 5: Integrate extension execution into the Pass1 pipeline and CLI config surface

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/pipeline.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/config.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/cli.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/scoring.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_cli_progress.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_pass1_adaptive_runtime.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_scoring.py`

- [ ] **Step 1: Add config fields for enabled extension spec paths / loaded specs**

Keep them additive and optional so old callers remain valid.

- [ ] **Step 2: Add CLI flags for repeatable `--label-extension`**

Add parser support and wire the loaded spec list into runtime config.

- [ ] **Step 3: Update `label_one()` to run extensions after core labels settle**

Ensure:
- core labels unchanged
- extension failures non-fatal
- extension monitor data captured
- shared runtime / concurrency limits reused safely

- [ ] **Step 4: Update workload/progress estimates for optional extra calls**

Adjust pass1 estimated calls and progress tests so enabled extensions increase estimated work without breaking no-extension behavior.

- [ ] **Step 5: Add Pass2 isolation regression coverage**

Add tests proving extension labels are excluded from Pass2 prompt inputs, rarity/scoring behavior, and no-extension baseline outputs.

- [ ] **Step 6: Run focused pipeline/CLI/scoring tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_cli_progress.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_pass1_adaptive_runtime.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_scoring.py -q -k "extension or prompt or rarity or scoring"`

Expected: PASS.

- [ ] **Step 7: Commit the pipeline integration**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/pipeline.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/config.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/cli.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_cli_progress.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_pass1_adaptive_runtime.py

git commit -m "feat: integrate extension labeling into pass1"
```

---

## Chunk 3: Persist extension labels in inline mode and maintenance flows

### Task 6: Extend inline turn/data_label persistence to keep `label_extensions`

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_pass1.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_scoring.py`
- Test: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py`
- Test: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_pass1.py`
- Test: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_labels.py`
- Test: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_scoring.py`

- [ ] **Step 1: Add failing tests for inline turn-record roundtrip**

Cover:
- `turn_record.label_extensions` preserved after compaction
- cached `labeled.json` rebuilt from inline rows retains extensions
- old rows without extensions still load cleanly

- [ ] **Step 2: Update sample artifact building to attach `label_extensions`**

Keep `labels` and `label_extensions` separate in sample payloads.

- [ ] **Step 3: Update turn record compaction helpers to preserve extension payloads**

Add explicit allow-list handling rather than relying on accidental passthrough.

- [ ] **Step 4: Record extension spec fingerprint in inline meta**

Store a stable additive field under `data_label.meta` without changing old fields.

- [ ] **Step 5: Run the focused inline/runtime tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_pass1.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_labels.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_scoring.py -q -k inline`

Expected: PASS.

- [ ] **Step 6: Commit the inline persistence changes**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_pass1.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_scoring.py

git commit -m "feat: persist extension labels in inline runs"
```

### Task 7: Keep recompute/regenerate compatible and avoid unnecessary Pass2 invalidation

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/recompute.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_pass1.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_recompute.py`

- [ ] **Step 1: Add failing tests for recompute preserving extension aggregates**

Cover:
- recompute from labeled/scored data retains extension stats
- regenerate-dashboard can render extension sections from recomputed stats

- [ ] **Step 2: Add a targeted test proving extension-only updates do not clear Pass2 by default**

Make sure the default v1 behavior leaves scoring intact when core labels are unchanged.

- [ ] **Step 3: Implement recompute aggregation for extension stats**

Keep core stats unchanged while adding parallel `extension_stats`.

- [ ] **Step 4: Implement conservative invalidation logic in inline merge flow**

Only clear Pass2 when core labels change, not just because extension labels changed.

- [ ] **Step 5: Run focused recompute tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_recompute.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the maintenance-flow compatibility work**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/recompute.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_pass1.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_recompute.py

git commit -m "feat: preserve extension labels in maintenance flows"
```

---

## Chunk 4: Add extension stats, dashboard sections, and explorer support

### Task 8: Implement extension statistics builders for sample and conversation views

**Files:**
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_stats.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/pipeline.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_scopes.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_scopes.py`

- [ ] **Step 1: Add failing tests for `extension_stats` shape in pass1 stats**

Cover:
- per-extension `matched/success/failed`
- per-field distributions
- confidence summaries
- no pollution of core `tag_distributions`

- [ ] **Step 2: Implement extension stats builder from sample payloads**

Keep it independent from core taxonomy builders.

- [ ] **Step 3: Thread extension stats through scope merges**

Ensure folder/global dashboard scopes aggregate extension sections correctly.

- [ ] **Step 4: Run focused stats tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_scopes.py -q -k extension_stats`

Expected: PASS.

- [ ] **Step 5: Commit the stats builder work**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_stats.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/pipeline.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_scopes.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py

git commit -m "feat: add extension labeling stats"
```

### Task 9: Render extension sections in Pass1 dashboards and explorer payloads

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_aggregation.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard.js`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_explorer.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_explorer.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_e2e_mock.py`

- [ ] **Step 1: Add failing dashboard tests for extension sections**

Cover:
- dashboard payload contains `extension_sections`
- no-extension runs hide the section cleanly
- multi-extension runs render multiple sections in a stable order

- [ ] **Step 2: Update scope detail payload builders to emit extension sections**

Keep them separate from core `distributions`, `coverage`, and `cross_matrix`.

- [ ] **Step 3: Update dashboard JS to render extension cards/tables**

Render:
- overview cards
- field distribution tables
- confidence summary where present

- [ ] **Step 4: Update explorer flattening to expose extension tags without mutating core label chips**

A minimal v1 behavior is enough: extension tags searchable/filterable, but clearly separated from core labels.

- [ ] **Step 5: Run focused dashboard/e2e tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_explorer.py /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_e2e_mock.py -q -k extension`

Expected: PASS.

- [ ] **Step 6: Commit the dashboard work**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_aggregation.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/visualize_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard.js \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/dashboard_explorer.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_e2e_mock.py

git commit -m "feat: render extension sections in pass1 dashboards"
```

---

## Chunk 5: Add launcher UX, docs, and regression coverage

### Task 10: Add interactive launcher support for multiple extension specs

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/launcher.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_launcher.py`

- [ ] **Step 1: Add failing launcher tests for extension prompts and generated argv**

Cover:
- prompt to enable extension labeling
- collection of multiple extension paths
- dry-run summary includes each `--label-extension`

- [ ] **Step 2: Implement launcher prompts and argv wiring**

Make the flow user-friendly and optional.

- [ ] **Step 3: Run focused launcher tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_launcher.py -q`

Expected: PASS.

- [ ] **Step 4: Commit the launcher UX work**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/launcher.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_launcher.py

git commit -m "feat: add launcher support for extension labeling"
```

### Task 11: Document the new feature and add opt-in export coverage

**Files:**
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/README.md`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/README.zh-CN.md`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/docs/guides/how-sft-label-works.md`
- Create: `/Users/lxs/.codex/worktrees/e8fb/sft-label/docs/guides/pass1-extension-labeling.md`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/export_review.py`
- Modify: `/Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_export_review.py`

- [ ] **Step 1: Add failing export-review tests for opt-in extension fields**

Keep default CSV unchanged, but add an explicit opt-in mode or sidecar path for extension columns.

- [ ] **Step 2: Implement opt-in extension review export support**

Do not break the default header expected by existing users.

- [ ] **Step 3: Update docs with examples for multiple extension specs**

Document:
- spec file format
- CLI usage
- launcher path
- dashboard behavior
- compatibility caveats

- [ ] **Step 4: Run focused export/doc-adjacent tests**

Run: `uv run pytest /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_export_review.py -q`

Expected: PASS.

- [ ] **Step 5: Commit docs and export updates**

```bash
git add /Users/lxs/.codex/worktrees/e8fb/sft-label/README.md \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/README.zh-CN.md \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/docs/guides/how-sft-label-works.md \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/docs/guides/pass1-extension-labeling.md \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/tools/export_review.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_export_review.py

git commit -m "docs: add pass1 extension labeling guide"
```

---

## Chunk 6: Final verification

### Task 12: Run the targeted verification suite and summarize remaining gaps

**Files:**
- No new files unless failures require follow-up edits

- [ ] **Step 1: Run the extension-focused test suite**

Run: `uv run pytest \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_schema.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_label_extensions_runtime.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_explorer.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_scopes.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_recompute.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_launcher.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_export_review.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_scoring.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_e2e_mock.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_pass1.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_scoring.py -q`

Expected: PASS.

- [ ] **Step 2: Run a broader affected-area regression suite**

Run: `uv run pytest \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_cli_progress.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_pass1_adaptive_runtime.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_visualize_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_explorer.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_dashboard_scopes.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_recompute.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_launcher.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_export_review.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_scoring.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_e2e_mock.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_pass1.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_labels.py \
  /Users/lxs/.codex/worktrees/e8fb/sft-label/tests/test_inline_scoring.py -q`

Expected: PASS.

- [ ] **Step 3: Run validation and one representative dry-run command**

Run: `uv run sft-label validate`

Expected: PASS.

Run: `uv run sft-label start --dry-run`

Expected: dry-run completes and can surface extension-labeling prompts when selected.

- [ ] **Step 4: Commit final follow-up fixes if needed**

```bash
git add -A

git commit -m "test: finish pass1 extension labeling coverage"
```

---

## Suggested subagent execution order

### Wave 0 (must happen first)
1. Task 1 — schema fixtures + contract tests
2. Task 2 — runtime behavior tests

### Parallel wave 1 (after tests/fixtures exist)
1. Task 3 — schema helpers
2. Task 10 — launcher tests + prompts skeleton

### Parallel wave 2 (after Task 3 defines schema interfaces)
1. Task 4 — runtime helpers
2. Task 5 — pipeline + CLI integration

### Parallel wave 3 (after runtime contract lands)
1. Task 6 — inline persistence
2. Task 8 — extension stats builder

### Parallel wave 4 (after stats/persistence contracts land)
1. Task 7 — recompute + invalidation rules
2. Task 9 — dashboard + explorer rendering
3. Task 11 — docs + export coverage

### Final wave
1. Task 12 — verification + cleanup

This ordering keeps write scopes mostly disjoint and suits subagent-driven implementation well.

