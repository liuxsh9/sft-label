# Pass 1 extension labeling

Pass 1 extension labeling lets you attach **additional, domain-specific tags** on top of the core 9-dimension taxonomy. Extensions run **after** the core Pass 1 calls and never change the core taxonomy or Pass 2 semantics.

If you want a quicker presentation-style walkthrough before reading this reference, start with [Pass 1 extension labeling intro](../pass1-extension-labeling-intro.md).

## When to use it

Use extension labeling when you need extra metadata for a specific subset of samples (e.g., UI-heavy data, Web UI mix analysis, mobile tasks, or domain-specific heuristics).

## Enabling extension labeling

Use `--label-extension` once per extension spec file:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension extensions/my_web_extension.yaml \
  --label-extension extensions/my_mobile_extension.yaml
```

The flag is repeatable, and **each spec must have a unique `id`**. Specs that do not match a sample’s trigger are recorded as `status=skipped` and `matched=false`.

If you use the interactive launcher (`uv run sft-label start`), extension selection now supports:

- auto-loading the repo-root `extensions/` directory,
- a single YAML path,
- or a directory path (lists `.yaml` / `.yml` files and lets you select **one or more**).

The repo-root `extensions/` directory now ships with `ui_web_analysis_v1.example.yaml` as a starter example. You can keep your own specs there and the launcher will pick up the latest files automatically. The launcher also prints a strong cost reminder: **each enabled extension adds extra extension-labeling calls for matched samples**, so do not turn on highly domain-specific extensions across the full dataset unless you first filter or sample it down.

## Extension spec files

An extension spec is a YAML file with:

- `id` and `spec_version` (required)
- `prompt` (required)
- `schema` (required)
- optional `trigger` rules that gate when the extension runs
- optional `output` settings (confidence and unmapped handling)

Example:

```yaml
id: ui_fine_labels
spec_version: v1
display_name: UI Fine Labels
trigger:
  domain_any_of: [web-frontend]
prompt: |
  You are a UI labeling assistant. Return JSON only.
schema:
  component_type:
    type: multi_enum
    options: [form, table, modal, chart]
  visual_complexity:
    type: enum
    options: [low, medium, high]
output:
  include_confidence: true
  allow_unmapped: true
```

Starter locations:

- Default launcher starter in repo-root `extensions/`: `extensions/ui_web_analysis_v1.example.yaml`
- Reference example specs for direct CLI / copying: `docs/examples/extensions/...`

Reference example specs:

- Minimal starter: `docs/examples/extensions/ui_fine_labels_minimal_v1.yaml`
- Richer UI variant: `docs/examples/extensions/ui_fine_labels_v1.yaml`
- Web UI analysis / mix-optimization example: `docs/examples/extensions/ui_web_analysis_v1.yaml`

Recommended rollout:

1. Start with the minimal spec to verify triggers + dashboard surfacing.
2. Add the richer spec once field coverage and consistency look stable.
3. Keep multiple specs side-by-side with repeated `--label-extension` flags.

### Schema field types

- `enum`: single select string
- `multi_enum`: array of strings

Return field ids exactly as defined in `schema`. If confidence is enabled, return a `confidence` object keyed by field id.

## Multiple extension configs

You can pass multiple `--label-extension` specs in a single run. Each spec is evaluated independently against the core labels:

- If the trigger matches, the extension runs and records labels.
- If it does not match, the extension is recorded as `skipped`.
- Failures and invalid outputs are isolated to the specific spec.

### Extension diagnostics and follow-up guidance

Before the first extension call, the CLI prints a compact preflight summary that lists every registered spec, whether it has a trigger, and the prompt/schema warnings that matter before you spend tokens. That block is your first checkpoint for new extensions: confirm the spec looks focused, note any triggerless or oversized prompts, and split or tighten the schema before scaling up.

After the run completes, a follow-up diagnostics section highlights per-spec match counts, failed/invalid statuses, and unmapped rows. Use that together with the dashboard drawer and `export-review --include-extensions` to inspect low-confidence fields or suspicious value distributions before widening the rollout.

Both diagnostics sections reference the same metadata that ends up in `stats_labeling.json` so you can quickly trace a warning back to the dashboard or the `runtime_events` log for deeper investigation.

## Inline (mirrored JSONL) behavior

When you run in mirrored inline mode, extension payloads are embedded under:

```
extra_info.unique_info.data_label.turns[].label_extensions
```

The spec metadata is captured in:

```
extra_info.unique_info.data_label.meta.extension_specs
```

Recompute/regenerate workflows **reuse these embedded payloads** and do not re-run the LLM.

## Dashboard visibility

Extension aggregates are stored in `stats_labeling.json` (and summary stats) under:

```
extension_stats.specs.<spec_id>
```

`regenerate-dashboard` uses the existing stats files, so extension counts remain available without re-labeling.

## Export review CSV (optional)

`export-review` can include extension columns when you opt in:

```bash
uv run sft-label export-review --input run_dir --output review.csv --include-extensions
```

This keeps the default CSV format unchanged while letting you include extension fields when needed. When enabled, the export includes per-spec status/matched metadata, spec version/hash when present, flattened label/confidence columns, and a normalized unmapped summary column for each extension.

If Pass 2 later runs with `--extension-rarity-mode preview|bonus_only`, the same opt-in export can populate additive extension-rarity columns such as:

- `extension_rarity_preview_score`
- `extension_rarity_preview_confidence`
- `extension_rarity_preview_matched_specs`
- `extension_rarity_preview_baseline_source`

Interpretation:

- `preview` is diagnostic-only and does **not** change legacy `rarity / value_score / selection_score`;
- `preview` populates only the `extension_rarity_preview_*` columns above;
- `bonus_only` can additionally write `rarity_v2_score`, `value_score_v2`, and `selection_score_v2`;
- local-only extension baselines remain diagnostic and do not produce a non-zero bonus.
- the dashboard shows these additive extension-rarity/V2 metrics in sample mode only for now, so conversation mode keeps the legacy aggregation semantics.

## Recommended user-side config patterns

### 1) Minimal / safe first rollout

Use a small reference example directly (or copy it into your own `extensions/` folder first):

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension docs/examples/extensions/ui_fine_labels_minimal_v1.yaml
```

This is the best first step when you want to validate:

- trigger scope,
- prompt/schema contract quality,
- extension visibility in dashboard + review CSV.

### 2) Richer UI analytics rollout

Use the richer example when you want more slicing dimensions:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension docs/examples/extensions/ui_fine_labels_v1.yaml
```

### 3) Web UI dataset analysis / distribution optimization

Use the Web-only analysis starter when you want to understand **dataset mix**, not answer quality:

```bash
uv run sft-label run \
  --input filtered_web_ui_data.jsonl \
  --label-extension extensions/ui_web_analysis_v1.example.yaml
```

This example is useful when you want to answer questions like:

- Are we over-concentrated in CRUD/forms and under-covered in dashboards, builders, or admin workflows?
- Do we mostly have low-state UI rows, while missing multi-source async/state-coordination samples?
- How much of the Web UI subset actually exercises design-system, responsive, accessibility, or dense-data constraints?
- Are we overly concentrated in one framework ecosystem?

This example intentionally keeps only `domain_any_of: [web-frontend]` active. The other trigger keys can stay in the YAML as empty placeholders so users can see which routing dimensions exist without narrowing recall by default.

Interpret it this way:

- **Analysis-oriented default:** only the broad domain trigger is active, so more Web UI rows are included in the audit.
- **Stricter trigger mode:** if you later fill in `language_any_of`, `intent_any_of`, `task_any_of`, `context_any_of`, or `difficulty_any_of`, you are explicitly saying that you trust core Pass 1 enough to use those labels as hard extension-routing gates.

Trigger semantics for this example:

- empty list `[]` = this trigger dimension does **not** participate in filtering
- non-empty `*_any_of` = that dimension becomes a **hard precondition**
- values inside one `*_any_of` list are matched with **OR**
- different trigger dimensions combine with **AND**
- if the trigger does not match, the extension is recorded as `status=skipped` and `matched=false`

Keep this example **Web-only**. If you also need mobile analysis, create a separate mobile extension so the prompt, trigger, stats, and cost profile remain interpretable.

This example also assumes the input is already mostly a **Web UI subset**. If your data still contains routing/config/build-only frontend tasks or non-visual web work, filter first or tighten the trigger before you trust the resulting distributions.

#### How to turn this example into your own extension

1. Copy `extensions/ui_web_analysis_v1.example.yaml` to a new file with a new `id`.
2. Keep the schema to roughly **3-5 fields** unless you have a strong reason to expand it.
3. Start with only the broadest trigger active (usually `domain_any_of`) so you can inspect recall first.
4. Run a small filtered sample, then inspect `matched/skipped`, dashboard distributions, and `export-review --include-extensions`.
5. Only after that, tighten triggers or split into a second extension if you need stricter routing or a different domain.

What to change first:

- **must change:** `id`, `display_name`, and the schema/prompt so they match your own analysis target
- **usually keep first:** `output.include_confidence`, `output.allow_unmapped`
- **change later if needed:** `dashboard.group`, `dashboard.priority`, and stricter trigger dimensions

Example output shape:

```json
{
  "ui_surface_type": "data-dense-dashboard",
  "interaction_pattern": "search-filter-explore",
  "state_data_complexity": "remote-data-binding",
  "ui_constraint_focus": ["responsive-layout", "dense-data-layout"],
  "frontend_stack_shape": "react-next-ecosystem",
  "confidence": {
    "ui_surface_type": 0.92,
    "interaction_pattern": 0.88
  }
}
```

### 4) Multiple extension configs coexisting

You can enable several domain-specific extensions at once:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension extensions/my_web_extension.yaml \
  --label-extension extensions/my_mobile_extension.yaml
```

Each extension:

- keeps its own `id`,
- runs independently,
- writes its own payload under `label_extensions.<id>`,
- gets its own aggregate section in dashboard stats.

### First-run checklist

1.  Start with one small spec (`extensions/ui_web_analysis_v1.example.yaml` from the default launcher folder, or a copied reference spec) on a small input and make sure the extension shows up in the dashboard and review exports.
2.  Confirm the trigger matches roughly the expected samples (`matched` vs. `skipped` counts should align with your domain).
3.  Drill down into a handful of matched samples and read `label_extensions.<spec_id>` in the drawer, checking field values, confidence, and unmapped arrays.
4.  Export that subset via `uv run sft-label export-review --include-extensions` to see exactly which columns downstream reviewers will consume.
5.  Once the minimal flow is stable, add richer fields or a second extension spec.
6.  Keep an eye on the preflight diagnostics block when you run or `start`; if the follow-up diagnostics highlight failed/invalid or unmapped issues, fix them before scaling the extension to more data.

### Recommended spec size

- Aim for about 5 fields per extension so manual review stays reasonable.
- Keep enum / multi_enum option lists in the low double digits; if you accrue more than ~20 options it’s a sign the field should be split or simplified.
- Treat the serialized `prompt + schema` as the thing you are trying to keep under ~2,000 characters, which leaves room inside the compact budget for other conversation context.
- If a prompt is longer than that, trim the prose, reduce example sections, or offload part of the logic to another spec.

### Compact-mode advisory

1.  Compact mode uses an 8,000-character conversation budget. We do not gate it, but we recommend keeping each extension’s prompt+schema ≤ 2,000 characters so the payload stays comfortable.
2.  When you pick compact, include a quick summary of the current prompt length and schema size alongside the recommendation so you can adjust before the run.
3.  If you are over the suggestion, focus on tightening instructions, reducing option counts, or splitting functionality into a separate extension.

### Validation checklist

Before trusting an extension rollout, verify the following:
- Trigger scope looks sane; `matched` / `skipped` should align with the subset you intended.
- Value distributions per field are not degenerate; you should see multiple reasonable options, not a single dominant label.
- Low-confidence entries and unmapped rows stay within your domain’s tolerance.
- Drill-down samples actually match the positions you care about so you can spot-check 10–20 cases manually.
- Exported review CSV with `--include-extensions` shows the right columns.

### When to split into multiple extensions

- Keep each spec focused on a single domain or intent so the runtime diagnostics remain easy to interpret.
- Prefer separate specs instead of one oversize schema when a new trigger, monitoring, or follow-up workflow is needed.
- Mount each extension behind a specific trigger rather than bundling multiple unrelated domains into the same file. That keeps the dashboard, export stats, and follow-up diagnostics per spec clean.
- The domains are different enough that one prompt/schema would be overloaded (web vs. mobile vs. infrastructure).
- A single spec would exceed ~5 fields, contain dozens of option values, or produce a prompt+schema near the 2,000-character guideline.
- You need separate trigger rules, enablement, or monitoring per intent.
- You want isolated stats per schema so you can make independent stability decisions.

### Web UI dataset analysis & distribution optimization

“UI SFT data” refers to the subset of samples where the assistant is working on explicit **Web / desktop browser UI surfaces**—pages, dashboards, admin consoles, interactive panels, builders, settings flows, or component/design-system work—so you can reason about coverage, complexity, and engineering constraints as you curate a training mix.

The default Web-only analysis starter (`extensions/ui_web_analysis_v1.example.yaml`, with a matching reference copy under `docs/examples/extensions/ui_web_analysis_v1.yaml`) adds labels for:

- **`ui_surface_type`** — which Web surface family the sample belongs to, so you can spot over-concentration in one kind of UI.
- **`interaction_pattern`** — whether the sample is mostly display, form/config, search/filter/explore, CRUD-heavy, or builder/editing oriented.
- For `interaction_pattern`, treat dashboards and analytics surfaces with filters/exploration as `search-filter-explore`; reserve `dense-crud-operations` for row-level management workflows where create/update/delete is the dominant interaction.
- **`state_data_complexity`** — whether the work is static, local-state only, async-data driven, validation-heavy, or requires multi-source coordination.
- **`ui_constraint_focus`** — whether the sample meaningfully exercises design-system consistency, responsive layout, accessibility semantics, dense data layout, or UI performance.
- **`frontend_stack_shape`** — which Web ecosystem shape the sample belongs to, so framework skew is visible during analysis.

Those labels exist to help you analyze coverage gaps and optimize dataset ratios, not to create more instructions for every sample. Each enabled extension fires an extra extension-labeling call for every matching sample, so **do not enable domain-personalized or mobile-specific extensions across your entire dataset** unless you first narrow the inputs with filters, smaller subsets, or targeted sampling. Mobile surfaces should live in a separate extension because the signals, triggers, prompt wording, and responsiveness constraints differ; mixing them would dilute both the Web-focused dashboard view and the analysis prompts.

For this analysis-oriented Web example, the recommended default is to keep only `domain_any_of` active and leave the other trigger dimensions empty. If you later fill them in, treat that as a deliberate tradeoff: you are tightening recall in exchange for trusting core Pass 1 routing precision more aggressively.

### Cost / risk quick check

- Extra extension calls ≈ `matched_samples × enabled_extension_specs`
- Broader triggers increase recall, but also increase extra calls, latency, and review surface
- Stricter triggers reduce cost, but increase the risk of silent skips caused by imperfect Pass 1 routing
- Start with filtering, sampling, or `--limit`, then validate on roughly **20-100 rows** before widening
