# Pass 1 extension labeling

Pass 1 extension labeling lets you attach **additional, domain-specific tags** on top of the core 9-dimension taxonomy. Extensions run **after** the core Pass 1 calls and never change the core taxonomy or Pass 2 semantics.

## When to use it

Use extension labeling when you need extra metadata for a specific subset of samples (e.g., UI-heavy data, mobile tasks, domain-specific heuristics).

## Enabling extension labeling

Use `--label-extension` once per extension spec file:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension extensions/ui_fine_labels.yaml \
  --label-extension extensions/mobile_fine_labels.yaml
```

The flag is repeatable, and **each spec must have a unique `id`**. Specs that do not match a sample’s trigger are recorded as `status=skipped` and `matched=false`.

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

Bundled example specs:

- Minimal starter: `docs/examples/extensions/ui_fine_labels_minimal_v1.yaml`
- Richer UI variant: `docs/examples/extensions/ui_fine_labels_v1.yaml`

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

This keeps the default CSV format unchanged while letting you include extension fields when needed.

## Recommended user-side config patterns

### 1) Minimal / safe first rollout

Use the bundled minimal example directly:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension docs/examples/extensions/ui_fine_labels_minimal_v1.yaml
```

This is the best first step when you want to validate:

- trigger hit rate,
- prompt/schema contract quality,
- extension visibility in dashboard + review CSV.

### 2) Richer UI analytics rollout

Use the richer example when you want more slicing dimensions:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension docs/examples/extensions/ui_fine_labels_v1.yaml
```

### 3) Multiple extension configs coexisting

You can enable several domain-specific extensions at once:

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension docs/examples/extensions/ui_fine_labels_minimal_v1.yaml \
  --label-extension docs/examples/extensions/ui_fine_labels_v1.yaml
```

Each extension:

- keeps its own `id`,
- runs independently,
- writes its own payload under `label_extensions.<id>`,
- gets its own aggregate section in dashboard stats.

### First-run checklist

1.  Start with the minimal spec (`docs/examples/extensions/ui_fine_labels_minimal_v1.yaml`) on a small input and make sure the extension shows up in the dashboard and review exports.
2.  Confirm the trigger matches roughly the expected samples (`matched` vs. `skipped` counts should align with your domain).
3.  Drill down into a handful of matched samples and read `label_extensions.<spec_id>` in the drawer, checking field values, confidence, and unmapped arrays.
4.  Export that subset via `uv run sft-label export-review --include-extensions` to see exactly which columns downstream reviewers will consume.
5.  Once the minimal flow is stable, add richer fields or a second extension spec.

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
- Trigger hit rate looks sane; there should be a meaningful difference between `matched` and `skipped`.
- Value distributions per field are not degenerate; you should see multiple reasonable options, not a single dominant label.
- Low-confidence entries and unmapped rows stay within your domain’s tolerance.
- Drill-down samples actually match the positions you care about so you can spot-check 10–20 cases manually.
- Exported review CSV with `--include-extensions` shows the right columns.

### When to split into multiple extensions

- The domains are different enough that one prompt/schema would be overloaded (web vs. mobile vs. infrastructure).
- A single spec would exceed ~5 fields, contain dozens of option values, or produce a prompt+schema near the 2,000-character guideline.
- You need separate trigger rules, enablement, or monitoring per intent.
- You want isolated stats per schema so you can make independent stability decisions.
