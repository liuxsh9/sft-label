# Interactive launcher guide

The interactive launcher is exposed as:

```bash
uv run sft-label start
```

It is the recommended way to begin if you do not already know the exact combination of flags you need.

## What the launcher can launch

Current workflow groups include:

- **Pipeline**
  - Pass 1 labeling
  - Pass 1 + Pass 2
  - Pass 1 + Pass 2 + Pass 3 semantic clustering
  - Smart resume
  - Pass 2 scoring only
  - Pass 3 semantic clustering only
- **Data curation**
  - filtering
- **Maintenance**
  - recompute stats
  - refresh rarity
  - regenerate dashboards
  - validate taxonomy
  - analyze unmapped tags
  - optimize legacy layout
- **Service**
  - dashboard service maintenance
- **Export**
  - semantic export
  - review CSV/TSV export

## What it asks

The launcher asks only for the inputs needed by the selected workflow.

For `run`, it can guide you through:

- new run vs resume
- when to use “Smart resume” vs “Resume specific run directory”
- input path and optional output path
- inline mode (`refresh`, `incremental`, `migrate`, `recompute`)
- optional extension labeling via:
  - one YAML file path,
  - one directory path with **one-or-more** YAML selections,
  - or the shortcut `examples` for built-in extension examples
- whether to chain Pass 2 scoring
- optional rarity stats
- prompt mode
- sample limit
- concurrency / RPS / timeout / retries
- adaptive runtime / recovery sweep toggles
- one-off environment overrides for `LITELLM_BASE` / `LITELLM_KEY`

Concurrency caps default to 200 across `run`, `score`, and smart-resume flows, but you can quickly pick one of the 25 / 50 / 150 / 200 / 300 presets or type a custom value. The RPS max limit prompt likewise accepts a free-form numeric entry if you need to cap request rate yourself.

## Useful flags

```bash
uv run sft-label start --dry-run
uv run sft-label start --lang en
uv run sft-label start --lang zh
```

- `--dry-run`: print the generated command and stop
- `--lang`: switch the interactive prompt language

## Launch summary behavior

Before anything runs, the launcher prints:

- the exact generated CLI command
- any one-off environment overrides, with secrets masked

That means the launcher is a safe way to discover the plain CLI command you want to save in scripts later.

### Extension diagnostics and review exports

When `start` asks for extension specs, you can now:

- paste a single extension YAML path,
- paste a directory path and choose one or more `.yaml` / `.yml` files,
- or type `examples` to browse the built-in `docs/examples/extensions/` directory.

This keeps the flow simple for first-time users while still supporting multiple extensions in one run.

If you enable Pass 1 extensions via `start`, the launcher now surfaces the same diagnostics block that the CLI prints: a preflight summary showing each spec’s trigger presence plus prompt/schema warnings so you can catch oversized or triggerless specs before the run starts. After the job finishes, a follow-up diagnostics section highlights match counts, failed/invalid statuses, and unmapped rows per spec so you can decide whether to revisit prompts, schema options, or trigger rules before scaling. For low-confidence fields, inspect the dashboard drawer or `export-review --include-extensions` output on a small sample.

When you need to inspect the extension columns in export form, run the generated `export-review` command with `--include-extensions` (`uv run sft-label export-review --input <run_dir> --output review.csv --include-extensions`). That flag keeps the default review CSV unchanged unless you opt in, while letting downstream reviewers see the additional extension fields.

Important: every enabled extension adds extra extension-labeling calls for matched samples. Avoid turning on strongly domain-specific extensions for the full dataset; filter or sample first, especially for personalized domain labels. If you need mobile analysis, create a separate mobile extension rather than mixing mobile and Web labels into one spec.

### Web UI dataset analysis guidance

If you want to inspect “UI SFT data” (samples that explicitly target Web/desktop UI surfaces such as dashboards, landing pages, interactive panels, admin consoles, builders, and configuration tools), enable the Web-only analysis example (`docs/examples/extensions/ui_web_analysis_v1.yaml`). It surfaces coverage/distribution signals: surface categories, interaction mix, state/data complexity, engineering constraints, and ecosystem shape. These labels help you spot whether your dataset is overconcentrated in CRUD/forms or missing richer workflows before you narrow or rebalance the mix.

That example intentionally keeps only `domain_any_of` active by default. The other trigger dimensions are left empty in the YAML as visible placeholders. If you later fill them in, you are explicitly choosing to trust core Pass 1 routing quality enough to use those labels as hard preconditions for the extension.

In this example, empty trigger lists mean “do not filter on this dimension.” If you make a trigger list non-empty, that dimension becomes a hard gate. Within one `*_any_of` list the values are OR'ed together; across different trigger dimensions the conditions combine with AND. If the trigger misses, the extension shows up as `status=skipped` and `matched=false`.

Each enabled extension fires an extra extension-labeling call per sample, so **don’t enable domain-personalized or mobile-specific extensions across your entire dataset without pre-filtering or sampling**. Mobile surfaces belong in their own extension because their prompts, triggers, and responsiveness requirements differ; mixing them dilutes the Web-specific analysis while multiplying per-sample costs.

After the dashboard decisions (auto-publish, service exposure, etc.) are settled, the launcher now prints a richer execution overview—command recap, concurrency / RPS caps, dashboard service state, and auto-publish choices—before asking for final confirmation to run the job.

## Auto-publishing dashboards from `start`

When the generated workflow is `run`, `score`, or `regenerate-dashboard`, the launcher can ask:

> Auto-publish dashboards to the static service after completion?

This prompt now defaults to **Yes**, so the recommended path is to publish dashboards when the service is configured.

If you answer yes:

- it picks an existing configured dashboard service, or helps initialize one if none exists
- it can start or restart the service before the job runs
- when the run finishes, it publishes the generated dashboard bundle and prints stable URLs
- on first bootstrap, it asks for a concise exposure mode (`local`, `LAN`, or `public`) instead of raw host details first

Important details:

- the auto-bootstrap path initializes a **PM2-backed** default service
- `LAN` / `public` use `0.0.0.0`; `public` is where you should provide a reverse-proxy or externally shared base URL
- if you prefer the lightweight builtin server for local use, initialize it manually with `dashboard-service init --service-type builtin`

## When to stop using the launcher

Move to direct CLI commands when:

- you already know the exact flags you want
- you are scripting runs in CI or cron
- you want reproducible shell snippets checked into docs or automation

A common pattern is:

1. use `sft-label start --dry-run`
2. copy the generated command
3. run or script that command directly
