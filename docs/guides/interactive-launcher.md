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
  - Pass 1 + Pass 2 + Pass 4 semantic clustering
  - Pass 2 scoring only
  - Pass 4 semantic clustering only
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
- input path and optional output path
- inline mode (`refresh`, `incremental`, `migrate`, `recompute`)
- whether to chain Pass 2 scoring
- optional rarity stats
- prompt mode
- sample limit
- concurrency / RPS / timeout / retries
- one-off environment overrides for `LITELLM_BASE` / `LITELLM_KEY`

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

## Auto-publishing dashboards from `start`

When the generated workflow is `run`, `score`, or `regenerate-dashboard`, the launcher can ask:

> Auto-publish dashboards to the static service after completion?

If you answer yes:

- it picks an existing configured dashboard service, or helps initialize one if none exists
- it can start or restart the service before the job runs
- when the run finishes, it publishes the generated dashboard bundle and prints stable URLs

Important details:

- the auto-bootstrap path initializes a **PM2-backed** default service
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
