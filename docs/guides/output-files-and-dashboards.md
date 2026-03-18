# Output files and dashboards

This guide explains what a run writes and how dashboards are served locally or published behind a shared static service.

## Standard run layout

A typical standard run looks like this:

```text
<run_dir>/
  labeled.json
  scored.json
  stats_labeling.json
  stats_scoring.json
  conversation_scores.json
  dashboards/
    dashboard_labeling.html
    dashboard_labeling.data/
      manifest.json
      ...
    dashboard_scoring.html
    dashboard_scoring.data/
      manifest.json
      ...
    _dashboard_static/v1/
      dashboard.js
      dashboard.css
```

Notes:

- `scored.json` only exists if you ran Pass 2
- `conversation_scores.json` appears when conversation aggregation is available
- dashboard HTML files are lightweight bootstraps; most data lives in the adjacent `.data/` folder
- `_dashboard_static/v1/` contains the JS/CSS runtime for local viewing

## Mirrored inline JSONL layout

For mirrored inline datasets, the source rows remain the source of truth and labels are embedded into the dataset tree.

Typical layout:

```text
<run_root>/
  <dataset_root>/
    ... mirrored input files with embedded data_label ...
  meta_label_data/
    checkpoint.json
    summary_stats_labeling.json
    summary_stats_scoring.json
    conversation_scores.json
    files/
      ... per-source caches and per-file artifacts ...
    dashboards/
      dashboard_labeling_<scope>.html
      dashboard_labeling_<scope>.data/
      dashboard_scoring_<scope>.html
      dashboard_scoring_<scope>.data/
```

## Rebuilding stats and dashboards offline

If you manually edited outputs, merged results, or migrated inline labels, you do not need to call the LLM again.

```bash
uv run sft-label recompute-stats --input <run_dir>
uv run sft-label regenerate-dashboard --input <run_dir>
```

For inline mirrored runs, recomputed artifacts are written under `meta_label_data/`.

## Opening dashboards locally

Just open the generated HTML files in your browser.

Standard runs:

- `dashboards/dashboard_labeling.html`
- `dashboards/dashboard_scoring.html`

Inline mirrored runs:

- `meta_label_data/dashboards/dashboard_labeling*.html`
- `meta_label_data/dashboards/dashboard_scoring*.html`

## Shared dashboard hosting

The project supports a machine-level static dashboard service so you can publish stable URLs for past runs.

## Service backends

Two service backends are supported:

- `builtin`: uses Python's built-in `http.server`; good for local debugging
- `pm2`: recommended for longer-lived services and easy restart management

## Initialize a service

### Builtin local service

```bash
uv run sft-label dashboard-service init \
  --name local \
  --web-root ~/sft-label-dashboard \
  --service-type builtin \
  --host 127.0.0.1 \
  --port 8765
```

### PM2-backed service

```bash
uv run sft-label dashboard-service init \
  --name prod \
  --web-root ~/sft-label-dashboard \
  --service-type pm2 \
  --public-base-url https://dash.example.com
```

## Manage the service

```bash
uv run sft-label dashboard-service list
uv run sft-label dashboard-service status --name local
uv run sft-label dashboard-service start --name local
uv run sft-label dashboard-service restart --name local
uv run sft-label dashboard-service stop --name local
uv run sft-label dashboard-service set-default --name local
```

If `start` or `restart` detects that the configured port is already owned by another process, interactive TTY sessions will show the conflicting PID/command and prompt for a replacement port instead of exiting immediately. When that happens, the service config is updated in place. Direct share URLs such as `http://192.168.1.25:8765` are rewritten to the new port automatically, while custom reverse-proxy URLs such as `https://dash.example.com/base` are preserved as-is.

## Publish an existing run

```bash
uv run sft-label dashboard-service register-run --name local --run-dir <run_dir>
```

Publishing does three things:

1. copies dashboard HTML files into the service web root
2. copies each matching `dashboard_*.data/` directory
3. rewrites the HTML to use shared service assets instead of the local `_dashboard_static/v1/` path

Published layout under the service root:

```text
<web_root>/
  assets/v1/
    dashboard.js
    dashboard.css
  runs/<run-id>/
    dashboard_labeling.html
    dashboard_labeling.data/
    dashboard_scoring.html
    dashboard_scoring.data/
```

The `register-run` command prints primary URLs for each published dashboard.

## Auto-publishing from the interactive launcher

If you use `uv run sft-label start` and choose auto-publish:

- the launcher can select a configured service
- it can start the service before the main job when the configured service is not already running
- after `run`, `score`, or `regenerate-dashboard`, it publishes the resulting dashboards automatically
- if the service port is already occupied, it shows the conflicting process and lets you choose a new port before retrying

## Which URLs are considered primary

When a run contains multiple dashboard HTML files, the publisher picks one primary URL per dashboard type:

- labeling dashboards prefer the non-sidebar, non-nested-scope global dashboard
- scoring dashboards prefer the top-level global scoring dashboard

The full set still exists under the published run directory.
