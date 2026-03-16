# Getting started

This guide is the fastest path from a fresh checkout to a successful local run.

## Prerequisites

- Python 3.9+
- [`uv`](https://github.com/astral-sh/uv)
- An OpenAI-compatible endpoint exposed through LiteLLM-style environment variables:
  - `LITELLM_BASE`
  - `LITELLM_KEY`

## Install

From the repo root:

```bash
uv sync --extra dev
```

If you also want the dataset download/conversion scripts in `scripts/`:

```bash
uv sync --extra dev --extra data
```

## Verify the checkout

```bash
uv run sft-label validate
uv run pytest tests/test_cli_start.py tests/test_dashboard_service.py -q
```

## Set your model endpoint

```bash
export LITELLM_BASE="http://localhost:4000/v1"
export LITELLM_KEY="your-key"
```

If these variables only exist in your interactive shell startup files, run commands through an interactive shell:

```bash
zsh -ic 'cd /path/to/sft-label && uv run sft-label start'
```

## Recommended first command

```bash
uv run sft-label start
```

Why start here?

- it surfaces the supported workflows
- it only asks for the arguments required by the workflow you selected
- it previews the exact command before execution
- it can auto-publish dashboards after the run

If you want to inspect the generated command without running it:

```bash
uv run sft-label start --dry-run
```

## Minimal direct CLI examples

### Repo smoke test (recommended)

```bash
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10
```

### Pass 1 only

```bash
uv run sft-label run --input tests/fixtures/smoke_test.json --limit 5
```

### Pass 1 + Pass 2

```bash
uv run sft-label run --input tests/fixtures/smoke_test.json --score --limit 5
```

### Score an existing labeled file

```bash
uv run sft-label score --input labeled.json
```

## What success looks like

A successful standard run usually gives you:

- `labeled.json` after Pass 1
- `scored.json` after Pass 2
- `stats_labeling.json` and/or `stats_scoring.json`
- `conversation_scores.json` for multi-turn scoring outputs
- dashboard HTML under `dashboards/`

A successful mirrored inline run writes the mirrored dataset plus metadata under `meta_label_data/`.

See [Output files and dashboards](output-files-and-dashboards.md) for the full layout.

## Troubleshooting the first run

### `sft-label` cannot reach the model endpoint

Check:

- `LITELLM_BASE`
- `LITELLM_KEY`
- firewall / proxy / localhost routing
- whether your endpoint supports the request size implied by the chosen prompt mode

For smaller payloads, prefer compact prompts:

```bash
uv run sft-label run --input data.json --score --prompt-mode compact
```

### You edited labels and want updated dashboards without new LLM calls

```bash
uv run sft-label recompute-stats --input <run_dir>
uv run sft-label regenerate-dashboard --input <run_dir>
```

### You only want to validate the repo without running the model

```bash
uv run sft-label validate
uv run pytest
```
