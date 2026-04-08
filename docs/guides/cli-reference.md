# CLI 参考手册 / CLI Reference

Complete parameter reference for all `sft-label` subcommands. Maps every interactive launcher prompt to its direct CLI flag equivalent with defaults.

Related guides:

- [Interactive launcher guide](interactive-launcher.md) — step-by-step interactive workflow walkthrough
- [Common workflows](common-workflows.md) — recipe-oriented usage patterns
- [Getting started](getting-started.md) — first-time setup

---

## 概述 / Overview

### Two startup modes

| Mode | Command | When to use |
|---|---|---|
| **Interactive** | `uv run sft-label start` | First-time users, exploring options, building commands |
| **Direct CLI** | `uv run sft-label run ...` | Scripting, CI/CD, AI agents, advanced users |

Both modes produce identical results — the interactive launcher builds the same argv that direct CLI uses.

**Tip:** Use `uv run sft-label start --dry-run` to discover the exact command, then copy it for scripting.

### `start` flags

| Flag | Default | Description |
|---|---|---|
| `--dry-run` | off | Print generated command without executing |
| `--lang` | `zh` | Prompt language: `zh` or `en` |
| `--en` | off | Shorthand for `--lang en` |

### All subcommands

| Command | Description |
|---|---|
| `start` | Interactive launcher — builds and runs a command for you |
| `run` | Pass 1 labeling (optionally chains Pass 2/3) |
| `score` | Pass 2 value scoring on pre-labeled data |
| `validate` | Validate taxonomy definitions |
| `semantic-cluster` | Pass 3: trajectory semantic clustering |
| `export-semantic` | Export semantic clustering rows |
| `filter` | Pass 4: filter scored data by value/quality criteria |
| `export-review` | Export labeled data to review CSV/TSV |
| `recompute-stats` | Rebuild stats from existing outputs (no LLM) |
| `refresh-rarity` | Recompute rarity/value fields (no LLM) |
| `regenerate-dashboard` | Regenerate HTML dashboards from stats |
| `complete-postprocess` | Finish deferred Pass 2 postprocess offline |
| `optimize-layout` | Normalize legacy output layout |
| `analyze-unmapped` | Inspect out-of-pool tags |
| `dashboard-service` | Manage shared dashboard hosting |

---

## 快速命令速查 / Quick Recipes

Environment variables must be set before running pipeline commands:

```bash
export LITELLM_BASE="http://localhost:4000/v1"
export LITELLM_KEY="your-key"
```

Or inline:

```bash
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score
```

### Common recipes

```bash
# 1. Pass 1 only (labeling)
uv run sft-label run --input data.json

# 2. Pass 1 + Pass 2 (labeling + scoring) — most common
uv run sft-label run --input data.json --score

# 3. Pass 1 + 2 + 3 (full pipeline with semantic clustering)
uv run sft-label run --input data.json --score --semantic-cluster

# 4. Pass 2 only (score existing labeled data)
uv run sft-label score --input labeled.json

# 5. Resume an interrupted run
uv run sft-label run --resume /path/to/run_dir --score

# 6. Smart resume (resume + add new input)
uv run sft-label run --input new_data.json --resume /path/to/run_dir --score

# 7. With label extensions
uv run sft-label run --input data.json --score \
  --label-extension extensions/my_ext.yaml \
  --label-extension extensions/another.yaml

# 8. Inline JSONL mode (labels written alongside dataset rows)
uv run sft-label run --input /path/to/dataset_dir/ --mode refresh --score

# 9. Migrate labels from an older run
uv run sft-label run --input /path/to/dataset_dir/ --mode migrate \
  --migrate-from /path/to/old_run/ --score

# 10. Full production run with all defaults explicit
uv run sft-label run --input data/ --score \
  --prompt-mode compact \
  --rollout-preset compact_safe \
  --sparse-preset b \
  --concurrency 200 \
  --rps-limit 20 \
  --rps-warmup 30 \
  --request-timeout 90 \
  --max-retries 3 \
  --adaptive-runtime \
  --recovery-sweep

# 11. Filter scored data for training
uv run sft-label filter --input run_dir --value-min 7 --format training

# 12. Offline maintenance
uv run sft-label recompute-stats --input run_dir
uv run sft-label regenerate-dashboard --input run_dir

# 13. Smoke test with bundled fixtures
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10
```

---

## `run` — Pass 1 标注 / Pass 1 Labeling

Run the tag labeling pipeline. Optionally chain Pass 2 scoring (`--score`) and/or Pass 3 semantic clustering (`--semantic-cluster`).

### Parameter reference

| CLI Flag | Type | Default | Description | Interactive prompt |
|---|---|---|---|---|
| `--input` | path | *(required unless `--resume`)* | Input file (.json/.jsonl) or directory | 输入文件或目录路径 |
| `--output` | path | sibling of input | Output directory | 输出目录 |
| `--resume` | path | none | Resume from existing run directory | 续跑目录 |
| `--mode` | choice | `refresh` | Inline mode: `refresh`, `incremental`, `migrate`, `recompute` | 标注模式 |
| `--migrate-from` | path | none | Migration source (requires `--mode migrate`) | 迁移来源 |
| `--limit` | int | `0` (all) | Max samples per file | 样本限制 |
| `--shuffle` | flag | off | Shuffle samples before processing | 是否打乱样本 |
| `--no-arbitration` | flag | off (arbitration on) | Disable low-confidence arbitration pass | 是否启用仲裁 |
| `--score` | flag | off | Chain Pass 2 value scoring after labeling | *(workflow selection)* |
| `--semantic-cluster` | flag | off | Chain Pass 3 semantic clustering | *(workflow selection)* |
| `--tag-stats` | path | auto-discover | Stats file for Pass 2 rarity baseline | 稀有度统计文件路径 |
| `--rarity-mode` | choice | `absolute` | Rarity normalization: `absolute`, `percentile` | 稀有度归一化模式 |
| `--extension-rarity-mode` | choice | `off` | Extension rarity: `off`, `preview`, `bonus_only` | Extension rarity mode |
| `--prompt-mode` | choice | `compact` | Prompt payload size: `compact` (smaller), `full` | 提示模式 |
| `--rollout-preset` | choice | `compact_safe` | Multi-turn rollout: `compact_safe`, `baseline_control`, `planner_hybrid` | Multi-turn rollout preset |
| `--sparse-preset` | choice | `b` | Sparse multi-turn: `current`, `a`, `b`, `c` | Sparse multi-turn labeling preset |
| `--model` | string | `gpt-4o-mini` | LLM model name (sets both labeling and scoring) | 模型覆写 |
| `--concurrency` | int | none (Pass 1: 200, Pass 2: 500) | Override LLM concurrency — sets both passes to same value | 最大并发数 |
| `--rps-limit` | float | `20` | Max LLM requests/sec | RPS 上限 |
| `--rps-warmup` | float | `30` | Seconds to ramp from 1 rps to full rps | RPS 预热秒数 |
| `--request-timeout` | int | `90` | Per-request timeout in seconds | 请求超时秒数 |
| `--max-retries` | int | `3` | Max retries per request | 最大重试次数 |
| `--adaptive-runtime` / `--no-adaptive-runtime` | flag pair | on | Enable/disable adaptive LLM pressure control | 自适应运行时 |
| `--recovery-sweep` / `--no-recovery-sweep` | flag pair | on | Enable/disable end-of-phase retry for infra failures | 恢复扫描 |
| `--label-extension` | path | none | Label extension spec (repeatable) | 是否加载 Label Extension |

> **Note — concurrency:** When `--concurrency` is **not passed**, Pass 1 uses 200 and Pass 2 uses 500 (their independent defaults). When `--concurrency N` **is passed**, both passes are set to N. The interactive launcher defaults to 200, which means it lowers Pass 2 from 500 to 200. See [Appendix](#附录--pipelineconfig-库模式差异--appendix-pipelineconfig-library-mode-differences) for details.

> **Note:** `--model` sets both `labeling_model` and `scoring_model` to the same value. In library mode you can set them independently.

---

## `score` — Pass 2 评分 / Pass 2 Value Scoring

Run value scoring on pre-labeled data. Computes complexity, quality, reasoning, and rarity scores.

### Parameter reference

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Labeled file or directory |
| `--tag-stats` | path | auto-discover | Stats file for rarity baseline |
| `--rarity-mode` | choice | `absolute` | Rarity normalization: `absolute`, `percentile` |
| `--extension-rarity-mode` | choice | `off` | Extension rarity: `off`, `preview`, `bonus_only` |
| `--limit` | int | `0` (all) | Max samples to score |
| `--resume` | flag | off | Skip already-scored samples in scored.jsonl |
| `--prompt-mode` | choice | `compact` | Prompt payload size: `compact`, `full` |
| `--rollout-preset` | choice | `compact_safe` | Multi-turn rollout preset |
| `--sparse-preset` | choice | `b` | Sparse multi-turn labeling preset |
| `--model` | string | `gpt-4o-mini` | LLM model name |
| `--concurrency` | int | none (default: 500) | Override LLM concurrency |
| `--rps-limit` | float | `20` | Max LLM requests/sec |
| `--rps-warmup` | float | `30` | RPS warmup seconds |
| `--request-timeout` | int | `90` | Per-request timeout seconds |
| `--max-retries` | int | `3` | Max retries per request |
| `--adaptive-runtime` / `--no-adaptive-runtime` | flag pair | on | Adaptive LLM pressure control |
| `--recovery-sweep` / `--no-recovery-sweep` | flag pair | on | End-of-phase recovery sweep |

> **Note:** `score` only runs Pass 2. Without `--concurrency`, the effective default is **500** (`DEFAULT_SCORING_CONCURRENCY`). When `--concurrency N` is passed, it overrides to N.

### Key differences from `run`

- `--input` is **required** (enforced by argparse, not optional)
- `--resume` is a **boolean flag** (skip scored samples), not a directory path
- No `--output`, `--shuffle`, `--mode`, `--migrate-from`, `--no-arbitration`, `--label-extension`, `--semantic-cluster` flags

---

## 运行时参数详解 / Shared Runtime Parameters

### Rollout presets

| Preset | Description | Use case |
|---|---|---|
| `compact_safe` | Conservative multi-turn handling with compact payloads | **Recommended production default** |
| `baseline_control` | Original/rollback behavior for comparison | Regression testing, control experiments |
| `planner_hybrid` | Semantic-aware multi-turn planning | **Experimental**, may change |

### Sparse presets

Control how multi-turn slices are sampled for labeling:

| Preset | Description |
|---|---|
| `current` | Historical defaults |
| `a` | Conservative (more slices labeled) |
| `b` | **Balanced default** |
| `c` | Aggressive (fewer slices, more inheritance) |

### Prompt mode

| Mode | Description |
|---|---|
| `compact` | Smaller payloads, fewer few-shot examples. **CLI/interactive default.** Best for size-limited endpoints. |
| `full` | Full prompts with all examples. Use when endpoint can tolerate larger payloads. |

### Adaptive runtime

When enabled (default), automatically adjusts concurrency and RPS under provider pressure:

- Degrades concurrency and RPS when error/timeout rates exceed thresholds
- Recovers gradually when stability improves
- Disables arbitration during degraded state to reduce load

### Recovery sweep

When enabled (default), performs one conservative retry pass at the end of each pipeline stage:

- Targets only infra-style failures (timeouts, rate limits)
- Uses reduced concurrency (25% of normal) and reduced RPS (25%)
- Does not retry content/validation failures

---

## 环境变量 / Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LITELLM_BASE` | `http://localhost:4000/v1` | LLM API base URL |
| `LITELLM_KEY` | `""` (empty) | LLM API key |

These must be set before running any pipeline command (`run`, `score`).

**Option 1 — shell profile (persistent):**

```bash
# Add to ~/.zshrc or ~/.bashrc
export LITELLM_BASE="http://your-llm-endpoint:4000/v1"
export LITELLM_KEY="sk-your-key"
```

**Option 2 — inline (one-off):**

```bash
LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run sft-label run --input data.json --score
```

The interactive launcher can also set one-off overrides through the "Override LITELLM_BASE/KEY" prompt.

---

## `filter` — 数据筛选 / Data Filtering

Select high-value samples from scored artifacts.

### Output options

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Scored file, inline JSONL, or run directory |
| `--output` | path | auto alongside input | Output file path |
| `--format` | choice | `scored` | Output format: `scored` (with labels) or `training` (training-ready) |
| `--preserve-structure` | flag | off | Mirror input folder structure (directory mode) |

### Sample-level criteria

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--value-min` | float | none | Min `value_score` to retain |
| `--threshold` | float | none | Alias for `--value-min` (backward compat) |
| `--selection-min` | float | none | Min `selection_score` to retain |
| `--correctness-min` | float | none | Min `quality.correctness` score |
| `--include-tags` | string | none | Require at least one tag (comma-separated `dim:tag`) |
| `--exclude-tags` | string | none | Exclude samples with any tag (comma-separated `dim:tag`) |
| `--difficulty` | string | none | Allowed difficulty levels (comma-separated) |
| `--thinking-mode` | choice | none | Filter by thinking mode: `slow` or `fast` |
| `--exclude-inherited` | flag | off | Exclude samples with inherited labels |
| `--include-unscored` | flag | off | Keep samples without value scores |
| `--verify-source` | path | none | Only retain samples from this source file |

### Conversation-level criteria

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--conv-value-min` | float | none | Min conversation value score |
| `--conv-selection-min` | float | none | Min conversation selection score |
| `--conv-value-v2-min` | float | none | Min conversation value_v2 score |
| `--conv-selection-v2-min` | float | none | Min conversation selection_v2 score |
| `--peak-complexity-min` | float | none | Min peak complexity across turns |
| `--trajectory-structure-min` | float | none | Min trajectory structure score |
| `--rarity-confidence-min` | float | none | Min conversation rarity confidence |
| `--observed-turn-ratio-min` | float | none | Min directly labeled turn ratio |
| `--turn-count-min` | int | none | Min turns in conversation |
| `--turn-count-max` | int | none | Max turns in conversation |

### Turn-level pruning

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--turn-value-min` | float | none | Min per-turn `value_score` |
| `--turn-quality-min` | float | none | Min per-turn `quality.overall` |
| `--max-pruned-ratio` | float | `0.5` | Max fraction of turns to prune per conversation |
| `--no-keep-first-last` | flag | off | Allow pruning first/last turns |

### Examples

```bash
# Basic value filter
uv run sft-label filter --input run_dir --value-min 7

# Training-format export
uv run sft-label filter --input run_dir --value-min 7 --format training

# Conversation-level filter
uv run sft-label filter --input run_dir --conv-value-min 7 --peak-complexity-min 6

# Multi-condition filter
uv run sft-label filter --input scored.json \
  --value-min 6 \
  --difficulty advanced,expert \
  --exclude-inherited \
  --thinking-mode slow
```

---

## 其他子命令速查 / Other Subcommands

### `semantic-cluster` — Pass 3 语义聚类

```bash
uv run sft-label semantic-cluster --input <run_dir_or_file>
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Input file/dir (prefers scored/labeled) |
| `--output` | path | input dir | Output directory |
| `--limit` | int | `0` (all) | Max samples to process |
| `--resume` | flag | off | Check against existing manifest |
| `--no-export-representatives` | flag | off | Skip representative windows output |
| `--semantic-long-turn-threshold` | int | auto | Min turns for sliding windows |
| `--semantic-window-size` | int | auto | Window body size in turns |
| `--semantic-window-stride` | int | auto | Sliding stride in turns |
| `--semantic-pinned-prefix-max-turns` | int | auto | Max turns in pinned prefix |
| `--semantic-embedding-provider` | choice | auto | Backend: `local` or `api` |
| `--semantic-embedding-model` | string | auto | Embedding model id |
| `--semantic-embedding-dim` | int | auto | Vector dimension (local) |
| `--semantic-embedding-batch-size` | int | auto | Embedding batch size |
| `--semantic-embedding-max-workers` | int | auto | Worker count (local) |
| `--semantic-semhash-bits` | int | auto | SemHash bit width |
| `--semantic-semhash-seed` | int | auto | SemHash random seed |
| `--semantic-semhash-bands` | int | auto | SemHash band count |
| `--semantic-hamming-radius` | int | auto | Hamming radius for candidates |
| `--semantic-ann-top-k` | int | auto | Max refined neighbors |
| `--semantic-ann-sim-threshold` | float | auto | Cosine similarity threshold |
| `--semantic-output-prefix` | string | auto | Artifact filename prefix |

### `export-semantic`

```bash
uv run sft-label export-semantic --input <cluster_dir> --output rows.jsonl
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Directory with clustering artifacts |
| `--output` | path | *(required)* | Output JSONL path |
| `--include-all` | flag | off | Include non-representative rows |

### `recompute-stats`

Rebuild stats from existing outputs without LLM calls.

```bash
uv run sft-label recompute-stats --input <run_dir>
uv run sft-label recompute-stats --input scored.json --pass 2
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Labeled/scored file or run directory |
| `--pass` | choice | `both` | Which stats: `1`, `2`, or `both` |
| `--output` | path | same as input | Output directory |
| `--workers` | int | `8` | File-level parallel workers |

### `refresh-rarity`

Recompute rarity/value/selection fields without LLM calls.

```bash
uv run sft-label refresh-rarity --input <run_dir_or_file>
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Scored file or run directory |
| `--tag-stats` | path | auto | Cross-dataset rarity baseline |
| `--output` | path | in-place | Output directory |
| `--mode` | choice | auto | Rarity normalization: `absolute`, `percentile` |
| `--extension-rarity-mode` | choice | `off` | Extension rarity: `off`, `preview`, `bonus_only` |
| `--workers` | int | `8` | File-level parallel workers |

### `regenerate-dashboard`

Rebuild HTML dashboards from existing stats/data files.

```bash
uv run sft-label regenerate-dashboard --input <run_dir>
uv run sft-label regenerate-dashboard --input <run_dir> --pass 1
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Run directory with stats/data |
| `--pass` | choice | `both` | Which dashboards: `1`, `2`, or `both` |
| `--open` | flag | off | Open in browser after generation |
| `--workers` | int | `8` | Parallel workers for batch mode |

### `complete-postprocess`

Finish deferred Pass 2 conversation aggregation and dashboards offline.

```bash
uv run sft-label complete-postprocess --input <run_dir>
uv run sft-label complete-postprocess --input <run_dir> --scope all
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Run directory with scored outputs |
| `--scope` | choice | `global` | `global` (safe for huge runs) or `all` (includes per-file dashboards) |
| `--open` | flag | off | Open last dashboard in browser |
| `--workers` | int | `8` | File-level workers |

### `export-review`

Export labeled data to review CSV/TSV.

```bash
uv run sft-label export-review --input <run_dir> --output review.csv
uv run sft-label export-review --input <run_dir> --output review.csv --include-extensions
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Labeled JSON/JSONL, inline JSONL, or run directory |
| `--output` | path | *(required)* | Output CSV/TSV path |
| `--monitor` | path | `""` | Monitor JSONL file (optional) |
| `--format` | choice | infer from ext | `csv` or `tsv` |
| `--include-extensions` | flag | off | Include extension label columns |

### `analyze-unmapped`

Inspect out-of-pool tags from labeled outputs.

```bash
uv run sft-label analyze-unmapped --input <run_dir>
uv run sft-label analyze-unmapped --input <run_dir> --dimension task --top 50
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Labeled/scored file, run dir, or stats file |
| `--dimension` | string | none (shows all) | Show one dimension only (e.g. `task`, `concept`) |
| `--top` | int | `20` | Max tags per dimension |
| `--examples` | int | `2` | Max sample examples per tag |
| `--stats-only` | flag | off | Only read from stats file |

### `optimize-layout`

Normalize legacy output naming/layout.

```bash
uv run sft-label optimize-layout --input <old_run_dir>          # dry-run
uv run sft-label optimize-layout --input <old_run_dir> --apply  # execute
```

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | path | *(required)* | Run directory root |
| `--apply` | flag | off (dry-run) | Execute planned operations |
| `--prune-legacy` | flag | off | Delete legacy alias files |
| `--manifest` | path | auto | Manifest output path |

### `validate`

Validate taxonomy definitions. No arguments.

```bash
uv run sft-label validate
```

### `dashboard-service`

Manage shared dashboard static services.

```bash
uv run sft-label dashboard-service <action> [options]
```

**Actions:**

| Action | Key flags | Description |
|---|---|---|
| `init` | `--web-root PATH` (required), `--name`, `--host`, `--port`, `--service-type`, `--public-base-url`, `--pm2-name` | Initialize or update a service |
| `list` | *(none)* | List configured services |
| `status` | `--name` | Show service status |
| `start` | `--name` | Start service |
| `restart` | `--name` | Restart service |
| `stop` | `--name` | Stop service |
| `runs` | `--name` | List published runs |
| `set-default` | `--name` (required) | Set default service |
| `register-run` | `--run-dir PATH` (required), `--name` | Publish a run directory |

**`init` defaults:** name=`default`, host=`127.0.0.1`, port=`8765`, service-type=`pm2`

---

## 非交互式用法 / Non-interactive Usage (AI Agent / Automation)

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | User error (bad arguments, missing files, runtime failure) |
| `2` | Pipeline aborted (sustained auth/billing errors — resumable with `--resume`) |

### Core rules

1. **Never use `start`** — it requires interactive TTY input. Always call subcommands directly (`run`, `score`, `filter`, etc.).
2. **Set environment variables** — `LITELLM_BASE` and `LITELLM_KEY` must be configured before pipeline commands.
3. **Validate first** — use `--limit 5` or `--limit 10` for a small test run before processing large datasets.
4. **Check `--help`** — every subcommand supports `uv run sft-label <command> --help`.

### Typical agent workflow

```bash
# Step 1: Validate connectivity with a small run
uv run sft-label run \
  --input tests/fixtures/e2e_folder_test/ \
  --score \
  --limit 5

# Step 2: Full production run
uv run sft-label run \
  --input /path/to/dataset/ \
  --score \
  --prompt-mode compact \
  --rollout-preset compact_safe \
  --concurrency 200

# Step 3: Filter results
uv run sft-label filter \
  --input /path/to/run_dir/ \
  --value-min 7 \
  --format training

# Step 4: Export for review
uv run sft-label export-review \
  --input /path/to/run_dir/ \
  --output review.csv
```

### All defaults for a typical `run --score`

When using `uv run sft-label run --input <path> --score` with no other flags, these defaults apply:

| Parameter | Value |
|---|---|
| mode | `refresh` |
| prompt_mode | `compact` |
| rollout_preset | `compact_safe` |
| sparse_preset | `b` |
| concurrency (Pass 1) | `200` |
| concurrency (Pass 2) | `500` |
| rps_limit | `20` req/s |
| rps_warmup | `30` seconds |
| request_timeout | `90` seconds |
| max_retries | `3` |
| adaptive_runtime | on |
| recovery_sweep | on |
| model | `gpt-4o-mini` |
| rarity_mode | `absolute` |
| arbitration | on |
| shuffle | off |
| limit | `0` (all samples) |

These are **identical** to what the interactive launcher selects when you press Enter through all prompts, **except** the interactive launcher explicitly sets `--concurrency 200` which overrides Pass 2 from 500 to 200.

---

## 附录 — PipelineConfig 库模式差异 / Appendix: PipelineConfig Library-Mode Differences

When using `sft-label` as a Python library via `from sft_label import run, PipelineConfig`, be aware of these differences from CLI defaults:

| Parameter | CLI default (no flag) | `PipelineConfig()` default | Notes |
|---|---|---|---|
| `prompt_mode` | `compact` | `full` | Must set `prompt_mode="compact"` explicitly to match CLI behavior |
| `concurrency` (Pass 1) | `200` | `200` | Same in both modes |
| `scoring_concurrency` (Pass 2) | `500` (no flag) or `N` (with `--concurrency N`) | `500` | CLI `--concurrency N` overrides both passes to N; without the flag, Pass 2 stays at 500 |
| rollout/sparse presets | applied automatically | not applied | Call `apply_rollout_preset(config, "compact_safe")` and `apply_sparse_preset(config, "b")` explicitly |

**Library-mode example matching CLI defaults:**

```python
from sft_label import run, PipelineConfig
from sft_label.config import apply_rollout_preset, apply_sparse_preset

config = PipelineConfig(prompt_mode="compact")
apply_rollout_preset(config, "compact_safe")
apply_sparse_preset(config, "b")

# Now config matches `uv run sft-label run --input ... --score` defaults
```
