# Common workflows

This page collects the most common commands you will reuse after your first run.

## 1. Label a dataset

```bash
uv run sft-label run --input data.json
```

## 2. Label and score in one pass

```bash
uv run sft-label run --input data.json --score
```

Use compact prompts for smaller model payloads:

```bash
uv run sft-label run --input data.json --score --prompt-mode compact
```

## 3. Run on a directory

```bash
uv run sft-label run --input tests/fixtures/e2e_folder_test/ --score --limit 10
```

This is the recommended smoke-sized directory example inside the repo.

## 4. Resume an interrupted run

```bash
uv run sft-label run --resume <run_dir>
```

## 5. Recompute stats after manual edits

```bash
uv run sft-label recompute-stats --input <run_dir>
```

Choose only one pass if needed:

```bash
uv run sft-label recompute-stats --input <run_dir> --pass 1
uv run sft-label recompute-stats --input <run_dir> --pass 2
```

## 6. Regenerate dashboards

```bash
uv run sft-label regenerate-dashboard --input <run_dir>
```

Open them automatically after regeneration:

```bash
uv run sft-label regenerate-dashboard --input <run_dir> --open
```

## 7. Score an existing labeled artifact

```bash
uv run sft-label score --input labeled.json
uv run sft-label score --input labeled.json --tag-stats global_stats.json
```

## 8. Filter high-value samples

```bash
uv run sft-label filter --input scored.json --value-min 6
uv run sft-label filter --input scored.json --value-min 7 --format training
uv run sft-label filter --input scored.json --selection-min 7 --exclude-inherited
```

## 9. Filter multi-turn data by conversation metrics

```bash
uv run sft-label filter --input scored.json --conv-value-min 7
uv run sft-label filter --input scored.json --conv-selection-min 6 --peak-complexity-min 6
uv run sft-label filter --input scored.json --turn-value-min 5 --turn-count-min 3
```

## 10. Validate taxonomy definitions

```bash
uv run sft-label validate
```

## 11. Export review sheets

```bash
uv run sft-label export-review --input labeled.json --output review.csv
uv run sft-label export-review --input <run_dir> --output review.tsv
```

## 12. Publish dashboards

```bash
uv run sft-label dashboard-service init --web-root ~/sft-label-dashboard --service-type builtin
uv run sft-label dashboard-service start
uv run sft-label dashboard-service register-run --run-dir <run_dir>
```

## 13. Work with mirrored inline JSONL runs

### Refresh labels in-place

```bash
uv run sft-label run --input <dataset_dir> --mode refresh
```

### Incrementally fill missing labels

```bash
uv run sft-label run --input <existing_run_dir> --mode incremental
```

### Migrate labels from another run, then fill gaps

```bash
uv run sft-label run --input <dataset_dir> --mode migrate --migrate-from <old_run>
```

### Recompute inline stats and dashboards without LLM calls

```bash
uv run sft-label run --input <existing_run_dir> --mode recompute
```
