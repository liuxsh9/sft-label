# sft-label

SFT Capability Taxonomy & Auto-Labeling Pipeline.

Automated labeling pipeline for SFT code-generation training data, using a structured 9-category taxonomy (v2.0, 221 tags).

## Install

```bash
pip install -e .
# or with uv
uv sync
```

## Usage

### CLI

```bash
# Run labeling pipeline
sft-label run --input data.json --model gpt-4o-mini --concurrency 50

# Run on a directory
sft-label run --input data_dir/ --output results/

# Resume interrupted run
sft-label run --input data_dir/ --resume results/20250101_120000_gpt-4o-mini/

# Validate taxonomy
sft-label validate

# Export to review CSV
sft-label export-review --input labeled.json --output review.csv
```

### Library

```python
import asyncio
from sft_label import run, PipelineConfig

config = PipelineConfig(
    model="gpt-4o-mini",
    concurrency=50,
    litellm_base="http://localhost:4000/v1",
    litellm_key="your-key",
)

stats = asyncio.run(run("data.json", config=config))
print(f"Labeled {stats['success']}/{stats['total_samples']} samples")
```

## Architecture

```
Input (ShareGPT JSON / Pangu JSONL)
  │
  ├─ Format detection + normalization (preprocessing.py)
  ├─ Multi-turn slicing: each reply → one sample
  ├─ Call 1 (LLM): Intent, Language, Domain, Task, Difficulty
  ├─ Call 2 (LLM): Concept, Agentic, Constraint, Context
  ├─ Validation: tag pool check, cross-dimension consistency
  ├─ Arbitration (optional): re-run low-confidence dimensions
  └─ Output: labeled JSON/JSONL + stats + dashboard
```

## Taxonomy

9 orthogonal categories, 221 tags:

| Category   | Tags | Select |
|-----------|------|--------|
| Intent     | 5    | single |
| Difficulty | 4    | single |
| Context    | 10   | single |
| Language   | 75   | multi  |
| Domain     | 38   | multi  |
| Task       | 21   | multi  |
| Concept    | 25   | multi  |
| Agentic    | 23   | multi  |
| Constraint | 20   | multi  |

## Development

```bash
uv sync --extra dev
uv run pytest
uv run sft-label validate
```

## Environment Variables

- `LITELLM_BASE` — LLM proxy base URL (default: `http://localhost:4000/v1`)
- `LITELLM_KEY` — API key for the LLM proxy

## License

[Apache License 2.0](LICENSE)
