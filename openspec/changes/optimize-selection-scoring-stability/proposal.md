## Why

Recent real-data review on Nemotron SWE shows Pass 2 is directionally correct, but selection is overly driven by intra-class rank and polished final-summary style. This makes mid-trajectory tool actions too sticky in value, and makes summary-heavy final answers overrepresented in top selection buckets.

## What Changes

- Tighten Pass 2 scoring so tool-call-only or low-information trajectory slices receive lower quality/selection without changing prompt payload size.
- Add stage-aware scoring/selection heuristics that distinguish opener, exploration, implementation, verification, and final-summary slices using local features rather than longer prompts.
- Rebalance selection so diversity/rarity has more effect relative to intra-class rank, while keeping existing score semantics stable enough for production.
- Add explicit guardrails that prevent summary-style polish alone from earning top selection unless the sample also contains concrete technical evidence.
- Expand repo-repair domain inference coverage using preprocessing and post-LLM heuristics instead of larger prompts.
- Add regression fixtures and evaluation slices from the audited Nemotron SWE sample set to prevent future drift.

## Capabilities

### New Capabilities
- `selection-stability`: Stable post-scoring heuristics and ranking rules for trajectory stage, tool-action penalties, summary-evidence checks, and diversity-aware selection.

### Modified Capabilities
- `value-scoring`: Selection/value aggregation behavior changes to include stage-aware and evidence-aware adjustments while preserving existing output schema.
- `value-prompts`: Prompt behavior changes to tighten decision boundaries without increasing prompt length or adding new large examples.

## Impact

- Affected code: `src/sft_label/scoring.py`, `src/sft_label/conversation.py`, `src/sft_label/preprocessing.py`, `src/sft_label/prompts_value.py`, dashboard stats/export code, and related tests.
- Production constraint: prompt length MUST NOT increase; prompt edits must be budget-neutral or smaller to avoid payload-based firewall rejection.
- Evaluation: add audited Nemotron SWE cases plus unit/e2e regression checks for value/selection ordering.
