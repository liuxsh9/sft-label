## Context

The sft-label project has a mature Pass 1 pipeline that labels SFT training data across 9 taxonomy dimensions (223 tags) via two LLM calls per sample. It processes 750K+ samples with async concurrency, sparse sampling for multi-turn data, and watermark-based file loading for directory mode.

Pass 1 answers "what is this data?". Pass 2 needs to answer "how valuable is this data?" — enabling data selection for more efficient SFT training. The scoring system must integrate cleanly with Pass 1's output and pipeline infrastructure.

Key constraints:
- Must reuse Pass 1's async pipeline infrastructure (semaphore, retry logic, monitor tracking)
- Must not modify Pass 1 output format or behavior
- Must handle both slow-thinking (explicit COT in `<think>` blocks) and fast-thinking (reasoning in response) data
- Rarity computation depends on tag distributions from Pass 1 or an external stats file
- Dashboard must be self-contained HTML (same approach as Pass 1)

## Goals / Non-Goals

**Goals:**
- Score every sample on complexity, quality, reasoning, and rarity (1-10 scale each)
- Produce a single composite value_score per sample with configurable weights
- Support continuous execution (Pass 1 → Pass 2) and standalone execution on pre-labeled data
- Generate dashboards enabling data selection decisions (file ranking, coverage impact, threshold simulation)
- Handle COT content intelligently — preserve it for evaluation, use smart truncation with middle-fragment sampling

**Non-Goals:**
- Token-level value scoring (emerging research, tooling immature)
- Model-intrinsic scoring (IFD/perplexity-based methods — requires target model access)
- Automatic data selection/filtering (we score and visualize; the human decides)
- Modifying Pass 1's taxonomy or tag definitions
- Real-time/streaming scoring

## Decisions

### 1. Single LLM call for scoring (not two)

**Choice**: One call evaluates complexity + quality + reasoning together.

**Rationale**: These dimensions are deeply interrelated — assessing quality requires understanding complexity, and reasoning evaluation requires seeing both. Splitting would double cost (~$150+ per 750K samples) with minimal accuracy gain. The single-call prompt is ~800 tokens, well within budget.

**Alternative considered**: Separate calls for "objective" (correctness) vs "subjective" (quality) scoring. Rejected because correctness assessment inherently requires judging the full response in context.

### 2. COT-preserving smart truncation with middle-fragment sampling

**Choice**: Head (30%) + 2-3 evenly-spaced middle fragments (10% each) + tail (30%) of COT content, with position markers like `[... 2800 chars omitted, fragment at 35% ...]`.

**Rationale**: Simple head+tail truncation misses critical reasoning transitions in the middle of long COT (10K+ chars). Research shows reasoning quality often hinges on key pivot points (error recognition, strategy changes) that occur mid-chain. Sampling 2-3 fragments captures these while keeping total budget manageable.

**Alternative considered**: LLM-based COT summarization before scoring. Rejected due to added cost and latency (extra call per sample).

### 3. Rarity as pure computation (no LLM)

**Choice**: Compute rarity from tag IDF + combo IDF using Pass 1's `stats.json::tag_distributions`, normalize to 1-10 via percentile mapping.

**Rationale**: Rarity is a statistical property of the dataset, not a judgment call. Using LLM for this would be wasteful and less accurate than direct computation. The existing tag distributions in `stats.json` provide exactly the data needed.

**Formula**:
```
tag_idf(t) = log2(N / (count(t) + 1))
dim_rarity(sample, dim) = mean(idf(t) for t in sample.tags[dim])
weighted_rarity = Σ(w[dim] × dim_rarity) / Σ(w[dim])
combo_key = (intent, difficulty, sorted(concept[:3]))
combo_idf = log2(N / (combo_count[combo_key] + 1))
rarity = 0.7 × weighted_rarity + 0.3 × combo_idf
```

### 4. Separate output files (scored.json, not overwriting labeled.json)

**Choice**: Pass 2 outputs to `scored.json` (contains labeled data + value scores), leaving `labeled.json` untouched.

**Rationale**: Keeps Pass 1 and Pass 2 outputs independently usable and recoverable. Users can re-run Pass 2 with different weights/prompts without affecting Pass 1 results.

### 5. Independent dashboard files

**Choice**: `dashboard_value.html` as separate file from `dashboard.html`.

**Rationale**: Pass 2 can run independently of Pass 1. Dashboard data sources differ (scored.json vs labeled.json). Avoids modifying existing visualization code.

### 6. Thinking mode auto-detection

**Choice**: Auto-detect slow/fast thinking via presence of `<think>`, `<thinking>`, or `[unused16]...[unused17]` markers in the raw (pre-strip) conversation. Pass this as `thinking_mode: "slow"|"fast"` to the scoring prompt.

**Rationale**: The prompt adapts its reasoning evaluation criteria based on thinking mode. Slow thinking evaluates COT clarity, consistency, self-correction. Fast thinking evaluates reasoning integration within the response.

### 7. Stats reference input for rarity

**Choice**: Accept `--tag-stats` CLI argument or auto-discover `stats.json` in same directory as input.

**Rationale**: When scoring new data against an existing corpus, users need to provide the full corpus distribution. When running continuously (Pass 1 → Pass 2), the just-produced stats.json is used automatically.

## Risks / Trade-offs

**[LLM scoring variance]** → LLM-as-judge scores can vary ±1-2 points between runs. Mitigation: Use low temperature (0.1), provide detailed scoring anchors with concrete code examples per score level. Consider averaging 2 runs for critical decisions (configurable, off by default).

**[Middle bias in 1-10 scoring]** → LLMs tend to cluster scores around 5-7. Mitigation: Detailed rubric anchors at every 2-point interval with specific code examples. Post-hoc percentile normalization in stats computation.

**[COT truncation losing key reasoning]** → 2-3 middle fragments may still miss the critical pivot point. Mitigation: Include original COT length in meta so the LLM can factor in "this was a very long reasoning chain" even without seeing all of it.

**[Rarity score staleness]** → If tag distributions change significantly between Pass 1 runs, rarity scores from different batches won't be comparable. Mitigation: `stats_ref` metadata on every sample enables tracking which distribution was used.

**[Cost at scale]** → 750K samples × 1 LLM call each = significant cost. Mitigation: Can use cheaper models (gpt-4o-mini) for scoring since it's evaluation not generation. Support sparse scoring (score representative subset, inherit for similar samples — future enhancement).
