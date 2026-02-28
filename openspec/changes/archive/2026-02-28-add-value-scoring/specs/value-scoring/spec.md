## ADDED Requirements

### Requirement: Rarity computation from tag distributions
The system SHALL compute a rarity score (1-10) for each sample using tag IDF values from a tag distribution stats file. The rarity score SHALL combine weighted per-dimension tag IDF (α=0.7) with cross-dimension combo IDF (1-α=0.3). Each dimension SHALL have a configurable weight (defaults: concept=2.0, domain=1.5, agentic=1.5, language/task/constraint=1.0, intent/difficulty/context=0.3-0.5). Raw rarity values SHALL be normalized to 1-10 scale via percentile mapping across all scored samples.

#### Scenario: Rarity from auto-discovered stats
- **WHEN** scoring `labeled.json` and `stats.json` exists in the same directory
- **THEN** the system SHALL read `tag_distributions` from `stats.json` and compute rarity for each sample

#### Scenario: Rarity from explicit stats file
- **WHEN** `--tag-stats global_stats.json` is provided
- **THEN** the system SHALL use the specified file's `tag_distributions` for rarity computation

#### Scenario: No stats available
- **WHEN** no stats file is found or specified
- **THEN** the system SHALL set `rarity.score` to `null` for all samples and emit a warning

#### Scenario: Stats reference metadata
- **WHEN** rarity is computed for a sample
- **THEN** `value.rarity.stats_ref` SHALL contain `source` (file path), `total_samples` (N), and `timestamp` (ISO format)

### Requirement: COT-preserving smart truncation
The system SHALL provide a truncation mode that preserves chain-of-thought content for quality evaluation. The total budget (default 20K chars) SHALL be allocated as: instruction 15%, COT 45%, response 35%, meta 5%. For COT content exceeding its budget, the system SHALL keep head (30% of COT budget) + 2-3 evenly-spaced middle fragments (10% each) + tail (30% of COT budget), with position markers between gaps.

#### Scenario: Slow-thinking sample truncation
- **WHEN** a sample contains `<think>...</think>` blocks and total content exceeds budget
- **THEN** COT content SHALL be truncated using head + middle fragments + tail strategy with markers like `[... N chars omitted, fragment at X% ...]`

#### Scenario: Fast-thinking sample truncation
- **WHEN** a sample has no COT markers
- **THEN** COT budget (45%) SHALL merge into response budget (total 80%), and response SHALL use head 30% + middle fragments + tail 30%

#### Scenario: Short content no truncation
- **WHEN** total conversation content fits within budget
- **THEN** no truncation SHALL occur and full content SHALL be passed to the LLM

### Requirement: Thinking mode auto-detection
The system SHALL auto-detect whether a sample uses slow-thinking or fast-thinking by checking for `<think>`, `<thinking>`, or `[unused16]...[unused17]` markers in the raw conversation data before any stripping.

#### Scenario: Slow thinking detected
- **WHEN** conversation contains `<think>` or `[unused16]` markers
- **THEN** `thinking_mode` SHALL be set to `"slow"` and COT content SHALL be preserved for truncation

#### Scenario: Fast thinking default
- **WHEN** no COT markers are found
- **THEN** `thinking_mode` SHALL be set to `"fast"`

### Requirement: LLM-based value scoring
The system SHALL make one LLM call per sample to evaluate complexity (1-10), quality (1-10), and reasoning quality (1-10), each with sub-scores. The call SHALL receive the truncated conversation, Pass 1 tags as context, and meta information (thinking_mode, original lengths).

#### Scenario: Successful scoring
- **WHEN** the LLM returns valid JSON with all required fields
- **THEN** the system SHALL parse complexity (instruction/reasoning/implementation/overall), quality (correctness/code_quality/explanation/completeness/overall), reasoning (adapts to thinking_mode), flags array, and confidence score

#### Scenario: LLM call failure with retry
- **WHEN** the LLM call fails or returns invalid JSON
- **THEN** the system SHALL retry up to `SAMPLE_MAX_RETRIES` times, then mark the sample as failed

### Requirement: Value score aggregation
The system SHALL compute a composite `value_score` per sample as a weighted sum of complexity.overall, quality.overall, reasoning.overall, and rarity.score (normalized). Default weights: complexity=0.25, quality=0.35, reasoning=0.15, rarity=0.25. All weights SHALL be configurable via PipelineConfig.

#### Scenario: All dimensions available
- **WHEN** LLM scoring and rarity computation both succeed
- **THEN** `value_score` SHALL be the weighted mean of all four dimension scores

#### Scenario: Rarity unavailable
- **WHEN** rarity is null (no stats file)
- **THEN** `value_score` SHALL be computed from the three LLM-scored dimensions only, with weights renormalized

### Requirement: Standalone score CLI subcommand
The system SHALL provide `sft-label score` subcommand that runs Pass 2 on pre-labeled data. It SHALL accept `--input` (required, path to labeled.json or directory), `--tag-stats` (optional), `--model`, `--concurrency`, and `--limit`.

#### Scenario: Score single file
- **WHEN** `sft-label score --input run_dir/labeled.json`
- **THEN** the system SHALL read labeled samples, compute rarity, run LLM scoring, and output scored.json + stats_value.json + dashboard_value.html in the same directory

#### Scenario: Score with external stats
- **WHEN** `sft-label score --input labeled.json --tag-stats global_stats.json`
- **THEN** the system SHALL use `global_stats.json` for rarity computation instead of auto-discovery

### Requirement: Continuous Pass 1 + Pass 2 execution
The system SHALL support `sft-label run --input data.json --score` to run tag labeling followed by value scoring in a single invocation. Pass 2 SHALL automatically use Pass 1's `stats.json` for rarity computation.

#### Scenario: Continuous mode
- **WHEN** `--score` flag is provided with `sft-label run`
- **THEN** after Pass 1 completes, the system SHALL automatically invoke Pass 2 on the Pass 1 output using the just-produced stats.json

### Requirement: Output file structure
Pass 2 SHALL write output files alongside Pass 1 outputs. Per-file outputs: `scored[_prefix].json`, `scored[_prefix].jsonl`, `stats_value[_prefix].json`, `monitor_value[_prefix].jsonl`, `dashboard_value[_prefix].html`, `failed_value[_prefix].jsonl`. Global outputs: `summary_stats_value.json`, `dashboard_value_<name>.html`.

#### Scenario: Per-sample output structure
- **WHEN** a sample is scored
- **THEN** the output SHALL contain the original `labels` field (unchanged) plus a new `value` field with complexity, quality, reasoning, rarity (with stats_ref), flags, thinking_mode, value_score, and confidence

#### Scenario: Directory mode output
- **WHEN** scoring a directory of labeled files
- **THEN** each file SHALL produce prefixed output files, plus global summary_stats_value.json and dashboard_value_<name>.html

### Requirement: Scoring configuration
PipelineConfig SHALL be extended with scoring-specific fields: value scoring weights (complexity/quality/reasoning/rarity), rarity dimension weights, rarity combo alpha, value truncation budget and ratios. Module-level defaults SHALL be defined in config.py.

#### Scenario: Custom weights
- **WHEN** `PipelineConfig(value_weights={"complexity": 0.3, "quality": 0.4, "reasoning": 0.1, "rarity": 0.2})`
- **THEN** the composite value_score SHALL use the custom weights

#### Scenario: Default configuration
- **WHEN** no scoring config overrides are provided
- **THEN** defaults SHALL be used: complexity=0.25, quality=0.35, reasoning=0.15, rarity=0.25
