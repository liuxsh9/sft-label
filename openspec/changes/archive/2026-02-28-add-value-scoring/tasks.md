## 1. Configuration

- [x] 1.1 Add scoring constants to `config.py`: VALUE_WEIGHTS, RARITY_WEIGHTS, RARITY_COMBO_ALPHA, VALUE_TRUNCATION_BUDGET, VALUE_TRUNCATION_* ratios, KNOWN_FLAGS
- [x] 1.2 Extend `PipelineConfig` dataclass with scoring fields: value_weights, rarity_weights, rarity_combo_alpha, value_truncation_budget

## 2. COT-Preserving Truncation

- [x] 2.1 Add `detect_thinking_mode()` to `preprocessing.py` — scan raw conversations for `<think>`, `<thinking>`, `[unused16]` markers, return `"slow"` or `"fast"`
- [x] 2.2 Add `extract_cot_content()` to `preprocessing.py` — extract COT blocks from conversations without stripping them, return (cot_text, cot_length, response_without_cot)
- [x] 2.3 Implement `truncate_with_fragments()` in `preprocessing.py` — head + 2-3 evenly-spaced middle fragments + tail truncation with position markers `[... N chars omitted, fragment at X% ...]`
- [x] 2.4 Implement `truncate_for_scoring()` in `preprocessing.py` — orchestrates budget allocation (instruction 15% / COT 45% / response 35% / meta 5%), delegates to `truncate_with_fragments()`, handles fast-thinking mode (COT budget → response)

## 3. Rarity Computation

- [x] 3.1 Create `scoring.py` with `load_tag_stats()` — reads tag_distributions from stats.json, validates structure, returns (distributions_dict, total_samples, timestamp)
- [x] 3.2 Implement `compute_tag_idf()` in `scoring.py` — computes log2(N / (count + 1)) for every tag across all dimensions, returns {dim: {tag: idf}}
- [x] 3.3 Implement `compute_sample_rarity()` in `scoring.py` — per-sample weighted tag IDF + combo IDF, returns raw rarity score + stats_ref metadata
- [x] 3.4 Implement `normalize_rarity_scores()` in `scoring.py` — percentile-maps raw rarity values to 1-10 scale across all scored samples

## 4. Scoring Prompt

- [x] 4.1 Create `prompts_value.py` with `SCORING_SYSTEM` prompt — role, three evaluation dimensions with 2-point-interval anchors, flag vocabulary, JSON output format, thinking mode conditional instructions
- [x] 4.2 Add `SCORING_FEWSHOT` in `prompts_value.py` — 2-3 examples: high-complexity/high-quality, low-complexity/high-quality, high-complexity/low-quality (with bugs)
- [x] 4.3 Implement `build_scoring_messages()` in `prompts_value.py` — constructs system + few-shot + user message with `<meta>` block (thinking_mode, lengths, tags) and `<conversation>` block (truncated with COT)

## 5. Core Scoring Pipeline

- [x] 5.1 Implement `score_one()` in `scoring.py` — per-sample scoring: detect thinking mode, truncate for scoring, build messages, call LLM, parse/validate response, merge with precomputed rarity, compute value_score
- [x] 5.2 Implement `validate_score_response()` in `scoring.py` — check all required fields present, scores are int 1-10, confidence is float 0-1, flags from known vocabulary (track unknown)
- [x] 5.3 Implement `compute_value_score()` in `scoring.py` — weighted aggregation of complexity.overall + quality.overall + reasoning.overall + rarity.score, handles null rarity (renormalize weights)
- [x] 5.4 Implement `compute_value_stats()` in `scoring.py` — aggregate statistics: score distributions (mean/std/percentiles), sub-score means, value_by_tag, thinking_mode_stats, flag_counts, flag_value_impact, selection_thresholds, coverage_at_thresholds

## 6. Pipeline Integration

- [x] 6.1 Implement `run_scoring()` in `scoring.py` — async entry point: load labeled data + stats, compute rarity for all samples, run score_one() with concurrency semaphore, write outputs (scored.json, stats_value.json, monitor_value.jsonl, failed_value.jsonl)
- [x] 6.2 Implement `run_scoring_directory()` in `scoring.py` — directory mode: iterate labeled files, per-file scoring + output, produce summary_stats_value.json
- [x] 6.3 Add `score` subcommand to `cli.py` — args: --input (required), --tag-stats, --model, --concurrency, --limit
- [x] 6.4 Add `--score` flag to `run` subcommand in `cli.py` — after Pass 1 completes, auto-invoke run_scoring() with Pass 1's stats.json
- [x] 6.5 Wire continuous mode in `pipeline.py` — after run() completes, if score=True, call run_scoring() with output_dir paths

## 7. Dashboard

- [x] 7.1 Create `tools/visualize_value.py` with `compute_value_viz_data()` — transform scored data + stats into dashboard JSON: overview cards, score distributions (histogram bins), sub-score breakdowns, value×tag cross-analysis, thinking mode comparison, flag analysis
- [x] 7.2 Implement per-file dashboard HTML template in `visualize_value.py` — self-contained HTML with 6 sections: Value Overview Cards, Score Distributions, Sub-score Breakdown, Value×Tag Cross-Analysis (quality by difficulty, value by domain, complexity vs quality scatter, rarity vs quality quadrant), Thinking Mode Analysis, Flag Analysis
- [x] 7.3 Implement global dashboard HTML template — adds: File Ranking Table (sortable), Coverage Impact Analysis, Data Selection Simulator (pre-computed thresholds)
- [x] 7.4 Implement `generate_value_dashboard()` — entry point matching Pass 1's `generate_dashboard()` signature, handles both per-file (full data) and global (stats-only) modes

## 8. Testing

- [x] 8.1 Create `tests/fixtures/smoke_test_value.json` — 3-5 test samples with COT content (slow-thinking with `<think>` blocks + fast-thinking), covering diverse complexity/quality levels
- [x] 8.2 Write unit tests for truncation in `tests/test_scoring.py` — test detect_thinking_mode(), extract_cot_content(), truncate_with_fragments(), truncate_for_scoring() for both slow/fast modes, edge cases (no truncation needed, empty COT)
- [x] 8.3 Write unit tests for rarity in `tests/test_scoring.py` — test compute_tag_idf(), compute_sample_rarity(), normalize_rarity_scores(), null stats handling
- [x] 8.4 Write unit tests for score validation in `tests/test_scoring.py` — test validate_score_response() with valid/invalid/partial responses, unknown flags
- [x] 8.5 Write integration test: end-to-end scoring of smoke test data via API call (using LITELLM_BASE/KEY env vars), verify output structure, stats computation, dashboard generation

## 9. Iteration and Prompt Tuning

- [x] 9.1 Run scoring on smoke test fixtures, review LLM outputs for score distribution, anchor calibration, and thinking mode adaptation
- [x] 9.2 Tune prompt anchors and few-shot examples based on scoring results — adjust if middle-bias detected, refine flag vocabulary, improve COT evaluation criteria
- [ ] 9.3 Run scoring on a larger sample (~100-500 from real data), validate cross-analysis outputs, verify dashboard renders correctly
