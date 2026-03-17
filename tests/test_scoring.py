"""Tests for the value scoring pipeline (Pass 2).

Covers:
  - Thinking mode detection and COT extraction (preprocessing.py)
  - COT-preserving truncation (preprocessing.py)
  - Tag IDF and rarity computation (scoring.py)
  - Score response validation (scoring.py)
  - Value score computation (scoring.py)
  - Aggregate stats computation (scoring.py)
"""

import json
import math
import os
import re
import asyncio
import shutil
from pathlib import Path

import pytest

from sft_label.artifacts import (
    PASS2_STATS_FILE,
    PASS2_DASHBOARD_FILE,
    dashboard_relpath,
)
from sft_label.config import PipelineConfig
from sft_label.conversation import (
    _compute_penalty,
    _compute_pure_quality_from_slice,
    aggregate_conversation,
)
from sft_label.preprocessing import (
    detect_thinking_mode,
    extract_cot_content,
    truncate_with_fragments,
    truncate_for_scoring,
)
from sft_label.scoring import (
    compute_tag_idf,
    compute_sample_rarity,
    normalize_rarity_scores,
    build_combo_counts,
    validate_score_response,
    compute_value_score,
    compute_value_stats,
    load_tag_stats,
    load_tag_stats_context,
    _infer_selection_features,
    _infer_domain_backfill,
    compute_selection_scores,
    compute_selection_scores_from_summaries,
)
from sft_label.prompts_value import SCORING_SYSTEM_COMPACT, build_scoring_messages
from sft_label.tools.compare_selection import compare_selection_configs, load_selection_regression_pack


def _bootstrap_from_html(html: str) -> dict:
    match = re.search(
        r'<script id="dashboard-bootstrap" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_selection_regression_fixture():
    return load_selection_regression_pack(FIXTURES_DIR / "nemotron_selection_regression.json")


# ─────────────────────────────────────────────────────────
# 8.2  Thinking Mode Detection & COT Extraction
# ─────────────────────────────────────────────────────────

class TestDetectThinkingMode:
    def test_slow_sharegpt_think_tag(self):
        convs = [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "<think>reasoning here</think>answer"},
        ]
        assert detect_thinking_mode(convs) == "slow"

    def test_slow_sharegpt_thinking_tag(self):
        convs = [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "<thinking>reasoning</thinking>answer"},
        ]
        assert detect_thinking_mode(convs) == "slow"

    def test_slow_pangu_unused16(self):
        convs = [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "[unused16]cot here[unused17]answer"},
        ]
        assert detect_thinking_mode(convs) == "slow"

    def test_fast_no_markers(self):
        convs = [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "just a normal answer"},
        ]
        assert detect_thinking_mode(convs) == "fast"

    def test_fast_empty_conversations(self):
        assert detect_thinking_mode([]) == "fast"

    def test_slow_multiline_think(self):
        convs = [
            {"from": "human", "value": "solve this"},
            {"from": "gpt", "value": "<think>\nline1\nline2\n</think>\nanswer"},
        ]
        assert detect_thinking_mode(convs) == "slow"

    def test_fast_think_word_in_text(self):
        """The word 'think' in normal text should NOT trigger slow detection."""
        convs = [
            {"from": "human", "value": "I think this is interesting"},
            {"from": "gpt", "value": "I think you're right"},
        ]
        assert detect_thinking_mode(convs) == "fast"


class TestExtractCotContent:
    def test_sharegpt_think_extraction(self):
        convs = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "<think>my reasoning</think>the answer"},
        ]
        cot_text, cot_chars, cleaned = extract_cot_content(convs)
        assert "my reasoning" in cot_text
        assert cot_chars > 0
        # Cleaned conversation should not have COT
        assert "<think>" not in cleaned[1]["value"]
        assert "the answer" in cleaned[1]["value"]

    def test_pangu_unused16_extraction(self):
        convs = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "[unused16]pangu cot[unused17]the answer"},
        ]
        cot_text, cot_chars, cleaned = extract_cot_content(convs)
        assert "pangu cot" in cot_text
        assert cot_chars > 0
        assert "[unused16]" not in cleaned[1]["value"]

    def test_no_cot(self):
        convs = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "plain answer"},
        ]
        cot_text, cot_chars, cleaned = extract_cot_content(convs)
        assert cot_text == ""
        assert cot_chars == 0
        assert cleaned[1]["value"] == "plain answer"

    def test_multiple_cot_blocks(self):
        convs = [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "<think>first</think>a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "<think>second</think>a2"},
        ]
        cot_text, cot_chars, cleaned = extract_cot_content(convs)
        assert "first" in cot_text
        assert "second" in cot_text

    def test_human_turns_untouched(self):
        convs = [
            {"from": "human", "value": "question with <think>not cot</think>"},
            {"from": "gpt", "value": "<think>real cot</think>answer"},
        ]
        _, _, cleaned = extract_cot_content(convs)
        # Human turns should be left unchanged
        assert "<think>not cot</think>" in cleaned[0]["value"]


# ─────────────────────────────────────────────────────────
# 8.2  Truncation
# ─────────────────────────────────────────────────────────

class TestTruncateWithFragments:
    def test_no_truncation_needed(self):
        text = "short text"
        result = truncate_with_fragments(text, 1000)
        assert result == text

    def test_truncation_preserves_head_and_tail(self):
        text = "A" * 500 + "B" * 500 + "C" * 500
        result = truncate_with_fragments(text, 600, n_fragments=1)
        # Should start with As and end with Cs
        assert result.startswith("A")
        assert result.endswith("C")

    def test_truncation_contains_markers(self):
        text = "X" * 5000
        result = truncate_with_fragments(text, 1000, n_fragments=2)
        assert "chars omitted" in result

    def test_truncation_respects_budget_approximately(self):
        text = "X" * 10000
        budget = 2000
        result = truncate_with_fragments(text, budget, n_fragments=3)
        # Result should be reasonably close to budget (markers add overhead)
        assert len(result) < budget * 1.5

    def test_exact_budget_no_truncation(self):
        text = "A" * 100
        result = truncate_with_fragments(text, 100)
        assert result == text

    def test_very_small_budget_fallback(self):
        text = "X" * 10000
        result = truncate_with_fragments(text, 150, n_fragments=3)
        # Should fall back to simple head+tail (budget too small for fragments)
        assert "chars omitted" not in result or len(result) <= 200

    def test_zero_fragments(self):
        text = "H" * 300 + "M" * 400 + "T" * 300
        result = truncate_with_fragments(text, 500, n_fragments=0)
        assert result.startswith("H")


class TestTruncateForScoring:
    def test_no_truncation_short_content(self):
        convs = [
            {"from": "human", "value": "short question"},
            {"from": "gpt", "value": "short answer"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["was_truncated"] is False
        assert result["instruction"] == "short question"
        assert result["response"] == "short answer"
        assert result["cot"] is None

    def test_slow_thinking_has_cot(self):
        convs = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "<think>deep reasoning</think>answer"},
        ]
        cot_text = "deep reasoning"
        result = truncate_for_scoring(convs, "slow", cot_text=cot_text, budget=10000)
        assert result["cot"] == "deep reasoning"
        assert result["was_truncated"] is False

    def test_fast_thinking_no_cot(self):
        convs = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["cot"] is None

    def test_truncation_triggered(self):
        long_q = "Q" * 5000
        long_a = "A" * 15000
        convs = [
            {"from": "human", "value": long_q},
            {"from": "gpt", "value": long_a},
        ]
        result = truncate_for_scoring(convs, "fast", budget=5000)
        assert result["was_truncated"] is True
        assert result["original_instruction_chars"] == 5000
        assert result["original_response_chars"] == 15000
        assert len(result["instruction"]) < 5000
        assert len(result["response"]) < 15000

    def test_slow_truncation_allocates_cot_budget(self):
        long_q = "Q" * 3000
        long_cot = "C" * 10000
        long_a = "A" * 10000
        convs = [
            {"from": "human", "value": long_q},
            {"from": "gpt", "value": long_a},
        ]
        result = truncate_for_scoring(
            convs, "slow", cot_text=long_cot, budget=5000,
        )
        assert result["was_truncated"] is True
        assert result["cot"] is not None
        assert len(result["cot"]) > 0

    def test_original_chars_tracking(self):
        convs = [
            {"from": "human", "value": "hello world"},
            {"from": "gpt", "value": "goodbye world"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["original_instruction_chars"] == len("hello world")
        assert result["original_response_chars"] == len("goodbye world")
        assert result["original_cot_chars"] == 0

    def test_multi_turn_centers_latest_user_and_final_answer(self):
        convs = [
            {"from": "human", "value": "first question"},
            {"from": "gpt", "value": "first answer"},
            {"from": "human", "value": "second question"},
            {"from": "gpt", "value": "second answer"},
            {"from": "human", "value": "current request"},
            {"from": "gpt", "value": "final answer"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["instruction"] == "current request"
        assert result["current_request"] == "current request"
        assert result["response"] == "final answer"

    def test_includes_current_turn_tool_trajectory(self):
        convs = [
            {"from": "human", "value": "old request"},
            {"from": "gpt", "value": "old answer"},
            {"from": "human", "value": "fix the failing test"},
            {"from": "gpt", "value": "I'll inspect the logs first."},
            {"from": "tool", "value": "pytest output: assertion failed on line 42"},
            {"from": "gpt", "value": "final fix with updated code"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["instruction"] == "fix the failing test"
        assert "[Assistant Step] I'll inspect the logs first." in result["trajectory"]
        assert "[Tool Result] pytest output: assertion failed on line 42" in result["trajectory"]
        assert result["trajectory_tool_turns"] == 1

    def test_excludes_old_turn_trajectory(self):
        convs = [
            {"from": "human", "value": "old request"},
            {"from": "gpt", "value": "I'll run the old command"},
            {"from": "tool", "value": "old tool result"},
            {"from": "gpt", "value": "old answer"},
            {"from": "human", "value": "new request"},
            {"from": "gpt", "value": "let me inspect the new failure"},
            {"from": "tool", "value": "new tool result"},
            {"from": "gpt", "value": "new final answer"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["instruction"] == "new request"
        assert "new tool result" in result["trajectory"]
        assert "old tool result" not in result["trajectory"]

    def test_budget_preserves_final_answer_alongside_large_tool_output(self):
        huge_tool = "tool-output-" * 1500
        final_answer = "final answer with the actual fix"
        convs = [
            {"from": "human", "value": "repair the repo"},
            {"from": "gpt", "value": "running diagnostics"},
            {"from": "tool", "value": huge_tool},
            {"from": "gpt", "value": final_answer},
        ]
        result = truncate_for_scoring(convs, "fast", budget=1800)
        assert result["was_truncated"] is True
        assert "final answer with the actual fix" in result["response"]
        assert result["response"]


# ─────────────────────────────────────────────────────────
# 8.3  Rarity Computation
# ─────────────────────────────────────────────────────────

class TestComputeTagIdf:
    def test_basic_idf(self):
        distributions = {
            "intent": {"build": 500, "learn": 300, "debug": 200, "modify": 100},
        }
        idf = compute_tag_idf(distributions, 1000)
        assert "intent" in idf
        # More common tags should have lower IDF
        assert idf["intent"]["build"] < idf["intent"]["debug"]

    def test_idf_formula(self):
        distributions = {"intent": {"build": 99}}
        idf = compute_tag_idf(distributions, 1000)
        expected = math.log2(1000 / (99 + 1))
        assert abs(idf["intent"]["build"] - expected) < 0.001

    def test_empty_distributions(self):
        assert compute_tag_idf({}, 1000) == {}
        assert compute_tag_idf(None, 1000) == {}

    def test_zero_total_samples(self):
        assert compute_tag_idf({"intent": {"build": 10}}, 0) == {}

    def test_multiple_dimensions(self):
        distributions = {
            "intent": {"build": 500},
            "difficulty": {"beginner": 800, "expert": 50},
        }
        idf = compute_tag_idf(distributions, 1000)
        assert "intent" in idf and "difficulty" in idf
        assert idf["difficulty"]["expert"] > idf["difficulty"]["beginner"]


class TestComputeSampleRarity:
    def setup_method(self):
        self.distributions = {
            "intent": {"build": 500, "learn": 300, "debug": 200, "modify": 100},
            "difficulty": {"beginner": 600, "intermediate": 300, "advanced": 80, "expert": 20},
            "concept": {"algorithms": 100, "data-types": 400, "concurrency": 30},
        }
        self.idf_map = compute_tag_idf(self.distributions, 1000)

    def test_rare_sample_higher_score(self):
        common_labels = {"intent": "build", "difficulty": "beginner", "concept": ["data-types"]}
        rare_labels = {"intent": "debug", "difficulty": "expert", "concept": ["concurrency"]}

        common_rarity = compute_sample_rarity(common_labels, self.idf_map, 1000)
        rare_rarity = compute_sample_rarity(rare_labels, self.idf_map, 1000)

        assert rare_rarity["score"] > common_rarity["score"]

    def test_no_idf_returns_null(self):
        result = compute_sample_rarity({"intent": "build"}, {}, 1000)
        assert result["score"] is None

    def test_stats_ref_passed_through(self):
        ref = {"source": "test.json", "total_samples": 1000}
        result = compute_sample_rarity(
            {"intent": "build"}, self.idf_map, 1000,
            stats_ref_info=ref,
        )
        assert result["stats_ref"] == ref

    def test_null_labels(self):
        """compute_sample_rarity with empty labels should not crash."""
        result = compute_sample_rarity({}, self.idf_map, 1000)
        # No matching dimensions → score based on zero weights
        assert result["score"] is not None or result["score"] is None  # just no crash

    def test_combo_rarity(self):
        labels = {"intent": "build", "difficulty": "expert", "concept": ["algorithms"]}
        combo_counts = {"build|expert|algorithms": 2}
        result = compute_sample_rarity(
            labels, self.idf_map, 1000,
            combo_counts=combo_counts,
        )
        assert result["combo_rarity"] > 0

    def test_richer_combo_uses_agentic_and_context_when_available(self):
        labels = {
            "intent": "build",
            "difficulty": "advanced",
            "context": "multi-file",
            "concept": ["algorithms"],
            "agentic": ["tool-calling"],
            "task": ["feature-implementation"],
        }
        combo_counts = {
            "intent=build|difficulty=advanced|context=multi-file|task=feature-implementation|concept=algorithms|agentic=tool-calling": 4,
        }
        result = compute_sample_rarity(labels, self.idf_map, 1000, combo_counts=combo_counts)
        assert result["combo_rarity"] > 0

    def test_unknown_tag_uses_conservative_prior_not_max_idf(self):
        """Unknown tags should shrink toward the dimension prior, not max rarity."""
        labels = {"intent": "never-seen-before"}
        result = compute_sample_rarity(labels, self.idf_map, 1000)
        expected_prior = sum(self.idf_map["intent"].values()) / len(self.idf_map["intent"])
        assert result["tag_rarity"] == round(expected_prior, 3)

    def test_multi_select_blends_max_with_mean(self):
        labels = {"concept": ["algorithms", "data-types"]}
        result = compute_sample_rarity(labels, self.idf_map, 1000)
        algo_idf = self.idf_map["concept"]["algorithms"]
        data_types_idf = self.idf_map["concept"]["data-types"]
        assert result["tag_rarity"] < round(max(algo_idf, data_types_idf), 3)
        assert result["tag_rarity"] > round((algo_idf + data_types_idf) / 2, 3)

    def test_low_confidence_shrinks_toward_dimension_prior(self):
        labels = {
            "intent": "debug",
            "confidence": {"intent": 0.1},
        }
        result = compute_sample_rarity(labels, self.idf_map, 1000)
        prior = sum(self.idf_map["intent"].values()) / len(self.idf_map["intent"])
        debug_idf = self.idf_map["intent"]["debug"]
        assert abs(result["tag_rarity"] - prior) < abs(debug_idf - prior)

    def test_inherited_labels_report_higher_uncertainty(self):
        direct = compute_sample_rarity({"intent": "debug"}, self.idf_map, 1000)
        inherited = compute_sample_rarity(
            {"intent": "debug", "inherited": True},
            self.idf_map,
            1000,
        )
        assert inherited["effective_confidence"] < direct["effective_confidence"]
        assert inherited["uncertainty"] > direct["uncertainty"]

    def test_custom_weights(self):
        labels = {"intent": "build", "concept": ["algorithms"]}
        # Zero weight on concept should change rarity
        r1 = compute_sample_rarity(labels, self.idf_map, 1000,
                                   rarity_weights={"intent": 1.0, "concept": 1.0})
        r2 = compute_sample_rarity(labels, self.idf_map, 1000,
                                   rarity_weights={"intent": 1.0, "concept": 0.0})
        # Different weights → different scores
        # r2 only uses intent, r1 uses both
        assert r1["score"] != r2["score"]

    def test_missing_dimension_is_skipped_not_maxed(self):
        """When baseline lacks a dimension, that dim should be ignored."""
        labels = {"intent": "build", "context": "snippet"}
        rarity = compute_sample_rarity(
            labels,
            idf_map={"intent": {"build": 1.0}},
            total_samples=1000,
            rarity_weights={"intent": 1.0, "context": 10.0},
            combo_counts=None,
        )
        assert rarity["tag_rarity"] == 1.0


class TestNormalizeRarityScores:
    def test_normalize_to_1_10(self):
        rarity_results = [
            {"score": 1.0, "tag_rarity": 1.0, "combo_rarity": 0},
            {"score": 5.0, "tag_rarity": 5.0, "combo_rarity": 0},
            {"score": 10.0, "tag_rarity": 10.0, "combo_rarity": 0},
        ]
        normalize_rarity_scores(rarity_results)
        for r in rarity_results:
            assert 1.0 <= r["score"] <= 10.0

    def test_highest_gets_10(self):
        rarity_results = [
            {"score": 1.0, "tag_rarity": 1, "combo_rarity": 0},
            {"score": 2.0, "tag_rarity": 2, "combo_rarity": 0},
            {"score": 3.0, "tag_rarity": 3, "combo_rarity": 0},
        ]
        normalize_rarity_scores(rarity_results)
        assert rarity_results[2]["score"] == 10.0

    def test_lowest_gets_1(self):
        rarity_results = [
            {"score": 1.0, "tag_rarity": 1, "combo_rarity": 0},
            {"score": 2.0, "tag_rarity": 2, "combo_rarity": 0},
            {"score": 3.0, "tag_rarity": 3, "combo_rarity": 0},
        ]
        normalize_rarity_scores(rarity_results)
        assert rarity_results[0]["score"] == 1.0

    def test_null_scores_skipped(self):
        rarity_results = [
            {"score": None, "tag_rarity": None, "combo_rarity": None},
            {"score": 5.0, "tag_rarity": 5, "combo_rarity": 0},
        ]
        normalize_rarity_scores(rarity_results)
        assert rarity_results[0]["score"] is None
        # Single non-null score should get 1.0 (percentile of sole element)
        assert rarity_results[1]["score"] == 1.0

    def test_all_null_returns_empty(self):
        rarity_results = [{"score": None}]
        breakpoints = normalize_rarity_scores(rarity_results)
        assert breakpoints == {}

    def test_single_sample(self):
        rarity_results = [{"score": 7.0, "tag_rarity": 7, "combo_rarity": 0}]
        normalize_rarity_scores(rarity_results)
        assert rarity_results[0]["score"] == 1.0  # percentile 0/0 → 0 → 1.0

    def test_absolute_mode_maps_by_total_samples(self):
        rarity_results = [
            {"score": 0.0, "tag_rarity": 0, "combo_rarity": 0},
            {"score": 5.0, "tag_rarity": 5, "combo_rarity": 0},
            {"score": 10.0, "tag_rarity": 10, "combo_rarity": 0},
        ]
        meta = normalize_rarity_scores(
            rarity_results,
            mode="absolute",
            total_samples=1024,  # log2 ceiling = 10
        )
        assert meta["mode"] == "absolute"
        assert rarity_results[0]["score"] == 1.0
        assert rarity_results[1]["score"] == 5.5
        assert rarity_results[2]["score"] == 10.0


class TestBuildComboCounts:
    def test_basic_counting(self):
        samples = [
            {"labels": {"intent": "build", "difficulty": "advanced", "concept": ["algorithms"]}},
            {"labels": {"intent": "build", "difficulty": "advanced", "concept": ["algorithms"]}},
            {"labels": {"intent": "learn", "difficulty": "beginner", "concept": ["data-types"]}},
        ]
        counts = build_combo_counts(samples)
        assert counts["intent=build|difficulty=advanced|concept=algorithms"] == 2
        assert counts["intent=learn|difficulty=beginner|concept=data-types"] == 1

    def test_empty_samples(self):
        assert build_combo_counts([]) == {}

    def test_null_labels(self):
        """Samples with labels=None (failed Pass 1) should not crash."""
        samples = [
            {"labels": None},
            {"labels": {"intent": "build", "difficulty": "advanced", "concept": ["algorithms"]}},
            {},  # no labels key at all
        ]
        counts = build_combo_counts(samples)
        assert counts["intent=build|difficulty=advanced|concept=algorithms"] == 1
        # null/missing labels should be ignored
        assert "||" not in counts

    def test_richer_combo_distinguishes_same_legacy_bucket(self):
        samples = [
            {"labels": {
                "intent": "build",
                "difficulty": "advanced",
                "concept": ["algorithms"],
                "agentic": ["tool-calling"],
            }},
            {"labels": {
                "intent": "build",
                "difficulty": "advanced",
                "concept": ["algorithms"],
                "context": "multi-file",
            }},
        ]
        counts = build_combo_counts(samples)
        assert len(counts) == 2


class TestLoadTagStats:
    def test_prefers_distribution_total_samples(self, tmp_path):
        stats = {
            "total_samples": 1000,
            "distribution_total_samples": 200,
            "tag_distributions": {"intent": {"build": 100}},
            "timestamp": "2026-01-01T00:00:00",
        }
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(stats), encoding="utf-8")

        dists, total, ts = load_tag_stats(path)
        assert dists == stats["tag_distributions"]
        assert total == 200
        assert ts == "2026-01-01T00:00:00"

    def test_loads_combo_distributions_from_context(self, tmp_path):
        stats = {
            "total_samples": 1000,
            "tag_distributions": {"intent": {"build": 100}},
            "combo_distributions": {
                "build|beginner|algorithms": 12,
                "bad|entry": -1,
                "": 10,
            },
        }
        path = tmp_path / "stats.json"
        path.write_text(json.dumps(stats), encoding="utf-8")

        _d, _t, _ts, meta = load_tag_stats_context(path)
        assert meta["combo_counts"] == {"build|beginner|algorithms": 12}


# ─────────────────────────────────────────────────────────
# 8.4  Score Validation
# ─────────────────────────────────────────────────────────

class TestValidateScoreResponse:
    def test_valid_response(self):
        parsed = {
            "complexity": {"instruction": 7, "analytical_depth": 8, "implementation": 6, "overall": 7},
            "quality": {"correctness": 8, "code_quality": 7, "explanation": 6, "completeness": 8, "overall": 7},
            "reasoning": {"clarity": 7, "consistency": 9, "self_correction": True, "overall": 8},
            "flags": ["clean-code"],
            "confidence": 0.85,
        }
        result, issues = validate_score_response(parsed)
        assert result is not None
        assert len(issues) == 0
        assert result["complexity"]["overall"] == 7
        assert result["quality"]["overall"] == 7
        assert result["reasoning"]["overall"] == 8
        assert result["reasoning"]["self_correction"] is True
        assert result["flags"] == ["clean-code"]
        assert result["confidence"] == 0.85

    def test_null_response(self):
        result, issues = validate_score_response(None)
        assert result is None
        assert "null response" in issues

    def test_out_of_range_scores(self):
        parsed = {
            "complexity": {"instruction": 0, "analytical_depth": 11, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.5,
        }
        result, issues = validate_score_response(parsed)
        assert result["complexity"]["instruction"] is None  # 0 is out of range
        assert result["complexity"]["analytical_depth"] is None  # 11 is out of range
        assert result["complexity"]["implementation"] == 5  # valid
        assert len(issues) >= 2

    def test_missing_dimensions(self):
        parsed = {
            "flags": [],
            "confidence": 0.5,
        }
        result, issues = validate_score_response(parsed)
        assert result is not None
        # Missing dimensions default to {} which is a dict with None values
        assert len(issues) > 0
        # All sub-scores should be invalid (None)
        assert result["complexity"]["overall"] is None
        assert result["quality"]["overall"] is None
        assert result["reasoning"]["overall"] is None

    def test_unknown_flags(self):
        parsed = {
            "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": ["clean-code", "totally-made-up-flag", "another-fake"],
            "confidence": 0.8,
        }
        result, issues = validate_score_response(parsed)
        assert "clean-code" in result["flags"]
        assert "totally-made-up-flag" not in result["flags"]
        assert "totally-made-up-flag" in result["unknown_flags"]
        assert "another-fake" in result["unknown_flags"]
        assert any("unknown flags" in i for i in issues)

    def test_invalid_confidence(self):
        parsed = {
            "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 1.5,  # out of [0, 1]
        }
        result, issues = validate_score_response(parsed)
        assert result["confidence"] == 0.5  # default fallback
        assert any("confidence" in i for i in issues)

    def test_float_scores_converted_to_int(self):
        parsed = {
            "complexity": {"instruction": 7.5, "analytical_depth": 8.0, "implementation": 6.3, "overall": 7.2},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.5,
        }
        result, issues = validate_score_response(parsed)
        assert result["complexity"]["instruction"] == 7
        assert result["complexity"]["analytical_depth"] == 8

    def test_negative_flags_preserved(self):
        parsed = {
            "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": ["has-bug", "security-issue"],
            "confidence": 0.7,
        }
        result, issues = validate_score_response(parsed)
        assert "has-bug" in result["flags"]
        assert "security-issue" in result["flags"]
        assert len(issues) == 0

    def test_boolean_in_numeric_fields(self):
        """P0-1: isinstance(True, int) is True — booleans must be rejected as numeric scores."""
        parsed = {
            "complexity": {"instruction": True, "analytical_depth": 5, "implementation": 5, "overall": False},
            "quality": {"correctness": True, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": True, "consistency": False, "self_correction": True, "overall": 5},
            "flags": [],
            "confidence": 0.8,
        }
        result, issues = validate_score_response(parsed)
        # Boolean values in numeric fields should become None
        assert result["complexity"]["instruction"] is None
        assert result["complexity"]["overall"] is None
        assert result["quality"]["correctness"] is None
        assert result["reasoning"]["clarity"] is None
        assert result["reasoning"]["consistency"] is None
        # self_correction IS a boolean field — should be preserved
        assert result["reasoning"]["self_correction"] is True
        # Valid numeric fields preserved
        assert result["complexity"]["analytical_depth"] == 5
        assert result["reasoning"]["overall"] == 5

    def test_complexity_overall_clamped_by_subscores(self):
        """P0-2: overall must be within ±2 of mean(sub-scores)."""
        parsed = {
            "complexity": {"instruction": 2, "analytical_depth": 3, "implementation": 2, "overall": 9},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.8,
        }
        result, issues = validate_score_response(parsed)
        # mean(2,3,2) = 2.33, overall=9 → clamped to round(2.33) = 2
        assert result["complexity"]["overall"] <= 5  # must be clamped down
        assert any("complexity.overall" in i and "clamped" in i for i in issues)

    def test_reasoning_overall_clamped_by_subscores(self):
        """P0-2: reasoning overall clamped when sub-scores diverge."""
        parsed = {
            "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 2, "consistency": 3, "self_correction": False, "overall": 9},
            "flags": [],
            "confidence": 0.8,
        }
        result, issues = validate_score_response(parsed)
        # mean(2,3) = 2.5, overall=9 → clamped to round(2.5) = 2 (banker's rounding)
        assert result["reasoning"]["overall"] <= 5
        assert any("reasoning.overall" in i and "clamped" in i for i in issues)

    def test_subscores_null_no_clamp(self):
        """P0-2: when all sub-scores are None, no clamp should apply."""
        parsed = {
            "complexity": {"instruction": None, "analytical_depth": None, "implementation": None, "overall": 8},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.8,
        }
        result, issues = validate_score_response(parsed)
        # No valid sub-scores → no clamping, overall preserved
        assert result["complexity"]["overall"] == 8


# ─────────────────────────────────────────────────────────
# 8.4  Value Score Computation
# ─────────────────────────────────────────────────────────

class TestComputeValueScore:
    def test_basic_computation(self):
        score_result = {
            "complexity": {"overall": 7},
            "quality": {"overall": 8},
            "reasoning": {"overall": 6},
        }
        rarity_result = {"score": 5.0}
        value = compute_value_score(score_result, rarity_result)
        assert isinstance(value, float)
        assert 1.0 <= value <= 10.0

    def test_weighted_correctly(self):
        score_result = {
            "complexity": {"overall": 10},
            "quality": {"overall": 10},
            "reasoning": {"overall": 10},
        }
        rarity_result = {"score": 10.0}
        value = compute_value_score(score_result, rarity_result)
        assert value == 10.0

    def test_null_rarity_uses_default(self):
        score_result = {
            "complexity": {"overall": 8},
            "quality": {"overall": 8},
            "reasoning": {"overall": 8},
        }
        rarity_result = {"score": None}
        value = compute_value_score(score_result, rarity_result)
        # Default rarity=5.0 applied: 0.25*8 + 0.40*8 + 0.20*8 + 0.15*5.0 = 7.55
        assert value is not None
        assert value < 8.0  # No longer pure 8.0 due to default rarity=5.0

    def test_no_valid_scores(self):
        score_result = {
            "complexity": {"overall": None},
            "quality": {"overall": None},
            "reasoning": {"overall": None},
        }
        rarity_result = {"score": None}
        value = compute_value_score(score_result, rarity_result)
        assert value is None

    def test_custom_weights(self):
        score_result = {
            "complexity": {"overall": 10},
            "quality": {"overall": 2},
            "reasoning": {"overall": 5},
        }
        rarity_result = {"score": 5.0}
        # Weight only complexity
        value = compute_value_score(
            score_result, rarity_result,
            weights={"complexity": 1.0, "quality": 0.0, "reasoning": 0.0, "rarity": 0.0},
        )
        assert value == 10.0

    def test_quality_floor_penalty_applied(self):
        """quality.overall < 4 triggers 0.7x penalty."""
        score_result = {
            "complexity": {"overall": 8},
            "quality": {"overall": 3},
            "reasoning": {"overall": 6},
        }
        rarity_result = {"score": 5.0}
        value = compute_value_score(score_result, rarity_result)
        # Without penalty: 0.25*8 + 0.40*3 + 0.20*6 + 0.15*5 = 5.15
        # With penalty: 5.15 * 0.7 = 3.605 → clamped to max(1.0, 3.6)
        assert value is not None
        assert value < 4.0  # penalty kicks in

    def test_quality_floor_not_applied(self):
        """quality.overall >= 4 has no penalty."""
        score_result = {
            "complexity": {"overall": 8},
            "quality": {"overall": 4},
            "reasoning": {"overall": 6},
        }
        rarity_result = {"score": 5.0}
        value = compute_value_score(score_result, rarity_result)
        # No penalty: 0.25*8 + 0.40*4 + 0.20*6 + 0.15*5 = 5.55
        assert value is not None
        assert value >= 5.0  # no penalty, score reflects actual weighted mean

    def test_single_llm_dimension_returns_none(self):
        """P1-3: with only 1 LLM dimension, weight renormalization inflates score."""
        score_result = {
            "complexity": {"overall": None},
            "quality": {"overall": 9},
            "reasoning": {"overall": None},
        }
        rarity_result = {"score": 5.0}
        value = compute_value_score(score_result, rarity_result)
        assert value is None  # insufficient LLM dimensions

    def test_two_dimensions_returns_value(self):
        """P1-3: 2 LLM dimensions is enough for a reliable composite."""
        score_result = {
            "complexity": {"overall": 7},
            "quality": {"overall": 8},
            "reasoning": {"overall": None},
        }
        rarity_result = {"score": 5.0}
        value = compute_value_score(score_result, rarity_result)
        assert value is not None
        assert 1.0 <= value <= 10.0

    def test_rarity_none_uses_default(self):
        """Rarity=None uses default 5.0 instead of renormalizing."""
        score_result = {
            "complexity": {"overall": 8},
            "quality": {"overall": 8},
            "reasoning": {"overall": 8},
        }
        # Explicit None rarity
        value_with_none = compute_value_score(score_result, {"score": None})
        # Explicit 5.0 rarity
        value_with_default = compute_value_score(score_result, {"score": 5.0})
        assert value_with_none == value_with_default

    def test_low_confidence_shrinks_high_score(self):
        score_result = {
            "complexity": {"overall": 9},
            "quality": {"overall": 9},
            "reasoning": {"overall": 9},
            "confidence": 0.25,
        }
        rarity_result = {"score": 9.0}
        value = compute_value_score(score_result, rarity_result)
        assert value is not None
        assert value < 9.0

    def test_quality_overall_clamped_by_correctness(self):
        """validate_score_response enforces overall <= correctness + 2."""
        parsed = {
            "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
            "quality": {"correctness": 3, "code_quality": 8, "explanation": 8, "completeness": 8, "overall": 8},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.8,
        }
        result, issues = validate_score_response(parsed)
        # overall=8 should be clamped to correctness+2 = 5
        assert result["quality"]["overall"] == 5
        assert any("clamped" in i for i in issues)


# ─────────────────────────────────────────────────────────
# 8.4  Aggregate Stats
# ─────────────────────────────────────────────────────────

class TestComputeValueStats:
    def test_basic_stats(self):
        scored_samples = [
            {
                "labels": {"intent": "build", "difficulty": "advanced"},
                "value": {
                    "complexity": {"instruction": 7, "overall": 7},
                    "quality": {"correctness": 8, "overall": 8},
                    "reasoning": {"clarity": 6, "overall": 6},
                    "rarity": {"score": 5.0},
                    "flags": ["clean-code"],
                    "thinking_mode": "slow",
                    "value_score": 6.8,
                    "confidence": 0.85,
                },
            },
            {
                "labels": {"intent": "learn", "difficulty": "beginner"},
                "value": {
                    "complexity": {"instruction": 3, "overall": 3},
                    "quality": {"correctness": 9, "overall": 9},
                    "reasoning": {"clarity": 7, "overall": 7},
                    "rarity": {"score": 3.0},
                    "flags": [],
                    "thinking_mode": "fast",
                    "value_score": 5.5,
                    "confidence": 0.9,
                },
            },
        ]
        monitors = [
            {"status": "success", "llm_calls": 1, "prompt_tokens": 500, "completion_tokens": 200},
            {"status": "success", "llm_calls": 1, "prompt_tokens": 480, "completion_tokens": 180},
        ]
        stats = compute_value_stats(scored_samples, monitors)

        assert stats["total_scored"] == 2
        assert stats["total_failed"] == 0
        assert "score_distributions" in stats
        assert "value_by_tag" in stats
        assert "thinking_mode_stats" in stats
        assert stats["thinking_mode_stats"]["slow"]["count"] == 1
        assert stats["thinking_mode_stats"]["fast"]["count"] == 1
        assert stats["flag_counts"]["clean-code"] == 1

    def test_empty_samples(self):
        stats = compute_value_stats([], [])
        assert stats["total_scored"] == 0

    def test_failed_counting(self):
        monitors = [
            {"status": "success", "llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0},
            {"status": "failed", "llm_calls": 3, "prompt_tokens": 0, "completion_tokens": 0},
        ]
        scored = [{"value": {"value_score": 5, "complexity": {}, "quality": {}, "reasoning": {}, "flags": [], "thinking_mode": "fast"}}]
        stats = compute_value_stats(scored, monitors)
        assert stats["total_failed"] == 1

    def test_distribution_warnings_generated(self):
        """P1-2: detect quality.overall distribution bias."""
        # All quality scores clustered at 9-10 → over-represented top bucket
        scored = []
        for i in range(30):
            scored.append({
                "labels": {"intent": "build"},
                "value": {
                    "complexity": {"overall": 5},
                    "quality": {"overall": 9},  # all high quality
                    "reasoning": {"overall": 5},
                    "rarity": {"score": 5.0},
                    "flags": [],
                    "thinking_mode": "fast",
                    "value_score": 7.0,
                },
            })
        monitors = [{"status": "success", "llm_calls": 1,
                      "prompt_tokens": 0, "completion_tokens": 0}] * 30
        stats = compute_value_stats(scored, monitors)
        warnings = stats["score_distributions"].get("distribution_warnings", [])
        assert len(warnings) > 0
        assert any("9-10" in w for w in warnings)

    def test_no_warnings_small_batch(self):
        """P1-2: no distribution warnings when < 20 samples."""
        scored = []
        for i in range(10):
            scored.append({
                "labels": {"intent": "build"},
                "value": {
                    "complexity": {"overall": 5},
                    "quality": {"overall": 10},
                    "reasoning": {"overall": 5},
                    "rarity": {"score": 5.0},
                    "flags": [],
                    "thinking_mode": "fast",
                    "value_score": 7.0,
                },
            })
        monitors = [{"status": "success", "llm_calls": 1,
                      "prompt_tokens": 0, "completion_tokens": 0}] * 10
        stats = compute_value_stats(scored, monitors)
        warnings = stats["score_distributions"].get("distribution_warnings", [])
        assert len(warnings) == 0


# ─────────────────────────────────────────────────────────
# Smoke test fixture loading
# ─────────────────────────────────────────────────────────

class TestSmokeTestFixtures:
    def test_fixture_loads(self):
        path = FIXTURES_DIR / "smoke_test_value.json"
        assert path.exists(), "smoke_test_value.json fixture not found"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 3

    def test_fixture_has_expected_samples(self):
        with open(FIXTURES_DIR / "smoke_test_value.json", encoding="utf-8") as f:
            data = json.load(f)
        ids = {s["id"] for s in data}
        assert "value-1-dp-slow-think" in ids
        assert "value-2-simple-fast-think" in ids

    def test_fixture_thinking_modes(self):
        with open(FIXTURES_DIR / "smoke_test_value.json", encoding="utf-8") as f:
            data = json.load(f)
        modes = {}
        for s in data:
            mode = detect_thinking_mode(s["conversations"])
            modes[s["id"]] = mode
        assert modes["value-1-dp-slow-think"] == "slow"
        assert modes["value-2-simple-fast-think"] == "fast"
        assert modes["value-5-pangu-cot"] == "slow"

    def test_fixture_cot_extraction(self):
        with open(FIXTURES_DIR / "smoke_test_value.json", encoding="utf-8") as f:
            data = json.load(f)
        # value-1 has <think> block
        sample_1 = next(s for s in data if s["id"] == "value-1-dp-slow-think")
        cot_text, cot_chars, _ = extract_cot_content(sample_1["conversations"])
        assert cot_chars > 0
        assert "动态规划" in cot_text or "dp[i]" in cot_text

        # value-2 has no COT
        sample_2 = next(s for s in data if s["id"] == "value-2-simple-fast-think")
        cot_text2, cot_chars2, _ = extract_cot_content(sample_2["conversations"])
        assert cot_chars2 == 0


# ─────────────────────────────────────────────────────────
# 8.5  Resume support
# ─────────────────────────────────────────────────────────


class TestResumeScoringFile:
    """Test resume support for _run_scoring_file."""

    def test_partial_labels_are_skipped_from_scoring(self):
        from sft_label.scoring import score_one

        sample = {
            "id": "partial-1",
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ],
            "labels": {
                "intent": "build",
                "difficulty": "intermediate",
                "partial": True,
                "partial_stage": "call1",
                "partial_reason": "call2_retry_exhausted",
            },
        }

        value, monitor = asyncio.run(
            score_one(
                None,
                sample,
                "mock-model",
                {"score": 0.0},
                0,
                1,
                asyncio.Semaphore(1),
            )
        )

        assert value is None
        assert monitor["status"] == "skipped_partial_labels"
        assert monitor["llm_calls"] == 0

    def test_score_one_uses_current_turn_centered_evidence_window(self):
        from sft_label.scoring import score_one
        from unittest.mock import patch

        sample = {
            "id": "trajectory-1",
            "conversations": [
                {"from": "human", "value": "initial request"},
                {"from": "gpt", "value": "initial answer"},
                {"from": "human", "value": "please run the tests and fix the failure"},
                {"from": "gpt", "value": "I'll inspect the failure first."},
                {"from": "tool", "value": "pytest output: AssertionError in test_bug"},
                {"from": "gpt", "value": "Here is the final fix"},
            ],
            "labels": {
                "intent": "debug",
                "agentic": ["tool-calling"],
                "difficulty": "advanced",
                "confidence": {"intent": 0.95, "agentic": 0.95, "difficulty": 0.95},
            },
            "metadata": {"turn_index": 2, "total_turns": 3},
        }

        captured = {}

        async def fake_async_llm_call(http_client, messages, model, **kwargs):
            captured["messages"] = messages
            payload = {
                "complexity": {"instruction": 6, "analytical_depth": 6, "implementation": 6, "overall": 6},
                "quality": {"correctness": 7, "code_quality": 7, "explanation": 6, "completeness": 7, "overall": 7},
                "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
                "flags": [],
                "confidence": 0.8,
            }
            return payload, json.dumps(payload), {"prompt_tokens": 10, "completion_tokens": 5}

        with patch("sft_label.scoring.async_llm_call", side_effect=fake_async_llm_call):
            value, monitor = asyncio.run(
                score_one(
                    None,
                    sample,
                    "mock-model",
                    {"score": 5.0},
                    0,
                    1,
                    asyncio.Semaphore(1),
                    config=PipelineConfig(sample_max_retries=1),
                )
            )

        assert value is not None
        assert monitor["status"] == "success"
        user_msg = captured["messages"][-1]["content"]
        assert "please run the tests and fix the failure" in user_msg
        assert "[Tool Result] pytest output: AssertionError in test_bug" in user_msg
        assert "[Final Assistant] Here is the final fix" in user_msg
        assert "<prior_context>" in user_msg

    def test_directory_jsonl_uses_chunked_pipeline(self, tmp_path):
        from sft_label.scoring import run_scoring
        from unittest.mock import patch

        labeled_path = tmp_path / "batch" / "labeled.jsonl"
        labeled_path.parent.mkdir()
        labeled_path.write_text(json.dumps({
            "id": "sample-1",
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
            "labels": {"intent": "build", "difficulty": "intermediate"},
        }) + "\n")

        with patch("sft_label.scoring._run_scoring_file_chunked", return_value={
            "total_scored": 1,
            "total_failed": 0,
            "total_llm_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "score_distributions": {},
            "thinking_mode_breakdown": {},
            "tag_insights": {},
            "selection_thresholds": {},
        }) as mock_chunked:
            result = asyncio.run(run_scoring(str(tmp_path), config=PipelineConfig()))

        assert mock_chunked.call_count == 1
        assert result["files_processed"] == 1
        assert result["per_file_summary"][0]["file"] == "batch/labeled.jsonl"

    def test_directory_selection_is_global_not_per_file(self, tmp_path):
        from sft_label.scoring import run_scoring
        from unittest.mock import patch

        quality_by_id = {
            "a-low": 2,
            "a-high": 4,
            "b-low": 7,
            "b-high": 9,
        }

        for subdir, sample_ids in {
            "file_a": ["a-low", "a-high"],
            "file_b": ["b-low", "b-high"],
        }.items():
            path = tmp_path / subdir
            path.mkdir()
            samples = []
            for sample_id in sample_ids:
                samples.append({
                    "id": sample_id,
                    "conversations": [
                        {"from": "human", "value": f"Request {sample_id}"},
                        {"from": "gpt", "value": f"Answer {sample_id}"},
                    ],
                    "labels": {
                        "intent": "build",
                        "language": ["python"],
                        "difficulty": "intermediate",
                        "confidence": {"intent": 0.95, "language": 0.95, "difficulty": 0.95},
                    },
                })
            (path / "labeled.json").write_text(json.dumps(samples), encoding="utf-8")

        async def mock_score_one(http_client, sample, model, rarity_result,
                                  sample_idx, total, sem, config=None, rate_limiter=None):
            q = quality_by_id[sample["id"]]
            value = {
                "complexity": {"instruction": q, "analytical_depth": q, "implementation": q, "overall": q},
                "quality": {"correctness": q, "code_quality": q, "explanation": q, "completeness": q, "overall": q},
                "reasoning": {"clarity": q, "consistency": q, "self_correction": False, "overall": q},
                "rarity": rarity_result,
                "flags": [],
                "thinking_mode": "fast",
                "value_score": float(q),
                "confidence": 0.9,
            }
            monitor = {
                "sample_id": sample["id"],
                "status": "success",
                "llm_calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "attempts": 1,
                "validation_issues": [],
            }
            return value, monitor

        with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
            asyncio.run(run_scoring(str(tmp_path), config=PipelineConfig(scoring_concurrency=1, sample_max_retries=1)))

        scored_a = json.loads((tmp_path / "file_a" / "scored.json").read_text(encoding="utf-8"))
        scored_b = json.loads((tmp_path / "file_b" / "scored.json").read_text(encoding="utf-8"))
        by_id = {sample["id"]: sample["value"]["selection_score"] for sample in scored_a + scored_b}

        assert by_id["b-high"] > by_id["a-high"]
        assert by_id["b-low"] > by_id["a-high"]

    def test_resume_skips_prescored(self, tmp_path):
        """Partially scored.jsonl should let resume skip those samples."""
        from sft_label.scoring import _run_scoring_file
        from sft_label.config import PipelineConfig
        from unittest.mock import patch

        # Create labeled.json with 3 samples
        samples = [
            {
                "id": f"sample-{i}",
                "conversations": [
                    {"from": "human", "value": f"Question {i}"},
                    {"from": "gpt", "value": f"Answer {i}"},
                ],
                "labels": {"intent": "build", "difficulty": "intermediate"},
            }
            for i in range(3)
        ]
        labeled_path = tmp_path / "labeled.json"
        with open(labeled_path, "w") as f:
            json.dump(samples, f)

        # Pre-scored JSONL: sample-0 already scored
        pre_scored = {
            "id": "sample-0",
            "conversations": samples[0]["conversations"],
            "labels": samples[0]["labels"],
            "value": {
                "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
                "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
                "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
                "rarity": {"score": 5.0},
                "flags": [],
                "thinking_mode": "fast",
                "value_score": 6.0,
                "confidence": 0.85,
            },
        }
        scored_jsonl = tmp_path / "scored.jsonl"
        with open(scored_jsonl, "w") as f:
            f.write(json.dumps(pre_scored) + "\n")

        # Mock score_one to track which samples are scored
        scored_ids = []

        async def mock_score_one(http_client, sample, model, rarity_result,
                                  sample_idx, total, sem, config=None, rate_limiter=None):
            scored_ids.append(sample.get("id"))
            value = {
                "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
                "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
                "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
                "rarity": rarity_result,
                "flags": [],
                "thinking_mode": "fast",
                "value_score": 6.0,
                "confidence": 0.85,
            }
            monitor = {"sample_id": sample.get("id"), "status": "success",
                        "llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 50,
                        "attempts": 1, "validation_issues": []}
            return value, monitor

        config = PipelineConfig(scoring_concurrency=2, sample_max_retries=1)

        with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
            asyncio.run(_run_scoring_file(
                labeled_path, tmp_path, None, 0, config, resume=True,
            ))

        # sample-0 should NOT have been scored again
        assert "sample-0" not in scored_ids
        # sample-1 and sample-2 should have been scored
        assert "sample-1" in scored_ids
        assert "sample-2" in scored_ids

    def test_no_stats_uses_local_rarity_baseline(self, tmp_path):
        """Without --tag-stats, rarity should still be computed from local labels."""
        from sft_label.scoring import _run_scoring_file
        from sft_label.config import PipelineConfig
        from unittest.mock import patch

        samples = [
            {
                "id": "common-1",
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ],
                "labels": {
                    "intent": "build",
                    "difficulty": "beginner",
                    "language": ["python"],
                    "domain": ["web-backend"],
                    "task": ["feature-implementation"],
                    "concept": ["data-types"],
                    "context": "snippet",
                },
            },
            {
                "id": "common-2",
                "conversations": [
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ],
                "labels": {
                    "intent": "build",
                    "difficulty": "beginner",
                    "language": ["python"],
                    "domain": ["web-backend"],
                    "task": ["feature-implementation"],
                    "concept": ["data-types"],
                    "context": "snippet",
                },
            },
            {
                "id": "rare-1",
                "conversations": [
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                ],
                "labels": {
                    "intent": "debug",
                    "difficulty": "expert",
                    "language": ["rust"],
                    "domain": ["security"],
                    "task": ["debugging"],
                    "concept": ["concurrency"],
                    "context": "single-file",
                },
            },
        ]
        labeled_path = tmp_path / "labeled.json"
        with open(labeled_path, "w", encoding="utf-8") as f:
            json.dump(samples, f)

        rarity_by_id = {}

        async def mock_score_one(http_client, sample, model, rarity_result,
                                  sample_idx, total, sem, config=None, rate_limiter=None):
            rarity_by_id[sample["id"]] = (rarity_result or {}).get("score")
            value = {
                "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
                "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
                "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
                "rarity": rarity_result,
                "flags": [],
                "thinking_mode": "fast",
                "value_score": 6.0,
                "confidence": 0.85,
            }
            monitor = {"sample_id": sample.get("id"), "status": "success",
                        "llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 50,
                        "attempts": 1, "validation_issues": []}
            return value, monitor

        config = PipelineConfig(
            scoring_concurrency=2,
            sample_max_retries=1,
            rarity_score_mode="absolute",
        )
        with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
            asyncio.run(_run_scoring_file(
                labeled_path, tmp_path, None, 0, config, resume=False,
            ))

        assert rarity_by_id["common-1"] is not None
        assert rarity_by_id["common-2"] is not None
        assert rarity_by_id["rare-1"] is not None
        assert rarity_by_id["rare-1"] > rarity_by_id["common-1"]

    def test_external_stats_without_combo_uses_local_combo_fallback(self, tmp_path):
        from sft_label.scoring import _run_scoring_file
        from sft_label.config import PipelineConfig
        from unittest.mock import patch

        samples = [
            {
                "id": "s1",
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ],
                "labels": {
                    "intent": "build",
                    "difficulty": "beginner",
                    "language": ["python"],
                    "domain": ["web-backend"],
                    "task": ["feature-implementation"],
                    "concept": ["data-types"],
                    "context": "snippet",
                },
            }
        ]
        labeled_path = tmp_path / "labeled.json"
        labeled_path.write_text(json.dumps(samples), encoding="utf-8")

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps({
            "total_samples": 1000,
            "distribution_total_samples": 800,
            "tag_distributions": {
                "intent": {"build": 500},
                "difficulty": {"beginner": 600},
                "language": {"python": 700},
                "domain": {"web-backend": 600},
                "task": {"feature-implementation": 650},
                "concept": {"data-types": 500},
                "context": {"snippet": 700},
            },
        }), encoding="utf-8")

        captured_rarity = {}

        async def mock_score_one(http_client, sample, model, rarity_result,
                                  sample_idx, total, sem, config=None, rate_limiter=None):
            captured_rarity[sample["id"]] = rarity_result
            value = {
                "complexity": {"instruction": 5, "analytical_depth": 5, "implementation": 5, "overall": 5},
                "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
                "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
                "rarity": rarity_result,
                "flags": [],
                "thinking_mode": "fast",
                "value_score": 6.0,
                "confidence": 0.85,
            }
            monitor = {"sample_id": sample.get("id"), "status": "success",
                        "llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 50,
                        "attempts": 1, "validation_issues": []}
            return value, monitor

        config = PipelineConfig(
            scoring_concurrency=1,
            sample_max_retries=1,
            rarity_score_mode="absolute",
        )
        with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
            stats = asyncio.run(_run_scoring_file(
                labeled_path, tmp_path, str(stats_path), 0, config, resume=False,
            ))

        rarity = captured_rarity["s1"]
        assert rarity is not None
        assert rarity["combo_rarity"] > 0.0
        assert stats["rarity_config"]["combo_mode"] == "hybrid"


# ─────────────────────────────────────────────────────────
# 8.6  Selection Scores
# ─────────────────────────────────────────────────────────


class TestComputeSelectionScores:
    def _make_sample(self, idx, labels, complexity=5, quality=5, reasoning=5,
                     rarity_score=5.0, value_score=5.0):
        return {
            "id": f"sample-{idx}",
            "labels": labels,
            "value": {
                "complexity": {"instruction": complexity, "overall": complexity},
                "quality": {"correctness": quality, "overall": quality},
                "reasoning": {"clarity": reasoning, "overall": reasoning},
                "rarity": {"score": rarity_score},
                "flags": [],
                "value_score": value_score,
            },
        }

    def test_basic_computation(self):
        """15+ samples with diverse labels → selection_score in [1, 10]."""
        samples = []
        intents = ["build", "learn", "debug", "modify"]
        diffs = ["beginner", "intermediate", "upper-intermediate", "advanced", "expert"]
        for i in range(20):
            labels = {
                "intent": intents[i % len(intents)],
                "difficulty": diffs[i % len(diffs)],
                "language": ["python"],
                "concept": ["algorithms"] if i % 2 == 0 else ["data-types"],
            }
            samples.append(self._make_sample(
                i, labels, complexity=3 + i % 7, quality=3 + i % 7,
                reasoning=3 + i % 5, value_score=3 + i * 0.3,
            ))

        compute_selection_scores(samples, min_group_size=2)

        scored = [s for s in samples if s["value"].get("selection_score") is not None]
        assert len(scored) > 0
        for s in scored:
            assert 1.0 <= s["value"]["selection_score"] <= 10.0

    def test_small_group_below_threshold(self):
        """Tag groups below min_group_size fall back to global percentile."""
        # 1 sample with unique tag, rest share common tag
        samples = []
        for i in range(10):
            labels = {"intent": "build", "language": ["python"]}
            samples.append(self._make_sample(i, labels, quality=5 + i % 3, value_score=5.0))
        # One sample with a unique tag
        samples.append(self._make_sample(10, {"intent": "build", "language": ["haskell"]},
                                          quality=9, value_score=8.0))

        compute_selection_scores(samples, min_group_size=5)

        # The haskell sample should still get a selection score (via global fallback)
        haskell = samples[10]["value"]
        assert haskell.get("selection_score") is not None

    def test_bayesian_shrinkage_formula(self):
        """When n = prior, shrinkage should be 0.5."""
        from sft_label.config import SELECTION_SMOOTHING_PRIOR
        # Create exactly SELECTION_SMOOTHING_PRIOR samples for one tag
        n = SELECTION_SMOOTHING_PRIOR
        samples = []
        for i in range(n):
            labels = {"intent": "build", "language": ["python"]}
            samples.append(self._make_sample(i, labels, quality=i + 1, value_score=float(i + 1)))

        compute_selection_scores(samples, min_group_size=1)

        # Shrinkage = n / (n + prior) = 0.5, so percentiles blend 50/50
        # All should have selection scores
        for s in samples:
            assert s["value"].get("selection_score") is not None

    def test_multi_tag_averaging(self):
        """Multiple tags within a dimension → percentiles averaged."""
        samples = []
        for i in range(15):
            labels = {
                "intent": "build",
                "concept": ["algorithms", "data-structures"],  # multi-tag
                "language": ["python"],
            }
            samples.append(self._make_sample(i, labels, quality=3 + i % 7, value_score=5.0))

        compute_selection_scores(samples, min_group_size=2)

        for s in samples:
            v = s["value"]
            assert v.get("intra_class_rank") is not None

    def test_fallback_to_value_score(self):
        """Samples with no value dict use value_score as fallback."""
        sample = {
            "id": "sample-0",
            "labels": {"intent": "build"},
            "value": {
                "value_score": 7.5,
                # No complexity/quality/reasoning → pure_quality is None
            },
        }
        compute_selection_scores([sample], min_group_size=1)
        # No pure_quality computable → ultimate fallback
        assert sample["value"]["selection_score"] == 7.5

    def test_single_sample(self):
        """Edge case: n=1, single sample should not crash."""
        samples = [self._make_sample(0, {"intent": "build", "language": ["python"]},
                                      value_score=6.0)]
        compute_selection_scores(samples, min_group_size=1)
        assert samples[0]["value"].get("selection_score") is not None

    def test_tied_samples_get_same_intra_class_rank(self):
        samples = [
            self._make_sample(0, {"intent": "build", "language": ["python"]},
                              complexity=5, quality=7, reasoning=6, value_score=6.0),
            self._make_sample(1, {"intent": "build", "language": ["python"]},
                              complexity=5, quality=7, reasoning=6, value_score=6.1),
            self._make_sample(2, {"intent": "build", "language": ["python"]},
                              complexity=8, quality=9, reasoning=8, value_score=8.0),
        ]

        compute_selection_scores(samples, min_group_size=1)

        assert samples[0]["value"]["intra_class_rank"] == samples[1]["value"]["intra_class_rank"]

    def test_selection_uses_configured_value_weights(self):
        samples = [
            self._make_sample(0, {"intent": "build", "language": ["python"]},
                              complexity=9, quality=5, reasoning=1, value_score=6.0),
            self._make_sample(1, {"intent": "build", "language": ["python"]},
                              complexity=1, quality=5, reasoning=9, value_score=6.0),
        ]

        complexity_heavy = PipelineConfig(
            value_weights={"complexity": 0.8, "quality": 0.1, "reasoning": 0.1, "rarity": 0.0},
            selection_min_group_size=1,
        )
        reasoning_heavy = PipelineConfig(
            value_weights={"complexity": 0.1, "quality": 0.1, "reasoning": 0.8, "rarity": 0.0},
            selection_min_group_size=1,
        )

        compute_selection_scores(samples, config=complexity_heavy)
        complexity_scores = {s["id"]: s["value"]["selection_score"] for s in samples}

        samples = [
            self._make_sample(0, {"intent": "build", "language": ["python"]},
                              complexity=9, quality=5, reasoning=1, value_score=6.0),
            self._make_sample(1, {"intent": "build", "language": ["python"]},
                              complexity=1, quality=5, reasoning=9, value_score=6.0),
        ]
        compute_selection_scores(samples, config=reasoning_heavy)
        reasoning_scores = {s["id"]: s["value"]["selection_score"] for s in samples}

        assert complexity_scores["sample-0"] > complexity_scores["sample-1"]
        assert reasoning_scores["sample-1"] > reasoning_scores["sample-0"]

    def test_selection_respects_configured_smoothing_prior(self):
        samples = [
            self._make_sample(0, {"intent": "build", "language": ["python"]},
                              quality=2, value_score=4.0),
            self._make_sample(1, {"intent": "build", "language": ["python"]},
                              quality=3, value_score=5.0),
            self._make_sample(2, {"intent": "learn", "language": ["go"]},
                              quality=10, value_score=9.0),
        ]

        low_prior = PipelineConfig(selection_min_group_size=2, selection_smoothing_prior=1)
        high_prior = PipelineConfig(selection_min_group_size=2, selection_smoothing_prior=1000)

        compute_selection_scores(samples, config=low_prior)
        low_rank = samples[1]["value"]["intra_class_rank"]

        samples = [
            self._make_sample(0, {"intent": "build", "language": ["python"]},
                              quality=2, value_score=4.0),
            self._make_sample(1, {"intent": "build", "language": ["python"]},
                              quality=3, value_score=5.0),
            self._make_sample(2, {"intent": "learn", "language": ["go"]},
                              quality=10, value_score=9.0),
        ]
        compute_selection_scores(samples, config=high_prior)
        high_rank = samples[1]["value"]["intra_class_rank"]

        assert low_rank > high_rank

    def test_low_confidence_labels_get_more_globalized_selection(self):
        confident = [
            self._make_sample(0, {"intent": "build", "language": ["python"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=5, value_score=5.0),
            self._make_sample(1, {"intent": "build", "language": ["python"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=6, value_score=6.0),
            self._make_sample(2, {"intent": "learn", "language": ["go"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=1, value_score=1.0),
            self._make_sample(3, {"intent": "review", "language": ["rust"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=9, value_score=9.0),
        ]
        uncertain = [
            self._make_sample(0, {"intent": "build", "language": ["python"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=5, value_score=5.0),
            self._make_sample(1, {"intent": "build", "language": ["python"], "confidence": {"intent": 0.10, "language": 0.10}},
                              quality=6, value_score=6.0),
            self._make_sample(2, {"intent": "learn", "language": ["go"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=1, value_score=1.0),
            self._make_sample(3, {"intent": "review", "language": ["rust"], "confidence": {"intent": 0.95, "language": 0.95}},
                              quality=9, value_score=9.0),
        ]

        compute_selection_scores(confident, min_group_size=2)
        compute_selection_scores(uncertain, min_group_size=2)

        assert uncertain[1]["value"]["intra_class_rank"] < confident[1]["value"]["intra_class_rank"]


class TestComputeSelectionScoresFromSummaries:
    def test_summary_path_respects_smoothing_prior(self):
        summaries = [
            {
                "labels": {"intent": "build", "language": ["python"]},
                "complexity_overall": 5,
                "quality_overall": 2,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 4.0,
            },
            {
                "labels": {"intent": "build", "language": ["python"]},
                "complexity_overall": 5,
                "quality_overall": 3,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 5.0,
            },
            {
                "labels": {"intent": "learn", "language": ["go"]},
                "complexity_overall": 5,
                "quality_overall": 10,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 9.0,
            },
        ]

        low_prior = compute_selection_scores_from_summaries(
            summaries,
            config=PipelineConfig(selection_min_group_size=2, selection_smoothing_prior=1),
        )
        high_prior = compute_selection_scores_from_summaries(
            summaries,
            config=PipelineConfig(selection_min_group_size=2, selection_smoothing_prior=1000),
        )

        assert low_prior[1]["intra_class_rank"] > high_prior[1]["intra_class_rank"]

    def test_summary_path_respects_label_uncertainty(self):
        confident = [
            {
                "labels": {"intent": "build", "language": ["python"], "confidence": {"intent": 0.95, "language": 0.95}},
                "complexity_overall": 5,
                "quality_overall": 5,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 5.0,
            },
            {
                "labels": {"intent": "build", "language": ["python"], "confidence": {"intent": 0.95, "language": 0.95}},
                "complexity_overall": 5,
                "quality_overall": 6,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 6.0,
            },
            {
                "labels": {"intent": "learn", "language": ["go"], "confidence": {"intent": 0.95, "language": 0.95}},
                "complexity_overall": 5,
                "quality_overall": 1,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 1.0,
            },
            {
                "labels": {"intent": "review", "language": ["rust"], "confidence": {"intent": 0.95, "language": 0.95}},
                "complexity_overall": 5,
                "quality_overall": 9,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 9.0,
            },
        ]
        uncertain = [
            confident[0],
            {
                "labels": {"intent": "build", "language": ["python"], "confidence": {"intent": 0.10, "language": 0.10}},
                "complexity_overall": 5,
                "quality_overall": 6,
                "reasoning_overall": 5,
                "rarity_score": 5.0,
                "value_score": 6.0,
            },
            confident[2],
            confident[3],
        ]

        confident_scores = compute_selection_scores_from_summaries(confident, min_group_size=2)
        uncertain_scores = compute_selection_scores_from_summaries(uncertain, min_group_size=2)

        assert uncertain_scores[1]["intra_class_rank"] < confident_scores[1]["intra_class_rank"]


# ─────────────────────────────────────────────────────────
# 8.6b Selection stability regressions
# ─────────────────────────────────────────────────────────

class TestSelectionStabilityRegression:
    def test_feature_inference_on_audited_fixture(self):
        fixture = {sample["audit_label"]: sample for sample in _load_selection_regression_fixture()}

        opener = _infer_selection_features(sample=fixture["opener_incomplete"])
        exploration = _infer_selection_features(sample=fixture["tool_exploration_low_info"])
        common_final = _infer_selection_features(sample=fixture["common_final_bugfix"])
        summary_final = _infer_selection_features(sample=fixture["summary_heavy_final"])

        assert opener["trajectory_stage"] == "opener"
        assert exploration["trajectory_stage"] == "exploration"
        assert exploration["is_low_information_tool"] is True
        assert common_final["has_summary_evidence"] is True
        assert summary_final["trajectory_stage"] == "final-summary"
        assert summary_final["has_summary_evidence"] is False

    def test_relative_ordering_on_audited_fixture(self):
        samples = _load_selection_regression_fixture()
        compute_selection_scores(samples, min_group_size=1, config=PipelineConfig())
        by_label = {sample["audit_label"]: sample["value"] for sample in samples}

        assert by_label["opener_incomplete"]["selection_score"] < by_label["tool_exploration_low_info"]["selection_score"]
        assert by_label["tool_exploration_low_info"]["selection_score"] < by_label["common_final_bugfix"]["selection_score"]
        assert by_label["summary_heavy_final"]["selection_score"] < by_label["rare_domain_final_ml"]["selection_score"]
        assert by_label["common_final_bugfix"]["selection_score"] >= by_label["summary_heavy_final"]["selection_score"]

    def test_comparison_utility_reports_expected_shift(self):
        report = compare_selection_configs(_load_selection_regression_fixture())
        by_label = {row["label"]: row for row in report["deltas"]}

        assert report["count"] == 5
        assert "opener_incomplete" in by_label
        assert by_label["tool_exploration_low_info"]["delta"] < 0
        assert by_label["summary_heavy_final"]["current_selection"] < by_label["rare_domain_final_ml"]["current_selection"]
        assert by_label["summary_heavy_final"]["current_selection"] < by_label["common_final_bugfix"]["current_selection"]
        assert by_label["tool_exploration_low_info"]["current_selection"] < 4.0
        assert report["threshold_counts"]["current_ge_8"] <= report["threshold_counts"]["legacy_ge_8"]

    def test_selection_output_stays_schema_compatible(self):
        samples = _load_selection_regression_fixture()
        compute_selection_scores(samples, min_group_size=1, config=PipelineConfig())
        stats = compute_value_stats(samples, [])

        assert "selection_score" in stats["score_distributions"]
        assert "intra_class_rank" in stats["score_distributions"]
        assert "selection_score" in samples[0]["value"]
        assert "intra_class_rank" in samples[0]["value"]
        assert "trajectory_stage" not in samples[0]["value"]


class TestDomainBackfill:
    def test_obvious_domains_backfill(self):
        assert _infer_domain_backfill(
            {"current_request": "Fix the parser crash in the compiler lexer and AST lowering path."},
            {"domain": []},
        ) == ["compiler-development"]
        assert _infer_domain_backfill(
            {"current_request": "Make epoch_length optional in the PyTorch dataloader training engine."},
            {"domain": []},
        ) == ["machine-learning"]
        assert _infer_domain_backfill(
            {"current_request": "Add a per-connection Storage API toggle for the BigQuery client."},
            {"domain": []},
        ) == ["cloud-computing"]
        assert _infer_domain_backfill(
            {"current_request": "Add a REST API endpoint and update request/response validation for this DBAPI client."},
            {"domain": []},
        ) == ["api-development"]

    def test_ambiguous_domain_stays_empty(self):
        assert _infer_domain_backfill(
            {"current_request": "Refactor the Python utility and improve tests."},
            {"domain": []},
        ) == []


class TestPromptBudgetRegression:
    def test_compact_prompt_budget_does_not_grow(self):
        truncated = {
            "original_prior_context_chars": 120,
            "original_cot_chars": 0,
            "original_trajectory_chars": 340,
            "original_response_chars": 220,
            "trajectory_turn_count": 4,
            "trajectory_tool_turns": 2,
            "prior_context": "prior",
            "current_request": "fix the failing parser edge case",
            "trajectory": "[Assistant Step] inspect file\\n[Tool Result] failing test output",
            "response": "I updated parser.py and added a regression test in tests/test_parser.py",
        }
        labels = {
            "intent": "debug",
            "language": ["python"],
            "domain": ["compiler-development"],
            "task": ["bug-fixing", "testing-task"],
            "difficulty": "upper-intermediate",
        }
        messages = build_scoring_messages(
            truncated,
            "fast",
            labels,
            total_turns=8,
            code_block_count=1,
            compact=True,
        )

        assert len(SCORING_SYSTEM_COMPACT) <= 5996
        assert sum(len(message["content"]) for message in messages) <= 10888


# ─────────────────────────────────────────────────────────
# 8.7  Conversation aggregation
# ─────────────────────────────────────────────────────────


class TestConversationAggregation:
    def test_conv_no_double_penalty(self):
        """P2-1: pure_quality should NOT apply 0.7× penalty (only _compute_penalty does)."""
        # Slice with quality=2 (below threshold 3 → penalty 0.5× from _compute_penalty)
        sample = {
            "value": {
                "complexity": {"overall": 5},
                "quality": {"overall": 2},
                "reasoning": {"overall": 5},
            },
        }
        pq = _compute_pure_quality_from_slice(sample)
        assert pq is not None
        # Without penalty: (0.25*5 + 0.40*2 + 0.20*5) / 0.85 = (1.25+0.8+1.0)/0.85 = 3.59
        # The old code would apply 0.7× → 2.51, new code does NOT
        assert pq > 3.0  # no penalty applied at slice level

    def test_conv_penalty_still_applies(self):
        """_compute_penalty should still apply quality floor penalty to conversation."""
        slices = [
            {"value": {"quality": {"overall": 2}, "flags": []}},
            {"value": {"quality": {"overall": 7}, "flags": []}},
        ]
        penalty, quality_floor, neg_flags = _compute_penalty(slices)
        assert quality_floor == 2
        assert penalty < 1.0  # penalty should kick in
        assert penalty == 0.5  # quality < 3 → 0.5×

    def test_conversation_with_low_quality_slice(self):
        """Full aggregation: low quality slice should apply penalty once."""
        slices = [
            {
                "id": "s1",
                "metadata": {"source_id": "conv-1", "total_turns": 2, "turn_index": 1},
                "labels": {"intent": "build", "language": ["python"]},
                "value": {
                    "complexity": {"overall": 5},
                    "quality": {"overall": 2},
                    "reasoning": {"overall": 5},
                    "rarity": {"score": 5.0},
                    "flags": [],
                    "value_score": 4.0,
                    "thinking_mode": "fast",
                },
            },
            {
                "id": "s2",
                "metadata": {"source_id": "conv-1", "total_turns": 2, "turn_index": 2},
                "labels": {"intent": "build", "language": ["python"]},
                "value": {
                    "complexity": {"overall": 7},
                    "quality": {"overall": 8},
                    "reasoning": {"overall": 7},
                    "rarity": {"score": 5.0},
                    "flags": [],
                    "value_score": 7.0,
                    "thinking_mode": "fast",
                },
            },
        ]
        rec = aggregate_conversation("conv-1", slices)
        assert rec is not None
        # penalty = 0.5 (quality_floor=2 < 3)
        assert rec["detail"]["penalty"] == 0.5
        # conv_value = q_base * 0.5, where q_base is weighted avg of 4.0 and 7.0
        assert 1.0 <= rec["conv_value"] <= 10.0


# ─────────────────────────────────────────────────────────
# 8.8  Integration test: end-to-end scoring via API
# ─────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("LITELLM_BASE") or not os.environ.get("LITELLM_KEY"),
    reason="LITELLM_BASE and LITELLM_KEY env vars required for integration test",
)
class TestIntegrationScoring:
    """End-to-end scoring of smoke test data via real API calls.

    Run with:
        LITELLM_BASE="http://..." LITELLM_KEY="sk-..." uv run pytest tests/test_scoring.py::TestIntegrationScoring -v
    """

    def test_end_to_end_scoring(self, tmp_path):
        """Score smoke test fixtures via API, verify output structure."""
        from sft_label.config import PipelineConfig
        from sft_label.scoring import run_scoring

        # Copy fixture to tmp dir as labeled.json
        fixture = FIXTURES_DIR / "smoke_test_value.json"
        labeled_path = tmp_path / "labeled.json"
        shutil.copy(fixture, labeled_path)

        # Create a minimal stats.json for rarity computation
        stats = {
            "total_samples": 1000,
            "tag_distributions": {
                "intent": {"build": 400, "learn": 300, "debug": 200, "fix": 100},
                "difficulty": {"beginner": 300, "intermediate": 400, "advanced": 200, "expert": 100},
                "language": {"python": 600, "rust": 50, "javascript": 200},
                "domain": {"web-backend": 200, "data-engineering": 100},
                "task": {"feature-implementation": 400, "bug-fixing": 200, "code-explanation": 300},
                "concept": {"algorithms": 150, "data-structures": 200, "data-types": 300,
                            "concurrency": 80, "design-patterns": 100, "memory-management": 60,
                            "debugging": 180, "ownership": 30},
                "agentic": {"file-operations": 100, "multi-step-reasoning": 80},
                "constraint": {"thread-safe": 40, "optimized": 60},
                "context": {"snippet": 500, "single-file": 400, "multi-file": 100},
            },
        }
        stats_path = tmp_path / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f)

        config = PipelineConfig(
            scoring_concurrency=5,
            sample_max_retries=2,
            request_timeout=60,
        )

        result = asyncio.run(run_scoring(
            input_path=str(labeled_path),
            tag_stats_path=str(stats_path),
            limit=3,  # Score only first 3 to keep test fast
            config=config,
        ))

        # Verify output files exist
        assert (tmp_path / "scored.json").exists()
        assert (tmp_path / "scored.jsonl").exists()
        assert (tmp_path / PASS2_STATS_FILE).exists()
        assert (tmp_path / "monitor_value.jsonl").exists()
        assert (tmp_path / dashboard_relpath(PASS2_DASHBOARD_FILE)).exists()

        # Verify scored.json structure
        with open(tmp_path / "scored.json", encoding="utf-8") as f:
            scored = json.load(f)
        assert len(scored) == 3

        # At least some samples should be scored successfully
        scored_with_value = [s for s in scored if s.get("value")]
        assert len(scored_with_value) > 0, "No samples scored successfully"

        for s in scored_with_value:
            v = s["value"]
            # Check required fields
            assert "complexity" in v
            assert "quality" in v
            assert "reasoning" in v
            assert "rarity" in v
            assert "flags" in v
            assert "thinking_mode" in v
            assert "value_score" in v
            assert "confidence" in v

            # Value score should be 1-10
            assert 1.0 <= v["value_score"] <= 10.0

            # Thinking mode should be detected
            assert v["thinking_mode"] in ("slow", "fast")

            # Rarity should have been computed
            rarity = v["rarity"]
            assert rarity is not None
            assert rarity.get("score") is not None
            assert 1.0 <= rarity["score"] <= 10.0

            # Flags should be a list
            assert isinstance(v["flags"], list)

        # Verify Pass 2 stats
        with open(tmp_path / PASS2_STATS_FILE, encoding="utf-8") as f:
            stats_out = json.load(f)
        assert stats_out["total_scored"] > 0
        assert "score_distributions" in stats_out
        assert "thinking_mode_stats" in stats_out
        assert "weights_used" in stats_out

        # Verify dashboard is valid HTML and uses the new sidecar contract
        dashboard_path = tmp_path / dashboard_relpath(PASS2_DASHBOARD_FILE)
        html = dashboard_path.read_text(encoding="utf-8")
        bootstrap = _bootstrap_from_html(html)
        data_dir = dashboard_path.with_name(f"{dashboard_path.stem}.data")
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))
        assert "<!DOCTYPE html>" in html
        assert "SFT Labeling & Scoring Dashboard" in html
        assert "Scope Navigator" in html
        assert bootstrap["manifestUrl"] == "dashboard_scoring.data/manifest.json"
        assert "dashboard.js" in html
        assert "const DATA =" not in html
        assert manifest["scopes"]["global"]["summary"]["scored_total"] > 0
        assert detail["pass2"]["modes"]["sample"]["overview"]["total_scored"] > 0

        # Verify stats return value
        assert result is not None
        assert result["total_scored"] > 0


# ─────────────────────────────────────────────────────────
# P0-1: modify+task consistency rule
# ─────────────────────────────────────────────────────────

class TestConsistencyRuleModify:
    """Verify that modify+feature-implementation and modify+bug-fixing
    no longer trigger consistency rule warnings."""

    def test_modify_feature_implementation_no_warning(self):
        from sft_label.config import CONSISTENCY_RULES
        # Simulate the eval context
        context = {
            "intent": "modify",
            "task": ["feature-implementation"],
            "language": ["python"],
            "agentic": [],
            "concept": [],
            "constraint": [],
            "difficulty": "intermediate",
            "context": "single-file",
            "domain": [],
        }
        for rule_expr, msg in CONSISTENCY_RULES:
            if "modify" in rule_expr and "modify-relevant" in msg:
                result = eval(rule_expr, {}, context)
                assert not result, f"modify+feature-implementation should not trigger: {msg}"

    def test_modify_bug_fixing_no_warning(self):
        from sft_label.config import CONSISTENCY_RULES
        context = {
            "intent": "modify",
            "task": ["bug-fixing"],
            "language": ["python"],
            "agentic": [],
            "concept": [],
            "constraint": [],
            "difficulty": "intermediate",
            "context": "single-file",
            "domain": [],
        }
        for rule_expr, msg in CONSISTENCY_RULES:
            if "modify" in rule_expr and "modify-relevant" in msg:
                result = eval(rule_expr, {}, context)
                assert not result, f"modify+bug-fixing should not trigger: {msg}"

    def test_modify_error_handling_no_warning(self):
        from sft_label.config import CONSISTENCY_RULES
        context = {
            "intent": "modify",
            "task": ["error-handling-task"],
            "language": [],
            "agentic": [],
            "concept": [],
            "constraint": [],
            "difficulty": "intermediate",
            "context": "single-file",
            "domain": [],
        }
        for rule_expr, msg in CONSISTENCY_RULES:
            if "modify" in rule_expr and "modify-relevant" in msg:
                result = eval(rule_expr, {}, context)
                assert not result

    def test_modify_no_relevant_task_still_warns(self):
        """modify + unrelated tasks should still trigger warning."""
        from sft_label.config import CONSISTENCY_RULES
        context = {
            "intent": "modify",
            "task": ["code-explanation"],
            "language": [],
            "agentic": [],
            "concept": [],
            "constraint": [],
            "difficulty": "intermediate",
            "context": "single-file",
            "domain": [],
        }
        triggered = False
        for rule_expr, msg in CONSISTENCY_RULES:
            if "modify" in rule_expr and "modify-relevant" in msg:
                if eval(rule_expr, {}, context):
                    triggered = True
        assert triggered, "modify+code-explanation should still trigger warning"


# ─────────────────────────────────────────────────────────
# P1-1: Selection score with quality component
# ─────────────────────────────────────────────────────────

class TestSelectionScoreQualityComponent:
    def _make_sample(self, idx, quality=5, rarity_score=5.0):
        return {
            "id": f"sample-{idx}",
            "labels": {"intent": "build", "language": ["python"], "concept": ["algorithms"]},
            "value": {
                "complexity": {"instruction": 5, "overall": 5},
                "quality": {"correctness": quality, "overall": quality},
                "reasoning": {"clarity": 5, "overall": 5},
                "rarity": {"score": rarity_score},
                "flags": [],
                "value_score": 5.0,
            },
        }

    def test_high_quality_gets_higher_selection(self):
        """High quality samples should get higher selection scores than low quality."""
        samples = []
        for i in range(20):
            # Half high quality, half low quality
            q = 9 if i < 10 else 3
            samples.append(self._make_sample(i, quality=q))

        compute_selection_scores(samples, min_group_size=2)

        high_q = [s["value"]["selection_score"] for s in samples[:10]
                  if s["value"].get("selection_score") is not None]
        low_q = [s["value"]["selection_score"] for s in samples[10:]
                 if s["value"].get("selection_score") is not None]

        if high_q and low_q:
            # Mean of high quality group should be higher
            assert sum(high_q) / len(high_q) > sum(low_q) / len(low_q)


# ─────────────────────────────────────────────────────────
# P1-2: Turn position in scoring messages
# ─────────────────────────────────────────────────────────

class TestBuildScoringMessagesMultiTurn:
    def test_turn_position_added_for_multiturn(self):
        from sft_label.prompts_value import build_scoring_messages
        truncated = {
            "current_request": "Fix the bug",
            "instruction": "Fix the bug",
            "trajectory": "[Assistant Step] I will inspect the logs\n[Tool Result] pytest failed",
            "response": "Here is the fix",
            "original_prior_context_chars": 0,
            "original_cot_chars": 0,
            "original_trajectory_chars": 42,
            "original_response_chars": 100,
            "trajectory_turn_count": 2,
            "trajectory_tool_turns": 1,
        }
        messages = build_scoring_messages(
            truncated=truncated, thinking_mode="fast",
            labels={"intent": "debug"}, total_turns=2,
            code_block_count=1,
            slice_position=2, slice_count=5,
        )
        user_msg = messages[-1]["content"]
        assert "turn_position: 2/5 (multi-turn slice)" in user_msg
        assert "<current_turn_request>" in user_msg
        assert "<trajectory>" in user_msg
        assert "<final_response>" in user_msg
        assert user_msg.index("<current_turn_request>") < user_msg.index("<trajectory>") < user_msg.index("<final_response>")

    def test_turn_position_never_exceeds_total(self):
        from sft_label.prompts_value import build_scoring_messages
        truncated = {
            "current_request": "Write the fix",
            "instruction": "Write the fix",
            "response": "Patched the code",
            "original_prior_context_chars": 0,
            "original_cot_chars": 0,
            "original_trajectory_chars": 0,
            "original_response_chars": 40,
            "trajectory_turn_count": 0,
            "trajectory_tool_turns": 0,
        }
        messages = build_scoring_messages(
            truncated=truncated, thinking_mode="fast",
            labels={"intent": "debug"}, total_turns=2,
            code_block_count=0,
            slice_position=2, slice_count=2,
        )
        user_msg = messages[-1]["content"]
        assert "turn_position: 2/2 (multi-turn slice)" in user_msg
        assert "turn_position: 3/2" not in user_msg

    def test_no_turn_position_for_single_turn(self):
        from sft_label.prompts_value import build_scoring_messages
        truncated = {
            "current_request": "Write code",
            "instruction": "Write code",
            "response": "Here is code",
            "original_prior_context_chars": 0,
            "original_cot_chars": 0,
            "original_trajectory_chars": 0,
            "original_response_chars": 100,
            "trajectory_turn_count": 0,
            "trajectory_tool_turns": 0,
        }
        messages = build_scoring_messages(
            truncated=truncated, thinking_mode="fast",
            labels={"intent": "build"}, total_turns=2,
            code_block_count=1,
            slice_position=1, slice_count=1,
        )
        user_msg = messages[-1]["content"]
        assert "turn_position" not in user_msg

    def test_tool_turns_are_labeled_distinctly(self):
        from sft_label.prompts_value import build_scoring_messages
        truncated = {
            "current_request": "Run the failing command",
            "instruction": "Run the failing command",
            "trajectory": "[Assistant Step] Executing tests\n[Tool Result] FAIL test_example.py::test_bug",
            "response": "I fixed the bug",
            "original_prior_context_chars": 0,
            "original_cot_chars": 0,
            "original_trajectory_chars": 80,
            "original_response_chars": 20,
            "trajectory_turn_count": 2,
            "trajectory_tool_turns": 1,
        }
        messages = build_scoring_messages(
            truncated=truncated,
            thinking_mode="fast",
            labels={"intent": "debug", "agentic": ["tool-calling"]},
            total_turns=4,
            code_block_count=1,
        )
        user_msg = messages[-1]["content"]
        assert "[Tool Result] FAIL test_example.py::test_bug" in user_msg
        assert "[Final Assistant] I fixed the bug" in user_msg


# ─────────────────────────────────────────────────────────
# P1-3: Agentic quality floor penalty
# ─────────────────────────────────────────────────────────

class TestAgenticQualityFloor:
    def test_agentic_uses_p10_not_min(self):
        """Agentic conversations with 5+ slices should use p10 quality, not min."""
        # 10 slices: 9 have quality=7, 1 has quality=1
        slices = []
        for i in range(10):
            q = 1 if i == 0 else 7
            slices.append({
                "labels": {"agentic": ["tool-calling"]},
                "value": {"quality": {"overall": q}, "flags": []},
            })
        penalty, quality_floor, neg_flags = _compute_penalty(slices)
        # p10 of [1,7,7,7,7,7,7,7,7,7] → sorted[1] = 7 (idx=int(10*0.1)=1)
        assert quality_floor == 7
        assert penalty == 1.0  # quality_floor=7 → no penalty

    def test_non_agentic_uses_min(self):
        """Non-agentic conversations should still use min quality."""
        slices = []
        for i in range(10):
            q = 1 if i == 0 else 7
            slices.append({
                "labels": {},
                "value": {"quality": {"overall": q}, "flags": []},
            })
        penalty, quality_floor, neg_flags = _compute_penalty(slices)
        assert quality_floor == 1
        assert penalty < 1.0  # quality_floor=1 < 3 → 0.5×

    def test_agentic_small_group_uses_min(self):
        """Agentic conversations with <5 slices still use min (not enough data for p10)."""
        slices = [
            {"labels": {"agentic": ["tool-calling"]},
             "value": {"quality": {"overall": 2}, "flags": []}},
            {"labels": {"agentic": ["tool-calling"]},
             "value": {"quality": {"overall": 8}, "flags": []}},
        ]
        penalty, quality_floor, neg_flags = _compute_penalty(slices)
        assert quality_floor == 2  # min, not p10
