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
from pathlib import Path

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
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


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

    def test_multi_turn_uses_first_human_last_gpt(self):
        convs = [
            {"from": "human", "value": "first question"},
            {"from": "gpt", "value": "first answer"},
            {"from": "human", "value": "second question"},
            {"from": "gpt", "value": "final answer"},
        ]
        result = truncate_for_scoring(convs, "fast", budget=10000)
        assert result["instruction"] == "first question"
        assert result["response"] == "final answer"


# ─────────────────────────────────────────────────────────
# 8.3  Rarity Computation
# ─────────────────────────────────────────────────────────

class TestComputeTagIdf:
    def test_basic_idf(self):
        distributions = {
            "intent": {"build": 500, "learn": 300, "debug": 200},
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
            "intent": {"build": 500, "learn": 300, "debug": 200},
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

    def test_combo_rarity(self):
        labels = {"intent": "build", "difficulty": "expert", "concept": ["algorithms"]}
        combo_counts = {"build|expert|algorithms": 2}
        result = compute_sample_rarity(
            labels, self.idf_map, 1000,
            combo_counts=combo_counts,
        )
        assert result["combo_rarity"] > 0

    def test_unknown_tag_gets_max_idf(self):
        """Tags not in distributions should get max IDF (log2(N))."""
        labels = {"intent": "never-seen-before"}
        result = compute_sample_rarity(labels, self.idf_map, 1000)
        expected_max = math.log2(1000)
        assert result["tag_rarity"] == round(expected_max, 3)

    def test_multi_select_averages(self):
        labels = {"concept": ["algorithms", "data-types"]}
        result = compute_sample_rarity(labels, self.idf_map, 1000)
        # Should average the IDF of both tags
        assert result["tag_rarity"] > 0

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


class TestBuildComboCounts:
    def test_basic_counting(self):
        samples = [
            {"labels": {"intent": "build", "difficulty": "advanced", "concept": ["algorithms"]}},
            {"labels": {"intent": "build", "difficulty": "advanced", "concept": ["algorithms"]}},
            {"labels": {"intent": "learn", "difficulty": "beginner", "concept": ["data-types"]}},
        ]
        counts = build_combo_counts(samples)
        assert counts["build|advanced|algorithms"] == 2
        assert counts["learn|beginner|data-types"] == 1

    def test_empty_samples(self):
        assert build_combo_counts([]) == {}


# ─────────────────────────────────────────────────────────
# 8.4  Score Validation
# ─────────────────────────────────────────────────────────

class TestValidateScoreResponse:
    def test_valid_response(self):
        parsed = {
            "complexity": {"instruction": 7, "reasoning": 8, "implementation": 6, "overall": 7},
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
            "complexity": {"instruction": 0, "reasoning": 11, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.5,
        }
        result, issues = validate_score_response(parsed)
        assert result["complexity"]["instruction"] is None  # 0 is out of range
        assert result["complexity"]["reasoning"] is None  # 11 is out of range
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
            "complexity": {"instruction": 5, "reasoning": 5, "implementation": 5, "overall": 5},
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
            "complexity": {"instruction": 5, "reasoning": 5, "implementation": 5, "overall": 5},
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
            "complexity": {"instruction": 7.5, "reasoning": 8.0, "implementation": 6.3, "overall": 7.2},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": [],
            "confidence": 0.5,
        }
        result, issues = validate_score_response(parsed)
        assert result["complexity"]["instruction"] == 7
        assert result["complexity"]["reasoning"] == 8

    def test_negative_flags_preserved(self):
        parsed = {
            "complexity": {"instruction": 5, "reasoning": 5, "implementation": 5, "overall": 5},
            "quality": {"correctness": 5, "code_quality": 5, "explanation": 5, "completeness": 5, "overall": 5},
            "reasoning": {"clarity": 5, "consistency": 5, "overall": 5},
            "flags": ["has-bug", "security-issue"],
            "confidence": 0.7,
        }
        result, issues = validate_score_response(parsed)
        assert "has-bug" in result["flags"]
        assert "security-issue" in result["flags"]
        assert len(issues) == 0


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

    def test_null_rarity_renormalizes(self):
        score_result = {
            "complexity": {"overall": 8},
            "quality": {"overall": 8},
            "reasoning": {"overall": 8},
        }
        rarity_result = {"score": None}
        value = compute_value_score(score_result, rarity_result)
        assert value == 8.0  # All available scores are 8

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
# 8.5  Integration test: end-to-end scoring via API
# ─────────────────────────────────────────────────────────

import os
import asyncio
import tempfile
import shutil
import pytest


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
        assert (tmp_path / "stats_value.json").exists()
        assert (tmp_path / "monitor_value.jsonl").exists()
        assert (tmp_path / "dashboard_value.html").exists()

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

        # Verify stats_value.json
        with open(tmp_path / "stats_value.json", encoding="utf-8") as f:
            stats_out = json.load(f)
        assert stats_out["total_scored"] > 0
        assert "score_distributions" in stats_out
        assert "thinking_mode_stats" in stats_out
        assert "weights_used" in stats_out

        # Verify dashboard is valid HTML
        html = (tmp_path / "dashboard_value.html").read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "SFT Value Scoring Dashboard" in html

        # Verify stats return value
        assert result is not None
        assert result["total_scored"] > 0
