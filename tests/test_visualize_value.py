from __future__ import annotations

import json

from sft_label.artifacts import PASS2_STATS_FILE, PASS2_SUMMARY_STATS_FILE
from sft_label.tools.visualize_value import compute_value_viz_data, _compute_conv_viz_data
from sft_label.tools.visualize_value import generate_value_dashboard


def test_compute_value_viz_data_conversation_mode_counts_multiturn_once():
    samples = [
        {
            "id": "conv-1-s1",
            "metadata": {"source_id": "conv-1", "source_file": "train.jsonl", "turn_index": 1, "total_turns": 2},
            "labels": {
                "intent": "build",
                "difficulty": "intermediate",
                "agentic": ["planning"],
                "context": "snippet",
            },
            "value": {
                "value_score": 5.0,
                "selection_score": 5.5,
                "thinking_mode": "fast",
                "confidence": 0.8,
                "flags": [],
                "quality": {"overall": 5.0},
                "complexity": {"overall": 4.0},
                "reasoning": {"overall": 5.0},
                "rarity": {"score": 4.0},
            },
        },
        {
            "id": "conv-1-s2",
            "metadata": {"source_id": "conv-1", "source_file": "train.jsonl", "turn_index": 2, "total_turns": 2},
            "labels": {
                "intent": "debug",
                "difficulty": "advanced",
                "agentic": ["planning", "file-operations"],
                "context": "repository",
            },
            "value": {
                "value_score": 8.0,
                "selection_score": 8.5,
                "thinking_mode": "fast",
                "confidence": 0.9,
                "flags": ["has-bug"],
                "quality": {"overall": 8.0},
                "complexity": {"overall": 7.0},
                "reasoning": {"overall": 8.0},
                "rarity": {"score": 7.0},
            },
        },
    ]
    conv_records = [
        {
            "conversation_id": "conv-1",
            "conversation_key": "train.jsonl::conv-1",
            "source_file": "train.jsonl",
            "turn_count": 2,
            "conv_value": 7.2,
            "conv_selection": 7.8,
            "peak_complexity": 7,
            "conv_rarity": 6.4,
            "thinking_mode": "fast",
            "merged_labels": {
                "intent": "debug",
                "difficulty": "advanced",
                "agentic": ["planning", "file-operations"],
                "context": "repository",
            },
            "detail": {"score_confidence": 0.85},
        }
    ]
    stats = {"total_scored": 2, "total_failed": 0, "total_tokens": 20}

    viz = compute_value_viz_data(samples, stats, conv_records)

    assert viz["modes"]["sample"]["overview"]["total_scored"] == 2
    assert viz["modes"]["conversation"]["overview"]["total_scored"] == 1
    assert viz["modes"]["conversation"]["value_by_tag"]["agentic"]["planning"]["n"] == 1
    assert viz["modes"]["conversation"]["value_by_tag"]["agentic"]["file-operations"]["n"] == 1
    assert viz["modes"]["conversation"]["flag_counts"]["has-bug"] == 1


def test_compute_value_viz_data_exposes_prompt_mode_budget():
    viz = compute_value_viz_data([], {
        "total_scored": 0,
        "total_failed": 0,
        "score_distributions": {},
        "prompt_mode": "compact",
        "compact_prompt": True,
        "value_truncation_budget": 14000,
    })

    assert viz["overview"]["prompt_mode"] == "compact"
    assert viz["overview"]["compact_prompt"] is True
    assert viz["overview"]["value_truncation_budget"] == 14000
    assert viz["modes"]["sample"]["overview"]["prompt_mode"] == "compact"
    assert viz["modes"]["conversation"]["overview"]["value_truncation_budget"] == 14000


def test_compute_conv_viz_data_aggregates_new_diagnostics():
    conv = _compute_conv_viz_data([
        {
            "conv_value": 6.0,
            "conv_selection": 7.0,
            "peak_complexity": 8.0,
            "turn_count": 4,
            "observed_turn_ratio": 0.5,
            "inherited_turn_ratio": 0.5,
            "rarity_confidence": 0.7,
            "compression_gap": 2.0,
            "late_turn_gain": 1.0,
            "tool_turn_ratio": 0.6,
            "unique_tool_count": 2,
            "unique_file_count": 3,
            "detail": {
                "turn_value_std": 1.2,
                "test_related_turn_count": 2,
                "edit_related_turn_count": 1,
            },
        },
        {
            "conv_value": 8.0,
            "conv_selection": 9.0,
            "peak_complexity": 9.0,
            "turn_count": 6,
            "observed_turn_ratio": 1.0,
            "inherited_turn_ratio": 0.0,
            "rarity_confidence": 0.9,
            "compression_gap": 3.0,
            "late_turn_gain": 2.0,
            "tool_turn_ratio": 0.4,
            "unique_tool_count": 4,
            "unique_file_count": 5,
            "detail": {
                "turn_value_std": 0.8,
                "test_related_turn_count": 4,
                "edit_related_turn_count": 3,
            },
        },
    ])

    assert conv["mean_peak_minus_mean"] == 2.5
    assert conv["mean_late_turn_gain"] == 1.5
    assert conv["mean_tool_turn_ratio"] == 0.5
    assert conv["mean_unique_tool_count"] == 3.0
    assert conv["mean_unique_file_count"] == 4.0
    assert conv["mean_turn_value_std"] == 1.0
    assert conv["mean_test_related_turns"] == 3.0
    assert conv["mean_edit_related_turns"] == 2.0


def test_generate_value_dashboard_stats_only_single_scope_uses_serialized_stats(tmp_path):
    stats = {
        "total_scored": 3,
        "total_failed": 1,
        "total_tokens": 123,
        "input_file": "demo.jsonl",
        "weights_used": {"quality": 0.4},
        "selection_thresholds": {"top_10pct": {"threshold": 8.0, "count": 1}},
        "score_distributions": {
            "value_score": {"mean": 6.5, "min": 4.0, "max": 9.0, "p50": 6.0},
            "selection_score": {"mean": 7.1, "min": 5.0, "max": 9.0, "p50": 7.0},
            "complexity_overall": {"mean": 6.0, "min": 3.0, "max": 8.0, "p50": 6.0},
            "quality_overall": {"mean": 7.0, "min": 4.0, "max": 9.0, "p50": 7.0},
            "reasoning_overall": {"mean": 6.3, "min": 4.0, "max": 8.0, "p50": 6.0},
            "rarity_score": {"mean": 5.5, "min": 2.0, "max": 8.0, "p50": 5.0},
            "confidence": {"mean": 0.8, "min": 0.6, "max": 0.9, "p50": 0.8},
        },
        "histograms": {
            "value_score": [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            "selection_score": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            "complexity_overall": [0] * 10,
            "quality_overall": [0] * 10,
            "reasoning_overall": [0] * 10,
            "rarity_score": [0] * 10,
        },
        "thinking_mode_stats": {
            "fast": {"count": 3, "mean_value": 6.5, "mean_quality": 7.0, "mean_reasoning": 6.3},
        },
        "value_by_tag": {"intent": {"build": {"mean": 6.5, "n": 3}}},
        "selection_by_tag": {"intent": {"build": {"mean": 7.1, "n": 3}}},
        "coverage_at_thresholds": {"5.0": {"retained": 2, "pct": 0.67, "tags_lost": [], "coverage": 1.0}},
        "confidence_histogram": [0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
        "flag_counts": {"has-bug": 1},
        "flag_value_impact": {"has-bug": {"mean_value": 5.0, "count": 1}},
        "per_file_summary": [{"file": "demo.jsonl", "count": 3, "mean_value": 6.5, "mean_complexity": 6.0, "mean_quality": 7.0, "mean_rarity": 5.5, "mean_selection": 7.1}],
    }
    (tmp_path / PASS2_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")

    out = generate_value_dashboard(tmp_path, scored_file="missing.json", stats_file=PASS2_STATS_FILE, quiet=True)
    data_dir = out.with_name(f"{out.stem}.data")
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    sample_mode = detail["pass2"]["modes"]["sample"]
    conversation_mode = detail["pass2"]["modes"]["conversation"]
    assert detail["summary"]["scored_total"] == 3
    assert sample_mode["overview"]["total_scored"] == 3
    assert sample_mode["overview"]["mean_value"] == 6.5
    assert sample_mode["score_distributions"]["selection_score"]["mean"] == 7.1
    assert sample_mode["per_file_summary"][0]["file"] == "demo.jsonl"
    assert conversation_mode["overview"]["total_scored"] == 3
    assert conversation_mode["overview"]["mean_selection"] == 7.1


def test_generate_value_dashboard_stats_only_summary_scope_uses_serialized_stats(tmp_path):
    stats = {
        "total_scored": 5,
        "total_failed": 0,
        "total_tokens": 555,
        "input_path": "dataset",
        "score_distributions": {
            "value_score": {"mean": 6.9, "min": 4.0, "max": 9.0, "p50": 7.0},
            "selection_score": {"mean": 7.4, "min": 5.0, "max": 9.0, "p50": 7.0},
            "complexity_overall": {"mean": 6.2, "min": 4.0, "max": 8.0, "p50": 6.0},
            "quality_overall": {"mean": 7.1, "min": 5.0, "max": 9.0, "p50": 7.0},
            "reasoning_overall": {"mean": 6.8, "min": 4.0, "max": 9.0, "p50": 7.0},
            "rarity_score": {"mean": 5.8, "min": 2.0, "max": 8.0, "p50": 6.0},
            "confidence": {"mean": 0.82, "min": 0.6, "max": 0.95, "p50": 0.8},
        },
        "histograms": {"value_score": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]},
        "per_file_summary": [{"file": "part-1.jsonl", "count": 5, "mean_value": 6.9, "mean_complexity": 6.2, "mean_quality": 7.1, "mean_rarity": 5.8, "mean_selection": 7.4}],
    }
    (tmp_path / PASS2_SUMMARY_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")

    out = generate_value_dashboard(tmp_path, scored_file=None, stats_file=PASS2_SUMMARY_STATS_FILE, quiet=True)
    data_dir = out.with_name(f"{out.stem}.data")
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert detail["summary"]["scored_total"] == 5
    assert detail["pass2"]["modes"]["sample"]["overview"]["total_scored"] == 5
    assert detail["pass2"]["modes"]["sample"]["overview"]["mean_value"] == 6.9
    assert detail["pass2"]["modes"]["conversation"]["overview"]["mean_selection"] == 7.4
