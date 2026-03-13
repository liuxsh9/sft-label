from __future__ import annotations

from sft_label.tools.visualize_value import compute_value_viz_data, _compute_conv_viz_data


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
