from __future__ import annotations

from sft_label.tools.visualize_value import compute_value_viz_data


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
