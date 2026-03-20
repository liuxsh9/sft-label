from __future__ import annotations

import json

from sft_label.artifacts import PASS2_STATS_FILE, PASS2_SUMMARY_STATS_FILE
from sft_label.tools import visualize_value as visualize_value_module
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
    sample_summary = viz["modes"]["sample"]["per_file_summary"][0]
    conversation_summary = viz["modes"]["conversation"]["per_file_summary"][0]
    assert sample_summary["mean_rarity"] == 5.5
    assert sample_summary["keep_rate_7"] == 0.5
    assert sample_summary["keep_rates"] == {"4.0": 1.0, "5.0": 1.0, "6.0": 0.5, "7.0": 0.5}
    assert sample_summary["mean_turns"] == 2.0
    assert conversation_summary["mean_rarity"] == 6.4
    assert conversation_summary["keep_rate_7"] == 1.0
    assert conversation_summary["keep_rates"] == {"4.0": 1.0, "5.0": 1.0, "6.0": 1.0, "7.0": 1.0}
    assert conversation_summary["mean_turns"] == 2.0


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


def test_compute_value_viz_data_surfaces_extension_rarity_v2_metrics():
    samples = [
        {
            "id": "ext-1",
            "metadata": {"source_file": "sample.jsonl"},
            "labels": {"intent": "build", "difficulty": "advanced", "context": "snippet"},
            "value": {
                "value_score": 6.5,
                "selection_score": 6.8,
                "value_score_v2": 6.9,
                "selection_score_v2": 7.1,
                "thinking_mode": "fast",
                "confidence": 0.8,
                "quality": {"overall": 6.0},
                "complexity": {"overall": 5.0},
                "reasoning": {"overall": 5.0},
                "rarity": {"score": 4.0},
                "rarity_extension": {"score": 7.4, "baseline_source": "external"},
                "rarity_v2": {"score": 4.6, "extension_bonus": 0.3},
            },
        }
    ]
    stats = {
        "total_scored": 1,
        "total_failed": 0,
        "total_tokens": 10,
        "extension_rarity_config": {
            "mode": "bonus_only",
            "baseline_source": "external",
            "min_extension_baseline_total": 200,
        },
    }

    viz = compute_value_viz_data(samples, stats)

    sample_mode = viz["modes"]["sample"]
    conversation_mode = viz["modes"]["conversation"]
    assert sample_mode["score_distributions"]["extension_rarity_score"]["mean"] == 7.4
    assert sample_mode["score_distributions"]["rarity_v2_score"]["mean"] == 4.6
    assert sample_mode["score_distributions"]["value_score_v2"]["mean"] == 6.9
    assert sample_mode["score_distributions"]["selection_score_v2"]["mean"] == 7.1
    assert sample_mode["overview"]["extension_rarity_mode"] == "bonus_only"
    assert sample_mode["overview"]["extension_baseline_source"] == "external"
    assert "extension_rarity_score" not in conversation_mode["score_distributions"]
    assert "rarity_v2_score" not in conversation_mode["score_distributions"]
    assert "value_score_v2" not in conversation_mode["histograms"]
    assert "selection_score_v2" not in conversation_mode["histograms"]
    assert conversation_mode["overview"]["mean_extension_rarity"] is None
    assert conversation_mode["overview"]["mean_rarity_v2"] is None
    assert "mean_extension_rarity" not in conversation_mode["per_file_summary"][0]
    assert "mean_selection_v2" not in conversation_mode["per_file_summary"][0]


def test_compute_value_viz_data_stats_only_hides_empty_extension_rarity_v2_metrics():
    viz = compute_value_viz_data([], {
        "total_scored": 0,
        "total_failed": 0,
        "score_distributions": {
            "value_score": {"mean": 6.2, "min": 6.2, "max": 6.2, "p50": 6.2},
            "selection_score": {"mean": 6.5, "min": 6.5, "max": 6.5, "p50": 6.5},
            "extension_rarity_score": {},
            "rarity_v2_score": {},
            "value_score_v2": {},
            "selection_score_v2": {},
        },
        "histograms": {
            "value_score": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "selection_score": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "extension_rarity_score": [0] * 10,
            "rarity_v2_score": [0] * 10,
            "value_score_v2": [0] * 10,
            "selection_score_v2": [0] * 10,
        },
    })

    sample_mode = viz["modes"]["sample"]

    assert "extension_rarity_score" not in sample_mode["score_distributions"]
    assert "rarity_v2_score" not in sample_mode["score_distributions"]
    assert "value_score_v2" not in sample_mode["score_distributions"]
    assert "selection_score_v2" not in sample_mode["score_distributions"]
    assert "extension_rarity_score" not in sample_mode["histograms"]
    assert "rarity_v2_score" not in sample_mode["histograms"]
    assert "value_score_v2" not in sample_mode["histograms"]
    assert "selection_score_v2" not in sample_mode["histograms"]
    assert sample_mode["overview"]["mean_extension_rarity"] is None
    assert sample_mode["overview"]["mean_rarity_v2"] is None
    assert sample_mode["overview"]["mean_value_v2"] is None
    assert sample_mode["overview"]["mean_selection_v2"] is None
    assert sample_mode["overview"]["extension_rarity_mode"] is None
    assert sample_mode["overview"]["extension_baseline_source"] is None


def test_compute_value_viz_data_single_turn_defaults_mean_turns_to_one():
    samples = [
        {
            "id": "single-1",
            "metadata": {"source_file": "single.jsonl"},
            "labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
            "value": {
                "value_score": 6.0,
                "selection_score": 6.4,
                "thinking_mode": "fast",
                "confidence": 0.8,
                "quality": {"overall": 6.0},
                "complexity": {"overall": 5.0},
                "reasoning": {"overall": 5.0},
                "rarity": {"score": 4.0},
            },
        }
    ]

    viz = compute_value_viz_data(samples, {"total_scored": 1, "total_failed": 0, "total_tokens": 10})

    sample_summary = viz["modes"]["sample"]["per_file_summary"][0]
    assert sample_summary["mean_turns"] == 1.0


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
                "clarification_turn_count": 1,
                "recovery_success_ratio": 0.5,
                "tool_result_success_ratio": 0.5,
                "tool_repeat_ratio": 0.25,
                "verification_turn_count": 2,
                "verification_after_edit_count": 1,
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
                "clarification_turn_count": 0,
                "recovery_success_ratio": 1.0,
                "tool_result_success_ratio": 0.75,
                "tool_repeat_ratio": 0.0,
                "verification_turn_count": 4,
                "verification_after_edit_count": 2,
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
    assert conv["mean_clarification_turns"] == 0.5
    assert conv["mean_recovery_success_ratio"] == 0.75
    assert conv["mean_tool_result_success_ratio"] == 0.625
    assert conv["mean_tool_repeat_ratio"] == 0.125
    assert conv["mean_verification_turns"] == 3.0
    assert conv["mean_verification_after_edit_turns"] == 1.5


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


def test_generate_value_dashboard_tree_payload_uses_stats_without_preloading_leaf_samples(tmp_path, monkeypatch):
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "scored.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "metadata": {"source_file": "code/sample.jsonl", "source_id": "conv-1"},
                "labels": {"intent": "build", "difficulty": "intermediate", "context": "snippet"},
                "value": {
                    "value_score": 6.5,
                    "selection_score": 7.1,
                    "thinking_mode": "fast",
                    "quality": {"overall": 6.0},
                    "complexity": {"overall": 6.0},
                    "reasoning": {"overall": 6.0},
                    "rarity": {"score": 5.5},
                    "confidence": 0.8,
                },
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "conversation_scores.json").write_text(
        json.dumps(
            [
                {
                    "conversation_id": "conv-1",
                    "conversation_key": "code/sample.jsonl::conv-1",
                    "source_file": "code/sample.jsonl",
                    "turn_count": 2,
                    "conv_value": 6.5,
                    "conv_selection": 7.1,
                    "peak_complexity": 7,
                }
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / PASS2_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample.jsonl",
                "total_scored": 1,
                "total_failed": 0,
                "score_distributions": {
                    "value_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                    "selection_score": {"mean": 7.1, "min": 7.1, "max": 7.1},
                    "complexity_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "quality_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "confidence": {"mean": 0.8, "min": 0.8, "max": 0.8},
                },
                "histograms": {},
                "flag_counts": {},
                "value_by_tag": {},
                "selection_by_tag": {},
            }
        ),
        encoding="utf-8",
    )
    (meta_root / PASS2_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "input_path": "dataset",
                "total_scored": 1,
                "total_failed": 0,
                "score_distributions": {
                    "value_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                    "selection_score": {"mean": 7.1, "min": 7.1, "max": 7.1},
                    "complexity_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "quality_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "confidence": {"mean": 0.8, "min": 0.8, "max": 0.8},
                },
                "histograms": {},
                "flag_counts": {},
                "per_file_summary": [
                    {
                        "file": "code/sample.jsonl",
                        "count": 1,
                        "mean_value": 6.5,
                        "mean_complexity": 6.0,
                        "mean_quality": 6.0,
                        "mean_selection": 7.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fail_load(_path):
        raise AssertionError("tree payload should not preload leaf sample files")

    monkeypatch.setattr(visualize_value_module, "load_data_file", _fail_load)

    out = generate_value_dashboard(meta_root, scored_file=None, stats_file=PASS2_SUMMARY_STATS_FILE, quiet=True)
    data_dir = out.with_name(f"{out.stem}.data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    detail = json.loads((data_dir / "scopes" / "file_code_sample_jsonl.json").read_text(encoding="utf-8"))

    assert manifest["scopes"]["file:code/sample.jsonl"]["explorer"]["sample_count"] == 1
    assert detail["conversation"]["mean_conv_value"] == 6.5


def test_generate_value_dashboard_tree_payload_aggregates_conversation_mode_for_parent_scopes(tmp_path):
    meta_root = tmp_path / "meta_label_data"
    code_leaf = meta_root / "files" / "code" / "sample_a"
    docs_leaf = meta_root / "files" / "docs" / "sample_b"
    code_leaf.mkdir(parents=True, exist_ok=True)
    docs_leaf.mkdir(parents=True, exist_ok=True)

    code_rows = [
        {
            "id": "code-1",
            "metadata": {"source_file": "code/sample_a.jsonl", "source_id": "conv-1"},
            "labels": {"intent": "build", "difficulty": "intermediate", "context": "snippet"},
            "value": {
                "value_score": 4.0,
                "selection_score": 4.4,
                "thinking_mode": "fast",
                "quality": {"overall": 4.0},
                "complexity": {"overall": 4.0},
                "reasoning": {"overall": 4.0},
                "rarity": {"score": 4.0},
                "confidence": 0.7,
            },
        },
        {
            "id": "code-2",
            "metadata": {"source_file": "code/sample_a.jsonl", "source_id": "conv-1"},
            "labels": {"intent": "debug", "difficulty": "advanced", "context": "repository"},
            "value": {
                "value_score": 8.0,
                "selection_score": 8.2,
                "thinking_mode": "fast",
                "quality": {"overall": 8.0},
                "complexity": {"overall": 8.0},
                "reasoning": {"overall": 8.0},
                "rarity": {"score": 8.0},
                "confidence": 0.9,
            },
        },
    ]
    docs_rows = [
        {
            "id": "docs-1",
            "metadata": {"source_file": "docs/sample_b.jsonl", "source_id": "conv-2"},
            "labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
            "value": {
                "value_score": 5.0,
                "selection_score": 5.4,
                "thinking_mode": "fast",
                "quality": {"overall": 5.0},
                "complexity": {"overall": 5.0},
                "reasoning": {"overall": 5.0},
                "rarity": {"score": 5.0},
                "confidence": 0.8,
            },
        }
    ]
    (code_leaf / "scored.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in code_rows) + "\n",
        encoding="utf-8",
    )
    (docs_leaf / "scored.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in docs_rows) + "\n",
        encoding="utf-8",
    )
    (code_leaf / "conversation_scores.json").write_text(
        json.dumps(
            [
                {
                    "conversation_id": "conv-1",
                    "conversation_key": "code/sample_a.jsonl::conv-1",
                    "source_file": "code/sample_a.jsonl",
                    "turn_count": 2,
                    "conv_value": 7.0,
                    "conv_selection": 7.5,
                    "peak_complexity": 8.0,
                    "conv_rarity": 6.6,
                    "thinking_mode": "fast",
                    "merged_labels": {"intent": "debug", "difficulty": "advanced", "context": "repository"},
                }
            ]
        ),
        encoding="utf-8",
    )
    (docs_leaf / "conversation_scores.json").write_text(
        json.dumps(
            [
                {
                    "conversation_id": "conv-2",
                    "conversation_key": "docs/sample_b.jsonl::conv-2",
                    "source_file": "docs/sample_b.jsonl",
                    "turn_count": 1,
                    "conv_value": 5.0,
                    "conv_selection": 5.4,
                    "peak_complexity": 5.0,
                    "conv_rarity": 5.0,
                    "thinking_mode": "fast",
                    "merged_labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
                }
            ]
        ),
        encoding="utf-8",
    )
    (code_leaf / PASS2_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample_a.jsonl",
                "total_scored": 2,
                "total_failed": 0,
                "score_distributions": {
                    "value_score": {"mean": 6.0},
                    "selection_score": {"mean": 6.3},
                    "complexity_overall": {"mean": 6.0},
                    "quality_overall": {"mean": 6.0},
                    "reasoning_overall": {"mean": 6.0},
                    "rarity_score": {"mean": 6.0},
                    "confidence": {"mean": 0.8},
                },
                "histograms": {},
                "flag_counts": {},
                "value_by_tag": {},
                "selection_by_tag": {},
            }
        ),
        encoding="utf-8",
    )
    (docs_leaf / PASS2_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "docs/sample_b.jsonl",
                "total_scored": 1,
                "total_failed": 0,
                "score_distributions": {
                    "value_score": {"mean": 5.0},
                    "selection_score": {"mean": 5.4},
                    "complexity_overall": {"mean": 5.0},
                    "quality_overall": {"mean": 5.0},
                    "reasoning_overall": {"mean": 5.0},
                    "rarity_score": {"mean": 5.0},
                    "confidence": {"mean": 0.8},
                },
                "histograms": {},
                "flag_counts": {},
                "value_by_tag": {},
                "selection_by_tag": {},
            }
        ),
        encoding="utf-8",
    )
    (meta_root / PASS2_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "input_path": "dataset",
                "total_scored": 3,
                "total_failed": 0,
                "score_distributions": {
                    "value_score": {"mean": 5.7},
                    "selection_score": {"mean": 6.0},
                    "complexity_overall": {"mean": 5.7},
                    "quality_overall": {"mean": 5.7},
                    "reasoning_overall": {"mean": 5.7},
                    "rarity_score": {"mean": 5.7},
                    "confidence": {"mean": 0.8},
                },
                "histograms": {},
                "flag_counts": {},
                "per_file_summary": [
                    {"file": "code/sample_a.jsonl", "count": 2, "mean_value": 6.0, "mean_complexity": 6.0, "mean_quality": 6.0, "mean_selection": 6.3},
                    {"file": "docs/sample_b.jsonl", "count": 1, "mean_value": 5.0, "mean_complexity": 5.0, "mean_quality": 5.0, "mean_selection": 5.4},
                ],
            }
        ),
        encoding="utf-8",
    )

    out = generate_value_dashboard(meta_root, scored_file=None, stats_file=PASS2_SUMMARY_STATS_FILE, quiet=True)
    data_dir = out.with_name(f"{out.stem}.data")
    code_detail = json.loads((data_dir / "scopes" / "file_code_sample_a_jsonl.json").read_text(encoding="utf-8"))
    global_detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert code_detail["pass2"]["modes"]["conversation"]["overview"]["total_scored"] == 1
    assert global_detail["pass2"]["modes"]["conversation"]["overview"]["total_scored"] == 2
    assert global_detail["summary_modes"]["conversation"]["scored_total"] == 2
    assert global_detail["conversation"]["total"] == 2
    assert global_detail["conversation"]["mean_conv_value"] == 6.0
