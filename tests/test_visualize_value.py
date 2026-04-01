from __future__ import annotations

import json
from sft_label.artifacts import PASS1_STATS_FILE, PASS1_SUMMARY_STATS_FILE, PASS2_STATS_FILE, PASS2_SUMMARY_STATS_FILE
from sft_label.conversation import aggregate_conversations
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
        "postprocess": {
            "conversation_scores": {"status": "completed"},
            "dashboard": {"status": "completed"},
        },
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
        "postprocess": {
            "conversation_scores": {"status": "completed"},
            "dashboard": {"status": "completed"},
        },
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
    assert "conversation" not in ((detail.get("pass2") or {}).get("modes") or {})
    assert detail["summary_modes"]["conversation"]["scored_total"] == 0


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
                    "merged_labels": {
                        "intent": "build",
                        "difficulty": "intermediate",
                        "task": ["feature-implementation"],
                        "context": "snippet",
                    },
                    "detail": {
                        "quality_overall": 6.0,
                        "reasoning_overall": 6.0,
                        "observed_turns": 1,
                        "inherited_turns": 1,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / PASS2_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": str(artifact_dir / "labeled.jsonl"),
                "total_scored": 1,
                "total_failed": 0,
                "score_distributions": {
                    "value_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                    "selection_score": {"mean": 7.1, "min": 7.1, "max": 7.1},
                    "complexity_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "quality_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "reasoning_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "confidence": {"mean": 0.8, "min": 0.8, "max": 0.8},
                },
                "histograms": {},
                "flag_counts": {},
                "value_by_tag": {},
                "selection_by_tag": {},
                "thinking_mode_stats": {
                    "fast": {"count": 1, "mean_value": 6.5, "mean_quality": 6.0, "mean_reasoning": 6.0},
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / PASS1_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample.jsonl",
                "total_samples": 2,
                "success": 2,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 2}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
                "sparse_labeled": 1,
                "sparse_inherited": 1,
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
                "postprocess": {
                    "dashboard": {"status": "completed"},
                    "conversation_scores": {"status": "completed"},
                },
                "score_distributions": {
                    "value_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                    "selection_score": {"mean": 7.1, "min": 7.1, "max": 7.1},
                    "complexity_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "quality_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "reasoning_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
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
                "thinking_mode_stats": {
                    "fast": {"count": 1, "mean_value": 6.5, "mean_quality": 6.0, "mean_reasoning": 6.0},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta_root / PASS1_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "input_path": "dataset",
                "total_samples": 2,
                "success": 2,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 2}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
                "sparse_labeled": 1,
                "sparse_inherited": 1,
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
    assert detail["pass2"]["modes"]["sample"]["overview"]["input_file"] == "code/sample.jsonl"
    assert detail["pass2"]["modes"]["conversation"]["overview"]["input_file"] == "code/sample.jsonl"
    assert detail["pass1"]["modes"]["conversation"]["total"] == 1
    assert detail["pass1"]["modes"]["conversation"]["overview"]["llm_labeled_units"] == 1
    assert detail["pass1"]["modes"]["conversation"]["overview"]["inherited_units"] == 1
    assert detail["pass1"]["modes"]["conversation"]["distributions"]["task"] == {"feature-implementation": 1}
    assert detail["pass2"]["modes"]["conversation"]["overview"]["mean_quality"] == 6.0
    assert detail["pass2"]["modes"]["conversation"]["score_distributions"]["reasoning_overall"]["mean"] == 6.0
    assert detail["pass2"]["modes"]["conversation"]["per_file_summary"][0]["mean_quality"] == 6.0
    assert detail["pass2"]["modes"]["conversation"]["thinking_mode_stats"]["fast"]["mean_quality"] == 6.0


def test_tree_payload_keeps_pass1_conversation_without_pass2_records(tmp_path, monkeypatch):
    pass1_data_path = tmp_path / "pass1.jsonl"
    pass1_data_path.write_text(
        json.dumps({"metadata": {"conversation_uid": "conv-1", "total_turns": 2}}) + "\n",
        encoding="utf-8",
    )
    pass2_data_path = tmp_path / "pass2.jsonl"
    pass2_data_path.write_text(
        json.dumps({"metadata": {"source_id": "inline-1", "total_turns": 1}}) + "\n",
        encoding="utf-8",
    )

    leaf_path = "files/sample.jsonl"
    file_scope_id = f"file:{leaf_path}"
    conversation_stats = {"_inject_conversation": True}

    def fake_build_scope_tree(*args, **kwargs):
        return {
            "root_id": "global",
            "scopes": {
                "global": {
                    "id": "global",
                    "label": "global",
                    "kind": "global",
                    "path": "",
                    "parent_id": None,
                    "children": [file_scope_id],
                    "descendant_files": [leaf_path],
                    "raw_pass1": conversation_stats,
                    "raw_pass2": {"total_scored": 1},
                    "raw_conversations": [],
                },
                file_scope_id: {
                    "id": file_scope_id,
                    "label": "sample.jsonl",
                    "kind": "file",
                    "path": leaf_path,
                    "parent_id": "global",
                    "children": [],
                    "descendant_files": [leaf_path],
                    "raw_pass1": conversation_stats,
                    "raw_pass2": {"total_scored": 1},
                    "raw_conversations": [],
                    "pass1_data_path": str(pass1_data_path),
                    "pass2_data_path": str(pass2_data_path),
                },
            },
        }

    original_compute = visualize_value_module.compute_viz_data

    def fake_compute_viz_data(samples, stats, conv_records=None):
        payload = dict(original_compute(samples, stats))
        if stats and stats.get("_inject_conversation"):
            modes = dict(payload.get("modes") or {})
            modes["conversation"] = {
                "overview": {"total_scored": 2},
                "total": 2,
                "mode_id": "conversation",
            }
            payload["modes"] = modes
        return payload

    monkeypatch.setattr(visualize_value_module, "build_scope_tree", fake_build_scope_tree)
    def fake_infer(path):
        if path == str(pass1_data_path):
            return "multi"
        if path == str(pass2_data_path):
            return "single"
        return None

    monkeypatch.setattr(visualize_value_module, "infer_scope_turn_kind_from_path", fake_infer)
    monkeypatch.setattr(visualize_value_module, "compute_viz_data", fake_compute_viz_data)

    payload, _, _ = visualize_value_module._tree_payload(tmp_path, include_conversations=True)
    global_scope = payload["scopes"]["global"]
    conversation_mode = ((global_scope.get("pass1") or {}).get("modes") or {}).get("conversation")

    assert conversation_mode is not None
    assert conversation_mode["total"] == 2
    assert global_scope["summary_modes"]["conversation"]["pass1_total"] == 2


def test_tree_payload_merges_pass1_and_pass2_turn_kinds(tmp_path, monkeypatch):
    pass1_data_path = tmp_path / "pass1.jsonl"
    pass1_data_path.write_text(
        json.dumps({"metadata": {"conversation_uid": "conv-1", "total_turns": 2}}) + "\n",
        encoding="utf-8",
    )
    pass2_data_path = tmp_path / "pass2.jsonl"
    pass2_data_path.write_text(
        json.dumps({"metadata": {"source_id": "inline-1", "total_turns": 1}}) + "\n",
        encoding="utf-8",
    )

    leaf_path = "files/sample.jsonl"
    file_scope_id = f"file:{leaf_path}"

    def fake_build_scope_tree(*args, **kwargs):
        return {
            "root_id": "global",
            "scopes": {
                "global": {
                    "id": "global",
                    "label": "global",
                    "kind": "global",
                    "path": "",
                    "parent_id": None,
                    "children": [file_scope_id],
                    "descendant_files": [leaf_path],
                    "raw_pass1": {"total_samples": 1},
                    "raw_pass2": {"total_scored": 1},
                    "raw_conversations": [],
                },
                file_scope_id: {
                    "id": file_scope_id,
                    "label": "sample.jsonl",
                    "kind": "file",
                    "path": leaf_path,
                    "parent_id": "global",
                    "children": [],
                    "descendant_files": [leaf_path],
                    "raw_pass1": {"total_samples": 1},
                    "raw_pass2": {"total_scored": 1},
                    "raw_conversations": [],
                    "pass1_data_path": str(pass1_data_path),
                    "pass2_data_path": str(pass2_data_path),
                },
            },
        }

    monkeypatch.setattr(visualize_value_module, "build_scope_tree", fake_build_scope_tree)

    payload, _, _ = visualize_value_module._tree_payload(tmp_path, include_conversations=False)
    file_scope = payload["scopes"][file_scope_id]

    assert file_scope["turn_kind"] == "mixed"


def test_generate_value_dashboard_tree_resolves_deferred_policy_before_tree_loading(tmp_path, monkeypatch):
    meta_root = tmp_path / "meta_label_data"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / PASS2_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "total_scored": 25000,
                "postprocess": {
                    "conversation_scores": {"status": "deferred", "reason": "samples=25000>=20000"},
                    "dashboard": {"status": "deferred", "reason": "samples=25000>=20000"},
                },
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def _fake_tree_payload(_run_dir, *, include_conversations=True):
        captured["include_conversations"] = include_conversations
        payload = {
            "title": "SFT Labeling & Scoring Dashboard",
            "title_key": "dashboard_title_scoring",
            "subtitle": "test",
            "root_id": "global",
            "default_scope_id": "global",
            "initially_expanded": ["global"],
            "scopes": {
                "global": {
                    "id": "global",
                    "label": "global",
                    "kind": "global",
                    "path": "",
                    "parent_id": None,
                    "children": [],
                    "descendant_files": [],
                    "pass1": None,
                    "pass2": None,
                    "conversation": None,
                    "turn_kind": "mixed",
                    "summary": {},
                }
            },
        }
        return payload, [], {"total_scored": 25000}

    monkeypatch.setattr(visualize_value_module, "_tree_payload", _fake_tree_payload)
    monkeypatch.setattr(visualize_value_module, "_write_dashboard_bundle", lambda *args, **kwargs: None)

    generate_value_dashboard(meta_root, scored_file=None, stats_file=PASS2_SUMMARY_STATS_FILE, quiet=True)

    assert captured["include_conversations"] is True


def test_generate_value_dashboard_tree_fail_closes_when_completed_metadata_has_no_conversation_artifact(tmp_path):
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / PASS2_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample.jsonl",
                "total_scored": 1,
                "total_failed": 0,
                "postprocess": {
                    "conversation_scores": {"status": "completed"},
                    "dashboard": {"status": "completed"},
                },
                "score_distributions": {
                    "value_score": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "selection_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                },
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
                "postprocess": {
                    "conversation_scores": {"status": "completed"},
                    "dashboard": {"status": "completed"},
                },
                "score_distributions": {
                    "value_score": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "selection_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                },
                "per_file_summary": [
                    {"file": "code/sample.jsonl", "count": 1, "mean_value": 6.0, "mean_selection": 6.5}
                ],
            }
        ),
        encoding="utf-8",
    )

    out = generate_value_dashboard(meta_root, scored_file=None, stats_file=PASS2_SUMMARY_STATS_FILE, quiet=True)
    data_dir = out.with_name(f"{out.stem}.data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert manifest["scopes"]["global"]["has_conversation"] is False
    assert detail["summary_modes"]["conversation"]["scored_total"] == 0
    assert "conversation" not in ((detail.get("pass2") or {}).get("modes") or {})


def test_compute_value_viz_data_conversation_mode_deduplicates_single_reply_conv_records():
    samples = [
        {
            "id": "row-1",
            "metadata": {
                "source_id": "inline-row-1",
                "source_file": "inline.jsonl",
                "turn_index": 1,
                "total_turns": 1,
            },
            "labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
            "value": {
                "value_score": 6.0,
                "selection_score": 6.2,
                "thinking_mode": "fast",
                "confidence": 0.8,
                "quality": {"overall": 6.0},
                "complexity": {"overall": 5.0},
                "reasoning": {"overall": 5.0},
                "rarity": {"score": 4.0},
            },
        }
    ]
    conv_records = [
        {
            "conversation_id": "inline-row-1",
            "conversation_key": "inline.jsonl::inline-row-1",
            "source_file": "inline.jsonl",
            "turn_count": 1,
            "conv_value": 6.1,
            "conv_selection": 6.3,
            "peak_complexity": 5,
            "merged_labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
            "detail": {"score_confidence": 0.8},
        }
    ]

    viz = compute_value_viz_data(samples, {"total_scored": 1, "total_failed": 0}, conv_records)

    assert viz["modes"]["sample"]["overview"]["total_scored"] == 1
    assert viz["modes"]["conversation"]["overview"]["total_scored"] == 1
    assert viz["modes"]["conversation"]["per_file_summary"][0]["count"] == 1


def test_resolve_explorer_policy_keeps_completed_large_run_enabled():
    policy = visualize_value_module._resolve_explorer_policy(
        {
            "total_scored": 25000,
            "postprocess": {
                "dashboard": {"status": "completed", "mode": "complete-postprocess"},
                "conversation_scores": {"status": "completed", "mode": "complete-postprocess"},
            },
        }
    )

    assert policy["enabled"] is True


def test_resolve_explorer_policy_fail_closed_for_missing_pending_and_failed_metadata():
    missing_policy = visualize_value_module._resolve_explorer_policy({"total_scored": 1})
    pending_policy = visualize_value_module._resolve_explorer_policy(
        {"total_scored": 1, "postprocess": {"dashboard": {"status": "pending"}}}
    )
    failed_policy = visualize_value_module._resolve_explorer_policy(
        {"total_scored": 1, "postprocess": {"dashboard": {"status": "failed", "reason": "worker-crashed"}}}
    )

    assert missing_policy["enabled"] is False
    assert missing_policy["mode"] == "missing"
    assert pending_policy == {"enabled": False, "mode": "pending"}
    assert failed_policy == {"enabled": False, "mode": "failed", "reason": "worker-crashed"}


def test_turn_kind_treats_inline_single_reply_source_id_as_single():
    assert visualize_value_module.infer_scope_turn_kind(
        [
            {
                "metadata": {
                    "source_id": "inline-row-1",
                    "source_file": "inline.jsonl",
                    "turn_index": 1,
                    "total_turns": 1,
                }
            }
        ]
    ) == "single"


def test_turn_kind_from_path_treats_inline_single_reply_source_id_as_single(tmp_path):
    data_path = tmp_path / "scored.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_id": "inline-row-1",
                    "source_file": "inline.jsonl",
                    "turn_index": 1,
                    "total_turns": 1,
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert visualize_value_module.infer_scope_turn_kind_from_path(data_path) == "single"


def test_generate_value_dashboard_single_scope_normalizes_input_file_to_source_path(tmp_path):
    scored_row = {
        "id": "sample-1",
        "metadata": {"source_file": "code/sample.jsonl", "source_id": "conv-1", "turn_index": 1, "total_turns": 2},
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
    }
    (tmp_path / "scored.jsonl").write_text(
        json.dumps(scored_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "conversation_scores.json").write_text(
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
                    "detail": {"quality_overall": 6.0, "reasoning_overall": 6.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / PASS2_STATS_FILE).write_text(
        json.dumps(
                {
                    "input_file": str(tmp_path / "labeled.jsonl"),
                    "total_scored": 1,
                    "total_failed": 0,
                    "postprocess": {
                        "conversation_scores": {"status": "completed"},
                        "dashboard": {"status": "completed"},
                    },
                    "score_distributions": {
                    "value_score": {"mean": 6.5, "min": 6.5, "max": 6.5},
                    "selection_score": {"mean": 7.1, "min": 7.1, "max": 7.1},
                    "complexity_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "quality_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
                    "reasoning_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
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

    out = generate_value_dashboard(tmp_path, scored_file="scored.jsonl", stats_file=PASS2_STATS_FILE, quiet=True)
    data_dir = out.with_name(f"{out.stem}.data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert manifest["scopes"]["global"]["path"] == "code/sample.jsonl"
    assert detail["pass2"]["modes"]["sample"]["overview"]["input_file"] == "code/sample.jsonl"
    assert detail["pass2"]["modes"]["conversation"]["overview"]["input_file"] == "code/sample.jsonl"


def test_generate_value_dashboard_tree_payload_aggregates_conversation_mode_for_parent_scopes(tmp_path):
    meta_root = tmp_path / "meta_label_data"
    code_leaf = meta_root / "files" / "code" / "sample_a"
    docs_leaf = meta_root / "files" / "docs" / "sample_b"
    code_leaf.mkdir(parents=True, exist_ok=True)
    docs_leaf.mkdir(parents=True, exist_ok=True)

    code_rows = [
        {
            "id": "code-1",
            "metadata": {"source_file": "code/sample_a.jsonl", "source_id": "conv-1", "turn_index": 1, "total_turns": 2},
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
            "metadata": {"source_file": "code/sample_a.jsonl", "source_id": "conv-1", "turn_index": 2, "total_turns": 2},
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
            "metadata": {"source_file": "docs/sample_b.jsonl", "source_id": "conv-2", "turn_index": 1, "total_turns": 2},
            "labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
            "value": {
                "value_score": 3.0,
                "selection_score": 3.4,
                "thinking_mode": "fast",
                "quality": {"overall": 3.0},
                "complexity": {"overall": 3.0},
                "reasoning": {"overall": 3.0},
                "rarity": {"score": 3.0},
                "confidence": 0.8,
            },
        },
        {
            "id": "docs-2",
            "metadata": {"source_file": "docs/sample_b.jsonl", "source_id": "conv-2", "turn_index": 2, "total_turns": 2},
            "labels": {"intent": "build", "difficulty": "beginner", "context": "snippet"},
            "value": {
                "value_score": 7.0,
                "selection_score": 7.4,
                "thinking_mode": "fast",
                "quality": {"overall": 5.0},
                "complexity": {"overall": 7.0},
                "reasoning": {"overall": 5.0},
                "rarity": {"score": 7.0},
                "confidence": 0.8,
            },
        },
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
        json.dumps(aggregate_conversations(code_rows)),
        encoding="utf-8",
    )
    (docs_leaf / "conversation_scores.json").write_text(
        json.dumps(aggregate_conversations(docs_rows)),
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
                    "total_scored": 2,
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
                    "postprocess": {
                        "dashboard": {"status": "completed"},
                        "conversation_scores": {"status": "completed"},
                    },
                    "total_scored": 4,
                    "total_failed": 0,
                    "score_distributions": {
                        "value_score": {"mean": 5.5},
                        "selection_score": {"mean": 5.85},
                        "complexity_overall": {"mean": 5.5},
                        "quality_overall": {"mean": 5.5},
                        "reasoning_overall": {"mean": 5.5},
                        "rarity_score": {"mean": 5.5},
                        "confidence": {"mean": 0.8},
                    },
                    "histograms": {},
                    "flag_counts": {},
                    "per_file_summary": [
                        {"file": "code/sample_a.jsonl", "count": 2, "mean_value": 6.0, "mean_complexity": 6.0, "mean_quality": 6.0, "mean_selection": 6.3},
                        {"file": "docs/sample_b.jsonl", "count": 2, "mean_value": 5.0, "mean_complexity": 5.0, "mean_quality": 5.0, "mean_selection": 5.4},
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
    assert code_detail["pass2"]["modes"]["conversation"]["overview"]["mean_quality"] == 6.67
    assert global_detail["pass2"]["modes"]["conversation"]["overview"]["mean_quality"] == 5.5
    assert global_detail["pass2"]["modes"]["conversation"]["score_distributions"]["quality_overall"]["mean"] == 5.5
    assert global_detail["summary_modes"]["conversation"]["scored_total"] == 2
    assert global_detail["conversation"]["total"] == 2
    assert global_detail["conversation"]["mean_conv_value"] == 5.24


def test_tree_payload_global_conversation_mode_includes_single_turn_samples(tmp_path):
    """Global conversation mode should include single-turn samples from root-level conversation_scores.json."""
    meta_root = tmp_path / "meta_label_data"
    leaf_dir = meta_root / "files" / "code" / "mixed"
    leaf_dir.mkdir(parents=True, exist_ok=True)

    # Multi-turn slices (2 slices for conv-1)
    multi_turn_slices = [
        {
            "id": "mt-1",
            "metadata": {"source_file": "code/mixed.jsonl", "source_id": "conv-1", "turn_index": 1, "total_turns": 2, "conversation_uid": "conv-1"},
            "labels": {"intent": "build", "difficulty": "intermediate", "context": "snippet"},
            "value": {
                "value_score": 5.0, "selection_score": 5.5, "thinking_mode": "fast",
                "quality": {"overall": 5.0}, "complexity": {"overall": 5.0},
                "reasoning": {"overall": 5.0}, "rarity": {"score": 5.0}, "confidence": 0.8,
            },
        },
        {
            "id": "mt-2",
            "metadata": {"source_file": "code/mixed.jsonl", "source_id": "conv-1", "turn_index": 2, "total_turns": 2, "conversation_uid": "conv-1"},
            "labels": {"intent": "debug", "difficulty": "advanced", "context": "repository"},
            "value": {
                "value_score": 7.0, "selection_score": 7.5, "thinking_mode": "fast",
                "quality": {"overall": 7.0}, "complexity": {"overall": 7.0},
                "reasoning": {"overall": 7.0}, "rarity": {"score": 7.0}, "confidence": 0.9,
            },
        },
    ]
    # Single-turn sample
    single_turn_sample = {
        "id": "st-1",
        "metadata": {"source_file": "code/mixed.jsonl", "source_id": "single-1"},
        "labels": {"intent": "explain", "difficulty": "beginner", "context": "snippet"},
        "value": {
            "value_score": 4.0, "selection_score": 4.5, "thinking_mode": "fast",
            "quality": {"overall": 4.0}, "complexity": {"overall": 3.0},
            "reasoning": {"overall": 4.0}, "rarity": {"score": 3.0}, "confidence": 0.7,
        },
    }
    all_samples = multi_turn_slices + [single_turn_sample]

    (leaf_dir / "scored.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in all_samples) + "\n",
        encoding="utf-8",
    )

    # Per-file conversation_scores.json: only multi-turn (matches aggregate_conversations behavior)
    from sft_label.conversation import aggregate_conversations
    per_file_conv = aggregate_conversations(all_samples)  # only returns multi-turn
    assert len(per_file_conv) == 1, "aggregate_conversations should only return 1 multi-turn record"
    (leaf_dir / "conversation_scores.json").write_text(
        json.dumps(per_file_conv), encoding="utf-8",
    )

    # Root-level conversation_scores.json: both multi-turn AND single-turn
    from sft_label.inline_scoring import _single_turn_conversation_update
    single_turn_conv = _single_turn_conversation_update(single_turn_sample)
    root_conv = per_file_conv + [single_turn_conv]
    assert len(root_conv) == 2, "root-level should have both multi-turn and single-turn"
    (meta_root / "conversation_scores.json").write_text(
        json.dumps(root_conv), encoding="utf-8",
    )

    # Stats files
    (leaf_dir / PASS2_STATS_FILE).write_text(
        json.dumps({
            "input_file": "code/mixed.jsonl",
            "total_scored": 3,
            "total_failed": 0,
            "score_distributions": {
                "value_score": {"mean": 5.33},
                "selection_score": {"mean": 5.83},
                "complexity_overall": {"mean": 5.0},
                "quality_overall": {"mean": 5.33},
                "reasoning_overall": {"mean": 5.33},
                "confidence": {"mean": 0.8},
            },
            "histograms": {},
            "flag_counts": {},
            "value_by_tag": {},
            "selection_by_tag": {},
        }),
        encoding="utf-8",
    )
    (meta_root / PASS2_SUMMARY_STATS_FILE).write_text(
        json.dumps({
            "input_path": "dataset",
            "total_scored": 3,
            "total_failed": 0,
            "postprocess": {
                "dashboard": {"status": "completed"},
                "conversation_scores": {"status": "completed"},
            },
            "score_distributions": {
                "value_score": {"mean": 5.33},
                "selection_score": {"mean": 5.83},
                "complexity_overall": {"mean": 5.0},
                "quality_overall": {"mean": 5.33},
                "reasoning_overall": {"mean": 5.33},
                "confidence": {"mean": 0.8},
            },
            "histograms": {},
            "flag_counts": {},
            "per_file_summary": [{"file": "code/mixed.jsonl", "count": 3, "mean_value": 5.33}],
        }),
        encoding="utf-8",
    )

    from sft_label.tools.visualize_value import _tree_payload
    manifest, _, _ = _tree_payload(meta_root)
    global_scope = manifest["scopes"]["global"]
    conv_mode = global_scope["pass2"]["modes"]["conversation"]

    # BUG: Before fix, this would be 1 (only multi-turn).
    # After fix, should be 2 (1 multi-turn conv + 1 single-turn unit).
    assert conv_mode["overview"]["total_scored"] == 2

    # summary_modes should also reflect the correct count
    assert global_scope["summary_modes"]["conversation"]["scored_total"] == 2
