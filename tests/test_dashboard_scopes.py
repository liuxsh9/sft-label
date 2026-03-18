from __future__ import annotations

import json
from pathlib import Path

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
)
from sft_label.tools import dashboard_aggregation as dashboard_aggregation_module
from sft_label.tools.dashboard_aggregation import infer_scope_turn_kind_from_path
from sft_label.tools.dashboard_scopes import build_scope_tree, merge_label_stats


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_merge_label_stats_preserves_combo_distributions():
    merged = merge_label_stats(
        [
            {
                "total_samples": 2,
                "success": 2,
                "failed": 0,
                "tag_distributions": {"intent": {"build": 2}},
                "combo_distributions": {"intent=build|difficulty=beginner": 2},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            },
            {
                "total_samples": 1,
                "success": 1,
                "failed": 0,
                "tag_distributions": {"intent": {"debug": 1}},
                "combo_distributions": {"intent=debug|difficulty=expert": 1},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            },
        ]
    )

    assert merged is not None
    assert merged["combo_distributions"] == {
        "intent=build|difficulty=beginner": 2,
        "intent=debug|difficulty=expert": 1,
    }


def test_build_scope_tree_deduplicates_inline_artifact_leaf_paths(tmp_path):
    run_root = tmp_path / "dataset_labeled_20260311_120000"
    meta_root = run_root / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"
    source_file = tmp_path / "fixtures" / "dataset" / "code" / "sample.jsonl"

    _write_json(
        artifact_dir / PASS1_STATS_FILE,
        {
            "input_file": str(source_file),
            "total_samples": 3,
            "success": 3,
            "tag_distributions": {},
            "confidence_stats": {},
            "cross_matrix": {},
            "unmapped_tags": {},
        },
    )
    _write_json(
        artifact_dir / PASS2_STATS_FILE,
        {
            "input_file": str(artifact_dir / "labeled.json"),
            "total_scored": 3,
            "total_failed": 0,
            "total_llm_calls": 3,
            "total_prompt_tokens": 30,
            "total_completion_tokens": 15,
            "total_tokens": 45,
            "score_distributions": {
                "value_score": {"mean": 6.5, "min": 5.0, "max": 8.0},
            },
            "histograms": {},
            "flag_counts": {},
        },
    )
    _write_jsonl(
        artifact_dir / "scored.jsonl",
        [
            {
                "id": "sample-1",
                "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}],
                "labels": {"intent": "learn", "language": ["python"], "difficulty": "beginner"},
                "value": {"value_score": 6.5, "quality": {"overall": 6.0}},
            }
        ],
    )
    _write_json(
        artifact_dir / "conversation_scores.json",
        [
            {
                "conversation_id": "sample-1",
                "conversation_key": "code/sample.jsonl::sample-1",
                "source_file": "code/sample.jsonl",
                "turn_count": 2,
                "conv_value": 6.5,
                "conv_selection": 6.2,
                "peak_complexity": 7,
            }
        ],
    )
    _write_json(
        meta_root / PASS1_SUMMARY_STATS_FILE,
        {
            "input_path": str(source_file.parents[1]),
            "total_samples": 3,
            "success": 3,
            "tag_distributions": {},
            "confidence_stats": {},
            "cross_matrix": {},
            "unmapped_tags": {},
        },
    )
    _write_json(
        meta_root / PASS2_SUMMARY_STATS_FILE,
        {
            "input_path": str(meta_root / "files"),
            "total_scored": 3,
            "total_failed": 0,
            "per_file_summary": [
                {
                    "file": "labeled.json",
                    "count": 3,
                    "mean_value": 6.5,
                    "mean_complexity": 0,
                    "mean_quality": 0,
                    "mean_selection": 0,
                }
            ],
            "score_distributions": {
                "value_score": {"mean": 6.5, "min": 5.0, "max": 8.0},
            },
            "histograms": {},
            "flag_counts": {},
        },
    )

    tree = build_scope_tree(meta_root)
    file_scopes = [scope for scope in tree["scopes"].values() if scope["kind"] == "file"]
    dir_scopes = [scope for scope in tree["scopes"].values() if scope["kind"] == "dir"]

    assert tree["scopes"]["global"]["label"] == "dataset"
    assert len(file_scopes) == 1
    assert len(dir_scopes) == 1
    assert file_scopes[0]["path"] == "code/sample.jsonl"
    assert file_scopes[0]["pass2_data_path"].endswith("scored.jsonl")
    assert file_scopes[0]["conversation_data_path"].endswith("conversation_scores.json")
    assert all(not scope["path"].endswith("labeled.json") for scope in file_scopes)
    assert tree["scopes"]["global"]["raw_pass2"]["per_file_summary"][0]["file"] == "code/sample.jsonl"


def test_build_scope_tree_does_not_duplicate_leaf_conversations_into_parent_scopes(tmp_path):
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"

    _write_json(
        artifact_dir / PASS2_STATS_FILE,
        {
            "input_file": "code/sample.jsonl",
            "total_scored": 1,
            "total_failed": 0,
            "total_llm_calls": 1,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "total_tokens": 15,
            "score_distributions": {"value_score": {"mean": 6.0, "min": 6.0, "max": 6.0}},
            "histograms": {},
            "flag_counts": {},
        },
    )
    _write_json(
        artifact_dir / "conversation_scores.json",
        [
            {
                "conversation_id": "conv-1",
                "conversation_key": "code/sample.jsonl::conv-1",
                "source_file": "code/sample.jsonl",
                "turn_count": 2,
                "conv_value": 6.0,
                "conv_selection": 6.3,
                "peak_complexity": 7,
            }
        ],
    )
    _write_json(
        meta_root / PASS2_SUMMARY_STATS_FILE,
        {
            "input_path": "dataset",
            "total_scored": 1,
            "total_failed": 0,
            "score_distributions": {"value_score": {"mean": 6.0, "min": 6.0, "max": 6.0}},
            "histograms": {},
            "flag_counts": {},
        },
    )

    tree = build_scope_tree(meta_root)

    assert tree["scopes"]["file:code/sample.jsonl"]["raw_conversations"] == [
        {
            "conversation_id": "conv-1",
            "conversation_key": "code/sample.jsonl::conv-1",
            "source_file": "code/sample.jsonl",
            "turn_count": 2,
            "conv_value": 6.0,
            "conv_selection": 6.3,
            "peak_complexity": 7,
        }
    ]
    assert tree["scopes"]["dir:code"]["raw_conversations"] == []
    assert tree["scopes"]["global"]["raw_conversations"] == []


def test_infer_scope_turn_kind_from_path_streams_json_array_without_json_load(tmp_path, monkeypatch):
    json_path = tmp_path / "labeled.json"
    json_path.write_text(
        json.dumps(
            [
                {"id": "sample-1", "metadata": {"source_id": "conv-1"}},
                {"id": "sample-2", "metadata": {}},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def _fail_json_load(*_args, **_kwargs):
        raise AssertionError("iter_data_file should not call json.load for .json artifacts")

    monkeypatch.setattr(dashboard_aggregation_module.json, "load", _fail_json_load)

    assert infer_scope_turn_kind_from_path(json_path) == "mixed"


def test_build_scope_tree_finds_legacy_suffixed_labeled_artifacts(tmp_path):
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"

    _write_json(
        artifact_dir / "stats_labeling_sample.json",
        {
            "input_file": "code/sample.jsonl",
            "total_samples": 1,
            "success": 1,
            "tag_distributions": {},
            "confidence_stats": {},
            "cross_matrix": {},
            "unmapped_tags": {},
        },
    )
    _write_jsonl(
        artifact_dir / "labeled_sample.jsonl",
        [
            {
                "id": "sample-1",
                "metadata": {"source_id": "conv-1", "source_file": "code/sample.jsonl"},
                "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}],
                "labels": {"intent": "learn", "language": ["python"], "difficulty": "beginner"},
            }
        ],
    )
    _write_json(
        meta_root / PASS1_SUMMARY_STATS_FILE,
        {
            "input_path": "dataset",
            "total_samples": 1,
            "success": 1,
            "tag_distributions": {},
            "confidence_stats": {},
            "cross_matrix": {},
            "unmapped_tags": {},
        },
    )

    tree = build_scope_tree(meta_root)
    file_scope = tree["scopes"]["file:code/sample.jsonl"]

    assert file_scope["pass1_data_path"].endswith("labeled_sample.jsonl")
