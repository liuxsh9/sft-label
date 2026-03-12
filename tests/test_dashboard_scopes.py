from __future__ import annotations

import json
from pathlib import Path

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
)
from sft_label.tools.dashboard_scopes import build_scope_tree


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


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
