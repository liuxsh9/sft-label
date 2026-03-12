from __future__ import annotations

import json

from sft_label.artifacts import PASS1_STATS_FILE, PASS1_SUMMARY_STATS_FILE
from sft_label.tools.visualize_labels import compute_viz_data, generate_dashboard


def _sample_with_unmapped(sample_id: str, query: str, unmapped: list[dict]) -> dict:
    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": query},
            {"from": "gpt", "value": "ok"},
        ],
        "labels": {
            "intent": "build",
            "difficulty": "intermediate",
            "language": ["python"],
            "domain": [],
            "concept": [],
            "task": ["feature-implementation"],
            "agentic": [],
            "constraint": [],
            "context": "snippet",
            "confidence": {"intent": 0.9},
            "unmapped": unmapped,
        },
    }


def test_compute_viz_data_includes_unmapped_examples():
    samples = [
        _sample_with_unmapped(
            "sample-1",
            "Add WebGPU rendering support to this pipeline.",
            [{"dimension": "task", "value": "gpu-porting"}],
        ),
        _sample_with_unmapped(
            "sample-2",
            "Keep shader tests green while porting.",
            [{"dimension": "task", "value": "gpu-porting"}],
        ),
    ]
    stats = {
        "total_samples": 2,
        "tag_distributions": {"task": {"feature-implementation": 2}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {"task:gpu-porting": 2},
        "unmapped_unique_count": 1,
    }

    viz = compute_viz_data(samples, stats)

    assert viz["unmapped_details"]["total_occurrences"] == 2
    task_rows = viz["unmapped_details"]["by_dimension"]["task"]
    assert task_rows[0]["label"] == "gpu-porting"
    assert task_rows[0]["count"] == 2
    assert task_rows[0]["examples"][0]["id"] == "sample-1"
    assert "WebGPU rendering support" in task_rows[0]["examples"][0]["query"]


def test_generate_dashboard_renders_unmapped_section(tmp_path):
    samples = [
        _sample_with_unmapped(
            "sample-1",
            "Add WebGPU rendering support to this pipeline.",
            [{"dimension": "task", "value": "gpu-porting"}],
        )
    ]
    stats = {
        "total_samples": 1,
        "input_file": "labeled.json",
        "success_rate": 1.0,
        "total_tokens": 123,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 1,
        "tag_distributions": {"task": {"feature-implementation": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {"task:gpu-porting": 1},
    }
    (tmp_path / "labeled.json").write_text(json.dumps(samples), encoding="utf-8")
    (tmp_path / PASS1_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")

    out = generate_dashboard(tmp_path)
    html = out.read_text(encoding="utf-8")

    assert "Unmapped Tags" in html
    assert "gpu-porting" in html
    assert "sample-1" in html
    assert "WebGPU rendering support" in html


def test_generate_dashboard_tree_payload_keeps_unmapped_counts(tmp_path):
    stats = {
        "input_path": "dataset",
        "total_samples": 2,
        "success_rate": 1.0,
        "total_tokens": 10,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 1,
        "tag_distributions": {},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {"task:gpu-porting": 2},
    }
    (tmp_path / PASS1_SUMMARY_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")

    out = generate_dashboard(tmp_path, labeled_file=None, stats_file=PASS1_SUMMARY_STATS_FILE)
    html = out.read_text(encoding="utf-8")

    assert "Unmapped Tags" in html
    assert "gpu-porting" in html
    assert "No example sample loaded for this scope." in html
