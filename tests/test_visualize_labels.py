from __future__ import annotations

import json
import re

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


def test_generate_dashboard_stats_only_single_scope_uses_stats_and_explorer_assets(tmp_path):
    sample = _sample_with_unmapped(
        "sample-1",
        "Add WebGPU rendering support to this pipeline.",
        [{"dimension": "task", "value": "gpu-porting"}],
    )
    (tmp_path / "labeled.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = {
        "total_samples": 1,
        "input_file": "train.jsonl",
        "success_rate": 1.0,
        "total_tokens": 123,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 1,
        "tag_distributions": {"task": {"feature-implementation": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {"task:gpu-porting": 1},
    }
    (tmp_path / PASS1_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")

    out = generate_dashboard(tmp_path, labeled_file=None, stats_file=PASS1_STATS_FILE)
    html = out.read_text(encoding="utf-8")
    assets_dir = out.with_suffix("").with_name(f"{out.stem}.assets")

    assert assets_dir.exists()
    assert any(path.name.startswith("preview_") for path in assets_dir.iterdir())
    assert any(path.name.startswith("detail_") for path in assets_dir.iterdir())

    match = re.search(r"const DATA = (.*?);\nconst STATE", html, re.S)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["scopes"]["global"]["summary"]["pass1_total"] == 1
    assert payload["scopes"]["global"]["pass1"]["unmapped_details"]["total_occurrences"] == 1
    assert payload["explorer"]["enabled"] is True
    assert payload["scopes"]["global"]["explorer"]["sample_count"] == 1
    assert payload["scopes"]["global"].get("pass2") is None
    assert 'scopeHasPass2Data(scope)' in html


def test_compute_viz_data_conversation_mode_merges_multiturn_tags():
    samples = [
        {
            "id": "conv-1-s1",
            "metadata": {"source_id": "conv-1", "turn_index": 1, "total_turns": 2},
            "conversations": [{"from": "human", "value": "first"}, {"from": "gpt", "value": "ok"}],
            "labels": {
                "intent": "build",
                "difficulty": "intermediate",
                "language": ["python"],
                "domain": [],
                "concept": [],
                "task": ["feature-implementation"],
                "agentic": ["planning"],
                "constraint": [],
                "context": "snippet",
                "confidence": {"intent": 0.8, "agentic": 0.7, "difficulty": 0.8},
                "unmapped": [],
            },
        },
        {
            "id": "conv-1-s2",
            "metadata": {"source_id": "conv-1", "turn_index": 2, "total_turns": 2},
            "conversations": [{"from": "human", "value": "second"}, {"from": "gpt", "value": "ok"}],
            "labels": {
                "intent": "debug",
                "difficulty": "advanced",
                "language": ["python", "sql"],
                "domain": [],
                "concept": [],
                "task": ["bug-fixing"],
                "agentic": ["planning", "file-operations"],
                "constraint": [],
                "context": "repository",
                "confidence": {"intent": 0.9, "agentic": 0.9, "difficulty": 0.9},
                "unmapped": [],
            },
        },
    ]
    stats = {
        "total_samples": 2,
        "success_rate": 1.0,
        "total_tokens": 0,
        "arbitrated_rate": 0.0,
        "tag_distributions": {
            "agentic": {"planning": 2, "file-operations": 1},
            "intent": {"build": 1, "debug": 1},
            "difficulty": {"intermediate": 1, "advanced": 1},
        },
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
    }

    viz = compute_viz_data(samples, stats)

    assert viz["modes"]["sample"]["total"] == 2
    assert viz["modes"]["conversation"]["total"] == 1
    assert viz["modes"]["conversation"]["distributions"]["agentic"] == {
        "planning": 1,
        "file-operations": 1,
    }
    assert viz["modes"]["conversation"]["cross_matrix"]["data"]["debug|advanced"] == 1


def test_generate_dashboard_renders_aggregation_toggle(tmp_path):
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

    assert 'aggregationMode: "sample"' in html
    assert 'data-aggregation-mode' in html
    assert 'language_toggle_help' in html
    assert 'dashboard.locale' in html
    assert 'Aggregation mode' in html
