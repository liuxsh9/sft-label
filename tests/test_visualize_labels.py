from __future__ import annotations

import json
import os
import re

from sft_label.artifacts import PASS1_STATS_FILE, PASS1_SUMMARY_STATS_FILE
from sft_label.tools import visualize_labels as visualize_labels_module
from sft_label.tools.dashboard_aggregation import infer_scope_turn_kind, infer_scope_turn_kind_from_path
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


def _bootstrap_from_html(html: str) -> dict:
    match = re.search(
        r'<script id="dashboard-bootstrap" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


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


def test_compute_viz_data_exposes_prompt_mode_budget():
    viz = compute_viz_data([], {
        "total_samples": 0,
        "tag_distributions": {},
        "confidence_stats": {},
        "cross_matrix": {},
        "prompt_mode": "compact",
        "compact_prompt": True,
        "conversation_char_budget": 8000,
        "fewshot_variant": "compact",
    })

    assert viz["overview"]["prompt_mode"] == "compact"
    assert viz["overview"]["compact_prompt"] is True
    assert viz["overview"]["conversation_char_budget"] == 8000
    assert viz["overview"]["fewshot_variant"] == "compact"


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
    bootstrap = _bootstrap_from_html(html)
    data_dir = out.with_name(f"{out.stem}.data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert bootstrap["manifestUrl"] == "dashboard_labeling.data/manifest.json"
    assert bootstrap["manifestScriptUrl"] == "dashboard_labeling.data/manifest.js"
    assert (data_dir / "manifest.js").exists()
    assert (data_dir / "scopes" / "global.js").exists()
    assert manifest["scopes"]["global"]["summary"]["pass1_total"] == 1
    assert detail["pass1"]["modes"]["sample"]["unmapped_details"]["by_dimension"]["task"][0]["label"] == "gpu-porting"
    assert detail["pass1"]["modes"]["sample"]["unmapped_details"]["by_dimension"]["task"][0]["examples"][0]["id"] == "sample-1"


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
    _bootstrap_from_html(html)
    data_dir = out.with_name(f"{out.stem}.data")
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    rows = detail["pass1"]["modes"]["sample"]["unmapped_details"]["by_dimension"]["task"]
    assert rows[0]["label"] == "gpu-porting"
    assert rows[0]["count"] == 2
    assert rows[0]["examples"] == []


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
    bootstrap = _bootstrap_from_html(html)
    data_dir = out.with_name(f"{out.stem}.data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))
    explorer_dir = data_dir / "explorer"

    assert bootstrap["manifestUrl"] == "dashboard_labeling.data/manifest.json"
    assert "dashboard.js" in html
    assert "const DATA =" not in html
    assert explorer_dir.exists()
    assert any(path.name.startswith("preview_") for path in explorer_dir.iterdir())
    assert any(path.name.startswith("detail_") for path in explorer_dir.iterdir())
    assert manifest["scopes"]["global"]["summary"]["pass1_total"] == 1
    assert manifest["explorer"]["enabled"] is True
    assert manifest["scopes"]["global"]["explorer"]["sample_count"] == 1
    assert detail["pass1"]["modes"]["sample"]["unmapped_details"]["total_occurrences"] == 1
    assert detail["pass1"]["modes"]["sample"].get("conf_matrix") is None
    assert detail.get("pass2") is None


def test_generate_dashboard_stats_only_single_scope_prefers_conversation_stats_artifact(tmp_path, monkeypatch):
    (tmp_path / "labeled.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "metadata": {"source_file": "train.jsonl", "source_id": "conv-1", "turn_index": 1},
                "conversations": [{"from": "human", "value": "first"}, {"from": "gpt", "value": "ok"}],
                "labels": _sample_with_unmapped("sample-1", "first", [])["labels"],
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "conversation_stats_labeling.json").write_text(
        json.dumps(
            {
                "total": 1,
                "distribution_total": 1,
                "input_file": "train.jsonl",
                "distributions": {"intent": {"build": 1}},
                "unmapped_details": {"total_occurrences": 0, "by_dimension": {}},
                "confidence_stats": {},
                "conf_matrix": [],
                "coverage": {"intent": 1},
                "cross_matrix": {"rows": [], "cols": [], "data": {}},
                "overview": {"success_rate": 1.0, "unmapped_unique": 0},
                "unit_label": "units",
                "mode_id": "conversation",
            }
        ),
        encoding="utf-8",
    )
    stats = {
        "total_samples": 1,
        "input_file": "train.jsonl",
        "success_rate": 1.0,
        "total_tokens": 123,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 0,
        "tag_distributions": {"intent": {"build": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
    }
    (tmp_path / PASS1_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")
    fresh_time = (tmp_path / PASS1_STATS_FILE).stat().st_mtime + 5
    os.utime(tmp_path / "conversation_stats_labeling.json", (fresh_time, fresh_time))

    def _fail_load(_path):
        raise AssertionError("stats-only single-scope dashboard should use conversation_stats_labeling.json")

    monkeypatch.setattr(visualize_labels_module, "load_data_file", _fail_load)

    out = generate_dashboard(tmp_path, labeled_file=None, stats_file=PASS1_STATS_FILE)
    data_dir = out.with_name(f"{out.stem}.data")
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert detail["pass1"]["modes"]["conversation"]["total"] == 1


def test_generate_dashboard_stats_only_single_scope_ignores_stale_conversation_stats_artifact(tmp_path, monkeypatch):
    sample = {
        "id": "sample-1",
        "metadata": {"source_file": "train.jsonl", "source_id": "conv-1", "turn_index": 1},
        "conversations": [{"from": "human", "value": "first"}, {"from": "gpt", "value": "ok"}],
        "labels": _sample_with_unmapped("sample-1", "first", [])["labels"],
    }
    data_path = tmp_path / "labeled.jsonl"
    data_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    conversation_stats_path = tmp_path / "conversation_stats_labeling.json"
    conversation_stats_path.write_text(
        json.dumps(
            {
                "total": 99,
                "distribution_total": 99,
                "input_file": "train.jsonl",
                "distributions": {"intent": {"stale": 99}},
                "unmapped_details": {"total_occurrences": 0, "by_dimension": {}},
                "confidence_stats": {},
                "conf_matrix": [],
                "coverage": {"intent": 1},
                "cross_matrix": {"rows": [], "cols": [], "data": {}},
                "overview": {"success_rate": 1.0, "unmapped_unique": 0},
                "unit_label": "units",
                "mode_id": "conversation",
            }
        ),
        encoding="utf-8",
    )
    stats = {
        "total_samples": 1,
        "input_file": "train.jsonl",
        "success_rate": 1.0,
        "total_tokens": 123,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 0,
        "tag_distributions": {"intent": {"build": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
    }
    stats_path = tmp_path / PASS1_STATS_FILE
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    stale_time = stats_path.stat().st_mtime - 5
    os.utime(conversation_stats_path, (stale_time, stale_time))

    called = {"count": 0}
    real_load_data_file = visualize_labels_module.load_data_file

    def _tracking_load(path):
        called["count"] += 1
        return real_load_data_file(path)

    monkeypatch.setattr(visualize_labels_module, "load_data_file", _tracking_load)

    out = generate_dashboard(tmp_path, labeled_file=None, stats_file=PASS1_STATS_FILE)
    data_dir = out.with_name(f"{out.stem}.data")
    detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert called["count"] == 1
    assert detail["pass1"]["modes"]["conversation"]["total"] == 1
    assert detail["pass1"]["modes"]["conversation"]["distributions"]["intent"] == {"build": 1}


def test_generate_dashboard_tree_payload_uses_stats_without_preloading_leaf_samples(tmp_path, monkeypatch):
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "labeled.jsonl").write_text(
        json.dumps(
            _sample_with_unmapped(
                "sample-1",
                "Add WebGPU rendering support to this pipeline.",
                [{"dimension": "task", "value": "gpu-porting"}],
            ),
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / PASS1_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample.jsonl",
                "total_samples": 1,
                "success_rate": 1.0,
                "tag_distributions": {"task": {"feature-implementation": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {"task:gpu-porting": 1},
            }
        ),
        encoding="utf-8",
    )
    (meta_root / PASS1_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "input_path": "dataset",
                "total_samples": 1,
                "success_rate": 1.0,
                "tag_distributions": {"task": {"feature-implementation": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {"task:gpu-porting": 1},
            }
        ),
        encoding="utf-8",
    )

    def _fail_load(_path):
        raise AssertionError("tree payload should not preload leaf sample files")

    monkeypatch.setattr(visualize_labels_module, "load_data_file", _fail_load)

    out = generate_dashboard(meta_root, labeled_file=None, stats_file=PASS1_SUMMARY_STATS_FILE)
    data_dir = out.with_name(f"{out.stem}.data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["scopes"]["global"]["summary"]["pass1_total"] == 1
    assert manifest["scopes"]["file:code/sample.jsonl"]["explorer"]["sample_count"] == 1


def test_generate_dashboard_tree_payload_prefers_conversation_stats_artifact_over_leaf_reload(tmp_path, monkeypatch):
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "labeled.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "metadata": {"source_file": "code/sample.jsonl", "source_id": "conv-1", "turn_index": 1},
                "conversations": [{"from": "human", "value": "first"}, {"from": "gpt", "value": "ok"}],
                "labels": _sample_with_unmapped("sample-1", "first", [])["labels"],
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / PASS1_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample.jsonl",
                "total_samples": 1,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "conversation_stats_labeling.json").write_text(
        json.dumps(
            {
                "total": 1,
                "distribution_total": 1,
                "input_file": "code/sample.jsonl",
                "distributions": {"intent": {"build": 1}},
                "unmapped_details": {"total_occurrences": 0, "by_dimension": {}},
                "confidence_stats": {},
                "conf_matrix": [],
                "coverage": {"intent": 1},
                "cross_matrix": {"rows": [], "cols": [], "data": {}},
                "overview": {"success_rate": 1.0, "unmapped_unique": 0},
                "unit_label": "units",
                "mode_id": "conversation",
            }
        ),
        encoding="utf-8",
    )
    (meta_root / PASS1_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "input_path": "dataset",
                "total_samples": 1,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            }
        ),
        encoding="utf-8",
    )

    def _fail_iter(_path):
        raise AssertionError("dashboard should use conversation_stats_labeling.json instead of reloading leaf samples")

    monkeypatch.setattr(visualize_labels_module, "_load_scope_samples", _fail_iter)

    out = generate_dashboard(meta_root, labeled_file=None, stats_file=PASS1_SUMMARY_STATS_FILE)
    data_dir = out.with_name(f"{out.stem}.data")
    detail = json.loads((data_dir / "scopes" / "file_code_sample_jsonl.json").read_text(encoding="utf-8"))

    assert detail["pass1"]["modes"]["conversation"]["total"] == 1
    assert detail["summary_modes"]["conversation"]["pass1_total"] == 1


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

    assert 'dashboard-bootstrap' in html
    assert 'dashboard.js' in html
    assert 'dashboard.css' in html
    assert 'dashboard.locale' not in html


def test_generate_dashboard_reuses_shared_static_base_url_across_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SFT_LABEL_DASHBOARD_STATIC_BASE_URL", "/static/sft-label-dashboard/v1")
    stats = {
        "total_samples": 1,
        "input_file": "labeled.json",
        "success_rate": 1.0,
        "total_tokens": 1,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 0,
        "tag_distributions": {"task": {"feature-implementation": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
    }
    sample = _sample_with_unmapped("sample-1", "Add logging.", [])

    run_outputs = []
    for run_name in ("run_a", "run_b"):
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "labeled.json").write_text(json.dumps([sample]), encoding="utf-8")
        (run_dir / PASS1_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")
        run_outputs.append(generate_dashboard(run_dir))

    htmls = [path.read_text(encoding="utf-8") for path in run_outputs]
    assert all("/static/sft-label-dashboard/v1/dashboard.js" in html for html in htmls)
    assert all("/static/sft-label-dashboard/v1/dashboard.css" in html for html in htmls)
    assert not any((path.parent / "_dashboard_static").exists() for path in run_outputs)


def test_generated_dashboard_uses_sidebar_runtime_assets_for_compact_meta(tmp_path):
    stats = {
        "total_samples": 1,
        "input_file": "labeled.json",
        "success_rate": 1.0,
        "total_tokens": 1,
        "arbitrated_rate": 0.0,
        "unmapped_unique_count": 0,
        "tag_distributions": {"task": {"feature-implementation": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
    }
    sample = _sample_with_unmapped("sample-1", "Add logging.", [])
    (tmp_path / "labeled.json").write_text(json.dumps([sample]), encoding="utf-8")
    (tmp_path / PASS1_STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")

    out = generate_dashboard(tmp_path)
    css = (out.parent / "_dashboard_static" / "v1" / "dashboard.css").read_text(encoding="utf-8")
    js = (out.parent / "_dashboard_static" / "v1" / "dashboard.js").read_text(encoding="utf-8")

    assert '.tree-kind-dot {' in css
    assert '.tree-sub {' in css
    assert 'N ${fmtInt' in js
    assert '${escapeHtml(scopeKindLabel(scope.kind))}</span>' not in js


def test_generate_dashboard_tree_payload_aggregates_conversation_mode_from_leaf_samples(tmp_path):
    meta_root = tmp_path / "meta_label_data"
    code_leaf = meta_root / "files" / "code" / "sample_a"
    docs_leaf = meta_root / "files" / "docs" / "sample_b"
    code_leaf.mkdir(parents=True, exist_ok=True)
    docs_leaf.mkdir(parents=True, exist_ok=True)

    code_rows = [
        {
            "id": "code-1",
            "metadata": {"source_file": "code/sample_a.jsonl", "source_id": "conv-1", "turn_index": 1, "total_turns": 2},
            "conversations": [{"from": "human", "value": "first"}, {"from": "gpt", "value": "ok"}],
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
                "confidence": {"intent": 0.8},
                "unmapped": [],
            },
        },
        {
            "id": "code-2",
            "metadata": {"source_file": "code/sample_a.jsonl", "source_id": "conv-1", "turn_index": 2, "total_turns": 2},
            "conversations": [{"from": "human", "value": "second"}, {"from": "gpt", "value": "ok"}],
            "labels": {
                "intent": "debug",
                "difficulty": "advanced",
                "language": ["python"],
                "domain": [],
                "concept": [],
                "task": ["bug-fixing"],
                "agentic": [],
                "constraint": [],
                "context": "repository",
                "confidence": {"intent": 0.9},
                "unmapped": [],
            },
        },
    ]
    docs_rows = [
        {
            "id": "docs-1",
            "metadata": {"source_file": "docs/sample_b.jsonl"},
            "conversations": [{"from": "human", "value": "single"}, {"from": "gpt", "value": "ok"}],
            "labels": {
                "intent": "build",
                "difficulty": "beginner",
                "language": ["python"],
                "domain": [],
                "concept": [],
                "task": ["feature-implementation"],
                "agentic": [],
                "constraint": [],
                "context": "snippet",
                "confidence": {"intent": 0.7},
                "unmapped": [],
            },
        }
    ]

    (code_leaf / "labeled.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in code_rows) + "\n",
        encoding="utf-8",
    )
    (docs_leaf / "labeled.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in docs_rows) + "\n",
        encoding="utf-8",
    )

    (code_leaf / PASS1_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "code/sample_a.jsonl",
                "total_samples": 2,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 1, "debug": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            }
        ),
        encoding="utf-8",
    )
    (docs_leaf / PASS1_STATS_FILE).write_text(
        json.dumps(
            {
                "input_file": "docs/sample_b.jsonl",
                "total_samples": 1,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            }
        ),
        encoding="utf-8",
    )
    (meta_root / PASS1_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "input_path": "dataset",
                "total_samples": 3,
                "success_rate": 1.0,
                "tag_distributions": {"intent": {"build": 2, "debug": 1}},
                "confidence_stats": {},
                "cross_matrix": {},
                "unmapped_tags": {},
            }
        ),
        encoding="utf-8",
    )

    out = generate_dashboard(meta_root, labeled_file=None, stats_file=PASS1_SUMMARY_STATS_FILE)
    data_dir = out.with_name(f"{out.stem}.data")
    code_detail = json.loads((data_dir / "scopes" / "file_code_sample_a_jsonl.json").read_text(encoding="utf-8"))
    dir_detail = json.loads((data_dir / "scopes" / "dir_code.json").read_text(encoding="utf-8"))
    global_detail = json.loads((data_dir / "scopes" / "global.json").read_text(encoding="utf-8"))

    assert code_detail["pass1"]["modes"]["conversation"]["total"] == 1
    assert dir_detail["pass1"]["modes"]["conversation"]["total"] == 1
    assert global_detail["pass1"]["modes"]["conversation"]["total"] == 2
    assert global_detail["summary_modes"]["conversation"]["pass1_total"] == 2
    assert global_detail["pass1"]["modes"]["conversation"]["distributions"]["intent"] == {"build": 1, "debug": 1}


def test_infer_scope_turn_kind_distinguishes_single_multi_and_mixed():
    assert infer_scope_turn_kind([
        {"metadata": {}},
        {"metadata": {"source_id": None}},
    ]) == "single"
    assert infer_scope_turn_kind([
        {"metadata": {"source_id": "conv-1"}},
        {"metadata": {"source_id": "conv-2"}},
    ]) == "multi"
    assert infer_scope_turn_kind([
        {"metadata": {}},
        {"metadata": {"source_id": "conv-1"}},
    ]) == "mixed"


def test_infer_scope_turn_kind_from_json_path_streams_without_json_load(tmp_path, monkeypatch):
    data_path = tmp_path / "labeled.json"
    data_path.write_text(
        json.dumps(
            [
                {"metadata": {}},
                {"metadata": {"source_id": "conv-1"}},
            ]
        ),
        encoding="utf-8",
    )

    def _fail_json_load(*args, **kwargs):
        raise AssertionError("turn-kind inference should not full-load JSON arrays")

    monkeypatch.setattr("sft_label.tools.dashboard_aggregation.json.load", _fail_json_load)

    assert infer_scope_turn_kind_from_path(data_path) == "mixed"
