from __future__ import annotations

import json

from sft_label.tools.analyze_unmapped import _discover_legacy_sample_files, run_unmapped_analysis


def test_run_unmapped_analysis_reports_sample_examples(tmp_path, capsys):
    path = tmp_path / "labeled.json"
    payload = [
        {
            "id": "sample-1",
            "conversations": [
                {"from": "human", "value": "Please add WebGPU support to this renderer."},
                {"from": "gpt", "value": "Sure."},
            ],
            "labels": {
                "intent": "build",
                "unmapped": [
                    {"dimension": "domain", "value": "graphics-programming"},
                    {"dimension": "task", "value": "gpu-porting"},
                ],
            },
        },
        {
            "id": "sample-2",
            "conversations": [
                {"from": "human", "value": "Port the shader pipeline and keep tests green."},
                {"from": "gpt", "value": "Done."},
            ],
            "labels": {
                "intent": "modify",
                "unmapped": [
                    {"dimension": "task", "value": "gpu-porting"},
                ],
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = run_unmapped_analysis(path, top=5, examples=1)
    out = capsys.readouterr().out

    assert summary["source_kind"] == "file"
    assert summary["total_occurrences"] == 3
    assert "Unmapped Tag Analysis" in out
    assert "DOMAIN" in out
    assert "TASK" in out
    assert "gpu-porting" in out
    assert "eg. [sample-1] Please add WebGPU support to this renderer." in out


def test_run_unmapped_analysis_falls_back_to_stats_file(tmp_path, capsys):
    stats_path = tmp_path / "stats_labeling.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_samples": 12,
                "distribution_total_samples": 10,
                "unmapped_tags": {
                    "task:gpu-porting": 4,
                    "domain:graphics-programming": 2,
                },
            },
            handle,
        )

    summary = run_unmapped_analysis(tmp_path, top=10)
    out = capsys.readouterr().out

    assert summary["source_kind"] == "stats"
    assert summary["total_occurrences"] == 6
    assert "Unmapped Tag Summary" in out
    assert "gpu-porting" in out
    assert "graphics-programming" in out


def test_run_unmapped_analysis_prefers_labeled_over_scored_in_directory(tmp_path, capsys):
    labeled_path = tmp_path / "labeled.json"
    scored_path = tmp_path / "scored.json"
    sample = {
        "id": "sample-1",
        "conversations": [{"from": "human", "value": "Inspect the same sample once."}],
        "labels": {"unmapped": [{"dimension": "task", "value": "gpu-porting"}]},
    }
    with open(labeled_path, "w", encoding="utf-8") as handle:
        json.dump([sample], handle)
    with open(scored_path, "w", encoding="utf-8") as handle:
        json.dump([sample], handle)

    summary = run_unmapped_analysis(tmp_path, top=5, examples=0)
    out = capsys.readouterr().out

    assert summary["source_kind"] == "directory"
    assert summary["total_occurrences"] == 1
    assert "1 occurrences, 1 unique tags" in out


def test_legacy_sample_discovery_prefers_jsonl_siblings(tmp_path):
    labeled_json = tmp_path / "labeled.json"
    labeled_jsonl = tmp_path / "labeled.jsonl"
    sample = {
        "id": "sample-1",
        "conversations": [{"from": "human", "value": "Inspect the streamed sample."}],
        "labels": {"unmapped": [{"dimension": "task", "value": "gpu-porting"}]},
    }
    with open(labeled_json, "w", encoding="utf-8") as handle:
        json.dump([sample], handle)
    labeled_jsonl.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    files = _discover_legacy_sample_files(tmp_path)

    assert len(files) == 1
    assert files[0].suffix == ".jsonl"
