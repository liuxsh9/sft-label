from __future__ import annotations

import csv
import json
from pathlib import Path

from sft_label.inline_pass1 import merge_pass1_results
from sft_label.inline_rows import build_row_sample_bundle, flatten_row_sample_bundles
from sft_label.tools.export_review import _discover_legacy_labeled_files, export_review


def _legacy_sample(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "world"},
        ],
        "labels": {
            "intent": "build",
            "difficulty": "intermediate",
            "language": ["python"],
            "domain": ["web-backend"],
            "task": ["feature-implementation"],
            "concept": ["algorithms"],
            "agentic": [],
            "constraint": [],
            "context": "snippet",
            "confidence": {"intent": 0.9, "difficulty": 0.8},
        },
        "labeling_monitor": {
            "llm_calls": 2,
            "elapsed_seconds": 0.5,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
        },
    }


def _legacy_sample_with_extension(sample_id: str) -> dict:
    sample = _legacy_sample(sample_id)
    sample["labels"]["label_extensions"] = {
        "ui_fine_labels": {
            "status": "success",
            "matched": True,
            "spec_version": "v1",
            "spec_hash": "sha256:ui-v1",
            "labels": {"component_type": ["form", "modal"], "visual_complexity": "high"},
            "confidence": {"component_type": 0.76},
            "unmapped": [],
        }
    }
    return sample


def _inline_labels(intent: str = "build") -> dict:
    return {
        "intent": intent,
        "language": ["python"],
        "domain": ["web-backend"],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "concept": ["algorithms"],
        "agentic": [],
        "constraint": [],
        "context": "snippet",
        "confidence": {"intent": 0.9, "language": 0.9, "difficulty": 0.9},
        "unmapped": [],
    }


def _inline_monitor() -> dict:
    return {
        "sample_id": "sample-1",
        "status": "success",
        "llm_calls": 2,
        "total_prompt_tokens": 12,
        "total_completion_tokens": 8,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": 0,
        "elapsed_seconds": 0.5,
    }


def _write_inline_rows(source_file: Path) -> None:
    bundle = build_row_sample_bundle(
        {
            "id": "conv-inline",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        },
        source_file,
        1,
    )
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])
    merged = merge_pass1_results(
        [bundle],
        samples,
        [_inline_labels("build"), _inline_labels("modify")],
        [_inline_monitor(), _inline_monitor()],
        sample_to_bundle,
        source_file=source_file,
    )
    source_file.parent.mkdir(parents=True, exist_ok=True)
    with open(source_file, "w", encoding="utf-8") as f:
        for row in merged.rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestExportReview:
    def test_legacy_discovery_prefers_jsonl_siblings(self, tmp_path):
        (tmp_path / "labeled.json").write_text(json.dumps([_legacy_sample("legacy-json")]), encoding="utf-8")
        (tmp_path / "labeled.jsonl").write_text(
            json.dumps(_legacy_sample("legacy-jsonl"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        files = _discover_legacy_labeled_files(tmp_path)

        assert len(files) == 1
        assert files[0].suffix == ".jsonl"

    def test_legacy_labeled_json_exports_csv(self, tmp_path):
        input_file = tmp_path / "labeled.json"
        input_file.write_text(json.dumps([_legacy_sample("legacy-1")]), encoding="utf-8")
        output_file = tmp_path / "review.csv"

        summary = export_review(str(input_file), str(output_file))

        assert summary["rows"] == 1
        with open(output_file, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["id"] == "legacy-1"
        assert rows[0]["intent"] == "build"
        assert rows[0]["source_file"].endswith("labeled.json")

    def test_inline_mirrored_jsonl_exports_turn_rows(self, tmp_path):
        run_root = tmp_path / "dataset_labeled_20260311_120000"
        source_file = run_root / "dataset" / "train.jsonl"
        output_file = tmp_path / "review.tsv"
        _write_inline_rows(source_file)

        summary = export_review(str(source_file), str(output_file))

        assert summary["rows"] == 2
        with open(output_file, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert rows[0]["source_file"] == str(source_file)
        assert rows[0]["data_id"]
        assert rows[0]["turn_index"] == "1"
        assert rows[1]["turn_index"] == "2"

    def test_export_review_includes_extension_spec_metadata_and_normalized_unmapped(self, tmp_path):
        sample = _legacy_sample("legacy-3")
        sample["labels"]["label_extensions"] = {
            "ui_fine_labels": {
                "status": "invalid",
                "matched": True,
                "spec_version": "v2",
                "spec_hash": "sha256:ui-v2",
                "labels": {"component_type": ["modal"]},
                "confidence": {"component_type": 0.61},
                "unmapped": [
                    {"dimension": "component_type", "value": "sheet"},
                    {"dimension": "component_type", "value": "sheet"},
                    {"dimension": "visual_complexity", "value": "very-high"},
                ],
            }
        }
        input_file = tmp_path / "labeled.json"
        input_file.write_text(json.dumps([sample]), encoding="utf-8")
        output_file = tmp_path / "review.csv"

        export_review(str(input_file), str(output_file), include_extensions=True)

        with open(output_file, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert row["ext_ui_fine_labels_spec_version"] == "v2"
        assert row["ext_ui_fine_labels_spec_hash"] == "sha256:ui-v2"
        assert row["ext_ui_fine_labels_unmapped"] == "component_type:sheet, visual_complexity:very-high"


    def test_export_review_can_include_extension_columns(self, tmp_path):
        input_file = tmp_path / "labeled.json"
        input_file.write_text(json.dumps([_legacy_sample_with_extension("legacy-2")]), encoding="utf-8")
        output_file = tmp_path / "review.csv"

        summary = export_review(str(input_file), str(output_file), include_extensions=True)

        assert summary["rows"] == 1
        with open(output_file, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert row["ext_ui_fine_labels_status"] == "success"
        assert row["ext_ui_fine_labels_matched"] == "True"
        assert row["ext_ui_fine_labels_labels_component_type"] == "form, modal"
        assert row["ext_ui_fine_labels_labels_visual_complexity"] == "high"
        assert row["ext_ui_fine_labels_confidence_component_type"] == "0.76"
