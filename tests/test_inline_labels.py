"""Tests for inline row labeling helpers and mirrored run layout."""

from __future__ import annotations


import pytest

from sft_label.inline_labels import (
    DATA_LABEL_SCHEMA_VERSION,
    build_data_label,
    compact_conversation_record,
    compute_data_id,
    ensure_data_id,
    ensure_data_label,
    get_data_id,
    get_data_label,
    get_unique_info,
    mark_migrated,
    mark_stage_timestamp,
    replace_data_label,
    upsert_turn_record,
)
from sft_label.run_layout import InlineRunLayout, resolve_run_root


def _pangu_row():
    return {
        "meta_prompt": ["You are helpful"],
        "data": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    }


class TestInlineDataId:
    def test_compute_data_id_is_stable_across_unrelated_metadata(self):
        row_a = _pangu_row()
        row_a["extra_info"] = {"foo": "bar"}

        row_b = _pangu_row()
        row_b["extra_info"] = {"foo": "baz", "unique_info": {"existing": 1}}
        row_b["id"] = "different-id"

        assert compute_data_id(row_a) == compute_data_id(row_b)

    def test_compute_data_id_changes_when_training_content_changes(self):
        row_a = _pangu_row()
        row_b = _pangu_row()
        row_b["data"][0]["content"] = "different"

        assert compute_data_id(row_a) != compute_data_id(row_b)

    def test_ensure_data_id_writes_under_unique_info(self):
        row = _pangu_row()
        row["extra_info"] = {"notes": "keep me"}

        data_id = ensure_data_id(row)

        assert data_id == get_data_id(row)
        assert row["extra_info"]["notes"] == "keep me"
        assert row["extra_info"]["unique_info"]["data_id"] == data_id


class TestInlineDataLabel:
    def test_replace_data_label_preserves_sibling_fields(self):
        row = _pangu_row()
        row["extra_info"] = {
            "keep_extra": 1,
            "unique_info": {
                "keep_unique": {"a": 1},
                "data_label": {"old": True},
            },
        }

        replace_data_label(
            row,
            label_version="v-test",
            source_format="pangu",
            mode="refresh",
            timestamp="2026-03-11T10:00:00",
        )

        assert row["extra_info"]["keep_extra"] == 1
        assert row["extra_info"]["unique_info"]["keep_unique"] == {"a": 1}
        data_label = row["extra_info"]["unique_info"]["data_label"]
        assert data_label["meta"]["label_version"] == "v-test"
        assert data_label["meta"]["mode"] == "refresh"

    def test_ensure_data_label_creates_default_schema(self):
        row = _pangu_row()

        data_label = ensure_data_label(
            row,
            label_version="inline-test",
            source_format="pangu",
            mode="incremental",
            timestamp="2026-03-11T11:00:00",
        )

        assert get_data_label(row) == data_label
        assert data_label["meta"]["schema_version"] == DATA_LABEL_SCHEMA_VERSION
        assert data_label["meta"]["label_version"] == "inline-test"
        assert data_label["meta"]["source_format"] == "pangu"
        assert data_label["turns"] == []
        assert data_label["conversation"] == {}

    def test_upsert_turn_record_keeps_turns_sorted_and_replaces_duplicates(self):
        data_label = build_data_label(timestamp="2026-03-11T12:00:00")

        upsert_turn_record(data_label, {"turn_index": 2, "labels": {"intent": "build"}})
        upsert_turn_record(data_label, {"turn_index": 1, "labels": {"intent": "debug"}})
        upsert_turn_record(data_label, {"turn_index": 2, "labels": {"intent": "modify"}})

        assert [turn["turn_index"] for turn in data_label["turns"]] == [1, 2]
        assert data_label["turns"][1]["labels"]["intent"] == "modify"

    def test_mark_helpers_update_meta_fields(self):
        data_label = build_data_label(timestamp="2026-03-11T12:30:00")

        mark_stage_timestamp(data_label, "pass1", timestamp="2026-03-11T12:31:00")
        mark_migrated(
            data_label,
            migration_source="archive/run-a",
            timestamp="2026-03-11T12:32:00",
        )

        assert data_label["meta"]["pass1_at"] == "2026-03-11T12:31:00"
        assert data_label["meta"]["migrated_at"] == "2026-03-11T12:32:00"
        assert data_label["meta"]["migration_source"] == "archive/run-a"
        assert data_label["meta"]["updated_at"] == "2026-03-11T12:32:00"

    def test_get_unique_info_rejects_non_dict_parents(self):
        row = _pangu_row()
        row["extra_info"] = "bad"

        with pytest.raises(TypeError):
            get_unique_info(row, create=True)

    def test_compact_conversation_record_keeps_inline_diagnostics_but_not_heavy_fields(self):
        compact = compact_conversation_record(
            {
                "conversation_id": "conv-1",
                "conversation_key": "train.jsonl::conv-1",
                "source_file": "train.jsonl",
                "turn_count": 4,
                "conv_value": 7.2,
                "conv_selection": 6.8,
                "peak_complexity": 8,
                "conv_rarity": 6.1,
                "observed_turn_ratio": 0.5,
                "inherited_turn_ratio": 0.5,
                "rarity_confidence": 0.7,
                "compression_gap": 2.1,
                "late_turn_gain": 1.4,
                "tool_turn_ratio": 0.75,
                "unique_tool_count": 3,
                "unique_file_count": 5,
                "thinking_mode": "fast",
                "thinking_mode_summary": "all_fast",
                "thinking_mode_counts": {"fast": 4, "slow": 0, "unknown": 0},
                "thinking_mode_ratio": {"fast": 1.0, "slow": 0.0, "unknown": 0.0},
                "detail": {
                    "top_k_mean": 8.4,
                    "bottom_k_mean": 5.1,
                    "turn_value_std": 1.23,
                    "unique_tools": ["bash", "apply_patch"],
                    "test_related_turn_count": 2,
                    "edit_related_turn_count": 3,
                    "bash_execution_turn_count": 2,
                },
                "merged_labels": {"task": ["bug-fixing"]},
                "slices": [{"id": "x"}],
            }
        )

        assert compact["conversation_key"] == "train.jsonl::conv-1"
        assert compact["compression_gap"] == 2.1
        assert compact["tool_turn_ratio"] == 0.75
        assert compact["unique_tool_count"] == 3
        assert compact["thinking_mode_summary"] == "all_fast"
        assert compact["thinking_mode_counts"]["fast"] == 4
        assert compact["thinking_mode_ratio"]["fast"] == 1.0
        assert compact["detail"]["top_k_mean"] == 8.4
        assert compact["detail"]["unique_tools"] == ["bash", "apply_patch"]
        assert "merged_labels" not in compact
        assert "slices" not in compact


class TestInlineRunLayout:
    def test_resolve_run_root_uses_inline_naming(self, tmp_path):
        input_dir = tmp_path / "llm_sft_data_v1"
        input_dir.mkdir()

        run_root = resolve_run_root(input_dir, timestamp="20260311_120000")

        assert run_root == tmp_path / "llm_sft_data_v1_labeled_20260311_120000"

    def test_resolve_run_root_can_override_base_dir(self, tmp_path):
        input_dir = tmp_path / "old_run" / "llm_sft_data_v1"
        input_dir.mkdir(parents=True)

        run_root = resolve_run_root(
            input_dir,
            timestamp="20260311_120000",
            base_dir=tmp_path,
        )

        assert run_root == tmp_path / "llm_sft_data_v1_labeled_20260311_120000"

    def test_directory_layout_paths(self, tmp_path):
        input_dir = tmp_path / "dataset"
        source_file = input_dir / "code" / "sample.jsonl"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("")

        layout = InlineRunLayout.from_paths(
            input_dir,
            tmp_path / "dataset_labeled_20260311_120000",
        )

        assert layout.dataset_root == layout.run_root / "dataset"
        assert layout.mirrored_dataset_path(source_file) == (
            layout.run_root / "dataset" / "code" / "sample.jsonl"
        )
        assert layout.file_artifact_path(source_file, "stats_labeling.json") == (
            layout.run_root
            / "meta_label_data"
            / "files"
            / "code"
            / "sample"
            / "stats_labeling.json"
        )
        assert layout.cache_path("flattened", "sample.jsonl") == (
            layout.run_root / "meta_label_data" / "cache" / "flattened" / "sample.jsonl"
        )
        assert layout.dashboard_path("dashboard_labeling.html") == (
            layout.run_root / "meta_label_data" / "dashboards" / "dashboard_labeling.html"
        )

    def test_single_file_layout_paths(self, tmp_path):
        input_file = tmp_path / "train.jsonl"
        input_file.write_text("")

        layout = InlineRunLayout.from_paths(
            input_file,
            tmp_path / "train_labeled_20260311_120000",
        )

        assert layout.dataset_root == layout.run_root / "train"
        assert layout.mirrored_dataset_path(input_file) == (
            layout.run_root / "train" / "train.jsonl"
        )
        assert layout.file_artifact_dir(input_file) == (
            layout.run_root / "meta_label_data" / "files" / "train"
        )

    def test_single_file_layout_rejects_unexpected_source_path(self, tmp_path):
        input_file = tmp_path / "train.jsonl"
        other_file = tmp_path / "other.jsonl"
        input_file.write_text("")
        other_file.write_text("")
        layout = InlineRunLayout.from_paths(
            input_file,
            tmp_path / "train_labeled_20260311_120000",
        )

        with pytest.raises(ValueError):
            layout.relative_source_path(other_file)
