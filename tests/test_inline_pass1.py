from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig
from sft_label.inline_migration import InlineMigrationMatch
from sft_label.inline_pass1 import InlineBundlePlan, merge_pass1_results, prepare_inline_pass1_batch
from sft_label.inline_rows import RowSampleBundle, build_row_sample_bundle, flatten_row_sample_bundles
from sft_label.label_extensions_schema import ExtensionSpec, SchemaField


def _full_labels(intent: str = "build") -> dict:
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
        "confidence": {"intent": 0.9},
        "unmapped": [],
    }


def _monitor(status: str = "success", llm_calls: int = 2) -> dict:
    return {
        "sample_id": "sample-1",
        "status": status,
        "llm_calls": llm_calls,
        "total_prompt_tokens": 12,
        "total_completion_tokens": 8,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": 0,
        "elapsed_seconds": 0.5,
    }


def _extension_payload(label: str = "form") -> dict:
    return {
        "ui_fine_labels": {
            "status": "success",
            "matched": True,
            "spec_version": "v1",
            "spec_hash": "sha256:ui-v1",
            "labels": {"component_type": [label]},
            "confidence": {"component_type": 0.8},
            "unmapped": [],
            "monitor": {"status": "success", "llm_calls": 1},
        }
    }


def _extension_spec(spec_hash_suffix: str = "v1") -> ExtensionSpec:
    return ExtensionSpec(
        id="ui_fine_labels",
        spec_version="v1",
        display_name="UI Fine Labels",
        description=None,
        enabled=True,
        trigger={"domain_any_of": ("web-backend",)},
        prompt="Label UI",
        schema={"component_type": SchemaField(field_type="multi_enum", options=("form", "modal"))},
        output={"include_confidence": True, "allow_unmapped": True},
        dashboard={},
        source=f"/tmp/spec-{spec_hash_suffix}.yaml",
    )


def _incremental_tool_trajectory_samples(total_turns: int = 13) -> list[dict]:
    samples = []
    for i in range(total_turns):
        conversations = [{"from": "human", "value": "Diagnose the same flaky test failure."}]
        for j in range(i):
            conversations.append({
                "from": "tool",
                "value": f"pytest repeat run {j}: same flaky stack trace",
            })
        conversations.append({
            "from": "gpt",
            "value": "The root cause is the same timeout race; keep the same fix plan.",
        })
        samples.append({
            "id": f"conv_t{i + 1}",
            "metadata": {"source_id": "conv", "turn_index": i + 1, "total_turns": total_turns},
            "conversations": conversations,
        })
    return samples


def test_merge_pass1_results_preserves_sibling_fields_on_single_turn_row(tmp_path):
    row = {
        "meta_prompt": ["You are helpful"],
        "data": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "extra_info": {
            "keep_extra": 1,
            "unique_info": {"keep_unique": {"a": 1}},
        },
    }
    bundle = build_row_sample_bundle(row, tmp_path / "train.jsonl", 1)
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])

    result = merge_pass1_results(
        [bundle],
        samples,
        [_full_labels()],
        [_monitor()],
        sample_to_bundle,
        source_file=tmp_path / "train.jsonl",
    )

    merged_row = result.rows[0]
    unique_info = merged_row["extra_info"]["unique_info"]
    assert merged_row["extra_info"]["keep_extra"] == 1
    assert unique_info["keep_unique"] == {"a": 1}
    assert unique_info["data_id"].startswith("sha256:v1:")
    assert unique_info["data_label"]["conversation"]["turn_count"] == 1
    assert unique_info["data_label"]["turns"][0]["labels"]["intent"] == "build"
    assert unique_info["data_label"]["meta"]["pass1_at"] is not None


def test_merge_pass1_results_keeps_multiturn_turn_order_and_inheritance(tmp_path):
    row = {
        "id": "conv-1",
        "conversations": [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ],
    }
    bundle = build_row_sample_bundle(row, tmp_path / "multi.jsonl", 2)
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])
    inherited_labels = _full_labels("modify")
    inherited_labels["inherited"] = True
    inherited_labels["inherited_from"] = samples[0]["id"]

    result = merge_pass1_results(
        [bundle],
        samples,
        [_full_labels("build"), inherited_labels],
        [_monitor(), None],
        sample_to_bundle,
        source_file=tmp_path / "multi.jsonl",
    )

    turns = result.rows[0]["extra_info"]["unique_info"]["data_label"]["turns"]
    assert [turn["turn_index"] for turn in turns] == [1, 2]
    assert turns[1]["inherited"] is True
    assert turns[1]["inherited_from"] == samples[0]["id"]
    assert turns[1]["labeling_monitor"]["status"] == "inherited"
    assert "metadata" not in turns[0]
    assert "label_state" not in turns[0]
    conversation = result.rows[0]["extra_info"]["unique_info"]["data_label"]["conversation"]
    assert conversation["turn_count"] == 2
    assert "inherited_turn_count" not in conversation


def test_merge_pass1_results_refresh_replaces_old_scoring_state(tmp_path):
    row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        "extra_info": {
            "keep_extra": "ok",
            "unique_info": {
                "keep_unique": {"old": True},
                "data_label": {
                    "meta": {
                        "schema_version": "1",
                        "label_version": "legacy-v0",
                        "source_format": "pangu",
                        "mode": "refresh",
                        "created_at": "2026-03-11T09:00:00",
                        "updated_at": "2026-03-11T09:00:00",
                        "pass1_at": "2026-03-11T09:00:00",
                        "pass2_at": "2026-03-11T09:10:00",
                        "recomputed_at": None,
                        "migrated_at": None,
                        "migration_source": None,
                    },
                    "turns": [{
                        "turn_index": 1,
                        "labels": _full_labels("legacy"),
                        "value": {"value_score": 8.2},
                        "scoring_monitor": {"status": "success"},
                    }],
                    "conversation": {"conv_value": 8.2, "turn_count": 1},
                },
            },
        },
    }
    bundle = build_row_sample_bundle(row, tmp_path / "train.jsonl", 1)
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])

    result = merge_pass1_results(
        [bundle],
        samples,
        [_full_labels("fresh")],
        [_monitor()],
        sample_to_bundle,
        source_file=tmp_path / "train.jsonl",
        mode="refresh",
    )

    merged = result.rows[0]["extra_info"]["unique_info"]
    assert merged["keep_unique"] == {"old": True}
    turn = merged["data_label"]["turns"][0]
    assert turn["labels"]["intent"] == "fresh"
    assert "value" not in turn
    assert "scoring_monitor" not in turn
    assert "conv_value" not in merged["data_label"]["conversation"]


def test_merge_pass1_results_promotes_label_extensions_to_samples_and_turns(tmp_path):
    row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
    }
    bundle = build_row_sample_bundle(row, tmp_path / "train.jsonl", 1)
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])
    labels = _full_labels("fresh")
    labels["label_extensions"] = _extension_payload()

    result = merge_pass1_results(
        [bundle],
        samples,
        [labels],
        [_monitor()],
        sample_to_bundle,
        source_file=tmp_path / "train.jsonl",
    )

    sample_record = result.samples[0]
    assert sample_record["labels"]["intent"] == "fresh"
    assert "label_extensions" not in sample_record["labels"]
    assert sample_record["label_extensions"]["ui_fine_labels"]["labels"]["component_type"] == ["form"]

    data_label = result.rows[0]["extra_info"]["unique_info"]["data_label"]
    assert data_label["meta"]["extension_specs"]["ui_fine_labels"]["spec_hash"] == "sha256:ui-v1"
    assert data_label["turns"][0]["label_extensions"]["ui_fine_labels"]["labels"]["component_type"] == ["form"]


def test_merge_pass1_results_preserves_pass2_when_only_extensions_change(tmp_path):
    row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        "extra_info": {
            "unique_info": {
                "data_label": {
                    "meta": {
                        "schema_version": "1",
                        "label_version": "inline-v1",
                        "source_format": "pangu",
                        "mode": "refresh",
                        "created_at": "2026-03-11T09:00:00",
                        "updated_at": "2026-03-11T09:00:00",
                        "pass1_at": "2026-03-11T09:00:00",
                        "pass2_at": "2026-03-11T09:10:00",
                        "recomputed_at": None,
                        "migrated_at": None,
                        "migration_source": None,
                    },
                    "turns": [{
                        "turn_index": 1,
                        "labels": _full_labels("fresh"),
                        "label_extensions": _extension_payload("form"),
                        "value": {"value_score": 8.2},
                        "scoring_monitor": {"status": "success"},
                    }],
                    "conversation": {"conv_value": 8.2, "turn_count": 1},
                },
            },
        },
    }
    bundle = build_row_sample_bundle(row, tmp_path / "train.jsonl", 1)
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])
    samples[0].setdefault("metadata", {})["turn_index"] = 1
    new_labels = _full_labels("fresh")
    new_labels["label_extensions"] = _extension_payload("modal")

    result = merge_pass1_results(
        [bundle],
        samples,
        [new_labels],
        [_monitor()],
        sample_to_bundle,
        source_file=tmp_path / "train.jsonl",
        bundle_plans=[InlineBundlePlan(
            local_label_indices=[0],
            local_inherit_map={},
            preserved_labels=[],
            use_existing_data_label=True,
            pass1_changed=True,
            invalidate_pass2=False,
        )],
        updated_sample_indices={0},
    )

    turn = result.rows[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]
    assert turn["value"]["value_score"] == 8.2
    assert turn["scoring_monitor"]["status"] == "success"
    assert turn["label_extensions"]["ui_fine_labels"]["labels"]["component_type"] == ["modal"]


def test_prepare_inline_pass1_batch_copies_migration_provenance(tmp_path):
    target_row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
    }
    bundle = build_row_sample_bundle(target_row, tmp_path / "target.jsonl", 1)
    source_data_label = {
        "meta": {
            "schema_version": "1",
            "label_version": "inline-v1",
            "source_format": "pangu",
            "mode": "refresh",
            "created_at": "2026-03-11T08:00:00",
            "updated_at": "2026-03-11T08:00:00",
            "pass1_at": "2026-03-11T08:00:00",
            "pass2_at": "2026-03-11T08:10:00",
            "recomputed_at": None,
            "migrated_at": None,
            "migration_source": None,
        },
        "turns": [{
            "turn_index": 1,
            "labels": _full_labels("copied"),
            "value": {"value_score": 7.5},
        }],
        "conversation": {"conv_value": 7.5, "turn_count": 1},
    }
    migration_index = {
        bundle.data_id: InlineMigrationMatch(
            data_id=bundle.data_id,
            data_label=source_data_label,
            source_ref="old/train.jsonl:1",
        )
    }

    prepared = prepare_inline_pass1_batch(
        [bundle],
        mode="migrate",
        migration_index=migration_index,
    )

    migrated_meta = prepared.bundles[0].raw_row["extra_info"]["unique_info"]["data_label"]["meta"]
    assert prepared.stats["rows_migrated"] == 1
    assert prepared.bundle_plans[0].migrated is True
    assert prepared.labels[0]["intent"] == "copied"
    assert migrated_meta["migration_source"] == "old/train.jsonl:1"
    assert migrated_meta["migrated_at"] is not None


def test_prepare_inline_pass1_batch_reruns_extension_only_changes_without_invalidating_pass2(tmp_path):
    row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        "extra_info": {
            "unique_info": {
                "data_label": {
                    "meta": {
                        "schema_version": "1",
                        "label_version": "inline-v1",
                        "source_format": "pangu",
                        "mode": "refresh",
                        "created_at": "2026-03-11T09:00:00",
                        "updated_at": "2026-03-11T09:00:00",
                        "pass1_at": "2026-03-11T09:00:00",
                        "pass2_at": "2026-03-11T09:10:00",
                        "extension_specs": {
                            "ui_fine_labels": {
                                "spec_version": "v1",
                                "spec_hash": "sha256:old",
                            }
                        },
                    },
                    "turns": [{
                        "turn_index": 1,
                        "labels": _full_labels("build"),
                        "label_extensions": _extension_payload("form"),
                        "value": {"value_score": 8.2},
                    }],
                    "conversation": {"conv_value": 8.2, "turn_count": 1},
                }
            }
        },
    }
    bundle = build_row_sample_bundle(row, tmp_path / "train.jsonl", 1)

    prepared = prepare_inline_pass1_batch(
        [bundle],
        mode="incremental",
        extension_specs=[_extension_spec("new")],
    )

    assert prepared.label_indices == [0]
    assert prepared.bundle_plans[0].pass1_changed is True
    assert prepared.bundle_plans[0].invalidate_pass2 is False


def test_prepare_inline_pass1_batch_refresh_preserves_sparse_inheritance_for_tool_growth(tmp_path):
    bundle = RowSampleBundle(
        raw_row={"id": "conv"},
        source_path=tmp_path / "train.jsonl",
        row_number=1,
        data_id="conv",
        samples=_incremental_tool_trajectory_samples(),
    )
    config = PipelineConfig(
        prompt_mode="compact",
        planner_enabled=True,
        planner_metadata_only=False,
    )
    sparse_kwargs = dict(
        full_label_count=config.sparse_full_label_count,
        gap_multiplier=config.sparse_gap_multiplier,
        min_gap=config.sparse_min_gap,
        max_gap=config.sparse_max_gap,
        threshold=config.sparse_threshold,
        planner_enabled=config.planner_enabled,
        planner_metadata_only=config.planner_metadata_only,
        planner_policy=config.planner_policy,
        planner_boundary_threshold=config.planner_boundary_threshold,
        planner_min_segment_size=config.planner_min_segment_size,
        planner_max_anchor_gap=config.planner_max_anchor_gap,
        planner_fallback_boundary_ratio=config.planner_fallback_boundary_ratio,
    )

    prepared = prepare_inline_pass1_batch(
        [bundle],
        mode="refresh",
        sparse_kwargs=sparse_kwargs,
    )

    assert len(prepared.samples) == 13
    assert len(prepared.label_indices) < len(prepared.samples)
    assert prepared.inherit_map


@pytest.mark.asyncio
async def test_run_single_jsonl_writes_mirrored_inline_dataset(tmp_path):
    from sft_label.pipeline import run

    input_path = tmp_path / "train.jsonl"
    rows = [
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "id": "conv-2",
            "conversations": [
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
                {"from": "human", "value": "q3"},
                {"from": "gpt", "value": "a3"},
            ],
        },
    ]
    with open(input_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        turn_index = (sample.get("metadata") or {}).get("turn_index", 1)
        return sample_idx, _full_labels(f"turn-{turn_index}"), _monitor()

    output_dir = tmp_path / "inline-output"
    config = PipelineConfig(concurrency=1)

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(input_path),
            output=str(output_dir),
            config=config,
        )

    run_dir = Path(stats["run_dir"])
    dataset_path = run_dir / "train" / "train.jsonl"
    meta_dir = run_dir / "meta_label_data" / "files" / "train"

    assert dataset_path.exists()
    assert (meta_dir / "stats_labeling.json").exists()
    assert (meta_dir / "monitor.jsonl").exists()
    assert (meta_dir / "labeled.jsonl").exists()

    merged_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert len(merged_rows) == 2
    assert merged_rows[0]["extra_info"]["unique_info"]["data_label"]["conversation"]["turn_count"] == 1
    assert merged_rows[1]["extra_info"]["unique_info"]["data_label"]["conversation"]["turn_count"] == 2
    assert len(merged_rows[1]["extra_info"]["unique_info"]["data_label"]["turns"]) == 2


@pytest.mark.asyncio
async def test_run_single_jsonl_chunked_preserves_original_row_order(tmp_path):
    from sft_label.pipeline import run

    input_path = tmp_path / "train.jsonl"
    rows = [
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "slow-row"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "fast-row"},
                {"role": "assistant", "content": "a2"},
            ],
        },
    ]
    with open(input_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        user_text = next(
            turn["value"] for turn in sample["conversations"] if turn["from"] == "human"
        )
        if user_text == "slow-row":
            await asyncio.sleep(0.05)
        return sample_idx, _full_labels(user_text), _monitor()

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(input_path),
            output=str(tmp_path / "ordered-out"),
            config=PipelineConfig(concurrency=2, chunk_size=1, max_active_chunks=2),
        )

    dataset_path = Path(stats["run_dir"]) / "train" / "train.jsonl"
    merged_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert [row["data"][0]["content"] for row in merged_rows] == ["slow-row", "fast-row"]
    assert merged_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]["labels"]["intent"] == "slow-row"
    assert merged_rows[1]["extra_info"]["unique_info"]["data_label"]["turns"][0]["labels"]["intent"] == "fast-row"


@pytest.mark.asyncio
async def test_run_directory_jsonl_mirrors_source_tree(tmp_path):
    from sft_label.pipeline import run

    input_dir = tmp_path / "dataset"
    code_file = input_dir / "code" / "a.jsonl"
    multi_file = input_dir / "multi" / "b.jsonl"
    code_file.parent.mkdir(parents=True)
    multi_file.parent.mkdir(parents=True)

    code_file.write_text(
        json.dumps({
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    multi_file.write_text(
        json.dumps({
            "id": "conv-b",
            "conversations": [
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
                {"from": "human", "value": "q3"},
                {"from": "gpt", "value": "a3"},
            ],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        turn_index = (sample.get("metadata") or {}).get("turn_index", 1)
        return sample_idx, _full_labels(f"turn-{turn_index}"), _monitor()

    output_dir = tmp_path / "dir-output"
    config = PipelineConfig(concurrency=1)

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(input_dir),
            output=str(output_dir),
            config=config,
        )

    run_dir = Path(stats["run_dir"])
    assert (run_dir / "dataset" / "code" / "a.jsonl").exists()
    assert (run_dir / "dataset" / "multi" / "b.jsonl").exists()
    assert (run_dir / "meta_label_data" / "files" / "code" / "a" / "stats_labeling.json").exists()
    assert (run_dir / "meta_label_data" / "files" / "multi" / "b" / "stats_labeling.json").exists()
    assert (run_dir / "meta_label_data" / "checkpoint.json").exists()
    assert (run_dir / "meta_label_data" / "summary_stats_labeling.json").exists()


@pytest.mark.asyncio
async def test_incremental_run_preserves_line_count_and_skips_completed_rows(tmp_path):
    from sft_label.inline_labels import compute_data_id
    from sft_label.pipeline import run

    input_path = tmp_path / "inline.jsonl"
    row_done = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
    }
    row_done["extra_info"] = {
        "unique_info": {
            "data_id": compute_data_id(row_done),
            "data_label": {
                "meta": {
                    "schema_version": "1",
                    "label_version": "inline-v1",
                    "source_format": "pangu",
                    "mode": "refresh",
                    "created_at": "2026-03-11T08:00:00",
                    "updated_at": "2026-03-11T08:00:00",
                    "pass1_at": "2026-03-11T08:00:00",
                    "pass2_at": "2026-03-11T08:10:00",
                    "recomputed_at": None,
                    "migrated_at": None,
                    "migration_source": None,
                },
                "turns": [{
                    "turn_index": 1,
                    "labels": _full_labels("existing"),
                    "value": {"value_score": 6.8},
                }],
                "conversation": {"conv_value": 6.8, "turn_count": 1},
            },
        }
    }
    row_missing = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
    }
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in (row_done, row_missing)) + "\n",
        encoding="utf-8",
    )

    called = []

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        called.append(sample.get("id"))
        return sample_idx, _full_labels("fresh"), _monitor()

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(input_path),
            output=str(tmp_path / "out"),
            config=PipelineConfig(concurrency=1),
            mode="incremental",
        )

    dataset_path = Path(stats["run_dir"]) / "inline" / "inline.jsonl"
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert len(called) == 1
    rows = [json.loads(line) for line in lines]
    assert rows[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]["labels"]["intent"] == "existing"
    assert rows[0]["extra_info"]["unique_info"]["data_label"]["conversation"]["conv_value"] == 6.8
    assert "merged_labels" not in rows[0]["extra_info"]["unique_info"]["data_label"]["conversation"]
    assert rows[1]["extra_info"]["unique_info"]["data_label"]["turns"][0]["labels"]["intent"] == "fresh"


@pytest.mark.asyncio
async def test_migrate_run_copies_matching_rows_and_marks_provenance(tmp_path):
    from sft_label.pipeline import run

    source_root = tmp_path / "source_labeled_20260311_120000"
    source_dataset = source_root / "dataset"
    source_dataset.mkdir(parents=True)
    source_file = source_dataset / "train.jsonl"

    source_row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        "extra_info": {
            "unique_info": {
                "data_label": {
                    "meta": {
                        "schema_version": "1",
                        "label_version": "inline-v1",
                        "source_format": "pangu",
                        "mode": "refresh",
                        "created_at": "2026-03-11T08:00:00",
                        "updated_at": "2026-03-11T08:00:00",
                        "pass1_at": "2026-03-11T08:00:00",
                        "pass2_at": "2026-03-11T08:10:00",
                        "recomputed_at": None,
                        "migrated_at": None,
                        "migration_source": None,
                    },
                    "turns": [{
                        "turn_index": 1,
                        "labels": _full_labels("copied"),
                        "value": {"value_score": 7.1},
                    }],
                    "conversation": {"conv_value": 7.1, "turn_count": 1},
                },
            }
        },
    }
    source_bundle = build_row_sample_bundle(source_row, source_file, 1)
    source_file.write_text(json.dumps(source_row, ensure_ascii=False) + "\n", encoding="utf-8")
    (source_root / "meta_label_data").mkdir()

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_file = target_dir / "train.jsonl"
    target_rows = [
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        },
    ]
    target_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in target_rows) + "\n",
        encoding="utf-8",
    )

    called = []

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        called.append(sample.get("id"))
        return sample_idx, _full_labels("new"), _monitor()

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(target_dir),
            output=str(tmp_path / "migrated"),
            config=PipelineConfig(concurrency=1),
            mode="migrate",
            migrate_from=str(source_root),
        )

    run_dir = Path(stats["run_dir"])
    migrated_rows = [
        json.loads(line)
        for line in (run_dir / "target" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(called) == 1
    copied = migrated_rows[0]["extra_info"]["unique_info"]["data_label"]
    filled = migrated_rows[1]["extra_info"]["unique_info"]["data_label"]
    assert migrated_rows[0]["extra_info"]["unique_info"]["data_id"] == source_bundle.data_id
    assert copied["turns"][0]["labels"]["intent"] == "copied"
    assert copied["conversation"]["conv_value"] == 7.1
    assert "merged_labels" not in copied["conversation"]
    assert copied["meta"]["migration_source"] == "train.jsonl:1"
    assert filled["turns"][0]["labels"]["intent"] == "new"


@pytest.mark.asyncio
async def test_directory_run_bounds_inflight_submission_by_watermark(tmp_path):
    from sft_label.pipeline import run

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    input_file = input_dir / "many.json"
    payload = [
        {
            "id": f"sample-{i}",
            "conversations": [
                {"from": "human", "value": f"q{i}"},
                {"from": "gpt", "value": f"a{i}"},
            ],
        }
        for i in range(24)
    ]
    input_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    peak_inflight = 0
    inflight = 0
    real_ensure_future = asyncio.ensure_future

    def tracking_ensure_future(coro):
        nonlocal inflight, peak_inflight
        task = real_ensure_future(coro)
        inflight += 1
        peak_inflight = max(peak_inflight, inflight)

        def _mark_done(_task):
            nonlocal inflight
            inflight -= 1

        task.add_done_callback(_mark_done)
        return task

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        await asyncio.sleep(0.01)
        return sample_idx, _full_labels("bounded"), _monitor()

    config = PipelineConfig(
        concurrency=2,
        dir_pipeline_watermark=1.0,
        dir_pipeline_max_files=1,
        enable_adaptive_runtime=False,
    )

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one), patch(
        "sft_label.pipeline.asyncio.ensure_future",
        new=tracking_ensure_future,
    ):
        await run(
            input_path=str(input_dir),
            output=str(tmp_path / "out"),
            config=config,
        )

    assert peak_inflight <= 2


@pytest.mark.asyncio
async def test_directory_large_jsonl_reuses_chunked_path(tmp_path):
    from sft_label.pipeline import run

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    input_file = input_dir / "train.jsonl"
    rows = [
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
        }
        for i in range(3)
    ]
    input_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("directory mode should delegate large JSONL to chunked pipeline")

    async def fake_chunked(*args, **kwargs):
        return {
            "total_samples": 3,
            "distribution_total_samples": 3,
            "success": 3,
            "failed": 0,
            "success_rate": 1.0,
            "total_llm_calls": 6,
            "avg_calls_per_sample": 2.0,
            "total_prompt_tokens": 1,
            "total_completion_tokens": 1,
            "total_tokens": 2,
            "arbitrated_count": 0,
            "arbitrated_rate": 0.0,
            "validation_issue_count": 0,
            "consistency_warning_count": 0,
            "unmapped_tags": {},
            "unmapped_unique_count": 0,
            "canonicalization_counts": {},
            "canonicalization_total_count": 0,
            "canonicalization_unique_count": 0,
            "confidence_stats": {},
            "low_confidence_frequency": {},
            "tag_distributions": {},
            "combo_distributions": {},
            "cross_matrix": {},
            "total_elapsed_seconds": 0.1,
            "input_file": str(input_file),
            "mode": "refresh",
            "chunked": True,
            "chunk_size": 1,
            "chunks_processed": 3,
        }

    with patch("sft_label.pipeline.label_one", side_effect=fail_if_called), patch(
        "sft_label.pipeline._run_one_file_chunked",
        side_effect=fake_chunked,
    ) as mocked_chunked:
        await run(
            input_path=str(input_dir),
            output=str(tmp_path / "out"),
            config=PipelineConfig(
                concurrency=1,
                chunk_size=1,
                dir_pipeline_max_files=1,
                enable_adaptive_runtime=False,
            ),
        )

    assert mocked_chunked.await_count == 1


@pytest.mark.asyncio
async def test_directory_large_jsonl_chunked_files_are_admitted_with_rolling_overlap(tmp_path):
    from sft_label.pipeline import run

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    for file_idx in range(3):
        input_file = input_dir / f"train_{file_idx}.jsonl"
        rows = [
            {
                "meta_prompt": ["system"],
                "data": [
                    {"role": "user", "content": f"q{file_idx}-{row_idx}"},
                    {"role": "assistant", "content": f"a{file_idx}-{row_idx}"},
                ],
            }
            for row_idx in range(2)
        ]
        input_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    started: list[str] = []
    finished: list[str] = []
    events: list[tuple[str, str]] = []
    active = 0
    peak_active = 0

    async def fake_chunked(input_path, *args, **kwargs):
        nonlocal active, peak_active
        name = Path(input_path).name
        started.append(name)
        events.append(("start", name))
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.05 if name == "train_0.jsonl" else 0.01)
            return {
                "total_samples": 2,
                "distribution_total_samples": 2,
                "success": 2,
                "failed": 0,
                "success_rate": 1.0,
                "total_llm_calls": 4,
                "avg_calls_per_sample": 2.0,
                "total_prompt_tokens": 1,
                "total_completion_tokens": 1,
                "total_tokens": 2,
                "arbitrated_count": 0,
                "arbitrated_rate": 0.0,
                "validation_issue_count": 0,
                "consistency_warning_count": 0,
                "unmapped_tags": {},
                "unmapped_unique_count": 0,
                "canonicalization_counts": {},
                "canonicalization_total_count": 0,
                "canonicalization_unique_count": 0,
                "confidence_stats": {},
                "low_confidence_frequency": {},
                "tag_distributions": {},
                "combo_distributions": {},
                "cross_matrix": {},
                "total_elapsed_seconds": 0.1,
                "input_file": str(input_path),
                "mode": "refresh",
                "chunked": True,
                "chunk_size": 1,
                "chunks_processed": 2,
            }
        finally:
            finished.append(name)
            events.append(("finish", name))
            active -= 1

    with patch("sft_label.pipeline._run_one_file_chunked", side_effect=fake_chunked):
        await run(
            input_path=str(input_dir),
            output=str(tmp_path / "out"),
            config=PipelineConfig(
                concurrency=2,
                chunk_size=1,
                dir_pipeline_max_files=2,
                dir_pipeline_watermark=1.0,
                enable_adaptive_runtime=False,
            ),
        )

    assert peak_active == 2
    assert started[:2] == ["train_0.jsonl", "train_1.jsonl"]
    assert "train_2.jsonl" in started
    assert events.index(("start", "train_2.jsonl")) < events.index(("finish", "train_0.jsonl"))


@pytest.mark.asyncio
async def test_directory_large_jsonl_chunked_smoke_preserves_full_row_count(tmp_path):
    from sft_label.pipeline import run

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    input_file = input_dir / "train.jsonl"
    row_count = 512
    rows = [
        {
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
        }
        for i in range(row_count)
    ]
    input_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        return sample_idx, _full_labels(f"intent-{sample_idx}"), _monitor()

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(input_dir),
            output=str(tmp_path / "out"),
            config=PipelineConfig(
                concurrency=4,
                chunk_size=64,
                max_active_chunks=2,
                dir_pipeline_max_files=1,
                enable_adaptive_runtime=False,
            ),
        )

    run_dir = Path(stats["run_dir"])
    dataset_path = run_dir / "dataset" / "train.jsonl"
    merged_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(merged_rows) == row_count
    assert merged_rows[0]["data"][0]["content"] == "q0"
    assert merged_rows[-1]["data"][0]["content"] == f"q{row_count - 1}"
    assert stats["total_samples"] == row_count


@pytest.mark.asyncio
async def test_pass1_writes_conversation_stats_artifact_for_inline_jsonl(tmp_path):
    from sft_label.pipeline import run

    input_path = tmp_path / "train.jsonl"
    rows = [
        {
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        turn_index = (sample.get("metadata") or {}).get("turn_index", 1)
        return sample_idx, _full_labels(f"turn-{turn_index}"), _monitor()

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one):
        stats = await run(
            input_path=str(input_path),
            output=str(tmp_path / "out"),
            config=PipelineConfig(concurrency=1, enable_adaptive_runtime=False),
        )

    artifact_dir = Path(stats["run_dir"]) / "meta_label_data" / "files" / "train"
    conversation_stats = json.loads(
        (artifact_dir / "conversation_stats_labeling.json").read_text(encoding="utf-8")
    )

    assert conversation_stats["mode_id"] == "conversation"
    assert conversation_stats["total"] == 1
    assert conversation_stats["distributions"]["intent"] == {"turn-2": 1}


@pytest.mark.asyncio
async def test_pass1_chunked_conversation_stats_artifact_deduplicates_same_source_across_rows(tmp_path):
    from sft_label.pipeline import _write_pass1_conversation_stats_from_path

    artifact_dir = tmp_path / "meta"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = artifact_dir / "labeled.jsonl"
    labeled_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {
                    "id": "conv-1-turn-1",
                    "metadata": {"source_id": "conv-1", "source_file": "train.jsonl", "turn_index": 1, "total_turns": 2},
                    "conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}],
                    "labels": _full_labels("turn-1"),
                },
                {
                    "id": "conv-1-turn-2",
                    "metadata": {"source_id": "conv-1", "source_file": "train.jsonl", "turn_index": 2, "total_turns": 2},
                    "conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}],
                    "labels": _full_labels("turn-2"),
                },
            ]
        ) + "\n",
        encoding="utf-8",
    )
    stats = {
        "input_file": "train.jsonl",
        "total_samples": 2,
        "success_rate": 1.0,
        "total_tokens": 0,
        "arbitrated_rate": 0.0,
        "tag_distributions": {"intent": {"turn-1": 1, "turn-2": 1}},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
    }

    _write_pass1_conversation_stats_from_path(artifact_dir, labeled_path, stats)

    conversation_stats = json.loads(
        (artifact_dir / "conversation_stats_labeling.json").read_text(encoding="utf-8")
    )

    assert conversation_stats["total"] == 1
    assert conversation_stats["distributions"]["intent"] == {"turn-2": 1}


@pytest.mark.asyncio
async def test_pass1_conversation_stats_sidecar_failure_does_not_abort_run(tmp_path):
    from sft_label.pipeline import run

    input_path = tmp_path / "train.jsonl"
    rows = [
        {
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
            ],
        }
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    async def mock_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                             config=None, rate_limiter=None):
        return sample_idx, _full_labels("turn-1"), _monitor()

    with patch("sft_label.pipeline.label_one", side_effect=mock_label_one), \
            patch("sft_label.pipeline._compute_pass1_conversation_mode", side_effect=RuntimeError("boom")):
        stats = await run(
            input_path=str(input_path),
            output=str(tmp_path / "out"),
            config=PipelineConfig(concurrency=1, enable_adaptive_runtime=False),
        )

    artifact_dir = Path(stats["run_dir"]) / "meta_label_data" / "files" / "train"
    assert (artifact_dir / "stats_labeling.json").exists()


def test_flush_file_and_chunk_collectors_release_heavy_references(tmp_path):
    from sft_label.inline_rows import build_row_sample_bundle
    from sft_label.pipeline import (
        ChunkCollector,
        FileCollector,
        StatsAccumulator,
        _flush_chunk,
        flush_file_output,
    )

    row = {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
    }
    source_file = tmp_path / "train.jsonl"
    bundle = build_row_sample_bundle(row, source_file, 1)
    samples, sample_to_bundle = flatten_row_sample_bundles([bundle])

    collector = FileCollector(
        file_idx=0,
        abs_path=source_file,
        rel_path=Path("train.jsonl"),
        output_dir=tmp_path / "meta" / "train",
        prefix="train",
        total=len(samples),
        samples=samples,
        row_bundles=[bundle],
        sample_to_bundle=sample_to_bundle,
        dataset_output_path=tmp_path / "dataset" / "train.jsonl",
        run_failure_log_path=tmp_path / "failures.jsonl",
        config=PipelineConfig(enable_adaptive_runtime=False),
        inline_output=True,
        label_count=1,
        inherit_map={},
        updated_sample_indices={0},
        bundle_plans=[],
        plan_stats={},
    )
    collector.labels = [_full_labels()]
    collector.monitors = [_monitor()]

    flush_file_output(collector, tmp_path, checkpoint_path=None, pprint=lambda _msg: None)

    assert collector.completed is True
    assert collector.samples is None
    assert collector.labels is None
    assert collector.monitors is None
    assert collector.row_bundles is None
    assert collector.sample_to_bundle is None
    assert collector.inherit_map == {}
    assert collector.updated_sample_indices == set()
    assert collector.bundle_plans is None
    assert collector.plan_stats == {}

    chunk = ChunkCollector(
        chunk_idx=0,
        samples=samples,
        row_bundles=[bundle],
        sample_to_bundle=sample_to_bundle,
        label_count=1,
        inherit_map={},
        updated_sample_indices={0},
        bundle_plans=[],
        plan_stats={},
    )
    chunk.labels = [_full_labels()]
    chunk.monitors = [_monitor()]
    stats_acc = StatsAccumulator()

    out_rows = open(tmp_path / "chunk_rows.jsonl", "w", encoding="utf-8")
    out_labeled = open(tmp_path / "chunk_labeled.jsonl", "w", encoding="utf-8")
    out_monitor = open(tmp_path / "chunk_monitor.jsonl", "w", encoding="utf-8")
    out_failed = open(tmp_path / "chunk_failed.jsonl", "w", encoding="utf-8")
    try:
        _flush_chunk(
            chunk,
            out_rows,
            out_labeled,
            out_monitor,
            out_failed,
            stats_acc,
            source_file,
            pprint=lambda _msg: None,
            run_failure_log_path=tmp_path / "failures.jsonl",
        )
    finally:
        out_rows.close()
        out_labeled.close()
        out_monitor.close()
        out_failed.close()

    assert chunk.completed is True
    assert chunk.samples is None
    assert chunk.labels is None
    assert chunk.monitors is None
    assert chunk.row_bundles is None
    assert chunk.sample_to_bundle is None
    assert chunk.inherit_map == {}
    assert chunk.updated_sample_indices == set()
    assert chunk.bundle_plans is None
    assert chunk.plan_stats == {}


def test_estimate_directory_workload_streams_inline_jsonl_planning(monkeypatch, tmp_path):
    from sft_label.pipeline import estimate_directory_workload

    events = []
    real_prepare = prepare_inline_pass1_batch
    source_path = tmp_path / "large.jsonl"

    bundles = [
        RowSampleBundle(
            raw_row={"id": f"row-{idx}", "conversations": []},
            source_path=source_path,
            row_number=idx + 1,
            data_id=f"row-{idx}",
            samples=[{"id": f"sample-{idx}", "conversations": []}],
        )
        for idx in range(3)
    ]

    def fake_iter(_input_path, limit=0):
        for idx, bundle in enumerate(bundles):
            events.append(("yield", idx))
            yield bundle

    def tracking_prepare(batch_bundles, **kwargs):
        events.append(("prepare", len(batch_bundles)))
        return real_prepare(batch_bundles, **kwargs)

    monkeypatch.setattr("sft_label.pipeline.iter_row_sample_bundles_from_jsonl", fake_iter)
    monkeypatch.setattr("sft_label.pipeline.prepare_inline_pass1_batch", tracking_prepare)

    estimate = estimate_directory_workload(
        [(source_path, Path("large.jsonl"))],
        config=PipelineConfig(chunk_size=1, enable_adaptive_runtime=False),
    )

    assert estimate.files_planned == 1
    assert estimate.total_samples == 3
    assert estimate.total_labeled_samples == 3
    assert events[:4] == [("yield", 0), ("prepare", 1), ("yield", 1), ("prepare", 1)]
