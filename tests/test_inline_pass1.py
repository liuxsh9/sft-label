from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig
from sft_label.inline_migration import InlineMigrationMatch
from sft_label.inline_pass1 import merge_pass1_results, prepare_inline_pass1_batch
from sft_label.inline_rows import build_row_sample_bundle, flatten_row_sample_bundles


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
