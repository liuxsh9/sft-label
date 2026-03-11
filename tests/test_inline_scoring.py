from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig
from sft_label.inline_labels import build_turn_id
from sft_label.inline_pass1 import merge_pass1_results
from sft_label.inline_rows import build_row_sample_bundle, flatten_row_sample_bundles
from sft_label.inline_scoring import infer_inline_scoring_target, load_inline_scoring_file


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
        "confidence": {"intent": 0.9, "language": 0.9, "difficulty": 0.9},
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


def _score_value(sample_id: str, rarity_result: dict | None = None) -> dict:
    return {
        "complexity": {"instruction": 6, "analytical_depth": 6, "implementation": 6, "overall": 6},
        "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
        "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
        "rarity": rarity_result or {"score": 5.0},
        "flags": [],
        "thinking_mode": "fast",
        "value_score": 6.5 if sample_id.endswith("2") else 6.0,
        "confidence": 0.85,
    }


def _inline_rows_for_file(source_file: Path, rows: list[dict], labels_per_row: list[list[dict]]) -> list[dict]:
    merged_rows = []
    for row_number, (row, row_labels) in enumerate(zip(rows, labels_per_row), start=1):
        bundle = build_row_sample_bundle(row, source_file, row_number)
        samples, sample_to_bundle = flatten_row_sample_bundles([bundle])
        monitors = [_monitor() if labels else None for labels in row_labels]
        merged = merge_pass1_results(
            [bundle],
            samples,
            row_labels,
            monitors,
            sample_to_bundle,
            source_file=source_file,
        )
        merged_rows.extend(merged.rows)
    return merged_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_inline_scoring_file_rehydrates_embedded_turn_labels(tmp_path):
    source_file = tmp_path / "train.jsonl"
    rows = _inline_rows_for_file(
        source_file,
        [{
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }],
        [[_full_labels("build"), _full_labels("modify")]],
    )
    _write_jsonl(source_file, rows)

    bundles, samples, sample_to_bundle = load_inline_scoring_file(source_file)

    assert len(bundles) == 1
    assert len(samples) == 2
    assert sample_to_bundle == [0, 0]
    assert samples[0]["labels"]["intent"] == "build"
    assert samples[1]["labels"]["intent"] == "modify"
    assert samples[1]["metadata"]["turn_index"] == 2
    assert samples[0]["id"] == build_turn_id(bundles[0].data_id, 1)
    assert samples[1]["id"] == build_turn_id(bundles[0].data_id, 2)


@pytest.mark.asyncio
async def test_run_scoring_inline_file_updates_mirrored_rows(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "train_labeled_20260311_120000"
    source_file = run_root / "train" / "train.jsonl"
    rows = _inline_rows_for_file(
        source_file,
        [{
            "id": "conv-2",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }],
        [[_full_labels("build"), _full_labels("modify")]],
    )
    _write_jsonl(source_file, rows)

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        return _score_value(sample["id"], rarity_result), {
            "sample_id": sample["id"],
            "status": "success",
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "attempts": 1,
            "validation_issues": [],
        }

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        stats = await run_scoring(str(source_file), config=PipelineConfig(scoring_concurrency=1))

    assert stats["total_scored"] == 2
    artifact_dir = run_root / "meta_label_data" / "files" / "train"
    assert (artifact_dir / "labeled.jsonl").exists()
    assert (artifact_dir / "scored.jsonl").exists()
    assert (artifact_dir / "monitor_value.jsonl").exists()
    assert (artifact_dir / "stats_scoring.json").exists()

    updated_rows = [json.loads(line) for line in source_file.read_text(encoding="utf-8").splitlines()]
    turns = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"]
    assert turns[0]["value"]["value_score"] is not None
    assert turns[0]["value"]["selection_score"] is not None
    assert turns[1]["value"]["selection_score"] is not None
    assert "metadata" not in turns[0]
    conversation = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["conversation"]
    assert conversation["conv_value"] is not None
    assert conversation["conv_selection"] is not None
    assert "merged_labels" not in conversation
    assert "detail" not in conversation
    assert "slices" not in conversation


@pytest.mark.asyncio
async def test_run_scoring_inline_run_dir_writes_meta_summary(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    file_a = run_root / "dataset" / "code" / "a.jsonl"
    file_b = run_root / "dataset" / "multi" / "b.jsonl"

    rows_a = _inline_rows_for_file(
        file_a,
        [{
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "qa"},
                {"role": "assistant", "content": "aa"},
            ],
        }],
        [[_full_labels("build")]],
    )
    rows_b = _inline_rows_for_file(
        file_b,
        [{
            "id": "conv-b",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }],
        [[_full_labels("debug"), _full_labels("modify")]],
    )
    _write_jsonl(file_a, rows_a)
    _write_jsonl(file_b, rows_b)

    target = infer_inline_scoring_target(run_root)
    assert target is not None

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        return _score_value(sample["id"], rarity_result), {
            "sample_id": sample["id"],
            "status": "success",
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "attempts": 1,
            "validation_issues": [],
        }

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        summary = await run_scoring(str(run_root), config=PipelineConfig(scoring_concurrency=1))

    assert summary["files_processed"] == 2
    assert (run_root / "meta_label_data" / "summary_stats_scoring.json").exists()
    assert (run_root / "meta_label_data" / "conversation_scores.json").exists()
    assert len(list(run_root.glob("dashboard_scoring*.html"))) >= 1

    updated_b = [json.loads(line) for line in file_b.read_text(encoding="utf-8").splitlines()]
    turns = updated_b[0]["extra_info"]["unique_info"]["data_label"]["turns"]
    assert turns[0]["value"]["selection_score"] is not None
    assert turns[1]["value"]["selection_score"] is not None


@pytest.mark.asyncio
async def test_run_scoring_inline_run_dir_resume_skips_embedded_scored_samples(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    file_a = run_root / "dataset" / "code" / "a.jsonl"
    rows_a = _inline_rows_for_file(
        file_a,
        [{
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "qa"},
                {"role": "assistant", "content": "aa"},
            ],
        }],
        [[_full_labels("build")]],
    )
    turns = rows_a[0]["extra_info"]["unique_info"]["data_label"]["turns"]
    turns[0]["value"] = _score_value("resume-sample")
    rows_a[0]["extra_info"]["unique_info"]["data_label"]["conversation"] = {
        "turn_count": 1,
        "conv_value": turns[0]["value"]["value_score"],
        "conv_selection": 6.0,
    }
    _write_jsonl(file_a, rows_a)

    calls = {"count": 0}

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        calls["count"] += 1
        return _score_value(sample["id"], rarity_result), {
            "sample_id": sample["id"],
            "status": "success",
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "attempts": 1,
            "validation_issues": [],
        }

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(scoring_concurrency=1, chunk_size=1, max_active_chunks=1),
            resume=True,
        )

    assert summary["files_processed"] == 1
    assert calls["count"] == 0
    updated_rows = [json.loads(line) for line in file_a.read_text(encoding="utf-8").splitlines()]
    updated_value = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]["value"]
    assert updated_value["value_score"] == rows_a[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]["value"]["value_score"]


@pytest.mark.asyncio
async def test_run_scoring_inline_resume_matches_legacy_sample_id_cache(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    file_a = run_root / "dataset" / "code" / "a.jsonl"
    rows_a = _inline_rows_for_file(
        file_a,
        [{
            "meta_prompt": ["system"],
            "data": [
                {"role": "user", "content": "qa"},
                {"role": "assistant", "content": "aa"},
            ],
        }],
        [[_full_labels("build")]],
    )
    turn = rows_a[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]
    turn["sample_id"] = "legacy-inline-1"
    turn.pop("value", None)
    _write_jsonl(file_a, rows_a)

    artifact_dir = run_root / "meta_label_data" / "files" / "code" / "a"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cached_value = _score_value("legacy-inline-1")
    (artifact_dir / "scored.jsonl").write_text(
        json.dumps({"id": "legacy-inline-1", "value": cached_value}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    calls = {"count": 0}

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        calls["count"] += 1
        return _score_value(sample["id"], rarity_result), {
            "sample_id": sample["id"],
            "status": "success",
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "attempts": 1,
            "validation_issues": [],
        }

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(scoring_concurrency=1, chunk_size=1, max_active_chunks=1),
            resume=True,
        )

    assert summary["files_processed"] == 1
    assert calls["count"] == 0
    updated_rows = [json.loads(line) for line in file_a.read_text(encoding="utf-8").splitlines()]
    updated_turn = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]
    data_id = updated_rows[0]["extra_info"]["unique_info"]["data_id"]
    assert updated_turn["sample_id"] == build_turn_id(data_id, 1)
    assert updated_turn["value"]["value_score"] == cached_value["value_score"]
