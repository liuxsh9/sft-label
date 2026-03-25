from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.artifacts import DASHBOARDS_DIRNAME
from sft_label.config import PipelineConfig
from sft_label.inline_labels import build_turn_id
from sft_label.inline_pass1 import merge_pass1_results, prepare_inline_pass1_batch
from sft_label.inline_rows import build_row_sample_bundle, flatten_row_sample_bundles
from sft_label.preprocessing import apply_sparse_sampling
from sft_label.inline_scoring import (
    InlineScoringTarget,
    discover_inline_jsonl_files,
    infer_inline_scoring_target,
    load_inline_scoring_file,
)
from sft_label.run_layout import InlineRunLayout


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


def _extension_payload() -> dict:
    return {
        "ui_fine_labels": {
            "status": "success",
            "matched": True,
            "spec_version": "v1",
            "spec_hash": "sha256:ui-v1",
            "labels": {"component_type": ["form"]},
            "confidence": {"component_type": 0.8},
            "unmapped": [],
            "monitor": {"status": "success", "llm_calls": 1},
        }
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
    assert samples[0]["metadata"]["planner_policy"] == "compact_semantic_anchor_v1"
    assert samples[1]["metadata"]["slice_position"] == 2
    assert samples[1]["metadata"]["slice_count"] == 2
    assert samples[1]["metadata"]["conversation_uid"]


def test_prepare_inline_pass1_refresh_preserves_sparse_inheritance_for_tool_trajectory_growth():
    conversations = [{"from": "human", "value": "Debug the flaky retry pipeline end to end."}]
    for i in range(13):
        conversations.append({"from": "tool", "value": f"$ pytest tests/test_retry.py::test_case_{i}\nFAILED timeout"})
        conversations.append({"from": "gpt", "value": "I inspected the failure and tightened the retry handling."})

    bundle = build_row_sample_bundle(
        {"id": "inline-tool-trajectory-growth", "conversations": conversations},
        Path("synthetic.jsonl"),
        1,
    )
    config = PipelineConfig(planner_enabled=True, planner_metadata_only=False, prompt_mode="compact")
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

    prepared = prepare_inline_pass1_batch([bundle], mode="refresh", sparse_kwargs=sparse_kwargs)

    assert len(prepared.samples) == 13
    assert len(prepared.inherit_map) > 0
    assert len(prepared.label_indices) < len(prepared.samples)


def test_prepare_inline_pass1_refresh_matches_direct_sparse_plan():
    conversations = [{"from": "human", "value": "Debug the flaky retry pipeline end to end."}]
    for i in range(13):
        conversations.append({"from": "tool", "value": f"$ pytest tests/test_retry.py::test_case_{i}\nFAILED timeout"})
        conversations.append({"from": "gpt", "value": "I inspected the failure and tightened the retry handling."})

    bundle = build_row_sample_bundle(
        {"id": "inline-direct-consistency", "conversations": conversations},
        Path("synthetic.jsonl"),
        1,
    )
    config = PipelineConfig(planner_enabled=True, planner_metadata_only=False, prompt_mode="compact")
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

    direct_label_indices, direct_inherit_map = apply_sparse_sampling(bundle.samples, **sparse_kwargs)
    prepared = prepare_inline_pass1_batch([bundle], mode="refresh", sparse_kwargs=sparse_kwargs)

    assert sorted(prepared.label_indices) == sorted(direct_label_indices)
    assert prepared.inherit_map == direct_inherit_map


def test_infer_inline_scoring_target_does_not_misclassify_standard_run_file(tmp_path):
    run_root = tmp_path / "dataset_labeled_20260318_120000"
    standard_file = run_root / "batch" / "labeled.jsonl"
    standard_file.parent.mkdir(parents=True)
    standard_file.write_text("", encoding="utf-8")

    target = infer_inline_scoring_target(standard_file)

    assert target is None


def test_infer_inline_scoring_target_ignores_sidecar_only_dataset_dir(tmp_path):
    run_root = tmp_path / "dataset_labeled_20260318_120000"
    dataset_dir = run_root / "batch"
    dataset_dir.mkdir(parents=True)
    (run_root / "meta_label_data").mkdir()
    (dataset_dir / "._ghost.jsonl").write_bytes(b"\x00\x01")

    target = infer_inline_scoring_target(run_root)

    assert target is None


def test_infer_inline_scoring_target_ignores_generated_manifest_dir(tmp_path):
    run_root = tmp_path / "dataset_labeled_20260318_120000"
    dataset_dir = run_root / "dataset"
    manifest_dir = run_root / "manifest"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    (run_root / "meta_label_data").mkdir()
    (dataset_dir / "train.jsonl").write_text('{"id":"ok"}\n', encoding="utf-8")
    (manifest_dir / "monitor_value.jsonl").write_text('{"id":"generated"}\n', encoding="utf-8")

    target = infer_inline_scoring_target(run_root)

    assert target is not None
    assert target.target_path == run_root.resolve()
    assert target.layout.dataset_root == dataset_dir.resolve()


def test_discover_inline_jsonl_files_ignores_macos_sidecars(tmp_path):
    run_root = tmp_path / "train_labeled_20260311_120000"
    dataset_root = run_root / "train"
    dataset_root.mkdir(parents=True)
    (run_root / "meta_label_data").mkdir()
    good = dataset_root / "train.jsonl"
    good.write_text('{"id":"ok"}\n', encoding="utf-8")
    (dataset_root / "._train.jsonl").write_bytes(b"\x00\x01")

    target = InlineScoringTarget(
        layout=InlineRunLayout.from_paths(dataset_root, run_root),
        target_path=run_root,
    )

    assert discover_inline_jsonl_files(target) == [good.resolve()]


def test_load_inline_scoring_file_rehydrates_label_extensions(tmp_path):
    source_file = tmp_path / "train.jsonl"
    row = {
        "id": "conv-1",
        "conversations": [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
        ],
        "extra_info": {
            "unique_info": {
                "data_label": {
                    "meta": {"schema_version": "1", "label_version": "inline-v1"},
                    "turns": [{
                        "turn_index": 1,
                        "sample_id": "s1",
                        "labels": _full_labels("build"),
                        "label_extensions": _extension_payload(),
                        "labeling_monitor": {"status": "success"},
                    }],
                    "conversation": {"turn_count": 1},
                }
            }
        },
    }
    _write_jsonl(source_file, [row])

    _, samples, _ = load_inline_scoring_file(source_file)

    assert samples[0]["label_extensions"]["ui_fine_labels"]["labels"]["component_type"] == ["form"]


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
        stats = await run_scoring(
            str(source_file),
            config=PipelineConfig(scoring_concurrency=1, enable_adaptive_runtime=True),
        )

    assert stats["total_scored"] == 2
    artifact_dir = run_root / "meta_label_data" / "files" / "train"
    assert (artifact_dir / "labeled.jsonl").exists()
    assert (artifact_dir / "scored.jsonl").exists()
    assert (artifact_dir / "monitor_value.jsonl").exists()
    assert (artifact_dir / "stats_scoring.json").exists()
    stats_payload = json.loads((artifact_dir / "stats_scoring.json").read_text(encoding="utf-8"))
    assert stats_payload["file"] == "train.jsonl"
    assert stats_payload["adaptive_runtime"]["enabled"] is True

    updated_rows = [json.loads(line) for line in source_file.read_text(encoding="utf-8").splitlines()]
    turns = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"]
    assert turns[0]["value"]["value_score"] is not None
    assert turns[0]["value"]["selection_score"] is not None
    assert turns[1]["value"]["selection_score"] is not None
    assert "metadata" not in turns[0]
    conversation = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["conversation"]
    assert conversation["conv_value"] is not None
    assert conversation["conv_selection"] is not None
    assert conversation["conversation_key"]
    assert (
        conversation["conversation_key"].endswith("::conv-2")
        or conversation["conversation_key"].startswith("conversation_uid:")
    )
    assert conversation["source_file"].endswith("train.jsonl")
    assert "compression_gap" in conversation
    assert "tool_turn_ratio" in conversation
    assert conversation["detail"]["top_k_mean"] is not None
    assert "turn_value_std" in conversation["detail"]
    assert "merged_labels" not in conversation
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
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(scoring_concurrency=1, enable_adaptive_runtime=True),
        )

    assert summary["files_processed"] == 2
    assert (run_root / "meta_label_data" / "summary_stats_scoring.json").exists()
    assert (run_root / "meta_label_data" / "conversation_scores.json").exists()
    dashboards_dir = run_root / "meta_label_data" / DASHBOARDS_DIRNAME
    assert (dashboards_dir / "dashboard_scoring_dataset.html").exists()
    assert not (dashboards_dir / "dashboard_scoring.html").exists()
    assert {row["file"] for row in summary.get("per_file_summary", [])} == {"code/a.jsonl", "multi/b.jsonl"}

    canonical_conv = json.loads((run_root / "meta_label_data" / "conversation_scores.json").read_text(encoding="utf-8"))
    assert canonical_conv
    multi_turn_record = next(
        record for record in canonical_conv
        if (record.get("merged_labels") or {}).get("intent") in {"debug", "modify"}
    )
    assert "slices" in multi_turn_record
    assert multi_turn_record["detail"]["conv_intra_class_rank"] is not None

    file_conv = json.loads(
        (run_root / "meta_label_data" / "files" / "multi" / "b" / "conversation_scores.json").read_text(
            encoding="utf-8"
        )
    )
    assert file_conv
    assert file_conv[0]["merged_labels"]["intent"] in {"debug", "modify"}

    scoring_dashboard = dashboards_dir / "dashboard_scoring_dataset.data"
    manifest = json.loads((scoring_dashboard / "manifest.json").read_text(encoding="utf-8"))
    detail = json.loads((scoring_dashboard / "scopes" / "global.json").read_text(encoding="utf-8"))
    assert manifest["scopes"]["global"]["summary_modes"]["conversation"]["pass1_total"] == 2
    assert detail["pass1"]["modes"]["conversation"]["distribution_total"] == 2
    assert detail["pass1"]["modes"]["conversation"]["distributions"]["intent"] in (
        {"build": 1, "debug": 1},
        {"build": 1, "modify": 1},
    )

    updated_b = [json.loads(line) for line in file_b.read_text(encoding="utf-8").splitlines()]
    turns = updated_b[0]["extra_info"]["unique_info"]["data_label"]["turns"]
    assert turns[0]["value"]["selection_score"] is not None
    assert turns[1]["value"]["selection_score"] is not None
    conversation = updated_b[0]["extra_info"]["unique_info"]["data_label"]["conversation"]
    assert conversation["conversation_key"]
    assert (
        conversation["conversation_key"].endswith("::conv-b")
        or conversation["conversation_key"].startswith("conversation_uid:")
    )
    assert conversation["detail"]["turn_value_std"] is not None


@pytest.mark.asyncio
async def test_run_scoring_inline_run_dir_repairs_deferred_postprocess_metadata(tmp_path):
    from sft_label.artifacts import PASS2_SUMMARY_STATS_FILE
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    source_file = run_root / "dataset" / "sample_1.jsonl"

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
        [[_full_labels("build"), _full_labels("debug")]],
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
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(
                scoring_concurrency=1,
                enable_adaptive_runtime=True,
                pass2_heavy_postprocess_mode="auto",
                pass2_heavy_postprocess_sample_threshold=1,
                pass2_heavy_postprocess_file_bytes_threshold=10**12,
            ),
        )

    assert summary["run_dir"] == str(run_root)
    assert summary["postprocess"]["conversation_scores"]["status"] == "completed"
    assert summary["postprocess"]["dashboard"]["status"] == "completed"
    assert (run_root / "meta_label_data" / "conversation_scores.json").exists()

    persisted_summary = json.loads(
        (run_root / "meta_label_data" / PASS2_SUMMARY_STATS_FILE).read_text(encoding="utf-8")
    )
    assert persisted_summary["run_dir"] == str(run_root)
    assert persisted_summary["postprocess"]["conversation_scores"]["status"] == "completed"
    assert persisted_summary["postprocess"]["dashboard"]["status"] == "completed"

    dashboard_detail = json.loads(
        (
            run_root
            / "meta_label_data"
            / "dashboards"
            / "dashboard_scoring_dataset.data"
            / "scopes"
            / "global.json"
        ).read_text(encoding="utf-8")
    )
    assert dashboard_detail["pass2"]["modes"]["sample"]["overview"]["total_scored"] == 2
    assert dashboard_detail["pass2"]["modes"]["conversation"]["overview"]["total_scored"] == 1
    assert dashboard_detail["summary_modes"]["conversation"]["scored_total"] == 1


@pytest.mark.asyncio
async def test_run_scoring_inline_run_dir_counts_single_reply_rows_as_conversations(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    source_file = run_root / "dataset" / "sample_2.jsonl"

    rows = _inline_rows_for_file(
        source_file,
        [
            {
                "id": "conv-single",
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                ],
            },
            {
                "id": "conv-multi",
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ],
            },
        ],
        [
            [_full_labels("build")],
            [_full_labels("debug"), _full_labels("modify")],
        ],
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
        await run_scoring(
            str(run_root),
            config=PipelineConfig(
                scoring_concurrency=1,
                enable_adaptive_runtime=True,
                pass2_heavy_postprocess_mode="always",
            ),
        )

    conv_records = json.loads(
        (run_root / "meta_label_data" / "conversation_scores.json").read_text(encoding="utf-8")
    )
    assert len(conv_records) == 2
    assert {record["turn_count"] for record in conv_records} == {1, 2}

    dashboard_detail = json.loads(
        (
            run_root
            / "meta_label_data"
            / "dashboards"
            / "dashboard_scoring_dataset.data"
            / "scopes"
            / "global.json"
        ).read_text(encoding="utf-8")
    )
    assert dashboard_detail["pass1"]["modes"]["conversation"]["total"] == 2
    assert dashboard_detail["pass2"]["modes"]["sample"]["overview"]["total_scored"] == 3
    assert dashboard_detail["pass2"]["modes"]["conversation"]["overview"]["total_scored"] == 2
    assert dashboard_detail["summary_modes"]["conversation"]["pass1_total"] == 2
    assert dashboard_detail["summary_modes"]["conversation"]["scored_total"] == 2


@pytest.mark.asyncio
async def test_run_scoring_existing_inline_run_dir_prefers_inline_layout_over_legacy_root_markers(tmp_path):
    from sft_label.artifacts import PASS1_SUMMARY_STATS_FILE, PASS2_SUMMARY_STATS_FILE
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    source_file = run_root / "dataset" / "train.jsonl"
    rows = _inline_rows_for_file(
        source_file,
        [{
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
            ],
        }],
        [[_full_labels("build")]],
    )
    _write_jsonl(source_file, rows)

    meta_root = run_root / "meta_label_data"
    meta_root.mkdir(parents=True, exist_ok=True)
    meta_summary = meta_root / PASS1_SUMMARY_STATS_FILE
    meta_summary.write_text(
        json.dumps({
            "total_samples": 100,
            "distribution_total_samples": 80,
            "tag_distributions": {"intent": {"build": 80}},
        }),
        encoding="utf-8",
    )

    # Simulate a previously mis-run Pass 2 that left top-level legacy markers behind.
    (run_root / PASS2_SUMMARY_STATS_FILE).write_text("{}", encoding="utf-8")
    manifest_dir = run_root / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "stats_labeling_manifest.json").write_text(
        json.dumps({
            "total_samples": 1,
            "distribution_total_samples": 1,
            "tag_distributions": {"intent": {"build": 1}},
        }),
        encoding="utf-8",
    )

    async def _fake_inline(*args, **kwargs):
        target = args[0]
        assert target.layout.meta_root == meta_root.resolve()
        return {
            "files_processed": 1,
            "rarity_config": {"stats_ref": str(meta_summary.resolve())},
        }

    with patch("sft_label.scoring._run_inline_scoring_directory", side_effect=_fake_inline) as mock_inline, \
            patch("sft_label.scoring._run_scoring_directory", side_effect=AssertionError("standard directory scoring should not be used")):
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(scoring_concurrency=1),
        )

    assert mock_inline.call_count == 1
    assert summary["rarity_config"]["stats_ref"] == str(meta_summary.resolve())


@pytest.mark.asyncio
async def test_inline_directory_prefers_chunked_jsonl_scoring(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    file_a = run_root / "dataset" / "code" / "a.jsonl"

    rows_a = _inline_rows_for_file(
        file_a,
        [{
            "id": "conv-a",
            "conversations": [
                {"from": "human", "value": "qa"},
                {"from": "gpt", "value": "aa"},
            ],
        }],
        [[_full_labels("build")]],
    )
    _write_jsonl(file_a, rows_a)

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
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(
                scoring_concurrency=1,
                chunk_size=1,
                max_active_chunks=1,
                enable_adaptive_runtime=False,
                enable_stage_recovery_sweep=False,
            ),
        )

    stats_files = list((run_root / "meta_label_data" / "files").rglob("stats_scoring.json"))
    assert len(stats_files) == 1
    per_file_stats = json.loads(stats_files[0].read_text(encoding="utf-8"))
    assert per_file_stats.get("chunked") is True
    assert summary["files_processed"] == 1


@pytest.mark.asyncio
async def test_inline_directory_large_jsonl_chunked_smoke_scores_all_rows(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    file_a = run_root / "dataset" / "code" / "a.jsonl"
    row_count = 256

    rows = [
        {
            "id": f"conv-{i}",
            "conversations": [
                {"from": "human", "value": f"q{i}"},
                {"from": "gpt", "value": f"a{i}"},
            ],
        }
        for i in range(row_count)
    ]
    merged_rows = _inline_rows_for_file(
        file_a,
        rows,
        [[_full_labels("build")] for _ in range(row_count)],
    )
    _write_jsonl(file_a, merged_rows)

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
        summary = await run_scoring(
            str(run_root),
            config=PipelineConfig(
                scoring_concurrency=4,
                chunk_size=64,
                max_active_chunks=2,
                enable_adaptive_runtime=False,
                enable_stage_recovery_sweep=False,
            ),
        )

    stats_files = list((run_root / "meta_label_data" / "files").rglob("stats_scoring.json"))
    assert len(stats_files) == 1
    per_file_stats = json.loads(stats_files[0].read_text(encoding="utf-8"))
    scored_lines = [
        line for line in (stats_files[0].parent / "scored.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert per_file_stats["total_scored"] == row_count
    assert len(scored_lines) == row_count
    assert summary["files_processed"] == 1


@pytest.mark.asyncio
async def test_inline_directory_chunked_summary_preserves_flags_thinking_mode_and_exact_single_file_means(tmp_path):
    from sft_label.scoring import run_scoring

    run_root = tmp_path / "dataset_labeled_20260311_120000"
    file_a = run_root / "dataset" / "code" / "a.jsonl"
    rows_a = _inline_rows_for_file(
        file_a,
        [{
            "id": "conv-a",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }],
        [[_full_labels("build"), _full_labels("modify")]],
    )
    _write_jsonl(file_a, rows_a)

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        is_second = sample_idx == 1
        return {
            "complexity": {"instruction": 6, "analytical_depth": 6, "implementation": 6, "overall": 6},
            "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
            "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
            "rarity": rarity_result or {"score": 5.0},
            "flags": ["incomplete"] if is_second else ["has-bug"],
            "thinking_mode": "fast" if is_second else "slow",
            "value_score": 6.4 if is_second else 5.9,
            "confidence": 0.85,
        }, {
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
            config=PipelineConfig(
                scoring_concurrency=1,
                chunk_size=1,
                max_active_chunks=1,
                enable_adaptive_runtime=False,
                enable_stage_recovery_sweep=False,
            ),
        )

    stats_path = run_root / "meta_label_data" / "files" / "code" / "a" / "stats_scoring.json"
    summary_path = run_root / "meta_label_data" / "summary_stats_scoring.json"
    per_file_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    summary_stats = json.loads(summary_path.read_text(encoding="utf-8"))

    assert per_file_stats["flag_counts"] == {"has-bug": 1, "incomplete": 1}
    assert per_file_stats["flag_value_impact"] == {
        "has-bug": {"mean_value": 5.9, "count": 1},
        "incomplete": {"mean_value": 6.4, "count": 1},
    }
    assert per_file_stats["thinking_mode_stats"]["slow"]["count"] == 1
    assert per_file_stats["thinking_mode_stats"]["fast"]["count"] == 1

    assert summary_stats["flag_counts"] == per_file_stats["flag_counts"]
    assert summary_stats["flag_value_impact"] == per_file_stats["flag_value_impact"]
    assert summary_stats["thinking_mode_stats"] == per_file_stats["thinking_mode_stats"]
    assert summary_stats["score_distributions"]["value_score"] == per_file_stats["score_distributions"]["value_score"]
    assert summary["score_distributions"]["value_score"] == per_file_stats["score_distributions"]["value_score"]


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
            config=PipelineConfig(
                scoring_concurrency=1,
                chunk_size=1,
                max_active_chunks=1,
                enable_adaptive_runtime=True,
            ),
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
            config=PipelineConfig(
                scoring_concurrency=1,
                chunk_size=1,
                max_active_chunks=1,
                enable_adaptive_runtime=True,
            ),
            resume=True,
        )

    assert summary["files_processed"] == 1
    assert calls["count"] == 0
    updated_rows = [json.loads(line) for line in file_a.read_text(encoding="utf-8").splitlines()]
    updated_turn = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"][0]
    data_id = updated_rows[0]["extra_info"]["unique_info"]["data_id"]
    assert updated_turn["sample_id"] == build_turn_id(data_id, 1)
    assert updated_turn["value"]["value_score"] == cached_value["value_score"]
