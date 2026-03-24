import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from sft_label.config import PipelineConfig
from sft_label import scoring_large_run as scoring_large_run_module
from sft_label.scoring import run_scoring


def _write_labeled_file(path: Path, sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for sample_id in sample_ids:
        payload.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": f"Question {sample_id}"},
                    {"from": "gpt", "value": f"Answer {sample_id}"},
                ],
                "labels": {
                    "intent": "build",
                    "difficulty": "intermediate",
                    "language": ["python"],
                },
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _selection_snapshot(run_dir: Path) -> dict[str, tuple[float | None, float | None]]:
    snapshot = {}
    for scored_file in sorted(run_dir.glob("*/scored.json")):
        rows = json.loads(scored_file.read_text(encoding="utf-8"))
        for row in rows:
            value = row.get("value") or {}
            snapshot[row["id"]] = (
                value.get("selection_score"),
                value.get("intra_class_rank"),
            )
    return snapshot


def test_directory_deferred_mode_preserves_finalized_global_selection_semantics(tmp_path):
    quality_by_id = {
        "a-low": 2,
        "a-high": 4,
        "b-low": 7,
        "b-high": 9,
    }

    def _prepare_inputs(root: Path) -> None:
        _write_labeled_file(root / "batch_a" / "labeled.json", ["a-low", "a-high"])
        _write_labeled_file(root / "batch_b" / "labeled.json", ["b-low", "b-high"])

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        q = quality_by_id[sample["id"]]
        value = {
            "complexity": {"overall": q},
            "quality": {"overall": q},
            "reasoning": {"overall": q},
            "rarity": rarity_result,
            "flags": [],
            "thinking_mode": "fast",
            "value_score": float(q),
            "confidence": 0.86,
        }
        monitor = {
            "sample_id": sample["id"],
            "status": "success",
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "attempts": 1,
            "validation_issues": [],
        }
        return value, monitor

    always_dir = tmp_path / "always"
    defer_dir = tmp_path / "defer"
    _prepare_inputs(always_dir)
    _prepare_inputs(defer_dir)

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one), patch(
        "sft_label.tools.visualize_value.generate_value_dashboard",
        side_effect=lambda *args, **kwargs: None,
    ):
        always_summary = asyncio.run(
            run_scoring(
                str(always_dir),
                config=PipelineConfig(
                    scoring_concurrency=1,
                    sample_max_retries=1,
                    enable_adaptive_runtime=False,
                    enable_stage_recovery_sweep=False,
                    pass2_heavy_postprocess_mode="always",
                ),
            )
        )

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one), \
            patch(
                "sft_label.conversation.merge_conversation_record_batches",
                side_effect=AssertionError("conversation aggregation should be deferred"),
            ), \
            patch(
                "sft_label.tools.visualize_value.generate_value_dashboard",
                side_effect=AssertionError("dashboard generation should be deferred"),
            ):
        summary = asyncio.run(
            run_scoring(
                str(defer_dir),
                config=PipelineConfig(
                    scoring_concurrency=1,
                    sample_max_retries=1,
                    enable_adaptive_runtime=False,
                    enable_stage_recovery_sweep=False,
                    pass2_heavy_postprocess_mode="defer",
                ),
            )
        )

    assert summary["postprocess"]["conversation_scores"]["status"] == "deferred"
    assert summary["postprocess"]["dashboard"]["status"] == "deferred"
    assert always_summary["postprocess"]["conversation_scores"]["status"] == "completed"
    assert always_summary["postprocess"]["dashboard"]["status"] == "completed"
    assert _selection_snapshot(defer_dir) == _selection_snapshot(always_dir)


def test_directory_global_selection_rewrite_restreams_per_file_summaries(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    batch_a = output_dir / "batch_a"
    batch_b = output_dir / "batch_b"
    batch_a.mkdir(parents=True)
    batch_b.mkdir(parents=True)
    scored_a = batch_a / "scored.jsonl"
    scored_b = batch_b / "scored.jsonl"
    scored_a.write_text("", encoding="utf-8")
    scored_b.write_text("", encoding="utf-8")

    stream_calls = {scored_a: 0, scored_b: 0}

    def _stream(scored_path, **_kwargs):
        path = Path(scored_path)
        stream_calls[path] += 1
        return [
            {
                "value_score": 5.0 if path == scored_a else 7.0,
                "pure_quality": 5.0 if path == scored_a else 7.0,
                "rarity_score": 5.0,
                "labels": {"intent": "build"},
            }
        ], 1

    monkeypatch.setattr(scoring_large_run_module, "stream_selection_summaries", _stream)
    monkeypatch.setattr(
        scoring_large_run_module,
        "rewrite_scored_jsonl_selection",
        lambda _scored_path, _selection_results, cursor, **_kwargs: cursor + 1,
    )
    monkeypatch.setattr(
        scoring_large_run_module,
        "rewrite_scored_json_sibling_from_jsonl",
        lambda *_args, **_kwargs: None,
    )

    scoring_large_run_module.rewrite_directory_global_selection(
        output_dir=output_dir,
        input_dir=output_dir,
        config=PipelineConfig(),
        pass2_stats_file="stats_scoring.json",
        pass2_dashboard_file="dashboard_scoring.html",
        discover_scored_output_files=lambda _out: [scored_a, scored_b],
        selection_summary_from_sample=lambda _sample, config=None: {},
        compute_selection_scores_from_summaries=lambda summaries, config=None: [
            {"selection_score": float(index + 1), "intra_class_rank": float(index + 1)}
            for index, _ in enumerate(summaries)
        ],
        apply_v2_scores=lambda samples, config=None: None,
        load_scored_samples=lambda _path: [],
        write_scored_samples=lambda _path, _samples: None,
        compute_value_stats_from_summaries=lambda file_summaries, monitor_totals, scored_count: {
            "total_scored": scored_count,
            "score_distributions": {},
        },
        load_monitor_totals=lambda _dir: {},
        load_existing_pass2_stats=lambda _dir: {},
        relative_file_label=lambda file_label_source, _input_dir: Path(file_label_source).name,
        generate_dashboard=False,
    )

    assert stream_calls[scored_a] == 2
    assert stream_calls[scored_b] == 2
