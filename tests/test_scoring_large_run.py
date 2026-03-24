import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from sft_label.config import PipelineConfig
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
