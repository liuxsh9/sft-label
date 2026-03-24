import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from sft_label.config import PipelineConfig
from sft_label.scoring import run_scoring


def _write_labeled_file(path: Path, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
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
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_directory_deferred_mode_skips_global_rewrite_and_keeps_selection_fields(tmp_path):
    _write_labeled_file(tmp_path / "batch_a" / "labeled.json", "a-1")
    _write_labeled_file(tmp_path / "batch_b" / "labeled.json", "b-1")

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        value = {
            "complexity": {"overall": 5},
            "quality": {"overall": 7},
            "reasoning": {"overall": 6},
            "rarity": rarity_result,
            "flags": [],
            "thinking_mode": "fast",
            "value_score": 6.3,
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

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one), \
            patch(
                "sft_label.scoring._rewrite_directory_global_selection",
                side_effect=AssertionError("global rewrite should be deferred in large-run mode"),
            ), \
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
                str(tmp_path),
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

    for batch in ("batch_a", "batch_b"):
        scored_rows = json.loads((tmp_path / batch / "scored.json").read_text(encoding="utf-8"))
        assert scored_rows
        for row in scored_rows:
            value = row.get("value") or {}
            assert "selection_score" in value
            assert "intra_class_rank" in value
