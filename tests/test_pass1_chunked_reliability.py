from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig


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


def _monitor(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "status": "success",
        "llm_calls": 2,
        "total_prompt_tokens": 12,
        "total_completion_tokens": 8,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": 0,
        "elapsed_seconds": 0.01,
    }


@pytest.mark.asyncio
async def test_chunked_non_inline_durability_keeps_existing_final_files_until_finalize(tmp_path, monkeypatch):
    import sft_label.pipeline as pipeline

    input_path = tmp_path / "source.jsonl"
    rows = [
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
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    final_dataset_path = output_dir / input_path.name
    final_labeled_path = output_dir / "labeled.jsonl"
    final_monitor_path = output_dir / "monitor.jsonl"
    final_failed_path = output_dir / "failed_samples.jsonl"

    final_dataset_path.write_text("old-dataset\n", encoding="utf-8")
    final_labeled_path.write_text("old-labeled\n", encoding="utf-8")
    final_monitor_path.write_text("old-monitor\n", encoding="utf-8")
    final_failed_path.write_text("old-failed\n", encoding="utf-8")

    async def fake_label_one(http_client, sample, model, sample_idx, total, sem,
                             enable_arbitration=True, config=None, rate_limiter=None,
                             llm_progress_cb=None, progress_event_cb=None):
        return sample_idx, _full_labels(), _monitor(sample.get("id", f"sample-{sample_idx}"))

    original_flush = pipeline._flush_chunk
    flush_calls = {"count": 0}

    def interrupted_flush(*args, **kwargs):
        original_flush(*args, **kwargs)
        flush_calls["count"] += 1
        if flush_calls["count"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pipeline, "_flush_chunk", interrupted_flush)

    with patch("sft_label.pipeline.label_one", side_effect=fake_label_one), pytest.raises(
        RuntimeError,
        match="simulated interruption",
    ):
        await pipeline._run_one_file_chunked(
            input_path=input_path,
            output_dir=output_dir,
            http_client=None,
            sem=asyncio.Semaphore(1),
            model="mock-model",
            enable_arbitration=False,
            config=PipelineConfig(
                concurrency=1,
                chunk_size=1,
                max_active_chunks=1,
                enable_adaptive_runtime=False,
                enable_stage_recovery_sweep=False,
            ),
        )

    assert final_dataset_path.read_text(encoding="utf-8") == "old-dataset\n"
    assert final_labeled_path.read_text(encoding="utf-8") == "old-labeled\n"
    assert final_monitor_path.read_text(encoding="utf-8") == "old-monitor\n"
    assert final_failed_path.read_text(encoding="utf-8") == "old-failed\n"

    assert (output_dir / f".{input_path.name}.next").exists()
    assert (output_dir / ".labeled.jsonl.next").exists()
    assert (output_dir / ".monitor.jsonl.next").exists()
    assert (output_dir / ".failed_samples.jsonl.next").exists()
