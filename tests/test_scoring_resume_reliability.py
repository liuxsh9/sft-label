import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

from sft_label.config import PipelineConfig
from sft_label.scoring import _pass2_checkpoint_path, _run_scoring_file_chunked


def _sample(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": f"Q:{sample_id}"},
            {"from": "gpt", "value": f"A:{sample_id}"},
        ],
        "labels": {"intent": "build", "difficulty": "intermediate"},
    }


def _value_payload(*, score: float, rarity_score: float, confidence: float) -> dict:
    return {
        "complexity": {
            "instruction": 5,
            "analytical_depth": 5,
            "implementation": 5,
            "overall": 5,
        },
        "quality": {
            "correctness": 7,
            "code_quality": 7,
            "explanation": 7,
            "completeness": 7,
            "overall": 7,
        },
        "reasoning": {
            "clarity": 6,
            "consistency": 6,
            "self_correction": False,
            "overall": 6,
        },
        "rarity": {"score": rarity_score},
        "flags": [],
        "thinking_mode": "fast",
        "value_score": score,
        "confidence": confidence,
    }


def _monitor(sample_id: str, *, status: str = "success", retryable_infra: bool | None = None) -> dict:
    payload = {
        "sample_id": sample_id,
        "status": status,
        "llm_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "attempts": 1,
        "validation_issues": [],
    }
    if retryable_infra is not None:
        payload["retryable_infra"] = retryable_infra
    return payload


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _quiet_chunked_config(**overrides) -> PipelineConfig:
    base = dict(
        scoring_concurrency=1,
        chunk_size=1,
        max_active_chunks=2,
        enable_adaptive_runtime=False,
        pass2_heavy_postprocess_mode="defer",
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_chunked_resume_fastpath_materializes_checkpoint_only_artifacts(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample(f"sample-{idx}") for idx in range(3)]
    _write_jsonl(labeled_path, samples)

    checkpoint_scored = _pass2_checkpoint_path(tmp_path, "scored.jsonl")
    checkpoint_monitor = _pass2_checkpoint_path(tmp_path, "monitor_value.jsonl")
    _write_jsonl(
        checkpoint_scored,
        [
            {
                **sample,
                "value": _value_payload(score=6.0 + idx, rarity_score=4.0 + idx, confidence=0.8),
            }
            for idx, sample in enumerate(samples)
        ],
    )
    _write_jsonl(checkpoint_monitor, [_monitor(sample["id"]) for sample in samples])

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during checkpoint-only resume finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=False),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    assert (tmp_path / "scored.jsonl").exists()
    assert (tmp_path / "monitor_value.jsonl").exists()


def test_chunked_recovery_sweep_keeps_failure_outputs_consistent_after_recovery(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-recovered"), _sample("sample-failed")]
    _write_jsonl(labeled_path, samples)

    calls: dict[str, int] = {}

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        sid = sample["id"]
        calls[sid] = calls.get(sid, 0) + 1

        if sid == "sample-recovered" and calls[sid] == 1:
            return None, {
                **_monitor(sid, status="failed", retryable_infra=True),
                "error": "timeout",
                "error_class": "timeout",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        if sid == "sample-failed":
            return None, {
                **_monitor(sid, status="failed", retryable_infra=False),
                "error": "invalid_request",
                "error_class": "input_error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        return _value_payload(score=7.2, rarity_score=5.0, confidence=0.9), _monitor(sid)

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(
                    enable_stage_recovery_sweep=True,
                    recovery_sweep_max_passes=1,
                ),
                resume=False,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    failed_ids = {row["id"] for row in _read_jsonl(tmp_path / "failed_value.jsonl")}
    failure_log_ids = {row["sample_id"] for row in _read_jsonl(tmp_path / "score_failures.jsonl")}

    assert failed_ids == {"sample-failed"}
    assert failure_log_ids == {"sample-failed"}


def test_chunked_resume_fastpath_honors_working_checkpoint_final_precedence(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample(f"sample-{idx}") for idx in range(2)]
    _write_jsonl(labeled_path, samples)

    final_scored = [
        {
            **sample,
            "value": _value_payload(score=1.5, rarity_score=1.0, confidence=0.1),
        }
        for sample in samples
    ]
    final_monitors = [_monitor(sample["id"], status="final-source") for sample in samples]
    _write_jsonl(tmp_path / "scored.jsonl", final_scored)
    _write_jsonl(tmp_path / "monitor_value.jsonl", final_monitors)

    checkpoint_scored = [
        {
            **sample,
            "value": _value_payload(score=8.5, rarity_score=6.0, confidence=0.99),
        }
        for sample in samples
    ]
    checkpoint_monitors = [_monitor(sample["id"], status="checkpoint-source") for sample in samples]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), checkpoint_scored)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), checkpoint_monitors)

    working_scored = [
        {
            **sample,
            "value": _value_payload(score=9.6, rarity_score=7.0, confidence=0.97),
        }
        for sample in samples
    ]
    working_monitors = [_monitor(sample["id"], status="working-source") for sample in samples]
    _write_jsonl(tmp_path / ".scored.jsonl.next", working_scored)
    _write_jsonl(tmp_path / ".monitor_value.jsonl.next", working_monitors)

    # Force freshness ordering opposite of precedence to ensure finalize order is authoritative.
    os.utime(tmp_path / ".scored.jsonl.next", (100, 100))
    os.utime(tmp_path / ".monitor_value.jsonl.next", (100, 100))
    os.utime(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), (200, 200))
    os.utime(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), (200, 200))
    os.utime(tmp_path / "scored.jsonl", (300, 300))
    os.utime(tmp_path / "monitor_value.jsonl", (300, 300))

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during resume fast-path finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=False),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    final_scored_rows = _read_jsonl(tmp_path / "scored.jsonl")
    final_monitor_rows = _read_jsonl(tmp_path / "monitor_value.jsonl")
    assert {row["value"]["value_score"] for row in final_scored_rows} == {9.6}
    assert {row["status"] for row in final_monitor_rows} == {"working-source"}
