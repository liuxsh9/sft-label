from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig
from sft_label.llm_runtime import AdaptiveLLMRuntime, OutcomeClass, RequestOutcome
from sft_label.pipeline import label_one


def _sample_row(user_text: str = "q1") -> dict:
    return {
        "meta_prompt": ["system"],
        "data": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "a1"},
        ],
    }


def _full_labels() -> dict:
    return {
        "intent": "build",
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


def _monitor(status: str, *, retryable_infra: bool | None = None) -> dict:
    m = {
        "sample_id": "sample-000000",
        "index": 0,
        "llm_calls": 2,
        "total_prompt_tokens": 1,
        "total_completion_tokens": 1,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": 0,
        "status": status,
        "elapsed_seconds": 0.01,
    }
    if retryable_infra is not None:
        m["retryable_infra"] = retryable_infra
    return m


@pytest.mark.asyncio
async def test_pass1_chunked_recovery_sweep_recovers_infra_failure(tmp_path):
    from sft_label.pipeline import run

    input_path = tmp_path / "train.jsonl"
    input_path.write_text(json.dumps(_sample_row("q1"), ensure_ascii=False) + "\n", encoding="utf-8")

    calls: list[str] = []

    async def fake_label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True, config=None, rate_limiter=None):
        # First attempt fails with retryable infra; sweep re-calls and succeeds.
        calls.append(sample.get("id", ""))
        if len(calls) == 1:
            return sample_idx, None, _monitor("call1_failed", retryable_infra=True)
        return sample_idx, _full_labels(), _monitor("success")

    config = PipelineConfig(
        concurrency=1,
        chunk_size=1,
        max_active_chunks=1,
        enable_adaptive_runtime=True,
        enable_stage_recovery_sweep=True,
        recovery_sweep_max_passes=1,
    )

    with patch("sft_label.pipeline.label_one", side_effect=fake_label_one):
        stats = await run(input_path=str(input_path), output=str(tmp_path / "out"), config=config)

    assert stats["success"] == 1
    run_dir = Path(stats["run_dir"])
    meta_dir = run_dir / "meta_label_data" / "files" / "train"
    failed = meta_dir / "failed_samples.jsonl"
    assert failed.exists() is False
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_label_one_skips_arbitration_when_runtime_degraded():
    # Force runtime into degraded state before labeling.
    runtime = AdaptiveLLMRuntime(
        base_concurrency=2,
        base_rps=10.0,
        min_concurrency=1,
        min_rps=0.5,
        window_requests=10,
        window_seconds=10.0,
        degrade_timeout_rate=0.01,
        open_timeout_rate=2.0,
        degrade_overload_rate=0.9,
        open_overload_rate=0.9,
        degrade_abnormal_rate=0.9,
        open_base_cooldown=0.01,
        open_max_cooldown=0.02,
        probe_success_required=1,
    )
    runtime.observe(RequestOutcome(classification=OutcomeClass.TIMEOUT, error="timeout"))
    assert runtime.state == "degraded"

    config = PipelineConfig(
        sample_max_retries=0,
        max_retries=0,
        enable_adaptive_runtime=True,
        disable_arbitration_when_degraded=True,
        request_quick_retries=0,
    )
    setattr(config, "_adaptive_runtime", runtime)

    call1_ok = {
        "intent": "build",
        "language": ["python"],
        "domain": [],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "confidence": {"intent": 0.1},  # trigger low-conf arbitration
        "unmapped": [],
    }
    call2_ok = {
        "concept": ["algorithms"],
        "agentic": [],
        "constraint": [],
        "context": "snippet",
        "confidence": {"concept": 0.1},  # trigger low-conf arbitration
        "unmapped": [],
    }

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        # Only call1 + call2 should happen; arbitration calls should be skipped.
        idx = fake_async_llm_call.calls
        fake_async_llm_call.calls += 1
        if idx == 0:
            return call1_ok, json.dumps(call1_ok), {"prompt_tokens": 1, "completion_tokens": 1, "status_code": 200}
        return call2_ok, json.dumps(call2_ok), {"prompt_tokens": 1, "completion_tokens": 1, "status_code": 200}

    fake_async_llm_call.calls = 0

    sample = {
        "id": "s-1",
        "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
    }

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        _, labels, monitor = await label_one(
            http_client=None,
            sample=sample,
            model="mock",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=True,
            config=config,
            rate_limiter=None,
        )

    assert labels is not None
    assert monitor.get("arbitrated") is False
    assert monitor.get("arbitration_skipped") == "runtime:degraded"
    assert fake_async_llm_call.calls == 2


@pytest.mark.asyncio
async def test_label_one_non_adaptive_path_respects_shared_semaphore():
    active_calls = 0
    peak_active_calls = 0

    call1_ok = {
        "intent": "build",
        "language": ["python"],
        "domain": [],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "confidence": {"intent": 0.9},
        "unmapped": [],
    }
    call2_ok = {
        "concept": ["algorithms"],
        "agentic": [],
        "constraint": [],
        "context": "snippet",
        "confidence": {"concept": 0.9},
        "unmapped": [],
    }

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        nonlocal active_calls, peak_active_calls
        active_calls += 1
        peak_active_calls = max(peak_active_calls, active_calls)
        try:
            await asyncio.sleep(0.01)
            message_blob = json.dumps(messages, ensure_ascii=False)
            payload = call2_ok if '"concept"' in message_blob or '"context"' in message_blob else call1_ok
            return payload, json.dumps(payload), {"prompt_tokens": 1, "completion_tokens": 1, "status_code": 200}
        finally:
            active_calls -= 1

    config = PipelineConfig(
        sample_max_retries=0,
        max_retries=0,
        enable_adaptive_runtime=False,
        request_quick_retries=0,
    )
    shared_sem = asyncio.Semaphore(1)
    samples = [
        {
            "id": "s-1",
            "conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}],
        },
        {
            "id": "s-2",
            "conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}],
        },
    ]

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        await asyncio.gather(
            *[
                label_one(
                    http_client=None,
                    sample=sample,
                    model="mock",
                    sample_idx=idx,
                    total=len(samples),
                    sem=shared_sem,
                    enable_arbitration=False,
                    config=config,
                    rate_limiter=None,
                )
                for idx, sample in enumerate(samples)
            ]
        )

    assert peak_active_calls == 1
