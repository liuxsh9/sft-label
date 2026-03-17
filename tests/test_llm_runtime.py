from __future__ import annotations

import asyncio
import time

import pytest

from sft_label.llm_runtime import (
    AdaptiveLLMRuntime,
    AdaptiveRateLimiter,
    DynamicConcurrencyGate,
    OutcomeClass,
    classify_exception,
    classify_http_result,
)


@pytest.mark.asyncio
async def test_dynamic_concurrency_gate_shrink_and_expand():
    gate = DynamicConcurrencyGate(initial_limit=3, min_limit=1, max_limit=8)
    p1 = await gate.acquire()
    p2 = await gate.acquire()
    p3 = await gate.acquire()
    assert gate.in_flight == 3
    assert gate.limit == 3

    acquired = asyncio.Event()

    async def waiter():
        permit = await gate.acquire()
        acquired.set()
        permit.release()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.02)
    assert acquired.is_set() is False

    gate.set_limit(1)
    p1.release()
    await asyncio.sleep(0.02)
    assert acquired.is_set() is False
    p2.release()
    await asyncio.sleep(0.02)
    assert acquired.is_set() is False
    p3.release()
    await asyncio.wait_for(acquired.wait(), timeout=0.2)
    await task

    gate.set_limit(2)
    hold = await gate.acquire()
    acquired_expand = asyncio.Event()

    async def waiter_expand():
        permit = await gate.acquire()
        acquired_expand.set()
        permit.release()

    task_expand = asyncio.create_task(waiter_expand())
    await asyncio.sleep(0.02)
    await asyncio.wait_for(acquired_expand.wait(), timeout=0.2)
    hold.release()
    await task_expand


@pytest.mark.asyncio
async def test_dynamic_concurrency_gate_pause_resume():
    gate = DynamicConcurrencyGate(initial_limit=1, min_limit=1, max_limit=4)
    gate.pause()

    acquired = asyncio.Event()

    async def waiter():
        permit = await gate.acquire()
        acquired.set()
        permit.release()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.02)
    assert acquired.is_set() is False

    gate.resume()
    await asyncio.wait_for(acquired.wait(), timeout=0.2)
    await task


@pytest.mark.asyncio
async def test_adaptive_rate_limiter_pause_and_retarget():
    limiter = AdaptiveRateLimiter(target_rps=5.0, burst=1, initial_tokens=0.0)

    t0 = time.perf_counter()
    await limiter.acquire()
    slow_elapsed = time.perf_counter() - t0
    assert slow_elapsed >= 0.15

    limiter.set_target_rps(100.0)
    t1 = time.perf_counter()
    await limiter.acquire()
    fast_elapsed = time.perf_counter() - t1
    assert fast_elapsed < 0.08

    limiter.pause_for(0.05)
    t2 = time.perf_counter()
    await limiter.acquire()
    paused_elapsed = time.perf_counter() - t2
    assert paused_elapsed >= 0.045


@pytest.mark.asyncio
async def test_adaptive_runtime_state_transitions_and_snapshot():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=8,
        base_rps=20.0,
        min_concurrency=1,
        min_rps=0.5,
        window_requests=6,
        window_seconds=10.0,
        degrade_timeout_rate=0.2,
        open_timeout_rate=0.75,
        degrade_overload_rate=0.2,
        open_overload_rate=0.6,
        degrade_abnormal_rate=0.2,
        open_base_cooldown=0.02,
        open_max_cooldown=0.05,
        probe_success_required=2,
    )

    assert runtime.state == "healthy"
    runtime.observe(classify_http_result(200))
    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    assert runtime.state == "degraded"

    runtime.observe(classify_http_result(429))
    runtime.observe(classify_http_result(503))
    runtime.observe(classify_http_result(429))
    runtime.observe(classify_http_result(429))
    assert runtime.state == "open"
    assert runtime.snapshot()["gate_paused"] is True

    await asyncio.sleep(0.03)
    permit = await runtime.acquire(stage="pass1", sample_id="s-1")
    permit.release()
    assert runtime.state == "probing"

    runtime.observe(classify_http_result(200))
    runtime.observe(classify_http_result(200))
    assert runtime.state in {"degraded", "healthy"}

    for _ in range(6):
        runtime.observe(classify_http_result(200))
    snap = runtime.snapshot()
    assert snap["state"] in {"healthy", "degraded"}
    assert snap["base_concurrency"] == 8
    assert "rates" in snap


def test_classify_http_result_basics():
    ok = classify_http_result(200)
    assert ok.classification == OutcomeClass.SUCCESS
    assert ok.is_retryable is False

    overload = classify_http_result(429)
    assert overload.classification == OutcomeClass.OVERLOAD
    assert overload.is_retryable is True
    assert overload.is_infra_failure is True

    server = classify_http_result(503)
    assert server.classification == OutcomeClass.SERVER_ERROR
    assert server.is_retryable is True

    malformed = classify_http_result(200, parse_error=True)
    assert malformed.classification == OutcomeClass.ABNORMAL_RESPONSE
    assert malformed.is_infra_failure is True

    content_filtered = classify_http_result(400, error_text="content_filter blocked")
    assert content_filtered.classification == OutcomeClass.CONTENT_FILTERED
    assert content_filtered.is_retryable is False

    auth = classify_http_result(401)
    assert auth.classification == OutcomeClass.AUTH_ERROR
    assert auth.is_retryable is False
