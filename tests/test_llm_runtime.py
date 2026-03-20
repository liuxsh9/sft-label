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
async def test_adaptive_rate_limiter_clips_burst_when_target_rps_is_reduced():
    limiter = AdaptiveRateLimiter(target_rps=20.0, burst=20, initial_tokens=20.0)

    limiter.set_target_rps(2.0)

    t0 = time.perf_counter()
    await limiter.acquire()
    first_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    await limiter.acquire()
    second_elapsed = time.perf_counter() - t1

    t2 = time.perf_counter()
    await limiter.acquire()
    third_elapsed = time.perf_counter() - t2

    assert first_elapsed < 0.05
    assert second_elapsed < 0.05
    assert third_elapsed >= 0.35


def test_adaptive_runtime_observation_floor_prevents_early_degrade_and_open():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=8,
        base_rps=20.0,
        min_observations_degraded=3,
        min_observations_open=5,
        min_failures_degraded=2,
        min_failures_open=2,
        degrade_timeout_rate=0.2,
        open_timeout_rate=0.2,
        degrade_overload_rate=0.2,
        open_overload_rate=0.2,
        degrade_abnormal_rate=0.2,
        open_abnormal_rate=0.2,
    )

    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    snap = runtime.snapshot()
    assert snap["state"] == "healthy"
    assert snap["effective_concurrency"] == 8
    assert snap["effective_rps"] == 20.0
    assert snap["rates"]["timeout_count"] == 1

    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    snap = runtime.snapshot()
    assert snap["state"] == "healthy"

    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    snap = runtime.snapshot()
    assert snap["state"] == "degraded"

    runtime.observe(classify_http_result(429))
    assert runtime.snapshot()["state"] == "degraded"

    runtime.observe(classify_http_result(429))
    assert runtime.snapshot()["state"] == "open"


def test_adaptive_runtime_can_open_on_abnormal_response():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=4,
        base_rps=10.0,
        min_observations_degraded=3,
        min_observations_open=4,
        min_failures_degraded=2,
        min_failures_open=2,
        degrade_abnormal_rate=0.25,
        open_abnormal_rate=0.5,
        open_base_cooldown=0.01,
        open_max_cooldown=0.02,
    )

    runtime.observe(classify_http_result(200))
    runtime.observe(classify_http_result(200, parse_error=True))
    runtime.observe(classify_http_result(200, parse_error=True))
    assert runtime.snapshot()["state"] == "degraded"

    runtime.observe(classify_http_result(200, parse_error=True))
    assert runtime.snapshot()["state"] == "open"


@pytest.mark.asyncio
async def test_adaptive_runtime_probing_non_infra_outcome_falls_back_to_degraded():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=4,
        base_rps=10.0,
        min_concurrency=1,
        min_rps=0.5,
        min_observations_degraded=1,
        min_observations_open=1,
        min_failures_degraded=1,
        min_failures_open=1,
        open_timeout_rate=0.5,
        open_base_cooldown=0.01,
        open_max_cooldown=0.02,
        probe_success_required=2,
    )

    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    assert runtime.state == "open"

    await asyncio.sleep(0.02)
    permit = await runtime.acquire(stage="pass1", sample_id="s-1")
    permit.release()
    assert runtime.state == "probing"

    runtime.observe(classify_http_result(400, error_text="content_filter blocked"))
    snap = runtime.snapshot()
    assert snap["state"] == "degraded"
    assert snap["effective_concurrency"] == 2
    assert snap["effective_rps"] == 6.0


@pytest.mark.asyncio
async def test_adaptive_runtime_probe_fallback_does_not_immediately_reopen_from_stale_window():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=4,
        base_rps=10.0,
        min_concurrency=1,
        min_rps=0.5,
        min_observations_degraded=1,
        min_observations_open=2,
        min_failures_degraded=1,
        min_failures_open=1,
        open_timeout_rate=2.0,
        open_overload_rate=0.5,
        open_base_cooldown=0.01,
        open_max_cooldown=0.02,
        probe_success_required=2,
    )

    runtime.observe(classify_http_result(429))
    runtime.observe(classify_http_result(429))
    assert runtime.state == "open"

    await asyncio.sleep(0.02)
    permit = await runtime.acquire(stage="pass1", sample_id="s-2")
    permit.release()
    assert runtime.state == "probing"

    runtime.observe(classify_http_result(400, error_text="content_filter blocked"))
    assert runtime.state == "degraded"

    runtime.observe(classify_http_result(200))
    assert runtime.state == "degraded"


@pytest.mark.asyncio
async def test_adaptive_runtime_state_transitions_and_snapshot():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=8,
        base_rps=20.0,
        min_concurrency=1,
        min_rps=0.5,
        min_observations_degraded=3,
        min_observations_open=5,
        min_failures_degraded=2,
        min_failures_open=2,
        window_requests=6,
        window_seconds=10.0,
        degrade_timeout_rate=0.2,
        open_timeout_rate=0.75,
        degrade_overload_rate=0.2,
        open_overload_rate=0.6,
        degrade_abnormal_rate=0.2,
        open_abnormal_rate=0.5,
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

    for _ in range(11):
        runtime.observe(classify_http_result(200))
    snap = runtime.snapshot()
    assert snap["state"] == "healthy"
    assert snap["base_concurrency"] == 8
    assert snap["effective_concurrency"] == 8
    assert snap["effective_rps"] == 20.0
    assert "rates" in snap


@pytest.mark.asyncio
async def test_adaptive_runtime_eventually_recovers_to_full_speed_after_open_state():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=10,
        base_rps=24.0,
        min_concurrency=1,
        min_rps=1.0,
        min_observations_degraded=2,
        min_observations_open=3,
        min_failures_degraded=2,
        min_failures_open=2,
        window_requests=8,
        window_seconds=10.0,
        degrade_timeout_rate=0.2,
        open_timeout_rate=0.5,
        degrade_overload_rate=0.2,
        open_overload_rate=0.5,
        degrade_abnormal_rate=0.2,
        open_abnormal_rate=0.75,
        degrade_concurrency_factor=0.4,
        degrade_rps_factor=0.5,
        recovery_concurrency_step=3,
        recovery_rps_step=4.0,
        open_base_cooldown=0.02,
        open_max_cooldown=0.05,
        probe_success_required=2,
    )

    runtime.observe(classify_http_result(429))
    runtime.observe(classify_http_result(429))
    runtime.observe(classify_http_result(503))
    assert runtime.state == "open"
    assert runtime.snapshot()["effective_concurrency"] == 1
    assert runtime.snapshot()["effective_rps"] == 1.0

    await asyncio.sleep(0.03)
    permit = await runtime.acquire(stage="pass1", sample_id="recover-full")
    permit.release()
    assert runtime.state == "probing"

    runtime.observe(classify_http_result(200))
    assert runtime.state == "probing"
    runtime.observe(classify_http_result(200))
    snap = runtime.snapshot()
    assert snap["state"] == "degraded"
    assert snap["effective_concurrency"] == 4
    assert snap["effective_rps"] == 12.0

    # Sustained healthy observations should eventually restore base limits and healthy state.
    for _ in range(10):
        runtime.observe(classify_http_result(200))

    snap = runtime.snapshot()
    assert snap["state"] == "healthy"
    assert snap["effective_concurrency"] == runtime.base_concurrency == 10
    assert snap["effective_rps"] == runtime.base_rps == 24.0
    assert runtime.gate.limit == runtime.base_concurrency
    assert runtime.rate_limiter.target_rps == runtime.base_rps


@pytest.mark.asyncio
async def test_adaptive_runtime_zero_degrade_thresholds_can_still_recover_to_healthy():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=6,
        base_rps=12.0,
        min_concurrency=1,
        min_rps=1.0,
        min_observations_degraded=1,
        min_observations_open=2,
        min_failures_degraded=1,
        min_failures_open=2,
        degrade_timeout_rate=0.0,
        degrade_overload_rate=0.0,
        degrade_abnormal_rate=0.0,
        open_timeout_rate=1.0,
        open_overload_rate=1.0,
        open_abnormal_rate=1.0,
        recovery_concurrency_step=2,
        recovery_rps_step=2.0,
        open_base_cooldown=0.01,
        open_max_cooldown=0.02,
        probe_success_required=1,
    )

    runtime.observe(classify_http_result(429))
    runtime.observe(classify_http_result(429))
    assert runtime.state == "open"

    await asyncio.sleep(0.02)
    permit = await runtime.acquire(stage="pass1", sample_id="zero-threshold")
    permit.release()
    runtime.observe(classify_http_result(200))
    assert runtime.state == "degraded"

    for _ in range(6):
        runtime.observe(classify_http_result(200))

    snap = runtime.snapshot()
    assert snap["state"] == "healthy"
    assert snap["effective_concurrency"] == runtime.base_concurrency == 6
    assert snap["effective_rps"] == runtime.base_rps == 12.0


def test_adaptive_runtime_default_style_thresholds_do_not_trip_on_single_failure():
    runtime = AdaptiveLLMRuntime(
        base_concurrency=8,
        base_rps=20.0,
        min_observations_degraded=3,
        min_observations_open=5,
        min_failures_degraded=2,
        min_failures_open=2,
        degrade_timeout_rate=0.05,
        open_timeout_rate=0.20,
        degrade_overload_rate=0.05,
        open_overload_rate=0.15,
        degrade_abnormal_rate=0.04,
        open_abnormal_rate=0.60,
    )

    runtime.observe(classify_http_result(200))
    runtime.observe(classify_http_result(200))
    runtime.observe(classify_exception(asyncio.TimeoutError("timeout")))
    assert runtime.state == "healthy"

    runtime = AdaptiveLLMRuntime(
        base_concurrency=8,
        base_rps=20.0,
        min_observations_degraded=3,
        min_observations_open=5,
        min_failures_degraded=2,
        min_failures_open=2,
        degrade_timeout_rate=0.05,
        open_timeout_rate=0.20,
        degrade_overload_rate=0.05,
        open_overload_rate=0.15,
        degrade_abnormal_rate=0.04,
        open_abnormal_rate=0.60,
    )
    for _ in range(4):
        runtime.observe(classify_http_result(200))
    runtime.observe(classify_http_result(429))
    assert runtime.state == "healthy"


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
