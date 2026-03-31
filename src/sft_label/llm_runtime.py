from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class OutcomeClass(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    OVERLOAD = "overload"
    SERVER_ERROR = "server_error"
    ABNORMAL_RESPONSE = "abnormal_response"
    CONTENT_FILTERED = "content_filtered"
    AUTH_ERROR = "auth_error"
    INPUT_ERROR = "input_error"
    TRANSIENT_ERROR = "transient_error"


@dataclass(slots=True)
class RequestOutcome:
    classification: OutcomeClass
    status_code: int | None = None
    error: str = ""
    latency_ms: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.classification is OutcomeClass.SUCCESS

    @property
    def is_retryable(self) -> bool:
        return self.classification in {
            OutcomeClass.TIMEOUT,
            OutcomeClass.OVERLOAD,
            OutcomeClass.SERVER_ERROR,
            OutcomeClass.ABNORMAL_RESPONSE,
            OutcomeClass.TRANSIENT_ERROR,
        }

    @property
    def is_infra_failure(self) -> bool:
        return self.classification in {
            OutcomeClass.TIMEOUT,
            OutcomeClass.OVERLOAD,
            OutcomeClass.SERVER_ERROR,
            OutcomeClass.ABNORMAL_RESPONSE,
            OutcomeClass.TRANSIENT_ERROR,
        }

    @property
    def is_overload(self) -> bool:
        return self.classification in {OutcomeClass.OVERLOAD, OutcomeClass.SERVER_ERROR}

    @property
    def is_abnormal(self) -> bool:
        return self.classification is OutcomeClass.ABNORMAL_RESPONSE


_CONTENT_FILTER_KEYWORDS = (
    "content_filter",
    "content policy",
    "content_policy",
    "responsibleaipolicyviolation",
    "content_management_policy",
    "moderation",
)

_INPUT_ERROR_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "invalid_request_error",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def classify_http_result(
    status_code: int,
    *,
    error_text: str = "",
    parse_error: bool = False,
    validation_error: bool = False,
    content_filtered: bool = False,
    latency_ms: float | None = None,
) -> RequestOutcome:
    if 200 <= status_code < 300:
        if parse_error or validation_error:
            return RequestOutcome(
                classification=OutcomeClass.ABNORMAL_RESPONSE,
                status_code=status_code,
                error=error_text,
                latency_ms=latency_ms,
            )
        return RequestOutcome(
            classification=OutcomeClass.SUCCESS,
            status_code=status_code,
            latency_ms=latency_ms,
        )

    if status_code == 401:
        return RequestOutcome(
            classification=OutcomeClass.AUTH_ERROR,
            status_code=status_code,
            error=error_text,
            latency_ms=latency_ms,
        )

    if status_code == 429:
        return RequestOutcome(
            classification=OutcomeClass.OVERLOAD,
            status_code=status_code,
            error=error_text,
            latency_ms=latency_ms,
        )

    if status_code in (500, 502, 503, 504):
        return RequestOutcome(
            classification=OutcomeClass.SERVER_ERROR,
            status_code=status_code,
            error=error_text,
            latency_ms=latency_ms,
        )

    if status_code == 400:
        if content_filtered or _contains_any(error_text, _CONTENT_FILTER_KEYWORDS):
            return RequestOutcome(
                classification=OutcomeClass.CONTENT_FILTERED,
                status_code=status_code,
                error=error_text,
                latency_ms=latency_ms,
            )
        if _contains_any(error_text, _INPUT_ERROR_KEYWORDS):
            return RequestOutcome(
                classification=OutcomeClass.INPUT_ERROR,
                status_code=status_code,
                error=error_text,
                latency_ms=latency_ms,
            )
        return RequestOutcome(
            classification=OutcomeClass.TRANSIENT_ERROR,
            status_code=status_code,
            error=error_text,
            latency_ms=latency_ms,
        )

    if status_code == 403:
        return RequestOutcome(
            classification=OutcomeClass.TRANSIENT_ERROR,
            status_code=status_code,
            error=error_text,
            latency_ms=latency_ms,
        )

    if 400 <= status_code < 500:
        return RequestOutcome(
            classification=OutcomeClass.INPUT_ERROR,
            status_code=status_code,
            error=error_text,
            latency_ms=latency_ms,
        )

    return RequestOutcome(
        classification=OutcomeClass.TRANSIENT_ERROR,
        status_code=status_code,
        error=error_text,
        latency_ms=latency_ms,
    )


def classify_exception(exc: Exception, *, latency_ms: float | None = None) -> RequestOutcome:
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, asyncio.TimeoutError):
        return RequestOutcome(
            classification=OutcomeClass.TIMEOUT,
            error=message,
            latency_ms=latency_ms,
        )
    return RequestOutcome(
        classification=OutcomeClass.TRANSIENT_ERROR,
        error=message,
        latency_ms=latency_ms,
    )


class _GatePermit:
    __slots__ = ("_gate", "_released")

    def __init__(self, gate: DynamicConcurrencyGate):
        self._gate = gate
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._gate.release()

    async def __aenter__(self) -> _GatePermit:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


class DynamicConcurrencyGate:
    def __init__(self, initial_limit: int, min_limit: int = 1, max_limit: int | None = None):
        if initial_limit <= 0:
            raise ValueError("initial_limit must be > 0")
        self._min_limit = max(min_limit, 1)
        self._max_limit = max_limit if max_limit is not None else max(initial_limit, self._min_limit)
        self._limit = max(self._min_limit, min(initial_limit, self._max_limit))
        self._in_flight = 0
        self._paused = False
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def paused(self) -> bool:
        return self._paused

    def set_limit(self, new_limit: int) -> None:
        clipped = max(self._min_limit, min(int(new_limit), self._max_limit))
        self._limit = clipped
        self._notify_waiters()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._notify_waiters()

    async def acquire(self) -> _GatePermit:
        async with self._cond:
            while self._paused or self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1
        return _GatePermit(self)

    def release(self) -> None:
        async def _release() -> None:
            async with self._cond:
                self._in_flight = max(0, self._in_flight - 1)
                self._cond.notify_all()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._in_flight = max(0, self._in_flight - 1)
            return
        loop.create_task(_release())

    def _notify_waiters(self) -> None:
        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(_notify())


class AdaptiveRateLimiter:
    def __init__(
        self,
        target_rps: float,
        *,
        burst: int | None = None,
        initial_tokens: float = 1.0,
    ):
        if target_rps <= 0:
            raise ValueError("target_rps must be > 0")
        self._target_rps = float(target_rps)
        self._burst_cap = burst if burst is not None else max(int(math.ceil(target_rps)), 1)
        self._burst = self._burst_cap
        self._tokens = max(0.0, min(float(initial_tokens), float(self._burst)))
        self._last_refill = time.monotonic()
        self._pause_until = 0.0
        self._lock = asyncio.Lock()

    @property
    def target_rps(self) -> float:
        return self._target_rps

    @property
    def pause_until(self) -> float:
        return self._pause_until

    def set_target_rps(self, target_rps: float) -> None:
        if target_rps <= 0:
            raise ValueError("target_rps must be > 0")
        self._target_rps = float(target_rps)
        self._burst = max(1, min(self._burst_cap, int(math.ceil(self._target_rps))))
        self._tokens = min(self._tokens, float(self._burst))

    def pause_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._pause_until = max(self._pause_until, time.monotonic() + seconds)

    async def acquire(self, *, probe: bool = False) -> None:
        del probe
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._pause_until:
                    wait_for = self._pause_until - now
                else:
                    elapsed = max(0.0, now - self._last_refill)
                    self._tokens = min(self._burst, self._tokens + elapsed * self._target_rps)
                    self._last_refill = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait_for = (1.0 - self._tokens) / self._target_rps
            await asyncio.sleep(max(wait_for, 0.0))


@dataclass(slots=True)
class RuntimePermit:
    runtime: AdaptiveLLMRuntime
    gate_permit: _GatePermit
    stage: str
    sample_id: str
    queue_wait_ms: float
    state_at_acquire: str
    snapshot_at_acquire: dict
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.gate_permit.release()

    async def __aenter__(self) -> RuntimePermit:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


class AdaptiveLLMRuntime:
    def __init__(
        self,
        *,
        base_concurrency: int,
        base_rps: float,
        min_concurrency: int = 1,
        min_rps: float = 0.5,
        window_requests: int = 50,
        window_seconds: float = 20.0,
        degrade_timeout_rate: float = 0.05,
        open_timeout_rate: float = 0.20,
        degrade_overload_rate: float = 0.05,
        open_overload_rate: float = 0.15,
        degrade_abnormal_rate: float = 0.04,
        open_abnormal_rate: float = 0.60,
        min_observations_degraded: int = 3,
        min_observations_open: int = 12,
        min_failures_degraded: int = 2,
        min_failures_open: int = 4,
        open_base_cooldown: float = 15.0,
        open_max_cooldown: float = 60.0,
        degrade_concurrency_factor: float = 0.5,
        degrade_rps_factor: float = 0.6,
        recovery_concurrency_step: int = 2,
        recovery_rps_step: float = 1.0,
        probe_success_required: int = 3,
    ):
        if base_concurrency <= 0:
            raise ValueError("base_concurrency must be > 0")
        if base_rps <= 0:
            raise ValueError("base_rps must be > 0")

        self.base_concurrency = int(base_concurrency)
        self.base_rps = float(base_rps)
        self.min_concurrency = max(int(min_concurrency), 1)
        self.min_rps = max(float(min_rps), 0.01)

        self.window_requests = max(int(window_requests), 1)
        self.window_seconds = max(float(window_seconds), 1.0)
        self.degrade_timeout_rate = max(float(degrade_timeout_rate), 0.0)
        self.open_timeout_rate = max(float(open_timeout_rate), 0.0)
        self.degrade_overload_rate = max(float(degrade_overload_rate), 0.0)
        self.open_overload_rate = max(float(open_overload_rate), 0.0)
        self.degrade_abnormal_rate = max(float(degrade_abnormal_rate), 0.0)
        self.open_abnormal_rate = max(float(open_abnormal_rate), 0.0)
        self.min_observations_degraded = max(int(min_observations_degraded), 1)
        self.min_observations_open = max(int(min_observations_open), self.min_observations_degraded)
        self.min_failures_degraded = max(int(min_failures_degraded), 1)
        self.min_failures_open = max(int(min_failures_open), self.min_failures_degraded)
        self.open_base_cooldown = max(float(open_base_cooldown), 0.0)
        self.open_max_cooldown = max(float(open_max_cooldown), self.open_base_cooldown)
        self.degrade_concurrency_factor = max(float(degrade_concurrency_factor), 0.01)
        self.degrade_rps_factor = max(float(degrade_rps_factor), 0.01)
        self.recovery_concurrency_step = max(int(recovery_concurrency_step), 1)
        self.recovery_rps_step = max(float(recovery_rps_step), 0.01)
        self.probe_success_required = max(int(probe_success_required), 1)

        self.state = "healthy"
        self._cooldown_until = 0.0
        self._open_streak = 0
        self._probe_successes = 0
        self._healthy_streak = 0
        self._effective_concurrency = self.base_concurrency
        self._effective_rps = self.base_rps

        self.gate = DynamicConcurrencyGate(
            initial_limit=self.base_concurrency,
            min_limit=self.min_concurrency,
            max_limit=self.base_concurrency,
        )
        self.rate_limiter = AdaptiveRateLimiter(
            target_rps=self.base_rps,
            burst=max(int(math.ceil(self.base_rps)), 1),
            initial_tokens=1.0,
        )
        self._events: deque[dict] = deque(maxlen=500)
        self._window: deque[tuple[float, RequestOutcome]] = deque()

    async def acquire(self, *, stage: str = "", sample_id: str = "", probe: bool = False) -> RuntimePermit:
        start = time.perf_counter()
        while True:
            local_probe = probe
            if self.state == "open":
                now = time.monotonic()
                if now >= self._cooldown_until:
                    self._enter_probing(reason="cooldown_elapsed")
                else:
                    local_probe = True

            gate_permit = await self.gate.acquire()
            try:
                await self.rate_limiter.acquire(probe=local_probe or self.state == "probing")
            except Exception:
                gate_permit.release()
                raise

            if self.state == "open" and time.monotonic() >= self._cooldown_until:
                gate_permit.release()
                self._enter_probing(reason="cooldown_elapsed")
                continue

            wait_ms = (time.perf_counter() - start) * 1000.0
            snap = self.snapshot()
            return RuntimePermit(
                runtime=self,
                gate_permit=gate_permit,
                stage=stage,
                sample_id=sample_id,
                queue_wait_ms=wait_ms,
                state_at_acquire=self.state,
                snapshot_at_acquire=snap,
            )

    def observe(self, outcome: RequestOutcome) -> None:
        now = time.monotonic()
        self._window.append((now, outcome))
        self._prune_window(now)
        rates = self._window_rates()

        if self.state == "probing":
            if outcome.is_success:
                self._probe_successes += 1
                if self._probe_successes >= self.probe_success_required:
                    self._enter_degraded(reason="probe_success")
            elif outcome.is_infra_failure:
                self._enter_open(reason="probe_failure")
            else:
                self._enter_degraded(reason="probe_terminal_outcome")
            return

        if self.state in {"healthy", "degraded"}:
            should_open = (
                rates["total"] >= self.min_observations_open
                and (
                    (
                        rates["timeout_count"] >= self.min_failures_open
                        and rates["timeout_rate"] >= self.open_timeout_rate
                    )
                    or (
                        rates["overload_count"] >= self.min_failures_open
                        and rates["overload_rate"] >= self.open_overload_rate
                    )
                    or (
                        rates["abnormal_count"] >= self.min_failures_open
                        and rates["abnormal_rate"] >= self.open_abnormal_rate
                    )
                )
            )
            if should_open:
                self._enter_open(reason="open_threshold")
                return

        if self.state == "healthy":
            should_degrade = (
                rates["total"] >= self.min_observations_degraded
                and (
                    (
                        rates["timeout_count"] >= self.min_failures_degraded
                        and rates["timeout_rate"] >= self.degrade_timeout_rate
                    )
                    or (
                        rates["overload_count"] >= self.min_failures_degraded
                        and rates["overload_rate"] >= self.degrade_overload_rate
                    )
                    or (
                        rates["abnormal_count"] >= self.min_failures_degraded
                        and rates["abnormal_rate"] >= self.degrade_abnormal_rate
                    )
                )
            )
            if should_degrade:
                self._enter_degraded(reason="degrade_threshold")
            return

        if self.state == "degraded":
            is_healthy_window = (
                rates["timeout_rate"] <= self.degrade_timeout_rate * 0.5
                and rates["overload_rate"] <= self.degrade_overload_rate * 0.5
                and rates["abnormal_rate"] <= self.degrade_abnormal_rate * 0.5
            )
            if is_healthy_window:
                self._healthy_streak += 1
                self._increase_limits()
                if (
                    self._healthy_streak >= 3
                    and self._effective_concurrency >= self.base_concurrency
                    and self._effective_rps >= self.base_rps
                ):
                    self._enter_healthy(reason="recovered")
            else:
                self._healthy_streak = 0

    def snapshot(self) -> dict:
        rates = self._window_rates()
        cooldown_remaining = max(self._cooldown_until - time.monotonic(), 0.0)
        return {
            "state": self.state,
            "base_concurrency": self.base_concurrency,
            "base_rps": self.base_rps,
            "effective_concurrency": self._effective_concurrency,
            "effective_rps": self._effective_rps,
            "in_flight": self.gate.in_flight,
            "gate_paused": self.gate.paused,
            "cooldown_until": self._cooldown_until,
            "cooldown_remaining_seconds": cooldown_remaining,
            "rates": rates,
            "window_size": len(self._window),
            "events_recorded": len(self._events),
        }

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def _prune_window(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        while len(self._window) > self.window_requests:
            self._window.popleft()

    def _window_rates(self) -> dict:
        total = len(self._window)
        if total == 0:
            return {
                "total": 0,
                "timeout_rate": 0.0,
                "timeout_count": 0,
                "overload_rate": 0.0,
                "overload_count": 0,
                "abnormal_rate": 0.0,
                "abnormal_count": 0,
                "infra_failure_rate": 0.0,
            }

        timeouts = 0
        overloads = 0
        abnormals = 0
        infra_failures = 0
        for _, outcome in self._window:
            if outcome.classification is OutcomeClass.TIMEOUT:
                timeouts += 1
            if outcome.classification in {OutcomeClass.OVERLOAD, OutcomeClass.SERVER_ERROR}:
                overloads += 1
            if outcome.classification is OutcomeClass.ABNORMAL_RESPONSE:
                abnormals += 1
            if outcome.is_infra_failure:
                infra_failures += 1

        return {
            "total": total,
            "timeout_rate": timeouts / total,
            "timeout_count": timeouts,
            "overload_rate": overloads / total,
            "overload_count": overloads,
            "abnormal_rate": abnormals / total,
            "abnormal_count": abnormals,
            "infra_failure_rate": infra_failures / total,
        }

    def _enter_healthy(self, *, reason: str) -> None:
        self.state = "healthy"
        self._probe_successes = 0
        self._healthy_streak = 0
        self._open_streak = 0
        self.gate.resume()
        self._set_effective_limits(self.base_concurrency, self.base_rps)
        self._record_event("state", reason, {"state": self.state})

    def _enter_degraded(self, *, reason: str) -> None:
        self.state = "degraded"
        self._probe_successes = 0
        self._healthy_streak = 0
        self.gate.resume()
        degraded_concurrency = max(
            self.min_concurrency,
            int(math.floor(self.base_concurrency * self.degrade_concurrency_factor)),
        )
        degraded_rps = max(self.min_rps, self.base_rps * self.degrade_rps_factor)
        self._set_effective_limits(degraded_concurrency, degraded_rps)
        self._record_event("state", reason, {"state": self.state})

    def _enter_open(self, *, reason: str) -> None:
        self.state = "open"
        self._probe_successes = 0
        self._healthy_streak = 0
        self._open_streak += 1
        cooldown = min(self.open_base_cooldown * (2 ** (self._open_streak - 1)), self.open_max_cooldown)
        self._cooldown_until = time.monotonic() + cooldown
        self.gate.resume()
        self._set_effective_limits(self.min_concurrency, self.min_rps)
        self._record_event("state", reason, {"state": self.state, "cooldown_seconds": cooldown})

    def _enter_probing(self, *, reason: str) -> None:
        self.state = "probing"
        self._probe_successes = 0
        self._healthy_streak = 0
        self._window.clear()
        self.gate.resume()
        self._set_effective_limits(self.min_concurrency, self.min_rps)
        self._record_event("state", reason, {"state": self.state})

    def _increase_limits(self) -> None:
        next_concurrency = min(
            self.base_concurrency,
            self._effective_concurrency + self.recovery_concurrency_step,
        )
        next_rps = min(self.base_rps, self._effective_rps + self.recovery_rps_step)
        self._set_effective_limits(next_concurrency, next_rps)

    def _set_effective_limits(self, concurrency: int, rps: float) -> None:
        self._effective_concurrency = int(max(self.min_concurrency, min(concurrency, self.base_concurrency)))
        self._effective_rps = float(max(self.min_rps, min(rps, self.base_rps)))
        # Never set gate limit below current in-flight count: doing so creates
        # a deadly embrace where the gate blocks ALL new acquisitions because
        # in_flight >= limit, but existing in-flight requests cannot complete
        # because the API may be down (which is what triggered the open state).
        # Use in_flight + 1 so at least one new probe request can enter.
        # The gate will naturally tighten as in-flight requests drain.
        gate_limit = max(self._effective_concurrency, self.gate.in_flight + 1)
        self.gate.set_limit(gate_limit)
        self.rate_limiter.set_target_rps(self._effective_rps)

    def _record_event(self, kind: str, reason: str, payload: dict) -> None:
        self._events.append(
            {
                "ts": time.time(),
                "kind": kind,
                "reason": reason,
                "state": self.state,
                "effective_concurrency": self._effective_concurrency,
                "effective_rps": self._effective_rps,
                "payload": payload,
            }
        )
