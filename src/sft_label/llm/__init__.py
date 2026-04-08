"""Unified LLM communication infrastructure for sft-label.

This package consolidates LLM transport, rate-limiting, adaptive runtime,
and progress tracking that is shared by both Pass 1 (labeling) and
Pass 2 (scoring) pipelines.
"""
from __future__ import annotations

# ─── Runtime (circuit breaker, outcome classification) ────────────
from sft_label.llm.runtime import (  # noqa: F401
    AdaptiveLLMRuntime,
    AdaptiveRateLimiter as _AdaptiveRateLimiter_Runtime,
    DynamicConcurrencyGate,
    FatalFailureMonitor,
    OutcomeClass,
    PipelineAbortError,
    RequestOutcome,
    RuntimePermit,
    classify_exception,
    classify_http_result,
    monitor_to_outcome_class,
)

# ─── Transport (HTTP calls, rate limiting) ────────────────────────
from sft_label.llm.transport import (  # noqa: F401
    AsyncRateLimiter,
    RequestStats,
    async_llm_call,
)

# ─── Progress (ETA, display helpers) ─────────────────────────────
from sft_label.llm.progress import (  # noqa: F401
    RuntimeEtaEstimator,
    _decorate_run_info_with_runtime,
    _format_runtime_progress,
    _update_llm_task_progress,
    format_progress_info,
    parse_run_progress,
)

# ─── Factory (runtime construction) ──────────────────────────────
from sft_label.llm.factory import (  # noqa: F401
    _build_adaptive_runtime,
    _config_adaptive_runtime,
    _make_recovery_config,
    _outcome_from_usage,
)

# ─── HTTP helpers (connection pool sizing) ────────────────────────
from sft_label.llm.http import (  # noqa: F401
    _build_http_client_limits,
    _chunked_output_fd_budget,
)
