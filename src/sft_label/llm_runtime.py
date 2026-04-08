"""Backward compatibility — module moved to sft_label.llm.runtime."""
from __future__ import annotations

from sft_label.llm.runtime import *  # noqa: F401,F403
from sft_label.llm.runtime import (  # explicit re-exports for type checkers
    AdaptiveLLMRuntime,
    AdaptiveRateLimiter,
    DynamicConcurrencyGate,
    FatalFailureMonitor,
    OutcomeClass,
    PipelineAbortError,
    RequestOutcome,
    RuntimePermit,
    _GatePermit,
    classify_exception,
    classify_http_result,
    monitor_to_outcome_class,
)
