from __future__ import annotations

from dataclasses import replace as _dc_replace

from sft_label.config import REQUEST_TIMEOUT
from sft_label.llm.runtime import AdaptiveLLMRuntime, classify_http_result, RequestOutcome, OutcomeClass


def _config_adaptive_runtime(config) -> AdaptiveLLMRuntime | None:
    """Return the shared adaptive runtime attached to this PipelineConfig, if any."""
    if config is None:
        return None
    if not getattr(config, "enable_adaptive_runtime", False):
        return None
    runtime = getattr(config, "_adaptive_runtime", None)
    return runtime if isinstance(runtime, AdaptiveLLMRuntime) else None


def _build_adaptive_runtime(config) -> AdaptiveLLMRuntime:
    """Build an AdaptiveLLMRuntime from config fields.

    Note: When rps_limit is unset/unlimited (<= 0), we still enable the runtime
    and treat the runtime RPS limiter as effectively non-binding until it
    degrades/open-circuits (it will still clamp to min_rps during probe/open).
    """
    base_rps = float(getattr(config, "rps_limit", 0.0) or 0.0)
    if base_rps <= 0:
        # Avoid disabling adaptive behavior entirely. Concurrency is the main
        # pressure valve; RPS becomes relevant in probing/open states.
        base_rps = float(max(getattr(config, "concurrency", 1), 1)) * 1000.0
    return AdaptiveLLMRuntime(
        base_concurrency=int(getattr(config, "concurrency", 1) or 1),
        base_rps=base_rps,
        min_concurrency=int(getattr(config, "adaptive_min_concurrency", 1) or 1),
        min_rps=float(getattr(config, "adaptive_min_rps", 0.5) or 0.5),
        window_requests=int(getattr(config, "adaptive_window_requests", 50) or 50),
        window_seconds=float(getattr(config, "adaptive_window_seconds", 20.0) or 20.0),
        degrade_timeout_rate=float(getattr(config, "adaptive_timeout_rate_degraded", 0.05) or 0.05),
        open_timeout_rate=float(getattr(config, "adaptive_timeout_rate_open", 0.2) or 0.2),
        degrade_overload_rate=float(getattr(config, "adaptive_overload_rate_degraded", 0.05) or 0.05),
        open_overload_rate=float(getattr(config, "adaptive_overload_rate_open", 0.15) or 0.15),
        degrade_abnormal_rate=float(getattr(config, "adaptive_abnormal_rate_degraded", 0.04) or 0.04),
        open_abnormal_rate=float(getattr(config, "adaptive_abnormal_rate_open", 0.60) or 0.60),
        min_observations_degraded=int(getattr(config, "adaptive_min_observations_degraded", 3) or 3),
        min_observations_open=int(getattr(config, "adaptive_min_observations_open", 12) or 12),
        min_failures_degraded=int(getattr(config, "adaptive_min_failures_degraded", 2) or 2),
        min_failures_open=int(getattr(config, "adaptive_min_failures_open", 4) or 4),
        open_base_cooldown=float(getattr(config, "adaptive_open_base_cooldown", 15.0) or 15.0),
        open_max_cooldown=float(getattr(config, "adaptive_open_max_cooldown", 60.0) or 60.0),
        degrade_concurrency_factor=float(getattr(config, "adaptive_degrade_concurrency_factor", 0.5) or 0.5),
        degrade_rps_factor=float(getattr(config, "adaptive_degrade_rps_factor", 0.6) or 0.6),
        recovery_concurrency_step=int(getattr(config, "adaptive_recovery_concurrency_step", 2) or 2),
        recovery_rps_step=float(getattr(config, "adaptive_recovery_rps_step", 1.0) or 1.0),
    )


def _make_recovery_config(config):
    """Build a conservative config for recovery sweeps."""
    if config is None:
        return None
    new_conc = max(1, int(round((config.concurrency or 1) * (config.recovery_sweep_concurrency_factor or 0.25))))
    new_rps = config.rps_limit
    if isinstance(new_rps, (int, float)) and new_rps > 0:
        new_rps = float(new_rps) * float(config.recovery_sweep_rps_factor or 0.25)
        new_rps = max(new_rps, float(getattr(config, "adaptive_min_rps", 0.5) or 0.5))
    new_timeout = int(round((config.request_timeout or REQUEST_TIMEOUT) * float(config.recovery_sweep_timeout_multiplier or 1.5)))
    recovery_config = _dc_replace(
        config,
        concurrency=new_conc,
        request_timeout=new_timeout,
        rps_limit=new_rps,
    )
    # Ensure recovery does not run arbitration unless explicitly allowed.
    if getattr(config, "recovery_sweep_disable_arbitration", True):
        setattr(recovery_config, "_force_disable_arbitration", True)
    return recovery_config


def _outcome_from_usage(usage: dict, *, default_status: int | None = None) -> RequestOutcome:
    status_code = usage.get("status_code", default_status)
    error_text = usage.get("error_response") or usage.get("error") or ""
    if status_code is not None:
        return classify_http_result(
            int(status_code),
            error_text=error_text,
            parse_error=bool(usage.get("parse_error")),
            validation_error=bool(usage.get("validation_error")),
            content_filtered=bool(usage.get("content_filtered")),
        )
    exc_type = str(usage.get("exception_type") or "")
    err = str(usage.get("error") or "")
    low = f"{exc_type} {err}".lower()
    if "timeout" in low:
        return RequestOutcome(classification=OutcomeClass.TIMEOUT, error=err)
    return RequestOutcome(classification=OutcomeClass.TRANSIENT_ERROR, error=err)
