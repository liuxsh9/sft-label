"""
Value Scoring Pipeline (Pass 2)

Evaluates SFT training data value through:
  1. Rarity computation — tag IDF + combo rarity from existing tag distributions
  2. LLM-based scoring — complexity, quality, reasoning assessment
  3. Weighted aggregation — composite value_score

Runs after Pass 1 (tag labeling) and produces scored.json, stats_scoring.json,
dashboard_scoring.html per file.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import asyncio
import random
import re
import inspect
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import httpx

from sft_label.config import (
    SAMPLE_MAX_RETRIES, VALUE_WEIGHTS, RARITY_WEIGHTS, RARITY_COMBO_ALPHA, RARITY_SCORE_MODE,
    VALUE_TRUNCATION_BUDGET, COMPACT_VALUE_TRUNCATION_BUDGET,
    KNOWN_FLAGS, CHUNK_SIZE, MAX_ACTIVE_CHUNKS,
    SELECTION_INTRA_WEIGHT, SELECTION_QUALITY_WEIGHT, SELECTION_RARITY_WEIGHT,
    SELECTION_MIN_GROUP_SIZE, SELECTION_SMOOTHING_PRIOR,
    ENABLE_VALUE_STABILITY, ENABLE_SELECTION_STABILITY, ENABLE_DOMAIN_BACKFILL,
    SELECTION_STAGE_VALUE_MULTIPLIERS, SELECTION_STAGE_SELECTION_MULTIPLIERS,
    SELECTION_LOW_INFO_TOOL_PENALTY, SELECTION_SUMMARY_NO_EVIDENCE_PENALTY,
    SELECTION_SUMMARY_EVIDENCE_BONUS,
    ENABLE_SELECTIVE_SCORING, SELECTIVE_SCORING_POLICY, SELECTIVE_SCORING_MIN_TURNS,
    SELECTIVE_SCORING_DRIFT_INTERVAL, SELECTIVE_SCORING_ESTIMATE_CONFIDENCE_CAP,
    FILE_RANKING_KEEP_RATE_THRESHOLD, FILE_RANKING_KEEP_RATE_THRESHOLDS,
    PipelineConfig,
)
from sft_label.preprocessing import (
    detect_thinking_mode, extract_cot_content, truncate_for_scoring,
    count_code_blocks,
)
from sft_label.pipeline import (
    async_llm_call,
    AsyncRateLimiter,
    RuntimeEtaEstimator,
    format_progress_info,
    parse_run_progress,
)
from sft_label.progress_heartbeat import run_with_heartbeat
from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
    PASS2_DASHBOARD_FILE,
    pass2_global_dashboard_filename,
)
from sft_label.inline_scoring import (
    InlineScoringTarget,
    discover_inline_jsonl_files,
    infer_inline_scoring_target,
    load_inline_scoring_file,
    update_inline_row_with_scored_samples,
    write_inline_labeled_cache,
)
from sft_label.labels import is_partial_labels, is_usable_labels
from sft_label.label_extensions_stats import aggregate_extension_stats
from sft_label.progress_display import create_pipeline_progress
from sft_label.score_confidence import apply_score_confidence, score_confidence

try:
    from sft_label.llm_runtime import (
        AdaptiveLLMRuntime,
        OutcomeClass,
        RequestOutcome,
        classify_exception,
        classify_http_result,
    )
except Exception:  # pragma: no cover - runtime module lands in parallel task
    AdaptiveLLMRuntime = None
    OutcomeClass = None
    RequestOutcome = None
    classify_exception = None
    classify_http_result = None


DIRECTORY_JSON_STREAMING_THRESHOLD_BYTES = 64 * 1024 * 1024


# ─────────────────────────────────────────────────────────
# Rarity computation
# ─────────────────────────────────────────────────────────


def _cfg_bool(config, key, default):
    if config is None:
        return default
    value = getattr(config, key, default)
    if isinstance(value, bool):
        return value
    return default


def _cfg_number(config, key, default):
    if config is None:
        return default
    value = getattr(config, key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value
    return default


def _instantiate_runtime(config=None, *, concurrency=1) -> AdaptiveLLMRuntime | None:
    """Construct the shared adaptive runtime (or return None when disabled/unavailable)."""
    if AdaptiveLLMRuntime is None:
        return None
    if not _cfg_bool(config, "enable_adaptive_runtime", True):
        return None
    base_concurrency = max(int(concurrency), 1)
    # rps_limit=0 means "unlimited" in the existing config; approximate it with a high ceiling
    # so concurrency becomes the primary limiter.
    base_rps = float(_cfg_number(config, "rps_limit", 0.0))
    if base_rps <= 0:
        base_rps = float(max(50, base_concurrency * 10))
    return AdaptiveLLMRuntime(
        base_concurrency=base_concurrency,
        base_rps=base_rps,
        min_concurrency=max(int(_cfg_number(config, "adaptive_min_concurrency", 1)), 1),
        min_rps=float(_cfg_number(config, "adaptive_min_rps", 0.5)),
        window_requests=int(_cfg_number(config, "adaptive_window_requests", 50)),
        window_seconds=float(_cfg_number(config, "adaptive_window_seconds", 20.0)),
        degrade_timeout_rate=float(_cfg_number(config, "adaptive_timeout_rate_degraded", 0.05)),
        open_timeout_rate=float(_cfg_number(config, "adaptive_timeout_rate_open", 0.20)),
        degrade_overload_rate=float(_cfg_number(config, "adaptive_overload_rate_degraded", 0.05)),
        open_overload_rate=float(_cfg_number(config, "adaptive_overload_rate_open", 0.15)),
        degrade_abnormal_rate=float(_cfg_number(config, "adaptive_abnormal_rate_degraded", 0.04)),
        open_abnormal_rate=float(_cfg_number(config, "adaptive_abnormal_rate_open", 0.60)),
        min_observations_degraded=int(_cfg_number(config, "adaptive_min_observations_degraded", 3)),
        min_observations_open=int(_cfg_number(config, "adaptive_min_observations_open", 5)),
        min_failures_degraded=int(_cfg_number(config, "adaptive_min_failures_degraded", 2)),
        min_failures_open=int(_cfg_number(config, "adaptive_min_failures_open", 2)),
        open_base_cooldown=float(_cfg_number(config, "adaptive_open_base_cooldown", 15.0)),
        open_max_cooldown=float(_cfg_number(config, "adaptive_open_max_cooldown", 120.0)),
        degrade_concurrency_factor=float(_cfg_number(config, "adaptive_degrade_concurrency_factor", 0.5)),
        degrade_rps_factor=float(_cfg_number(config, "adaptive_degrade_rps_factor", 0.6)),
        recovery_concurrency_step=int(_cfg_number(config, "adaptive_recovery_concurrency_step", 2)),
        recovery_rps_step=float(_cfg_number(config, "adaptive_recovery_rps_step", 1.0)),
    )


@asynccontextmanager
async def _runtime_permit(runtime, *, stage, sample_id):
    """Acquire/release runtime permit across runtime implementations."""
    permit = None
    if runtime is not None and hasattr(runtime, "acquire"):
        acquired = runtime.acquire(stage=stage, sample_id=sample_id)
        permit = await acquired if inspect.isawaitable(acquired) else acquired
    try:
        yield permit
    finally:
        if permit is not None and hasattr(permit, "release"):
            try:
                released = permit.release()
                if inspect.isawaitable(released):
                    await released
            except Exception:
                pass
        elif runtime is not None and hasattr(runtime, "release"):
            try:
                released = runtime.release(permit=permit)
                if inspect.isawaitable(released):
                    await released
            except TypeError:
                try:
                    released = runtime.release()
                    if inspect.isawaitable(released):
                        await released
                except Exception:
                    pass
            except Exception:
                pass


@asynccontextmanager
async def _borrow_http_client(client):
    yield client


def _notify_runtime(runtime, outcome: RequestOutcome | None) -> None:
    if runtime is None or outcome is None or not hasattr(runtime, "observe"):
        return
    try:
        runtime.observe(outcome)
    except Exception:
        pass


def _runtime_snapshot(runtime):
    if runtime is None:
        return None
    snap = None
    if hasattr(runtime, "snapshot"):
        try:
            snap = runtime.snapshot()
        except Exception:
            snap = None
    elif hasattr(runtime, "to_dict"):
        try:
            snap = runtime.to_dict()
        except Exception:
            snap = None
    if isinstance(snap, dict):
        return snap
    if snap is None:
        return None
    return {"value": str(snap)}


_HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b")


def _extract_http_status(raw: str | None, usage: dict | None) -> int | None:
    usage = usage or {}
    status = usage.get("status_code")
    if isinstance(status, bool):
        status = None
    if isinstance(status, int):
        return status
    if isinstance(status, str):
        try:
            return int(status)
        except ValueError:
            pass
    for text in (
        usage.get("error") or "",
        raw or "",
    ):
        match = _HTTP_STATUS_RE.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _is_deployment_cooldown(raw: str | None, usage: dict | None) -> bool:
    usage = usage or {}
    blob = " ".join(
        str(x or "") for x in (usage.get("error"), usage.get("error_response"), raw)
    ).lower()
    return "deployment cooldown" in blob or "no deployments available" in blob or "cooldown_list" in blob


def _is_timeout_error(raw: str | None, usage: dict | None) -> bool:
    usage = usage or {}
    blob = f"{usage.get('error') or ''} {raw or ''}".lower()
    return any(token in blob for token in ("timeout", "readtimeout", "connecttimeout", "timeouterror"))


def _looks_like_html_response(text: str | None) -> bool:
    blob = str(text or "").lstrip().lower()
    return blob.startswith("<!doctype html") or blob.startswith("<html")


def _summarize_first_failure(monitor: dict | None) -> str | None:
    if not monitor:
        return None
    sample_id = monitor.get("sample_id") or "unknown"
    attempts = monitor.get("attempts")
    err = str(monitor.get("error") or "unknown").strip()
    http_status = monitor.get("http_status") or _extract_http_status(err, monitor)
    parts = [f"  [!] First failure: sample={sample_id}"]
    if attempts not in (None, ""):
        parts.append(f"attempts={attempts}")
    if http_status is not None:
        parts.append(f"http={http_status}")
    if err:
        compact_err = " ".join(err.split())
        parts.append(f"err={compact_err[:200]}")
    if _looks_like_html_response(monitor.get("error_response")):
        parts.append("html_response")
    return " ".join(parts)


def _classify_pass2_attempt(
    parsed,
    raw,
    usage,
    score_result,
    *,
    latency_ms: float | None,
) -> tuple[str, bool, RequestOutcome | None]:
    """Classify Pass 2 attempt result for recovery sweep + runtime feedback."""
    usage = usage or {}
    status_code = _extract_http_status(raw, usage)
    content_filtered = bool(usage.get("content_filtered"))
    error_text = str(usage.get("error") or raw or "")
    parse_error = bool(usage.get("parse_error"))

    # Score schema/parse failures are treated as abnormal responses (health-relevant).
    # NOTE: validate_score_response() always returns a dict, so callers should pass
    # score_result=None for "malformed" / "insufficient usable scores" cases.
    validation_error = parsed is not None and score_result is None

    if classify_http_result is not None and status_code is not None:
        outcome = classify_http_result(
            status_code,
            error_text=error_text,
            parse_error=parse_error,
            validation_error=validation_error,
            content_filtered=content_filtered,
            latency_ms=latency_ms,
        )
    else:
        if parsed is None and parse_error and RequestOutcome is not None:
            outcome = RequestOutcome(
                classification=OutcomeClass.ABNORMAL_RESPONSE,
                status_code=status_code or 200,
                error=error_text,
                latency_ms=latency_ms,
            )
        elif parsed is None and _is_timeout_error(raw, usage) and RequestOutcome is not None:
            outcome = RequestOutcome(
                classification=OutcomeClass.TIMEOUT,
                status_code=status_code,
                error=error_text,
                latency_ms=latency_ms,
            )
        elif validation_error and RequestOutcome is not None:
            outcome = RequestOutcome(
                classification=OutcomeClass.ABNORMAL_RESPONSE,
                status_code=status_code or 200,
                error=error_text,
                latency_ms=latency_ms,
            )
        elif parsed is None and RequestOutcome is not None:
            outcome = RequestOutcome(
                classification=OutcomeClass.TRANSIENT_ERROR,
                status_code=status_code,
                error=error_text,
                latency_ms=latency_ms,
            )
        elif RequestOutcome is not None:
            outcome = RequestOutcome(
                classification=OutcomeClass.SUCCESS,
                status_code=status_code or 200,
                latency_ms=latency_ms,
            )
        else:
            outcome = None

    # Override: pipeline async_llm_call marks deployment cooldown as non-retryable, but it is infra-retryable.
    if _is_deployment_cooldown(raw, usage) and outcome is not None:
        if RequestOutcome is not None:
            outcome = RequestOutcome(
                classification=OutcomeClass.OVERLOAD,
                status_code=outcome.status_code,
                error=error_text,
                latency_ms=latency_ms,
                extra={"deployment_cooldown": True},
            )

    retryable_infra = bool(outcome.is_infra_failure) if outcome is not None else False
    if content_filtered:
        return "content_filtered", False, outcome
    if outcome is not None and outcome.classification == OutcomeClass.AUTH_ERROR:
        return "auth_error", False, outcome
    if outcome is not None and outcome.is_success:
        return "ok", False, outcome
    if outcome is not None and retryable_infra:
        if outcome.classification == OutcomeClass.ABNORMAL_RESPONSE:
            return "abnormal_response", True, outcome
        if outcome.classification == OutcomeClass.TIMEOUT:
            return "timeout", True, outcome
        if outcome.classification in (OutcomeClass.OVERLOAD, OutcomeClass.SERVER_ERROR):
            return "overload", True, outcome
        return "infra_retryable_error", True, outcome
    if usage.get("non_retryable"):
        return "non_retryable_error", False, outcome
    return "unknown_error", True, outcome

def _relative_file_label(path, root):
    """Render a stable file label relative to the batch input root."""
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _should_recovery_retry(monitor: dict | None) -> bool:
    if not monitor:
        return False
    if monitor.get("retryable_infra") is True:
        return True
    # Backward compatible: older monitors may not include retryable_infra yet.
    error_class = str(monitor.get("error_class") or "").strip().lower()
    return error_class in {
        "infra_retryable_error",
        "timeout",
        "overload",
        "abnormal_response",
        "unknown_error",
    }


def _recovery_config(config: PipelineConfig) -> PipelineConfig:
    sweep = copy.copy(config)
    factor = float(getattr(config, "recovery_sweep_concurrency_factor", 0.25) or 0.25)
    rps_factor = float(getattr(config, "recovery_sweep_rps_factor", 0.25) or 0.25)
    sweep.scoring_concurrency = max(1, int(round(config.scoring_concurrency * factor)))
    # rps_limit=0 means unlimited; clamp it during recovery to reduce pressure.
    base_rps = float(getattr(config, "rps_limit", 0.0) or 0.0)
    if base_rps > 0:
        sweep.rps_limit = max(0.5, float(base_rps) * rps_factor)
    else:
        sweep.rps_limit = float(max(1, sweep.scoring_concurrency))
    mult = float(getattr(config, "recovery_sweep_timeout_multiplier", 1.5) or 1.5)
    try:
        sweep.request_timeout = int(round(float(config.request_timeout) * mult))
    except Exception:
        pass
    # Keep sample retries conservative in recovery passes.
    try:
        sweep.sample_max_retries = max(1, min(int(getattr(config, "sample_max_retries", SAMPLE_MAX_RETRIES)), 2))
    except Exception:
        pass
    setattr(sweep, "_recovery_sweep", True)
    return sweep


async def _run_pass2_recovery_sweep_in_memory(
    *,
    http_client: httpx.AsyncClient,
    samples: list,
    rarity_results: list,
    all_values: list,
    all_monitors: list,
    config: PipelineConfig,
) -> dict:
    """Retry infra-retryable failures before finalizing Pass 2 artifacts."""
    if not getattr(config, "enable_stage_recovery_sweep", True):
        return {"enabled": False, "attempted": 0, "recovered": 0}

    failed_indices = [
        i for i, v in enumerate(all_values)
        if v is None and _should_recovery_retry(all_monitors[i] if i < len(all_monitors) else None)
    ]
    if not failed_indices:
        return {"enabled": True, "attempted": 0, "recovered": 0}

    max_passes = int(getattr(config, "recovery_sweep_max_passes", 1) or 1)
    attempted = 0
    recovered = 0

    for _pass in range(max_passes):
        if not failed_indices:
            break

        sweep_config = _recovery_config(config)
        sweep_runtime = _instantiate_runtime(sweep_config, concurrency=sweep_config.scoring_concurrency)
        setattr(sweep_config, "_adaptive_runtime", sweep_runtime)

        sem = asyncio.Semaphore(sweep_config.scoring_concurrency)
        rate_limiter = (
            AsyncRateLimiter(sweep_config.rps_limit, warmup=sweep_config.rps_warmup)
            if sweep_runtime is None and sweep_config.rps_limit > 0
            else None
        )

        async def _retry_one(idx: int):
            nonlocal attempted, recovered
            attempted += 1
            value, monitor = await score_one(
                http_client,
                samples[idx],
                sweep_config.scoring_model,
                rarity_results[idx],
                idx,
                len(samples),
                sem,
                config=sweep_config,
                rate_limiter=rate_limiter,
            )
            if value:
                all_values[idx] = value
                recovered += 1
            all_monitors[idx] = monitor

        tasks = [asyncio.create_task(_retry_one(i)) for i in failed_indices]
        for coro in asyncio.as_completed(tasks):
            await coro

        failed_indices = [
            i for i in failed_indices
            if all_values[i] is None and _should_recovery_retry(all_monitors[i] if i < len(all_monitors) else None)
        ]

    return {"enabled": True, "attempted": attempted, "recovered": recovered}


def _init_monitor_totals():
    return {
        "total_failed": 0,
        "total_estimated": 0,
        "total_llm_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }


def _accumulate_monitor_totals(monitor_totals, monitor):
    if not monitor:
        return
    status = monitor.get("status")
    if status == "failed":
        monitor_totals["total_failed"] += 1
    if status == "estimated_selective":
        monitor_totals["total_estimated"] += 1
    monitor_totals["total_llm_calls"] += monitor.get("llm_calls", 0) or 0
    monitor_totals["total_prompt_tokens"] += monitor.get("prompt_tokens", 0) or 0
    monitor_totals["total_completion_tokens"] += monitor.get("completion_tokens", 0) or 0


def _rebuild_chunked_summaries_and_monitors(scored_path: Path, monitor_path: Path):
    """Rebuild summaries + compact monitor totals after recovery sweep edits."""
    monitor_totals = _init_monitor_totals()
    summaries = []
    with open(scored_path, "r", encoding="utf-8") as fs, open(monitor_path, "r", encoding="utf-8") as fm:
        for line_s, line_m in zip(fs, fm):
            sample = json.loads(line_s)
            monitor = json.loads(line_m)
            _accumulate_monitor_totals(monitor_totals, monitor)
            if sample.get("value"):
                summaries.append(_selection_summary_from_sample(sample))
    return summaries, monitor_totals


async def _run_pass2_recovery_sweep_chunked(
    *,
    output_dir: Path,
    raw_rarities: list,
    config: PipelineConfig,
) -> dict:
    """Retry infra failures for chunked Pass 2 outputs and patch scored/monitor files."""
    if not getattr(config, "enable_stage_recovery_sweep", True):
        return {"enabled": False, "attempted": 0, "recovered": 0}

    scored_path = Path(output_dir) / "scored.jsonl"
    monitor_path = Path(output_dir) / "monitor_value.jsonl"
    if not scored_path.exists() or not monitor_path.exists():
        return {"enabled": True, "attempted": 0, "recovered": 0}

    retry_items: list[tuple[int, dict]] = []
    with open(scored_path, "r", encoding="utf-8") as fs, open(monitor_path, "r", encoding="utf-8") as fm:
        for idx, (line_s, line_m) in enumerate(zip(fs, fm)):
            sample = json.loads(line_s)
            monitor = json.loads(line_m)
            if sample.get("value") is not None:
                continue
            if idx >= len(raw_rarities):
                continue
            if not _should_recovery_retry(monitor):
                continue
            retry_items.append((idx, sample))

    if not retry_items:
        return {"enabled": True, "attempted": 0, "recovered": 0}

    max_passes = int(getattr(config, "recovery_sweep_max_passes", 1) or 1)
    attempted = 0
    recovered = 0

    # Only keep minimal sample payloads in memory during retries.
    for _pass in range(max_passes):
        if not retry_items:
            break

        sweep_config = _recovery_config(config)
        sweep_runtime = _instantiate_runtime(sweep_config, concurrency=sweep_config.scoring_concurrency)
        setattr(sweep_config, "_adaptive_runtime", sweep_runtime)

        sem = asyncio.Semaphore(sweep_config.scoring_concurrency)
        rate_limiter = (
            AsyncRateLimiter(sweep_config.rps_limit, warmup=sweep_config.rps_warmup)
            if sweep_runtime is None and sweep_config.rps_limit > 0
            else None
        )

        recovered_map: dict[int, tuple[dict, dict]] = {}

        async with httpx.AsyncClient(
            proxy=None,
            timeout=sweep_config.request_timeout,
            limits=httpx.Limits(
                max_connections=sweep_config.scoring_concurrency + 10,
                max_keepalive_connections=sweep_config.scoring_concurrency,
            ),
        ) as client:

            async def _retry_one(idx: int, sample: dict):
                nonlocal attempted, recovered
                attempted += 1
                rarity = raw_rarities[idx] if idx < len(raw_rarities) else {"score": None}
                value, monitor = await score_one(
                    client,
                    sample,
                    sweep_config.scoring_model,
                    rarity,
                    idx,
                    len(raw_rarities),
                    sem,
                    config=sweep_config,
                    rate_limiter=rate_limiter,
                )
                if value:
                    recovered += 1
                    recovered_map[idx] = (value, monitor)
                else:
                    recovered_map[idx] = (None, monitor)

            tasks = [asyncio.create_task(_retry_one(idx, sample)) for idx, sample in retry_items]
            for coro in asyncio.as_completed(tasks):
                await coro

        # Patch scored.jsonl + monitor_value.jsonl (aligned line-for-line).
        scored_tmp = scored_path.with_suffix(".tmp")
        monitor_tmp = monitor_path.with_suffix(".tmp")
        failed_tmp = (Path(output_dir) / "failed_value.jsonl.tmp")
        failures_tmp = (Path(output_dir) / "score_failures.jsonl.tmp")
        failures_written = 0
        failed_written = 0

        with open(scored_path, "r", encoding="utf-8") as fs, open(monitor_path, "r", encoding="utf-8") as fm, \
             open(scored_tmp, "w", encoding="utf-8") as out_s, open(monitor_tmp, "w", encoding="utf-8") as out_m, \
             open(failed_tmp, "w", encoding="utf-8") as out_failed, open(failures_tmp, "w", encoding="utf-8") as out_failures:
            for idx, (line_s, line_m) in enumerate(zip(fs, fm)):
                sample = json.loads(line_s)
                monitor = json.loads(line_m)
                if idx in recovered_map:
                    value, new_monitor = recovered_map[idx]
                    if value:
                        sample["value"] = value
                    monitor = new_monitor or monitor
                # rewrite
                out_s.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_m.write(json.dumps(monitor, ensure_ascii=False) + "\n")
                if sample.get("value") is None:
                    out_failed.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    failed_written += 1
                    record = {
                        "sample_id": sample.get("id", f"sample-{idx}"),
                        "status": monitor.get("status", "no_result") if isinstance(monitor, dict) else "no_result",
                        "error": (monitor.get("error", "") if isinstance(monitor, dict) else ""),
                        "error_response": (monitor.get("error_response", "")[:1000] if isinstance(monitor, dict) else ""),
                        "attempts": (monitor.get("attempts", 0) if isinstance(monitor, dict) else 0),
                        "error_class": (monitor.get("error_class") if isinstance(monitor, dict) else None),
                        "retryable_infra": (monitor.get("retryable_infra") if isinstance(monitor, dict) else None),
                        "http_status": (monitor.get("http_status") if isinstance(monitor, dict) else None),
                        "runtime_state": (monitor.get("runtime_state") if isinstance(monitor, dict) else None),
                    }
                    out_failures.write(json.dumps(record, ensure_ascii=False) + "\n")
                    failures_written += 1

        os.replace(scored_tmp, scored_path)
        os.replace(monitor_tmp, monitor_path)

        failed_path = Path(output_dir) / "failed_value.jsonl"
        failures_path = Path(output_dir) / "score_failures.jsonl"
        if failed_written > 0:
            os.replace(failed_tmp, failed_path)
        else:
            try:
                failed_tmp.unlink()
            except OSError:
                pass
            if failed_path.exists():
                failed_path.unlink()
        if failures_written > 0:
            os.replace(failures_tmp, failures_path)
        else:
            try:
                failures_tmp.unlink()
            except OSError:
                pass
            if failures_path.exists():
                failures_path.unlink()

        # Recompute retry set for next pass (based on patched monitors).
        retry_items = []
        with open(scored_path, "r", encoding="utf-8") as fs, open(monitor_path, "r", encoding="utf-8") as fm:
            for idx, (line_s, line_m) in enumerate(zip(fs, fm)):
                sample = json.loads(line_s)
                monitor = json.loads(line_m)
                if sample.get("value") is None and _should_recovery_retry(monitor):
                    retry_items.append((idx, sample))

    return {"enabled": True, "attempted": attempted, "recovered": recovered}


def _coerce_positive_int(value):
    """Parse a positive integer from JSON-ish input; return 0 when invalid."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        iv = int(value)
        return iv if iv > 0 else 0
    return 0


def _write_json_atomic(path, payload):
    """Write JSON via a temp file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path, records):
    """Write JSONL via a temp file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _resolved_scoring_prompt_budget(config=None):
    """Return effective prompt mode metadata for Pass 2 observability."""
    compact = config.prompt_mode == "compact" if config else False
    budget = config.value_truncation_budget if config else VALUE_TRUNCATION_BUDGET
    if compact and config and budget == VALUE_TRUNCATION_BUDGET:
        budget = COMPACT_VALUE_TRUNCATION_BUDGET
    return compact, budget


def _annotate_scoring_prompt_stats(stats, config=None):
    compact, budget = _resolved_scoring_prompt_budget(config)
    stats["prompt_mode"] = "compact" if compact else "full"
    stats["compact_prompt"] = compact
    stats["value_truncation_budget"] = budget
    return stats


def _normalize_combo_counts(raw_counts):
    """Normalize combo counts map loaded from stats.json.

    Returns None when missing/invalid.
    """
    if not isinstance(raw_counts, dict):
        return None
    cleaned = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or not key:
            continue
        count = _coerce_positive_int(value)
        if count > 0:
            cleaned[key] = count
    return cleaned or None


def load_tag_stats_context(stats_path):
    """Load rarity baseline context from a Pass 1 stats file.

    Returns (distributions, distribution_total_samples, timestamp, meta) where
    meta may include:
      - stats_total_samples
      - distribution_total_samples
      - combo_counts
    Returns (None, 0, None, {}) on failure.
    """
    path = Path(stats_path)
    if not path.exists():
        return None, 0, None, {}

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    distributions = stats.get("tag_distributions")
    if not distributions:
        return None, 0, None, {}

    stats_total_samples = _coerce_positive_int(stats.get("total_samples", 0))
    distribution_total_samples = _coerce_positive_int(
        stats.get("distribution_total_samples", stats_total_samples)
    )
    if distribution_total_samples <= 0:
        distribution_total_samples = stats_total_samples

    combo_counts = _normalize_combo_counts(stats.get("combo_distributions"))
    timestamp = stats.get("timestamp", None)

    return distributions, distribution_total_samples, timestamp, {
        "stats_total_samples": stats_total_samples,
        "distribution_total_samples": distribution_total_samples,
        "combo_counts": combo_counts,
        "extension_stats": stats.get("extension_stats") or None,
    }


def load_tag_stats(stats_path):
    """Load tag_distributions from a Pass 1 stats file.

    Uses `distribution_total_samples` when present; falls back to
    `total_samples` for backward compatibility.

    Returns (distributions, total_samples, timestamp) or (None, 0, None) on failure.
    """
    distributions, total_samples, timestamp, _meta = load_tag_stats_context(stats_path)
    return distributions, total_samples, timestamp


_RARITY_DEFAULT_CONFIDENCE = 0.75
_RARITY_INHERITED_CONFIDENCE_PENALTY = 0.65
_RARITY_MULTI_TAG_MAX_BLEND = 0.60
_SELECTION_CONFIDENCE_FLOOR = 0.25
_EXTENSION_RARITY_BLEND_WEIGHT = 0.10
_EXTENSION_RARITY_BONUS_CAP = 0.50


def _coerce_unit_float(value, default):
    """Clamp a possibly-invalid float-like value into [0, 1]."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _dimension_label_confidence(labels, dim, default=_RARITY_DEFAULT_CONFIDENCE):
    """Return the effective confidence for one label dimension."""
    if not isinstance(labels, dict):
        return default
    confidence = default
    confidence_map = labels.get("confidence")
    if isinstance(confidence_map, dict):
        confidence = _coerce_unit_float(confidence_map.get(dim), default)
    if labels.get("inherited"):
        confidence *= _RARITY_INHERITED_CONFIDENCE_PENALTY
    return max(0.0, min(1.0, confidence))


def _sample_label_confidence(labels, dims=None, default=_RARITY_DEFAULT_CONFIDENCE):
    """Return the mean confidence across populated label dimensions."""
    if not isinstance(labels, dict):
        return default
    if dims is None:
        dims = [dim for dim in RARITY_WEIGHTS.keys() if labels.get(dim) is not None]
    confidences = []
    for dim in dims:
        tags = labels.get(dim)
        if tags is None:
            continue
        if isinstance(tags, list) and not tags:
            continue
        if isinstance(tags, str) and not tags:
            continue
        confidences.append(_dimension_label_confidence(labels, dim, default=default))
    if not confidences:
        return default * (_RARITY_INHERITED_CONFIDENCE_PENALTY if labels.get("inherited") else 1.0)
    return sum(confidences) / len(confidences)


def _dimension_rarity_prior(dim_idfs):
    """Use the dimension mean as a conservative prior for missing/uncertain tags."""
    if not dim_idfs:
        return 0.0
    values = [v for v in dim_idfs.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _aggregate_multi_tag_rarity(values):
    """Blend max and mean to reduce single-tag max-IDF amplification."""
    if not values:
        return 0.0
    max_value = max(values)
    mean_value = sum(values) / len(values)
    return (_RARITY_MULTI_TAG_MAX_BLEND * max_value
            + (1.0 - _RARITY_MULTI_TAG_MAX_BLEND) * mean_value)


def compute_tag_idf(distributions, total_samples):
    """Compute IDF for every tag across all dimensions.

    idf(tag) = log2(N / (count + 1))

    Returns {dim: {tag: idf_value}}.
    """
    if not distributions or total_samples <= 0:
        return {}

    idf_map = {}
    for dim, tag_counts in distributions.items():
        idf_map[dim] = {}
        for tag, count in tag_counts.items():
            # Multi-select dimensions can have count >= total_samples; clamp at 0
            # because "negative rarity" is not meaningful for downstream scoring.
            idf_map[dim][tag] = max(0.0, math.log2(total_samples / (count + 1)))

    return idf_map


def compute_extension_field_idf(value_counts, baseline_total):
    """Compute IDF map for one extension field from successful matched baselines."""
    baseline_total = _coerce_positive_int(baseline_total)
    if not isinstance(value_counts, dict) or baseline_total <= 0:
        return {}

    idf_map = {}
    for value, count in value_counts.items():
        if value in (None, ""):
            continue
        count = _coerce_positive_int(count)
        idf_map[value] = max(0.0, math.log2(baseline_total / (count + 1)))
    return idf_map


def compute_extension_field_idf_map(field_value_distributions, baseline_total):
    """Compute per-field extension IDF maps."""
    if not isinstance(field_value_distributions, dict):
        return {}
    idf_map = {}
    for field, value_counts in field_value_distributions.items():
        field_idfs = compute_extension_field_idf(value_counts, baseline_total)
        if field_idfs:
            idf_map[field] = field_idfs
    return idf_map


def _extension_field_names(spec_payload, baseline):
    config_schema = ((baseline or {}).get("config") or {}).get("schema")
    if isinstance(config_schema, dict) and config_schema:
        return list(config_schema.keys())

    names = []
    for source in (
        ((baseline or {}).get("field_value_distributions") or {}),
        ((baseline or {}).get("field_presence_counts") or {}),
        ((spec_payload or {}).get("labels") or {}),
    ):
        if not isinstance(source, dict):
            continue
        for key in source.keys():
            if key not in names:
                names.append(key)
    return names


def _extension_field_prior(field_idfs):
    return _dimension_rarity_prior(field_idfs)


def compute_extension_field_rarity(value, field_idfs, confidence=_RARITY_DEFAULT_CONFIDENCE):
    """Compute one extension field rarity with confidence shrinkage to the field prior."""
    if not isinstance(field_idfs, dict) or not field_idfs:
        return 0.0

    prior = _extension_field_prior(field_idfs)
    confidence = _coerce_unit_float(confidence, _RARITY_DEFAULT_CONFIDENCE)

    if isinstance(value, list):
        observed_values = [
            field_idfs.get(item, prior)
            for item in value
            if item not in (None, "")
        ]
        if not observed_values:
            return 0.0
        observed_rarity = _aggregate_multi_tag_rarity(observed_values)
    else:
        if value in (None, ""):
            return 0.0
        observed_rarity = field_idfs.get(value, prior)

    return confidence * observed_rarity + (1.0 - confidence) * prior


def normalize_extension_rarity_score(raw_score, baseline_total, mode="absolute", raw_population=None):
    """Normalize one extension rarity score onto the shared 1-10 scale."""
    if raw_score is None:
        return None

    mode = str(mode or "absolute").strip().lower()
    if mode == "percentile" and raw_population:
        values = sorted(float(value) for value in raw_population if isinstance(value, (int, float)))
        if values:
            rank = 0
            for i, value in enumerate(values):
                if value >= raw_score:
                    rank = i
                    break
            else:
                rank = len(values) - 1
            percentile = rank / max(len(values) - 1, 1)
            return round(1 + percentile * 9, 1)

    ceiling = max(math.log2(max(_coerce_positive_int(baseline_total), 2)), 1e-9)
    raw = max(0.0, min(float(raw_score), ceiling))
    return round(1 + (raw / ceiling) * 9, 1)


def resolve_extension_rarity_mode(config):
    """Resolve extension rarity mode from config with safe fallback."""
    mode = getattr(config, "extension_rarity_mode", "off") if config is not None else "off"
    mode = str(mode or "off").strip().lower()
    if mode not in {"off", "preview", "bonus_only"}:
        return "off"
    return mode


def _extract_sample_extension_payloads(sample):
    if not isinstance(sample, dict):
        return {}
    payload = sample.get("label_extensions")
    if isinstance(payload, dict) and payload:
        return payload
    labels = sample.get("labels") or {}
    payload = labels.get("label_extensions")
    if isinstance(payload, dict) and payload:
        return payload
    return {}


def compute_extension_spec_rarity(spec_payload, baseline, *, normalization_mode="absolute", default_confidence=_RARITY_DEFAULT_CONFIDENCE):
    """Compute extension rarity for one spec from persisted spec-local baseline data."""
    payload = spec_payload if isinstance(spec_payload, dict) else {}
    baseline = baseline if isinstance(baseline, dict) else {}
    status = payload.get("status")
    matched = bool(payload.get("matched"))
    spec_hash = payload.get("spec_hash") or baseline.get("spec_hash")
    baseline_total = _coerce_positive_int(baseline.get("baseline_total", 0))

    result = {
        "status": status,
        "matched": matched,
        "spec_version": payload.get("spec_version") or baseline.get("spec_version"),
        "spec_hash": spec_hash,
        "baseline_total": baseline_total,
        "raw_score": None,
        "score": None,
        "confidence": 0.0,
    }

    if status != "success" or not matched or baseline_total <= 0:
        return result

    labels = payload.get("labels") if isinstance(payload.get("labels"), dict) else {}
    confidence_map = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
    field_idf_map = compute_extension_field_idf_map(
        baseline.get("field_value_distributions"),
        baseline_total,
    )
    field_names = _extension_field_names(payload, baseline)
    if not field_names:
        field_names = list(field_idf_map.keys())

    field_rarities = []
    populated_confidences = []
    populated_fields = 0

    for field in field_names:
        value = labels.get(field)
        if value in (None, "") or (isinstance(value, list) and not [item for item in value if item not in (None, "")]):
            continue
        field_idfs = field_idf_map.get(field)
        if not field_idfs:
            continue
        field_confidence = _coerce_unit_float(confidence_map.get(field), default_confidence)
        field_rarity = compute_extension_field_rarity(value, field_idfs, confidence=field_confidence)
        field_rarities.append(field_rarity)
        populated_confidences.append(field_confidence)
        populated_fields += 1

    if not field_rarities:
        return result

    raw_score = sum(field_rarities) / len(field_rarities)
    denominator = max(len(field_names), 1)
    populated_ratio = populated_fields / denominator
    spec_confidence = (sum(populated_confidences) / len(populated_confidences)) * populated_ratio

    result["raw_score"] = round(raw_score, 4)
    result["score"] = normalize_extension_rarity_score(
        raw_score,
        baseline_total=baseline_total,
        mode=normalization_mode,
    )
    result["confidence"] = round(spec_confidence, 3)
    return result


def compute_sample_extension_rarity(sample, extension_stats, *, config=None, baseline_source="external"):
    """Compute aggregate extension rarity for one sample from spec-local baselines."""
    mode = resolve_extension_rarity_mode(config)
    payloads = _extract_sample_extension_payloads(sample)
    specs = (extension_stats or {}).get("specs") or {}
    min_baseline_total = _coerce_positive_int(
        getattr(config, "min_extension_baseline_total", 0) if config is not None else 0
    )

    result = {
        "mode": mode,
        "score": None,
        "confidence": 0.0,
        "matched_specs": 0,
        "baseline_source": baseline_source,
        "source_eligible": baseline_source != "local",
        "support_sufficient": False,
        "specs": {},
    }
    if mode == "off" or not payloads:
        return result

    weighted_score_sum = 0.0
    weight_total = 0.0
    confidence_sum = 0.0
    confidence_count = 0

    for spec_id, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        spec_stats = specs.get(spec_id) or {}
        spec_hash = str(payload.get("spec_hash") or spec_stats.get("spec_hash") or "")
        baseline = copy.deepcopy(((spec_stats.get("baselines") or {}).get(spec_hash)) or {})
        if spec_stats.get("config") and "config" not in baseline:
            baseline["config"] = spec_stats.get("config")

        spec_result = compute_extension_spec_rarity(
            payload,
            baseline,
            normalization_mode=resolve_rarity_mode(config),
        )
        support_sufficient = spec_result["baseline_total"] >= min_baseline_total
        spec_result["support_sufficient"] = support_sufficient
        spec_result["source_eligible"] = baseline_source != "local"
        result["specs"][spec_id] = spec_result

        if spec_result["matched"]:
            result["matched_specs"] += 1
        if spec_result["score"] is None:
            continue
        if not support_sufficient:
            continue

        spec_confidence = _coerce_unit_float(spec_result.get("confidence"), 0.0)
        confidence_sum += spec_confidence
        confidence_count += 1
        weighted_score_sum += spec_result["score"] * max(spec_confidence, 1e-9)
        weight_total += max(spec_confidence, 1e-9)
        result["support_sufficient"] = result["support_sufficient"] or support_sufficient

    if weight_total > 0:
        result["score"] = round(weighted_score_sum / weight_total, 1)
    if confidence_count > 0:
        result["confidence"] = round(confidence_sum / confidence_count, 3)

    return result


def augment_rarity_result(core_rarity, sample, *, extension_stats=None, config=None, baseline_source="external"):
    """Additive rarity V2 wrapper that preserves legacy core rarity semantics."""
    mode = resolve_extension_rarity_mode(config)
    if mode == "off":
        return core_rarity

    core_copy = copy.deepcopy(core_rarity) if isinstance(core_rarity, dict) else {"score": None}
    result = copy.deepcopy(core_copy)
    result["rarity_core"] = copy.deepcopy(core_copy)

    extension_rarity = compute_sample_extension_rarity(
        sample,
        extension_stats,
        config=config,
        baseline_source=baseline_source,
    )
    result["rarity_extension"] = extension_rarity

    if mode == "bonus_only":
        core_score = core_copy.get("score")
        extension_score = extension_rarity.get("score")
        extension_gate = 0.0
        if (
            isinstance(extension_score, (int, float))
            and extension_rarity.get("support_sufficient")
            and extension_rarity.get("source_eligible")
        ):
            extension_gate = _coerce_unit_float(extension_rarity.get("confidence"), 0.0)
        extension_bonus = 0.0
        if isinstance(extension_score, (int, float)):
            extension_bonus = _EXTENSION_RARITY_BLEND_WEIGHT * extension_gate * max(0.0, extension_score - 5.0)
            extension_bonus = min(max(extension_bonus, 0.0), _EXTENSION_RARITY_BONUS_CAP)
        rarity_v2_score = core_score
        if isinstance(core_score, (int, float)):
            rarity_v2_score = round(max(1.0, min(10.0, core_score + extension_bonus)), 1)
        result["rarity_v2"] = {
            "score": rarity_v2_score,
            "core_score": core_score,
            "extension_bonus": round(extension_bonus, 3),
            "blend_mode": "bonus_only",
            "extension_gate": round(extension_gate, 3),
        }

    return result


def _refresh_augmented_rarity_after_core_normalization(rarity_result):
    """Sync additive V2 rarity fields after legacy core score normalization."""
    if not isinstance(rarity_result, dict):
        return rarity_result

    core_view = {
        key: value
        for key, value in rarity_result.items()
        if key not in {"rarity_core", "rarity_extension", "rarity_v2"}
    }
    if isinstance(rarity_result.get("rarity_core"), dict):
        rarity_result["rarity_core"] = copy.deepcopy(core_view)

    rarity_v2 = rarity_result.get("rarity_v2")
    extension_rarity = rarity_result.get("rarity_extension") or {}
    if isinstance(rarity_v2, dict):
        core_score = core_view.get("score")
        extension_score = extension_rarity.get("score")
        extension_gate = 0.0
        if (
            isinstance(extension_score, (int, float))
            and extension_rarity.get("support_sufficient")
            and extension_rarity.get("source_eligible")
        ):
            extension_gate = _coerce_unit_float(extension_rarity.get("confidence"), 0.0)
        extension_bonus = 0.0
        if isinstance(extension_score, (int, float)):
            extension_bonus = _EXTENSION_RARITY_BLEND_WEIGHT * extension_gate * max(0.0, extension_score - 5.0)
            extension_bonus = min(max(extension_bonus, 0.0), _EXTENSION_RARITY_BONUS_CAP)
        rarity_v2["core_score"] = core_score
        rarity_v2["extension_bonus"] = round(extension_bonus, 3)
        rarity_v2["extension_gate"] = round(extension_gate, 3)
        if isinstance(core_score, (int, float)):
            rarity_v2["score"] = round(max(1.0, min(10.0, core_score + extension_bonus)), 1)
    return rarity_result


def compute_sample_rarity(sample_labels, idf_map, total_samples,
                          rarity_weights=None, combo_alpha=None,
                          combo_counts=None, stats_ref_info=None):
    """Compute rarity score for a single sample.

    Returns dict with score, tag_rarity, combo_rarity, stats_ref.
    Score is raw (not normalized to 1-10 yet).
    """
    if rarity_weights is None:
        rarity_weights = RARITY_WEIGHTS
    if combo_alpha is None:
        combo_alpha = RARITY_COMBO_ALPHA
    if not isinstance(sample_labels, dict):
        sample_labels = {}

    if not idf_map:
        return {
            "score": None,
            "tag_rarity": None,
            "combo_rarity": None,
            "stats_ref": stats_ref_info,
        }

    # Weighted tag IDF
    weighted_sum = 0.0
    weight_total = 0.0

    for dim, weight in rarity_weights.items():
        tags = sample_labels.get(dim)
        if tags is None:
            continue

        # Missing dimension in baseline: skip instead of treating everything as
        # maximally rare. This protects external baseline scenarios with sparse
        # or incomplete dimensions.
        dim_idfs = idf_map.get(dim)
        if not dim_idfs:
            continue

        dim_prior = _dimension_rarity_prior(dim_idfs)
        dim_confidence = _dimension_label_confidence(sample_labels, dim)

        if isinstance(tags, list):
            if not tags:
                continue
            tag_idf_values = [
                dim_idfs.get(t, dim_prior)
                for t in tags
                if isinstance(t, str) and t
            ]
            if not tag_idf_values:
                continue
            observed_rarity = _aggregate_multi_tag_rarity(tag_idf_values)
        else:
            # Single-select dimension
            observed_rarity = dim_idfs.get(tags, dim_prior)

        dim_rarity = dim_confidence * observed_rarity + (1 - dim_confidence) * dim_prior

        weighted_sum += weight * dim_rarity
        weight_total += weight

    tag_rarity = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Combo rarity
    combo_rarity = 0.0
    if combo_counts is not None:
        combo_count = _lookup_combo_count(combo_counts, sample_labels)
        if combo_count is not None:
            raw_combo_rarity = math.log2(total_samples / (combo_count + 1))
            combo_confidence = _sample_label_confidence(sample_labels, dims=list(rarity_weights.keys()))
            combo_rarity = (
                combo_confidence * raw_combo_rarity
                + (1 - combo_confidence) * tag_rarity
            )

    raw_score = combo_alpha * tag_rarity + (1 - combo_alpha) * combo_rarity

    return {
        "score": raw_score,
        "tag_rarity": round(tag_rarity, 3),
        "combo_rarity": round(combo_rarity, 3),
        "effective_confidence": round(_sample_label_confidence(sample_labels), 3),
        "uncertainty": round(1.0 - _sample_label_confidence(sample_labels), 3),
        "stats_ref": stats_ref_info,
    }


def _labels_have_rarity_signal(labels, dims=None):
    """Return True when labels contain at least one non-empty rarity tag."""
    if not is_usable_labels(labels):
        return False
    target_dims = list(dims) if dims is not None else list(RARITY_WEIGHTS.keys())
    for dim in target_dims:
        tags = labels.get(dim)
        if isinstance(tags, list):
            if any(isinstance(t, str) and t for t in tags):
                return True
        elif isinstance(tags, str) and tags:
            return True
    return False


def _normalize_concepts(labels):
    """Normalize concept labels into sorted string tags."""
    concepts = labels.get("concept", [])
    if isinstance(concepts, str):
        concepts = [concepts]
    if not isinstance(concepts, list):
        return []
    return sorted([c for c in concepts if isinstance(c, str) and c])


def _normalize_combo_dim(labels, dim, limit):
    """Normalize one combo-key dimension into a bounded list of string tags."""
    value = labels.get(dim)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    cleaned = sorted({item for item in value if isinstance(item, str) and item})
    if limit > 0:
        return cleaned[:limit]
    return cleaned


def _legacy_combo_key_from_labels(labels):
    """Backward-compatible combo key from (intent, difficulty, concept[:3])."""
    intent = labels.get("intent", "")
    difficulty = labels.get("difficulty", "")
    intent = intent if isinstance(intent, str) else ""
    difficulty = difficulty if isinstance(difficulty, str) else ""
    concepts = _normalize_concepts(labels)[:3]
    if not intent and not difficulty and not concepts:
        return None
    return f"{intent}|{difficulty}|{','.join(concepts)}"


def _combo_key_from_labels(labels):
    """Build a richer combo key for multi-turn/code-SFT rarity."""
    parts = []
    for dim in ("intent", "difficulty", "context"):
        value = labels.get(dim)
        if isinstance(value, str) and value:
            parts.append(f"{dim}={value}")

    for dim, limit in (
        ("language", 2),
        ("domain", 2),
        ("task", 2),
        ("concept", 3),
        ("agentic", 2),
        ("constraint", 2),
    ):
        values = _normalize_combo_dim(labels, dim, limit)
        if values:
            parts.append(f"{dim}={','.join(values)}")

    if not parts:
        return None
    return "|".join(parts)


def _combo_keys_from_labels(labels):
    """Return candidate combo keys, supporting both richer and legacy baselines."""
    primary = _combo_key_from_labels(labels)
    legacy = _legacy_combo_key_from_labels(labels)
    keys = []
    if primary:
        keys.append(primary)
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys


def _lookup_combo_count(combo_counts, labels):
    """Resolve the most conservative matching combo count across key versions."""
    best_count = None
    for key in _combo_keys_from_labels(labels):
        if key in combo_counts:
            count = combo_counts[key]
            best_count = count if best_count is None else max(best_count, count)
    return best_count


def _update_combo_counts_from_labels(combo_counts, labels):
    """In-place combo count update; returns True when one combo is counted."""
    if isinstance(labels, dict) and labels.get("inherited"):
        return False
    if not _labels_have_rarity_signal(labels):
        return False
    combo_key = _combo_key_from_labels(labels)
    if not combo_key:
        return False
    combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
    return True


def build_combo_counts(samples):
    """Build combo occurrence counts from labeled samples for combo rarity."""
    counts = {}
    for s in samples:
        labels = s.get("labels") or {}
        _update_combo_counts_from_labels(counts, labels)
    return counts


def _resolve_combo_baseline(external_combo_counts, local_combo_counts=None):
    """Resolve combo counts, preferring external stats then local fallback."""
    if external_combo_counts:
        return external_combo_counts, "external", None

    local_combo_counts = _normalize_combo_counts(local_combo_counts)
    if local_combo_counts:
        return (
            local_combo_counts,
            "hybrid",
            "  External stats has no combo_distributions; using local combo fallback",
        )

    return (
        None,
        "disabled",
        "  External stats has no combo_distributions; combo rarity disabled",
    )


def build_tag_distributions(samples, rarity_weights=None):
    """Build per-dimension tag distributions from labeled samples."""
    dims = list((rarity_weights or RARITY_WEIGHTS).keys())
    distributions = {dim: {} for dim in dims}
    total = 0

    for s in samples:
        labels = s.get("labels") or {}
        if labels.get("inherited"):
            continue
        if not _labels_have_rarity_signal(labels, dims=dims):
            continue
        total += 1
        for dim in dims:
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                if not tags:
                    continue
                for tag in tags:
                    if not isinstance(tag, str) or not tag:
                        continue
                    distributions[dim][tag] = distributions[dim].get(tag, 0) + 1
            else:
                if isinstance(tags, str) and tags:
                    distributions[dim][tags] = distributions[dim].get(tags, 0) + 1

    # Remove empty dimensions
    distributions = {dim: counts for dim, counts in distributions.items() if counts}
    return distributions, total


def _update_distributions_from_labels(distributions, labels, dims):
    """In-place update for streaming distribution construction.

    Returns:
        bool: True when at least one tag is counted into distributions.
    """
    if not is_usable_labels(labels):
        return False
    if labels.get("inherited"):
        return False
    touched = False
    for dim in dims:
        tags = labels.get(dim)
        if tags is None:
            continue
        dim_dist = distributions.setdefault(dim, {})
        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, str) or not tag:
                    continue
                dim_dist[tag] = dim_dist.get(tag, 0) + 1
                touched = True
        else:
            if isinstance(tags, str) and tags:
                dim_dist[tags] = dim_dist.get(tags, 0) + 1
                touched = True
    return touched


def normalize_rarity_scores(rarity_results, mode="percentile", total_samples=0):
    """Normalize raw rarity scores to 1-10 scale.

    Modes:
      - percentile: within-batch percentile mapping (legacy behavior)
      - absolute: absolute mapping by log2(total_samples) ceiling

    Modifies rarity dicts in-place, setting score to normalized value.
    Returns normalization metadata.
    """
    raw_scores = [r["score"] for r in rarity_results if r["score"] is not None]
    if not raw_scores:
        return {}

    if mode == "absolute":
        n = max(int(total_samples), 2)
        raw_ceiling = max(math.log2(n), 1e-9)
        for r in rarity_results:
            if r["score"] is None:
                continue
            raw = max(0.0, min(float(r["score"]), raw_ceiling))
            r["score"] = round(1 + (raw / raw_ceiling) * 9, 1)
        return {"mode": "absolute", "raw_ceiling": round(raw_ceiling, 4), "total_samples": n}

    raw_scores_sorted = sorted(raw_scores)
    n = len(raw_scores_sorted)

    for r in rarity_results:
        if r["score"] is None:
            continue
        # Find percentile rank
        rank = 0
        for i, v in enumerate(raw_scores_sorted):
            if v >= r["score"]:
                rank = i
                break
        else:
            rank = n - 1
        percentile = rank / max(n - 1, 1)
        # Map to 1-10
        r["score"] = round(1 + percentile * 9, 1)

    # Return breakpoints for stats
    breakpoints = {}
    for pct in [10, 25, 50, 75, 90]:
        idx = int(n * pct / 100)
        breakpoints[f"p{pct}"] = raw_scores_sorted[min(idx, n - 1)]
    breakpoints["mode"] = "percentile"
    return breakpoints


def resolve_rarity_mode(config):
    """Resolve rarity normalization mode from config with safe fallback."""
    mode = getattr(config, "rarity_score_mode", RARITY_SCORE_MODE) or RARITY_SCORE_MODE
    mode = str(mode).strip().lower()
    if mode not in {"absolute", "percentile"}:
        return RARITY_SCORE_MODE
    return mode


# ─────────────────────────────────────────────────────────
# Score validation
# ─────────────────────────────────────────────────────────

def _clamp_overall_to_subscores(dim_dict, sub_keys, dim_name, issues):
    """Clamp overall to within ±2 of mean(sub_scores).

    When |overall - mean(sub_scores)| > 2.0, sets overall = round(mean).
    Operates in-place on dim_dict; appends to issues list.
    """
    overall = dim_dict.get("overall")
    if overall is None:
        return
    valid_subs = [dim_dict[k] for k in sub_keys if dim_dict.get(k) is not None]
    if not valid_subs:
        return
    mean_sub = sum(valid_subs) / len(valid_subs)
    if abs(overall - mean_sub) > 2.0:
        clamped = max(1, min(10, round(mean_sub)))
        issues.append(f"{dim_name}.overall {overall} clamped to {clamped} "
                      f"(mean sub-scores={mean_sub:.1f})")
        dim_dict["overall"] = clamped


def validate_score_response(parsed):
    """Validate LLM scoring response.

    Returns (cleaned_result, issues) where issues is a list of strings.
    """
    issues = []
    if parsed is None:
        return None, ["null response"]
    if not isinstance(parsed, dict):
        return None, [f"response not a dict: {type(parsed).__name__}"]

    result = {}

    # Validate complexity
    complexity = parsed.get("complexity", {})
    if isinstance(complexity, dict):
        for key in ("instruction", "analytical_depth", "implementation", "overall"):
            val = complexity.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and 1 <= val <= 10:
                complexity[key] = int(val)
            else:
                complexity[key] = None
                issues.append(f"complexity.{key} invalid: {val}")
        result["complexity"] = complexity
        _clamp_overall_to_subscores(
            complexity, ["instruction", "analytical_depth", "implementation"],
            "complexity", issues)
    else:
        result["complexity"] = {"instruction": None, "analytical_depth": None, "implementation": None, "overall": None}
        issues.append("complexity not a dict")

    # Validate quality
    quality = parsed.get("quality", {})
    if isinstance(quality, dict):
        for key in ("correctness", "code_quality", "explanation", "completeness", "overall"):
            val = quality.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and 1 <= val <= 10:
                quality[key] = int(val)
            else:
                quality[key] = None
                issues.append(f"quality.{key} invalid: {val}")
        result["quality"] = quality
        # Enforce overall <= correctness + 2 (prompt constraint)
        q_correct = quality.get("correctness")
        q_overall = quality.get("overall")
        if (q_correct is not None and q_overall is not None
                and q_overall > q_correct + 2):
            quality["overall"] = q_correct + 2
            issues.append(f"quality.overall {q_overall} clamped to {quality['overall']} (correctness+2 rule)")
        _clamp_overall_to_subscores(
            quality, ["correctness", "code_quality", "explanation", "completeness"],
            "quality", issues)
    else:
        result["quality"] = {"correctness": None, "code_quality": None, "explanation": None, "completeness": None, "overall": None}
        issues.append("quality not a dict")

    # Validate reasoning
    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict):
        for key in ("clarity", "consistency", "overall"):
            val = reasoning.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and 1 <= val <= 10:
                reasoning[key] = int(val)
            else:
                reasoning[key] = None
                if val is not None:
                    issues.append(f"reasoning.{key} invalid: {val}")
        # self_correction can be bool
        sc = reasoning.get("self_correction")
        if isinstance(sc, bool):
            reasoning["self_correction"] = sc
        else:
            reasoning["self_correction"] = None
        result["reasoning"] = reasoning
        _clamp_overall_to_subscores(
            reasoning, ["clarity", "consistency"],
            "reasoning", issues)
    else:
        result["reasoning"] = {"clarity": None, "consistency": None, "self_correction": None, "overall": None}
        issues.append("reasoning not a dict")

    # Validate flags
    flags = parsed.get("flags", [])
    if isinstance(flags, list):
        known = []
        unknown = []
        for f in flags:
            if isinstance(f, str):
                if f in KNOWN_FLAGS:
                    known.append(f)
                else:
                    unknown.append(f)
        result["flags"] = known
        result["unknown_flags"] = unknown
        if unknown:
            issues.append(f"unknown flags: {unknown}")
    else:
        result["flags"] = []
        result["unknown_flags"] = []

    # Validate confidence
    conf = parsed.get("confidence", 0.5)
    if isinstance(conf, (int, float)) and 0 <= conf <= 1:
        result["confidence"] = round(float(conf), 2)
    else:
        result["confidence"] = 0.5
        issues.append(f"confidence invalid: {conf}")

    # Extract optional rationale (when ENABLE_RATIONALE is on)
    rationale = parsed.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        result["rationale"] = rationale.strip()[:500]

    return result, issues


# ─────────────────────────────────────────────────────────
# Value score computation
# ─────────────────────────────────────────────────────────

def compute_value_score(score_result, rarity_result, weights=None):
    """Compute weighted composite value_score.

    Uses default rarity=5.0 when rarity is missing (instead of renormalizing
    weights) to maintain cross-batch comparability.  Applies quality floor
    penalty: quality.overall < 4 → value *= 0.7.

    Returns float 1-10 or None if no valid scores.
    """
    if weights is None:
        weights = VALUE_WEIGHTS

    components = {}

    # Extract overall scores
    if score_result:
        c = score_result.get("complexity", {})
        if c and c.get("overall") is not None:
            components["complexity"] = c["overall"]
        q = score_result.get("quality", {})
        if q and q.get("overall") is not None:
            components["quality"] = q["overall"]
        r = score_result.get("reasoning", {})
        if r and r.get("overall") is not None:
            components["reasoning"] = r["overall"]

    if rarity_result and rarity_result.get("score") is not None:
        components["rarity"] = rarity_result["score"]

    if not components:
        return None

    # Require at least 2 LLM dimensions for a reliable composite score;
    # with only 1, weight renormalization inflates a single dimension
    llm_dims = sum(1 for k in ("complexity", "quality", "reasoning")
                   if k in components)
    if llm_dims < 2:
        return None

    # Apply rarity default only when we have at least one LLM score
    if "rarity" not in components and "rarity" in weights and weights["rarity"] > 0:
        components["rarity"] = 5.0

    # Weighted mean
    total_weight = sum(weights.get(k, 0) for k in components)
    if total_weight <= 0:
        return None

    value = sum(weights.get(k, 0) * v for k, v in components.items()) / total_weight

    # Quality floor penalty: buggy code should not pass threshold
    quality_overall = components.get("quality")
    quality_weight = weights.get("quality", 0)
    if quality_overall is not None and quality_overall < 4 and quality_weight > 0:
        value *= 0.7

    value = apply_score_confidence(value, score_confidence(score_result))
    return round(max(1.0, min(10.0, value)), 1)


def _attach_augmented_rarity_fields(value_payload, rarity_result):
    """Persist additive rarity V2 fields onto a value payload without changing legacy keys."""
    if not isinstance(value_payload, dict) or not isinstance(rarity_result, dict):
        return value_payload
    if isinstance(rarity_result.get("rarity_core"), dict):
        value_payload["rarity_core"] = copy.deepcopy(rarity_result["rarity_core"])
    if isinstance(rarity_result.get("rarity_extension"), dict):
        value_payload["rarity_extension"] = copy.deepcopy(rarity_result["rarity_extension"])
    if isinstance(rarity_result.get("rarity_v2"), dict):
        value_payload["rarity_v2"] = copy.deepcopy(rarity_result["rarity_v2"])
    return value_payload


def _clear_augmented_rarity_fields(value_payload, *, keep_preview=False):
    """Remove additive extension/V2 rarity fields from a value payload."""
    if not isinstance(value_payload, dict):
        return value_payload
    if not keep_preview:
        value_payload.pop("rarity_core", None)
        value_payload.pop("rarity_extension", None)
    value_payload.pop("rarity_v2", None)
    value_payload.pop("value_score_v2", None)
    value_payload.pop("selection_score_v2", None)
    return value_payload


# ─────────────────────────────────────────────────────────
# Selection score (intra-class quality ranking)
# ─────────────────────────────────────────────────────────

_SELECTION_DIMS = ["intent", "language", "domain", "concept", "task",
                   "agentic", "constraint", "context", "difficulty"]
_STAGE_ORDER = ("opener", "exploration", "implementation", "verification", "final-summary")
_SUMMARY_MARKERS = (
    "## summary", "summary", "final review", "changes made", "what i implemented",
    "implemented the requested", "successfully implemented", "here's a summary",
)
_ROOT_CAUSE_MARKERS = (
    "root cause", "caused by", "this issue was caused", "the bug was", "the failure was due",
)
_FIX_MARKERS = (
    "added", "updated", "changed", "removed", "fixed", "implemented", "refactored",
    "renamed", "patched", "handled",
)
_VERIFICATION_MARKERS = (
    "pytest", "test", "tests passed", "verified", "verification", "regression",
    "exit code: 0", "all tests pass", "passes now", "confirmed",
)
_FILE_PATH_RE = re.compile(
    r"(?:(?:/workspace/)?[\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|sql|yml|yaml|json|toml|ini|cfg|c|cc|cpp|h|hpp|cs|rb|php|swift|kt|scala))",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(
    r"`?[A-Za-z_][\w.]{2,}`?\s*(?:\(|class\b|def\b|method\b|function\b)",
    re.IGNORECASE,
)
_TOOL_CALL_RE = re.compile(r"<tool_call>|\"name\"\s*:\s*\"[^\"]+\"", re.IGNORECASE)
_LOW_INFO_TOOL_RE = re.compile(
    r"\b(view|grep|ls|find|pwd|tree|cat|head|tail|sed|awk|pip install|npm install|poetry install|pytest)\b",
    re.IGNORECASE,
)
_DOMAIN_HINTS = {
    "compiler-development": (
        "compiler", "parser", "lexer", "token", "ast", "bytecode", "transpile",
        "transpilation", "interpreter", "macro", "typecheck",
    ),
    "machine-learning": (
        "pytorch", "tensorflow", "torch", "tensor", "epoch_length", "optimizer",
        "training", "inference", "dataloader", "gradient", "loss",
    ),
    "cloud-computing": (
        "aws", "gcp", "azure", "cloud storage", "storage api", "lambda", "iam",
        "bigquery", "s3", "gcs", "pubsub",
    ),
    "api-development": (
        "api", "endpoint", "request", "response", "rest", "graphql",
        "dbapi", "header",
    ),
    "devops": (
        "docker", "kubernetes", "helm", "terraform", "github actions", "jenkins",
        "ci", "cd", "deploy", "build profile",
    ),
}


def _clamp_score(value, default=None):
    if value is None:
        return default
    return max(1.0, min(10.0, float(value)))


def _selection_view_from_sample(sample, config=None):
    """Extract current-turn-centered text used by selection stability heuristics."""
    preset = sample.get("selection_view")
    if isinstance(preset, dict):
        return {
            "current_request": preset.get("current_request", ""),
            "trajectory": preset.get("trajectory", ""),
            "response": preset.get("response", ""),
            "trajectory_turn_count": preset.get("trajectory_turn_count", 0),
            "trajectory_tool_turns": preset.get("trajectory_tool_turns", 0),
        }

    conversations = sample.get("conversations") or []
    if not conversations:
        return {
            "current_request": "",
            "trajectory": "",
            "response": "",
            "trajectory_turn_count": 0,
            "trajectory_tool_turns": 0,
        }

    metadata = sample.get("metadata") or {}
    thinking_mode = metadata.get("thinking_mode") or detect_thinking_mode(conversations)
    cot_text = metadata.get("cot_text", "")
    if not cot_text:
        cot_text, _chars, _cleaned = extract_cot_content(conversations)
    budget = 4000
    if config and getattr(config, "value_truncation_budget", None):
        budget = min(int(config.value_truncation_budget), 4000)
    truncated = truncate_for_scoring(
        conversations,
        thinking_mode,
        cot_text=cot_text,
        budget=budget,
    )
    return {
        "current_request": truncated.get("current_request") or truncated.get("instruction") or "",
        "trajectory": truncated.get("trajectory") or "",
        "response": truncated.get("response") or "",
        "trajectory_turn_count": truncated.get("trajectory_turn_count", 0),
        "trajectory_tool_turns": truncated.get("trajectory_tool_turns", 0),
    }


def _text_signal_count(text, patterns):
    lowered = (text or "").lower()
    return sum(1 for marker in patterns if marker in lowered)


def _infer_selection_features(*, sample=None, summary=None, config=None):
    """Derive deterministic stage/evidence features from local sample text."""
    if summary is None:
        summary = {}
    metadata = (sample or {}).get("metadata") or summary.get("metadata") or {}
    view = summary.get("selection_view")
    if not isinstance(view, dict):
        view = _selection_view_from_sample(sample or {}, config=config)

    request = view.get("current_request", "") or ""
    trajectory = view.get("trajectory", "") or ""
    response = view.get("response", "") or ""
    combined = "\n".join(part for part in (request, trajectory, response) if part)
    response.lower()
    combined.lower()

    slice_position = metadata.get("slice_position")
    slice_count = metadata.get("slice_count") or summary.get("slice_count")
    if not isinstance(slice_position, int):
        slice_position = metadata.get("turn_index")
    if not isinstance(slice_position, int):
        slice_position = summary.get("turn_index")
    if not isinstance(slice_count, int):
        slice_count = metadata.get("total_turns") or summary.get("total_turns")
    if not isinstance(slice_count, int):
        slice_count = 1
    position_ratio = (slice_position / slice_count) if slice_position and slice_count else 1.0

    trajectory_turn_count = view.get("trajectory_turn_count") or 0
    trajectory_tool_turns = view.get("trajectory_tool_turns") or 0
    tool_density = (
        trajectory_tool_turns / max(trajectory_turn_count, 1)
        if trajectory_turn_count else 0.0
    )

    summary_marker_hits = _text_signal_count(response, _SUMMARY_MARKERS)
    file_hits = len(_FILE_PATH_RE.findall(response))
    symbol_hits = len(_SYMBOL_RE.findall(response))
    root_cause_hits = _text_signal_count(response, _ROOT_CAUSE_MARKERS)
    fix_hits = _text_signal_count(response, _FIX_MARKERS)
    verification_hits = _text_signal_count(response, _VERIFICATION_MARKERS)
    workflow_verification_hits = _text_signal_count(combined, _VERIFICATION_MARKERS)
    evidence_categories = sum(
        1 for value in (
            file_hits > 0,
            symbol_hits > 0,
            root_cause_hits > 0,
            fix_hits > 0,
            verification_hits > 0,
        ) if value
    )
    has_summary_evidence = evidence_categories >= 2 or (
        file_hits > 0 and (fix_hits > 0 or verification_hits > 0)
    )

    tool_like = bool(_TOOL_CALL_RE.search(response) or _TOOL_CALL_RE.search(trajectory))
    low_info_action_hits = len(_LOW_INFO_TOOL_RE.findall("\n".join((response, trajectory))))
    response_words = len(re.findall(r"[A-Za-z_]{2,}", response))
    tool_only_response = tool_like and response.lstrip().startswith(("{", "<tool_call>"))
    is_low_information_tool = (
        (tool_only_response and low_info_action_hits > 0 and response_words < 80)
        or (
            tool_like
            and low_info_action_hits > 0
            and not has_summary_evidence
            and response_words < 80
        )
    )

    if summary_marker_hits > 0 and position_ratio >= 0.7:
        trajectory_stage = "final-summary"
    elif position_ratio <= 0.18 and not trajectory and not has_summary_evidence:
        trajectory_stage = "opener"
    elif is_low_information_tool or (tool_density >= 0.6 and not has_summary_evidence):
        trajectory_stage = "exploration"
    elif workflow_verification_hits > 0 and position_ratio >= 0.5:
        trajectory_stage = "verification"
    else:
        trajectory_stage = "implementation"

    return {
        "trajectory_stage": trajectory_stage,
        "position_ratio": round(position_ratio, 3),
        "tool_density": round(tool_density, 3),
        "is_low_information_tool": is_low_information_tool,
        "summary_marker_hits": summary_marker_hits,
        "has_summary_evidence": has_summary_evidence,
        "evidence_categories": evidence_categories,
        "verification_hits": workflow_verification_hits,
        "selection_view": view,
    }


def _selection_modifier_multiplier(features, *, for_selection):
    """Compute bounded deterministic multiplier for value/selection."""
    stage = features.get("trajectory_stage") or "implementation"
    base = (
        SELECTION_STAGE_SELECTION_MULTIPLIERS if for_selection
        else SELECTION_STAGE_VALUE_MULTIPLIERS
    ).get(stage, 1.0)

    if features.get("is_low_information_tool"):
        base *= SELECTION_LOW_INFO_TOOL_PENALTY

    if stage == "final-summary":
        if features.get("has_summary_evidence"):
            if for_selection:
                base *= SELECTION_SUMMARY_EVIDENCE_BONUS
        else:
            base *= SELECTION_SUMMARY_NO_EVIDENCE_PENALTY

    return max(0.50, min(1.10, base))


def _stability_enabled(config, attr_name, default):
    return getattr(config, attr_name, default) if config else default


def _selection_weights(config=None, intra_weight=None):
    intra = intra_weight if intra_weight is not None else (
        config.selection_intra_weight if config else SELECTION_INTRA_WEIGHT
    )
    quality = (
        config.selection_quality_weight if config else SELECTION_QUALITY_WEIGHT
    )
    rarity = (
        config.selection_rarity_weight if config else SELECTION_RARITY_WEIGHT
    )
    weights = {
        "intra": max(0.0, float(intra or 0.0)),
        "quality": max(0.0, float(quality or 0.0)),
        "rarity": max(0.0, float(rarity or 0.0)),
    }
    total = sum(weights.values())
    if total <= 0:
        return {"intra": 1.0, "quality": 0.0, "rarity": 0.0}
    return {key: value / total for key, value in weights.items()}


def _compose_selection_score(intra_class_rank, pq_scaled, rarity_score, value_score, features,
                             config=None, intra_weight=None):
    """Blend intra-rank, quality, rarity, then apply bounded stage modifiers."""
    if intra_class_rank is None and value_score is not None:
        intra_class_rank = value_score
    components = {
        "intra": _clamp_score(intra_class_rank),
        "quality": _clamp_score(pq_scaled),
        "rarity": _clamp_score(rarity_score),
    }
    weights = _selection_weights(config=config, intra_weight=intra_weight)
    active = {name: score for name, score in components.items() if score is not None}
    if not active:
        return value_score
    active_weight_total = sum(weights[name] for name in active)
    if active_weight_total <= 0:
        return value_score
    base = sum(weights[name] * score for name, score in active.items()) / active_weight_total
    if _stability_enabled(config, "enable_selection_stability", ENABLE_SELECTION_STABILITY):
        base *= _selection_modifier_multiplier(features, for_selection=True)
    return round(max(1.0, min(10.0, base)), 2)


def _apply_value_stability(value_score, features, config=None):
    """Apply deterministic post-LLM value adjustment without changing schema."""
    if value_score is None:
        return None
    if not _stability_enabled(config, "enable_value_stability", ENABLE_VALUE_STABILITY):
        return round(max(1.0, min(10.0, value_score)), 1)
    adjusted = value_score * _selection_modifier_multiplier(features, for_selection=False)
    return round(max(1.0, min(10.0, adjusted)), 1)


def _infer_domain_backfill(view, labels):
    """Conservatively infer a missing domain from strong lexical evidence."""
    if not isinstance(labels, dict):
        return []
    existing = labels.get("domain")
    if isinstance(existing, list) and existing:
        return existing
    text = "\n".join(
        part for part in (
            (view or {}).get("current_request", ""),
            (view or {}).get("trajectory", ""),
            (view or {}).get("response", ""),
        ) if part
    ).lower()
    if not text:
        return []

    scores = {}
    for domain, keywords in _DOMAIN_HINTS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 2 if len(keyword) > 6 else 1
        if domain == "api-development" and "api" in text:
            score += 1 if any(term in text for term in ("endpoint", "request", "response", "rest", "graphql", "dbapi")) else 0
            if not any(term in text for term in ("endpoint", "request", "response", "rest", "graphql", "dbapi")):
                score = max(0, score - 1)
        if domain == "cloud-computing" and any(term in text for term in ("bigquery", "storage api", "s3", "gcs", "lambda", "iam")):
            score += 2
        if score > 0:
            scores[domain] = score

    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_domain, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -1
    if top_score < 3 or top_score <= second_score:
        return []
    return [top_domain]


def _maybe_backfill_domain(sample, config=None):
    """Mutate sample labels with a conservative inferred domain when empty."""
    if not sample or not _stability_enabled(config, "enable_domain_backfill", ENABLE_DOMAIN_BACKFILL):
        return
    labels = sample.get("labels")
    if not isinstance(labels, dict):
        labels = {}
        sample["labels"] = labels
    domains = labels.get("domain")
    if isinstance(domains, list) and domains:
        return
    inferred = _infer_domain_backfill(_selection_view_from_sample(sample, config=config), labels)
    if inferred:
        labels["domain"] = inferred


def _extract_pure_quality(value_dict, quality_weights=None):
    """Compute quality score without rarity from a value dict.

    Uses the same VALUE_WEIGHTS but excludes rarity and renormalizes.
    """
    if quality_weights is None:
        quality_weights = {k: v for k, v in VALUE_WEIGHTS.items() if k != "rarity"}

    components = {}
    c = value_dict.get("complexity", {})
    if isinstance(c, dict) and c.get("overall") is not None:
        components["complexity"] = c["overall"]
    q = value_dict.get("quality", {})
    if isinstance(q, dict) and q.get("overall") is not None:
        components["quality"] = q["overall"]
    r = value_dict.get("reasoning", {})
    if isinstance(r, dict) and r.get("overall") is not None:
        components["reasoning"] = r["overall"]

    if not components:
        return None

    total_w = sum(quality_weights.get(k, 0) for k in components)
    if total_w <= 0:
        return None

    pq = sum(quality_weights.get(k, 0) * v for k, v in components.items()) / total_w

    # Quality floor penalty: consistent with compute_value_score
    quality_overall = components.get("quality")
    if quality_overall is not None and quality_overall < 4:
        pq *= 0.7

    pq = apply_score_confidence(pq, score_confidence(value_dict))
    return pq


def _selection_quality_weights(config=None):
    """Resolve quality-only weights from config.value_weights."""
    base_weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS
    return {k: v for k, v in base_weights.items() if k != "rarity"}


def _selection_dim_confidence(labels, dim):
    """Confidence used for selection shrinkage and fusion."""
    return max(_SELECTION_CONFIDENCE_FLOOR, _dimension_label_confidence(labels, dim))


def _selection_summary_from_sample(sample, config=None):
    """Extract compact fields needed for global selection ranking."""
    value = sample.get("value") or {}
    selection_view = _selection_view_from_sample(sample)
    metadata = sample.get("metadata") or {}
    labels = sample.get("labels") or {}
    selection_features = _infer_selection_features(
        sample=sample,
        summary={
            "selection_view": selection_view,
            "metadata": {
                "turn_index": metadata.get("turn_index"),
                "total_turns": metadata.get("total_turns"),
            },
        },
        config=config,
    )
    selection_features = dict(selection_features or {})
    selection_features.pop("selection_view", None)
    inferred_domain = _infer_domain_backfill(selection_view, labels)
    return {
        "labels": labels,
        "complexity_overall": (value.get("complexity") or {}).get("overall"),
        "quality_overall": (value.get("quality") or {}).get("overall"),
        "reasoning_overall": (value.get("reasoning") or {}).get("overall"),
        "rarity_score": (value.get("rarity") or {}).get("score"),
        "extension_rarity_score": (value.get("rarity_extension") or {}).get("score"),
        "rarity_v2_score": (value.get("rarity_v2") or {}).get("score"),
        "value_score": value.get("value_score"),
        "value_score_v2": value.get("value_score_v2"),
        "confidence": value.get("confidence"),
        "selection_score_v2": value.get("selection_score_v2"),
        "selection_features": selection_features,
        "inferred_domain": inferred_domain,
        "metadata": {
            "turn_index": metadata.get("turn_index"),
            "total_turns": metadata.get("total_turns"),
        },
    }


def _coerce_mean_turn_count(metadata):
    if not isinstance(metadata, dict):
        return 1
    for key in ("total_turns", "trajectory_turn_count"):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def _compute_keep_rates(selection_scores, thresholds=FILE_RANKING_KEEP_RATE_THRESHOLDS):
    scores = [score for score in selection_scores if isinstance(score, (int, float))]
    total = len(scores)
    keep_rates = {}
    for threshold in thresholds:
        key = f"{float(threshold):.1f}"
        keep_rates[key] = round(
            sum(1 for score in scores if score >= float(threshold)) / max(total, 1),
            4,
        ) if total else 0
    return keep_rates


def _compute_percentiles(indexed_values):
    """Assign tie-aware percentiles to (idx, value) pairs."""
    if not indexed_values:
        return {}

    ordered = sorted(indexed_values, key=lambda item: (item[1], item[0]))
    n = len(ordered)
    scale = max(n - 1, 1)
    percentiles = {}

    pos = 0
    while pos < n:
        end = pos
        value = ordered[pos][1]
        while end + 1 < n and ordered[end + 1][1] == value:
            end += 1
        percentile = ((pos + end) / 2) / scale
        for member_pos in range(pos, end + 1):
            percentiles[ordered[member_pos][0]] = percentile
        pos = end + 1

    return percentiles


def compute_selection_scores(samples, rarity_weights=None,
                              intra_weight=None, min_group_size=None,
                              config=None):
    """Compute intra-class quality rankings and selection scores.

    Post-scoring pass (no LLM). For each tag in each dimension, ranks samples
    by pure quality (quality without rarity) within the tag group, then fuses
    across dimensions using rarity_weights.

    Modifies samples in-place: adds selection_score and intra_class_rank to
    each sample's value dict.

    Args:
        samples: list of sample dicts with 'value' and 'labels' fields
        rarity_weights: per-dimension weights for fusion (default: RARITY_WEIGHTS)
        intra_weight: weight for intra-class quality vs rarity (default: 0.75)
        min_group_size: min samples per tag for percentile computation (default: 30)
        config: PipelineConfig override
    """
    summaries = []
    for sample in samples:
        if sample.get("value"):
            summaries.append(_selection_summary_from_sample(sample))
        else:
            summaries.append({
                "labels": sample.get("labels") or {},
                "value_score": (sample.get("value") or {}).get("value_score"),
            })

    selection_results = compute_selection_scores_from_summaries(
        summaries,
        rarity_weights=rarity_weights,
        intra_weight=intra_weight,
        min_group_size=min_group_size,
        config=config,
    )

    for sample, result in zip(samples, selection_results):
        value = sample.get("value")
        if value is None:
            continue
        value["selection_score"] = result["selection_score"]
        value["intra_class_rank"] = result["intra_class_rank"]


def apply_v2_scores(samples, config=None):
    """Add bonus-only V2 value/selection fields without changing legacy outputs."""
    if resolve_extension_rarity_mode(config) != "bonus_only":
        return

    weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS
    quality_weights = _selection_quality_weights(config)

    for sample in samples:
        value = sample.get("value")
        if not isinstance(value, dict):
            continue
        rarity_v2 = value.get("rarity_v2")
        if not isinstance(rarity_v2, dict):
            continue

        score_result = {
            "complexity": value.get("complexity"),
            "quality": value.get("quality"),
            "reasoning": value.get("reasoning"),
            "confidence": value.get("confidence"),
        }
        value_score_v2 = compute_value_score(score_result, rarity_v2, weights)
        summary = _selection_summary_from_sample(sample, config=config)
        features = summary.get("selection_features") or {}
        value_score_v2 = _apply_value_stability(value_score_v2, features, config=config)
        if value_score_v2 is not None:
            value["value_score_v2"] = value_score_v2

        pq_scaled = _extract_pure_quality(value, quality_weights)
        selection_score_v2 = _compose_selection_score(
            value.get("intra_class_rank"),
            pq_scaled,
            rarity_v2.get("score"),
            value_score_v2,
            features,
            config=config,
        )
        if selection_score_v2 is not None:
            value["selection_score_v2"] = selection_score_v2


def compute_selection_scores_from_summaries(summaries, rarity_weights=None,
                                             intra_weight=None,
                                             min_group_size=None, config=None):
    """Compute selection scores from lightweight summary dicts (chunked mode).

    Same algorithm as compute_selection_scores but works on summary dicts
    instead of full sample dicts.

    Args:
        summaries: list of dicts with keys:
            complexity_overall, quality_overall, reasoning_overall,
            rarity_score, labels, value_score
    Returns:
        list of dicts: {selection_score, intra_class_rank} per summary
    """
    if rarity_weights is None:
        rarity_weights = (config.rarity_weights if config and config.rarity_weights
                          else RARITY_WEIGHTS)
    if intra_weight is None:
        intra_weight = (config.selection_intra_weight if config
                        else SELECTION_INTRA_WEIGHT)
    if min_group_size is None:
        min_group_size = (config.selection_min_group_size if config
                          else SELECTION_MIN_GROUP_SIZE)

    quality_weights = _selection_quality_weights(config)
    normalized_labels = []
    features_list = []
    for summary in summaries:
        labels = copy.deepcopy(summary.get("labels") or {})
        inferred = summary.get("inferred_domain")
        if not inferred:
            inferred = _infer_domain_backfill(summary.get("selection_view"), labels)
        if inferred and not labels.get("domain"):
            labels["domain"] = inferred
        normalized_labels.append(labels)
        features = summary.get("selection_features")
        if not isinstance(features, dict):
            features = _infer_selection_features(summary=summary, config=config)
        features_list.append(features)

    label_confidences = [
        {dim: _selection_dim_confidence(labels, dim) for dim in _SELECTION_DIMS}
        for labels in normalized_labels
    ]

    # Step 1: pure_quality from summary fields
    pure_qualities = []
    for s in summaries:
        components = {}
        if s.get("complexity_overall") is not None:
            components["complexity"] = s["complexity_overall"]
        if s.get("quality_overall") is not None:
            components["quality"] = s["quality_overall"]
        if s.get("reasoning_overall") is not None:
            components["reasoning"] = s["reasoning_overall"]
        if components:
            tw = sum(quality_weights.get(k, 0) for k in components)
            pq = sum(quality_weights.get(k, 0) * v for k, v in components.items()) / tw if tw > 0 else None
            # Quality floor penalty: consistent with compute_value_score
            if pq is not None and components.get("quality") is not None and components["quality"] < 4:
                pq *= 0.7
            pq = apply_score_confidence(pq, score_confidence(s))
        else:
            pq = None
        pure_qualities.append(pq)

    # Step 2: Global percentile (needed for Bayesian shrinkage)
    valid_pq = [(i, pq) for i, pq in enumerate(pure_qualities) if pq is not None]
    global_percentiles = _compute_percentiles(valid_pq)

    # Step 3: Per-tag percentile with Bayesian shrinkage
    smoothing_prior = (config.selection_smoothing_prior if config
                       else SELECTION_SMOOTHING_PRIOR)
    dim_percentiles = [{} for _ in summaries]

    for dim in _SELECTION_DIMS:
        tag_groups = {}
        for i, s in enumerate(summaries):
            if pure_qualities[i] is None:
                continue
            labels = normalized_labels[i]
            if not is_usable_labels(labels):
                continue
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_groups.setdefault(t, []).append((i, pure_qualities[i]))
            else:
                tag_groups.setdefault(tags, []).append((i, pure_qualities[i]))

        for tag, members in tag_groups.items():
            if len(members) < min_group_size:
                continue
            effective_n = sum(label_confidences[idx].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                              for idx, _pq in members)
            shrinkage = effective_n / (effective_n + smoothing_prior)
            tag_percentiles = _compute_percentiles(members)
            for idx, _pq in members:
                tag_pct = tag_percentiles[idx]
                g_pct = global_percentiles.get(idx, 0.5)
                sample_conf = label_confidences[idx].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                percentile = (
                    sample_conf * (shrinkage * tag_pct + (1 - shrinkage) * g_pct)
                    + (1 - sample_conf) * g_pct
                )
                # Multi-select: average across tags in this dimension
                if dim in dim_percentiles[idx]:
                    prev, cnt = dim_percentiles[idx][dim]
                    dim_percentiles[idx][dim] = (prev + percentile, cnt + 1)
                else:
                    dim_percentiles[idx][dim] = (percentile, 1)

    # Step 4: Fusion
    results = []
    for i, s in enumerate(summaries):
        percs = dim_percentiles[i]

        if percs:
            weighted_sum = 0.0
            weight_total = 0.0
            for dim, (pct_sum, pct_cnt) in percs.items():
                w = rarity_weights.get(dim, 1.0) * label_confidences[i].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                weighted_sum += w * (pct_sum / pct_cnt)  # mean percentile
                weight_total += w
            if weight_total > 0:
                fused = weighted_sum / weight_total
                intra_class_rank = round(fused * 9 + 1, 2)
            else:
                intra_class_rank = None
        else:
            if i in global_percentiles:
                intra_class_rank = round(global_percentiles[i] * 9 + 1, 2)
            else:
                intra_class_rank = None

        rarity_score = s.get("rarity_score")
        pq = pure_qualities[i]
        pq_scaled = max(1.0, min(10.0, pq)) if pq is not None else None
        selection_score = _compose_selection_score(
            intra_class_rank,
            pq_scaled,
            rarity_score,
            s.get("value_score"),
            features_list[i],
            config=config,
            intra_weight=intra_weight,
        )

        results.append({
            "selection_score": selection_score,
            "intra_class_rank": intra_class_rank,
        })

    return results


# ─────────────────────────────────────────────────────────
# Per-sample scoring
# ─────────────────────────────────────────────────────────

_SELECTIVE_COMPLEXITY_PRIOR = {
    "beginner": 3.5,
    "intermediate": 5.0,
    "upper-intermediate": 6.0,
    "advanced": 7.0,
    "expert": 8.0,
}
_DISABLED_SELECTIVE_POLICIES = {"off", "disabled", "all", "full"}


def _resolve_selective_scoring_policy(config=None):
    enabled = getattr(config, "enable_selective_scoring", ENABLE_SELECTIVE_SCORING)
    if not enabled:
        return None
    policy = getattr(config, "selective_scoring_policy", SELECTIVE_SCORING_POLICY)
    if policy is None:
        return None
    policy = str(policy).strip().lower()
    if not policy or policy in _DISABLED_SELECTIVE_POLICIES:
        return None
    return policy


def _required_turn_anchors(total_turns, drift_interval):
    """Return required turn indices for adaptive selective scoring."""
    if total_turns <= 0:
        return set()
    anchors = {1, total_turns}
    step = max(drift_interval, 1)
    for idx in range(1, total_turns + 1, step):
        anchors.add(idx)
    return anchors


def _selective_scoring_decision(sample, config=None):
    """Decide whether this sample requires an LLM scoring call."""
    policy = _resolve_selective_scoring_policy(config)
    if policy is None:
        return {
            "requires_llm": True,
            "policy": "full",
            "reason": "selective_scoring_disabled",
            "anchor_distance": None,
        }

    if policy != "multiturn_adaptive_v1":
        return {
            "requires_llm": True,
            "policy": policy,
            "reason": "unknown_policy_fallback",
            "anchor_distance": None,
        }

    metadata = sample.get("metadata") or {}
    labels = sample.get("labels") or {}
    source_id = metadata.get("source_id")
    total_turns = _coerce_positive_int(metadata.get("total_turns"))
    turn_index = _coerce_positive_int(metadata.get("turn_index"))
    min_turns = max(
        2,
        _coerce_positive_int(getattr(config, "selective_scoring_min_turns", SELECTIVE_SCORING_MIN_TURNS)),
    )
    drift_interval = max(
        1,
        _coerce_positive_int(getattr(config, "selective_scoring_drift_interval", SELECTIVE_SCORING_DRIFT_INTERVAL)),
    )

    if not source_id or total_turns <= 1 or turn_index <= 0:
        return {
            "requires_llm": True,
            "policy": policy,
            "reason": "missing_multiturn_metadata",
            "anchor_distance": None,
        }

    if total_turns < min_turns:
        return {
            "requires_llm": True,
            "policy": policy,
            "reason": "below_min_turns_threshold",
            "anchor_distance": None,
        }

    if not labels.get("inherited"):
        return {
            "requires_llm": True,
            "policy": policy,
            "reason": "non_inherited_turn",
            "anchor_distance": None,
        }

    anchors = _required_turn_anchors(total_turns, drift_interval)
    if turn_index in anchors:
        return {
            "requires_llm": True,
            "policy": policy,
            "reason": "required_anchor_turn",
            "anchor_distance": 0,
        }

    anchor_distance = min(abs(turn_index - idx) for idx in anchors) if anchors else None
    return {
        "requires_llm": False,
        "policy": policy,
        "reason": "inherited_mid_turn_estimate",
        "anchor_distance": anchor_distance,
    }


def _build_selective_estimate_score(sample, rarity_result, decision, config=None):
    """Build conservative pseudo scores for non-LLM selective scoring turns."""
    labels = sample.get("labels") or {}
    difficulty = labels.get("difficulty")
    difficulty_key = str(difficulty).strip().lower() if isinstance(difficulty, str) else ""
    complexity_prior = _SELECTIVE_COMPLEXITY_PRIOR.get(difficulty_key, 5.0)
    anchor_distance = decision.get("anchor_distance") or 0
    inherited = bool(labels.get("inherited"))

    complexity = complexity_prior - min(anchor_distance * 0.15, 1.0)
    quality = 5.0 - (0.8 if inherited else 0.0) - min(anchor_distance * 0.30, 1.2)
    reasoning = complexity - 0.6 - min(anchor_distance * 0.12, 0.5)

    complexity = round(max(1.0, min(10.0, complexity)), 1)
    quality = round(max(1.0, min(10.0, quality)), 1)
    reasoning = round(max(1.0, min(10.0, reasoning)), 1)

    conf_cap = getattr(
        config,
        "selective_scoring_estimate_confidence_cap",
        SELECTIVE_SCORING_ESTIMATE_CONFIDENCE_CAP,
    )
    if not isinstance(conf_cap, (int, float)) or isinstance(conf_cap, bool):
        conf_cap = SELECTIVE_SCORING_ESTIMATE_CONFIDENCE_CAP
    conf_cap = max(0.2, min(1.0, float(conf_cap)))

    label_conf = _sample_label_confidence(labels, default=0.65)
    confidence = min(conf_cap, label_conf * 0.72)
    if inherited:
        confidence -= 0.06
    confidence -= min(anchor_distance * 0.03, 0.2)
    confidence = round(max(0.2, min(conf_cap, confidence)), 2)

    estimated_score = {
        "complexity": {
            "instruction": complexity,
            "analytical_depth": complexity,
            "implementation": complexity,
            "overall": complexity,
        },
        "quality": {
            "correctness": quality,
            "code_quality": quality,
            "explanation": quality,
            "completeness": quality,
            "overall": quality,
        },
        "reasoning": {
            "clarity": reasoning,
            "consistency": reasoning,
            "self_correction": False,
            "overall": reasoning,
        },
        "confidence": confidence,
    }

    weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS
    value_score = compute_value_score(estimated_score, rarity_result, weights)
    features = _infer_selection_features(sample=sample, config=config)
    value_score = _apply_value_stability(value_score, features, config=config)

    value = {
        "complexity": estimated_score["complexity"],
        "quality": estimated_score["quality"],
        "reasoning": estimated_score["reasoning"],
        "rarity": rarity_result,
        "flags": [],
        "value_score": value_score,
        "confidence": confidence,
        "estimation": {
            "llm_scored": False,
            "policy": decision.get("policy"),
            "reason": decision.get("reason"),
            "anchor_distance": decision.get("anchor_distance"),
            "uncertainty": round(1.0 - confidence, 2),
        },
    }
    return _attach_augmented_rarity_fields(value, rarity_result)


async def score_one(http_client, sample, model, rarity_result,
                    sample_idx, total, sem, config=None, rate_limiter=None):
    """Score a single sample: truncate, call LLM, validate, compute value_score.

    Retry loop is OUTSIDE the semaphore so failed/retrying samples don't block
    other tasks from acquiring a slot (matching Pass 1's label_one pattern).

    Returns (value_dict, monitor_dict).
    """
    from sft_label.prompts_value import build_scoring_messages

    _max_retries = config.sample_max_retries if config else SAMPLE_MAX_RETRIES
    _weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS
    runtime = getattr(config, "_adaptive_runtime", None) if config is not None else None
    request_quick_retries = getattr(config, "request_quick_retries", 1) if config is not None else 1

    sample_id = sample.get("id", f"sample-{sample_idx}")
    conversations = sample.get("conversations", [])
    labels = sample.get("labels") or {}
    metadata = sample.get("metadata") or {}

    _maybe_backfill_domain(sample, config=config)
    labels = sample.get("labels") or labels

    monitor = {
        "sample_id": sample_id,
        "llm_calls": 0,
        "status": "pending",
        "attempts": 0,
        "error": None,
        "validation_issues": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    if is_partial_labels(labels):
        monitor["status"] = "skipped_partial_labels"
        monitor["error"] = labels.get("partial_reason", "partial labels")
        return None, monitor

    # Detect thinking mode: prefer metadata (Pangu COT stripped during Pass 1),
    # fallback to scanning conversations (ShareGPT retains COT markers)
    thinking_mode = metadata.get("thinking_mode") or detect_thinking_mode(conversations)

    # Extract COT: prefer metadata.cot_text (preserved from Pangu preprocessing),
    # fallback to extracting from conversations (ShareGPT)
    saved_cot = metadata.get("cot_text", "")
    if saved_cot:
        cot_text = saved_cot
    else:
        cot_text, _, _ = extract_cot_content(conversations)

    selective_decision = _selective_scoring_decision(sample, config=config)
    if not selective_decision.get("requires_llm", True):
        value = _build_selective_estimate_score(
            sample,
            rarity_result,
            selective_decision,
            config=config,
        )
        value["thinking_mode"] = thinking_mode
        monitor["status"] = "estimated_selective"
        monitor["selective_scoring"] = {
            "policy": selective_decision.get("policy"),
            "reason": selective_decision.get("reason"),
            "anchor_distance": selective_decision.get("anchor_distance"),
        }
        return value, monitor

    # Truncate for scoring
    _compact, _budget = _resolved_scoring_prompt_budget(config)
    truncated = truncate_for_scoring(
        conversations, thinking_mode, cot_text=cot_text,
        budget=_budget,
    )

    # Build messages
    code_block_count = count_code_blocks(
        " ".join(t.get("value", "") for t in conversations)
    )
    total_turns = len(conversations)

    # Multi-turn slice position (for incomplete flag calibration)
    slice_position = metadata.get("slice_position")
    slice_count = metadata.get("slice_count")
    turn_index = metadata.get("turn_index")
    total_turns_meta = metadata.get("total_turns")

    messages = build_scoring_messages(
        truncated=truncated,
        thinking_mode=thinking_mode,
        labels=labels,
        total_turns=total_turns,
        code_block_count=code_block_count,
        enable_rationale=config.enable_rationale if config else False,
        slice_position=slice_position,
        slice_count=slice_count,
        turn_index=turn_index,
        total_turns_meta=total_turns_meta,
        compact=_compact,
    )

    # LLM call with retry — retry OUTSIDE admission gates so failed/retrying samples
    # do not block other tasks from acquiring a slot.
    for attempt in range(_max_retries):
        if attempt > 0:
            # Backoff outside semaphore so slot is free for others
            base_wait = 2 ** attempt * 2
            await asyncio.sleep(base_wait + random.uniform(0, base_wait))

        permit = None
        t0 = time.perf_counter()
        if runtime is not None:
            async with _runtime_permit(runtime, stage="pass2_score", sample_id=sample_id) as permit:
                monitor["attempts"] = attempt + 1
                parsed, raw, usage = await async_llm_call(
                    http_client,
                    messages,
                    model,
                    temperature=0.1,
                    max_tokens=800,
                    max_retries=max(int(request_quick_retries), 0),
                    config=config,
                    rate_limiter=None,
                )
        else:
            async with sem:
                monitor["attempts"] = attempt + 1
                parsed, raw, usage = await async_llm_call(
                    http_client,
                    messages,
                    model,
                    temperature=0.1,
                    max_tokens=800,
                    max_retries=max(int(request_quick_retries), 0),
                    config=config,
                    rate_limiter=rate_limiter,
                )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        monitor["llm_calls"] += 1
        monitor["prompt_tokens"] += (usage or {}).get("prompt_tokens", 0)
        monitor["completion_tokens"] += (usage or {}).get("completion_tokens", 0)
        monitor["last_latency_ms"] = round(latency_ms, 2)
        if permit is not None and hasattr(permit, "queue_wait_ms"):
            monitor["queue_wait_ms"] = round(float(getattr(permit, "queue_wait_ms", 0.0) or 0.0), 2)
            monitor["runtime_state"] = getattr(permit, "state_at_acquire", None) or getattr(runtime, "state", None)
        elif runtime is not None:
            monitor["runtime_state"] = getattr(runtime, "state", None)

        if parsed is None:
            monitor["error"] = raw[:300] if raw else "null response"
            monitor["error_response"] = usage.get("error_response") or (raw[:500] if raw else "")
            # Detect COT mimicry — LLM echoed COT format instead of JSON
            if raw and raw.lstrip().startswith(("```COT", "```cot", "```Cot",
                                                "[COT]", "COT\n", "«cot»")):
                monitor["error"] = f"cot_mimicry: {raw[:200]}"
            error_class, retryable_infra, outcome = _classify_pass2_attempt(
                parsed, raw, usage, None, latency_ms=latency_ms
            )
            monitor["error_class"] = error_class
            monitor["retryable_infra"] = retryable_infra
            if outcome is not None:
                monitor["http_status"] = outcome.status_code
            _notify_runtime(runtime, outcome)
            if usage.get("non_retryable") and not retryable_infra:
                break
            continue

        score_result, issues = validate_score_response(parsed)
        monitor["validation_issues"] = issues

        value_score = compute_value_score(score_result, rarity_result, _weights)
        if value_score is None:
            # Treat structurally invalid / unusable scoring outputs as abnormal infra failures
            # so recovery sweeps can retry them. (validate_score_response returns a dict even
            # when all scores are None.)
            monitor["error"] = "abnormal_response: insufficient valid scores"
            error_class, retryable_infra, outcome = _classify_pass2_attempt(
                parsed, raw, usage, None, latency_ms=latency_ms
            )
            monitor["error_class"] = error_class
            monitor["retryable_infra"] = retryable_infra
            if outcome is not None:
                monitor["http_status"] = outcome.status_code
            _notify_runtime(runtime, outcome)
            continue

        # Success (valid usable value_score)
        error_class, retryable_infra, outcome = _classify_pass2_attempt(
            parsed, raw, usage, score_result, latency_ms=latency_ms
        )
        monitor["error_class"] = error_class
        monitor["retryable_infra"] = retryable_infra
        if outcome is not None:
            monitor["http_status"] = outcome.status_code
        _notify_runtime(runtime, outcome)
        features = _infer_selection_features(
            sample=sample,
            summary={
                "selection_view": {
                    "current_request": truncated.get("current_request") or truncated.get("instruction") or "",
                    "trajectory": truncated.get("trajectory") or "",
                    "response": truncated.get("response") or "",
                    "trajectory_turn_count": truncated.get("trajectory_turn_count", 0),
                    "trajectory_tool_turns": truncated.get("trajectory_tool_turns", 0),
                },
                "metadata": {
                    "turn_index": metadata.get("turn_index"),
                    "total_turns": metadata.get("total_turns"),
                },
            },
            config=config,
        )
        value_score = _apply_value_stability(value_score, features, config=config)

        value = {
            "complexity": score_result.get("complexity"),
            "quality": score_result.get("quality"),
            "reasoning": score_result.get("reasoning"),
            "rarity": rarity_result,
            "flags": score_result.get("flags", []),
            "thinking_mode": thinking_mode,
            "value_score": value_score,
            "confidence": score_result.get("confidence", 0.5),
        }
        monitor["status"] = "success"
        return _attach_augmented_rarity_fields(value, rarity_result), monitor

    # All retries exhausted
    monitor["status"] = "failed"
    return None, monitor


# ─────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────

def _percentiles(values, pcts=(10, 25, 50, 75, 90)):
    """Compute percentiles from a list of values."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    result = {"mean": round(sum(s) / n, 2), "std": 0.0, "min": s[0], "max": s[-1]}
    if n > 1:
        mean = result["mean"]
        result["std"] = round((sum((x - mean) ** 2 for x in s) / (n - 1)) ** 0.5, 2)
    for p in pcts:
        idx = min(int(n * p / 100), n - 1)
        result[f"p{p}"] = round(s[idx], 2)
    return result


def _histogram_bins(values):
    """Compute histogram bins (1-10) from a list of numeric values."""
    bins = [0] * 10
    for v in values:
        if isinstance(v, (int, float)) and 1 <= v <= 10:
            idx = min(int(v) - 1, 9)
            bins[idx] += 1
    return bins


def _percentiles_from_histogram_bins(bins, pcts=(10, 25, 50, 75, 90)):
    """Approximate percentiles from integer-score histogram bins."""
    if not bins or len(bins) != 10:
        return {}
    total = sum(bins)
    if total <= 0:
        return {}

    mean = sum((idx + 1) * count for idx, count in enumerate(bins)) / total
    result = {
        "mean": round(mean, 2),
        "std": 0.0,
        "min": next((idx + 1 for idx, count in enumerate(bins) if count > 0), 0),
        "max": next((10 - idx for idx, count in enumerate(reversed(bins)) if count > 0), 0),
    }
    if total > 1:
        variance = sum(count * ((idx + 1) - mean) ** 2 for idx, count in enumerate(bins)) / (total - 1)
        result["std"] = round(variance ** 0.5, 2)

    for pct in pcts:
        target = min(int(total * pct / 100), total - 1)
        seen = 0
        score = 10
        for idx, count in enumerate(bins):
            seen += count
            if seen > target:
                score = idx + 1
                break
        result[f"p{pct}"] = round(float(score), 2)
    return result


def compute_value_stats(scored_samples, all_monitors, include_raw_scores=True):
    """Compute aggregate statistics for value scoring.

    Returns dict matching Pass 2 stats structure.
    """
    values = [s.get("value", {}) for s in scored_samples if s.get("value")]

    total_scored = len(values)
    total_failed = sum(1 for m in all_monitors if m.get("status") == "failed")
    total_estimated = sum(1 for m in all_monitors if m.get("status") == "estimated_selective")

    # Score distributions
    def extract_scores(key_path):
        result = []
        for v in values:
            obj = v
            for k in key_path.split("."):
                obj = obj.get(k, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if isinstance(obj, (int, float)):
                result.append(obj)
        return result

    score_distributions = {
        "value_score": _percentiles(extract_scores("value_score")),
        "selection_score": _percentiles(extract_scores("selection_score")),
        "intra_class_rank": _percentiles(extract_scores("intra_class_rank")),
        "complexity_overall": _percentiles(extract_scores("complexity.overall")),
        "quality_overall": _percentiles(extract_scores("quality.overall")),
        "reasoning_overall": _percentiles(extract_scores("reasoning.overall")),
        "rarity_score": _percentiles(extract_scores("rarity.score")),
        "extension_rarity_score": _percentiles(extract_scores("rarity_extension.score")),
        "rarity_v2_score": _percentiles(extract_scores("rarity_v2.score")),
        "value_score_v2": _percentiles(extract_scores("value_score_v2")),
        "selection_score_v2": _percentiles(extract_scores("selection_score_v2")),
    }

    # Distribution bias detection (quality.overall)
    distribution_warnings = []
    quality_scores = extract_scores("quality.overall")
    if len(quality_scores) >= 20:
        n_q = len(quality_scores)
        buckets = {
            "1-3": sum(1 for s in quality_scores if 1 <= s <= 3) / n_q * 100,
            "4-6": sum(1 for s in quality_scores if 4 <= s <= 6) / n_q * 100,
            "7-8": sum(1 for s in quality_scores if 7 <= s <= 8) / n_q * 100,
            "9-10": sum(1 for s in quality_scores if 9 <= s <= 10) / n_q * 100,
        }
        expected = {"1-3": 15, "4-6": 50, "7-8": 30, "9-10": 5}
        for bucket, actual_pct in buckets.items():
            exp_pct = expected[bucket]
            deviation = actual_pct - exp_pct
            if abs(deviation) > 20:
                direction = "over" if deviation > 0 else "under"
                distribution_warnings.append(
                    f"quality.overall bucket {bucket}: {actual_pct:.1f}% "
                    f"({direction}-represented vs expected {exp_pct}%, "
                    f"deviation={deviation:+.1f}pp)"
                )
    score_distributions["distribution_warnings"] = distribution_warnings

    # Confidence distribution
    confidence_scores = extract_scores("confidence")
    score_distributions["confidence"] = _percentiles(confidence_scores)

    score_arrays = {
        "value_score": extract_scores("value_score"),
        "selection_score": extract_scores("selection_score"),
        "intra_class_rank": extract_scores("intra_class_rank"),
        "complexity_overall": extract_scores("complexity.overall"),
        "quality_overall": extract_scores("quality.overall"),
        "reasoning_overall": extract_scores("reasoning.overall"),
        "rarity_score": extract_scores("rarity.score"),
        "extension_rarity_score": extract_scores("rarity_extension.score"),
        "rarity_v2_score": extract_scores("rarity_v2.score"),
        "value_score_v2": extract_scores("value_score_v2"),
        "selection_score_v2": extract_scores("selection_score_v2"),
    }

    # Histogram bins (1-10) for dashboard charts
    histograms = {k: _histogram_bins(v) for k, v in score_arrays.items()}

    # Sub-score means
    sub_score_means = {
        "complexity": {},
        "quality": {},
        "reasoning": {},
    }
    for dim in ("complexity", "quality", "reasoning"):
        obj_template = values[0].get(dim, {}) if values else {}
        for key in obj_template:
            scores = extract_scores(f"{dim}.{key}")
            scores = [s for s in scores if isinstance(s, (int, float))]
            if scores:
                sub_score_means[dim][key] = round(sum(scores) / len(scores), 2)

    # Value by tag (mean value per tag per dimension)
    tag_dims = ["intent", "difficulty", "domain", "concept", "task", "agentic", "constraint", "context"]
    value_by_tag = {}
    selection_by_tag = {}
    for dim in tag_dims:
        tag_values = {}  # {tag: [value_scores]}
        tag_selections = {}  # {tag: [selection_scores]}
        for s in scored_samples:
            v = s.get("value", {})
            vs = v.get("value_score")
            ss = v.get("selection_score")
            if vs is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_values.setdefault(t, []).append(vs)
                    if ss is not None:
                        tag_selections.setdefault(t, []).append(ss)
            else:
                tag_values.setdefault(tags, []).append(vs)
                if ss is not None:
                    tag_selections.setdefault(tags, []).append(ss)
        value_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_values.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }
        selection_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_selections.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }

    # Thinking mode stats
    slow_values = [v for v in values if v.get("thinking_mode") == "slow"]
    fast_values = [v for v in values if v.get("thinking_mode") == "fast"]
    thinking_mode_stats = {
        "slow": {
            "count": len(slow_values),
            "mean_value": round(sum(v.get("value_score", 0) or 0 for v in slow_values) / max(len(slow_values), 1), 2),
            "mean_quality": round(sum((v.get("quality", {}).get("overall") or 0) for v in slow_values) / max(len(slow_values), 1), 2),
            "mean_reasoning": round(sum((v.get("reasoning", {}).get("overall") or 0) for v in slow_values) / max(len(slow_values), 1), 2),
        },
        "fast": {
            "count": len(fast_values),
            "mean_value": round(sum(v.get("value_score", 0) or 0 for v in fast_values) / max(len(fast_values), 1), 2),
            "mean_quality": round(sum((v.get("quality", {}).get("overall") or 0) for v in fast_values) / max(len(fast_values), 1), 2),
            "mean_reasoning": round(sum((v.get("reasoning", {}).get("overall") or 0) for v in fast_values) / max(len(fast_values), 1), 2),
        },
    }

    # Flag counts and impact
    flag_counts = {}
    flag_value_sums = {}
    for v in values:
        for f in v.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
            flag_value_sums.setdefault(f, []).append(v.get("value_score", 0) or 0)

    flag_value_impact = {
        f: {"mean_value": round(sum(vs) / len(vs), 2), "count": len(vs)}
        for f, vs in flag_value_sums.items()
    }

    # Selection thresholds
    all_value_scores = sorted([v.get("value_score") for v in values if v.get("value_score") is not None])
    n_vs = len(all_value_scores)
    selection_thresholds = {}
    for pct_label, pct in [("top_10pct", 90), ("top_25pct", 75), ("top_50pct", 50)]:
        if n_vs > 0:
            idx = min(int(n_vs * pct / 100), n_vs - 1)
            threshold = all_value_scores[idx]
            count = sum(1 for v in all_value_scores if v >= threshold)
            selection_thresholds[pct_label] = {"threshold": round(threshold, 1), "count": count}

    # Coverage at thresholds
    coverage_at_thresholds = {}
    for threshold_val in [5.0, 6.0, 7.0, 8.0]:
        retained = [s for s in scored_samples
                    if s.get("value", {}).get("value_score") is not None
                    and s["value"]["value_score"] >= threshold_val]
        all_tags = set()
        retained_tags = set()
        for s in scored_samples:
            labels = s.get("labels") or {}
            for dim in tag_dims:
                t = labels.get(dim)
                if isinstance(t, list):
                    all_tags.update(f"{dim}:{x}" for x in t)
                elif t:
                    all_tags.add(f"{dim}:{t}")
        for s in retained:
            labels = s.get("labels") or {}
            for dim in tag_dims:
                t = labels.get(dim)
                if isinstance(t, list):
                    retained_tags.update(f"{dim}:{x}" for x in t)
                elif t:
                    retained_tags.add(f"{dim}:{t}")

        lost = sorted(all_tags - retained_tags)
        coverage_at_thresholds[str(threshold_val)] = {
            "retained": len(retained),
            "pct": round(len(retained) / max(total_scored, 1), 2),
            "tags_lost": lost[:20],
            "coverage": round(len(retained_tags) / max(len(all_tags), 1), 3),
        }

    keep_rates = _compute_keep_rates([value.get("selection_score") for value in values])
    keep_rate_7 = keep_rates.get(f"{FILE_RANKING_KEEP_RATE_THRESHOLD:.1f}", 0)
    turn_counts = [
        turn_count
        for sample in scored_samples
        if (turn_count := _coerce_mean_turn_count(sample.get("metadata"))) is not None
    ]
    mean_turns = round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0

    # LLM usage
    total_llm_calls = sum(m.get("llm_calls", 0) for m in all_monitors)
    total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in all_monitors)
    total_completion_tokens = sum(m.get("completion_tokens", 0) for m in all_monitors)

    stats = {
        "total_scored": total_scored,
        "total_failed": total_failed,
        "total_estimated": total_estimated,
        "total_llm_calls": total_llm_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "score_distributions": score_distributions,
        "histograms": histograms,
        "sub_score_means": sub_score_means,
        "value_by_tag": value_by_tag,
        "selection_by_tag": selection_by_tag,
        "thinking_mode_stats": thinking_mode_stats,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "flag_value_impact": flag_value_impact,
        "selection_thresholds": selection_thresholds,
        "coverage_at_thresholds": coverage_at_thresholds,
        "keep_rates": keep_rates,
        "keep_rate_7": keep_rate_7,
        "mean_turns": mean_turns,
    }
    if include_raw_scores:
        stats["_raw_scores"] = score_arrays
    return stats


# ─────────────────────────────────────────────────────────
# Pipeline entry points
# ─────────────────────────────────────────────────────────

def _inline_dashboard_filename(layout: InlineScoringTarget, source_file: Path) -> str:
    rel = layout.layout.relative_source_path(source_file).with_suffix("")
    stem = "__".join(rel.parts) if rel.parts else source_file.stem
    return f"dashboard_scoring_{stem}.html"


def _load_scored_artifact_samples(artifact_dir: Path):
    """Load scored samples from a per-file artifact directory."""
    artifact_dir = Path(artifact_dir)
    json_path = artifact_dir / "scored.json"
    jsonl_path = artifact_dir / "scored.jsonl"
    if jsonl_path.exists():
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []


def _load_monitor_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    path = Path(path)
    if not path.exists():
        return lookup
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            monitor = json.loads(line)
            sample_id = monitor.get("sample_id")
            if sample_id:
                lookup[sample_id] = monitor
    return lookup


def _load_resumed_value_cache(output_dir: Path) -> dict[str, dict]:
    """Load existing scored values keyed by sample id from scored.jsonl."""
    resumed_values: dict[str, dict] = {}
    scored_jsonl_path = Path(output_dir) / "scored.jsonl"
    if not scored_jsonl_path.exists():
        return resumed_values

    with open(scored_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scored_sample = json.loads(line)
            sample_id = scored_sample.get("id", "")
            value = scored_sample.get("value")
            if sample_id and isinstance(value, dict):
                resumed_values[sample_id] = value
    return resumed_values


def _existing_resume_value(sample: dict, resumed_values: dict[str, dict]) -> dict | None:
    """Return an existing score from inline payload or an on-disk scored cache."""
    embedded_value = sample.get("value")
    if isinstance(embedded_value, dict):
        return copy.deepcopy(embedded_value)

    sample_id = sample.get("id", "")
    cached_value = resumed_values.get(sample_id)
    if isinstance(cached_value, dict):
        return copy.deepcopy(cached_value)

    legacy_sample_id = (sample.get("metadata") or {}).get("legacy_sample_id", "")
    if legacy_sample_id and legacy_sample_id != sample_id:
        cached_value = resumed_values.get(legacy_sample_id)
        if isinstance(cached_value, dict):
            return copy.deepcopy(cached_value)
    return None


def _looks_like_legacy_scoring_run_dir(input_path) -> bool:
    """Return True when a directory already looks like a standard run output."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        return False
    for marker in (
        PASS1_SUMMARY_STATS_FILE,
        PASS2_SUMMARY_STATS_FILE,
        PASS1_STATS_FILE,
        PASS2_STATS_FILE,
        "labeled.json",
        "labeled.jsonl",
        "scored.json",
        "scored.jsonl",
    ):
        if (input_path / marker).exists():
            return True
    return False


def _resumed_monitor(sample_id: str) -> dict:
    """Synthetic scoring monitor for resume-skipped samples."""
    return {
        "sample_id": sample_id,
        "status": "resumed",
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "attempts": 0,
    }


def _sync_inline_scored_cache_to_dataset(source_file: Path, artifact_dir: Path, limit: int = 0):
    """Copy scored cache results back into the mirrored inline dataset rows."""
    source_file = Path(source_file)
    artifact_dir = Path(artifact_dir)
    scored_samples = _load_scored_artifact_samples(artifact_dir)
    bundles, _samples, sample_to_bundle = load_inline_scoring_file(source_file, limit=limit)
    if len(scored_samples) != len(sample_to_bundle):
        raise ValueError(
            f"Scored cache/sample mismatch for {source_file}: "
            f"{len(scored_samples)} scored vs {len(sample_to_bundle)} scoreable turns"
        )

    samples_by_bundle = [[] for _ in bundles]
    for sample, bundle_idx in zip(scored_samples, sample_to_bundle):
        samples_by_bundle[bundle_idx].append(sample)

    monitor_lookup = _load_monitor_lookup(artifact_dir / "monitor_value.jsonl")
    updated_rows = {}
    conversation_records = []
    for bundle_idx, bundle in enumerate(bundles):
        updated_row, conversation = update_inline_row_with_scored_samples(
            bundle.raw_row,
            samples_by_bundle[bundle_idx],
            monitor_lookup=monitor_lookup,
        )
        updated_rows[bundle.row_number] = updated_row
        if conversation and len(samples_by_bundle[bundle_idx]) >= 2:
            conversation_records.append(conversation)

    tmp_path = source_file.with_name(f".{source_file.name}.tmp")
    with open(source_file, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for row_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = updated_rows.get(row_number)
            if payload is None:
                payload = json.loads(stripped)
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(tmp_path, source_file)

    if conversation_records:
        _write_json_atomic(artifact_dir / "conversation_scores.json", conversation_records)
    else:
        conv_path = artifact_dir / "conversation_scores.json"
        if conv_path.exists():
            conv_path.unlink()

    return scored_samples, conversation_records


def _generate_inline_file_dashboard(target: InlineScoringTarget, source_file: Path):
    artifact_dir = target.layout.file_artifact_dir(source_file)
    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        scored_name = "scored.json" if (artifact_dir / "scored.json").exists() else "scored.jsonl"
        generate_value_dashboard(
            artifact_dir,
            scored_file=scored_name,
            stats_file=PASS2_STATS_FILE,
            output_file=str(target.layout.dashboard_path(_inline_dashboard_filename(target, source_file))),
            quiet=True,
        )
    except Exception:
        pass


async def _run_inline_scoring_file(target: InlineScoringTarget, source_file: Path,
                                   tag_stats_path, limit, config, resume=False,
                                   llm_progress_cb=None):
    """Score one mirrored inline JSONL file via rebuildable meta caches."""
    artifact_dir = target.layout.file_artifact_dir(source_file)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _bundles, cached_samples = write_inline_labeled_cache(source_file, artifact_dir, limit=limit)

    effective_tag_stats = tag_stats_path
    if effective_tag_stats is None:
        candidate = artifact_dir / PASS1_STATS_FILE
        if candidate.exists():
            effective_tag_stats = str(candidate)
        else:
            summary_candidate = target.layout.meta_root / PASS1_SUMMARY_STATS_FILE
            if summary_candidate.exists():
                effective_tag_stats = str(summary_candidate)

    stats = await _run_scoring_file(
        artifact_dir / "labeled.jsonl",
        artifact_dir,
        effective_tag_stats,
        limit,
        config,
        resume=resume,
        llm_progress_cb=llm_progress_cb,
        file_label=target.layout.relative_source_path(source_file).as_posix(),
    )

    scored_samples, conversation_records = _sync_inline_scored_cache_to_dataset(
        source_file,
        artifact_dir,
        limit=limit,
    )
    _generate_inline_file_dashboard(target, source_file)

    stats["input_file"] = str(source_file)
    stats["mirrored_file"] = str(source_file)
    stats["file"] = target.layout.relative_source_path(source_file).as_posix()
    stats["total_scored"] = len([sample for sample in scored_samples if sample.get("value")])
    if conversation_records:
        stats["conversation_records"] = len(conversation_records)
    _write_json_atomic(artifact_dir / PASS2_STATS_FILE, {k: v for k, v in stats.items() if k != "_raw_scores"})
    return stats


async def _run_inline_scoring_directory(target: InlineScoringTarget, tag_stats_path,
                                        limit, config, resume=False,
                                        llm_progress_cb=None):
    """Score a mirrored inline dataset tree using meta caches under meta_label_data."""
    source_files = discover_inline_jsonl_files(target)
    for source_file in source_files:
        write_inline_labeled_cache(
            source_file,
            target.layout.file_artifact_dir(source_file),
            limit=limit,
        )

    effective_tag_stats = tag_stats_path
    if effective_tag_stats is None:
        summary_candidate = target.layout.meta_root / PASS1_SUMMARY_STATS_FILE
        if summary_candidate.exists():
            effective_tag_stats = str(summary_candidate)

    summary = await _run_scoring_directory(
        target.layout.file_meta_root,
        target.layout.meta_root,
        effective_tag_stats,
        limit,
        config,
        resume=resume,
        llm_progress_cb=llm_progress_cb,
    )

    corrected_file_stats = []
    all_conv_records = []
    for source_file in source_files:
        artifact_dir = target.layout.file_artifact_dir(source_file)
        scored_samples, conv_records = _sync_inline_scored_cache_to_dataset(
            source_file,
            artifact_dir,
            limit=limit,
        )
        if conv_records:
            all_conv_records.extend(conv_records)
        _generate_inline_file_dashboard(target, source_file)

        stats = _load_existing_pass2_stats(artifact_dir)
        if stats:
            stats["input_file"] = str(source_file)
            stats["mirrored_file"] = str(source_file)
            stats["file"] = target.layout.relative_source_path(source_file).as_posix()
            stats["total_scored"] = len([sample for sample in scored_samples if sample.get("value")])
            if conv_records:
                stats["conversation_records"] = len(conv_records)
            _write_json_atomic(
                artifact_dir / PASS2_STATS_FILE,
                {k: v for k, v in stats.items() if k != "_raw_scores"},
            )
            corrected_file_stats.append(stats)

    if corrected_file_stats:
        corrected_summary = _merge_value_stats(corrected_file_stats)
        for key in (
            "elapsed_seconds",
            "model",
            "files_processed",
            "planned_files",
            "planned_samples",
            "planned_baseline_llm_calls",
            "planned_initial_llm_calls",
            "planning_elapsed_seconds",
            "rarity_config",
            "http_request_stats",
            "prompt_mode",
            "compact_prompt",
            "value_truncation_budget",
        ):
            if key in summary:
                corrected_summary[key] = summary[key]
        corrected_summary["input_path"] = str(target.target_path)
        _write_json_atomic(target.layout.meta_root / PASS2_SUMMARY_STATS_FILE, corrected_summary)
        summary = corrected_summary

    if all_conv_records:
        _write_json_atomic(target.layout.meta_root / "conversation_scores.json", all_conv_records)

    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        generate_value_dashboard(
            target.layout.meta_root,
            scored_file=None,
            stats_file=PASS2_SUMMARY_STATS_FILE,
            output_file=str(target.layout.dashboard_path(
                pass2_global_dashboard_filename(target.layout.dataset_root_name)
            )),
            quiet=True,
        )
    except Exception:
        pass

    summary["run_dir"] = str(target.layout.run_root)
    summary["input_path"] = str(target.target_path)
    return summary

def _create_progress():
    """Create a Rich progress bar for scoring."""
    return create_pipeline_progress()


async def run_scoring(input_path, output_dir=None, tag_stats_path=None,
                      limit=0, config=None, resume=False, llm_progress_cb=None,
                      precomputed_workload_estimate=None):
    """Run value scoring (Pass 2) on pre-labeled data.

    Args:
        input_path: Path to labeled.json or directory of labeled files
        output_dir: Where to write outputs (default: same directory as input)
        tag_stats_path: Path to Pass 1 stats for rarity computation
        limit: Max samples to score (0 = all)
        config: PipelineConfig override
        resume: If True, skip samples that already have scores in scored.jsonl
    """
    if config is None:
        config = PipelineConfig()

    input_path = Path(input_path)
    if _looks_like_legacy_scoring_run_dir(input_path):
        inline_target = None
        print(f"  Pass 2 input layout: standard run dir ({input_path.name})")
    else:
        inline_target = run_with_heartbeat(
            "Resolving Pass 2 input layout",
            lambda: infer_inline_scoring_target(input_path),
        )
        if inline_target is not None:
            if input_path.is_file():
                print(f"  Pass 2 input layout: inline mirrored file ({input_path.name})")
            elif input_path == inline_target.layout.run_root:
                print(f"  Pass 2 input layout: inline mirrored run ({input_path.name})")
            else:
                print(f"  Pass 2 input layout: inline mirrored dataset ({input_path.name})")
        elif input_path.is_dir():
            print(f"  Pass 2 input layout: standard directory ({input_path.name})")
        else:
            print(f"  Pass 2 input layout: single file ({input_path.name})")

    if inline_target is not None:
        if input_path.is_file():
            return await _run_inline_scoring_file(
                inline_target,
                input_path.resolve(),
                tag_stats_path,
                limit,
                config,
                resume=resume,
                llm_progress_cb=llm_progress_cb,
            )
        return await _run_inline_scoring_directory(
            inline_target,
            tag_stats_path,
            limit,
            config,
            resume=resume,
            llm_progress_cb=llm_progress_cb,
        )

    # Determine if directory or single file
    if input_path.is_dir():
        return await _run_scoring_directory(
            input_path, output_dir, tag_stats_path, limit, config,
            resume=resume,
            llm_progress_cb=llm_progress_cb,
            precomputed_workload_estimate=precomputed_workload_estimate,
        )
    else:
        return await _run_scoring_file(
            input_path, output_dir, tag_stats_path, limit, config,
            resume=resume, llm_progress_cb=llm_progress_cb,
        )


async def _run_scoring_file_chunked(input_path, output_dir, tag_stats_path,
                                     limit, config, resume=False,
                                     llm_progress_cb=None,
                                     file_label=None,
                                     show_progress=True,
                                     combo_counts_override=None,
                                     combo_mode_override=None,
                                     shared_http_client=None,
                                     shared_semaphore=None,
                                     shared_rate_limiter=None,
                                     shared_runtime=None,
                                     progress_hook=None,
                                     print_summary=True,
                                     generate_dashboard=True,
                                     quiet=False):
    """Score a JSONL labeled file in chunks to bound memory.

    Two-pass approach:
      Pass A (lightweight, no LLM): Stream through JSONL to compute rarity.
      Pass B (chunked, watermark LLM): Read chunks, attach pre-computed rarity,
             run LLM scoring, write scored JSONL incrementally.
    """
    from sft_label.pipeline import iter_chunks_from_jsonl

    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resumed_values = _load_resumed_value_cache(output_dir) if resume else {}
    if resumed_values and not quiet:
        print(f"  Resume: loaded {len(resumed_values)} pre-scored samples from scored.jsonl")

    chunk_size = config.chunk_size if config else CHUNK_SIZE
    max_active = config.max_active_chunks if config else MAX_ACTIVE_CHUNKS

    # ── Load tag stats for rarity ──
    rarity_mode = resolve_rarity_mode(config)
    extension_rarity_mode = resolve_extension_rarity_mode(config)
    stats_ref_info = None
    idf_map = {}
    total_stats_samples = 0
    combo_counts = None
    combo_mode = "disabled"
    baseline_combo_counts = None
    extension_stats_context = None
    extension_baseline_source = "unavailable"

    if tag_stats_path is None:
        candidate = input_path.parent / PASS1_STATS_FILE
        if candidate.exists():
            tag_stats_path = str(candidate)

    stats_source = tag_stats_path
    if stats_source:
        distributions, total_stats_samples, ts, meta = load_tag_stats_context(stats_source)
        if distributions:
            idf_map = compute_tag_idf(distributions, total_stats_samples)
            baseline_combo_counts = meta.get("combo_counts")
            extension_stats_context = meta.get("extension_stats")
            if extension_stats_context:
                extension_baseline_source = "external"
            stats_ref_info = {
                "source": str(stats_source),
                "total_samples": total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            missing_dims = [
                d for d in (config.rarity_weights or RARITY_WEIGHTS).keys()
                if d not in idf_map
            ]
            if missing_dims:
                print("  Warning: rarity baseline missing dimensions; ignored in IDF: "
                      + ", ".join(missing_dims))
        else:
            print(f"  Warning: {stats_source} has no tag_distributions, fallback to local baseline")
    else:
        print("  No external tag stats found, fallback to local rarity baseline")

    # ── Pass A: Compute rarity (stream, no LLM) ──
    raw_rarities = []
    local_combo_counts = {}
    extension_baseline_samples = [] if extension_rarity_mode != "off" and not extension_stats_context else None
    processed_count = 0
    usable_count = 0
    local_dims = list((config.rarity_weights or RARITY_WEIGHTS).keys())
    local_distributions = {dim: {} for dim in local_dims}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            _maybe_backfill_domain(sample, config=config)
            labels = sample.get("labels") or {}
            if _update_distributions_from_labels(local_distributions, labels, local_dims):
                usable_count += 1
            _update_combo_counts_from_labels(local_combo_counts, labels)
            if extension_baseline_samples is not None:
                extension_payloads = _extract_sample_extension_payloads(sample)
                if extension_payloads:
                    extension_baseline_samples.append({"label_extensions": copy.deepcopy(extension_payloads)})
            processed_count += 1
            del sample
            if limit > 0 and processed_count >= limit:
                break

    if not idf_map and usable_count > 0:
        # Fallback: build IDF from current labeled file so duplicates do not
        # collapse into pseudo-uniform percentile-only rarity.
        local_distributions = {d: c for d, c in local_distributions.items() if c}
        if local_distributions:
            total_stats_samples = usable_count
            idf_map = compute_tag_idf(local_distributions, total_stats_samples)
            stats_source = f"{input_path}#local"
            stats_ref_info = {
                "source": stats_source,
                "total_samples": total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }
            combo_counts = local_combo_counts
            combo_mode = "local"
            print(f"  Using local rarity baseline ({total_stats_samples} samples)")
    elif idf_map:
        combo_counts, combo_mode, combo_msg = _resolve_combo_baseline(
            baseline_combo_counts,
            combo_counts_override or local_combo_counts,
        )
        if combo_mode_override and combo_mode == "hybrid":
            combo_mode = combo_mode_override
        if combo_msg:
            print(combo_msg)

    if not idf_map:
        print("  Warning: rarity unavailable (no valid tag distributions)")
    if extension_rarity_mode != "off" and not extension_stats_context and extension_baseline_samples:
        extension_stats_context = aggregate_extension_stats(extension_baseline_samples)
        if extension_stats_context:
            extension_baseline_source = "local"

    # Second lightweight pass to compute per-sample rarity (needs combo_counts)
    sample_idx = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            _maybe_backfill_domain(sample, config=config)
            labels = sample.get("labels") or {}
            rarity = compute_sample_rarity(
                labels, idf_map, total_stats_samples,
                rarity_weights=config.rarity_weights or RARITY_WEIGHTS,
                combo_alpha=config.rarity_combo_alpha,
                combo_counts=combo_counts,
                stats_ref_info=stats_ref_info,
            )
            rarity = augment_rarity_result(
                rarity,
                sample,
                extension_stats=extension_stats_context,
                config=config,
                baseline_source=extension_baseline_source,
            )
            raw_rarities.append(rarity)
            del sample
            sample_idx += 1
            if limit > 0 and sample_idx >= limit:
                break

    normalize_rarity_scores(
        raw_rarities,
        mode=rarity_mode,
        total_samples=total_stats_samples,
    )
    for rarity in raw_rarities:
        _refresh_augmented_rarity_after_core_normalization(rarity)
    total = len(raw_rarities)

    # ── Pass B: Chunked LLM scoring ──
    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    if not quiet:
        print(f"  Scoring {total} samples | model={config.scoring_model} concurrency={config.scoring_concurrency} {_rps}")

    runtime = shared_runtime
    if runtime is None:
        runtime = _instantiate_runtime(config, concurrency=config.scoring_concurrency)
        setattr(config, "_adaptive_runtime", runtime)
    elif config is not None:
        setattr(config, "_adaptive_runtime", runtime)
    rate_limiter = shared_rate_limiter
    if rate_limiter is None:
        rate_limiter = (
            AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup)
            if runtime is None and config.rps_limit > 0
            else None
        )
    sem = shared_semaphore or asyncio.Semaphore(config.scoring_concurrency)
    scored_count = 0
    failed_count = 0
    first_error_logged = False
    start_time = time.time()

    # Lightweight per-sample summaries for final stats
    score_summaries = []
    monitor_totals = _init_monitor_totals()

    out_scored = open(output_dir / "scored.jsonl", "w", encoding="utf-8")
    out_monitor = open(output_dir / "monitor_value.jsonl", "w", encoding="utf-8")
    out_failed = open(output_dir / "failed_value.jsonl", "w", encoding="utf-8")
    out_failures_log = open(output_dir / "score_failures.jsonl", "w", encoding="utf-8")

    try:
        client_ctx = (
            _borrow_http_client(shared_http_client)
            if shared_http_client is not None
            else httpx.AsyncClient(
                proxy=None,
                timeout=config.request_timeout,
                limits=httpx.Limits(
                    max_connections=config.scoring_concurrency + 10,
                    max_keepalive_connections=config.scoring_concurrency,
                ),
            )
        )
        async with client_ctx as client:

            # Chunk state tracking
            global_offset = 0
            active_chunks = {}  # chunk_idx -> {samples, pending_count, done_samples, offset}
            pending_futures = set()
            chunk_gen = iter_chunks_from_jsonl(input_path, chunk_size, limit=limit)
            chunks_loaded = 0
            gen_exhausted = False
            next_chunk_to_flush = 0
            skipped_count = 0

            async def _tagged_score(idx, chunk_idx, sample, rarity_result):
                nonlocal scored_count, failed_count, first_error_logged
                value, monitor = await score_one(
                    client, sample, config.scoring_model, rarity_result,
                    idx, total, sem, config=config, rate_limiter=rate_limiter,
                )
                return chunk_idx, idx, value, monitor, sample

            def load_next_scoring_chunk():
                nonlocal chunks_loaded, global_offset, gen_exhausted, skipped_count
                try:
                    raw_records = next(chunk_gen)
                except StopIteration:
                    gen_exhausted = True
                    return None, 0

                # These are already-labeled samples from the JSONL
                samples = []
                for raw in raw_records:
                    _maybe_backfill_domain(raw, config=config)
                    samples.append(raw)

                chunk_idx = chunks_loaded
                offset = global_offset
                global_offset += len(samples)
                chunks_loaded += 1

                active_chunks[chunk_idx] = {
                    "samples": samples,
                    "offset": offset,
                    "total": len(samples),
                    "done": 0,
                    "values": [None] * len(samples),
                    "monitors": [None] * len(samples),
                    "flushed": False,
                }

                # Submit scoring tasks
                resumed_in_chunk = 0
                for i, sample in enumerate(samples):
                    abs_idx = offset + i
                    resumed_value = _existing_resume_value(sample, resumed_values)
                    if resumed_value is not None:
                        active_chunks[chunk_idx]["values"][i] = resumed_value
                        active_chunks[chunk_idx]["monitors"][i] = _resumed_monitor(
                            sample.get("id", f"sample-{abs_idx}")
                        )
                        active_chunks[chunk_idx]["done"] += 1
                        resumed_in_chunk += 1
                        skipped_count += 1
                        continue
                    rarity_result = raw_rarities[abs_idx] if abs_idx < len(raw_rarities) else {"score": None, "tag_rarity": None, "combo_rarity": None, "stats_ref": stats_ref_info}
                    fut = asyncio.ensure_future(
                        _tagged_score(abs_idx, chunk_idx, sample, rarity_result)
                    )
                    pending_futures.add(fut)

                return chunk_idx, resumed_in_chunk

            chunk_submission_watermark = max(
                config.scoring_concurrency,
                int(config.scoring_concurrency * config.dir_pipeline_watermark),
            )

            def maybe_load_more_scoring():
                resumed_loaded = 0
                active_count = sum(1 for c in active_chunks.values()
                                   if not c.get("flushed") and c["done"] < c["total"])
                while (not gen_exhausted
                       and len(pending_futures) < chunk_submission_watermark
                       and active_count < max_active):
                    chunk_idx, resumed_in_chunk = load_next_scoring_chunk()
                    if chunk_idx is None:
                        break
                    resumed_loaded += resumed_in_chunk
                    if active_chunks[chunk_idx]["done"] < active_chunks[chunk_idx]["total"]:
                        active_count += 1
                return resumed_loaded

            def flush_scoring_chunk(chunk_idx):
                nonlocal scored_count, failed_count
                chunk_data = active_chunks[chunk_idx]
                for i, sample in enumerate(chunk_data["samples"]):
                    value = chunk_data["values"][i]
                    monitor = chunk_data["monitors"][i]

                    if value:
                        sample["value"] = value
                        scored_count += 1
                        # Summary for final stats
                        summary = _selection_summary_from_sample(sample, config=config)
                        summary.update({
                            "flags": value.get("flags", []),
                            "thinking_mode": value.get("thinking_mode"),
                        })
                        score_summaries.append(summary)
                    else:
                        failed_count += 1
                        out_failed.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        # Failure log record
                        record = {
                            "sample_id": sample.get("id", f"sample-{chunk_data['offset'] + i}"),
                            "status": monitor["status"] if monitor else "no_result",
                            "error": (monitor.get("error", "") if monitor else "no monitor record"),
                            "error_response": (monitor.get("error_response", "")[:1000] if monitor else ""),
                            "attempts": (monitor.get("attempts", 0) if monitor else 0),
                            "error_class": (monitor.get("error_class") if monitor else None),
                            "retryable_infra": (monitor.get("retryable_infra") if monitor else None),
                            "http_status": (monitor.get("http_status") if monitor else None),
                            "runtime_state": (monitor.get("runtime_state") if monitor else None),
                        }
                        out_failures_log.write(json.dumps(record, ensure_ascii=False) + "\n")

                    out_scored.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    if monitor:
                        _accumulate_monitor_totals(monitor_totals, monitor)
                        out_monitor.write(json.dumps(monitor, ensure_ascii=False) + "\n")

                # Release
                chunk_data["samples"] = None
                chunk_data["values"] = None
                chunk_data["monitors"] = None
                chunk_data["flushed"] = True

            def flush_ready_scoring_chunks():
                nonlocal next_chunk_to_flush
                while True:
                    chunk_data = active_chunks.get(next_chunk_to_flush)
                    if chunk_data is None or chunk_data.get("flushed"):
                        break
                    if chunk_data["done"] < chunk_data["total"]:
                        break
                    flush_scoring_chunk(next_chunk_to_flush)
                    next_chunk_to_flush += 1

            progress_ctx = _create_progress() if show_progress else nullcontext(None)
            with progress_ctx as progress:
                task = progress.add_task("Pass 2", total=total, info="") if progress is not None else None

                # Initial load
                resumed_loaded = maybe_load_more_scoring()
                if resumed_loaded:
                    if progress_hook is not None:
                        progress_hook({
                            "type": "resumed",
                            "count": resumed_loaded,
                            "skipped_count": skipped_count,
                            "label": file_label or input_path.name,
                        })
                    if progress is not None:
                        progress.update(task, advance=resumed_loaded, info=f"skipped {skipped_count} resumed")
                flush_ready_scoring_chunks()

                while pending_futures or not gen_exhausted:
                    if not pending_futures:
                        resumed_loaded = maybe_load_more_scoring()
                        if resumed_loaded:
                            if progress_hook is not None:
                                progress_hook({
                                    "type": "resumed",
                                    "count": resumed_loaded,
                                    "skipped_count": skipped_count,
                                    "label": file_label or input_path.name,
                                })
                            if progress is not None:
                                progress.update(task, advance=resumed_loaded, info=f"skipped {skipped_count} resumed")
                        flush_ready_scoring_chunks()
                        if not pending_futures:
                            if gen_exhausted:
                                break
                            continue
                    done, pending_futures = await asyncio.wait(
                        pending_futures, return_when=asyncio.FIRST_COMPLETED)

                    for fut in done:
                        chunk_idx, abs_idx, value, monitor, _ = fut.result()
                        chunk_data = active_chunks[chunk_idx]
                        local_idx = abs_idx - chunk_data["offset"]

                        chunk_data["values"][local_idx] = value
                        chunk_data["monitors"][local_idx] = monitor
                        chunk_data["done"] += 1

                        if not value and not first_error_logged and monitor:
                            summary = _summarize_first_failure(monitor)
                            if summary:
                                print(summary)
                            first_error_logged = True

                        _info = format_progress_info(
                            ok_count=scored_count + (1 if value else 0),
                            fail_count=failed_count + (0 if value else 1),
                            request_stats=rate_limiter.stats if rate_limiter else None,
                        )
                        if llm_progress_cb and monitor and progress_hook is None:
                            run_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass2")
                            if run_info:
                                _info = f"{_info} • {run_info}"
                        if progress_hook is not None:
                            progress_hook({
                                "type": "sample",
                                "count": 1,
                                "label": file_label or input_path.name,
                                "monitor": monitor,
                                "value": value,
                            })
                        if progress is not None:
                            progress.update(task, advance=1, info=_info)
                    flush_ready_scoring_chunks()
                    resumed_loaded = maybe_load_more_scoring()
                    if resumed_loaded:
                        if progress_hook is not None:
                            progress_hook({
                                "type": "resumed",
                                "count": resumed_loaded,
                                "skipped_count": skipped_count,
                                "label": file_label or input_path.name,
                            })
                        if progress is not None:
                            progress.update(task, advance=resumed_loaded, info=f"skipped {skipped_count} resumed")
                    flush_ready_scoring_chunks()

                flush_ready_scoring_chunks()

    finally:
        out_scored.close()
        out_monitor.close()
        out_failed.close()
        out_failures_log.close()

    elapsed = time.time() - start_time

    sweep = await _run_pass2_recovery_sweep_chunked(
        output_dir=output_dir,
        raw_rarities=raw_rarities,
        config=config,
    )
    if sweep and sweep.get("recovered"):
        # After patching scored/monitor files, rebuild summaries/monitors in file order
        # so selection/stats align with the rewritten outputs.
        score_summaries, monitor_totals = _rebuild_chunked_summaries_and_monitors(
            output_dir / "scored.jsonl",
            output_dir / "monitor_value.jsonl",
        )

    # Remove empty failed file
    failed_path = output_dir / "failed_value.jsonl"
    if failed_path.exists() and failed_path.stat().st_size == 0:
        failed_path.unlink()

    # Remove empty failure log
    failures_log_path = output_dir / "score_failures.jsonl"
    if failures_log_path.exists() and failures_log_path.stat().st_size == 0:
        failures_log_path.unlink()

    # ── Compute selection scores from summaries ──
    selection_results = compute_selection_scores_from_summaries(
        score_summaries, config=config)

    # Attach selection scores back to summaries (for stats computation)
    for summary, sel in zip(score_summaries, selection_results):
        summary["selection_score"] = sel["selection_score"]
        summary["intra_class_rank"] = sel["intra_class_rank"]

    # Re-stream scored.jsonl to attach selection_score and intra_class_rank
    scored_path = output_dir / "scored.jsonl"
    scored_tmp = output_dir / "scored.jsonl.tmp"
    scored_idx = 0  # index into selection_results (only for samples with value)
    with open(scored_path, "r", encoding="utf-8") as fin, \
         open(scored_tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("value"):
                if scored_idx < len(selection_results):
                    sample["value"]["selection_score"] = selection_results[scored_idx]["selection_score"]
                    sample["value"]["intra_class_rank"] = selection_results[scored_idx]["intra_class_rank"]
                    apply_v2_scores([sample], config=config)
                    scored_idx += 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    scored_tmp.rename(scored_path)

    # ── Compute stats from summaries ──
    stats = _compute_value_stats_from_summaries(score_summaries, monitor_totals, total)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["model"] = config.scoring_model
    stats["input_file"] = str(input_path)
    stats["file"] = file_label or stats.get("file") or input_path.name
    if rate_limiter:
        stats["http_request_stats"] = rate_limiter.stats.to_dict()
    if runtime is not None:
        stats["adaptive_runtime"] = {
            "enabled": True,
            "final": _runtime_snapshot(runtime),
        }
    if sweep:
        stats["recovery_sweep"] = sweep
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": str(stats_source) if stats_source else None,
        "total_samples_in_distribution": total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
        "combo_mode": combo_mode,
        "score_mode": rarity_mode,
    }
    stats["extension_rarity_config"] = {
        "mode": resolve_extension_rarity_mode(config),
        "baseline_source": "external" if stats_source else "local",
        "min_extension_baseline_total": getattr(config, "min_extension_baseline_total", None),
    }
    stats["chunked"] = True
    _annotate_scoring_prompt_stats(stats, config)

    stats_path = output_dir / PASS2_STATS_FILE
    stats_to_write = {k: v for k, v in stats.items() if k != "_raw_scores"}
    tmp_stats = stats_path.with_suffix(".tmp.json")
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats_to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp_stats, stats_path)

    # Conversation-level aggregation (re-read scored.jsonl for chunked mode)
    try:
        def _aggregate_chunked_conversations():
            from sft_label.conversation import aggregate_conversations, write_conversation_scores

            scored_path = output_dir / "scored.jsonl"
            if not scored_path.exists():
                return None

            def _iter_scored_samples():
                with open(scored_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)

            conv_records = aggregate_conversations(_iter_scored_samples())
            if conv_records:
                write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
            return conv_records

        run_with_heartbeat("Aggregating conversation scores", _aggregate_chunked_conversations)
    except Exception as e:
        print(f"  Warning: conversation aggregation failed: {e}")

    # Dashboard
    if generate_dashboard:
        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            run_with_heartbeat(
                "Generating scoring dashboard",
                lambda: generate_value_dashboard(
                    output_dir,
                    scored_file="scored.jsonl",
                    stats_file=PASS2_STATS_FILE,
                    output_file=PASS2_DASHBOARD_FILE,
                    quiet=quiet,
                ),
            )
        except Exception as e:
            print(f"  Warning: dashboard generation failed: {e}")

    if print_summary:
        print_scoring_summary(stats, output_dir)

    return stats


def _compute_value_stats_from_summaries(summaries, monitor_totals, total_input):
    """Compute value stats from lightweight per-sample summaries.

    Produces the same structure as compute_value_stats but without needing
    all scored samples in memory.
    """
    total_scored = len(summaries)
    if isinstance(monitor_totals, dict):
        total_failed = int(monitor_totals.get("total_failed", 0) or 0)
        total_estimated = int(monitor_totals.get("total_estimated", 0) or 0)
        total_llm_calls = int(monitor_totals.get("total_llm_calls", 0) or 0)
        total_prompt_tokens = int(monitor_totals.get("total_prompt_tokens", 0) or 0)
        total_completion_tokens = int(monitor_totals.get("total_completion_tokens", 0) or 0)
    else:
        monitors = monitor_totals or []
        total_failed = sum(1 for m in monitors if m.get("status") == "failed")
        total_estimated = sum(1 for m in monitors if m.get("status") == "estimated_selective")
        total_llm_calls = sum(m.get("llm_calls", 0) for m in monitors)
        total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in monitors)
        total_completion_tokens = sum(m.get("completion_tokens", 0) for m in monitors)

    # Score distributions
    def _gather(key):
        return [s[key] for s in summaries if s.get(key) is not None]

    score_distributions = {
        "value_score": _percentiles(_gather("value_score")),
        "selection_score": _percentiles(_gather("selection_score")),
        "intra_class_rank": _percentiles(_gather("intra_class_rank")),
        "complexity_overall": _percentiles(_gather("complexity_overall")),
        "quality_overall": _percentiles(_gather("quality_overall")),
        "reasoning_overall": _percentiles(_gather("reasoning_overall")),
        "rarity_score": _percentiles(_gather("rarity_score")),
        "extension_rarity_score": _percentiles(_gather("extension_rarity_score")),
        "rarity_v2_score": _percentiles(_gather("rarity_v2_score")),
        "value_score_v2": _percentiles(_gather("value_score_v2")),
        "selection_score_v2": _percentiles(_gather("selection_score_v2")),
    }

    # Histogram bins (1-10) for dashboard charts
    _hist_keys = ["value_score", "selection_score", "intra_class_rank",
                  "complexity_overall", "quality_overall", "reasoning_overall",
                  "rarity_score", "extension_rarity_score", "rarity_v2_score",
                  "value_score_v2", "selection_score_v2"]
    histograms = {k: _histogram_bins(_gather(k)) for k in _hist_keys}

    # Thinking mode stats
    slow = [s for s in summaries if s.get("thinking_mode") == "slow"]
    fast = [s for s in summaries if s.get("thinking_mode") == "fast"]
    thinking_mode_stats = {
        "slow": {
            "count": len(slow),
            "mean_value": round(sum(s.get("value_score", 0) or 0 for s in slow) / max(len(slow), 1), 2),
            "mean_quality": round(sum(s.get("quality_overall", 0) or 0 for s in slow) / max(len(slow), 1), 2),
            "mean_reasoning": round(sum(s.get("reasoning_overall", 0) or 0 for s in slow) / max(len(slow), 1), 2),
        },
        "fast": {
            "count": len(fast),
            "mean_value": round(sum(s.get("value_score", 0) or 0 for s in fast) / max(len(fast), 1), 2),
            "mean_quality": round(sum(s.get("quality_overall", 0) or 0 for s in fast) / max(len(fast), 1), 2),
            "mean_reasoning": round(sum(s.get("reasoning_overall", 0) or 0 for s in fast) / max(len(fast), 1), 2),
        },
    }

    # Flag counts
    flag_counts = {}
    flag_value_sums = {}
    for s in summaries:
        for f in s.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
            flag_value_sums.setdefault(f, []).append(s.get("value_score", 0) or 0)

    flag_value_impact = {
        f: {"mean_value": round(sum(vs) / len(vs), 2), "count": len(vs)}
        for f, vs in flag_value_sums.items()
    }

    # Selection thresholds
    all_value_scores = sorted([s["value_score"] for s in summaries
                                if s.get("value_score") is not None])
    n_vs = len(all_value_scores)
    selection_thresholds = {}
    for pct_label, pct in [("top_10pct", 90), ("top_25pct", 75), ("top_50pct", 50)]:
        if n_vs > 0:
            idx = min(int(n_vs * pct / 100), n_vs - 1)
            threshold = all_value_scores[idx]
            count = sum(1 for v in all_value_scores if v >= threshold)
            selection_thresholds[pct_label] = {"threshold": round(threshold, 1), "count": count}

    # Value and selection by tag
    tag_dims = ["intent", "difficulty", "domain", "concept", "task",
                "agentic", "constraint", "context"]
    value_by_tag = {}
    selection_by_tag = {}
    for dim in tag_dims:
        tag_values = {}
        tag_selections = {}
        for s in summaries:
            vs = s.get("value_score")
            ss = s.get("selection_score")
            if vs is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_values.setdefault(t, []).append(vs)
                    if ss is not None:
                        tag_selections.setdefault(t, []).append(ss)
            else:
                tag_values.setdefault(tags, []).append(vs)
                if ss is not None:
                    tag_selections.setdefault(tags, []).append(ss)
        value_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_values.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }
        selection_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_selections.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }

    keep_rates = _compute_keep_rates([s.get("selection_score") for s in summaries])
    keep_rate_7 = keep_rates.get(f"{FILE_RANKING_KEEP_RATE_THRESHOLD:.1f}", 0)
    turn_counts = [
        turn_count
        for s in summaries
        if (turn_count := _coerce_mean_turn_count((s.get("metadata") or {}))) is not None
    ]
    mean_turns = round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0

    return {
        "total_scored": total_scored,
        "total_failed": total_failed,
        "total_estimated": total_estimated,
        "total_llm_calls": total_llm_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "score_distributions": score_distributions,
        "histograms": histograms,
        "value_by_tag": value_by_tag,
        "selection_by_tag": selection_by_tag,
        "thinking_mode_stats": thinking_mode_stats,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "flag_value_impact": flag_value_impact,
        "selection_thresholds": selection_thresholds,
        "keep_rates": keep_rates,
        "keep_rate_7": keep_rate_7,
        "mean_turns": mean_turns,
    }


def print_scoring_summary(stats, run_dir, is_batch=False):
    """Print final scoring summary to stdout (mirrors Pass 1's print_summary)."""
    print(f"\n{'='*80}")
    label = "BATCH SCORING COMPLETE" if is_batch else "SCORING COMPLETE"
    elapsed = stats.get('elapsed_seconds', 0)
    print(f"{label} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")

    total_scored = stats.get('total_scored', 0)
    total_failed = stats.get('total_failed', 0)
    total_estimated = stats.get('total_estimated', 0)
    total = total_scored + total_failed
    success_rate = total_scored / total * 100 if total > 0 else 0
    if is_batch:
        print(f"Files:       {stats.get('files_processed', '?')}")
    print(f"Scored:      {total_scored}/{total} ({success_rate:.1f}%)")
    if total_estimated:
        print(f"Estimated:   {total_estimated}")
    print(f"LLM calls:   {stats.get('total_llm_calls', 0)}")
    total_tokens = stats.get('total_tokens', 0)
    print(f"Tokens:      {total_tokens:,}")
    if elapsed > 0 and total_scored > 0:
        print(f"Throughput:  {total_scored / elapsed:.1f} samples/sec")

    # HTTP request stats
    http = stats.get("http_request_stats")
    if http and http.get("total_http_requests", 0) > 0:
        t = http["total_http_requests"]
        ok = http["success"]
        rate = http["success_rate"]
        errs = http.get("errors", {})
        timeouts = http.get("timeouts", 0)
        err_parts = [f"{code}×{n}" for code, n in errs.items()]
        if timeouts:
            err_parts.append(f"timeout×{timeouts}")
        err_str = f" — errors: {', '.join(err_parts)}" if err_parts else ""
        print(f"HTTP:        {ok}/{t} ({rate}%){err_str}")

    # Score distributions
    dists = stats.get("score_distributions", {})
    dims = [
        ("value_score", "Value"),
        ("complexity_overall", "Complexity"),
        ("quality_overall", "Quality"),
        ("reasoning_overall", "Reasoning"),
        ("rarity_score", "Rarity"),
        ("selection_score", "Selection"),
    ]
    print("\nScore distributions (mean ± std):")
    for key, label in dims:
        d = dists.get(key, {})
        mean = d.get("mean", 0)
        std = d.get("std", 0)
        p50 = d.get("p50", 0)
        bar = "█" * int(mean)
        print(f"  {label:12s} {mean:5.2f} ±{std:4.2f}  p50={p50:.1f}  {bar}")

    # Thinking mode breakdown
    ts = stats.get("thinking_mode_stats", {})
    slow = ts.get("slow", {})
    fast = ts.get("fast", {})
    if slow.get("count", 0) > 0 or fast.get("count", 0) > 0:
        print("\nThinking modes:")
        if slow.get("count", 0) > 0:
            print(f"  slow:  {slow['count']:4d}  val={slow.get('mean_value', 0):.2f}  "
                  f"qual={slow.get('mean_quality', 0):.2f}  "
                  f"reas={slow.get('mean_reasoning', 0):.2f}")
        if fast.get("count", 0) > 0:
            print(f"  fast:  {fast['count']:4d}  val={fast.get('mean_value', 0):.2f}  "
                  f"qual={fast.get('mean_quality', 0):.2f}  "
                  f"reas={fast.get('mean_reasoning', 0):.2f}")

    # Top flags
    flag_counts = stats.get("flag_counts", {})
    if flag_counts:
        top_flags = list(flag_counts.items())[:8]
        flags_str = ", ".join(f"{f}({n})" for f, n in top_flags)
        print(f"\nFlags: {flags_str}")

    # Selection thresholds
    thresholds = stats.get("selection_thresholds", {})
    if thresholds:
        parts = []
        for k in ["top_10pct", "top_25pct", "top_50pct"]:
            t = thresholds.get(k, {})
            if t:
                parts.append(f"{k.replace('_', ' ')}≥{t.get('threshold', 0):.1f}({t.get('count', 0)})")
        if parts:
            print(f"Thresholds: {', '.join(parts)}")

    # Per-file summary in batch mode
    per_file = stats.get("per_file_summary", [])
    if is_batch and per_file:
        print("\nPer-file scores:")
        for pf in per_file:
            print(f"  {pf.get('file', '?'):30s}  n={pf.get('count', 0):4d}  "
                  f"val={pf.get('mean_value', 0):.2f}  "
                  f"qual={pf.get('mean_quality', 0):.2f}  "
                  f"sel={pf.get('mean_selection', 0):.2f}")

    print(f"\nRun dir: {run_dir}")


async def _run_scoring_file(input_path, output_dir, tag_stats_path, limit, config, resume=False,
                            llm_progress_cb=None, file_label=None, generate_dashboard=True):
    """Score a single labeled file."""
    input_path = Path(input_path)

    # Dispatch to chunked pipeline for JSONL files (memory-bounded)
    if input_path.suffix == ".jsonl":
        return await _run_scoring_file_chunked(
            input_path, output_dir, tag_stats_path, limit, config,
            resume=resume,
            llm_progress_cb=llm_progress_cb,
            file_label=file_label,
            generate_dashboard=generate_dashboard,
        )

    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled data (only .json reaches here; .jsonl dispatched above)
    print(f"Loading labeled data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit > 0:
        samples = samples[:limit]
    for sample in samples:
        _maybe_backfill_domain(sample, config=config)
    total = len(samples)

    # Resume: load previously scored samples from scored.jsonl
    resumed_values = _load_resumed_value_cache(output_dir) if resume else {}
    if resumed_values:
        print(f"  Resume: loaded {len(resumed_values)} pre-scored samples from scored.jsonl")

    # Load tag stats for rarity
    rarity_mode = resolve_rarity_mode(config)
    extension_rarity_mode = resolve_extension_rarity_mode(config)
    stats_ref_info = None
    idf_map = {}
    total_stats_samples = 0
    combo_counts = None
    combo_mode = "disabled"
    baseline_combo_counts = None
    extension_stats_context = None
    extension_baseline_source = "unavailable"

    if tag_stats_path:
        stats_source = tag_stats_path
    else:
        # Auto-discover Pass 1 stats in same directory
        candidate = input_path.parent / PASS1_STATS_FILE
        stats_source = str(candidate) if candidate.exists() else None

    if stats_source:
        distributions, total_stats_samples, ts, meta = load_tag_stats_context(stats_source)
        if distributions:
            idf_map = compute_tag_idf(distributions, total_stats_samples)
            baseline_combo_counts = meta.get("combo_counts")
            extension_stats_context = meta.get("extension_stats")
            if extension_stats_context:
                extension_baseline_source = "external"
            stats_ref_info = {
                "source": str(stats_source),
                "total_samples": total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            missing_dims = [
                d for d in (config.rarity_weights or RARITY_WEIGHTS).keys()
                if d not in idf_map
            ]
            if missing_dims:
                print("  Warning: rarity baseline missing dimensions; ignored in IDF: "
                      + ", ".join(missing_dims))
        else:
            print(f"  Warning: {stats_source} has no tag_distributions, fallback to local baseline")
    else:
        print("  No external tag stats found, fallback to local rarity baseline")

    local_combo_counts = None
    if idf_map and not baseline_combo_counts and samples:
        local_combo_counts = build_combo_counts(samples)
        combo_counts, combo_mode, combo_msg = _resolve_combo_baseline(
            baseline_combo_counts,
            local_combo_counts,
        )
        if combo_msg:
            print(combo_msg)

    if not idf_map and samples:
        local_distributions, local_total = build_tag_distributions(
            samples, rarity_weights=config.rarity_weights or RARITY_WEIGHTS
        )
        if local_distributions and local_total > 0:
            total_stats_samples = local_total
            idf_map = compute_tag_idf(local_distributions, total_stats_samples)
            stats_source = f"{input_path}#local"
            stats_ref_info = {
                "source": stats_source,
                "total_samples": total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }
            combo_counts = local_combo_counts or build_combo_counts(samples)
            combo_mode = "local"
            print(f"  Using local rarity baseline ({total_stats_samples} samples)")
    elif idf_map and baseline_combo_counts:
        combo_counts = baseline_combo_counts
        combo_mode = "external"

    if extension_rarity_mode != "off" and not extension_stats_context:
        local_extension_stats = aggregate_extension_stats(samples)
        if local_extension_stats:
            extension_stats_context = local_extension_stats
            extension_baseline_source = "local"

    if not idf_map:
        print("  Warning: rarity unavailable (no valid tag distributions)")

    # Compute rarity for all samples
    rarity_results = []
    for s in samples:
        labels = s.get("labels") or {}
        rarity = compute_sample_rarity(
            labels, idf_map, total_stats_samples,
            rarity_weights=config.rarity_weights or RARITY_WEIGHTS,
            combo_alpha=config.rarity_combo_alpha,
            combo_counts=combo_counts,
            stats_ref_info=stats_ref_info,
        )
        rarity = augment_rarity_result(
            rarity,
            s,
            extension_stats=extension_stats_context,
            config=config,
            baseline_source=extension_baseline_source,
        )
        rarity_results.append(rarity)

    # Normalize rarity to 1-10
    normalize_rarity_scores(
        rarity_results,
        mode=rarity_mode,
        total_samples=total_stats_samples,
    )
    for rarity in rarity_results:
        _refresh_augmented_rarity_after_core_normalization(rarity)

    runtime = _instantiate_runtime(config, concurrency=config.scoring_concurrency)
    setattr(config, "_adaptive_runtime", runtime)

    # Run LLM scoring
    rate_limiter = (
        AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup)
        if runtime is None and config.rps_limit > 0
        else None
    )
    sem = asyncio.Semaphore(config.scoring_concurrency)
    all_monitors = [None] * total
    all_values = [None] * total

    # Pre-fill resumed values
    skipped_count = 0
    if resume:
        for idx, s in enumerate(samples):
            sid = s.get("id", "")
            resumed_value = _existing_resume_value(s, resumed_values)
            if resumed_value is not None:
                all_values[idx] = resumed_value
                all_monitors[idx] = _resumed_monitor(sid)
                skipped_count += 1

    scored_count = skipped_count
    failed_count = 0
    first_error_logged = False
    start_time = time.time()

    to_score = [i for i in range(total) if all_values[i] is None]
    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    print(f"  Scoring {len(to_score)} samples (skipped {skipped_count} resumed) "
          f"| model={config.scoring_model} concurrency={config.scoring_concurrency} {_rps}")

    sweep = None
    async with httpx.AsyncClient(
        proxy=None,
        timeout=config.request_timeout,
        limits=httpx.Limits(
            max_connections=config.scoring_concurrency + 10,
            max_keepalive_connections=config.scoring_concurrency,
        ),
    ) as client:
        async def score_task(idx):
            nonlocal scored_count, failed_count, first_error_logged
            value, monitor = await score_one(
                client, samples[idx], config.scoring_model, rarity_results[idx],
                idx, total, sem, config=config, rate_limiter=rate_limiter,
            )
            all_values[idx] = value
            all_monitors[idx] = monitor
            if value:
                scored_count += 1
            else:
                failed_count += 1
                if not first_error_logged and monitor:
                    summary = _summarize_first_failure(monitor)
                    if summary:
                        print(summary)
                    first_error_logged = True
            return idx

        with _create_progress() as progress:
            task = progress.add_task("Pass 2", total=len(to_score), info="")
            submission_watermark = max(
                config.scoring_concurrency,
                int(config.scoring_concurrency * config.dir_pipeline_watermark),
            )
            next_submit_idx = 0
            pending_tasks = set()
            inflight_count = 0

            def submit_more_tasks():
                nonlocal next_submit_idx, inflight_count
                while next_submit_idx < len(to_score) and inflight_count < submission_watermark:
                    sample_idx = to_score[next_submit_idx]
                    next_submit_idx += 1
                    pending_tasks.add(asyncio.ensure_future(score_task(sample_idx)))
                    inflight_count += 1

            submit_more_tasks()
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                inflight_count = max(inflight_count - len(done), 0)
                for finished in done:
                    idx = finished.result()
                    _info = format_progress_info(
                        ok_count=scored_count,
                        fail_count=failed_count,
                        request_stats=rate_limiter.stats if rate_limiter else None,
                    )
                    monitor = all_monitors[idx]
                    if llm_progress_cb and monitor:
                        run_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass2")
                        if run_info:
                            _info = f"{_info} • {run_info}"
                    progress.update(task, advance=1, info=_info)
                submit_more_tasks()

        sweep = await _run_pass2_recovery_sweep_in_memory(
            http_client=client,
            samples=samples,
            rarity_results=rarity_results,
            all_values=all_values,
            all_monitors=all_monitors,
            config=config,
        )

    elapsed = time.time() - start_time

    # Attach value to samples
    for i, s in enumerate(samples):
        if all_values[i]:
            s["value"] = all_values[i]

    # Compute selection scores (intra-class quality ranking, no LLM)
    compute_selection_scores(samples, config=config)
    apply_v2_scores(samples, config=config)

    # Write outputs (atomic: write to .tmp then rename)
    scored_path = output_dir / "scored.json"
    tmp_scored = scored_path.with_suffix(".tmp.json")
    with open(tmp_scored, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp_scored, scored_path)

    scored_jsonl_path = output_dir / "scored.jsonl"
    tmp_jsonl = scored_jsonl_path.with_suffix(".tmp.jsonl")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, scored_jsonl_path)

    # Monitor
    monitor_path = output_dir / "monitor_value.jsonl"
    with open(monitor_path, "w", encoding="utf-8") as f:
        for m in all_monitors:
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Failed samples
    failed_samples = [s for i, s in enumerate(samples) if all_values[i] is None]
    if failed_samples:
        failed_path = output_dir / "failed_value.jsonl"
        with open(failed_path, "w", encoding="utf-8") as f:
            for s in failed_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Failure log (aligned with Pass 1 format)
    failed_indices = [i for i, v in enumerate(all_values) if v is None]
    if failed_indices:
        with open(output_dir / "score_failures.jsonl", "w", encoding="utf-8") as f:
            for i in failed_indices:
                m = all_monitors[i]
                record = {
                    "sample_id": samples[i].get("id", f"sample-{i}"),
                    "status": m["status"] if m else "no_result",
                    "error": (m.get("error", "") if m else "no monitor record"),
                    "error_response": (m.get("error_response", "")[:1000] if m else ""),
                    "attempts": (m.get("attempts", 0) if m else 0),
                    "error_class": (m.get("error_class") if m else None),
                    "retryable_infra": (m.get("retryable_infra") if m else None),
                    "http_status": (m.get("http_status") if m else None),
                    "runtime_state": (m.get("runtime_state") if m else None),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Stats
    stats = compute_value_stats(samples, [m for m in all_monitors if m])
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["model"] = config.scoring_model
    stats["input_file"] = str(input_path)
    stats["file"] = file_label or stats.get("file") or input_path.name
    if rate_limiter:
        stats["http_request_stats"] = rate_limiter.stats.to_dict()
    if runtime is not None:
        stats["adaptive_runtime"] = {
            "enabled": True,
            "final": _runtime_snapshot(runtime),
        }
    if sweep:
        stats["recovery_sweep"] = sweep
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": str(stats_source) if stats_source else None,
        "total_samples_in_distribution": total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
        "combo_mode": combo_mode,
        "score_mode": rarity_mode,
    }
    stats["extension_rarity_config"] = {
        "mode": resolve_extension_rarity_mode(config),
        "baseline_source": "external" if stats_source else "local",
        "min_extension_baseline_total": getattr(config, "min_extension_baseline_total", None),
    }
    _annotate_scoring_prompt_stats(stats, config)

    stats_path = output_dir / PASS2_STATS_FILE
    stats_to_write = {k: v for k, v in stats.items() if k != "_raw_scores"}
    tmp_stats = stats_path.with_suffix(".tmp.json")
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats_to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp_stats, stats_path)

    # Conversation-level aggregation
    try:
        def _aggregate_conversations_in_memory():
            from sft_label.conversation import aggregate_conversations, write_conversation_scores

            conv_records = aggregate_conversations(samples)
            if conv_records:
                write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
            return conv_records

        run_with_heartbeat("Aggregating conversation scores", _aggregate_conversations_in_memory)
    except Exception as e:
        print(f"  Warning: conversation aggregation failed: {e}")

    # Dashboard
    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        run_with_heartbeat(
            "Generating scoring dashboard",
            lambda: generate_value_dashboard(
                output_dir,
                scored_file="scored.json",
                stats_file=PASS2_STATS_FILE,
                output_file=PASS2_DASHBOARD_FILE,
            ),
        )
    except Exception as e:
        print(f"  Warning: dashboard generation failed: {e}")

    stats["elapsed_seconds"] = round(elapsed, 1)
    if rate_limiter:
        stats["http_request_stats"] = rate_limiter.stats.to_dict()
    print_scoring_summary(stats, output_dir)

    return stats


@dataclass
class _ScoringFileCollector:
    """Track per-file scoring results for cross-file scoring pipeline."""
    file_idx: int
    labeled_path: Path
    output_dir: Path
    samples: list
    rarity_results: list
    total: int
    stats_source: str = None
    idf_map: dict = field(default_factory=dict)
    total_stats_samples: int = 0
    stats_ref_info: dict = None
    combo_mode: str = "disabled"
    done: int = 0
    ok: int = 0
    fail: int = 0
    values: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    score_indices: list = field(default_factory=list)
    submit_cursor: int = 0
    completed: bool = False

    def __post_init__(self):
        self.values = [None] * self.total
        self.monitors = [None] * self.total


@dataclass
class ScoringDirectoryWorkloadEstimate:
    """Pre-run workload estimate for directory scoring."""
    files_planned: int
    total_samples: int
    baseline_total_llm_calls: int
    initial_estimated_llm_calls: int
    scan_elapsed_seconds: float


def _iter_labeled_samples_streaming(labeled_path, limit=0):
    """Stream labeled samples from .jsonl or a top-level JSON array."""
    labeled_path = Path(labeled_path)
    seen = 0

    if labeled_path.suffix == ".jsonl":
        with open(labeled_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
                    seen += 1
                    if limit > 0 and seen >= limit:
                        return
        return

    decoder = json.JSONDecoder()
    started = False
    eof = False
    buffer = ""
    with open(labeled_path, "r", encoding="utf-8") as f:
        while True:
            if not eof:
                chunk = f.read(65536)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if not started:
                    if buffer[0] != "[":
                        return
                    started = True
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    row, consumed = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        return
                    break
                buffer = buffer[consumed:]
                if isinstance(row, dict):
                    yield row
                    seen += 1
                    if limit > 0 and seen >= limit:
                        return

            if eof:
                return


def _count_scoring_samples_in_file(labeled_path, limit=0):
    """Count samples in a labeled file with per-file limit applied."""
    count = 0
    for _sample in _iter_labeled_samples_streaming(labeled_path, limit=limit):
        count += 1
    return count


def _count_required_llm_calls_in_file(labeled_path, limit=0, config=None):
    """Count samples that still require an LLM scoring call."""
    if _resolve_selective_scoring_policy(config) is None:
        return _count_scoring_samples_in_file(labeled_path, limit=limit)

    calls = 0
    for sample in _iter_labeled_samples_streaming(labeled_path, limit=limit):
        if _selective_scoring_decision(sample, config=config).get("requires_llm", True):
            calls += 1
    return calls


def _should_stream_json_directory_input(labeled_path) -> bool:
    """Return True when a resident JSON file should be converted to temp JSONL."""
    labeled_path = Path(labeled_path)
    if labeled_path.suffix == ".jsonl":
        return False
    try:
        return labeled_path.stat().st_size >= DIRECTORY_JSON_STREAMING_THRESHOLD_BYTES
    except OSError:
        return False


def _spill_json_array_to_jsonl(source_path, *, limit=0):
    """Convert a large JSON array input into a temp JSONL file for chunked scoring."""
    source_path = Path(source_path)
    temp_path = source_path.with_name(f".{source_path.stem}.streaming.jsonl")
    with open(temp_path, "w", encoding="utf-8") as f:
        for sample in _iter_labeled_samples_streaming(source_path, limit=limit):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return temp_path


def estimate_scoring_directory_workload(labeled_files, *, limit=0, config=None):
    """Estimate scoring workload before directory-mode execution."""
    start = time.time()
    total_samples = 0
    required_llm_calls = 0
    for labeled_path in labeled_files:
        total_samples += _count_scoring_samples_in_file(labeled_path, limit=limit)
        required_llm_calls += _count_required_llm_calls_in_file(
            labeled_path,
            limit=limit,
            config=config,
        )

    baseline_calls = required_llm_calls
    # Pass 2 has at most one scoring call per selected slice; retries may increase calls.
    sample_retries = max(getattr(config, "sample_max_retries", SAMPLE_MAX_RETRIES), 1)
    retry_factor = 1.0 + min(0.2, 0.05 * (sample_retries - 1))
    initial_est_calls = int(round(required_llm_calls * retry_factor))
    initial_est_calls = max(initial_est_calls, baseline_calls)

    return ScoringDirectoryWorkloadEstimate(
        files_planned=len(labeled_files),
        total_samples=total_samples,
        baseline_total_llm_calls=baseline_calls,
        initial_estimated_llm_calls=initial_est_calls,
        scan_elapsed_seconds=round(time.time() - start, 2),
    )


def _discover_scored_output_files(root_dir):
    """Find logical scored outputs in a run tree, preferring JSONL over JSON."""
    root_dir = Path(root_dir)
    candidates = []
    for pattern in ("scored*.json", "scored*.jsonl", "**/scored*.json", "**/scored*.jsonl"):
        candidates.extend(root_dir.glob(pattern))

    def _logical_json_family_key(path: Path):
        name = path.name
        if name.endswith(".jsonl"):
            name = name[:-6]
        elif name.endswith(".json"):
            name = name[:-5]
        return path.parent, name

    preferred = {}
    for path in sorted(set(candidates)):
        key = _logical_json_family_key(path)
        existing = preferred.get(key)
        if existing is None:
            preferred[key] = path
            continue
        if existing.suffix == ".json" and path.suffix == ".jsonl":
            preferred[key] = path

    return sorted(preferred.values())


def _discover_labeled_input_files(input_dir):
    """Find labeled inputs in a run tree, preferring JSONL when siblings coexist."""
    input_dir = Path(input_dir)
    candidates = []
    for pattern in (
        "labeled*.json",
        "labeled*.jsonl",
        "*/labeled*.json",
        "*/labeled*.jsonl",
        "**/labeled*.json",
        "**/labeled*.jsonl",
    ):
        candidates.extend(input_dir.glob(pattern))

    def _logical_json_family_key(path: Path):
        name = path.name
        if name.endswith(".jsonl"):
            name = name[:-6]
        elif name.endswith(".json"):
            name = name[:-5]
        return path.parent, name

    preferred = {}
    for path in sorted(set(candidates)):
        key = _logical_json_family_key(path)
        existing = preferred.get(key)
        if existing is None:
            preferred[key] = path
            continue
        if existing.suffix == ".json" and path.suffix == ".jsonl":
            preferred[key] = path

    return sorted(preferred.values())


def _split_integer_budget(total: int, parts: int) -> list[int]:
    """Split a positive integer budget across workers without exceeding the total."""
    total = max(int(total), 1)
    parts = max(int(parts), 1)
    parts = min(parts, total)
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _load_scored_samples(path):
    """Load scored samples from .json or .jsonl."""
    path = Path(path)
    if path.suffix == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _write_scored_samples(primary_path, samples):
    """Write scored outputs back to disk, keeping JSON/JSONL siblings in sync."""
    primary_path = Path(primary_path)
    dir_path = primary_path.parent
    json_path = dir_path / "scored.json"
    jsonl_path = dir_path / "scored.jsonl"

    if json_path.exists() or primary_path.suffix == ".json":
        tmp_json = json_path.with_suffix(".tmp.json")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        os.replace(tmp_json, json_path)

    if jsonl_path.exists() or primary_path.suffix == ".jsonl":
        tmp_jsonl = jsonl_path.with_suffix(".tmp.jsonl")
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(tmp_jsonl, jsonl_path)


def _load_monitor_records(dir_path):
    """Load scoring monitors for per-file stats recomputation."""
    monitor_path = Path(dir_path) / "monitor_value.jsonl"
    if not monitor_path.exists():
        return []

    monitors = []
    with open(monitor_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                monitors.append(json.loads(line))
    return monitors


def _load_existing_pass2_stats(dir_path):
    """Load existing per-file Pass 2 stats if present."""
    stats_path = Path(dir_path) / PASS2_STATS_FILE
    if not stats_path.exists():
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rewrite_directory_global_selection(output_dir, input_dir, config, pprint=print):
    """Recompute selection globally across directory outputs and rewrite files."""
    scored_files = _discover_scored_output_files(output_dir)
    if not scored_files:
        return []

    pprint(f"  Recomputing global selection across {len(scored_files)} scored file(s)")

    summaries = []
    file_entries = []
    for scored_path in scored_files:
        samples = _load_scored_samples(scored_path)
        scored_count = 0
        for sample in samples:
            if sample.get("value"):
                summaries.append(_selection_summary_from_sample(sample, config=config))
                scored_count += 1
        file_entries.append({
            "path": scored_path,
            "scored_count": scored_count,
        })

    selection_results = compute_selection_scores_from_summaries(summaries, config=config)

    updated_stats = []
    cursor = 0
    for entry in file_entries:
        scored_path = entry["path"]
        samples = _load_scored_samples(scored_path)
        for sample in samples:
            value = sample.get("value")
            if not value:
                continue
            if cursor >= len(selection_results):
                break
            selection = selection_results[cursor]
            value["selection_score"] = selection["selection_score"]
            value["intra_class_rank"] = selection["intra_class_rank"]
            apply_v2_scores([sample], config=config)
            cursor += 1

        _write_scored_samples(scored_path, samples)

        monitors = _load_monitor_records(scored_path.parent)
        stats = compute_value_stats(samples, monitors, include_raw_scores=False)
        existing_stats = _load_existing_pass2_stats(scored_path.parent)
        for key in (
            "elapsed_seconds",
            "model",
            "input_file",
            "http_request_stats",
            "weights_used",
            "rarity_config",
            "chunked",
        ):
            if key in existing_stats:
                stats[key] = existing_stats[key]

        input_file = existing_stats.get("input_file")
        file_label_source = Path(input_file) if input_file else scored_path
        stats["file"] = _relative_file_label(file_label_source, input_dir)

        stats_path = scored_path.parent / PASS2_STATS_FILE
        tmp_stats = stats_path.with_suffix(".tmp.json")
        with open(tmp_stats, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "_raw_scores"},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp_stats, stats_path)

        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            generate_value_dashboard(
                scored_path.parent,
                scored_file="scored.json" if (scored_path.parent / "scored.json").exists() else "scored.jsonl",
                stats_file=PASS2_STATS_FILE,
                output_file=PASS2_DASHBOARD_FILE,
                quiet=True,
            )
        except Exception:
            pass
        updated_stats.append(stats)

    return updated_stats


def _flush_scoring_file(collector, config, pprint=print, file_label=None, generate_dashboard=True):
    """Write all scoring outputs for a completed file and release memory.

    Returns the stats dict.
    """
    samples = collector.samples
    all_values = collector.values
    all_monitors = collector.monitors
    output_dir = collector.output_dir

    # Attach value to samples
    for i, s in enumerate(samples):
        if all_values[i]:
            s["value"] = all_values[i]

    # Compute selection scores (intra-class quality ranking, no LLM)
    compute_selection_scores(samples, config=config)
    apply_v2_scores(samples, config=config)

    # Write outputs (atomic: write to .tmp then rename)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_path = output_dir / "scored.json"
    tmp_scored = scored_path.with_suffix(".tmp.json")
    with open(tmp_scored, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp_scored, scored_path)

    scored_jsonl_path = output_dir / "scored.jsonl"
    tmp_jsonl = scored_jsonl_path.with_suffix(".tmp.jsonl")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, scored_jsonl_path)

    monitor_path = output_dir / "monitor_value.jsonl"
    with open(monitor_path, "w", encoding="utf-8") as f:
        for m in all_monitors:
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    failed_samples = [s for i, s in enumerate(samples) if all_values[i] is None]
    if failed_samples:
        failed_path = output_dir / "failed_value.jsonl"
        with open(failed_path, "w", encoding="utf-8") as f:
            for s in failed_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Failure log (aligned with Pass 1 format)
    failed_indices = [i for i, v in enumerate(all_values) if v is None]
    if failed_indices:
        with open(output_dir / "score_failures.jsonl", "w", encoding="utf-8") as f:
            for i in failed_indices:
                m = all_monitors[i]
                record = {
                    "sample_id": samples[i].get("id", f"sample-{i}"),
                    "source_file": str(collector.labeled_path),
                    "status": m["status"] if m else "no_result",
                    "error": (m.get("error", "") if m else "no monitor record"),
                    "error_response": (m.get("error_response", "")[:1000] if m else ""),
                    "attempts": (m.get("attempts", 0) if m else 0),
                    "error_class": (m.get("error_class") if m else None),
                    "retryable_infra": (m.get("retryable_infra") if m else None),
                    "http_status": (m.get("http_status") if m else None),
                    "runtime_state": (m.get("runtime_state") if m else None),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    valid_monitors = [m for m in all_monitors if m]
    stats = compute_value_stats(samples, valid_monitors, include_raw_scores=False)
    stats["model"] = config.scoring_model
    stats["input_file"] = str(collector.labeled_path)
    stats["file"] = file_label or stats.get("file") or collector.labeled_path.name
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    runtime = getattr(config, "_adaptive_runtime", None)
    if runtime is not None:
        stats["adaptive_runtime"] = {
            "enabled": True,
            "final": _runtime_snapshot(runtime),
        }
    sweep = getattr(collector, "recovery_sweep", None)
    if sweep:
        stats["recovery_sweep"] = sweep
    stats["rarity_config"] = {
        "stats_ref": collector.stats_source,
        "total_samples_in_distribution": collector.total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
        "combo_mode": collector.combo_mode,
        "score_mode": resolve_rarity_mode(config),
    }
    stats["extension_rarity_config"] = {
        "mode": resolve_extension_rarity_mode(config),
        "baseline_source": "external" if collector.stats_source else "local",
        "min_extension_baseline_total": getattr(config, "min_extension_baseline_total", None),
    }
    _annotate_scoring_prompt_stats(stats, config)

    stats_path = output_dir / PASS2_STATS_FILE
    stats_to_write = {k: v for k, v in stats.items() if k != "_raw_scores"}
    tmp_stats = stats_path.with_suffix(".tmp.json")
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats_to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp_stats, stats_path)

    # Conversation-level aggregation
    try:
        from sft_label.conversation import aggregate_conversations, write_conversation_scores
        conv_records = aggregate_conversations(samples)
        if conv_records:
            write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
    except Exception:
        pass

    if generate_dashboard:
        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            generate_value_dashboard(output_dir, scored_file="scored.json",
                                     stats_file=PASS2_STATS_FILE,
                                     output_file=PASS2_DASHBOARD_FILE,
                                     quiet=True)
        except Exception:
            pass

    # Release memory
    collector.samples = None
    collector.values = None
    collector.monitors = None
    collector.rarity_results = None
    collector.score_indices = []
    collector.submit_cursor = 0
    collector.idf_map = {}
    collector.stats_ref_info = None
    collector.completed = True

    return stats


async def _run_scoring_directory(input_dir, output_dir, tag_stats_path, limit, config,
                                 resume=False, llm_progress_cb=None,
                                 precomputed_workload_estimate=None):
    """Score all labeled files in a directory with cross-file parallelism.

    Uses watermark-based file loading (same pattern as Pass 1's
    run_directory_pipeline) to keep the semaphore saturated across files.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)

    # Find labeled files (flat, nested, recursive). Prefer JSONL siblings.
    labeled_files = _discover_labeled_input_files(input_dir)

    if not labeled_files:
        print(f"No labeled*.json/jsonl files found in {input_dir}")
        return {}

    rarity_mode = resolve_rarity_mode(config)

    print(f"Found {len(labeled_files)} labeled files in {input_dir}")
    workload_estimate = precomputed_workload_estimate
    if workload_estimate is None:
        workload_estimate = run_with_heartbeat(
            "Estimating Pass 2 workload",
            lambda: estimate_scoring_directory_workload(
                labeled_files,
                limit=limit,
                config=config,
            ),
        )
    print(
        "  Plan | "
        f"{workload_estimate.files_planned} files, "
        f"{workload_estimate.total_samples} samples, "
        f"llm~{workload_estimate.initial_estimated_llm_calls}, "
        f"scan {workload_estimate.scan_elapsed_seconds:.1f}s"
    )

    # Use Pass 1 summary stats for rarity if no explicit stats provided
    if tag_stats_path is None:
        candidate = input_dir / PASS1_SUMMARY_STATS_FILE
        if candidate.exists():
            tag_stats_path = str(candidate)
        else:
            # Try per-file stats (flat, one-level, or recursive)
            stats_files = sorted(input_dir.glob("stats_labeling*.json"))
            if not stats_files:
                stats_files = sorted(input_dir.glob("*/stats_labeling*.json"))
            if not stats_files:
                stats_files = sorted(input_dir.glob("**/stats_labeling*.json"))
            if stats_files:
                tag_stats_path = str(stats_files[0])

    # Load global tag stats once (shared across all files)
    global_idf_map = {}
    global_total_stats_samples = 0
    global_stats_ref_info = None
    global_stats_source = str(tag_stats_path) if tag_stats_path else None
    global_combo_counts = None
    global_combo_mode = "disabled"

    def _scan_local_distributions():
        dims = list((config.rarity_weights or RARITY_WEIGHTS).keys())
        distributions = {dim: {} for dim in dims}
        total = 0
        combo_counts = {}
        for labeled_path in labeled_files:
            for sample in _iter_labeled_samples_streaming(labeled_path, limit=limit):
                labels = sample.get("labels") or {}
                if _update_distributions_from_labels(distributions, labels, dims):
                    total += 1
                _update_combo_counts_from_labels(combo_counts, labels)
        distributions = {d: c for d, c in distributions.items() if c}
        return distributions, total, combo_counts

    local_distributions = None
    local_combo_counts = None

    if tag_stats_path:
        distributions, global_total_stats_samples, ts, meta = load_tag_stats_context(tag_stats_path)
        if distributions:
            global_idf_map = compute_tag_idf(distributions, global_total_stats_samples)
            global_stats_ref_info = {
                "source": str(tag_stats_path),
                "total_samples": global_total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            missing_dims = [
                d for d in (config.rarity_weights or RARITY_WEIGHTS).keys()
                if d not in global_idf_map
            ]
            if missing_dims:
                print("  Warning: rarity baseline missing dimensions; ignored in IDF: "
                      + ", ".join(missing_dims))
            global_combo_counts = meta.get("combo_counts")
            if global_combo_counts:
                global_combo_mode = "external"
        else:
            print(f"  Warning: {tag_stats_path} has no tag_distributions, fallback to local baseline")

    if not global_idf_map or not global_combo_counts:
        local_distributions, local_total, local_combo_counts = _scan_local_distributions()

    if global_idf_map and not global_combo_counts:
        global_combo_counts, global_combo_mode, combo_msg = _resolve_combo_baseline(
            None,
            local_combo_counts,
        )
        if combo_msg:
            print(combo_msg)

    if not global_idf_map:
        if local_distributions and local_total > 0:
            global_total_stats_samples = local_total
            global_idf_map = compute_tag_idf(local_distributions, local_total)
            global_combo_counts = local_combo_counts
            global_combo_mode = "local"
            global_stats_source = f"{input_dir}#local"
            global_stats_ref_info = {
                "source": global_stats_source,
                "total_samples": global_total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }
            print(f"  Using local rarity baseline ({local_total} samples)")
        else:
            print("  Warning: rarity unavailable (no valid tag distributions)")

    shared_tag_stats_path = str(tag_stats_path) if tag_stats_path else None
    temp_tag_stats_path = None
    if shared_tag_stats_path is None and local_distributions and global_total_stats_samples > 0:
        temp_tag_stats_path = output_dir / ".directory_rarity_stats.json"
        with open(temp_tag_stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "tag_distributions": local_distributions,
                "combo_distributions": global_combo_counts or {},
                "total_samples": global_total_stats_samples,
                "distribution_total_samples": global_total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
        shared_tag_stats_path = str(temp_tag_stats_path)

    streaming_inputs = [(p, p, None) for p in labeled_files if p.suffix == ".jsonl"]
    resident_files = []
    for labeled_path in labeled_files:
        if labeled_path.suffix == ".jsonl":
            continue
        if _should_stream_json_directory_input(labeled_path):
            temp_jsonl = _spill_json_array_to_jsonl(labeled_path, limit=limit)
            streaming_inputs.append((labeled_path, temp_jsonl, temp_jsonl))
            print(
                f"  Large JSON detected; using streamed temp JSONL for "
                f"{labeled_path.relative_to(input_dir)}"
            )
        else:
            resident_files.append(labeled_path)

    concurrency = config.scoring_concurrency
    watermark = max(concurrency, int(concurrency * config.dir_pipeline_watermark))
    max_active = config.dir_pipeline_max_files
    runtime = _instantiate_runtime(config, concurrency=concurrency)
    setattr(config, "_adaptive_runtime", runtime)
    rate_limiter = (
        AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup)
        if runtime is None and config.rps_limit > 0
        else None
    )
    sem = asyncio.Semaphore(concurrency)
    pprint = print

    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    print(f"  Scoring: model={config.scoring_model} concurrency={concurrency} {_rps}")

    batch_start = time.time()

    # --- Helper: wrap score_one with file/sample tracking ---
    async def _tagged_score(client, sample, rarity_result, file_idx, sample_idx,
                            total_in_file):
        value, monitor = await score_one(
            client, sample, config.scoring_model, rarity_result,
            sample_idx, total_in_file, sem, config=config, rate_limiter=rate_limiter,
        )
        return file_idx, sample_idx, value, monitor

    # --- Load a file and prepare its work queue ---
    def load_and_prepare(file_idx, labeled_path):
        # Load samples
        with open(labeled_path, "r", encoding="utf-8") as f:
            if labeled_path.suffix == ".jsonl":
                samples = [json.loads(line) for line in f if line.strip()]
            else:
                samples = json.load(f)
        if limit > 0:
            samples = samples[:limit]
        total = len(samples)

        # Per-file rarity: always use one shared combo baseline for consistency
        combo_counts = global_combo_counts if global_idf_map else None
        rarity_results = []
        for s in samples:
            labels = s.get("labels") or {}
            rarity = compute_sample_rarity(
                labels, global_idf_map, global_total_stats_samples,
                rarity_weights=config.rarity_weights or RARITY_WEIGHTS,
                combo_alpha=config.rarity_combo_alpha,
                combo_counts=combo_counts,
                stats_ref_info=global_stats_ref_info,
            )
            rarity_results.append(rarity)
        normalize_rarity_scores(
            rarity_results,
            mode=rarity_mode,
            total_samples=global_total_stats_samples,
        )
        for rarity in rarity_results:
            _refresh_augmented_rarity_after_core_normalization(rarity)

        file_output_dir = labeled_path.parent
        collector = _ScoringFileCollector(
            file_idx=file_idx,
            labeled_path=labeled_path,
            output_dir=file_output_dir,
            samples=samples,
            rarity_results=rarity_results,
            total=total,
            stats_source=global_stats_source,
            idf_map=global_idf_map,
            total_stats_samples=global_total_stats_samples,
            stats_ref_info=global_stats_ref_info,
            combo_mode=global_combo_mode,
        )
        resumed_values = _load_resumed_value_cache(file_output_dir) if resume else {}
        if resumed_values:
            pprint(
                f"  Resume: loaded {len(resumed_values)} pre-scored samples for "
                f"{labeled_path.relative_to(input_dir)}"
            )

        resumed_loaded = 0
        # Prepare scoring indices (bounded submission happens separately)
        for i in range(total):
            resumed_value = _existing_resume_value(samples[i], resumed_values)
            if resumed_value is not None:
                collector.values[i] = resumed_value
                collector.monitors[i] = _resumed_monitor(samples[i].get("id", f"sample-{i}"))
                collector.done += 1
                collector.ok += 1
                resumed_loaded += 1
                continue
            collector.score_indices.append(i)

        return collector, resumed_loaded

    # --- Try to load more files if below watermark ---
    def maybe_load_more(pending_futures, collectors, next_to_load, extra_active_files=0):
        active_count = sum(1 for c in collectors.values() if not c.completed) + extra_active_files
        resumed_loaded = 0
        while (next_to_load < len(resident_files)
               and len(pending_futures) < watermark
               and active_count < max_active):
            labeled_path = resident_files[next_to_load]
            c, file_resumed = load_and_prepare(next_to_load, labeled_path)
            collectors[c.file_idx] = c
            next_to_load += 1
            active_count += 1
            resumed_loaded += file_resumed
        return next_to_load, resumed_loaded

    def submit_more_tasks(pending_futures, collectors, client):
        submitted = 0
        capacity = max(watermark - len(pending_futures), 0)
        if capacity <= 0:
            return submitted

        active_collectors = [
            c for c in sorted(collectors.values(), key=lambda item: item.file_idx)
            if not c.completed
        ]
        while capacity > 0:
            progressed = False
            for collector in active_collectors:
                if collector.submit_cursor >= len(collector.score_indices):
                    continue
                sample_idx = collector.score_indices[collector.submit_cursor]
                collector.submit_cursor += 1
                fut = asyncio.ensure_future(
                    _tagged_score(
                        client,
                        collector.samples[sample_idx],
                        collector.rarity_results[sample_idx],
                        collector.file_idx,
                        sample_idx,
                        collector.total,
                    )
                )
                pending_futures.add(fut)
                submitted += 1
                capacity -= 1
                progressed = True
                if capacity <= 0:
                    break
            if not progressed:
                break
        return submitted

    # --- Main watermark-driven loop ---
    collectors = {}
    pending_futures = set()
    all_file_stats = []
    next_to_load = 0
    first_error_logged = False
    eta_tracker = RuntimeEtaEstimator(
        total_labeled_samples=workload_estimate.total_samples,
        initial_estimated_calls=workload_estimate.initial_estimated_llm_calls,
    )

    try:
        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_connections=concurrency + 10,
                max_keepalive_connections=concurrency,
            ),
        ) as client:
            with _create_progress() as progress:
                global_llm_info = None
                streaming_ok = 0
                streaming_fail = 0
                file_task = progress.add_task("Files", total=len(labeled_files), info="")
                sample_task = progress.add_task(
                    "Pass 2",
                    total=workload_estimate.total_samples,
                    visible=bool(labeled_files),
                    info="starting...",
                )
                llm_task = progress.add_task(
                    "LLM",
                    total=max(eta_tracker.estimated_total_calls, 1),
                    visible=bool(labeled_files),
                    info=eta_tracker.info_line(),
                )
                pprint = progress.console.print

                def _combined_counts():
                    resident_ok = sum(cc.ok for cc in collectors.values())
                    resident_fail = sum(cc.fail for cc in collectors.values())
                    return resident_ok + streaming_ok, resident_fail + streaming_fail

                def _update_llm_progress(monitor=None):
                    nonlocal global_llm_info
                    if llm_progress_cb and monitor:
                        global_llm_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass2")
                    if monitor is not None:
                        eta_tracker.update(monitor.get("llm_calls", 0))
                    global_counts = parse_run_progress(global_llm_info) if global_llm_info else None
                    if global_counts:
                        g_done, g_total = global_counts
                        progress.update(
                            llm_task,
                            total=max(g_total, 1),
                            completed=min(g_done, g_total),
                            info=global_llm_info,
                        )
                    else:
                        progress.update(
                            llm_task,
                            total=max(eta_tracker.estimated_total_calls, eta_tracker.calls_done, 1),
                            completed=eta_tracker.calls_done,
                            info=eta_tracker.info_line(),
                        )

                def _update_sample_progress(*, advance, label, monitor=None, skipped_info=None):
                    total_ok, total_fail = _combined_counts()
                    info = skipped_info or format_progress_info(
                        ok_count=total_ok,
                        fail_count=total_fail,
                        label=label,
                        request_stats=rate_limiter.stats if rate_limiter else None,
                    )
                    progress.update(sample_task, advance=advance, info=info)
                    _update_llm_progress(monitor)

                def _streaming_progress_hook(event):
                    nonlocal streaming_ok, streaming_fail
                    label = event.get("label")
                    if event.get("type") == "resumed":
                        skipped_count = event.get("skipped_count", event.get("count", 0))
                        _update_sample_progress(
                            advance=event.get("count", 0),
                            label=label,
                            skipped_info=f"skipped {skipped_count} resumed",
                        )
                        return
                    if event.get("type") != "sample":
                        return
                    if event.get("value"):
                        streaming_ok += 1
                    else:
                        streaming_fail += 1
                    _update_sample_progress(
                        advance=event.get("count", 1),
                        label=label,
                        monitor=event.get("monitor"),
                    )

                async def flush_completed_collectors():
                    flushed = 0
                    for collector in collectors.values():
                        if collector.done < collector.total or collector.completed:
                            continue
                        sweep = await _run_pass2_recovery_sweep_in_memory(
                            http_client=client,
                            samples=collector.samples,
                            rarity_results=collector.rarity_results,
                            all_values=collector.values,
                            all_monitors=collector.monitors,
                            config=config,
                        )
                        setattr(collector, "recovery_sweep", sweep)
                        collector.ok = sum(1 for v in collector.values if v)
                        collector.fail = sum(1 for v in collector.values if v is None)
                        stats = _flush_scoring_file(
                            collector,
                            config,
                            pprint=pprint,
                            file_label=_relative_file_label(collector.labeled_path, input_dir),
                            generate_dashboard=False,
                        )
                        stats["file"] = stats.get("file") or _relative_file_label(collector.labeled_path, input_dir)
                        all_file_stats.append(stats)
                        progress.update(file_task, advance=1)
                        flushed += 1
                    return flushed

                streaming_parallelism = min(
                    len(streaming_inputs),
                    max(int(max_active), 1),
                    max(int(concurrency), 1),
                ) if streaming_inputs else 0
                active_streaming_tasks = {}
                next_streaming_idx = 0

                async def _run_streaming_file(original_path, scoring_input_path):
                    file_config = copy.copy(config)
                    if runtime is not None:
                        setattr(file_config, "_adaptive_runtime", runtime)
                    stats = await _run_scoring_file_chunked(
                        scoring_input_path,
                        original_path.parent,
                        shared_tag_stats_path,
                        limit,
                        file_config,
                        resume=resume,
                        llm_progress_cb=llm_progress_cb,
                        file_label=_relative_file_label(original_path, input_dir),
                        show_progress=False,
                        combo_counts_override=global_combo_counts,
                        combo_mode_override=global_combo_mode,
                        shared_http_client=client,
                        shared_semaphore=sem,
                        shared_rate_limiter=rate_limiter,
                        shared_runtime=runtime,
                        progress_hook=_streaming_progress_hook,
                        print_summary=False,
                        generate_dashboard=False,
                        quiet=True,
                    )
                    stats["file"] = stats.get("file") or _relative_file_label(original_path, input_dir)
                    return stats

                def _resident_active_count():
                    return sum(1 for c in collectors.values() if not c.completed)

                def _update_active_file_info():
                    streaming_names = list(active_streaming_tasks.values())
                    resident_names = [
                        c.labeled_path.name for c in collectors.values() if not c.completed
                    ]
                    active_names = streaming_names + resident_names
                    progress.update(file_task, info=", ".join(active_names)[:60] if active_names else "done")

                def submit_more_streaming_tasks():
                    nonlocal next_streaming_idx
                    started = 0
                    while (
                        next_streaming_idx < len(streaming_inputs)
                        and len(active_streaming_tasks) < streaming_parallelism
                        and (_resident_active_count() + len(active_streaming_tasks)) < max_active
                    ):
                        original_path, scoring_input_path, _temp_input_path = streaming_inputs[next_streaming_idx]
                        next_streaming_idx += 1
                        task = asyncio.create_task(_run_streaming_file(original_path, scoring_input_path))
                        active_streaming_tasks[task] = _relative_file_label(original_path, input_dir)
                        started += 1
                    return started

                async def top_up_work():
                    nonlocal next_to_load
                    while True:
                        started_streaming = submit_more_streaming_tasks()
                        next_to_load, resumed_loaded = maybe_load_more(
                            pending_futures,
                            collectors,
                            next_to_load,
                            extra_active_files=len(active_streaming_tasks),
                        )
                        if resumed_loaded:
                            _update_sample_progress(
                                advance=resumed_loaded,
                                label="resident",
                                skipped_info=f"skipped {resumed_loaded} resumed",
                            )
                        submit_more_tasks(pending_futures, collectors, client)
                        flushed = await flush_completed_collectors()
                        if not (started_streaming or resumed_loaded or flushed):
                            break
                    _update_active_file_info()

                await top_up_work()

                while (
                    pending_futures
                    or active_streaming_tasks
                    or next_to_load < len(resident_files)
                    or next_streaming_idx < len(streaming_inputs)
                ):
                    waitables = set(pending_futures) | set(active_streaming_tasks.keys())
                    if not waitables:
                        await top_up_work()
                        waitables = set(pending_futures) | set(active_streaming_tasks.keys())
                        if not waitables:
                            break

                    done, _pending = await asyncio.wait(
                        waitables, return_when=asyncio.FIRST_COMPLETED
                    )
                    pending_futures.difference_update(done)

                    for fut in done:
                        if fut in active_streaming_tasks:
                            active_streaming_tasks.pop(fut, None)
                            stats = fut.result()
                            all_file_stats.append(stats)
                            progress.update(file_task, advance=1)
                            continue

                        file_idx, sample_idx, value, monitor = fut.result()
                        c = collectors[file_idx]

                        if 0 <= sample_idx < c.total:
                            c.values[sample_idx] = value
                            c.monitors[sample_idx] = monitor

                        c.done += 1
                        if value:
                            c.ok += 1
                        else:
                            c.fail += 1
                            if not first_error_logged and monitor:
                                summary = _summarize_first_failure(monitor)
                                if summary:
                                    pprint(summary)
                                first_error_logged = True

                        _update_sample_progress(
                            advance=1,
                            label=c.labeled_path.name,
                            monitor=monitor,
                        )

                    await top_up_work()

                await flush_completed_collectors()
                _update_active_file_info()
    finally:
        if temp_tag_stats_path and temp_tag_stats_path.exists():
            temp_tag_stats_path.unlink()
        for _original, _input, temp_input_path in streaming_inputs:
            if temp_input_path is not None and temp_input_path.exists():
                temp_input_path.unlink()

    elapsed = time.time() - batch_start

    if all_file_stats:
        rewritten_file_stats = _rewrite_directory_global_selection(
            output_dir=output_dir,
            input_dir=input_dir,
            config=config,
            pprint=pprint,
        )
        if rewritten_file_stats:
            all_file_stats = rewritten_file_stats

    # Write global summary
    if all_file_stats:
        summary = _merge_value_stats(all_file_stats)
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["model"] = config.scoring_model
        summary["input_path"] = str(input_dir)
        summary["files_processed"] = len(all_file_stats)
        summary["planned_files"] = workload_estimate.files_planned
        summary["planned_samples"] = workload_estimate.total_samples
        summary["planned_baseline_llm_calls"] = workload_estimate.baseline_total_llm_calls
        summary["planned_initial_llm_calls"] = workload_estimate.initial_estimated_llm_calls
        summary["planning_elapsed_seconds"] = workload_estimate.scan_elapsed_seconds
        summary["rarity_config"] = {
            "stats_ref": global_stats_source,
            "total_samples_in_distribution": global_total_stats_samples,
            "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
            "combo_alpha": config.rarity_combo_alpha,
            "combo_mode": global_combo_mode,
            "score_mode": rarity_mode,
        }
        summary["extension_rarity_config"] = {
            "mode": resolve_extension_rarity_mode(config),
            "baseline_source": "external" if global_stats_source else "local",
            "min_extension_baseline_total": getattr(config, "min_extension_baseline_total", None),
        }
        if rate_limiter:
            summary["http_request_stats"] = rate_limiter.stats.to_dict()
        if runtime is not None:
            summary["adaptive_runtime"] = {
                "enabled": True,
                "final": _runtime_snapshot(runtime),
            }
        _annotate_scoring_prompt_stats(summary, config)

        summary_path = output_dir / PASS2_SUMMARY_STATS_FILE
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Global conversation-level aggregation across all files
        try:
            from sft_label.conversation import (
                merge_conversation_record_batches,
                write_conversation_scores,
            )

            conv_batches = []
            for conv_path in sorted(output_dir.rglob("conversation_scores.json")):
                if conv_path.parent == output_dir:
                    continue
                with open(conv_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    conv_batches.append(data)

            conv_records = merge_conversation_record_batches(conv_batches)
            if conv_records:
                write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
        except Exception as e:
            print(f"  Warning: global conversation aggregation failed: {e}")

        # Global dashboard
        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            dir_name = input_dir.name
            generate_value_dashboard(output_dir, scored_file=None,
                                     stats_file=PASS2_SUMMARY_STATS_FILE,
                                     output_file=pass2_global_dashboard_filename(dir_name),
                                     quiet=True)
        except Exception as e:
            print(f"  Warning: global dashboard generation failed: {e}")

        print_scoring_summary(summary, output_dir, is_batch=True)

    return summary if all_file_stats else {}


def _merge_value_stats(file_stats_list):
    """Merge per-file value stats into a global summary."""
    merged = {
        "total_scored": sum(s.get("total_scored", 0) for s in file_stats_list),
        "total_failed": sum(s.get("total_failed", 0) for s in file_stats_list),
        "total_estimated": sum(s.get("total_estimated", 0) for s in file_stats_list),
        "total_llm_calls": sum(s.get("total_llm_calls", 0) for s in file_stats_list),
        "total_prompt_tokens": sum(s.get("total_prompt_tokens", 0) for s in file_stats_list),
        "total_completion_tokens": sum(s.get("total_completion_tokens", 0) for s in file_stats_list),
        "total_tokens": sum(s.get("total_tokens", 0) for s in file_stats_list),
    }

    # Per-file summary
    merged["per_file_summary"] = [
        {
            "file": s.get("file", "unknown"),
            "count": s.get("total_scored", 0),
            "mean_value": s.get("score_distributions", {}).get("value_score", {}).get("mean", 0),
            "mean_complexity": s.get("score_distributions", {}).get("complexity_overall", {}).get("mean", 0),
            "mean_quality": s.get("score_distributions", {}).get("quality_overall", {}).get("mean", 0),
            "mean_rarity": s.get("score_distributions", {}).get("rarity_score", {}).get("mean", 0),
            "mean_selection": s.get("score_distributions", {}).get("selection_score", {}).get("mean", 0),
            "keep_rates": dict(s.get("keep_rates") or {}),
            "keep_rate_7": s.get("keep_rate_7", 0),
            "mean_turns": s.get("mean_turns", 0),
        }
        for s in file_stats_list
    ]

    # Merge score distributions: use raw scores when available for exact percentiles
    all_dist_keys = set()
    for s in file_stats_list:
        all_dist_keys.update(s.get("score_distributions", {}).keys())
    merged_distributions = {}
    for key in all_dist_keys:
        # Collect raw values from all files (attached by compute_value_stats)
        all_values = []
        for s in file_stats_list:
            raw = s.get("_raw_scores", {}).get(key)
            if raw:
                all_values.extend(raw)

        if all_values:
            # Exact percentiles from raw data
            merged_distributions[key] = _percentiles(all_values)
        else:
            # Fallback without raw arrays: use merged histograms when available.
            merged_bins = [0] * 10
            has_bins = False
            for s in file_stats_list:
                bins = s.get("histograms", {}).get(key)
                if isinstance(bins, list) and len(bins) == 10:
                    has_bins = True
                    for i, value in enumerate(bins):
                        merged_bins[i] += value
            if has_bins and sum(merged_bins) > 0:
                merged_distributions[key] = _percentiles_from_histogram_bins(merged_bins)
                continue

            # Last-resort fallback: weighted mean + global min/max.
            total_n = 0
            total_sum = 0.0
            global_min = float("inf")
            global_max = float("-inf")
            for s in file_stats_list:
                dist = s.get("score_distributions", {}).get(key, {})
                n = s.get("total_scored", 0)
                if not dist or n == 0 or not isinstance(dist, dict):
                    continue
                mean = dist.get("mean", 0)
                total_sum += mean * n
                total_n += n
                if "min" in dist:
                    global_min = min(global_min, dist["min"])
                if "max" in dist:
                    global_max = max(global_max, dist["max"])
            if total_n > 0:
                merged_distributions[key] = {
                    "mean": round(total_sum / total_n, 2),
                    "min": global_min if global_min != float("inf") else 0,
                    "max": global_max if global_max != float("-inf") else 0,
                    "total_n": total_n,
                }
    merged["score_distributions"] = merged_distributions

    # Merge histogram bins: sum across files
    merged_histograms = {}
    for s in file_stats_list:
        for key, bins in s.get("histograms", {}).items():
            if key not in merged_histograms:
                merged_histograms[key] = [0] * 10
            for i, v in enumerate(bins):
                merged_histograms[key][i] += v
    merged["histograms"] = merged_histograms

    # Merge thinking mode stats: sum counts, weighted means
    merged_thinking = {"slow": {"count": 0, "sum_value": 0.0, "sum_quality": 0.0, "sum_reasoning": 0.0},
                       "fast": {"count": 0, "sum_value": 0.0, "sum_quality": 0.0, "sum_reasoning": 0.0}}
    for s in file_stats_list:
        ts = s.get("thinking_mode_stats", {})
        for mode in ("slow", "fast"):
            ms = ts.get(mode, {})
            n = ms.get("count", 0)
            merged_thinking[mode]["count"] += n
            merged_thinking[mode]["sum_value"] += ms.get("mean_value", 0) * n
            merged_thinking[mode]["sum_quality"] += ms.get("mean_quality", 0) * n
            merged_thinking[mode]["sum_reasoning"] += ms.get("mean_reasoning", 0) * n
    merged["thinking_mode_stats"] = {
        mode: {
            "count": d["count"],
            "mean_value": round(d["sum_value"] / max(d["count"], 1), 2),
            "mean_quality": round(d["sum_quality"] / max(d["count"], 1), 2),
            "mean_reasoning": round(d["sum_reasoning"] / max(d["count"], 1), 2),
        }
        for mode, d in merged_thinking.items()
    }

    # Merge flag counts: sum across files
    merged_flags = {}
    for s in file_stats_list:
        for flag, count in s.get("flag_counts", {}).items():
            merged_flags[flag] = merged_flags.get(flag, 0) + count
    merged["flag_counts"] = dict(sorted(merged_flags.items(), key=lambda x: -x[1]))

    # Merge value_by_tag: weighted mean across files
    all_tag_dims = set()
    for s in file_stats_list:
        all_tag_dims.update(s.get("value_by_tag", {}).keys())
    merged_by_tag = {}
    for dim in all_tag_dims:
        tag_accum = {}  # tag -> {sum, count}
        for s in file_stats_list:
            dim_data = s.get("value_by_tag", {}).get(dim, {})
            for tag, info in dim_data.items():
                if tag not in tag_accum:
                    tag_accum[tag] = {"sum": 0.0, "count": 0}
                n = info.get("n", 0)
                tag_accum[tag]["sum"] += info.get("mean", 0) * n
                tag_accum[tag]["count"] += n
        merged_by_tag[dim] = {
            t: {"mean": round(a["sum"] / max(a["count"], 1), 2), "n": a["count"]}
            for t, a in sorted(tag_accum.items(),
                                key=lambda x: -x[1]["sum"] / max(x[1]["count"], 1))
        }
    merged["value_by_tag"] = merged_by_tag

    # Merge selection_by_tag: same pattern as value_by_tag
    all_sel_dims = set()
    for s in file_stats_list:
        all_sel_dims.update(s.get("selection_by_tag", {}).keys())
    merged_sel_by_tag = {}
    for dim in all_sel_dims:
        tag_accum = {}
        for s in file_stats_list:
            dim_data = s.get("selection_by_tag", {}).get(dim, {})
            for tag, info in dim_data.items():
                if tag not in tag_accum:
                    tag_accum[tag] = {"sum": 0.0, "count": 0}
                n = info.get("n", 0)
                tag_accum[tag]["sum"] += info.get("mean", 0) * n
                tag_accum[tag]["count"] += n
        merged_sel_by_tag[dim] = {
            t: {"mean": round(a["sum"] / max(a["count"], 1), 2), "n": a["count"]}
            for t, a in sorted(tag_accum.items(),
                                key=lambda x: -x[1]["sum"] / max(x[1]["count"], 1))
        }
    merged["selection_by_tag"] = merged_sel_by_tag

    total_scored = merged["total_scored"]
    merged_keep_rates = {
        f"{float(threshold):.1f}": round(
            sum(
                ((s.get("keep_rates") or {}).get(f"{float(threshold):.1f}", 0) or 0)
                * (s.get("total_scored", 0) or 0)
                for s in file_stats_list
            ) / max(total_scored, 1),
            4,
        )
        for threshold in FILE_RANKING_KEEP_RATE_THRESHOLDS
    }
    merged["keep_rates"] = merged_keep_rates
    merged["keep_rate_7"] = round(
        sum((s.get("keep_rate_7", 0) or 0) * (s.get("total_scored", 0) or 0) for s in file_stats_list) / max(total_scored, 1),
        4,
    )
    merged["mean_turns"] = round(
        sum((s.get("mean_turns", 0) or 0) * (s.get("total_scored", 0) or 0) for s in file_stats_list) / max(total_scored, 1),
        2,
    )

    return merged


# Public alias for external use (recompute module)
merge_value_stats = _merge_value_stats
