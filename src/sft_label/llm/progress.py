from __future__ import annotations

import math
import time
from collections import deque

from sft_label.llm.factory import _config_adaptive_runtime


def format_progress_info(ok_count, fail_count=0, label=None, request_stats=None):
    """Build a compact progress info string for live terminal updates."""
    parts = [f"ok={ok_count}"]
    if fail_count:
        parts.append(f"fail={fail_count}")
    if label:
        parts.append(label)
    if request_stats is not None:
        http = request_stats.summary_line()
        if http:
            parts.append(http)
    return " • ".join(parts)


def _format_runtime_progress(runtime_snapshot: dict | None) -> str | None:
    if not runtime_snapshot:
        return None
    state = runtime_snapshot.get("state")
    eff_concurrency = runtime_snapshot.get("effective_concurrency")
    eff_rps = runtime_snapshot.get("effective_rps")
    in_flight = runtime_snapshot.get("in_flight")
    if state is None or eff_concurrency is None or eff_rps is None:
        return None
    parts = [f"runtime {state}"]
    cooldown_remaining = runtime_snapshot.get("cooldown_remaining_seconds")
    if cooldown_remaining is None:
        cooldown_until = runtime_snapshot.get("cooldown_until")
        if cooldown_until is not None:
            try:
                cooldown_remaining = max(float(cooldown_until) - time.monotonic(), 0.0)
            except (TypeError, ValueError):
                cooldown_remaining = None
    if state == "open" and cooldown_remaining and cooldown_remaining > 0:
        parts.append(f"cooldown={int(math.ceil(cooldown_remaining))}s")
    parts.extend(
        [
            f"c={int(eff_concurrency)}",
            f"rps={float(eff_rps):.1f}",
            f"in_flight={int(in_flight or 0)}",
        ]
    )
    return " ".join(parts)


def _decorate_run_info_with_runtime(run_info: str | None, config) -> str | None:
    runtime = _config_adaptive_runtime(config)
    if runtime is None:
        return run_info
    runtime_info = _format_runtime_progress(runtime.snapshot())
    if not runtime_info:
        return run_info
    if run_info:
        return f"{run_info} • {runtime_info}"
    return runtime_info


def parse_run_progress(info):
    """Parse 'run <done>/<total> ...' and return (done, total) if available."""
    if not info or not info.startswith("run "):
        return None
    token = info.split(" ", 2)[1] if " " in info else ""
    if "/" not in token:
        return None
    done_s, total_s = token.split("/", 1)
    try:
        done = max(int(done_s), 0)
        total = max(int(total_s), 0)
    except ValueError:
        return None
    if total <= 0:
        return None
    return done, total


def _update_llm_task_progress(progress, llm_task, run_info, eta_tracker=None, config=None):
    """Refresh the global LLM progress task from tracker info or local ETA fallback."""
    if not progress or llm_task is None:
        return

    global_counts = parse_run_progress(run_info) if run_info else None
    if global_counts:
        g_done, g_total = global_counts
        progress.update(
            llm_task,
            total=max(g_total, 1),
            completed=min(g_done, g_total),
            info=run_info,
        )
        return

    if eta_tracker is not None:
        info = _decorate_run_info_with_runtime(eta_tracker.info_line(), config)
        progress.update(
            llm_task,
            total=max(eta_tracker.estimated_total_calls, eta_tracker.calls_done, 1),
            completed=eta_tracker.calls_done,
            info=info,
        )


class RuntimeEtaEstimator:
    """Adaptive ETA tracker based on observed LLM-call throughput."""

    def __init__(self, total_labeled_samples: int, initial_estimated_calls: int):
        self.total_labeled_samples = max(int(total_labeled_samples), 0)
        self.samples_done = 0
        self.calls_done = 0
        self.avg_calls_per_sample = 0.0
        self.calls_per_sec = 0.0
        self.estimated_total_calls = max(int(initial_estimated_calls), 0)
        self._start_time = time.time()
        self._avg_cps = 0.0
        self._recent_cps = 0.0
        self._eta_cps = 0.0
        self._recent_window_seconds = 30.0
        self._recent_samples = deque()
        self._warmup_samples = min(max(8, self.total_labeled_samples // 50), 30)

    def _effective_eta_cps(self) -> float:
        if self._avg_cps <= 0:
            return self._recent_cps
        if self._recent_cps <= 0:
            return self._avg_cps
        return (self._avg_cps * 0.75) + (self._recent_cps * 0.25)

    @staticmethod
    def _fmt_duration(seconds):
        if seconds is None:
            return "--:--"
        sec = max(int(seconds), 0)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(self, sample_llm_calls: int):
        calls = max(int(sample_llm_calls or 0), 0)
        self.samples_done += 1
        self.calls_done += calls

        if self.samples_done > 0:
            self.avg_calls_per_sample = self.calls_done / self.samples_done

        if self.total_labeled_samples > 0 and self.samples_done >= self._warmup_samples:
            target_total = int(round(self.avg_calls_per_sample * self.total_labeled_samples))
            target_total = max(target_total, self.calls_done)
            alpha = 0.2
            blended = int(round((1.0 - alpha) * self.estimated_total_calls + alpha * target_total))
            self.estimated_total_calls = max(blended, self.calls_done)
        else:
            self.estimated_total_calls = max(self.estimated_total_calls, self.calls_done)

        self.record_call_delta(calls)

    def record_call_delta(self, calls: int):
        delta = max(int(calls or 0), 0)
        now = time.time()
        elapsed = max(now - self._start_time, 1e-6)
        self._avg_cps = self.calls_done / elapsed
        self._recent_samples.append((now, delta))
        cutoff = now - self._recent_window_seconds
        while self._recent_samples and self._recent_samples[0][0] < cutoff:
            self._recent_samples.popleft()
        recent_calls = sum(item_delta for _, item_delta in self._recent_samples)
        recent_span = max(min(elapsed, self._recent_window_seconds), 1e-6)
        self._recent_cps = recent_calls / recent_span
        self._eta_cps = self._effective_eta_cps()
        self.calls_per_sec = self._eta_cps

    def eta_seconds(self):
        effective_cps = self.calls_per_sec if self.calls_per_sec > 0 else self._effective_eta_cps()
        if effective_cps <= 0:
            return None
        remaining = max(self.estimated_total_calls - self.calls_done, 0)
        return remaining / effective_cps

    def info_line(self):
        rate = f"{self.calls_per_sec:.1f}/s" if self.calls_per_sec > 0 else "warming"
        eta = self._fmt_duration(self.eta_seconds())
        avg = f"{self.avg_calls_per_sample:.2f}/sample" if self.samples_done > 0 else "n/a"
        if self._avg_cps > 0 or self._recent_cps > 0:
            return (
                f"eta {eta} • eta_rate {rate} • recent {self._recent_cps:.1f}/s "
                f"• avg {self._avg_cps:.1f}/s • calls {avg}"
            )
        return f"eta {eta} • eta_rate {rate} • calls {avg}"
