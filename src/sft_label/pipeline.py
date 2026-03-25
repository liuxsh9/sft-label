"""
SFT Auto-Labeling Pipeline — Concurrent Version

Processes SFT data through the labeling pipeline with high concurrency:
  1. Preprocessing (structural signal extraction) — local, instant
  2. Call 1 — Intent, Language, Domain, Task, Difficulty — concurrent LLM
  3. Call 2 — Concept, Agentic, Constraint, Context — concurrent LLM (depends on Call 1)
  4. Validation — local, instant
  5. Optional arbitration for low-confidence labels — concurrent LLM

Supports single file or directory input. Directory mode processes files serially
(samples within each file run concurrently) with checkpoint-based resume.

Usage:
  sft-label run [--input FILE_OR_DIR] [--model MODEL] [--concurrency N]
  sft-label run --resume <run_dir>/
"""

from __future__ import annotations

import json
import time
import asyncio
import argparse
import random
import re
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace

import httpx

from sft_label.prompts import (
    CALL1_SYSTEM, CALL1_FEWSHOT, CALL2_SYSTEM, CALL2_FEWSHOT,
    CALL1_SYSTEM_COMPACT, CALL2_SYSTEM_COMPACT,
    CALL1_FEWSHOT_COMPACT, CALL2_FEWSHOT_COMPACT,
    TAG_POOLS, SINGLE_SELECT, MULTI_SELECT
)
from sft_label.fs_artifacts import is_ignored_fs_artifact
from sft_label.preprocessing import preprocess, format_signals_for_prompt, normalize_and_slice, truncate_conversations_for_labeling, apply_sparse_sampling
from sft_label.inline_pass1 import (
    append_rows_jsonl,
    apply_inherited_labels,
    build_sample_artifacts,
    build_unmapped_event_records,
    merge_pass1_results,
    prepare_inline_pass1_batch,
)
from sft_label.inline_migration import load_inline_migration_index
from sft_label.inline_labels import DEFAULT_LABEL_VERSION
from sft_label.inline_rows import (
    flatten_row_sample_bundles,
    iter_row_sample_bundle_chunks_from_jsonl,
    iter_row_sample_bundles_from_jsonl,
)
from sft_label.inline_scoring import infer_inline_scoring_target
from sft_label.progress_heartbeat import run_with_heartbeat
from sft_label.run_layout import InlineRunLayout, resolve_run_root
from sft_label.config import (
    LITELLM_BASE, LITELLM_KEY, CONFIDENCE_THRESHOLD, CONSISTENCY_RULES,
    DEFAULT_LABELING_MODEL, DEFAULT_CONCURRENCY, MAX_RETRIES, SAMPLE_MAX_RETRIES,
    REQUEST_TIMEOUT, REQUEST_TIMEOUT_ESCALATION,
    MAX_CONVERSATION_CHARS, COMPACT_CONVERSATION_CHARS,
    COMPACT_LABELING_REQUEST_BYTES,
    DIR_PIPELINE_WATERMARK, DIR_PIPELINE_MAX_FILES,
    CHUNK_SIZE, MAX_ACTIVE_CHUNKS,
    PipelineConfig,
)
from sft_label.artifacts import (
    PASS1_DASHBOARD_FILE,
    PASS1_CONVERSATION_STATS_FILE,
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    pass1_stats_filename,
    pass1_dashboard_filename,
    prune_dashboard_bundles,
)
from sft_label.labels import is_partial_labels, is_usable_labels
from sft_label.tag_canonicalization import (
    TAG_ALIASES,
    canonicalization_stat_key,
    make_canonicalization_event,
)
from sft_label.llm_runtime import (
    AdaptiveLLMRuntime,
    OutcomeClass,
    RequestOutcome,
    classify_http_result,
)
from sft_label.label_extensions import run_label_extensions
from sft_label.label_extensions_schema import load_extension_specs
from sft_label.label_extensions_stats import aggregate_extension_stats, merge_extension_stats
from sft_label.http_limits import estimate_pass1_extra_connections, resolve_httpx_connection_limits
from sft_label.progress_display import create_pipeline_progress

try:
    from sft_label.llm_runtime import AdaptiveLLMRuntime  # type: ignore
except Exception:  # pragma: no cover - optional dependency while feature is rolling out
    AdaptiveLLMRuntime = None


INLINE_RUN_MODES = {"incremental", "refresh", "migrate", "recompute"}
WORKLOAD_ESTIMATION_CPU_FRACTION = 0.8


def _resolved_extension_specs(config: PipelineConfig | None) -> list:
    if config is None:
        return []

    specs = getattr(config, "extension_specs", None)
    if specs is None:
        specs = getattr(config, "label_extension_specs", None)
    if specs is None:
        paths = list(getattr(config, "extension_spec_paths", None) or [])
        specs = load_extension_specs(paths) if paths else []

    resolved = list(specs or [])
    config.extension_specs = resolved
    setattr(config, "label_extension_specs", resolved)
    return resolved


# ─────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────

def create_progress():
    """Create a Rich progress bar display for labeling."""
    return create_pipeline_progress()


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


def _update_llm_task_progress(progress, llm_task, run_info, eta_tracker=None):
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
        progress.update(
            llm_task,
            total=max(eta_tracker.estimated_total_calls, eta_tracker.calls_done, 1),
            completed=eta_tracker.calls_done,
            info=eta_tracker.info_line(),
        )


def _build_http_client_limits(
    concurrency: int,
    *,
    chunked_output_files: int = 0,
    pprint=print,
) -> httpx.Limits:
    extra_connections = estimate_pass1_extra_connections(
        chunked_output_files=chunked_output_files,
    )
    max_connections, max_keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=concurrency,
        extra_connections=extra_connections,
    )
    if capped:
        pprint(
            "  [warn] lowering HTTP connection pool to "
            f"max_connections={max_connections}, keepalive={max_keepalive} "
            f"for requested concurrency={concurrency} to avoid FD exhaustion"
        )
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
    )


def _chunked_output_fd_budget(*, is_directory: bool, input_path: Path | None, config: PipelineConfig | None) -> int:
    if is_directory:
        max_files = config.dir_pipeline_max_files if config else DIR_PIPELINE_MAX_FILES
        return max(int(max_files or 0), 0)
    if input_path is not None and str(input_path).endswith(".jsonl"):
        return 1
    return 0


# ─────────────────────────────────────────────────────────
# Directory discovery & checkpoint
# ─────────────────────────────────────────────────────────

def discover_input_files(input_path):
    """Discover input files. Returns [(abs_path, rel_path_or_None)].

    - File → [(path, None)]  (single-file mode, no relative path)
    - Directory → [(abs, rel), ...] sorted by relative path
    """
    p = Path(input_path)
    if p.is_file():
        return [(p.resolve(), None)]

    files = sorted(
        f.resolve()
        for ext in ("*.json", "*.jsonl")
        for f in p.rglob(ext)
        if f.is_file() and not is_ignored_fs_artifact(f)
    )
    base = p.resolve()
    return [(f, f.relative_to(base)) for f in files]


def load_checkpoint(checkpoint_path):
    """Load existing checkpoint or return None."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def create_checkpoint(checkpoint_path, files, *, metadata=None):
    """Create a fresh checkpoint for a batch run."""
    ckpt = {
        "status": "in_progress",
        "completed": [],
        "failed": {},
        "total_files": len(files),
    }
    if metadata:
        ckpt["metadata"] = dict(metadata)
    _write_checkpoint(checkpoint_path, ckpt)
    return ckpt


def update_checkpoint(checkpoint_path, rel_path_str, success=True, error_msg=None):
    """Mark a file as completed or failed in the checkpoint."""
    ckpt = load_checkpoint(checkpoint_path) or {}
    if success:
        if rel_path_str not in ckpt.get("completed", []):
            ckpt.setdefault("completed", []).append(rel_path_str)
        ckpt.get("failed", {}).pop(rel_path_str, None)
    else:
        ckpt.setdefault("failed", {})[rel_path_str] = error_msg or "unknown"
    done = len(ckpt.get("completed", [])) + len(ckpt.get("failed", {}))
    if done >= ckpt.get("total_files", 0):
        ckpt["status"] = "done"
    _write_checkpoint(checkpoint_path, ckpt)
    return ckpt


def _write_checkpoint(checkpoint_path, ckpt):
    _write_json_atomic(checkpoint_path, ckpt)


def _numeric_summary(values):
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not cleaned:
        return {}
    total = sum(cleaned)
    return {
        "count": len(cleaned),
        "sum": round(total, 3),
        "mean": round(total / len(cleaned), 3),
        "min": round(min(cleaned), 3),
        "max": round(max(cleaned), 3),
    }


class OnlineNumericSummary:
    """Constant-memory numeric summary accumulator."""

    __slots__ = ("count", "total", "minimum", "maximum")

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        num = float(value)
        self.count += 1
        self.total += num
        if self.minimum is None or num < self.minimum:
            self.minimum = num
        if self.maximum is None or num > self.maximum:
            self.maximum = num

    def as_dict(self):
        if self.count <= 0:
            return {}
        return {
            "count": self.count,
            "sum": round(self.total, 3),
            "mean": round(self.total / self.count, 3),
            "min": round(float(self.minimum), 3),
            "max": round(float(self.maximum), 3),
        }


def _merge_numeric_summary(summaries):
    valid = [s for s in summaries if isinstance(s, dict) and s.get("count")]
    if not valid:
        return {}
    total_count = sum(int(s.get("count", 0) or 0) for s in valid)
    total_sum = sum(float(s.get("sum", 0.0) or 0.0) for s in valid)
    return {
        "count": total_count,
        "sum": round(total_sum, 3),
        "mean": round(total_sum / max(total_count, 1), 3),
        "min": round(min(float(s.get("min", 0.0) or 0.0) for s in valid), 3),
        "max": round(max(float(s.get("max", 0.0) or 0.0) for s in valid), 3),
    }


def _merge_compact_prompt_bytes(payloads):
    valid = [payload for payload in payloads if isinstance(payload, dict)]
    if not valid:
        return {}
    merged = {}
    for stage in ("call1", "call2"):
        stage_payloads = [payload.get(stage) for payload in valid if isinstance(payload.get(stage), dict)]
        if not stage_payloads:
            continue
        summary = _merge_numeric_summary(stage_payloads)
        if not summary:
            continue
        hard_cap_hits = sum(int(item.get("hard_cap_hits", 0) or 0) for item in stage_payloads)
        summary["limit_bytes"] = max(int(item.get("limit_bytes", 0) or 0) for item in stage_payloads)
        summary["hard_cap_hits"] = hard_cap_hits
        summary["hard_cap_hit_rate"] = round(hard_cap_hits / max(summary["count"], 1), 4)
        merged[stage] = summary
    return merged


def _merge_planner_stats(payloads):
    valid = [payload for payload in payloads if isinstance(payload, dict)]
    if not valid:
        return {}

    anchor_priority_counts = {}
    anchor_reason_counts = {}
    for item in valid:
        for key, value in (item.get("anchor_priority_counts") or {}).items():
            anchor_priority_counts[key] = anchor_priority_counts.get(key, 0) + int(value or 0)
        for key, value in (item.get("anchor_reason_counts") or {}).items():
            anchor_reason_counts[key] = anchor_reason_counts.get(key, 0) + int(value or 0)

    conversations_with_metadata = sum(int(item.get("conversations_with_metadata", 0) or 0) for item in valid)
    fallback_conversations = sum(int(item.get("fallback_conversations", 0) or 0) for item in valid)
    inherit_edges = sum(int(item.get("inherit_edges", 0) or 0) for item in valid)
    cross_segment_inherit_edges = sum(int(item.get("cross_segment_inherit_edges", 0) or 0) for item in valid)

    merged = {
        "conversations_total": sum(int(item.get("conversations_total", 0) or 0) for item in valid),
        "conversations_with_metadata": conversations_with_metadata,
        "fallback_conversations": fallback_conversations,
        "fallback_rate": round(fallback_conversations / max(conversations_with_metadata, 1), 4),
        "inherit_edges": inherit_edges,
        "cross_segment_inherit_edges": cross_segment_inherit_edges,
        "cross_segment_inherit_ratio": round(cross_segment_inherit_edges / max(inherit_edges, 1), 4),
        "anchor_priority_counts": dict(sorted(anchor_priority_counts.items(), key=lambda item: (-item[1], item[0]))),
        "anchor_reason_counts": dict(sorted(anchor_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
    }

    for key in ("segments_per_conversation", "boundary_score"):
        summary = _merge_numeric_summary([item.get(key) for item in valid])
        if summary:
            merged[key] = summary
    return merged


def merge_stats(all_file_stats):
    """Merge per-file stats into a global summary."""
    merged = {
        "total_samples": 0, "success": 0, "failed": 0,
        "distribution_total_samples": 0,
        "total_llm_calls": 0, "total_prompt_tokens": 0,
        "total_completion_tokens": 0, "total_tokens": 0,
        "arbitrated_count": 0,
        "validation_issue_count": 0, "consistency_warning_count": 0,
        "unmapped_unique_count": 0,
        "canonicalization_total_count": 0,
        "canonicalization_unique_count": 0,
        "total_elapsed_seconds": 0,
        "sparse_labeled": 0, "sparse_inherited": 0,
        "rows_total": 0, "rows_skipped": 0, "rows_changed": 0,
        "rows_migrated": 0, "rows_pass2_invalidated": 0,
        "preserved_samples": 0,
        "tag_distributions": {},
        "combo_distributions": {},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
        "canonicalization_counts": {},
        "files_processed": len(all_file_stats),
    }
    for st in all_file_stats:
        for k in ("total_samples", "success", "failed", "total_llm_calls",
                   "total_prompt_tokens", "total_completion_tokens", "total_tokens",
                   "arbitrated_count", "validation_issue_count", "consistency_warning_count",
                   "canonicalization_total_count",
                   "sparse_labeled", "sparse_inherited",
                   "distribution_total_samples",
                   "rows_total", "rows_skipped", "rows_changed",
                   "rows_migrated", "rows_pass2_invalidated",
                   "preserved_samples"):
            merged[k] += st.get(k, 0)
        merged["total_elapsed_seconds"] += st.get("total_elapsed_seconds", 0)
        # Merge tag distributions
        for dim, dist in st.get("tag_distributions", {}).items():
            if dim not in merged["tag_distributions"]:
                merged["tag_distributions"][dim] = {}
            for tag, count in dist.items():
                merged["tag_distributions"][dim][tag] = merged["tag_distributions"][dim].get(tag, 0) + count
        for combo_key, count in st.get("combo_distributions", {}).items():
            merged["combo_distributions"][combo_key] = merged["combo_distributions"].get(combo_key, 0) + count
        # Merge unmapped
        for tag, count in st.get("unmapped_tags", {}).items():
            merged["unmapped_tags"][tag] = merged["unmapped_tags"].get(tag, 0) + count
        for key, count in st.get("canonicalization_counts", {}).items():
            merged["canonicalization_counts"][key] = merged["canonicalization_counts"].get(key, 0) + count
        # Merge confidence_stats (weighted by count for proper averaging)
        for dim, cs in st.get("confidence_stats", {}).items():
            if dim not in merged["confidence_stats"]:
                merged["confidence_stats"][dim] = {"sum": 0, "count": 0, "min": cs.get("min", 1), "max": cs.get("max", 0), "below_threshold": 0}
            entry = merged["confidence_stats"][dim]
            n = cs.get("count", 0)
            if n == 0:
                continue
            entry["sum"] += cs["mean"] * n
            entry["count"] += n
            entry["min"] = min(entry["min"], cs.get("min", 1))
            entry["max"] = max(entry["max"], cs.get("max", 0))
            entry["below_threshold"] += cs.get("below_threshold", 0)
        # Merge cross_matrix (intent × difficulty)
        for key, count in st.get("cross_matrix", {}).items():
            merged["cross_matrix"][key] = merged["cross_matrix"].get(key, 0) + count

    # Sort distributions and unmapped
    for dim in merged["tag_distributions"]:
        merged["tag_distributions"][dim] = dict(sorted(merged["tag_distributions"][dim].items(), key=lambda x: -x[1]))
    merged["combo_distributions"] = dict(sorted(merged["combo_distributions"].items(), key=lambda x: -x[1]))
    merged["unmapped_tags"] = dict(sorted(merged["unmapped_tags"].items(), key=lambda x: -x[1]))
    merged["unmapped_unique_count"] = len(merged["unmapped_tags"])
    merged["canonicalization_counts"] = dict(sorted(merged["canonicalization_counts"].items(), key=lambda x: (-x[1], x[0])))
    merged["canonicalization_unique_count"] = len(merged["canonicalization_counts"])
    extension_stats = merge_extension_stats(all_file_stats)
    if extension_stats:
        merged["extension_stats"] = extension_stats
    planner_stats = _merge_planner_stats([st.get("planner_stats") for st in all_file_stats if st.get("planner_stats")])
    if planner_stats:
        merged["planner_stats"] = planner_stats
    compact_prompt_bytes = _merge_compact_prompt_bytes(
        [st.get("compact_prompt_bytes") for st in all_file_stats if st.get("compact_prompt_bytes")]
    )
    if compact_prompt_bytes:
        merged["compact_prompt_bytes"] = compact_prompt_bytes

    # Finalize confidence_stats: convert accumulated sum/count to mean
    for dim, entry in merged["confidence_stats"].items():
        n = entry.get("count", 0)
        if n > 0:
            merged["confidence_stats"][dim] = {
                "mean": round(entry["sum"] / n, 3),
                "min": round(entry["min"], 3),
                "max": round(entry["max"], 3),
                "below_threshold": entry["below_threshold"],
                "count": n,
            }

    total = merged["total_samples"]
    merged["success_rate"] = round(merged["success"] / max(total, 1), 4)
    merged["avg_calls_per_sample"] = round(merged["total_llm_calls"] / max(total, 1), 2)
    merged["arbitrated_rate"] = round(merged["arbitrated_count"] / max(total, 1), 4)

    return merged


def resolve_run_dir(output, input_path, *, base_dir=None):
    """Backward-compatible wrapper around the inline run-layout helper."""
    return resolve_run_root(input_path, output=output, base_dir=base_dir)


def _checkpoint_path_for_run(run_dir, layout: InlineRunLayout | None = None) -> Path:
    run_dir = Path(run_dir)
    if layout is not None:
        return layout.meta_root / "checkpoint.json"
    candidate = run_dir / "meta_label_data" / "checkpoint.json"
    if candidate.exists():
        return candidate
    return run_dir / "checkpoint.json"


def _summary_path_for_run(run_dir, layout: InlineRunLayout | None = None) -> Path:
    run_dir = Path(run_dir)
    if layout is not None:
        return layout.meta_root / PASS1_SUMMARY_STATS_FILE
    candidate = run_dir / "meta_label_data" / PASS1_SUMMARY_STATS_FILE
    if candidate.exists():
        return candidate
    return run_dir / PASS1_SUMMARY_STATS_FILE


def _normalize_run_input(input_path) -> tuple[Path, object | None]:
    """Normalize inline run roots to their mirrored dataset path."""
    if input_path is None:
        return None, None
    resolved = Path(input_path).resolve()
    inline_target = infer_inline_scoring_target(resolved)
    if inline_target is None:
        return resolved, None
    if inline_target.target_path == inline_target.layout.run_root:
        return inline_target.layout.dataset_root, inline_target
    return inline_target.target_path, inline_target


def _resolve_mode(mode: str) -> str:
    resolved = str(mode or "refresh").strip().lower()
    if resolved not in INLINE_RUN_MODES:
        raise ValueError(f"unsupported run mode: {mode}")
    return resolved


def _write_json_atomic(path, payload):
    """Write JSON via a temp file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _write_jsonl_atomic(path, records):
    """Write JSONL via a temp file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


PASS1_CHUNKED_WORKING_SUFFIX = ".next"


def _pass1_chunked_working_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}{PASS1_CHUNKED_WORKING_SUFFIX}")


def _finalize_pass1_chunked_working_files(path_pairs):
    for final_path, working_path in path_pairs:
        final_path = Path(final_path)
        working_path = Path(working_path)
        if not working_path.exists():
            continue
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(working_path, final_path)


# ─────────────────────────────────────────────────────────
# Rate limiter + HTTP request stats
# ─────────────────────────────────────────────────────────

class RequestStats:
    """Track HTTP request outcomes for real-time display and summary."""
    __slots__ = ('success', 'errors', 'timeouts')

    def __init__(self):
        self.success = 0
        self.errors = {}   # status_code (int) -> count
        self.timeouts = 0

    def record(self, status_code: int):
        if 200 <= status_code < 300:
            self.success += 1
        else:
            self.errors[status_code] = self.errors.get(status_code, 0) + 1

    def record_timeout(self):
        self.timeouts += 1

    @property
    def total(self):
        return self.success + sum(self.errors.values()) + self.timeouts

    def summary_line(self):
        """One-line summary for progress display."""
        t = self.total
        if t == 0:
            return ""
        parts = [f"✓{self.success}"]
        for code in sorted(self.errors):
            parts.append(f"{code}×{self.errors[code]}")
        if self.timeouts:
            parts.append(f"timeout×{self.timeouts}")
        rate = self.success / t * 100
        return f"http({' '.join(parts)} {rate:.0f}%)"

    def to_dict(self):
        t = self.total
        return {
            "total_http_requests": t,
            "success": self.success,
            "errors": {str(k): v for k, v in sorted(self.errors.items())},
            "timeouts": self.timeouts,
            "success_rate": round(self.success / t * 100, 1) if t > 0 else 0,
        }


class AsyncRateLimiter:
    """Token bucket rate limiter for async contexts with warmup support.

    During warmup, effective RPS ramps linearly from 1 to the target RPS.
    Initial tokens = 1 (cold start) to avoid burst on startup.
    """

    def __init__(self, rps: float, burst: int | None = None, warmup: float = 0.0):
        self._rps = rps
        self._burst = burst if burst is not None else max(int(rps), 1)
        self._tokens = 1.0  # Cold start: only 1 token initially (no burst)
        self._last_refill = 0.0
        self._start_time = 0.0
        self._warmup = warmup
        self._lock = asyncio.Lock()
        self.stats = RequestStats()

    def _effective_rps(self, now):
        """Current RPS considering warmup ramp."""
        if self._warmup <= 0 or self._start_time == 0.0:
            return self._rps
        elapsed = now - self._start_time
        if elapsed >= self._warmup:
            return self._rps
        # Linear ramp: 1 → rps over warmup seconds
        progress = elapsed / self._warmup
        return 1.0 + (self._rps - 1.0) * progress

    async def acquire(self):
        while True:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                if self._last_refill == 0.0:
                    self._last_refill = now
                    self._start_time = now
                elapsed = now - self._last_refill
                effective_rps = self._effective_rps(now)
                self._tokens = min(self._burst, self._tokens + elapsed * effective_rps)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / effective_rps
            await asyncio.sleep(wait)  # Sleep OUTSIDE lock


# ─────────────────────────────────────────────────────────
# Async LLM calls
# ─────────────────────────────────────────────────────────

async def async_llm_call(http_client, messages, model, temperature=0.1, max_tokens=1000, max_retries=MAX_RETRIES,
                         config=None, rate_limiter=None):
    """Async LLM call with retry + jitter. Returns (parsed_json, raw_content, usage)."""
    _base = config.litellm_base if config else LITELLM_BASE
    _key = config.litellm_key if config else LITELLM_KEY
    _timeout = config.request_timeout if config else REQUEST_TIMEOUT
    _escalation = (config.request_timeout_escalation if config and config.request_timeout_escalation
                   else REQUEST_TIMEOUT_ESCALATION)
    url = f"{_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_key}",
        "Content-Type": "application/json",
        "User-Agent": "sft-label/0.1.0",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Reasoning models (o1, o3, gpt-5*) don't support temperature or max_tokens
    _model_lower = model.lower()
    if any(p in _model_lower for p in ("o1", "o3", "gpt-5")):
        payload.pop("temperature", None)
        payload["max_completion_tokens"] = payload.pop("max_tokens")
    last_error = None
    last_error_response = None

    runtime = getattr(config, "_adaptive_runtime", None) if config is not None else None
    adaptive_mode = runtime is not None

    for attempt in range(max_retries + 1):
        _http_recorded = False
        # Adaptive timeout: escalate on retries
        attempt_timeout = (_escalation[attempt] if _escalation and attempt < len(_escalation)
                           else _timeout)
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()
            resp = await http_client.post(url, json=payload, headers=headers, timeout=attempt_timeout)
            if rate_limiter is not None:
                rate_limiter.stats.record(resp.status_code)
                _http_recorded = True
            if resp.status_code == 403:
                # Content filtered by upstream WAF/provider — retry once to rule out transient proxy issues
                resp_body = resp.text
                # Capture diagnostic headers to identify which layer returned the 403:
                #   x-litellm-* → LiteLLM or upstream provider
                #   via/x-squid/x-cache → corporate proxy
                #   server → nginx/uvicorn/squid/etc
                diag_keys = ("server", "via", "x-squid-error", "x-cache",
                             "content-type", "x-litellm-version",
                             "x-litellm-model-group", "x-litellm-call-id")
                diag_headers = {k: resp.headers[k] for k in diag_keys if k in resp.headers}
                is_html = "text/html" in resp.headers.get("content-type", "")
                # Build diagnostic hint that flows with the error string everywhere
                diag_src = ("litellm/upstream" if "x-litellm-version" in diag_headers
                            else "proxy/WAF" if any(k in diag_headers for k in ("via", "x-squid-error", "x-cache"))
                            else f"server={diag_headers.get('server', '?')}")
                diag_hint = f" [source={diag_src}]"
                if is_html:
                    diag_hint += " [HTML response — likely WAF/proxy block page]"
                last_error = f"HTTP 403: {resp_body[:300]}{diag_hint}"
                if attempt < 1:
                    await asyncio.sleep(3 + random.uniform(0, 2))
                    continue
                return None, last_error, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": last_error,
                    "error_response": resp_body,
                    "error_diag_headers": diag_headers,
                    "error_is_html": is_html,
                    "non_retryable": True,
                }
            if resp.status_code in (429, 500, 502, 503, 504):
                resp_text = resp.text[:500]
                resp_lower = resp_text.lower()
                # LiteLLM "No deployments available" — all deployments are in cooldown,
                # often caused by upstream content-filter 400s.  Retrying immediately
                # only prolongs the cooldown cycle. In adaptive mode, we want the
                # outer controller to open the circuit and pause; in legacy mode,
                # treat as non-retryable to avoid local retry storms.
                if "no deployments available" in resp_lower or "cooldown_list" in resp_lower:
                    return None, f"HTTP {resp.status_code}: {resp_text[:300]}", {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP {resp.status_code} (deployment cooldown): {resp_text[:300]}",
                        "error_response": resp.text,
                        "provider_cooldown": True,
                        "non_retryable": False if adaptive_mode else True,
                    }
                # Rate limited or server error — exponential backoff with jitter
                if adaptive_mode:
                    return None, f"HTTP {resp.status_code}: {resp_text[:300]}", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP {resp.status_code}: {resp_text[:300]}",
                        "error_response": resp.text,
                        "non_retryable": False,
                    }
                base_wait = min(2 ** attempt * 3 + 2, 60)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                last_error = f"HTTP {resp.status_code}: {resp_text[:200]}"
                last_error_response = resp.text
                if attempt < max_retries:
                    await asyncio.sleep(wait)
                    continue
            if resp.status_code == 400:
                error_text = resp.text[:500]
                error_lower = error_text.lower()
                # Genuinely non-retryable: the request content itself is invalid.
                # Keep this list narrow — in multi-hop proxy chains, broad keywords
                # like "invalid_request" or "model_not_found" can match transient
                # supplier routing errors (e.g. "Unknown model: gpt4ominicep").
                #
                # Content moderation keywords (Azure + OpenAI + LiteLLM wrappers):
                #   content_filter / content_policy — Azure RAI filter
                #   responsibleaipolicyviolation — Azure innererror code
                #   content_management_policy — Azure alternate wording
                #   moderation — OpenAI moderation endpoint flag
                _NON_RETRYABLE_400_KEYWORDS = (
                    "context_length_exceeded", "maximum context length",
                    "content_policy", "content_filter",
                    "responsibleaipolicyviolation", "content_management_policy",
                    "moderation",
                )
                if any(kw in error_lower for kw in _NON_RETRYABLE_400_KEYWORDS):
                    return None, f"HTTP 400: {error_text[:300]}", {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP 400 (content filtered): {error_text[:300]}",
                        "error_response": resp.text,
                        "content_filtered": True,
                        "non_retryable": True,
                    }
                # Likely a transient proxy/supplier error — retry with backoff
                if adaptive_mode:
                    return None, f"HTTP 400: {error_text[:300]}", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP 400 (transient): {error_text[:300]}",
                        "error_response": resp.text,
                        "non_retryable": False,
                    }
                last_error = f"HTTP 400 (transient): {error_text[:200]}"
                last_error_response = resp.text
                if attempt < max_retries:
                    base_wait = min(2 ** attempt * 3 + 2, 60)
                    await asyncio.sleep(base_wait + random.uniform(0, base_wait * 0.5))
                    continue
            if resp.status_code == 401:
                # Auth failure — not retryable
                error_text = resp.text[:300]
                return None, f"HTTP 401: {error_text}", {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": f"HTTP 401: {error_text}",
                    "error_response": resp.text,
                    "non_retryable": True,
                }
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            usage_dict = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "status_code": resp.status_code,
            }

            # Parse JSON
            json_str = content
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.startswith("```") and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                json_str = "\n".join(json_lines)

            parsed = json.loads(json_str)
            return parsed, content, usage_dict

        except (json.JSONDecodeError, KeyError) as e:
            last_error = f"ParseError: {e}"
            if adaptive_mode:
                return None, content if 'content' in locals() else "", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "status_code": 200,
                    "error": last_error,
                    "parse_error": True,
                    "non_retryable": False,
                }
            if attempt < max_retries:
                await asyncio.sleep(2 + random.uniform(0, 2))
                continue
            return None, content if 'content' in locals() else "", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "status_code": 200,
                "error": last_error,
                "parse_error": True,
            }
        except Exception as e:
            if rate_limiter is not None and not _http_recorded:
                rate_limiter.stats.record_timeout()
            last_error = f"{type(e).__name__}: {e}"
            if adaptive_mode:
                return None, str(e), {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": last_error,
                    "exception_type": type(e).__name__,
                    "non_retryable": False,
                }
            if attempt < max_retries:
                base_wait = min(2 ** attempt * 3 + 2, 60)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                await asyncio.sleep(wait)
                continue
            return None, str(e), {"prompt_tokens": 0, "completion_tokens": 0, "error": last_error, "exception_type": type(e).__name__}

    return None, f"max retries exceeded: {last_error}", {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": last_error or "max_retries",
        "error_response": last_error_response,
    }

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
        min_observations_open=int(getattr(config, "adaptive_min_observations_open", 5) or 5),
        min_failures_degraded=int(getattr(config, "adaptive_min_failures_degraded", 2) or 2),
        min_failures_open=int(getattr(config, "adaptive_min_failures_open", 2) or 2),
        open_base_cooldown=float(getattr(config, "adaptive_open_base_cooldown", 15.0) or 15.0),
        open_max_cooldown=float(getattr(config, "adaptive_open_max_cooldown", 120.0) or 120.0),
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


# ─────────────────────────────────────────────────────────
# Prompt builders (inline to avoid cross-import issues with async)
# ─────────────────────────────────────────────────────────

def build_call1_messages(conversation_json, preprocessed_signals, compact=False):
    user_content = f"""<conversation>
{conversation_json}
</conversation>

<preprocessed_signals>
{preprocessed_signals}
</preprocessed_signals>"""
    fewshot = CALL1_FEWSHOT_COMPACT if compact else CALL1_FEWSHOT
    system = CALL1_SYSTEM_COMPACT if compact else CALL1_SYSTEM
    messages = [{"role": "system", "content": system}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})
    return messages


def build_call2_messages(conversation_json, preprocessed_signals, call1_result, compact=False):
    call1_str = json.dumps(call1_result, ensure_ascii=False) if isinstance(call1_result, dict) else str(call1_result)
    user_content = f"""<conversation>
{conversation_json}
</conversation>

<call1_result>
{call1_str}
</call1_result>

<preprocessed_signals>
{preprocessed_signals}
</preprocessed_signals>"""
    fewshot = CALL2_FEWSHOT_COMPACT if compact else CALL2_FEWSHOT
    system = CALL2_SYSTEM_COMPACT if compact else CALL2_SYSTEM
    messages = [{"role": "system", "content": system}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})
    return messages


# ─────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────

TAG_DIMENSION_INDEX = {}
for _dim, _pool in TAG_POOLS.items():
    for _tag in _pool:
        TAG_DIMENSION_INDEX.setdefault(_tag, set()).add(_dim)

# Tags that belong to a DIFFERENT category — drop from the current dimension
# and log a warning, but do NOT add to unmapped (they're valid, just misplaced).
CROSS_CATEGORY_CORRECTIONS = {
    "domain": {
        # These are concepts or tasks, not domains
        "algorithm", "algorithms", "data-structure", "data-structures",
        "documentation", "debugging",
    },
}


def _append_canonicalization(canonicalized, source_dim, source_value, target_dim, canonical_value, reason):
    """Record a canonicalization event, deduplicated within one sample."""
    if canonicalized is None:
        return
    source_value = "" if source_value is None else str(source_value).strip()
    canonical_value = "" if canonical_value is None else str(canonical_value).strip()
    if not source_value or not canonical_value:
        return
    event = make_canonicalization_event(
        source_dimension=source_dim or "?",
        source_value=source_value,
        target_dimension=target_dim or source_dim or "?",
        canonical_value=canonical_value,
        reason=reason,
    )
    if event not in canonicalized:
        canonicalized.append(event)


def _resolve_aliases(dim, values, canonicalized=None):
    """Resolve aliases and deduplicate, preserving order.

    Only applies alias mapping when the original value is NOT already
    a valid tag in the target dimension's pool (e.g., concept:security
    should stay as-is, not become "cybersecurity").
    """
    pool = TAG_POOLS.get(dim, set())
    seen = set()
    resolved = []
    for v in values:
        # Only map alias if the raw value is not already valid in this pool
        canonical = v if v in pool else TAG_ALIASES.get(v, v)
        if canonical != v and canonical in pool:
            _append_canonicalization(canonicalized, dim, v, dim, canonical, "alias")
            canonical_for_dim = canonical
        else:
            canonical_for_dim = v
        # Cross-category correction: silently drop misplaced tags
        if dim in CROSS_CATEGORY_CORRECTIONS and canonical_for_dim in CROSS_CATEGORY_CORRECTIONS[dim]:
            continue
        if canonical_for_dim not in seen:
            seen.add(canonical_for_dim)
            resolved.append(canonical_for_dim)
    return resolved


def _sanitize_confidence(confidence, dims, issues):
    """Keep only numeric per-dimension confidence scores in [0, 1]."""
    if confidence is None:
        return {}
    if not isinstance(confidence, dict):
        issues.append("confidence: expected object")
        return {}

    cleaned = {}
    for dim in dims:
        score = confidence.get(dim)
        if score is None:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            issues.append(f"confidence.{dim}: invalid type")
            continue
        score = float(score)
        if 0.0 <= score <= 1.0:
            cleaned[dim] = round(score, 4)
        else:
            issues.append(f"confidence.{dim}: out of range")
    return cleaned


CONTENT_FILTER_REPLACEMENTS = (
    (r"\bplease\s+reason\s+step\s+by\s+step\b", "briefly explain your approach"),
    (r"\breason\s+step\s+by\s+step\b", "briefly explain the approach"),
    (r"\bthink\s+step\s+by\s+step\b", "explain briefly"),
    (r"\bstep\s+by\s+step\b", "succinctly"),
    (r"\berroneous code\b", "reference code"),
    (r"\bbuggy code\b", "reference code"),
    (r"\bincrease misdirection\b", "add context"),
    (r"\bmisdirection\b", "extra context"),
)


def sanitize_conversations_for_content_filter(conversations):
    """Apply tiny wording softeners before one content-filter retry."""
    sanitized = []
    changed = False
    for turn in conversations:
        cloned = dict(turn)
        for key in ("value", "content"):
            text = cloned.get(key)
            if not isinstance(text, str) or not text:
                continue
            updated = text
            for pattern, replacement in CONTENT_FILTER_REPLACEMENTS:
                updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
            if updated != text:
                cloned[key] = updated
                changed = True
        sanitized.append(cloned)
    return sanitized, changed


def _candidate_tag_values(value):
    """Return raw + alias-resolved candidates for rescue checks."""
    raw = "" if value is None else str(value).strip()
    if not raw or _is_empty_sentinel(raw):
        return []
    candidates = [raw]
    alias = TAG_ALIASES.get(raw)
    if alias and alias not in candidates:
        candidates.append(alias)
    return candidates


def _is_empty_sentinel(value):
    """Return True for placeholder outputs that should mean empty."""
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {
        "",
        "none",
        "unspecified",
        "unknown",
        "n/a",
        "not applicable",
        "multiple languages detected",
        "user query not present",
        "user context not provided",
    }:
        return True
    return normalized.startswith((
        "no specific ",
        "no programming language ",
        "no code ",
        "no domain ",
        "no task ",
        "no difficulty ",
        "no context ",
    ))


def _normalize_optional_single_value(value):
    """Normalize explicit empty sentinels to the empty-string shape."""
    return "" if _is_empty_sentinel(value) else value


def _normalize_optional_multi_values(values):
    """Drop explicit empty sentinels from multi-select outputs."""
    return [value for value in values if not _is_empty_sentinel(value)]


def _find_cross_dim_tag(value, exclude_dim=None):
    """Find a uniquely matching tag in another dimension."""
    for candidate in _candidate_tag_values(value):
        matched_dims = sorted(
            dim for dim in TAG_DIMENSION_INDEX.get(candidate, set())
            if dim != exclude_dim
        )
        if len(matched_dims) == 1:
            return matched_dims[0], candidate
    return None, None


def _queue_rescued_tag(rescued, dim, value):
    """Stage a rescued tag for merge into cleaned output."""
    if dim in SINGLE_SELECT:
        rescued.setdefault(dim, value)
        return
    bucket = rescued.setdefault(dim, [])
    if value not in bucket:
        bucket.append(value)


def _try_rescue_unmapped_value(value, source_dim, rescued, current_dims, canonicalized=None, reason="cross_dimension_rescue"):
    """Rescue values that already exist elsewhere in the taxonomy."""
    target_dim, rescued_value = _find_cross_dim_tag(value, exclude_dim=source_dim)
    if not target_dim:
        return False
    _queue_rescued_tag(rescued, target_dim, rescued_value)
    _append_canonicalization(
        canonicalized,
        source_dim or "?",
        value,
        target_dim,
        rescued_value,
        reason,
    )
    return True


def _normalize_unmapped_items(items):
    """Normalize LLM unmapped output to {dimension, value} objects."""
    normalized = []
    for item in items:
        if isinstance(item, dict):
            dim = str(item.get("dimension", "?") or "?")
            value = str(item.get("value", "") or "").strip()
        else:
            dim = "?"
            value = str(item or "").strip()
        if value and not _is_empty_sentinel(value):
            normalized.append({"dimension": dim, "value": value})
    return normalized


def _build_partial_labels(call1_cleaned, reason):
    """Wrap Call 1 output so downstream stages can block partial labels."""
    labels = {
        d: call1_cleaned.get(d, [] if d in MULTI_SELECT else "")
        for d in ["intent", "language", "domain", "task", "difficulty"]
    }
    labels["confidence"] = dict(call1_cleaned.get("confidence", {}))
    labels["unmapped"] = list(call1_cleaned.get("unmapped", []))
    labels["canonicalized"] = list(call1_cleaned.get("canonicalized", []))
    labels["partial"] = True
    labels["partial_stage"] = "call1"
    labels["partial_reason"] = reason
    return labels


def validate_tags(result, call_name="call1"):
    issues = []
    raw_unmapped = result.get("unmapped", [])
    if not isinstance(raw_unmapped, list):
        raw_unmapped = []
    cleaned = dict(result)
    rescued = {}
    pending_unmapped = []
    canonicalized = []

    dims = (["intent", "language", "domain", "task", "difficulty"] if call_name == "call1"
            else ["concept", "agentic", "constraint", "context"])

    for dim in dims:
        if dim not in result:
            issues.append(f"Missing: {dim}")
            cleaned[dim] = "" if dim in SINGLE_SELECT else []
            continue

        pool = TAG_POOLS.get(dim, set())
        if dim in SINGLE_SELECT:
            raw_val = _normalize_optional_single_value(result[dim])
            if raw_val is None:
                raw_val = ""
            elif isinstance(raw_val, (list, dict, set, tuple)):
                issues.append(f"{dim}: invalid type {type(raw_val).__name__}")
                raw_val = ""
            elif not isinstance(raw_val, str):
                issues.append(f"{dim}: invalid type {type(raw_val).__name__}")
                raw_val = str(raw_val)
            val = raw_val if (not raw_val or raw_val in pool) else TAG_ALIASES.get(raw_val, raw_val)
            if val and val not in pool:
                if not _try_rescue_unmapped_value(val, dim, rescued, dims, canonicalized=canonicalized):
                    issues.append(f"{dim}: '{val}' not in pool")
                    pending_unmapped.append({"dimension": dim, "value": val})
                cleaned[dim] = ""
            else:
                cleaned[dim] = val
        else:
            raw = result[dim] if isinstance(result[dim], list) else [result[dim]]
            raw = _normalize_optional_multi_values(raw)
            resolved = _resolve_aliases(dim, raw, canonicalized=canonicalized)
            valid = []
            for v in resolved:
                if v in pool:
                    valid.append(v)
                else:
                    if not _try_rescue_unmapped_value(v, dim, rescued, dims, canonicalized=canonicalized):
                        issues.append(f"{dim}: '{v}' not in pool")
                        pending_unmapped.append({"dimension": dim, "value": v})
            cleaned[dim] = valid

    for item in _normalize_unmapped_items(raw_unmapped):
        dim = item["dimension"]
        value = item["value"]
        if dim in dims:
            same_dim_hit = next(
                (candidate for candidate in _candidate_tag_values(value) if candidate in TAG_POOLS.get(dim, set())),
                None,
            )
            if same_dim_hit:
                _queue_rescued_tag(rescued, dim, same_dim_hit)
                if same_dim_hit != value:
                    _append_canonicalization(
                        canonicalized,
                        dim,
                        value,
                        dim,
                        same_dim_hit,
                        "raw_unmapped_alias",
                    )
                continue
        if _try_rescue_unmapped_value(
            value,
            dim if dim in TAG_POOLS else None,
            rescued,
            dims,
            canonicalized=canonicalized,
            reason="raw_unmapped_cross_dimension_rescue",
        ):
            continue
        pending_unmapped.append(item)

    for dim, rescued_value in rescued.items():
        if dim in SINGLE_SELECT:
            current = cleaned.get(dim)
            if not current:
                cleaned[dim] = rescued_value
            continue
        merged = list(cleaned.get(dim, []))
        for value in rescued_value:
            if value not in merged:
                merged.append(value)
        cleaned[dim] = merged

    # Filter out LLM explanation sentences — unmapped should be short tag IDs
    unmapped = [
        item for item in pending_unmapped
        if len(item.get("value", "")) <= 60 and item.get("value", "").count(" ") <= 3
    ]

    cleaned["confidence"] = _sanitize_confidence(result.get("confidence"), dims, issues)
    cleaned["unmapped"] = unmapped
    cleaned["canonicalized"] = canonicalized
    return cleaned, issues


def check_consistency(labels):
    warnings = []
    ns = {
        "intent": labels.get("intent", ""),
        "difficulty": labels.get("difficulty", ""),
        "context": labels.get("context", ""),
        "language": labels.get("language", []),
        "domain": labels.get("domain", []),
        "concept": labels.get("concept", []),
        "task": labels.get("task", []),
        "agentic": labels.get("agentic", []),
        "constraint": labels.get("constraint", []),
        "len": len,
    }
    for condition, message in CONSISTENCY_RULES:
        try:
            if eval(condition, {"__builtins__": {}}, ns):
                warnings.append(message)
        except Exception:
            pass
    return warnings


def find_low_confidence_dims(labels, threshold=CONFIDENCE_THRESHOLD):
    low = []
    confidence = labels.get("confidence", {})
    if not isinstance(confidence, dict):
        return low
    for dim, score in confidence.items():
        if isinstance(score, (int, float)) and score < threshold:
            low.append((dim, score))
    return low


def _resolved_labeling_prompt_budget(config=None):
    """Return effective prompt mode metadata for Pass 1 observability."""
    compact = config.prompt_mode == "compact" if config else False
    budget = config.max_conversation_chars if config else MAX_CONVERSATION_CHARS
    if compact and config and config.max_conversation_chars == MAX_CONVERSATION_CHARS:
        budget = COMPACT_CONVERSATION_CHARS
    return compact, budget


def _message_payload_bytes(messages):
    """Return UTF-8 payload size for the full request body."""
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))


def _labeling_truncation_kwargs(config=None, *, max_total_chars):
    kwargs = {"max_total_chars": max_total_chars}
    if config:
        kwargs.update(
            head_ratio=config.truncation_head_ratio,
            last_response_ratio=config.truncation_last_response_ratio,
            per_turn_ratio=config.truncation_per_turn_ratio,
        )
    return kwargs


def _build_labeling_messages_with_budget(
    conversations,
    *,
    signals_str,
    compact,
    max_chars,
    config,
    call1_context=None,
):
    """Build Pass 1 messages and shrink conversation chars until byte-safe."""
    compact_byte_limit = None
    if compact:
        compact_byte_limit = int(getattr(config, "compact_labeling_request_bytes", COMPACT_LABELING_REQUEST_BYTES) or COMPACT_LABELING_REQUEST_BYTES)

    current_chars = int(max_chars)
    last_payload_bytes = None
    while True:
        truncated_convs, was_truncated = truncate_conversations_for_labeling(
            conversations,
            **_labeling_truncation_kwargs(config, max_total_chars=current_chars),
        )
        conversation_json = json.dumps(truncated_convs, ensure_ascii=False)
        if call1_context is None:
            messages = build_call1_messages(conversation_json, signals_str, compact=compact)
        else:
            messages = build_call2_messages(conversation_json, signals_str, call1_context, compact=compact)
        last_payload_bytes = _message_payload_bytes(messages)
        if compact_byte_limit is None or last_payload_bytes <= compact_byte_limit or current_chars <= 2000:
            return truncated_convs, conversation_json, messages, was_truncated, last_payload_bytes
        current_chars = max(2000, int(current_chars * 0.85))


def _annotate_labeling_prompt_stats(stats, config=None):
    compact, budget = _resolved_labeling_prompt_budget(config)
    stats["prompt_mode"] = "compact" if compact else "full"
    stats["compact_prompt"] = compact
    stats["conversation_char_budget"] = budget
    stats["fewshot_variant"] = "compact" if compact else "full"
    return stats


# ─────────────────────────────────────────────────────────
# Per-sample pipeline (async)
# ─────────────────────────────────────────────────────────

async def label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                    config=None, rate_limiter=None, llm_progress_cb=None, progress_event_cb=None):
    """Label a single sample with sample-level retry on failure."""
    _max_retries_sample = config.sample_max_retries if config else SAMPLE_MAX_RETRIES
    _max_retries = config.max_retries if config else MAX_RETRIES
    _conf_threshold = config.confidence_threshold if config else CONFIDENCE_THRESHOLD
    _compact, _max_chars = _resolved_labeling_prompt_budget(config)
    runtime = _config_adaptive_runtime(config)
    if getattr(config, "_force_disable_arbitration", False):
        enable_arbitration = False
    start = time.time()

    conversations = sample.get("conversations", [])
    truncated_convs, effective_conversations_json, _initial_msgs, was_truncated, assembled_prompt_bytes = _build_labeling_messages_with_budget(
        conversations,
        signals_str="",
        compact=_compact,
        max_chars=_max_chars,
        config=config,
    )
    effective_conversations = truncated_convs
    sanitized_convs = None

    _cached_call1 = None  # Cache successful Call 1 across sample retries
    last_run_info = None

    def _record_llm_progress(delta_calls: int):
        nonlocal last_run_info
        delta = max(int(delta_calls or 0), 0)
        if delta <= 0:
            return last_run_info
        if llm_progress_cb:
            last_run_info = llm_progress_cb(delta, "pass1")
        if progress_event_cb:
            progress_event_cb({
                "kind": "llm_delta",
                "delta_calls": delta,
                "run_info": last_run_info,
            })
        return last_run_info

    async def _call(stage: str, messages, *, temperature=0.1, max_tokens=1000):
        if runtime is None:
            async with sem:
                parsed, raw, usage = await async_llm_call(
                    http_client,
                    messages,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=_max_retries,
                    config=config,
                    rate_limiter=rate_limiter,
                )
            return parsed, raw, usage, _outcome_from_usage(usage, default_status=usage.get("status_code"))

        quick_retries = int(getattr(config, "request_quick_retries", 0) or 0)
        last = None
        for quick_attempt in range(quick_retries + 1):
            permit = await runtime.acquire(stage=stage, sample_id=sample.get("id", f"sample-{sample_idx}"))
            try:
                parsed, raw, usage = await async_llm_call(
                    http_client,
                    messages,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=0,
                    config=config,
                    rate_limiter=None,
                )
            finally:
                permit.release()
            outcome = _outcome_from_usage(usage, default_status=usage.get("status_code"))
            runtime.observe(outcome)
            usage = dict(usage)
            usage["queue_wait_ms"] = permit.queue_wait_ms
            usage["runtime_state_at_acquire"] = permit.state_at_acquire
            usage["runtime_state"] = runtime.state
            last = (parsed, raw, usage, outcome)
            if parsed is not None:
                break
            if outcome.classification is OutcomeClass.TRANSIENT_ERROR and quick_attempt < quick_retries:
                await asyncio.sleep(0)
                continue
            break
        return last

    for sample_attempt in range(_max_retries_sample + 1):
        if sample_attempt > 0:
            if runtime is None:
                base_wait = 2 ** sample_attempt * 2
                await asyncio.sleep(base_wait + random.uniform(0, base_wait))
            else:
                await asyncio.sleep(0)

        monitor = {
            "sample_id": sample.get("id", f"sample-{sample_idx}"),
            "index": sample_idx,
            "llm_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "validation_issues": [],
            "consistency_warnings": [],
            "low_confidence_dims": [],
            "arbitrated": False,
            "sample_attempt": sample_attempt,
            "status": "success",
        }

        try:
            # Preprocess (uses original conversations for signal extraction)
            signals = preprocess(sample)
            signals_str = format_signals_for_prompt(signals)
            # Use truncated conversations for LLM prompt
            conversations_json = effective_conversations_json
            monitor["assembled_prompt_bytes"] = assembled_prompt_bytes
            if was_truncated:
                monitor["truncated"] = True

            # Call 1 — reuse cached result if available from a previous attempt
            if _cached_call1 is not None:
                call1_cleaned, msgs1 = _cached_call1
                monitor["call1_cached"] = True
            else:
                truncated_convs, conversations_json, msgs1, was_truncated, assembled_prompt_bytes = _build_labeling_messages_with_budget(
                    conversations,
                    signals_str=signals_str,
                    compact=_compact,
                    max_chars=_max_chars,
                    config=config,
                )
                effective_conversations = truncated_convs
                effective_conversations_json = conversations_json
                monitor["assembled_prompt_bytes"] = assembled_prompt_bytes
                if _compact and assembled_prompt_bytes > int(getattr(config, "compact_labeling_request_bytes", COMPACT_LABELING_REQUEST_BYTES) or COMPACT_LABELING_REQUEST_BYTES):
                    monitor["status"] = "prompt_budget_exceeded"
                    monitor["error"] = f"assembled labeling prompt exceeds {config.compact_labeling_request_bytes} bytes"
                    monitor["elapsed_seconds"] = round(time.time() - start, 2)
                    return sample_idx, None, monitor
                call1_result, call1_raw, usage1, outcome1 = await _call("pass1.call1", msgs1)
                monitor["llm_calls"] += 1
                _record_llm_progress(1)
                monitor["total_prompt_tokens"] += usage1.get("prompt_tokens", 0)
                monitor["total_completion_tokens"] += usage1.get("completion_tokens", 0)
                if "queue_wait_ms" in usage1:
                    monitor["queue_wait_ms"] = usage1.get("queue_wait_ms")
                if usage1.get("runtime_state") is not None:
                    monitor["runtime_state"] = usage1.get("runtime_state")

                if call1_result is None and usage1.get("content_filtered"):
                    if sanitized_convs is None:
                        sanitized_convs, changed = sanitize_conversations_for_content_filter(truncated_convs)
                        if changed:
                            pass
                    if sanitized_convs:
                        _sanitized_convs, sanitized_conversations_json, retry_msgs1, _retry_truncated, retry_prompt_bytes = _build_labeling_messages_with_budget(
                            sanitized_convs,
                            signals_str=signals_str,
                            compact=_compact,
                            max_chars=_max_chars,
                            config=config,
                        )
                    else:
                        sanitized_conversations_json = None
                    if sanitized_conversations_json and sanitized_conversations_json != conversations_json:
                        retry_result, retry_raw, retry_usage, retry_outcome1 = await _call(
                            "pass1.call1.sanitized",
                            retry_msgs1,
                        )
                        monitor["llm_calls"] += 1
                        _record_llm_progress(1)
                        monitor["total_prompt_tokens"] += retry_usage.get("prompt_tokens", 0)
                        monitor["total_completion_tokens"] += retry_usage.get("completion_tokens", 0)
                        monitor["assembled_prompt_bytes"] = retry_prompt_bytes
                        monitor["content_filter_retry"] = "call1"
                        if retry_result is not None:
                            call1_result, call1_raw, usage1, outcome1 = retry_result, retry_raw, retry_usage, retry_outcome1
                            msgs1 = retry_msgs1
                            conversations_json = sanitized_conversations_json
                            effective_conversations = sanitized_convs
                            effective_conversations_json = sanitized_conversations_json
                        else:
                            call1_result, call1_raw, usage1, outcome1 = retry_result, retry_raw, retry_usage, retry_outcome1

                if call1_result is None:
                    monitor["status"] = "call1_failed"
                    monitor["error"] = usage1.get("error", "unknown")
                    monitor["error_response"] = usage1.get("error_response") or (call1_raw[:500] if call1_raw else "")
                    monitor["error_class"] = outcome1.classification.value
                    monitor["retryable_infra"] = bool(outcome1.is_infra_failure)
                    monitor["http_status"] = outcome1.status_code
                    if usage1.get("non_retryable"):
                        monitor["elapsed_seconds"] = round(time.time() - start, 2)
                        return sample_idx, None, monitor
                    if sample_attempt < _max_retries_sample:
                        continue
                    monitor["elapsed_seconds"] = round(time.time() - start, 2)
                    return sample_idx, None, monitor

                call1_cleaned, call1_issues = validate_tags(call1_result, "call1")
                monitor["validation_issues"].extend(call1_issues)
                _cached_call1 = (call1_cleaned, msgs1)

            # Call 2 (depends on Call 1)
            call1_context = {d: call1_cleaned[d] for d in ["intent", "language", "domain", "task", "difficulty"] if d in call1_cleaned}
            call2_source_conversations = effective_conversations
            _call2_convs, conversations_json, msgs2, _call2_truncated, assembled_prompt_bytes = _build_labeling_messages_with_budget(
                call2_source_conversations,
                signals_str=signals_str,
                compact=_compact,
                max_chars=_max_chars,
                config=config,
                call1_context=call1_context,
            )
            monitor["assembled_prompt_bytes_call2"] = assembled_prompt_bytes
            if _compact and assembled_prompt_bytes > int(getattr(config, "compact_labeling_request_bytes", COMPACT_LABELING_REQUEST_BYTES) or COMPACT_LABELING_REQUEST_BYTES):
                monitor["status"] = "prompt_budget_exceeded"
                monitor["error"] = f"assembled labeling prompt exceeds {config.compact_labeling_request_bytes} bytes"
                monitor["elapsed_seconds"] = round(time.time() - start, 2)
                return sample_idx, None, monitor
            call2_result, call2_raw, usage2, outcome2 = await _call("pass1.call2", msgs2)
            monitor["llm_calls"] += 1
            _record_llm_progress(1)
            monitor["total_prompt_tokens"] += usage2.get("prompt_tokens", 0)
            monitor["total_completion_tokens"] += usage2.get("completion_tokens", 0)

            if call2_result is None and usage2.get("content_filtered"):
                if sanitized_convs is None:
                    sanitized_convs, changed = sanitize_conversations_for_content_filter(truncated_convs)
                    if changed:
                        pass
                if sanitized_convs:
                    _sanitized_convs, sanitized_conversations_json, retry_msgs2, _retry_truncated, retry_prompt_bytes = _build_labeling_messages_with_budget(
                        sanitized_convs,
                        signals_str=signals_str,
                        compact=_compact,
                        max_chars=_max_chars,
                        config=config,
                        call1_context=call1_context,
                    )
                else:
                    sanitized_conversations_json = None
                if sanitized_conversations_json and sanitized_conversations_json != conversations_json:
                    retry_result, retry_raw, retry_usage, retry_outcome2 = await _call(
                        "pass1.call2.sanitized",
                        retry_msgs2,
                    )
                    monitor["llm_calls"] += 1
                    _record_llm_progress(1)
                    monitor["total_prompt_tokens"] += retry_usage.get("prompt_tokens", 0)
                    monitor["total_completion_tokens"] += retry_usage.get("completion_tokens", 0)
                    monitor["assembled_prompt_bytes_call2"] = retry_prompt_bytes
                    monitor["content_filter_retry"] = "call2"
                    if retry_result is not None:
                        call2_result, call2_raw, usage2, outcome2 = retry_result, retry_raw, retry_usage, retry_outcome2
                        conversations_json = sanitized_conversations_json
                        effective_conversations_json = sanitized_conversations_json
                    else:
                        call2_result, call2_raw, usage2, outcome2 = retry_result, retry_raw, retry_usage, retry_outcome2

            if call2_result is None:
                monitor["status"] = "call2_failed"
                monitor["error"] = usage2.get("error", "unknown")
                monitor["error_response"] = usage2.get("error_response") or (call2_raw[:500] if call2_raw else "")
                monitor["error_class"] = outcome2.classification.value
                monitor["retryable_infra"] = bool(outcome2.is_infra_failure)
                monitor["http_status"] = outcome2.status_code
                monitor["elapsed_seconds"] = round(time.time() - start, 2)
                if usage2.get("non_retryable"):
                    return sample_idx, _build_partial_labels(call1_cleaned, "call2_non_retryable_failure"), monitor
                if sample_attempt < _max_retries_sample:
                    continue
                return sample_idx, _build_partial_labels(call1_cleaned, "call2_retry_exhausted"), monitor

            call2_cleaned, call2_issues = validate_tags(call2_result, "call2")
            monitor["validation_issues"].extend(call2_issues)

            # Merge
            labels = {}
            for d in ["intent", "language", "domain", "task", "difficulty"]:
                labels[d] = call1_cleaned.get(d, [] if d in MULTI_SELECT else "")
            for d in ["concept", "agentic", "constraint", "context"]:
                labels[d] = call2_cleaned.get(d, [] if d in MULTI_SELECT else "")
            labels["confidence"] = {
                **(call1_cleaned.get("confidence", {}) if isinstance(call1_cleaned.get("confidence"), dict) else {}),
                **(call2_cleaned.get("confidence", {}) if isinstance(call2_cleaned.get("confidence"), dict) else {}),
            }
            labels["unmapped"] = call1_cleaned.get("unmapped", []) + call2_cleaned.get("unmapped", [])
            labels["canonicalized"] = list(call1_cleaned.get("canonicalized", [])) + list(call2_cleaned.get("canonicalized", []))
            for payload in (call1_cleaned, call2_cleaned):
                for d in ["intent", "language", "domain", "task", "difficulty", "concept", "agentic", "constraint", "context"]:
                    if d not in payload:
                        continue
                    if d in SINGLE_SELECT:
                        if not labels.get(d) and payload.get(d):
                            labels[d] = payload[d]
                        continue
                    for value in payload.get(d, []) or []:
                        if value not in labels[d]:
                            labels[d].append(value)

            warnings = check_consistency(labels)
            monitor["consistency_warnings"] = warnings

            low_conf = find_low_confidence_dims(labels, threshold=_conf_threshold)
            monitor["low_confidence_dims"] = [{"dim": d, "conf": s} for d, s in low_conf]

            arb_allowed = bool(enable_arbitration)
            if runtime is not None and getattr(config, "disable_arbitration_when_degraded", False):
                if runtime.state != "healthy":
                    arb_allowed = False
                    monitor["arbitration_skipped"] = f"runtime:{runtime.state}"

            if low_conf and arb_allowed:
                monitor["arbitrated"] = True
                call1_dims = {"intent", "language", "domain", "task", "difficulty"}
                call2_dims = {"concept", "agentic", "constraint", "context"}

                if any(d in call1_dims for d, _ in low_conf):
                    re1, _, u1, _ = await _call("pass1.call1.arb", msgs1, temperature=0.3, max_tokens=1000)
                    monitor["llm_calls"] += 1
                    _record_llm_progress(1)
                    monitor["total_prompt_tokens"] += u1.get("prompt_tokens", 0)
                    monitor["total_completion_tokens"] += u1.get("completion_tokens", 0)
                    if re1:
                        re1_clean, _ = validate_tags(re1, "call1")
                        for d, _ in low_conf:
                            if d in call1_dims and d in re1_clean:
                                labels[d] = re1_clean[d]
                                labels["confidence"][d] = re1_clean.get("confidence", {}).get(d, 0)

                if any(d in call2_dims for d, _ in low_conf):
                    re2, _, u2, _ = await _call("pass1.call2.arb", msgs2, temperature=0.3, max_tokens=1000)
                    monitor["llm_calls"] += 1
                    _record_llm_progress(1)
                    monitor["total_prompt_tokens"] += u2.get("prompt_tokens", 0)
                    monitor["total_completion_tokens"] += u2.get("completion_tokens", 0)
                    if re2:
                        re2_clean, _ = validate_tags(re2, "call2")
                        for d, _ in low_conf:
                            if d in call2_dims and d in re2_clean:
                                labels[d] = re2_clean[d]
                                labels["confidence"][d] = re2_clean.get("confidence", {}).get(d, 0)

            extension_specs = _resolved_extension_specs(config)
            if extension_specs:
                extension_usage: dict[str, dict] = {}

                async def _extension_llm_caller(messages, spec):
                    parsed, raw, usage, _outcome = await _call(
                        f"pass1.extension.{spec.id}",
                        messages,
                        temperature=0.1,
                        max_tokens=800,
                    )
                    extension_usage[spec.id] = dict(usage)
                    if parsed is None:
                        raise RuntimeError(usage.get("error") or raw or "extension labeling failed")
                    return parsed

                extension_payloads = await run_label_extensions(
                    conversation_json=conversations_json,
                    preprocessed_signals=signals_str,
                    core_labels=labels,
                    extension_specs=extension_specs,
                    llm_caller=_extension_llm_caller,
                )
                for spec_id, payload in extension_payloads.items():
                    usage = extension_usage.get(spec_id) or {}
                    extension_calls = int((payload.get("monitor") or {}).get("llm_calls", 0))
                    monitor["llm_calls"] += extension_calls
                    _record_llm_progress(extension_calls)
                    monitor["total_prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                    monitor["total_completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                    if isinstance(payload.get("monitor"), dict):
                        payload["monitor"]["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
                        payload["monitor"]["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
                        if usage.get("runtime_state") is not None:
                            payload["monitor"]["runtime_state"] = usage.get("runtime_state")
                labels["label_extensions"] = extension_payloads
                monitor["label_extensions"] = {
                    spec_id: {
                        "status": payload.get("status"),
                        "matched": payload.get("matched"),
                        "llm_calls": (payload.get("monitor") or {}).get("llm_calls", 0),
                    }
                    for spec_id, payload in extension_payloads.items()
                }

            monitor["elapsed_seconds"] = round(time.time() - start, 2)
            return sample_idx, labels, monitor

        except Exception as e:
            monitor["status"] = f"error: {str(e)[:100]}"
            if sample_attempt < _max_retries_sample:
                continue
            monitor["elapsed_seconds"] = round(time.time() - start, 2)
            return sample_idx, None, monitor

    # Exhausted retries
    monitor = {
        "sample_id": sample.get("id", f"sample-{sample_idx}"),
        "index": sample_idx,
        "llm_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": _max_retries_sample,
        "status": "timeout",
        "elapsed_seconds": round(time.time() - start, 2),
    }
    return sample_idx, None, monitor


def compute_stats(all_monitors, all_labels, inherit_map=None, extension_specs=None):
    """Compute aggregate statistics.

    When inherit_map is provided, inherited samples are:
    - Counted toward success (if non-None) and total_samples
    - Excluded from tag_distributions, confidence_stats, cross_matrix
      (to avoid inflating tag counts from duplicated labels)
    - Reported separately as sparse_labeled / sparse_inherited
    """
    total = len(all_labels)
    success = sum(1 for labels in all_labels if is_usable_labels(labels))
    total_calls = sum(m["llm_calls"] for m in all_monitors)
    total_pt = sum(m["total_prompt_tokens"] for m in all_monitors)
    total_ct = sum(m["total_completion_tokens"] for m in all_monitors)
    arbitrated = sum(1 for m in all_monitors if m["arbitrated"])

    inherited_indices = set(inherit_map.keys()) if inherit_map else set()
    distribution_total = sum(
        1 for idx, labels in enumerate(all_labels)
        if is_usable_labels(labels) and idx not in inherited_indices
    )

    combo_counts = {}
    distributions = {}
    for dim in ["intent", "language", "domain", "concept", "task", "agentic", "constraint", "context", "difficulty"]:
        dist = {}
        for idx, labels in enumerate(all_labels):
            if not is_usable_labels(labels) or idx in inherited_indices:
                continue
            val = labels.get(dim, [])
            if isinstance(val, list):
                for v in val:
                    dist[v] = dist.get(v, 0) + 1
            elif val:
                dist[val] = dist.get(val, 0) + 1
        distributions[dim] = dict(sorted(dist.items(), key=lambda x: -x[1]))

    for idx, labels in enumerate(all_labels):
        if not is_usable_labels(labels) or idx in inherited_indices:
            continue
        combo_key = _combo_key_from_labels(labels)
        if combo_key:
            combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1

    all_unmapped = {}
    canonicalization_counts = {}
    for idx, labels in enumerate(all_labels):
        if not is_usable_labels(labels) or idx in inherited_indices:
            continue
        for item in labels.get("unmapped", []):
            key = f"{item.get('dimension', '?')}:{item.get('value', '?')}" if isinstance(item, dict) else str(item)
            all_unmapped[key] = all_unmapped.get(key, 0) + 1
        for event in labels.get("canonicalized", []):
            if isinstance(event, dict):
                key = canonicalization_stat_key(event)
                canonicalization_counts[key] = canonicalization_counts.get(key, 0) + 1

    conf_stats = {}
    for dim in ["intent", "language", "domain", "task", "difficulty", "concept", "agentic", "constraint", "context"]:
        scores = [labels["confidence"][dim] for idx, labels in enumerate(all_labels)
                  if is_usable_labels(labels) and idx not in inherited_indices
                  and "confidence" in labels and isinstance(labels["confidence"].get(dim), (int, float))]
        if scores:
            conf_stats[dim] = {
                "mean": round(sum(scores) / len(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "below_threshold": sum(1 for s in scores if s < CONFIDENCE_THRESHOLD),
                "count": len(scores),
            }

    # Intent × Difficulty cross matrix
    cross = {}
    for idx, labels in enumerate(all_labels):
        if not is_usable_labels(labels) or idx in inherited_indices:
            continue
        intent = labels.get("intent", "?")
        diff = labels.get("difficulty", "?")
        key = f"{intent}|{diff}"
        cross[key] = cross.get(key, 0) + 1

    stats = {
        "total_samples": total,
        "distribution_total_samples": distribution_total,
        "success": success,
        "failed": total - success,
        "success_rate": round(success / max(total, 1), 4),
        "total_llm_calls": total_calls,
        "avg_calls_per_sample": round(total_calls / max(total, 1), 2),
        "total_prompt_tokens": total_pt,
        "total_completion_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
        "arbitrated_count": arbitrated,
        "arbitrated_rate": round(arbitrated / max(total, 1), 4),
        "validation_issue_count": sum(1 for m in all_monitors if m["validation_issues"]),
        "consistency_warning_count": sum(1 for m in all_monitors if m["consistency_warnings"]),
        "unmapped_tags": dict(sorted(all_unmapped.items(), key=lambda x: -x[1])),
        "unmapped_unique_count": len(all_unmapped),
        "canonicalization_counts": dict(sorted(canonicalization_counts.items(), key=lambda x: (-x[1], x[0]))),
        "canonicalization_total_count": sum(canonicalization_counts.values()),
        "canonicalization_unique_count": len(canonicalization_counts),
        "confidence_stats": conf_stats,
        "low_confidence_frequency": dict(sorted(
            {d: sum(1 for m in all_monitors for lc in m.get("low_confidence_dims", []) if lc["dim"] == d)
             for d in set(lc["dim"] for m in all_monitors for lc in m.get("low_confidence_dims", []))}.items(),
            key=lambda x: -x[1])),
        "tag_distributions": distributions,
        "combo_distributions": dict(sorted(combo_counts.items(), key=lambda x: -x[1])),
        "cross_matrix": cross,
    }
    extension_stats = aggregate_extension_stats(
        [{"labels": labels} for labels in all_labels if is_usable_labels(labels)],
        extension_specs=extension_specs,
    )
    if extension_stats:
        stats["extension_stats"] = extension_stats
    return stats


# ─────────────────────────────────────────────────────────
# Streaming I/O + cross-file helpers
# ─────────────────────────────────────────────────────────

def iter_samples_from_file(
    input_path,
    limit=0,
    shuffle=False,
    return_row_bundles=False,
    *,
    annotate_planner_metadata=True,
):
    """Load and normalize samples from a file with minimal memory overhead.

    JSONL: line-by-line read + normalize_and_slice (memory = 1 raw line at a time).
    JSON:  json.load then del raw (can't avoid full load, but releases raw ASAP).

    Returns:
      - default: (samples_list, n_raw)
      - with return_row_bundles=True: (samples_list, n_raw, row_bundles, sample_to_bundle)
    """
    input_path = Path(input_path)
    samples = []
    n_raw = 0

    if str(input_path).endswith(".jsonl"):
        if return_row_bundles:
            row_bundles = []
            sample_budget = 0
            for bundle in iter_row_sample_bundles_from_jsonl(
                input_path,
                limit=0,
                annotate_planner_metadata=annotate_planner_metadata,
            ):
                projected = sample_budget + len(bundle.samples)
                if limit > 0 and row_bundles and projected > limit:
                    break
                row_bundles.append(bundle)
                sample_budget = projected
                if limit > 0 and sample_budget >= limit:
                    break
            n_raw = len(row_bundles)
            samples, sample_to_bundle = flatten_row_sample_bundles(row_bundles)
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_raw += 1
                    raw = json.loads(line)
                    samples.extend(
                        normalize_and_slice(
                            raw,
                            source_file=input_path,
                            source_row=n_raw,
                            annotate_planner_metadata=annotate_planner_metadata,
                        )
                    )
                    del raw
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
        if not isinstance(raw_samples, list):
            raw_samples = [raw_samples]
        n_raw = len(raw_samples)
        for row_idx, s in enumerate(raw_samples, start=1):
            samples.extend(
                normalize_and_slice(
                    s,
                    source_file=input_path,
                    source_row=row_idx,
                    annotate_planner_metadata=annotate_planner_metadata,
                )
            )
        del raw_samples

    for i, s in enumerate(samples):
        if not s.get("id"):
            s["id"] = f"sample-{i:04d}"

    if shuffle:
        random.shuffle(samples)
    if limit > 0 and not (return_row_bundles and str(input_path).endswith(".jsonl")):
        samples = samples[:limit]

    if return_row_bundles and str(input_path).endswith(".jsonl"):
        return samples, n_raw, row_bundles, sample_to_bundle

    return samples, n_raw


@dataclass
class FileCollector:
    """Track per-file labeling results for cross-file pipeline."""
    file_idx: int
    abs_path: Path
    rel_path: Path          # relative path within input dir
    output_dir: Path
    prefix: str             # file stem for output naming
    total: int
    samples: list
    row_bundles: list | None = None
    sample_to_bundle: list | None = None
    dataset_output_path: Path | None = None
    run_failure_log_path: Path | None = None
    config: PipelineConfig | None = None
    inline_output: bool = False
    label_count: int = 0    # actual LLM labels (sparse)
    inherit_map: dict = field(default_factory=dict)
    updated_sample_indices: set[int] = field(default_factory=set)
    bundle_plans: list | None = None
    plan_stats: dict = field(default_factory=dict)
    run_mode: str = "refresh"
    sparse_info: str = ""   # progress bar context string
    done: int = 0
    ok: int = 0
    fail: int = 0
    labels: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    completed: bool = False
    submit_order: list[int] = field(default_factory=list)
    submit_cursor: int = 0
    chunked_delegate: bool = False
    live_progress: bool = False

    def __post_init__(self):
        # Pre-allocate result slots
        self.labels = [None] * self.total
        self.monitors = [None] * self.total

    def has_pending_submission(self) -> bool:
        return self.submit_cursor < len(self.submit_order)


@dataclass
class DirectoryWorkloadEstimate:
    """Pre-run workload estimate for directory mode."""
    files_planned: int
    total_raw_conversations: int
    total_samples: int
    total_labeled_samples: int
    total_inherited_samples: int
    baseline_total_llm_calls: int
    initial_estimated_llm_calls: int
    scan_elapsed_seconds: float


def _compute_pass1_conversation_mode(labeled_samples: list[dict], stats: dict) -> dict | None:
    """Compute Pass 1 conversation-mode dashboard stats from labeled samples."""
    from sft_label.tools.dashboard_aggregation import build_pass1_conversation_mode

    return build_pass1_conversation_mode(labeled_samples, stats)


def _write_pass1_conversation_stats(
    output_dir: Path,
    labeled_samples: list[dict] | None,
    stats: dict,
) -> None:
    """Persist lightweight Pass 1 conversation-mode stats for dashboard tree views."""
    conversation_mode = _compute_pass1_conversation_mode(labeled_samples or [], stats)
    if isinstance(conversation_mode, dict):
        _write_json_atomic(Path(output_dir) / PASS1_CONVERSATION_STATS_FILE, conversation_mode)


def _write_pass1_conversation_stats_from_path(output_dir: Path, labeled_path: Path, stats: dict) -> None:
    """Persist lightweight Pass 1 conversation-mode stats by streaming final artifacts."""
    from sft_label.tools.dashboard_aggregation import build_pass1_conversation_mode_from_iter, iter_data_file

    conversation_mode = build_pass1_conversation_mode_from_iter(iter_data_file(labeled_path), stats)
    if isinstance(conversation_mode, dict):
        _write_json_atomic(Path(output_dir) / PASS1_CONVERSATION_STATS_FILE, conversation_mode)


def _safe_write_pass1_conversation_stats(
    output_dir: Path,
    *,
    labeled_samples: list[dict] | None = None,
    labeled_path: Path | None = None,
    stats: dict,
    pprint=print,
) -> None:
    try:
        if labeled_path is not None:
            _write_pass1_conversation_stats_from_path(output_dir, labeled_path, stats)
        else:
            _write_pass1_conversation_stats(output_dir, labeled_samples, stats)
    except Exception as exc:
        pprint(f"  [warn] skipped conversation sidecar: {type(exc).__name__}: {exc}")


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
        self._smoothed_cps = 0.0
        self._warmup_samples = min(max(8, self.total_labeled_samples // 50), 30)

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

        elapsed = max(time.time() - self._start_time, 1e-6)
        instant_cps = self.calls_done / elapsed
        if self._smoothed_cps <= 0:
            self._smoothed_cps = instant_cps
        else:
            self._smoothed_cps = 0.2 * instant_cps + 0.8 * self._smoothed_cps
        self.calls_per_sec = self._smoothed_cps

    def eta_seconds(self):
        if self.calls_per_sec <= 0:
            return None
        remaining = max(self.estimated_total_calls - self.calls_done, 0)
        return remaining / self.calls_per_sec

    def info_line(self):
        rate = f"{self.calls_per_sec:.1f}/s" if self.calls_per_sec > 0 else "warming"
        eta = self._fmt_duration(self.eta_seconds())
        avg = f"{self.avg_calls_per_sample:.2f}/sample" if self.samples_done > 0 else "n/a"
        return f"eta {eta} • rate {rate} • avg {avg}"


def estimate_directory_workload(
    dir_files,
    *,
    limit=0,
    shuffle=False,
    parallelize: bool = False,
    config=None,
    completed_set=None,
    enable_arbitration=True,
    mode: str = "refresh",
    migration_index: dict | None = None,
    label_version: str = DEFAULT_LABEL_VERSION,
):
    """Estimate global workload before directory-mode execution."""
    completed_set = completed_set or set()
    start = time.time()
    total_raw = 0
    total_samples = 0
    total_labeled = 0
    total_inherited = 0
    files_planned = 0

    _sparse_kw = {}
    if config:
        _sparse_kw = dict(
            full_label_count=config.sparse_full_label_count,
            gap_multiplier=config.sparse_gap_multiplier,
            min_gap=config.sparse_min_gap,
            max_gap=config.sparse_max_gap,
            threshold=config.sparse_threshold,
            planner_enabled=config.planner_enabled,
            planner_metadata_only=config.planner_metadata_only,
            planner_policy=config.planner_policy,
            planner_boundary_threshold=config.planner_boundary_threshold,
            planner_min_segment_size=config.planner_min_segment_size,
            planner_max_anchor_gap=config.planner_max_anchor_gap,
            planner_fallback_boundary_ratio=config.planner_fallback_boundary_ratio,
        )

    extension_specs = _resolved_extension_specs(config)
    annotate_planner_metadata = bool(getattr(config, "planner_enabled", False))

    pending_items = []
    for abs_path, rel_path in dir_files:
        rel_str = str(rel_path)
        if rel_str in completed_set:
            continue
        files_planned += 1
        pending_items.append(
            {
                "abs_path": str(abs_path),
                "rel_path": rel_str,
                "limit": limit,
                "shuffle": shuffle,
                "mode": mode,
                "label_version": label_version,
                "sparse_kwargs": dict(_sparse_kw),
                "extension_specs": extension_specs,
                "migration_index": migration_index,
                "annotate_planner_metadata": annotate_planner_metadata,
            }
        )

    workers = (
        _resolve_workload_estimation_workers(files_planned)
        if parallelize and migration_index is None
        else 1
    )

    # Preserve RNG state so pre-scan doesn't perturb runtime shuffle behavior.
    rng_state = random.getstate()
    try:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                file_results = executor.map(_estimate_directory_workload_for_file, pending_items)
                for result in file_results:
                    total_raw += result["total_raw_conversations"]
                    total_samples += result["total_samples"]
                    total_labeled += result["total_labeled_samples"]
                    total_inherited += result["total_inherited_samples"]
        else:
            for item in pending_items:
                result = _estimate_directory_workload_for_file(item)
                total_raw += result["total_raw_conversations"]
                total_samples += result["total_samples"]
                total_labeled += result["total_labeled_samples"]
                total_inherited += result["total_inherited_samples"]
    finally:
        random.setstate(rng_state)

    baseline_calls = total_labeled * 2
    # Call1 + Call2 are guaranteed. Arbitration may add extra calls.
    initial_calls_per_sample = 2.2 if enable_arbitration else 2.0
    initial_est_calls = int(round(total_labeled * initial_calls_per_sample))
    initial_est_calls = max(initial_est_calls, baseline_calls)

    return DirectoryWorkloadEstimate(
        files_planned=files_planned,
        total_raw_conversations=total_raw,
        total_samples=total_samples,
        total_labeled_samples=total_labeled,
        total_inherited_samples=total_inherited,
        baseline_total_llm_calls=baseline_calls,
        initial_estimated_llm_calls=initial_est_calls,
        scan_elapsed_seconds=round(time.time() - start, 2),
    )


def _resolve_workload_estimation_workers(file_count: int) -> int:
    """Auto-size Pass 1 estimation workers to ~80% of available CPUs."""
    if file_count <= 1:
        return 1
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if not main_file or str(main_file).startswith("<"):
        return 1
    cpu_total = os.cpu_count() or 1
    target = max(1, int(cpu_total * WORKLOAD_ESTIMATION_CPU_FRACTION))
    return max(1, min(file_count, target))


def _estimate_directory_workload_for_file(item: dict) -> dict:
    """Estimate Pass 1 workload counts for a single input file."""
    abs_path = Path(item["abs_path"])
    limit = item.get("limit", 0)
    shuffle = bool(item.get("shuffle", False))
    mode = item.get("mode", "refresh")
    label_version = item.get("label_version", DEFAULT_LABEL_VERSION)
    sparse_kwargs = dict(item.get("sparse_kwargs") or {})
    extension_specs = list(item.get("extension_specs") or [])
    migration_index = item.get("migration_index")
    annotate_planner_metadata = bool(item.get("annotate_planner_metadata", True))

    total_raw = 0
    total_samples = 0
    total_labeled = 0
    total_inherited = 0

    if str(abs_path).endswith(".jsonl"):
        for row_bundle in iter_row_sample_bundles_from_jsonl(
            abs_path,
            limit=limit,
            annotate_planner_metadata=annotate_planner_metadata,
        ):
            prepared = prepare_inline_pass1_batch(
                [row_bundle],
                mode=mode,
                label_version=label_version,
                sparse_kwargs=sparse_kwargs,
                extension_specs=extension_specs,
                migration_index=migration_index,
            )
            total_raw += len(prepared.bundles)
            total_samples += len(prepared.samples)
            total_labeled += len(prepared.label_indices)
            total_inherited += len(prepared.inherit_map)
        return {
            "total_raw_conversations": total_raw,
            "total_samples": total_samples,
            "total_labeled_samples": total_labeled,
            "total_inherited_samples": total_inherited,
        }

    samples, n_raw = iter_samples_from_file(
        abs_path,
        limit=limit,
        shuffle=shuffle,
        annotate_planner_metadata=annotate_planner_metadata,
    )
    label_indices, inherit_map = apply_sparse_sampling(samples, **sparse_kwargs)
    return {
        "total_raw_conversations": n_raw,
        "total_samples": len(samples),
        "total_labeled_samples": len(label_indices),
        "total_inherited_samples": len(inherit_map),
    }


def flush_file_output(collector, run_dir, checkpoint_path, pprint=print, generate_dashboard=True):
    """Write all outputs for a completed file and release memory.

    Writes labeled.json/jsonl, monitor.jsonl, stats_labeling.json, dashboard.
    Updates checkpoint. Deletes heavy data from collector to free memory.
    Returns the stats dict.
    """
    samples = collector.samples
    all_labels = collector.labels
    all_monitors = collector.monitors
    output_dir = collector.output_dir
    prefix = collector.prefix

    resolved_labels = apply_inherited_labels(samples, all_labels, collector.inherit_map)
    all_labels[:] = resolved_labels

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if collector.inline_output else (f"_{prefix}" if prefix else "")

    labeled_json = f"labeled{suffix}.json"
    labeled_jsonl = f"labeled{suffix}.jsonl"
    monitor_file = f"monitor{suffix}.jsonl"
    stats_file = pass1_stats_filename(suffix)
    dashboard_file = pass1_dashboard_filename(suffix)
    failed_samples_file = f"failed_samples{suffix}.jsonl"
    unmapped_events_file = f"unmapped_events{suffix}.jsonl"

    if collector.inline_output:
        merge_result = merge_pass1_results(
            collector.row_bundles or [],
            samples,
            resolved_labels,
            all_monitors,
            collector.sample_to_bundle or [],
            source_file=collector.abs_path,
            mode=collector.run_mode,
            bundle_plans=collector.bundle_plans,
            updated_sample_indices=collector.updated_sample_indices,
        )
        _write_jsonl_atomic(collector.dataset_output_path, merge_result.rows)
        _write_json_atomic(output_dir / labeled_json, merge_result.samples)
        _write_jsonl_atomic(output_dir / labeled_jsonl, merge_result.samples)
        _write_jsonl_atomic(output_dir / monitor_file, merge_result.monitor_records)
        _write_jsonl_atomic(output_dir / unmapped_events_file, build_unmapped_event_records(merge_result.samples))
        labeled_samples_for_dashboard = merge_result.samples
        if merge_result.failed_samples:
            _write_jsonl_atomic(output_dir / failed_samples_file, merge_result.failed_samples)
    else:
        rendered_samples, monitor_records, failed_samples = build_sample_artifacts(
            samples,
            resolved_labels,
            all_monitors,
            source_file=collector.abs_path,
        )
        _write_json_atomic(output_dir / labeled_json, rendered_samples)
        _write_jsonl_atomic(output_dir / labeled_jsonl, rendered_samples)
        _write_jsonl_atomic(output_dir / monitor_file, monitor_records)
        _write_jsonl_atomic(output_dir / unmapped_events_file, build_unmapped_event_records(rendered_samples))
        labeled_samples_for_dashboard = rendered_samples
        if failed_samples:
            _write_jsonl_atomic(output_dir / failed_samples_file, failed_samples)

    # Inherited samples have no monitor — they are not failures
    inherited_indices = set(collector.inherit_map.keys())
    failed_indices = [
        i for i, labels in enumerate(all_labels)
        if (labels is None or is_partial_labels(labels)) and i not in inherited_indices
    ]

    # Append to global failure log at run_dir root
    failure_records = []
    for i in failed_indices:
        m = all_monitors[i]
        record = {
            "sample_id": samples[i].get("id", f"sample-{i}"),
            "source_file": str(collector.abs_path),
            "status": m["status"] if m else "no_result",
            "error": (m.get("error", "") if m else "no monitor record"),
            "error_response": (m.get("error_response", "")[:1000] if m else ""),
            "attempts": (m.get("sample_attempt", 0) + 1 if m else 0),
        }
        failure_records.append(record)
    if failure_records:
        failure_log_path = collector.run_failure_log_path or (run_dir / "failures.jsonl")
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log_path, "a", encoding="utf-8") as f:
            for r in failure_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute and write stats
    valid_monitors = [m for m in all_monitors if m is not None]
    stats = compute_stats(
        valid_monitors,
        all_labels,
        inherit_map=collector.inherit_map,
        extension_specs=_resolved_extension_specs(collector.config),
    )
    stats["input_file"] = str(collector.abs_path)
    stats["mode"] = collector.run_mode
    if collector.plan_stats:
        stats.update(collector.plan_stats)
    sparse_inherited = len(collector.inherit_map)
    if sparse_inherited > 0:
        stats["sparse_labeled"] = collector.label_count
        stats["sparse_inherited"] = sparse_inherited
    _annotate_labeling_prompt_stats(stats, collector.config)

    stats_path = output_dir / stats_file
    _write_json_atomic(stats_path, stats)
    _safe_write_pass1_conversation_stats(
        output_dir,
        labeled_samples=labeled_samples_for_dashboard,
        stats=stats,
        pprint=pprint,
    )

    # Per-file dashboard
    if generate_dashboard:
        try:
            from sft_label.tools.visualize_labels import generate_dashboard
            generate_dashboard(output_dir, labeled_file=labeled_json,
                               stats_file=stats_file, output_file=dashboard_file,
                               quiet=True)
        except Exception:
            pass

    stats["success"]
    stats["total_tokens"]

    # Print failure details — collapsed single-line summary
    failed_count = len(failed_indices)
    if failed_count > 0:
        failed_monitors = [all_monitors[i] for i in failed_indices if all_monitors[i] and all_monitors[i]["status"] != "success"]
        error_groups = {}
        for m in failed_monitors:
            err_type = m["status"]
            error_groups[err_type] = error_groups.get(err_type, 0) + 1
        timeout_count = sum(1 for i, m in enumerate(all_monitors) if m is None and i not in inherited_indices)
        if timeout_count > 0:
            error_groups["no_result"] = timeout_count
        err_parts = [f"{t}×{n}" for t, n in sorted(error_groups.items())]
        rel_name = getattr(collector, 'rel_path', None)
        file_label = str(rel_name) if rel_name else "file"
        pprint(f"  ✗ {file_label}: {failed_count} failed [{', '.join(err_parts)}]")

    # Update checkpoint
    if checkpoint_path:
        rel_str = str(collector.rel_path)
        update_checkpoint(checkpoint_path, rel_str, success=True)

    # Release memory
    collector.samples = None
    collector.labels = None
    collector.monitors = None
    collector.row_bundles = None
    collector.sample_to_bundle = None
    collector.inherit_map = {}
    collector.updated_sample_indices = set()
    collector.bundle_plans = None
    collector.plan_stats = {}
    collector.submit_order = []
    collector.submit_cursor = 0
    collector.config = None
    collector.completed = True

    return stats


# ─────────────────────────────────────────────────────────
# Chunked JSONL pipeline (memory-bounded for large files)
# ─────────────────────────────────────────────────────────

def iter_chunks_from_jsonl(path, chunk_size, limit=0):
    """Yield lists of parsed JSONL records, at most chunk_size per yield.

    Each yielded list contains raw parsed dicts (one per line).
    Respects limit (total raw lines cap, 0 = unlimited).
    """
    count = 0
    chunk = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk.append(json.loads(line))
            count += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if limit > 0 and count >= limit:
                break
    if chunk:
        yield chunk


class StatsAccumulator:
    """Incremental version of compute_stats() for chunked processing.

    Accumulates results chunk-by-chunk and produces the same dict structure
    as compute_stats() via finalize().
    """

    DIMS = ["intent", "language", "domain", "concept", "task",
            "agentic", "constraint", "context", "difficulty"]

    def __init__(self, extension_specs=None, compact_labeling_request_bytes=None):
        self.total = 0
        self.distribution_total = 0
        self.success = 0
        self.total_calls = 0
        self.total_pt = 0
        self.total_ct = 0
        self.arbitrated = 0
        self.validation_issue_count = 0
        self.consistency_warning_count = 0
        self.canonicalization_total_count = 0

        self.distributions = {dim: {} for dim in self.DIMS}
        self.combo_counts = {}
        self.unmapped = {}
        self.canonicalization_counts = {}
        self.cross_matrix = {}

        # Confidence: sum/count/min/max per dim for running mean
        self.conf_acc = {}

        # Low-confidence frequency
        self.low_conf_freq = {}

        # Sparse tracking
        self.sparse_labeled = 0
        self.sparse_inherited = 0
        self.extension_stats_batches = []
        self.extension_specs = extension_specs or []
        self.compact_labeling_request_bytes = int(compact_labeling_request_bytes or 0)

        # Planner rollout observability
        self.planner_conversations_total = 0
        self.planner_conversations_with_metadata = 0
        self.planner_fallback_conversations = 0
        self.planner_segment_counts = OnlineNumericSummary()
        self.planner_boundary_scores = OnlineNumericSummary()
        self.planner_anchor_priority_counts = {}
        self.planner_anchor_reason_counts = {}
        self.planner_inherit_edges = 0
        self.planner_cross_segment_inherit_edges = 0

        # Compact prompt byte tracking
        self.prompt_bytes_call1 = OnlineNumericSummary()
        self.prompt_bytes_call2 = OnlineNumericSummary()
        self.prompt_bytes_call1_hard_cap_hits = 0
        self.prompt_bytes_call2_hard_cap_hits = 0

    def update(self, all_labels, all_monitors, inherit_map=None, samples=None):
        """Ingest one chunk's labels and monitors.

        Inherited samples (from inherit_map) are counted toward total/success
        but excluded from tag_distributions, confidence, and cross_matrix
        to avoid inflating counts from duplicated labels.
        """
        inherited_indices = set(inherit_map.keys()) if inherit_map else set()
        self.total += len(all_labels)
        self.success += sum(1 for labels in all_labels if is_usable_labels(labels))

        for m in all_monitors:
            if m is None:
                continue
            self.total_calls += m["llm_calls"]
            self.total_pt += m["total_prompt_tokens"]
            self.total_ct += m["total_completion_tokens"]
            if m["arbitrated"]:
                self.arbitrated += 1
            if m["validation_issues"]:
                self.validation_issue_count += 1
            if m["consistency_warnings"]:
                self.consistency_warning_count += 1
            for lc in m.get("low_confidence_dims", []):
                d = lc["dim"]
                self.low_conf_freq[d] = self.low_conf_freq.get(d, 0) + 1
            call1_bytes = m.get("assembled_prompt_bytes")
            if isinstance(call1_bytes, (int, float)) and not isinstance(call1_bytes, bool):
                self.prompt_bytes_call1.add(int(call1_bytes))
                if self.compact_labeling_request_bytes and int(call1_bytes) > self.compact_labeling_request_bytes:
                    self.prompt_bytes_call1_hard_cap_hits += 1
            call2_bytes = m.get("assembled_prompt_bytes_call2")
            if isinstance(call2_bytes, (int, float)) and not isinstance(call2_bytes, bool):
                self.prompt_bytes_call2.add(int(call2_bytes))
                if self.compact_labeling_request_bytes and int(call2_bytes) > self.compact_labeling_request_bytes:
                    self.prompt_bytes_call2_hard_cap_hits += 1

        for idx, labels in enumerate(all_labels):
            if not is_usable_labels(labels) or idx in inherited_indices:
                continue
            self.distribution_total += 1
            for dim in self.DIMS:
                val = labels.get(dim, [])
                if isinstance(val, list):
                    for v in val:
                        self.distributions[dim][v] = self.distributions[dim].get(v, 0) + 1
                elif val:
                    self.distributions[dim][val] = self.distributions[dim].get(val, 0) + 1
            combo_key = _combo_key_from_labels(labels)
            if combo_key:
                self.combo_counts[combo_key] = self.combo_counts.get(combo_key, 0) + 1
            # Unmapped
            for item in labels.get("unmapped", []):
                key = f"{item.get('dimension', '?')}:{item.get('value', '?')}" if isinstance(item, dict) else str(item)
                self.unmapped[key] = self.unmapped.get(key, 0) + 1
            for event in labels.get("canonicalized", []):
                if isinstance(event, dict):
                    key = canonicalization_stat_key(event)
                    self.canonicalization_counts[key] = self.canonicalization_counts.get(key, 0) + 1
                    self.canonicalization_total_count += 1
            # Confidence
            conf = labels.get("confidence")
            if isinstance(conf, dict):
                for dim in self.DIMS:
                    score = conf.get(dim)
                    if isinstance(score, (int, float)):
                        if dim not in self.conf_acc:
                            self.conf_acc[dim] = {"sum": 0.0, "count": 0, "min": score, "max": score, "below": 0}
                        acc = self.conf_acc[dim]
                        acc["sum"] += score
                        acc["count"] += 1
                        acc["min"] = min(acc["min"], score)
                        acc["max"] = max(acc["max"], score)
                        if score < CONFIDENCE_THRESHOLD:
                            acc["below"] += 1
            # Cross matrix
            intent = labels.get("intent", "?")
            diff = labels.get("difficulty", "?")
            key = f"{intent}|{diff}"
            self.cross_matrix[key] = self.cross_matrix.get(key, 0) + 1

        if inherit_map:
            self.sparse_inherited += len(inherit_map)

        if samples:
            self._update_planner_stats(samples, inherit_map=inherit_map)

        extension_stats = aggregate_extension_stats(
            [{"labels": labels} for labels in all_labels if is_usable_labels(labels)],
            extension_specs=self.extension_specs,
        )
        if extension_stats:
            self.extension_stats_batches.append({"extension_stats": extension_stats})

    def set_sparse_labeled(self, count):
        """Track total sparse-labeled count across chunks."""
        self.sparse_labeled += count

    def _update_planner_stats(self, samples, *, inherit_map=None):
        groups = {}
        for idx, sample in enumerate(samples):
            meta = sample.get("metadata") or {}
            conversation_key = meta.get("conversation_uid") or meta.get("source_id")
            total_turns = meta.get("total_turns", 1)
            if not conversation_key or total_turns <= 1:
                continue
            groups.setdefault(conversation_key, []).append((idx, sample))

        self.planner_conversations_total += len(groups)
        for members in groups.values():
            metas = [(sample.get("metadata") or {}) for _idx, sample in members]
            planner_metas = [meta for meta in metas if meta.get("planner_policy")]
            if not planner_metas:
                continue
            self.planner_conversations_with_metadata += 1
            if any(meta.get("planner_fallback") for meta in planner_metas):
                self.planner_fallback_conversations += 1

            segment_ids = {meta.get("segment_id") for meta in planner_metas if meta.get("segment_id") is not None}
            if segment_ids:
                self.planner_segment_counts.add(len(segment_ids))

            for meta in planner_metas:
                boundary_score = meta.get("boundary_score")
                if isinstance(boundary_score, (int, float)) and not isinstance(boundary_score, bool):
                    self.planner_boundary_scores.add(float(boundary_score))
                anchor_priority = str(meta.get("anchor_priority") or "").strip().lower()
                if anchor_priority:
                    self.planner_anchor_priority_counts[anchor_priority] = (
                        self.planner_anchor_priority_counts.get(anchor_priority, 0) + 1
                    )
                anchor_reason = str(meta.get("anchor_reason") or "").strip().lower()
                if anchor_reason:
                    self.planner_anchor_reason_counts[anchor_reason] = (
                        self.planner_anchor_reason_counts.get(anchor_reason, 0) + 1
                    )

        for target_idx, source_idx in (inherit_map or {}).items():
            if not (0 <= target_idx < len(samples) and 0 <= source_idx < len(samples)):
                continue
            target_meta = samples[target_idx].get("metadata") or {}
            source_meta = samples[source_idx].get("metadata") or {}
            if not target_meta.get("planner_policy") or not source_meta.get("planner_policy"):
                continue
            self.planner_inherit_edges += 1
            if target_meta.get("segment_id") != source_meta.get("segment_id"):
                self.planner_cross_segment_inherit_edges += 1

    def finalize(self):
        """Produce the same dict structure as compute_stats()."""
        # Sort distributions
        sorted_dist = {}
        for dim in self.DIMS:
            sorted_dist[dim] = dict(sorted(self.distributions[dim].items(), key=lambda x: -x[1]))

        conf_stats = {}
        for dim, acc in self.conf_acc.items():
            n = acc["count"]
            if n > 0:
                conf_stats[dim] = {
                    "mean": round(acc["sum"] / n, 3),
                    "min": round(acc["min"], 3),
                    "max": round(acc["max"], 3),
                    "below_threshold": acc["below"],
                    "count": n,
                }

        total = self.total
        result = {
            "total_samples": total,
            "distribution_total_samples": self.distribution_total,
            "success": self.success,
            "failed": total - self.success,
            "success_rate": round(self.success / max(total, 1), 4),
            "total_llm_calls": self.total_calls,
            "avg_calls_per_sample": round(self.total_calls / max(total, 1), 2),
            "total_prompt_tokens": self.total_pt,
            "total_completion_tokens": self.total_ct,
            "total_tokens": self.total_pt + self.total_ct,
            "arbitrated_count": self.arbitrated,
            "arbitrated_rate": round(self.arbitrated / max(total, 1), 4),
            "validation_issue_count": self.validation_issue_count,
            "consistency_warning_count": self.consistency_warning_count,
            "unmapped_tags": dict(sorted(self.unmapped.items(), key=lambda x: -x[1])),
            "unmapped_unique_count": len(self.unmapped),
            "canonicalization_counts": dict(sorted(self.canonicalization_counts.items(), key=lambda x: (-x[1], x[0]))),
            "canonicalization_total_count": self.canonicalization_total_count,
            "canonicalization_unique_count": len(self.canonicalization_counts),
            "confidence_stats": conf_stats,
            "low_confidence_frequency": dict(sorted(self.low_conf_freq.items(), key=lambda x: -x[1])),
            "tag_distributions": sorted_dist,
            "combo_distributions": dict(sorted(self.combo_counts.items(), key=lambda x: -x[1])),
            "cross_matrix": self.cross_matrix,
        }
        extension_stats = merge_extension_stats(self.extension_stats_batches)
        if extension_stats:
            result["extension_stats"] = extension_stats
        if self.sparse_inherited > 0:
            result["sparse_labeled"] = self.sparse_labeled
            result["sparse_inherited"] = self.sparse_inherited
        planner_stats = {
            "conversations_total": self.planner_conversations_total,
            "conversations_with_metadata": self.planner_conversations_with_metadata,
            "fallback_conversations": self.planner_fallback_conversations,
            "fallback_rate": round(self.planner_fallback_conversations / max(self.planner_conversations_with_metadata, 1), 4),
            "inherit_edges": self.planner_inherit_edges,
            "cross_segment_inherit_edges": self.planner_cross_segment_inherit_edges,
            "cross_segment_inherit_ratio": round(
                self.planner_cross_segment_inherit_edges / max(self.planner_inherit_edges, 1),
                4,
            ),
            "anchor_priority_counts": dict(sorted(self.planner_anchor_priority_counts.items(), key=lambda item: (-item[1], item[0]))),
            "anchor_reason_counts": dict(sorted(self.planner_anchor_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        }
        segments_summary = self.planner_segment_counts.as_dict()
        if segments_summary:
            planner_stats["segments_per_conversation"] = segments_summary
        boundary_summary = self.planner_boundary_scores.as_dict()
        if boundary_summary:
            planner_stats["boundary_score"] = boundary_summary
        if self.planner_conversations_total:
            result["planner_stats"] = planner_stats

        compact_prompt_bytes = {}
        call1_summary = self.prompt_bytes_call1.as_dict()
        if call1_summary:
            call1_summary["limit_bytes"] = self.compact_labeling_request_bytes
            call1_summary["hard_cap_hits"] = self.prompt_bytes_call1_hard_cap_hits
            call1_summary["hard_cap_hit_rate"] = round(
                self.prompt_bytes_call1_hard_cap_hits / max(call1_summary["count"], 1), 4
            )
            compact_prompt_bytes["call1"] = call1_summary
        call2_summary = self.prompt_bytes_call2.as_dict()
        if call2_summary:
            call2_summary["limit_bytes"] = self.compact_labeling_request_bytes
            call2_summary["hard_cap_hits"] = self.prompt_bytes_call2_hard_cap_hits
            call2_summary["hard_cap_hit_rate"] = round(
                self.prompt_bytes_call2_hard_cap_hits / max(call2_summary["count"], 1), 4
            )
            compact_prompt_bytes["call2"] = call2_summary
        if compact_prompt_bytes:
            result["compact_prompt_bytes"] = compact_prompt_bytes

        return result


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


def _combo_key_from_labels(labels):
    """Build the richer combo key used by Pass 2 rarity baselines."""
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


@dataclass
class ChunkCollector:
    """Track per-chunk labeling results for chunked JSONL pipeline."""
    chunk_idx: int
    samples: list
    row_bundles: list | None = None
    sample_to_bundle: list | None = None
    label_count: int = 0
    inherit_map: dict = field(default_factory=dict)
    updated_sample_indices: set[int] = field(default_factory=set)
    bundle_plans: list | None = None
    plan_stats: dict = field(default_factory=dict)
    run_mode: str = "refresh"
    done: int = 0
    ok: int = 0
    fail: int = 0
    labels: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    completed: bool = False
    sample_id_offset: int = 0  # global offset for sample IDs
    submit_order: list[int] = field(default_factory=list)
    submit_cursor: int = 0

    def __post_init__(self):
        n = len(self.samples)
        self.labels = [None] * n
        self.monitors = [None] * n

    def has_pending_submission(self) -> bool:
        return self.submit_cursor < len(self.submit_order)


async def _run_pass1_recovery_sweep(
    *,
    samples: list,
    all_labels: list,
    all_monitors: list,
    candidate_indices: list[int],
    http_client,
    model: str,
    config,
    enable_arbitration: bool,
) -> int:
    """Retry infra-failed samples once (or a few passes) before finalizing outputs."""
    if not config or not getattr(config, "enable_stage_recovery_sweep", False):
        return 0
    if not getattr(config, "enable_adaptive_runtime", False):
        return 0
    if _config_adaptive_runtime(config) is None:
        return 0
    max_passes = int(getattr(config, "recovery_sweep_max_passes", 1) or 0)
    if max_passes <= 0:
        return 0

    recovered_total = 0
    recovery_config = _make_recovery_config(config)
    if recovery_config is None:
        return 0
    recovery_runtime = _build_adaptive_runtime(recovery_config)
    setattr(recovery_config, "_adaptive_runtime", recovery_runtime)
    recovery_sem = asyncio.Semaphore(max(1, int(getattr(recovery_config, "concurrency", 1) or 1)))

    for _ in range(max_passes):
        retry_indices = []
        for i in candidate_indices:
            labels = all_labels[i]
            if labels is not None and not is_partial_labels(labels):
                continue
            m = all_monitors[i] or {}
            if not m.get("retryable_infra", False):
                continue
            retry_indices.append(i)

        if not retry_indices:
            break

        results = await asyncio.gather(
            *[
                label_one(
                    http_client,
                    samples[i],
                    model,
                    i,
                    len(samples),
                    recovery_sem,
                    enable_arbitration=enable_arbitration,
                    config=recovery_config,
                    rate_limiter=None,
                )
                for i in retry_indices
            ]
        )
        for (_, labels, monitor), idx in zip(results, retry_indices):
            before = all_labels[idx]
            all_labels[idx] = labels
            all_monitors[idx] = monitor
            if (before is None or is_partial_labels(before)) and (labels is not None and not is_partial_labels(labels)):
                recovered_total += 1

    return recovered_total


def _flush_chunk(chunk, out_rows, out_labeled, out_monitor, out_failed, out_unmapped, stats_acc,
                 input_path, pprint=print, run_failure_log_path=None):
    """Finalize a completed chunk: inherit, attach labels, write, release memory."""
    samples = chunk.samples
    all_labels = chunk.labels
    all_monitors = chunk.monitors

    resolved_labels = apply_inherited_labels(samples, all_labels, chunk.inherit_map)
    all_labels[:] = resolved_labels

    merge_result = merge_pass1_results(
        chunk.row_bundles or [],
        samples,
        resolved_labels,
        all_monitors,
        chunk.sample_to_bundle or [],
        source_file=input_path,
        mode=chunk.run_mode,
        bundle_plans=chunk.bundle_plans,
        updated_sample_indices=chunk.updated_sample_indices,
    )

    append_rows_jsonl(out_rows, merge_result.rows)
    for sample in merge_result.samples:
        out_labeled.write(json.dumps(sample, ensure_ascii=False) + "\n")
    for monitor in merge_result.monitor_records:
        out_monitor.write(json.dumps(monitor, ensure_ascii=False) + "\n")
    for failed in merge_result.failed_samples:
        out_failed.write(json.dumps(failed, ensure_ascii=False) + "\n")
    for event in build_unmapped_event_records(merge_result.samples):
        out_unmapped.write(json.dumps(event, ensure_ascii=False) + "\n")

    if run_failure_log_path and merge_result.failed_samples:
        run_failure_log_path = Path(run_failure_log_path)
        run_failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(run_failure_log_path, "a", encoding="utf-8") as f:
            for idx, failed in enumerate(merge_result.failed_samples):
                monitor = failed.get("labeling_monitor") or {}
                f.write(json.dumps({
                    "sample_id": failed.get("id", f"sample-{idx}"),
                    "source_file": str(input_path),
                    "status": monitor.get("status", "no_result"),
                    "error": monitor.get("error", ""),
                    "error_response": monitor.get("error_response", ""),
                    "attempts": monitor.get("sample_attempt", 0) + 1 if monitor else 0,
                    "error_class": monitor.get("error_class"),
                    "retryable_infra": monitor.get("retryable_infra"),
                    "http_status": monitor.get("http_status"),
                    "runtime_state": monitor.get("runtime_state"),
                }, ensure_ascii=False) + "\n")

    # Update stats accumulator
    stats_acc.update(all_labels, all_monitors, inherit_map=chunk.inherit_map, samples=samples)
    stats_acc.set_sparse_labeled(chunk.label_count)

    # Release memory
    chunk.samples = None
    chunk.labels = None
    chunk.monitors = None
    chunk.row_bundles = None
    chunk.sample_to_bundle = None
    chunk.inherit_map = {}
    chunk.updated_sample_indices = set()
    chunk.bundle_plans = None
    chunk.plan_stats = {}
    chunk.completed = True


def _effective_chunk_row_count(chunk_size: int, watermark: int) -> int:
    """Cap giant JSONL chunks so large runs flush incrementally."""
    requested = max(int(chunk_size or 0), 1)
    target = max(int(watermark or 0) * 2, 128)
    return max(1, min(requested, target))


async def _run_one_file_chunked(input_path, output_dir, http_client, sem, model,
                                 enable_arbitration=True, limit=0, shuffle=False,
                                 progress=None, sample_task=None, config=None,
                                 rate_limiter=None, llm_progress_cb=None,
                                 dataset_output_path=None,
                                 run_failure_log_path=None,
                                 inline_output: bool = False,
                                 mode: str = "refresh",
                                 migration_index: dict | None = None,
                                 progress_event_cb=None,
                                 directory_delegate: bool = False):
    """Chunked JSONL labeling: watermark-based processing for large files.

    Processes JSONL files in chunks to bound memory usage. Each chunk is
    independently: read → slice → sparse sample → label → inherit → write.

    Output: labeled.jsonl, monitor.jsonl, failed_samples.jsonl, stats_labeling.json, dashboard_labeling.html
    (No labeled.json — too large to hold in memory for JSON serialization)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pprint = progress.console.print if progress else print

    requested_chunk_size = config.chunk_size if config else CHUNK_SIZE
    max_active = config.max_active_chunks if config else MAX_ACTIVE_CHUNKS
    if directory_delegate:
        # A directory-delegated JSONL file is already a heavyweight producer that
        # can keep the global semaphore saturated on its own. Limit it to one
        # active chunk so we do not multiply large-file memory/IO pressure.
        max_active = min(max_active, 1)
    concurrency = config.concurrency if config else DEFAULT_CONCURRENCY
    watermark = max(
        int(concurrency * (config.dir_pipeline_watermark if config else DIR_PIPELINE_WATERMARK)),
        1,
    )
    chunk_size = _effective_chunk_row_count(requested_chunk_size, watermark)

    _sparse_kw = {}
    if config:
        _sparse_kw = dict(full_label_count=config.sparse_full_label_count,
                          gap_multiplier=config.sparse_gap_multiplier,
                          min_gap=config.sparse_min_gap,
                          max_gap=config.sparse_max_gap,
                          threshold=config.sparse_threshold,
                          planner_enabled=config.planner_enabled,
                          planner_metadata_only=config.planner_metadata_only,
                          planner_policy=config.planner_policy,
                          planner_boundary_threshold=config.planner_boundary_threshold,
                          planner_min_segment_size=config.planner_min_segment_size,
                          planner_max_anchor_gap=config.planner_max_anchor_gap,
                          planner_fallback_boundary_ratio=config.planner_fallback_boundary_ratio)

    stats_acc = StatsAccumulator(
        extension_specs=_resolved_extension_specs(config),
        compact_labeling_request_bytes=(
            getattr(config, "compact_labeling_request_bytes", COMPACT_LABELING_REQUEST_BYTES)
            if config is not None
            else COMPACT_LABELING_REQUEST_BYTES
        ),
    )
    chunk_gen = iter_row_sample_bundle_chunks_from_jsonl(input_path, chunk_size, limit=limit)
    plan_totals = {
        "rows_total": 0,
        "rows_skipped": 0,
        "rows_changed": 0,
        "rows_migrated": 0,
        "rows_pass2_invalidated": 0,
        "preserved_samples": 0,
    }

    # Open output file handles
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_output_path is None:
        dataset_output_path = output_dir / input_path.name
    final_dataset_path = Path(dataset_output_path)
    final_labeled_path = output_dir / "labeled.jsonl"
    final_monitor_path = output_dir / "monitor.jsonl"
    final_failed_path = output_dir / "failed_samples.jsonl"

    use_working_sidecars = not inline_output
    if use_working_sidecars:
        dataset_write_path = _pass1_chunked_working_path(final_dataset_path)
        labeled_write_path = _pass1_chunked_working_path(final_labeled_path)
        monitor_write_path = _pass1_chunked_working_path(final_monitor_path)
        failed_write_path = _pass1_chunked_working_path(final_failed_path)
    else:
        dataset_write_path = final_dataset_path
        labeled_write_path = final_labeled_path
        monitor_write_path = final_monitor_path
        failed_write_path = final_failed_path

    dataset_write_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_write_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_write_path.parent.mkdir(parents=True, exist_ok=True)
    failed_write_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows = open(dataset_write_path, "w", encoding="utf-8")
    out_labeled = open(labeled_write_path, "w", encoding="utf-8")
    out_monitor = open(monitor_write_path, "w", encoding="utf-8")
    out_failed = open(failed_write_path, "w", encoding="utf-8")
    out_unmapped = open(output_dir / "unmapped_events.jsonl", "w", encoding="utf-8")

    try:
        # --- Helper: wrap label_one with chunk tracking ---
        async def _tagged_label(coro, chunk_idx, sample_idx):
            _, labels, monitor = await coro
            return chunk_idx, sample_idx, labels, monitor

        # --- Load a chunk and submit its tasks ---
        chunks_loaded = 0
        global_sample_offset = 0

        def load_next_chunk(pending_futures):
            nonlocal chunks_loaded, global_sample_offset
            try:
                row_bundles = next(chunk_gen)
            except StopIteration:
                return None

            prepared = prepare_inline_pass1_batch(
                row_bundles,
                mode=mode,
                label_version=DEFAULT_LABEL_VERSION,
                sparse_kwargs=_sparse_kw,
                extension_specs=_resolved_extension_specs(config),
                migration_index=migration_index,
            )
            samples = prepared.samples
            sample_to_bundle = prepared.sample_to_bundle

            # Assign global IDs
            for i, s in enumerate(samples):
                if not s.get("id"):
                    s["id"] = f"sample-{global_sample_offset + i:06d}"

            label_indices = prepared.label_indices
            inherit_map = prepared.inherit_map
            label_count = len(label_indices)
            sparse_inherited = len(inherit_map)

            chunk_idx = chunks_loaded
            chunks_loaded_display = chunks_loaded + 1

            collector = ChunkCollector(
                chunk_idx=chunk_idx,
                samples=samples,
                row_bundles=prepared.bundles,
                sample_to_bundle=sample_to_bundle,
                label_count=label_count,
                inherit_map=inherit_map,
                updated_sample_indices=prepared.updated_sample_indices,
                bundle_plans=prepared.bundle_plans,
                plan_stats=prepared.stats,
                run_mode=mode,
                sample_id_offset=global_sample_offset,
            )
            collector.labels = list(prepared.labels)
            for key in plan_totals:
                plan_totals[key] += prepared.stats.get(key, 0)

            global_sample_offset += len(samples)

            if sparse_inherited > 0:
                pprint(f"  [chunk {chunks_loaded_display}] {len(samples)} samples "
                       f"(sparse: {label_count} labeled + {sparse_inherited} inherited)")
            else:
                pprint(f"  [chunk {chunks_loaded_display}] {len(samples)} samples")

            # Update progress bar
            if progress and sample_task is not None:
                current_total = progress.tasks[sample_task].total or 0
                progress.update(sample_task, total=current_total + label_count, visible=True)

            submit_order = list(label_indices)
            random.shuffle(submit_order)
            collector.submit_order = submit_order
            collector.submit_cursor = 0

            chunks_loaded += 1  # noqa: F841 (nonlocal assignment)
            return collector

        # --- Try to load more chunks if below watermark ---
        def maybe_load_more(pending_futures, collectors):
            active_count = sum(1 for c in collectors.values() if not c.completed)
            loaded = []
            while (len(pending_futures) < watermark
                   and active_count < max_active):
                c = load_next_chunk(pending_futures)
                if c is None:
                    break
                collectors[c.chunk_idx] = c
                loaded.append(c)
                active_count += 1
            return loaded

        def top_up_submissions(pending_futures, collectors):
            if len(pending_futures) >= watermark:
                return
            active = [
                c for c in collectors.values()
                if not c.completed and c.has_pending_submission()
            ]
            while len(pending_futures) < watermark and active:
                submitted = 0
                for c in active:
                    if len(pending_futures) >= watermark:
                        break
                    if not c.has_pending_submission():
                        continue
                    idx = c.submit_order[c.submit_cursor]
                    c.submit_cursor += 1
                    coro = label_one(
                        http_client,
                        c.samples[idx],
                        model,
                        idx,
                        len(c.samples),
                        sem,
                        enable_arbitration=enable_arbitration,
                        config=config,
                        rate_limiter=rate_limiter,
                        llm_progress_cb=llm_progress_cb,
                        progress_event_cb=progress_event_cb,
                    )
                    fut = asyncio.ensure_future(_tagged_label(coro, c.chunk_idx, idx))
                    pending_futures.add(fut)
                    submitted += 1
                if submitted == 0:
                    break
                active = [
                    c for c in collectors.values()
                    if not c.completed and c.has_pending_submission()
                ]

        # --- Main watermark-driven loop ---
        collectors = {}
        pending_futures = set()
        next_chunk_to_flush = 0

        async def flush_ready_chunks():
            nonlocal next_chunk_to_flush
            while True:
                c = collectors.get(next_chunk_to_flush)
                if c is None or c.completed:
                    break
                if c.done < c.label_count:
                    break
                # Retry infra failures once, within the chunk boundary, so we can
                # merge/rewrite outputs deterministically.
                if config and getattr(config, "enable_stage_recovery_sweep", False):
                    inherited = set(c.inherit_map.keys()) if c.inherit_map else set()
                    candidate = [
                        i for i in range(len(c.labels))
                        if i not in inherited and (c.labels[i] is None or is_partial_labels(c.labels[i]))
                    ]
                    recovered = await _run_pass1_recovery_sweep(
                        samples=c.samples,
                        all_labels=c.labels,
                        all_monitors=c.monitors,
                        candidate_indices=candidate,
                        http_client=http_client,
                        model=model,
                        config=config,
                        enable_arbitration=enable_arbitration,
                    )
                    if recovered:
                        c.ok = sum(1 for i in range(len(c.labels)) if is_usable_labels(c.labels[i]))
                        c.fail = sum(
                            1 for i in range(len(c.labels))
                            if (c.labels[i] is None or is_partial_labels(c.labels[i])) and i not in inherited
                        )
                _flush_chunk(c, out_rows, out_labeled, out_monitor, out_failed, out_unmapped,
                             stats_acc, input_path, pprint=pprint,
                             run_failure_log_path=run_failure_log_path)
                pprint(f"  [chunk {c.chunk_idx + 1}] ✓ {c.ok} ok, {c.fail} fail — flushed")
                next_chunk_to_flush += 1

        if progress and sample_task is not None:
            progress.update(sample_task, total=0, completed=0, visible=True, info="loading...")

        # Initial load
        maybe_load_more(pending_futures, collectors)
        top_up_submissions(pending_futures, collectors)
        await flush_ready_chunks()

        file_start = time.time()

        while pending_futures:
            done, pending_futures = await asyncio.wait(
                pending_futures, return_when=asyncio.FIRST_COMPLETED)

            for fut in done:
                chunk_idx, sample_idx, labels, monitor = fut.result()
                c = collectors[chunk_idx]

                if 0 <= sample_idx < len(c.labels):
                    c.labels[sample_idx] = labels
                    c.monitors[sample_idx] = monitor

                c.done += 1
                if labels:
                    c.ok += 1
                else:
                    c.fail += 1

                if progress and sample_task is not None:
                    info = format_progress_info(
                        c.ok,
                        c.fail,
                        label=f"chunk {c.chunk_idx + 1}",
                        request_stats=rate_limiter.stats if rate_limiter else None,
                    )
                    if llm_progress_cb and monitor:
                        run_info = llm_progress_cb(0, "pass1")
                        if run_info:
                            info = f"{info} • {run_info}"
                    progress.update(sample_task, advance=1, info=info)
                if progress_event_cb:
                    progress_event_cb({
                        "kind": "sample_complete",
                        "chunk_idx": c.chunk_idx,
                        "sample_idx": sample_idx,
                        "labels": labels,
                        "monitor": monitor,
                    })

            await flush_ready_chunks()
            # After processing batch, check if we should load more chunks
            maybe_load_more(pending_futures, collectors)
            top_up_submissions(pending_futures, collectors)
            await flush_ready_chunks()

        await flush_ready_chunks()

        file_elapsed = time.time() - file_start

    finally:
        out_rows.close()
        out_labeled.close()
        out_monitor.close()
        out_failed.close()
        out_unmapped.close()

    if use_working_sidecars:
        _finalize_pass1_chunked_working_files(
            (
                (final_dataset_path, dataset_write_path),
                (final_labeled_path, labeled_write_path),
                (final_monitor_path, monitor_write_path),
                (final_failed_path, failed_write_path),
            )
        )

    # Remove empty failed file
    failed_path = final_failed_path
    if failed_path.exists() and failed_path.stat().st_size == 0:
        failed_path.unlink()

    # Finalize and write stats
    stats = stats_acc.finalize()
    stats["total_elapsed_seconds"] = round(file_elapsed, 1)
    stats["input_file"] = str(input_path)
    stats["mode"] = mode
    stats["chunked"] = True
    stats["chunk_size"] = chunk_size
    stats["chunks_processed"] = chunks_loaded
    stats.update(plan_totals)
    _annotate_labeling_prompt_stats(stats, config)

    stats_path = output_dir / PASS1_STATS_FILE
    _write_json_atomic(stats_path, stats)
    _safe_write_pass1_conversation_stats(
        output_dir,
        labeled_path=final_labeled_path,
        stats=stats,
        pprint=pprint,
    )

    # Dashboard (stats-only mode — no labeled.json)
    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        generate_dashboard(output_dir, labeled_file=None,
                           stats_file=PASS1_STATS_FILE, output_file=pass1_dashboard_filename())
    except Exception:
        pass

    pprint(f"  ✓ {stats['success']}/{stats['total_samples']} success, "
           f"{file_elapsed:.1f}s, {stats['total_tokens']:,} tokens")

    return stats


async def run_one_file(input_path, output_dir, http_client, sem, model,
                       enable_arbitration=True, limit=0, shuffle=False,
                       file_prefix=None, progress=None, sample_task=None,
                       config=None, rate_limiter=None, llm_progress_cb=None,
                       dataset_output_path=None, run_failure_log_path=None,
                       inline_output=False, mode: str = "refresh",
                       migration_index: dict | None = None):
    """Label a single file. Writes outputs to output_dir. Returns stats dict.

    file_prefix: if set, output files are named e.g. labeled_<prefix>.json
                 instead of labeled.json (avoids name collisions in batch mode).
    """
    # Dispatch to chunked pipeline for JSONL files (memory-bounded)
    if str(input_path).endswith(".jsonl") and not file_prefix:
        return await _run_one_file_chunked(
            input_path, output_dir, http_client, sem, model,
            enable_arbitration=enable_arbitration, limit=limit, shuffle=shuffle,
            progress=progress, sample_task=sample_task, config=config,
            rate_limiter=rate_limiter, llm_progress_cb=llm_progress_cb,
            dataset_output_path=dataset_output_path,
            run_failure_log_path=run_failure_log_path,
            inline_output=inline_output,
            mode=mode,
            migration_index=migration_index,
        )

    # Load input — streaming for JSONL
    row_bundles = None
    sample_to_bundle = None
    prepared_batch = None
    _sparse_kw = {}
    if config:
        _sparse_kw = dict(full_label_count=config.sparse_full_label_count,
                          gap_multiplier=config.sparse_gap_multiplier,
                          min_gap=config.sparse_min_gap,
                          max_gap=config.sparse_max_gap,
                          threshold=config.sparse_threshold,
                          planner_enabled=config.planner_enabled,
                          planner_metadata_only=config.planner_metadata_only,
                          planner_policy=config.planner_policy,
                          planner_boundary_threshold=config.planner_boundary_threshold,
                          planner_min_segment_size=config.planner_min_segment_size,
                          planner_max_anchor_gap=config.planner_max_anchor_gap,
                          planner_fallback_boundary_ratio=config.planner_fallback_boundary_ratio)
    if inline_output and str(input_path).endswith(".jsonl"):
        raw_row_bundles = list(iter_row_sample_bundles_from_jsonl(input_path, limit=limit))
        prepared_batch = prepare_inline_pass1_batch(
            raw_row_bundles,
            mode=mode,
            label_version=DEFAULT_LABEL_VERSION,
            sparse_kwargs=_sparse_kw,
            extension_specs=_resolved_extension_specs(config),
            migration_index=migration_index,
        )
        samples = prepared_batch.samples
        n_raw = len(prepared_batch.bundles)
        row_bundles = prepared_batch.bundles
        sample_to_bundle = prepared_batch.sample_to_bundle
    else:
        samples, n_raw = iter_samples_from_file(input_path, limit=limit, shuffle=shuffle)

    total = len(samples)
    pprint = progress.console.print if progress else print

    # Sparse sampling: only label a subset of multi-turn slices
    if prepared_batch is not None:
        label_indices = list(prepared_batch.label_indices)
        inherit_map = dict(prepared_batch.inherit_map)
    else:
        label_indices, inherit_map = apply_sparse_sampling(samples, **_sparse_kw)
    label_count = len(label_indices)
    sparse_inherited = len(inherit_map)
    if sparse_inherited > 0:
        pprint(f"  ({n_raw} conversations → {total} samples, sparse: {label_count} labeled + {sparse_inherited} inherited)")
    else:
        pprint(f"  ({n_raw} conversations → {total} samples)")

    # Set up progress bar task — reset timer for this file (track actual labels, not total)
    sparse_info = f" ({total} total, {round(sparse_inherited/total*100)}% sparse)" if sparse_inherited > 0 else ""
    if progress and sample_task is not None:
        progress.reset(sample_task, total=label_count, completed=0, visible=True, info="starting..." + sparse_info)

    # Pre-allocate result slots
    if prepared_batch is not None:
        all_labels = list(prepared_batch.labels)
    else:
        all_labels = [None] * total
    all_monitors = [None] * total

    concurrency = config.concurrency if config else DEFAULT_CONCURRENCY
    submit_window = max(
        int(concurrency * (config.dir_pipeline_watermark if config else DIR_PIPELINE_WATERMARK)),
        1,
    )

    # Submit tasks in shuffled order — only for indices that need labeling
    submit_order = list(label_indices)
    random.shuffle(submit_order)
    submit_cursor = 0
    pending_tasks = set()

    def top_up_submissions():
        nonlocal submit_cursor
        while submit_cursor < len(submit_order) and len(pending_tasks) < submit_window:
            idx = submit_order[submit_cursor]
            submit_cursor += 1
            pending_tasks.add(asyncio.ensure_future(label_one(
                http_client, samples[idx], model, idx, total, sem,
                enable_arbitration=enable_arbitration,
                config=config, rate_limiter=rate_limiter,
                llm_progress_cb=llm_progress_cb,
            )))

    done_count = 0
    ok_count = 0
    fail_count = 0
    file_start = time.time()
    top_up_submissions()
    while pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            sample_idx, labels, monitor = await task

            all_labels[sample_idx] = labels
            all_monitors[sample_idx] = monitor
            done_count += 1

            if labels:
                ok_count += 1
            else:
                fail_count += 1

            if progress and sample_task is not None:
                info = format_progress_info(
                    ok_count,
                    fail_count,
                    label=sparse_info.strip() if sparse_info else None,
                    request_stats=rate_limiter.stats if rate_limiter else None,
                )
                if llm_progress_cb and monitor:
                    run_info = llm_progress_cb(0, "pass1")
                    if run_info:
                        info = f"{info} • {run_info}"
                progress.update(sample_task, advance=1, info=info)
            else:
                # Fallback: per-sample print (no progress bar)
                sid = monitor["sample_id"]
                calls = monitor["llm_calls"]
                elapsed = monitor.get("elapsed_seconds", 0)
                status = monitor["status"]

                if labels:
                    intent = labels.get("intent", "?")
                    diff = labels.get("difficulty", "?")
                    langs = ",".join(labels.get("language", [])[:3])
                    n_tags = sum(
                        (len(labels[d]) if isinstance(labels.get(d), list) else (1 if labels.get(d) else 0))
                        for d in ["intent", "language", "domain", "concept", "task", "constraint", "agentic", "context", "difficulty"]
                    )
                    arb = " [ARB]" if monitor["arbitrated"] else ""
                    print(f"  [{done_count:4d}/{total}] {sid:20s} | {calls} calls {elapsed:5.1f}s | {intent:6s} {diff:12s} | {langs:20s} | {n_tags:2d} tags{arb}")
                else:
                    print(f"  [{done_count:4d}/{total}] {sid:20s} | {calls} calls {elapsed:5.1f}s | FAILED: {status}")
        top_up_submissions()

    file_elapsed = time.time() - file_start

    # One conservative sweep over infra-retryable failures (before inheritance/output).
    if config and getattr(config, "enable_stage_recovery_sweep", False):
        inherited_indices = set(inherit_map.keys())
        candidate = [
            i for i in label_indices
            if i not in inherited_indices and (all_labels[i] is None or is_partial_labels(all_labels[i]))
        ]
        if candidate:
            recovered = await _run_pass1_recovery_sweep(
                samples=samples,
                all_labels=all_labels,
                all_monitors=all_monitors,
                candidate_indices=candidate,
                http_client=http_client,
                model=model,
                config=config,
                enable_arbitration=enable_arbitration,
            )
            if recovered:
                pprint(f"  Recovery sweep: recovered {recovered}/{len(candidate)} samples")

    resolved_labels = apply_inherited_labels(samples, all_labels, inherit_map)
    all_labels[:] = resolved_labels

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if inline_output else (f"_{file_prefix}" if file_prefix else "")

    labeled_json = f"labeled{suffix}.json"
    labeled_jsonl = f"labeled{suffix}.jsonl"
    monitor_file = f"monitor{suffix}.jsonl"
    stats_file = pass1_stats_filename(suffix)
    dashboard_file = pass1_dashboard_filename(suffix)
    failed_samples_file = f"failed_samples{suffix}.jsonl"
    unmapped_events_file = f"unmapped_events{suffix}.jsonl"

    if inline_output:
        merge_result = merge_pass1_results(
            row_bundles or [],
            samples,
            resolved_labels,
            all_monitors,
            sample_to_bundle or [],
            source_file=input_path,
            mode=mode,
            bundle_plans=prepared_batch.bundle_plans if prepared_batch else None,
            updated_sample_indices=prepared_batch.updated_sample_indices if prepared_batch else None,
        )
        _write_jsonl_atomic(dataset_output_path, merge_result.rows)
        _write_json_atomic(output_dir / labeled_json, merge_result.samples)
        _write_jsonl_atomic(output_dir / labeled_jsonl, merge_result.samples)
        _write_jsonl_atomic(output_dir / monitor_file, merge_result.monitor_records)
        _write_jsonl_atomic(output_dir / unmapped_events_file, build_unmapped_event_records(merge_result.samples))
        labeled_samples_for_dashboard = merge_result.samples
        if merge_result.failed_samples:
            _write_jsonl_atomic(output_dir / failed_samples_file, merge_result.failed_samples)
    else:
        rendered_samples, monitor_records, failed_samples = build_sample_artifacts(
            samples,
            resolved_labels,
            all_monitors,
            source_file=input_path,
        )
        _write_json_atomic(output_dir / labeled_json, rendered_samples)
        _write_jsonl_atomic(output_dir / labeled_jsonl, rendered_samples)
        _write_jsonl_atomic(output_dir / monitor_file, monitor_records)
        _write_jsonl_atomic(output_dir / unmapped_events_file, build_unmapped_event_records(rendered_samples))
        labeled_samples_for_dashboard = rendered_samples
        if failed_samples:
            _write_jsonl_atomic(output_dir / failed_samples_file, failed_samples)

    # Write failed samples (original, without labels) for easy retry
    # Inherited samples have no monitor — they are not failures
    inherited_indices = set(inherit_map.keys())
    failed_indices = [
        i for i, labels in enumerate(all_labels)
        if (labels is None or is_partial_labels(labels)) and i not in inherited_indices
    ]
    if failed_indices:
        with open(output_dir / failed_samples_file, "w", encoding="utf-8") as f:
            for i in failed_indices:
                s = dict(samples[i])
                s.pop("labels", None)
                s.pop("labeling_monitor", None)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write failure log
    if failed_indices:
        failure_log_path = Path(run_failure_log_path) if run_failure_log_path else (output_dir / "failures.jsonl")
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log_path, "w", encoding="utf-8") as f:
            for i in failed_indices:
                m = all_monitors[i]
                record = {
                    "sample_id": samples[i].get("id", f"sample-{i}"),
                    "source_file": str(input_path),
                    "status": m["status"] if m else "timeout",
                    "error": (m.get("error", "") if m else "exceeded sample timeout"),
                    "error_response": (m.get("error_response", "")[:1000] if m else ""),
                    "attempts": (m.get("sample_attempt", 0) + 1 if m else 0),
                    "error_class": (m.get("error_class") if m else None),
                    "retryable_infra": (m.get("retryable_infra") if m else None),
                    "http_status": (m.get("http_status") if m else None),
                    "runtime_state": (m.get("runtime_state") if m else None),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Compute and write stats
    valid_monitors = [m for m in all_monitors if m is not None]
    stats = compute_stats(
        valid_monitors,
        all_labels,
        inherit_map=inherit_map,
        extension_specs=_resolved_extension_specs(config),
    )
    stats["total_elapsed_seconds"] = round(file_elapsed, 1)
    stats["input_file"] = str(input_path)
    stats["mode"] = mode
    if prepared_batch is not None:
        stats.update(prepared_batch.stats)
    if sparse_inherited > 0:
        stats["sparse_labeled"] = label_count
        stats["sparse_inherited"] = sparse_inherited
    _annotate_labeling_prompt_stats(stats, config)

    stats_path = output_dir / stats_file
    _write_json_atomic(stats_path, stats)
    _safe_write_pass1_conversation_stats(
        output_dir,
        labeled_samples=labeled_samples_for_dashboard,
        stats=stats,
        pprint=pprint,
    )

    # Per-file dashboard
    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        generate_dashboard(output_dir, labeled_file=labeled_json,
                           stats_file=stats_file, output_file=dashboard_file)
    except Exception:
        pass

    success = stats["success"]
    total_tokens = stats["total_tokens"]
    pprint(f"  ✓ {success}/{total} success, {file_elapsed:.1f}s, {total_tokens:,} tokens")

    # Print failure details for debugging — batch into single print to avoid progress bar flicker
    failed_count = len(failed_indices)
    if failed_count > 0:
        lines = []
        failed_monitors = [all_monitors[i] for i in failed_indices if all_monitors[i] and all_monitors[i]["status"] != "success"]
        # Group by error type
        error_groups = {}
        for m in failed_monitors:
            err_type = m["status"]
            error_groups.setdefault(err_type, []).append(m)
        lines.append(f"  ✗ {failed_count} failed:")
        for err_type, monitors in sorted(error_groups.items()):
            lines.append(f"    [{err_type}] ×{len(monitors)}")
            for m in monitors[:3]:  # show up to 3 per type
                sid = m.get("sample_id", "?")
                err = m.get("error", "")
                resp = m.get("error_response", "")
                attempts = m.get("sample_attempt", 0) + 1
                detail = err[:120]
                if resp and resp != err:
                    detail += f" | response: {resp[:80]}"
                lines.append(f"      {sid} (attempt {attempts}): {detail}")
            if len(monitors) > 3:
                lines.append(f"      ... and {len(monitors) - 3} more")
        # Count timeouts (samples with no monitor entry)
        timeout_count = sum(1 for i, m in enumerate(all_monitors) if m is None and i not in inherited_indices)
        if timeout_count > 0:
            lines.append(f"    [no_result] ×{timeout_count}")
        pprint("\n".join(lines))

    return stats


def _jsonl_exceeds_chunk_threshold(path: Path, *, chunk_size: int, limit: int = 0) -> bool:
    """Return True when a JSONL file likely benefits from chunked execution."""
    threshold = max(int(chunk_size), 1)
    if limit > 0:
        threshold = min(threshold, int(limit))
    if threshold <= 0:
        return False

    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if seen > threshold:
                return True
            if limit > 0 and seen >= limit:
                break
    return False


def _should_delegate_directory_jsonl_to_chunked(
    abs_path: Path,
    *,
    inline_jsonl: bool,
    config: PipelineConfig | None,
    limit: int,
) -> bool:
    if not inline_jsonl or not str(abs_path).endswith(".jsonl"):
        return False
    chunk_size = config.chunk_size if config else CHUNK_SIZE
    return _jsonl_exceeds_chunk_threshold(abs_path, chunk_size=chunk_size, limit=limit)


def _labeled_samples_from_stats(stats: dict) -> int:
    if "sparse_labeled" in stats:
        return int(stats.get("sparse_labeled", 0) or 0)
    total = int(stats.get("total_samples", 0) or 0)
    inherited = int(stats.get("sparse_inherited", 0) or 0)
    return max(total - inherited, 0)


async def run_directory_pipeline(dir_files, run_dir, model, concurrency,
                                 checkpoint_path, completed_set=None,
                                 limit=0, shuffle=False,
                                 progress=None, file_task=None, sample_task=None,
                                 http_client=None, sem=None, enable_arbitration=True,
                                 config=None, rate_limiter=None,
                                 llm_task=None, workload_estimate=None,
                                 llm_progress_cb=None, layout: InlineRunLayout | None = None,
                                 mode: str = "refresh",
                                 migration_index: dict | None = None):
    """Cross-file pipeline with watermark-based file loading.

    Instead of processing files serially, loads new files whenever in-flight
    task count drops below a watermark (concurrency * DIR_PIPELINE_WATERMARK).
    This keeps the semaphore saturated even when some files have long-tail
    samples in retry/backoff. Memory is bounded by DIR_PIPELINE_MAX_FILES.

    Returns list of per-file stats dicts.
    """
    watermark = int(concurrency * (config.dir_pipeline_watermark if config else DIR_PIPELINE_WATERMARK))
    max_active = config.dir_pipeline_max_files if config else DIR_PIPELINE_MAX_FILES
    completed_set = completed_set or set()
    pprint = progress.console.print if progress else print

    # Separate already-completed files (for resume) from pending ones
    pending_files = []
    skipped_stats = []
    for i, (abs_path, rel_path) in enumerate(dir_files):
        rel_str = str(rel_path)
        if rel_str in completed_set:
            # Load existing stats for summary
            file_out_dir = layout.file_artifact_dir(abs_path) if layout else (run_dir / rel_path.with_suffix(""))
            prefix = rel_path.stem
            existing_stats = file_out_dir / (PASS1_STATS_FILE if layout else f"stats_{prefix}.json")
            if existing_stats.exists():
                with open(existing_stats, "r", encoding="utf-8") as f:
                    skipped_stats.append(json.load(f))
            if progress and file_task is not None:
                progress.update(file_task, advance=1)
        else:
            pending_files.append((i, abs_path, rel_path))

    if not pending_files:
        return skipped_stats

    # --- Helper to wrap background tasks with source metadata ---
    async def _tagged_label(coro, file_idx, sample_idx):
        _, labels, monitor = await coro
        return {
            "kind": "sample",
            "file_idx": file_idx,
            "sample_idx": sample_idx,
            "labels": labels,
            "monitor": monitor,
        }

    async def _tagged_chunked_file(coro, file_idx):
        stats = await coro
        return {
            "kind": "chunked_file",
            "file_idx": file_idx,
            "stats": stats,
        }

    # --- Load a file into a FileCollector ---
    def load_and_submit(file_entry, pending_futures):
        orig_idx, abs_path, rel_path = file_entry
        str(rel_path)
        inline_jsonl = layout is not None and str(abs_path).endswith(".jsonl")
        file_out_dir = layout.file_artifact_dir(abs_path) if inline_jsonl else (run_dir / rel_path.with_suffix(""))
        dataset_output_path = layout.mirrored_dataset_path(abs_path) if inline_jsonl else None
        run_failure_log_path = layout.run_artifact_path("failures.jsonl") if layout else (run_dir / "failures.jsonl")
        prefix = rel_path.stem

        _sparse_kw = {}
        if config:
            _sparse_kw = dict(full_label_count=config.sparse_full_label_count,
                              gap_multiplier=config.sparse_gap_multiplier,
                              min_gap=config.sparse_min_gap,
                              max_gap=config.sparse_max_gap,
                              threshold=config.sparse_threshold,
                              planner_enabled=config.planner_enabled,
                              planner_metadata_only=config.planner_metadata_only,
                              planner_policy=config.planner_policy,
                              planner_boundary_threshold=config.planner_boundary_threshold,
                              planner_min_segment_size=config.planner_min_segment_size,
                              planner_max_anchor_gap=config.planner_max_anchor_gap,
                              planner_fallback_boundary_ratio=config.planner_fallback_boundary_ratio)

        use_chunked_delegate = _should_delegate_directory_jsonl_to_chunked(
            abs_path,
            inline_jsonl=inline_jsonl,
            config=config,
            limit=limit,
        )
        if use_chunked_delegate:
            def _chunked_progress_event(event):
                if not isinstance(event, dict):
                    return
                kind = event.get("kind")
                if kind == "llm_delta":
                    _record_live_llm_delta(event.get("delta_calls", 0), event.get("run_info"))
                    return
                if kind == "sample_complete" and progress and sample_task is not None:
                    if workload_estimate is None:
                        current_total = progress.tasks[sample_task].total or 0
                        progress.update(
                            sample_task,
                            total=current_total + 1,
                            advance=1,
                            info=f"{rel_path.name} (chunked)",
                        )
                    else:
                        progress.update(sample_task, advance=1, info=f"{rel_path.name} (chunked)")

            collector = FileCollector(
                file_idx=orig_idx,
                abs_path=abs_path,
                rel_path=rel_path,
                output_dir=file_out_dir,
                prefix=rel_path.stem,
                total=0,
                samples=[],
                row_bundles=None,
                sample_to_bundle=None,
                dataset_output_path=dataset_output_path,
                run_failure_log_path=run_failure_log_path,
                config=config,
                inline_output=inline_jsonl,
                label_count=0,
                inherit_map={},
                updated_sample_indices=set(),
                bundle_plans=None,
                plan_stats={},
                run_mode=mode,
                sparse_info="",
                chunked_delegate=True,
                live_progress=True,
            )
            fut = asyncio.ensure_future(
                _tagged_chunked_file(
                    _run_one_file_chunked(
                        abs_path,
                        file_out_dir,
                        http_client,
                        sem,
                        model,
                        enable_arbitration=enable_arbitration,
                        limit=limit,
                        shuffle=shuffle,
                        progress=None,
                        sample_task=None,
                        config=config,
                        rate_limiter=rate_limiter,
                        llm_progress_cb=llm_progress_cb,
                        dataset_output_path=dataset_output_path,
                        run_failure_log_path=run_failure_log_path,
                        inline_output=inline_jsonl,
                        mode=mode,
                        migration_index=migration_index,
                        progress_event_cb=_chunked_progress_event,
                        directory_delegate=True,
                    ),
                    orig_idx,
                )
            )
            pending_futures.add(fut)
            return collector

        row_bundles = None
        sample_to_bundle = None
        prepared = None
        if inline_jsonl:
            raw_row_bundles = list(iter_row_sample_bundles_from_jsonl(abs_path, limit=limit))
            prepared = prepare_inline_pass1_batch(
                raw_row_bundles,
                mode=mode,
                label_version=DEFAULT_LABEL_VERSION,
                sparse_kwargs=_sparse_kw,
                extension_specs=_resolved_extension_specs(config),
                migration_index=migration_index,
            )
            samples = prepared.samples
            row_bundles = prepared.bundles
            sample_to_bundle = prepared.sample_to_bundle
        else:
            samples, _ = iter_samples_from_file(
                abs_path, limit=limit, shuffle=shuffle)

        # Sparse sampling
        if prepared is not None:
            label_indices = list(prepared.label_indices)
            inherit_map = dict(prepared.inherit_map)
        else:
            label_indices, inherit_map = apply_sparse_sampling(samples, **_sparse_kw)
        label_count = len(label_indices)
        sparse_inherited = len(inherit_map)

        collector = FileCollector(
            file_idx=orig_idx,
            abs_path=abs_path,
            rel_path=rel_path,
            output_dir=file_out_dir,
            prefix=prefix,
            total=len(samples),
            samples=samples,
            row_bundles=row_bundles,
            sample_to_bundle=sample_to_bundle,
            dataset_output_path=dataset_output_path,
            run_failure_log_path=run_failure_log_path,
            config=config,
            inline_output=inline_jsonl,
            label_count=label_count,
            inherit_map=inherit_map,
            updated_sample_indices=prepared.updated_sample_indices if prepared else set(),
            bundle_plans=prepared.bundle_plans if prepared else None,
            plan_stats=prepared.stats if prepared else {},
            run_mode=mode,
            sparse_info=(
                f" ({len(samples)} total, {round(sparse_inherited/len(samples)*100)}% sparse)"
                if sparse_inherited > 0 and len(samples) > 0
                else ""
            ),
        )
        if prepared is not None:
            collector.labels = list(prepared.labels)

        # Fallback behavior: if no pre-scan estimate, grow total dynamically.
        if progress and sample_task is not None and workload_estimate is None:
            current_total = progress.tasks[sample_task].total or 0
            progress.update(sample_task, total=current_total + label_count, visible=True)

        submit_order = list(label_indices)
        random.shuffle(submit_order)
        collector.submit_order = submit_order
        collector.submit_cursor = 0

        return collector

    def _collector_active_work(c: FileCollector) -> int:
        if c.completed:
            return 0
        if c.chunked_delegate:
            # Large delegated JSONL files internally top up to the same request
            # watermark as normal directory scheduling. Treat each as a full
            # watermark unit so the outer scheduler admits them conservatively.
            return max(watermark, 1)
        return max(c.label_count - c.done, 0)

    # --- Try to load more files if active work is below watermark ---
    def maybe_load_more(pending_futures, collectors, file_queue, next_to_load):
        active_count = sum(1 for c in collectors.values() if not c.completed)
        active_work = sum(_collector_active_work(c) for c in collectors.values())
        while (
            next_to_load < len(file_queue)
            and active_count < max_active
            and active_work < watermark
        ):
            entry = file_queue[next_to_load]
            next_to_load += 1
            new_c = load_and_submit(entry, pending_futures)
            collectors[new_c.file_idx] = new_c
            active_count += 1
            active_work += _collector_active_work(new_c)
        return next_to_load

    def top_up_submissions(pending_futures, collectors):
        if len(pending_futures) >= watermark:
            return
        active = [
            c for c in collectors.values()
            if not c.completed and not c.chunked_delegate and c.has_pending_submission()
        ]
        while len(pending_futures) < watermark and active:
            submitted = 0
            for c in active:
                if len(pending_futures) >= watermark:
                    break
                if not c.has_pending_submission():
                    continue
                idx = c.submit_order[c.submit_cursor]
                c.submit_cursor += 1
                coro = label_one(
                    http_client,
                    c.samples[idx],
                    model,
                    idx,
                    c.total,
                    sem,
                    enable_arbitration=enable_arbitration,
                    config=config,
                    rate_limiter=rate_limiter,
                    llm_progress_cb=llm_progress_cb,
                )
                fut = asyncio.ensure_future(_tagged_label(coro, c.file_idx, idx))
                pending_futures.add(fut)
                submitted += 1
            if submitted == 0:
                break
            active = [
                c for c in collectors.values()
                if not c.completed and not c.chunked_delegate and c.has_pending_submission()
            ]

    # --- Main loop: watermark-driven ---
    collectors = {}  # file_idx -> FileCollector
    pending_futures = set()
    all_file_stats = list(skipped_stats)
    file_queue = list(pending_files)
    next_to_load = 0
    eta_tracker = None
    if workload_estimate and workload_estimate.total_labeled_samples > 0:
        eta_tracker = RuntimeEtaEstimator(
            total_labeled_samples=workload_estimate.total_labeled_samples,
            initial_estimated_calls=workload_estimate.initial_estimated_llm_calls,
        )
    global_llm_info = None

    def _record_live_llm_delta(delta_calls: int, run_info: str | None):
        nonlocal global_llm_info
        delta = max(int(delta_calls or 0), 0)
        if eta_tracker and delta > 0:
            eta_tracker.calls_done += delta
            eta_tracker.estimated_total_calls = max(eta_tracker.estimated_total_calls, eta_tracker.calls_done)
            elapsed = max(time.time() - eta_tracker._start_time, 1e-6)
            instant_cps = eta_tracker.calls_done / elapsed
            if eta_tracker._smoothed_cps <= 0:
                eta_tracker._smoothed_cps = instant_cps
            else:
                eta_tracker._smoothed_cps = 0.2 * instant_cps + 0.8 * eta_tracker._smoothed_cps
            eta_tracker.calls_per_sec = eta_tracker._smoothed_cps
        if run_info:
            global_llm_info = run_info
        _update_llm_task_progress(progress, llm_task, global_llm_info, eta_tracker=eta_tracker)

    if progress and sample_task is not None:
        initial_sample_total = (
            workload_estimate.total_labeled_samples if workload_estimate else 0
        )
        progress.update(sample_task, total=initial_sample_total, completed=0, visible=True, info="starting...")
    if progress and llm_task is not None:
        if eta_tracker:
            progress.update(
                llm_task,
                total=max(eta_tracker.estimated_total_calls, 1),
                completed=0,
                visible=True,
                info=eta_tracker.info_line(),
            )
        else:
            progress.update(llm_task, visible=False)

    # Initial load
    next_to_load = maybe_load_more(pending_futures, collectors, file_queue, next_to_load)
    top_up_submissions(pending_futures, collectors)

    if progress and file_task is not None:
        active_names = [str(c.rel_path) for c in collectors.values() if not c.completed]
        progress.update(file_task, info=", ".join(active_names)[:60])

    # Process results as they complete
    while pending_futures:
        done, pending_futures = await asyncio.wait(
            pending_futures, return_when=asyncio.FIRST_COMPLETED)

        for fut in done:
            result = fut.result()
            result_kind = result.get("kind")
            if result_kind == "sample":
                file_idx = result["file_idx"]
                sample_idx = result["sample_idx"]
                labels = result["labels"]
                monitor = result["monitor"]
                c = collectors[file_idx]

                if 0 <= sample_idx < c.total:
                    c.labels[sample_idx] = labels
                    c.monitors[sample_idx] = monitor

                c.done += 1
                if labels:
                    c.ok += 1
                else:
                    c.fail += 1

                # Update samples progress bar
                if progress and sample_task is not None:
                    label = c.rel_path.name
                    if c.sparse_info:
                        label += c.sparse_info
                    info = format_progress_info(
                        c.ok,
                        c.fail,
                        label=label,
                        request_stats=rate_limiter.stats if rate_limiter else None,
                    )
                    if llm_progress_cb and monitor:
                        global_llm_info = llm_progress_cb(0, "pass1")
                    progress.update(sample_task, advance=1, info=info)
                if eta_tracker:
                    eta_tracker.samples_done += 1
                    if not llm_progress_cb:
                        eta_tracker.update(monitor.get("llm_calls", 0) if monitor else 0)
                    _update_llm_task_progress(progress, llm_task, global_llm_info, eta_tracker=eta_tracker)

                # Check if this file is fully done (compare against label_count, not total)
                if c.done >= c.label_count and not c.completed:
                    stats = flush_file_output(
                        c,
                        run_dir,
                        checkpoint_path,
                        pprint=pprint,
                        generate_dashboard=False,
                    )
                    all_file_stats.append(stats)

                    if progress and file_task is not None:
                        progress.update(file_task, advance=1)
                    collectors.pop(file_idx, None)
            elif result_kind == "chunked_file":
                file_idx = result["file_idx"]
                stats = result["stats"]
                c = collectors.get(file_idx)
                if c is None:
                    continue

                c.samples = None
                c.labels = None
                c.monitors = None
                c.row_bundles = None
                c.sample_to_bundle = None
                c.bundle_plans = None
                c.plan_stats = {}
                c.completed = True

                update_checkpoint(checkpoint_path, str(c.rel_path), success=True)
                all_file_stats.append(stats)

                labeled_done = _labeled_samples_from_stats(stats)
                if progress and sample_task is not None and labeled_done > 0 and not c.live_progress:
                    if workload_estimate is None:
                        current_total = progress.tasks[sample_task].total or 0
                        progress.update(
                            sample_task,
                            total=current_total + labeled_done,
                            advance=labeled_done,
                            info=f"{c.rel_path.name} (chunked)",
                        )
                    else:
                        progress.update(sample_task, advance=labeled_done, info=f"{c.rel_path.name} (chunked)")
                if eta_tracker and not c.live_progress:
                    calls_done = int(stats.get("total_llm_calls", 0) or 0)
                    eta_tracker.calls_done += max(calls_done, 0)
                    eta_tracker.samples_done += max(labeled_done, 0)
                    if eta_tracker.samples_done > 0:
                        eta_tracker.avg_calls_per_sample = eta_tracker.calls_done / eta_tracker.samples_done
                    eta_tracker.estimated_total_calls = max(eta_tracker.estimated_total_calls, eta_tracker.calls_done)
                    elapsed = max(time.time() - eta_tracker._start_time, 1e-6)
                    instant_cps = eta_tracker.calls_done / elapsed
                    if eta_tracker._smoothed_cps <= 0:
                        eta_tracker._smoothed_cps = instant_cps
                    else:
                        eta_tracker._smoothed_cps = 0.2 * instant_cps + 0.8 * eta_tracker._smoothed_cps
                    eta_tracker.calls_per_sec = eta_tracker._smoothed_cps
                    _update_llm_task_progress(progress, llm_task, global_llm_info, eta_tracker=eta_tracker)
                if progress and file_task is not None:
                    progress.update(file_task, advance=1)

                collectors.pop(file_idx, None)

        # After processing batch of completions, check if we should load more files
        next_to_load = maybe_load_more(pending_futures, collectors, file_queue, next_to_load)
        top_up_submissions(pending_futures, collectors)

        if progress and file_task is not None:
            active_names = [str(cc.rel_path) for cc in collectors.values()
                            if not cc.completed]
            progress.update(file_task, info=", ".join(active_names)[:60] if active_names else "done")

    # Handle any files with 0 labels to submit (edge case: 0 samples or all inherited)
    for file_idx, c in list(collectors.items()):
        if not c.completed and c.label_count == 0:
            stats = flush_file_output(
                c,
                run_dir,
                checkpoint_path,
                pprint=pprint,
                generate_dashboard=False,
            )
            all_file_stats.append(stats)
            if progress and file_task is not None:
                progress.update(file_task, advance=1)
            collectors.pop(file_idx, None)

    return all_file_stats


def print_summary(stats, run_dir, is_batch=False):
    """Print final summary to stdout."""
    print(f"\n{'='*80}")
    label = "BATCH LABELING COMPLETE" if is_batch else "LABELING COMPLETE"
    elapsed = stats.get('total_elapsed_seconds', 0)
    print(f"{label} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")
    if is_batch:
        print(f"Files:       {stats.get('files_processed', '?')}")
    if stats.get("mode"):
        print(f"Mode:        {stats['mode']}")
    print(f"Success:     {stats['success']}/{stats['total_samples']} ({stats.get('success_rate', 0)*100:.1f}%)")
    print(f"LLM calls:   {stats['total_llm_calls']} total, {stats.get('avg_calls_per_sample', 0):.1f} avg/sample")
    print(f"Tokens:      {stats['total_tokens']:,}")
    print(f"Arbitrated:  {stats['arbitrated_count']} ({stats.get('arbitrated_rate', 0)*100:.1f}%)")
    print(f"Unmapped:    {stats.get('unmapped_unique_count', 0)} unique out-of-pool tags")
    sparse_labeled = stats.get('sparse_labeled', 0)
    sparse_inherited = stats.get('sparse_inherited', 0)
    if sparse_inherited > 0:
        saving = round(sparse_inherited / (sparse_labeled + sparse_inherited) * 100)
        print(f"Sparse:      {sparse_labeled} labeled + {sparse_inherited} inherited ({saving}% saved)")
    total_samples = stats.get('total_samples', 0)
    if elapsed > 0 and total_samples > 0:
        print(f"Throughput:  {total_samples / elapsed:.1f} samples/sec")

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

    print("\nConfidence (mean):")
    for dim, cs in stats.get("confidence_stats", {}).items():
        bar = "█" * int(cs["mean"] * 20)
        print(f"  {dim:15s} {cs['mean']:.3f} {bar}")

    print("\nTop distributions:")
    for dim in ["intent", "difficulty", "domain", "concept"]:
        dist = stats.get("tag_distributions", {}).get(dim, {})
        top5 = list(dist.items())[:5]
        top_str = ", ".join(f"{k}({v})" for k, v in top5)
        print(f"  {dim:15s} {top_str}")

    if stats.get("unmapped_tags"):
        print("\nUnmapped tags (top 10):")
        for tag, count in list(stats["unmapped_tags"].items())[:10]:
            print(f"  {tag}: {count}")

    print(f"\nRun dir: {run_dir}")


def _write_global_summary(all_file_stats, run_dir, input_path, model, concurrency, batch_start,
                          rate_limiter=None, workload_estimate=None,
                          layout: InlineRunLayout | None = None,
                          mode: str = "refresh",
                          migrate_from: str | None = None,
                          migration_stats: dict | None = None,
                          config: PipelineConfig | None = None):
    """Write global summary stats + dashboard for a batch run."""
    batch_elapsed = time.time() - batch_start
    summary = merge_stats(all_file_stats) if all_file_stats else {
        "total_samples": 0, "distribution_total_samples": 0,
        "success": 0, "failed": 0, "success_rate": 0,
        "total_llm_calls": 0, "total_tokens": 0, "arbitrated_count": 0,
        "unmapped_unique_count": 0,
        "canonicalization_total_count": 0,
        "canonicalization_unique_count": 0,
        "tag_distributions": {},
        "unmapped_tags": {},
        "canonicalization_counts": {},
        "files_processed": 0,
    }
    summary["model"] = model
    summary["concurrency"] = concurrency
    summary["total_elapsed_seconds"] = round(batch_elapsed, 1)
    summary["timestamp"] = datetime.now().isoformat()
    summary["input_path"] = str(input_path)
    summary["run_dir"] = str(run_dir)
    summary["mode"] = mode
    if migrate_from:
        summary["migrate_from"] = str(migrate_from)
    if migration_stats:
        summary["migration_indexed_rows"] = migration_stats.get("indexed_rows", 0)
        summary["migration_unique_data_ids"] = migration_stats.get("unique_data_ids", 0)
        duplicate_data_ids = migration_stats.get("duplicate_data_ids", 0)
        if duplicate_data_ids:
            summary["migration_duplicate_data_ids"] = duplicate_data_ids
    if workload_estimate is not None:
        summary["planned_files"] = workload_estimate.files_planned
        summary["planned_raw_conversations"] = workload_estimate.total_raw_conversations
        summary["planned_samples"] = workload_estimate.total_samples
        summary["planned_labeled_samples"] = workload_estimate.total_labeled_samples
        summary["planned_inherited_samples"] = workload_estimate.total_inherited_samples
        summary["planned_baseline_llm_calls"] = workload_estimate.baseline_total_llm_calls
        summary["planned_initial_llm_calls"] = workload_estimate.initial_estimated_llm_calls
        summary["planning_elapsed_seconds"] = workload_estimate.scan_elapsed_seconds
    if rate_limiter:
        summary["http_request_stats"] = rate_limiter.stats.to_dict()
    _annotate_labeling_prompt_stats(summary, config)

    summary_path = _summary_path_for_run(run_dir, layout=layout)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        dashboard_source = layout.meta_root if layout is not None else run_dir
        dashboard_output = (
            layout.dashboard_path(PASS1_DASHBOARD_FILE) if layout is not None
            else (run_dir / "dashboards" / PASS1_DASHBOARD_FILE)
        )
        generated_output = run_with_heartbeat(
            "Generating labeling dashboard",
            lambda: generate_dashboard(
                dashboard_source,
                labeled_file=None,
                stats_file=PASS1_SUMMARY_STATS_FILE,
                output_file=str(dashboard_output),
                quiet=True,
            ),
        )
        if layout is None:
            prune_dashboard_bundles(
                run_dir,
                keep_paths=[dashboard_output],
                kind="labeling",
                recursive=True,
            )
        print(f"\nGlobal dashboard generated: {generated_output}")
    except Exception as e:
        print(f"\nGlobal dashboard generation skipped: {e}")

    print_summary(summary, run_dir, is_batch=True)


def _print_migration_warning(migration_stats: dict | None):
    """Warn when migration source contains ambiguous duplicate data_ids."""
    if not migration_stats:
        return
    duplicate_data_ids = int(migration_stats.get("duplicate_data_ids", 0) or 0)
    if duplicate_data_ids <= 0:
        return
    print(
        "  Warning | "
        f"migration source has {duplicate_data_ids} duplicate data_id rows; "
        "the first match wins for copy-forward."
    )


async def run(
    input_path: str | Path,
    *,
    output: str | Path | None = None,
    resume: str | Path | None = None,
    limit: int = 0,
    shuffle: bool = False,
    enable_arbitration: bool = True,
    config: PipelineConfig | None = None,
    mode: str = "refresh",
    migrate_from: str | Path | None = None,
    llm_progress_cb=None,
    precomputed_workload_estimate=None,
    parallelize_workload_estimation: bool = False,
) -> dict:
    """Library entry point for the labeling pipeline.

    Args:
        input_path: Path to input file or directory.
        output: Output directory. None = sibling of input, "runs" = labeling/data/runs/.
        resume: Path to an existing run directory to resume from.
        limit: Max samples per file (0 = all).
        shuffle: Randomly shuffle samples before processing.
        enable_arbitration: Enable low-confidence arbitration pass.
        config: Pipeline configuration. Defaults to PipelineConfig().

    Returns:
        Stats dict with labeling results.

    Raises:
        FileNotFoundError: If input/resume path doesn't exist or has no valid files.
        ValueError: If resume is used with non-directory input.
    """
    if config is None:
        config = PipelineConfig()
    _resolved_extension_specs(config)

    mode = _resolve_mode(mode)

    normalized_input = None
    inline_input_target = None
    if input_path is not None:
        normalized_input, inline_input_target = _normalize_run_input(input_path)

    if mode == "recompute":
        if resume:
            raise ValueError("--resume is not supported with --mode recompute")
        if normalized_input is None:
            raise ValueError("--input is required for --mode recompute")

        from sft_label.tools.recompute import (
            run_recompute,
            run_refresh_rarity,
            run_regenerate_dashboard,
        )

        written = run_recompute(normalized_input, pass_num="both")
        if any(key in written for key in ("stats_value", "summary_stats_value")):
            run_refresh_rarity(normalized_input, config=config)
        dashboards = run_regenerate_dashboard(normalized_input, pass_num="both")
        run_root = (
            inline_input_target.layout.run_root
            if inline_input_target is not None
            else Path(normalized_input).resolve()
        )
        return {
            "status": "recomputed",
            "mode": mode,
            "run_dir": str(run_root),
            "input_path": str(normalized_input),
            "written": written,
            "dashboards": dashboards,
        }

    # Create shared adaptive runtime (optional). When enabled, concurrency/rps
    # act as *maximums* and the runtime will lower pressure on instability.
    if config and getattr(config, "enable_adaptive_runtime", False):
        setattr(config, "_adaptive_runtime", _build_adaptive_runtime(config))

    # Create shared rate limiter (legacy path). In adaptive mode we rely on
    # runtime admission control instead of per-request token buckets.
    rate_limiter = None
    if _config_adaptive_runtime(config) is None:
        rate_limiter = AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup) if config.rps_limit > 0 else None

    # ── Resume mode ──────────────────────────────────────
    if resume:
        run_dir = Path(resume).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume path does not exist: {run_dir}")
        checkpoint_path = _checkpoint_path_for_run(run_dir)
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint.json in {run_dir}")
        if ckpt.get("status") == "done":
            print(f"All files already completed in {run_dir}")
            return {"status": "already_done", "run_dir": str(run_dir)}

        ckpt_meta = ckpt.get("metadata") or {}
        # Recover settings from summary_stats or checkpoint
        summary_path = _summary_path_for_run(run_dir)
        prev_summary = {}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
        recovered_input = prev_summary.get("input_path") or ckpt_meta.get("input_path")
        if recovered_input is None and input_path is not None:
            recovered_input = str(input_path)
        if recovered_input is None:
            raise FileNotFoundError(
                f"missing {PASS1_SUMMARY_STATS_FILE} in {run_dir}; "
                "please run 'sft-label optimize-layout --input <run_dir> --apply' "
                "for legacy runs or provide --input explicitly when resuming"
            )

        _input_path, _inline_target = _normalize_run_input(recovered_input)
        _model = prev_summary.get("model", config.labeling_model)
        _mode = _resolve_mode(prev_summary.get("mode", ckpt_meta.get("mode", mode)))
        _migrate_from = prev_summary.get("migrate_from") or ckpt_meta.get("migrate_from") or (
            str(migrate_from) if migrate_from else None
        )

        layout = InlineRunLayout.from_paths(_input_path, run_dir)

        files = discover_input_files(_input_path)
        dir_files = [(a, r) for a, r in files if r is not None]
        if not dir_files:
            raise ValueError("--resume only works with directory-mode runs")

        completed = set(ckpt.get("completed", []))
        _concurrency = config.concurrency
        migration_index = None
        migration_stats = None
        if _mode == "migrate":
            if not _migrate_from:
                raise ValueError("resume metadata is missing migrate_from for migrate mode")
            migration_index, migration_stats = load_inline_migration_index(_migrate_from)
            _print_migration_warning(migration_stats)

        print(
            f"Pipeline RESUME | {run_dir} | mode={_mode} model={_model} "
            f"concurrency={_concurrency} | {len(completed)}/{len(dir_files)} files done"
        )
        workload_estimate = estimate_directory_workload(
            dir_files,
            limit=limit,
            shuffle=shuffle,
            parallelize=parallelize_workload_estimation,
            config=config,
            completed_set=completed,
            enable_arbitration=enable_arbitration,
            mode=_mode,
            migration_index=migration_index,
        )
        print(
            "  Plan | "
            f"{workload_estimate.files_planned} pending files, "
            f"{workload_estimate.total_samples} samples "
            f"({workload_estimate.total_labeled_samples} labeled, {workload_estimate.total_inherited_samples} inherited), "
            f"pass1_llm~{workload_estimate.initial_estimated_llm_calls}, "
            f"scan {workload_estimate.scan_elapsed_seconds:.1f}s"
        )

        batch_start = time.time()

        client_limits = _build_http_client_limits(
            _concurrency,
            chunked_output_files=_chunked_output_fd_budget(
                is_directory=True,
                input_path=_input_path,
                config=config,
            ),
            pprint=print,
        )
        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=client_limits,
        ) as http_client:
            sem = asyncio.Semaphore(_concurrency)
            n = len(dir_files)
            with create_progress() as progress:
                file_task = progress.add_task("Files", total=n, info="")
                sample_task = progress.add_task(
                    "Pass 1",
                    total=workload_estimate.total_labeled_samples,
                    visible=workload_estimate.total_labeled_samples > 0,
                    info="starting...",
                )
                llm_task = progress.add_task(
                    "LLM (P1+P2)",
                    total=max(workload_estimate.initial_estimated_llm_calls, 1),
                    visible=workload_estimate.total_labeled_samples > 0,
                    info="starting...",
                )
                all_file_stats = await run_directory_pipeline(
                    dir_files, run_dir, _model, _concurrency,
                    checkpoint_path, completed_set=completed,
                    limit=limit, shuffle=shuffle,
                    progress=progress, file_task=file_task, sample_task=sample_task,
                    llm_task=llm_task, workload_estimate=workload_estimate,
                    http_client=http_client, sem=sem,
                    enable_arbitration=enable_arbitration,
                    config=config, rate_limiter=rate_limiter,
                    llm_progress_cb=llm_progress_cb,
                    layout=layout,
                    mode=_mode,
                    migration_index=migration_index,
                )

        # Write global summary
        _write_global_summary(all_file_stats, run_dir, _input_path, _model, _concurrency, batch_start,
                              rate_limiter=rate_limiter, workload_estimate=workload_estimate,
                              layout=layout, mode=_mode, migrate_from=_migrate_from,
                              migration_stats=migration_stats, config=config)
        summary_path = _summary_path_for_run(run_dir, layout=layout)
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Normal mode ──────────────────────────────────────
    if normalized_input is None:
        raise ValueError("--input is required")
    _input_path = normalized_input
    files = discover_input_files(_input_path)
    is_directory = _input_path.is_dir()
    migration_index = None
    migration_stats = None
    if mode == "migrate":
        if not migrate_from:
            raise ValueError("--migrate-from is required for --mode migrate")
        migration_index, migration_stats = load_inline_migration_index(migrate_from)
        _print_migration_warning(migration_stats)

    # Determine output directory
    base_dir = (
        inline_input_target.layout.run_root.parent
        if inline_input_target is not None
        else None
    )
    run_dir = resolve_run_dir(str(output) if output is not None else None, _input_path, base_dir=base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    layout = InlineRunLayout.from_paths(_input_path, run_dir)

    _concurrency = config.concurrency

    _arb = "off" if not enable_arbitration else f"threshold={config.confidence_threshold}"
    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    _src = f"directory, {len(files)} files" if is_directory else "single file"
    print(
        f"Pipeline | {_input_path} ({_src}) | mode={mode} "
        f"model={config.labeling_model} concurrency={_concurrency} {_rps} arb={_arb}"
    )
    print(f"  run_dir={run_dir}")
    if migration_stats:
        print(
            "  Migrate | "
            f"{migration_stats['unique_data_ids']} unique ids from {migration_stats['source_input']}"
        )

    if is_directory:
        # ── Directory mode: cross-file pipeline ──
        dir_files = [(a, r) for a, r in files if r is not None]
        if not dir_files:
            raise FileNotFoundError("No .json/.jsonl files found in directory")

        checkpoint_path = _checkpoint_path_for_run(run_dir, layout=layout)
        create_checkpoint(
            checkpoint_path,
            dir_files,
            metadata={
                "mode": mode,
                "input_path": str(_input_path),
                "migrate_from": str(migrate_from) if migrate_from else None,
                "created_at": datetime.now().isoformat(),
            },
        )
        workload_estimate = precomputed_workload_estimate
        if workload_estimate is None:
            workload_estimate = run_with_heartbeat(
                "Estimating Pass 1 workload",
                lambda: estimate_directory_workload(
                    dir_files,
                    limit=limit,
                    shuffle=shuffle,
                    parallelize=parallelize_workload_estimation,
                    config=config,
                    completed_set=None,
                    enable_arbitration=enable_arbitration,
                    mode=mode,
                    migration_index=migration_index,
                ),
            )
        print(
            "  Plan | "
            f"{workload_estimate.files_planned} files, "
            f"{workload_estimate.total_samples} samples "
            f"({workload_estimate.total_labeled_samples} labeled, {workload_estimate.total_inherited_samples} inherited), "
            f"pass1_llm~{workload_estimate.initial_estimated_llm_calls}, "
            f"scan {workload_estimate.scan_elapsed_seconds:.1f}s"
        )
        batch_start = time.time()

        client_limits = _build_http_client_limits(
            _concurrency,
            chunked_output_files=_chunked_output_fd_budget(
                is_directory=True,
                input_path=_input_path,
                config=config,
            ),
            pprint=print,
        )
        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=client_limits,
        ) as http_client:
            sem = asyncio.Semaphore(_concurrency)
            n = len(dir_files)
            with create_progress() as progress:
                file_task = progress.add_task("Files", total=n, info="")
                sample_task = progress.add_task(
                    "Pass 1",
                    total=workload_estimate.total_labeled_samples,
                    visible=workload_estimate.total_labeled_samples > 0,
                    info="starting...",
                )
                llm_task = progress.add_task(
                    "LLM (P1+P2)",
                    total=max(workload_estimate.initial_estimated_llm_calls, 1),
                    visible=workload_estimate.total_labeled_samples > 0,
                    info="starting...",
                )
                all_file_stats = await run_directory_pipeline(
                    dir_files, run_dir, config.labeling_model, _concurrency,
                    checkpoint_path,
                    limit=limit, shuffle=shuffle,
                    progress=progress, file_task=file_task, sample_task=sample_task,
                    llm_task=llm_task, workload_estimate=workload_estimate,
                    http_client=http_client, sem=sem,
                    enable_arbitration=enable_arbitration,
                    config=config, rate_limiter=rate_limiter,
                    llm_progress_cb=llm_progress_cb,
                    layout=layout,
                    mode=mode,
                    migration_index=migration_index,
                )

        _write_global_summary(all_file_stats, run_dir, _input_path, config.labeling_model, _concurrency, batch_start,
                              rate_limiter=rate_limiter, workload_estimate=workload_estimate,
                              layout=layout, mode=mode,
                              migrate_from=str(migrate_from) if migrate_from else None,
                              migration_stats=migration_stats, config=config)
        summary_path = _summary_path_for_run(run_dir, layout=layout)
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        # ── Single-file mode: backward compatible ────────
        batch_start = time.time()
        client_limits = _build_http_client_limits(
            _concurrency,
            chunked_output_files=_chunked_output_fd_budget(
                is_directory=False,
                input_path=_input_path,
                config=config,
            ),
            pprint=print,
        )
        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=client_limits,
        ) as http_client:
            sem = asyncio.Semaphore(_concurrency)
            with create_progress() as progress:
                sample_task = progress.add_task("Pass 1", total=None, info="starting...")
                stats = await run_one_file(
                    _input_path,
                    layout.file_artifact_dir(_input_path) if str(_input_path).endswith(".jsonl") else run_dir,
                    http_client,
                    sem,
                    config.labeling_model,
                    enable_arbitration=enable_arbitration,
                    limit=limit, shuffle=shuffle,
                    progress=progress, sample_task=sample_task,
                    config=config, rate_limiter=rate_limiter,
                    llm_progress_cb=llm_progress_cb,
                    dataset_output_path=layout.mirrored_dataset_path(_input_path) if str(_input_path).endswith(".jsonl") else None,
                    run_failure_log_path=layout.run_artifact_path("failures.jsonl") if str(_input_path).endswith(".jsonl") else None,
                    inline_output=str(_input_path).endswith(".jsonl"),
                    mode=mode,
                    migration_index=migration_index,
                )

        stats["model"] = config.labeling_model
        stats["concurrency"] = _concurrency
        stats["timestamp"] = datetime.now().isoformat()
        stats["run_dir"] = str(run_dir)
        stats["mode"] = mode
        if migrate_from:
            stats["migrate_from"] = str(migrate_from)
        if migration_stats:
            stats["migration_indexed_rows"] = migration_stats.get("indexed_rows", 0)
            stats["migration_unique_data_ids"] = migration_stats.get("unique_data_ids", 0)
            duplicate_data_ids = migration_stats.get("duplicate_data_ids", 0)
            if duplicate_data_ids:
                stats["migration_duplicate_data_ids"] = duplicate_data_ids
        if rate_limiter:
            stats["http_request_stats"] = rate_limiter.stats.to_dict()

        # Overwrite stats with enriched version
        stats_path = (
            layout.file_artifact_path(_input_path, PASS1_STATS_FILE)
            if str(_input_path).endswith(".jsonl")
            else (run_dir / PASS1_STATS_FILE)
        )
        _write_json_atomic(stats_path, stats)

        print_summary(stats, run_dir)
        if str(_input_path).endswith(".jsonl"):
            print(f"Dataset: {layout.mirrored_dataset_path(_input_path)}")
            print(f"Meta:    {layout.file_artifact_dir(_input_path)}")
            print(f"Stats:   {layout.file_artifact_path(_input_path, PASS1_STATS_FILE)}")
            print(f"Monitor: {layout.file_artifact_path(_input_path, 'monitor.jsonl')}")
        else:
            print(f"Output:  {run_dir / 'labeled.json'}")
            print(f"JSONL:   {run_dir / 'labeled.jsonl'}")
            print(f"Stats:   {run_dir / PASS1_STATS_FILE}")
            print(f"Monitor: {run_dir / 'monitor.jsonl'}")
        return stats


async def run_pipeline(args):
    """CLI entry point. Delegates to run()."""
    config = PipelineConfig(labeling_model=args.model, concurrency=args.concurrency)
    return await run(
        input_path=args.input,
        output=args.output,
        resume=args.resume,
        limit=args.limit,
        shuffle=args.shuffle,
        enable_arbitration=not args.no_arbitration,
        config=config,
        mode=getattr(args, "mode", "refresh"),
        migrate_from=getattr(args, "migrate_from", None),
    )


def main():
    parser = argparse.ArgumentParser(description="SFT Auto-Labeling Pipeline (Concurrent)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory: omit for sibling of input, or an explicit path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing run directory (reads checkpoint.json)")
    parser.add_argument("--mode", type=str, default="refresh",
                        choices=sorted(INLINE_RUN_MODES))
    parser.add_argument("--migrate-from", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_LABELING_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples per file (0 = all). In directory mode, applies to each file independently")
    parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle samples before slicing")
    parser.add_argument("--no-arbitration", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(run_pipeline(args))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
