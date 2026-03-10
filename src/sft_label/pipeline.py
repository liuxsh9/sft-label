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
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import httpx
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)

from sft_label.prompts import (
    CALL1_SYSTEM, CALL1_FEWSHOT, CALL2_SYSTEM, CALL2_FEWSHOT,
    CALL1_SYSTEM_COMPACT, CALL2_SYSTEM_COMPACT,
    CALL1_FEWSHOT_COMPACT, CALL2_FEWSHOT_COMPACT,
    TAG_POOLS, SINGLE_SELECT, MULTI_SELECT
)
from sft_label.preprocessing import preprocess, format_signals_for_prompt, normalize_and_slice, truncate_conversations_for_labeling, apply_sparse_sampling
from sft_label.config import (
    LITELLM_BASE, LITELLM_KEY, CONFIDENCE_THRESHOLD, CONSISTENCY_RULES,
    DEFAULT_LABELING_MODEL, DEFAULT_CONCURRENCY, MAX_RETRIES, SAMPLE_MAX_RETRIES,
    REQUEST_TIMEOUT, REQUEST_TIMEOUT_ESCALATION,
    MAX_CONVERSATION_CHARS,
    DIR_PIPELINE_WATERMARK, DIR_PIPELINE_MAX_FILES,
    CHUNK_SIZE, MAX_ACTIVE_CHUNKS,
    PipelineConfig,
)
from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_STATS_FILE_LEGACY,
    PASS1_SUMMARY_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE_LEGACY,
    pass1_stats_filename,
    pass1_stats_legacy_filename,
    pass1_dashboard_filename,
    pass1_dashboard_legacy_filename,
    pass1_global_dashboard_filename,
    pass1_global_dashboard_legacy_filename,
    find_first_existing,
    sync_legacy_aliases,
)
from sft_label.labels import is_partial_labels, is_usable_labels


# ─────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────

def create_progress():
    """Create a Rich progress bar display for labeling."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•", style="dim"),
        TimeElapsedColumn(),
        TextColumn("•", style="dim"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}", style="cyan"),
    )


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
        if f.is_file()
    )
    base = p.resolve()
    return [(f, f.relative_to(base)) for f in files]


def load_checkpoint(checkpoint_path):
    """Load existing checkpoint or return None."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def create_checkpoint(checkpoint_path, files):
    """Create a fresh checkpoint for a batch run."""
    ckpt = {
        "status": "in_progress",
        "completed": [],
        "failed": {},
        "total_files": len(files),
    }
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
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


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
        "total_elapsed_seconds": 0,
        "sparse_labeled": 0, "sparse_inherited": 0,
        "tag_distributions": {},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
        "files_processed": len(all_file_stats),
    }
    for st in all_file_stats:
        for k in ("total_samples", "success", "failed", "total_llm_calls",
                   "total_prompt_tokens", "total_completion_tokens", "total_tokens",
                   "arbitrated_count", "validation_issue_count", "consistency_warning_count",
                   "sparse_labeled", "sparse_inherited",
                   "distribution_total_samples"):
            merged[k] += st.get(k, 0)
        merged["total_elapsed_seconds"] += st.get("total_elapsed_seconds", 0)
        # Merge tag distributions
        for dim, dist in st.get("tag_distributions", {}).items():
            if dim not in merged["tag_distributions"]:
                merged["tag_distributions"][dim] = {}
            for tag, count in dist.items():
                merged["tag_distributions"][dim][tag] = merged["tag_distributions"][dim].get(tag, 0) + count
        # Merge unmapped
        for tag, count in st.get("unmapped_tags", {}).items():
            merged["unmapped_tags"][tag] = merged["unmapped_tags"].get(tag, 0) + count
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
    merged["unmapped_tags"] = dict(sorted(merged["unmapped_tags"].items(), key=lambda x: -x[1]))
    merged["unmapped_unique_count"] = len(merged["unmapped_tags"])

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


def resolve_run_dir(output, input_path):
    """Determine the run output directory based on output setting.

    Two modes:
      - None (default):
        - Directory input: <parent>/<dir_name>-labeled-<timestamp>/
        - Single file input: <parent>/<stem>-labeled-<timestamp>/
      - explicit path: use as-is (absolute or relative to cwd)
    """
    if output is not None:
        return Path(output).resolve()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = input_path.parent
    if input_path.is_dir():
        name = input_path.resolve().name
    else:
        name = input_path.stem
    return parent / f"{name}-labeled-{run_ts}"


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
                # only prolongs the cooldown cycle, so treat as non-retryable.
                if "no deployments available" in resp_lower or "cooldown_list" in resp_lower:
                    return None, f"HTTP {resp.status_code}: {resp_text[:300]}", {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "error": f"HTTP {resp.status_code} (deployment cooldown): {resp_text[:300]}",
                        "error_response": resp.text,
                        "non_retryable": True,
                    }
                # Rate limited or server error — exponential backoff with jitter
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
                        "error": f"HTTP 400 (content filtered): {error_text[:300]}",
                        "error_response": resp.text,
                        "non_retryable": True,
                    }
                # Likely a transient proxy/supplier error — retry with backoff
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
            if attempt < max_retries:
                await asyncio.sleep(2 + random.uniform(0, 2))
                continue
            return None, content if 'content' in locals() else "", {"prompt_tokens": 0, "completion_tokens": 0, "error": last_error}
        except Exception as e:
            if rate_limiter is not None and not _http_recorded:
                rate_limiter.stats.record_timeout()
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                base_wait = min(2 ** attempt * 3 + 2, 60)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                await asyncio.sleep(wait)
                continue
            return None, str(e), {"prompt_tokens": 0, "completion_tokens": 0, "error": last_error}

    return None, f"max retries exceeded: {last_error}", {"prompt_tokens": 0, "completion_tokens": 0, "error": last_error or "max_retries", "error_response": last_error_response}


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

# Alias → canonical tag mapping.  Applied before pool validation so that
# common LLM outputs (underscore variants, abbreviations, sub-topic names)
# are silently resolved instead of becoming unmapped.
TAG_ALIASES = {
    # language
    "c++": "cpp",
    "c#": "csharp",
    "f#": "fsharp",
    # concept — algorithm sub-topics
    "dynamic-programming": "algorithms",
    "dp": "algorithms",
    "graph-theory": "algorithms",
    "geometry": "algorithms",
    "combinatorics": "algorithms",
    "bit-manipulation": "algorithms",
    "string-manipulation": "algorithms",
    "string-algorithms": "algorithms",
    "mathematics": "algorithms",
    "greedy": "algorithms",
    "backtracking": "algorithms",
    "divide-and-conquer": "algorithms",
    # agentic — format variants
    "execute_python_code": "code-execution",
    "execute_python": "code-execution",
    "execute_code": "code-execution",
    "run_code": "code-execution",
    "execute_bash": "bash-execution",
    "run_bash": "bash-execution",
    "shell_execution": "bash-execution",
    # task
    "test-creation": "testing-task",
    "write-tests": "testing-task",
    # domain
    "security": "cybersecurity",
    # intent — modify aliases
    "refactor": "modify",
    "optimize": "modify",
    "transform": "modify",
    # language — common abbreviations
    "objc": "objective-c",
    "objective_c": "objective-c",
    "js": "javascript",
    "ts": "typescript",
    "rb": "ruby",
    "py": "python",
    # concept — additional mappings
    "regex": "algorithms",
    "regular-expressions": "algorithms",
    "async-await": "concurrency",
    "system-programming": "concurrency",
}

# Tags that belong to a DIFFERENT category — drop from the current dimension
# and log a warning, but do NOT add to unmapped (they're valid, just misplaced).
CROSS_CATEGORY_CORRECTIONS = {
    "domain": {
        # These are concepts or tasks, not domains
        "algorithm", "algorithms", "data-structure", "data-structures",
        "documentation", "debugging",
    },
}


def _resolve_aliases(dim, values):
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
        # Cross-category correction: silently drop misplaced tags
        if dim in CROSS_CATEGORY_CORRECTIONS and canonical in CROSS_CATEGORY_CORRECTIONS[dim]:
            continue
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
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


def _build_partial_labels(call1_cleaned, reason):
    """Wrap Call 1 output so downstream stages can block partial labels."""
    labels = {
        d: call1_cleaned.get(d, [] if d in MULTI_SELECT else "")
        for d in ["intent", "language", "domain", "task", "difficulty"]
    }
    labels["confidence"] = dict(call1_cleaned.get("confidence", {}))
    labels["unmapped"] = list(call1_cleaned.get("unmapped", []))
    labels["partial"] = True
    labels["partial_stage"] = "call1"
    labels["partial_reason"] = reason
    return labels


def validate_tags(result, call_name="call1"):
    issues = []
    unmapped = result.get("unmapped", [])
    if not isinstance(unmapped, list):
        unmapped = []
    cleaned = dict(result)

    dims = (["intent", "language", "domain", "task", "difficulty"] if call_name == "call1"
            else ["concept", "agentic", "constraint", "context"])

    for dim in dims:
        if dim not in result:
            issues.append(f"Missing: {dim}")
            cleaned[dim] = "" if dim in SINGLE_SELECT else []
            continue

        pool = TAG_POOLS.get(dim, set())
        if dim in SINGLE_SELECT:
            raw_val = result[dim]
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
                issues.append(f"{dim}: '{val}' not in pool")
                unmapped.append({"dimension": dim, "value": val})
                cleaned[dim] = ""
            else:
                cleaned[dim] = val
        else:
            raw = result[dim] if isinstance(result[dim], list) else [result[dim]]
            resolved = _resolve_aliases(dim, raw)
            valid = []
            for v in resolved:
                if v in pool:
                    valid.append(v)
                else:
                    issues.append(f"{dim}: '{v}' not in pool")
                    unmapped.append({"dimension": dim, "value": v})
            cleaned[dim] = valid

    # Normalize string items to dict format (LLM may return plain strings)
    unmapped = [
        item if isinstance(item, dict) else {"dimension": "?", "value": item}
        for item in unmapped
    ]
    # Filter out LLM explanation sentences — unmapped should be short tag IDs
    unmapped = [
        item for item in unmapped
        if len(item.get("value", "")) <= 60 and item.get("value", "").count(" ") <= 3
    ]

    cleaned["confidence"] = _sanitize_confidence(result.get("confidence"), dims, issues)
    cleaned["unmapped"] = unmapped
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


# ─────────────────────────────────────────────────────────
# Per-sample pipeline (async)
# ─────────────────────────────────────────────────────────

async def label_one(http_client, sample, model, sample_idx, total, sem, enable_arbitration=True,
                    config=None, rate_limiter=None):
    """Label a single sample with sample-level retry on failure."""
    _max_retries_sample = config.sample_max_retries if config else SAMPLE_MAX_RETRIES
    _max_retries = config.max_retries if config else MAX_RETRIES
    _conf_threshold = config.confidence_threshold if config else CONFIDENCE_THRESHOLD
    _compact = config.prompt_mode == "compact" if config else False
    start = time.time()

    # Conversation truncation budget — compact mode uses smaller budget for firewall limits
    _max_chars = config.max_conversation_chars if config else MAX_CONVERSATION_CHARS
    if _compact and config and config.max_conversation_chars == MAX_CONVERSATION_CHARS:
        from sft_label.config import COMPACT_CONVERSATION_CHARS
        _max_chars = COMPACT_CONVERSATION_CHARS

    # Truncate oversized conversations before sending to LLM
    conversations = sample.get("conversations", [])
    _trunc_kw = {"max_total_chars": _max_chars}
    if config:
        _trunc_kw.update(head_ratio=config.truncation_head_ratio,
                         last_response_ratio=config.truncation_last_response_ratio,
                         per_turn_ratio=config.truncation_per_turn_ratio)
    truncated_convs, was_truncated = truncate_conversations_for_labeling(
        conversations, **_trunc_kw)

    _cached_call1 = None  # Cache successful Call 1 across sample retries

    for sample_attempt in range(_max_retries_sample + 1):
        if sample_attempt > 0:
            # Back off before retry with jitter, outside semaphore so we don't block others
            base_wait = 2 ** sample_attempt * 2
            await asyncio.sleep(base_wait + random.uniform(0, base_wait))

        async with sem:
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
                conversations_json = json.dumps(truncated_convs, ensure_ascii=False)
                if was_truncated:
                    monitor["truncated"] = True

                # Call 1 — reuse cached result if available from a previous attempt
                if _cached_call1 is not None:
                    call1_cleaned, msgs1 = _cached_call1
                    monitor["call1_cached"] = True
                else:
                    msgs1 = build_call1_messages(conversations_json, signals_str, compact=_compact)
                    call1_result, call1_raw, usage1 = await async_llm_call(http_client, msgs1, model,
                                                                          max_retries=_max_retries, config=config,
                                                                          rate_limiter=rate_limiter)
                    monitor["llm_calls"] += 1
                    monitor["total_prompt_tokens"] += usage1["prompt_tokens"]
                    monitor["total_completion_tokens"] += usage1["completion_tokens"]

                    if call1_result is None:
                        monitor["status"] = "call1_failed"
                        monitor["error"] = usage1.get("error", "unknown")
                        monitor["error_response"] = usage1.get("error_response") or (call1_raw[:500] if call1_raw else "")
                        if usage1.get("non_retryable"):
                            return sample_idx, None, monitor
                        if sample_attempt < _max_retries_sample:
                            continue
                        return sample_idx, None, monitor

                    call1_cleaned, call1_issues = validate_tags(call1_result, "call1")
                    monitor["validation_issues"].extend(call1_issues)
                    # Cache successful Call 1 for potential sample retry
                    _cached_call1 = (call1_cleaned, msgs1)

                # Call 2 (depends on Call 1)
                call1_context = {d: call1_cleaned[d] for d in ["intent", "language", "domain", "task", "difficulty"] if d in call1_cleaned}
                msgs2 = build_call2_messages(conversations_json, signals_str, call1_context, compact=_compact)
                call2_result, call2_raw, usage2 = await async_llm_call(http_client, msgs2, model,
                                                                      max_retries=_max_retries, config=config,
                                                                      rate_limiter=rate_limiter)
                monitor["llm_calls"] += 1
                monitor["total_prompt_tokens"] += usage2["prompt_tokens"]
                monitor["total_completion_tokens"] += usage2["completion_tokens"]

                if call2_result is None:
                    monitor["status"] = "call2_failed"
                    monitor["error"] = usage2.get("error", "unknown")
                    monitor["error_response"] = usage2.get("error_response") or (call2_raw[:500] if call2_raw else "")
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

                # Consistency
                warnings = check_consistency(labels)
                monitor["consistency_warnings"] = warnings

                # Arbitration
                low_conf = find_low_confidence_dims(labels, threshold=_conf_threshold)
                monitor["low_confidence_dims"] = [{"dim": d, "conf": s} for d, s in low_conf]

                if low_conf and enable_arbitration:
                    monitor["arbitrated"] = True
                    # Re-run relevant call(s)
                    call1_dims = {"intent", "language", "domain", "task", "difficulty"}
                    call2_dims = {"concept", "agentic", "constraint", "context"}

                    if any(d in call1_dims for d, _ in low_conf):
                        re1, _, u1 = await async_llm_call(http_client, msgs1, model, temperature=0.3,
                                                          max_retries=_max_retries, config=config,
                                                          rate_limiter=rate_limiter)
                        monitor["llm_calls"] += 1
                        monitor["total_prompt_tokens"] += u1["prompt_tokens"]
                        monitor["total_completion_tokens"] += u1["completion_tokens"]
                        if re1:
                            re1_clean, _ = validate_tags(re1, "call1")
                            for d, _ in low_conf:
                                if d in call1_dims and d in re1_clean:
                                    labels[d] = re1_clean[d]
                                    labels["confidence"][d] = re1_clean.get("confidence", {}).get(d, 0)

                    if any(d in call2_dims for d, _ in low_conf):
                        re2, _, u2 = await async_llm_call(http_client, msgs2, model, temperature=0.3,
                                                          max_retries=_max_retries, config=config,
                                                          rate_limiter=rate_limiter)
                        monitor["llm_calls"] += 1
                        monitor["total_prompt_tokens"] += u2["prompt_tokens"]
                        monitor["total_completion_tokens"] += u2["completion_tokens"]
                        if re2:
                            re2_clean, _ = validate_tags(re2, "call2")
                            for d, _ in low_conf:
                                if d in call2_dims and d in re2_clean:
                                    labels[d] = re2_clean[d]
                                    labels["confidence"][d] = re2_clean.get("confidence", {}).get(d, 0)

            except Exception as e:
                monitor["status"] = f"error: {str(e)[:100]}"
                if sample_attempt < _max_retries_sample:
                    continue
                return sample_idx, None, monitor

            monitor["elapsed_seconds"] = round(time.time() - start, 2)
            return sample_idx, labels, monitor

        # Should not reach here, but just in case
        monitor["elapsed_seconds"] = round(time.time() - start, 2)
        return sample_idx, None, monitor


def compute_stats(all_monitors, all_labels, inherit_map=None):
    """Compute aggregate statistics.

    When inherit_map is provided, inherited samples are:
    - Counted toward success (if non-None) and total_samples
    - Excluded from tag_distributions, confidence_stats, cross_matrix
      (to avoid inflating tag counts from duplicated labels)
    - Reported separately as sparse_labeled / sparse_inherited
    """
    total = len(all_monitors)
    success = sum(1 for m in all_monitors if m["status"] == "success")
    total_calls = sum(m["llm_calls"] for m in all_monitors)
    total_pt = sum(m["total_prompt_tokens"] for m in all_monitors)
    total_ct = sum(m["total_completion_tokens"] for m in all_monitors)
    arbitrated = sum(1 for m in all_monitors if m["arbitrated"])

    inherited_indices = set(inherit_map.keys()) if inherit_map else set()
    distribution_total = sum(
        1 for idx, labels in enumerate(all_labels)
        if is_usable_labels(labels) and idx not in inherited_indices
    )

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

    all_unmapped = {}
    for idx, labels in enumerate(all_labels):
        if not is_usable_labels(labels) or idx in inherited_indices:
            continue
        for item in labels.get("unmapped", []):
            key = f"{item.get('dimension', '?')}:{item.get('value', '?')}" if isinstance(item, dict) else str(item)
            all_unmapped[key] = all_unmapped.get(key, 0) + 1

    conf_stats = {}
    for dim in ["intent", "language", "domain", "task", "difficulty", "concept", "agentic", "constraint", "context"]:
        scores = [l["confidence"][dim] for idx, l in enumerate(all_labels)
                  if is_usable_labels(l) and idx not in inherited_indices
                  and "confidence" in l and isinstance(l["confidence"].get(dim), (int, float))]
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

    return {
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
        "confidence_stats": conf_stats,
        "low_confidence_frequency": dict(sorted(
            {d: sum(1 for m in all_monitors for lc in m.get("low_confidence_dims", []) if lc["dim"] == d)
             for d in set(lc["dim"] for m in all_monitors for lc in m.get("low_confidence_dims", []))}.items(),
            key=lambda x: -x[1])),
        "tag_distributions": distributions,
        "cross_matrix": cross,
    }


# ─────────────────────────────────────────────────────────
# Streaming I/O + cross-file helpers
# ─────────────────────────────────────────────────────────

def iter_samples_from_file(input_path, limit=0, shuffle=False):
    """Load and normalize samples from a file with minimal memory overhead.

    JSONL: line-by-line read + normalize_and_slice (memory = 1 raw line at a time).
    JSON:  json.load then del raw (can't avoid full load, but releases raw ASAP).

    Returns: (samples_list, n_raw)
    """
    input_path = Path(input_path)
    samples = []
    n_raw = 0

    if str(input_path).endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_raw += 1
                raw = json.loads(line)
                samples.extend(normalize_and_slice(raw))
                del raw
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
        if not isinstance(raw_samples, list):
            raw_samples = [raw_samples]
        n_raw = len(raw_samples)
        for s in raw_samples:
            samples.extend(normalize_and_slice(s))
        del raw_samples

    for i, s in enumerate(samples):
        if not s.get("id"):
            s["id"] = f"sample-{i:04d}"

    if shuffle:
        random.shuffle(samples)
    if limit > 0:
        samples = samples[:limit]

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
    label_count: int = 0    # actual LLM labels (sparse)
    inherit_map: dict = field(default_factory=dict)
    sparse_info: str = ""   # progress bar context string
    done: int = 0
    ok: int = 0
    fail: int = 0
    labels: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    completed: bool = False

    def __post_init__(self):
        # Pre-allocate result slots
        self.labels = [None] * self.total
        self.monitors = [None] * self.total


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
    config=None,
    completed_set=None,
    enable_arbitration=True,
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
        )

    # Preserve RNG state so pre-scan doesn't perturb runtime shuffle behavior.
    rng_state = random.getstate()
    try:
        for abs_path, rel_path in dir_files:
            rel_str = str(rel_path)
            if rel_str in completed_set:
                continue
            files_planned += 1

            samples, n_raw = iter_samples_from_file(abs_path, limit=limit, shuffle=shuffle)
            label_indices, inherit_map = apply_sparse_sampling(samples, **_sparse_kw)

            total_raw += n_raw
            total_samples += len(samples)
            total_labeled += len(label_indices)
            total_inherited += len(inherit_map)
            del samples
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


def flush_file_output(collector, run_dir, checkpoint_path, pprint=print):
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
    total = collector.total

    # Inherit labels for sparse-sampled slices
    for unlabeled_idx, source_idx in collector.inherit_map.items():
        if all_labels[source_idx] is not None:
            inherited = dict(all_labels[source_idx])
            inherited["inherited"] = True
            inherited["inherited_from"] = samples[source_idx].get("id")
            all_labels[unlabeled_idx] = inherited

    # Attach labels to samples
    for idx, sample in enumerate(samples):
        sample["labels"] = all_labels[idx]
        sample.setdefault("metadata", {})["source_file"] = str(collector.abs_path)
        if all_monitors[idx]:
            sample["labeling_monitor"] = {
                "llm_calls": all_monitors[idx]["llm_calls"],
                "arbitrated": all_monitors[idx]["arbitrated"],
                "validation_issues": all_monitors[idx]["validation_issues"],
                "consistency_warnings": all_monitors[idx]["consistency_warnings"],
            }

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{prefix}" if prefix else ""

    labeled_json = f"labeled{suffix}.json"
    labeled_jsonl = f"labeled{suffix}.jsonl"
    monitor_file = f"monitor{suffix}.jsonl"
    stats_file = pass1_stats_filename(suffix)
    stats_file_legacy = pass1_stats_legacy_filename(suffix)
    dashboard_file = pass1_dashboard_filename(suffix)
    dashboard_file_legacy = pass1_dashboard_legacy_filename(suffix)
    failed_samples_file = f"failed_samples{suffix}.jsonl"

    with open(output_dir / labeled_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    with open(output_dir / labeled_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(output_dir / monitor_file, "w", encoding="utf-8") as f:
        for m in all_monitors:
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Write failed samples (original, without labels) for easy retry
    # Inherited samples have no monitor — they are not failures
    inherited_indices = set(collector.inherit_map.keys())
    failed_indices = [
        i for i, l in enumerate(all_labels)
        if (l is None or is_partial_labels(l)) and i not in inherited_indices
    ]
    if failed_indices:
        with open(output_dir / failed_samples_file, "w", encoding="utf-8") as f:
            for i in failed_indices:
                s = dict(samples[i])
                s.pop("labels", None)
                s.pop("labeling_monitor", None)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

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
        with open(run_dir / "failures.jsonl", "a", encoding="utf-8") as f:
            for r in failure_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute and write stats
    valid_monitors = [m for m in all_monitors if m is not None]
    stats = compute_stats(valid_monitors, all_labels, inherit_map=collector.inherit_map)
    stats["input_file"] = str(collector.abs_path)
    sparse_inherited = len(collector.inherit_map)
    if sparse_inherited > 0:
        stats["sparse_labeled"] = collector.label_count
        stats["sparse_inherited"] = sparse_inherited
        # Adjust totals: inherited samples count as success
        stats["total_samples"] = total
        inherited_ok = sum(1 for i in inherited_indices if is_usable_labels(all_labels[i]))
        stats["success"] += inherited_ok
        stats["failed"] = stats["total_samples"] - stats["success"]
        stats["success_rate"] = round(stats["success"] / max(total, 1), 4)

    stats_path = output_dir / stats_file
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    sync_legacy_aliases(stats_path, [stats_file_legacy])

    # Per-file dashboard
    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        generate_dashboard(output_dir, labeled_file=labeled_json,
                           stats_file=stats_file, output_file=dashboard_file)
        sync_legacy_aliases(output_dir / dashboard_file, [dashboard_file_legacy])
    except Exception:
        pass

    success = stats["success"]
    total_tokens = stats["total_tokens"]

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

    def __init__(self):
        self.total = 0
        self.distribution_total = 0
        self.success = 0
        self.total_calls = 0
        self.total_pt = 0
        self.total_ct = 0
        self.arbitrated = 0
        self.validation_issue_count = 0
        self.consistency_warning_count = 0

        self.distributions = {dim: {} for dim in self.DIMS}
        self.unmapped = {}
        self.cross_matrix = {}

        # Confidence: sum/count/min/max per dim for running mean
        self.conf_acc = {}

        # Low-confidence frequency
        self.low_conf_freq = {}

        # Sparse tracking
        self.sparse_labeled = 0
        self.sparse_inherited = 0

    def update(self, all_labels, all_monitors, inherit_map=None):
        """Ingest one chunk's labels and monitors.

        Inherited samples (from inherit_map) are counted toward total/success
        but excluded from tag_distributions, confidence, and cross_matrix
        to avoid inflating counts from duplicated labels.
        """
        inherited_indices = set(inherit_map.keys()) if inherit_map else set()

        for m in all_monitors:
            if m is None:
                continue
            self.total += 1
            if m["status"] == "success":
                self.success += 1
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
            # Unmapped
            for item in labels.get("unmapped", []):
                key = f"{item.get('dimension', '?')}:{item.get('value', '?')}" if isinstance(item, dict) else str(item)
                self.unmapped[key] = self.unmapped.get(key, 0) + 1
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

        # Count inherited samples toward total (they have labels but no monitor)
        if inherit_map:
            inherited_indices = set(inherit_map.keys())
            inherited_ok = sum(
                1 for i in inherited_indices
                if i < len(all_labels) and is_usable_labels(all_labels[i])
            )
            self.sparse_inherited += len(inherit_map)
            self.total += len(inherit_map)
            self.success += inherited_ok

    def set_sparse_labeled(self, count):
        """Track total sparse-labeled count across chunks."""
        self.sparse_labeled += count

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
            "confidence_stats": conf_stats,
            "low_confidence_frequency": dict(sorted(self.low_conf_freq.items(), key=lambda x: -x[1])),
            "tag_distributions": sorted_dist,
            "cross_matrix": self.cross_matrix,
        }
        if self.sparse_inherited > 0:
            result["sparse_labeled"] = self.sparse_labeled
            result["sparse_inherited"] = self.sparse_inherited

        return result


@dataclass
class ChunkCollector:
    """Track per-chunk labeling results for chunked JSONL pipeline."""
    chunk_idx: int
    samples: list
    label_count: int = 0
    inherit_map: dict = field(default_factory=dict)
    done: int = 0
    ok: int = 0
    fail: int = 0
    labels: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    completed: bool = False
    sample_id_offset: int = 0  # global offset for sample IDs

    def __post_init__(self):
        n = len(self.samples)
        self.labels = [None] * n
        self.monitors = [None] * n


def _flush_chunk(chunk, out_labeled, out_monitor, out_failed, stats_acc,
                 input_path, pprint=print):
    """Finalize a completed chunk: inherit, attach labels, write, release memory."""
    samples = chunk.samples
    all_labels = chunk.labels
    all_monitors = chunk.monitors

    # Inherit labels for sparse-sampled slices
    for unlabeled_idx, source_idx in chunk.inherit_map.items():
        if all_labels[source_idx] is not None:
            inherited = dict(all_labels[source_idx])
            inherited["inherited"] = True
            inherited["inherited_from"] = samples[source_idx].get("id")
            all_labels[unlabeled_idx] = inherited

    # Attach labels to samples
    for idx, sample in enumerate(samples):
        sample["labels"] = all_labels[idx]
        sample.setdefault("metadata", {})["source_file"] = str(input_path)
        if all_monitors[idx]:
            sample["labeling_monitor"] = {
                "llm_calls": all_monitors[idx]["llm_calls"],
                "arbitrated": all_monitors[idx]["arbitrated"],
                "validation_issues": all_monitors[idx]["validation_issues"],
                "consistency_warnings": all_monitors[idx]["consistency_warnings"],
            }

    # Write labeled samples (append)
    for sample in samples:
        out_labeled.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write monitors (append)
    for m in all_monitors:
        if m:
            out_monitor.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Write failed samples
    inherited_indices = set(chunk.inherit_map.keys())
    for i, l in enumerate(all_labels):
        if (l is None or is_partial_labels(l)) and i not in inherited_indices:
            s = dict(samples[i])
            s.pop("labels", None)
            s.pop("labeling_monitor", None)
            out_failed.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Update stats accumulator
    stats_acc.update(all_labels, all_monitors, inherit_map=chunk.inherit_map)
    stats_acc.set_sparse_labeled(chunk.label_count)

    # Release memory
    chunk.samples = None
    chunk.labels = None
    chunk.monitors = None
    chunk.completed = True


async def _run_one_file_chunked(input_path, output_dir, http_client, sem, model,
                                 enable_arbitration=True, limit=0, shuffle=False,
                                 progress=None, sample_task=None, config=None,
                                 rate_limiter=None, llm_progress_cb=None):
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

    chunk_size = config.chunk_size if config else CHUNK_SIZE
    max_active = config.max_active_chunks if config else MAX_ACTIVE_CHUNKS
    concurrency = config.concurrency if config else DEFAULT_CONCURRENCY
    watermark = int(concurrency * (config.dir_pipeline_watermark if config else DIR_PIPELINE_WATERMARK))

    _sparse_kw = {}
    if config:
        _sparse_kw = dict(full_label_count=config.sparse_full_label_count,
                          gap_multiplier=config.sparse_gap_multiplier,
                          min_gap=config.sparse_min_gap,
                          max_gap=config.sparse_max_gap,
                          threshold=config.sparse_threshold)

    stats_acc = StatsAccumulator()
    chunk_gen = iter_chunks_from_jsonl(input_path, chunk_size, limit=limit)

    # Open output file handles
    out_labeled = open(output_dir / "labeled.jsonl", "w", encoding="utf-8")
    out_monitor = open(output_dir / "monitor.jsonl", "w", encoding="utf-8")
    out_failed = open(output_dir / "failed_samples.jsonl", "w", encoding="utf-8")

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
                raw_records = next(chunk_gen)
            except StopIteration:
                return None

            # Normalize and slice
            samples = []
            for raw in raw_records:
                samples.extend(normalize_and_slice(raw))
            del raw_records

            # Assign global IDs
            for i, s in enumerate(samples):
                if not s.get("id"):
                    s["id"] = f"sample-{global_sample_offset + i:06d}"

            # Sparse sampling
            label_indices, inherit_map = apply_sparse_sampling(samples, **_sparse_kw)
            label_count = len(label_indices)
            sparse_inherited = len(inherit_map)

            chunk_idx = chunks_loaded
            chunks_loaded_val = chunks_loaded
            chunks_loaded_display = chunks_loaded + 1

            collector = ChunkCollector(
                chunk_idx=chunk_idx,
                samples=samples,
                label_count=label_count,
                inherit_map=inherit_map,
                sample_id_offset=global_sample_offset,
            )

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

            # Submit tasks (shuffled)
            submit_order = list(label_indices)
            random.shuffle(submit_order)
            for idx in submit_order:
                coro = label_one(
                    http_client, samples[idx], model, idx, len(samples), sem,
                    enable_arbitration=enable_arbitration,
                    config=config, rate_limiter=rate_limiter,
                )
                fut = asyncio.ensure_future(_tagged_label(coro, chunk_idx, idx))
                pending_futures.add(fut)

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

        # --- Main watermark-driven loop ---
        collectors = {}
        pending_futures = set()

        if progress and sample_task is not None:
            progress.update(sample_task, total=0, completed=0, visible=True, info="loading...")

        # Initial load
        maybe_load_more(pending_futures, collectors)

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
                        run_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass1")
                        if run_info:
                            info = f"{info} • {run_info}"
                    progress.update(sample_task, advance=1, info=info)

                # Check if chunk is fully done
                if c.done >= c.label_count and not c.completed:
                    _flush_chunk(c, out_labeled, out_monitor, out_failed,
                                 stats_acc, input_path, pprint=pprint)
                    pprint(f"  [chunk {c.chunk_idx + 1}] ✓ {c.ok} ok, {c.fail} fail — flushed")

            # After processing batch, check if we should load more chunks
            maybe_load_more(pending_futures, collectors)

        # Handle chunks with 0 labels (all inherited)
        for c in collectors.values():
            if not c.completed and c.label_count == 0:
                _flush_chunk(c, out_labeled, out_monitor, out_failed,
                             stats_acc, input_path, pprint=pprint)

        file_elapsed = time.time() - file_start

    finally:
        out_labeled.close()
        out_monitor.close()
        out_failed.close()

    # Remove empty failed file
    failed_path = output_dir / "failed_samples.jsonl"
    if failed_path.exists() and failed_path.stat().st_size == 0:
        failed_path.unlink()

    # Finalize and write stats
    stats = stats_acc.finalize()
    stats["total_elapsed_seconds"] = round(file_elapsed, 1)
    stats["input_file"] = str(input_path)
    stats["chunked"] = True
    stats["chunk_size"] = chunk_size
    stats["chunks_processed"] = chunks_loaded

    stats_path = output_dir / PASS1_STATS_FILE
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    sync_legacy_aliases(stats_path, [PASS1_STATS_FILE_LEGACY])

    # Dashboard (stats-only mode — no labeled.json)
    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        generate_dashboard(output_dir, labeled_file=None,
                           stats_file=PASS1_STATS_FILE, output_file=pass1_dashboard_filename())
        sync_legacy_aliases(
            output_dir / pass1_dashboard_filename(),
            [pass1_dashboard_legacy_filename()],
        )
    except Exception:
        pass

    pprint(f"  ✓ {stats['success']}/{stats['total_samples']} success, "
           f"{file_elapsed:.1f}s, {stats['total_tokens']:,} tokens")

    return stats


async def run_one_file(input_path, output_dir, http_client, sem, model,
                       enable_arbitration=True, limit=0, shuffle=False,
                       file_prefix=None, progress=None, sample_task=None,
                       config=None, rate_limiter=None, llm_progress_cb=None):
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
        )

    # Load input — streaming for JSONL
    samples, n_raw = iter_samples_from_file(input_path, limit=limit, shuffle=shuffle)

    total = len(samples)
    pprint = progress.console.print if progress else print

    # Sparse sampling: only label a subset of multi-turn slices
    _sparse_kw = {}
    if config:
        _sparse_kw = dict(full_label_count=config.sparse_full_label_count,
                          gap_multiplier=config.sparse_gap_multiplier,
                          min_gap=config.sparse_min_gap,
                          max_gap=config.sparse_max_gap,
                          threshold=config.sparse_threshold)
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
    all_labels = [None] * total
    all_monitors = [None] * total

    # Submit tasks in shuffled order — only for indices that need labeling
    submit_order = list(label_indices)
    random.shuffle(submit_order)
    tasks = []
    for idx in submit_order:
        tasks.append(label_one(
            http_client, samples[idx], model, idx, total, sem,
            enable_arbitration=enable_arbitration,
            config=config, rate_limiter=rate_limiter,
        ))

    done_count = 0
    ok_count = 0
    fail_count = 0
    file_start = time.time()
    for coro in asyncio.as_completed(tasks):
        sample_idx, labels, monitor = await coro

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
                run_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass1")
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

    file_elapsed = time.time() - file_start

    # Inherit labels for sparse-sampled slices
    for unlabeled_idx, source_idx in inherit_map.items():
        if all_labels[source_idx] is not None:
            inherited = dict(all_labels[source_idx])
            inherited["inherited"] = True
            inherited["inherited_from"] = samples[source_idx].get("id")
            all_labels[unlabeled_idx] = inherited

    # Attach labels to samples
    for idx, sample in enumerate(samples):
        sample["labels"] = all_labels[idx]
        sample.setdefault("metadata", {})["source_file"] = str(input_path)
        if all_monitors[idx]:
            sample["labeling_monitor"] = {
                "llm_calls": all_monitors[idx]["llm_calls"],
                "arbitrated": all_monitors[idx]["arbitrated"],
                "validation_issues": all_monitors[idx]["validation_issues"],
                "consistency_warnings": all_monitors[idx]["consistency_warnings"],
            }

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{file_prefix}" if file_prefix else ""

    labeled_json = f"labeled{suffix}.json"
    labeled_jsonl = f"labeled{suffix}.jsonl"
    monitor_file = f"monitor{suffix}.jsonl"
    stats_file = pass1_stats_filename(suffix)
    stats_file_legacy = pass1_stats_legacy_filename(suffix)
    dashboard_file = pass1_dashboard_filename(suffix)
    dashboard_file_legacy = pass1_dashboard_legacy_filename(suffix)
    failed_samples_file = f"failed_samples{suffix}.jsonl"

    with open(output_dir / labeled_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    with open(output_dir / labeled_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(output_dir / monitor_file, "w", encoding="utf-8") as f:
        for m in all_monitors:
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Write failed samples (original, without labels) for easy retry
    # Inherited samples have no monitor — they are not failures
    inherited_indices = set(inherit_map.keys())
    failed_indices = [
        i for i, l in enumerate(all_labels)
        if (l is None or is_partial_labels(l)) and i not in inherited_indices
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
        with open(output_dir / "failures.jsonl", "w", encoding="utf-8") as f:
            for i in failed_indices:
                m = all_monitors[i]
                record = {
                    "sample_id": samples[i].get("id", f"sample-{i}"),
                    "source_file": str(input_path),
                    "status": m["status"] if m else "timeout",
                    "error": (m.get("error", "") if m else "exceeded sample timeout"),
                    "error_response": (m.get("error_response", "")[:1000] if m else ""),
                    "attempts": (m.get("sample_attempt", 0) + 1 if m else 0),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Compute and write stats
    valid_monitors = [m for m in all_monitors if m is not None]
    stats = compute_stats(valid_monitors, all_labels, inherit_map=inherit_map)
    stats["total_elapsed_seconds"] = round(file_elapsed, 1)
    stats["input_file"] = str(input_path)
    if sparse_inherited > 0:
        stats["sparse_labeled"] = label_count
        stats["sparse_inherited"] = sparse_inherited
        # Adjust totals: inherited samples count as success
        stats["total_samples"] = total
        inherited_ok = sum(1 for i in inherited_indices if is_usable_labels(all_labels[i]))
        stats["success"] += inherited_ok
        stats["failed"] = stats["total_samples"] - stats["success"]
        stats["success_rate"] = round(stats["success"] / max(total, 1), 4)

    stats_path = output_dir / stats_file
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    sync_legacy_aliases(stats_path, [stats_file_legacy])

    # Per-file dashboard
    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        generate_dashboard(output_dir, labeled_file=labeled_json,
                           stats_file=stats_file, output_file=dashboard_file)
        sync_legacy_aliases(output_dir / dashboard_file, [dashboard_file_legacy])
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


async def run_directory_pipeline(dir_files, run_dir, model, concurrency,
                                 checkpoint_path, completed_set=None,
                                 limit=0, shuffle=False,
                                 progress=None, file_task=None, sample_task=None,
                                 http_client=None, sem=None, enable_arbitration=True,
                                 config=None, rate_limiter=None,
                                 llm_task=None, workload_estimate=None,
                                 llm_progress_cb=None):
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
            file_out_dir = run_dir / rel_path.with_suffix("")
            prefix = rel_path.stem
            existing_stats = file_out_dir / f"stats_{prefix}.json"
            if existing_stats.exists():
                with open(existing_stats, "r", encoding="utf-8") as f:
                    skipped_stats.append(json.load(f))
            if progress and file_task is not None:
                progress.update(file_task, advance=1)
        else:
            pending_files.append((i, abs_path, rel_path))

    if not pending_files:
        return skipped_stats

    # --- Helper to wrap label_one with file/sample tracking ---
    async def _tagged_label(coro, file_idx, sample_idx):
        _, labels, monitor = await coro
        return file_idx, sample_idx, labels, monitor

    # --- Load a file into a FileCollector and submit its tasks ---
    def load_and_submit(file_entry, pending_futures):
        orig_idx, abs_path, rel_path = file_entry
        rel_str = str(rel_path)
        file_out_dir = run_dir / rel_path.with_suffix("")
        prefix = rel_path.stem

        samples, n_raw = iter_samples_from_file(
            abs_path, limit=limit, shuffle=shuffle)

        # Sparse sampling
        _sparse_kw = {}
        if config:
            _sparse_kw = dict(full_label_count=config.sparse_full_label_count,
                              gap_multiplier=config.sparse_gap_multiplier,
                              min_gap=config.sparse_min_gap,
                              max_gap=config.sparse_max_gap,
                              threshold=config.sparse_threshold)
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
            label_count=label_count,
            inherit_map=inherit_map,
            sparse_info=f" ({len(samples)} total, {round(sparse_inherited/len(samples)*100)}% sparse)" if sparse_inherited > 0 else "",
        )

        # Fallback behavior: if no pre-scan estimate, grow total dynamically.
        if progress and sample_task is not None and workload_estimate is None:
            current_total = progress.tasks[sample_task].total or 0
            progress.update(sample_task, total=current_total + label_count, visible=True)

        # Submit only label_indices (shuffled to avoid convoy effect)
        submit_order = list(label_indices)
        random.shuffle(submit_order)
        for idx in submit_order:
            coro = label_one(
                http_client, samples[idx], model, idx, len(samples), sem,
                enable_arbitration=enable_arbitration,
                config=config, rate_limiter=rate_limiter,
            )
            fut = asyncio.ensure_future(_tagged_label(coro, orig_idx, idx))
            pending_futures.add(fut)

        return collector

    # --- Try to load more files if below watermark and within memory limit ---
    def maybe_load_more(pending_futures, collectors, file_queue, next_to_load):
        active_count = sum(1 for c in collectors.values() if not c.completed)
        while (next_to_load < len(file_queue)
               and len(pending_futures) < watermark
               and active_count < max_active):
            entry = file_queue[next_to_load]
            next_to_load += 1
            new_c = load_and_submit(entry, pending_futures)
            collectors[new_c.file_idx] = new_c
            active_count += 1
        return next_to_load

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

    if progress and file_task is not None:
        active_names = [str(c.rel_path) for c in collectors.values() if not c.completed]
        progress.update(file_task, info=", ".join(active_names)[:60])

    # Process results as they complete
    while pending_futures:
        done, pending_futures = await asyncio.wait(
            pending_futures, return_when=asyncio.FIRST_COMPLETED)

        for fut in done:
            file_idx, sample_idx, labels, monitor = fut.result()
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
                    global_llm_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass1")
                progress.update(sample_task, advance=1, info=info)
            if eta_tracker:
                eta_tracker.update(monitor.get("llm_calls", 0) if monitor else 0)
                if progress and llm_task is not None:
                    llm_info = eta_tracker.info_line()
                    if global_llm_info:
                        llm_info = f"{llm_info} • {global_llm_info}"
                    progress.update(
                        llm_task,
                        total=max(eta_tracker.estimated_total_calls, eta_tracker.calls_done, 1),
                        completed=eta_tracker.calls_done,
                        info=llm_info,
                    )

            # Check if this file is fully done (compare against label_count, not total)
            if c.done >= c.label_count and not c.completed:
                stats = flush_file_output(c, run_dir, checkpoint_path, pprint=pprint)
                all_file_stats.append(stats)

                if progress and file_task is not None:
                    progress.update(file_task, advance=1)

        # After processing batch of completions, check if we should load more files
        next_to_load = maybe_load_more(pending_futures, collectors, file_queue, next_to_load)

        if progress and file_task is not None:
            active_names = [str(cc.rel_path) for cc in collectors.values()
                            if not cc.completed]
            progress.update(file_task, info=", ".join(active_names)[:60] if active_names else "done")

    # Handle any files with 0 labels to submit (edge case: 0 samples or all inherited)
    for c in collectors.values():
        if not c.completed and c.label_count == 0:
            stats = flush_file_output(c, run_dir, checkpoint_path, pprint=pprint)
            all_file_stats.append(stats)
            if progress and file_task is not None:
                progress.update(file_task, advance=1)

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

    print(f"\nConfidence (mean):")
    for dim, cs in stats.get("confidence_stats", {}).items():
        bar = "█" * int(cs["mean"] * 20)
        print(f"  {dim:15s} {cs['mean']:.3f} {bar}")

    print(f"\nTop distributions:")
    for dim in ["intent", "difficulty", "domain", "concept"]:
        dist = stats.get("tag_distributions", {}).get(dim, {})
        top5 = list(dist.items())[:5]
        top_str = ", ".join(f"{k}({v})" for k, v in top5)
        print(f"  {dim:15s} {top_str}")

    if stats.get("unmapped_tags"):
        print(f"\nUnmapped tags (top 10):")
        for tag, count in list(stats["unmapped_tags"].items())[:10]:
            print(f"  {tag}: {count}")

    print(f"\nRun dir: {run_dir}")


def _write_global_summary(all_file_stats, run_dir, input_path, model, concurrency, batch_start,
                          rate_limiter=None, workload_estimate=None):
    """Write global summary stats + dashboard for a batch run."""
    batch_elapsed = time.time() - batch_start
    summary = merge_stats(all_file_stats) if all_file_stats else {
        "total_samples": 0, "distribution_total_samples": 0,
        "success": 0, "failed": 0, "success_rate": 0,
        "total_llm_calls": 0, "total_tokens": 0, "arbitrated_count": 0,
        "unmapped_unique_count": 0, "tag_distributions": {}, "unmapped_tags": {},
        "files_processed": 0,
    }
    summary["model"] = model
    summary["concurrency"] = concurrency
    summary["total_elapsed_seconds"] = round(batch_elapsed, 1)
    summary["timestamp"] = datetime.now().isoformat()
    summary["input_path"] = str(input_path)
    summary["run_dir"] = str(run_dir)
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

    summary_path = run_dir / PASS1_SUMMARY_STATS_FILE
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    sync_legacy_aliases(summary_path, [PASS1_SUMMARY_STATS_FILE_LEGACY])

    input_name = input_path.name
    global_dashboard = pass1_global_dashboard_filename(input_name)
    try:
        from sft_label.tools.visualize_labels import generate_dashboard
        generate_dashboard(run_dir, labeled_file=None,
                           stats_file=PASS1_SUMMARY_STATS_FILE,
                           output_file=global_dashboard)
        sync_legacy_aliases(
            run_dir / global_dashboard,
            [pass1_global_dashboard_legacy_filename(input_name)],
        )
        print(f"\nGlobal dashboard generated: {run_dir / global_dashboard}")
    except Exception as e:
        print(f"\nGlobal dashboard generation skipped: {e}")

    print_summary(summary, run_dir, is_batch=True)


async def run(
    input_path: str | Path,
    *,
    output: str | Path | None = None,
    resume: str | Path | None = None,
    limit: int = 0,
    shuffle: bool = False,
    enable_arbitration: bool = True,
    config: PipelineConfig | None = None,
    llm_progress_cb=None,
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

    # Create shared rate limiter (one per run, shared across all LLM calls)
    rate_limiter = AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup) if config.rps_limit > 0 else None

    # ── Resume mode ──────────────────────────────────────
    if resume:
        run_dir = Path(resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume path does not exist: {run_dir}")
        checkpoint_path = run_dir / "checkpoint.json"
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint.json in {run_dir}")
        if ckpt.get("status") == "done":
            print(f"All files already completed in {run_dir}")
            return {"status": "already_done", "run_dir": str(run_dir)}

        # Recover settings from summary_stats or checkpoint
        summary_path = find_first_existing(
            run_dir,
            [PASS1_SUMMARY_STATS_FILE, PASS1_SUMMARY_STATS_FILE_LEGACY],
        )
        if summary_path:
            with open(summary_path, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
            _input_path = Path(prev_summary.get("input_path", str(input_path)))
            _model = prev_summary.get("model", config.labeling_model)
        else:
            _input_path = Path(input_path)
            _model = config.labeling_model

        files = discover_input_files(_input_path)
        dir_files = [(a, r) for a, r in files if r is not None]
        if not dir_files:
            raise ValueError("--resume only works with directory-mode runs")

        completed = set(ckpt.get("completed", []))
        _concurrency = config.concurrency

        print(f"Pipeline RESUME | {run_dir} | model={_model} concurrency={_concurrency} | {len(completed)}/{len(dir_files)} files done")
        workload_estimate = estimate_directory_workload(
            dir_files,
            limit=limit,
            shuffle=shuffle,
            config=config,
            completed_set=completed,
            enable_arbitration=enable_arbitration,
        )
        print(
            "  Plan | "
            f"{workload_estimate.files_planned} pending files, "
            f"{workload_estimate.total_samples} samples "
            f"({workload_estimate.total_labeled_samples} labeled, {workload_estimate.total_inherited_samples} inherited), "
            f"llm~{workload_estimate.initial_estimated_llm_calls}, "
            f"scan {workload_estimate.scan_elapsed_seconds:.1f}s"
        )

        batch_start = time.time()

        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_connections=_concurrency + 10,
                max_keepalive_connections=_concurrency,
            ),
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
                    "LLM",
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
                )

        # Write global summary
        _write_global_summary(all_file_stats, run_dir, _input_path, _model, _concurrency, batch_start,
                              rate_limiter=rate_limiter, workload_estimate=workload_estimate)
        summary_path = run_dir / PASS1_SUMMARY_STATS_FILE
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Normal mode ──────────────────────────────────────
    _input_path = Path(input_path)
    files = discover_input_files(_input_path)
    is_directory = _input_path.is_dir()

    # Determine output directory
    run_dir = resolve_run_dir(str(output) if output is not None else None, _input_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    _concurrency = config.concurrency

    _arb = "off" if not enable_arbitration else f"threshold={config.confidence_threshold}"
    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    _src = f"directory, {len(files)} files" if is_directory else "single file"
    print(f"Pipeline | {_input_path} ({_src}) | model={config.labeling_model} concurrency={_concurrency} {_rps} arb={_arb}")
    print(f"  run_dir={run_dir}")

    if is_directory:
        # ── Directory mode: cross-file pipeline ──
        dir_files = [(a, r) for a, r in files if r is not None]
        if not dir_files:
            raise FileNotFoundError("No .json/.jsonl files found in directory")

        checkpoint_path = run_dir / "checkpoint.json"
        create_checkpoint(checkpoint_path, dir_files)
        workload_estimate = estimate_directory_workload(
            dir_files,
            limit=limit,
            shuffle=shuffle,
            config=config,
            completed_set=None,
            enable_arbitration=enable_arbitration,
        )
        print(
            "  Plan | "
            f"{workload_estimate.files_planned} files, "
            f"{workload_estimate.total_samples} samples "
            f"({workload_estimate.total_labeled_samples} labeled, {workload_estimate.total_inherited_samples} inherited), "
            f"llm~{workload_estimate.initial_estimated_llm_calls}, "
            f"scan {workload_estimate.scan_elapsed_seconds:.1f}s"
        )
        batch_start = time.time()

        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_connections=_concurrency + 10,
                max_keepalive_connections=_concurrency,
            ),
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
                    "LLM",
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
                )

        _write_global_summary(all_file_stats, run_dir, _input_path, config.labeling_model, _concurrency, batch_start,
                              rate_limiter=rate_limiter, workload_estimate=workload_estimate)
        summary_path = run_dir / PASS1_SUMMARY_STATS_FILE
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        # ── Single-file mode: backward compatible ────────
        batch_start = time.time()
        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_connections=_concurrency + 10,
                max_keepalive_connections=_concurrency,
            ),
        ) as http_client:
            sem = asyncio.Semaphore(_concurrency)
            with create_progress() as progress:
                sample_task = progress.add_task("Pass 1", total=None, info="starting...")
                stats = await run_one_file(
                    _input_path, run_dir, http_client, sem, config.labeling_model,
                    enable_arbitration=enable_arbitration,
                    limit=limit, shuffle=shuffle,
                    progress=progress, sample_task=sample_task,
                    config=config, rate_limiter=rate_limiter,
                    llm_progress_cb=llm_progress_cb,
                )

        stats["model"] = config.labeling_model
        stats["concurrency"] = _concurrency
        stats["timestamp"] = datetime.now().isoformat()
        stats["run_dir"] = str(run_dir)
        if rate_limiter:
            stats["http_request_stats"] = rate_limiter.stats.to_dict()

        # Overwrite stats with enriched version
        stats_path = run_dir / PASS1_STATS_FILE
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        sync_legacy_aliases(stats_path, [PASS1_STATS_FILE_LEGACY])

        print_summary(stats, run_dir)
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
    )


def main():
    parser = argparse.ArgumentParser(description="SFT Auto-Labeling Pipeline (Concurrent)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory: omit for sibling of input, or an explicit path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing run directory (reads checkpoint.json)")
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
