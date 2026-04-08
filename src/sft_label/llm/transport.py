from __future__ import annotations

import asyncio
import json
import random
import time

import httpx

from sft_label.config import (
    LITELLM_BASE,
    LITELLM_KEY,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    REQUEST_TIMEOUT_ESCALATION,
)


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
        parts = [f"\u2713{self.success}"]
        for code in sorted(self.errors):
            parts.append(f"{code}\u00d7{self.errors[code]}")
        if self.timeouts:
            parts.append(f"timeout\u00d7{self.timeouts}")
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
            if float(attempt_timeout or 0) > 0:
                resp = await asyncio.wait_for(
                    http_client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=attempt_timeout,
                    ),
                    timeout=float(attempt_timeout),
                )
            else:
                resp = await http_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=attempt_timeout,
                )
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
            if resp.status_code == 402:
                # Billing exhausted — not retryable (same as 401)
                error_text = resp.text[:300]
                return None, f"HTTP 402: {error_text}", {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": f"HTTP 402 (billing): {error_text}",
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
