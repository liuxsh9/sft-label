"""
Conversation-Level Value Aggregation (Post-scoring, no LLM)

Aggregates per-turn slice scores into conversation-level metrics for
multi-turn conversations. Enables conversation-as-a-unit filtering.

Runs after Pass 2 scoring. Pure computation — no LLM calls, no async.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from sft_label.config import (
    CONV_CONFIDENCE_INHERITED,
    CONV_QUALITY_PENALTIES,
    CONV_QUALITY_PENALTY_DEFAULT,
    CONV_FLAG_PENALTY_BASE,
    CONV_AGENTIC_QUALITY_PERCENTILE,
    CONV_RARITY_MEAN_WEIGHT,
    CONV_RARITY_PEAK_WEIGHT,
    CONV_RARITY_DIVERSITY_BONUS,
    CONV_COVERAGE_CONFIDENCE_FLOOR,
    VALUE_WEIGHTS,
    RARITY_WEIGHTS,
    SELECTION_INTRA_WEIGHT,
    SELECTION_QUALITY_WEIGHT,
    SELECTION_MIN_GROUP_SIZE,
    SELECTION_SMOOTHING_PRIOR,
    KNOWN_FLAGS_NEGATIVE,
    ENABLE_CONVERSATION_V2,
    CONV_V2_TURN_THRESHOLD,
    CONV_V2_TOOL_THRESHOLD,
    CONV_V2_TOP_K,
    CONV_V2_BOTTOM_K,
    CONV_V2_HIGH_VALUE_THRESHOLD,
    CONV_V2_LOW_QUALITY_THRESHOLD,
    CONV_V2_VALUE_UPLIFT_CAP,
    CONV_V2_SELECTION_UPLIFT_CAP,
    CONV_V2_DOWNSIDE_CAP,
)
from sft_label.labels import LABEL_META_KEYS
from sft_label.preprocessing import build_conversation_uid
from sft_label.score_confidence import apply_score_confidence, score_confidence


def build_conversation_key(source_id, source_file=None, conversation_uid=None):
    """Build a stable conversation key across files."""
    if conversation_uid:
        return conversation_uid
    if not source_id:
        return None
    if source_file:
        return f"{source_file}::{source_id}"
    return source_id


def sample_conversation_key(sample):
    """Build the conversation key for a scored sample."""
    meta = sample.get("metadata") or {}
    conversation_uid = meta.get("conversation_uid")
    if conversation_uid:
        return conversation_uid
    if is_trajectory_object(meta):
        return build_conversation_uid(
            sample,
            source_file=meta.get("source_file"),
            source_row=meta.get("source_row"),
        )
    return build_conversation_key(
        meta.get("source_id"),
        meta.get("source_file"),
        None,
    )


def is_trajectory_object(meta):
    """Whether metadata marks this sample as a single-slice trajectory object."""
    return bool((meta or {}).get("trajectory_object"))


def is_conversation_object(meta):
    """Whether metadata should be treated as a conversation-level unit."""
    meta = meta or {}
    if not meta.get("source_id"):
        return False
    if meta.get("total_turns", 1) > 1:
        return True
    return is_trajectory_object(meta)


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


# ─── Grouping ────────────────────────────────────────────

def group_by_conversation(samples):
    """Group samples by conversation-like units.

    Returns dict {conversation_key: [slices sorted by turn_index]}.
    Includes regular multi-turn slices and trajectory objects.
    """
    groups = defaultdict(list)
    for s in samples:
        meta = s.get("metadata") or {}
        conv_key = sample_conversation_key(s)
        if conv_key and is_conversation_object(meta):
            groups[conv_key].append(s)

    # Sort each group by turn_index
    for conv_key in groups:
        groups[conv_key].sort(
            key=lambda s: (s.get("metadata") or {}).get("turn_index", 0)
        )

    return dict(groups)


# ─── Weighting helpers ───────────────────────────────────

def _position_weight(i, n):
    """Linear position weight: 1.0 for first turn → 2.0 for last turn.

    Args:
        i: 0-based turn index within the conversation
        n: total number of slices in the conversation
    """
    if n <= 1:
        return 1.0
    return 1.0 + 1.0 * i / (n - 1)


def _effective_weights(slices):
    """Compute effective weight per slice: position_weight × confidence.

    Inherited slices get reduced confidence (CONV_CONFIDENCE_INHERITED).
    """
    n = len(slices)
    weights = []
    for i, s in enumerate(slices):
        pw = _position_weight(i, n)
        labels = s.get("labels") or {}
        conf = CONV_CONFIDENCE_INHERITED if labels.get("inherited") else 1.0
        weights.append(pw * conf)
    return weights


def _position_weights(slices):
    """Return raw position weights without inherited confidence discount."""
    n = len(slices)
    return [_position_weight(i, n) for i, _s in enumerate(slices)]


def _weighted_average(slices, weights, extract_fn):
    """Generic weighted average over slices, skipping None values."""
    total_w = 0.0
    total_v = 0.0
    for s, w in zip(slices, weights):
        val = extract_fn(s)
        if val is not None:
            total_w += w
            total_v += w * val
    if total_w <= 0:
        return None
    return total_v / total_w


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mu = _mean(values) or 0.0
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _top_k_mean(values, k):
    if not values:
        return None
    top = sorted(values, reverse=True)[:max(1, min(k, len(values)))]
    return _mean(top)


def _bottom_k_mean(values, k):
    if not values:
        return None
    bottom = sorted(values)[:max(1, min(k, len(values)))]
    return _mean(bottom)


def _round_or_none(value, digits=2):
    if value is None:
        return None
    return round(value, digits)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _scale_01_to_10(value):
    return 1.0 + 9.0 * _clamp(value, 0.0, 1.0)


def _apply_v2_guardrail(candidate, baseline, uplift_cap, downside_cap):
    lower = max(1.0, baseline - downside_cap)
    upper = min(10.0, baseline + uplift_cap)
    return _clamp(candidate, lower, upper)


def _extract_turn_metrics(slices):
    """Compute additive diagnostics for turn-level value compression."""
    value_scores = []
    quality_scores = []
    for s in slices:
        value = s.get("value") or {}
        score = value.get("value_score")
        if score is not None:
            value_scores.append(score)
        quality = (value.get("quality") or {}).get("overall")
        if quality is not None:
            quality_scores.append(quality)

    if not value_scores:
        return {}

    value_mean = _mean(value_scores)
    value_max = max(value_scores)
    top_k_mean = _top_k_mean(value_scores, 5)
    bottom_k_mean = _bottom_k_mean(value_scores, 3)

    k = min(5, max(1, len(value_scores) // 4))
    early_mean = _mean(value_scores[:k])
    late_mean = _mean(value_scores[-k:])

    return {
        "turn_value_mean": value_mean,
        "turn_value_min": min(value_scores),
        "turn_value_max": value_max,
        "turn_value_median": _median(value_scores),
        "turn_value_std": _std(value_scores),
        "turn_quality_std": _std(quality_scores) if quality_scores else None,
        "top_k_mean": top_k_mean,
        "bottom_k_mean": bottom_k_mean,
        "peak_minus_mean": (value_max - value_mean) if value_mean is not None else None,
        "late_turn_gain": (
            (late_mean - early_mean)
            if early_mean is not None and late_mean is not None
            else None
        ),
    }


def _compute_conv_turn_signal(slices):
    """Compress per-turn values into a trajectory-friendly signal.

    Top turns dominate, but bottom turns still add mild downside pressure.
    """
    value_scores = []
    quality_scores = []
    for s in slices:
        value = s.get("value") or {}
        score = value.get("value_score")
        if score is not None:
            value_scores.append(score)
        quality = (value.get("quality") or {}).get("overall")
        if quality is not None:
            quality_scores.append(quality)

    if not value_scores:
        return None, {}

    top_k_mean = _top_k_mean(value_scores, CONV_V2_TOP_K)
    bottom_k_mean = _bottom_k_mean(value_scores, CONV_V2_BOTTOM_K)
    median_score = _median(value_scores)

    raw_signal = (
        0.70 * (top_k_mean or median_score or 5.5)
        + 0.20 * (median_score or top_k_mean or 5.5)
        + 0.10 * (bottom_k_mean or median_score or 5.5)
    )

    low_quality_ratio = 0.0
    if quality_scores:
        low_quality_ratio = (
            sum(1 for q in quality_scores if q < CONV_V2_LOW_QUALITY_THRESHOLD)
            / len(quality_scores)
        )

    if low_quality_ratio <= 0.15:
        tail_penalty = 1.0
    elif low_quality_ratio <= 0.30:
        tail_penalty = 0.95
    else:
        tail_penalty = 0.90

    high_value_ratio = (
        sum(1 for score in value_scores if score >= CONV_V2_HIGH_VALUE_THRESHOLD)
        / len(value_scores)
    )
    signal = raw_signal * tail_penalty
    return signal, {
        "conv_turn_signal_raw": raw_signal,
        "conv_turn_signal": signal,
        "conv_turn_signal_tail_penalty": tail_penalty,
        "high_value_turn_ratio": high_value_ratio,
        "low_quality_turn_ratio": low_quality_ratio,
        "turn_value_median": median_score,
    }


def _slice_thinking_mode(sample):
    value = sample.get("value") or {}
    meta = sample.get("metadata") or {}
    mode = value.get("thinking_mode") or meta.get("thinking_mode")
    if mode in {"fast", "slow"}:
        return mode
    return None


def _summarize_thinking_modes(slices):
    counts = {"fast": 0, "slow": 0, "unknown": 0}
    for sample in slices:
        mode = _slice_thinking_mode(sample)
        if mode == "fast":
            counts["fast"] += 1
        elif mode == "slow":
            counts["slow"] += 1
        else:
            counts["unknown"] += 1

    total = len(slices)
    ratio = {
        key: (count / total if total else 0.0)
        for key, count in counts.items()
    }

    if total > 0 and counts["fast"] == total:
        summary = "all_fast"
        thinking_mode = "fast"
    elif total > 0 and counts["slow"] == total:
        summary = "all_slow"
        thinking_mode = "slow"
    elif counts["fast"] == 0 and counts["slow"] == 0:
        summary = "unknown"
        thinking_mode = None
    else:
        summary = "mixed"
        thinking_mode = "mixed"

    return thinking_mode, summary, counts, ratio


_TEST_TEXT_RE = re.compile(
    r"\b("
    r"pytest|unittest|go\s+test|cargo\s+test|npm\s+test|pnpm\s+test|yarn\s+test|"
    r"vitest|jest|mocha|ctest|gradle\s+test|mvn\s+test|phpunit|rspec|tox|nosetests"
    r"|test|tests|testing"
    r")\b",
    re.IGNORECASE,
)
_EDIT_TEXT_RE = re.compile(
    r"("
    r"apply_patch|str_replace_editor|write_file|cat\s+>\s+|tee\s+|patch\b|"
    r"diff --git|\*\*\* Update File:|\*\*\* Add File:|\*\*\* Delete File:|"
    r"sed\s+-i\b|perl\s+-pi\b|\bpatched?\b"
    r")",
    re.IGNORECASE,
)
_BASH_TOOL_NAMES = {
    "bash", "execute_bash", "shell", "terminal", "command",
    "run_shell_command", "run_command",
}
_PATH_RE = re.compile(r"(?<![\w.-])((?:\.{1,2}/|/)?(?:[\w.-]+/)+[\w.-]+)")
_BASENAME_RE = re.compile(
    r"(?<![/\w.-])([\w.-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|scala|rb|php|"
    r"c|cc|cpp|h|hpp|cs|swift|m|mm|sql|ya?ml|json|toml|ini|cfg|conf|md|sh|zsh|"
    r"bash|ps1|xml|html|css|gradle))(?![\w.-])",
    re.IGNORECASE,
)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_CLARIFICATION_TEXT_RE = re.compile(
    r"\b("
    r"should\s+i|do\s+you\s+want|would\s+you\s+like|can\s+you\s+confirm|"
    r"can\s+you\s+clarify|which\s+one|which\s+should|is\s+it\s+okay|"
    r"before\s+i\s+(?:change|proceed|update)|confirm\s+whether"
    r")\b",
    re.IGNORECASE,
)
_FAILURE_TEXT_RE = re.compile(
    r"\b("
    r"fail(?:ed|ure)?|error|traceback|exception|assertionerror|keyerror|"
    r"typeerror|valueerror|syntaxerror|runtimeerror|no such file|not found"
    r")\b",
    re.IGNORECASE,
)
_SUCCESS_TEXT_RE = re.compile(
    r"\b("
    r"pass(?:ed|es)?|success(?:ful|fully)?|succeeded|fixed|resolved|"
    r"updated|patched|written|created|green|verified|done|completed"
    r")\b",
    re.IGNORECASE,
)
_VERIFICATION_TEXT_RE = re.compile(
    r"\b("
    r"verify|verified|verification|confirm|confirmed|assert|assertion|"
    r"pytest|test|tests|testing|green|pass(?:ed|es)?"
    r")\b",
    re.IGNORECASE,
)


def _normalize_role(turn):
    role = str(turn.get("role") or turn.get("from") or "").strip().lower()
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    return role


def _coerce_turn_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or value.get("value") or "")
    return str(value)


def _extract_tool_names_from_text(text):
    names = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        if not name and isinstance(payload.get("function"), dict):
            name = payload["function"].get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _extract_tool_payloads(text):
    payloads = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _tool_signature(payload, fallback_name=""):
    name = ""
    if isinstance(payload, dict):
        name = payload.get("name") or ""
        if not name and isinstance(payload.get("function"), dict):
            name = payload["function"].get("name") or ""
    name = str(name or fallback_name or "").strip()
    if not name:
        return ""

    arguments = {}
    if isinstance(payload, dict):
        arguments = (
            payload.get("arguments")
            or payload.get("parameters")
            or payload.get("args")
            or {}
        )

    if isinstance(arguments, dict):
        compact = {
            key: arguments[key]
            for key in (
                "command",
                "path",
                "file_path",
                "file",
                "target_file",
                "target",
                "uri",
                "url",
                "q",
                "query",
            )
            if key in arguments
        }
        arg_value = compact or arguments
    else:
        arg_value = arguments

    try:
        encoded = json.dumps(arg_value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        encoded = str(arg_value)
    return f"{name}:{encoded}"


def _extract_file_refs(text):
    refs = set()
    for pattern in (_PATH_RE, _BASENAME_RE):
        for match in pattern.finditer(text or ""):
            candidate = str(match.group(1)).strip("`'\"()[]{}:;,")
            if not candidate or candidate in {".", "..", "/"}:
                continue
            refs.add(candidate)
    return refs


def _conversation_from_slices(slices):
    for s in reversed(slices):
        turns = s.get("conversations")
        if isinstance(turns, list) and turns:
            return turns
    return []


def _normalize_freeform_text(text):
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized.strip(" .;:,!?")


def _extract_structure_signals(slices):
    """Approximate deterministic structure metrics from the final full trajectory."""
    if not slices:
        return {}

    full_conversation = _conversation_from_slices(slices)

    if not full_conversation:
        return {
            "tool_turn_count": 0,
            "tool_turn_ratio": 0.0,
            "unique_tool_count": 0,
            "unique_tools": [],
            "unique_file_count": 0,
            "test_related_turn_count": 0,
            "edit_related_turn_count": 0,
            "bash_execution_turn_count": 0,
            "assistant_turn_count": 0,
            "final_turn_role": "",
            "final_turn_text": "",
        }

    tool_turn_count = 0
    test_related_turn_count = 0
    edit_related_turn_count = 0
    bash_execution_turn_count = 0
    assistant_turn_count = 0
    unique_tools = set()
    unique_files = set()
    final_turn_role = ""
    final_turn_text = ""

    for turn in full_conversation:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(turn)
        text = _coerce_turn_text(
            turn.get("value", turn.get("content"))
        )
        final_turn_role = role
        final_turn_text = text
        tool_name = str(turn.get("name") or turn.get("tool_name") or "").strip()
        tool_names = set(_extract_tool_names_from_text(text))
        if tool_name:
            tool_names.add(tool_name)

        is_tool_turn = (role == "tool") or bool(tool_names)
        if is_tool_turn:
            tool_turn_count += 1
        if role == "assistant":
            assistant_turn_count += 1

        unique_tools.update(name for name in tool_names if name)
        unique_files.update(_extract_file_refs(text))

        lower = text.lower()
        if _TEST_TEXT_RE.search(lower):
            test_related_turn_count += 1
        if _EDIT_TEXT_RE.search(lower):
            edit_related_turn_count += 1

        if any(name.lower() in _BASH_TOOL_NAMES for name in tool_names) or text.lstrip().startswith("$ "):
            bash_execution_turn_count += 1

    turn_total = len(full_conversation) or 1
    return {
        "tool_turn_count": tool_turn_count,
        "tool_turn_ratio": tool_turn_count / turn_total,
        "unique_tool_count": len(unique_tools),
        "unique_tools": sorted(unique_tools),
        "unique_file_count": len(unique_files),
        "test_related_turn_count": test_related_turn_count,
        "edit_related_turn_count": edit_related_turn_count,
        "bash_execution_turn_count": bash_execution_turn_count,
        "assistant_turn_count": assistant_turn_count,
        "final_turn_role": final_turn_role,
        "final_turn_text": final_turn_text,
    }


def _extract_trajectory_quality_features(slices):
    """Extract additive diagnostic-only quality features from the final trajectory."""
    full_conversation = _conversation_from_slices(slices)
    if not full_conversation:
        return {
            "clarification_turn_count": 0,
            "clarification_ratio": 0.0,
            "tool_call_count": 0,
            "tool_result_count": 0,
            "tool_success_count": 0,
            "tool_failure_count": 0,
            "tool_result_success_ratio": 0.0,
            "redundant_tool_call_count": 0,
            "tool_repeat_ratio": 0.0,
            "assistant_repeat_turn_count": 0,
            "recovery_attempt_count": 0,
            "recovery_success_count": 0,
            "recovery_success_ratio": 0.0,
            "verification_turn_count": 0,
            "verification_success_count": 0,
            "verification_after_edit_count": 0,
            "has_final_verification": False,
        }

    clarification_turn_count = 0
    tool_call_count = 0
    tool_result_count = 0
    tool_success_count = 0
    tool_failure_count = 0
    redundant_tool_call_count = 0
    assistant_repeat_turn_count = 0
    verification_turn_count = 0
    verification_success_count = 0
    verification_after_edit_count = 0
    edit_open = False

    seen_tool_signatures = set()
    seen_assistant_texts = set()
    pending_recovery = 0
    recovery_success_count = 0

    for turn in full_conversation:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(turn)
        text = _coerce_turn_text(turn.get("value", turn.get("content")))
        lower = text.lower()
        tool_name = str(turn.get("name") or turn.get("tool_name") or "").strip()
        tool_payloads = _extract_tool_payloads(text)
        tool_signatures = [
            _tool_signature(payload, fallback_name=tool_name)
            for payload in tool_payloads
        ]
        tool_signatures = [signature for signature in tool_signatures if signature]
        if not tool_signatures and role == "assistant" and tool_name:
            tool_signatures = [_tool_signature({}, fallback_name=tool_name)]

        is_edit_turn = bool(_EDIT_TEXT_RE.search(lower)) or any(
            signature.startswith(("apply_patch:", "str_replace_editor:", "write_file:"))
            for signature in tool_signatures
        )
        is_verification_turn = bool(_VERIFICATION_TEXT_RE.search(lower) or _TEST_TEXT_RE.search(lower))

        if role == "assistant" and "?" in text and _CLARIFICATION_TEXT_RE.search(lower):
            clarification_turn_count += 1

        if role == "assistant" and not tool_signatures:
            normalized_text = _normalize_freeform_text(text)
            if normalized_text:
                if normalized_text in seen_assistant_texts:
                    assistant_repeat_turn_count += 1
                else:
                    seen_assistant_texts.add(normalized_text)

        if role == "assistant" and tool_signatures:
            tool_call_count += len(tool_signatures)
            for signature in tool_signatures:
                if signature in seen_tool_signatures:
                    redundant_tool_call_count += 1
                else:
                    seen_tool_signatures.add(signature)

        if role == "tool":
            tool_result_count += 1
            is_failure = bool(_FAILURE_TEXT_RE.search(lower))
            is_success = bool(_SUCCESS_TEXT_RE.search(lower)) and not is_failure
            if is_failure:
                tool_failure_count += 1
                pending_recovery += 1
            elif is_success:
                tool_success_count += 1
                if pending_recovery > 0:
                    recovery_success_count += pending_recovery
                    pending_recovery = 0

        if is_edit_turn and pending_recovery > 0:
            recovery_success_count += pending_recovery
            pending_recovery = 0
            edit_open = True
        elif is_edit_turn:
            edit_open = True

        if is_verification_turn:
            verification_turn_count += 1
            if bool(_SUCCESS_TEXT_RE.search(lower)) and not bool(_FAILURE_TEXT_RE.search(lower)):
                verification_success_count += 1
            if edit_open:
                verification_after_edit_count += 1
                edit_open = False

    turn_total = len(full_conversation) or 1
    recovery_attempt_count = tool_failure_count
    final_turn_text = _coerce_turn_text(
        (full_conversation[-1] or {}).get("value", (full_conversation[-1] or {}).get("content"))
    )
    final_lower = final_turn_text.lower()
    has_final_verification = bool(_VERIFICATION_TEXT_RE.search(final_lower) and _SUCCESS_TEXT_RE.search(final_lower))

    return {
        "clarification_turn_count": clarification_turn_count,
        "clarification_ratio": clarification_turn_count / turn_total,
        "tool_call_count": tool_call_count,
        "tool_result_count": tool_result_count,
        "tool_success_count": tool_success_count,
        "tool_failure_count": tool_failure_count,
        "tool_result_success_ratio": (
            tool_success_count / tool_result_count if tool_result_count else 0.0
        ),
        "redundant_tool_call_count": redundant_tool_call_count,
        "tool_repeat_ratio": (
            redundant_tool_call_count / tool_call_count if tool_call_count else 0.0
        ),
        "assistant_repeat_turn_count": assistant_repeat_turn_count,
        "recovery_attempt_count": recovery_attempt_count,
        "recovery_success_count": min(recovery_success_count, recovery_attempt_count),
        "recovery_success_ratio": (
            min(recovery_success_count, recovery_attempt_count) / recovery_attempt_count
            if recovery_attempt_count
            else 0.0
        ),
        "verification_turn_count": verification_turn_count,
        "verification_success_count": verification_success_count,
        "verification_after_edit_count": verification_after_edit_count,
        "has_final_verification": has_final_verification,
    }


def _compute_trajectory_structure_score(
    slices,
    merged_labels,
    structure_signals,
    turn_metrics,
    turn_signal_detail,
    turn_count_hint=None,
):
    """Estimate conversation-level trajectory quality without extra LLM calls."""
    turn_count = len(slices) or 1
    if isinstance(turn_count_hint, int) and turn_count_hint > 0:
        turn_count = max(turn_count, turn_count_hint)
    unique_tool_count = structure_signals.get("unique_tool_count", 0) or 0
    tool_turn_ratio = structure_signals.get("tool_turn_ratio", 0.0) or 0.0
    unique_file_count = structure_signals.get("unique_file_count", 0) or 0
    test_related_turn_count = structure_signals.get("test_related_turn_count", 0) or 0
    edit_related_turn_count = structure_signals.get("edit_related_turn_count", 0) or 0
    final_turn_text = structure_signals.get("final_turn_text", "") or ""

    high_value_turn_ratio = turn_signal_detail.get("high_value_turn_ratio", 0.0) or 0.0
    low_quality_turn_ratio = turn_signal_detail.get("low_quality_turn_ratio", 0.0) or 0.0
    late_turn_gain = max(0.0, turn_metrics.get("late_turn_gain") or 0.0)
    peak_minus_mean = max(0.0, turn_metrics.get("peak_minus_mean") or 0.0)
    final_quality = ((slices[-1].get("value") or {}).get("quality") or {}).get("overall")
    final_value = (slices[-1].get("value") or {}).get("value_score")

    edit_ratio = min(1.0, edit_related_turn_count / max(turn_count, 1))
    test_ratio = min(1.0, test_related_turn_count / max(turn_count, 1))
    file_ratio = min(1.0, unique_file_count / 4.0)
    tool_diversity = min(1.0, unique_tool_count / 3.0)
    late_gain_norm = min(1.0, late_turn_gain / 2.5)
    compression_norm = min(1.0, peak_minus_mean / 2.5)
    final_quality_norm = min(1.0, (final_quality or final_value or 5.5) / 10.0)
    oversaturation = _clamp((tool_turn_ratio - 0.70) / 0.30, 0.0, 1.0)
    evidence_density = min(1.0, 0.45 * edit_ratio + 0.35 * test_ratio + 0.20 * file_ratio)
    final_lower = final_turn_text.lower()
    resolution_keywords = (
        "fixed", "resolved", "pass", "passed", "green", "success",
        "completed", "done", "implemented", "updated",
    )
    resolution_hint = 1.0 if any(token in final_lower for token in resolution_keywords) else 0.0

    progress_raw = _clamp(
        0.40 * high_value_turn_ratio
        + 0.25 * late_gain_norm
        + 0.20 * edit_ratio
        + 0.15 * compression_norm,
        0.0,
        1.0,
    )

    if low_quality_turn_ratio <= 0.10:
        recovery_raw = _clamp(0.65 + 0.20 * test_ratio + 0.15 * edit_ratio, 0.0, 1.0)
    else:
        recovery_raw = _clamp(
            0.35 * late_gain_norm + 0.35 * test_ratio + 0.30 * edit_ratio,
            0.0,
            1.0,
        )

    if unique_tool_count <= 0:
        tool_effectiveness_raw = _clamp(0.45 + 0.35 * evidence_density + 0.20 * (1.0 - oversaturation), 0.0, 1.0)
    else:
        tool_effectiveness_raw = _clamp(
            0.45 * tool_diversity + 0.30 * evidence_density + 0.25 * (1.0 - oversaturation),
            0.0,
            1.0,
        )

    resolution_raw = _clamp(
        0.45 * final_quality_norm
        + 0.25 * evidence_density
        + 0.15 * late_gain_norm
        + 0.15 * resolution_hint,
        0.0,
        1.0,
    )

    non_redundancy_raw = _clamp(
        0.45 * high_value_turn_ratio
        + 0.35 * (1.0 - low_quality_turn_ratio)
        + 0.20 * (1.0 - oversaturation),
        0.0,
        1.0,
    )

    progress_score = _scale_01_to_10(progress_raw)
    recovery_score = _scale_01_to_10(recovery_raw)
    tool_effectiveness_score = _scale_01_to_10(tool_effectiveness_raw)
    resolution_score = _scale_01_to_10(resolution_raw)
    non_redundancy_score = _scale_01_to_10(non_redundancy_raw)

    trajectory_structure_score = (
        0.25 * progress_score
        + 0.20 * recovery_score
        + 0.20 * tool_effectiveness_score
        + 0.20 * resolution_score
        + 0.15 * non_redundancy_score
    )

    trajectory_v2_enabled = (
        ENABLE_CONVERSATION_V2 and (
            turn_count >= CONV_V2_TURN_THRESHOLD
            or unique_tool_count >= CONV_V2_TOOL_THRESHOLD
            or bool((merged_labels or {}).get("agentic"))
        )
    )

    return trajectory_structure_score, {
        "trajectory_v2_enabled": trajectory_v2_enabled,
        "progress_score": progress_score,
        "recovery_score": recovery_score,
        "tool_effectiveness_score": tool_effectiveness_score,
        "resolution_score": resolution_score,
        "non_redundancy_score": non_redundancy_score,
    }


# ─── Penalty computation ─────────────────────────────────

def _compute_penalty(slices):
    """Compute quality penalty from floor quality and negative flags.

    Returns (penalty, quality_floor, negative_flags_list).
    """
    # Find quality floor (min quality across slices)
    quality_values = []
    for s in slices:
        v = s.get("value") or {}
        q = (v.get("quality") or {}).get("overall")
        if q is not None:
            quality_values.append(q)

    quality_floor = min(quality_values) if quality_values else None

    # For conversations with agentic behaviors, use p10 quality instead of min.
    # Agent trajectories naturally have low-quality "error" turns (tool returns
    # error, exploratory dead ends) that are normal exploration, not quality failures.
    has_agentic = any(
        (s.get("labels") or {}).get("agentic")
        for s in slices
    )
    if has_agentic and len(quality_values) >= 5:
        sorted_q = sorted(quality_values)
        p10_idx = max(0, int(len(sorted_q) * CONV_AGENTIC_QUALITY_PERCENTILE))
        quality_floor = sorted_q[p10_idx]

    # Quality floor penalty
    q_penalty = CONV_QUALITY_PENALTY_DEFAULT
    if quality_floor is not None:
        for threshold in sorted(CONV_QUALITY_PENALTIES.keys()):
            if quality_floor < threshold:
                q_penalty = CONV_QUALITY_PENALTIES[threshold]
                break

    # Collect unique negative flags across all slices
    neg_flags = set()
    for s in slices:
        v = s.get("value") or {}
        flags = v.get("flags") or []
        for f in flags:
            if f in KNOWN_FLAGS_NEGATIVE:
                neg_flags.add(f)

    neg_flags_list = sorted(neg_flags)

    # Flag penalty: 0.95 ^ count
    flag_penalty = CONV_FLAG_PENALTY_BASE ** len(neg_flags_list)

    penalty = q_penalty * flag_penalty
    return penalty, quality_floor, neg_flags_list


# ─── Label merging ───────────────────────────────────────

_SINGLE_SELECT_DIMS = {"intent", "difficulty", "context"}
_MULTI_SELECT_DIMS = {"language", "domain", "task", "concept", "agentic", "constraint"}
_LABEL_META_KEYS = LABEL_META_KEYS


def _merge_labels(slices):
    """Merge labels across slices.

    Single-select dims: last slice wins.
    Multi-select dims: union of all tags.
    """
    merged = {}
    for s in slices:
        labels = s.get("labels") or {}
        for dim, val in labels.items():
            if dim in _LABEL_META_KEYS:
                continue
            if dim in _SINGLE_SELECT_DIMS:
                merged[dim] = val
            elif dim in _MULTI_SELECT_DIMS:
                existing = merged.get(dim, [])
                if not isinstance(existing, list):
                    existing = [existing]
                if isinstance(val, list):
                    for v in val:
                        if v not in existing:
                            existing.append(v)
                elif val not in existing:
                    existing.append(val)
                merged[dim] = existing
            else:
                # Unknown dim: last-wins
                merged[dim] = val
    return merged


def _label_signature(labels):
    """Build a stable label-state signature for trajectory diversity."""
    if not isinstance(labels, dict):
        return ()

    parts = []
    for dim in sorted(_SINGLE_SELECT_DIMS | _MULTI_SELECT_DIMS):
        if dim in _SINGLE_SELECT_DIMS:
            value = labels.get(dim)
            if isinstance(value, str) and value:
                parts.append((dim, value))
            continue

        value = labels.get(dim)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        cleaned = tuple(sorted({item for item in value if isinstance(item, str) and item}))
        if cleaned:
            parts.append((dim, cleaned))

    return tuple(parts)


def _coverage_metrics(slices):
    """Estimate how much of a conversation is directly labeled vs inherited."""
    if not slices:
        return {
            "observed_turns": 0,
            "inherited_turns": 0,
            "observed_turn_ratio": 0.0,
            "inherited_turn_ratio": 0.0,
            "observed_weight_ratio": 0.0,
            "coverage_confidence": CONV_COVERAGE_CONFIDENCE_FLOOR,
        }

    raw_weights = _position_weights(slices)
    total_weight = sum(raw_weights) or 1.0

    observed_turns = 0
    inherited_turns = 0
    observed_weight = 0.0

    for s, raw_weight in zip(slices, raw_weights):
        labels = s.get("labels") or {}
        if labels.get("inherited"):
            inherited_turns += 1
        else:
            observed_turns += 1
            observed_weight += raw_weight

    observed_turn_ratio = observed_turns / len(slices)
    observed_weight_ratio = observed_weight / total_weight
    coverage_confidence = (
        CONV_COVERAGE_CONFIDENCE_FLOOR
        + (1.0 - CONV_COVERAGE_CONFIDENCE_FLOOR) * observed_weight_ratio
    )

    return {
        "observed_turns": observed_turns,
        "inherited_turns": inherited_turns,
        "observed_turn_ratio": observed_turn_ratio,
        "inherited_turn_ratio": inherited_turns / len(slices),
        "observed_weight_ratio": observed_weight_ratio,
        "coverage_confidence": min(1.0, max(0.0, coverage_confidence)),
    }


def _compute_conv_rarity(slices, weights, score_conf):
    """Aggregate conversation rarity with diversity and inheritance shrinkage."""
    rarity_values = []
    for s, weight in zip(slices, weights):
        rarity = ((s.get("value") or {}).get("rarity") or {}).get("score")
        if rarity is None:
            continue
        labels = s.get("labels") or {}
        rarity_values.append((rarity, weight, bool(labels.get("inherited"))))

    if not rarity_values:
        return None, {}

    total_weight = sum(weight for _rarity, weight, _inherited in rarity_values) or 1.0
    rarity_mean = sum(rarity * weight for rarity, weight, _inherited in rarity_values) / total_weight

    observed_rarities = [rarity for rarity, _weight, inherited in rarity_values if not inherited]
    rarity_peak = max(observed_rarities) if observed_rarities else max(rarity for rarity, _weight, _inherited in rarity_values)

    signatures = {
        _label_signature(s.get("labels") or {})
        for s in slices
        if not (s.get("labels") or {}).get("inherited")
    }
    signatures.discard(())
    signature_count = len(signatures)
    transition_scale = max(len(slices) - 1, 1)
    diversity_ratio = min(1.0, max(0.0, (signature_count - 1) / transition_scale))
    diversity_bonus = CONV_RARITY_DIVERSITY_BONUS * diversity_ratio

    coverage = _coverage_metrics(slices)
    rarity_conf = min(
        1.0,
        0.5 * coverage["coverage_confidence"] + 0.5 * max(0.0, min(1.0, score_conf or 1.0)),
    )

    raw_rarity = (
        CONV_RARITY_MEAN_WEIGHT * rarity_mean
        + CONV_RARITY_PEAK_WEIGHT * rarity_peak
        + diversity_bonus
    )
    conv_rarity = apply_score_confidence(min(10.0, raw_rarity), rarity_conf)
    conv_rarity = max(1.0, min(10.0, conv_rarity))

    return conv_rarity, {
        "rarity_mean": rarity_mean,
        "rarity_peak": rarity_peak,
        "rarity_diversity_bonus": diversity_bonus,
        "label_signature_count": signature_count,
        "rarity_confidence": rarity_conf,
        **coverage,
    }


def _compute_conv_extension_v2_rarity(slices, weights, score_conf, legacy_conv_rarity):
    """Aggregate extension-aware conversation rarity from gated sample-level rarity_v2 deltas."""
    rarity_v2_bonus_values = []
    saw_sample_v2 = False
    for s, weight in zip(slices, weights):
        value = s.get("value") or {}
        rarity_v2 = (value.get("rarity_v2") or {}).get("score")
        if not isinstance(rarity_v2, (int, float)):
            continue
        saw_sample_v2 = True
        core_rarity = (value.get("rarity") or {}).get("score")
        if not isinstance(core_rarity, (int, float)):
            continue
        bonus = max(0.0, float(rarity_v2) - float(core_rarity))
        rarity_v2_bonus_values.append((bonus, weight))

    if not saw_sample_v2:
        return None
    if not rarity_v2_bonus_values:
        return legacy_conv_rarity

    total_weight = sum(weight for _bonus, weight in rarity_v2_bonus_values) or 1.0
    bonus_mean = sum(bonus * weight for bonus, weight in rarity_v2_bonus_values) / total_weight
    bonus_peak = max(bonus for bonus, _weight in rarity_v2_bonus_values)
    bonus_confidence = max(0.0, min(1.0, score_conf or 1.0))
    conv_bonus = (0.7 * bonus_mean + 0.3 * bonus_peak) * bonus_confidence

    if not isinstance(legacy_conv_rarity, (int, float)):
        return max(1.0, min(10.0, conv_bonus))

    conv_rarity_extension_v2 = legacy_conv_rarity + conv_bonus
    conv_rarity_extension_v2 = max(legacy_conv_rarity, conv_rarity_extension_v2)
    return max(1.0, min(10.0, conv_rarity_extension_v2))


# ─── Pure quality from slice ─────────────────────────────

def _compute_pure_quality_from_slice(s):
    """Compute pure quality (without rarity) from a slice's value dict.

    Uses VALUE_WEIGHTS excluding rarity, renormalized.
    """
    v = s.get("value") or {}
    components = {}

    c = (v.get("complexity") or {}).get("overall")
    if c is not None:
        components["complexity"] = c

    q = (v.get("quality") or {}).get("overall")
    if q is not None:
        components["quality"] = q

    r = (v.get("reasoning") or {}).get("overall")
    if r is not None:
        components["reasoning"] = r

    if not components:
        return None

    quality_weights = {k: w for k, w in VALUE_WEIGHTS.items() if k != "rarity"}
    total_w = sum(quality_weights.get(k, 0) for k in components)
    if total_w <= 0:
        return None

    pq = sum(quality_weights.get(k, 0) * v for k, v in components.items()) / total_w

    # NOTE: No quality floor penalty here — penalty is applied only at the
    # conversation level via _compute_penalty() to avoid double-penalizing.
    # The sample-level _extract_pure_quality (in scoring.py) keeps its own
    # 0.7× penalty because it feeds directly into selection_score.

    pq = apply_score_confidence(pq, score_confidence(v))
    return pq


# ─── Per-conversation aggregation ────────────────────────

def aggregate_conversation(conversation_key, slices):
    """Aggregate scores for a single conversation.

    Returns a record dict, or None if insufficient data.
    """
    if not slices:
        return None

    first_meta = (slices[0].get("metadata") or {}) if slices else {}
    single_slice_trajectory = len(slices) == 1 and is_trajectory_object(first_meta)
    if len(slices) < 2 and not single_slice_trajectory:
        return None

    source_id = first_meta.get("source_id") or conversation_key
    source_file = first_meta.get("source_file")
    conversation_uid = first_meta.get("conversation_uid") or conversation_key

    weights = _effective_weights(slices)

    # Weighted average of value_score → q_base_v1
    q_base_v1 = _weighted_average(
        slices, weights,
        lambda s: (s.get("value") or {}).get("value_score")
    )
    if q_base_v1 is None:
        return None

    # Max-pooling blend: prevent a single bad early turn from
    # drowning out genuinely good later turns
    valid_values = [
        (s.get("value") or {}).get("value_score")
        for s in slices
    ]
    valid_values = [v for v in valid_values if v is not None]
    q_base = q_base_v1
    if valid_values:
        q_base = 0.7 * q_base_v1 + 0.3 * max(valid_values)

    # Penalty
    penalty, quality_floor, neg_flags = _compute_penalty(slices)

    score_conf = _weighted_average(
        slices, weights,
        lambda s: score_confidence(s.get("value") or {}, default=1.0)
    )

    # conv_value = q_base × penalty, clamped to [1, 10]
    conv_value = apply_score_confidence(q_base * penalty, score_conf)
    conv_value = max(1.0, min(10.0, conv_value))

    # Peak complexity
    complexities = []
    for s in slices:
        v = s.get("value") or {}
        c = (v.get("complexity") or {}).get("overall")
        if c is not None:
            complexities.append(c)
    peak_complexity = max(complexities) if complexities else None
    quality_overall = _weighted_average(
        slices,
        weights,
        lambda s: ((s.get("value") or {}).get("quality") or {}).get("overall"),
    )
    reasoning_overall = _weighted_average(
        slices,
        weights,
        lambda s: ((s.get("value") or {}).get("reasoning") or {}).get("overall"),
    )

    # Conversation rarity: weighted mean + peak + label-state diversity,
    # then shrink when the conversation is dominated by inherited slices.
    conv_rarity, rarity_detail = _compute_conv_rarity(slices, weights, score_conf)
    conv_rarity_extension_v2 = _compute_conv_extension_v2_rarity(
        slices,
        weights,
        score_conf,
        conv_rarity,
    )

    # Pure quality (for selection score computation)
    pure_quality = _weighted_average(
        slices, weights,
        _compute_pure_quality_from_slice
    )

    # Thinking mode: summarize across slices (fast/slow/mixed)
    thinking_mode, thinking_mode_summary, thinking_mode_counts, thinking_mode_ratio = _summarize_thinking_modes(slices)

    # Merged labels
    merged_labels = _merge_labels(slices)

    # Turn count
    turn_count = len(slices)
    traj_turn_count = first_meta.get("trajectory_turn_count")
    if single_slice_trajectory and isinstance(traj_turn_count, int) and traj_turn_count > turn_count:
        turn_count = traj_turn_count
    turn_metrics = _extract_turn_metrics(slices)
    structure_signals = _extract_structure_signals(slices)
    quality_features = _extract_trajectory_quality_features(slices)
    conv_turn_signal, turn_signal_detail = _compute_conv_turn_signal(slices)
    trajectory_structure_score, structure_detail = _compute_trajectory_structure_score(
        slices,
        merged_labels,
        structure_signals,
        turn_metrics,
        turn_signal_detail,
        turn_count_hint=turn_count,
    )
    trajectory_v2_enabled = bool(structure_detail.get("trajectory_v2_enabled"))

    if trajectory_v2_enabled and conv_turn_signal is not None:
        rarity_term = conv_rarity if conv_rarity is not None else conv_value
        conv_value_v2_raw = 0.80 * conv_turn_signal + 0.20 * rarity_term
        conv_value_v2 = apply_score_confidence(conv_value_v2_raw, score_conf)
        conv_value_v2 = max(conv_value, conv_value_v2)
        conv_value_v2 = _apply_v2_guardrail(
            max(1.0, min(10.0, conv_value_v2)),
            conv_value,
            CONV_V2_VALUE_UPLIFT_CAP,
            CONV_V2_DOWNSIDE_CAP,
        )
    else:
        conv_value_v2 = conv_value

    # Slice details
    slice_details = []
    for i, (s, w) in enumerate(zip(slices, weights)):
        v = s.get("value") or {}
        labels = s.get("labels") or {}
        meta = s.get("metadata") or {}
        slice_details.append({
            "id": s.get("id", ""),
            "turn_index": meta.get("turn_index", i),
            "value_score": v.get("value_score"),
            "confidence": CONV_CONFIDENCE_INHERITED if labels.get("inherited") else 1.0,
            "effective_weight": round(w, 3),
        })

    observed_turn_ratio = rarity_detail.get("observed_turn_ratio")
    inherited_turn_ratio = rarity_detail.get("inherited_turn_ratio")
    rarity_confidence = rarity_detail.get("rarity_confidence")
    compression_gap = turn_metrics.get("peak_minus_mean")
    late_turn_gain = turn_metrics.get("late_turn_gain")

    return {
        "conversation_id": source_id,
        "conversation_uid": conversation_uid,
        "conversation_key": build_conversation_key(source_id, source_file, conversation_uid),
        "source_file": source_file,
        "turn_count": turn_count,
        "conv_value": round(conv_value, 2),
        "conv_value_v2": round(conv_value_v2, 2),
        "conv_selection": None,  # filled by compute_conv_selection_scores
        "conv_selection_v2": None,  # filled by compute_conv_selection_scores
        "conv_selection_extension_v2": None,  # filled by compute_conv_selection_scores
        "peak_complexity": peak_complexity,
        "conv_rarity": round(conv_rarity, 2) if conv_rarity is not None else None,
        "conv_rarity_extension_v2": (
            round(conv_rarity_extension_v2, 2)
            if conv_rarity_extension_v2 is not None else None
        ),
        "trajectory_structure_score": round(trajectory_structure_score, 2),
        "observed_turn_ratio": round(observed_turn_ratio, 2) if observed_turn_ratio is not None else None,
        "inherited_turn_ratio": round(inherited_turn_ratio, 2) if inherited_turn_ratio is not None else None,
        "rarity_confidence": round(rarity_confidence, 2) if rarity_confidence is not None else None,
        "compression_gap": _round_or_none(compression_gap),
        "late_turn_gain": _round_or_none(late_turn_gain),
        "tool_turn_ratio": _round_or_none(structure_signals.get("tool_turn_ratio")),
        "unique_tool_count": structure_signals.get("unique_tool_count"),
        "unique_file_count": structure_signals.get("unique_file_count"),
        "thinking_mode": thinking_mode,
        "thinking_mode_summary": thinking_mode_summary,
        "thinking_mode_counts": thinking_mode_counts,
        "thinking_mode_ratio": thinking_mode_ratio,
        "merged_labels": merged_labels,
        "detail": {
            "q_base": round(q_base, 2),
            "q_base_v1": round(q_base_v1, 2),
            "penalty": round(penalty, 3),
            "quality_floor": quality_floor,
            "negative_flags": neg_flags,
            "pure_quality": round(pure_quality, 2) if pure_quality is not None else None,
            "score_confidence": round(score_conf, 2) if score_conf is not None else None,
            "observed_turns": rarity_detail.get("observed_turns"),
            "inherited_turns": rarity_detail.get("inherited_turns"),
            "observed_weight_ratio": round(rarity_detail["observed_weight_ratio"], 3) if "observed_weight_ratio" in rarity_detail else None,
            "coverage_confidence": round(rarity_detail["coverage_confidence"], 3) if "coverage_confidence" in rarity_detail else None,
            "rarity_mean": round(rarity_detail["rarity_mean"], 2) if "rarity_mean" in rarity_detail else None,
            "rarity_peak": round(rarity_detail["rarity_peak"], 2) if "rarity_peak" in rarity_detail else None,
            "rarity_diversity_bonus": round(rarity_detail["rarity_diversity_bonus"], 3) if "rarity_diversity_bonus" in rarity_detail else None,
            "label_signature_count": rarity_detail.get("label_signature_count"),
            "turn_value_mean": _round_or_none(turn_metrics.get("turn_value_mean")),
            "turn_value_min": _round_or_none(turn_metrics.get("turn_value_min")),
            "turn_value_max": _round_or_none(turn_metrics.get("turn_value_max")),
            "turn_value_median": _round_or_none(turn_metrics.get("turn_value_median")),
            "turn_value_std": _round_or_none(turn_metrics.get("turn_value_std"), 3),
            "turn_quality_std": _round_or_none(turn_metrics.get("turn_quality_std"), 3),
            "top_k_mean": _round_or_none(turn_metrics.get("top_k_mean")),
            "bottom_k_mean": _round_or_none(turn_metrics.get("bottom_k_mean")),
            "peak_minus_mean": _round_or_none(turn_metrics.get("peak_minus_mean")),
            "late_turn_gain": _round_or_none(turn_metrics.get("late_turn_gain")),
            "quality_overall": _round_or_none(quality_overall),
            "reasoning_overall": _round_or_none(reasoning_overall),
            "conv_turn_signal_raw": _round_or_none(turn_signal_detail.get("conv_turn_signal_raw")),
            "conv_turn_signal": _round_or_none(turn_signal_detail.get("conv_turn_signal")),
            "conv_turn_signal_tail_penalty": _round_or_none(turn_signal_detail.get("conv_turn_signal_tail_penalty"), 3),
            "high_value_turn_ratio": _round_or_none(turn_signal_detail.get("high_value_turn_ratio"), 3),
            "low_quality_turn_ratio": _round_or_none(turn_signal_detail.get("low_quality_turn_ratio"), 3),
            "tool_turn_count": structure_signals.get("tool_turn_count"),
            "unique_tools": structure_signals.get("unique_tools"),
            "test_related_turn_count": structure_signals.get("test_related_turn_count"),
            "edit_related_turn_count": structure_signals.get("edit_related_turn_count"),
            "bash_execution_turn_count": structure_signals.get("bash_execution_turn_count"),
            "trajectory_v2_enabled": trajectory_v2_enabled,
            "progress_score": _round_or_none(structure_detail.get("progress_score")),
            "recovery_score": _round_or_none(structure_detail.get("recovery_score")),
            "tool_effectiveness_score": _round_or_none(structure_detail.get("tool_effectiveness_score")),
            "resolution_score": _round_or_none(structure_detail.get("resolution_score")),
            "non_redundancy_score": _round_or_none(structure_detail.get("non_redundancy_score")),
            "clarification_turn_count": quality_features.get("clarification_turn_count"),
            "clarification_ratio": _round_or_none(quality_features.get("clarification_ratio"), 3),
            "tool_call_count": quality_features.get("tool_call_count"),
            "tool_result_count": quality_features.get("tool_result_count"),
            "tool_success_count": quality_features.get("tool_success_count"),
            "tool_failure_count": quality_features.get("tool_failure_count"),
            "tool_result_success_ratio": _round_or_none(quality_features.get("tool_result_success_ratio"), 3),
            "redundant_tool_call_count": quality_features.get("redundant_tool_call_count"),
            "tool_repeat_ratio": _round_or_none(quality_features.get("tool_repeat_ratio"), 3),
            "assistant_repeat_turn_count": quality_features.get("assistant_repeat_turn_count"),
            "recovery_attempt_count": quality_features.get("recovery_attempt_count"),
            "recovery_success_count": quality_features.get("recovery_success_count"),
            "recovery_success_ratio": _round_or_none(quality_features.get("recovery_success_ratio"), 3),
            "verification_turn_count": quality_features.get("verification_turn_count"),
            "verification_success_count": quality_features.get("verification_success_count"),
            "verification_after_edit_count": quality_features.get("verification_after_edit_count"),
            "has_final_verification": quality_features.get("has_final_verification"),
            "conv_intra_class_rank": None,  # filled later
            "selection_confidence": None,  # filled later
            "conv_value_v2_delta": _round_or_none(conv_value_v2 - conv_value),
            "conv_selection_extension_v2_delta": None,  # filled later
        },
        "slices": slice_details,
    }


# ─── Conversation-level selection scores ─────────────────

_CONV_SELECTION_DIMS = ["intent", "language", "domain", "concept", "task",
                        "agentic", "constraint", "context", "difficulty"]


def compute_conv_selection_scores(records):
    """Compute per-tag percentile of pure_quality → conv_selection.

    Similar to sample-level selection scores but operates on conversation
    records. Uses Bayesian shrinkage to handle small tag groups.
    """
    if not records:
        return

    min_group_size = max(1, SELECTION_MIN_GROUP_SIZE)
    smoothing_prior = max(1, SELECTION_SMOOTHING_PRIOR // 5)  # lighter prior for fewer records
    rarity_weights = RARITY_WEIGHTS
    intra_weight = SELECTION_INTRA_WEIGHT

    # Global percentile (needed for Bayesian shrinkage)
    valid_pq = []
    for i, rec in enumerate(records):
        pq = (rec.get("detail") or {}).get("pure_quality")
        if pq is not None:
            valid_pq.append((i, pq))

    global_percentiles = _compute_percentiles(valid_pq)

    # Per-record percentiles across dimensions
    dim_percentiles = [{} for _ in records]

    for dim in _CONV_SELECTION_DIMS:
        tag_groups = defaultdict(list)
        for i, rec in enumerate(records):
            pq = (rec.get("detail") or {}).get("pure_quality")
            if pq is None:
                continue
            labels = rec.get("merged_labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_groups[t].append((i, pq))
            else:
                tag_groups[tags].append((i, pq))

        for tag, members in tag_groups.items():
            if len(members) < min_group_size:
                continue
            n = len(members)
            shrinkage = n / (n + smoothing_prior)
            tag_percentiles = _compute_percentiles(members)
            for idx, _pq in members:
                tag_pct = tag_percentiles[idx]
                g_pct = global_percentiles.get(idx, 0.5)
                percentile = shrinkage * tag_pct + (1 - shrinkage) * g_pct
                if dim in dim_percentiles[idx]:
                    prev, cnt = dim_percentiles[idx][dim]
                    dim_percentiles[idx][dim] = (prev + percentile, cnt + 1)
                else:
                    dim_percentiles[idx][dim] = (percentile, 1)

    # Fuse into intra_class_rank per record
    quality_weight = SELECTION_QUALITY_WEIGHT
    rarity_fuse_weight = 1.0 - intra_weight - quality_weight

    for i, rec in enumerate(records):
        percs = dim_percentiles[i]
        if percs:
            total_w = 0.0
            total_v = 0.0
            for dim, (pct_sum, pct_cnt) in percs.items():
                w = rarity_weights.get(dim, 1.0)
                total_w += w
                total_v += w * (pct_sum / pct_cnt)  # arithmetic mean percentile
            if total_w <= 0:
                continue
            intra_rank = total_v / total_w
            intra_rank_scaled = 1.0 + 9.0 * intra_rank  # scale to 1-10
        else:
            global_pct = global_percentiles.get(i)
            if global_pct is None:
                continue
            intra_rank_scaled = 1.0 + 9.0 * global_pct

        # Absolute quality component
        pq = (rec.get("detail") or {}).get("pure_quality")
        pq_scaled = max(1.0, min(10.0, pq)) if pq is not None else 5.5

        # Fuse with rarity and quality
        rarity = rec.get("conv_rarity")
        if rarity is not None:
            conv_selection = (intra_weight * intra_rank_scaled +
                              quality_weight * pq_scaled +
                              rarity_fuse_weight * rarity)
        else:
            conv_selection = ((intra_weight + rarity_fuse_weight * 0.5) * intra_rank_scaled +
                              (quality_weight + rarity_fuse_weight * 0.5) * pq_scaled)

        selection_conf = min(
            1.0,
            0.5 * max(0.0, min(1.0, (rec.get("rarity_confidence") or 1.0))) +
            0.5 * max(0.0, min(1.0, ((rec.get("detail") or {}).get("score_confidence") or 1.0))),
        )
        conv_selection = apply_score_confidence(conv_selection, selection_conf)
        conv_selection = max(1.0, min(10.0, conv_selection))
        rec["conv_selection"] = round(conv_selection, 2)
        structure_score = rec.get("trajectory_structure_score")
        trajectory_v2_enabled = bool((rec.get("detail") or {}).get("trajectory_v2_enabled"))
        if trajectory_v2_enabled and structure_score is not None:
            rarity_term = rarity if rarity is not None else pq_scaled
            conv_selection_v2 = (
                0.40 * intra_rank_scaled
                + 0.25 * rarity_term
                + 0.35 * structure_score
            )
            conv_selection_v2 = apply_score_confidence(conv_selection_v2, selection_conf)
            conv_selection_v2 = max(conv_selection, conv_selection_v2)
            conv_selection_v2 = _apply_v2_guardrail(
                max(1.0, min(10.0, conv_selection_v2)),
                conv_selection,
                CONV_V2_SELECTION_UPLIFT_CAP,
                CONV_V2_DOWNSIDE_CAP,
            )
        else:
            conv_selection_v2 = conv_selection
        rec["conv_selection_v2"] = round(conv_selection_v2, 2)
        rec["detail"]["conv_intra_class_rank"] = round(intra_rank_scaled, 2)
        rec["detail"]["selection_confidence"] = round(selection_conf, 2)
        rec["detail"]["conv_selection_v2_delta"] = _round_or_none(conv_selection_v2 - conv_selection)

        extension_rarity = rec.get("conv_rarity_extension_v2")
        if extension_rarity is not None:
            conv_selection_extension_v2 = (
                intra_weight * intra_rank_scaled +
                quality_weight * pq_scaled +
                rarity_fuse_weight * extension_rarity
            )
            conv_selection_extension_v2 = apply_score_confidence(
                conv_selection_extension_v2,
                selection_conf,
            )
            conv_selection_extension_v2 = max(conv_selection, conv_selection_extension_v2)
        else:
            conv_selection_extension_v2 = conv_selection
        conv_selection_extension_v2 = max(1.0, min(10.0, conv_selection_extension_v2))
        rec["conv_selection_extension_v2"] = round(conv_selection_extension_v2, 2)
        rec["detail"]["conv_selection_extension_v2_delta"] = _round_or_none(
            conv_selection_extension_v2 - conv_selection
        )


# ─── Top-level entry points ─────────────────────────────

def _conversation_record_sort_key(record):
    return (
        str(record.get("conversation_key") or ""),
        str(record.get("conversation_id") or ""),
    )


def finalize_conversation_records(records):
    """Apply global selection scoring and deterministic ordering to records."""
    if not records:
        return []

    compute_conv_selection_scores(records)
    records.sort(key=_conversation_record_sort_key)
    return records


def merge_conversation_record_batches(record_batches):
    """Merge per-file conversation batches without reloading scored samples."""
    records = []
    for batch in record_batches:
        if batch:
            records.extend(batch)
    return finalize_conversation_records(records)


def _aggregate_conversations_streaming(samples):
    """Aggregate conversation records from a streaming iterator.

    Assumes samples for the same conversation key are contiguous.
    """
    records = []
    current_key = None
    current_slices = []

    def _flush_active():
        nonlocal current_key, current_slices
        if not current_slices or current_key is None:
            current_key = None
            current_slices = []
            return
        current_slices.sort(
            key=lambda s: (s.get("metadata") or {}).get("turn_index", 0)
        )
        rec = aggregate_conversation(current_key, current_slices)
        if rec is not None:
            records.append(rec)
        current_key = None
        current_slices = []

    for sample in samples:
        meta = sample.get("metadata") or {}
        conv_key = sample_conversation_key(sample)
        if not conv_key or not is_conversation_object(meta):
            continue
        if current_key is None:
            current_key = conv_key
            current_slices = [sample]
            continue
        if conv_key != current_key:
            _flush_active()
            current_key = conv_key
            current_slices = [sample]
            continue
        current_slices.append(sample)

    _flush_active()
    return finalize_conversation_records(records)


def aggregate_conversations(samples):
    """Top-level: group → aggregate each → compute selection → return list.

    Args:
        samples: list of scored sample dicts (from Pass 2 output)

    Returns:
        list of conversation record dicts, or empty list if no multi-turn.
    """
    if isinstance(samples, (list, tuple)):
        groups = group_by_conversation(samples)
        if not groups:
            return []

        records = []
        for source_id, slices in groups.items():
            rec = aggregate_conversation(source_id, slices)
            if rec is not None:
                records.append(rec)

        return finalize_conversation_records(records)

    return _aggregate_conversations_streaming(samples)


def write_conversation_scores(records, path):
    """Write conversation_scores.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
