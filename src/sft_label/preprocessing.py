"""
Preprocessing Module

Extracts structural signals from SFT conversations before LLM labeling.
These signals are embedded into the prompt to help the LLM focus on semantic
judgment rather than basic information extraction.

Supports two input formats:
  - ShareGPT: {"conversations": [{"from": "human/gpt/tool", "value": "..."}]}
  - Pangu:    {"data": [{"role": "user/assistant/tool", "content": "..."}]}

Signals extracted:
  - Language detection from code blocks
  - Tool role detection (agentic signals)
  - Code block count and content analysis
  - Turn count and conversation structure
  - Keyword-based hints
  - Token estimation
  - Last turn extraction
"""

import hashlib
import json
import re

from sft_label.config import (
    MAX_CONVERSATION_CHARS, TRUNCATION_HEAD_RATIO,
    TRUNCATION_LAST_RESPONSE_RATIO, TRUNCATION_PER_TURN_RATIO,
    VALUE_TRUNCATION_BUDGET, VALUE_TRUNCATION_INSTRUCTION_RATIO,
    VALUE_TRUNCATION_COT_RATIO, VALUE_TRUNCATION_RESPONSE_RATIO,
    VALUE_TRUNCATION_FRAGMENT_COUNT,
    ENABLE_COMPACT_SEMANTIC_PLANNER,
    PLANNER_POLICY,
    PLANNER_BOUNDARY_THRESHOLD,
    PLANNER_MIN_SEGMENT_SIZE,
    PLANNER_MAX_ANCHOR_GAP,
    PLANNER_FALLBACK_BOUNDARY_RATIO,
)


# ─────────────────────────────────────────────────────────
# Format normalization
# ─────────────────────────────────────────────────────────

# CoT / thinking block patterns
_PANGU_COT_RE = re.compile(r'\[unused16\].*?\[unused17\]', re.DOTALL)
_PANGU_TOKENS_RE = re.compile(r'\[unused(?:9|10|11|12|13|14|15|16|17)\]')
_NO_THINK_RE = re.compile(r'\s*/no_think')
_SHAREGPT_COT_RE = re.compile(r'<(?:think|thinking)>.*?</(?:think|thinking)>', re.DOTALL)

# Training-data XML tags that can cause prompt injection (e.g., <solution>, <tool_call>)
_TRAINING_XML_TAGS_RE = re.compile(
    r'</?(?:solution|tool_call|tool_response|issue|patch|code_change|'
    r'file_change|original_code|new_code|bug_fixes|tests|'
    r'diff|hunk|context|change|fix|'
    r'command|output|result|reasoning|response)(?:\s[^>]*)?>',
    re.IGNORECASE,
)

# Markdown COT blocks (```COT ... ```) that LLMs may mimic in output
_MARKDOWN_COT_BLOCK_RE = re.compile(
    r'```(?:COT|REASONING|THINKING)\b.*?```',
    re.IGNORECASE | re.DOTALL,
)


def sanitize_training_markers(text: str) -> str:
    """Strip training-specific XML tags and markdown COT blocks (keep content between tags).

    These tags (e.g. <solution>, <tool_call>) appear in SWE repair and tool-use
    training data.  When sent to the labeling/scoring LLM they cause prompt
    injection — the LLM mimics the XML format instead of outputting valid JSON.

    Markdown COT blocks (```COT...```) similarly cause the LLM to echo COT
    format instead of producing structured JSON output.
    """
    text = _TRAINING_XML_TAGS_RE.sub('', text)
    text = _MARKDOWN_COT_BLOCK_RE.sub('', text)
    return text

PANGU_ROLE_MAP = {"user": "human", "assistant": "gpt", "tool": "tool"}

CONVERSATION_UID_VERSION = "v1"


def _meta_prompt_to_system_turns(meta_prompt):
    """Convert Pangu meta_prompt into explicit leading system turns."""
    if meta_prompt is None:
        return []

    if isinstance(meta_prompt, (list, tuple)):
        items = meta_prompt
    else:
        items = [meta_prompt]

    turns = []
    for item in items:
        text = str(item).strip()
        if text:
            turns.append({"from": "system", "value": text})
    return turns


def _system_turns_to_text(turns):
    """Concatenate leading system turns into a Pangu-compatible meta_prompt string."""
    parts = [turn.get("value", "").strip() for turn in turns if turn.get("value", "").strip()]
    return "\n\n".join(parts)


def _split_leading_system_turns(conversations):
    """Split off consecutive leading system turns."""
    idx = 0
    while idx < len(conversations) and conversations[idx].get("from") == "system":
        idx += 1
    return conversations[:idx], conversations[idx:]


def _truncate_turns_to_budget(turns, budget):
    """Preserve turns in order, truncating the last kept turn if needed to fit budget."""
    kept = []
    remaining = budget
    for turn in turns:
        if remaining <= 0:
            break
        value = turn.get("value", "")
        if len(value) <= remaining:
            kept.append(dict(turn))
            remaining -= len(value)
            continue
        if remaining < len(_TRUNCATION_MARKER) + 300:
            truncated_value = value[:remaining]
        else:
            truncated_value = _truncate_text(value, remaining)
        kept.append({**turn, "value": truncated_value})
        remaining = 0
        break
    return kept, remaining


def _resolve_pangu_meta_prompt(leading_system_turns, metadata):
    """Choose export meta_prompt, preserving original Pangu shape when possible."""
    raw_meta_prompt = metadata.get("pangu_meta_prompt")
    if leading_system_turns:
        system_text = _system_turns_to_text(leading_system_turns)
        if raw_meta_prompt is not None and _system_turns_to_text(_meta_prompt_to_system_turns(raw_meta_prompt)) == system_text:
            return raw_meta_prompt
        return system_text
    if raw_meta_prompt is not None:
        return raw_meta_prompt
    return metadata.get("system_prompt", "")


def _normalize_role_value(role):
    role_text = str(role or "").strip().lower()
    if role_text in {"user", "human"}:
        return "human"
    if role_text in {"assistant", "gpt"}:
        return "gpt"
    if role_text == "tool":
        return "tool"
    if role_text == "system":
        return "system"
    return role_text


def _build_invalid_turn_error(message, *, source_name, turn_index, source_file=None, source_row=None):
    bits = [f"{message} at {source_name}[{turn_index}]"]
    if source_file is not None:
        bits.append(f"source_file={source_file}")
    if source_row is not None:
        bits.append(f"source_row={source_row}")
    return ValueError(", ".join(bits))


def _coerce_selected_text(value, *, source_name, turn_index, field_name, source_file=None, source_row=None):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value_type = type(value).__name__
        raise _build_invalid_turn_error(
            f"invalid {field_name} type {value_type}",
            source_name=source_name,
            turn_index=turn_index,
            source_file=source_file,
            source_row=source_row,
        )
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_role_content_turn(
    turn,
    *,
    source_name,
    turn_index,
    source_file=None,
    source_row=None,
):
    role = _normalize_role_value(turn.get("role"))
    value = _coerce_selected_text(
        turn.get("content"),
        source_name=source_name,
        turn_index=turn_index,
        field_name="content",
        source_file=source_file,
        source_row=source_row,
    )
    return {"from": role, "value": value}


def _normalize_internal_turn(
    turn,
    *,
    source_name,
    turn_index,
    source_file=None,
    source_row=None,
):
    role_value = turn.get("from")
    if "from" not in turn and "role" in turn:
        role_value = turn.get("role")

    value_source = turn.get("value")
    if "value" not in turn and "content" in turn:
        value_source = turn.get("content")

    role = _normalize_role_value(role_value)
    value = _coerce_selected_text(
        value_source,
        source_name=source_name,
        turn_index=turn_index,
        field_name="value",
        source_file=source_file,
        source_row=source_row,
    )
    return {"from": role, "value": value}


def _normalize_conversations_turns(turns, *, source_file=None, source_row=None):
    normalized_turns = []
    provenance = None

    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise _build_invalid_turn_error(
                f"turn must be object, got {type(turn).__name__}",
                source_name="conversations",
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )

        has_internal = ("from" in turn) or ("value" in turn)
        has_role_content = ("role" in turn) or ("content" in turn)

        if has_internal:
            normalized_turn = _normalize_internal_turn(
                turn,
                source_name="conversations",
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )
            turn_source = "sharegpt"
        elif has_role_content:
            normalized_turn = _normalize_role_content_turn(
                turn,
                source_name="conversations",
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )
            turn_source = "openai_conversations"
        else:
            normalized_turn = {"from": "", "value": ""}
            turn_source = None

        if provenance is None and turn_source and normalized_turn["from"] and normalized_turn["value"]:
            provenance = turn_source
        normalized_turns.append(normalized_turn)

    return normalized_turns, provenance or "sharegpt"


def _normalize_declared_role_content_turns(source_name, turns, *, source_file=None, source_row=None):
    normalized_turns = []
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise _build_invalid_turn_error(
                f"turn must be object, got {type(turn).__name__}",
                source_name=source_name,
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )
        has_role = "role" in turn
        has_content = "content" in turn
        has_internal_fields = ("from" in turn) or ("value" in turn)

        if has_internal_fields:
            raise _build_invalid_turn_error(
                "hybrid turn shape with from/value is not allowed",
                source_name=source_name,
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )

        if not has_role or not has_content:
            missing_fields = []
            if not has_role:
                missing_fields.append("role")
            if not has_content:
                missing_fields.append("content")
            raise _build_invalid_turn_error(
                f"missing required field(s): {', '.join(missing_fields)}",
                source_name=source_name,
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )

        normalized_turns.append(
            _normalize_role_content_turn(
                turn,
                source_name=source_name,
                turn_index=idx,
                source_file=source_file,
                source_row=source_row,
            )
        )
    return normalized_turns


def _has_semantically_non_empty_turns(normalized_turns):
    for turn in normalized_turns:
        role = str(turn.get("from") or "").strip()
        value = turn.get("value")
        if isinstance(value, str):
            value = value.strip()
        if role and value:
            return True
    return False


def _analyze_turn_source(sample, source_name, *, source_file=None, source_row=None):
    if source_name not in sample:
        return None

    raw_turns = sample.get(source_name)
    routing_format = "pangu" if source_name == "data" else "sharegpt" if source_name == "conversations" else "openai_messages"
    provenance = "pangu" if source_name == "data" else "openai_messages"

    if not isinstance(raw_turns, list):
        return {
            "source_name": source_name,
            "routing_format": routing_format,
            "legal": False,
            "non_empty": False,
            "error": ValueError(f"{source_name} must be a list"),
        }

    try:
        if source_name == "conversations":
            normalized_turns, provenance = _normalize_conversations_turns(
                raw_turns,
                source_file=source_file,
                source_row=source_row,
            )
        else:
            normalized_turns = _normalize_declared_role_content_turns(
                source_name,
                raw_turns,
                source_file=source_file,
                source_row=source_row,
            )
    except ValueError as exc:
        return {
            "source_name": source_name,
            "routing_format": routing_format,
            "legal": False,
            "non_empty": False,
            "error": exc,
        }

    return {
        "source_name": source_name,
        "routing_format": routing_format,
        "provenance": provenance,
        "legal": True,
        "non_empty": _has_semantically_non_empty_turns(normalized_turns),
        "normalized_turns": normalized_turns,
    }


def _resolve_turn_source_and_provenance(sample, *, source_file=None, source_row=None):
    analyses = []
    for source_name in ("data", "conversations", "messages"):
        analyzed = _analyze_turn_source(
            sample,
            source_name,
            source_file=source_file,
            source_row=source_row,
        )
        if analyzed is not None:
            analyses.append(analyzed)

    non_empty_legal = next((item for item in analyses if item["legal"] and item["non_empty"]), None)
    if non_empty_legal:
        return {
            "routing_format": non_empty_legal["routing_format"],
            "provenance": non_empty_legal["provenance"],
            "normalized_turns": non_empty_legal["normalized_turns"],
        }

    first_legal_empty = next(
        ((idx, item) for idx, item in enumerate(analyses) if item["legal"]),
        None,
    )
    if first_legal_empty:
        legal_empty_idx, legal_empty_item = first_legal_empty
        blocking_illegal = next(
            (item for idx, item in enumerate(analyses) if idx < legal_empty_idx and not item["legal"]),
            None,
        )
        if blocking_illegal:
            raise blocking_illegal["error"]
        return {
            "routing_format": legal_empty_item["routing_format"],
            "provenance": legal_empty_item["provenance"],
            "normalized_turns": legal_empty_item["normalized_turns"],
        }

    first_illegal = next((item for item in analyses if not item["legal"]), None)
    if first_illegal:
        raise first_illegal["error"]

    return {
        "routing_format": "unknown",
        "provenance": None,
        "normalized_turns": [],
    }


def _normalize_non_pangu_sample(sample, resolved):
    normalized = {
        key: value
        for key, value in sample.items()
        if key not in {"data", "conversations", "messages"}
    }
    normalized["conversations"] = list(resolved.get("normalized_turns", []))
    metadata = dict(sample.get("metadata") or {})
    if resolved.get("provenance"):
        metadata["original_format"] = resolved["provenance"]
    normalized["metadata"] = metadata
    return normalized


def _apply_non_pangu_cot_semantics(normalized):
    """Apply non-Pangu COT extraction/stripping semantics in-place."""
    raw_convs = normalized.get("conversations")
    if not isinstance(raw_convs, list):
        return normalized

    reply_cot = []
    thinking_mode = detect_thinking_mode(raw_convs)
    if thinking_mode == "slow":
        cot_text, _, _ = extract_cot_content(raw_convs)
        normalized.setdefault("metadata", {})["thinking_mode"] = "slow"
        if cot_text:
            normalized["metadata"]["cot_text"] = cot_text

    for turn in raw_convs:
        if turn.get("from") != "gpt":
            continue
        reply_cot.append(extract_turn_cot_text(turn.get("value", "")))
        if turn.get("value"):
            turn["value"] = strip_cot(turn["value"])

    if reply_cot:
        normalized.setdefault("metadata", {})["assistant_cot_by_reply"] = reply_cot

    return normalized


def detect_format(sample):
    """Detect sample format using centralized routing rules."""
    resolved = _resolve_turn_source_and_provenance(sample)
    return resolved["routing_format"]


def strip_cot(text):
    """Remove CoT/thinking blocks from text. Handles both Pangu and ShareGPT patterns."""
    # Pangu: [unused16]thinking content[unused17] → remove entire block
    text = _PANGU_COT_RE.sub('', text)
    # ShareGPT: <think>...</think> or <thinking>...</thinking>
    text = _SHAREGPT_COT_RE.sub('', text)
    return text.strip()


def extract_turn_cot_text(text):
    """Extract explicit CoT text from a single assistant turn."""
    cot_parts = []
    for match in _SHAREGPT_COT_RE.finditer(text):
        inner = match.group()
        inner = re.sub(r'^<(?:think|thinking)>', '', inner)
        inner = re.sub(r'</(?:think|thinking)>$', '', inner)
        if inner.strip():
            cot_parts.append(inner.strip())
    for match in _PANGU_COT_RE.finditer(text):
        inner = match.group()
        inner = inner.replace('[unused16]', '').replace('[unused17]', '')
        if inner.strip():
            cot_parts.append(inner.strip())
    return "\n\n".join(cot_parts)


def _strip_pangu_tokens(text):
    """Remove Pangu training tokens and CoT content."""
    text = strip_cot(text)
    text = _NO_THINK_RE.sub('', text)
    text = _PANGU_TOKENS_RE.sub('', text)
    return text.strip()


def _canonical_role(turn):
    role = str(turn.get("from") or turn.get("role") or "").strip().lower()
    if role in {"user", "human"}:
        return "human"
    if role in {"assistant", "gpt"}:
        return "gpt"
    if role == "tool":
        return "tool"
    return role


def _detect_single_reply_trajectory_object(conversations):
    """Detect long tool-heavy traces that should be grouped as trajectory objects."""
    roles = [_canonical_role(turn) for turn in conversations if isinstance(turn, dict)]
    if not roles:
        return None

    assistant_turn_count = sum(1 for role in roles if role == "gpt")
    tool_turn_count = sum(1 for role in roles if role == "tool")
    turn_count = len(roles)

    if assistant_turn_count != 1:
        return None
    if turn_count < 5 or tool_turn_count < 2:
        return None

    return {
        "trajectory_object": True,
        "trajectory_object_kind": "single_reply_tool_trajectory",
        "trajectory_turn_count": turn_count,
        "trajectory_tool_turn_count": tool_turn_count,
        "trajectory_assistant_turn_count": assistant_turn_count,
    }


def slice_multiturn(conversations):
    """Slice multi-turn conversations into training-aligned samples.

    Each assistant reply becomes one sample, with full preceding context.
    Mirrors how SFT training expands multi-turn into pyramid-shaped samples:
      [H1] → A1
      [H1, A1, H2] → A2
      [H1, A1, H2, A2, H3] → A3

    Single-turn (1 human + 1 gpt) returns as-is in a list.
    """
    # Find all assistant reply indices
    reply_indices = [i for i, t in enumerate(conversations) if t["from"] == "gpt"]

    if len(reply_indices) <= 1:
        return [conversations]

    slices = []
    for idx in reply_indices:
        # Context = everything up to and including this assistant reply
        slices.append(conversations[:idx + 1])

    return slices


def normalize_pangu(sample):
    """Convert Pangu format sample to internal ShareGPT format.

    - Strips training tokens ([unused*], /no_think)
    - Maps roles (user→human, assistant→gpt)
    - Detects pseudo multi-turn (already a single training sample)
    - Preserves meta_prompt and tools in metadata
    - Preserves raw COT text in metadata for Pass 2 scoring
    """
    data = sample.get("data", [])
    conversations = _meta_prompt_to_system_turns(sample.get("meta_prompt"))
    is_pseudo_multiturn = False
    cot_parts = []
    assistant_cot_by_reply = []

    for turn in data:
        role = PANGU_ROLE_MAP.get(turn.get("role", ""), turn.get("role", ""))
        content = turn.get("content", "")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        # Detect pseudo multi-turn (don't expand, just strip tokens)
        if role == "human" and "[unused10]" in content:
            is_pseudo_multiturn = True

        # Extract COT before stripping (for Pass 2 scoring)
        if role == "gpt":
            turn_cot = extract_turn_cot_text(content)
            assistant_cot_by_reply.append(turn_cot)
            if turn_cot:
                cot_parts.append(turn_cot)

        conversations.append({
            "from": role,
            "value": _strip_pangu_tokens(content),
        })

    # Build normalized sample
    normalized = {
        "id": sample.get("id", ""),
        "conversations": conversations,
        "metadata": sample.get("metadata", {}),
    }

    # Preserve Pangu-specific info in metadata
    pangu_meta = {}
    if "meta_prompt" in sample:
        pangu_meta["pangu_meta_prompt"] = sample.get("meta_prompt")
        system_prompt = _system_turns_to_text(_meta_prompt_to_system_turns(sample.get("meta_prompt")))
        if system_prompt:
            pangu_meta["system_prompt"] = system_prompt
    if "tools" in sample:
        pangu_meta["tool_definitions"] = sample.get("tools", "")
    pangu_meta["is_pseudo_multiturn"] = is_pseudo_multiturn
    pangu_meta["original_format"] = "pangu"
    if is_pseudo_multiturn:
        # Preserve raw Pangu data for faithful training format reconstruction
        pangu_meta["raw_pangu_data"] = sample.get("data", [])
    if cot_parts:
        pangu_meta["thinking_mode"] = "slow"
        pangu_meta["cot_text"] = "\n\n".join(cot_parts)
    if assistant_cot_by_reply:
        pangu_meta["assistant_cot_by_reply"] = assistant_cot_by_reply
    normalized["metadata"] = {**normalized["metadata"], **pangu_meta}

    return normalized


def normalize_and_slice(sample, *, source_file=None, source_row=None, annotate_planner_metadata=True):
    """Auto-detect format, normalize, and slice multi-turn into training samples.

    Returns a list of samples. Single-turn and pseudo multi-turn return [1 sample].
    True multi-turn returns [N samples], one per assistant reply.
    Each slice has id suffixed with turn number (e.g. "id_t1", "id_t2").
    """
    resolved = _resolve_turn_source_and_provenance(
        sample,
        source_file=source_file,
        source_row=source_row,
    )
    fmt = resolved["routing_format"]
    if fmt == "pangu":
        normalized = normalize_pangu(sample)
    elif fmt in {"sharegpt", "openai_messages"}:
        normalized = _normalize_non_pangu_sample(sample, resolved)
        _apply_non_pangu_cot_semantics(normalized)
    else:
        normalized = dict(sample)

    conversations = normalized.get("conversations", [])
    is_pseudo = normalized.get("metadata", {}).get("is_pseudo_multiturn", False)
    if source_file is not None:
        normalized.setdefault("metadata", {}).setdefault("source_file", str(source_file))
    if source_row is not None:
        normalized.setdefault("metadata", {}).setdefault("source_row", int(source_row))

    # Pseudo multi-turn: already one training sample, don't slice
    if is_pseudo:
        return [normalized]

    slices = slice_multiturn(conversations)
    trajectory_meta = _detect_single_reply_trajectory_object(conversations)

    base_meta = normalized.get("metadata", {})
    conversation_uid = base_meta.get("conversation_uid")
    if not conversation_uid and (len(slices) > 1 or trajectory_meta):
        conversation_uid = build_conversation_uid(
            normalized,
            source_file=source_file,
            source_row=source_row,
        )

    if len(slices) == 1:
        if not normalized.setdefault("metadata", {}).get("thinking_mode"):
            normalized["metadata"]["thinking_mode"] = "fast"
        if conversation_uid:
            normalized.setdefault("metadata", {})["conversation_uid"] = conversation_uid
        if trajectory_meta:
            enriched = dict(normalized)
            meta = dict(enriched.get("metadata") or {})
            source_id = meta.get("source_id") or enriched.get("id")
            if source_id:
                meta["source_id"] = source_id
            meta.setdefault("turn_index", 1)
            meta.setdefault("total_turns", 1)
            if conversation_uid:
                meta.setdefault("conversation_uid", conversation_uid)
            meta.update(trajectory_meta)
            enriched["metadata"] = meta
            return [enriched]
        return [normalized]

    # Build one sample per slice
    results = []
    base_id = normalized.get("id", "")
    reply_cot = list(base_meta.get("assistant_cot_by_reply", []))
    for i, conv_slice in enumerate(slices):
        slice_index = i
        slice_position = i + 1
        slice_count = len(slices)
        slice_meta = {
            **base_meta,
            "source_id": base_id,
            "turn_index": slice_position,
            "total_turns": slice_count,
            "slice_index": slice_index,
            "slice_position": slice_position,
            "slice_count": slice_count,
            "source_turn_count": len(conversations),
        }
        if conversation_uid:
            slice_meta["conversation_uid"] = conversation_uid
        final_turn_cot = reply_cot[i] if i < len(reply_cot) else ""
        # Preserve COT only when it belongs to this slice's final assistant reply.
        if final_turn_cot:
            slice_meta["thinking_mode"] = "slow"
            slice_meta["cot_text"] = final_turn_cot
        else:
            slice_meta["thinking_mode"] = "fast"
            slice_meta.pop("cot_text", None)

        s = {
            "id": f"{base_id}_t{i+1}",
            "conversations": conv_slice,
            "metadata": slice_meta,
        }
        results.append(s)

    if annotate_planner_metadata:
        annotate_multiturn_planner_metadata(results)
    return results


def build_conversation_uid(sample, *, source_file=None, source_row=None):
    """Build a stable UID for one source conversation row.

    `source_file` + `source_row` disambiguate duplicate source_id/content rows
    inside the same dataset file while keeping the key deterministic.
    """
    metadata = sample.get("metadata") or {}
    existing = metadata.get("conversation_uid")
    if existing:
        return existing

    conversations = sample.get("conversations") or []
    canonical_turns = [
        {
            "from": turn.get("from"),
            "value": turn.get("value", ""),
        }
        for turn in conversations
        if isinstance(turn, dict)
    ]
    payload = {
        "id": sample.get("id"),
        "turns": canonical_turns,
    }
    if source_file is not None:
        payload["source_file"] = str(source_file)
    if source_row is not None:
        payload["source_row"] = int(source_row)

    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"conversation_uid:{CONVERSATION_UID_VERSION}:{digest}"


# Keep backward-compatible alias
def normalize_sample(sample, *, source_file=None, source_row=None):
    """Normalize without slicing (for tools that don't need multi-turn expansion)."""
    resolved = _resolve_turn_source_and_provenance(
        sample,
        source_file=source_file,
        source_row=source_row,
    )
    fmt = resolved["routing_format"]
    if fmt == "pangu":
        return normalize_pangu(sample)
    if fmt in {"sharegpt", "openai_messages"}:
        normalized = _normalize_non_pangu_sample(sample, resolved)
        _apply_non_pangu_cot_semantics(normalized)
        if not normalized.setdefault("metadata", {}).get("thinking_mode"):
            normalized["metadata"]["thinking_mode"] = "fast"
        return normalized
    return sample


def to_pangu_pseudo_multiturn(sample):
    """Convert a normalized ShareGPT-format sample back to Pangu pseudo-multi-turn training format.

    Takes a labeled/scored sample and reconstructs the Pangu training format:
    - Prior turns packed into the user message with [unused10][unused9] separators
    - Human turns in prior context get /no_think suffix
    - GPT turns in prior context get 助手：[unused16][unused17] prefix
    - Role labels (用户：, 助手：) added
    - COT restored when the current final assistant turn has attributable cot_text

    For pseudo-multiturn samples (already packed before normalization),
    uses raw_pangu_data from metadata for faithful reconstruction.

    Returns dict with {"data": [...], "meta_prompt": ..., "tools": ...}
    """
    conversations = sample.get("conversations", [])
    metadata = sample.get("metadata", {})
    leading_system_turns, conversations = _split_leading_system_turns(conversations)
    meta_prompt = _resolve_pangu_meta_prompt(leading_system_turns, metadata)

    if not conversations:
        return {"data": [], "meta_prompt": meta_prompt, "tools": metadata.get("tool_definitions", "")}

    # Pseudo-multiturn: use raw data directly if available
    if metadata.get("is_pseudo_multiturn") and metadata.get("raw_pangu_data"):
        return {
            "data": metadata["raw_pangu_data"],
            "meta_prompt": meta_prompt,
            "tools": metadata.get("tool_definitions", ""),
        }

    # Find last assistant reply and its preceding human turn
    last_reply_idx = None
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("from") == "gpt":
            last_reply_idx = i
            break

    if last_reply_idx is None:
        # No assistant reply — return as-is in Pangu format
        return {
            "data": [{"role": "user", "content": conversations[0].get("value", "")}],
            "meta_prompt": meta_prompt,
            "tools": metadata.get("tool_definitions", ""),
        }

    # Find the human turn preceding the last reply
    last_user_idx = None
    for i in range(last_reply_idx - 1, -1, -1):
        if conversations[i].get("from") == "human":
            last_user_idx = i
            break

    # Pack prior turns (before last_user_idx) into a single string
    prior_parts = []
    prior_end = last_user_idx if last_user_idx is not None else last_reply_idx
    for i in range(prior_end):
        turn = conversations[i]
        role = turn.get("from", "")
        content = turn.get("value", "")
        if role == "human":
            prior_parts.append(f"用户：{content} /no_think")
        elif role == "gpt":
            prior_parts.append(f"助手：[unused16][unused17]{content}")
        elif role == "tool":
            prior_parts.append(content)

    # Pack trajectory turns (between last user request and final assistant reply)
    trajectory_parts = []
    if last_user_idx is not None:
        for i in range(last_user_idx + 1, last_reply_idx):
            turn = conversations[i]
            role = turn.get("from", "")
            content = turn.get("value", "")
            if role == "human":
                trajectory_parts.append(f"用户：{content} /no_think")
            elif role == "gpt":
                trajectory_parts.append(f"助手：[unused16][unused17]{content}")
            elif role == "tool":
                trajectory_parts.append(content)

    # Build user content
    last_user_content = ""
    if last_user_idx is not None:
        last_user_content = conversations[last_user_idx].get("value", "")

    if prior_parts:
        packed_prior = "[unused10][unused9]".join(prior_parts)
        user_content = f"{packed_prior}[unused10][unused9]用户：{last_user_content}"
    else:
        user_content = last_user_content

    # Build last assistant content
    response = conversations[last_reply_idx].get("value", "")
    cot_text = metadata.get("cot_text", "")

    if cot_text:
        # Restore COT whenever it is attributable to the current final reply.
        assistant_content = f"[unused16]{cot_text}[unused17]{response}"
    else:
        # Fast thinking (no attributable COT)
        assistant_content = f"[unused16][unused17]{response}"

    if trajectory_parts:
        packed_trajectory = "[unused10][unused9]".join(trajectory_parts)
        assistant_content = f"{packed_trajectory}[unused10][unused9]{assistant_content}"

    result = {
        "data": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta_prompt": meta_prompt,
        "tools": metadata.get("tool_definitions", ""),
    }

    return result


# ─────────────────────────────────────────────────────────
# Conversation truncation for labeling
# ─────────────────────────────────────────────────────────

_TRUNCATION_MARKER = "\n\n[... content truncated for labeling ...]\n\n"


def _truncate_text(text, max_chars, keep_head_ratio=0.3):
    """Truncate a single text block, keeping head + tail with a marker in between."""
    if len(text) <= max_chars:
        return text
    marker_len = len(_TRUNCATION_MARKER)
    head_chars = int(max_chars * keep_head_ratio)
    tail_chars = max_chars - head_chars - marker_len
    if tail_chars < 200:
        tail_chars = 200
        head_chars = max_chars - tail_chars - marker_len
    if head_chars < 100:
        head_chars = 100
    return text[:head_chars] + _TRUNCATION_MARKER + text[-tail_chars:]


def truncate_conversations_for_labeling(conversations, max_total_chars=None,
                                        head_ratio=None, last_response_ratio=None,
                                        per_turn_ratio=None):
    """Truncate conversations to fit within model context for labeling.

    Budget allocation (configurable in config.py):
    - First human turn (task context): head_ratio of budget
    - Last gpt turn (labeling target): last_response_ratio of budget
    - Remaining turns: share the rest, filled from the end backward

    For ≤2 turns: split budget proportionally between query and response,
    giving the response slightly more room.

    Returns (truncated_conversations, was_truncated).
    """
    if max_total_chars is None:
        max_total_chars = MAX_CONVERSATION_CHARS
    if head_ratio is None:
        head_ratio = TRUNCATION_HEAD_RATIO
    if last_response_ratio is None:
        last_response_ratio = TRUNCATION_LAST_RESPONSE_RATIO
    if per_turn_ratio is None:
        per_turn_ratio = TRUNCATION_PER_TURN_RATIO

    # Sanitize training-specific XML tags before truncation
    conversations = [
        {**t, "value": sanitize_training_markers(t.get("value", ""))}
        for t in conversations
    ]

    total_chars = sum(len(t.get("value", "")) for t in conversations)
    if total_chars <= max_total_chars:
        return conversations, False

    leading_system_turns, remaining_conversations = _split_leading_system_turns(conversations)
    if leading_system_turns:
        kept_system_turns, remaining_budget = _truncate_turns_to_budget(leading_system_turns, max_total_chars)
        if not remaining_conversations or remaining_budget <= 0:
            return kept_system_turns, True
        truncated_rest, _ = truncate_conversations_for_labeling(
            remaining_conversations,
            max_total_chars=remaining_budget,
            head_ratio=head_ratio,
            last_response_ratio=last_response_ratio,
            per_turn_ratio=per_turn_ratio,
        )
        return kept_system_turns + truncated_rest, True

    n = len(conversations)

    # --- Single turn or two turns (query + response) ---
    if n <= 2:
        # Give response 60% of budget, query 40%
        if n == 1:
            val = _truncate_text(conversations[0].get("value", ""), max_total_chars)
            return [{**conversations[0], "value": val}], True
        query_budget = int(max_total_chars * 0.4)
        response_budget = max_total_chars - query_budget
        result = []
        for i, t in enumerate(conversations):
            budget = response_budget if t["from"] == "gpt" else query_budget
            val = t.get("value", "")
            if len(val) > budget:
                val = _truncate_text(val, budget)
            result.append({**t, "value": val})
        return result, True

    # --- Multi-turn ---
    # Identify key turns
    first_human_idx = next((i for i, t in enumerate(conversations) if t["from"] == "human"), 0)
    last_gpt_idx = next((i for i in range(n - 1, -1, -1) if conversations[i]["from"] == "gpt"), n - 1)

    # Reserve budgets for key turns
    first_budget = int(max_total_chars * head_ratio)
    last_resp_budget = int(max_total_chars * last_response_ratio)
    middle_budget = max_total_chars - first_budget - last_resp_budget

    # 1. Truncate first human turn
    first_turn = dict(conversations[first_human_idx])
    first_val = first_turn.get("value", "")
    if len(first_val) > first_budget:
        first_val = _truncate_text(first_val, first_budget)
    first_turn["value"] = first_val

    # 2. Truncate last gpt turn
    last_turn = dict(conversations[last_gpt_idx])
    last_val = last_turn.get("value", "")
    if len(last_val) > last_resp_budget:
        last_val = _truncate_text(last_val, last_resp_budget)
    last_turn["value"] = last_val

    # 3. Fill middle from the end backward (excluding first_human and last_gpt)
    #    This preserves the most recent context leading up to the last response.
    middle_indices = [i for i in range(n) if i != first_human_idx and i != last_gpt_idx]
    per_turn_cap = int(max_total_chars * per_turn_ratio)

    tail_turns = []  # (original_index, turn_dict)
    tail_chars = 0
    for i in reversed(middle_indices):
        val = conversations[i].get("value", "")
        # Cap individual turn
        if len(val) > per_turn_cap:
            val = _truncate_text(val, per_turn_cap)
        if tail_chars + len(val) > middle_budget:
            # Try to fit a truncated version
            remaining = middle_budget - tail_chars
            if remaining > 500:
                val = _truncate_text(conversations[i].get("value", ""), remaining)
                tail_turns.append((i, {**conversations[i], "value": val}))
            break
        tail_turns.append((i, {**conversations[i], "value": val}))
        tail_chars += len(val)

    tail_turns.reverse()

    # 4. Assemble: first_human + [omission marker] + kept middle turns + last_gpt
    result = [first_turn]

    {i for i, _ in tail_turns}
    # Count omitted turns between first_human and the earliest kept middle turn
    omitted = len(middle_indices) - len(tail_turns)
    if omitted > 0:
        result.append({"from": "system", "value": f"[... {omitted} middle turns omitted for labeling ...]"})

    for _, turn in tail_turns:
        result.append(turn)

    # Ensure last_gpt is at the end (it might already be in tail_turns if last_gpt_idx was in middle)
    if not tail_turns or tail_turns[-1][0] != last_gpt_idx:
        result.append(last_turn)

    return result, True


# ─────────────────────────────────────────────────────────
# Value scoring: thinking-mode detection & COT-preserving truncation
# ─────────────────────────────────────────────────────────

def detect_thinking_mode(conversations):
    """Detect whether a sample uses slow-thinking (explicit COT) or fast-thinking.

    Scans raw conversation text for <think>, <thinking>, or [unused16] markers.
    Returns "slow" or "fast".
    """
    for turn in conversations:
        val = turn.get("value", "")
        if _SHAREGPT_COT_RE.search(val) or _PANGU_COT_RE.search(val):
            return "slow"
    return "fast"


def extract_cot_content(conversations):
    """Extract COT blocks from conversations without stripping them.

    Returns:
        cot_text: concatenated COT content (without tags)
        cot_chars: total char count of COT content
        conversations_without_cot: conversations with COT blocks removed from GPT turns
    """
    cot_parts = []
    cleaned_convs = []

    for turn in conversations:
        val = turn.get("value", "")
        if turn.get("from") == "gpt":
            # Collect COT content
            for match in _SHAREGPT_COT_RE.finditer(val):
                # Strip the outer tags
                inner = match.group()
                inner = re.sub(r'^<(?:think|thinking)>', '', inner)
                inner = re.sub(r'</(?:think|thinking)>$', '', inner)
                cot_parts.append(inner.strip())
            for match in _PANGU_COT_RE.finditer(val):
                inner = match.group()
                inner = inner.replace('[unused16]', '').replace('[unused17]', '')
                cot_parts.append(inner.strip())
            # Remove COT from the response
            clean_val = strip_cot(val)
            cleaned_convs.append({**turn, "value": clean_val})
        else:
            cleaned_convs.append(turn)

    cot_text = "\n\n".join(cot_parts)
    return cot_text, len(cot_text), cleaned_convs


def truncate_with_fragments(text, budget, n_fragments=3, head_ratio=0.3, tail_ratio=0.3):
    """Truncate text keeping head + evenly-spaced middle fragments + tail.

    Each gap is annotated with position and omitted char count.
    """
    if len(text) <= budget:
        return text

    # Reserve space for markers (approx)
    marker_template = "\n[... {} chars omitted, fragment at {}% ...]\n"
    marker_overhead = len(marker_template.format("99999", "99")) * (n_fragments + 1)
    usable = budget - marker_overhead
    if usable < 200:
        # Too small for fragment approach, fall back to head+tail
        return _truncate_text(text, budget, keep_head_ratio=head_ratio)

    head_chars = int(usable * head_ratio)
    tail_chars = int(usable * tail_ratio)
    mid_total = usable - head_chars - tail_chars
    frag_size = mid_total // max(n_fragments, 1)

    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""

    # Sample middle fragments at evenly-spaced positions
    mid_start = head_chars
    mid_end = len(text) - tail_chars
    mid_range = mid_end - mid_start

    fragments = []
    if mid_range > 0 and n_fragments > 0:
        for i in range(n_fragments):
            # Position at (i+1)/(n+1) of the middle range
            pos = mid_start + int((i + 1) * mid_range / (n_fragments + 1))
            frag_start = max(mid_start, pos - frag_size // 2)
            frag_end = min(mid_end, frag_start + frag_size)
            pct = int(100 * (frag_start - 0) / len(text))
            fragments.append((frag_start, frag_end, pct))

    # Assemble
    parts = [head]
    prev_end = head_chars
    for frag_start, frag_end, pct in fragments:
        omitted = frag_start - prev_end
        if omitted > 0:
            parts.append(f"\n[... {omitted} chars omitted, fragment at {pct}% ...]\n")
        parts.append(text[frag_start:frag_end])
        prev_end = frag_end

    # Gap before tail
    omitted = (len(text) - tail_chars) - prev_end
    if omitted > 0:
        parts.append(f"\n[... {omitted} chars omitted ...]\n")
    if tail:
        parts.append(tail)

    return "".join(parts)


def _find_last_assistant_idx(conversations):
    """Return the last assistant turn index, or the last turn as fallback."""
    for idx in range(len(conversations) - 1, -1, -1):
        if conversations[idx].get("from") == "gpt":
            return idx
    return len(conversations) - 1 if conversations else -1


def _find_current_request_idx(conversations, last_assistant_idx):
    """Return the human turn that the final assistant response is answering."""
    if last_assistant_idx <= 0:
        return 0 if conversations else -1
    for idx in range(last_assistant_idx - 1, -1, -1):
        if conversations[idx].get("from") == "human":
            return idx
    for idx, turn in enumerate(conversations):
        if turn.get("from") == "human":
            return idx
    return max(0, last_assistant_idx - 1)


def _format_scoring_turn(turn, final=False):
    """Render one conversation turn for the Pass 2 evidence window."""
    role = turn.get("from")
    if role == "human":
        label = "Human"
    elif role == "tool":
        label = "Tool Result"
    elif role == "gpt":
        label = "Final Assistant" if final else "Assistant Step"
    else:
        label = (role or "Turn").replace("-", " ").title()

    text = sanitize_training_markers(turn.get("value", ""))
    return f"[{label}] {text}" if text else f"[{label}]"


def _extract_scoring_segments(conversations):
    """Split a sample into prior context, current request, trajectory, and final answer."""
    if not conversations:
        return {
            "prior_context": "",
            "current_request": "",
            "trajectory": "",
            "response": "",
            "trajectory_turn_count": 0,
            "trajectory_tool_turns": 0,
        }

    last_assistant_idx = _find_last_assistant_idx(conversations)
    if last_assistant_idx < 0:
        return {
            "prior_context": "",
            "current_request": "",
            "trajectory": "",
            "response": "",
            "trajectory_turn_count": 0,
            "trajectory_tool_turns": 0,
        }

    request_idx = _find_current_request_idx(conversations, last_assistant_idx)
    prior_turns = conversations[:request_idx]
    request_turn = conversations[request_idx] if 0 <= request_idx < len(conversations) else {}
    trajectory_turns = conversations[request_idx + 1:last_assistant_idx]
    response_turn = conversations[last_assistant_idx]

    return {
        "prior_context": "\n".join(_format_scoring_turn(turn) for turn in prior_turns),
        "current_request": sanitize_training_markers(request_turn.get("value", "")),
        "trajectory": "\n".join(_format_scoring_turn(turn) for turn in trajectory_turns),
        "response": sanitize_training_markers(response_turn.get("value", "")),
        "trajectory_turn_count": len(trajectory_turns),
        "trajectory_tool_turns": sum(1 for turn in trajectory_turns if turn.get("from") == "tool"),
    }


def truncate_for_scoring(conversations, thinking_mode, cot_text="",
                         budget=None, instruction_ratio=None,
                         cot_ratio=None, response_ratio=None,
                         n_fragments=None):
    """Orchestrate COT-preserving truncation for value scoring.

    Budget allocation:
      - Current user request: 15%
      - COT (slow thinking only): 45%
      - Remaining evidence pool: prior context + current-turn trajectory + final answer
      - Meta is handled externally: 5%

    For fast thinking (no COT), the COT budget merges into the evidence pool.

    Returns:
        dict with keys: instruction/current_request, prior_context, trajectory,
              cot (or None), response, was_truncated, original_*_chars
    """
    if budget is None:
        budget = VALUE_TRUNCATION_BUDGET
    if instruction_ratio is None:
        instruction_ratio = VALUE_TRUNCATION_INSTRUCTION_RATIO
    if cot_ratio is None:
        cot_ratio = VALUE_TRUNCATION_COT_RATIO
    if response_ratio is None:
        response_ratio = VALUE_TRUNCATION_RESPONSE_RATIO
    if n_fragments is None:
        n_fragments = VALUE_TRUNCATION_FRAGMENT_COUNT

    # Content budget (exclude ~5% meta overhead)
    content_budget = int(budget * 0.95)

    # Extract a current-turn-centered evidence window.
    segments = _extract_scoring_segments(conversations)
    prior_context = segments["prior_context"]
    instruction = segments["current_request"]
    trajectory = segments["trajectory"]
    response = segments["response"]
    trajectory_turn_count = segments["trajectory_turn_count"]
    trajectory_tool_turns = segments["trajectory_tool_turns"]

    # For fast thinking, strip COT from response if any
    if thinking_mode == "fast":
        response_clean = response
        cot_text_final = ""
    else:
        response_clean = strip_cot(response)
        cot_text_final = cot_text

    # Sanitize training-specific XML tags
    prior_context = sanitize_training_markers(prior_context)
    instruction = sanitize_training_markers(instruction)
    trajectory = sanitize_training_markers(trajectory)
    response_clean = sanitize_training_markers(response_clean)
    if cot_text_final:
        cot_text_final = sanitize_training_markers(cot_text_final)

    original_prior_context_chars = len(prior_context)
    original_instruction_chars = len(instruction)
    original_trajectory_chars = len(trajectory)
    original_cot_chars = len(cot_text_final)
    original_response_chars = len(response_clean)

    total_original = (
        original_prior_context_chars
        + original_instruction_chars
        + original_trajectory_chars
        + original_cot_chars
        + original_response_chars
    )
    was_truncated = total_original > content_budget

    if not was_truncated:
        return {
            "prior_context": prior_context,
            "current_request": instruction,
            "instruction": instruction,
            "trajectory": trajectory,
            "cot": cot_text_final if thinking_mode == "slow" else None,
            "response": response_clean,
            "was_truncated": False,
            "original_prior_context_chars": original_prior_context_chars,
            "original_instruction_chars": original_instruction_chars,
            "original_trajectory_chars": original_trajectory_chars,
            "original_cot_chars": original_cot_chars,
            "original_response_chars": original_response_chars,
            "trajectory_turn_count": trajectory_turn_count,
            "trajectory_tool_turns": trajectory_tool_turns,
        }

    # Allocate budgets
    if thinking_mode == "slow" and cot_text_final:
        instr_budget = int(content_budget * instruction_ratio)
        cot_budget = int(content_budget * cot_ratio)
        evidence_budget = content_budget - instr_budget - cot_budget
    else:
        # Fast thinking: no COT, merge COT budget into current-turn evidence
        instr_budget = int(content_budget * instruction_ratio)
        cot_budget = 0
        evidence_budget = content_budget - instr_budget

    evidence_weights = []
    if prior_context:
        evidence_weights.append(("prior_context", 0.20))
    if trajectory:
        evidence_weights.append(("trajectory", 0.45))
    if response_clean:
        evidence_weights.append(("response", 0.35))

    evidence_budgets = {
        "prior_context": 0,
        "trajectory": 0,
        "response": 0,
    }
    if evidence_weights and evidence_budget > 0:
        total_weight = sum(weight for _, weight in evidence_weights)
        allocated = 0
        for name, weight in evidence_weights[:-1]:
            share = int(evidence_budget * (weight / total_weight))
            evidence_budgets[name] = share
            allocated += share
        last_name = evidence_weights[-1][0]
        evidence_budgets[last_name] = max(0, evidence_budget - allocated)

    # Truncate each section
    trunc_instruction = _truncate_text(instruction, instr_budget) if len(instruction) > instr_budget else instruction
    prior_budget = evidence_budgets["prior_context"]
    if prior_context and prior_budget > 0:
        trunc_prior_context = (
            _truncate_text(prior_context, prior_budget, keep_head_ratio=0.15)
            if len(prior_context) > prior_budget else prior_context
        )
    else:
        trunc_prior_context = prior_context if prior_budget != 0 else ""

    if thinking_mode == "slow" and cot_text_final and cot_budget > 0:
        trunc_cot = truncate_with_fragments(cot_text_final, cot_budget, n_fragments=n_fragments)
    else:
        trunc_cot = None

    trajectory_budget = evidence_budgets["trajectory"]
    if trajectory and trajectory_budget > 0:
        trunc_trajectory = truncate_with_fragments(
            trajectory, trajectory_budget,
            n_fragments=n_fragments,
            head_ratio=0.25,
            tail_ratio=0.35,
        )
    else:
        trunc_trajectory = trajectory if trajectory_budget != 0 else ""

    response_budget = evidence_budgets["response"]
    if thinking_mode == "fast" and response_budget > 0:
        trunc_response = truncate_with_fragments(
            response_clean, response_budget, n_fragments=n_fragments,
            head_ratio=0.45, tail_ratio=0.25,
        )
    elif response_budget > 0:
        trunc_response = _truncate_text(response_clean, response_budget) if len(response_clean) > response_budget else response_clean
    else:
        trunc_response = response_clean if response_budget != 0 else ""

    return {
        "prior_context": trunc_prior_context,
        "current_request": trunc_instruction,
        "instruction": trunc_instruction,
        "trajectory": trunc_trajectory,
        "cot": trunc_cot,
        "response": trunc_response,
        "was_truncated": True,
        "original_prior_context_chars": original_prior_context_chars,
        "original_instruction_chars": original_instruction_chars,
        "original_trajectory_chars": original_trajectory_chars,
        "original_cot_chars": original_cot_chars,
        "original_response_chars": original_response_chars,
        "trajectory_turn_count": trajectory_turn_count,
        "trajectory_tool_turns": trajectory_tool_turns,
    }


# ─────────────────────────────────────────────────────────
# Language detection patterns
# ─────────────────────────────────────────────────────────

# Map of code fence language identifiers → taxonomy language tag IDs
FENCE_LANG_MAP = {
    "python": "python", "py": "python", "python3": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript", "jsx": "javascript",
    "java": "java",
    "go": "go", "golang": "go",
    "rust": "rust", "rs": "rust",
    "c": "c", "h": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp", "hpp": "cpp",
    "csharp": "csharp", "cs": "csharp", "c#": "csharp",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin", "kt": "kotlin",
    "scala": "scala",
    "sql": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql", "sqlite": "sql",
    "html": "html",
    "css": "css", "scss": "css", "sass": "css", "less": "css",
    "shell": "shell", "bash": "shell", "sh": "shell", "zsh": "shell",
    "dockerfile": "dockerfile", "docker": "dockerfile",
    "yaml": "yaml", "yml": "yaml",
    "json": "json", "jsonc": "json",
    "toml": "toml",
    "xml": "xml",
    "markdown": "markdown", "md": "markdown",
    "hcl": "hcl", "terraform": "hcl", "tf": "hcl",
    "lua": "lua",
    "r": "r",
    "dart": "dart",
    "elixir": "elixir", "ex": "elixir",
    "erlang": "erlang", "erl": "erlang",
    "haskell": "haskell", "hs": "haskell",
    "ocaml": "ocaml", "ml": "ocaml",
    "perl": "perl", "pl": "perl",
    "solidity": "solidity", "sol": "solidity",
    "verilog": "verilog", "v": "verilog",
    "zig": "zig",
    "makefile": "makefile", "make": "makefile",
    "cmake": "cmake",
    "nginx": "nginx-config",
    "ini": "ini",
    "properties": "properties",
    "latex": "latex", "tex": "latex",
    "matlab": "matlab",
    "julia": "julia", "jl": "julia",
    "clojure": "clojure", "clj": "clojure",
    "fsharp": "fsharp", "fs": "fsharp", "f#": "fsharp",
    "powershell": "powershell", "ps1": "powershell",
    "prolog": "prolog",
    "scheme": "scheme",
    "lisp": "lisp",
    "assembly": "assembly", "asm": "assembly", "nasm": "assembly",
    "fortran": "fortran", "f90": "fortran",
    "cobol": "cobol",
    "groovy": "groovy",
    "ada": "ada",
    "nim": "nim",
    "crystal": "crystal",
    "vyper": "vyper",
}

# Framework → language inference
FRAMEWORK_LANG_MAP = {
    "react": "typescript", "next.js": "typescript", "nextjs": "typescript",
    "vue": "typescript", "nuxt": "typescript", "angular": "typescript",
    "svelte": "typescript", "sveltekit": "typescript",
    "django": "python", "flask": "python", "fastapi": "python", "tornado": "python",
    "spring": "java", "spring boot": "java", "springboot": "java",
    "express": "javascript", "express.js": "javascript", "koa": "javascript",
    "rails": "ruby", "ruby on rails": "ruby", "sinatra": "ruby",
    "laravel": "php", "symfony": "php",
    "gin": "go", "echo": "go", "fiber": "go",
    "actix": "rust", "axum": "rust", "tokio": "rust", "rocket": "rust",
    "swiftui": "swift", "uikit": "swift",
    "jetpack compose": "kotlin", "ktor": "kotlin",
    "flutter": "dart",
    "pytorch": "python", "tensorflow": "python", "keras": "python",
    "pandas": "python", "numpy": "python", "scikit-learn": "python",
    "tailwind": "css", "bootstrap": "css",
    "terraform": "hcl", "ansible": "yaml", "helm": "yaml",
    "docker compose": "yaml", "docker-compose": "yaml",
    "prisma": "typescript", "drizzle": "typescript",
    "sqlalchemy": "python", "alembic": "python",
    "playwright": "typescript", "cypress": "typescript", "selenium": "python",
    "pygame": "python",
}

# Tool name → agentic tag mapping
TOOL_NAME_MAP = {
    # File operations
    "read_file": "file-operations", "write_file": "file-operations",
    "edit_file": "file-operations", "read": "file-operations",
    "write": "file-operations", "edit": "file-operations",
    "glob": "file-operations", "grep": "file-operations",
    "search_files": "file-operations", "list_files": "file-operations",
    "cat": "file-operations", "ls": "file-operations",

    # Shell / bash
    "bash": "bash-execution", "shell": "bash-execution",
    "terminal": "bash-execution", "execute_command": "bash-execution",
    "run_command": "bash-execution",

    # Code execution
    "python": "code-execution", "execute_code": "code-execution",
    "run_code": "code-execution", "jupyter": "code-execution",
    "repl": "code-execution",

    # Git
    "git": "git-operations",

    # Web
    "web_search": "web-search", "search": "web-search",
    "browser": "web-search", "fetch_url": "web-search",

    # Build
    "build": "build-execution", "compile": "build-execution",
    "make": "build-execution",

    # Test
    "test": "test-running", "run_tests": "test-running",
    "pytest": "test-running",

    # DB
    "sql": "database-query", "query": "database-query",

    # Package management
    "install": "dependency-installation", "pip": "dependency-installation",
    "npm": "dependency-installation", "yarn": "dependency-installation",
}


def detect_code_fence_languages(text):
    """Extract language tags from markdown code fences."""
    pattern = r'```(\w[\w+#.-]*)'
    matches = re.findall(pattern, text)
    languages = set()
    for match in matches:
        lang = match.lower().strip()
        if lang in FENCE_LANG_MAP:
            languages.add(FENCE_LANG_MAP[lang])
    return sorted(languages)


def detect_framework_languages(text):
    """Infer languages from framework/library mentions."""
    text_lower = text.lower()
    languages = set()
    for framework, lang in FRAMEWORK_LANG_MAP.items():
        if framework in text_lower:
            languages.add(lang)
    return sorted(languages)


def count_code_blocks(text):
    """Count number of code blocks in text."""
    return len(re.findall(r'```', text)) // 2


def extract_tool_signals(conversations):
    """Extract agentic signals from tool role messages."""
    tool_names = []
    agentic_tags = set()

    for turn in conversations:
        if turn.get("from") == "tool":
            value = turn.get("value", "")

            # Try to detect tool name from common patterns
            # Pattern: "$ command ..." (shell)
            if value.strip().startswith("$") or value.strip().startswith("#"):
                tool_names.append("bash")
                agentic_tags.add("bash-execution")

                # Check specific commands within bash
                cmd = value.strip().lstrip("$ ").split()[0] if value.strip().lstrip("$ ") else ""
                if cmd in ("cat", "ls", "find", "head", "tail", "grep"):
                    agentic_tags.add("file-operations")
                elif cmd in ("git",):
                    agentic_tags.add("git-operations")
                elif cmd in ("npm", "pip", "yarn", "pnpm", "cargo", "go"):
                    subargs = value.strip().lstrip("$ ").split()
                    if len(subargs) > 1 and subargs[1] in ("install", "add", "get"):
                        agentic_tags.add("dependency-installation")
                elif cmd in ("pytest", "jest", "go test", "cargo test"):
                    agentic_tags.add("test-running")
                elif cmd in ("docker", "make", "cargo build", "npm run build"):
                    agentic_tags.add("build-execution")
                elif cmd in ("python", "node", "ruby", "go run"):
                    agentic_tags.add("code-execution")

            # Pattern: structured tool call JSON-like
            if '"name"' in value or '"tool"' in value:
                for tool_key, tag in TOOL_NAME_MAP.items():
                    if tool_key in value.lower():
                        tool_names.append(tool_key)
                        agentic_tags.add(tag)

    return sorted(set(tool_names)), sorted(agentic_tags)


def detect_behavioral_patterns(conversations):
    """Detect agentic behavioral patterns from conversation structure."""
    patterns = set()
    len(conversations)
    gpt_turns = [t for t in conversations if t.get("from") == "gpt"]
    tool_turns = [t for t in conversations if t.get("from") == "tool"]

    # Multi-file coordination: multiple file operations on different paths
    file_paths = set()
    for t in tool_turns:
        val = t.get("value", "")
        paths = re.findall(r'(?:cat|read|write|edit)\s+(\S+\.\w+)', val)
        paths += re.findall(r'(?:>\s*)(\S+\.\w+)', val)
        file_paths.update(paths)
    if len(file_paths) >= 2:
        patterns.add("multi-file-coordination")

    # Iterative refinement: multiple tool calls suggesting try-fix-retry
    if len(tool_turns) >= 3:
        patterns.add("iterative-refinement")

    # Planning: first gpt turn contains plan-like language
    if gpt_turns:
        first_gpt = gpt_turns[0].get("value", "").lower()
        plan_signals = ["let me", "first", "step 1", "plan", "i'll start by",
                        "先", "首先", "步骤", "计划", "我来"]
        if any(s in first_gpt for s in plan_signals):
            patterns.add("planning")

    # Multi-step reasoning: long gpt response with sequential logic
    for t in gpt_turns:
        val = t.get("value", "")
        if len(val) > 500:
            reasoning_signals = ["because", "therefore", "this means", "so we need",
                                 "因为", "所以", "这意味着", "因此", "根本原因"]
            if sum(1 for s in reasoning_signals if s in val.lower()) >= 2:
                patterns.add("multi-step-reasoning")
                break

    # Error recovery: error message followed by a fix attempt
    for i, t in enumerate(conversations):
        val = t.get("value", "").lower()
        if any(w in val for w in ["error", "failed", "exception", "traceback", "panic", "报错"]):
            if i + 1 < len(conversations):
                patterns.add("error-recovery")
                break

    return sorted(patterns)


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for mixed en/zh content."""
    return len(text) // 4


def extract_last_turn(conversations):
    """Extract the last user query and last assistant response."""
    last_human = ""
    last_gpt = ""
    for turn in reversed(conversations):
        if turn.get("from") == "human" and not last_human:
            last_human = turn.get("value", "")
        if turn.get("from") == "gpt" and not last_gpt:
            last_gpt = turn.get("value", "")
        if last_human and last_gpt:
            break
    return last_human, last_gpt


def detect_keywords(text):
    """Detect domain/topic keywords for context hints."""
    text_lower = text.lower()
    hits = []
    keyword_groups = {
        "web": ["react", "vue", "angular", "next.js", "express", "django", "flask",
                "html", "css", "api", "rest", "graphql", "frontend", "backend"],
        "devops": ["docker", "kubernetes", "k8s", "terraform", "ansible", "ci/cd",
                   "github actions", "jenkins", "helm", "nginx", "prometheus", "grafana"],
        "database": ["sql", "postgresql", "mysql", "mongodb", "redis", "sqlite",
                     "database", "query", "index", "migration", "schema"],
        "ml": ["pytorch", "tensorflow", "model", "training", "neural", "dataset",
               "classification", "regression", "epoch", "loss", "optimizer"],
        "security": ["auth", "oauth", "jwt", "xss", "csrf", "injection", "encryption",
                     "password", "token", "vulnerability", "security"],
        "mobile": ["ios", "android", "swift", "kotlin", "flutter", "react native"],
        "systems": ["kernel", "memory", "allocator", "lock-free", "atomic", "assembly",
                    "embedded", "firmware", "rtos"],
    }
    for group, keywords in keyword_groups.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            hits.append((group, matches))
    return hits


def generate_sparse_schedule(n, full_label_count=8, gap_multiplier=1.3, min_gap=2, max_gap=8, threshold=12):
    """Front-dense, back-sparse sampling schedule for pyramid slices.

    n <= threshold: label all (return [0..n-1])
    n > threshold: first full_label_count all labeled, then gap grows ~gap_multiplier
                   (capped at max_gap), last always labeled
    """
    if n <= threshold:
        return list(range(n))

    schedule = list(range(full_label_count))  # first N always labeled
    pos = full_label_count
    gap = float(min_gap)
    while pos < n - 1:
        schedule.append(pos)
        gap *= gap_multiplier
        gap = min(gap, max_gap)
        pos = min(pos + max(int(gap), min_gap), n - 1)

    # Always include the last slice
    if schedule[-1] != n - 1:
        schedule.append(n - 1)

    return schedule


_FILE_SCOPE_RE = re.compile(r'(?:[\w./-]+\.[A-Za-z0-9]{1,8}|[\w./-]+/[\w./-]+)')


def _normalized_char_ngrams(text, n=3, max_ngrams=256):
    """Build a lightweight language-agnostic shingle set for semantic drift checks."""
    normalized = re.sub(r'\s+', ' ', (text or '').lower()).strip()
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    total = len(normalized) - n + 1
    step = max(total // max_ngrams, 1)
    ngrams = []
    for i in range(0, total, step):
        gram = normalized[i:i + n]
        if gram.strip():
            ngrams.append(gram)
        if len(ngrams) >= max_ngrams:
            break
    return set(ngrams)


def _jaccard_similarity(left, right):
    """Return Jaccard similarity between two sets."""
    if not left or not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _current_turn_signature(sample):
    """Build lightweight structural signals for inheritance decisions."""
    convs = sample.get("conversations", [])
    segments = _extract_scoring_segments(convs)

    request = segments["current_request"]
    trajectory = segments["trajectory"]
    response = segments["response"]
    window_text = "\n".join(part for part in (request, trajectory, response) if part)

    keyword_groups = {group for group, _matches in detect_keywords(request)}
    fence_langs = set(detect_code_fence_languages(response))
    if not fence_langs and window_text:
        fence_langs = set(detect_code_fence_languages(window_text))

    role_pattern = []
    last_assistant_idx = _find_last_assistant_idx(convs)
    request_idx = _find_current_request_idx(convs, last_assistant_idx) if last_assistant_idx >= 0 else -1
    if request_idx >= 0 and last_assistant_idx >= request_idx:
        role_pattern = [turn.get("from", "") for turn in convs[request_idx:last_assistant_idx + 1]]

    return {
        "keyword_groups": keyword_groups,
        "fence_langs": fence_langs,
        "trajectory_tool_turns": segments["trajectory_tool_turns"],
        "trajectory_turn_count": segments["trajectory_turn_count"],
        "code_block_count": count_code_blocks(window_text),
        "has_file_scope": bool(_FILE_SCOPE_RE.search(window_text)),
        "role_pattern": tuple(role_pattern),
        "request_len": len(request.strip()),
        "response_len": len(response.strip()),
        "window_len": len(window_text.strip()),
        "request_ngrams": _normalized_char_ngrams(request),
        "response_ngrams": _normalized_char_ngrams(response),
        "window_ngrams": _normalized_char_ngrams(window_text),
    }


def _planner_boundary_score(source_sig, target_sig):
    """Return deterministic boundary score/reasons for adjacent local views."""
    score = 0.0
    reasons = []

    s_langs = source_sig["fence_langs"]
    t_langs = target_sig["fence_langs"]
    if s_langs and t_langs and s_langs != t_langs:
        score += 3.0
        reasons.append("language_change")

    if source_sig["trajectory_tool_turns"] != target_sig["trajectory_tool_turns"]:
        score += 2.5
        reasons.append("tool_pattern_change")
    if source_sig["role_pattern"] != target_sig["role_pattern"] and (
        source_sig["trajectory_turn_count"] or target_sig["trajectory_turn_count"]
    ):
        score += 2.0
        reasons.append("role_pattern_change")

    s_code_blocks = source_sig["code_block_count"]
    t_code_blocks = target_sig["code_block_count"]
    if (s_code_blocks == 0) != (t_code_blocks == 0):
        score += 2.0
        reasons.append("code_density_flip")
    elif abs(s_code_blocks - t_code_blocks) >= 2:
        score += 1.0
        reasons.append("code_density_shift")

    if source_sig["has_file_scope"] != target_sig["has_file_scope"]:
        score += 2.0
        reasons.append("file_scope_change")

    s_keywords = source_sig["keyword_groups"]
    t_keywords = target_sig["keyword_groups"]
    if s_keywords and t_keywords and s_keywords != t_keywords:
        score += 1.5
        reasons.append("keyword_shift")

    request_similarity = _jaccard_similarity(
        source_sig["request_ngrams"], target_sig["request_ngrams"]
    )
    if (request_similarity is not None
            and min(source_sig["request_len"], target_sig["request_len"]) >= 24
            and min(len(source_sig["request_ngrams"]), len(target_sig["request_ngrams"])) >= 8
            and request_similarity < 0.25):
        score += 1.5
        reasons.append("request_similarity_drop")
    elif (request_similarity is not None
            and min(source_sig["request_len"], target_sig["request_len"]) >= 12
            and request_similarity < 0.10):
        score += 2.0
        reasons.append("short_request_shift")

    response_similarity = _jaccard_similarity(
        source_sig["response_ngrams"], target_sig["response_ngrams"]
    )
    if (response_similarity is not None
            and min(source_sig["response_len"], target_sig["response_len"]) >= 32
            and min(len(source_sig["response_ngrams"]), len(target_sig["response_ngrams"])) >= 8
            and response_similarity < 0.20):
        score += 1.5
        reasons.append("response_similarity_drop")
    elif (response_similarity is not None
            and min(source_sig["response_len"], target_sig["response_len"]) >= 16
            and response_similarity < 0.08):
        score += 2.0
        reasons.append("short_response_shift")

    window_similarity = _jaccard_similarity(
        source_sig["window_ngrams"], target_sig["window_ngrams"]
    )
    if (window_similarity is not None
            and min(source_sig["window_len"], target_sig["window_len"]) >= 48
            and min(len(source_sig["window_ngrams"]), len(target_sig["window_ngrams"])) >= 12
            and window_similarity < 0.22):
        score += 1.0
        reasons.append("window_similarity_drop")

    return score, reasons


def _merge_tiny_segments(segments, min_segment_size):
    if len(segments) <= 1:
        return segments

    merged = []
    for segment in segments:
        if merged and len(segment) < min_segment_size:
            merged[-1].extend(segment)
            continue
        merged.append(list(segment))

    if len(merged) > 1 and len(merged[0]) < min_segment_size:
        merged[1] = merged[0] + merged[1]
        merged = merged[1:]

    return merged


def _annotate_planner_group(
    ordered_samples,
    *,
    planner_policy=PLANNER_POLICY,
    boundary_threshold=PLANNER_BOUNDARY_THRESHOLD,
    min_segment_size=PLANNER_MIN_SEGMENT_SIZE,
    max_anchor_gap=PLANNER_MAX_ANCHOR_GAP,
    fallback_boundary_ratio=PLANNER_FALLBACK_BOUNDARY_RATIO,
):
    if len(ordered_samples) <= 1:
        return

    signatures = [_current_turn_signature(sample) for sample in ordered_samples]
    boundary_scores = [0.0] * len(ordered_samples)
    boundary_reasons = [[] for _ in ordered_samples]
    boundary_before = [False] * len(ordered_samples)

    for idx in range(1, len(ordered_samples)):
        score, reasons = _planner_boundary_score(signatures[idx - 1], signatures[idx])
        boundary_scores[idx] = round(score, 3)
        boundary_reasons[idx] = reasons
        boundary_before[idx] = score >= boundary_threshold

    boundary_count = sum(1 for flag in boundary_before[1:] if flag)
    boundary_ratio = boundary_count / max(1, len(ordered_samples) - 1)
    planner_fallback = (
        len(ordered_samples) >= 8
        and boundary_count >= 3
        and boundary_ratio > float(fallback_boundary_ratio)
    )

    if planner_fallback:
        first_turn = (ordered_samples[0].get("metadata") or {}).get("turn_index") or 1
        last_turn = (ordered_samples[-1].get("metadata") or {}).get("turn_index") or len(ordered_samples)
        for sample in ordered_samples:
            meta = sample.setdefault("metadata", {})
            turn_index = meta.get("turn_index") or first_turn
            meta["planner_policy"] = planner_policy
            meta["planner_confidence"] = 0.0
            meta["planner_fallback"] = True
            meta["segment_id"] = 0
            meta["segment_turn_start"] = first_turn
            meta["segment_turn_end"] = last_turn
            meta["boundary_score"] = 0.0
            meta["anchor_priority"] = "required" if turn_index in {first_turn, last_turn} else "inherit"
            meta["anchor_reason"] = "segment_end" if turn_index == last_turn else ("segment_start" if turn_index == first_turn else "inherit")
            meta["anchor_distance"] = min(abs(turn_index - first_turn), abs(last_turn - turn_index))
            meta["inherit_group_id"] = "segment-0"
        return

    segments = []
    start = 0
    for idx in range(1, len(ordered_samples)):
        if boundary_before[idx]:
            segments.append(list(range(start, idx)))
            start = idx
    segments.append(list(range(start, len(ordered_samples))))
    segments = _merge_tiny_segments(segments, max(1, int(min_segment_size)))

    anchor_positions = set()
    position_to_segment = {}
    for segment_id, positions in enumerate(segments):
        for pos in positions:
            position_to_segment[pos] = segment_id
        anchor_positions.add(positions[0])
        anchor_positions.add(positions[-1])
        if len(positions) > max(1, int(max_anchor_gap)):
            anchor_positions.add(positions[len(positions) // 2])

    planner_confidence = round(max(0.0, min(1.0, 1.0 - (boundary_ratio * 0.8))), 3)

    for segment_id, positions in enumerate(segments):
        start_pos = positions[0]
        end_pos = positions[-1]
        start_turn = (ordered_samples[start_pos].get("metadata") or {}).get("turn_index") or (start_pos + 1)
        end_turn = (ordered_samples[end_pos].get("metadata") or {}).get("turn_index") or (end_pos + 1)
        segment_anchor_positions = sorted(anchor_positions & set(positions))

        for pos in positions:
            sample = ordered_samples[pos]
            meta = sample.setdefault("metadata", {})
            anchor_distance = min(abs(pos - anchor) for anchor in segment_anchor_positions) if segment_anchor_positions else None

            anchor_reason = "inherit"
            anchor_priority = "inherit"
            if pos == start_pos and pos == end_pos:
                anchor_reason = "segment_end"
                anchor_priority = "required"
            elif pos == start_pos:
                anchor_reason = "segment_boundary" if pos > 0 else "segment_start"
                anchor_priority = "boundary" if pos > 0 else "required"
            elif pos == end_pos:
                anchor_reason = "segment_end"
                anchor_priority = "required"
            elif pos in anchor_positions:
                anchor_reason = "segment_mid"
                anchor_priority = "required"
            elif boundary_scores[pos] >= boundary_threshold:
                anchor_reason = "segment_boundary"
                anchor_priority = "boundary"

            meta["segment_id"] = segment_id
            meta["segment_turn_start"] = start_turn
            meta["segment_turn_end"] = end_turn
            meta["boundary_score"] = boundary_scores[pos]
            meta["anchor_priority"] = anchor_priority
            meta["anchor_reason"] = anchor_reason
            meta["anchor_distance"] = anchor_distance
            meta["inherit_group_id"] = f"segment-{segment_id}"
            meta["planner_policy"] = planner_policy
            meta["planner_confidence"] = planner_confidence
            meta["planner_fallback"] = False
            if boundary_reasons[pos]:
                meta["boundary_reasons"] = list(boundary_reasons[pos])


def annotate_multiturn_planner_metadata(
    samples,
    *,
    planner_policy=PLANNER_POLICY,
    boundary_threshold=PLANNER_BOUNDARY_THRESHOLD,
    min_segment_size=PLANNER_MIN_SEGMENT_SIZE,
    max_anchor_gap=PLANNER_MAX_ANCHOR_GAP,
    fallback_boundary_ratio=PLANNER_FALLBACK_BOUNDARY_RATIO,
):
    """Add deterministic planner metadata to multi-turn samples in place."""
    groups = {}
    for idx, sample in enumerate(samples):
        meta = sample.get("metadata") or {}
        conversation_key = meta.get("conversation_uid") or meta.get("source_id")
        total_turns = meta.get("total_turns", 1)
        if not conversation_key or total_turns <= 1:
            continue
        turn_index = meta.get("turn_index", idx + 1)
        groups.setdefault(conversation_key, []).append((turn_index, sample))

    for members in groups.values():
        members.sort(key=lambda item: item[0])
        ordered_samples = [sample for _turn_index, sample in members]
        _annotate_planner_group(
            ordered_samples,
            planner_policy=planner_policy,
            boundary_threshold=boundary_threshold,
            min_segment_size=min_segment_size,
            max_anchor_gap=max_anchor_gap,
            fallback_boundary_ratio=fallback_boundary_ratio,
        )


def _should_force_label(source_sample, target_sample):
    """Check if target's signals differ enough from source to warrant fresh labeling.

    Prevents label inheritance when the adjacent slice has materially different
    characteristics (language change, tool role presence change).
    """
    source_sig = _current_turn_signature(source_sample)
    target_sig = _current_turn_signature(target_sample)

    score, _reasons = _planner_boundary_score(source_sig, target_sig)
    return score >= min(PLANNER_BOUNDARY_THRESHOLD, 2.0)


def _should_force_planner_segment_label(source_sample, target_sample):
    """Conservative divergence guard retained inside planner segments."""
    source_sig = _current_turn_signature(source_sample)
    target_sig = _current_turn_signature(target_sample)
    _score, reasons = _planner_boundary_score(source_sig, target_sig)
    high_risk_reasons = {
        "language_change",
        "file_scope_change",
        "code_density_flip",
    }
    return bool(high_risk_reasons & set(reasons))


_HARD_PATH_BOUNDARY_REASONS = {
    "language_change",
    "file_scope_change",
    "code_density_flip",
    "keyword_shift",
    "short_request_shift",
    "request_similarity_drop",
    "short_response_shift",
    "response_similarity_drop",
}


def _group_hard_boundary_edges(members, samples):
    """Return per-edge hard-boundary flags for adjacent samples within one group."""
    hard_edges = [False] * len(members)
    if len(members) <= 1:
        return hard_edges

    signatures = [
        _current_turn_signature(samples[global_idx])
        for global_idx, _turn_idx in members
    ]
    for pos in range(1, len(members)):
        _score, reasons = _planner_boundary_score(signatures[pos - 1], signatures[pos])
        hard_edges[pos] = bool(_HARD_PATH_BOUNDARY_REASONS & set(reasons))
    return hard_edges


def _path_has_hard_boundary(hard_edges, left_pos, right_pos):
    """Check whether any adjacent hop on the path crosses a hard boundary."""
    lo = min(left_pos, right_pos) + 1
    hi = max(left_pos, right_pos)
    for pos in range(lo, hi + 1):
        if 0 <= pos < len(hard_edges) and hard_edges[pos]:
            return True
    return False


def _choose_inheritance_source(pos_in_group, labeled_positions, hard_edges):
    """Prefer forward inheritance, but fall back backward when it is the only soft path."""
    source_pos = None
    for lp in labeled_positions:
        if lp > pos_in_group and not _path_has_hard_boundary(hard_edges, pos_in_group, lp):
            source_pos = lp
            break
    if source_pos is not None:
        return source_pos

    for lp in reversed(labeled_positions):
        if lp < pos_in_group and not _path_has_hard_boundary(hard_edges, pos_in_group, lp):
            return lp
    return None


def _legacy_sparse_group_plan(members, label_indices, inherit_map, samples,
                              full_label_count, gap_multiplier, min_gap, max_gap, threshold):
    n = len(members)
    schedule = generate_sparse_schedule(n, full_label_count, gap_multiplier, min_gap, max_gap, threshold)
    schedule_set = set(schedule)
    hard_edges = _group_hard_boundary_edges(members, samples)

    labeled_positions = []
    for pos_in_group, (global_idx, _) in enumerate(members):
        if pos_in_group in schedule_set:
            label_indices.add(global_idx)
            labeled_positions.append(pos_in_group)

    for pos_in_group, (global_idx, _) in enumerate(members):
        if pos_in_group in schedule_set:
            continue
        source_pos = _choose_inheritance_source(pos_in_group, labeled_positions, hard_edges)
        if source_pos is not None:
            source_global_idx = members[source_pos][0]
            inherit_map[global_idx] = source_global_idx
        else:
            label_indices.add(global_idx)


def apply_sparse_sampling(samples, full_label_count=8, gap_multiplier=1.3, min_gap=2, max_gap=8, threshold=12,
                          planner_enabled=ENABLE_COMPACT_SEMANTIC_PLANNER, planner_metadata_only=False,
                          planner_policy=PLANNER_POLICY, planner_boundary_threshold=PLANNER_BOUNDARY_THRESHOLD,
                          planner_min_segment_size=PLANNER_MIN_SEGMENT_SIZE, planner_max_anchor_gap=PLANNER_MAX_ANCHOR_GAP,
                          planner_fallback_boundary_ratio=PLANNER_FALLBACK_BOUNDARY_RATIO):
    """Apply sparse sampling to multi-turn pyramid slices.

    Single-turn samples (no conversation identity or total_turns=1) are always labeled.
    Multi-turn slices are grouped by conversation_uid (fallback: source_id), sorted by turn_index,
    and sampled via generate_sparse_schedule.

    Returns:
        label_indices: set[int] — global indices that need actual LLM labeling
        inherit_map: dict[int, int] — {unlabeled_idx: source_idx} for inheritance

    Inheritance strategy: look forward (inherit from next labeled slice).
    Reason: slice_i is a prefix of slice_j (j>i), so j's labels are more complete.
    Last unlabeled slice with no successor inherits backward.
    """
    label_indices = set()
    inherit_map = {}

    # Group by stable conversation identity; track global index
    groups = {}  # conversation_key -> [(global_idx, turn_index)]
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        conversation_uid = meta.get("conversation_uid")
        source_id = meta.get("source_id")
        total_turns = meta.get("total_turns", 1)
        conversation_key = conversation_uid or source_id

        if not conversation_key or total_turns <= 1:
            # Single-turn: always label
            label_indices.add(i)
        else:
            turn_idx = meta.get("turn_index", 1)
            groups.setdefault(conversation_key, []).append((i, turn_idx))

    if planner_enabled:
        annotate_multiturn_planner_metadata(
            samples,
            planner_policy=planner_policy,
            boundary_threshold=planner_boundary_threshold,
            min_segment_size=planner_min_segment_size,
            max_anchor_gap=planner_max_anchor_gap,
            fallback_boundary_ratio=planner_fallback_boundary_ratio,
        )

    # For each multi-turn group, apply sparse schedule
    for _conversation_key, members in groups.items():
        # Sort by turn_index
        members.sort(key=lambda x: x[1])
        n = len(members)
        if n <= threshold:
            for global_idx, _turn_idx in members:
                label_indices.add(global_idx)
            continue

        planner_ready = planner_enabled and not planner_metadata_only
        planner_ready = planner_ready and all(
            (samples[global_idx].get("metadata") or {}).get("planner_policy") == planner_policy
            for global_idx, _turn_idx in members
        )
        planner_ready = planner_ready and not any(
            (samples[global_idx].get("metadata") or {}).get("planner_fallback")
            for global_idx, _turn_idx in members
        )

        if not planner_ready:
            _legacy_sparse_group_plan(
                members, label_indices, inherit_map, samples,
                full_label_count, gap_multiplier, min_gap, max_gap, threshold,
            )
            continue

        legacy_group_label_indices = set()
        legacy_group_inherit_map = {}
        _legacy_sparse_group_plan(
            members, legacy_group_label_indices, legacy_group_inherit_map, samples,
            full_label_count, gap_multiplier, min_gap, max_gap, threshold,
        )

        planner_group_label_indices = set()
        planner_group_inherit_map = {}
        positions_by_segment = {}
        for pos_in_group, (global_idx, _turn_idx) in enumerate(members):
            meta = samples[global_idx].get("metadata") or {}
            segment_id = meta.get("segment_id", 0)
            positions_by_segment.setdefault(segment_id, []).append(pos_in_group)

        required_positions = set()
        for positions in positions_by_segment.values():
            required_positions.add(positions[0])
            required_positions.add(positions[-1])
            if len(positions) > max(1, int(planner_max_anchor_gap)):
                required_positions.add(positions[len(positions) // 2])

        if required_positions:
            for pos_in_group in sorted(required_positions):
                planner_group_label_indices.add(members[pos_in_group][0])

        for segment_positions in positions_by_segment.values():
            labeled_positions = [pos for pos in segment_positions if pos in required_positions]
            hard_edges = _group_hard_boundary_edges(
                [members[pos_in_group] for pos_in_group in segment_positions],
                samples,
            )
            for pos_in_group in segment_positions:
                if pos_in_group in required_positions:
                    continue

                local_pos = segment_positions.index(pos_in_group)
                local_labeled_positions = [
                    segment_positions.index(lp)
                    for lp in labeled_positions
                ]
                source_local_pos = _choose_inheritance_source(
                    local_pos,
                    local_labeled_positions,
                    hard_edges,
                )
                source_pos = (
                    segment_positions[source_local_pos]
                    if source_local_pos is not None
                    else None
                )
                if source_pos is None:
                    planner_group_label_indices.add(members[pos_in_group][0])
                    continue

                global_idx = members[pos_in_group][0]
                source_global_idx = members[source_pos][0]
                planner_group_inherit_map[global_idx] = source_global_idx

        if len(planner_group_label_indices) >= len(legacy_group_label_indices):
            label_indices.update(legacy_group_label_indices)
            inherit_map.update(legacy_group_inherit_map)
        else:
            label_indices.update(planner_group_label_indices)
            inherit_map.update(planner_group_inherit_map)

    return label_indices, inherit_map


def preprocess(sample):
    """
    Full preprocessing pipeline for one SFT sample.

    Args:
        sample: dict in ShareGPT or Pangu format (auto-detected)

    Returns:
        dict with all extracted signals
    """
    sample = normalize_sample(sample)
    conversations = sample.get("conversations", [])
    full_text = " ".join(t.get("value", "") for t in conversations)

    # Language detection
    fence_langs = detect_code_fence_languages(full_text)
    framework_langs = detect_framework_languages(full_text)
    all_detected_langs = sorted(set(fence_langs + framework_langs))

    # Tool / agentic detection
    has_tool_roles = any(t.get("from") == "tool" for t in conversations)
    tool_names, tool_agentic_tags = extract_tool_signals(conversations)
    behavioral_patterns = detect_behavioral_patterns(conversations)

    # Conversation structure
    total_turns = len(conversations)
    code_block_count = count_code_blocks(full_text)
    last_query, last_response = extract_last_turn(conversations)

    # Keywords
    keyword_hits = detect_keywords(full_text)

    # Token estimation
    est_tokens = estimate_tokens(full_text)

    signals = {
        "detected_languages": all_detected_langs,
        "fence_languages": fence_langs,
        "framework_languages": framework_langs,
        "has_tool_roles": has_tool_roles,
        "tool_names": tool_names,
        "tool_agentic_tags": tool_agentic_tags,
        "behavioral_patterns": behavioral_patterns,
        "total_turns": total_turns,
        "code_block_count": code_block_count,
        "est_tokens": est_tokens,
        "keyword_hits": keyword_hits,
        "last_query_preview": last_query[:200] if last_query else "",
        "last_response_length": len(last_response),
    }

    return signals


def format_signals_for_prompt(signals):
    """Format preprocessed signals as a string to embed in the LLM prompt."""
    lines = []
    if signals["detected_languages"]:
        lines.append(f"detected_languages: {signals['detected_languages']}")
    lines.append(f"has_tool_roles: {str(signals['has_tool_roles']).lower()}")
    if signals["tool_names"]:
        lines.append(f"tool_names: {signals['tool_names']}")
    if signals["tool_agentic_tags"]:
        lines.append(f"tool_agentic_tags_detected: {signals['tool_agentic_tags']}")
    if signals["behavioral_patterns"]:
        lines.append(f"behavioral_patterns_detected: {signals['behavioral_patterns']}")
    lines.append(f"code_block_count: {signals['code_block_count']}")
    lines.append(f"total_turns: {signals['total_turns']}")
    lines.append(f"est_tokens: {signals['est_tokens']}")
    if signals["keyword_hits"]:
        kw_str = "; ".join(f"{g}: {', '.join(kws)}" for g, kws in signals["keyword_hits"])
        lines.append(f"keyword_hints: {kw_str}")
    return "\n".join(lines)
