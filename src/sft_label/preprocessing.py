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


def detect_format(sample):
    """Detect whether sample is ShareGPT or Pangu format."""
    if "conversations" in sample:
        return "sharegpt"
    if "data" in sample:
        return "pangu"
    return "unknown"


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


def normalize_and_slice(sample, *, source_file=None, source_row=None):
    """Auto-detect format, normalize, and slice multi-turn into training samples.

    Returns a list of samples. Single-turn and pseudo multi-turn return [1 sample].
    True multi-turn returns [N samples], one per assistant reply.
    Each slice has id suffixed with turn number (e.g. "id_t1", "id_t2").
    """
    fmt = detect_format(sample)
    if fmt == "pangu":
        normalized = normalize_pangu(sample)
    else:
        normalized = dict(sample)
        if "conversations" in normalized:
            raw_convs = normalized["conversations"]
            reply_cot = []
            # Detect and save COT metadata before stripping (Pangu parity)
            thinking_mode = detect_thinking_mode(raw_convs)
            if thinking_mode == "slow":
                cot_text, _, _ = extract_cot_content(raw_convs)
                normalized.setdefault("metadata", {})["thinking_mode"] = "slow"
                if cot_text:
                    normalized["metadata"]["cot_text"] = cot_text
            for turn in raw_convs:
                if turn.get("from") == "gpt":
                    reply_cot.append(extract_turn_cot_text(turn.get("value", "")))
            if reply_cot:
                normalized.setdefault("metadata", {})["assistant_cot_by_reply"] = reply_cot
            # Strip CoT from conversations
            for turn in raw_convs:
                if turn.get("from") == "gpt" and turn.get("value"):
                    turn["value"] = strip_cot(turn["value"])

    conversations = normalized.get("conversations", [])
    is_pseudo = normalized.get("metadata", {}).get("is_pseudo_multiturn", False)

    # Pseudo multi-turn: already one training sample, don't slice
    if is_pseudo:
        return [normalized]

    slices = slice_multiturn(conversations)

    base_meta = normalized.get("metadata", {})
    conversation_uid = base_meta.get("conversation_uid")
    if not conversation_uid and len(slices) > 1:
        conversation_uid = build_conversation_uid(
            normalized,
            source_file=source_file,
            source_row=source_row,
        )

    if len(slices) == 1:
        if conversation_uid:
            normalized.setdefault("metadata", {})["conversation_uid"] = conversation_uid
        trajectory_meta = _detect_single_reply_trajectory_object(conversations)
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
def normalize_sample(sample):
    """Normalize without slicing (for tools that don't need multi-turn expansion)."""
    fmt = detect_format(sample)
    if fmt == "pangu":
        return normalize_pangu(sample)
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


def _should_force_label(source_sample, target_sample):
    """Check if target's signals differ enough from source to warrant fresh labeling.

    Prevents label inheritance when the adjacent slice has materially different
    characteristics (language change, tool role presence change).
    """
    source_sig = _current_turn_signature(source_sample)
    target_sig = _current_turn_signature(target_sample)

    # Compare last-turn code languages
    s_langs = source_sig["fence_langs"]
    t_langs = target_sig["fence_langs"]
    if s_langs and t_langs and s_langs != t_langs:
        return True

    # Tool execution pattern changed materially.
    if source_sig["trajectory_tool_turns"] != target_sig["trajectory_tool_turns"]:
        return True
    if source_sig["role_pattern"] != target_sig["role_pattern"] and (
        source_sig["trajectory_turn_count"] or target_sig["trajectory_turn_count"]
    ):
        return True

    # Current-turn evidence moved between plain text and code-heavy.
    s_code_blocks = source_sig["code_block_count"]
    t_code_blocks = target_sig["code_block_count"]
    if (s_code_blocks == 0) != (t_code_blocks == 0):
        return True
    if abs(s_code_blocks - t_code_blocks) >= 2:
        return True

    # File-scope changes usually indicate a different coordination burden.
    if source_sig["has_file_scope"] != target_sig["has_file_scope"]:
        return True

    # Query focus changed between materially different keyword groups.
    s_keywords = source_sig["keyword_groups"]
    t_keywords = target_sig["keyword_groups"]
    if s_keywords and t_keywords and s_keywords != t_keywords:
        return True

    # Semantic drift: similar structure but materially different request/answer.
    request_similarity = _jaccard_similarity(
        source_sig["request_ngrams"], target_sig["request_ngrams"]
    )
    if (request_similarity is not None
            and min(source_sig["request_len"], target_sig["request_len"]) >= 24
            and min(len(source_sig["request_ngrams"]), len(target_sig["request_ngrams"])) >= 8
            and request_similarity < 0.25):
        return True

    response_similarity = _jaccard_similarity(
        source_sig["response_ngrams"], target_sig["response_ngrams"]
    )
    if (response_similarity is not None
            and min(source_sig["response_len"], target_sig["response_len"]) >= 32
            and min(len(source_sig["response_ngrams"]), len(target_sig["response_ngrams"])) >= 8
            and response_similarity < 0.20):
        return True

    window_similarity = _jaccard_similarity(
        source_sig["window_ngrams"], target_sig["window_ngrams"]
    )
    if (window_similarity is not None
            and min(source_sig["window_len"], target_sig["window_len"]) >= 48
            and min(len(source_sig["window_ngrams"]), len(target_sig["window_ngrams"])) >= 12
            and window_similarity < 0.22):
        return True

    return False


def apply_sparse_sampling(samples, full_label_count=8, gap_multiplier=1.3, min_gap=2, max_gap=8, threshold=12):
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

    # For each multi-turn group, apply sparse schedule
    for _conversation_key, members in groups.items():
        # Sort by turn_index
        members.sort(key=lambda x: x[1])
        n = len(members)
        schedule = generate_sparse_schedule(n, full_label_count, gap_multiplier, min_gap, max_gap, threshold)
        schedule_set = set(schedule)

        # Mark labeled indices
        labeled_positions = []
        for pos_in_group, (global_idx, _) in enumerate(members):
            if pos_in_group in schedule_set:
                label_indices.add(global_idx)
                labeled_positions.append(pos_in_group)

        # Build inherit_map: unlabeled → nearest labeled (forward first, backward fallback)
        # Signal change detection: if target differs from source, force fresh labeling
        for pos_in_group, (global_idx, _) in enumerate(members):
            if pos_in_group in schedule_set:
                continue
            # Find next labeled position (forward)
            source_pos = None
            for lp in labeled_positions:
                if lp > pos_in_group:
                    source_pos = lp
                    break
            # Fallback: previous labeled position (backward)
            if source_pos is None:
                for lp in reversed(labeled_positions):
                    if lp < pos_in_group:
                        source_pos = lp
                        break
            if source_pos is not None:
                source_global_idx = members[source_pos][0]
                # Check for signal divergence before inheriting
                if _should_force_label(samples[source_global_idx], samples[global_idx]):
                    label_indices.add(global_idx)
                else:
                    inherit_map[global_idx] = source_global_idx

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
