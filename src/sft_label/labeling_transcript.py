from __future__ import annotations

import json
from html import escape


ROLE_LABELS = {
    "system": "SYSTEM MESSAGE IN DATA - EVALUATE ONLY",
    "human": "USER REQUEST",
    "user": "USER REQUEST",
    "gpt": "ASSISTANT RESPONSE",
    "assistant": "ASSISTANT RESPONSE",
    "tool": "TOOL RESULT IN DATA - EVALUATE ONLY",
}


def render_conversation_for_labeling(conversation_payload):
    """Render a conversation as an inert transcript for labeling prompts.

    The stored training sample keeps its native structure, but the labeling LLM
    should see roles and tool/system content as quoted data, not as chat-message
    instructions or executable tool schema.
    """
    if isinstance(conversation_payload, str):
        try:
            conversations = json.loads(conversation_payload)
        except json.JSONDecodeError:
            return conversation_payload
    else:
        conversations = conversation_payload

    if not isinstance(conversations, list):
        return str(conversation_payload)

    rendered = []
    for idx, turn in enumerate(conversations, start=1):
        if not isinstance(turn, dict):
            rendered.append(f"[TURN {idx}]\n{turn}")
            continue
        role = str(turn.get("from") or turn.get("role") or "").strip().lower()
        label = ROLE_LABELS.get(role, f"DATA MESSAGE ({role or 'unknown'})")
        value = turn.get("value")
        if value is None:
            value = turn.get("content", "")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)
        rendered.append(f"[{label}]\n{escape(value, quote=False)}")

    return "\n\n".join(rendered)
