from __future__ import annotations

import json
from typing import Awaitable, Callable, Mapping

from sft_label.labeling_transcript import render_conversation_for_labeling
from sft_label.label_extensions_schema import ExtensionSpec, SchemaField, match_extension_specs, spec_hash

ExtensionLLMCaller = Callable[[list[dict], ExtensionSpec], Awaitable[dict]]


def build_extension_messages(
    conversation_json: str,
    preprocessed_signals: str,
    core_labels: Mapping[str, object],
    spec: ExtensionSpec,
) -> list[dict]:
    """Build a deterministic extension-labeling prompt for one spec."""

    conversation_text = render_conversation_for_labeling(conversation_json)
    schema_payload = {
        field_name: {
            "type": field.field_type,
            "options": list(field.options),
            **({"description": field.description} if field.description else {}),
        }
        for field_name, field in spec.schema.items()
    }
    user_content = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n"
        f"<core_labels>\n{json.dumps(dict(core_labels), ensure_ascii=False)}\n</core_labels>\n\n"
        f"<preprocessed_signals>\n{preprocessed_signals}\n</preprocessed_signals>\n\n"
        f"<extension_schema>\n{json.dumps(schema_payload, ensure_ascii=False)}\n</extension_schema>"
    )
    system_content = (
        f"{spec.prompt.strip()}\n\n"
        "Return ONLY valid JSON. Use the schema field ids exactly as given. "
        "For enum fields return a string or empty string. For multi_enum fields return an array of strings. "
        "If confidence is available, return a confidence object keyed by field id with values in [0,1]."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def run_label_extensions(
    *,
    conversation_json: str,
    preprocessed_signals: str,
    core_labels: Mapping[str, object],
    extension_specs: list[ExtensionSpec],
    llm_caller: ExtensionLLMCaller,
) -> dict[str, dict]:
    """Execute all enabled extension specs for a single sample."""

    matched_ids = {spec.id for spec in match_extension_specs(extension_specs, core_labels)}
    results: dict[str, dict] = {}

    for spec in extension_specs:
        base_payload = {
            "status": "skipped",
            "matched": spec.id in matched_ids,
            "spec_version": spec.spec_version,
            "spec_hash": spec_hash(spec),
            "labels": {},
            "confidence": {},
            "unmapped": [],
            "monitor": {"status": "skipped", "llm_calls": 0},
        }
        if not spec.enabled or spec.id not in matched_ids:
            results[spec.id] = base_payload
            continue

        messages = build_extension_messages(conversation_json, preprocessed_signals, core_labels, spec)
        try:
            response = await llm_caller(messages, spec)
        except Exception as exc:  # pragma: no cover - exercised in tests
            failed_payload = dict(base_payload)
            failed_payload.update(
                status="failed",
                monitor={"status": "failed", "llm_calls": 1, "error": str(exc)},
            )
            results[spec.id] = failed_payload
            continue

        validated = validate_extension_response(response, spec)
        validated["matched"] = True
        validated["spec_version"] = spec.spec_version
        validated["spec_hash"] = spec_hash(spec)
        validated["monitor"] = {"status": validated["status"], "llm_calls": 1}
        results[spec.id] = validated

    return results


def validate_extension_response(response: Mapping[str, object], spec: ExtensionSpec) -> dict[str, object]:
    """Validate and normalize one extension response payload."""

    labels: dict[str, object] = {}
    unmapped: list[dict[str, str]] = []
    include_confidence = bool(spec.output.get("include_confidence", True))
    allow_unmapped = bool(spec.output.get("allow_unmapped", True))
    confidence = _normalize_confidence(response.get("confidence"), spec) if include_confidence else {}
    invalid = False

    for field_name, field in spec.schema.items():
        raw_value = response.get(field_name)
        normalized, field_invalid, field_unmapped = _normalize_field(raw_value, field_name, field)
        labels[field_name] = normalized
        if field_invalid:
            invalid = True
        if allow_unmapped:
            unmapped.extend(field_unmapped)

    return {
        "status": "invalid" if invalid else "success",
        "labels": labels,
        "confidence": confidence,
        "unmapped": unmapped,
    }


def _normalize_field(raw_value: object, field_name: str, field: SchemaField) -> tuple[object, bool, list[dict[str, str]]]:
    if field.field_type == "multi_enum":
        values = _coerce_multi_values(raw_value)
        normalized: list[str] = []
        invalid = False
        unmapped: list[dict[str, str]] = []
        for value in values:
            if value in field.options:
                if value not in normalized:
                    normalized.append(value)
            else:
                invalid = True
                unmapped.append({"dimension": field_name, "value": value})
        return normalized, invalid, unmapped

    if field.field_type == "enum":
        if raw_value in (None, ""):
            return "", False, []
        value = str(raw_value)
        if value in field.options:
            return value, False, []
        return "", True, [{"dimension": field_name, "value": value}]

    raise ValueError(f"unsupported extension field type: {field.field_type}")


def _coerce_multi_values(raw_value: object) -> list[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if item not in (None, "")]
    return [str(raw_value)]


def _normalize_confidence(raw_confidence: object, spec: ExtensionSpec) -> dict[str, float]:
    if not isinstance(raw_confidence, Mapping):
        return {}

    confidence: dict[str, float] = {}
    for field_name in spec.schema:
        score = raw_confidence.get(field_name)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        score = max(0.0, min(1.0, float(score)))
        confidence[field_name] = score
    return confidence


__all__ = [
    "ExtensionLLMCaller",
    "build_extension_messages",
    "run_label_extensions",
    "validate_extension_response",
]
