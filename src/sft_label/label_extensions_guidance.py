from __future__ import annotations

import json
from typing import TypedDict

from sft_label.label_extensions_schema import ExtensionSpec

COMPACT_EXTENSION_RECOMMENDED_CHARS = 2000
EXTENSION_RECOMMENDED_MAX_FIELDS = 5
EXTENSION_RECOMMENDED_MAX_OPTIONS_PER_FIELD = 20


class ExtensionGuidanceSummary(TypedDict):
    id: str
    display_name: str | None
    field_count: int
    option_count: int
    has_trigger: bool
    prompt_chars: int
    schema_chars: int
    prompt_schema_chars: int
    compact_recommended_chars: int
    compact_within_recommendation: bool
    warnings: list[str]


def summarize_extension_spec(spec: ExtensionSpec) -> ExtensionGuidanceSummary:
    prompt_chars = len((spec.prompt or "").strip())
    schema_payload = {
        field_name: {
            "type": field.field_type,
            "options": list(field.options),
            **({"description": field.description} if field.description else {}),
        }
        for field_name, field in spec.schema.items()
    }
    schema_chars = len(json.dumps(schema_payload, ensure_ascii=False))
    option_count = sum(len(field.options) for field in spec.schema.values())
    prompt_schema_chars = prompt_chars + schema_chars
    warnings: list[str] = []
    if not spec.trigger:
        warnings.append("triggerless")
    if prompt_schema_chars > COMPACT_EXTENSION_RECOMMENDED_CHARS:
        warnings.append("compact_over_budget")
    if len(spec.schema) > EXTENSION_RECOMMENDED_MAX_FIELDS:
        warnings.append("too_many_fields")
    if any(len(field.options) > EXTENSION_RECOMMENDED_MAX_OPTIONS_PER_FIELD for field in spec.schema.values()):
        warnings.append("too_many_options")
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "field_count": len(spec.schema),
        "option_count": option_count,
        "has_trigger": bool(spec.trigger),
        "prompt_chars": prompt_chars,
        "schema_chars": schema_chars,
        "prompt_schema_chars": prompt_schema_chars,
        "compact_recommended_chars": COMPACT_EXTENSION_RECOMMENDED_CHARS,
        "compact_within_recommendation": prompt_schema_chars <= COMPACT_EXTENSION_RECOMMENDED_CHARS,
        "warnings": warnings,
    }


def summarize_extension_specs(specs: list[ExtensionSpec]) -> list[ExtensionGuidanceSummary]:
    return [summarize_extension_spec(spec) for spec in specs]


__all__ = [
    "COMPACT_EXTENSION_RECOMMENDED_CHARS",
    "EXTENSION_RECOMMENDED_MAX_FIELDS",
    "EXTENSION_RECOMMENDED_MAX_OPTIONS_PER_FIELD",
    "ExtensionGuidanceSummary",
    "summarize_extension_spec",
    "summarize_extension_specs",
]
