from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

DEFAULT_OUTPUT = {
    "include_confidence": True,
    "allow_unmapped": True,
}

_TRIGGER_SUFFIXES = {
    "_any_of": "any",
    "_all_of": "all",
}
_SUPPORTED_FIELD_TYPES = {"enum", "multi_enum"}
_SUPPORTED_TRIGGER_LABELS = {"domain", "language", "intent", "task", "context", "difficulty"}


@dataclasses.dataclass(frozen=True)
class SchemaField:
    field_type: str
    options: tuple[str, ...]
    description: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SchemaField":
        if not isinstance(data, Mapping):
            raise ValueError("schema field must be a mapping")

        field_type = data.get("type")
        if not field_type:
            raise ValueError("schema field missing 'type'")
        field_type = str(field_type)
        if field_type not in _SUPPORTED_FIELD_TYPES:
            raise ValueError(f"unsupported schema field type '{field_type}'")

        options = data.get("options")
        if options is None:
            raise ValueError("schema field missing 'options'")
        if isinstance(options, str):
            raise ValueError("schema field 'options' must be a sequence")

        normalized: list[str] = []
        seen: set[str] = set()
        for entry in options:
            if entry is None:
                continue
            option = str(entry).strip()
            if not option:
                raise ValueError("blank option is not allowed in schema field")
            if option in seen:
                raise ValueError(f"duplicate option '{option}' in schema field")
            seen.add(option)
            normalized.append(option)

        if not normalized:
            raise ValueError("schema field must define at least one option")

        description = data.get("description")
        return cls(field_type=field_type, options=tuple(normalized), description=str(description) if description is not None else None)


@dataclasses.dataclass(frozen=True)
class ExtensionSpec:
    id: str
    spec_version: str
    display_name: str | None
    description: str | None
    enabled: bool
    trigger: dict[str, Sequence[str]]
    prompt: str
    schema: dict[str, SchemaField]
    output: dict[str, bool]
    dashboard: dict[str, object]
    source: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, object], source: str | None = None) -> "ExtensionSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("extension spec must be a mapping")

        spec_id = raw.get("id")
        if not spec_id:
            raise ValueError("extension spec missing 'id'")

        spec_version = raw.get("spec_version")
        if not spec_version:
            raise ValueError("extension spec missing 'spec_version'")

        prompt = raw.get("prompt")
        if not prompt:
            raise ValueError("extension spec missing 'prompt'")

        schema_raw = raw.get("schema")
        if not isinstance(schema_raw, Mapping):
            raise ValueError("extension spec missing 'schema'")

        schema: dict[str, SchemaField] = {}
        for field_name, field_value in schema_raw.items():
            normalized_field_name = str(field_name).strip()
            if not normalized_field_name:
                raise ValueError("schema field name must be a non-empty string")
            if normalized_field_name in schema:
                raise ValueError(f"duplicate schema field name '{normalized_field_name}'")
            schema[normalized_field_name] = SchemaField.from_dict(field_value)  # type: ignore[arg-type]

        enabled_raw = raw.get("enabled")
        enabled = True if enabled_raw is None else bool(enabled_raw)

        return cls(
            id=str(spec_id),
            spec_version=str(spec_version),
            display_name=str(raw["display_name"]) if "display_name" in raw and raw["display_name"] is not None else None,
            description=str(raw["description"]) if "description" in raw and raw["description"] is not None else None,
            enabled=enabled,
            trigger=_normalize_trigger(raw.get("trigger")),
            prompt=str(prompt),
            schema=schema,
            output=_normalize_output(raw.get("output")),
            dashboard=_ensure_mapping(raw.get("dashboard")),
            source=source,
        )


def load_extension_spec_file(path: Path) -> ExtensionSpec:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("extension spec file must contain a mapping")
    return ExtensionSpec.from_dict(raw, source=str(path))


def load_extension_specs(paths: Sequence[Path | str]) -> list[ExtensionSpec]:
    specs: list[ExtensionSpec] = []
    seen: set[str] = set()
    for candidate in paths:
        path = Path(candidate)
        spec = load_extension_spec_file(path)
        if spec.id in seen:
            raise ValueError(f"duplicate extension spec id '{spec.id}'")
        seen.add(spec.id)
        specs.append(spec)
    return specs


def spec_hash(spec: ExtensionSpec) -> str:
    payload: dict[str, object] = {
        "id": spec.id,
        "spec_version": spec.spec_version,
        "prompt": spec.prompt,
        "enabled": spec.enabled,
        "display_name": spec.display_name,
        "description": spec.description,
        "schema": {
            name: {
                "type": field.field_type,
                "options": list(field.options),
                "description": field.description,
            }
            for name, field in sorted(spec.schema.items())
        },
        "trigger": {
            key: list(values)
            for key, values in sorted(spec.trigger.items())
        },
        "output": {key: spec.output[key] for key in sorted(spec.output)},
        "dashboard": {key: spec.dashboard[key] for key in sorted(spec.dashboard)},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def matches_trigger(trigger: Mapping[str, Sequence[str]], labels: Mapping[str, object]) -> bool:
    if not trigger:
        return True

    for key, values in trigger.items():
        label_name, mode = _split_trigger_key(key)
        required = set(values)
        if not required:
            continue

        label_values = _label_values(labels.get(label_name))
        if not label_values:
            return False

        if mode == "any":
            if not required.intersection(label_values):
                return False
        else:
            if not required.issubset(label_values):
                return False

    return True


def match_extension_specs(extension_specs: Sequence[ExtensionSpec], labels: Mapping[str, object]) -> list[ExtensionSpec]:
    matches: list[ExtensionSpec] = []
    for spec in extension_specs:
        if not spec.enabled:
            continue
        if matches_trigger(spec.trigger, labels):
            matches.append(spec)
    return matches


def _normalize_trigger(raw: object | None) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("trigger must be a mapping")

    normalized: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        trigger_key = str(key).strip()
        label_name, _mode = _split_trigger_key(trigger_key)
        if label_name not in _SUPPORTED_TRIGGER_LABELS:
            raise ValueError(f"unsupported trigger key '{trigger_key}'")
        values = _sequence_to_tuple(value)
        if values:
            normalized[trigger_key] = values
    return normalized


def _sequence_to_tuple(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        filtered = [str(item) for item in value if item is not None]
        return tuple(filtered)
    return (str(value),)


def _label_values(value: object | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple)):
        return {str(item) for item in value if item is not None}
    return {str(value)}


def _split_trigger_key(key: str) -> tuple[str, str]:
    for suffix, mode in _TRIGGER_SUFFIXES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], mode
    raise ValueError(f"unsupported trigger key '{key}'")


def _normalize_output(raw: object | None) -> dict[str, bool]:
    if raw is None:
        return dict(DEFAULT_OUTPUT)
    if not isinstance(raw, Mapping):
        raise ValueError("output must be a mapping")

    normalized = dict(DEFAULT_OUTPUT)
    for field in DEFAULT_OUTPUT:
        if field in raw:
            normalized[field] = bool(raw[field])
    return normalized


def _ensure_mapping(raw: object | None) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("expected a mapping")
    return dict(raw)


__all__ = [
    "ExtensionSpec",
    "SchemaField",
    "load_extension_spec_file",
    "load_extension_specs",
    "spec_hash",
    "matches_trigger",
    "match_extension_specs",
]
