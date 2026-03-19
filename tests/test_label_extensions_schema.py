from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from sft_label import label_extensions_guidance as guidance
from sft_label import label_extensions_schema as schema

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "extensions"
UI_SPEC_PATH = FIXTURE_DIR / "ui_fine_labels.yaml"
MOBILE_SPEC_PATH = FIXTURE_DIR / "mobile_fine_labels.yaml"


def _write_temp_spec(tmp_path: Path, spec: dict, name: str = "temp-spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _load_raw_spec(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.fixture(name="extension_specs")
def fixture_extension_specs() -> list[schema.ExtensionSpec]:
    return schema.load_extension_specs([UI_SPEC_PATH, MOBILE_SPEC_PATH])


def test_load_fixture_spec_normalizes_schema_fields() -> None:
    spec = schema.load_extension_spec_file(UI_SPEC_PATH)

    assert spec.id == "ui_fine_labels"
    assert spec.spec_version == "v1"
    assert spec.dashboard["group"] == "ui"
    assert spec.trigger["difficulty_any_of"] == ("beginner", "intermediate", "advanced")

    component_field = spec.schema["component_type"]
    assert component_field.field_type == "multi_enum"
    assert component_field.options[0] == "form"

    visual_field = spec.schema["visual_complexity"]
    assert visual_field.field_type == "enum"
    assert visual_field.options == ("low", "medium", "high")

    assert spec.output["include_confidence"] is True
    assert spec.output["allow_unmapped"] is True


def test_required_fields_are_enforced(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw.pop("schema")
    bad_path = _write_temp_spec(tmp_path, raw, "missing-schema.yaml")

    with pytest.raises(ValueError):
        schema.load_extension_spec_file(bad_path)


def test_duplicate_options_are_rejected(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw["schema"]["component_type"]["options"].append("form")
    bad_path = _write_temp_spec(tmp_path, raw, "duplicate-option.yaml")

    with pytest.raises(ValueError):
        schema.load_extension_spec_file(bad_path)


def test_unsupported_schema_field_type_is_rejected(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw["schema"]["component_type"]["type"] = "free_text"
    bad_path = _write_temp_spec(tmp_path, raw, "unsupported-type.yaml")

    with pytest.raises(ValueError, match="unsupported schema field type"):
        schema.load_extension_spec_file(bad_path)


def test_duplicate_extension_ids_are_rejected(tmp_path: Path) -> None:
    raw_ui = _load_raw_spec(UI_SPEC_PATH)
    copy_path = _write_temp_spec(tmp_path, raw_ui, "ui-copy.yaml")

    with pytest.raises(ValueError):
        schema.load_extension_specs([UI_SPEC_PATH, copy_path])


def test_defaults_apply_when_optional_sections_missing(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw.pop("trigger", None)
    raw.pop("output", None)
    raw.pop("dashboard", None)
    raw.pop("enabled", None)
    path = _write_temp_spec(tmp_path, raw, "defaults.yaml")

    spec = schema.load_extension_spec_file(path)

    assert spec.enabled is True
    assert spec.output == {"include_confidence": True, "allow_unmapped": True}
    assert spec.dashboard == {}
    assert spec.trigger == {}


def test_missing_trigger_matches_every_sample(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw.pop("trigger", None)
    path = _write_temp_spec(tmp_path, raw, "no-trigger.yaml")
    spec = schema.load_extension_spec_file(path)

    assert schema.matches_trigger(spec.trigger, {"language": "typescript", "domain": ["web-frontend"]})


def test_spec_hash_is_stable_for_same_content() -> None:
    spec = schema.load_extension_spec_file(UI_SPEC_PATH)
    first_hash = schema.spec_hash(spec)
    second_hash = schema.spec_hash(spec)

    assert first_hash == second_hash


def test_spec_hash_changes_when_prompt_changes() -> None:
    spec = schema.load_extension_spec_file(UI_SPEC_PATH)
    mutated = dataclasses.replace(spec, prompt=spec.prompt + "\nextra note")

    assert schema.spec_hash(mutated) != schema.spec_hash(spec)


def test_spec_hash_ignores_schema_ordering() -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    schema_entries = list(raw["schema"].items())
    reordered = dict(raw)
    reordered["schema"] = {k: v for k, v in reversed(schema_entries)}

    reordered_spec = schema.ExtensionSpec.from_dict(reordered, source="reordered.yaml")
    normal_spec = schema.load_extension_spec_file(UI_SPEC_PATH)

    assert schema.spec_hash(reordered_spec) == schema.spec_hash(normal_spec)


@pytest.mark.parametrize(
    "trigger_key, trigger_values, labels",
    [
        ("language_any_of", ["typescript"], {"language": ["typescript"]}),
        ("intent_any_of", ["build"], {"intent": "build"}),
        ("task_any_of", ["feature-implementation"], {"task": ["feature-implementation"]}),
        ("context_any_of", ["snippet"], {"context": "snippet"}),
        ("difficulty_any_of", ["intermediate"], {"difficulty": "intermediate"}),
    ],
)
def test_any_of_triggers_match(trigger_key: str, trigger_values: list[str], labels: dict[str, object]) -> None:
    trigger = {trigger_key: trigger_values}
    assert schema.matches_trigger(trigger, labels)


def test_domain_all_of_trigger_matches() -> None:
    trigger = {"domain_all_of": ["mobile", "frontend"]}
    matching_labels = {"domain": ["mobile", "frontend"]}
    missing_labels = {"domain": ["mobile"]}

    assert schema.matches_trigger(trigger, matching_labels)
    assert not schema.matches_trigger(trigger, missing_labels)


def test_domain_any_of_extension_matching(extension_specs: list[schema.ExtensionSpec]) -> None:
    labels = {
        "domain": ["web-frontend"],
        "language": ["typescript"],
        "intent": "build",
        "task": ["feature-implementation"],
        "context": "snippet",
        "difficulty": "intermediate",
    }

    matched_ids = [spec.id for spec in schema.match_extension_specs(extension_specs, labels)]

    assert matched_ids == ["ui_fine_labels"]


def test_domain_all_of_extension_matching(extension_specs: list[schema.ExtensionSpec]) -> None:
    labels = {
        "domain": ["mobile", "frontend"],
        "language": ["kotlin"],
        "intent": "build",
        "task": ["mobile-ui"],
        "context": "mobile-app",
        "difficulty": "intermediate",
    }

    matched_ids = [spec.id for spec in schema.match_extension_specs(extension_specs, labels)]

    assert matched_ids == ["mobile_fine_labels"]


def test_unmatched_sample_is_skipped(extension_specs: list[schema.ExtensionSpec]) -> None:
    labels = {
        "domain": ["data-science"],
        "language": ["python"],
        "intent": "learn",
        "task": ["analysis"],
        "context": "snippet",
        "difficulty": "beginner",
    }

    assert schema.match_extension_specs(extension_specs, labels) == []


def test_multiple_extensions_can_match(extension_specs: list[schema.ExtensionSpec]) -> None:
    labels = {
        "domain": ["web-frontend", "frontend", "mobile"],
        "language": ["typescript", "kotlin"],
        "intent": "build",
        "task": ["feature-implementation", "mobile-ui"],
        "context": "mobile-app",
        "difficulty": "intermediate",
    }

    matched_ids = [spec.id for spec in schema.match_extension_specs(extension_specs, labels)]

    assert matched_ids == ["ui_fine_labels", "mobile_fine_labels"]


def test_invalid_trigger_key_is_rejected_at_load_time(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw["trigger"] = {"domain_only": ["web-frontend"]}
    bad_path = _write_temp_spec(tmp_path, raw, "invalid-trigger.yaml")

    with pytest.raises(ValueError, match="unsupported trigger key"):
        schema.load_extension_spec_file(bad_path)


def test_blank_schema_field_name_is_rejected(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    field = raw["schema"].pop("component_type")
    raw["schema"]["   "] = field
    bad_path = _write_temp_spec(tmp_path, raw, "blank-field-name.yaml")

    with pytest.raises(ValueError, match="schema field name"):
        schema.load_extension_spec_file(bad_path)


def test_blank_option_value_is_rejected(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw["schema"]["component_type"]["options"].append("   ")
    bad_path = _write_temp_spec(tmp_path, raw, "blank-option.yaml")

    with pytest.raises(ValueError, match="blank option"):
        schema.load_extension_spec_file(bad_path)


def test_guidance_summary_emits_size_warnings(tmp_path: Path) -> None:
    raw = _load_raw_spec(UI_SPEC_PATH)
    raw.pop("trigger", None)
    raw["prompt"] = ("Label UI data with detailed heuristics. " * 120).strip()
    for idx in range(6):
        raw["schema"][f"extra_field_{idx}"] = {
            "type": "multi_enum",
            "options": [f"option_{n}" for n in range(21)],
        }
    path = _write_temp_spec(tmp_path, raw, "warning-heavy.yaml")

    spec = schema.load_extension_spec_file(path)
    summary = guidance.summarize_extension_spec(spec)

    assert "triggerless" in summary["warnings"]
    assert "compact_over_budget" in summary["warnings"]
    assert "too_many_fields" in summary["warnings"]
    assert "too_many_options" in summary["warnings"]
