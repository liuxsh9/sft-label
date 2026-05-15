from __future__ import annotations

import json

import pytest

from sft_label.label_extensions import build_extension_messages, run_label_extensions
from sft_label.label_extensions_schema import ExtensionSpec


def _build_spec(raw: dict, source: str) -> ExtensionSpec:
    return ExtensionSpec.from_dict(raw, source=source)


def _ui_spec() -> ExtensionSpec:
    return _build_spec(
        {
            "id": "ui_fine_labels",
            "spec_version": "v1",
            "display_name": "UI Fine Labels",
            "prompt": "Label UI samples.",
            "trigger": {
                "domain_any_of": ["web-frontend"],
                "task_any_of": ["feature-implementation"],
            },
            "schema": {
                "component_type": {"type": "multi_enum", "options": ["card", "table", "form"]},
                "interaction_pattern": {"type": "multi_enum", "options": ["crud", "dashboard"]},
                "visual_complexity": {"type": "enum", "options": ["low", "medium", "high"]},
            },
        },
        "ui.yaml",
    )


def _mobile_spec() -> ExtensionSpec:
    return _build_spec(
        {
            "id": "mobile_fine_labels",
            "spec_version": "v1",
            "display_name": "Mobile Fine Labels",
            "prompt": "Label mobile UI samples.",
            "trigger": {
                "domain_any_of": ["mobile-frontend"],
                "difficulty_any_of": ["intermediate"],
            },
            "schema": {
                "platform": {"type": "multi_enum", "options": ["ios", "android"]},
                "screen_type": {"type": "enum", "options": ["detail", "list"]},
            },
        },
        "mobile.yaml",
    )


def _noisy_mobile_spec() -> ExtensionSpec:
    return _build_spec(
        {
            "id": "mobile_noisy_labels",
            "spec_version": "v1",
            "display_name": "Mobile Fine Labels (No Confidence/Unmapped)",
            "prompt": "Label mobile UI samples.",
            "trigger": {
                "domain_any_of": ["mobile-frontend"],
                "difficulty_any_of": ["intermediate"],
            },
            "schema": {
                "platform": {"type": "multi_enum", "options": ["ios", "android"]},
                "screen_type": {"type": "enum", "options": ["detail", "list"]},
            },
            "output": {
                "include_confidence": False,
                "allow_unmapped": False,
            },
        },
        "mobile-noisy.yaml",
    )


def _core_labels(domains: list[str] | None = None) -> dict[str, object]:
    return {
        "intent": "build",
        "language": ["typescript"],
        "domain": domains or ["web-frontend", "mobile-frontend"],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "context": "snippet",
    }


@pytest.mark.asyncio
async def test_build_extension_messages_include_context_and_schema() -> None:
    spec = _ui_spec()
    messages = build_extension_messages(
        conversation_json=json.dumps([{"from": "human", "value": "Build a dashboard"}], ensure_ascii=False),
        preprocessed_signals="languages: [typescript]",
        core_labels=_core_labels(),
        spec=spec,
    )

    assert len(messages) == 2
    assert "Label UI samples." in messages[0]["content"]
    assert "<conversation>" in messages[1]["content"]
    assert '"from": "human"' not in messages[1]["content"]
    assert "[USER REQUEST]" in messages[1]["content"]
    assert "<core_labels>" in messages[1]["content"]
    assert "component_type" in messages[1]["content"]
    assert "languages: [typescript]" in messages[1]["content"]


@pytest.mark.asyncio
async def test_label_extensions_multi_extension_execution() -> None:
    specs = [_ui_spec(), _mobile_spec()]

    async def _fake_llm(_messages, spec: ExtensionSpec) -> dict:
        if spec.id == "ui_fine_labels":
            return {
                "component_type": ["table", "form"],
                "interaction_pattern": ["dashboard"],
                "visual_complexity": "high",
                "confidence": {"component_type": 0.82, "interaction_pattern": 0.78, "visual_complexity": 0.7},
            }
        return {
            "platform": ["ios"],
            "screen_type": "detail",
            "confidence": {"platform": 0.75, "screen_type": 0.73},
        }

    extensions = await run_label_extensions(
        conversation_json="[]",
        preprocessed_signals="languages: [typescript]",
        core_labels=_core_labels(),
        extension_specs=specs,
        llm_caller=_fake_llm,
    )

    assert set(extensions) == {"ui_fine_labels", "mobile_fine_labels"}
    assert extensions["ui_fine_labels"]["status"] == "success"
    assert extensions["mobile_fine_labels"]["status"] == "success"
    assert extensions["ui_fine_labels"]["labels"]["component_type"] == ["table", "form"]
    assert extensions["mobile_fine_labels"]["labels"]["platform"] == ["ios"]


@pytest.mark.asyncio
async def test_label_extensions_failure_isolation() -> None:
    core = _core_labels()
    before = json.dumps(core, sort_keys=True)

    async def _fake_llm(_messages, spec: ExtensionSpec) -> dict:
        if spec.id == "ui_fine_labels":
            raise RuntimeError("provider unavailable")
        return {"platform": ["android"], "screen_type": "list"}

    extensions = await run_label_extensions(
        conversation_json="[]",
        preprocessed_signals="signals",
        core_labels=core,
        extension_specs=[_ui_spec(), _mobile_spec()],
        llm_caller=_fake_llm,
    )

    assert json.dumps(core, sort_keys=True) == before
    assert extensions["ui_fine_labels"]["status"] == "failed"
    assert extensions["mobile_fine_labels"]["status"] == "success"


@pytest.mark.asyncio
async def test_label_extensions_invalid_enum_output() -> None:
    async def _fake_llm(_messages, _spec: ExtensionSpec) -> dict:
        return {
            "platform": ["android"],
            "screen_type": "sheet",
        }

    extensions = await run_label_extensions(
        conversation_json="[]",
        preprocessed_signals="signals",
        core_labels=_core_labels(),
        extension_specs=[_mobile_spec()],
        llm_caller=_fake_llm,
    )

    payload = extensions["mobile_fine_labels"]
    assert payload["status"] == "invalid"
    assert payload["labels"]["screen_type"] == ""
    assert payload["unmapped"] == [{"dimension": "screen_type", "value": "sheet"}]


@pytest.mark.asyncio
async def test_label_extensions_multi_enum_scalar_handling() -> None:
    async def _fake_llm(_messages, _spec: ExtensionSpec) -> dict:
        return {
            "component_type": "card",
            "interaction_pattern": "crud",
            "visual_complexity": "medium",
        }

    extensions = await run_label_extensions(
        conversation_json="[]",
        preprocessed_signals="signals",
        core_labels=_core_labels(domains=["web-frontend"]),
        extension_specs=[_ui_spec()],
        llm_caller=_fake_llm,
    )

    payload = extensions["ui_fine_labels"]
    assert payload["status"] == "success"
    assert payload["labels"]["component_type"] == ["card"]
    assert payload["labels"]["interaction_pattern"] == ["crud"]


@pytest.mark.asyncio
async def test_label_extensions_confidence_retention() -> None:
    async def _fake_llm(_messages, _spec: ExtensionSpec) -> dict:
        return {
            "component_type": ["form"],
            "interaction_pattern": ["crud"],
            "visual_complexity": "medium",
            "confidence": {"component_type": 0.82, "interaction_pattern": 0.78, "visual_complexity": 0.7},
        }

    extensions = await run_label_extensions(
        conversation_json="[]",
        preprocessed_signals="signals",
        core_labels=_core_labels(domains=["web-frontend"]),
        extension_specs=[_ui_spec()],
        llm_caller=_fake_llm,
    )

    payload = extensions["ui_fine_labels"]
    assert payload["confidence"] == {
        "component_type": pytest.approx(0.82),
        "interaction_pattern": pytest.approx(0.78),
        "visual_complexity": pytest.approx(0.7),
    }


@pytest.mark.asyncio
async def test_label_extensions_honor_output_controls() -> None:
    async def _fake_llm(_messages, _spec: ExtensionSpec) -> dict:
        return {
            "platform": ["android"],
            "screen_type": "sheet",
            "confidence": {"platform": 0.91, "screen_type": 0.33},
        }

    extensions = await run_label_extensions(
        conversation_json="[]",
        preprocessed_signals="signals",
        core_labels=_core_labels(),
        extension_specs=[_noisy_mobile_spec()],
        llm_caller=_fake_llm,
    )

    payload = extensions["mobile_noisy_labels"]
    assert payload["status"] == "invalid"
    assert payload["confidence"] == {}
    assert payload["unmapped"] == []
