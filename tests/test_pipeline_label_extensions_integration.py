from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig
from sft_label.label_extensions_stats import aggregate_extension_stats, merge_extension_stats
from sft_label.label_extensions_schema import load_extension_spec_file
from sft_label.pipeline import compute_stats, label_one, merge_stats


def _write_extension_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "ui_extension.yaml"
    spec_path.write_text(
        """
id: ui_fine_labels
spec_version: v1
trigger:
  domain_any_of: [web-frontend]
prompt: |
  Label UI data.
schema:
  component_type:
    type: multi_enum
    options: [form, modal, table]
dashboard:
  group: ui
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return spec_path


@pytest.mark.asyncio
async def test_label_one_runs_extension_labeling_after_core_pass1(tmp_path):
    spec = load_extension_spec_file(_write_extension_spec(tmp_path))
    config = PipelineConfig(sample_max_retries=0, max_retries=0)
    setattr(config, "label_extension_specs", [spec])

    call1_ok = {
        "intent": "build",
        "language": ["typescript"],
        "domain": ["web-frontend"],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "confidence": {"intent": 0.9},
        "unmapped": [],
    }
    call2_ok = {
        "concept": ["ui-design"],
        "agentic": [],
        "constraint": [],
        "context": "snippet",
        "confidence": {"concept": 0.8},
        "unmapped": [],
    }
    extension_ok = {
        "component_type": ["form", "modal"],
        "confidence": {"component_type": 0.76},
    }

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        idx = fake_async_llm_call.calls
        fake_async_llm_call.calls += 1
        payload = [call1_ok, call2_ok, extension_ok][idx]
        return payload, json.dumps(payload), {"prompt_tokens": 1, "completion_tokens": 1, "status_code": 200}

    fake_async_llm_call.calls = 0
    sample = {
        "id": "s-ext-1",
        "conversations": [
            {"from": "human", "value": "Build a modal form UI in TypeScript."},
            {"from": "gpt", "value": "Here is the component."},
        ],
    }

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        _, labels, monitor = await label_one(
            http_client=None,
            sample=sample,
            model="mock",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=False,
            config=config,
            rate_limiter=None,
        )

    assert labels is not None
    assert labels["intent"] == "build"
    assert labels["label_extensions"]["ui_fine_labels"]["status"] == "success"
    assert labels["label_extensions"]["ui_fine_labels"]["labels"]["component_type"] == ["form", "modal"]
    assert fake_async_llm_call.calls == 3
    assert monitor["llm_calls"] == 3


@pytest.mark.asyncio
async def test_label_one_loads_extension_specs_from_config_paths(tmp_path):
    spec_path = _write_extension_spec(tmp_path)
    config = PipelineConfig(sample_max_retries=0, max_retries=0)
    config.extension_spec_paths = [str(spec_path)]

    call1_ok = {
        "intent": "build",
        "language": ["typescript"],
        "domain": ["web-frontend"],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "confidence": {"intent": 0.9},
        "unmapped": [],
    }
    call2_ok = {
        "concept": ["ui-design"],
        "agentic": [],
        "constraint": [],
        "context": "snippet",
        "confidence": {"concept": 0.8},
        "unmapped": [],
    }
    extension_ok = {
        "component_type": ["modal"],
        "confidence": {"component_type": 0.76},
    }

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        idx = fake_async_llm_call.calls
        fake_async_llm_call.calls += 1
        payload = [call1_ok, call2_ok, extension_ok][idx]
        return payload, json.dumps(payload), {"prompt_tokens": 1, "completion_tokens": 1, "status_code": 200}

    fake_async_llm_call.calls = 0
    sample = {
        "id": "s-ext-2",
        "conversations": [
            {"from": "human", "value": "Build a modal form UI in TypeScript."},
            {"from": "gpt", "value": "Here is the component."},
        ],
    }

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        _, labels, monitor = await label_one(
            http_client=None,
            sample=sample,
            model="mock",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=False,
            config=config,
            rate_limiter=None,
        )

    assert labels is not None
    assert labels["label_extensions"]["ui_fine_labels"]["status"] == "success"
    assert labels["label_extensions"]["ui_fine_labels"]["labels"]["component_type"] == ["modal"]
    assert config.extension_specs is not None
    assert fake_async_llm_call.calls == 3
    assert monitor["llm_calls"] == 3


def test_compute_stats_and_merge_stats_include_extension_stats():
    extension_payload = {
        "ui_fine_labels": {
            "status": "success",
            "matched": True,
            "spec_version": "v1",
            "spec_hash": "sha256:ui-v1",
            "labels": {"component_type": ["form"], "visual_complexity": "medium"},
            "confidence": {"component_type": 0.8},
            "unmapped": [{"dimension": "component_type", "value": "unknown"}],
        }
    }
    labels = [
        {
            "intent": "build",
            "language": ["typescript"],
            "domain": ["web-frontend"],
            "task": ["feature-implementation"],
            "difficulty": "intermediate",
            "concept": ["ui-design"],
            "agentic": [],
            "constraint": [],
            "context": "snippet",
            "confidence": {"intent": 0.9},
            "label_extensions": extension_payload,
        }
    ]
    monitors = [
        {
            "llm_calls": 3,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "arbitrated": False,
            "validation_issues": [],
            "consistency_warnings": [],
        }
    ]

    per_file = compute_stats(monitors, labels)
    merged = merge_stats([per_file])

    assert per_file["extension_stats"]["specs"]["ui_fine_labels"]["total"] == 1
    assert per_file["extension_stats"]["specs"]["ui_fine_labels"]["matched"] == 1
    assert per_file["extension_stats"]["specs"]["ui_fine_labels"]["field_distributions"]["component_type"]["form"] == 1
    assert merged["extension_stats"]["specs"]["ui_fine_labels"]["status_counts"]["success"] == 1


def test_compute_stats_extension_baselines_are_keyed_by_spec_hash():
    def _payload(*, status, matched, spec_hash, labels=None, confidence=None):
        return {
            "ui_fine_labels": {
                "status": status,
                "matched": matched,
                "spec_version": "v1",
                "spec_hash": spec_hash,
                "labels": labels or {},
                "confidence": confidence or {},
                "unmapped": [],
            }
        }

    labels = [
        {"label_extensions": _payload(
            status="success",
            matched=True,
            spec_hash="sha256:ui-v1",
            labels={"component_type": ["form"], "visual_complexity": "medium"},
            confidence={"component_type": 0.8, "visual_complexity": 0.6},
        )},
        {"label_extensions": _payload(
            status="success",
            matched=True,
            spec_hash="sha256:ui-v1",
            labels={"component_type": ["modal"], "visual_complexity": "medium"},
            confidence={"component_type": 0.9},
        )},
        {"label_extensions": _payload(
            status="invalid",
            matched=True,
            spec_hash="sha256:ui-v1",
            labels={"component_type": ["table"]},
        )},
        {"label_extensions": _payload(
            status="failed",
            matched=True,
            spec_hash="sha256:ui-v1",
        )},
        {"label_extensions": _payload(
            status="skipped",
            matched=False,
            spec_hash="sha256:ui-v1",
        )},
        {"label_extensions": _payload(
            status="success",
            matched=True,
            spec_hash="sha256:ui-v2",
            labels={"component_type": ["table"]},
            confidence={"component_type": 0.7},
        )},
    ]
    monitors = [
        {
            "llm_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "arbitrated": False,
            "validation_issues": [],
            "consistency_warnings": [],
        }
        for _ in labels
    ]

    per_file = compute_stats(monitors, labels)
    merged = merge_stats([per_file])

    spec_stats = per_file["extension_stats"]["specs"]["ui_fine_labels"]
    assert spec_stats["status_counts"]["success"] == 3
    assert spec_stats["field_distributions"]["component_type"]["table"] == 2

    baseline_v1 = spec_stats["baselines"]["sha256:ui-v1"]
    assert baseline_v1["spec_version"] == "v1"
    assert baseline_v1["spec_hash"] == "sha256:ui-v1"
    assert baseline_v1["success"] == 2
    assert baseline_v1["invalid"] == 1
    assert baseline_v1["failed"] == 1
    assert baseline_v1["skipped"] == 1
    assert baseline_v1["baseline_total"] == 2
    assert baseline_v1["field_value_distributions"]["component_type"] == {"form": 1, "modal": 1}
    assert baseline_v1["field_presence_counts"] == {"component_type": 2, "visual_complexity": 2}

    baseline_v2 = merged["extension_stats"]["specs"]["ui_fine_labels"]["baselines"]["sha256:ui-v2"]
    assert baseline_v2["success"] == 1
    assert baseline_v2["baseline_total"] == 1
    assert baseline_v2["field_value_distributions"]["component_type"] == {"table": 1}


def test_merge_extension_stats_clears_top_level_singletons_for_mixed_hashes():
    def _sample(*, spec_hash, spec_version, component_type):
        return {
            "label_extensions": {
                "ui_fine_labels": {
                    "status": "success",
                    "matched": True,
                    "spec_version": spec_version,
                    "spec_hash": spec_hash,
                    "labels": {"component_type": [component_type]},
                    "confidence": {"component_type": 0.9},
                    "unmapped": [],
                }
            }
        }

    merged = merge_extension_stats([
        {"extension_stats": aggregate_extension_stats([_sample(
            spec_hash="sha256:ui-v1",
            spec_version="v1",
            component_type="form",
        )])},
        {"extension_stats": aggregate_extension_stats([_sample(
            spec_hash="sha256:ui-v2",
            spec_version="v2",
            component_type="table",
        )])},
    ])

    spec_stats = merged["specs"]["ui_fine_labels"]
    assert spec_stats["spec_hash"] is None
    assert spec_stats["spec_version"] is None
    assert spec_stats["config"] is None
    assert set(spec_stats["baselines"]) == {"sha256:ui-v1", "sha256:ui-v2"}
