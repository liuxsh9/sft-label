from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from sft_label.config import PipelineConfig
from sft_label.pipeline import label_one, sanitize_conversations_for_content_filter


CALL1_OK = {
    "intent": "build",
    "language": ["java"],
    "domain": [],
    "task": ["feature-implementation"],
    "difficulty": "intermediate",
    "confidence": {
        "intent": 0.9,
        "language": 0.95,
        "domain": 0.8,
        "task": 0.88,
        "difficulty": 0.82,
    },
    "unmapped": [],
}

CALL2_OK = {
    "concept": ["data-structures"],
    "agentic": [],
    "constraint": ["thread-safe"],
    "context": "snippet",
    "confidence": {
        "concept": 0.9,
        "agentic": 0.8,
        "constraint": 0.86,
        "context": 0.92,
    },
    "unmapped": [],
}

CONTENT_FILTER_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "error": "HTTP 400 (content filtered): blocked",
    "error_response": '{"error":{"code":"content_filter"}}',
    "content_filtered": True,
    "non_retryable": True,
}


def _sample():
    return {
        "id": "content-filter-sample",
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Please reason step by step and implement a thread-safe LRU cache. "
                    "Here is a piece of erroneous code as a reference to increase misdirection."
                ),
            },
            {"from": "gpt", "value": "I can help with that."},
        ],
    }


@pytest.mark.asyncio
async def test_label_one_retries_content_filter_with_sanitized_prompt():
    calls = []

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        payload = json.dumps(messages, ensure_ascii=False).lower()
        calls.append(payload)
        if len(calls) == 1:
            assert "reason step by step" in payload
            return None, "", dict(CONTENT_FILTER_USAGE)
        if len(calls) == 2:
            assert "reason step by step" not in payload
            assert "erroneous code" not in payload
            return CALL1_OK, json.dumps(CALL1_OK), {"prompt_tokens": 10, "completion_tokens": 5}
        return CALL2_OK, json.dumps(CALL2_OK), {"prompt_tokens": 8, "completion_tokens": 4}

    config = PipelineConfig(sample_max_retries=1, max_retries=1)

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        idx, labels, monitor = await label_one(
            http_client=None,
            sample=_sample(),
            model="mock-model",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=False,
            config=config,
        )

    assert idx == 0
    assert labels is not None
    assert labels["intent"] == "build"
    assert labels["constraint"] == ["thread-safe"]
    assert monitor["status"] == "success"
    assert monitor["content_filter_retry"] == "call1"
    assert monitor["llm_calls"] == 3


@pytest.mark.asyncio
async def test_label_one_does_not_retry_non_content_filter_failure():
    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        return None, "", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": "HTTP 401: unauthorized",
            "error_response": "unauthorized",
            "non_retryable": True,
        }

    config = PipelineConfig(sample_max_retries=1, max_retries=1)

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call) as mocked:
        idx, labels, monitor = await label_one(
            http_client=None,
            sample=_sample(),
            model="mock-model",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=False,
            config=config,
        )

    assert idx == 0
    assert labels is None
    assert monitor["status"] == "call1_failed"
    assert "content_filter_retry" not in monitor
    assert mocked.await_count == 1


def test_sanitize_conversations_for_content_filter_softens_trigger_phrases():
    sanitized, changed = sanitize_conversations_for_content_filter(_sample()["conversations"])

    assert changed is True
    user_text = sanitized[0]["value"].lower()
    assert "reason step by step" not in user_text
    assert "erroneous code" not in user_text
    assert "misdirection" not in user_text
    assert "briefly explain your approach" in user_text


@pytest.mark.asyncio
async def test_label_one_compact_prompt_stays_under_utf8_byte_limit():
    sample = {
        "id": "compact-cjk",
        "conversations": [
            {"from": "human", "value": "请修复超时问题并解释原因🙂" * 900},
            {"from": "gpt", "value": "我会先运行测试再修复问题。" * 900},
        ],
    }
    payload_sizes = []

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        payload_sizes.append(len(json.dumps(messages, ensure_ascii=False).encode("utf-8")))
        if len(payload_sizes) == 1:
            return CALL1_OK, json.dumps(CALL1_OK), {"prompt_tokens": 10, "completion_tokens": 5}
        return CALL2_OK, json.dumps(CALL2_OK), {"prompt_tokens": 8, "completion_tokens": 4}

    config = PipelineConfig(sample_max_retries=1, max_retries=1, prompt_mode="compact")

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        idx, labels, monitor = await label_one(
            http_client=None,
            sample=sample,
            model="mock-model",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=False,
            config=config,
        )

    assert idx == 0
    assert labels is not None
    assert monitor["status"] == "success"
    assert payload_sizes
    assert max(payload_sizes) <= config.compact_labeling_request_bytes


@pytest.mark.asyncio
async def test_label_one_propagates_sanitized_call1_context_into_call2():
    calls = []

    async def fake_async_llm_call(http_client, messages, model, **kwargs):
        payload = json.dumps(messages, ensure_ascii=False).lower()
        calls.append(payload)
        if len(calls) == 1:
            assert "reason step by step" in payload
            return None, "", dict(CONTENT_FILTER_USAGE)
        if len(calls) == 2:
            assert "reason step by step" not in payload
            return CALL1_OK, json.dumps(CALL1_OK), {"prompt_tokens": 10, "completion_tokens": 5}
        assert "reason step by step" not in payload
        return CALL2_OK, json.dumps(CALL2_OK), {"prompt_tokens": 8, "completion_tokens": 4}

    config = PipelineConfig(sample_max_retries=1, max_retries=1, prompt_mode="compact")

    with patch("sft_label.pipeline.async_llm_call", side_effect=fake_async_llm_call):
        idx, labels, monitor = await label_one(
            http_client=None,
            sample=_sample(),
            model="mock-model",
            sample_idx=0,
            total=1,
            sem=asyncio.Semaphore(1),
            enable_arbitration=False,
            config=config,
        )

    assert idx == 0
    assert labels is not None
    assert monitor["status"] == "success"
    assert len(calls) == 3
