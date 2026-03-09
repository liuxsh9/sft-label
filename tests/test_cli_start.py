"""Tests for interactive start command output helpers."""

from __future__ import annotations

from sft_label.cli import _format_start_env_lines, _is_sensitive_env_key, _mask_secret


def test_sensitive_env_key_detection():
    assert _is_sensitive_env_key("LITELLM_KEY") is True
    assert _is_sensitive_env_key("api_token") is True
    assert _is_sensitive_env_key("db_password") is True
    assert _is_sensitive_env_key("LITELLM_BASE") is False


def test_mask_secret_behavior():
    assert _mask_secret("") == "(empty)"
    assert _mask_secret("123456") == "******"
    assert _mask_secret("abcdefgh") == "ab***gh"


def test_format_start_env_lines_masks_sensitive_values():
    lines = _format_start_env_lines(
        {
            "LITELLM_KEY": "sk-1234567890",
            "LITELLM_BASE": "http://localhost:4000/v1",
        }
    )
    assert lines == [
        "  LITELLM_BASE=http://localhost:4000/v1",
        "  LITELLM_KEY=sk***90",
    ]
