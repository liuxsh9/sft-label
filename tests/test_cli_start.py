"""Tests for interactive start command output helpers."""

from __future__ import annotations

from types import SimpleNamespace

from sft_label.cli import (
    _apply_runtime_overrides,
    _format_start_env_lines,
    _is_sensitive_env_key,
    _mask_secret,
    build_parser,
)
from sft_label.config import PipelineConfig


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


def test_apply_runtime_overrides_updates_pipeline_config():
    config = PipelineConfig()
    args = SimpleNamespace(
        concurrency=123,
        scoring_concurrency=456,
        rps_limit=9.5,
        rps_warmup=12.0,
        request_timeout=77,
        max_retries=4,
    )
    _apply_runtime_overrides(config, args)
    assert config.concurrency == 123
    assert config.scoring_concurrency == 123
    assert config.rps_limit == 9.5
    assert config.rps_warmup == 12.0
    assert config.request_timeout == 77
    assert config.max_retries == 4


def test_apply_runtime_overrides_uses_legacy_scoring_alias_when_needed():
    config = PipelineConfig()
    args = SimpleNamespace(
        concurrency=None,
        scoring_concurrency=220,
        rps_limit=None,
        rps_warmup=None,
        request_timeout=None,
        max_retries=None,
    )
    _apply_runtime_overrides(config, args)
    assert config.concurrency == 220
    assert config.scoring_concurrency == 220


def test_parser_accepts_runtime_override_flags():
    parser = build_parser()
    run_args = parser.parse_args(
        [
            "run",
            "--input",
            "x.json",
            "--concurrency",
            "80",
            "--rps-limit",
            "25",
            "--rps-warmup",
            "20",
            "--request-timeout",
            "120",
            "--max-retries",
            "5",
        ]
    )
    assert run_args.concurrency == 80
    assert run_args.rps_limit == 25
    assert run_args.rps_warmup == 20
    assert run_args.request_timeout == 120
    assert run_args.max_retries == 5

    score_args = parser.parse_args(
        [
            "score",
            "--input",
            "x.json",
            "--concurrency",
            "700",
            "--rps-limit",
            "30",
            "--rps-warmup",
            "10",
            "--request-timeout",
            "90",
            "--max-retries",
            "3",
        ]
    )
    assert score_args.concurrency == 700
    assert score_args.rps_limit == 30
    assert score_args.rps_warmup == 10
    assert score_args.request_timeout == 90
    assert score_args.max_retries == 3


def test_parser_keeps_scoring_concurrency_legacy_alias():
    parser = build_parser()
    score_args = parser.parse_args(
        [
            "score",
            "--input",
            "x.json",
            "--scoring-concurrency",
            "620",
        ]
    )
    assert score_args.scoring_concurrency == 620
