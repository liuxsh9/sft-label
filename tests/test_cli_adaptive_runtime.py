from __future__ import annotations

from types import SimpleNamespace

from sft_label.cli import _apply_runtime_overrides, build_parser
from sft_label.config import PipelineConfig


def test_pipeline_config_exposes_adaptive_runtime_defaults():
    config = PipelineConfig()
    assert config.enable_adaptive_runtime is True
    assert config.enable_stage_recovery_sweep is True
    assert config.request_quick_retries >= 0
    assert config.adaptive_min_concurrency >= 1
    assert config.adaptive_open_base_cooldown > 0


def test_apply_runtime_overrides_updates_adaptive_runtime_flags():
    config = PipelineConfig()
    args = SimpleNamespace(
        concurrency=None,
        scoring_concurrency=None,
        rps_limit=None,
        rps_warmup=None,
        request_timeout=None,
        max_retries=None,
        rarity_mode=None,
        adaptive_runtime=False,
        recovery_sweep=False,
    )
    _apply_runtime_overrides(config, args)
    assert config.enable_adaptive_runtime is False
    assert config.enable_stage_recovery_sweep is False


def test_parser_accepts_adaptive_runtime_boolean_flags():
    parser = build_parser()

    run_args = parser.parse_args(
        [
            "run",
            "--input",
            "x.json",
            "--no-adaptive-runtime",
            "--no-recovery-sweep",
        ]
    )
    assert run_args.adaptive_runtime is False
    assert run_args.recovery_sweep is False

    score_args = parser.parse_args(
        [
            "score",
            "--input",
            "x.json",
            "--adaptive-runtime",
            "--recovery-sweep",
        ]
    )
    assert score_args.adaptive_runtime is True
    assert score_args.recovery_sweep is True
