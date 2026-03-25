"""Tests for interactive start command output helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from sft_label.cli import (
    _auto_publish_pass2_guard,
    _apply_runtime_overrides,
    _suggest_dashboard_share_base_url,
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


def test_suggest_dashboard_share_base_url_uses_detected_lan_host(monkeypatch):
    monkeypatch.setattr(
        "sft_label.cli._guess_local_network_host",
        lambda: "192.168.1.25",
    )

    assert _suggest_dashboard_share_base_url("local", 8765) is None
    assert _suggest_dashboard_share_base_url("lan", 8765) == "http://192.168.1.25:8765"
    assert _suggest_dashboard_share_base_url("public", 9000) == "http://192.168.1.25:9000"


def test_suggest_dashboard_share_base_url_returns_none_without_guess(monkeypatch):
    monkeypatch.setattr("sft_label.cli._guess_local_network_host", lambda: None)

    assert _suggest_dashboard_share_base_url("lan", 8765) is None
    assert _suggest_dashboard_share_base_url("public", 8765) is None


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
        rarity_mode="percentile",
        extension_rarity_mode=None,
        adaptive_runtime=None,
        recovery_sweep=None,
        rollout_preset="compact_safe",
    )
    _apply_runtime_overrides(config, args)
    assert config.concurrency == 123
    assert config.scoring_concurrency == 123
    assert config.rps_limit == 9.5
    assert config.rps_warmup == 12.0
    assert config.request_timeout == 77
    assert config.max_retries == 4
    assert config.rarity_score_mode == "percentile"


def test_apply_runtime_overrides_applies_rollout_preset():
    config = PipelineConfig(
        enable_selective_scoring=True,
        planner_enabled=False,
        planner_metadata_only=True,
        planner_pass2_enabled=False,
    )
    args = SimpleNamespace(
        concurrency=None,
        scoring_concurrency=None,
        rps_limit=None,
        rps_warmup=None,
        request_timeout=None,
        max_retries=None,
        rarity_mode=None,
        extension_rarity_mode=None,
        adaptive_runtime=None,
        recovery_sweep=None,
        rollout_preset="baseline_control",
    )

    _apply_runtime_overrides(config, args)

    assert config.enable_selective_scoring is False
    assert config.planner_enabled is False
    assert config.planner_metadata_only is True
    assert config.planner_pass2_enabled is False


def test_apply_runtime_overrides_uses_legacy_scoring_alias_when_needed():
    config = PipelineConfig()
    args = SimpleNamespace(
        concurrency=None,
        scoring_concurrency=220,
        rps_limit=None,
        rps_warmup=None,
        request_timeout=None,
        max_retries=None,
        rarity_mode=None,
        extension_rarity_mode=None,
        adaptive_runtime=None,
        recovery_sweep=None,
        rollout_preset="compact_safe",
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
            "--no-adaptive-runtime",
            "--no-recovery-sweep",
            "--rarity-mode",
            "percentile",
            "--rollout-preset",
            "planner_hybrid",
        ]
    )
    assert run_args.concurrency == 80
    assert run_args.rps_limit == 25
    assert run_args.rps_warmup == 20
    assert run_args.request_timeout == 120
    assert run_args.max_retries == 5
    assert run_args.adaptive_runtime is False
    assert run_args.recovery_sweep is False
    assert run_args.rarity_mode == "percentile"
    assert run_args.rollout_preset == "planner_hybrid"

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
            "--adaptive-runtime",
            "--recovery-sweep",
            "--rarity-mode",
            "percentile",
            "--rollout-preset",
            "baseline_control",
        ]
    )
    assert score_args.concurrency == 700
    assert score_args.rps_limit == 30
    assert score_args.rps_warmup == 10
    assert score_args.request_timeout == 90
    assert score_args.max_retries == 3
    assert score_args.adaptive_runtime is True
    assert score_args.recovery_sweep is True
    assert score_args.rarity_mode == "percentile"
    assert score_args.rollout_preset == "baseline_control"


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


def test_parser_accepts_maintenance_workers():
    parser = build_parser()

    recompute_args = parser.parse_args(
        ["recompute-stats", "--input", "run_dir", "--workers", "16"]
    )
    assert recompute_args.workers == 16

    refresh_args = parser.parse_args(
        ["refresh-rarity", "--input", "run_dir", "--workers", "32"]
    )
    assert refresh_args.workers == 32

    regen_args = parser.parse_args(
        ["regenerate-dashboard", "--input", "run_dir", "--workers", "4"]
    )
    assert regen_args.workers == 4

    complete_args = parser.parse_args(
        ["complete-postprocess", "--input", "run_dir", "--workers", "6", "--scope", "all"]
    )
    assert complete_args.workers == 6
    assert complete_args.scope == "all"


def test_parser_accepts_analyze_unmapped_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-unmapped",
            "--input",
            "run_dir",
            "--dimension",
            "task",
            "--top",
            "15",
            "--examples",
            "1",
            "--stats-only",
        ]
    )
    assert args.input == "run_dir"
    assert args.dimension == "task"
    assert args.top == 15
    assert args.examples == 1
    assert args.stats_only is True


def test_auto_publish_pass2_guard_uses_inline_specific_guidance(tmp_path):
    run_root = tmp_path / "dataset_labeled_20260325_090311"
    mirrored_file = run_root / "dataset" / "sample.jsonl"
    mirrored_file.parent.mkdir(parents=True, exist_ok=True)
    mirrored_file.write_text(json.dumps({"id": "row-1"}) + "\n", encoding="utf-8")

    dashboards_dir = run_root / "meta_label_data" / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    (dashboards_dir / "dashboard_scoring_dataset.html").write_text("<html></html>", encoding="utf-8")

    launched_args = SimpleNamespace(command="score")
    result = {
        "postprocess": {
            "conversation_scores": {"status": "missing"},
            "dashboard": {"status": "missing"},
        }
    }

    can_publish, message = _auto_publish_pass2_guard(
        launched_args,
        result,
        lang="en",
        run_dir=str(run_root),
    )

    assert can_publish is False
    assert message is not None
    assert "inline mirrored run" in message
    assert "Run `sft-label complete-postprocess --input" in message


def test_auto_publish_pass2_guard_regenerate_dashboard_fail_closed_on_root_meta_conflict(tmp_path):
    run_root = tmp_path / "demo_run"
    (run_root / "meta_label_data").mkdir(parents=True, exist_ok=True)
    (run_root / "summary_stats_scoring.json").write_text(
        json.dumps(
            {
                "postprocess": {
                    "conversation_scores": {"status": "completed"},
                    "dashboard": {"status": "completed"},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_root / "meta_label_data" / "summary_stats_scoring.json").write_text(
        json.dumps(
            {
                "postprocess": {
                    "conversation_scores": {"status": "deferred"},
                    "dashboard": {"status": "completed"},
                }
            }
        ),
        encoding="utf-8",
    )

    launched_args = SimpleNamespace(command="regenerate-dashboard", pass_num="both", input=str(run_root))
    can_publish, message = _auto_publish_pass2_guard(
        launched_args,
        result={"command": "regenerate-dashboard"},
        lang="en",
        run_dir=str(run_root),
    )

    assert can_publish is False
    assert message is not None
    assert "conflict" in message


def test_auto_publish_pass2_guard_regenerate_dashboard_reads_legacy_stats_postprocess(tmp_path):
    run_root = tmp_path / "demo_run"
    meta_dir = run_root / "meta_label_data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "stats_value.json").write_text(
        json.dumps(
            {
                "postprocess": {
                    "conversation_scores": {"status": "deferred"},
                    "dashboard": {"status": "completed"},
                }
            }
        ),
        encoding="utf-8",
    )

    launched_args = SimpleNamespace(command="regenerate-dashboard", pass_num="both", input=str(run_root))
    can_publish, message = _auto_publish_pass2_guard(
        launched_args,
        result={"command": "regenerate-dashboard"},
        lang="en",
        run_dir=str(run_root),
    )

    assert can_publish is False
    assert message is not None
    assert "deferred" in message


def test_auto_publish_pass2_guard_regenerate_dashboard_fail_closed_when_postprocess_block_missing(tmp_path):
    run_root = tmp_path / "demo_run"
    meta_dir = run_root / "meta_label_data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    dashboards_dir = meta_dir / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    (dashboards_dir / "dashboard_scoring_demo.html").write_text("<html></html>", encoding="utf-8")
    (meta_dir / "summary_stats_scoring.json").write_text(
        json.dumps({"total_scored": 1}),
        encoding="utf-8",
    )

    launched_args = SimpleNamespace(command="regenerate-dashboard", pass_num="both", input=str(run_root))
    can_publish, message = _auto_publish_pass2_guard(
        launched_args,
        result={"command": "regenerate-dashboard"},
        lang="en",
        run_dir=str(run_root),
    )

    assert can_publish is False
    assert message is not None
    assert "conversation_scores=missing" in message
    assert "dashboard=missing" in message


def test_auto_publish_pass2_guard_regenerate_dashboard_pass1_still_blocks_stale_scoring_publish(tmp_path):
    run_root = tmp_path / "demo_run"
    meta_dir = run_root / "meta_label_data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    dashboards_dir = meta_dir / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    (dashboards_dir / "dashboard_scoring_demo.html").write_text("<html></html>", encoding="utf-8")
    (meta_dir / "summary_stats_scoring.json").write_text(
        json.dumps({"total_scored": 1}),
        encoding="utf-8",
    )

    launched_args = SimpleNamespace(command="regenerate-dashboard", pass_num="1", input=str(run_root))
    can_publish, message = _auto_publish_pass2_guard(
        launched_args,
        result={"command": "regenerate-dashboard"},
        lang="en",
        run_dir=str(run_root),
    )

    assert can_publish is False
    assert message is not None
    assert "conversation_scores=missing" in message
