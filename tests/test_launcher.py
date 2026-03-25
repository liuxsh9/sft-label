"""Tests for interactive launcher command planning."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from sft_label.launcher import (
    LaunchPlan,
    _decode_utf8_input_byte,
    _ask_extension_spec_paths,
    build_launch_plan,
    format_command,
    format_env_prefix,
    set_language,
    sanitize_prompt_input,
)
from sft_label.cli import build_parser, cmd_dashboard_service, cmd_run, cmd_start
from sft_label.config import DEFAULT_CONCURRENCY, DEFAULT_ROLLOUT_PRESET
from sft_label.dashboard_service import (
    DashboardPortConflictError,
    init_dashboard_service,
    load_dashboard_service_store,
)


class StubIO:
    def __init__(self, answers):
        self._answers = iter(answers)
        self.outputs = []

    def input(self, prompt):
        self.outputs.append(prompt)
        try:
            return next(self._answers)
        except StopIteration as exc:
            raise AssertionError(f"Unexpected prompt: {prompt}") from exc

    def output(self, text):
        self.outputs.append(text)


def test_cmd_dashboard_service_start_prompts_for_new_port_after_conflict(monkeypatch, capsys, tmp_path):
    parser = build_parser()
    config_path = tmp_path / "dashboard_services.json"
    monkeypatch.setenv("SFT_LABEL_DASHBOARD_SERVICE_CONFIG", str(config_path))
    init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="0.0.0.0",
        port=8765,
        public_base_url="http://192.168.1.25:8765",
        config_path=config_path,
    )

    prompts = iter(["9000"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    attempts: list[int] = []

    def _fake_start(service):
        attempts.append(service.port)
        if service.port == 8765:
            raise DashboardPortConflictError(
                service_name=service.name,
                host=service.host,
                port=service.port,
                owner_pid=47920,
                owner_command="python -m http.server 8765",
                owned_by_service=False,
            )
        return {
            "name": service.name,
            "state": "running",
            "reachable": True,
            "url": service.base_url(),
            "public_url": service.share_base_url(),
        }

    monkeypatch.setattr("sft_label.cli.start_dashboard_service", _fake_start, raising=False)

    args = parser.parse_args(["dashboard-service", "start", "--name", "default"])
    status = cmd_dashboard_service(args)

    assert attempts == [8765, 9000]
    assert status["state"] == "running"
    store = load_dashboard_service_store(config_path)
    assert store.services["default"].port == 9000
    assert store.services["default"].public_base_url == "http://192.168.1.25:9000"
    out = capsys.readouterr().out
    assert "47920" in out
    assert "9000" in out


def test_cmd_dashboard_service_restart_exits_cleanly_without_tty(monkeypatch, capsys, tmp_path):
    parser = build_parser()
    config_path = tmp_path / "dashboard_services.json"
    monkeypatch.setenv("SFT_LABEL_DASHBOARD_SERVICE_CONFIG", str(config_path))
    init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="0.0.0.0",
        port=8765,
        public_base_url="http://192.168.1.25:8765",
        config_path=config_path,
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "sft_label.cli.restart_dashboard_service",
        lambda service: (_ for _ in ()).throw(
            DashboardPortConflictError(
                service_name=service.name,
                host=service.host,
                port=service.port,
                owner_pid=47920,
                owner_command="python -m http.server 8765",
                owned_by_service=False,
            )
        ),
        raising=False,
    )

    args = parser.parse_args(["dashboard-service", "restart", "--name", "default"])
    try:
        cmd_dashboard_service(args)
        raise AssertionError("Expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 1

    store = load_dashboard_service_store(config_path)
    assert store.services["default"].port == 8765
    out = capsys.readouterr().out
    assert "cannot bind 0.0.0.0:8765" in out


def test_cmd_dashboard_service_start_can_cancel_port_retry_with_ctrl_c(monkeypatch, capsys, tmp_path):
    parser = build_parser()
    config_path = tmp_path / "dashboard_services.json"
    monkeypatch.setenv("SFT_LABEL_DASHBOARD_SERVICE_CONFIG", str(config_path))
    init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="0.0.0.0",
        port=8765,
        public_base_url="http://192.168.1.25:8765",
        config_path=config_path,
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(
        "sft_label.cli.start_dashboard_service",
        lambda service: (_ for _ in ()).throw(
            DashboardPortConflictError(
                service_name=service.name,
                host=service.host,
                port=service.port,
                owner_pid=47920,
                owner_command="python -m http.server 8765",
                owned_by_service=False,
            )
        ),
        raising=False,
    )

    args = parser.parse_args(["dashboard-service", "start", "--name", "default"])
    try:
        cmd_dashboard_service(args)
        raise AssertionError("Expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 130

    store = load_dashboard_service_store(config_path)
    assert store.services["default"].port == 8765
    out = capsys.readouterr().out
    assert "已取消 dashboard 服务启动。" in out


def _write_extension_spec(path, *, prompt_text="Label UI data.", options=None, trigger=True, spec_id="ui_fine_labels"):
    options = options or ["form", "modal"]
    trigger_block = """
trigger:
  domain_any_of: [web-frontend]
""".strip() if trigger else ""
    path.write_text(
        (
            f"""
id: {spec_id}
spec_version: v1
display_name: UI Fine Labels
description: Fine-grained UI labels.
prompt: |
  {prompt_text}
schema:
  component_type:
    type: multi_enum
    description: Main UI components.
    options: [{", ".join(options)}]
{trigger_block}
""".strip()
            + "\n"
        ),
        encoding="utf-8",
    )


def _extension_run_answers(extension_paths: list[str]) -> list[str]:
    base_answers = ["3", "1", "dataset", "", ""] + [""] * 24
    return base_answers[:5] + _extension_manual_flow(extension_paths) + base_answers[5:] + [""]


def _extension_manual_flow(extension_paths: list[str]) -> list[str]:
    flow = ["y"]
    for index, path in enumerate(extension_paths):
        flow.extend(["2", path])
        flow.append("n" if index == len(extension_paths) - 1 else "y")
    return flow


def test_build_run_pass1_pass2_semantic_plan():
    io = StubIO(
        [
            "5",          # workflow: pass1+pass2+pass3
            "1",          # run mode: new
            "data.json",  # --input
            "",           # --output
            "",           # inline mode (refresh)
            "",           # extension labeling prompt (default no)
            "",           # --limit (default 0)
            "",           # shuffle (default no)
            "",           # arbitration (default yes)
            "",           # concurrency (default 200)
            "",           # rps-limit (default)
            "",           # adaptive runtime (default yes)
            "",           # recovery sweep (default yes)
            "",           # prompt mode (full)
            "",           # model
            "",           # tag-stats
            "",           # rarity mode (default absolute)
            "",           # extension rarity mode (default off)
            "n",          # env override
            "",           # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "run",
        "--input",
        "data.json",
        "--concurrency",
        "200",
        "--score",
        "--semantic-cluster",
    ]
    assert plan.env_overrides == {}


def test_build_run_plan_can_disable_adaptive_runtime_and_recovery_sweep():
    io = StubIO(
        [
            "3",          # workflow: run-pass1
            "1",          # run mode: new
            "data.json",  # --input
            "",           # --output
            "",           # inline mode (refresh)
            "",           # extension labeling prompt (default no)
            "",           # --limit (default 0)
            "",           # shuffle (default no)
            "",           # arbitration (default yes)
            "",           # concurrency (default 200)
            "",           # rps-limit (default)
            "n",          # adaptive runtime (disable)
            "n",          # recovery sweep (disable)
            "",           # prompt mode (full)
            "",           # model
            "n",          # env override
            "",           # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "run",
        "--input",
        "data.json",
        "--concurrency",
        "200",
        "--no-adaptive-runtime",
        "--no-recovery-sweep",
    ]


def test_build_run_plan_legacy_supports_custom_runtime_values():
    io = StubIO(
        [
            "3",            # workflow: run-pass1
            "1",            # run mode: new
            "data.json",    # --input
            "",             # --output
            "",             # inline mode (refresh)
            "",             # extension labeling prompt (default no)
            "",             # --limit
            "",             # shuffle
            "",             # arbitration
            "6",            # concurrency -> custom
            "260",          # custom concurrency
            "8",            # rps-limit -> custom
            "88.5",         # custom rps-limit
            "",             # adaptive runtime
            "",             # recovery sweep
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )

    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "run",
        "--input",
        "data.json",
        "--concurrency",
        "260",
        "--rps-limit",
        "88.5",
    ]


def test_build_score_plan_with_llm_env_override():
    io = StubIO(
        [
            "4",                      # workflow: score
            "labeled.json",           # --input
            "",                       # --tag-stats
            "",                       # rarity mode (default absolute)
            "",                       # extension rarity mode (default off)
            "",                       # --limit
            "",                       # --resume
            "",                       # concurrency (default 200)
            "",                       # rps-limit (default)
            "",                       # adaptive runtime (default yes)
            "",                       # recovery sweep (default yes)
            "",                       # prompt mode
            "",                       # model
            "y",                      # env override
            "http://proxy/v1",        # LITELLM_BASE
            "secret",                 # LITELLM_KEY
            "",                       # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == ["score", "--input", "labeled.json", "--concurrency", "200"]
    assert plan.env_overrides == {
        "LITELLM_BASE": "http://proxy/v1",
        "LITELLM_KEY": "secret",
    }


def test_build_score_plan_legacy_supports_custom_runtime_values():
    io = StubIO(
        [
            "4",            # workflow: score
            "labeled.json", # --input
            "",             # --tag-stats
            "",             # rarity mode
            "",             # extension rarity mode
            "",             # --limit
            "",             # --resume
            "6",            # concurrency -> custom
            "260",          # custom concurrency value
            "8",            # rps-limit -> custom
            "123.5",        # custom rps-limit value
            "",             # adaptive runtime
            "",             # recovery sweep
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )

    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "score",
        "--input",
        "labeled.json",
        "--concurrency",
        "260",
        "--rps-limit",
        "123.5",
    ]


def test_build_run_plan_switch_panel_emits_rollout_preset(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_switch_panel(input_fn, output_fn, title, fields):
        captured["field_keys"] = [field.key for field in fields]
        return {
            "rollout_preset": "planner_hybrid",
            "prompt_mode": "compact",
            "shuffle": "off",
            "arbitration": "on",
            "limit": "",
            "concurrency": str(DEFAULT_CONCURRENCY),
            "rps_limit": "",
            "rps_warmup": "",
            "adaptive_runtime": "on",
            "recovery_sweep": "on",
            "request_timeout": "",
            "max_retries": "",
            "model_override": "no",
            "env_override": "no",
        }

    monkeypatch.setattr("sft_label.launcher._supports_switch_panel", lambda input_fn: True)
    monkeypatch.setattr("sft_label.launcher._ask_switch_panel", _fake_switch_panel)

    io = StubIO(
        [
            "3",          # workflow: run-pass1
            "1",          # run mode: new
            "data.json",  # --input
            "",           # --output
            "",           # inline mode
            "",           # extension prompt
            "",           # extra flags
        ]
    )

    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert captured["field_keys"][:2] == ["rollout_preset", "prompt_mode"]
    assert plan.argv == [
        "run",
        "--input",
        "data.json",
        "--rollout-preset",
        "planner_hybrid",
        "--prompt-mode",
        "compact",
        "--concurrency",
        "200",
    ]


def test_build_score_plan_switch_panel_emits_default_rollout_preset(monkeypatch):
    def _fake_switch_panel(input_fn, output_fn, title, fields):
        return {
            "rollout_preset": DEFAULT_ROLLOUT_PRESET,
            "prompt_mode": "compact",
            "resume": "off",
            "rarity_mode": "absolute",
            "limit": "",
            "concurrency": str(DEFAULT_CONCURRENCY),
            "rps_limit": "",
            "rps_warmup": "",
            "adaptive_runtime": "on",
            "recovery_sweep": "on",
            "request_timeout": "",
            "max_retries": "",
            "model_override": "no",
            "env_override": "no",
        }

    monkeypatch.setattr("sft_label.launcher._supports_switch_panel", lambda input_fn: True)
    monkeypatch.setattr("sft_label.launcher._ask_switch_panel", _fake_switch_panel)

    io = StubIO(
        [
            "4",             # workflow: score
            "labeled.json",  # --input
            "",              # --tag-stats
            "",              # extension rarity mode
            "",              # extra flags
        ]
    )

    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "score",
        "--input",
        "labeled.json",
        "--rollout-preset",
        DEFAULT_ROLLOUT_PRESET,
        "--prompt-mode",
        "compact",
        "--concurrency",
        "200",
    ]


def test_build_smart_resume_plan_routes_to_score_for_labeled_run_dir(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path)]


def test_build_smart_resume_plan_routes_to_resumed_score_when_scored_exists(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path), "--resume"]


def test_build_smart_resume_plan_routes_to_resumed_score_when_hidden_next_artifact_exists(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / ".scored.jsonl.next").write_text("", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path), "--resume"]


def test_build_smart_resume_plan_routes_to_resumed_score_when_hidden_checkpoint_artifact_exists(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / ".scored.checkpoint.jsonl").write_text("", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path), "--resume"]


def test_build_smart_resume_plan_routes_to_resumed_score_when_visible_checkpoint_artifact_exists(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.batch_a.checkpoint.jsonl").write_text("", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path), "--resume"]


def test_build_smart_resume_plan_routes_to_run_resume_when_checkpoint_exists(tmp_path):
    (tmp_path / "checkpoint.json").write_text('{"status":"in_progress"}', encoding="utf-8")
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["run", "--resume", str(tmp_path)]


def test_build_smart_resume_plan_checkpoint_done_routes_to_score(tmp_path):
    (tmp_path / "checkpoint.json").write_text('{"status":"done"}', encoding="utf-8")
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path)]


def test_build_smart_resume_plan_routes_to_complete_postprocess_when_pass2_is_deferred(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"deferred"},"dashboard":{"status":"completed"}}}',
        encoding="utf-8",
    )

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_build_smart_resume_plan_routes_to_complete_postprocess_when_pass2_is_incomplete(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"completed"},"dashboard":{"status":"failed"}}}',
        encoding="utf-8",
    )

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_build_smart_resume_plan_prefers_pass2_resume_when_hidden_artifacts_exist_even_if_postprocess_incomplete(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (artifact_dir / ".scored.jsonl.next").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"deferred"},"dashboard":{"status":"completed"}}}',
        encoding="utf-8",
    )

    io = StubIO(
        [
            "2",               # workflow: smart resume
            str(tmp_path),     # run dir
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["score", "--concurrency", "200", "--input", str(tmp_path), "--resume"]


def test_build_smart_resume_plan_prefers_modern_postprocess_over_legacy_stats_without_postprocess(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "stats_scoring.json").write_text(
        '{"total_scored": 12, "mean_value_score": 6.3}',
        encoding="utf-8",
    )
    (tmp_path / "meta_label_data" / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"deferred"},"dashboard":{"status":"completed"}}}',
        encoding="utf-8",
    )

    io = StubIO(
        [
            "2",
            str(tmp_path),
            "",
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_build_smart_resume_plan_fail_closed_when_root_and_meta_postprocess_conflict(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"completed"},"dashboard":{"status":"completed"}}}',
        encoding="utf-8",
    )
    (tmp_path / "meta_label_data" / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"deferred"},"dashboard":{"status":"completed"}}}',
        encoding="utf-8",
    )

    io = StubIO(["2", str(tmp_path), ""])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_build_smart_resume_plan_fail_closed_when_postprocess_status_block_is_missing(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text(
        '{"postprocess":{"conversation_scores":{"status":"completed"}}}',
        encoding="utf-8",
    )

    io = StubIO(["2", str(tmp_path), ""])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_build_smart_resume_plan_fail_closed_when_entire_postprocess_block_is_missing(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text(
        '{"total_scored":1}',
        encoding="utf-8",
    )

    io = StubIO(["2", str(tmp_path), ""])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_build_smart_resume_plan_fail_closed_when_postprocess_stats_json_is_invalid(tmp_path):
    artifact_dir = tmp_path / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "scored.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary_stats_scoring.json").write_text("{invalid", encoding="utf-8")

    io = StubIO(["2", str(tmp_path), ""])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == ["complete-postprocess", "--input", str(tmp_path)]


def test_cmd_run_resume_with_score_forwards_resume_to_pass2(monkeypatch):
    captured = {}

    async def _fake_run(*args, **kwargs):
        return {"run_dir": "/tmp/fake-run"}

    async def _fake_score(*args, **kwargs):
        captured["resume"] = kwargs.get("resume")
        return {"run_dir": "/tmp/fake-run"}

    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))
    monkeypatch.setitem(sys.modules, "sft_label.scoring", SimpleNamespace(run_scoring=_fake_score))

    parser = build_parser()
    args = parser.parse_args(["run", "--resume", "/tmp/existing-run", "--score"])
    cmd_run(args)

    assert captured["resume"] is True


def test_build_filter_plan_enforces_at_least_one_criterion():
    io = StubIO(
        [
            "7",            # workflow: filter
            "scored.json",  # --input
            "",             # --output
            "",             # --format (scored)
            "",             # include-unscored
            "",             # --value-min
            "",             # --selection-min
            "",             # --difficulty
            "",             # thinking mode
            "",             # include-tags
            "",             # exclude-tags
            "",             # exclude-inherited
            "",             # verify-source
            "",             # conversation filters
            "",             # turn-level pruning
            "",             # fallback value-min -> default 6.0
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == ["filter", "--input", "scored.json", "--value-min", "6.0"]


def test_build_semantic_plan_api_provider_supports_env_override():
    io = StubIO(
        [
            "6",               # workflow: semantic
            "run_dir",         # --input
            "",                # --output
            "",                # --limit
            "",                # --resume
            "",                # export representatives
            "y",               # configure advanced semantic params
            "",                # --semantic-long-turn-threshold
            "",                # --semantic-window-size
            "",                # --semantic-window-stride
            "",                # --semantic-pinned-prefix-max-turns
            "3",               # provider: api
            "",                # --semantic-embedding-model
            "",                # --semantic-embedding-dim
            "",                # --semantic-embedding-batch-size
            "",                # --semantic-embedding-max-workers
            "",                # semhash bits
            "",                # --semantic-semhash-seed
            "",                # --semantic-semhash-bands
            "",                # --semantic-hamming-radius
            "",                # --semantic-ann-top-k
            "",                # --semantic-ann-sim-threshold
            "",                # --semantic-output-prefix
            "y",               # env override for api provider
            "http://embed/v1", # LITELLM_BASE
            "embed-key",       # LITELLM_KEY
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "semantic-cluster",
        "--input",
        "run_dir",
        "--semantic-embedding-provider",
        "api",
    ]
    assert plan.env_overrides == {
        "LITELLM_BASE": "http://embed/v1",
        "LITELLM_KEY": "embed-key",
    }


def test_build_dashboard_service_init_plan():
    io = StubIO(
        [
            "8",                     # workflow: dashboard maintenance
            "1",                     # action: init
            "",                      # name -> default
            "/srv/sft-label-dashboard",  # web root
            "",                      # host default
            "",                      # port default
            "",                      # service-type default pm2
            "https://dash.example.com",  # public base url
            "sft-label-dashboard-prod",  # pm2 name
            "",                      # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "dashboard-service",
        "init",
        "--name",
        "default",
        "--web-root",
        "/srv/sft-label-dashboard",
        "--public-base-url",
        "https://dash.example.com",
        "--pm2-name",
        "sft-label-dashboard-prod",
    ]


def test_build_dashboard_service_runs_plan():
    io = StubIO(
        [
            "8",      # workflow: dashboard maintenance
            "7",      # action: runs
            "prod",   # service name
            "",       # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "dashboard-service",
        "runs",
        "--name",
        "prod",
    ]


def test_build_launch_plan_can_cancel():
    io = StubIO(["0"])
    assert build_launch_plan(input_fn=io.input, output_fn=io.output) is None


def test_can_back_to_workflow_selection_with_zero_in_submenu():
    io = StubIO(
        [
            "1",  # select run-pass1 workflow
            "0",  # back from run mode selection
            "0",  # cancel at workflow menu
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is None
    assert any("已返回任务选择" in str(item) for item in io.outputs)


def test_can_back_to_workflow_selection_with_back_token():
    io = StubIO(
        [
            "1",          # select run-pass1 workflow
            "1",          # run mode: new
            "b",          # back from required input prompt
            "0",          # cancel at workflow menu
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is None
    assert any("已返回任务选择" in str(item) for item in io.outputs)


def test_format_helpers():
    assert format_command(["score", "--input", "a b.json"]) == "sft-label score --input 'a b.json'"
    assert (
        format_env_prefix({"LITELLM_KEY": "abc", "LITELLM_BASE": "http://x/v1"})
        == "LITELLM_BASE=http://x/v1 LITELLM_KEY=abc"
    )


def test_sanitize_prompt_input_removes_arrow_escape_sequences():
    clean, had_control = sanitize_prompt_input("\x1b[A")
    assert clean == ""
    assert had_control is True

    clean2, had_control2 = sanitize_prompt_input("abc\x1b[D")
    assert clean2 == "abc"
    assert had_control2 is True


def test_decode_utf8_input_byte_keeps_multibyte_chars():
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    parts = []
    for b in "中文路径".encode("utf-8"):
        parts.append(_decode_utf8_input_byte(decoder, bytes([b])))
    assert "".join(parts) == "中文路径"


def test_all_workflows_generate_parseable_argv(tmp_path):
    parser = build_parser()
    smart_resume_dir = tmp_path / "smart_resume_run"
    artifact_dir = smart_resume_dir / "meta_label_data" / "files" / "demo"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "labeled.json").write_text("[]", encoding="utf-8")
    workflow_answers = {
        # 1. run-pass1-pass2
        1: ["1", "data.json", "", "", "", "", "", "", "", "", "", "stats.json", "", "n", ""],
        # 2. smart-resume
        2: [str(smart_resume_dir), ""],
        # 3. run-pass1
        3: ["1", "data.json", "", "", "", "", "", "", "", "", "", "n", ""],
        # 4. score
        4: ["labeled.json", "", "", "", "", "", "", "", "", "n", ""],
        # 5. run-pass1-pass2-semantic
        5: ["1", "data.json", "", "", "", "", "", "", "", "", "", "stats.json", "", "n", ""],
        # 6. semantic
        6: ["run_dir", "", "", "", "", "", ""],
        # 7. filter
        7: ["scored.json", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        # 8. dashboard-service maintenance
        8: ["2", "", ""],  # action=status, name default, extra flags
        # 9. recompute-stats
        9: ["run_dir/", "", "", "", ""],
        # 10. regenerate-dashboard
        10: ["run_dir/", "", "", "", ""],
        # 11. refresh-rarity
        11: ["run_dir/", "", "", "", "", ""],
        # 12. analyze-unmapped
        12: ["run_dir/", "", "", "", "", ""],
        # 13. optimize-layout
        13: ["run_dir/", "", "", "", ""],
        # 14. validate
        14: [],
        # 15. export-semantic
        15: ["run_dir/", "out.jsonl", "", ""],
        # 16. export-review
        16: ["labeled.json", "review.csv", "", "", ""],
    }

    for wf_num, answers in workflow_answers.items():
        io = StubIO([str(wf_num)] + answers + [""] * 10)
        plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
        assert plan is not None
        parsed = parser.parse_args(plan.argv)
        assert parsed.command is not None


def test_llm_key_can_be_cleared_from_existing_env(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE", "http://example/v1")
    monkeypatch.setenv("LITELLM_KEY", "existing-key")

    io = StubIO(
        [
            "4",            # score
            "labeled.json", # --input
            "",             # --tag-stats
            "",             # rarity mode (default absolute)
            "",             # extension rarity mode (default off)
            "",             # --limit
            "",             # --resume
            "",             # concurrency (default 200)
            "",             # rps-limit (default)
            "",             # adaptive runtime (default yes)
            "",             # recovery sweep (default yes)
            "",             # prompt mode
            "",             # model
            "y",            # override env
            "",             # keep LITELLM_BASE default
            "-",            # clear key
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.env_overrides["LITELLM_BASE"] == "http://example/v1"
    assert plan.env_overrides["LITELLM_KEY"] == ""


def test_required_text_prompt_ignores_arrow_key_input(capsys):
    io = StubIO(
        [
            "3",          # workflow: run-pass1
            "1",          # run mode: new
            "\x1b[A",     # arrow key input on required --input field
            "data.json",  # actual --input
            "",           # --output
            "",           # inline mode
            "",           # extension labeling prompt (default no)
            "",           # --limit
            "",           # --shuffle
            "",           # --arbitration
            "",           # concurrency (default 200)
            "",           # rps-limit (default)
            "",           # adaptive runtime (default yes)
            "",           # recovery sweep (default yes)
            "",           # --prompt mode
            "",           # --model
            "n",          # env override
            "",           # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    captured = capsys.readouterr()

    assert plan is not None
    assert plan.argv[:3] == ["run", "--input", "data.json"]
    assert "检测到方向键/控制字符输入，已忽略" in captured.out


def test_build_run_plan_can_set_migrate_mode():
    io = StubIO(
        [
            "3",             # workflow: run-pass1
            "1",             # run mode: new
            "dataset",       # --input
            "",              # --output
            "3",             # inline mode: migrate
            "old-run",       # --migrate-from
            "",              # extension labeling prompt (default no)
            "",              # --limit
            "",              # --shuffle
            "",              # --arbitration
            "",              # concurrency (default 200)
            "",              # rps-limit (default)
            "",              # adaptive runtime (default yes)
            "",              # recovery sweep (default yes)
            "",              # --prompt mode
            "",              # --model
            "n",             # env override
            "",              # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv[:7] == [
        "run",
        "--input",
        "dataset",
        "--mode",
        "migrate",
        "--migrate-from",
        "old-run",
    ]


def test_build_run_plan_can_set_recompute_mode():
    io = StubIO(
        [
            "3",        # workflow: run-pass1
            "1",        # run mode: new
            "dataset",  # --input
            "",         # --output
            "4",        # inline mode: recompute
            "",         # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "run",
        "--input",
        "dataset",
        "--mode",
        "recompute",
    ]


def test_build_run_resume_plan_skips_output_prompt():
    io = StubIO(
        [
            "3",        # workflow: run-pass1
            "2",        # run mode: resume
            "run-dir",  # --resume
            "",         # also set --input? no
            "",         # extension labeling prompt (default no)
            "",         # --limit
            "",         # --shuffle
            "",         # --arbitration
            "",         # concurrency (default 200)
            "",         # rps-limit (default)
            "",         # adaptive runtime (default yes)
            "",         # recovery sweep (default yes)
            "",         # --prompt mode
            "",         # --model
            "n",        # env override
            "",         # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "run",
        "--resume",
        "run-dir",
        "--concurrency",
        "200",
    ]
    assert "--output" not in plan.argv
    assert not any("输出目录（可选）" in str(item) for item in io.outputs)
    assert any("继续指定 run 目录" in str(item) for item in io.outputs)


def test_build_launch_plan_can_render_english():
    io = StubIO(["14"])
    try:
        plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="en")
        assert plan is not None
        assert plan.argv == ["validate"]
        assert any("Interactive task launcher" in str(item) for item in io.outputs)
    finally:
        set_language("zh")


def test_chinese_prompt_uses_fullwidth_colon_without_english_suffix():
    io = StubIO(["0"])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="zh")
    assert plan is None
    assert any("请选择任务编号 [0-16, 默认 1]：" in str(item) for item in io.outputs)
    assert any("如果任务中断或报错" in str(item) and "智能续跑" in str(item) for item in io.outputs)
    assert not any("Select workflow number" in str(item) for item in io.outputs)


def test_smart_resume_moves_into_pipeline_group():
    io = StubIO(["0"])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="zh")
    assert plan is None
    output_lines = [str(item) for item in io.outputs]
    smart_resume_idx = next(i for i, line in enumerate(output_lines) if "智能续跑" in line)
    data_curation_idx = next(i for i, line in enumerate(output_lines) if "[数据整理" in line)
    assert smart_resume_idx < data_curation_idx
    assert not any("[恢复 / Recovery]" in line or "[恢复]" in line for line in output_lines)


def test_chinese_llm_override_prompt_keeps_full_key_name():
    io = StubIO(
        [
            "4",            # workflow: score
            "labeled.json", # --input
            "",             # --tag-stats
            "",             # rarity mode (default absolute)
            "",             # extension rarity mode (default off)
            "",             # --limit
            "",             # --resume
            "",             # concurrency (default 200)
            "",             # rps-limit (default)
            "",             # adaptive runtime (default yes)
            "",             # recovery sweep (default yes)
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="zh")
    assert plan is not None
    assert any("覆盖本次 LITELLM_BASE/LITELLM_KEY [y/N]：" in str(item) for item in io.outputs)
    assert not any("覆盖本次 LITELLM_BASE / LITELLM_KEY" in str(item) for item in io.outputs)


def test_build_optimize_layout_plan():
    io = StubIO(
        [
            "13",          # workflow: optimize-layout
            "run_dir",     # --input
            "y",           # --apply
            "y",           # --prune-legacy
            "plan.json",   # --manifest
            "",            # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "optimize-layout",
        "--input",
        "run_dir",
        "--apply",
        "--prune-legacy",
        "--manifest",
        "plan.json",
    ]


def test_build_analyze_unmapped_plan():
    io = StubIO(
        [
            "12",          # workflow: analyze-unmapped
            "run_dir",     # --input
            "task",        # --dimension
            "10",          # --top
            "1",           # --examples
            "y",           # --stats-only
            "",            # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "analyze-unmapped",
        "--input",
        "run_dir",
        "--dimension",
        "task",
        "--top",
        "10",
        "--examples",
        "1",
        "--stats-only",
    ]


def test_build_score_plan_can_set_percentile_rarity_mode():
    io = StubIO(
        [
            "4",            # workflow: score
            "labeled.json", # --input
            "",             # --tag-stats
            "2",            # rarity mode: percentile
            "",             # extension rarity mode: off (default)
            "",             # --limit
            "",             # --resume
            "",             # concurrency (default 200)
            "",             # rps-limit (default)
            "",             # adaptive runtime (default yes)
            "",             # recovery sweep (default yes)
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "score",
        "--input",
        "labeled.json",
        "--rarity-mode",
        "percentile",
        "--concurrency",
        "200",
    ]


def test_build_score_plan_can_set_extension_rarity_preview_mode():
    io = StubIO(
        [
            "4",            # workflow: score
            "labeled.json", # --input
            "",             # --tag-stats
            "",             # rarity mode: absolute
            "2",            # extension rarity mode: preview
            "",             # --limit
            "",             # --resume
            "",             # concurrency (default 200)
            "",             # rps-limit (default)
            "",             # adaptive runtime (default yes)
            "",             # recovery sweep (default yes)
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "score",
        "--input",
        "labeled.json",
        "--extension-rarity-mode",
        "preview",
        "--concurrency",
        "200",
    ]


def test_build_refresh_rarity_plan_can_set_extension_rarity_preview_mode():
    io = StubIO(
        [
            "11",           # workflow: refresh-rarity
            "run_dir",      # --input
            "",             # --tag-stats
            "",             # rarity mode: absolute
            "2",            # extension rarity mode: preview
            "",             # --output
            "",             # workers
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "refresh-rarity",
        "--input",
        "run_dir",
        "--extension-rarity-mode",
        "preview",
    ]


def test_build_recompute_plan_can_set_workers():
    io = StubIO(
        [
            "9",        # workflow: recompute-stats
            "run_dir",  # --input
            "",         # pass selection -> both
            "",         # --output
            "2",        # workers -> 4
            "",         # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "recompute-stats",
        "--input",
        "run_dir",
        "--workers",
        "4",
    ]



def test_cmd_start_can_auto_publish_dashboard(monkeypatch, capsys, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]))

    answers = iter(["", "y"])  # confirm, auto-publish yes
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr("sft_label.cli.dashboard_service_status", lambda svc: {"state": "running", "reachable": True, "url": svc.base_url(), "public_url": svc.share_base_url()}, raising=False)

    published = {
        "run_id": "demo_run",
        "dashboards": {
            "labeling": {"url": "http://127.0.0.1:8765/runs/demo_run/dashboard_labeling_demo.html"},
            "scoring": {"url": "http://127.0.0.1:8765/runs/demo_run/dashboard_scoring_demo.html"},
        },
    }
    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", lambda svc, run_dir, config_path=None: published, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert "dashboard_labeling_demo.html" in out
    assert "dashboard_scoring_demo.html" in out


def test_cmd_start_blank_dashboard_confirmation_defaults_to_yes_before_execute(monkeypatch, capsys, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json", "--concurrency", "200"]),
    )

    prompts: list[str] = []

    def _fake_input(prompt: str):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("sft_label.launcher.interactive_input", _fake_input, raising=False)

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url(), "public_url": svc.share_base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(service, run_dir, config_path=None):
        publish_calls["count"] += 1
        return {"run_id": "demo_run", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    auto_idx = next(i for i, prompt in enumerate(prompts) if "dashboard" in prompt.lower())
    execute_idx = next(i for i, prompt in enumerate(prompts) if "execute now" in prompt.lower() or "立即执行" in prompt)
    assert auto_idx < execute_idx
    assert publish_calls["count"] == 1
    assert "Dashboard plan" in out or "Dashboard 计划" in out


def test_cmd_start_auto_publish_uses_default_dashboard_service_without_prompt(monkeypatch, capsys, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]))

    answers = iter(["", "y"])  # confirm, auto publish
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)

    service_a = DashboardServiceConfig(name="a", web_root=str(tmp_path / "a"), host="127.0.0.1", port=8765)
    service_b = DashboardServiceConfig(
        name="b",
        web_root=str(tmp_path / "b"),
        host="127.0.0.1",
        port=9000,
        public_base_url="https://dash.example.com",
        service_type="pm2",
    )
    store = DashboardServiceStore(default_service="b", services={"a": service_a, "b": service_b})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr("sft_label.cli.dashboard_service_status", lambda svc: {"state": "running", "reachable": True, "url": svc.base_url(), "public_url": svc.share_base_url()}, raising=False)

    published_calls = {}

    def _fake_publish(service, run_dir, config_path=None):
        published_calls["service"] = service.name
        return {
            "run_id": "demo_run",
            "dashboards": {
                "labeling": {"url": "https://dash.example.com/runs/demo_run/dashboard_labeling_demo.html"},
            },
        }

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert published_calls["service"] == "b"
    assert "https://dash.example.com/runs/demo_run/dashboard_labeling_demo.html" in out
    assert "请选择 dashboard 服务" not in out


def test_cmd_start_dashboard_service_maintenance_can_continue_in_same_session(monkeypatch, capsys):
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    first_plan = LaunchPlan(argv=["dashboard-service", "status"], workflow_key="dashboard-service")
    next_plan = LaunchPlan(argv=["dashboard-service", "runs"], workflow_key="dashboard-service")

    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: first_plan)
    monkeypatch.setattr("sft_label.launcher.build_dashboard_service_plan", lambda **kwargs: next_plan)

    answers = iter(["", "y", "n"])  # confirm start, continue once, then exit
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)

    dispatched = []

    def _fake_dispatch(args, parser):
        dispatched.append((args.command, getattr(args, "dashboard_service_action", None)))
        return {}

    monkeypatch.setattr("sft_label.cli.dispatch_command", _fake_dispatch, raising=False)

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    assert dispatched == [
        ("dashboard-service", "status"),
        ("dashboard-service", "runs"),
    ]
    out = capsys.readouterr().out
    assert "dashboard-service status" in out
    assert "dashboard-service runs" in out


def test_cmd_start_wraps_auto_publish_with_heartbeat(monkeypatch, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]))

    answers = iter(["", "", ""])  # auto-publish default yes, final confirm yes, restart=no(default)
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr("sft_label.cli.dashboard_service_status", lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()}, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.publish_run_dashboards",
        lambda svc, run_dir, config_path=None: {"run_id": "demo_run", "dashboards": {}},
        raising=False,
    )

    wrapped_messages = []

    def _fake_heartbeat(message, fn, **kwargs):
        wrapped_messages.append(message)
        return fn()

    monkeypatch.setattr("sft_label.cli.run_with_heartbeat", _fake_heartbeat, raising=False)

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    assert any("Publishing dashboards" in message for message in wrapped_messages)


def test_cmd_start_dashboard_bootstrap_retries_with_new_port_after_conflict(monkeypatch, tmp_path):
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    config_path = tmp_path / "dashboard_services.json"
    monkeypatch.setenv("SFT_LABEL_DASHBOARD_SERVICE_CONFIG", str(config_path))
    init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="0.0.0.0",
        port=8765,
        public_base_url="http://192.168.1.25:8765",
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]),
    )

    answers = iter(["", "", "", "9000"])  # auto-publish default yes, start-now, final confirm, new port
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {
            "name": svc.name,
            "state": "stopped",
            "reachable": False,
            "url": svc.base_url(),
            "public_url": svc.share_base_url(),
        },
        raising=False,
    )

    attempts: list[int] = []

    def _fake_start(service):
        attempts.append(service.port)
        if service.port == 8765:
            raise DashboardPortConflictError(
                service_name=service.name,
                host=service.host,
                port=service.port,
                owner_pid=47920,
                owner_command="python -m http.server 8765",
                owned_by_service=False,
            )
        return {
            "name": service.name,
            "state": "running",
            "reachable": True,
            "url": service.base_url(),
            "public_url": service.share_base_url(),
        }

    monkeypatch.setattr("sft_label.cli.start_dashboard_service", _fake_start, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.publish_run_dashboards",
        lambda svc, run_dir, config_path=None: {"run_id": "demo_run", "dashboards": {}},
        raising=False,
    )

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    assert attempts == [8765, 9000]
    store = load_dashboard_service_store(config_path)
    assert store.services["default"].port == 9000
    assert store.services["default"].public_base_url == "http://192.168.1.25:9000"


def test_cmd_start_auto_publish_bootstraps_lan_service_with_shareable_url(monkeypatch, capsys, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]),
    )

    answers = iter([
        "",   # auto-publish default yes
        "",   # service name -> default
        "",   # web root -> default
        "2",  # exposure: LAN
        "",   # port -> default
        "",   # share URL -> accept guessed default
        "",   # start service now -> default yes
        "",   # final confirm execution
    ])
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    monkeypatch.setattr(
        "sft_label.cli.load_dashboard_service_store",
        lambda config_path=None: DashboardServiceStore(),
        raising=False,
    )
    monkeypatch.setattr(
        "sft_label.cli._guess_local_network_host",
        lambda: "192.168.1.25",
        raising=False,
    )

    init_calls = {}

    def _fake_init_dashboard_service(**kwargs):
        init_calls.update(kwargs)
        return DashboardServiceConfig(
            name=kwargs["name"],
            web_root=str(kwargs["web_root"]),
            host=kwargs["host"],
            port=kwargs["port"],
            service_type=kwargs["service_type"],
            public_base_url=kwargs.get("public_base_url"),
            pm2_name=kwargs.get("pm2_name"),
        )

    monkeypatch.setattr("sft_label.cli.init_dashboard_service", _fake_init_dashboard_service, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {
            "state": "stopped",
            "reachable": False,
            "url": svc.base_url(),
            "public_url": svc.share_base_url(),
        },
        raising=False,
    )
    monkeypatch.setattr(
        "sft_label.cli.start_dashboard_service",
        lambda svc: {
            "state": "running",
            "reachable": True,
            "url": svc.base_url(),
            "public_url": svc.share_base_url(),
        },
        raising=False,
    )
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.publish_run_dashboards",
        lambda svc, run_dir, config_path=None: {
            "run_id": "demo_run",
            "dashboards": {
                "labeling": {"url": "http://192.168.1.25:8765/runs/demo_run/dashboard_labeling_demo.html"},
            },
        },
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert init_calls["host"] == "0.0.0.0"
    assert init_calls["public_base_url"] == "http://192.168.1.25:8765"
    assert init_calls["service_type"] == "pm2"
    assert "http://192.168.1.25:8765" in out


def test_extension_labeling_prompt_flow_collects_paths(tmp_path):
    extension_paths = [str(tmp_path / "ui.yaml"), str(tmp_path / "mobile.yaml")]
    _write_extension_spec(Path(extension_paths[0]), prompt_text="Label UI data.")
    _write_extension_spec(Path(extension_paths[1]), prompt_text="Label mobile data.", spec_id="mobile_fine_labels")
    io = StubIO(_extension_run_answers(extension_paths))
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None

    prompt_log = " ".join(str(item).lower() for item in io.outputs)
    assert "extension labeling" in prompt_log or "扩展标注" in prompt_log
    assert "extension" in prompt_log and any(keyword in prompt_log for keyword in ("path", "路径", "spec", "规范"))


def test_extension_labeling_plan_includes_repeated_flags(tmp_path):
    extension_paths = [str(tmp_path / "ui.yaml"), str(tmp_path / "mobile.yaml")]
    _write_extension_spec(Path(extension_paths[0]), prompt_text="Label UI data.")
    _write_extension_spec(Path(extension_paths[1]), prompt_text="Label mobile data.", spec_id="mobile_fine_labels")
    io = StubIO(_extension_run_answers(extension_paths))
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None

    flag_positions = [idx for idx, arg in enumerate(plan.argv) if arg == "--label-extension"]
    assert len(flag_positions) == len(extension_paths)
    for index, position in enumerate(flag_positions):
        assert plan.argv[position + 1] == extension_paths[index]


def test_extension_labeling_prompt_flow_shows_spec_summary_and_compact_guidance(tmp_path):
    spec_path = tmp_path / "ui-extension.yaml"
    _write_extension_spec(spec_path, prompt_text=("Label UI data with frontend review hints. " * 80).strip())

    io = StubIO(_extension_run_answers([str(spec_path)]))
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    joined = "\n".join(str(item) for item in io.outputs)
    assert "ui_fine_labels" in joined
    assert "2000" in joined
    assert any(keyword in joined.lower() for keyword in ["prompt", "schema", "compact"])


def test_ask_extension_spec_paths_reprompts_on_duplicate_extension_id(tmp_path):
    first_path = tmp_path / "first.yaml"
    duplicate_id_path = tmp_path / "duplicate.yaml"
    unique_path = tmp_path / "unique.yaml"
    _write_extension_spec(first_path, spec_id="ui_fine_labels")
    _write_extension_spec(duplicate_id_path, spec_id="ui_fine_labels", prompt_text="Label duplicate UI data.")
    _write_extension_spec(unique_path, spec_id="mobile_fine_labels", prompt_text="Label mobile data.")

    io = StubIO(["y", "2", str(first_path), "y", "2", str(duplicate_id_path), "2", str(unique_path), "n"])
    paths = _ask_extension_spec_paths(io.input, io.output)

    assert paths == [str(first_path), str(unique_path)]
    joined = "\n".join(str(item) for item in io.outputs)
    assert "duplicate extension id" in joined.lower() or "重复" in joined


def test_ask_extension_spec_paths_prints_final_next_steps(tmp_path):
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _write_extension_spec(first_path, spec_id="ui_fine_labels")
    _write_extension_spec(second_path, spec_id="mobile_fine_labels", prompt_text="Label mobile data.")

    io = StubIO(["y", "2", str(first_path), "y", "2", str(second_path), "n"])
    paths = _ask_extension_spec_paths(io.input, io.output)

    assert paths == [str(first_path), str(second_path)]
    joined = "\n".join(str(item) for item in io.outputs)
    assert "2" in joined
    assert "include-extensions" in joined or "dashboard" in joined.lower()


def test_ask_extension_spec_paths_add_another_defaults_to_no(tmp_path):
    valid_path = tmp_path / "valid.yaml"
    _write_extension_spec(valid_path)

    io = StubIO(["y", "2", str(valid_path), ""])
    paths = _ask_extension_spec_paths(io.input, io.output)

    assert paths == [str(valid_path)]


def test_ask_extension_spec_paths_reprompts_until_valid_spec(tmp_path):
    invalid_path = tmp_path / "missing.yaml"
    valid_path = tmp_path / "valid.yaml"
    _write_extension_spec(valid_path)

    io = StubIO(["y", "2", str(invalid_path), "2", str(valid_path), "n"])
    paths = _ask_extension_spec_paths(io.input, io.output)

    assert paths == [str(valid_path)]
    joined = "\n".join(str(item) for item in io.outputs)
    assert "does not exist" in joined or "不存在" in joined


def test_ask_extension_spec_paths_can_select_multiple_specs_from_directory(tmp_path):
    spec_dir = tmp_path / "extensions"
    spec_dir.mkdir()
    first_path = spec_dir / "01-ui.yaml"
    second_path = spec_dir / "02-mobile.yaml"
    _write_extension_spec(first_path, spec_id="ui_fine_labels")
    _write_extension_spec(second_path, spec_id="mobile_fine_labels", prompt_text="Label mobile data.")

    io = StubIO(["y", "2", str(spec_dir), "1,2", "n"])
    paths = _ask_extension_spec_paths(io.input, io.output)

    assert paths == [str(first_path.resolve()), str(second_path.resolve())]
    joined = "\n".join(str(item) for item in io.outputs).lower()
    assert "directory" in joined or "目录" in joined
    assert "extra" in joined or "额外" in joined


def test_ask_extension_spec_paths_default_folder_option_uses_default_directory(monkeypatch, tmp_path):
    default_dir = tmp_path / "extensions"
    default_dir.mkdir()
    spec_path = default_dir / "ui-analysis.example.yaml"
    _write_extension_spec(spec_path, spec_id="ui_web_analysis")
    monkeypatch.setattr("sft_label.launcher._default_extension_specs_dir", lambda: default_dir)

    io = StubIO(["y", "1", "1", "n"])
    paths = _ask_extension_spec_paths(io.input, io.output)

    assert paths == [str(spec_path.resolve())]
    joined = "\n".join(str(item) for item in io.outputs).lower()
    assert "extensions" in joined


def test_export_review_prompt_flow_can_include_extension_columns():
    io = StubIO(["16", "labeled.json", "review.csv", "", "", "y", ""])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "export-review",
        "--input",
        "labeled.json",
        "--output",
        "review.csv",
        "--include-extensions",
    ]


def test_cmd_start_dry_run_shows_label_extension_flags(monkeypatch, capsys):
    parser = build_parser()
    plan = LaunchPlan(
        argv=[
            "run",
            "--input",
            "data.json",
            "--label-extension",
            "extensions/ui.yaml",
            "--label-extension",
            "extensions/mobile.yaml",
        ]
    )
    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: plan, raising=False)
    args = parser.parse_args(["start", "--dry-run"])

    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert out.count("--label-extension") >= 2
    assert "extensions/ui.yaml" in out
    assert "extensions/mobile.yaml" in out


def test_cmd_start_dry_run_shows_rollout_preset_summary(monkeypatch, capsys):
    parser = build_parser()
    plan = LaunchPlan(
        argv=[
            "run",
            "--input",
            "data.json",
            "--rollout-preset",
            "planner_hybrid",
        ]
    )
    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: plan, raising=False)
    args = parser.parse_args(["start", "--dry-run"])

    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert "planner_hybrid" in out
    assert "rollout" in out.lower() or "预设" in out
