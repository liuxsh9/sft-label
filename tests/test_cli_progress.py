import asyncio
import json
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sft_label.cli import (
    _CombinedLLMProgressTracker,
    _SemanticProgressPrinter,
    _estimate_end_to_end_llm_calls,
    build_parser,
    cmd_export_review,
    cmd_refresh_rarity,
    cmd_run,
    cmd_score,
    cmd_start,
)
from sft_label.config import PipelineConfig
from sft_label.progress_heartbeat import HeartbeatFrames, run_with_heartbeat


def _write_input(path: Path, n: int):
    data = []
    for i in range(n):
        data.append(
            {
                "id": f"c-{i}",
                "conversations": [
                    {"from": "human", "value": "Write python code."},
                    {"from": "gpt", "value": "```python\nprint(1)\n```"},
                ],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_estimate_end_to_end_llm_calls_single_file(tmp_path):
    input_file = tmp_path / "input.json"
    _write_input(input_file, 3)

    args = SimpleNamespace(
        input=str(input_file),
        resume=None,
        limit=0,
        shuffle=False,
        no_arbitration=False,
        mode="refresh",
        migrate_from=None,
    )
    config = PipelineConfig(sample_max_retries=1)
    plan = _estimate_end_to_end_llm_calls(args, config)

    assert plan is not None
    assert plan["pass1_labeled_samples"] == 3
    assert plan["pass2_samples"] == 3
    assert plan["pass1_est_calls"] == 7
    assert plan["pass2_est_calls"] == 3
    assert plan["total_est_calls"] == 10

def test_estimate_end_to_end_llm_calls_accounts_for_extensions(tmp_path):
    input_file = tmp_path / "input.json"
    _write_input(input_file, 3)

    args = SimpleNamespace(
        input=str(input_file),
        resume=None,
        limit=0,
        shuffle=False,
        no_arbitration=False,
        mode="refresh",
        migrate_from=None,
    )
    config = PipelineConfig(sample_max_retries=1)
    config.extension_spec_paths = ["ui.yaml", "mobile.yaml"]
    plan = _estimate_end_to_end_llm_calls(args, config)

    assert plan is not None
    assert plan["pass1_extension_calls"] == plan["pass1_labeled_samples"] * len(config.extension_spec_paths)
    assert plan["total_est_calls"] == (
        plan["pass1_est_calls"] + plan["pass1_extension_calls"] + plan["pass2_est_calls"]
    )


def test_combined_llm_progress_tracker_updates():
    tracker = _CombinedLLMProgressTracker(100)
    info = tracker.update(12, "pass1")
    assert "run 12/100" in info
    summary = tracker.summary_line()
    assert "p1=12" in summary
    assert "p2=0" in summary


def test_heartbeat_frames_cycle_from_one_to_six_and_reset():
    frames = HeartbeatFrames()
    observed = [frames.next_suffix() for _ in range(8)]
    assert observed == [".", "..", "...", "....", ".....", "......", ".", ".."]


def test_run_with_heartbeat_emits_stage_and_completes(capsys):
    result = run_with_heartbeat(
        "Estimating workload",
        lambda: time.sleep(0.08) or "done",
        interval=0.01,
    )

    assert result == "done"
    out = capsys.readouterr().out
    assert "Estimating workload." in out
    assert "Estimating workload......" in out
    assert out.endswith("\n")


def test_shared_pipeline_progress_factory_uses_stable_column_layout():
    from sft_label.progress_display import create_pipeline_progress

    progress = create_pipeline_progress()

    assert [type(column) for column in progress.columns] == [
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TextColumn,
        TimeElapsedColumn,
        TextColumn,
        TimeRemainingColumn,
        TextColumn,
    ]


def test_pass1_and_pass2_progress_factories_share_the_same_layout():
    from sft_label.pipeline import create_progress
    from sft_label.scoring import _create_progress

    pass1_progress = create_progress()
    pass2_progress = _create_progress()

    assert [type(column) for column in pass1_progress.columns] == [
        type(column) for column in pass2_progress.columns
    ]


def test_pipeline_directory_mode_creates_files_pass1_and_llm_tasks(monkeypatch, tmp_path):
    from sft_label.pipeline import run

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    _write_input(input_dir / "sample.json", 2)

    progress_tasks = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_task(self, description, **kwargs):
            progress_tasks.append(description)
            return len(progress_tasks)

        def update(self, *args, **kwargs):
            return None

        @property
        def console(self):
            return self

        def print(self, *args, **kwargs):
            return None

    async def _fake_directory_pipeline(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "sft_label.pipeline.estimate_directory_workload",
        lambda *args, **kwargs: type(
            "Estimate",
            (),
            {
                "files_planned": 1,
                "total_samples": 2,
                "total_labeled_samples": 2,
                "total_inherited_samples": 0,
                "initial_estimated_llm_calls": 4,
                "scan_elapsed_seconds": 0.0,
            },
        )(),
    )
    monkeypatch.setattr("sft_label.pipeline.create_progress", lambda: FakeProgress())
    monkeypatch.setattr("sft_label.pipeline.run_directory_pipeline", _fake_directory_pipeline)
    monkeypatch.setattr("sft_label.pipeline._write_global_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sft_label.pipeline._summary_path_for_run",
        lambda run_dir, layout=None: tmp_path / "summary.json",
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: FakeClient())
    (tmp_path / "summary.json").write_text(json.dumps({"run_dir": str(tmp_path / "run")}), encoding="utf-8")

    result = asyncio.run(
        run(
            input_path=str(input_dir),
            output=str(tmp_path / "run"),
            config=PipelineConfig(enable_adaptive_runtime=False),
        )
    )

    assert result["run_dir"] == str(tmp_path / "run")
    assert progress_tasks == ["Files", "Pass 1", "LLM"]


def test_cmd_run_semantic_cluster_failure_exits_cleanly(monkeypatch, capsys):
    async def _fake_run(*args, **kwargs):
        return {"run_dir": "/tmp/fake-run"}

    fake_pipeline = SimpleNamespace(run=_fake_run)

    def _fake_semantic(*args, **kwargs):
        raise ValueError("semantic failed")

    fake_semantic = SimpleNamespace(
        run_semantic_clustering=_fake_semantic,
        format_semantic_summary=lambda stats: "unused",
    )
    monkeypatch.setitem(sys.modules, "sft_label.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "sft_label.semantic_clustering", fake_semantic)

    parser = build_parser()
    args = parser.parse_args(["run", "--input", "input.json", "--semantic-cluster"])
    with pytest.raises(SystemExit) as exc:
        cmd_run(args)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Error: semantic failed" in out


def test_cmd_run_recompute_rejects_score(monkeypatch):
    parser = build_parser()
    args = parser.parse_args(["run", "--input", "input.json", "--mode", "recompute", "--score"])
    with pytest.raises(SystemExit) as exc:
        cmd_run(args)
    assert exc.value.code == 1


def test_cmd_run_wraps_estimation_and_reuses_precomputed_estimates(monkeypatch):
    captured = {}

    async def _fake_run(*args, **kwargs):
        captured["pipeline_workload"] = kwargs.get("precomputed_workload_estimate")
        return {"run_dir": "/tmp/fake-run"}

    async def _fake_score(*args, **kwargs):
        captured["scoring_workload"] = kwargs.get("precomputed_workload_estimate")
        return {"run_dir": "/tmp/fake-run"}

    estimate_plan = {
        "pass1_labeled_samples": 4,
        "pass2_samples": 7,
        "pass1_est_calls": 9,
        "pass2_est_calls": 8,
        "total_est_calls": 17,
        "pass1_workload_estimate": object(),
        "pass2_workload_estimate": object(),
    }

    heartbeat_labels = []

    def _fake_heartbeat(message, fn, **kwargs):
        heartbeat_labels.append(message)
        return fn()

    monkeypatch.setattr("sft_label.cli._estimate_end_to_end_llm_calls", lambda *a, **k: estimate_plan)
    monkeypatch.setattr("sft_label.cli.run_with_heartbeat", _fake_heartbeat)
    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))
    monkeypatch.setitem(sys.modules, "sft_label.scoring", SimpleNamespace(run_scoring=_fake_score))

    parser = build_parser()
    args = parser.parse_args(["run", "--input", "input.json", "--score"])
    result = cmd_run(args)

    assert result["run_dir"] == "/tmp/fake-run"
    assert captured["pipeline_workload"] is estimate_plan["pass1_workload_estimate"]
    assert captured["scoring_workload"] is estimate_plan["pass2_workload_estimate"]
    assert any("Estimating workload" in label for label in heartbeat_labels)


def test_cmd_run_loads_label_extension_specs_before_pipeline(monkeypatch, tmp_path):
    spec_path = tmp_path / "ui-extension.yaml"
    spec_path.write_text(
        """
id: ui_fine_labels
spec_version: v1
prompt: |
  Label UI data.
schema:
  component_type:
    type: multi_enum
    options: [form, modal]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    async def _fake_run(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return {"run_dir": "/tmp/fake-run"}

    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))

    parser = build_parser()
    args = parser.parse_args(["run", "--input", "input.json", "--label-extension", str(spec_path)])
    result = cmd_run(args)

    assert result["run_dir"] == "/tmp/fake-run"
    assert captured["config"].extension_spec_paths == [str(spec_path)]
    assert captured["config"].extension_specs is not None
    assert [spec.id for spec in captured["config"].extension_specs] == ["ui_fine_labels"]


def test_cmd_run_prints_extension_preflight_and_followup(monkeypatch, tmp_path, capsys):
    spec_path = tmp_path / "ui-extension.yaml"
    spec_path.write_text(
        """
id: ui_fine_labels
spec_version: v1
display_name: UI Fine Labels
description: Fine-grained UI tags.
prompt: |
  Label UI data with frontend review hints.
schema:
  component_type:
    type: multi_enum
    options: [form, modal, table]
trigger:
  domain_any_of: [web-frontend]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    async def _fake_run(*args, **kwargs):
        return {
            "run_dir": "/tmp/fake-run",
            "extension_stats": {
                "specs": {
                    "ui_fine_labels": {
                        "total": 10,
                        "matched": 7,
                        "status_counts": {"success": 7, "skipped": 3},
                        "unmapped_counts": {"component_type:card": 2},
                    }
                }
            },
        }

    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))

    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--input",
        "input.json",
        "--prompt-mode",
        "compact",
        "--label-extension",
        str(spec_path),
    ])
    result = cmd_run(args)

    assert result["run_dir"] == "/tmp/fake-run"
    out = capsys.readouterr().out
    assert "ui_fine_labels" in out
    assert "2000" in out
    assert "Inspect" in out or "dashboard" in out.lower()
    assert "--include-extensions" in out


def test_cmd_run_extension_followup_warns_when_match_rate_is_zero(monkeypatch, tmp_path, capsys):
    spec_path = tmp_path / "ui-extension.yaml"
    spec_path.write_text(
        """
id: ui_fine_labels
spec_version: v1
prompt: |
  Label UI data.
schema:
  component_type:
    type: multi_enum
    options: [form, modal]
trigger:
  domain_any_of: [web-frontend]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    async def _fake_run(*args, **kwargs):
        return {
            "run_dir": "/tmp/fake-run",
            "extension_stats": {
                "specs": {
                    "ui_fine_labels": {
                        "total": 12,
                        "matched": 0,
                        "status_counts": {"skipped": 12},
                        "unmapped_counts": {},
                    }
                }
            },
        }

    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))

    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--input",
        "input.json",
        "--label-extension",
        str(spec_path),
    ])
    cmd_run(args)

    out = capsys.readouterr().out.lower()
    assert "matched 0 / 12" in out
    assert "trigger" in out


def test_cmd_run_extension_followup_warns_when_everything_matches(monkeypatch, tmp_path, capsys):
    spec_path = tmp_path / "ui-extension.yaml"
    spec_path.write_text(
        """
id: ui_fine_labels
spec_version: v1
prompt: |
  Label UI data.
schema:
  component_type:
    type: multi_enum
    options: [form, modal]
trigger:
  domain_any_of: [web-frontend]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    async def _fake_run(*args, **kwargs):
        return {
            "run_dir": "/tmp/fake-run",
            "extension_stats": {
                "specs": {
                    "ui_fine_labels": {
                        "total": 8,
                        "matched": 8,
                        "status_counts": {"success": 8},
                        "unmapped_counts": {},
                    }
                }
            },
        }

    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))

    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--input",
        "input.json",
        "--label-extension",
        str(spec_path),
    ])
    cmd_run(args)

    out = capsys.readouterr().out.lower()
    assert "matched 8 / 8" in out
    assert "broad" in out or "过宽" in out


def test_run_parser_help_mentions_repeatable_extension_guidance():
    parser = build_parser()
    subparsers_action = next(item for item in parser._actions if item.dest == "command")
    run_parser = subparsers_action.choices["run"]
    action = next(item for item in run_parser._actions if getattr(item, "dest", None) == "label_extension")
    assert "repeatable" in action.help.lower()
    assert "unique" in action.help.lower()


def test_run_and_score_parser_default_prompt_mode_is_compact():
    parser = build_parser()
    args_run = parser.parse_args(["run", "--input", "input.json"])
    args_score = parser.parse_args(["score", "--input", "labeled.json"])

    assert args_run.prompt_mode == "compact"
    assert args_score.prompt_mode == "compact"
    assert args_run.extension_rarity_mode is None
    assert args_score.extension_rarity_mode is None


def test_run_score_and_refresh_parser_accept_extension_rarity_modes():
    parser = build_parser()
    args_run = parser.parse_args([
        "run",
        "--input",
        "input.json",
        "--extension-rarity-mode",
        "preview",
    ])
    args_score = parser.parse_args([
        "score",
        "--input",
        "labeled.json",
        "--extension-rarity-mode",
        "bonus_only",
    ])
    args_refresh = parser.parse_args([
        "refresh-rarity",
        "--input",
        "scored.json",
        "--extension-rarity-mode",
        "preview",
    ])

    assert args_run.extension_rarity_mode == "preview"
    assert args_score.extension_rarity_mode == "bonus_only"
    assert args_refresh.extension_rarity_mode == "preview"


def test_cmd_run_forwards_default_extension_rarity_mode_off(monkeypatch):
    captured = {}

    async def _fake_run(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return {"run_dir": "/tmp/fake-run"}

    monkeypatch.setitem(sys.modules, "sft_label.pipeline", SimpleNamespace(run=_fake_run))

    parser = build_parser()
    args = parser.parse_args(["run", "--input", "input.json"])
    result = cmd_run(args)

    assert result["run_dir"] == "/tmp/fake-run"
    assert captured["config"].extension_rarity_mode == "off"


def test_cmd_score_forwards_explicit_extension_rarity_mode(monkeypatch):
    captured = {}

    async def _fake_run_scoring(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return {"run_dir": "/tmp/fake-run"}

    monkeypatch.setitem(sys.modules, "sft_label.scoring", SimpleNamespace(run_scoring=_fake_run_scoring))

    parser = build_parser()
    args = parser.parse_args([
        "score",
        "--input",
        "labeled.json",
        "--extension-rarity-mode",
        "bonus_only",
    ])
    result = cmd_score(args)

    assert result["run_dir"] == "/tmp/fake-run"
    assert captured["config"].extension_rarity_mode == "bonus_only"


def test_apply_runtime_overrides_preserves_preconfigured_extension_rarity_mode_when_omitted():
    from sft_label.cli import _apply_runtime_overrides
    from sft_label.config import PipelineConfig

    config = PipelineConfig(extension_rarity_mode="preview")
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
    )

    _apply_runtime_overrides(config, args)

    assert config.extension_rarity_mode == "preview"


def test_cmd_refresh_rarity_forwards_explicit_extension_rarity_mode(monkeypatch):
    captured = {}

    def _fake_run_refresh_rarity(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return {"rarity_refreshed_samples": "3"}

    monkeypatch.setitem(
        sys.modules,
        "sft_label.tools.recompute",
        SimpleNamespace(run_refresh_rarity=_fake_run_refresh_rarity),
    )

    parser = build_parser()
    args = parser.parse_args([
        "refresh-rarity",
        "--input",
        "scored.json",
        "--extension-rarity-mode",
        "preview",
    ])
    cmd_refresh_rarity(args)

    assert captured["config"].extension_rarity_mode == "preview"


def test_semantic_progress_printer_shows_progress_without_spam(capsys):
    printer = _SemanticProgressPrinter()
    printer("start", "Semantic clustering started", None, None)
    printer("embed", "Embedding windows", 1, 10)
    printer("embed", "Embedding windows", 1, 10)  # duplicate percent, suppressed
    printer("embed", "Embedding windows", 5, 10)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any("[semantic:start]" in line for line in lines)
    embed_lines = [line for line in lines if "[semantic:embed]" in line]
    assert len(embed_lines) == 2



def test_dashboard_service_parser_supports_init_and_register_run():
    parser = build_parser()

    args = parser.parse_args([
        "dashboard-service",
        "init",
        "--name",
        "prod",
        "--web-root",
        "/srv/sft-label-dashboard",
        "--port",
        "9000",
    ])
    assert args.command == "dashboard-service"
    assert args.dashboard_service_action == "init"
    assert args.name == "prod"
    assert args.web_root == "/srv/sft-label-dashboard"
    assert args.port == 9000

    args = parser.parse_args([
        "dashboard-service",
        "set-default",
        "--name",
        "prod",
    ])
    assert args.dashboard_service_action == "set-default"
    assert args.name == "prod"

    args = parser.parse_args([
        "dashboard-service",
        "runs",
        "--name",
        "prod",
    ])
    assert args.dashboard_service_action == "runs"
    assert args.name == "prod"

    args = parser.parse_args([
        "dashboard-service",
        "register-run",
        "--name",
        "prod",
        "--run-dir",
        "/tmp/run-1",
    ])
    assert args.command == "dashboard-service"
    assert args.dashboard_service_action == "register-run"
    assert args.run_dir == "/tmp/run-1"

def test_run_parser_accepts_multiple_label_extensions():
    parser = build_parser()
    args = parser.parse_args(["run", "--label-extension", "ui.yaml", "--label-extension", "mobile.yaml"])
    assert args.command == "run"
    assert args.label_extension == ["ui.yaml", "mobile.yaml"]


def test_export_review_parser_accepts_include_extensions():
    parser = build_parser()
    args = parser.parse_args([
        "export-review",
        "--input",
        "run_dir",
        "--output",
        "review.csv",
        "--include-extensions",
    ])
    assert args.command == "export-review"
    assert args.include_extensions is True


def test_cmd_export_review_forwards_include_extensions(monkeypatch):
    captured = {}

    def _fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setitem(sys.modules, "sft_label.tools.export_review", SimpleNamespace(main=_fake_main))

    parser = build_parser()
    args = parser.parse_args([
        "export-review",
        "--input",
        "run_dir",
        "--output",
        "review.csv",
        "--include-extensions",
    ])
    cmd_export_review(args)

    assert "--include-extensions" in captured["argv"]


def test_dispatch_dashboard_service_register_run(monkeypatch, capsys, tmp_path):
    calls = {}

    def _fake_load(_config_path=None):
        from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
        service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
        return DashboardServiceStore(default_service="default", services={"default": service})

    def _fake_publish(service, run_dir, *, config_path=None):
        calls["service"] = service.name
        calls["run_dir"] = run_dir
        calls["config_path"] = config_path
        return {
            "run_id": "run-1",
            "dashboards": {
                "labeling": {"url": "http://127.0.0.1:8765/runs/run-1/dashboard_labeling.html"}
            },
        }

    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", _fake_load, raising=False)
    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)

    parser = build_parser()
    args = parser.parse_args(["dashboard-service", "register-run", "--run-dir", str(tmp_path / "run")])
    from sft_label.cli import dispatch_command
    result = dispatch_command(args, parser=parser)

    assert result["run_id"] == "run-1"
    assert calls["service"] == "default"
    assert calls["run_dir"] == str(tmp_path / "run")
    assert "dashboard_labeling.html" in capsys.readouterr().out


@pytest.mark.parametrize("blocked_status", ["deferred", "pending", "failed"])
def test_cmd_start_auto_publish_skips_when_pass2_postprocess_not_completed(
    monkeypatch,
    capsys,
    tmp_path,
    blocked_status,
):
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json", "--score"]),
    )

    answers = iter(["y", ""])  # auto-publish yes, execute yes
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(*args, **kwargs):
        publish_calls["count"] += 1
        return {"run_id": "demo", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dispatch_command",
        lambda args, parser: {
            "run_dir": str(run_dir),
            "score_summary": {
                "postprocess": {
                    "conversation_scores": {"status": blocked_status},
                    "dashboard": {"status": "completed"},
                }
            },
        },
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert publish_calls["count"] == 0
    assert blocked_status in out
    assert str(run_dir) in out
    assert "auto-publish" in out.lower()


def test_cmd_start_auto_publish_regenerate_dashboard_fail_closed_with_complete_postprocess_hint(
    monkeypatch,
    capsys,
    tmp_path,
):
    from sft_label.artifacts import PASS2_SUMMARY_STATS_FILE
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    run_dir = tmp_path / "demo run"
    run_dir.mkdir()
    (run_dir / PASS2_SUMMARY_STATS_FILE).write_text(
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

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["regenerate-dashboard", "--input", str(run_dir)]),
    )

    answers = iter(["y", ""])  # auto-publish yes, execute yes
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(*args, **kwargs):
        publish_calls["count"] += 1
        return {"run_id": "demo", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dispatch_command",
        lambda args, parser: {"command": "regenerate-dashboard", "run_dir": str(run_dir)},
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert publish_calls["count"] == 0
    assert "deferred" in out
    assert f"--input {shlex.quote(str(run_dir))}" in out


def test_cmd_start_auto_publish_regenerate_dashboard_pass1_allows_labeling_only_publish(
    monkeypatch,
    tmp_path,
):
    from sft_label.artifacts import PASS2_SUMMARY_STATS_FILE
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()
    (run_dir / PASS2_SUMMARY_STATS_FILE).write_text(
        json.dumps(
            {
                "postprocess": {
                    "conversation_scores": {"status": "deferred"},
                    "dashboard": {"status": "failed"},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["regenerate-dashboard", "--input", str(run_dir), "--pass", "1"]),
    )

    answers = iter(["y", ""])  # auto-publish yes, execute yes
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(*args, **kwargs):
        publish_calls["count"] += 1
        return {"run_id": "demo", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dispatch_command",
        lambda args, parser: {"command": "regenerate-dashboard", "run_dir": str(run_dir)},
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    assert publish_calls["count"] == 1


def test_cmd_start_auto_publish_regenerate_dashboard_inline_run_uses_meta_summary_postprocess(
    monkeypatch,
    capsys,
    tmp_path,
):
    from sft_label.artifacts import PASS2_SUMMARY_STATS_FILE
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    run_dir = tmp_path / "inline run"
    meta_dir = run_dir / "meta_label_data"
    meta_dir.mkdir(parents=True)
    (meta_dir / PASS2_SUMMARY_STATS_FILE).write_text(
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

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["regenerate-dashboard", "--input", str(run_dir)]),
    )

    answers = iter(["y", ""])  # auto-publish yes, execute yes
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(*args, **kwargs):
        publish_calls["count"] += 1
        return {"run_id": "demo", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dispatch_command",
        lambda args, parser: {"command": "regenerate-dashboard", "run_dir": str(run_dir)},
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert publish_calls["count"] == 0
    assert "deferred" in out
    assert f"--input {shlex.quote(str(run_dir))}" in out


def test_cmd_start_auto_publish_keeps_labeling_only_runs_publishable(monkeypatch, tmp_path):
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]),
    )

    answers = iter(["y", ""])  # auto-publish yes, execute yes
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(*args, **kwargs):
        publish_calls["count"] += 1
        return {"run_id": "demo", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dispatch_command",
        lambda args, parser: {"run_dir": str(run_dir)},
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    assert publish_calls["count"] == 1


@pytest.mark.parametrize(
    ("conversation_status", "dashboard_status"),
    [
        ("completed", "disabled"),
        ("disabled", "completed"),
    ],
)
def test_cmd_start_auto_publish_allows_scored_runs_when_pass2_postprocess_is_safe(
    monkeypatch,
    tmp_path,
    conversation_status,
    dashboard_status,
):
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
    from sft_label.launcher import LaunchPlan

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "sft_label.launcher.build_launch_plan",
        lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json", "--score"]),
    )

    answers = iter(["y", ""])  # auto-publish yes, execute yes
    monkeypatch.setattr(
        "sft_label.launcher.interactive_input",
        lambda prompt: next(answers),
        raising=False,
    )

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dashboard_service_status",
        lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()},
        raising=False,
    )

    publish_calls = {"count": 0}

    def _fake_publish(*args, **kwargs):
        publish_calls["count"] += 1
        return {"run_id": "demo", "dashboards": {}}

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr(
        "sft_label.cli.dispatch_command",
        lambda args, parser: {
            "run_dir": str(run_dir),
            "score_summary": {
                "postprocess": {
                    "conversation_scores": {"status": conversation_status},
                    "dashboard": {"status": dashboard_status},
                }
            },
        },
        raising=False,
    )

    args = parser.parse_args(["start", "--en"])
    cmd_start(args, parser)

    assert publish_calls["count"] == 1


def test_dispatch_dashboard_service_set_default_and_runs(monkeypatch, capsys, tmp_path):
    calls = {"default": None}

    def _fake_load(_config_path=None):
        from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore
        service = DashboardServiceConfig(
            name="prod",
            web_root=str(tmp_path / "web"),
            host="127.0.0.1",
            port=8765,
            published_runs=[{"run_id": "run-a", "dashboards": {"labeling": {"url": "https://dash/runs/run-a/dashboard.html"}}}],
        )
        return DashboardServiceStore(default_service="prod", services={"prod": service})

    def _fake_set_default(name, config_path=None):
        calls["default"] = name

    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", _fake_load, raising=False)
    monkeypatch.setattr("sft_label.cli.set_default_dashboard_service", _fake_set_default, raising=False)

    parser = build_parser()
    from sft_label.cli import dispatch_command

    args = parser.parse_args(["dashboard-service", "set-default", "--name", "prod"])
    result = dispatch_command(args, parser=parser)
    assert result["default_service"] == "prod"
    assert calls["default"] == "prod"

    args = parser.parse_args(["dashboard-service", "runs", "--name", "prod"])
    result = dispatch_command(args, parser=parser)
    assert result[0]["run_id"] == "run-a"
    assert "run-a" in capsys.readouterr().out
