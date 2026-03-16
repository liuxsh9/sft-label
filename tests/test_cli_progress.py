import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft_label.cli import (
    _CombinedLLMProgressTracker,
    _SemanticProgressPrinter,
    _estimate_end_to_end_llm_calls,
    build_parser,
    cmd_run,
)
from sft_label.config import PipelineConfig


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


def test_combined_llm_progress_tracker_updates():
    tracker = _CombinedLLMProgressTracker(100)
    info = tracker.update(12, "pass1")
    assert "run 12/100" in info
    summary = tracker.summary_line()
    assert "p1=12" in summary
    assert "p2=0" in summary


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
