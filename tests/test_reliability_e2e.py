from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

import sft_label.dashboard_service as dashboard_service_module
from sft_label.dashboard_service import (
    find_published_run,
    init_dashboard_service,
    list_published_runs,
    load_dashboard_service_store,
    publish_run_dashboards,
)
from sft_label.launcher import DEFAULT_CONCURRENCY, build_launch_plan


class _StubIO:
    def __init__(self, answers: list[str]):
        self._answers = iter(answers)
        self.outputs: list[str] = []

    def input(self, _prompt: str) -> str:
        return next(self._answers)

    def output(self, text: str = "") -> None:
        self.outputs.append(text)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_service(tmp_path: Path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )
    return service, config_path


def _write_dashboard_bundle(root: Path, filename: str, *, title: str = "Demo") -> Path:
    dashboards_dir = root / "meta_label_data" / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)

    html_path = dashboards_dir / filename
    data_dir = dashboards_dir / f"{html_path.stem}.data"
    data_dir.mkdir(parents=True, exist_ok=True)

    html_path.write_text(
        f"""<!DOCTYPE html>
<html>
<head>
  <title>{title}</title>
  <link rel=\"stylesheet\" href=\"_dashboard_static/v1/dashboard.css\">
</head>
<body>
<script id=\"dashboard-bootstrap\" type=\"application/json\">{{\"staticBaseUrl\": \"_dashboard_static/v1\", \"manifestUrl\": \"{html_path.stem}.data/manifest.json\"}}</script>
<script src=\"_dashboard_static/v1/dashboard.js\"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    (data_dir / "manifest.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    return html_path


def _write_scoring_summary(run_dir: Path, *, conversation_status: str, dashboard_status: str) -> None:
    payload = {
        "postprocess": {
            "conversation_scores": {"status": conversation_status},
            "dashboard": {"status": dashboard_status},
        }
    }
    (run_dir / "summary_stats_scoring.json").write_text(json.dumps(payload), encoding="utf-8")


def test_e2e_smart_resume_detects_hidden_pass2_artifacts(tmp_path):
    run_dir = tmp_path / "interrupted_run"
    artifacts = run_dir / "meta_label_data" / "files" / "demo"
    artifacts.mkdir(parents=True)
    (artifacts / "labeled.json").write_text("[]", encoding="utf-8")
    (artifacts / ".scored.jsonl.next").write_text("", encoding="utf-8")

    io = _StubIO([
        "2",  # workflow: smart resume
        str(run_dir),
        "",   # extra flags
    ])

    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.workflow_key == "smart-resume"
    assert plan.argv == [
        "score",
        "--concurrency",
        str(DEFAULT_CONCURRENCY),
        "--input",
        str(run_dir),
        "--resume",
    ]


def test_e2e_deferred_scoring_run_refuses_publish(tmp_path):
    service, config_path = _init_service(tmp_path)
    run_dir = tmp_path / "deferred_scoring_run"
    _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")
    _write_dashboard_bundle(run_dir, "dashboard_scoring_demo.html")
    _write_scoring_summary(run_dir, conversation_status="deferred", dashboard_status="completed")

    with pytest.raises(ValueError, match="conversation_scores"):
        publish_run_dashboards(service, run_dir, config_path=config_path)

    store = load_dashboard_service_store(config_path)
    assert list_published_runs(store.services["default"]) == []


def test_e2e_labeling_only_run_remains_publishable(tmp_path):
    service, config_path = _init_service(tmp_path)
    run_dir = tmp_path / "labeling_only_run"
    label_html = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")

    published = publish_run_dashboards(service, run_dir, config_path=config_path)

    assert "labeling" in published["dashboards"]
    assert "scoring" not in published["dashboards"]
    assert (Path(published["published_dir"]) / label_html.name).exists()

    store = load_dashboard_service_store(config_path)
    assert len(list_published_runs(store.services["default"])) == 1


def test_e2e_republish_failure_keeps_existing_public_bundle(monkeypatch, tmp_path):
    service, config_path = _init_service(tmp_path)
    run_dir = tmp_path / "republish_run"
    label_html = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html", title="old")

    first = publish_run_dashboards(service, run_dir, config_path=config_path)
    manifest_path = Path(first["published_dir"]) / f"{label_html.stem}.data" / "manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["title"] == "old"

    _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html", title="new")

    real_swap = dashboard_service_module._swap_published_directory
    calls = {"count": 0}

    def _boom_on_assets_swap(staging_root: Path, published_root: Path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("assets swap boom")
        return real_swap(staging_root, published_root)

    monkeypatch.setattr("sft_label.dashboard_service._swap_published_directory", _boom_on_assets_swap)

    with pytest.raises(OSError, match="assets swap boom"):
        publish_run_dashboards(service, run_dir, config_path=config_path)

    assert calls["count"] >= 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["title"] == "old"

    store = load_dashboard_service_store(config_path)
    found = find_published_run(store.services["default"], run_id=first["run_id"])
    assert found is not None
    assert found["published_at"] == first["published_at"]
