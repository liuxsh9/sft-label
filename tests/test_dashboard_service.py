from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path
from urllib.request import urlopen

import pytest

from sft_label.dashboard_service import (
    DashboardPortConflictError,
    _detect_port_conflict,
    _http_reachable,
    dashboard_service_status,
    DashboardServiceConfig,
    DashboardServiceStore,
    default_dashboard_service_config_path,
    find_published_run,
    init_dashboard_service,
    list_published_runs,
    load_dashboard_service_store,
    publish_run_dashboards,
    restart_dashboard_service,
    save_dashboard_service_store,
    set_default_dashboard_service,
    start_dashboard_service,
    stop_dashboard_service,
    sync_dashboard_service_runtime_assets,
    update_dashboard_service_port,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_dashboard_bundle(root: Path, filename: str, title: str = "Demo") -> Path:
    dashboards_dir = root / "meta_label_data" / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    html_path = dashboards_dir / filename
    data_dir = dashboards_dir / f"{html_path.stem}.data"
    data_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        """<!DOCTYPE html>
<html>
<head>
  <title>Demo</title>
  <link rel=\"stylesheet\" href=\"_dashboard_static/v1/dashboard.css\">
</head>
<body>
<script id=\"dashboard-bootstrap\" type=\"application/json\">{\"staticBaseUrl\": \"_dashboard_static/v1\", \"manifestUrl\": \"demo.data/manifest.json\"}</script>
<script src=\"_dashboard_static/v1/dashboard.js\"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    (data_dir / "manifest.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    return html_path


def _write_scoring_summary(
    run_dir: Path,
    *,
    conversation_status: str,
    dashboard_status: str,
) -> Path:
    payload = {
        "postprocess": {
            "conversation_scores": {"status": conversation_status},
            "dashboard": {"status": dashboard_status},
        }
    }
    path = run_dir / "summary_stats_scoring.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dashboard_service_config_roundtrip(tmp_path, monkeypatch):
    config_path = tmp_path / "dashboard_services.json"
    monkeypatch.setenv("SFT_LABEL_DASHBOARD_SERVICE_CONFIG", str(config_path))

    assert default_dashboard_service_config_path() == config_path

    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=8765,
        service_type="pm2",
        public_base_url="https://dash.example.com/base",
        pm2_name="sft-label-dashboard-prod",
        config_path=config_path,
    )
    assert service.name == "default"

    store = load_dashboard_service_store(config_path)
    assert store.default_service == "default"
    assert store.services["default"].web_root == str((tmp_path / "web").resolve())
    assert store.services["default"].service_type == "pm2"
    assert store.services["default"].public_base_url == "https://dash.example.com/base"
    assert store.services["default"].pm2_name == "sft-label-dashboard-prod"

    service_copy = DashboardServiceConfig.from_dict(service.to_dict())
    assert service_copy == service

    store.services["backup"] = DashboardServiceConfig(
        name="backup",
        web_root=str((tmp_path / "backup-web").resolve()),
        host="0.0.0.0",
        port=9000,
        service_type="builtin",
    )
    save_dashboard_service_store(store, config_path)

    reloaded = load_dashboard_service_store(config_path)
    assert reloaded == DashboardServiceStore.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    assert sorted(reloaded.services) == ["backup", "default"]


def test_load_dashboard_service_store_treats_empty_file_as_empty_store(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    config_path.write_text("", encoding="utf-8")

    store = load_dashboard_service_store(config_path)

    assert store.default_service is None
    assert store.services == {}


def test_set_default_dashboard_service_updates_store(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    init_dashboard_service(name="a", web_root=tmp_path / "a", config_path=config_path)
    init_dashboard_service(name="b", web_root=tmp_path / "b", config_path=config_path, set_default=False)

    set_default_dashboard_service("b", config_path=config_path)
    store = load_dashboard_service_store(config_path)

    assert store.default_service == "b"


def test_http_reachable_uses_loopback_probe_for_wildcard_host(monkeypatch):
    seen = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(url, timeout=0.5):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("sft_label.dashboard_service.urlopen", _fake_urlopen)

    service = DashboardServiceConfig(
        name="default",
        web_root="/tmp/demo",
        host="0.0.0.0",
        port=8765,
    )

    assert _http_reachable(service) is True
    assert seen["url"] == "http://127.0.0.1:8765"


@pytest.mark.parametrize("command", [start_dashboard_service, restart_dashboard_service])
def test_dashboard_service_lifecycle_helpers(tmp_path, command):
    port = _free_port()
    config_path = tmp_path / "dashboard_services.json"
    web_root = tmp_path / "web"
    (web_root / "index.html").parent.mkdir(parents=True, exist_ok=True)
    (web_root / "index.html").write_text("ok", encoding="utf-8")

    service = init_dashboard_service(
        name="default",
        web_root=web_root,
        host="127.0.0.1",
        port=port,
        config_path=config_path,
    )

    status = command(service)
    assert status["state"] == "running"
    assert status["reachable"] is True
    body = urlopen(service.base_url(), timeout=2).read().decode("utf-8")
    assert "ok" in body

    stopped = stop_dashboard_service(service)
    assert stopped["state"] == "stopped"


def test_start_dashboard_service_raises_port_conflict_for_foreign_listener(tmp_path):
    port = _free_port()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    try:
        service = DashboardServiceConfig(
            name="default",
            web_root=str(tmp_path / "web"),
            host="127.0.0.1",
            port=port,
        )

        with pytest.raises(DashboardPortConflictError) as exc:
            start_dashboard_service(service)

        assert exc.value.port == port
        assert exc.value.owner_pid == os.getpid()
        assert exc.value.owner_command
    finally:
        listener.close()


def test_update_dashboard_service_port_rewrites_only_direct_host_port_urls(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    direct = init_dashboard_service(
        name="direct",
        web_root=tmp_path / "direct-web",
        host="0.0.0.0",
        port=8765,
        public_base_url="http://192.168.1.25:8765",
        config_path=config_path,
    )
    custom = init_dashboard_service(
        name="custom",
        web_root=tmp_path / "custom-web",
        host="0.0.0.0",
        port=8765,
        public_base_url="https://dash.example.com/base",
        config_path=config_path,
        set_default=False,
    )

    updated_direct = update_dashboard_service_port(direct, 9000, config_path=config_path)
    updated_custom = update_dashboard_service_port(custom, 9100, config_path=config_path)

    assert updated_direct.port == 9000
    assert updated_direct.public_base_url == "http://192.168.1.25:9000"
    assert updated_custom.port == 9100
    assert updated_custom.public_base_url == "https://dash.example.com/base"


def test_detect_port_conflict_ignores_non_overlapping_specific_host_listener(monkeypatch, tmp_path):
    service = DashboardServiceConfig(
        name="default",
        web_root=str(tmp_path / "web"),
        host="127.0.0.1",
        port=8765,
    )
    monkeypatch.setattr(
        "sft_label.dashboard_service._listening_processes",
        lambda port: [{"pid": 555, "command": "python -m http.server", "name": "TCP 192.168.1.25:8765 (LISTEN)"}],
    )
    monkeypatch.setattr("sft_label.dashboard_service._expected_service_pids", lambda service: set())

    assert _detect_port_conflict(service) is None


def test_publish_run_dashboards_syncs_shared_assets_and_registry(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    port = _free_port()
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=port,
        public_base_url="https://dash.example.com",
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    label_html = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")
    score_html = _write_dashboard_bundle(run_dir, "dashboard_scoring_demo.html")
    _write_scoring_summary(run_dir, conversation_status="completed", dashboard_status="completed")

    assets_dir = sync_dashboard_service_runtime_assets(service)
    assert (assets_dir / "dashboard.js").exists()
    assert (assets_dir / "dashboard.css").exists()

    published = publish_run_dashboards(service, run_dir, config_path=config_path)
    run_id = published["run_id"]
    published_root = Path(service.web_root) / "runs" / run_id

    assert (published_root / label_html.name).exists()
    assert (published_root / score_html.name).exists()
    assert (published_root / f"{label_html.stem}.data" / "manifest.json").exists()
    assert (published_root / f"{score_html.stem}.data" / "manifest.json").exists()
    assert (Path(service.web_root) / "assets" / "v1" / "dashboard.js").exists()
    assert published["dashboards"]["labeling"]["url"] == f"https://dash.example.com/runs/{run_id}/{label_html.name}"
    assert published["dashboards"]["scoring"]["url"] == f"https://dash.example.com/runs/{run_id}/{score_html.name}"

    html = (published_root / label_html.name).read_text(encoding="utf-8")
    assert "../../assets/v1/dashboard.css" in html
    assert "_dashboard_static/v1" not in html

    store = load_dashboard_service_store(config_path)
    runs = list_published_runs(store.services["default"])
    assert [item["run_id"] for item in runs] == [run_id]
    found = find_published_run(store.services["default"], run_id=run_id)
    assert found is not None
    assert found["source_run_dir"] == str(run_dir.resolve())


def test_publish_registry_can_reuse_run_id_by_source_path(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "same_name_run"
    _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")
    first = publish_run_dashboards(service, run_dir, config_path=config_path)
    second = publish_run_dashboards(service, run_dir, config_path=config_path)
    assert second["run_id"] == first["run_id"]

    other_root = tmp_path / "nested" / "same_name_run"
    _write_dashboard_bundle(other_root, "dashboard_labeling_demo.html")
    third = publish_run_dashboards(service, other_root, config_path=config_path)
    assert third["run_id"] != first["run_id"]

    store = load_dashboard_service_store(config_path)
    assert len(list_published_runs(store.services["default"])) == 2


def test_publish_prefers_global_scoring_dashboard_for_primary_url(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    global_html = _write_dashboard_bundle(run_dir, "dashboard_scoring_demo.html")
    _write_dashboard_bundle(run_dir, "dashboard_scoring_code__part1.html")
    _write_dashboard_bundle(run_dir, "dashboard_scoring_demo_sidebar_b.html")
    _write_scoring_summary(run_dir, conversation_status="completed", dashboard_status="completed")

    published = publish_run_dashboards(service, run_dir, config_path=config_path)

    assert published["dashboards"]["scoring"]["filename"] == global_html.name


def test_publish_run_dashboards_keeps_previous_publish_when_bundle_copy_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    label_html = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html", title="old")
    first = publish_run_dashboards(service, run_dir, config_path=config_path)
    published_root = Path(first["published_dir"])
    manifest_path = published_root / f"{label_html.stem}.data" / "manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["title"] == "old"

    _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html", title="new")

    def _boom(src_html, dest_root):
        raise OSError("copy boom")

    monkeypatch.setattr("sft_label.dashboard_service._copy_dashboard_bundle", _boom)

    with pytest.raises(OSError, match="copy boom"):
        publish_run_dashboards(service, run_dir, config_path=config_path)

    assert (published_root / label_html.name).exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["title"] == "old"


@pytest.mark.parametrize(
    ("conversation_status", "dashboard_status", "expected"),
    [
        ("deferred", "completed", "conversation_scores"),
        ("completed", "failed", "dashboard"),
    ],
)
def test_publish_run_dashboards_rejects_incomplete_scoring_postprocess_state(
    tmp_path,
    conversation_status,
    dashboard_status,
    expected,
):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")
    _write_dashboard_bundle(run_dir, "dashboard_scoring_demo.html")
    _write_scoring_summary(
        run_dir,
        conversation_status=conversation_status,
        dashboard_status=dashboard_status,
    )

    with pytest.raises(ValueError, match=expected):
        publish_run_dashboards(service, run_dir, config_path=config_path)


def test_publish_run_dashboards_allows_labeling_only_when_pass2_not_applicable(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "label_only_run"
    label_html = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")

    published = publish_run_dashboards(service, run_dir, config_path=config_path)

    assert "labeling" in published["dashboards"]
    assert "scoring" not in published["dashboards"]
    assert (Path(published["published_dir"]) / label_html.name).exists()


def test_publish_run_dashboards_fails_closed_when_publish_lock_conflicts(monkeypatch, tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    label_html = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html", title="old")
    first = publish_run_dashboards(service, run_dir, config_path=config_path)
    published_root = Path(first["published_dir"])
    manifest_path = published_root / f"{label_html.stem}.data" / "manifest.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))

    _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html", title="new")

    def _lock_conflict(*args, **kwargs):
        raise RuntimeError("publish lock busy")

    monkeypatch.setattr(
        "sft_label.dashboard_service._acquire_publish_lock",
        _lock_conflict,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="publish lock busy"):
        publish_run_dashboards(service, run_dir, config_path=config_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == before
    store = load_dashboard_service_store(config_path)
    found = find_published_run(store.services["default"], run_id=first["run_id"])
    assert found is not None
    assert found["published_at"] == first["published_at"]


def test_publish_run_dashboards_rejects_missing_companion_data_dir(tmp_path):
    config_path = tmp_path / "dashboard_services.json"
    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        host="127.0.0.1",
        port=_free_port(),
        config_path=config_path,
    )

    run_dir = tmp_path / "demo_run"
    html_path = _write_dashboard_bundle(run_dir, "dashboard_labeling_demo.html")
    data_dir = html_path.with_name(f"{html_path.stem}.data")
    assert data_dir.exists()
    shutil.rmtree(data_dir)

    with pytest.raises(ValueError, match="missing companion data directory"):
        publish_run_dashboards(service, run_dir, config_path=config_path)

    store = load_dashboard_service_store(config_path)
    assert list_published_runs(store.services["default"]) == []
    runs_root = Path(service.web_root) / "runs"
    assert not runs_root.exists() or list(runs_root.iterdir()) == []


def test_pm2_lifecycle_helpers_use_pm2_commands(monkeypatch, tmp_path):
    calls = []
    state = {"status": "stopped"}

    class _Completed:
        def __init__(self, stdout=""):
            self.stdout = stdout
            self.returncode = 0

    def _fake_run(cmd, check=False, capture_output=False, text=False):
        calls.append(cmd)
        if cmd and cmd[0] == "lsof":
            return _Completed(stdout="")
        if cmd[:2] == ["ps", "-p"]:
            return _Completed(stdout="")
        if cmd[:2] == ["pm2", "jlist"]:
            if state["status"] == "running":
                return _Completed(stdout=json.dumps([{
                    "name": "sft-label-dashboard-prod",
                    "pid": 321,
                    "pm2_env": {"status": "online"},
                }]))
            return _Completed(stdout="[]")
        if cmd[:2] == ["pm2", "start"]:
            state["status"] = "running"
            return _Completed(stdout="started")
        if cmd[:2] == ["pm2", "restart"]:
            state["status"] = "running"
            return _Completed(stdout="restarted")
        if cmd[:2] == ["pm2", "stop"]:
            state["status"] = "stopped"
            return _Completed(stdout="stopped")
        if cmd[:2] == ["pm2", "save"]:
            return _Completed(stdout="saved")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("sft_label.dashboard_service.subprocess.run", _fake_run)
    monkeypatch.setattr("sft_label.dashboard_service._http_reachable", lambda service, timeout=0.5: True)

    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        service_type="pm2",
        pm2_name="sft-label-dashboard-prod",
        config_path=tmp_path / "dashboard_services.json",
    )

    status = start_dashboard_service(service)
    assert status["state"] == "running"
    assert any(cmd[:2] == ["pm2", "start"] for cmd in calls)
    assert any(cmd[:2] == ["pm2", "save"] for cmd in calls)

    ecosystem_path = Path(service.web_root) / ".sft-label" / "pm2.config.json"
    assert ecosystem_path.exists()

    calls.clear()
    status = restart_dashboard_service(service)
    assert status["state"] == "running"
    assert any(cmd[:2] == ["pm2", "restart"] for cmd in calls)

    calls.clear()
    status = stop_dashboard_service(service)
    assert status["state"] == "stopped"
    assert any(cmd[:2] == ["pm2", "stop"] for cmd in calls)


def test_pm2_online_status_is_running_even_if_probe_fails(monkeypatch, tmp_path):
    class _Completed:
        def __init__(self, stdout=""):
            self.stdout = stdout
            self.returncode = 0

    def _fake_run(cmd, check=False, capture_output=False, text=False):
        if cmd[:2] == ["pm2", "jlist"]:
            return _Completed(stdout=json.dumps([{
                "name": "sft-label-dashboard-prod",
                "pid": 654,
                "pm2_env": {"status": "online"},
            }]))
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("sft_label.dashboard_service.subprocess.run", _fake_run)
    monkeypatch.setattr("sft_label.dashboard_service._http_reachable", lambda service, timeout=0.5: False)

    service = init_dashboard_service(
        name="default",
        web_root=tmp_path / "web",
        service_type="pm2",
        pm2_name="sft-label-dashboard-prod",
        config_path=tmp_path / "dashboard_services.json",
    )

    status = dashboard_service_status(service)

    assert status["state"] == "running"
    assert status["reachable"] is False
