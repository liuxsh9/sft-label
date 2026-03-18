"""Machine-level dashboard static service management and run publishing."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.error import URLError
from urllib.request import urlopen

from sft_label.artifacts import DASHBOARD_RUNTIME_VERSION

DASHBOARD_SERVICE_CONFIG_ENV = "SFT_LABEL_DASHBOARD_SERVICE_CONFIG"
DEFAULT_DASHBOARD_SERVICE_NAME = "default"
CONFIG_VERSION = 1
_SERVICE_META_DIRNAME = ".sft-label"
_ASSETS_DIRNAME = "assets"
_RUNS_DIRNAME = "runs"
_RUNTIME_ASSET_FILENAMES = ("dashboard.js", "dashboard.css")
_RUNTIME_SOURCE_DIR = Path(__file__).resolve().parent / "tools"
_RUNTIME_PATH_RE = re.compile(r"(?P<quote>['\"])_dashboard_static/[^'\"]+(?P=quote)")


@dataclass(eq=True)
class DashboardServiceConfig:
    name: str
    web_root: str
    host: str = "127.0.0.1"
    port: int = 8765
    service_type: str = "builtin"
    public_base_url: str | None = None
    pm2_name: str | None = None
    published_runs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardServiceConfig":
        return cls(
            name=str(data["name"]),
            web_root=str(Path(data["web_root"]).expanduser()),
            host=str(data.get("host") or "127.0.0.1"),
            port=int(data.get("port") or 8765),
            service_type=str(data.get("service_type") or "builtin"),
            public_base_url=(str(data["public_base_url"]).rstrip("/") if data.get("public_base_url") else None),
            pm2_name=(str(data["pm2_name"]) if data.get("pm2_name") else None),
            published_runs=list(data.get("published_runs") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "web_root": self.web_root,
            "host": self.host,
            "port": self.port,
            "service_type": self.service_type,
            "public_base_url": self.public_base_url,
            "pm2_name": self.pm2_name,
            "published_runs": list(self.published_runs),
        }

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def share_base_url(self) -> str:
        return (self.public_base_url or self.base_url()).rstrip("/")

    @property
    def web_root_path(self) -> Path:
        return Path(self.web_root).expanduser().resolve()

    @property
    def service_meta_dir(self) -> Path:
        return self.web_root_path / _SERVICE_META_DIRNAME

    @property
    def pid_file(self) -> Path:
        return self.service_meta_dir / "http.pid"

    @property
    def log_file(self) -> Path:
        return self.service_meta_dir / "http.log"

    @property
    def pm2_process_name(self) -> str:
        return self.pm2_name or f"sft-label-dashboard-{self.name}"

    @property
    def pm2_config_file(self) -> Path:
        return self.service_meta_dir / "pm2.config.json"


class DashboardPortConflictError(ValueError):
    """Raised when a dashboard service cannot bind because another process owns the port."""

    def __init__(
        self,
        *,
        service_name: str,
        host: str,
        port: int,
        owner_pid: int | None,
        owner_command: str | None,
        owned_by_service: bool = False,
    ) -> None:
        self.service_name = service_name
        self.host = host
        self.port = int(port)
        self.owner_pid = owner_pid
        self.owner_command = owner_command
        self.owned_by_service = owned_by_service
        owner_bits: list[str] = []
        if owner_pid:
            owner_bits.append(f"pid={owner_pid}")
        if owner_command:
            owner_bits.append(owner_command)
        owner_desc = " | ".join(owner_bits) if owner_bits else "unknown process"
        super().__init__(
            f"Dashboard service '{service_name}' cannot bind {host}:{port}; port is already in use by {owner_desc}."
        )


@dataclass(eq=True)
class DashboardServiceStore:
    version: int = CONFIG_VERSION
    default_service: str | None = None
    services: dict[str, DashboardServiceConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardServiceStore":
        services_raw = data.get("services") or {}
        services = {
            name: DashboardServiceConfig.from_dict(cfg)
            for name, cfg in services_raw.items()
        }
        return cls(
            version=int(data.get("version") or CONFIG_VERSION),
            default_service=data.get("default_service"),
            services=services,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "default_service": self.default_service,
            "services": {name: svc.to_dict() for name, svc in self.services.items()},
        }


def default_dashboard_service_config_path() -> Path:
    override = os.getenv(DASHBOARD_SERVICE_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "sft-label" / "dashboard_services.json"


def load_dashboard_service_store(config_path: str | Path | None = None) -> DashboardServiceStore:
    path = Path(config_path) if config_path is not None else default_dashboard_service_config_path()
    if not path.exists():
        return DashboardServiceStore()
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return DashboardServiceStore()
    data = json.loads(raw)
    return DashboardServiceStore.from_dict(data)


def save_dashboard_service_store(
    store: DashboardServiceStore,
    config_path: str | Path | None = None,
) -> Path:
    path = Path(config_path) if config_path is not None else default_dashboard_service_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def init_dashboard_service(
    *,
    name: str = DEFAULT_DASHBOARD_SERVICE_NAME,
    web_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    service_type: str = "builtin",
    public_base_url: str | None = None,
    pm2_name: str | None = None,
    config_path: str | Path | None = None,
    set_default: bool = True,
) -> DashboardServiceConfig:
    store = load_dashboard_service_store(config_path)
    existing = store.services.get(name)
    service = DashboardServiceConfig(
        name=name,
        web_root=str(Path(web_root).expanduser().resolve()),
        host=host,
        port=int(port),
        service_type=service_type,
        public_base_url=public_base_url.rstrip("/") if public_base_url else None,
        pm2_name=pm2_name,
        published_runs=existing.published_runs if existing else [],
    )
    service.web_root_path.mkdir(parents=True, exist_ok=True)
    service.service_meta_dir.mkdir(parents=True, exist_ok=True)
    store.services[name] = service
    if set_default or not store.default_service:
        store.default_service = name
    save_dashboard_service_store(store, config_path)
    return service


def set_default_dashboard_service(
    name: str,
    *,
    config_path: str | Path | None = None,
) -> DashboardServiceStore:
    store = load_dashboard_service_store(config_path)
    if name not in store.services:
        raise ValueError(f"Dashboard service not found: {name}")
    store.default_service = name
    save_dashboard_service_store(store, config_path)
    return store


def _rewrite_direct_public_base_url_port(public_base_url: str | None, old_port: int, new_port: int) -> str | None:
    if not public_base_url:
        return public_base_url
    try:
        parsed = urlsplit(public_base_url)
    except ValueError:
        return public_base_url
    if parsed.scheme not in {"http", "https"}:
        return public_base_url
    try:
        parsed_port = parsed.port
    except ValueError:
        return public_base_url
    if parsed_port != int(old_port):
        return public_base_url
    if (parsed.path or "") not in {"", "/"} or parsed.query or parsed.fragment:
        return public_base_url
    hostname = parsed.hostname
    if not hostname:
        return public_base_url
    if ":" in hostname and not hostname.startswith("["):
        host_part = f"[{hostname}]"
    else:
        host_part = hostname
    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    netloc = f"{auth}{host_part}:{int(new_port)}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def update_dashboard_service_port(
    service: DashboardServiceConfig,
    port: int,
    *,
    config_path: str | Path | None = None,
) -> DashboardServiceConfig:
    store = load_dashboard_service_store(config_path)
    existing = store.services.get(service.name)
    if existing is None:
        raise ValueError(f"Dashboard service not found: {service.name}")
    old_port = int(existing.port)
    existing.port = int(port)
    existing.public_base_url = _rewrite_direct_public_base_url_port(
        existing.public_base_url,
        old_port,
        int(port),
    )
    save_dashboard_service_store(store, config_path)
    return existing


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(service: DashboardServiceConfig) -> int | None:
    try:
        raw = service.pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _http_reachable(service: DashboardServiceConfig, timeout: float = 0.5) -> bool:
    probe_host = service.host
    if probe_host == "0.0.0.0":
        probe_host = "127.0.0.1"
    elif probe_host in {"::", "[::]"}:
        probe_host = "::1"
    probe_url = f"http://{probe_host}:{service.port}"
    try:
        with urlopen(probe_url, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200)) < 500
    except (URLError, OSError, TimeoutError):
        return False


def _command_for_pid(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    command = (result.stdout or "").strip()
    return command or None


def _listening_processes(port: int) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode not in (0, 1):
        return []
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "command": _command_for_pid(pid) or parts[0],
                "name": " ".join(parts[8:]) if len(parts) > 8 else "",
            }
        )
    return rows


def _listener_host(listener_name: str) -> str | None:
    text = (listener_name or "").strip()
    if not text:
        return None
    match = re.search(r"TCP\s+(.+):\d+\s+\(LISTEN\)", text)
    if not match:
        return None
    host = match.group(1).strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _host_patterns_overlap(service_host: str, listener_host: str | None) -> bool:
    normalized_service = (service_host or "").strip()
    normalized_listener = (listener_host or "").strip()
    wildcard_hosts = {"*", "0.0.0.0", "::"}
    if not normalized_service or normalized_service in wildcard_hosts:
        return True
    if not normalized_listener or normalized_listener in wildcard_hosts:
        return True
    return normalized_service == normalized_listener


def _expected_service_pids(service: DashboardServiceConfig) -> set[int]:
    pids: set[int] = set()
    builtin_pid = _read_pid(service)
    if builtin_pid:
        pids.add(int(builtin_pid))
    if service.service_type == "pm2":
        try:
            info = _pm2_process_info(service)
        except ValueError:
            info = None
        if info:
            pid = info.get("pid") or (info.get("monit") or {}).get("pid")
            if pid:
                pids.add(int(pid))
    return pids


def _detect_port_conflict(service: DashboardServiceConfig) -> DashboardPortConflictError | None:
    listeners = _listening_processes(service.port)
    if not listeners:
        return None
    expected_pids = _expected_service_pids(service)
    for listener in listeners:
        pid = int(listener.get("pid") or 0) or None
        if pid is not None and pid in expected_pids:
            continue
        if not _host_patterns_overlap(service.host, _listener_host(str(listener.get("name") or ""))):
            continue
        return DashboardPortConflictError(
            service_name=service.name,
            host=service.host,
            port=service.port,
            owner_pid=pid,
            owner_command=listener.get("command"),
            owned_by_service=False,
        )
    return None


def _run_pm2(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise ValueError("pm2 is not installed or not found in PATH") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        detail = stderr or stdout or str(e)
        raise ValueError(f"pm2 command failed: {' '.join(args)}: {detail}") from e


def _write_pm2_config(service: DashboardServiceConfig) -> Path:
    service.service_meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "apps": [
            {
                "name": service.pm2_process_name,
                "script": sys.executable,
                "args": [
                    "-m",
                    "http.server",
                    str(service.port),
                    "--bind",
                    service.host,
                    "--directory",
                    str(service.web_root_path),
                ],
                "interpreter": "none",
                "cwd": str(service.web_root_path),
                "autorestart": True,
            }
        ]
    }
    service.pm2_config_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return service.pm2_config_file


def _pm2_process_info(service: DashboardServiceConfig) -> dict[str, Any] | None:
    result = _run_pm2(["pm2", "jlist"])
    try:
        data = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as e:
        raise ValueError("Failed to parse `pm2 jlist` output") from e
    for item in data:
        if str(item.get("name")) == service.pm2_process_name:
            return item
    return None


def dashboard_service_status(service: DashboardServiceConfig) -> dict[str, Any]:
    if service.service_type == "pm2":
        try:
            info = _pm2_process_info(service)
        except ValueError:
            return {
                "name": service.name,
                "state": "unavailable",
                "pid": None,
                "reachable": False,
                "url": service.base_url(),
                "public_url": service.share_base_url(),
                "web_root": str(service.web_root_path),
            }
        if not info:
            return {
                "name": service.name,
                "state": "stopped",
                "pid": None,
                "reachable": False,
                "url": service.base_url(),
                "public_url": service.share_base_url(),
                "web_root": str(service.web_root_path),
            }
        pm2_env = info.get("pm2_env") or {}
        raw_status = str(pm2_env.get("status") or "unknown")
        pid = info.get("pid") or (info.get("monit") or {}).get("pid")
        reachable = _http_reachable(service) if raw_status == "online" else False
        state = "running" if raw_status == "online" else (
            "starting" if raw_status == "launching" else raw_status
        )
        return {
            "name": service.name,
            "state": state,
            "pid": pid,
            "reachable": reachable,
            "url": service.base_url(),
            "public_url": service.share_base_url(),
            "web_root": str(service.web_root_path),
        }

    pid = _read_pid(service)
    alive = _pid_alive(pid)
    reachable = _http_reachable(service) if alive else False
    if alive and reachable:
        state = "running"
    elif alive:
        state = "starting"
    else:
        state = "stopped"
    if pid is not None and not alive and service.pid_file.exists():
        try:
            service.pid_file.unlink()
        except OSError:
            pass
    return {
        "name": service.name,
        "state": state,
        "pid": pid,
        "reachable": reachable,
        "url": service.base_url(),
        "public_url": service.share_base_url(),
        "web_root": str(service.web_root_path),
    }


def start_dashboard_service(service: DashboardServiceConfig, wait_timeout: float = 3.0) -> dict[str, Any]:
    conflict = _detect_port_conflict(service)
    if conflict is not None:
        raise conflict
    if service.service_type == "pm2":
        service.web_root_path.mkdir(parents=True, exist_ok=True)
        config_file = _write_pm2_config(service)
        if _pm2_process_info(service) is None:
            _run_pm2(["pm2", "start", str(config_file)])
        else:
            _run_pm2(["pm2", "restart", str(config_file)])
        _run_pm2(["pm2", "save"])
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            status = dashboard_service_status(service)
            if status["state"] == "running":
                return status
            time.sleep(0.1)
        return dashboard_service_status(service)

    status = dashboard_service_status(service)
    if status["state"] == "running":
        return status

    service.web_root_path.mkdir(parents=True, exist_ok=True)
    service.service_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(service.log_file, "ab") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(service.port),
                "--bind",
                service.host,
                "--directory",
                str(service.web_root_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    service.pid_file.write_text(str(proc.pid), encoding="utf-8")

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        status = dashboard_service_status(service)
        if status["state"] == "running":
            return status
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    return dashboard_service_status(service)


def stop_dashboard_service(service: DashboardServiceConfig, wait_timeout: float = 3.0) -> dict[str, Any]:
    if service.service_type == "pm2":
        if _pm2_process_info(service) is not None:
            _run_pm2(["pm2", "stop", service.pm2_process_name])
            _run_pm2(["pm2", "save"])
        return dashboard_service_status(service)

    pid = _read_pid(service)
    if not _pid_alive(pid):
        try:
            service.pid_file.unlink()
        except OSError:
            pass
        return dashboard_service_status(service)

    assert pid is not None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)

    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        deadline = time.time() + 1.0
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.05)

    try:
        service.pid_file.unlink()
    except OSError:
        pass
    return dashboard_service_status(service)


def restart_dashboard_service(service: DashboardServiceConfig, wait_timeout: float = 3.0) -> dict[str, Any]:
    conflict = _detect_port_conflict(service)
    if conflict is not None:
        raise conflict
    if service.service_type == "pm2":
        service.web_root_path.mkdir(parents=True, exist_ok=True)
        config_file = _write_pm2_config(service)
        if _pm2_process_info(service) is None:
            _run_pm2(["pm2", "start", str(config_file)])
        else:
            _run_pm2(["pm2", "restart", str(config_file)])
        _run_pm2(["pm2", "save"])
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            status = dashboard_service_status(service)
            if status["state"] == "running":
                return status
            time.sleep(0.1)
        return dashboard_service_status(service)

    stop_dashboard_service(service, wait_timeout=wait_timeout)
    return start_dashboard_service(service, wait_timeout=wait_timeout)


def sync_dashboard_service_runtime_assets(service: DashboardServiceConfig) -> Path:
    assets_dir = service.web_root_path / _ASSETS_DIRNAME / DASHBOARD_RUNTIME_VERSION
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name in _RUNTIME_ASSET_FILENAMES:
        shutil.copyfile(_RUNTIME_SOURCE_DIR / name, assets_dir / name)
    return assets_dir


def _resolve_dashboards_dir(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    candidates = [
        run_dir / "meta_label_data" / "dashboards",
        run_dir / "dashboards",
        run_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.html")):
            return candidate
    raise FileNotFoundError(f"No dashboard HTML files found under {run_dir}")


def _sanitize_run_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    clean = clean.strip("-._")
    return clean or "run"


def list_published_runs(service: DashboardServiceConfig) -> list[dict[str, Any]]:
    return list(service.published_runs)


def find_published_run(
    service: DashboardServiceConfig,
    *,
    run_id: str | None = None,
    source_run_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    source_str = str(Path(source_run_dir).expanduser().resolve()) if source_run_dir is not None else None
    for item in service.published_runs:
        if run_id is not None and item.get("run_id") == run_id:
            return item
        if source_str is not None and item.get("source_run_dir") == source_str:
            return item
    return None


def _allocate_run_id(service: DashboardServiceConfig, run_dir: Path) -> str:
    existing = find_published_run(service, source_run_dir=run_dir)
    if existing:
        return str(existing["run_id"])

    base = _sanitize_run_id(run_dir.name)
    used = {str(item.get("run_id")) for item in service.published_runs}
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


def _rewrite_dashboard_html_for_shared_assets(html_path: Path) -> None:
    text = html_path.read_text(encoding="utf-8")
    asset_rel = "../../assets/v1"
    text = text.replace("_dashboard_static/v1", asset_rel)
    text = text.replace('href="dashboard.css"', f'href="{asset_rel}/dashboard.css"')
    text = text.replace('src="dashboard.js"', f'src="{asset_rel}/dashboard.js"')
    html_path.write_text(text, encoding="utf-8")


def _classify_dashboard_key(filename: str) -> str:
    lower = filename.lower()
    if "scoring" in lower or "value" in lower:
        return "scoring"
    if "labeling" in lower:
        return "labeling"
    return Path(filename).stem


def _dashboard_priority(filename: str, key: str) -> tuple[int, int, str]:
    lower = filename.lower()
    has_nested_scope = "__" in filename
    is_sidebar_variant = "sidebar" in lower
    is_meta = "meta_label_data" in lower or lower.endswith("files.html")
    if key == "scoring":
        if not has_nested_scope and not is_sidebar_variant and not is_meta:
            return (0, len(filename), filename)
        if not has_nested_scope and not is_sidebar_variant:
            return (1, len(filename), filename)
        if not has_nested_scope:
            return (2, len(filename), filename)
        return (3, len(filename), filename)
    if key == "labeling":
        if not has_nested_scope and not is_sidebar_variant:
            return (0, len(filename), filename)
        return (1, len(filename), filename)
    return (10, len(filename), filename)


def _copy_dashboard_bundle(src_html: Path, dest_root: Path) -> Path:
    dest_html = dest_root / src_html.name
    shutil.copyfile(src_html, dest_html)
    data_dir = src_html.with_name(f"{src_html.stem}.data")
    if data_dir.is_dir():
        shutil.copytree(data_dir, dest_root / data_dir.name)
    _rewrite_dashboard_html_for_shared_assets(dest_html)
    return dest_html


def publish_run_dashboards(
    service: DashboardServiceConfig,
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    store = load_dashboard_service_store(config_path)
    configured = store.services.get(service.name)
    if configured is not None:
        service = configured

    sync_dashboard_service_runtime_assets(service)
    run_dir = Path(run_dir).expanduser().resolve()
    dashboards_dir = _resolve_dashboards_dir(run_dir)
    html_files = sorted(dashboards_dir.glob("*.html"))
    if not html_files:
        raise FileNotFoundError(f"No dashboard HTML files found under {dashboards_dir}")

    run_id = _allocate_run_id(service, run_dir)
    published_root = service.web_root_path / _RUNS_DIRNAME / run_id
    if published_root.exists():
        shutil.rmtree(published_root)
    published_root.mkdir(parents=True, exist_ok=True)

    dashboards: dict[str, dict[str, Any]] = {}
    for src_html in html_files:
        dest_html = _copy_dashboard_bundle(src_html, published_root)
        key = _classify_dashboard_key(src_html.name)
        candidate = {
            "filename": dest_html.name,
            "path": str(dest_html),
            "url": f"{service.share_base_url()}/{_RUNS_DIRNAME}/{run_id}/{dest_html.name}",
        }
        existing = dashboards.get(key)
        if existing is None or _dashboard_priority(candidate["filename"], key) < _dashboard_priority(existing["filename"], key):
            dashboards[key] = candidate

    published_record = {
        "run_id": run_id,
        "source_run_dir": str(run_dir),
        "dashboards_dir": str(dashboards_dir),
        "published_dir": str(published_root),
        "published_at": datetime.now().isoformat(),
        "dashboards": dashboards,
    }

    service.published_runs = [
        item for item in service.published_runs
        if item.get("run_id") != run_id and item.get("source_run_dir") != str(run_dir)
    ]
    service.published_runs.append(published_record)
    store.services[service.name] = service
    if not store.default_service:
        store.default_service = service.name
    save_dashboard_service_store(store, config_path)
    return published_record
