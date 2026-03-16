"""Shared HTML template helpers for interactive labeling/scoring dashboards."""

from __future__ import annotations

import json
from pathlib import Path

from sft_label.artifacts import DASHBOARD_RUNTIME_DIRNAME, DASHBOARD_RUNTIME_VERSION


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<link rel="stylesheet" href="__CSS_URL__">
</head>
<body>
<div class="shell">
  <div class="hero">
    <div class="hero-top">
      <div>
        <h1 id="hero-title"></h1>
        <div class="hero-sub" id="hero-subtitle"></div>
      </div>
      <div class="hero-stats" id="hero-stats"></div>
    </div>
  </div>
  <div class="layout">
    <aside class="sidebar panel" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-copy">
          <h2 id="sidebar-title">Scope Navigator</h2>
          <div class="sidebar-sub" id="sidebar-sub">Global, folders, and files</div>
        </div>
        <button class="sidebar-toggle" id="sidebar-toggle" type="button" aria-expanded="true" title="Collapse Navigator">&lt;</button>
      </div>
      <div class="sidebar-body" id="sidebar-body">
        <input id="scope-search" class="search" type="search" placeholder="Search folders or files">
        <div class="filter-row">
          <button class="filter-btn active" data-filter="all">All</button>
          <button class="filter-btn" data-filter="dir">Folders</button>
          <button class="filter-btn" data-filter="file">Files</button>
        </div>
        <div class="tree" id="tree"></div>
      </div>
    </aside>
    <main class="main">
      <section class="panel">
        <div class="scope-header">
          <div class="scope-title">
            <div class="breadcrumbs" id="breadcrumbs"></div>
            <h2 id="scope-title"></h2>
            <div class="scope-path" id="scope-path"></div>
          </div>
          <div class="scope-actions">
            <div class="scope-toolbar-extra" id="scope-toolbar-extra"></div>
            <button class="action-btn" id="go-parent">Up One Level</button>
            <button class="action-btn" id="go-global">Back To Global</button>
          </div>
        </div>
      </section>
      <section class="panel" id="scope-content"></section>
    </main>
  </div>
</div>
<script id="dashboard-bootstrap" type="application/json">__BOOTSTRAP__</script>
<script src="__JS_URL__"></script>
</body>
</html>"""


def _runtime_asset_contents(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text(encoding="utf-8")


def ensure_dashboard_runtime_assets(output_path: Path, static_base_url: str | None = None) -> str:
    """Ensure runtime assets exist locally unless an external static URL is configured.

    Returns the base URL/path that HTML should reference.
    """

    if static_base_url:
        return static_base_url.rstrip("/")

    output_path = Path(output_path)
    runtime_dir = output_path.parent / DASHBOARD_RUNTIME_DIRNAME / DASHBOARD_RUNTIME_VERSION
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dashboard.js", "dashboard.css"):
        target = runtime_dir / name
        target.write_text(_runtime_asset_contents(name), encoding="utf-8")
    return f"{DASHBOARD_RUNTIME_DIRNAME}/{DASHBOARD_RUNTIME_VERSION}"


def render_dashboard_html(payload: dict) -> str:
    """Render a lightweight bootstrap dashboard page."""

    bootstrap = payload.get("bootstrap") or {}
    static_base_url = str(bootstrap.get("staticBaseUrl") or "").rstrip("/")
    css_url = f"{static_base_url}/dashboard.css" if static_base_url else "dashboard.css"
    js_url = f"{static_base_url}/dashboard.js" if static_base_url else "dashboard.js"
    return (
        HTML_TEMPLATE
        .replace("__TITLE__", payload.get("title", "Interactive Dashboard"))
        .replace("__CSS_URL__", css_url)
        .replace("__JS_URL__", js_url)
        .replace("__BOOTSTRAP__", json.dumps(bootstrap, ensure_ascii=False))
    )
