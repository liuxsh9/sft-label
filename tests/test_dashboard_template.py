from __future__ import annotations

from pathlib import Path

from sft_label.tools.dashboard_template import render_dashboard_html


def test_render_dashboard_html_emits_bootstrap_shell_for_pass1_dashboard() -> None:
    payload = {
        "title": "Dashboard",
        "bootstrap": {
            "dashboardType": "labeling",
            "manifestUrl": "dashboard_labeling.data/manifest.json",
            "staticBaseUrl": "/static/sft-label-dashboard/v1",
            "defaultScopeId": "global",
            "rootId": "global",
        },
    }

    html = render_dashboard_html(payload)

    assert '<script id="dashboard-bootstrap" type="application/json">' in html
    assert "dashboard_labeling.data/manifest.json" in html
    assert "/static/sft-label-dashboard/v1/dashboard.css" in html
    assert "/static/sft-label-dashboard/v1/dashboard.js" in html
    assert "const DATA =" not in html
    assert '"dashboardType": "labeling"' in html


def test_render_dashboard_html_emits_bootstrap_shell_for_pass2_dashboard() -> None:
    payload = {
        "title": "Dashboard",
        "bootstrap": {
            "dashboardType": "scoring",
            "manifestUrl": "dashboard_scoring.data/manifest.json",
            "staticBaseUrl": "/static/sft-label-dashboard/v1",
            "defaultScopeId": "global",
            "rootId": "global",
        },
    }

    html = render_dashboard_html(payload)

    assert "dashboard_scoring.data/manifest.json" in html
    assert "/static/sft-label-dashboard/v1/dashboard.js" in html
    assert 'id="scope-toolbar-extra"' in html
    assert "Scope Navigator" in html
    assert "const DATA =" not in html


def test_dashboard_runtime_initializes_explorer_cache_before_manifest_load() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert js.index("const EXPLORER_CACHE =") < js.index("const DATA = await loadDashboardManifest();")


def test_dashboard_runtime_uses_compact_sidebar_meta_and_color_only_kind_markers() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert 'tree-kind-dot' in js
    assert 'tree-main' in js
    assert 'tree-sub' in js
    assert 'N ${fmtInt' in js
    assert 'V ${fmt(' in js
    assert 'S ${fmt(' in js
    assert '${escapeHtml(scopeKindLabel(scope.kind))}</span>' not in js


def test_dashboard_styles_define_compact_sidebar_row_layout() -> None:
    css = Path("src/sft_label/tools/dashboard.css").read_text(encoding="utf-8")

    assert '.tree-main {' in css
    assert '.tree-sub {' in css
    assert '.tree-kind-dot {' in css
    assert '.tree-meta-pill {' in css


def test_dashboard_runtime_uses_tighter_tree_indent_step() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert 'margin-left:${depth * 6}px' in js


def test_dashboard_runtime_has_bilingual_turn_kind_tags_and_english_intent_matrix_title() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert 'intent_difficulty: "Intent × Difficulty"' in js
    assert 'turn_kind_single: "1T"' in js
    assert 'turn_kind_multi: "MT"' in js
    assert 'turn_kind_mixed: "Mix"' in js
    assert 'turn_kind_single: "单"' in js
    assert 'turn_kind_multi: "多"' in js
    assert 'turn_kind_mixed: "混"' in js
    assert 'turnKindLabel(summary.turn_kind)' in js
