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
    assert 'id="hero-toolbar"' in html
    assert 'id="scope-toolbar-extra"' not in html
    assert "Scope Navigator" in html
    assert "const DATA =" not in html


def test_dashboard_runtime_renders_global_controls_into_hero_toolbar() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert 'document.getElementById("hero-toolbar").innerHTML = renderTagBarControlsInline();' in js
    assert 'document.getElementById("scope-toolbar-extra").innerHTML = renderTagBarControlsInline(scope);' not in js


def test_dashboard_styles_define_sticky_hero_toolbar_layout() -> None:
    css = Path("src/sft_label/tools/dashboard.css").read_text(encoding="utf-8")

    assert '.hero-toolbar {' in css
    assert '.hero-toolbar-group {' in css
    assert '.hero-toolbar .toolbar-inline-copy {' in css


def test_dashboard_runtime_toolbar_polish_uses_compact_group_markup() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert 'hero-toolbar-label' in js
    assert 'hero-toolbar-body' in js


def test_dashboard_styles_toolbar_polish_define_glass_group_layout() -> None:
    css = Path("src/sft_label/tools/dashboard.css").read_text(encoding="utf-8")

    assert '.hero-toolbar-label {' in css
    assert '.hero-toolbar-body {' in css
    assert 'backdrop-filter: blur(14px);' in css
    assert '.hero-toolbar .segmented-btn { padding: 6px 10px;' in css


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


def test_dashboard_runtime_file_ranking_includes_rarity_keep_rate_and_mean_turns() -> None:
    js = Path("src/sft_label/tools/dashboard.js").read_text(encoding="utf-8")

    assert 'keep_rates' in js
    assert 'mean_rarity' in js
    assert 'mean_turns' in js
    assert 'const KEEP_RATE_THRESHOLDS = ["4.0", "5.0", "6.0", "7.0"]' in js
    assert 'data-keep-rate-threshold' in js


def test_dashboard_styles_define_keep_rate_chip() -> None:
    css = Path("src/sft_label/tools/dashboard.css").read_text(encoding="utf-8")

    assert '.keep-rate-chip {' in css
    assert '.keep-rate-strong {' in css
