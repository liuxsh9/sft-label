from __future__ import annotations

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
