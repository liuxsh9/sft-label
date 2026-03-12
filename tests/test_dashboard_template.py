from __future__ import annotations

from sft_label.tools.dashboard_template import render_dashboard_html


def test_render_dashboard_html_adds_count_bars_for_pass1_tag_distributions() -> None:
    payload = {
        "title": "Dashboard",
        "root_id": "global",
        "default_scope_id": "global",
        "initially_expanded": ["global"],
        "scopes": {
            "global": {
                "id": "global",
                "label": "demo.jsonl",
                "kind": "file",
                "path": "demo.jsonl",
                "parent_id": None,
                "children": [],
                "descendant_files": ["demo.jsonl"],
                "summary": {"pass1_total": 3, "scored_total": 0, "mean_value": None, "file_count": 1},
                "pass1": {
                    "total": 3,
                    "overview": {"success_rate": 1.0, "total_tokens": 42, "arbitrated_rate": 0.0},
                    "distributions": {
                        "language": {"python": 12, "go": 4},
                    },
                    "confidence_stats": {},
                    "coverage": {},
                    "cross_matrix": {"rows": [], "cols": [], "data": {}},
                },
            }
        },
    }

    html = render_dashboard_html(payload)

    assert "rank-tag-wrap" in html
    assert "rank-tag-bar" in html
    assert 'tagBarMode: "relative"' in html
    assert 'id="scope-toolbar-extra"' in html
    assert "Tag Bars" in html
    assert "当前 section 内全部 tag 的计数做对数缩放" in html
    assert '["hidden", "隐藏", "Hide tag bars"]' in html
    assert '["relative", "归一化", "Scale by current panel max"]' in html
    assert '["global-log", "全局对数", "Scale by section-wide log max"]' in html
    assert 'barMetricKey: "count"' in html
    assert 'barMetricLabel: "Count"' in html
    assert "globalBarMax: globalTagCountMax" in html
    assert '"language": {"python": 12, "go": 4}' in html


def test_render_dashboard_html_adds_count_bars_for_pass2_tag_rankings() -> None:
    payload = {
        "title": "Dashboard",
        "root_id": "global",
        "default_scope_id": "global",
        "initially_expanded": ["global"],
        "scopes": {
            "global": {
                "id": "global",
                "label": "demo.jsonl",
                "kind": "file",
                "path": "demo.jsonl",
                "parent_id": None,
                "children": [],
                "descendant_files": ["demo.jsonl"],
                "summary": {"pass1_total": 0, "scored_total": 2, "mean_value": 7.5, "file_count": 1},
                "pass2": {
                    "overview": {"total_scored": 2, "total_failed": 0, "mean_value": 7.5},
                    "histograms": {},
                    "score_distributions": {},
                    "value_by_tag": {
                        "task": {
                            "debugging": {"mean": 8.2, "n": 7},
                            "feature-implementation": {"mean": 8.8, "n": 2},
                        }
                    },
                    "selection_by_tag": {},
                    "thinking_mode_stats": {},
                    "flag_counts": {},
                    "flag_value_impact": {},
                    "coverage_at_thresholds": {},
                    "weights_used": {},
                    "per_file_summary": [],
                },
            }
        },
    }

    html = render_dashboard_html(payload)

    assert "debugging" in html
    assert "feature-implementation" in html
    assert 'barMetricKey: "n"' in html
    assert "hexToRgba" in html
    assert "Math.log1p" in html
    assert '"n": 7' in html
