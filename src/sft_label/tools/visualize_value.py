"""
Value Scoring Dashboard Generator

Generates standalone HTML dashboards for Pass 2 value scoring results.
Includes: Value Overview Cards, Score Distributions, Sub-score Breakdown,
Value×Tag Cross-Analysis, Thinking Mode Analysis, Flag Analysis.

Global dashboard adds: File Ranking Table, Coverage Impact Analysis,
Data Selection Simulator.
"""

import json
from pathlib import Path


def load_value_run(run_dir, scored_file="scored.json", stats_file="stats_value.json"):
    """Load scored samples and value stats from a run directory."""
    samples = []
    if scored_file:
        scored_path = Path(run_dir) / scored_file
        if scored_path.exists():
            with open(scored_path, encoding="utf-8") as f:
                samples = json.load(f)

    stats = {}
    stats_path = Path(run_dir) / stats_file
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)

    return samples, stats


def compute_value_viz_data(samples, stats):
    """Transform scored data + stats into dashboard JSON."""
    viz = {}

    # Overview
    total_scored = stats.get("total_scored", len([s for s in samples if s.get("value")]))
    total_failed = stats.get("total_failed", 0)
    dists = stats.get("score_distributions", {})

    viz["overview"] = {
        "total_scored": total_scored,
        "total_failed": total_failed,
        "mean_value": dists.get("value_score", {}).get("mean", 0),
        "mean_complexity": dists.get("complexity_overall", {}).get("mean", 0),
        "mean_quality": dists.get("quality_overall", {}).get("mean", 0),
        "median_rarity": dists.get("rarity_score", {}).get("p50", 0),
        "total_tokens": stats.get("total_tokens", 0),
    }

    # Selection thresholds
    viz["selection_thresholds"] = stats.get("selection_thresholds", {})

    # Score distributions (histogram bins 1-10)
    score_keys = ["value_score", "complexity_overall", "quality_overall",
                  "reasoning_overall", "rarity_score"]
    histograms = {}
    if samples:
        for key in score_keys:
            bins = [0] * 10  # bins for 1-10
            for s in samples:
                v = s.get("value", {})
                # Navigate to the value
                if key == "value_score":
                    val = v.get("value_score")
                elif key == "rarity_score":
                    val = (v.get("rarity") or {}).get("score")
                else:
                    dim, sub = key.rsplit("_", 1)
                    val = (v.get(dim) or {}).get(sub)
                if isinstance(val, (int, float)) and 1 <= val <= 10:
                    idx = min(int(val) - 1, 9)
                    bins[idx] += 1
            histograms[key] = bins
    viz["histograms"] = histograms

    # Score distribution stats
    viz["score_distributions"] = dists

    # Sub-score means
    viz["sub_score_means"] = stats.get("sub_score_means", {})

    # Value by tag (cross-analysis)
    viz["value_by_tag"] = stats.get("value_by_tag", {})

    # Thinking mode
    viz["thinking_mode_stats"] = stats.get("thinking_mode_stats", {})

    # Flags
    viz["flag_counts"] = stats.get("flag_counts", {})
    viz["flag_value_impact"] = stats.get("flag_value_impact", {})

    # Coverage at thresholds
    viz["coverage_at_thresholds"] = stats.get("coverage_at_thresholds", {})

    # Weights
    viz["weights_used"] = stats.get("weights_used", {})

    # Per-file summary (global dashboard)
    viz["per_file_summary"] = stats.get("per_file_summary", [])

    return viz


# ─────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────

VALUE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFT Value Scoring Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }
h1 { font-size: 1.6em; margin-bottom: 8px; color: #fff; }
h2 { font-size: 1.2em; margin: 24px 0 12px; color: #a0a8c0; border-bottom: 1px solid #2a2d3a; padding-bottom: 6px; }
h3 { font-size: 1em; margin: 12px 0 8px; color: #8890a8; }

.cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.card { background: #1a1d2e; border-radius: 8px; padding: 16px 20px; min-width: 140px; flex: 1; }
.card .label { font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 1px; }
.card .value { font-size: 1.8em; font-weight: 700; margin-top: 4px; }
.card .sub { font-size: 0.8em; color: #666; margin-top: 2px; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.panel { background: #1a1d2e; border-radius: 8px; padding: 16px; }

.bar-row { display: flex; align-items: center; margin: 3px 0; font-size: 0.8em; }
.bar-label { width: 120px; text-align: right; padding-right: 8px; color: #8890a8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 18px; background: #252836; border-radius: 3px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.bar-val { min-width: 50px; text-align: right; padding-left: 6px; color: #888; font-size: 0.85em; }

.hist-row { display: flex; align-items: flex-end; gap: 2px; height: 100px; margin-top: 8px; }
.hist-bar { flex: 1; background: #4a6cf7; border-radius: 2px 2px 0 0; min-height: 2px; position: relative; }
.hist-bar:hover { opacity: 0.8; }
.hist-labels { display: flex; gap: 2px; font-size: 0.7em; color: #666; }
.hist-labels span { flex: 1; text-align: center; }

table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
th { background: #252836; padding: 8px; text-align: left; cursor: pointer; user-select: none; }
th:hover { background: #2e3248; }
td { padding: 8px; border-top: 1px solid #252836; }
tr:hover td { background: #1e2133; }

.quad-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.quad { background: #252836; border-radius: 6px; padding: 12px; text-align: center; }
.quad .q-label { font-size: 0.75em; color: #666; }
.quad .q-value { font-size: 1.4em; font-weight: 700; margin-top: 4px; }

.c-green { color: #4caf50; } .c-blue { color: #4a6cf7; } .c-orange { color: #ff9800; } .c-red { color: #f44336; }
.c-purple { color: #9c27b0; } .c-cyan { color: #00bcd4; }

.tag { display: inline-block; background: #252836; border-radius: 4px; padding: 2px 6px; font-size: 0.75em; margin: 1px; }
</style>
</head>
<body>
<h1>SFT Value Scoring Dashboard</h1>
<div id="dashboard"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;

function pct(n, total) { return total > 0 ? (n / total * 100).toFixed(1) : '0.0'; }
function scoreColor(v) {
    if (v >= 8) return '#4caf50';
    if (v >= 6) return '#4a6cf7';
    if (v >= 4) return '#ff9800';
    return '#f44336';
}

function renderHistogram(containerId, bins, label, color) {
    const max = Math.max(...bins, 1);
    let html = `<h3>${label}</h3><div class="hist-row">`;
    for (let i = 0; i < bins.length; i++) {
        const h = Math.max((bins[i] / max) * 100, 2);
        html += `<div class="hist-bar" style="height:${h}%;background:${color}" title="${i+1}: ${bins[i]}"></div>`;
    }
    html += '</div><div class="hist-labels">';
    for (let i = 1; i <= 10; i++) html += `<span>${i}</span>`;
    html += '</div>';
    return html;
}

function renderBarChart(items, maxVal, color, limit) {
    if (!items || items.length === 0) return '<p style="color:#666">No data</p>';
    const top = items.slice(0, limit || 15);
    const m = maxVal || Math.max(...top.map(x => x[1]), 1);
    let html = '';
    for (const [label, value, extra] of top) {
        const w = (value / m * 100).toFixed(1);
        const extraStr = extra ? ` (n=${extra})` : '';
        html += `<div class="bar-row">
            <div class="bar-label" title="${label}">${label}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
            <div class="bar-val">${typeof value === 'number' ? value.toFixed(1) : value}${extraStr}</div>
        </div>`;
    }
    return html;
}

function render() {
    const d = DATA;
    const o = d.overview || {};
    const el = document.getElementById('dashboard');
    let html = '';

    // === Section 1: Overview Cards ===
    html += '<h2>Value Overview</h2><div class="cards">';
    html += `<div class="card"><div class="label">Scored</div><div class="value">${o.total_scored || 0}</div><div class="sub">${o.total_failed || 0} failed</div></div>`;
    html += `<div class="card"><div class="label">Mean Value</div><div class="value" style="color:${scoreColor(o.mean_value)}">${(o.mean_value || 0).toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Complexity</div><div class="value" style="color:${scoreColor(o.mean_complexity)}">${(o.mean_complexity || 0).toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Quality</div><div class="value" style="color:${scoreColor(o.mean_quality)}">${(o.mean_quality || 0).toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Median Rarity</div><div class="value" style="color:${scoreColor(o.median_rarity)}">${(o.median_rarity || 0).toFixed(1)}</div></div>`;
    const sel = d.selection_thresholds || {};
    if (sel.top_10pct) {
        html += `<div class="card"><div class="label">Top 10%</div><div class="value c-green">${sel.top_10pct.count}</div><div class="sub">≥ ${sel.top_10pct.threshold}</div></div>`;
    }
    html += `<div class="card"><div class="label">Tokens Used</div><div class="value">${((o.total_tokens || 0) / 1000).toFixed(0)}K</div></div>`;
    html += '</div>';

    // === Section 2: Score Distributions ===
    html += '<h2>Score Distributions</h2><div class="grid">';
    const histColors = {'value_score': '#4a6cf7', 'complexity_overall': '#ff9800', 'quality_overall': '#4caf50', 'reasoning_overall': '#9c27b0', 'rarity_score': '#00bcd4'};
    const histLabels = {'value_score': 'Value Score', 'complexity_overall': 'Complexity', 'quality_overall': 'Quality', 'reasoning_overall': 'Reasoning', 'rarity_score': 'Rarity'};
    for (const [key, bins] of Object.entries(d.histograms || {})) {
        html += `<div class="panel">${renderHistogram(key, bins, histLabels[key] || key, histColors[key] || '#4a6cf7')}</div>`;
    }
    html += '</div>';

    // === Section 3: Sub-score Breakdown ===
    const sub = d.sub_score_means || {};
    if (Object.keys(sub).length > 0) {
        html += '<h2>Sub-score Breakdown</h2><div class="grid">';
        for (const [dim, scores] of Object.entries(sub)) {
            const items = Object.entries(scores).map(([k, v]) => [k, v]);
            html += `<div class="panel"><h3>${dim}</h3>${renderBarChart(items, 10, histColors[dim + '_overall'] || '#4a6cf7')}</div>`;
        }
        html += '</div>';
    }

    // === Section 4: Value × Tag Cross-Analysis ===
    const vbt = d.value_by_tag || {};
    if (Object.keys(vbt).length > 0) {
        html += '<h2>Value × Tag Cross-Analysis</h2><div class="grid">';
        const dimColors = {'difficulty': '#ff9800', 'intent': '#4a6cf7', 'domain': '#4caf50', 'concept': '#9c27b0', 'language': '#00bcd4', 'task': '#e91e63', 'agentic': '#ff5722', 'constraint': '#607d8b', 'context': '#795548'};
        for (const dim of ['difficulty', 'intent', 'domain', 'concept', 'language', 'task']) {
            const tags = vbt[dim];
            if (!tags || Object.keys(tags).length === 0) continue;
            const items = Object.entries(tags).map(([t, info]) => [t, info.mean, info.n]).sort((a, b) => b[1] - a[1]);
            html += `<div class="panel"><h3>Value by ${dim}</h3>${renderBarChart(items, 10, dimColors[dim] || '#4a6cf7', 15)}</div>`;
        }
        html += '</div>';
    }

    // === Section 5: Thinking Mode Analysis ===
    const tm = d.thinking_mode_stats || {};
    if (tm.slow || tm.fast) {
        html += '<h2>Thinking Mode Analysis</h2><div class="grid">';
        html += '<div class="panel"><table><thead><tr><th></th><th>Slow Thinking</th><th>Fast Thinking</th></tr></thead><tbody>';
        const s = tm.slow || {}; const f = tm.fast || {};
        html += `<tr><td>Count</td><td>${s.count || 0}</td><td>${f.count || 0}</td></tr>`;
        html += `<tr><td>Mean Value</td><td style="color:${scoreColor(s.mean_value)}">${(s.mean_value || 0).toFixed(1)}</td><td style="color:${scoreColor(f.mean_value)}">${(f.mean_value || 0).toFixed(1)}</td></tr>`;
        html += `<tr><td>Mean Quality</td><td>${(s.mean_quality || 0).toFixed(1)}</td><td>${(f.mean_quality || 0).toFixed(1)}</td></tr>`;
        html += `<tr><td>Mean Reasoning</td><td>${(s.mean_reasoning || 0).toFixed(1)}</td><td>${(f.mean_reasoning || 0).toFixed(1)}</td></tr>`;
        html += '</tbody></table></div>';
        html += '</div>';
    }

    // === Section 6: Flag Analysis ===
    const flags = d.flag_counts || {};
    if (Object.keys(flags).length > 0) {
        html += '<h2>Flag Analysis</h2><div class="grid">';
        const flagItems = Object.entries(flags).sort((a, b) => b[1] - a[1]);
        const maxFlag = Math.max(...flagItems.map(x => x[1]), 1);
        let flagHtml = '';
        for (const [flag, count] of flagItems) {
            const impact = (d.flag_value_impact || {})[flag];
            const meanV = impact ? impact.mean_value.toFixed(1) : '?';
            const isNeg = ['has-bug', 'security-issue', 'outdated-practice', 'incomplete', 'over-engineered', 'incorrect-output', 'poor-explanation'].includes(flag);
            const color = isNeg ? '#f44336' : '#4caf50';
            const w = (count / maxFlag * 100).toFixed(1);
            flagHtml += `<div class="bar-row">
                <div class="bar-label" title="${flag}">${flag}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
                <div class="bar-val">${count} (avg: ${meanV})</div>
            </div>`;
        }
        html += `<div class="panel"><h3>Flags (count & mean value)</h3>${flagHtml}</div>`;
        html += '</div>';
    }

    // === Section 7: Coverage Impact (if available) ===
    const cov = d.coverage_at_thresholds || {};
    if (Object.keys(cov).length > 0) {
        html += '<h2>Coverage Impact Analysis</h2>';
        html += '<div class="panel"><table><thead><tr><th>Threshold</th><th>Retained</th><th>%</th><th>Coverage</th><th>Tags Lost</th></tr></thead><tbody>';
        for (const [thresh, info] of Object.entries(cov).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
            const lost = (info.tags_lost || []).slice(0, 5).join(', ');
            const lostMore = (info.tags_lost || []).length > 5 ? ` +${info.tags_lost.length - 5} more` : '';
            html += `<tr><td>≥ ${thresh}</td><td>${info.retained}</td><td>${(info.pct * 100).toFixed(1)}%</td><td>${(info.coverage * 100).toFixed(1)}%</td><td><span class="tag">${lost}${lostMore}</span></td></tr>`;
        }
        html += '</tbody></table></div>';
    }

    // === Section 8: File Ranking (global dashboard) ===
    const pfs = d.per_file_summary || [];
    if (pfs.length > 1) {
        html += '<h2>File Ranking</h2>';
        html += '<div class="panel"><table id="file-table"><thead><tr>';
        html += '<th onclick="sortTable(0)">File</th><th onclick="sortTable(1)">Count</th><th onclick="sortTable(2)">Value</th><th onclick="sortTable(3)">Complexity</th><th onclick="sortTable(4)">Quality</th><th onclick="sortTable(5)">Rarity</th>';
        html += '</tr></thead><tbody>';
        for (const f of pfs.sort((a, b) => (b.mean_value || 0) - (a.mean_value || 0))) {
            html += `<tr>
                <td>${f.file}</td><td>${f.count}</td>
                <td style="color:${scoreColor(f.mean_value)}">${(f.mean_value || 0).toFixed(1)}</td>
                <td>${(f.mean_complexity || 0).toFixed(1)}</td>
                <td>${(f.mean_quality || 0).toFixed(1)}</td>
                <td>${(f.mean_rarity || 0).toFixed(1)}</td>
            </tr>`;
        }
        html += '</tbody></table></div>';
    }

    // Weights used
    const w = d.weights_used || {};
    if (Object.keys(w).length > 0) {
        html += '<h2>Configuration</h2><div class="panel">';
        html += '<p style="color:#666;font-size:0.85em">Value Score = ';
        html += Object.entries(w).map(([k, v]) => `${v}×${k}`).join(' + ');
        html += '</p></div>';
    }

    el.innerHTML = html;
}

function sortTable(col) {
    const table = document.getElementById('file-table');
    if (!table) return;
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const isNum = col > 0;
    rows.sort((a, b) => {
        const av = a.cells[col].textContent;
        const bv = b.cells[col].textContent;
        return isNum ? parseFloat(bv) - parseFloat(av) : av.localeCompare(bv);
    });
    rows.forEach(r => tbody.appendChild(r));
}

render();
</script>
</body>
</html>"""


def generate_value_dashboard(run_dir, scored_file="scored.json",
                              stats_file="stats_value.json",
                              output_file="dashboard_value.html"):
    """Generate a value scoring dashboard HTML file.

    Works in two modes:
    - Per-file: scored_file + stats_file (full data with histograms)
    - Global: scored_file=None, stats_file only (stats-based, no histograms)
    """
    run_dir = Path(run_dir)
    samples, stats = load_value_run(run_dir, scored_file, stats_file)
    viz_data = compute_value_viz_data(samples, stats)

    html = VALUE_HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__",
                                        json.dumps(viz_data, ensure_ascii=False))

    out = run_dir / output_file
    out.write_text(html, encoding="utf-8")
    print(f"  Dashboard: {out}")
