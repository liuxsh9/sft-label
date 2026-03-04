"""
Value Scoring Dashboard Generator

Generates standalone HTML dashboards for Pass 2 value scoring results.
When Pass 1 data (stats.json / labeled.json) is found in the same directory,
the dashboard includes Pass 1 sections (tag distributions, confidence, coverage)
at the top, with Pass 2 sections appended below.

Pass 1's standalone dashboard (dashboard.html) is unaffected.
"""

import json
from pathlib import Path


def load_value_run(run_dir, scored_file="scored.json", stats_file="stats_value.json"):
    """Load scored samples and value stats from a run directory.

    When scored_file is None (global/directory dashboard), automatically
    collects samples from scored*.json/jsonl files in subdirectories for histograms.
    """
    run_dir = Path(run_dir)
    samples = []
    if scored_file:
        scored_path = run_dir / scored_file
        if scored_path.exists():
            if scored_path.suffix == ".jsonl":
                with open(scored_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                with open(scored_path, encoding="utf-8") as f:
                    samples = json.load(f)
    else:
        # Global mode: collect from subdirectory scored files
        # Try .json first, fall back to .jsonl for dirs without .json
        json_paths = {}  # parent_dir -> path
        for pattern in ("*/scored*.json", "scored*.json"):
            for p in sorted(run_dir.glob(pattern)):
                if "summary" in p.name or "stats" in p.name or p.suffix == ".jsonl":
                    continue
                json_paths[p.parent] = p

        jsonl_paths = {}
        for pattern in ("*/scored*.jsonl", "scored*.jsonl"):
            for p in sorted(run_dir.glob(pattern)):
                if p.parent not in json_paths:
                    jsonl_paths[p.parent] = p

        for p in sorted(json_paths.values()):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
            except (json.JSONDecodeError, OSError):
                continue

        for p in sorted(jsonl_paths.values()):
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                continue

    stats = {}
    stats_path = run_dir / stats_file
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
        "input_file": stats.get("input_file") or stats.get("input_path", ""),
        "mean_value": dists.get("value_score", {}).get("mean", 0),
        "mean_complexity": dists.get("complexity_overall", {}).get("mean", 0),
        "mean_quality": dists.get("quality_overall", {}).get("mean", 0),
        "median_rarity": dists.get("rarity_score", {}).get("p50", 0),
        "mean_confidence": dists.get("confidence", {}).get("mean", 0),
        "total_tokens": stats.get("total_tokens", 0),
    }

    # Selection thresholds
    viz["selection_thresholds"] = stats.get("selection_thresholds", {})

    # Score distributions (histogram bins 1-10)
    score_keys = ["value_score", "complexity_overall", "quality_overall",
                  "reasoning_overall", "rarity_score", "selection_score"]
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
                elif key in ("selection_score", "intra_class_rank"):
                    val = v.get(key)
                else:
                    dim, sub = key.rsplit("_", 1)
                    val = (v.get(dim) or {}).get(sub)
                if isinstance(val, (int, float)) and 1 <= val <= 10:
                    idx = min(int(val) - 1, 9)
                    bins[idx] += 1
            histograms[key] = bins
    elif "histograms" in stats:
        # Fallback: use pre-computed histogram bins from stats
        histograms = stats["histograms"]
    viz["histograms"] = histograms

    # Confidence histogram (0.0-1.0 in 10 bins of 0.1)
    conf_bins = [0] * 10
    for s in samples:
        v = s.get("value", {})
        conf = v.get("confidence")
        if isinstance(conf, (int, float)) and 0 <= conf <= 1:
            idx = min(int(conf * 10), 9)
            conf_bins[idx] += 1
    viz["confidence_histogram"] = conf_bins

    # Score distribution stats (percentiles etc. from stats file)
    viz["score_distributions"] = dists

    # Sub-score means
    viz["sub_score_means"] = stats.get("sub_score_means", {})

    # Value by tag (cross-analysis)
    viz["value_by_tag"] = stats.get("value_by_tag", {})

    # Selection by tag (cross-analysis)
    viz["selection_by_tag"] = stats.get("selection_by_tag", {})

    # Selection vs Value 2D histogram (from samples)
    sv_heatmap = {}
    for s in samples:
        v = s.get("value", {})
        vs = v.get("value_score")
        ss = v.get("selection_score")
        if isinstance(vs, (int, float)) and isinstance(ss, (int, float)) and 1 <= vs <= 10 and 1 <= ss <= 10:
            v_bin = max(1, min(int(vs), 10))
            s_bin = max(1, min(int(ss), 10))
            key = f"{v_bin}|{s_bin}"
            sv_heatmap[key] = sv_heatmap.get(key, 0) + 1
    viz["selection_vs_value"] = sv_heatmap

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


def _load_conversation_scores(run_dir):
    """Load conversation_scores.json from run directory."""
    path = run_dir / "conversation_scores.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _compute_conv_viz_data(conv_records):
    """Compute visualization data for conversation-level aggregation."""
    if not conv_records:
        return None

    total = len(conv_records)
    values = [r["conv_value"] for r in conv_records if r.get("conv_value") is not None]
    selections = [r["conv_selection"] for r in conv_records if r.get("conv_selection") is not None]
    peaks = [r["peak_complexity"] for r in conv_records if r.get("peak_complexity") is not None]
    turns = [r["turn_count"] for r in conv_records]

    viz = {
        "total": total,
        "mean_conv_value": sum(values) / len(values) if values else 0,
        "mean_conv_selection": sum(selections) / len(selections) if selections else 0,
        "mean_peak_complexity": sum(peaks) / len(peaks) if peaks else 0,
        "mean_turns": sum(turns) / len(turns) if turns else 0,
    }

    # Histograms (bins 1-10)
    for key, data in [("conv_value", values), ("conv_selection", selections),
                       ("peak_complexity", peaks)]:
        bins = [0] * 10
        for v in data:
            if isinstance(v, (int, float)) and 1 <= v <= 10:
                idx = min(int(v) - 1, 9)
                bins[idx] += 1
        viz[f"{key}_hist"] = bins

    # Turn count distribution
    turn_counts = {}
    for t in turns:
        turn_counts[t] = turn_counts.get(t, 0) + 1
    viz["turn_distribution"] = turn_counts

    return viz


def _load_pass1_data(run_dir, is_global):
    """Try to load Pass 1 viz data from the same run directory.

    Returns viz_data dict or None if Pass 1 data is not available.
    """
    try:
        from sft_label.tools.visualize_labels import load_run, compute_viz_data

        if is_global:
            p1_stats_file = "summary_stats.json"
            p1_labeled_file = None
        else:
            p1_stats_file = "stats.json"
            p1_labeled_file = "labeled.json"

        p1_stats_path = run_dir / p1_stats_file
        if not p1_stats_path.exists():
            return None

        p1_samples, p1_stats = load_run(
            run_dir, labeled_file=p1_labeled_file, stats_file=p1_stats_file
        )
        if not p1_stats:
            return None

        return compute_viz_data(p1_samples, p1_stats)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# Combined HTML Template (Pass 1 + Pass 2)
# ─────────────────────────────────────────────────────────

COMBINED_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFT Labeling & Value Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f7; color: #1d1d1f; }
.header { background: linear-gradient(135deg, #1d1d1f 0%, #2d3748 100%); color: white; padding: 20px 28px; }
.header h1 { font-size: 1.4em; font-weight: 700; }
.header .sub { font-size: 0.82em; color: #a1a1aa; margin-top: 4px; }
.container { max-width: 1800px; margin: 0 auto; padding: 20px; }
.section { margin-bottom: 28px; }
.section h2 { font-size: 1.1em; font-weight: 600; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #e5e7eb; }
h3 { font-size: 0.95em; font-weight: 600; margin: 0 0 10px; color: #374151; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.card .label { font-size: 0.75em; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 1.6em; font-weight: 700; margin-top: 4px; }
.card .sub { font-size: 0.8em; color: #6b7280; margin-top: 2px; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.panel { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }

.bar-row { display: flex; align-items: center; margin-bottom: 4px; font-size: 0.82em; }
.bar-label { width: 130px; text-align: right; padding-right: 8px; color: #374151; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 18px; background: #f3f4f6; border-radius: 3px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.bar-val { min-width: 50px; text-align: right; padding-left: 6px; color: #6b7280; font-size: 0.82em; }
.bar-count { width: 50px; text-align: right; font-size: 0.78em; color: #6b7280; padding-left: 6px; }

.hist-row { display: flex; align-items: flex-end; gap: 2px; height: 100px; margin-top: 8px; }
.hist-bar { flex: 1; border-radius: 2px 2px 0 0; min-height: 2px; position: relative; }
.hist-bar:hover { opacity: 0.8; }
.hist-labels { display: flex; gap: 2px; font-size: 0.7em; color: #6b7280; }
.hist-labels span { flex: 1; text-align: center; }

table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
th { background: #f9fafb; padding: 8px; text-align: left; cursor: pointer; user-select: none; font-weight: 600; }
th:hover { background: #f3f4f6; }
td { padding: 8px; border-top: 1px solid #e5e7eb; }
tr:hover td { background: #f9fafb; }

.quad-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.quad { background: #f3f4f6; border-radius: 6px; padding: 12px; text-align: center; }
.quad .q-label { font-size: 0.75em; color: #6b7280; }
.quad .q-value { font-size: 1.4em; font-weight: 700; margin-top: 4px; }

.c-green { color: #16a34a; } .c-blue { color: #3b82f6; } .c-orange { color: #ea580c; } .c-red { color: #dc2626; }
.c-purple { color: #9333ea; } .c-cyan { color: #0891b2; }

.tag { display: inline-block; background: #f3f4f6; border-radius: 4px; padding: 2px 6px; font-size: 0.75em; margin: 1px; }

.pct-table { width: 100%; font-size: 0.82em; border-collapse: collapse; }
.pct-table th { background: #f9fafb; padding: 6px 10px; text-align: center; font-weight: 600; border: 1px solid #e5e7eb; }
.pct-table td { padding: 6px 10px; text-align: center; border: 1px solid #e5e7eb; }

/* Pass 1 specific styles */
.dim-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.dim-card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.dim-card h3 { font-size: 0.95em; font-weight: 600; margin-bottom: 10px; display: flex; justify-content: space-between; }
.dim-card h3 .badge { font-size: 0.75em; background: #f3f4f6; padding: 2px 8px; border-radius: 4px; font-weight: 500; color: #6b7280; }
.conf-table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
.conf-table th { padding: 6px 8px; text-align: center; font-weight: 600; background: #f9fafb; }
.conf-table td { padding: 5px 8px; text-align: center; border: 1px solid #f3f4f6; }
.conf-cell { border-radius: 4px; padding: 3px 6px; font-weight: 500; }
.cov-row { display: flex; align-items: center; margin-bottom: 6px; font-size: 0.82em; }
.cov-label { width: 90px; font-weight: 500; }
.cov-bar { flex: 1; height: 14px; background: #f3f4f6; border-radius: 3px; overflow: hidden; }
.cov-fill { height: 100%; background: #3b82f6; border-radius: 3px; }
.cov-text { width: 100px; text-align: right; font-size: 0.78em; color: #6b7280; }
.cross-table { border-collapse: collapse; font-size: 0.82em; }
.cross-table th, .cross-table td { padding: 6px 14px; text-align: center; border: 1px solid #e5e7eb; }
.cross-table th { background: #f9fafb; font-weight: 600; }
.cross-cell { border-radius: 4px; padding: 2px 8px; }
.unused-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.unused-tag { font-size: 0.72em; background: #fef2f2; color: #991b1b; padding: 2px 6px; border-radius: 3px; }

.bars-compact .bar-row { margin-bottom: 2px; font-size: 0.76em; }
.bars-compact .bar-track { height: 14px; }
.bars-compact .bar-label { width: 110px; }
.bars-compact .bar-val { font-size: 0.76em; }
.bars-compact .bar-row:nth-child(even) { background: #fafbfc; border-radius: 3px; }

.heatmap-table { border-collapse: collapse; font-size: 0.72em; width: 100%; table-layout: fixed; }
.heatmap-table th, .heatmap-table td { padding: 3px 2px; text-align: center; border: 1px solid #e5e7eb; overflow: hidden; }
.heatmap-table th { background: #f9fafb; font-weight: 600; }
.heatmap-table th:first-child { width: 24px; }

.pass-divider { border: none; border-top: 3px solid #e5e7eb; margin: 32px 0 24px; }
</style>
</head>
<body>
<div class="header">
  <h1>SFT Labeling & Value Dashboard</h1>
  <div class="sub" id="subtitle"></div>
</div>
<div class="container">
  <div id="pass1"></div>
  <div id="dashboard"></div>
</div>

<script>
const DATA_P1 = __DATA_P1_PLACEHOLDER__;
const DATA_P2 = __DATA_P2_PLACEHOLDER__;
const DATA_CONV = __DATA_CONV_PLACEHOLDER__;

// ── Shared utilities ──
function scoreColor(v) {
    if (v >= 8) return '#16a34a';
    if (v >= 6) return '#3b82f6';
    if (v >= 4) return '#ea580c';
    return '#dc2626';
}

function groupDims(dims, dataByDim, threshold) {
    // Group small dims (≤threshold tags) into pairs sharing one panel
    const groups = [];
    let pending = null;
    for (const dim of dims) {
        const data = dataByDim[dim];
        if (!data || Object.keys(data).length === 0) continue;
        if (Object.keys(data).length <= threshold) {
            if (pending) { groups.push([pending, dim]); pending = null; }
            else { pending = dim; }
        } else {
            if (pending) { groups.push([pending]); pending = null; }
            groups.push([dim]);
        }
    }
    if (pending) groups.push([pending]);
    return groups;
}

const PANEL_SEP = '<div style="margin:14px 0 10px;border-top:1px solid #e5e7eb"></div>';
const TAG_BADGE = (n) => `<span style="font-size:0.78em;color:#6b7280;font-weight:400">(${n} tags)</span>`;

// ── Pass 1 rendering ──
const P1_COLORS = {
    intent: '#3b82f6', language: '#8b5cf6', domain: '#06b6d4', task: '#f59e0b',
    difficulty: '#10b981', concept: '#ec4899', agentic: '#f97316', constraint: '#6366f1', context: '#14b8a6'
};

function confColor(v) {
    if (v >= 0.9) return '#dcfce7';
    if (v >= 0.8) return '#fef9c3';
    if (v >= 0.7) return '#fed7aa';
    return '#fecaca';
}

function renderPass1(d) {
    const el = document.getElementById('pass1');
    if (!d) return;

    const ov = d.overview || {};
    let html = '';

    // Overview cards
    html += '<div class="section"><h2>Labeling Overview</h2><div class="cards">';
    html += [
        ['Samples', d.total],
        ['Success', ((ov.success_rate || 0) * 100).toFixed(1) + '%'],
        ['Tokens', (ov.total_tokens || 0).toLocaleString()],
        ['Arbitrated', ((ov.arbitrated_rate || 0) * 100).toFixed(1) + '%'],
        ['Unmapped', ov.unmapped_unique || 0],
    ].map(([l, v]) => `<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');
    html += '</div></div>';

    // Tag distributions
    const distributions = d.distributions || {};
    if (Object.keys(distributions).length > 0) {
        html += '<div class="section"><h2>Tag Distributions</h2><div class="grid">';
        const prefOrder = ['intent', 'difficulty', 'context', 'language', 'domain', 'concept', 'task', 'agentic', 'constraint'];
        const allDims = prefOrder.filter(d => distributions[d]);
        for (const d of Object.keys(distributions)) { if (!allDims.includes(d)) allDims.push(d); }
        const groups = groupDims(allDims, distributions, 10);
        for (const group of groups) {
            html += '<div class="panel">';
            group.forEach((dim, g) => {
                if (g > 0) html += PANEL_SEP;
                const entries = Object.entries(distributions[dim]).sort((a, b) => b[1] - a[1]);
                const total = entries.reduce((s, e) => s + e[1], 0);
                const color = P1_COLORS[dim] || '#6b7280';
                const items = entries.map(([tag, count]) => [tag, count, `(${(count/total*100).toFixed(0)}%)`]);
                html += `<h3>${dim} ${TAG_BADGE(items.length)}</h3>${renderBarChart(items, null, color)}`;
            });
            html += '</div>';
        }
        html += '</div></div>';
    }

    // Confidence summary
    const confStats = d.confidence_stats || {};
    const confDims = Object.keys(confStats);
    if (confDims.length > 0) {
        html += '<div class="section"><h2>Confidence Summary</h2><div class="panel">';
        html += '<table class="conf-table"><tr><th>Dimension</th><th>Mean</th><th>Min</th><th>Max</th><th>Below Threshold</th></tr>';
        for (const dim of confDims) {
            const cs = confStats[dim];
            html += `<tr>
                <td style="font-weight:600">${dim}</td>
                <td><span class="conf-cell" style="background:${confColor(cs.mean)}">${cs.mean.toFixed(3)}</span></td>
                <td>${cs.min.toFixed(2)}</td><td>${cs.max.toFixed(2)}</td>
                <td>${cs.below_threshold}</td>
            </tr>`;
        }
        html += '</table></div></div>';
    }

    // Intent × Difficulty cross matrix
    const cm = d.cross_matrix || {};
    if (cm.rows && cm.rows.length > 0 && cm.cols && cm.cols.length > 0) {
        html += '<div class="section"><h2>Intent &times; Difficulty</h2><div class="panel">';
        html += '<table class="cross-table"><tr><th></th>';
        cm.cols.forEach(c => html += `<th>${c}</th>`);
        html += '<th>Total</th></tr>';
        for (const row of cm.rows) {
            html += `<tr><th>${row}</th>`;
            let rowTotal = 0;
            for (const col of cm.cols) {
                const v = cm.data[`${row}|${col}`] || 0;
                rowTotal += v;
                const bg = v > 0 ? 'background:#3b82f622' : '';
                html += `<td><span class="cross-cell" style="${bg}">${v || '-'}</span></td>`;
            }
            html += `<td style="font-weight:600">${rowTotal}</td></tr>`;
        }
        html += '</table></div></div>';
    }

    // Tag pool coverage
    const coverage = d.coverage || {};
    if (Object.keys(coverage).length > 0) {
        html += '<div class="section"><h2>Tag Pool Coverage</h2><div class="panel">';
        for (const dim of Object.keys(coverage)) {
            const c = coverage[dim];
            if (!c.pool_size) continue;
            const pct = (c.rate * 100).toFixed(0);
            html += `<div class="cov-row">
                <div class="cov-label">${dim}</div>
                <div class="cov-bar"><div class="cov-fill" style="width:${pct}%"></div></div>
                <div class="cov-text">${c.used}/${c.pool_size} (${pct}%)</div>
            </div>`;
            if (c.unused.length && c.unused.length <= 20) {
                html += `<div style="margin-left:90px;margin-bottom:8px"><div class="unused-tags">${c.unused.map(t => '<span class="unused-tag">' + t + '</span>').join('')}</div></div>`;
            } else if (c.unused.length > 20) {
                html += `<div style="margin-left:90px;margin-bottom:8px;font-size:0.75em;color:#991b1b">${c.unused.length} unused tags (expand dataset for better coverage)</div>`;
            }
        }
        html += '</div></div>';
    }

    // Divider between Pass 1 and Pass 2
    html += '<hr class="pass-divider">';

    el.innerHTML = html;
}

// ── Pass 2 rendering ──
function renderHistogram(bins, label, color, distInfo) {
    const max = Math.max(...bins, 1);
    let html = `<h3>${label}</h3>`;
    if (distInfo && distInfo.mean != null) {
        html += `<div style="font-size:0.9em;margin-bottom:8px;color:#6b7280">Mean: <strong style="color:${scoreColor(distInfo.mean)}">${distInfo.mean.toFixed(2)}</strong> &nbsp; Std: ${(distInfo.std || 0).toFixed(2)}</div>`;
    }
    html += '<div class="hist-row">';
    for (let i = 0; i < bins.length; i++) {
        const h = Math.max((bins[i] / max) * 100, 2);
        html += `<div class="hist-bar" style="height:${h}%;background:${color}" title="${i+1}: ${bins[i]}"></div>`;
    }
    html += '</div><div class="hist-labels">';
    for (let i = 1; i <= 10; i++) html += `<span>${i}</span>`;
    html += '</div>';
    return html;
}

function renderPercentileTable(key, distInfo, color) {
    const label = histLabels[key] || key;
    const pcts = ['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'];
    const pctLabels = ['Min', 'P10', 'P25', 'P50', 'P75', 'P90', 'Max'];
    let html = `<h3>${label}</h3>`;
    html += `<div style="font-size:0.9em;margin-bottom:8px;color:#6b7280">Mean: <strong style="color:${scoreColor(distInfo.mean || 0)}">${(distInfo.mean || 0).toFixed(2)}</strong> &nbsp; Std: ${(distInfo.std || 0).toFixed(2)}</div>`;
    html += '<table class="pct-table"><thead><tr>';
    for (const pl of pctLabels) html += `<th>${pl}</th>`;
    html += '</tr></thead><tbody><tr>';
    for (const p of pcts) html += `<td>${(distInfo[p] != null ? distInfo[p].toFixed(1) : '-')}</td>`;
    html += '</tr></tbody></table>';
    return html;
}

function renderBarChart(items, maxVal, color, limit) {
    if (!items || items.length === 0) return '<p style="color:#6b7280">No data</p>';
    const shown = limit ? items.slice(0, limit) : items;
    const m = maxVal || Math.max(...shown.map(x => x[1]), 1);
    const dense = shown.length > 20;
    let html = dense ? '<div class="bars-compact">' : '';
    for (const [label, value, extra] of shown) {
        const w = (value / m * 100).toFixed(1);
        const valStr = typeof value === 'number' ? (value % 1 === 0 ? String(value) : value.toFixed(1)) : String(value);
        const extraStr = extra != null ? (typeof extra === 'number' ? ` (n=${extra})` : ` ${extra}`) : '';
        html += `<div class="bar-row">
            <div class="bar-label" title="${label}">${label}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
            <div class="bar-val">${valStr}${extraStr}</div>
        </div>`;
    }
    if (dense) html += '</div>';
    return html;
}

function renderHeatmap(data) {
    const bins = Array.from({length: 10}, () => Array(10).fill(0));
    let maxCount = 0;
    for (const [key, count] of Object.entries(data)) {
        const [v, s] = key.split('|').map(Number);
        if (v >= 1 && v <= 10 && s >= 1 && s <= 10) {
            bins[s-1][v-1] = count;
            maxCount = Math.max(maxCount, count);
        }
    }
    let html = '<div style="display:flex;align-items:center;gap:8px">';
    html += '<div style="writing-mode:vertical-rl;transform:rotate(180deg);font-size:0.78em;color:#6b7280;font-weight:600">Selection Score</div>';
    html += '<div>';
    html += '<table class="heatmap-table"><thead><tr><th></th>';
    for (let i = 1; i <= 10; i++) html += `<th>${i}</th>`;
    html += '</tr></thead><tbody>';
    for (let s = 9; s >= 0; s--) {
        html += `<tr><th>${s+1}</th>`;
        for (let v = 0; v < 10; v++) {
            const count = bins[s][v];
            const intensity = maxCount > 0 ? count / maxCount : 0;
            const bg = count > 0 ? `rgba(124, 58, 237, ${0.08 + intensity * 0.85})` : '#f9fafb';
            const tc = intensity > 0.5 ? 'white' : '#374151';
            html += `<td style="background:${bg};color:${tc}">${count || ''}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    html += '<div style="text-align:center;font-size:0.78em;color:#6b7280;font-weight:600;margin-top:4px">Value Score</div>';
    html += '</div></div>';
    return html;
}

const histColors = {'value_score': '#3b82f6', 'complexity_overall': '#ea580c', 'quality_overall': '#16a34a', 'reasoning_overall': '#9333ea', 'rarity_score': '#0891b2', 'selection_score': '#db2777'};
const histLabels = {'value_score': 'Value Score', 'complexity_overall': 'Complexity', 'quality_overall': 'Quality', 'reasoning_overall': 'Reasoning', 'rarity_score': 'Rarity', 'selection_score': 'Selection Score'};

function renderPass2() {
    const d = DATA_P2;
    const o = d.overview || {};
    const el = document.getElementById('dashboard');
    let html = '';

    // === Section 1: Overview Cards ===
    html += '<div class="section"><h2>Value Overview</h2><div class="cards">';
    html += `<div class="card"><div class="label">Scored</div><div class="value">${o.total_scored || 0}</div><div class="sub">${o.total_failed || 0} failed</div></div>`;
    html += `<div class="card"><div class="label">Mean Value</div><div class="value" style="color:${scoreColor(o.mean_value)}">${(o.mean_value || 0).toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Complexity</div><div class="value" style="color:${scoreColor(o.mean_complexity)}">${(o.mean_complexity || 0).toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Quality</div><div class="value" style="color:${scoreColor(o.mean_quality)}">${(o.mean_quality || 0).toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Median Rarity</div><div class="value" style="color:${scoreColor(o.median_rarity)}">${(o.median_rarity || 0).toFixed(1)}</div></div>`;
    const selDist = (d.score_distributions || {}).selection_score || {};
    if (selDist.mean) {
        html += `<div class="card"><div class="label">Mean Selection</div><div class="value" style="color:${scoreColor(selDist.mean)}">${selDist.mean.toFixed(1)}</div></div>`;
    }
    const confDist = (d.score_distributions || {}).confidence || {};
    if (confDist.mean) {
        html += `<div class="card"><div class="label">Mean Confidence</div><div class="value">${confDist.mean.toFixed(2)}</div><div class="sub">p50=${(confDist.p50 || 0).toFixed(2)}</div></div>`;
    }
    const sel = d.selection_thresholds || {};
    if (sel.top_10pct) {
        html += `<div class="card"><div class="label">Top 10%</div><div class="value c-green">${sel.top_10pct.count}</div><div class="sub">&ge; ${sel.top_10pct.threshold}</div></div>`;
    }
    html += `<div class="card"><div class="label">Tokens Used</div><div class="value">${((o.total_tokens || 0) / 1000).toFixed(0)}K</div></div>`;
    html += '</div></div>';

    // === Section 2: Score Distributions ===
    const hasHistograms = Object.keys(d.histograms || {}).length > 0;
    const hasDistStats = Object.keys(d.score_distributions || {}).length > 0;
    if (hasHistograms || hasDistStats) {
        html += '<div class="section"><h2>Score Distributions</h2><div class="grid">';
        const scoreKeys = ['value_score', 'complexity_overall', 'quality_overall', 'reasoning_overall', 'rarity_score', 'selection_score'];
        if (hasHistograms) {
            for (const key of scoreKeys) {
                const bins = d.histograms[key];
                if (!bins) continue;
                const distInfo = (d.score_distributions || {})[key] || {};
                html += `<div class="panel">${renderHistogram(bins, histLabels[key] || key, histColors[key] || '#3b82f6', distInfo)}</div>`;
            }
        } else {
            for (const key of scoreKeys) {
                const distInfo = d.score_distributions[key];
                if (!distInfo) continue;
                html += `<div class="panel">${renderPercentileTable(key, distInfo, histColors[key] || '#3b82f6')}</div>`;
            }
        }
        html += '</div></div>';
    }

    // Confidence histogram (0-1 range, separate from 1-10 score histograms)
    const confBins = d.confidence_histogram;
    if (confBins && confBins.some(b => b > 0)) {
        html += '<div class="section"><h2>Confidence Distribution</h2><div class="grid">';
        const maxBin = Math.max(...confBins, 1);
        let confHtml = '<div class="panel"><h3>LLM Confidence (0.0 - 1.0)</h3><div style="display:flex;align-items:flex-end;gap:3px;height:120px;">';
        for (let i = 0; i < confBins.length; i++) {
            const h = Math.round(confBins[i] / maxBin * 100);
            const label = (i / 10).toFixed(1);
            confHtml += `<div style="flex:1;display:flex;flex-direction:column;align-items:center;">`;
            confHtml += `<span style="font-size:10px;color:#6b7280;">${confBins[i]}</span>`;
            confHtml += `<div style="width:100%;height:${h}px;background:#8b5cf6;border-radius:2px;"></div>`;
            confHtml += `<span style="font-size:9px;color:#9ca3af;margin-top:2px;">${label}</span>`;
            confHtml += `</div>`;
        }
        confHtml += '</div></div>';
        html += confHtml;
        html += '</div></div>';
    }

    // === Section 3: Sub-score Breakdown ===
    const sub = d.sub_score_means || {};
    if (Object.keys(sub).length > 0) {
        html += '<div class="section"><h2>Sub-score Breakdown</h2><div class="grid">';
        for (const [dim, scores] of Object.entries(sub)) {
            const items = Object.entries(scores).map(([k, v]) => [k, v]);
            html += `<div class="panel"><h3>${dim}</h3>${renderBarChart(items, 10, histColors[dim + '_overall'] || '#3b82f6')}</div>`;
        }
        html += '</div></div>';
    }

    // === Section 4: Value x Tag Cross-Analysis ===
    const vbt = d.value_by_tag || {};
    if (Object.keys(vbt).length > 0) {
        html += '<div class="section"><h2>Value &times; Tag Cross-Analysis</h2><div class="grid">';
        const dimColors = {'difficulty': '#ea580c', 'intent': '#3b82f6', 'domain': '#16a34a', 'concept': '#9333ea', 'language': '#0891b2', 'task': '#db2777', 'agentic': '#ea580c', 'constraint': '#475569', 'context': '#78716c'};
        const vGroups = groupDims(['difficulty', 'intent', 'domain', 'concept', 'language', 'task'], vbt, 10);
        for (const group of vGroups) {
            html += '<div class="panel">';
            group.forEach((dim, g) => {
                if (g > 0) html += PANEL_SEP;
                const items = Object.entries(vbt[dim]).map(([t, info]) => [t, info.mean, info.n]).sort((a, b) => b[1] - a[1]);
                html += `<h3>Value by ${dim} ${TAG_BADGE(items.length)}</h3>${renderBarChart(items, 10, dimColors[dim] || '#3b82f6')}`;
            });
            html += '</div>';
        }
        html += '</div></div>';
    }

    // === Section 5: Analysis (heatmap + thinking mode + flags) ===
    const svh = d.selection_vs_value || {};
    const tm = d.thinking_mode_stats || {};
    const flags = d.flag_counts || {};
    if (Object.keys(svh).length > 0 || tm.slow || tm.fast || Object.keys(flags).length > 0) {
        html += '<div class="section"><h2>Analysis</h2><div class="grid">';

        if (Object.keys(svh).length > 0) {
            html += '<div class="panel"><h3>Selection &times; Value</h3>';
            html += renderHeatmap(svh);
            html += '</div>';
        }

        if (tm.slow || tm.fast) {
            html += '<div class="panel"><h3>Thinking Mode</h3>';
            html += '<table><thead><tr><th></th><th>Slow</th><th>Fast</th></tr></thead><tbody>';
            const s = tm.slow || {}; const f = tm.fast || {};
            html += `<tr><td>Count</td><td>${s.count || 0}</td><td>${f.count || 0}</td></tr>`;
            html += `<tr><td>Value</td><td style="color:${scoreColor(s.mean_value)}">${(s.mean_value || 0).toFixed(1)}</td><td style="color:${scoreColor(f.mean_value)}">${(f.mean_value || 0).toFixed(1)}</td></tr>`;
            html += `<tr><td>Quality</td><td>${(s.mean_quality || 0).toFixed(1)}</td><td>${(f.mean_quality || 0).toFixed(1)}</td></tr>`;
            html += `<tr><td>Reasoning</td><td>${(s.mean_reasoning || 0).toFixed(1)}</td><td>${(f.mean_reasoning || 0).toFixed(1)}</td></tr>`;
            html += '</tbody></table></div>';
        }

        if (Object.keys(flags).length > 0) {
            const flagItems = Object.entries(flags).sort((a, b) => b[1] - a[1]);
            const maxFlag = Math.max(...flagItems.map(x => x[1]), 1);
            let flagHtml = '';
            for (const [flag, count] of flagItems) {
                const impact = (d.flag_value_impact || {})[flag];
                const meanV = impact && impact.mean_value != null ? impact.mean_value.toFixed(1) : '-';
                const isNeg = ['has-bug', 'security-issue', 'outdated-practice', 'incomplete', 'over-engineered', 'incorrect-output', 'poor-explanation'].includes(flag);
                const color = isNeg ? '#dc2626' : '#16a34a';
                const w = (count / maxFlag * 100).toFixed(1);
                flagHtml += `<div class="bar-row">
                    <div class="bar-label" title="${flag}">${flag}</div>
                    <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
                    <div class="bar-val">${count} (avg: ${meanV})</div>
                </div>`;
            }
            html += `<div class="panel"><h3>Flags</h3>${flagHtml}</div>`;
        }

        html += '</div></div>';
    }

    // === Section 6: Selection × Tag ===
    const sbt = d.selection_by_tag || {};
    if (Object.keys(sbt).length > 0) {
        html += '<div class="section"><h2>Selection &times; Tag</h2><div class="grid">';
        const selDimColors = {'difficulty': '#ea580c', 'intent': '#3b82f6', 'domain': '#16a34a', 'concept': '#9333ea', 'task': '#db2777', 'agentic': '#ea580c'};
        const sGroups = groupDims(['difficulty', 'intent', 'domain', 'concept', 'task', 'agentic'], sbt, 10);
        for (const group of sGroups) {
            html += '<div class="panel">';
            group.forEach((dim, g) => {
                if (g > 0) html += PANEL_SEP;
                const items = Object.entries(sbt[dim]).map(([t, info]) => [t, info.mean, info.n]).sort((a, b) => b[1] - a[1]);
                html += `<h3>Selection by ${dim} ${TAG_BADGE(items.length)}</h3>${renderBarChart(items, 10, selDimColors[dim] || '#db2777')}`;
            });
            html += '</div>';
        }
        html += '</div></div>';
    }

    // === Section 7: Coverage Impact (if available) ===
    const cov = d.coverage_at_thresholds || {};
    if (Object.keys(cov).length > 0) {
        html += '<div class="section"><h2>Coverage Impact Analysis</h2>';
        html += '<div class="panel"><table><thead><tr><th>Threshold</th><th>Retained</th><th>%</th><th>Coverage</th><th>Tags Lost</th></tr></thead><tbody>';
        for (const [thresh, info] of Object.entries(cov).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
            const lost = (info.tags_lost || []).slice(0, 5).join(', ');
            const lostMore = (info.tags_lost || []).length > 5 ? ` +${info.tags_lost.length - 5} more` : '';
            html += `<tr><td>&ge; ${thresh}</td><td>${info.retained}</td><td>${(info.pct * 100).toFixed(1)}%</td><td>${(info.coverage * 100).toFixed(1)}%</td><td><span class="tag">${lost}${lostMore}</span></td></tr>`;
        }
        html += '</tbody></table></div></div>';
    }

    // === Section 9: File Ranking (global dashboard) ===
    const pfs = d.per_file_summary || [];
    if (pfs.length > 1) {
        html += '<div class="section"><h2>File Ranking</h2>';
        html += '<div class="panel"><table id="file-table"><thead><tr>';
        html += '<th onclick="sortTable(0)">File</th><th onclick="sortTable(1)">Count</th><th onclick="sortTable(2)">Value</th><th onclick="sortTable(3)">Complexity</th><th onclick="sortTable(4)">Quality</th><th onclick="sortTable(5)">Rarity</th><th onclick="sortTable(6)">Selection</th>';
        html += '</tr></thead><tbody>';
        for (const f of pfs.sort((a, b) => (b.mean_value || 0) - (a.mean_value || 0))) {
            html += `<tr>
                <td>${f.file}</td><td>${f.count}</td>
                <td style="color:${scoreColor(f.mean_value)}">${(f.mean_value || 0).toFixed(1)}</td>
                <td>${(f.mean_complexity || 0).toFixed(1)}</td>
                <td>${(f.mean_quality || 0).toFixed(1)}</td>
                <td>${(f.mean_rarity || 0).toFixed(1)}</td>
                <td style="color:${scoreColor(f.mean_selection || 0)}">${(f.mean_selection || 0).toFixed(1)}</td>
            </tr>`;
        }
        html += '</tbody></table></div></div>';
    }

    // Weights used
    const w = d.weights_used || {};
    if (Object.keys(w).length > 0) {
        html += '<div class="section"><h2>Configuration</h2><div class="panel">';
        html += '<p style="color:#6b7280;font-size:0.85em">Value Score = ';
        html += Object.entries(w).map(([k, v]) => `${v}&times;${k}`).join(' + ');
        html += '</p></div></div>';
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

// ── Conversation-level rendering ──
function renderConversations() {
    if (!DATA_CONV) return;
    const el = document.getElementById('dashboard');
    let html = el.innerHTML;

    html += '<hr class="pass-divider">';
    html += '<div class="section"><h2>Conversation-Level Aggregation</h2>';

    // Overview cards
    html += '<div class="cards">';
    html += `<div class="card"><div class="label">Conversations</div><div class="value">${DATA_CONV.total}</div></div>`;
    html += `<div class="card"><div class="label">Mean Conv Value</div><div class="value" style="color:${scoreColor(DATA_CONV.mean_conv_value)}">${DATA_CONV.mean_conv_value.toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Conv Selection</div><div class="value" style="color:${scoreColor(DATA_CONV.mean_conv_selection)}">${DATA_CONV.mean_conv_selection.toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Peak Complexity</div><div class="value" style="color:${scoreColor(DATA_CONV.mean_peak_complexity)}">${DATA_CONV.mean_peak_complexity.toFixed(1)}</div></div>`;
    html += `<div class="card"><div class="label">Mean Turns</div><div class="value">${DATA_CONV.mean_turns.toFixed(1)}</div></div>`;
    html += '</div>';

    // Histograms
    const convHistKeys = [
        ['conv_value_hist', 'Conv Value', '#3b82f6'],
        ['conv_selection_hist', 'Conv Selection', '#db2777'],
        ['peak_complexity_hist', 'Peak Complexity', '#ea580c'],
    ];
    html += '<div class="grid">';
    for (const [key, label, color] of convHistKeys) {
        const bins = DATA_CONV[key];
        if (bins) {
            html += `<div class="panel">${renderHistogram(bins, label, color, {})}</div>`;
        }
    }

    // Turn distribution
    const td = DATA_CONV.turn_distribution || {};
    if (Object.keys(td).length > 0) {
        const items = Object.entries(td).map(([k, v]) => [k + ' turns', v]).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
        html += `<div class="panel"><h3>Turn Count Distribution</h3>${renderBarChart(items, null, '#0891b2')}</div>`;
    }
    html += '</div></div>';

    el.innerHTML = html;
}

// ── Build subtitle and render ──
const parts = [];
if (DATA_P1) parts.push(`${DATA_P1.total} labeled`);
parts.push(`${(DATA_P2.overview || {}).total_scored || 0} scored`);
const inputFile = (DATA_P2.overview || {}).input_file || (DATA_P1 && DATA_P1.input_file) || '';
if (inputFile) parts.push(inputFile);
document.getElementById('subtitle').textContent = parts.join(' \u00b7 ');

if (DATA_P1) renderPass1(DATA_P1);
renderPass2();
renderConversations();
</script>
</body>
</html>"""


def generate_value_dashboard(run_dir, scored_file="scored.json",
                              stats_file="stats_value.json",
                              output_file="dashboard_value.html",
                              quiet=False):
    """Generate a combined labeling + value scoring dashboard HTML file.

    Automatically discovers Pass 1 data (stats.json / labeled.json) in the
    same directory and includes labeling sections at the top. Falls back to
    Pass 2 only if Pass 1 data is not found.

    Works in two modes:
    - Per-file: scored_file + stats_file (full data with histograms)
    - Global: scored_file=None, stats_file only (stats-based, no histograms)
    """
    run_dir = Path(run_dir)
    samples, stats = load_value_run(run_dir, scored_file, stats_file)
    viz_data_p2 = compute_value_viz_data(samples, stats)

    # Try to load Pass 1 data from the same directory
    is_global = scored_file is None
    viz_data_p1 = _load_pass1_data(run_dir, is_global)

    # Load conversation-level aggregation data
    conv_records = _load_conversation_scores(run_dir)
    viz_data_conv = _compute_conv_viz_data(conv_records)

    html = COMBINED_HTML_TEMPLATE.replace(
        "__DATA_P1_PLACEHOLDER__",
        json.dumps(viz_data_p1, ensure_ascii=False) if viz_data_p1 else "null"
    ).replace(
        "__DATA_P2_PLACEHOLDER__",
        json.dumps(viz_data_p2, ensure_ascii=False)
    ).replace(
        "__DATA_CONV_PLACEHOLDER__",
        json.dumps(viz_data_conv, ensure_ascii=False) if viz_data_conv else "null"
    )

    out = run_dir / output_file
    out.write_text(html, encoding="utf-8")
    if not quiet:
        print(f"  Dashboard: {out}")
