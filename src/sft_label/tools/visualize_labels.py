"""
Label Statistics Dashboard Generator

Reads labeled.json from a pipeline run and generates a standalone HTML dashboard
with tag distributions, confidence heatmap, coverage analysis, and cross-dimension matrix.

Usage:
  python3 labeling/tools/visualize_labels.py labeling/data/runs/<run_dir>
  python3 labeling/tools/visualize_labels.py labeling/data/runs/<run_dir> --open
"""

import json
import sys
import argparse
import webbrowser
from pathlib import Path
from collections import Counter

from sft_label.prompts import TAG_POOLS, SINGLE_SELECT, MULTI_SELECT

DIMENSIONS = ["intent", "difficulty", "language", "domain", "concept",
              "task", "agentic", "constraint", "context"]


def load_run(run_dir: Path, labeled_file="labeled.json", stats_file="stats.json"):
    """Load samples and stats from a run directory.

    If labeled_file is None or doesn't exist, returns empty samples list
    (stats-only mode for global dashboards).
    """
    samples = []
    if labeled_file:
        labeled_path = run_dir / labeled_file
        if labeled_path.exists():
            with open(labeled_path, encoding="utf-8") as f:
                samples = json.load(f)

    stats = {}
    stats_path = run_dir / stats_file
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)

    return samples, stats


def compute_viz_data(samples, stats):
    """Compute all data needed for the dashboard."""
    # Tag distributions (from stats or recompute)
    distributions = stats.get("tag_distributions", {})

    # Confidence per dimension per sample (for heatmap)
    conf_matrix = []
    for s in samples:
        labels = s.get("labels") or {}
        conf = labels.get("confidence", {})
        conf_matrix.append({
            "id": s.get("id", "?"),
            "values": {d: conf.get(d, 0) for d in DIMENSIONS}
        })

    # Coverage: used tags vs pool
    coverage = {}
    for dim in DIMENSIONS:
        pool = set(TAG_POOLS.get(dim, []))
        used = set(distributions.get(dim, {}).keys())
        coverage[dim] = {
            "pool_size": len(pool),
            "used": len(used & pool),
            "unused": sorted(pool - used),
            "rate": len(used & pool) / len(pool) if pool else 0,
        }

    # Cross-dimension: intent × difficulty matrix
    cross = Counter()
    for s in samples:
        labels = s.get("labels") or {}
        intent = labels.get("intent", "?")
        diff = labels.get("difficulty", "?")
        cross[(intent, diff)] += 1

    # Fall back to stats cross_matrix when samples are empty (global dashboard)
    if not cross and stats.get("cross_matrix"):
        cross_data = stats["cross_matrix"]
        for key, count in cross_data.items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                cross[tuple(parts)] = count

    intents = sorted({k[0] for k in cross})
    diffs = ["beginner", "intermediate", "advanced", "expert"]
    cross_matrix = {
        "rows": intents,
        "cols": [d for d in diffs if any(cross.get((i, d), 0) for i in intents)],
        "data": {f"{i}|{d}": cross.get((i, d), 0) for i in intents for d in diffs},
    }

    return {
        "total": stats.get("total_samples", len(samples)) or len(samples),
        "distributions": distributions,
        "confidence_stats": stats.get("confidence_stats", {}),
        "conf_matrix": conf_matrix,
        "coverage": coverage,
        "cross_matrix": cross_matrix,
        "overview": {
            "success_rate": stats.get("success_rate", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "arbitrated_rate": stats.get("arbitrated_rate", 0),
            "unmapped_unique": stats.get("unmapped_unique_count", 0),
        },
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Label Statistics Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f7; color: #1d1d1f; }
.header { background: linear-gradient(135deg, #1d1d1f 0%, #2d3748 100%); color: white; padding: 20px 28px; }
.header h1 { font-size: 1.4em; font-weight: 700; }
.header .sub { font-size: 0.82em; color: #a1a1aa; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.overview { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.card .label { font-size: 0.75em; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 1.6em; font-weight: 700; margin-top: 4px; }
.section { margin-bottom: 28px; }
.section h2 { font-size: 1.1em; font-weight: 600; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #e5e7eb; }
.dim-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.dim-card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.dim-card h3 { font-size: 0.95em; font-weight: 600; margin-bottom: 10px; display: flex; justify-content: space-between; }
.dim-card h3 .badge { font-size: 0.75em; background: #f3f4f6; padding: 2px 8px; border-radius: 4px; font-weight: 500; color: #6b7280; }
.bar-row { display: flex; align-items: center; margin-bottom: 4px; font-size: 0.82em; }
.bar-label { width: 130px; text-align: right; padding-right: 8px; color: #374151; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; height: 18px; background: #f3f4f6; border-radius: 3px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.bar-count { width: 50px; text-align: right; font-size: 0.78em; color: #6b7280; padding-left: 6px; }
/* Confidence heatmap */
.conf-table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
.conf-table th { padding: 6px 8px; text-align: center; font-weight: 600; background: #f9fafb; }
.conf-table td { padding: 5px 8px; text-align: center; border: 1px solid #f3f4f6; }
.conf-cell { border-radius: 4px; padding: 3px 6px; font-weight: 500; }
/* Coverage */
.cov-row { display: flex; align-items: center; margin-bottom: 6px; font-size: 0.82em; }
.cov-label { width: 90px; font-weight: 500; }
.cov-bar { flex: 1; height: 14px; background: #f3f4f6; border-radius: 3px; overflow: hidden; }
.cov-fill { height: 100%; background: #3b82f6; border-radius: 3px; }
.cov-text { width: 100px; text-align: right; font-size: 0.78em; color: #6b7280; }
/* Cross matrix */
.cross-table { border-collapse: collapse; font-size: 0.82em; }
.cross-table th, .cross-table td { padding: 6px 14px; text-align: center; border: 1px solid #e5e7eb; }
.cross-table th { background: #f9fafb; font-weight: 600; }
.cross-cell { border-radius: 4px; padding: 2px 8px; }
/* Unused tags */
.unused-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.unused-tag { font-size: 0.72em; background: #fef2f2; color: #991b1b; padding: 2px 6px; border-radius: 3px; }
</style>
</head>
<body>
<div class="header">
  <h1>Label Statistics Dashboard</h1>
  <div class="sub" id="subtitle"></div>
</div>
<div class="container">
  <div class="overview" id="overview"></div>
  <div class="section"><h2>Tag Distributions</h2><div class="dim-grid" id="distributions"></div></div>
  <div class="section"><h2>Confidence Summary</h2><div class="card" id="confidence"></div></div>
  <div class="section"><h2>Intent × Difficulty</h2><div class="card" id="cross"></div></div>
  <div class="section"><h2>Tag Pool Coverage</h2><div class="card" id="coverage"></div></div>
</div>
<script>
const DATA = __DATA_PLACEHOLDER__;

const COLORS = {
  intent: '#3b82f6', language: '#8b5cf6', domain: '#06b6d4', task: '#f59e0b',
  difficulty: '#10b981', concept: '#ec4899', agentic: '#f97316', constraint: '#6366f1', context: '#14b8a6'
};

function confColor(v) {
  if (v >= 0.9) return '#dcfce7';
  if (v >= 0.8) return '#fef9c3';
  if (v >= 0.7) return '#fed7aa';
  return '#fecaca';
}

function render() {
  const d = DATA;
  document.getElementById('subtitle').textContent =
    `${d.total} samples`;

  // Overview cards
  const ov = d.overview;
  document.getElementById('overview').innerHTML = [
    ['Samples', d.total],
    ['Success', (ov.success_rate * 100).toFixed(1) + '%'],
    ['Tokens', ov.total_tokens.toLocaleString()],
    ['Arbitrated', (ov.arbitrated_rate * 100).toFixed(1) + '%'],
    ['Unmapped', ov.unmapped_unique],
  ].map(([l, v]) => `<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');

  // Distributions
  const distHtml = [];
  for (const dim of Object.keys(d.distributions)) {
    const dist = d.distributions[dim];
    const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
    const maxVal = entries.length ? entries[0][1] : 1;
    const total = entries.reduce((s, e) => s + e[1], 0);
    const color = COLORS[dim] || '#6b7280';
    let bars = entries.map(([tag, count]) => {
      const pct = (count / maxVal * 100).toFixed(0);
      return `<div class="bar-row">
        <div class="bar-label" title="${tag}">${tag}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="bar-count">${count} (${(count/total*100).toFixed(0)}%)</div>
      </div>`;
    }).join('');
    distHtml.push(`<div class="dim-card"><h3>${dim} <span class="badge">${entries.length} tags</span></h3>${bars}</div>`);
  }
  document.getElementById('distributions').innerHTML = distHtml.join('');

  // Confidence summary table
  const dims = Object.keys(d.confidence_stats);
  let confHtml = '<table class="conf-table"><tr><th>Dimension</th><th>Mean</th><th>Min</th><th>Max</th><th>Below Threshold</th></tr>';
  for (const dim of dims) {
    const cs = d.confidence_stats[dim];
    confHtml += `<tr>
      <td style="font-weight:600">${dim}</td>
      <td><span class="conf-cell" style="background:${confColor(cs.mean)}">${cs.mean.toFixed(3)}</span></td>
      <td>${cs.min.toFixed(2)}</td><td>${cs.max.toFixed(2)}</td>
      <td>${cs.below_threshold}</td>
    </tr>`;
  }
  confHtml += '</table>';
  document.getElementById('confidence').innerHTML = confHtml;

  // Cross matrix
  const cm = d.cross_matrix;
  let crossHtml = '<table class="cross-table"><tr><th></th>';
  cm.cols.forEach(c => crossHtml += `<th>${c}</th>`);
  crossHtml += '<th>Total</th></tr>';
  for (const row of cm.rows) {
    crossHtml += `<tr><th>${row}</th>`;
    let rowTotal = 0;
    for (const col of cm.cols) {
      const v = cm.data[`${row}|${col}`] || 0;
      rowTotal += v;
      const bg = v > 0 ? `background:${COLORS.intent}22` : '';
      crossHtml += `<td><span class="cross-cell" style="${bg}">${v || '-'}</span></td>`;
    }
    crossHtml += `<td style="font-weight:600">${rowTotal}</td></tr>`;
  }
  crossHtml += '</table>';
  document.getElementById('cross').innerHTML = crossHtml;

  // Coverage
  let covHtml = '';
  for (const dim of Object.keys(d.coverage)) {
    const c = d.coverage[dim];
    if (!c.pool_size) continue;
    const pct = (c.rate * 100).toFixed(0);
    covHtml += `<div class="cov-row">
      <div class="cov-label">${dim}</div>
      <div class="cov-bar"><div class="cov-fill" style="width:${pct}%"></div></div>
      <div class="cov-text">${c.used}/${c.pool_size} (${pct}%)</div>
    </div>`;
    if (c.unused.length && c.unused.length <= 20) {
      covHtml += `<div style="margin-left:90px;margin-bottom:8px"><div class="unused-tags">${c.unused.map(t => `<span class="unused-tag">${t}</span>`).join('')}</div></div>`;
    } else if (c.unused.length > 20) {
      covHtml += `<div style="margin-left:90px;margin-bottom:8px;font-size:0.75em;color:#991b1b">${c.unused.length} unused tags (expand dataset for better coverage)</div>`;
    }
  }
  document.getElementById('coverage').innerHTML = covHtml;
}

render();
</script>
</body>
</html>"""


def generate_dashboard(run_dir: Path, labeled_file="labeled.json",
                       stats_file="stats.json", output_file="dashboard.html") -> Path:
    """Generate dashboard HTML. Supports stats-only mode (no labeled.json)."""
    run_dir = Path(run_dir)
    samples, stats = load_run(run_dir, labeled_file=labeled_file, stats_file=stats_file)
    viz_data = compute_viz_data(samples, stats)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(viz_data, ensure_ascii=False))
    out = run_dir / output_file
    out.write_text(html, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate label statistics dashboard")
    parser.add_argument("run_dir", help="Path to pipeline run directory")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "labeled.json").exists():
        print(f"Error: {run_dir}/labeled.json not found")
        sys.exit(1)

    out = generate_dashboard(run_dir)
    print(f"Dashboard: {out}")
    if args.open:
        webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
