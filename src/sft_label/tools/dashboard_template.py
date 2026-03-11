"""Shared HTML template for interactive labeling/scoring dashboards."""

from __future__ import annotations

import json


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f4f3ee;
  --panel: #fffdf8;
  --panel-2: #f8f5eb;
  --text: #1d261f;
  --muted: #667264;
  --line: #dfd8c7;
  --accent: #1f6f5f;
  --accent-2: #c66b3d;
  --blue: #2563eb;
  --green: #15803d;
  --amber: #d97706;
  --red: #dc2626;
  --shadow: 0 10px 30px rgba(54, 49, 39, 0.08);
  --sidebar-open: clamp(280px, 18vw, 360px);
  --sidebar-closed: 72px;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: linear-gradient(180deg, #f7f4ea 0%, var(--bg) 100%); color: var(--text); font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif; }
body { min-height: 100vh; }
.shell { width: 100%; max-width: none; margin: 0 auto; padding: 24px clamp(18px, 1.8vw, 40px); }
.hero { background: linear-gradient(135deg, #18382f 0%, #35564d 42%, #78553c 100%); color: #f9f7f0; border-radius: 24px; padding: 24px 28px; box-shadow: var(--shadow); position: sticky; top: 0; z-index: 10; }
.hero-top { display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }
.hero h1 { margin: 0; font-size: 1.7rem; letter-spacing: 0.02em; }
.hero-sub { margin-top: 8px; color: rgba(249, 247, 240, 0.78); font-size: 0.96rem; }
.hero-stats { display: flex; gap: 10px; flex-wrap: wrap; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px; background: rgba(255,255,255,0.12); font-size: 0.84rem; }
.pill strong { font-size: 0.98rem; color: #fff; }
.layout { display: grid; grid-template-columns: var(--sidebar-open) minmax(0, 1fr); gap: clamp(18px, 1.5vw, 28px); margin-top: 18px; align-items: start; transition: grid-template-columns 220ms ease, gap 220ms ease; }
.shell.sidebar-collapsed .layout { grid-template-columns: var(--sidebar-closed) minmax(0, 1fr); gap: 16px; }
.sidebar { background: rgba(255, 253, 248, 0.88); backdrop-filter: blur(8px); border: 1px solid rgba(223,216,199,0.9); border-radius: 20px; box-shadow: var(--shadow); padding: 18px; height: calc(100vh - 190px); position: sticky; top: 148px; overflow: auto; transition: padding 220ms ease, background 220ms ease, box-shadow 220ms ease; }
.shell.sidebar-collapsed .sidebar { padding: 14px 10px; overflow: hidden; }
.sidebar h2, .main h2 { margin: 0; font-size: 1rem; }
.sidebar-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
.sidebar-copy { min-width: 0; flex: 1; transition: opacity 180ms ease, transform 220ms ease, max-width 220ms ease; }
.sidebar-sub { margin-top: 4px; color: var(--muted); font-size: 0.78rem; }
.sidebar-body { transition: opacity 180ms ease, transform 220ms ease, max-height 220ms ease, margin 220ms ease; transform-origin: top left; }
.sidebar-toggle { width: 36px; height: 36px; flex: 0 0 36px; border-radius: 999px; border: 1px solid rgba(223,216,199,0.95); background: rgba(255,255,255,0.82); color: var(--accent); cursor: pointer; font: inherit; font-size: 1rem; line-height: 1; box-shadow: 0 6px 16px rgba(54, 49, 39, 0.08); transition: transform 180ms ease, border-color 180ms ease, background 180ms ease; }
.sidebar-toggle:hover { transform: translateX(-1px); border-color: rgba(31,111,95,0.4); background: #fff; }
.shell.sidebar-collapsed .sidebar-header { justify-content: center; margin-bottom: 0; }
.shell.sidebar-collapsed .sidebar-copy { opacity: 0; transform: translateX(-8px); max-width: 0; overflow: hidden; pointer-events: none; }
.shell.sidebar-collapsed .sidebar-body { opacity: 0; transform: translateX(-8px); max-height: 0; overflow: hidden; margin: 0; pointer-events: none; }
.shell.sidebar-collapsed .sidebar-toggle:hover { transform: translateX(1px); }
.search { width: 100%; margin: 14px 0 10px; padding: 11px 13px; border-radius: 12px; border: 1px solid var(--line); background: #fff; color: var(--text); font: inherit; }
.filter-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.filter-btn { border: 1px solid var(--line); background: var(--panel); color: var(--muted); border-radius: 999px; padding: 7px 11px; font: inherit; cursor: pointer; }
.filter-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.tree { display: flex; flex-direction: column; gap: 4px; }
.tree-node { display: flex; flex-direction: column; gap: 4px; }
.tree-row { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 12px; cursor: pointer; color: var(--text); border: 1px solid transparent; }
.tree-row:hover { background: rgba(31, 111, 95, 0.08); }
.tree-row.active { background: rgba(31, 111, 95, 0.14); border-color: rgba(31, 111, 95, 0.22); }
.tree-indent { width: 14px; flex: 0 0 14px; }
.tree-toggle { width: 18px; text-align: center; color: var(--muted); font-size: 0.78rem; }
.tree-label { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.92rem; }
.tree-meta { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 0.75rem; }
.kind-badge { display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 2px 8px; font-size: 0.72rem; letter-spacing: 0.02em; }
.kind-global { background: rgba(31, 111, 95, 0.12); color: var(--accent); }
.kind-dir { background: rgba(198, 107, 61, 0.12); color: var(--accent-2); }
.kind-file { background: rgba(37, 99, 235, 0.12); color: var(--blue); }
.tree-children { display: none; padding-left: 12px; }
.tree-node.expanded > .tree-children { display: block; }
.main { min-width: 0; display: flex; flex-direction: column; gap: 18px; }
.panel { background: rgba(255, 253, 248, 0.95); border: 1px solid rgba(223,216,199,0.9); border-radius: 22px; box-shadow: var(--shadow); padding: 20px; }
.scope-header { display: flex; gap: 14px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }
.scope-title { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.scope-title h2 { font-size: 1.45rem; }
.scope-path { color: var(--muted); font-size: 0.92rem; word-break: break-all; }
.breadcrumbs { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; color: var(--muted); font-size: 0.84rem; }
.crumb-btn { border: none; background: none; padding: 0; color: var(--accent); cursor: pointer; font: inherit; }
.scope-actions { display: flex; gap: 8px; flex-wrap: wrap; }
.action-btn { border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 999px; padding: 8px 12px; font: inherit; cursor: pointer; }
.action-btn:hover { border-color: var(--accent); color: var(--accent); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border: 1px solid rgba(223,216,199,0.9); border-radius: 18px; padding: 14px; }
.card-label { font-size: 0.76rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.card-value { margin-top: 6px; font-size: 1.6rem; font-weight: 700; line-height: 1; }
.card-sub { margin-top: 6px; color: var(--muted); font-size: 0.82rem; }
.section-card { border: 1px solid rgba(223,216,199,0.9); border-radius: 18px; background: rgba(255,255,255,0.55); overflow: hidden; }
.section-card + .section-card { margin-top: 14px; }
.section-summary { list-style: none; display: flex; justify-content: space-between; align-items: center; gap: 12px; cursor: pointer; padding: 16px 18px; background: linear-gradient(90deg, rgba(31,111,95,0.08) 0%, rgba(198,107,61,0.06) 100%); }
.section-summary::-webkit-details-marker { display: none; }
.section-title { font-size: 1rem; font-weight: 700; }
.section-meta { color: var(--muted); font-size: 0.82rem; }
.section-body { padding: 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }
.mini-panel { border: 1px solid rgba(223,216,199,0.9); border-radius: 16px; background: rgba(255,253,248,0.9); padding: 14px; }
.mini-panel h4 { margin: 0 0 10px; font-size: 0.98rem; }
.bar-row { display: flex; gap: 8px; align-items: center; font-size: 0.84rem; margin-bottom: 5px; }
.bar-label { width: 120px; text-align: right; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; height: 16px; border-radius: 999px; background: #ece6d7; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-value { min-width: 82px; text-align: right; color: var(--muted); font-size: 0.78rem; }
.hist-row { display: flex; align-items: flex-end; gap: 4px; height: 120px; margin-top: 10px; }
.hist-bar { flex: 1; min-height: 2px; border-radius: 6px 6px 0 0; position: relative; }
.hist-labels { display: flex; gap: 4px; margin-top: 8px; color: var(--muted); font-size: 0.72rem; }
.hist-labels span { flex: 1; text-align: center; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.data-table th, .data-table td { padding: 10px 12px; border-top: 1px solid #e7e0d2; text-align: left; vertical-align: top; }
.data-table th { color: var(--muted); font-weight: 600; background: rgba(248,245,235,0.72); }
.data-table tr:hover td { background: rgba(31,111,95,0.05); }
.th-sort-btn { width: 100%; display: inline-flex; align-items: center; justify-content: space-between; gap: 8px; border: none; background: none; padding: 0; color: inherit; font: inherit; cursor: pointer; }
.th-sort-btn.active { color: var(--accent); }
.sort-indicator { color: var(--muted); font-size: 0.78rem; }
.compact-table th, .compact-table td { padding: 7px 8px; font-size: 0.8rem; }
.rank-cell { width: 42px; color: var(--muted); }
.metric-cell { white-space: nowrap; text-align: right; }
.link-btn { border: none; background: none; padding: 0; color: var(--accent); cursor: pointer; font: inherit; text-align: left; }
.tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.tag { display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; background: rgba(220, 38, 38, 0.08); color: #9a3412; font-size: 0.74rem; }
.note { color: var(--muted); font-size: 0.84rem; }
.empty { padding: 20px; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 14px; background: rgba(255,255,255,0.45); }
.value-strong { color: var(--green); }
.value-mid { color: var(--blue); }
.value-warn { color: var(--amber); }
.value-low { color: var(--red); }
@media (max-width: 1080px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; }
  .shell.sidebar-collapsed .layout { grid-template-columns: 1fr; }
  .shell.sidebar-collapsed .sidebar { padding: 14px 14px 10px; }
}
@media (min-width: 1800px) {
  .cards { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .grid { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
}
</style>
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
          <h2>Scope Navigator</h2>
          <div class="sidebar-sub">Global, folders, and files</div>
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
            <button class="action-btn" id="go-parent">Up One Level</button>
            <button class="action-btn" id="go-global">Back To Global</button>
          </div>
        </div>
      </section>
      <section class="panel" id="scope-content"></section>
    </main>
  </div>
</div>
<script>
const DATA = __DATA_PLACEHOLDER__;
const STATE = {
  currentId: DATA.default_scope_id || DATA.root_id,
  filter: "all",
  query: "",
  expanded: new Set(DATA.initially_expanded || [DATA.root_id]),
  sidebarCollapsed: false,
  fileRankingSort: { key: "mean_value", direction: "desc" },
};

function loadSidebarPreference() {
  try {
    return window.localStorage.getItem("dashboard.sidebarCollapsed") === "1";
  } catch (error) {
    return false;
  }
}

function applySidebarState() {
  const shell = document.querySelector(".shell");
  const button = document.getElementById("sidebar-toggle");
  const collapsed = STATE.sidebarCollapsed;
  shell.classList.toggle("sidebar-collapsed", collapsed);
  button.textContent = collapsed ? ">" : "<";
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");
  button.setAttribute("title", collapsed ? "Expand Navigator" : "Collapse Navigator");
}

function toggleSidebar() {
  STATE.sidebarCollapsed = !STATE.sidebarCollapsed;
  try {
    window.localStorage.setItem("dashboard.sidebarCollapsed", STATE.sidebarCollapsed ? "1" : "0");
  } catch (error) {
    // Ignore storage failures; sidebar toggle should still work.
  }
  applySidebarState();
}

const DIM_COLORS = {
  intent: "#2563eb",
  language: "#7c3aed",
  domain: "#0f766e",
  task: "#c2410c",
  difficulty: "#15803d",
  concept: "#be185d",
  agentic: "#9a3412",
  constraint: "#4f46e5",
  context: "#0d9488",
};

const HIST_COLORS = {
  value_score: "#2563eb",
  complexity_overall: "#c2410c",
  quality_overall: "#15803d",
  reasoning_overall: "#7c3aed",
  rarity_score: "#0f766e",
  selection_score: "#be185d",
};

const HIST_LABELS = {
  value_score: "Value Score",
  complexity_overall: "Complexity",
  quality_overall: "Quality",
  reasoning_overall: "Reasoning",
  rarity_score: "Rarity",
  selection_score: "Selection",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function scoreClass(value) {
  if (value >= 8) return "value-strong";
  if (value >= 6) return "value-mid";
  if (value >= 4) return "value-warn";
  return "value-low";
}

function confColor(value) {
  if (value >= 0.9) return "#d1fae5";
  if (value >= 0.8) return "#fef3c7";
  if (value >= 0.7) return "#fed7aa";
  return "#fecaca";
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function getScope(id) {
  return DATA.scopes[id];
}

function getCurrentScope() {
  return getScope(STATE.currentId) || getScope(DATA.root_id);
}

function toggleExpanded(id) {
  if (STATE.expanded.has(id)) STATE.expanded.delete(id);
  else STATE.expanded.add(id);
  renderTree();
}

function setScope(id) {
  if (!DATA.scopes[id]) return;
  STATE.currentId = id;
  let cursor = DATA.scopes[id];
  while (cursor && cursor.parent_id) {
    STATE.expanded.add(cursor.parent_id);
    cursor = DATA.scopes[cursor.parent_id];
  }
  render();
}

function matchesFilter(scope) {
  if (!scope) return false;
  if (STATE.filter === "all") return true;
  if (STATE.filter === "dir") return scope.kind === "dir" || scope.kind === "global";
  return scope.kind === "file";
}

function matchesQuery(scope) {
  if (!STATE.query) return true;
  const haystack = `${scope.label} ${scope.path}`.toLowerCase();
  return haystack.includes(STATE.query);
}

function subtreeMatches(id) {
  const scope = getScope(id);
  if (!scope) return false;
  if (matchesFilter(scope) && matchesQuery(scope)) return true;
  return (scope.children || []).some(subtreeMatches);
}

function renderTreeNode(id, depth = 0) {
  const scope = getScope(id);
  if (!scope || !subtreeMatches(id)) return "";

  const isExpanded = STATE.expanded.has(id) || id === DATA.root_id;
  const hasChildren = (scope.children || []).length > 0;
  const summary = scope.summary || {};
  const valueMean = summary.mean_value;
  const badgeClass = `kind-badge kind-${scope.kind === "global" ? "global" : (scope.kind === "dir" ? "dir" : "file")}`;
  const metaBits = [];
  if (summary.file_count) metaBits.push(`${summary.file_count} files`);
  if (summary.scored_total) metaBits.push(`v ${fmt(summary.mean_value, 1)}`);
  else if (summary.pass1_total) metaBits.push(`${summary.pass1_total} labeled`);

  let html = `<div class="tree-node ${isExpanded ? "expanded" : ""}">`;
  html += `<div class="tree-row ${STATE.currentId === id ? "active" : ""}" data-scope="${escapeHtml(id)}">`;
  html += `<span class="tree-indent" style="margin-left:${depth * 10}px"></span>`;
  if (hasChildren) {
    html += `<span class="tree-toggle" data-toggle="${escapeHtml(id)}">${isExpanded ? "▾" : "▸"}</span>`;
  } else {
    html += `<span class="tree-toggle"></span>`;
  }
  html += `<span class="${badgeClass}">${scope.kind}</span>`;
  html += `<span class="tree-label" title="${escapeHtml(scope.path || scope.label)}">${escapeHtml(scope.label)}</span>`;
  if (metaBits.length) {
    html += `<span class="tree-meta">${escapeHtml(metaBits.join(" · "))}</span>`;
  }
  html += `</div>`;
  if (hasChildren) {
    html += `<div class="tree-children">`;
    for (const childId of scope.children) {
      html += renderTreeNode(childId, depth + 1);
    }
    html += `</div>`;
  }
  html += `</div>`;
  return html;
}

function renderTree() {
  const tree = document.getElementById("tree");
  tree.innerHTML = renderTreeNode(DATA.root_id);

  tree.querySelectorAll("[data-scope]").forEach((node) => {
    node.addEventListener("click", (event) => {
      const toggleId = event.target.getAttribute("data-toggle");
      if (toggleId) {
        toggleExpanded(toggleId);
        event.stopPropagation();
        return;
      }
      const scopeId = node.getAttribute("data-scope");
      const scope = getScope(scopeId);
      if (scope && (scope.children || []).length > 0 && STATE.currentId === scopeId) {
        toggleExpanded(scopeId);
      }
      setScope(scopeId);
    });
  });
  tree.querySelectorAll("[data-toggle]").forEach((node) => {
    node.addEventListener("click", (event) => {
      toggleExpanded(node.getAttribute("data-toggle"));
      event.stopPropagation();
    });
  });
}

function renderHero() {
  document.getElementById("hero-title").textContent = DATA.title || "Interactive Dashboard";
  document.getElementById("hero-subtitle").textContent = DATA.subtitle || "";

  const root = getScope(DATA.root_id);
  const summary = (root && root.summary) || {};
  const heroStats = [
    ["Scopes", Object.keys(DATA.scopes || {}).length],
    ["Files", summary.file_count || 0],
  ];
  if (summary.pass1_total) heroStats.push(["Labeled", summary.pass1_total]);
  if (summary.scored_total) heroStats.push(["Scored", summary.scored_total]);
  if (summary.mean_value !== null && summary.mean_value !== undefined) heroStats.push(["Mean Value", fmt(summary.mean_value, 1)]);
  document.getElementById("hero-stats").innerHTML = heroStats
    .map(([label, value]) => `<div class="pill"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function breadcrumbIds(scope) {
  const ids = [];
  let cursor = scope;
  while (cursor) {
    ids.push(cursor.id);
    cursor = cursor.parent_id ? getScope(cursor.parent_id) : null;
  }
  return ids.reverse();
}

function renderBreadcrumbs(scope) {
  const parts = breadcrumbIds(scope).map((id, index, items) => {
    const item = getScope(id);
    if (index === items.length - 1) return `<span>${escapeHtml(item.label)}</span>`;
    return `<button class="crumb-btn" data-crumb="${escapeHtml(id)}">${escapeHtml(item.label)}</button><span>/</span>`;
  });
  document.getElementById("breadcrumbs").innerHTML = parts.join("");
  document.querySelectorAll("[data-crumb]").forEach((node) => {
    node.addEventListener("click", () => setScope(node.getAttribute("data-crumb")));
  });
}

function section(title, body, meta = "", open = true) {
  return `<details class="section-card" ${open ? "open" : ""}>
    <summary class="section-summary">
      <span class="section-title">${escapeHtml(title)}</span>
      <span class="section-meta">${escapeHtml(meta)}</span>
    </summary>
    <div class="section-body">${body}</div>
  </details>`;
}

function renderCards(cards) {
  if (!cards.length) return "";
  return `<div class="cards">${
    cards.map((card) => `<div class="card">
      <div class="card-label">${escapeHtml(card.label)}</div>
      <div class="card-value ${card.className || ""}">${escapeHtml(card.value)}</div>
      ${card.sub ? `<div class="card-sub">${escapeHtml(card.sub)}</div>` : ""}
    </div>`).join("")
  }</div>`;
}

function renderBarChart(items, color, maxValue = null) {
  if (!items || !items.length) return `<div class="empty">No data</div>`;
  const ceiling = maxValue || Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return items.map((item) => {
    const pct = ceiling > 0 ? ((Number(item.value) || 0) / ceiling) * 100 : 0;
    return `<div class="bar-row">
      <div class="bar-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${color}"></div></div>
      <div class="bar-value">${escapeHtml(item.display || item.value)}</div>
    </div>`;
  }).join("");
}

function renderHistogram(label, bins, color, stats) {
  const max = Math.max(...bins, 1);
  const bars = bins.map((count, index) => {
    const height = Math.max((count / max) * 100, 2);
    return `<div class="hist-bar" style="height:${height}%;background:${color}" title="${index + 1}: ${count}"></div>`;
  }).join("");
  const labels = bins.map((_, index) => `<span>${index + 1}</span>`).join("");
  const statLine = stats ? `<div class="note">mean=${fmt(stats.mean, 2)} · min=${fmt(stats.min, 1)} · max=${fmt(stats.max, 1)}</div>` : "";
  return `<div class="mini-panel">
    <h4>${escapeHtml(label)}</h4>
    ${statLine}
    <div class="hist-row">${bars}</div>
    <div class="hist-labels">${labels}</div>
  </div>`;
}

function renderRankingTable(items, metricLabel, valueRenderer) {
  if (!items || !items.length) return `<div class="empty">No data</div>`;
  const rows = items.map((item, index) => `<tr>
    <td class="rank-cell">${index + 1}</td>
    <td>${escapeHtml(item.label)}</td>
    <td class="metric-cell">${escapeHtml(valueRenderer(item))}</td>
  </tr>`).join("");
  return `<table class="data-table compact-table">
    <thead><tr><th>#</th><th>Tag</th><th>${escapeHtml(metricLabel)}</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

const FILE_RANKING_COLUMNS = {
  file: {label: "File", defaultDirection: "asc"},
  count: {label: "Count", defaultDirection: "desc"},
  mean_value: {label: "Value", defaultDirection: "desc"},
  mean_complexity: {label: "Complexity", defaultDirection: "desc"},
  mean_quality: {label: "Quality", defaultDirection: "desc"},
  mean_selection: {label: "Selection", defaultDirection: "desc"},
};

function fileRankingIndicator(key) {
  if (STATE.fileRankingSort.key !== key) return "↕";
  return STATE.fileRankingSort.direction === "asc" ? "↑" : "↓";
}

function renderFileRankingHeader(key) {
  const column = FILE_RANKING_COLUMNS[key];
  const active = STATE.fileRankingSort.key === key;
  return `<th><button class="th-sort-btn ${active ? "active" : ""}" data-file-sort="${escapeHtml(key)}" type="button">
    <span>${escapeHtml(column.label)}</span>
    <span class="sort-indicator">${fileRankingIndicator(key)}</span>
  </button></th>`;
}

function sortPerFileSummary(rows) {
  const {key, direction} = STATE.fileRankingSort;
  const factor = direction === "asc" ? 1 : -1;
  return rows.slice().sort((left, right) => {
    if (key === "file") {
      const cmp = String(left.file || "").localeCompare(String(right.file || ""), undefined, {
        numeric: true,
        sensitivity: "base",
      });
      if (cmp !== 0) return cmp * factor;
    } else {
      const delta = (Number(left[key]) || 0) - (Number(right[key]) || 0);
      if (delta !== 0) return delta * factor;
    }
    return String(left.file || "").localeCompare(String(right.file || ""), undefined, {
      numeric: true,
      sensitivity: "base",
    });
  });
}

function toggleFileRankingSort(key) {
  const column = FILE_RANKING_COLUMNS[key];
  if (!column) return;
  if (STATE.fileRankingSort.key === key) {
    STATE.fileRankingSort.direction = STATE.fileRankingSort.direction === "desc" ? "asc" : "desc";
  } else {
    STATE.fileRankingSort = {key, direction: column.defaultDirection};
  }
  renderScope();
}

function renderChildren(scope) {
  const children = (scope.children || []).map(getScope).filter(Boolean);
  if (!children.length) return "";

  const rows = children.map((child) => {
    const summary = child.summary || {};
    const meanValue = summary.mean_value;
    const meanHtml = meanValue === null || meanValue === undefined
      ? "-"
      : `<span class="${scoreClass(meanValue)}">${fmt(meanValue, 1)}</span>`;
    return `<tr>
      <td><button class="link-btn" data-jump="${escapeHtml(child.id)}">${escapeHtml(child.label)}</button></td>
      <td><span class="kind-badge kind-${child.kind === "global" ? "global" : (child.kind === "dir" ? "dir" : "file")}">${escapeHtml(child.kind)}</span></td>
      <td>${summary.file_count || 0}</td>
      <td>${summary.pass1_total || 0}</td>
      <td>${summary.scored_total || 0}</td>
      <td>${meanHtml}</td>
    </tr>`;
  }).join("");

  return section(
    "Children",
    `<table class="data-table">
      <thead><tr><th>Name</th><th>Type</th><th>Files</th><th>Labeled</th><th>Scored</th><th>Mean Value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`,
    `${children.length} direct children`,
    true,
  );
}

function renderPass1(pass1) {
  if (!pass1) return "";
  const sections = [];

  const cards = [];
  const overview = pass1.overview || {};
  cards.push({label: "Samples", value: pass1.total || 0});
  cards.push({label: "Success", value: `${(((overview.success_rate || 0) * 100).toFixed(1))}%`});
  cards.push({label: "Tokens", value: Number(overview.total_tokens || 0).toLocaleString()});
  cards.push({label: "Arbitrated", value: `${(((overview.arbitrated_rate || 0) * 100).toFixed(1))}%`});
  if (overview.unmapped_unique) cards.push({label: "Unmapped", value: overview.unmapped_unique});
  if (overview.sparse_inherited) {
    cards.push({label: "LLM Labeled", value: overview.sparse_labeled || 0});
    cards.push({label: "Inherited", value: overview.sparse_inherited || 0});
  }
  sections.push(section("Labeling Overview", renderCards(cards), `${pass1.total || 0} samples`, true));

  const distPanels = Object.entries(pass1.distributions || {}).map(([dim, dist]) => {
    const total = Object.values(dist || {}).reduce((sum, value) => sum + value, 0);
    const entries = Object.entries(dist || {})
      .sort((a, b) => (b[1] || 0) - (a[1] || 0) || a[0].localeCompare(b[0]))
      .map(([label, value]) => ({
        label,
        value,
      }));
    return `<div class="mini-panel"><h4>${escapeHtml(dim)}</h4>${renderRankingTable(entries, "Count", (entry) => (
      total > 0 ? `${entry.value} (${((entry.value / total) * 100).toFixed(1)}%)` : `${entry.value}`
    ))}</div>`;
  }).join("");
  if (distPanels) {
    sections.push(section("Tag Distributions", `<div class="grid">${distPanels}</div>`, "All tags, sorted by frequency", true));
  }

  const confStats = Object.entries(pass1.confidence_stats || {}).map(([dim, stats]) => {
    return `<tr>
      <td>${escapeHtml(dim)}</td>
      <td><span style="display:inline-block;padding:4px 8px;border-radius:999px;background:${confColor(stats.mean)}">${fmt(stats.mean, 3)}</span></td>
      <td>${fmt(stats.min, 2)}</td>
      <td>${fmt(stats.max, 2)}</td>
      <td>${stats.below_threshold ?? 0}</td>
    </tr>`;
  }).join("");
  if (confStats) {
    sections.push(section("Confidence", `<table class="data-table"><thead><tr><th>Dimension</th><th>Mean</th><th>Min</th><th>Max</th><th>Below Threshold</th></tr></thead><tbody>${confStats}</tbody></table>`, "", false));
  }

  const cross = pass1.cross_matrix || {};
  if ((cross.rows || []).length && (cross.cols || []).length) {
    let html = `<table class="data-table"><thead><tr><th>Intent \\ Difficulty</th>${cross.cols.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}<th>Total</th></tr></thead><tbody>`;
    for (const row of cross.rows) {
      let total = 0;
      html += `<tr><td>${escapeHtml(row)}</td>`;
      for (const col of cross.cols) {
        const value = cross.data[`${row}|${col}`] || 0;
        total += value;
        html += `<td>${value || "-"}</td>`;
      }
      html += `<td>${total}</td></tr>`;
    }
    html += `</tbody></table>`;
    sections.push(section("Intent × Difficulty", html, "", false));
  }

  const coverageItems = Object.entries(pass1.coverage || {}).map(([dim, item]) => {
    if (!item.pool_size) return "";
    const pct = (item.rate || 0) * 100;
    const tags = item.unused && item.unused.length && item.unused.length <= 24
      ? `<div class="tags">${item.unused.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>`
      : `<div class="note">${item.unused?.length || 0} unused tags</div>`;
    return `<div class="mini-panel">
      <h4>${escapeHtml(dim)}</h4>
      <div class="note">${item.used}/${item.pool_size} (${pct.toFixed(1)}%)</div>
      ${renderBarChart([{label: dim, value: pct, display: `${pct.toFixed(1)}%`}], "#2563eb", 100)}
      ${tags}
    </div>`;
  }).join("");
  if (coverageItems) {
    sections.push(section("Pool Coverage", `<div class="grid">${coverageItems}</div>`, "", false));
  }

  return sections.join("");
}

function renderPass2(pass2) {
  if (!pass2) return "";
  const sections = [];
  const overview = pass2.overview || {};
  const cards = [
    {label: "Scored", value: overview.total_scored || 0},
    {label: "Failed", value: overview.total_failed || 0},
    {label: "Mean Value", value: fmt(overview.mean_value, 1), className: scoreClass(overview.mean_value)},
    {label: "Complexity", value: fmt(overview.mean_complexity, 1), className: scoreClass(overview.mean_complexity)},
    {label: "Quality", value: fmt(overview.mean_quality, 1), className: scoreClass(overview.mean_quality)},
    {label: "Median Rarity", value: fmt(overview.median_rarity, 1), className: scoreClass(overview.median_rarity)},
    {label: "Confidence", value: fmt(overview.mean_confidence, 2)},
    {label: "Tokens", value: Number(overview.total_tokens || 0).toLocaleString()},
  ];
  sections.push(section("Scoring Overview", renderCards(cards), `${overview.total_scored || 0} scored`, true));

  const histograms = pass2.histograms || {};
  const scoreDistributions = pass2.score_distributions || {};
  const histogramBlocks = Object.entries(histograms).map(([key, bins]) => {
    if (!Array.isArray(bins) || !bins.length) return "";
    return renderHistogram(HIST_LABELS[key] || key, bins, HIST_COLORS[key] || "#2563eb", scoreDistributions[key]);
  }).join("");
  if (histogramBlocks) {
    sections.push(section("Score Distributions", `<div class="grid">${histogramBlocks}</div>`, "", true));
  }

  if ((pass2.confidence_histogram || []).some((value) => value > 0)) {
    const bins = pass2.confidence_histogram;
    const max = Math.max(...bins, 1);
    const bars = bins.map((count, idx) => {
      const height = Math.max((count / max) * 100, 2);
      return `<div class="hist-bar" style="height:${height}%;background:#7c3aed" title="${(idx / 10).toFixed(1)}-${((idx + 1) / 10).toFixed(1)}: ${count}"></div>`;
    }).join("");
    const labels = bins.map((_, idx) => `<span>${(idx / 10).toFixed(1)}</span>`).join("");
    sections.push(section("Confidence Distribution", `<div class="mini-panel"><h4>LLM Confidence</h4><div class="hist-row">${bars}</div><div class="hist-labels">${labels}</div></div>`, "", false));
  }

  const byTagSections = [];
  for (const [title, bucket] of [
    ["Value By Tag", pass2.value_by_tag || {}],
    ["Selection By Tag", pass2.selection_by_tag || {}],
  ]) {
    const panels = Object.entries(bucket).map(([dim, dist]) => {
      const items = Object.entries(dist || {})
        .sort((a, b) => (b[1]?.mean || 0) - (a[1]?.mean || 0) || a[0].localeCompare(b[0]))
        .map(([label, info]) => ({
          label,
          mean: info.mean,
          n: info.n || 0,
        }));
      return `<div class="mini-panel"><h4>${escapeHtml(dim)}</h4>${renderRankingTable(items, "Score", (item) => `${fmt(item.mean, 2)} (n=${item.n})`)}</div>`;
    }).join("");
    if (panels) {
      byTagSections.push(section(title, `<div class="grid">${panels}</div>`, "All tags, sorted by score", false));
    }
  }
  sections.push(byTagSections.join(""));

  const thinkingMode = pass2.thinking_mode_stats || {};
  const flags = pass2.flag_counts || {};
  if (Object.keys(thinkingMode).length || Object.keys(flags).length) {
    let html = `<div class="grid">`;
    if (Object.keys(thinkingMode).length) {
      const rows = Object.entries(thinkingMode).map(([mode, stats]) => `<tr>
        <td>${escapeHtml(mode)}</td>
        <td>${stats.count || 0}</td>
        <td>${fmt(stats.mean_value, 1)}</td>
        <td>${fmt(stats.mean_quality, 1)}</td>
        <td>${fmt(stats.mean_reasoning, 1)}</td>
      </tr>`).join("");
      html += `<div class="mini-panel"><h4>Thinking Mode</h4><table class="data-table"><thead><tr><th>Mode</th><th>Count</th><th>Value</th><th>Quality</th><th>Reasoning</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }
    if (Object.keys(flags).length) {
      const items = Object.entries(flags).slice(0, 18).map(([label, value]) => ({
        label,
        value,
        display: `${value}`,
      }));
      html += `<div class="mini-panel"><h4>Flags</h4>${renderBarChart(items, "#dc2626")}</div>`;
    }
    html += `</div>`;
    sections.push(section("Analysis", html, "", true));
  }

  const perFile = pass2.per_file_summary || [];
  if (perFile.length > 1) {
    const sortColumn = FILE_RANKING_COLUMNS[STATE.fileRankingSort.key] || FILE_RANKING_COLUMNS.mean_value;
    const rows = sortPerFileSummary(perFile)
      .map((row) => `<tr>
        <td>${escapeHtml(row.file)}</td>
        <td>${row.count || 0}</td>
        <td><span class="${scoreClass(row.mean_value)}">${fmt(row.mean_value, 1)}</span></td>
        <td>${fmt(row.mean_complexity, 1)}</td>
        <td>${fmt(row.mean_quality, 1)}</td>
        <td>${fmt(row.mean_selection, 1)}</td>
      </tr>`).join("");
    sections.push(section(
      "File Ranking",
      `<table class="data-table"><thead><tr>${
        ["file", "count", "mean_value", "mean_complexity", "mean_quality", "mean_selection"]
          .map(renderFileRankingHeader)
          .join("")
      }</tr></thead><tbody>${rows}</tbody></table>`,
      `${perFile.length} files · sorted by ${sortColumn.label}`,
      true,
    ));
  }

  const weights = pass2.weights_used || {};
  if (Object.keys(weights).length) {
    const formula = Object.entries(weights).map(([key, value]) => `${value}×${key}`).join(" + ");
    sections.push(section("Configuration", `<div class="note">${escapeHtml(formula)}</div>`, "", false));
  }

  return sections.join("");
}

function renderConversations(conversation) {
  if (!conversation) return "";
  const cards = [
    {label: "Conversations", value: conversation.total || 0},
    {label: "Conv Value", value: fmt(conversation.mean_conv_value, 1), className: scoreClass(conversation.mean_conv_value)},
    {label: "Conv Selection", value: fmt(conversation.mean_conv_selection, 1), className: scoreClass(conversation.mean_conv_selection)},
    {label: "Peak Complexity", value: fmt(conversation.mean_peak_complexity, 1), className: scoreClass(conversation.mean_peak_complexity)},
    {label: "Mean Turns", value: fmt(conversation.mean_turns, 1)},
  ];

  let body = renderCards(cards);
  const histBlocks = [];
  for (const [label, bins, color] of [
    ["Conv Value", conversation.conv_value_hist, "#2563eb"],
    ["Conv Selection", conversation.conv_selection_hist, "#be185d"],
    ["Peak Complexity", conversation.peak_complexity_hist, "#c2410c"],
  ]) {
    if (Array.isArray(bins) && bins.some((value) => value > 0)) {
      histBlocks.push(renderHistogram(label, bins, color, null));
    }
  }
  const turnDist = conversation.turn_distribution || {};
  if (Object.keys(turnDist).length) {
    const items = Object.entries(turnDist).map(([turns, count]) => ({
      label: `${turns} turns`,
      value: count,
      display: `${count}`,
    }));
    histBlocks.push(`<div class="mini-panel"><h4>Turn Distribution</h4>${renderBarChart(items, "#0f766e")}</div>`);
  }
  if (histBlocks.length) {
    body += `<div class="grid" style="margin-top:14px">${histBlocks.join("")}</div>`;
  }
  return section("Conversation Aggregation", body, "", false);
}

function renderScopeContent(scope) {
  const chunks = [];
  chunks.push(renderChildren(scope));
  chunks.push(renderPass1(scope.pass1));
  chunks.push(renderPass2(scope.pass2));
  chunks.push(renderConversations(scope.conversation));
  const html = chunks.filter(Boolean).join("");
  return html || `<div class="empty">No dashboard data for this scope.</div>`;
}

function renderScope() {
  const scope = getCurrentScope();
  if (!scope) return;

  document.getElementById("scope-title").textContent = scope.label;
  document.getElementById("scope-path").textContent = scope.path || "Global overview";
  renderBreadcrumbs(scope);
  document.getElementById("scope-content").innerHTML = renderScopeContent(scope);

  document.querySelectorAll("[data-jump]").forEach((node) => {
    node.addEventListener("click", () => setScope(node.getAttribute("data-jump")));
  });
  document.querySelectorAll("[data-file-sort]").forEach((node) => {
    node.addEventListener("click", () => toggleFileRankingSort(node.getAttribute("data-file-sort")));
  });

  const parentBtn = document.getElementById("go-parent");
  parentBtn.disabled = !scope.parent_id;
  parentBtn.onclick = () => {
    if (scope.parent_id) setScope(scope.parent_id);
  };
  const globalBtn = document.getElementById("go-global");
  globalBtn.disabled = scope.id === DATA.root_id;
  globalBtn.onclick = () => {
    if (scope.id !== DATA.root_id) setScope(DATA.root_id);
  };
}

function render() {
  renderHero();
  renderTree();
  renderScope();
}

document.getElementById("scope-search").addEventListener("input", (event) => {
  STATE.query = event.target.value.trim().toLowerCase();
  renderTree();
});

document.querySelectorAll(".filter-btn").forEach((button) => {
  button.addEventListener("click", () => {
    STATE.filter = button.getAttribute("data-filter");
    document.querySelectorAll(".filter-btn").forEach((node) => node.classList.toggle("active", node === button));
    renderTree();
  });
});

STATE.sidebarCollapsed = loadSidebarPreference();
applySidebarState();
document.getElementById("sidebar-toggle").addEventListener("click", toggleSidebar);

render();
</script>
</body>
</html>"""


def render_dashboard_html(payload: dict) -> str:
    """Render a static interactive dashboard page."""
    return (
        HTML_TEMPLATE
        .replace("__TITLE__", payload.get("title", "Interactive Dashboard"))
        .replace("__DATA_PLACEHOLDER__", json.dumps(payload, ensure_ascii=False))
    )
