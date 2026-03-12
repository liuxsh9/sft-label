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
.scope-toolbar-extra { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.action-btn { border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 999px; padding: 8px 12px; font: inherit; cursor: pointer; }
.action-btn:hover { border-color: var(--accent); color: var(--accent); }
.toolbar-inline-copy { color: var(--muted); font-size: 0.8rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border: 1px solid rgba(223,216,199,0.9); border-radius: 18px; padding: 14px; }
.card-label { font-size: 0.76rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.card-value { margin-top: 6px; font-size: 1.6rem; font-weight: 700; line-height: 1; }
.card-sub { margin-top: 6px; color: var(--muted); font-size: 0.82rem; }
.segmented { display: inline-flex; gap: 6px; flex-wrap: wrap; }
.segmented-btn { border: 1px solid rgba(223,216,199,0.95); background: rgba(255,255,255,0.82); color: var(--muted); border-radius: 999px; padding: 8px 12px; font: inherit; cursor: pointer; transition: border-color 180ms ease, color 180ms ease, background 180ms ease; }
.segmented-btn:hover { border-color: rgba(31,111,95,0.35); color: var(--accent); }
.segmented-btn.active { background: rgba(31,111,95,0.12); border-color: rgba(31,111,95,0.28); color: var(--accent); }
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
.data-table tr:hover .rank-tag-bar { opacity: 1; }
.th-sort-btn { width: 100%; display: inline-flex; align-items: center; justify-content: space-between; gap: 8px; border: none; background: none; padding: 0; color: inherit; font: inherit; cursor: pointer; }
.th-sort-btn.active { color: var(--accent); }
.sort-indicator { color: var(--muted); font-size: 0.78rem; }
.compact-table th, .compact-table td { padding: 7px 8px; font-size: 0.8rem; }
.rank-cell { width: 42px; color: var(--muted); }
.rank-tag-cell { min-width: 180px; }
.rank-tag-wrap { position: relative; min-height: 32px; display: flex; align-items: center; border-radius: 10px; overflow: hidden; }
.rank-tag-bar { position: absolute; inset: 4px auto 4px 4px; width: 0; border-radius: 8px; opacity: 0.82; pointer-events: none; transition: opacity 160ms ease, width 220ms ease; }
.rank-tag-content { position: relative; z-index: 1; min-width: 0; width: 100%; padding: 7px 10px; }
.rank-tag-link, .rank-tag-text { display: block; width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
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
.explorer-toolbar { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
.explorer-summary { color: var(--muted); font-size: 0.84rem; max-width: 780px; }
.explorer-form { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 14px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field label { color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; }
.field input, .field select { width: 100%; padding: 10px 12px; border-radius: 12px; border: 1px solid var(--line); background: #fff; color: var(--text); font: inherit; }
.field-check { display: flex; align-items: center; gap: 8px; padding-top: 26px; }
.field-check input { width: auto; }
.explorer-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.preset-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 0; }
.preset-btn { border: 1px solid rgba(31,111,95,0.18); background: rgba(31,111,95,0.08); color: var(--accent); border-radius: 999px; padding: 8px 12px; cursor: pointer; font: inherit; }
.preset-btn:hover { border-color: rgba(31,111,95,0.35); background: rgba(31,111,95,0.12); }
.result-meta { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin: 14px 0 10px; }
.result-count { color: var(--muted); font-size: 0.84rem; }
.result-table-wrap { overflow: auto; border: 1px solid rgba(223,216,199,0.9); border-radius: 16px; background: rgba(255,255,255,0.55); }
.preview-tags { display: flex; gap: 6px; flex-wrap: wrap; max-width: 380px; }
.preview-tag { display: inline-flex; align-items: center; padding: 3px 7px; border-radius: 999px; background: rgba(37, 99, 235, 0.08); color: #1d4ed8; font-size: 0.72rem; }
.preview-query { min-width: 260px; }
.preview-response { max-width: 360px; color: var(--muted); }
.drawer-backdrop { position: fixed; inset: 0; background: rgba(29,38,31,0.34); z-index: 49; }
.drawer { position: fixed; top: 0; right: 0; width: min(680px, 92vw); height: 100vh; background: #fffdf8; box-shadow: -18px 0 36px rgba(29,38,31,0.18); z-index: 50; display: flex; flex-direction: column; }
.drawer-header { padding: 18px 20px; border-bottom: 1px solid var(--line); display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.drawer-title { font-size: 1.12rem; font-weight: 700; }
.drawer-sub { color: var(--muted); font-size: 0.82rem; margin-top: 4px; word-break: break-all; }
.drawer-close { border: 1px solid var(--line); background: var(--panel); border-radius: 999px; width: 36px; height: 36px; cursor: pointer; font: inherit; }
.drawer-body { padding: 18px 20px 28px; overflow: auto; display: flex; flex-direction: column; gap: 14px; }
.drawer-section { border: 1px solid rgba(223,216,199,0.9); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.6); }
.drawer-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.drawer-collapse { border: 1px solid rgba(223,216,199,0.9); border-radius: 14px; background: rgba(250,248,241,0.9); padding: 10px 12px; }
.drawer-collapse summary { cursor: pointer; font-weight: 600; color: var(--text); }
.drawer-collapse[open] summary { margin-bottom: 10px; }
.kv { background: rgba(248,245,235,0.8); border-radius: 12px; padding: 10px; }
.kv-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kv-value { margin-top: 6px; font-size: 1rem; font-weight: 700; }
.conversation-turn { border-top: 1px solid #ece6d7; padding-top: 10px; margin-top: 10px; }
.conversation-turn:first-child { border-top: none; padding-top: 0; margin-top: 0; }
.turn-role { font-size: 0.74rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.turn-text { margin-top: 6px; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.8rem; line-height: 1.45; }
.drawer-truncate-note { margin-top: 10px; }
.drawer-json { white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.76rem; line-height: 1.45; }
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
<script>
const DATA = __DATA_PLACEHOLDER__;
const STATE = {
  currentId: DATA.default_scope_id || DATA.root_id,
  filter: "all",
  query: "",
  expanded: new Set(DATA.initially_expanded || [DATA.root_id]),
  sidebarCollapsed: false,
  tagBarMode: "relative",
  fileRankingSort: { key: "mean_value", direction: "desc" },
  explorer: {
    runId: 0,
    busy: false,
    scanned: 0,
    matched: 0,
    scopeTotal: 0,
    scopeDone: 0,
    status: "",
    results: [],
    sort: (DATA.explorer || {}).result_limit ? "quality_asc" : "sample_id_asc",
    limit: (DATA.explorer || {}).result_limit || 200,
    detailDocId: "",
    detailScopeId: "",
    detailData: null,
    detailLoading: false,
    queryModel: {
      tagQuery: "",
      language: "",
      intent: "",
      sourcePath: "",
      textQuery: "",
      flagQuery: "",
      convValueMin: "",
      convSelectionMin: "",
      peakComplexityMin: "",
      turnCountMin: "",
      minValue: "",
      maxValue: "",
      maxQuality: "",
      minSelection: "",
      maxConfidence: "",
      thinkingMode: "",
      hasFlags: false,
      includeInherited: false,
    },
  },
};
const EXPLORER_PREVIEW_CACHE_LIMIT = 6;
const EXPLORER_DETAIL_CACHE_LIMIT = 18;
const EXPLORER_PROGRESS_RENDER_MS = 120;
const EXPLORER_DRAWER_TURN_LIMIT = 12;
const EXPLORER_DRAWER_TEXT_LIMIT = 1600;
const EXPLORER_DRAWER_JSON_LIMIT = 24000;
const EXPLORER_CACHE = {
  scripts: new Map(),
  previews: new Map(),
  details: new Map(),
  previewAssets: new Map(),
  detailAssets: new Map(),
  lastProgressRender: 0,
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

function hexToRgba(hex, alpha) {
  const value = String(hex || "").replace("#", "").trim();
  if (value.length !== 6) return `rgba(37, 99, 235, ${alpha})`;
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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

function fmtInt(value) {
  return Number(value || 0).toLocaleString();
}

function parseNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function textPreview(value) {
  return escapeHtml(String(value || "").trim() || "-");
}

function truncateMiddle(value, maxLen) {
  const text = String(value || "");
  if (!maxLen || text.length <= maxLen) return text;
  const head = Math.max(Math.floor((maxLen - 1) * 0.65), 1);
  const tail = Math.max(maxLen - head - 1, 1);
  return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

function escapeAttrJson(value) {
  return escapeHtml(JSON.stringify(value));
}

function splitCsv(value) {
  return String(value || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function mergeCsvValues(existing, additions) {
  const seen = new Set();
  const merged = [];
  for (const part of [...splitCsv(existing), ...splitCsv(additions)]) {
    const key = part.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(part);
  }
  return merged.join(", ");
}

function explorerRegistry() {
  if (!window.__SFT_DASHBOARD_EXPLORER__) {
    window.__SFT_DASHBOARD_EXPLORER__ = { preview: {}, detail: {} };
  }
  return window.__SFT_DASHBOARD_EXPLORER__;
}

function persistDashboardState() {
  try {
    const state = {
      scope: STATE.currentId,
      tagBarMode: STATE.tagBarMode,
      explorer: {
        sort: STATE.explorer.sort,
        query: STATE.explorer.queryModel,
      },
    };
    const encoded = encodeURIComponent(JSON.stringify(state));
    const nextHash = `dashboard=${encoded}`;
    if (window.location.hash.slice(1) !== nextHash) {
      window.location.hash = nextHash;
    }
  } catch (error) {
    // Ignore hash persistence failures.
  }
}

function restoreDashboardState() {
  try {
    const raw = window.location.hash.replace(/^#/, "");
    if (!raw.startsWith("dashboard=")) return;
    const payload = JSON.parse(decodeURIComponent(raw.slice("dashboard=".length)));
    if (payload.scope && DATA.scopes[payload.scope]) {
      STATE.currentId = payload.scope;
    }
    if (payload.tagBarMode && ["hidden", "relative", "global-log"].includes(payload.tagBarMode)) {
      STATE.tagBarMode = payload.tagBarMode;
    }
    if (payload.explorer && typeof payload.explorer === "object") {
      if (payload.explorer.sort) STATE.explorer.sort = payload.explorer.sort;
      if (payload.explorer.query && typeof payload.explorer.query === "object") {
        STATE.explorer.queryModel = {
          ...STATE.explorer.queryModel,
          ...payload.explorer.query,
        };
      }
    }
    let cursor = getScope(STATE.currentId);
    while (cursor && cursor.parent_id) {
      STATE.expanded.add(cursor.parent_id);
      cursor = getScope(cursor.parent_id);
    }
  } catch (error) {
    // Ignore malformed hashes.
  }
}

function getScope(id) {
  return DATA.scopes[id];
}

function getCurrentScope() {
  return getScope(STATE.currentId) || getScope(DATA.root_id);
}

function resetExplorerView(resetQuery = false) {
  STATE.explorer.runId += 1;
  STATE.explorer.busy = false;
  STATE.explorer.scanned = 0;
  STATE.explorer.matched = 0;
  STATE.explorer.scopeTotal = 0;
  STATE.explorer.scopeDone = 0;
  STATE.explorer.status = "";
  STATE.explorer.results = [];
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  STATE.explorer.detailLoading = false;
  if (resetQuery) {
    STATE.explorer.queryModel = {
      tagQuery: "",
      language: "",
      intent: "",
      sourcePath: "",
      textQuery: "",
      flagQuery: "",
      convValueMin: "",
      convSelectionMin: "",
      peakComplexityMin: "",
      turnCountMin: "",
      minValue: "",
      maxValue: "",
      maxQuality: "",
      minSelection: "",
      maxConfidence: "",
      thinkingMode: "",
      hasFlags: false,
      includeInherited: false,
    };
    STATE.explorer.sort = "quality_asc";
  }
}

function sourcePathMatchesScope(scope, sourcePath) {
  const term = String(sourcePath || "").trim().toLowerCase();
  if (!term) return true;
  const haystack = `${scope.path || ""} ${scope.label || ""}`.toLowerCase();
  return haystack.includes(term);
}

function explorerFileScopeIds(scope, queryModel = STATE.explorer.queryModel) {
  if (!scope) return [];
  if (scope.explorer) return sourcePathMatchesScope(scope, (queryModel || {}).sourcePath) ? [scope.id] : [];
  return (scope.descendant_files || [])
    .map((path) => `file:${path}`)
    .filter((scopeId) => {
      const item = getScope(scopeId);
      return item && item.explorer && sourcePathMatchesScope(item, (queryModel || {}).sourcePath);
    });
}

function explorerTotalSamples(scope) {
  return explorerFileScopeIds(scope, {}).reduce((sum, scopeId) => {
    const explorer = (getScope(scopeId) || {}).explorer || {};
    return sum + Number(explorer.sample_count || 0);
  }, 0);
}

function scopeSupportsExplorer(scope) {
  return !!(DATA.explorer && DATA.explorer.enabled && explorerFileScopeIds(scope, {}).length);
}

function toggleExpanded(id) {
  if (STATE.expanded.has(id)) STATE.expanded.delete(id);
  else STATE.expanded.add(id);
  renderTree();
}

function setScope(id) {
  if (!DATA.scopes[id]) return;
  if (STATE.currentId !== id) resetExplorerView(false);
  STATE.currentId = id;
  let cursor = DATA.scopes[id];
  while (cursor && cursor.parent_id) {
    STATE.expanded.add(cursor.parent_id);
    cursor = DATA.scopes[cursor.parent_id];
  }
  persistDashboardState();
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

function renderBarChart(items, color, maxValue = null, actionBuilder = null) {
  if (!items || !items.length) return `<div class="empty">No data</div>`;
  const ceiling = maxValue || Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return items.map((item) => {
    const pct = ceiling > 0 ? ((Number(item.value) || 0) / ceiling) * 100 : 0;
    const labelHtml = actionBuilder
      ? `<button class="link-btn bar-label" type="button" data-explorer-patch="${escapeAttrJson(actionBuilder(item))}" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</button>`
      : `<div class="bar-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>`;
    return `<div class="bar-row">
      ${labelHtml}
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${color}"></div></div>
      <div class="bar-value">${escapeHtml(item.display || item.value)}</div>
    </div>`;
  }).join("");
}

function renderHistogram(label, bins, color, stats, bucketActionBuilder = null) {
  const max = Math.max(...bins, 1);
  const bars = bins.map((count, index) => {
    const height = Math.max((count / max) * 100, 2);
    const patch = bucketActionBuilder ? ` data-explorer-patch="${escapeAttrJson(bucketActionBuilder(index + 1, count))}"` : "";
    const role = bucketActionBuilder ? "button" : "";
    const cls = bucketActionBuilder ? "hist-bar link-btn" : "hist-bar";
    return `<div class="${cls}"${patch} style="height:${height}%;background:${color}" title="${index + 1}: ${count}" ${role ? `role="${role}"` : ""}></div>`;
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

function renderRankingTable(items, metricLabel, valueRenderer, actionBuilder = null, options = null) {
  if (!items || !items.length) return `<div class="empty">No data</div>`;
  const rankOptions = options || {};
  const barMetricKey = rankOptions.barMetricKey || "";
  const barColor = rankOptions.barColor || "#2563eb";
  const barMetricLabel = rankOptions.barMetricLabel || "Count";
  const globalBarMax = Number(rankOptions.globalBarMax) || 0;
  const barValues = barMetricKey
    ? items.map((item) => Number(item?.[barMetricKey]) || 0)
    : [];
  const localBarMax = barValues.length ? Math.max(...barValues, 1) : 0;
  const barMax = STATE.tagBarMode === "global-log" && globalBarMax > 0 ? globalBarMax : localBarMax;
  const rows = items.map((item, index) => `<tr>
    <td class="rank-cell">${index + 1}</td>
    <td class="rank-tag-cell">${renderRankingTagCell(item, actionBuilder, {
      barMetricKey,
      barMetricLabel,
      barMax,
      barColor,
    })}</td>
    <td class="metric-cell">${escapeHtml(valueRenderer(item))}</td>
  </tr>`).join("");
  return `<table class="data-table compact-table">
    <thead><tr><th>#</th><th>Tag</th><th>${escapeHtml(metricLabel)}</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderRankingTagCell(item, actionBuilder, options = null) {
  const rankOptions = options || {};
  const metricKey = rankOptions.barMetricKey || "";
  const rawMetric = metricKey ? Number(item?.[metricKey]) || 0 : 0;
  const barMax = Number(rankOptions.barMax) || 0;
  const mode = STATE.tagBarMode || "relative";
  let pct = 0;
  if (metricKey && barMax > 0 && mode !== "hidden") {
    if (mode === "global-log") {
      pct = (Math.log1p(Math.max(rawMetric, 0)) / Math.log1p(barMax)) * 100;
    } else {
      pct = (rawMetric / barMax) * 100;
    }
    pct = Math.max(6, pct);
  }
  const fill = metricKey && mode !== "hidden"
    ? `<div class="rank-tag-bar" style="width:${pct.toFixed(1)}%;background:linear-gradient(90deg, ${hexToRgba(rankOptions.barColor, 0.18)} 0%, ${hexToRgba(rankOptions.barColor, 0.10)} 85%, ${hexToRgba(rankOptions.barColor, 0.04)} 100%);border:1px solid ${hexToRgba(rankOptions.barColor, 0.12)}" title="${escapeHtml(`${item.label} · ${rankOptions.barMetricLabel || "Count"}: ${rawMetric}`)}"></div>`
    : "";
  const labelHtml = actionBuilder
    ? `<button class="link-btn rank-tag-link" type="button" data-explorer-patch="${escapeAttrJson(actionBuilder(item))}" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</button>`
    : `<span class="rank-tag-text" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span>`;
  return `<div class="rank-tag-wrap">${fill}<div class="rank-tag-content">${labelHtml}</div></div>`;
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

function explorerAssetPath(scopeExplorer, relpath) {
  return `${scopeExplorer.assets_dir}/${relpath}`;
}

function ensureExplorerScriptLoaded(path) {
  const cached = EXPLORER_CACHE.scripts.get(path);
  if (cached) return cached.promise;
  const script = document.createElement("script");
  const entry = { node: script, promise: null };
  entry.promise = new Promise((resolve, reject) => {
    script.src = path;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load explorer asset: ${path}`));
    document.head.appendChild(script);
  });
  EXPLORER_CACHE.scripts.set(path, entry);
  return entry.promise;
}

function dropExplorerScript(path) {
  const entry = EXPLORER_CACHE.scripts.get(path);
  if (entry && entry.node && entry.node.parentNode) {
    entry.node.parentNode.removeChild(entry.node);
  }
  EXPLORER_CACHE.scripts.delete(path);
}

function unloadScopePreview(scopeId) {
  const assets = EXPLORER_CACHE.previewAssets.get(scopeId) || [];
  const registry = explorerRegistry();
  for (const asset of assets) {
    delete registry.preview[asset.key];
    dropExplorerScript(asset.path);
  }
  EXPLORER_CACHE.previewAssets.delete(scopeId);
  EXPLORER_CACHE.previews.delete(scopeId);
}

function unloadDetailChunk(cacheKey) {
  const asset = EXPLORER_CACHE.detailAssets.get(cacheKey);
  const registry = explorerRegistry();
  if (asset) {
    delete registry.detail[asset.chunkKey];
    dropExplorerScript(asset.path);
  }
  EXPLORER_CACHE.detailAssets.delete(cacheKey);
  EXPLORER_CACHE.details.delete(cacheKey);
}

function retainScopePreview(scopeId, rows, assets) {
  if (EXPLORER_CACHE.previews.has(scopeId)) {
    EXPLORER_CACHE.previews.delete(scopeId);
  }
  EXPLORER_CACHE.previews.set(scopeId, rows);
  EXPLORER_CACHE.previewAssets.set(scopeId, assets);
  while (EXPLORER_CACHE.previews.size > EXPLORER_PREVIEW_CACHE_LIMIT) {
    const oldestScopeId = EXPLORER_CACHE.previews.keys().next().value;
    if (!oldestScopeId) break;
    unloadScopePreview(oldestScopeId);
  }
}

function maybeRenderExplorerProgress(force = false) {
  const now = Date.now();
  if (!force && (now - EXPLORER_CACHE.lastProgressRender) < EXPLORER_PROGRESS_RENDER_MS) return;
  EXPLORER_CACHE.lastProgressRender = now;
  renderScope();
}

async function processScopePreviewRows(scopeId, visitRow) {
  if (EXPLORER_CACHE.previews.has(scopeId)) {
    for (const row of EXPLORER_CACHE.previews.get(scopeId) || []) {
      if (visitRow(row) === false) return false;
    }
    return true;
  }
  const scope = getScope(scopeId);
  const explorer = (scope || {}).explorer;
  if (!explorer) return true;
  const registry = explorerRegistry();
  const chunks = explorer.preview_chunks || [];
  const cacheable = chunks.length <= 3;
  const rows = [];
  const assets = [];
  for (const chunk of chunks) {
    const path = explorerAssetPath(explorer, chunk.path);
    await ensureExplorerScriptLoaded(path);
    const chunkRows = registry.preview[chunk.key] || [];
    if (cacheable) {
      rows.push(...chunkRows);
      assets.push({ key: chunk.key, path });
    }
    for (const row of chunkRows) {
      if (visitRow(row) === false) {
        if (!cacheable) {
          delete registry.preview[chunk.key];
          dropExplorerScript(path);
        }
        return false;
      }
    }
    if (!cacheable) {
      delete registry.preview[chunk.key];
      dropExplorerScript(path);
    }
  }
  if (cacheable) retainScopePreview(scopeId, rows, assets);
  return true;
}

async function loadExplorerDetail(scopeId, detailChunkKey, docId) {
  const cacheKey = `${scopeId}|${detailChunkKey}`;
  if (!EXPLORER_CACHE.details.has(cacheKey)) {
    const scope = getScope(scopeId);
    const explorer = (scope || {}).explorer;
    const chunk = (explorer && (explorer.detail_chunks || []).find((item) => item.key === detailChunkKey)) || null;
    if (!explorer || !chunk) return null;
    const path = explorerAssetPath(explorer, chunk.path);
    await ensureExplorerScriptLoaded(path);
    const registry = explorerRegistry();
    EXPLORER_CACHE.details.set(cacheKey, registry.detail[detailChunkKey] || {});
    EXPLORER_CACHE.detailAssets.set(cacheKey, { chunkKey: detailChunkKey, path });
    while (EXPLORER_CACHE.details.size > EXPLORER_DETAIL_CACHE_LIMIT) {
      const oldestCacheKey = EXPLORER_CACHE.details.keys().next().value;
      if (!oldestCacheKey || oldestCacheKey === cacheKey) break;
      unloadDetailChunk(oldestCacheKey);
    }
  } else {
    const chunkPayload = EXPLORER_CACHE.details.get(cacheKey);
    const asset = EXPLORER_CACHE.detailAssets.get(cacheKey);
    EXPLORER_CACHE.details.delete(cacheKey);
    EXPLORER_CACHE.detailAssets.delete(cacheKey);
    EXPLORER_CACHE.details.set(cacheKey, chunkPayload);
    if (asset) EXPLORER_CACHE.detailAssets.set(cacheKey, asset);
  }
  const chunkPayload = EXPLORER_CACHE.details.get(cacheKey) || {};
  return chunkPayload[docId] || null;
}

function parseExplorerQueryFromDom() {
  return {
    tagQuery: (document.getElementById("explorer-tag-query") || {}).value || "",
    language: (document.getElementById("explorer-language") || {}).value || "",
    intent: (document.getElementById("explorer-intent") || {}).value || "",
    sourcePath: (document.getElementById("explorer-source-path") || {}).value || "",
    textQuery: (document.getElementById("explorer-text-query") || {}).value || "",
    flagQuery: (document.getElementById("explorer-flag-query") || {}).value || "",
    convValueMin: (document.getElementById("explorer-conv-value-min") || {}).value || "",
    convSelectionMin: (document.getElementById("explorer-conv-selection-min") || {}).value || "",
    peakComplexityMin: (document.getElementById("explorer-peak-complexity-min") || {}).value || "",
    turnCountMin: (document.getElementById("explorer-turn-count-min") || {}).value || "",
    minValue: (document.getElementById("explorer-min-value") || {}).value || "",
    maxValue: (document.getElementById("explorer-max-value") || {}).value || "",
    maxQuality: (document.getElementById("explorer-max-quality") || {}).value || "",
    minSelection: (document.getElementById("explorer-min-selection") || {}).value || "",
    maxConfidence: (document.getElementById("explorer-max-confidence") || {}).value || "",
    thinkingMode: (document.getElementById("explorer-thinking-mode") || {}).value || "",
    hasFlags: !!((document.getElementById("explorer-has-flags") || {}).checked),
    includeInherited: !!((document.getElementById("explorer-include-inherited") || {}).checked),
  };
}

function applyExplorerQueryPatch(patch) {
  const next = { ...STATE.explorer.queryModel };
  if (patch.tagQueryAppend) {
    next.tagQuery = mergeCsvValues(next.tagQuery, Array.isArray(patch.tagQueryAppend) ? patch.tagQueryAppend.join(", ") : patch.tagQueryAppend);
  }
  for (const [key, value] of Object.entries(patch || {})) {
    if (key === "tagQueryAppend") continue;
    next[key] = value;
  }
  STATE.explorer.queryModel = next;
  if (patch.sort) STATE.explorer.sort = patch.sort;
  persistDashboardState();
  renderScope();
}

function rowMatchesExplorerQuery(row, query) {
  if (!row) return false;
  const tagTerms = String(query.tagQuery || "")
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);
  const rowTags = new Set((row.flat_tags || []).map((item) => String(item).toLowerCase()));
  if (tagTerms.length && !tagTerms.every((term) => rowTags.has(term))) return false;

  const language = String(query.language || "").trim().toLowerCase();
  if (language) {
    const langs = new Set((row.languages || []).map((item) => String(item).toLowerCase()));
    if (!langs.has(language)) return false;
  }

  const intent = String(query.intent || "").trim().toLowerCase();
  if (intent && String(row.intent || "").toLowerCase() !== intent) return false;

  const sourcePath = String(query.sourcePath || "").trim().toLowerCase();
  if (sourcePath) {
    const sourceHaystack = `${row.scope_path || ""} ${row.source_file || ""}`.toLowerCase();
    if (!sourceHaystack.includes(sourcePath)) return false;
  }

  const flagQuery = String(query.flagQuery || "").trim().toLowerCase();
  if (flagQuery) {
    const flags = new Set((row.flags || []).map((item) => String(item).toLowerCase()));
    if (!flags.has(flagQuery)) return false;
  }

  const minValue = parseNumber(query.minValue);
  if (minValue !== null && (row.value_score === null || row.value_score === undefined || Number(row.value_score) < minValue)) return false;
  const convValueMin = parseNumber(query.convValueMin);
  if (convValueMin !== null && (row.conv_value === null || row.conv_value === undefined || Number(row.conv_value) < convValueMin)) return false;
  const convSelectionMin = parseNumber(query.convSelectionMin);
  if (convSelectionMin !== null && (row.conv_selection === null || row.conv_selection === undefined || Number(row.conv_selection) < convSelectionMin)) return false;
  const peakComplexityMin = parseNumber(query.peakComplexityMin);
  if (peakComplexityMin !== null && (row.peak_complexity === null || row.peak_complexity === undefined || Number(row.peak_complexity) < peakComplexityMin)) return false;
  const maxValue = parseNumber(query.maxValue);
  if (maxValue !== null && (row.value_score === null || row.value_score === undefined || Number(row.value_score) > maxValue)) return false;
  const maxQuality = parseNumber(query.maxQuality);
  if (maxQuality !== null && (row.quality_overall === null || row.quality_overall === undefined || Number(row.quality_overall) > maxQuality)) return false;
  const minSelection = parseNumber(query.minSelection);
  if (minSelection !== null && (row.selection_score === null || row.selection_score === undefined || Number(row.selection_score) < minSelection)) return false;
  const turnCountMin = parseNumber(query.turnCountMin);
  if (turnCountMin !== null && (row.turn_count === null || row.turn_count === undefined || Number(row.turn_count) < turnCountMin)) return false;
  const maxConfidence = parseNumber(query.maxConfidence);
  if (maxConfidence !== null && (row.confidence === null || row.confidence === undefined || Number(row.confidence) > maxConfidence)) return false;

  if (query.thinkingMode && String(row.thinking_mode || "") !== String(query.thinkingMode)) return false;
  if (!query.includeInherited && row.inherited) return false;
  if (query.hasFlags && !((row.flags || []).length > 0)) return false;

  const textQuery = String(query.textQuery || "").trim().toLowerCase();
  if (textQuery) {
    const haystack = [
      row.sample_id,
      row.query_preview,
      row.response_preview,
      row.source_file,
      row.scope_path,
    ].join(" ").toLowerCase();
    if (!haystack.includes(textQuery)) return false;
  }

  return true;
}

function compareExplorerRows(left, right, sortKey) {
  const numericSorts = {
    quality_asc: ["quality_overall", 1],
    value_asc: ["value_score", 1],
    confidence_asc: ["confidence", 1],
    selection_desc: ["selection_score", -1],
    conv_value_desc: ["conv_value", -1],
    conv_selection_desc: ["conv_selection", -1],
    turn_count_desc: ["turn_count", -1],
    rarity_desc: ["rarity_score", -1],
  };
  if (sortKey === "sample_id_asc") {
    return String(left.sample_id || "").localeCompare(String(right.sample_id || ""), undefined, { numeric: true, sensitivity: "base" });
  }
  const descriptor = numericSorts[sortKey] || numericSorts.quality_asc;
  const valueKey = descriptor[0];
  const direction = descriptor[1];
  const leftValue = Number(left[valueKey]);
  const rightValue = Number(right[valueKey]);
  const leftMissing = !Number.isFinite(leftValue);
  const rightMissing = !Number.isFinite(rightValue);
  if (leftMissing && rightMissing) {
    return String(left.sample_id || "").localeCompare(String(right.sample_id || ""), undefined, { numeric: true, sensitivity: "base" });
  }
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  if (leftValue !== rightValue) return (leftValue - rightValue) * direction;
  return String(left.sample_id || "").localeCompare(String(right.sample_id || ""), undefined, { numeric: true, sensitivity: "base" });
}

function pushExplorerResult(results, row, sortKey, limit) {
  results.push(row);
  results.sort((left, right) => compareExplorerRows(left, right, sortKey));
  if (results.length > limit) results.length = limit;
}

async function runExplorerQuery() {
  const scope = getCurrentScope();
  if (!scopeSupportsExplorer(scope)) return;
  const query = parseExplorerQueryFromDom();
  const scopeIds = explorerFileScopeIds(scope, query);
  STATE.explorer.queryModel = query;
  STATE.explorer.results = [];
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  STATE.explorer.busy = true;
  STATE.explorer.scanned = 0;
  STATE.explorer.matched = 0;
  STATE.explorer.scopeTotal = scopeIds.length;
  STATE.explorer.scopeDone = 0;
  const runId = ++STATE.explorer.runId;
  EXPLORER_CACHE.lastProgressRender = 0;
  persistDashboardState();
  renderScope();

  for (let index = 0; index < scopeIds.length; index += 1) {
    if (runId !== STATE.explorer.runId) return;
    const scopeId = scopeIds[index];
    STATE.explorer.status = `Scanning ${index + 1}/${scopeIds.length}: ${getScope(scopeId).label}`;
    maybeRenderExplorerProgress();
    const completed = await processScopePreviewRows(scopeId, (row) => {
      if (runId !== STATE.explorer.runId) return false;
      STATE.explorer.scanned += 1;
      if (rowMatchesExplorerQuery(row, query)) {
        STATE.explorer.matched += 1;
        pushExplorerResult(STATE.explorer.results, row, STATE.explorer.sort, STATE.explorer.limit);
      }
      return true;
    });
    if (!completed && runId !== STATE.explorer.runId) return;
    STATE.explorer.scopeDone = index + 1;
    maybeRenderExplorerProgress();
  }

  if (runId !== STATE.explorer.runId) return;
  STATE.explorer.busy = false;
  STATE.explorer.status = `Done. Matched ${fmtInt(STATE.explorer.matched)} rows across ${scopeIds.length} files.`;
  persistDashboardState();
  maybeRenderExplorerProgress(true);
}

function setExplorerPreset(preset) {
  const next = {
    tagQuery: "",
    language: "",
    intent: "",
    sourcePath: "",
    textQuery: "",
    flagQuery: "",
    convValueMin: "",
    convSelectionMin: "",
    peakComplexityMin: "",
    turnCountMin: "",
    minValue: "",
    maxValue: "",
    maxQuality: "",
    minSelection: "",
    maxConfidence: "",
    thinkingMode: "",
    hasFlags: false,
    includeInherited: false,
    ...preset,
  };
  STATE.explorer.queryModel = next;
  if (preset.sort) STATE.explorer.sort = preset.sort;
  STATE.explorer.results = [];
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  persistDashboardState();
  renderScope();
}

async function openExplorerDetail(docId, scopeId, detailChunkKey) {
  STATE.explorer.detailDocId = docId;
  STATE.explorer.detailScopeId = scopeId;
  STATE.explorer.detailLoading = true;
  STATE.explorer.detailData = null;
  persistDashboardState();
  renderScope();
  const detail = await loadExplorerDetail(scopeId, detailChunkKey, docId);
  STATE.explorer.detailLoading = false;
  STATE.explorer.detailData = detail;
  persistDashboardState();
  renderScope();
}

function closeExplorerDetail() {
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  STATE.explorer.detailLoading = false;
  persistDashboardState();
  renderScope();
}

function renderDrawerTurns(conversations) {
  const turns = Array.isArray(conversations) ? conversations : [];
  if (!turns.length) return `<div class="note">No conversation turns.</div>`;
  const hiddenCount = Math.max(turns.length - EXPLORER_DRAWER_TURN_LIMIT, 0);
  const visibleTurns = turns.slice(0, EXPLORER_DRAWER_TURN_LIMIT).map((turn) => `<div class="conversation-turn">
      <div class="turn-role">${escapeHtml(turn.from || "unknown")}</div>
      <div class="turn-text">${escapeHtml(truncateMiddle(turn.value || "", EXPLORER_DRAWER_TEXT_LIMIT))}</div>
    </div>`).join("");
  return `<details class="drawer-collapse" ${turns.length <= EXPLORER_DRAWER_TURN_LIMIT ? "open" : ""}>
      <summary>Conversation Preview (${fmtInt(Math.min(turns.length, EXPLORER_DRAWER_TURN_LIMIT))}/${fmtInt(turns.length)} turns)</summary>
      ${visibleTurns}
      ${hiddenCount ? `<div class="note drawer-truncate-note">Preview limited to first ${fmtInt(EXPLORER_DRAWER_TURN_LIMIT)} turns. ${fmtInt(hiddenCount)} more turns stay in raw JSON.</div>` : ""}
    </details>`;
}

function renderDrawerJson(detail) {
  const raw = JSON.stringify(detail, null, 2);
  const truncated = truncateMiddle(raw, EXPLORER_DRAWER_JSON_LIMIT);
  return `<details class="drawer-collapse">
      <summary>Raw JSON Preview${raw.length > EXPLORER_DRAWER_JSON_LIMIT ? ` (${fmtInt(EXPLORER_DRAWER_JSON_LIMIT)} char cap)` : ""}</summary>
      <div class="drawer-json">${escapeHtml(truncated)}</div>
      ${raw.length > EXPLORER_DRAWER_JSON_LIMIT ? `<div class="note drawer-truncate-note">Large payload truncated in drawer to keep long multi-turn samples responsive.</div>` : ""}
    </details>`;
}

function explorerPatchForTag(dim, label, sort = "quality_asc") {
  if (dim === "language") return { language: label, sort };
  if (dim === "intent") return { intent: label, sort };
  return { tagQueryAppend: [label], sort };
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

function renderTagBarControlsInline(scope) {
  if (!scope || (!scope.pass1 && !scope.pass2)) return "";
  const buttons = [
    ["hidden", "隐藏", "Hide tag bars"],
    ["relative", "归一化", "Scale by current panel max"],
    ["global-log", "全局对数", "Scale by section-wide log max"],
  ].map(([value, label, hint]) => (
    `<button class="segmented-btn ${STATE.tagBarMode === value ? "active" : ""}" type="button" data-tag-bar-mode="${escapeHtml(value)}" title="${escapeHtml(hint)}">${escapeHtml(label)}</button>`
  )).join("");
  return `<div class="toolbar-inline-copy" title="归一化按当前面板最大值缩放；全局对数按当前 section 内全部 tag 的计数做对数缩放。">Tag Bars</div><div class="segmented">${buttons}</div>`;
}

function renderPass1(pass1) {
  if (!pass1) return "";
  const sections = [];
  const globalTagCountMax = Object.values(pass1.distributions || {}).flatMap((dist) => (
    Object.values(dist || {}).map((value) => Number(value) || 0)
  )).reduce((maxValue, value) => Math.max(maxValue, value), 0);

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
        count: value,
      }));
    return `<div class="mini-panel"><h4>${escapeHtml(dim)}</h4>${renderRankingTable(entries, "Count", (entry) => (
      total > 0 ? `${entry.value} (${((entry.value / total) * 100).toFixed(1)}%)` : `${entry.value}`
    ), (entry) => explorerPatchForTag(dim, entry.label, "quality_asc"), {
      barMetricKey: "count",
      barMetricLabel: "Count",
      barColor: DIM_COLORS[dim] || "#2563eb",
      globalBarMax: globalTagCountMax,
    })}</div>`;
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
      ? `<div class="tags">${item.unused.map((tag) => `<button class="tag link-btn" type="button" data-explorer-patch="${escapeAttrJson(explorerPatchForTag(dim, tag, "quality_asc"))}">${escapeHtml(tag)}</button>`).join("")}</div>`
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
    let bucketAction = null;
    if (key === "value_score") bucketAction = (bucket) => ({ maxValue: String(bucket), sort: "value_asc" });
    if (key === "quality_overall") bucketAction = (bucket) => ({ maxQuality: String(bucket), sort: "quality_asc" });
    if (key === "selection_score") bucketAction = (bucket) => ({ minSelection: String(bucket), sort: "selection_desc" });
    if (key === "confidence") bucketAction = (bucket) => ({ maxConfidence: String((bucket / 10).toFixed(1)), sort: "confidence_asc" });
    return renderHistogram(HIST_LABELS[key] || key, bins, HIST_COLORS[key] || "#2563eb", scoreDistributions[key], bucketAction);
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
    const globalTagCountMax = Object.values(bucket).flatMap((dist) => (
      Object.values(dist || {}).map((info) => Number(info?.n) || 0)
    )).reduce((maxValue, value) => Math.max(maxValue, value), 0);
    const panels = Object.entries(bucket).map(([dim, dist]) => {
      const items = Object.entries(dist || {})
        .sort((a, b) => (b[1]?.mean || 0) - (a[1]?.mean || 0) || a[0].localeCompare(b[0]))
        .map(([label, info]) => ({
          label,
          mean: info.mean,
          n: info.n || 0,
        }));
      return `<div class="mini-panel"><h4>${escapeHtml(dim)}</h4>${renderRankingTable(items, "Score", (item) => `${fmt(item.mean, 2)} (n=${item.n})`, (item) => explorerPatchForTag(dim, item.label, title === "Selection By Tag" ? "selection_desc" : "quality_asc"), {
        barMetricKey: "n",
        barMetricLabel: "Count",
        barColor: DIM_COLORS[dim] || "#2563eb",
        globalBarMax: globalTagCountMax,
      })}</div>`;
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
      html += `<div class="mini-panel"><h4>Flags</h4>${renderBarChart(items, "#dc2626", null, (item) => ({ flagQuery: item.label, hasFlags: true, sort: "quality_asc" }))}</div>`;
    }
    html += `</div>`;
    sections.push(section("Analysis", html, "", true));
  }

  const perFile = pass2.per_file_summary || [];
  if (perFile.length > 1) {
    const sortColumn = FILE_RANKING_COLUMNS[STATE.fileRankingSort.key] || FILE_RANKING_COLUMNS.mean_value;
    const rows = sortPerFileSummary(perFile)
      .map((row) => `<tr>
        <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ sourcePath: row.file, sort: "quality_asc" })}">${escapeHtml(row.file)}</button></td>
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

  const thresholds = pass2.selection_thresholds || {};
  if (Object.keys(thresholds).length) {
    const rows = Object.entries(thresholds).map(([label, info]) => `<tr>
      <td>${escapeHtml(label)}</td>
      <td>${fmt(info.threshold, 1)}</td>
      <td>${info.count || 0}</td>
      <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ minSelection: String(info.threshold), sort: "selection_desc" })}">Inspect Samples</button></td>
    </tr>`).join("");
    sections.push(section("Selection Thresholds", `<table class="data-table"><thead><tr><th>Band</th><th>Threshold</th><th>Count</th><th>Action</th></tr></thead><tbody>${rows}</tbody></table>`, "", false));
  }

  const coverage = pass2.coverage_at_thresholds || {};
  if (Object.keys(coverage).length) {
    const rows = Object.entries(coverage).map(([threshold, info]) => `<tr>
      <td>${escapeHtml(threshold)}</td>
      <td>${info.retained || 0}</td>
      <td>${fmt((info.pct || 0) * 100, 1)}%</td>
      <td>${fmt((info.coverage || 0) * 100, 1)}%</td>
      <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ minValue: threshold, sort: "value_asc" })}">Inspect Retained</button></td>
    </tr>`).join("");
    sections.push(section("Coverage At Thresholds", `<table class="data-table"><thead><tr><th>Value Min</th><th>Retained</th><th>Sample %</th><th>Tag Coverage</th><th>Action</th></tr></thead><tbody>${rows}</tbody></table>`, "", false));
  }

  const flagImpact = pass2.flag_value_impact || {};
  if (Object.keys(flagImpact).length) {
    const rows = Object.entries(flagImpact).sort((a, b) => (a[1].mean_value || 0) - (b[1].mean_value || 0)).map(([flag, info]) => `<tr>
      <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ flagQuery: flag, hasFlags: true, sort: "quality_asc" })}">${escapeHtml(flag)}</button></td>
      <td>${fmt(info.mean_value, 2)}</td>
      <td>${info.count || 0}</td>
    </tr>`).join("");
    sections.push(section("Flag Impact", `<table class="data-table"><thead><tr><th>Flag</th><th>Mean Value</th><th>Count</th></tr></thead><tbody>${rows}</tbody></table>`, "", false));
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
      let bucketAction = null;
      if (label === "Conv Value") bucketAction = (bucket) => ({ convValueMin: String(bucket), sort: "conv_value_desc" });
      if (label === "Conv Selection") bucketAction = (bucket) => ({ convSelectionMin: String(bucket), sort: "conv_selection_desc" });
      if (label === "Peak Complexity") bucketAction = (bucket) => ({ peakComplexityMin: String(bucket), sort: "conv_value_desc" });
      histBlocks.push(renderHistogram(label, bins, color, null, bucketAction));
    }
  }
  const turnDist = conversation.turn_distribution || {};
  if (Object.keys(turnDist).length) {
    const items = Object.entries(turnDist).map(([turns, count]) => ({
      label: `${turns} turns`,
      value: count,
      display: `${count}`,
      turns: Number(turns),
    }));
    histBlocks.push(`<div class="mini-panel"><h4>Turn Distribution</h4>${renderBarChart(items, "#0f766e", null, (item) => ({ turnCountMin: String(item.turns), sort: "turn_count_desc" }))}</div>`);
  }
  if (histBlocks.length) {
    body += `<div class="grid" style="margin-top:14px">${histBlocks.join("")}</div>`;
  }
  return section("Conversation Aggregation", body, "", false);
}

function explorerSortOptions() {
  return [
    ["quality_asc", "Quality Asc"],
    ["value_asc", "Value Asc"],
    ["confidence_asc", "Confidence Asc"],
    ["selection_desc", "Selection Desc"],
    ["conv_value_desc", "Conv Value Desc"],
    ["conv_selection_desc", "Conv Selection Desc"],
    ["turn_count_desc", "Turn Count Desc"],
    ["rarity_desc", "Rarity Desc"],
    ["sample_id_asc", "Sample Id"],
  ];
}

function explorerQueryHasFilters(query) {
  if (!query) return false;
  return [
    query.tagQuery,
    query.language,
    query.intent,
    query.sourcePath,
    query.textQuery,
    query.flagQuery,
    query.convValueMin,
    query.convSelectionMin,
    query.peakComplexityMin,
    query.turnCountMin,
    query.minValue,
    query.maxValue,
    query.maxQuality,
    query.minSelection,
    query.maxConfidence,
    query.thinkingMode,
  ].some((value) => String(value || "").trim() !== "")
    || !!query.hasFlags
    || !!query.includeInherited;
}

function renderExplorerResults(rows) {
  if (!rows.length) {
    return `<div class="empty">Run a query or choose a preset to load sample previews.</div>`;
  }
  const body = rows.map((row) => {
    const tags = (row.flat_tags || []).slice(0, 6).map((tag) => `<span class="preview-tag">${escapeHtml(tag)}</span>`).join("");
    return `<tr>
      <td><button class="link-btn" data-explorer-detail="${escapeHtml(row.doc_id)}" data-explorer-scope="${escapeHtml(row.scope_id)}" data-explorer-detail-chunk="${escapeHtml(row.detail_chunk)}">${escapeHtml(row.sample_id || row.doc_id)}</button></td>
      <td class="preview-query" title="${escapeHtml(row.query_preview || "")}">${textPreview(row.query_preview)}</td>
      <td class="preview-response" title="${escapeHtml(row.response_preview || "")}">${textPreview(row.response_preview)}</td>
      <td><div class="preview-tags">${tags || `<span class="note">-</span>`}</div></td>
      <td>${row.value_score === null || row.value_score === undefined ? "-" : `<span class="${scoreClass(row.value_score)}">${fmt(row.value_score, 1)}</span>`}</td>
      <td>${row.quality_overall === null || row.quality_overall === undefined ? "-" : `<span class="${scoreClass(row.quality_overall)}">${fmt(row.quality_overall, 1)}</span>`}</td>
      <td>${row.selection_score === null || row.selection_score === undefined ? "-" : fmt(row.selection_score, 1)}</td>
      <td>${row.conv_value === null || row.conv_value === undefined ? "-" : `<span class="${scoreClass(row.conv_value)}">${fmt(row.conv_value, 1)}</span>`}</td>
      <td>${row.turn_count === null || row.turn_count === undefined ? "-" : fmtInt(row.turn_count)}</td>
      <td title="${escapeHtml(row.source_file || row.scope_path || "")}"><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ sourcePath: row.scope_path || row.source_file || "", sort: "quality_asc" })}">${escapeHtml(row.scope_path || row.source_file || "-")}</button></td>
    </tr>`;
  }).join("");
  return `<div class="result-table-wrap"><table class="data-table">
    <thead><tr><th>Sample</th><th>Query</th><th>Response</th><th>Tags</th><th>Value</th><th>Quality</th><th>Selection</th><th>Conv Value</th><th>Turns</th><th>Source</th></tr></thead>
    <tbody>${body}</tbody>
  </table></div>`;
}

function renderExplorerDrawer() {
  if (!STATE.explorer.detailDocId) return "";
  const detail = STATE.explorer.detailData;
  const loading = STATE.explorer.detailLoading;
  const header = loading
    ? `<div class="drawer-title">Loading sample…</div>`
    : `<div><div class="drawer-title">${escapeHtml((detail && detail.id) || STATE.explorer.detailDocId)}</div>
        <div class="drawer-sub">${escapeHtml((((detail || {}).metadata || {}).source_file) || STATE.explorer.detailScopeId || "")}</div></div>`;

  let body = `<div class="drawer-section"><div class="note">${loading ? "Loading full sample payload…" : escapeHtml((DATA.explorer || {}).detail_limit_notice || "")}</div></div>`;
  if (detail && !loading) {
    const labels = detail.labels || {};
    const value = detail.value || {};
    const flatTags = Object.entries(labels).flatMap(([key, val]) => {
      if (["confidence", "unmapped", "inherited", "inherited_from"].includes(key)) return [];
      if (Array.isArray(val)) return val.map((item) => `${key}:${item}`);
      return val ? [`${key}:${val}`] : [];
    });
    const cards = [
      ["Value", value.value_score],
      ["Quality", (value.quality || {}).overall],
      ["Selection", value.selection_score],
      ["Conv Value", (detail.conversation || {}).conv_value],
      ["Conv Sel", (detail.conversation || {}).conv_selection],
      ["Confidence", value.confidence],
      ["Thinking", value.thinking_mode || ((detail.metadata || {}).thinking_mode || "")],
      ["Turns", (detail.conversation || {}).turn_count || (detail.metadata || {}).total_turns || (detail.conversations || []).length],
      ["Peak Cplx", (detail.conversation || {}).peak_complexity],
    ].map(([label, raw]) => `<div class="kv"><div class="kv-label">${escapeHtml(label)}</div><div class="kv-value">${escapeHtml(raw === null || raw === undefined || raw === "" ? "-" : (typeof raw === "number" ? fmt(raw, label === "Confidence" ? 2 : 1) : raw))}</div></div>`).join("");
    body = `
      <div class="drawer-section"><div class="drawer-grid">${cards}</div></div>
      <div class="drawer-section"><h4>Tags</h4><div class="tags">${flatTags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("") || `<span class="note">No tags</span>`}</div></div>
      <div class="drawer-section"><h4>Conversation</h4>${renderDrawerTurns(detail.conversations || [])}</div>
      <div class="drawer-section"><h4>JSON</h4>${renderDrawerJson(detail)}</div>
    `;
  }

  return `<div class="drawer-backdrop" data-explorer-close="1"></div>
    <aside class="drawer">
      <div class="drawer-header">
        ${header}
        <button class="drawer-close" type="button" data-explorer-close="1">×</button>
      </div>
      <div class="drawer-body">${body}</div>
    </aside>`;
}

function renderExplorer(scope) {
  if (!scopeSupportsExplorer(scope)) return "";
  const totalSamples = explorerTotalSamples(scope);
  const query = STATE.explorer.queryModel;
  const candidateFileCount = explorerFileScopeIds(scope, query).length;
  const sortOptions = explorerSortOptions().map(([value, label]) => (
    `<option value="${escapeHtml(value)}" ${STATE.explorer.sort === value ? "selected" : ""}>${escapeHtml(label)}</option>`
  )).join("");
  const presets = [
    { id: "quality", label: "Lowest Quality", query: { sort: "quality_asc" } },
    { id: "value", label: "Lowest Value", query: { sort: "value_asc" } },
    { id: "confidence", label: "Low Confidence", query: { sort: "confidence_asc" } },
    { id: "python-debug", label: "Python + Debug", query: { tagQuery: "python, debug", sort: "quality_asc" } },
    { id: "flags", label: "Has Flags", query: { hasFlags: true, sort: "quality_asc" } },
    { id: "long-multiturn", label: "Long Multi-turn", query: { turnCountMin: "8", sort: "turn_count_desc" } },
  ];
  const status = STATE.explorer.busy
    ? `Scanning ${fmtInt(STATE.explorer.scopeDone)}/${fmtInt(STATE.explorer.scopeTotal)} files · ${fmtInt(STATE.explorer.scanned)} rows checked · ${fmtInt(STATE.explorer.matched)} matches`
    : (STATE.explorer.status || `Ready to scan ${fmtInt(totalSamples)} indexed previews in this scope.`);

  return section(
    "Sample Explorer",
    `<div class="explorer-toolbar">
      <div>
        <div class="explorer-summary">Progressively scan preview shards, stream large files chunk-by-chunk, keep the best ${fmtInt(STATE.explorer.limit)} matches in memory, load full sample details only when you open a drawer, and click tags / bars / threshold rows anywhere above to drill into the matching slices.</div>
        <div class="preset-row">${presets.map((preset) => `<button class="preset-btn" type="button" data-explorer-preset="${escapeHtml(preset.id)}">${escapeHtml(preset.label)}</button>`).join("")}</div>
      </div>
      <div class="result-count">${escapeHtml(status)}</div>
    </div>
    <div class="explorer-form">
      <div class="field"><label>Tags (comma)</label><input id="explorer-tag-query" type="text" value="${escapeHtml(query.tagQuery)}" placeholder="python, debug"></div>
      <div class="field"><label>Language</label><input id="explorer-language" type="text" value="${escapeHtml(query.language)}" placeholder="python"></div>
      <div class="field"><label>Intent</label><input id="explorer-intent" type="text" value="${escapeHtml(query.intent)}" placeholder="debug"></div>
      <div class="field"><label>Source File</label><input id="explorer-source-path" type="text" value="${escapeHtml(query.sourcePath)}" placeholder="multi_turn/coderforge"></div>
      <div class="field"><label>Text Contains</label><input id="explorer-text-query" type="text" value="${escapeHtml(query.textQuery)}" placeholder="OOM"></div>
      <div class="field"><label>Flag</label><input id="explorer-flag-query" type="text" value="${escapeHtml(query.flagQuery)}" placeholder="low-correctness"></div>
      <div class="field"><label>Conv Value ≥</label><input id="explorer-conv-value-min" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.convValueMin)}"></div>
      <div class="field"><label>Conv Sel ≥</label><input id="explorer-conv-selection-min" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.convSelectionMin)}"></div>
      <div class="field"><label>Peak Cplx ≥</label><input id="explorer-peak-complexity-min" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.peakComplexityMin)}"></div>
      <div class="field"><label>Turns ≥</label><input id="explorer-turn-count-min" type="number" min="1" step="1" value="${escapeHtml(query.turnCountMin)}"></div>
      <div class="field"><label>Value ≥</label><input id="explorer-min-value" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.minValue)}"></div>
      <div class="field"><label>Value ≤</label><input id="explorer-max-value" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.maxValue)}"></div>
      <div class="field"><label>Quality ≤</label><input id="explorer-max-quality" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.maxQuality)}"></div>
      <div class="field"><label>Selection ≥</label><input id="explorer-min-selection" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.minSelection)}"></div>
      <div class="field"><label>Confidence ≤</label><input id="explorer-max-confidence" type="number" min="0" max="1" step="0.01" value="${escapeHtml(query.maxConfidence)}"></div>
      <div class="field"><label>Thinking Mode</label><select id="explorer-thinking-mode"><option value="">Any</option><option value="fast" ${query.thinkingMode === "fast" ? "selected" : ""}>fast</option><option value="slow" ${query.thinkingMode === "slow" ? "selected" : ""}>slow</option></select></div>
      <div class="field"><label>Sort</label><select id="explorer-sort">${sortOptions}</select></div>
      <div class="field-check"><input id="explorer-has-flags" type="checkbox" ${query.hasFlags ? "checked" : ""}><label for="explorer-has-flags">Only samples with flags</label></div>
      <div class="field-check"><input id="explorer-include-inherited" type="checkbox" ${query.includeInherited ? "checked" : ""}><label for="explorer-include-inherited">Include inherited labels</label></div>
    </div>
    <div class="explorer-actions">
      <button class="action-btn" type="button" id="explorer-run">${STATE.explorer.busy ? "Scanning…" : "Run Query"}</button>
      <button class="action-btn" type="button" id="explorer-reset">Reset</button>
    </div>
    <div class="result-meta">
      <div class="result-count">Current scope: ${fmtInt(totalSamples)} indexed samples · ${fmtInt(candidateFileCount)} candidate files · showing top ${fmtInt(STATE.explorer.limit)} results sorted by ${escapeHtml((explorerSortOptions().find(([value]) => value === STATE.explorer.sort) || ["", STATE.explorer.sort])[1])}</div>
      <div class="result-count">${fmtInt(STATE.explorer.matched)} matches retained after scanning ${fmtInt(STATE.explorer.scanned)} rows</div>
    </div>
    ${renderExplorerResults(STATE.explorer.results)}
    ${renderExplorerDrawer()}`,
    `${fmtInt(totalSamples)} indexed samples`,
    true,
  );
}

function renderScopeContent(scope) {
  const chunks = [];
  chunks.push(renderChildren(scope));
  chunks.push(renderPass1(scope.pass1));
  chunks.push(renderPass2(scope.pass2));
  chunks.push(renderExplorer(scope));
  chunks.push(renderConversations(scope.conversation));
  const html = chunks.filter(Boolean).join("");
  return html || `<div class="empty">No dashboard data for this scope.</div>`;
}

function renderScope() {
  const scope = getCurrentScope();
  if (!scope) return;

  document.getElementById("scope-title").textContent = scope.label;
  document.getElementById("scope-path").textContent = scope.path || "Global overview";
  document.getElementById("scope-toolbar-extra").innerHTML = renderTagBarControlsInline(scope);
  renderBreadcrumbs(scope);
  document.getElementById("scope-content").innerHTML = renderScopeContent(scope);

  document.querySelectorAll("[data-jump]").forEach((node) => {
    node.addEventListener("click", () => setScope(node.getAttribute("data-jump")));
  });
  document.querySelectorAll("[data-file-sort]").forEach((node) => {
    node.addEventListener("click", () => toggleFileRankingSort(node.getAttribute("data-file-sort")));
  });
  document.querySelectorAll("[data-tag-bar-mode]").forEach((node) => {
    node.addEventListener("click", () => {
      STATE.tagBarMode = node.getAttribute("data-tag-bar-mode") || "relative";
      persistDashboardState();
      renderScope();
    });
  });
  const explorerRun = document.getElementById("explorer-run");
  if (explorerRun) {
    explorerRun.addEventListener("click", async () => {
      STATE.explorer.sort = (document.getElementById("explorer-sort") || {}).value || STATE.explorer.sort;
      await runExplorerQuery();
    });
  }
  const explorerReset = document.getElementById("explorer-reset");
  if (explorerReset) {
    explorerReset.addEventListener("click", () => {
      resetExplorerView(true);
      persistDashboardState();
      renderScope();
    });
  }
  const explorerSort = document.getElementById("explorer-sort");
  if (explorerSort) {
    explorerSort.addEventListener("change", () => {
      STATE.explorer.sort = explorerSort.value;
      persistDashboardState();
      if (STATE.explorer.results.length) {
        STATE.explorer.results.sort((left, right) => compareExplorerRows(left, right, STATE.explorer.sort));
        renderScope();
      }
    });
  }
  document.querySelectorAll("[data-explorer-patch]").forEach((node) => {
    node.addEventListener("click", async () => {
      const raw = node.getAttribute("data-explorer-patch");
      if (!raw) return;
      try {
        applyExplorerQueryPatch(JSON.parse(raw));
        await runExplorerQuery();
      } catch (error) {
        console.warn("Failed to apply explorer patch", error);
      }
    });
  });
  document.querySelectorAll("[data-explorer-preset]").forEach((node) => {
    node.addEventListener("click", async () => {
      const presetId = node.getAttribute("data-explorer-preset");
      if (presetId === "quality") setExplorerPreset({ sort: "quality_asc" });
      if (presetId === "value") setExplorerPreset({ sort: "value_asc" });
      if (presetId === "confidence") setExplorerPreset({ sort: "confidence_asc" });
      if (presetId === "python-debug") setExplorerPreset({ tagQuery: "python, debug", sort: "quality_asc" });
      if (presetId === "flags") setExplorerPreset({ hasFlags: true, sort: "quality_asc" });
      if (presetId === "long-multiturn") setExplorerPreset({ turnCountMin: "8", sort: "turn_count_desc" });
      await runExplorerQuery();
    });
  });
  document.querySelectorAll("[data-explorer-detail]").forEach((node) => {
    node.addEventListener("click", async () => {
      await openExplorerDetail(
        node.getAttribute("data-explorer-detail"),
        node.getAttribute("data-explorer-scope"),
        node.getAttribute("data-explorer-detail-chunk"),
      );
    });
  });
  document.querySelectorAll("[data-explorer-close]").forEach((node) => {
    node.addEventListener("click", closeExplorerDetail);
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

window.addEventListener("hashchange", () => {
  restoreDashboardState();
  render();
});

restoreDashboardState();
render();
if (explorerQueryHasFilters(STATE.explorer.queryModel) && scopeSupportsExplorer(getCurrentScope())) {
  runExplorerQuery();
}
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
