"""Interactive dashboard page for debugging the service.

Mounts a single self-contained HTML page at ``GET /dashboard``. The page
talks to the existing public endpoints only (healthz, /v1/models,
/v1/embeddings, /v1/databases family, etc.) — it does not call any
debug-only or hidden routes.

UI is **light-themed, minimalist**, Chinese-localised. The default
landing is an **overview** that summarises service health, model-load
state, and recent request activity. From the left sidebar the operator
can drill into the same endpoints the debug page used to expose: model
list, text/image/multimodal embeddings, rerank, model load/unload,
databases, collections, vectors, and search.

The font stack is unified on **Microsoft YaHei (微软雅黑)** with system
fallbacks; a separate mono stack is used only inside code panes and
HTTP headers.

A persistent **报文查看器** at the bottom shows the last HTTP request /
response in full (method, URL, headers, body, status, timing). A
collapsible **请求历史** sidebar on the right keeps the most recent 80
calls for quick re-inspection.

The signature visual is a row of dim-dots in the topbar — the dot
count equals the dimensions of the currently-loaded embedder, and a
subtle pulse ripples along the row when a request completes.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

# Marker for Edit operations: end-of-css / end-of-html-body / end-of-script.
_DASHBOARD_CSS_END = "/* ==== END_CSS ==== */"
_DASHBOARD_HTML_END = "<!-- ==== END_HTML ==== -->"
_DASHBOARD_JS_END = "/* ==== END_JS ==== */"

DASHBOARD_HTML_HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>vector-service · Dashboard</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23FFFFFF' stroke='%23E5E7EB' stroke-width='1'/%3E%3Ccircle cx='10' cy='16' r='2' fill='%230891B2'/%3E%3Ccircle cx='16' cy='12' r='2' fill='%230891B2' opacity='.7'/%3E%3Ccircle cx='16' cy='20' r='2' fill='%230891B2' opacity='.7'/%3E%3Ccircle cx='22' cy='16' r='2' fill='%230891B2' opacity='.4'/%3E%3C/svg%3E" />
<style>
/* ============ Tokens ============
   Light, minimalist theme. Surfaces stay near-white with subtle grey
   dividers; the teal accent is kept readable on a light background.
   Typography is unified on Microsoft YaHei (微软雅黑) with system
   fallbacks for non-CJK glyphs; a separate mono stack handles only
   code panes and HTTP headers. */
:root {
  --bg: #F7F8FA;
  --surface-1: #FFFFFF;
  --surface-2: #F2F4F7;
  --surface-3: #EAECF0;
  --border: #E5E7EB;
  --border-strong: #D1D5DB;
  --text: #1F2329;
  --text-dim: #4B5563;
  --text-muted: #9CA3AF;
  --accent: #0891B2;
  --accent-dim: #0E7490;
  --accent-soft: rgba(8, 145, 178, 0.08);
  --accent-glow: rgba(8, 145, 178, 0.30);
  --success: #059669;
  --success-soft: rgba(5, 150, 105, 0.10);
  --warn: #D97706;
  --warn-soft: rgba(217, 119, 6, 0.10);
  --danger: #DC2626;
  --danger-soft: rgba(220, 38, 38, 0.10);
  --code-bg: #F9FAFB;
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
  --shadow-md: 0 2px 8px rgba(15, 23, 42, 0.06);
  --shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.08);
  --shadow-accent: 0 0 0 1px rgba(8, 145, 178, 0.30), 0 0 12px rgba(8, 145, 178, 0.18);
  --font: "Microsoft YaHei", "微软雅黑", -apple-system, BlinkMacSystemFont,
          "Segoe UI", "PingFang SC", "Hiragino Sans GB", system-ui, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono",
          Consolas, "Source Han Mono SC", monospace;
  --r-sm: 4px;
  --r-md: 8px;
  --r-lg: 12px;
  --t-fast: 100ms ease-out;
  --t-base: 180ms cubic-bezier(.2, .8, .2, 1);
  --fz-xs: 11px;
  --fz-sm: 12px;
  --fz-md: 13px;
  --fz-lg: 14px;
  --fz-xl: 16px;
}

/* ============ Reset ============ */
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.55;
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  font-feature-settings: "tnum" 1, "ss01" 1;
  overflow: hidden;
}
a { color: var(--accent); text-decoration: none; transition: opacity var(--t-fast); }
a:hover { opacity: 0.8; }
button { font-family: inherit; }
input, textarea, select { font-family: inherit; color: inherit; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 5px;
  border: 2px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ============ Layout ============ */
.app {
  display: grid;
  grid-template-columns: 220px 1fr 320px;
  grid-template-rows: 64px 1fr 32px;
  grid-template-areas:
    "top    top     top"
    "side   main    log"
    "footer footer  footer";
  height: 100vh;
  min-width: 1024px;
}
.app--log-hidden { grid-template-columns: 220px 1fr 0; }
.app--log-hidden .log { display: none; }

/* ============ Topbar ============ */
.topbar {
  grid-area: top;
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 0 22px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, var(--surface-1) 0%, var(--bg) 100%);
  position: relative;
  overflow: hidden;
}
.topbar::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 600px 60px at 20% 50%, var(--accent-soft), transparent 70%);
  pointer-events: none;
  opacity: 0.6;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--mono);
  font-size: 15px;
  letter-spacing: 0.02em;
  color: var(--text);
  z-index: 1;
}
.brand-glyph {
  width: 22px; height: 22px;
  border-radius: var(--r-sm);
  background: var(--bg);
  border: 1px solid var(--accent);
  display: flex; align-items: center; justify-content: center;
  box-shadow: var(--shadow-accent);
}
.brand-glyph svg { width: 14px; height: 14px; }
.brand-name { font-weight: 600; }
.brand-sep { color: var(--text-muted); margin: 0 2px; }
.brand-tag {
  color: var(--accent);
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 3px 9px;
  border: 1px solid var(--accent-dim);
  border-radius: var(--r-sm);
  background: var(--accent-soft);
}

/* Signature: dim-dots row */
.signature {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-left: 8px;
  z-index: 1;
}
.dim-dots {
  display: flex;
  align-items: center;
  gap: 3px;
  height: 22px;
  padding: 0 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 99px;
  position: relative;
  overflow: hidden;
}
.dim-dot {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--text-muted);
  transition: background var(--t-base), transform var(--t-base);
  flex-shrink: 0;
}
.dim-dot--accent {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent-glow);
}
.dim-dot--pulse {
  animation: dot-pulse 600ms ease-out;
}
@keyframes dot-pulse {
  0% { transform: scale(1.6); box-shadow: 0 0 12px var(--accent-glow); }
  100% { transform: scale(1); box-shadow: none; }
}
.dim-dots-label {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}
.dim-dots-label .num {
  color: var(--accent);
  font-weight: 600;
}

.topbar-spacer { flex: 1; }

/* ============ Overview panel ============ */
.overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 22px;
}
@media (max-width: 1280px) {
  .overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.kpi {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  position: relative;
  overflow: hidden;
}
.kpi .label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.kpi .value {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.1;
  font-feature-settings: "tnum" 1;
}
.kpi .value .unit {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-dim);
  margin-left: 4px;
}
.kpi .sub {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
}
.kpi--ok .value { color: var(--success); }
.kpi--warn .value { color: var(--warn); }
.kpi--err .value { color: var(--danger); }
.kpi--accent .value { color: var(--accent); }

.bar {
  position: relative;
  height: 8px;
  background: var(--surface-3);
  border-radius: 4px;
  overflow: hidden;
}
.bar > span {
  display: block;
  height: 100%;
  background: var(--accent);
  transition: width var(--t-base);
  border-radius: inherit;
}
.bar--warn > span { background: var(--warn); }
.bar--err > span  { background: var(--danger); }

.gpu-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 12px 14px;
  margin-bottom: 8px;
}
.gpu-card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}
.gpu-card-head .name {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}
.gpu-card-head .meta {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-muted);
}
.gpu-card .row {
  display: grid;
  grid-template-columns: 80px 1fr 90px;
  gap: 10px;
  align-items: center;
  margin-bottom: 6px;
}
.gpu-card .row:last-child { margin-bottom: 0; }
.gpu-card .row label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  text-transform: none;
}
.gpu-card .row .num {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-dim);
  text-align: right;
}

.family-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px dashed var(--border);
}
.family-row:last-child { border-bottom: none; }
.family-row .name {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text);
  flex: 1;
  min-width: 0;
}
.family-row .id {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--accent);
  max-width: 50%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.family-row .id.empty { color: var(--text-muted); font-style: italic; }
.family-row .dim {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  flex-shrink: 0;
}

.info-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 12px;
}
.info-table th, .info-table td {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px dashed var(--border);
}
.info-table th {
  font-weight: 500;
  color: var(--text-muted);
  width: 30%;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 11px;
}
.info-table td { color: var(--text); }
.info-table td.muted { color: var(--text-muted); }

.db-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.db-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 99px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text);
}
.db-chip::before {
  content: "";
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
}

.indicators { display: flex; gap: 14px; z-index: 1; }
.led {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--mono); font-size: 12px;
  color: var(--text-dim); letter-spacing: 0.04em;
  padding: 5px 11px;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
}
.led::before {
  content: '';
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--text-muted);
}
.led.ok::before    { background: var(--success); box-shadow: 0 0 8px rgba(16,185,129,.6); }
.led.warn::before  { background: var(--warn);    box-shadow: 0 0 8px rgba(245,158,11,.6); }
.led.err::before   { background: var(--danger);  box-shadow: 0 0 8px rgba(239,68,68,.6); }
.led.checking::before {
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
@media (prefers-reduced-motion: reduce) {
  .led.checking::before, .dim-dot--pulse { animation: none; }
}

.topbar-actions { display: flex; gap: 6px; z-index: 1; }
.icon-btn {
  width: 30px; height: 30px;
  display: flex; align-items: center; justify-content: center;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  color: var(--text-dim);
  cursor: pointer;
  transition: all var(--t-fast);
}
.icon-btn:hover { border-color: var(--accent); color: var(--accent); }
.icon-btn.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }

/* ============ Sidebar ============ */
.sidebar {
  grid-area: side;
  background: var(--surface-1);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.sidebar-head {
  padding: 16px 18px 10px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.nav { flex: 1; overflow-y: auto; padding: 0 8px 16px; }
.nav-group { margin-bottom: 4px; }
.nav-group-label {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 10px 8px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
}
.nav-group-label::before {
  content: "";
  width: 3px; height: 3px;
  border-radius: 50%;
  background: var(--text-muted);
}
.nav-group-label .chev {
  margin-left: auto;
  transition: transform var(--t-fast);
}
.nav-group.collapsed .chev { transform: rotate(-90deg); }
.nav-group.collapsed .nav-items { display: none; }
.nav-items { display: flex; flex-direction: column; gap: 1px; }
.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px 8px 22px;
  font-size: 14px;
  color: var(--text-dim);
  cursor: pointer;
  border-radius: var(--r-sm);
  border-left: 2px solid transparent;
  margin-left: -2px;
  transition: all var(--t-fast);
}
.nav-item:hover { background: var(--surface-2); color: var(--text); }
.nav-item.active {
  background: var(--accent-soft);
  border-left-color: var(--accent);
  color: var(--accent);
  font-weight: 500;
}
.nav-item .badge {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 11px;
  padding: 1px 7px;
  border-radius: 99px;
  background: var(--surface-3);
  color: var(--text-muted);
}
.nav-item.active .badge {
  background: var(--accent);
  color: var(--bg);
}
.nav-item .dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 4px var(--success);
}

/* ============ Main ============ */
.main {
  grid-area: main;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
}
.breadcrumb {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 28px 8px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}
.breadcrumb .crumb-sep { color: var(--border-strong); }
.breadcrumb .crumb-current { color: var(--accent); }

.content {
  flex: 1;
  overflow-y: auto;
  padding: 12px 28px 28px;
}
.panel { display: none; }
.panel.active { display: block; animation: fade-in 200ms ease-out; }
@keyframes fade-in {
  from { opacity: 0; transform: translateY(-2px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ============ Sections ============ */
.section { margin-bottom: 32px; }
.section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.section-title {
  margin: 0;
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-sub {
  font-size: 13px;
  color: var(--text-muted);
}
.pill {
  display: inline-block;
  font-family: var(--mono);
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 99px;
  background: var(--surface-2);
  color: var(--text-dim);
  border: 1px solid var(--border);
  letter-spacing: 0.02em;
}
.pill.accent { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-dim); }
.pill.success { background: var(--success-soft); color: var(--success); border-color: var(--success); }
.pill.warn { background: var(--warn-soft); color: var(--warn); border-color: var(--warn); }
.pill.danger { background: var(--danger-soft); color: var(--danger); border-color: var(--danger); }

/* ============ Model cards (models panel) ============
   Loading/unloading is per-model (not per-family): each registered model
   id gets its own card. The card title shows the model_id, a small
   family tag marks the kind (embedder / image_embedder / multimodal /
   reranker), and the action button toggles load/unload for that exact
   id. Within a family only one model can be loaded at a time; the
   "已加载" state propagates from the server via GET /v1/models. */
.model-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 28px;
}
.model-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: all var(--t-base);
}
.model-card.loaded { border-color: var(--accent-dim); box-shadow: 0 0 0 1px rgba(34,211,238,.12) inset; }
.model-card.busy { opacity: 0.65; pointer-events: none; }
.model-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.model-title {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
}
.model-title .id {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.model-title .family-tag {
  font-weight: 400;
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 1px 6px;
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  background: var(--surface-2);
  flex-shrink: 0;
}
.status-pill {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.04em;
  padding: 2px 8px;
  border-radius: 99px;
  border: 1px solid var(--border);
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--text-muted);
  background: var(--surface-2);
}
.status-pill .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--text-muted);
}
.status-pill.loaded { color: var(--success); border-color: var(--success); background: var(--success-soft); }
.status-pill.loaded .dot { background: var(--success); box-shadow: 0 0 6px var(--success); animation: pulse-soft 2.4s ease-in-out infinite; }
@keyframes pulse-soft { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }

.model-current {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text);
  padding: 4px 0;
  min-height: 22px;
}
.model-current .dim { color: var(--text-dim); font-size: 12px; margin-left: 6px; }
.model-empty {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text-muted);
  font-style: italic;
}
.model-dots {
  height: 8px;
  display: flex;
  gap: 1px;
  overflow: hidden;
  border-radius: 2px;
  background: var(--bg);
  padding: 1px;
}
.model-dots .mini-dot {
  width: 2px;
  background: var(--border-strong);
  border-radius: 1px;
}
.model-dots .mini-dot.accent { background: var(--accent); box-shadow: 0 0 3px var(--accent-glow); }
.model-actions {
  display: flex;
  gap: 8px;
  padding-top: 4px;
  border-top: 1px dashed var(--border);
  margin-top: 4px;
}

/* ============ Forms ============ */
.row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
.row.split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.row.split > .row { margin-bottom: 0; }
.row.three { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 12px; }
.row.three > .row { margin-bottom: 0; }

label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
  font-weight: 600;
  display: flex;
  align-items: baseline;
  gap: 8px;
}
label .hint {
  text-transform: none;
  letter-spacing: 0;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 400;
}

input[type="text"], input[type="number"], select, textarea {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 10px 12px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  outline: none;
  transition: all var(--t-fast);
}
input:hover, select:hover, textarea:hover { border-color: var(--border-strong); }
input:focus, select:focus, textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
textarea { resize: vertical; min-height: 64px; line-height: 1.5; }
input[type="checkbox"] {
  accent-color: var(--accent);
  width: 14px; height: 14px;
}
input[type="file"] {
  padding: 6px;
  cursor: pointer;
}
input[type="file"]::-webkit-file-upload-button {
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 4px 10px;
  margin-right: 8px;
  cursor: pointer;
  font-family: var(--mono);
  font-size: 11px;
}
input[type="file"]::-webkit-file-upload-button:hover { border-color: var(--accent); color: var(--accent); }
.check-row { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-dim); }

/* ============ Buttons ============ */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.04em;
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-sm);
  padding: 8px 16px;
  cursor: pointer;
  transition: all var(--t-fast);
  white-space: nowrap;
}
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--bg);
  font-weight: 600;
}
.btn.primary:hover {
  background: var(--accent-dim);
  border-color: var(--accent-dim);
  color: var(--text);
  box-shadow: var(--shadow-accent);
}
.btn.ghost {
  background: transparent;
  border-style: dashed;
  border-color: var(--border-strong);
}
.btn.danger {
  color: var(--danger);
  border-color: var(--danger);
  background: var(--danger-soft);
}
.btn.danger:hover { background: var(--danger); color: var(--bg); }
.btn.sm { padding: 5px 11px; font-size: 11px; }

.actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 8px; }

/* ============ Lists ============ */
.list { display: flex; flex-direction: column; gap: 4px; margin-bottom: 16px; }
.list-item {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  background: var(--surface-1);
  font-family: var(--mono);
  font-size: 13px;
  cursor: pointer;
  transition: all var(--t-fast);
}
.list-item:hover { border-color: var(--accent); background: var(--surface-2); }
.list-item .name { color: var(--text); flex: 1; font-weight: 500; }
.list-item .meta { color: var(--text-dim); font-size: 12px; }

/* ============ Empty / error / loading ============ */
.empty {
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 13px;
  padding: 22px;
  border: 1px dashed var(--border-strong);
  border-radius: var(--r-md);
  text-align: center;
  background: var(--surface-1);
}
.empty.error { color: var(--danger); border-color: var(--danger); background: var(--danger-soft); }
.empty .hint { display: block; margin-top: 6px; font-size: 12px; color: var(--text-muted); }

.spinner {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid var(--border-strong);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 800ms linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ============ Field cards (structured rows) ============ */
.field-card {
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 12px 14px;
  background: var(--surface-1);
  margin-bottom: 8px;
  position: relative;
}
.field-card-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 10px;
}
.field-card-header .index-badge {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-muted);
  background: var(--surface-3);
  padding: 2px 8px;
  border-radius: var(--r-sm);
  letter-spacing: 0.04em;
}
.field-card-header .title {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text);
  font-weight: 600;
}
.field-card-header .remove {
  margin-left: auto;
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 16px;
  padding: 0 6px;
  cursor: pointer;
  line-height: 1;
}
.field-card-header .remove:hover { color: var(--danger); }
.field-card .grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}
.field-card .grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.field-card .field { display: flex; flex-direction: column; gap: 4px; }
.field-card .field label {
  font-size: 11px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}
.field-card .field input, .field-card .field select, .field-card .field textarea {
  padding: 7px 9px;
  font-size: 13px;
}
.field-card.dim-hidden .dim-cell,
.field-card.maxlen-hidden .maxlen-cell,
.field-card.params-hidden .params-cell { display: none; }

.add-row {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 11px;
  border: 1px dashed var(--border-strong);
  border-radius: var(--r-md);
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  margin-top: 4px;
  font-size: 13px;
  font-family: var(--mono);
  transition: all var(--t-fast);
}
.add-row:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }

.preset-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 12px;
}
.preset-grid .btn {
  text-align: left;
  padding: 10px 14px;
  font-family: var(--mono);
  font-size: 12px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 3px;
  height: auto;
}
.preset-grid .btn .preset-title { color: var(--text); font-weight: 600; }
.preset-grid .btn .preset-desc { color: var(--text-dim); font-size: 11px; }

details.collapsible {
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  background: var(--surface-1);
  margin-bottom: 12px;
}
details.collapsible > summary {
  cursor: pointer;
  padding: 12px 16px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-dim);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  list-style: none;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
details.collapsible > summary::-webkit-details-marker { display: none; }
details.collapsible > summary::before {
  content: '▸';
  color: var(--text-muted);
  display: inline-block;
  transition: transform var(--t-fast);
  font-size: 10px;
}
details.collapsible[open] > summary::before { transform: rotate(90deg); }
details.collapsible > .body { padding: 0 14px 14px; }

/* ============ Response pane ============ */
.response {
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  background: var(--surface-1);
  overflow: hidden;
  margin-top: 14px;
}
.response-head {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-dim);
}
.status-chip {
  padding: 2px 9px;
  border-radius: var(--r-sm);
  background: var(--surface-3);
  color: var(--text);
  font-weight: 600;
  letter-spacing: 0.04em;
  border: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 12px;
}
.status-chip.ok   { color: var(--success); border-color: var(--success); background: var(--success-soft); }
.status-chip.warn { color: var(--warn);    border-color: var(--warn);    background: var(--warn-soft); }
.status-chip.err  { color: var(--danger);  border-color: var(--danger);  background: var(--danger-soft); }
pre.code-pane {
  margin: 0;
  padding: 14px 16px;
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
  overflow: auto;
  max-height: 380px;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--code-bg);
}

/* ============ Traffic panel ============ */
.traffic {
  border-top: 1px solid var(--border);
  background: var(--surface-1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: 280px;
  flex-shrink: 0;
  transition: height var(--t-base);
}
.traffic.collapsed { height: 38px; }
.traffic.collapsed .traffic-body,
.traffic.collapsed .traffic-tabs { display: none; }
.traffic-head {
  padding: 0 16px;
  height: 38px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}
.traffic-head .title {
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
  font-weight: 600;
}
.traffic-head .sub {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-muted);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.traffic-head .right { display: flex; gap: 6px; }
.traffic-tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  padding: 0 16px;
  flex-shrink: 0;
}
.traffic-tab {
  padding: 8px 14px;
  background: none;
  border: none;
  cursor: pointer;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-dim);
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  letter-spacing: 0.04em;
}
.traffic-tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  font-weight: 600;
}
.traffic-body { flex: 1; overflow: auto; padding: 14px 16px; }
.traffic-section {
  display: grid;
  grid-template-columns: 90px 1fr;
  gap: 6px 14px;
  font-family: var(--mono);
  font-size: 13px;
  margin-bottom: 14px;
}
.traffic-section dt { color: var(--text-muted); }
.traffic-section dd { margin: 0; color: var(--text); word-break: break-all; }

/* ============ Log sidebar ============ */
.log {
  grid-area: log;
  background: var(--surface-1);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.log-head {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-dim);
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}
.log-list { flex: 1; overflow-y: auto; padding: 6px 8px; }
.log-item {
  padding: 8px 10px;
  border-radius: var(--r-sm);
  cursor: pointer;
  border: 1px solid var(--border);
  margin-bottom: 4px;
  background: var(--bg);
  transition: all var(--t-fast);
}
.log-item:hover { background: var(--surface-2); border-color: var(--border-strong); }
.log-item.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  box-shadow: 0 0 0 1px var(--accent-dim) inset;
}
.log-line { display: flex; align-items: center; gap: 6px; min-width: 0; }
.log-method {
  width: 46px;
  flex-shrink: 0;
  font-family: var(--mono);
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 0.04em;
}
.log-method.POST    { color: var(--accent); }
.log-method.PUT     { color: var(--warn); }
.log-method.DELETE  { color: var(--danger); }
.log-method.GET     { color: var(--success); }
.log-path {
  color: var(--text);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
  font-family: var(--mono);
  font-size: 12px;
}
.log-meta {
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 11px;
  padding-left: 52px;
  margin-top: 3px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

/* ============ Status bar ============ */
.statusbar {
  grid-area: footer;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 18px;
  background: var(--surface-1);
  border-top: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}
.statusbar .seg { display: flex; align-items: center; gap: 6px; }
.statusbar .seg .key { color: var(--text-muted); }
.statusbar .seg .val { color: var(--text-dim); }
.statusbar .seg .val.accent { color: var(--accent); }
.statusbar .right { margin-left: auto; display: flex; gap: 14px; }

/* ============ Tag tokens (search fields) ============ */
.tag-token {
  display: inline-block;
  font-family: var(--mono);
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 99px;
  background: var(--surface-2);
  color: var(--text-dim);
  border: 1px solid var(--border);
  cursor: pointer;
  user-select: none;
  margin: 0 4px 4px 0;
  transition: all var(--t-fast);
}
.tag-token:hover { border-color: var(--accent); color: var(--accent); }
.tag-token.active {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent);
}

/* ============ Rerank / emb result rows ============ */
.result-row {
  display: grid;
  grid-template-columns: 36px 70px 90px 1fr;
  gap: 12px;
  align-items: baseline;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  background: var(--surface-1);
  margin-bottom: 4px;
  font-family: var(--mono);
  font-size: 13px;
  transition: border-color var(--t-fast);
}
.result-row:hover { border-color: var(--accent); }
.result-rank  { color: var(--accent); font-weight: 600; }
.result-idx,
.result-score { color: var(--text-dim); font-size: 12px; }
.result-score { color: var(--text); font-weight: 600; }
.result-doc {
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ============ Hide & responsive ============ */
.hidden { display: none !important; }

@media (max-width: 1280px) {
  .model-grid { grid-template-columns: repeat(2, 1fr); }
}
"""

DASHBOARD_HTML_BODY = """
</style>
</head>
<body>
<div class="app" id="app">

  <!-- ============ Topbar ============ -->
  <header class="topbar">
    <div class="brand">
      <span class="brand-glyph">
        <svg viewBox="0 0 32 32" fill="none">
          <circle cx="10" cy="16" r="2" fill="#0891B2"/>
          <circle cx="16" cy="12" r="2" fill="#0891B2" opacity=".7"/>
          <circle cx="16" cy="20" r="2" fill="#0891B2" opacity=".7"/>
          <circle cx="22" cy="16" r="2" fill="#0891B2" opacity=".4"/>
        </svg>
      </span>
      <span class="brand-name">vector-service</span>
      <span class="brand-sep">·</span>
      <span class="brand-tag">Dashboard</span>
    </div>

    <div class="signature">
      <div class="dim-dots" id="dim-dots" aria-label="已加载 embedder 维度可视化"></div>
      <span class="dim-dots-label"><span class="num" id="dim-dots-num">—</span> 维</span>
    </div>

    <div class="topbar-spacer"></div>

    <div class="indicators">
      <div class="led" id="led-healthz"><span>healthz</span></div>
      <div class="led" id="led-readyz"><span>readyz</span></div>
    </div>

    <div class="topbar-actions">
      <button class="icon-btn" id="btn-toggle-log" title="切换请求历史侧栏" aria-label="切换请求历史">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="2" y="3" width="12" height="10" rx="1.5"/>
          <line x1="11" y1="3" x2="11" y2="13"/>
        </svg>
      </button>
    </div>
  </header>

  <!-- ============ Sidebar ============ -->
  <nav class="sidebar" aria-label="主导航">
    <div class="sidebar-head">导航 · Navigation</div>
    <div class="nav">

      <div class="nav-group" data-group="overview">
        <div class="nav-group-label">
          <span>概览</span>
          <svg class="chev" width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5">
            <polyline points="3,5 6,8 9,5"/>
          </svg>
        </div>
        <div class="nav-items">
          <div class="nav-item" data-view="overview">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="2" y="2" width="5" height="5" rx="1"/>
              <rect x="9" y="2" width="5" height="5" rx="1"/>
              <rect x="2" y="9" width="5" height="5" rx="1"/>
              <rect x="9" y="9" width="5" height="5" rx="1"/>
            </svg>
            <span>总览首页</span>
          </div>
        </div>
      </div>

      <div class="nav-group" data-group="models">
        <div class="nav-group-label">
          <span>模型</span>
          <svg class="chev" width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5">
            <polyline points="3,5 6,8 9,5"/>
          </svg>
        </div>
        <div class="nav-items">
          <div class="nav-item" data-view="models">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="2" y="4" width="12" height="8" rx="1.5"/>
              <circle cx="6" cy="8" r="1.2" fill="currentColor"/>
            </svg>
            <span>模型列表</span>
          </div>
          <div class="nav-item" data-view="embeddings">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M3 4h10M3 8h10M3 12h7"/>
            </svg>
            <span>文本嵌入</span>
          </div>
          <div class="nav-item" data-view="image-embeddings">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="2" y="3" width="12" height="10" rx="1.5"/>
              <circle cx="6" cy="7" r="1.2"/>
              <polyline points="4,12 7,9 10,11 14,7"/>
            </svg>
            <span>图像嵌入</span>
          </div>
          <div class="nav-item" data-view="multimodal-embeddings">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M2 4h6M2 8h6M2 12h6"/>
              <circle cx="12" cy="6" r="2.5"/>
              <rect x="9.5" y="9" width="5" height="5" rx="1"/>
            </svg>
            <span>图文嵌入</span>
          </div>
          <div class="nav-item" data-view="rerank">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <line x1="3" y1="4" x2="13" y2="4"/>
              <line x1="5" y1="8" x2="13" y2="8"/>
              <line x1="7" y1="12" x2="13" y2="12"/>
            </svg>
            <span>重排</span>
          </div>
          <div class="nav-item" data-view="text-similarity">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M3 6h6M3 10h6"/>
              <path d="M11 4l3 3-3 3M14 7H7" stroke-dasharray="2 2"/>
            </svg>
            <span>文本相似度</span>
          </div>
          <div class="nav-item" data-view="image-similarity">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="1.5" y="4" width="6" height="6" rx="1"/>
              <rect x="8.5" y="6" width="6" height="4" rx="1"/>
              <path d="M7.5 7h1" stroke-dasharray="1.5 1.5"/>
            </svg>
            <span>图像相似度</span>
          </div>
          <div class="nav-item" data-view="mm-similarity">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M2 5h5M2 8h5M2 11h3"/>
              <rect x="9" y="4" width="6" height="6" rx="1"/>
              <circle cx="11.5" cy="6.5" r="1"/>
              <path d="M9 10l1.5-1 1.5 1 2-1.5"/>
            </svg>
            <span>图文相似度</span>
          </div>
        </div>
      </div>

      <div class="nav-group" data-group="store">
        <div class="nav-group-label">
          <span>向量库</span>
          <svg class="chev" width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5">
            <polyline points="3,5 6,8 9,5"/>
          </svg>
        </div>
        <div class="nav-items">
          <div class="nav-item" data-view="databases">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <ellipse cx="8" cy="4" rx="5.5" ry="2"/>
              <path d="M2.5 4v4c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V4"/>
              <path d="M2.5 8v4c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V8"/>
            </svg>
            <span>数据库</span>
          </div>
          <div class="nav-item" data-view="collections">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="2.5" y="3" width="11" height="3" rx="1"/>
              <rect x="2.5" y="7" width="11" height="3" rx="1"/>
              <rect x="2.5" y="11" width="11" height="3" rx="1"/>
            </svg>
            <span>集合</span>
          </div>
          <div class="nav-item" data-view="vectors">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="4" cy="4" r="1.5"/>
              <circle cx="12" cy="4" r="1.5"/>
              <circle cx="4" cy="12" r="1.5"/>
              <circle cx="12" cy="12" r="1.5"/>
              <line x1="4" y1="4" x2="12" y2="12" stroke-dasharray="1.5 1.5"/>
              <line x1="12" y1="4" x2="4" y2="12" stroke-dasharray="1.5 1.5"/>
            </svg>
            <span>向量</span>
          </div>
          <div class="nav-item" data-view="search">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="7" cy="7" r="4"/>
              <line x1="10" y1="10" x2="13" y2="13"/>
            </svg>
            <span>检索</span>
          </div>
        </div>
      </div>

    </div>
  </nav>

  <!-- ============ Main ============ -->
  <main class="main">
    <nav class="breadcrumb" aria-label="路径">
      <span>vector-service</span>
      <span class="crumb-sep">›</span>
      <span id="crumb-cat">模型</span>
      <span class="crumb-sep" id="crumb-sub-sep" hidden>›</span>
      <span class="crumb-current" id="crumb-sub"></span>
    </nav>

    <div class="content" id="content">

      <!-- ===================== 总览 ===================== -->
      <div class="panel active" id="panel-overview">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">服务总览 <span class="pill accent">GET /v1/system/status</span></h3>
            <span class="section-sub">自动刷新 · 每 5 秒 · 含模型 / 向量库 / 机器指标</span>
          </div>

          <div class="overview-grid">
            <div class="kpi kpi--accent" id="kpi-version">
              <span class="label">版本</span>
              <span class="value" id="kpi-version-val">—</span>
              <span class="sub" id="kpi-version-sub">vector-service</span>
            </div>
            <div class="kpi" id="kpi-uptime">
              <span class="label">运行时长</span>
              <span class="value" id="kpi-uptime-val">—</span>
              <span class="sub" id="kpi-uptime-sub">进程启动以来</span>
            </div>
            <div class="kpi" id="kpi-models">
              <span class="label">已加载 / 已注册</span>
              <span class="value" id="kpi-models-val">—</span>
              <span class="sub" id="kpi-models-sub">跨所有族</span>
            </div>
            <div class="kpi" id="kpi-store">
              <span class="label">向量库</span>
              <span class="value" id="kpi-store-val">—</span>
              <span class="sub" id="kpi-store-sub">—</span>
            </div>
          </div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">模型注册与加载</h3>
            <span class="section-sub" id="models-summary-sub">—</span>
          </div>
          <div style="background:var(--surface-1);border:1px solid var(--border);border-radius:var(--r-md);padding:6px 16px;">
            <div class="family-row">
              <span class="name">embedder</span>
              <span class="id empty" id="fam-embedder-id">未加载</span>
              <span class="dim" id="fam-embedder-dim"></span>
            </div>
            <div class="family-row">
              <span class="name">image_embedder</span>
              <span class="id empty" id="fam-image-id">未加载</span>
              <span class="dim" id="fam-image-dim"></span>
            </div>
            <div class="family-row">
              <span class="name">multimodal_embedder</span>
              <span class="id empty" id="fam-mm-id">未加载</span>
              <span class="dim" id="fam-mm-dim"></span>
            </div>
            <div class="family-row">
              <span class="name">reranker</span>
              <span class="id empty" id="fam-reranker-id">未加载</span>
              <span class="dim" id="fam-reranker-dim"></span>
            </div>
          </div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">向量库连接状态</h3>
            <span class="section-sub" id="store-summary-sub">—</span>
          </div>
          <div style="background:var(--surface-1);border:1px solid var(--border);border-radius:var(--r-md);padding:14px 16px;">
            <table class="info-table">
              <tr><th>后端</th><td id="store-backend">—</td></tr>
              <tr><th>URI</th><td id="store-uri" class="muted">—</td></tr>
              <tr><th>状态</th><td id="store-status">—</td></tr>
              <tr><th>数据库数量</th><td id="store-db-count">—</td></tr>
            </table>
            <div class="db-chip-list" id="store-db-chips"></div>
          </div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">机器指标</h3>
            <span class="section-sub" id="os-summary-sub">—</span>
          </div>

          <div class="overview-grid" style="margin-bottom:14px;">
            <div class="kpi" id="kpi-cpu">
              <span class="label">CPU 使用率</span>
              <div class="bar" id="kpi-cpu-bar"><span></span></div>
              <span class="value" id="kpi-cpu-val">—</span>
              <span class="sub" id="kpi-cpu-sub">—</span>
            </div>
            <div class="kpi" id="kpi-mem">
              <span class="label">内存使用率</span>
              <div class="bar" id="kpi-mem-bar"><span></span></div>
              <span class="value" id="kpi-mem-val">—</span>
              <span class="sub" id="kpi-mem-sub">—</span>
            </div>
            <div class="kpi" id="kpi-disk">
              <span class="label">磁盘使用率</span>
              <div class="bar" id="kpi-disk-bar"><span></span></div>
              <span class="value" id="kpi-disk-val">—</span>
              <span class="sub" id="kpi-disk-sub">—</span>
            </div>
            <div class="kpi" id="kpi-proc">
              <span class="label">本进程 RSS</span>
              <span class="value" id="kpi-proc-val">—</span>
              <span class="sub" id="kpi-proc-sub">—</span>
            </div>
          </div>

          <div style="background:var(--surface-1);border:1px solid var(--border);border-radius:var(--r-md);padding:14px 16px;">
            <table class="info-table">
              <tr><th>操作系统</th><td id="os-system">—</td></tr>
              <tr><th>内核 / 版本</th><td id="os-release" class="muted">—</td></tr>
              <tr><th>架构</th><td id="os-machine">—</td></tr>
              <tr><th>主机名</th><td id="os-host">—</td></tr>
              <tr><th>Python</th><td id="os-python">—</td></tr>
            </table>
          </div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">GPU <span class="pill" id="gpu-count">0</span></h3>
            <span class="section-sub" id="gpu-sub">未检测到可用 GPU 时不显示设备卡</span>
          </div>
          <div id="gpu-list"><div class="empty">未检测到 GPU / pynvml 不可用。</div></div>
        </div>
      </div>

      <!-- ===================== 模型 ===================== -->
      <div class="panel" id="panel-models">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册模型 <span class="pill accent">GET /v1/models</span></h3>
            <span class="section-sub">加载 / 卸载均按 model_id 操作：POST /v1/models/{id}/load · /unload</span>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-refresh-models">刷新</button>
            <button class="btn sm" id="btn-models-auto-refresh">自动刷新：开</button>
          </div>
          <div class="model-grid" id="models-grid"></div>
        </div>
        <div class="empty hint">错误码速查：<code>409 model_busy</code>（同族并发 load/unload）/ <code>409 conflict_loaded</code>（同族不同 id，已加载）/ <code>409 not_loaded</code>（unload 空 slot）/ <code>503 model_load_failed</code>（权重下载/初始化失败）/ <code>404 model_not_found</code>（id 未注册）。</div>
        <div class="empty hint">点击列表中的模型可发起 <code>GET /v1/models/{id}</code> 并在底部报文面板查看响应。</div>
      </div>

      <!-- ===================== 嵌入 ===================== -->
      <div class="panel" id="panel-embeddings">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">文本嵌入 <span class="pill accent">POST /v1/embeddings</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>模型</label>
              <select id="emb-model"></select>
            </div>
            <div class="row">
              <label>类型 <span class="hint">单条字符串 或 字符串数组</span></label>
              <select id="emb-mode">
                <option value="text">单条</option>
                <option value="list">列表</option>
              </select>
            </div>
          </div>
          <div class="row">
            <label>输入文本 <span class="hint">type=list 时填 JSON 数组</span></label>
            <textarea id="emb-input" rows="3">你好世界</textarea>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-embed">生成嵌入</button>
          </div>
        </div>
      </div>

      <!-- ===================== 图像嵌入 ===================== -->
      <div class="panel" id="panel-image-embeddings">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册的图像嵌入后端 <span class="pill accent">GET /v1/models · type=image_embedder</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-image-emb-refresh-models">刷新</button>
          </div>
          <div class="list" id="image-emb-models-list"><div class="empty">点击"刷新"加载已注册的图像嵌入模型。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">生成图像嵌入 <span class="pill accent">POST /v1/image_embeddings</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>模型</label>
              <select id="image-emb-model"></select>
            </div>
            <div class="row">
              <label>输入方式</label>
              <select id="image-emb-mode">
                <option value="single">单张</option>
                <option value="list">多张（JSON 数组）</option>
              </select>
            </div>
          </div>

          <div class="row" id="image-emb-single-row">
            <label>选择图片 <span class="hint">本地文件 → 自动 base64 编码；MIME 自动从文件类型推断</span></label>
            <input type="file" id="image-emb-file" accept="image/png,image/jpeg,image/webp" />
          </div>

          <div class="row hidden" id="image-emb-list-row">
            <label>多张图片 <span class="hint">JSON 数组，每项 <code>{"data":"&lt;base64&gt;","mime":"image/png"}</code></span></label>
            <textarea id="image-emb-list" rows="6" placeholder='[{"data":"<base64>","mime":"image/png"}]'>[]</textarea>
          </div>

          <div class="row split" id="image-emb-mime-row">
            <div class="row">
              <label>MIME <span class="hint">留空 = 使用文件自身 type</span></label>
              <select id="image-emb-mime">
                <option value="">（自动）</option>
                <option value="image/png">image/png</option>
                <option value="image/jpeg">image/jpeg</option>
                <option value="image/webp">image/webp</option>
              </select>
            </div>
            <div class="row">
              <label>&nbsp;</label>
              <div id="image-emb-file-info" style="font-family:var(--mono);font-size:11px;color:var(--text-dim);">尚未选择文件。</div>
            </div>
          </div>

          <div class="actions">
            <button class="btn primary" id="btn-image-emb">生成嵌入</button>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3 class="section-title">结果</h3></div>
          <div id="image-emb-results"><div class="empty">尚无结果。点击"生成嵌入"查看响应。</div></div>
        </div>

        <div class="empty hint">base64 字符串不带 <code>data:</code> URI 前缀；MIME 必须与服务端 <code>allowed_mime</code> 配置一致。</div>
      </div>

      <!-- ===================== 图文嵌入 ===================== -->
      <div class="panel" id="panel-multimodal-embeddings">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册的图文嵌入后端 <span class="pill accent">GET /v1/models · type=multimodal_embedder</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-mm-emb-refresh-models">刷新</button>
          </div>
          <div class="list" id="mm-emb-models-list"><div class="empty">点击"刷新"加载已注册的图文嵌入模型。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">生成图文嵌入 <span class="pill accent">POST /v1/multimodal_embeddings</span></h3>
          </div>
          <div class="row">
            <label>模型</label>
            <select id="mm-emb-model"></select>
          </div>
          <div class="row">
            <label>输入项 <span class="hint">每项是文本或图片；输出顺序与请求顺序一一对应。文本与图片在同一向量空间（默认 512 维）。</span></label>
            <div id="mm-emb-items"></div>
            <button class="add-row" id="btn-mm-emb-add-text">＋ 添加文本项</button>
            <button class="add-row" id="btn-mm-emb-add-image" style="margin-top:6px;">＋ 添加图片项</button>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-mm-emb">生成嵌入</button>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3 class="section-title">结果</h3></div>
          <div id="mm-emb-results"><div class="empty">尚无结果。点击"生成嵌入"查看响应。</div></div>
        </div>

        <div class="empty hint">文本项仅支持中文（Chinese-CLIP 训练集）。base64 字符串不带 <code>data:</code> URI 前缀；MIME 必须与服务端 <code>VS_MULTIMODAL_EMBEDDING__ALLOWED_MIME</code> 一致。</div>
      </div>

      <!-- ===================== 文本相似度 ===================== -->
      <div class="panel" id="panel-text-similarity">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册的文本嵌入后端 <span class="pill accent">GET /v1/models · type=embedder</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-text-sim-refresh-models">刷新</button>
          </div>
          <div class="list" id="text-sim-models-list"><div class="empty">点击"刷新"加载已注册的文本嵌入模型。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">计算文本相似度 <span class="pill accent">POST /v1/text_similarity</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>模型</label>
              <select id="text-sim-model"></select>
            </div>
            <div class="row">
              <label>度量 <span class="hint">cosine / ip 越高越好 · l2 越低越好</span></label>
              <select id="text-sim-metric">
                <option value="cosine">cosine（默认）</option>
                <option value="ip">ip</option>
                <option value="l2">l2</option>
              </select>
            </div>
          </div>
          <div class="row">
            <label>查询文本 <span class="hint">单条字符串</span></label>
            <textarea id="text-sim-query" rows="2">无线鼠标</textarea>
          </div>
          <div class="row">
            <label>候选文档 <span class="hint">每行一条；服务端按行拆分</span></label>
            <textarea id="text-sim-docs" rows="5">蓝牙鼠标
机械键盘
蓝牙耳机
游戏手柄</textarea>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-text-sim">计算相似度</button>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3 class="section-title">结果</h3></div>
          <div id="text-sim-results"><div class="empty">尚无结果。点击"计算相似度"查看响应。</div></div>
        </div>

        <div class="empty hint">候选文档按行拆分；服务端会先调用 <code>embed_documents</code> 一次性嵌入查询与全部候选，再按所选度量计算两两分数。</div>
      </div>

      <!-- ===================== 图像相似度 ===================== -->
      <div class="panel" id="panel-image-similarity">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册的图像嵌入后端 <span class="pill accent">GET /v1/models · type=image_embedder</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-image-sim-refresh-models">刷新</button>
          </div>
          <div class="list" id="image-sim-models-list"><div class="empty">点击"刷新"加载已注册的图像嵌入模型。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">计算图像相似度 <span class="pill accent">POST /v1/image_similarity</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>模型</label>
              <select id="image-sim-model"></select>
            </div>
            <div class="row">
              <label>度量 <span class="hint">cosine / ip 越高越好 · l2 越低越好</span></label>
              <select id="image-sim-metric">
                <option value="cosine">cosine（默认）</option>
                <option value="ip">ip</option>
                <option value="l2">l2</option>
              </select>
            </div>
          </div>

          <div class="row" id="image-sim-single-row">
            <label>查询图片 <span class="hint">本地文件 → 自动 base64；MIME 自动从文件类型推断</span></label>
            <input type="file" id="image-sim-file" accept="image/png,image/jpeg,image/webp" />
          </div>

          <div class="row split" id="image-sim-mime-row">
            <div class="row">
              <label>MIME <span class="hint">留空 = 使用文件自身 type</span></label>
              <select id="image-sim-mime">
                <option value="">（自动）</option>
                <option value="image/png">image/png</option>
                <option value="image/jpeg">image/jpeg</option>
                <option value="image/webp">image/webp</option>
              </select>
            </div>
            <div class="row">
              <label>&nbsp;</label>
              <div id="image-sim-file-info" style="font-family:var(--mono);font-size:11px;color:var(--text-dim);">尚未选择文件。</div>
            </div>
          </div>

          <div class="row">
            <label>候选图片 <span class="hint">JSON 数组，每项 <code>{"data":"&lt;base64&gt;","mime":"image/png"}</code></span></label>
            <textarea id="image-sim-docs" rows="6" placeholder='[{"data":"<base64>","mime":"image/png"}]'>[]</textarea>
          </div>

          <div class="actions">
            <button class="btn primary" id="btn-image-sim">计算相似度</button>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3 class="section-title">结果</h3></div>
          <div id="image-sim-results"><div class="empty">尚无结果。点击"计算相似度"查看响应。</div></div>
        </div>

        <div class="empty hint">候选图片按 JSON 数组提交；查询走文件上传以便快速验证。base64 字符串不带 <code>data:</code> URI 前缀。</div>
      </div>

      <!-- ===================== 图文相似度 ===================== -->
      <div class="panel" id="panel-mm-similarity">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册的图文嵌入后端 <span class="pill accent">GET /v1/models · type=multimodal_embedder</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-mm-sim-refresh-models">刷新</button>
          </div>
          <div class="list" id="mm-sim-models-list"><div class="empty">点击"刷新"加载已注册的图文嵌入模型。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">计算图文相似度 <span class="pill accent">POST /v1/multimodal_similarity</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>模型</label>
              <select id="mm-sim-model"></select>
            </div>
            <div class="row">
              <label>度量 <span class="hint">cosine / ip 越高越好 · l2 越低越好</span></label>
              <select id="mm-sim-metric">
                <option value="cosine">cosine（默认）</option>
                <option value="ip">ip</option>
                <option value="l2">l2</option>
              </select>
            </div>
          </div>

          <div class="row">
            <label>查询项 <span class="hint">单条 JSON：<code>{"text":"..."}</code> 或 <code>{"image":{"data":"&lt;base64&gt;","mime":"image/png"}}</code></span></label>
            <textarea id="mm-sim-query" rows="2">{"text": "一只猫"}</textarea>
          </div>

          <div class="row">
            <label>候选项 <span class="hint">JSON 数组，每项与查询同构（text xor image）；跨模态有效（中文 ↔ 图）</span></label>
            <textarea id="mm-sim-docs" rows="5">[{"text": "狗"}, {"image": {"data": "<base64>", "mime": "image/png"}}]</textarea>
          </div>

          <div class="actions">
            <button class="btn primary" id="btn-mm-sim">计算相似度</button>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3 class="section-title">结果</h3></div>
          <div id="mm-sim-results"><div class="empty">尚无结果。点击"计算相似度"查看响应。</div></div>
        </div>

        <div class="empty hint">文本项仅支持中文（Chinese-CLIP 训练集）；查询与候选可以混合模态，因为 Chinese-CLIP 文本塔与图像塔在同一共享空间。</div>
      </div>

      <!-- ===================== 数据库 ===================== -->
      <div class="panel" id="panel-databases">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">数据库列表 <span class="pill accent">GET /v1/databases</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-refresh-dbs">刷新</button>
          </div>
          <div class="list" id="dbs-list"><div class="empty">点击"刷新"加载数据库。</div></div>
        </div>
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">新建数据库 <span class="pill accent">POST /v1/databases</span></h3>
          </div>
          <div class="row">
            <label>名称 <span class="hint">1-64 字符，字母 / 数字 / 下划线</span></label>
            <input type="text" id="new-db-name" placeholder="tenant-a" />
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-create-db">创建</button>
          </div>
        </div>
        <div class="empty hint">每个集合必须归属一个数据库；删除数据库会同时删除其下所有集合。</div>
      </div>

      <!-- ===================== 集合 ===================== -->
      <div class="panel" id="panel-collections">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">选择数据库</h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>当前数据库</label>
              <select id="colls-db"></select>
            </div>
            <div class="row">
              <label>&nbsp;</label>
              <button id="btn-colls-refresh-db" class="btn" style="align-self:flex-start;">↻ 重新加载数据库列表</button>
            </div>
          </div>
        </div>
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">集合列表 <span class="pill accent">GET /v1/databases/{db}/collections</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-refresh-colls">刷新</button>
          </div>
          <div class="list" id="colls-list"><div class="empty">选择数据库后点击"刷新"。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">新建集合 <span class="pill accent">POST /v1/databases/{db}/collections</span></h3>
          </div>

          <div class="row split">
            <div class="row">
              <label>集合名 <span class="hint">1-64 字符</span></label>
              <input type="text" id="new-coll-name" placeholder="products" />
            </div>
            <div class="row">
              <label>主键字段名 <span class="hint">必须出现在下方标量字段里，且 is_primary=true、dtype=varchar</span></label>
              <input type="text" id="new-coll-primary" value="id" />
            </div>
          </div>

          <details class="collapsible" open>
            <summary>标量字段（含主键）</summary>
            <div class="body">
              <div id="scalars-list"></div>
              <button class="add-row" id="btn-add-scalar">＋ 添加一个标量字段</button>
              <div class="preset-grid" style="margin-top:12px;">
                <button class="btn" data-preset-scalar="id+category+price">
                  <span class="preset-title">预设：id + category + price</span>
                  <span class="preset-desc">varchar 主键 + varchar 分类 + float 价格</span>
                </button>
                <button class="btn" data-preset-scalar="id+year">
                  <span class="preset-title">预设：id + year</span>
                  <span class="preset-desc">varchar 主键 + int32 年份</span>
                </button>
              </div>
            </div>
          </details>

          <details class="collapsible" open>
            <summary>向量字段</summary>
            <div class="body">
              <div id="vector-card"></div>
            </div>
          </details>

          <details class="collapsible">
            <summary>索引参数</summary>
            <div class="body">
              <div id="index-list"></div>
              <button class="add-row" id="btn-add-index">＋ 添加一个索引</button>
              <div class="preset-grid" style="margin-top:12px;">
                <button class="btn" data-preset-index="hnsw-cosine">
                  <span class="preset-title">HNSW · 余弦</span>
                  <span class="preset-desc">M=16, efConstruction=200</span>
                </button>
                <button class="btn" data-preset-index="ivf-l2">
                  <span class="preset-title">IVF_FLAT · 欧氏</span>
                  <span class="preset-desc">nlist=64</span>
                </button>
                <button class="btn" data-preset-index="diskann-ip">
                  <span class="preset-title">DISKANN · 内积</span>
                  <span class="preset-desc">默认参数</span>
                </button>
                <button class="btn" data-preset-index="default">
                  <span class="preset-title">使用默认</span>
                  <span class="preset-desc">由服务自动构建 HNSW + 向量字段的 metric</span>
                </button>
              </div>
            </div>
          </details>

          <div class="actions">
            <button class="btn primary" id="btn-create-coll">创建集合</button>
          </div>
        </div>
      </div>

      <!-- ===================== 向量 ===================== -->
      <div class="panel" id="panel-vectors">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">向量写入 <span class="pill accent">PUT /v1/databases/{db}/collections/{coll}/vectors</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>数据库</label>
              <select id="vec-db"></select>
            </div>
            <div class="row">
              <label>集合</label>
              <select id="vec-coll"></select>
            </div>
          </div>
          <div class="row split">
            <div class="row">
              <label>主键字段名</label>
              <input type="text" id="vec-primary" value="id" />
            </div>
            <div class="row">
              <label>向量字段名</label>
              <input type="text" id="vec-vecfield" value="vector" />
            </div>
          </div>
          <div class="row">
            <label>输入方式</label>
            <select id="vec-mode">
              <option value="texts">由服务端嵌入（texts）</option>
              <option value="vectors">直接提供向量</option>
            </select>
          </div>
          <div class="row">
            <label>主键值数组（ids）<span class="hint">JSON 数组，元素个数 = texts/vectors/fields 的长度</span></label>
            <textarea id="vec-ids" rows="2">["sku-1", "sku-2"]</textarea>
          </div>
          <div class="row" id="vec-texts-row">
            <label>待嵌入文本（texts）<span class="hint">JSON 数组，仅在「服务端嵌入」模式下使用</span></label>
            <textarea id="vec-texts" rows="3">["无线鼠标", "机械键盘"]</textarea>
          </div>
          <div class="row hidden" id="vec-embs-row">
            <label>已嵌入向量（vectors）<span class="hint">JSON 数组，每项一个数组，长度 = 向量 dim</span></label>
            <textarea id="vec-embs" rows="3">[[0.1, 0.2, 0.3, 0.4]]</textarea>
          </div>

          <details class="collapsible">
            <summary>附加字段（fields，按行对齐 ids）</summary>
            <div class="body">
              <div class="row">
                <label>字段集 <span class="hint">JSON 数组，每项 <code>{字段名: 值}</code>；可空</span></label>
                <textarea id="vec-fields" rows="3">[{"category":"mouse","price":29.9},{"category":"keyboard","price":119.0}]</textarea>
              </div>
            </div>
          </details>

          <div class="actions">
            <button class="btn primary" id="btn-upsert">写入（PUT）</button>
            <button class="btn" id="btn-fetch">按主键获取（POST）</button>
            <button class="btn danger" id="btn-delete">按主键删除（POST）</button>
          </div>
        </div>
      </div>

      <!-- ===================== 检索 ===================== -->
      <div class="panel" id="panel-search">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">向量检索 <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/search</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>数据库</label>
              <select id="srch-db"></select>
            </div>
            <div class="row">
              <label>集合</label>
              <select id="srch-coll"></select>
            </div>
          </div>
          <div class="row split">
            <div class="row">
              <label>主键字段名</label>
              <input type="text" id="srch-primary" value="id" />
            </div>
            <div class="row">
              <label>向量字段名</label>
              <input type="text" id="srch-vecfield" value="vector" />
            </div>
          </div>
          <div class="row split">
            <div class="row">
              <label>查询方式</label>
              <select id="srch-mode">
                <option value="text">服务端嵌入（query_text）</option>
                <option value="emb">直接提供向量（query_vector）</option>
              </select>
            </div>
            <div class="row">
              <label>top_k</label>
              <input type="number" id="srch-topk" min="1" max="1000" value="5" />
            </div>
          </div>
          <div class="row" id="srch-text-row">
            <label>查询文本</label>
            <input type="text" id="srch-text" value="电脑外设" />
          </div>
          <div class="row hidden" id="srch-emb-row">
            <label>查询向量 <span class="hint">JSON 数组，长度 = 向量 dim</span></label>
            <textarea id="srch-emb" rows="2">[0.1, 0.2, 0.3, 0.4]</textarea>
          </div>

          <details class="collapsible">
            <summary>过滤表达式（Milvus 原生）</summary>
            <div class="body">
              <div class="row">
                <label>filter_expr <span class="hint">例如 <code>category == 'mouse' and price &lt; 100</code>；空 = 不过滤</span></label>
                <textarea id="srch-filter" rows="2" placeholder="category == 'mouse' and price < 100"></textarea>
              </div>
              <div id="srch-filter-tokens" style="margin-bottom:8px;"></div>
              <div id="srch-filter-fields" class="tag-token-list"></div>
            </div>
          </details>

          <details class="collapsible">
            <summary>返回字段（output_fields）</summary>
            <div class="body">
              <div class="row">
                <label>输出字段 <span class="hint">留空 = 只返回主键。点击下方字段名可加入 / 移除</span></label>
                <div id="srch-output-tokens"></div>
                <div id="srch-output-fields" class="tag-token-list" style="margin-top:6px;"></div>
              </div>
            </div>
          </details>

          <div class="actions">
            <button class="btn primary" id="btn-search">检索</button>
          </div>
        </div>
      </div>

      <!-- ===================== 重排 ===================== -->
      <div class="panel" id="panel-rerank">
        <div class="section">
          <div class="section-head">
            <h3 class="section-title">已注册的 Reranker 后端 <span class="pill accent">GET /v1/models · type=reranker</span></h3>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-rerank-refresh-models">刷新</button>
          </div>
          <div class="list" id="rerank-models-list"><div class="empty">点击"刷新"加载已注册的 reranker。</div></div>
        </div>

        <div class="section">
          <div class="section-head">
            <h3 class="section-title">执行重排 <span class="pill accent">POST /v1/rerank</span></h3>
          </div>
          <div class="row split">
            <div class="row">
              <label>模型 <span class="hint">留空 = 使用服务端默认</span></label>
              <select id="rerank-model"></select>
            </div>
            <div class="row">
              <label>top_n <span class="hint">留空 = 由服务端决定</span></label>
              <input type="number" id="rerank-topn" min="1" placeholder="默认" />
            </div>
          </div>
          <div class="row">
            <label>查询文本（query）</label>
            <input type="text" id="rerank-query" placeholder="例如：电脑外设推荐" value="电脑外设推荐" />
          </div>
          <div class="row">
            <label>文档输入方式</label>
            <select id="rerank-mode">
              <option value="lines">每行一条（推荐）</option>
              <option value="json">JSON 数组</option>
            </select>
          </div>
          <div class="row">
            <label>文档列表（documents）</label>
            <textarea id="rerank-docs" rows="6">无线鼠标 静音版 支持蓝牙与 2.4G
机械键盘 红轴 87 键 RGB 背光
蓝牙耳机 主动降噪 续航 30 小时
游戏手柄 PC 与主机双模 霍尔摇杆
人体工学椅 网椅 可调节腰托</textarea>
          </div>
          <div class="actions">
            <button class="btn primary" id="btn-rerank">执行重排</button>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3 class="section-title">结果</h3></div>
          <div id="rerank-results"><div class="empty">尚无结果。点击"执行重排"查看响应。</div></div>
        </div>

        <div class="empty hint">响应中的 <code>results[].index</code> 是请求文档数组里的下标；前端会用它反查你提交的原文以方便查看。</div>
      </div>

    </div>

    <!-- ============ Traffic ============ -->
    <section class="traffic" id="traffic">
      <div class="traffic-head">
        <span class="title">报文</span>
        <span class="sub" id="traffic-summary">尚无调用</span>
        <div class="right">
          <button class="btn sm" id="btn-traffic-toggle">折叠</button>
          <button class="btn sm" id="btn-traffic-copy">复制</button>
          <button class="btn sm" id="btn-traffic-clear">清空</button>
        </div>
      </div>
      <div class="traffic-tabs" role="tablist">
        <button class="traffic-tab active" data-traffic="request">REQUEST</button>
        <button class="traffic-tab" data-traffic="response">RESPONSE</button>
      </div>
      <div class="traffic-body">
        <div id="traffic-request">
          <div class="empty">选择上方任一标签页的操作即可发起调用，本面板会实时展示请求与响应报文。</div>
        </div>
        <div id="traffic-response" hidden>
          <div class="empty">尚无响应。</div>
        </div>
      </div>
    </section>
  </main>

  <!-- ============ Log sidebar ============ -->
  <aside class="log" aria-label="请求历史">
    <div class="log-head">
      <span>请求历史</span>
      <button class="btn sm" id="btn-clear-log">清空</button>
    </div>
    <div class="log-list" id="log">
      <div class="empty" style="margin:8px;">还没有任何请求。</div>
    </div>
  </aside>

  <!-- ============ Status bar ============ -->
  <footer class="statusbar">
    <span class="seg"><span class="key">SERVICE</span><span class="val">vector-service</span></span>
    <span class="seg"><span class="key">VERSION</span><span class="val accent" id="sb-version">0.2.0</span></span>
    <span class="seg"><span class="key">EMBEDDING</span><span class="val" id="sb-embedding">bge-m3</span></span>
    <span class="seg"><span class="key">STORE</span><span class="val" id="sb-store">milvus</span></span>
    <span class="right">
      <span class="seg"><span class="key">LOADED</span><span class="val" id="sb-loaded">0/0</span></span>
      <span class="seg"><span class="key">UPTIME</span><span class="val" id="sb-uptime">0h 0m</span></span>
    </span>
  </footer>

</div>
""" + _DASHBOARD_HTML_END


# JavaScript kept as a separate constant for readability; concatenated at
# module load time. The script is ~1500 lines: existing panel logic is
# preserved (same function names, same DOM ids) plus a new overview
# subsystem, signature dim-dots, sidebar nav, and a sparkline renderer.
DASHBOARD_JS = r"""
<script>
(function () {
'use strict';

/* ============ Utilities ============ */
var $  = function (s, p) { return (p || document).querySelector(s); };
var $$ = function (s, p) { return Array.from((p || document).querySelectorAll(s)); };
var enc = function (s) { return encodeURIComponent(s); };
var safeParse = function (s) { try { return JSON.parse(s); } catch (_e) { return null; } };
var fmtJson = function (v) { try { return JSON.stringify(v, null, 2); } catch (_e) { return String(v); } };
var nowTs = function () { return new Date().toLocaleTimeString(); };
var escapeHtml = function (s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
};
var statusTextCN = function (code) {
  var map = {
    200: '成功', 201: '已创建', 204: '无内容',
    400: '请求错误', 401: '未授权', 403: '禁止访问', 404: '未找到',
    409: '冲突', 422: '参数错误',
    500: '服务器错误', 502: '网关不可达', 503: '服务不可用',
  };
  return map[code] || '';
};

/* ============ App state ============ */
var navState = { view: 'overview' };
var trafficState = { entries: [], activeIdx: -1 };
var modelsState = { data: [], auto: true, timer: null, busy: {} };
var overviewState = { data: null, timer: null };
var logState = { entries: [] };
var appStart = Date.now();

/* ============ HTTP wrapper ============ */
async function api(method, path, body) {
  var init = { method: method, headers: {} };
  var reqBodyText = '';
  if (body !== undefined && body !== null) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
    reqBodyText = JSON.stringify(body, null, 2);
  }
  var t0 = performance.now();
  var resp, text, error;
  try {
    resp = await fetch(path, init);
    text = await resp.text();
  } catch (e) {
    error = (e && e.message) || String(e);
  }
  var durationMs = Math.round(performance.now() - t0);
  var prettyText = text;
  var payload = text;
  if (text) {
    try {
      var parsed = JSON.parse(text);
      payload = parsed;
      prettyText = JSON.stringify(parsed, null, 2);
    } catch (_e) { /* leave as-is */ }
  }
  var respHeaders = {};
  if (resp && resp.headers) {
    resp.headers.forEach(function (v, k) {
      if (['content-type', 'content-length', 'x-request-id', 'date', 'server'].indexOf(k.toLowerCase()) >= 0) {
        respHeaders[k] = v;
      }
    });
  }
  var entry = {
    method: method, url: path,
    req: reqBodyText,
    reqHeaders: { 'Content-Type': body !== undefined ? 'application/json' : '(none)' },
    status: resp ? resp.status : 0,
    durationMs: durationMs,
    respSize: text ? text.length : 0,
    respHeaders: respHeaders,
    resp: prettyText,
    error: error,
    payload: payload,
    time: nowTs(),
  };
  appendLog(entry);
  appendTraffic(entry);
  pulseDimDots();
  if (error) throw new Error(error);
  return { status: resp.status, ok: resp.ok && resp.status < 400, payload: payload };
}

/* ============ Signature: dim-dots ============ */
var MAX_DIM_DOTS = 64; /* visual cap; real dims can be 768+ */

function currentEmbedderDim(data) {
  if (!data || !data.length) return 0;
  /* ``m.loaded`` is the canonical signal — for embedders the dimension
     is populated only when loaded, but for rerankers ``dimensions`` is
     permanently ``null`` so we have to look at ``m.loaded``. */
  var loaded = data.filter(function (m) { return m.type === 'embedder' && m.loaded; });
  if (!loaded.length) return 0;
  return loaded[0].dimensions || 0;
}

function renderDimDots(dim) {
  var root = $('#dim-dots');
  if (!root) return;
  var shown = Math.min(dim, MAX_DIM_DOTS);
  var html = '';
  for (var i = 0; i < shown; i++) {
    html += '<span class="dim-dot' + (i === 0 ? ' dim-dot--accent' : '') + '"></span>';
  }
  if (dim > MAX_DIM_DOTS) {
    html += '<span class="dim-dots-label" style="margin-left:6px;">+' + (dim - MAX_DIM_DOTS) + '</span>';
  }
  root.innerHTML = html || '<span class="dim-dots-label">未加载</span>';
  var num = $('#dim-dots-num');
  if (num) num.textContent = dim || '—';
}

var pulseTimer = null;
function pulseDimDots() {
  var dots = $$('#dim-dots .dim-dot');
  if (!dots.length || pulseTimer) return;
  var i = 0;
  pulseTimer = setInterval(function () {
    if (i >= dots.length) { clearInterval(pulseTimer); pulseTimer = null; return; }
    var d = dots[i];
    d.classList.add('dim-dot--pulse');
    setTimeout(function () { d.classList.remove('dim-dot--pulse'); }, 600);
    i++;
  }, 30);
}

/* ============ Log sidebar ============ */
var logEl = $('#log');
function appendLog(entry) {
  if (!logEl) return;
  var empty = logEl.querySelector('.empty');
  if (empty) empty.remove();
  logState.entries.unshift(entry);
  if (logState.entries.length > 80) logState.entries.pop();
  var item = document.createElement('div');
  item.className = 'log-item';
  var cls = !entry.status || entry.status >= 500 || entry.status === 0 ? 'err'
          : entry.status >= 400 ? 'warn'
          : entry.status >= 200 && entry.status < 300 ? 'ok' : '';
  var cn = statusTextCN(entry.status || 0);
  item.innerHTML =
    '<div class="log-line">' +
      '<span class="log-method ' + entry.method + '">' + entry.method + '</span>' +
      '<span class="log-path" title="' + escapeHtml(entry.url) + '">' + escapeHtml(entry.url) + '</span>' +
      '<span class="status-chip ' + cls + '">' + (entry.status || '×') + '</span>' +
    '</div>' +
    '<div class="log-meta">' +
      '<span>' + (entry.durationMs != null ? entry.durationMs + 'ms' : '—') + '</span>' +
      (cn ? '<span>· ' + cn + '</span>' : '') +
      (entry.error ? '<span style="color:var(--danger);">· ' + escapeHtml(entry.error) + '</span>' : '') +
      '<span style="margin-left:auto;">' + entry.time + '</span>' +
    '</div>';
  (function (e, el) {
    el.addEventListener('click', function () {
      trafficState.activeIdx = trafficState.entries.indexOf(e);
      setTrafficSummary();
      renderTrafficPanes();
      $$('.log-item').forEach(function (x) { x.classList.toggle('active', x === el); });
    });
  })(entry, item);
  logEl.insertBefore(item, logEl.firstChild);
  while (logEl.children.length > 80) logEl.removeChild(logEl.lastChild);
  $$('.log-item').forEach(function (x) { x.classList.remove('active'); });
  item.classList.add('active');
}
$('#btn-clear-log').addEventListener('click', function () {
  logState.entries = [];
  logEl.innerHTML = '<div class="empty" style="margin:8px;">还没有任何请求。</div>';
});

/* ============ Traffic panel ============ */
function setTrafficSummary() {
  var sub = $('#traffic-summary');
  if (!sub) return;
  if (trafficState.activeIdx < 0) { sub.textContent = '尚无调用'; return; }
  var e = trafficState.entries[trafficState.activeIdx];
  if (!e) return;
  var dur = e.durationMs != null ? e.durationMs + ' ms' : '';
  var size = e.respSize != null ? e.respSize + ' B' : '';
  sub.textContent = e.method + ' ' + e.url + '  ·  ' + (e.status || '—') + ' ' + statusTextCN(e.status || '') + '  ·  ' + dur + '  ·  ' + size;
}
function renderTrafficPanes() {
  var e = trafficState.activeIdx >= 0 ? trafficState.entries[trafficState.activeIdx] : null;
  var reqEl = $('#traffic-request');
  if (!e || !e.req) {
    reqEl.innerHTML = '<div class="empty">尚无请求。</div>';
  } else {
    reqEl.innerHTML =
      '<dl class="traffic-section">' +
        '<dt>方法</dt><dd>' + escapeHtml(e.method) + '</dd>' +
        '<dt>地址</dt><dd>' + escapeHtml(e.url) + '</dd>' +
        '<dt>请求头</dt><dd>' + renderHeaders(e.reqHeaders) + '</dd>' +
      '</dl>' +
      '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">请求体</div>' +
      '<pre class="code-pane">' + escapeHtml(e.req) + '</pre>';
  }
  var resEl = $('#traffic-response');
  if (!e || e.status == null) {
    resEl.innerHTML = '<div class="empty">尚无响应。</div>';
  } else {
    var errBlock = e.error
      ? '<div class="empty error">请求失败：' + escapeHtml(e.error) + '</div>'
      : '';
    resEl.innerHTML =
      errBlock +
      '<dl class="traffic-section">' +
        '<dt>状态</dt><dd>' + (e.status || '—') + ' ' + escapeHtml(statusTextCN(e.status || 0)) + '</dd>' +
        '<dt>耗时</dt><dd>' + (e.durationMs != null ? e.durationMs + ' ms' : '—') + '</dd>' +
        '<dt>字节</dt><dd>' + (e.respSize != null ? e.respSize + ' B' : '—') + '</dd>' +
        '<dt>响应头</dt><dd>' + renderHeaders(e.respHeaders) + '</dd>' +
      '</dl>' +
      '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">响应体</div>' +
      '<pre class="code-pane">' + escapeHtml(e.resp || '') + '</pre>';
  }
}
function renderHeaders(headers) {
  if (!headers) return '—';
  var lines = Object.keys(headers).map(function (k) { return escapeHtml(k) + ': ' + escapeHtml(String(headers[k])); });
  return lines.length ? lines.join('<br/>') : '—';
}
function appendTraffic(entry) {
  trafficState.entries.push(entry);
  if (trafficState.entries.length > 50) trafficState.entries.shift();
  trafficState.activeIdx = trafficState.entries.length - 1;
  setTrafficSummary();
  renderTrafficPanes();
}
$$('.traffic-tab').forEach(function (t) {
  t.addEventListener('click', function () {
    $$('.traffic-tab').forEach(function (x) { x.classList.toggle('active', x === t); });
    var which = t.dataset.traffic;
    $('#traffic-request').hidden  = which !== 'request';
    $('#traffic-response').hidden = which !== 'response';
  });
});
$('#btn-traffic-copy').addEventListener('click', async function () {
  var e = trafficState.activeIdx >= 0 ? trafficState.entries[trafficState.activeIdx] : null;
  if (!e) return;
  var text = [
    '=== REQUEST ===', e.method + ' ' + e.url,
    'Headers: ' + JSON.stringify(e.reqHeaders || {}, null, 2), '',
    e.req || '(no body)', '',
    '=== RESPONSE ===',
    'Status: ' + e.status + ' ' + statusTextCN(e.status || 0) + '  (' + (e.durationMs || '—') + ' ms, ' + (e.respSize || '—') + ' B)',
    'Headers: ' + JSON.stringify(e.respHeaders || {}, null, 2), '',
    e.resp || e.error || '(no body)',
  ].join('\n');
  try {
    await navigator.clipboard.writeText(text);
    var btn = $('#btn-traffic-copy');
    var old = btn.textContent; btn.textContent = '已复制 ✓';
    setTimeout(function () { btn.textContent = old; }, 1200);
  } catch (_e) { /* clipboard blocked */ }
});
$('#btn-traffic-clear').addEventListener('click', function () {
  trafficState.entries = []; trafficState.activeIdx = -1;
  setTrafficSummary();
  renderTrafficPanes();
  $$('.traffic-tab').forEach(function (x) { x.classList.toggle('active', x.dataset.traffic === 'request'); });
  $('#traffic-request').hidden = false; $('#traffic-response').hidden = true;
});
$('#btn-traffic-toggle').addEventListener('click', function () {
  var t = $('#traffic');
  t.classList.toggle('collapsed');
  this.textContent = t.classList.contains('collapsed') ? '展开' : '折叠';
});

/* ============ Health probes ============ */
var checking = false;
async function pollHealth() {
  if (checking) return;
  checking = true;
  var setLed = function (id, state) { $('#' + id).className = 'led ' + (state === 'ok' ? 'ok' : state === 'warn' ? 'warn' : 'err'); };
  $('#led-healthz').className = 'led checking';
  $('#led-readyz').className  = 'led checking';
  try {
    var h = await fetch('/healthz').then(function (r) { return r.json(); }).catch(function () { return null; });
    setLed('led-healthz', h && h.status === 'ok' ? 'ok' : 'err');
    var r = await fetch('/readyz').then(function (res) { return { ok: res.ok, status: res.status }; })
                                   .catch(function () { return { ok: false }; });
    setLed('led-readyz', r.ok ? 'ok' : 'warn');
  } finally {
    checking = false;
  }
}
pollHealth();
setInterval(pollHealth, 5000);

/* ============ Uptime ticker ============ */
setInterval(function () {
  var el = $('#sb-uptime');
  if (!el) return;
  var ms = Date.now() - appStart;
  var m = Math.floor(ms / 60000);
  var h = Math.floor(m / 60);
  el.textContent = h + 'h ' + (m % 60) + 'm';
}, 30000);

/* ============ Sidebar nav ============ */
function activateView(view) {
  navState.view = view;
  $$('.nav-item').forEach(function (n) { n.classList.toggle('active', n.dataset.view === view); });
  $$('.panel').forEach(function (p) { p.classList.toggle('active', p.id === 'panel-' + view); });
  var labels = {
    'overview': { cat: '概览', sub: '总览首页' },
    'models': { cat: '模型', sub: '模型列表' },
    'embeddings': { cat: '模型', sub: '文本嵌入' },
    'image-embeddings': { cat: '模型', sub: '图像嵌入' },
    'multimodal-embeddings': { cat: '模型', sub: '图文嵌入' },
    'rerank': { cat: '模型', sub: '重排' },
    'text-similarity': { cat: '模型', sub: '文本相似度' },
    'image-similarity': { cat: '模型', sub: '图像相似度' },
    'mm-similarity': { cat: '模型', sub: '图文相似度' },
    'databases': { cat: '向量库', sub: '数据库' },
    'collections': { cat: '向量库', sub: '集合' },
    'vectors': { cat: '向量库', sub: '向量' },
    'search': { cat: '向量库', sub: '检索' },
  };
  var l = labels[view] || { cat: '模型', sub: '模型列表' };
  $('#crumb-cat').textContent = l.cat;
  var subEl = $('#crumb-sub');
  var sepEl = $('#crumb-sub-sep');
  if (l.sub) { subEl.textContent = l.sub; subEl.hidden = false; sepEl.hidden = false; }
  else { subEl.textContent = ''; subEl.hidden = true; sepEl.hidden = true; }

  /* per-view data refresh */
  if (view === 'overview') { startOverviewPolling(); }
  else { stopOverviewPolling(); }
  if (view === 'models') { refreshModels(); }
  else if (view === 'embeddings') { refreshModels(); }
  else if (view === 'image-embeddings') { refreshImageModels(); }
  else if (view === 'multimodal-embeddings') { refreshMultimodalModels(); }
  else if (view === 'rerank') { refreshRerankModels(); }
  else if (view === 'text-similarity') { refreshTextSimModels(); }
  else if (view === 'image-similarity') { refreshImageSimModels(); }
  else if (view === 'mm-similarity') { refreshMmSimModels(); }
  else if (view === 'databases') { refreshDatabases(); }
  else if (view === 'collections') { refreshDatabases().then(refreshCollectionsInActive); }
  else if (view === 'vectors') { refreshDatabases().then(refreshCollectionsInActive); }
  else if (view === 'search') { refreshDatabases().then(refreshCollectionsInActive); }
}
$$('.nav-item').forEach(function (n) {
  n.addEventListener('click', function () { activateView(n.dataset.view); });
});
$$('.nav-group-label').forEach(function (g) {
  g.addEventListener('click', function () {
    g.parentElement.classList.toggle('collapsed');
  });
});
$('#btn-toggle-log').addEventListener('click', function () {
  var app = $('#app');
  app.classList.toggle('app--log-hidden');
  this.classList.toggle('active', app.classList.contains('app--log-hidden'));
});

/* ============ Models registry: cards + load/unload ============
   Rendering and actions are keyed by ``model.id``, not by family:
   - each card corresponds to one registered model;
   - load/unload buttons target that exact id via
     ``POST /v1/models/{id}/{load|unload}``;
   - the busy flag is keyed by id too, so only the affected card
     disables while a request is in flight;
   - the bottom status bar mirrors "已加载 / 已注册" counts so the
     operator can see registry size without leaving the panel. */
var FAMILY_LABELS = {
  embedder:            'embedder',
  image_embedder:      'image_embedder',
  multimodal_embedder: 'multimodal_embedder',
  reranker:            'reranker',
};
function familyLabel(type) { return FAMILY_LABELS[type] || type || 'unknown'; }

function currentLoadedOfFamily(models, type) {
  return models.find(function (m) { return m.type === type && m.loaded; }) || null;
}

function renderModelCard(m) {
  /* ``m.loaded`` is the canonical signal — for embedders / image /
     multimodal embedders ``dimensions`` happens to also imply "loaded",
     but rerankers' ``dimensions`` is permanently ``null`` so we must
     look at ``m.loaded`` to drive the 已加载 / 卸载 toggle. */
  var isLoaded = !!m.loaded;
  var busy = !!modelsState.busy[m.id];
  var cardClass = 'model-card'
    + (isLoaded ? ' loaded' : '')
    + (busy ? ' busy' : '');
  var statusText = isLoaded ? '已加载' : '未加载';
  var currentHtml = isLoaded
    ? '<div class="model-current">' +
        '<span class="dim">' + m.dimensions + ' 维</span>' +
      '</div>'
    : '<div class="model-empty">未加载 — 当前实例为 None</div>';
  var actionsHtml;
  if (isLoaded) {
    actionsHtml =
      '<button class="btn sm danger lc-card-unload" data-id="' + escapeHtml(m.id) + '">卸载</button>';
  } else {
    /* Loading a different id within the same family requires the
       currently-loaded sibling (if any) to be unloaded first; the
       server enforces this with 409 conflict_loaded. We surface that
       contract here as a tooltip so the operator isn't surprised. */
    var sibling = currentLoadedOfFamily(modelsState.data, m.type);
    var tip = sibling
      ? 'title="同家族已加载 ' + escapeHtml(sibling.id) + '，需先卸载"'
      : '';
    actionsHtml =
      '<button class="btn sm primary lc-card-load" data-id="' + escapeHtml(m.id) + '" ' + tip + '>加载</button>';
  }
  /* mini dim-dots visualisation */
  var dots = '';
  var totalDots = Math.min(isLoaded ? m.dimensions : 16, 32);
  var accentDots = isLoaded ? Math.min(8, totalDots) : 0;
  for (var i = 0; i < totalDots; i++) {
    dots += '<span class="mini-dot' + (i < accentDots ? ' accent' : '') + '"></span>';
  }
  return '<div class="' + cardClass + '" data-id="' + escapeHtml(m.id) + '" data-family="' + escapeHtml(m.type) + '" data-scope="models">' +
    '<div class="model-card-head">' +
      '<div class="model-title">' +
        '<span class="id" title="' + escapeHtml(m.id) + '">' + escapeHtml(m.id) + '</span>' +
        '<span class="family-tag">' + escapeHtml(familyLabel(m.type)) + '</span>' +
      '</div>' +
      '<span class="status-pill' + (isLoaded ? ' loaded' : '') + '">' +
        '<span class="dot"></span>' + statusText +
      '</span>' +
    '</div>' +
    '<div class="model-current">' + currentHtml + '</div>' +
    '<div class="model-dots">' + dots + '</div>' +
    '<div class="model-actions">' + actionsHtml + '</div>' +
  '</div>';
}

/* Minimal CSS.escape polyfill: ``data-id`` may contain dots that
   break querySelector. We only need to escape the few characters that
   appear in model ids (e.g. ``openai/clip-vit-l-14``). */
function cssEscape(s) {
  if (window.CSS && CSS.escape) return CSS.escape(s);
  return String(s).replace(/([!"#$%&'()*+,.\/:;<=>?@\[\\\]^`{|}~])/g, '\\$1');
}

function bindModelCardActions() {
  var sel = '#models-grid';
  $$('.lc-card-load', $(sel)).forEach(function (b) {
    b.addEventListener('click', function () { lifecycleAction('load', b.dataset.id); });
  });
  $$('.lc-card-unload', $(sel)).forEach(function (b) {
    b.addEventListener('click', function () { lifecycleAction('unload', b.dataset.id); });
  });
}

async function lifecycleAction(verb, id) {
  modelsState.busy[id] = true;
  /* Re-render the affected card so the user sees the busy state
     immediately, instead of waiting for the round-trip to land. */
  var card = document.querySelector('[data-scope="models"][data-id="' + cssEscape(id) + '"]');
  if (card) card.classList.add('busy');
  try {
    await api('POST', '/v1/models/' + enc(id) + '/' + verb);
  } catch (_e) { /* error in traffic panel */ }
  delete modelsState.busy[id];
  await refreshModels();
  refreshImageModels(); refreshMultimodalModels(); refreshRerankModels();
}

/* Render the registry into #models-grid + push loaded/total into the
   bottom status bar so the operator can see registry size without a
   dedicated KPI panel. */
function renderRegistryGrid(data) {
  var grid = $('#models-grid');
  if (!grid) return;
  grid.innerHTML = (data || []).map(renderModelCard).join('');
  bindModelCardActions();
  /* Use ``m.loaded`` so the bottom-bar count includes loaded rerankers
     (whose ``dimensions`` is permanently ``null``). */
  var loaded = (data || []).filter(function (m) { return m.loaded; }).length;
  var total = (data || []).length;
  var sb = $('#sb-loaded');
  if (sb) sb.textContent = loaded + '/' + total;
}

function setModelsAutoRefresh(on) {
  modelsState.auto = !!on;
  var btn = document.getElementById('btn-models-auto-refresh');
  if (btn) btn.textContent = '自动刷新：' + (modelsState.auto ? '开' : '关');
  if (modelsState.timer) { clearInterval(modelsState.timer); modelsState.timer = null; }
  if (modelsState.auto) {
    modelsState.timer = setInterval(function () {
      if (navState.view === 'models') refreshModels();
    }, 5000);
  }
}
$('#btn-models-auto-refresh').addEventListener('click', function () {
  setModelsAutoRefresh(!modelsState.auto);
});
setModelsAutoRefresh(true);

/* ============ Databases ============ */
async function refreshDatabases() {
  var dbs = [];
  try {
    var r = await api('GET', '/v1/databases');
    dbs = (r.payload && r.payload.databases) || [];
  } catch (_e) { /* leave empty */ }
  var list = $('#dbs-list');
  if (list) {
    list.innerHTML = '';
    if (!dbs.length) {
      list.innerHTML = '<div class="empty">暂无数据库。在下方表单创建第一个。</div>';
    } else {
      dbs.forEach(function (name) {
        var item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML =
          '<span class="name">' + escapeHtml(name) + '</span>' +
          '<button class="btn sm danger">删除</button>';
        item.querySelector('button').addEventListener('click', async function (e) {
          e.stopPropagation();
          if (!confirm('确认删除数据库 "' + name + '" 及其下所有集合？')) return;
          try { await api('DELETE', '/v1/databases/' + enc(name)); } catch (_e) {}
          refreshDatabases();
        });
        list.appendChild(item);
      });
    }
  }
  fillDbSelects(dbs);
  return dbs;
}
function fillDbSelects(dbs) {
  ['#colls-db', '#vec-db', '#srch-db'].forEach(function (sel) {
    var el = $(sel); if (!el) return;
    var prev = el.value;
    el.innerHTML = '';
    if (!dbs.length) {
      var o = document.createElement('option'); o.value = ''; o.textContent = '（暂无数据库）'; el.appendChild(o);
      return;
    }
    dbs.forEach(function (d) {
      var o = document.createElement('option'); o.value = d; o.textContent = d; el.appendChild(o);
    });
    if (prev && dbs.indexOf(prev) >= 0) el.value = prev;
  });
}
$('#btn-refresh-dbs').addEventListener('click', refreshDatabases);
$('#btn-colls-refresh-db').addEventListener('click', refreshDatabases);
$('#btn-create-db').addEventListener('click', async function () {
  var name = $('#new-db-name').value.trim();
  if (!name) { alert('请填写数据库名。'); return; }
  try {
    await api('POST', '/v1/databases', { name: name });
    $('#new-db-name').value = '';
    refreshDatabases();
  } catch (_e) {}
});

/* ============ Models / Embeddings ============ */
async function refreshModels() {
  var data = null;
  try {
    var r = await api('GET', '/v1/models');
    data = (r && r.payload && r.payload.data) || [];
  } catch (_e) { data = []; }
  /* Mirror latest model list into modelsState so card actions can read
     sibling state (e.g. the "同家族已加载 X" tip on the load button). */
  modelsState.data = data || [];
  /* Topbar dim-dots are derived from the live embedder dimensions. */
  var dim = currentEmbedderDim(data || []);
  renderDimDots(dim);
  /* Card grid (the only model list visible to the operator). */
  renderRegistryGrid(data);
  /* Embedding-model <select> still needs to be kept in sync for the
     "文本嵌入" panel. */
  var sel = $('#emb-model');
  if (sel) {
    var prev = sel.value;
    sel.innerHTML = '';
    (data || []).filter(function (m) { return m.type === 'embedder'; }).forEach(function (m) {
      var o = document.createElement('option'); o.value = m.id; o.textContent = m.id; sel.appendChild(o);
    });
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
  }
  return data || [];
}
$('#btn-refresh-models').addEventListener('click', refreshModels);
$('#btn-embed').addEventListener('click', async function () {
  var model = $('#emb-model').value;
  var mode  = $('#emb-mode').value;
  var inputRaw = $('#emb-input').value;
  var input;
  if (mode === 'list') {
    input = safeParse(inputRaw);
    if (!Array.isArray(input)) { alert('列表模式下，输入必须是 JSON 数组。'); return; }
  } else {
    input = inputRaw;
  }
  try { await api('POST', '/v1/embeddings', { model: model, input: input }); } catch (_e) {}
});

/* ============ Collections ============ */
function activeDb() {
  var v = $('#colls-db').value;
  if (!v) { alert('请先选择数据库。'); return null; }
  return v;
}
async function refreshCollectionsInActive() {
  var db = $('#colls-db').value;
  if (!db) {
    var list = $('#colls-list');
    if (list) list.innerHTML = '<div class="empty">选择数据库后点击"刷新"。</div>';
    return [];
  }
  var colls = [];
  try {
    var r = await api('GET', '/v1/databases/' + enc(db) + '/collections');
    colls = (r.payload && r.payload.collections) || [];
  } catch (_e) { colls = []; }
  var list = $('#colls-list');
  if (list) {
    list.innerHTML = '';
    if (!colls.length) {
      list.innerHTML = '<div class="empty">数据库 <code>' + escapeHtml(db) + '</code> 下暂无集合。</div>';
    } else {
      colls.forEach(function (name) {
        var item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML =
          '<span class="name">' + escapeHtml(db) + ' / ' + escapeHtml(name) + '</span>' +
          '<button class="btn sm danger">删除</button>';
        item.querySelector('button').addEventListener('click', async function (e) {
          e.stopPropagation();
          if (!confirm('确认删除集合 "' + db + '/' + name + '"？')) return;
          try { await api('DELETE', '/v1/databases/' + enc(db) + '/collections/' + enc(name)); } catch (_e) {}
          refreshCollectionsInActive();
        });
        list.appendChild(item);
      });
    }
  }
  var map = {}; map[$('#vec-db').value || db] = colls;
  fillCollSelects(map);
  rebuildSearchFieldTokens(db, colls);
  return colls;
}
function fillCollSelects(collsByDb) {
  var db = $('#vec-db').value;
  var colls = (collsByDb && collsByDb[db]) || [];
  ['#vec-coll', '#srch-coll'].forEach(function (sel) {
    var el = $(sel); if (!el) return;
    var prev = el.value;
    el.innerHTML = '';
    if (!colls.length) {
      var o = document.createElement('option'); o.value = ''; o.textContent = '（暂无集合）'; el.appendChild(o);
      return;
    }
    colls.forEach(function (c) {
      var o = document.createElement('option'); o.value = c; o.textContent = c; el.appendChild(o);
    });
    if (prev && colls.indexOf(prev) >= 0) el.value = prev;
  });
}
$('#colls-db').addEventListener('change', refreshCollectionsInActive);
$('#vec-db').addEventListener('change', refreshCollectionsInActive);
$('#btn-refresh-colls').addEventListener('click', refreshCollectionsInActive);

var searchFieldTokens = { outputs: new Set(), schemaCache: {} };

function rebuildSearchFieldTokens(dbName, colls) {
  searchFieldTokens.schemaCache = {};
  renderOutputTokens();
  renderFilterFieldTokens([]);
}

function renderOutputTokens() {
  var root = $('#srch-output-tokens');
  if (!root) return;
  var fields = collectCurrentSchemaFields();
  if (!fields.length) {
    root.innerHTML = '<div style="color:var(--text-muted);font-size:11px;">尚无字段。点下方"加载字段"按钮可拉取当前集合的 schema。</div>';
    return;
  }
  root.innerHTML = fields.map(function (f) {
    var active = searchFieldTokens.outputs.has(f) ? 'active' : '';
    return '<span class="tag-token ' + active + '" data-field="' + escapeHtml(f) + '">' + escapeHtml(f) + '</span>';
  }).join('');
  root.querySelectorAll('.tag-token').forEach(function (el) {
    el.addEventListener('click', function () {
      var f = el.dataset.field;
      if (searchFieldTokens.outputs.has(f)) searchFieldTokens.outputs.delete(f);
      else searchFieldTokens.outputs.add(f);
      renderOutputTokens();
    });
  });
}

function renderFilterFieldTokens(fields) {
  var root = $('#srch-filter-fields');
  if (!root) return;
  if (!fields.length) { root.innerHTML = ''; return; }
  root.innerHTML = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">点击字段名可插入到过滤表达式：</div>' +
    fields.map(function (f) { return '<span class="tag-token" data-insert="' + escapeHtml(f) + '">' + escapeHtml(f) + '</span>'; }).join('');
  root.querySelectorAll('.tag-token').forEach(function (el) {
    el.addEventListener('click', function () {
      var f = el.dataset.insert;
      var ta = $('#srch-filter');
      ta.value = (ta.value.trim() ? ta.value.trim() + ' and ' : '') + f + ' == ';
      ta.focus();
    });
  });
}

function collectCurrentSchemaFields() {
  var coll = $('#srch-coll').value;
  var db   = $('#srch-db').value;
  if (!db || !coll) return [];
  var k = db + '::' + coll;
  return searchFieldTokens.schemaCache[k] || [];
}

/* ============ Collection creation form ============ */
var SCALAR_DTYPES = ['bool','int8','int16','int32','int64','float','double','varchar','json'];
var METRICS = ['cosine','ip','l2'];
var INDEX_TYPES = ['HNSW','IVF_FLAT','IVF_SQ8','IVF_PQ','DISKANN','FLAT','ANNOY','AUTOINDEX'];

function defaultScalarRow() {
  return { name: '', dtype: 'varchar', is_primary: false, max_length: 64, nullable: false, default_value: '' };
}
function defaultIndexRow() {
  return { field_name: 'vector', metric_type: 'cosine', index_type: 'HNSW', params: { M: 16, efConstruction: 200 } };
}

function renderScalarCard(row, idx) {
  var card = document.createElement('div');
  card.className = 'field-card';
  var isVarchar = row.dtype === 'varchar';
  if (!isVarchar) card.classList.add('maxlen-hidden');
  card.innerHTML =
    '<div class="field-card-header">' +
      '<span class="index-badge">字段 #' + (idx + 1) + '</span>' +
      '<span class="title">' + (row.name || '（未命名）') + (row.is_primary ? ' · 主键' : '') + '</span>' +
      '<button class="remove" type="button" title="删除">×</button>' +
    '</div>' +
    '<div class="grid cols-3">' +
      '<div class="field"><label>字段名</label><input data-k="name" type="text" value="' + escapeHtml(row.name) + '" placeholder="id / category / price" /></div>' +
      '<div class="field"><label>类型</label>' +
        '<select data-k="dtype">' + SCALAR_DTYPES.map(function (t) { return '<option value="' + t + '"' + (t === row.dtype ? ' selected' : '') + '>' + t + '</option>'; }).join('') + '</select>' +
      '</div>' +
      '<div class="field maxlen-cell"><label>max_length（varchar 必填）</label><input data-k="max_length" type="number" min="1" max="65535" value="' + (row.max_length || 64) + '" /></div>' +
      '<div class="field"><label class="check-row"><input data-k="is_primary" type="checkbox"' + (row.is_primary ? ' checked' : '') + ' /> 作为主键</label></div>' +
      '<div class="field"><label class="check-row"><input data-k="nullable" type="checkbox"' + (row.nullable ? ' checked' : '') + ' /> 允许空值</label></div>' +
    '</div>';
  bindFieldCard(card, row);
  card.querySelector('.remove').addEventListener('click', function () { card.remove(); });
  return card;
}

function renderIndexCard(row, idx) {
  var card = document.createElement('div');
  card.className = 'field-card';
  card.innerHTML =
    '<div class="field-card-header">' +
      '<span class="index-badge">索引 #' + (idx + 1) + '</span>' +
      '<span class="title">' + row.index_type + ' · ' + row.metric_type + ' → ' + row.field_name + '</span>' +
      '<button class="remove" type="button" title="删除">×</button>' +
    '</div>' +
    '<div class="grid cols-3">' +
      '<div class="field"><label>字段名</label><input data-k="field_name" type="text" value="' + escapeHtml(row.field_name) + '" /></div>' +
      '<div class="field"><label>度量</label>' +
        '<select data-k="metric_type">' + METRICS.map(function (t) { return '<option value="' + t + '"' + (t === row.metric_type ? ' selected' : '') + '>' + t + '</option>'; }).join('') + '</select>' +
      '</div>' +
      '<div class="field"><label>索引类型</label>' +
        '<select data-k="index_type">' + INDEX_TYPES.map(function (t) { return '<option value="' + t + '"' + (t === row.index_type ? ' selected' : '') + '>' + t + '</option>'; }).join('') + '</select>' +
      '</div>' +
    '</div>' +
    '<div class="grid" style="margin-top:8px;">' +
      '<div class="field params-cell"><label>参数（JSON 对象）</label>' +
        '<textarea data-k="params" rows="2">' + escapeHtml(JSON.stringify(row.params || {})) + '</textarea>' +
      '</div>' +
    '</div>';
  bindFieldCard(card, row);
  card.querySelector('.remove').addEventListener('click', function () { card.remove(); });
  return card;
}

function bindFieldCard(card, row) {
  card.querySelectorAll('[data-k]').forEach(function (el) {
    el.addEventListener('input', function () {
      var k = el.dataset.k;
      if (el.type === 'checkbox') row[k] = el.checked;
      else if (el.type === 'number') row[k] = el.value === '' ? null : Number(el.value);
      else row[k] = el.value;
      if (k === 'dtype') {
        card.classList.toggle('maxlen-hidden', el.value !== 'varchar');
      }
      var title = card.querySelector('.title');
      if (title) {
        if ('field_name' in row) {
          title.textContent = row.index_type + ' · ' + row.metric_type + ' → ' + row.field_name;
        } else {
          title.textContent = (row.name || '（未命名）') + (row.is_primary ? ' · 主键' : '');
        }
      }
    });
  });
}

var scalarRows = [defaultScalarRow()];
scalarRows[0].name = 'id';
scalarRows[0].is_primary = true;
var indexRows = [defaultIndexRow()];

function renderScalars() {
  var root = $('#scalars-list');
  root.innerHTML = '';
  scalarRows.forEach(function (r, i) { root.appendChild(renderScalarCard(r, i)); });
}
function renderIndices() {
  var root = $('#index-list');
  root.innerHTML = '';
  indexRows.forEach(function (r, i) { root.appendChild(renderIndexCard(r, i)); });
}
function renderVector() {
  var root = $('#vector-card');
  root.innerHTML =
    '<div class="field-card">' +
      '<div class="field-card-header">' +
        '<span class="index-badge">向量字段</span>' +
        '<span class="title">唯一 · 不可改</span>' +
      '</div>' +
      '<div class="grid cols-3">' +
        '<div class="field"><label>字段名</label><input id="vec-field-name" type="text" value="vector" /></div>' +
        '<div class="field"><label>维度</label><input id="vec-field-dim" type="number" min="1" max="32768" value="1024" /></div>' +
        '<div class="field"><label>度量</label><select id="vec-field-metric">' +
          METRICS.map(function (t) { return '<option value="' + t + '"' + (t === 'cosine' ? ' selected' : '') + '>' + t + '</option>'; }).join('') +
        '</select></div>' +
      '</div>' +
    '</div>';
}
renderScalars(); renderIndices(); renderVector();

$('#btn-add-scalar').addEventListener('click', function () { scalarRows.push(defaultScalarRow()); renderScalars(); });
$('#btn-add-index').addEventListener('click', function () { indexRows.push(defaultIndexRow()); renderIndices(); });
$$('[data-preset-scalar]').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var which = btn.getAttribute('data-preset-scalar');
    if (which === 'id+category+price') {
      scalarRows = [
        { name: 'id', dtype: 'varchar', is_primary: true, max_length: 64, nullable: false, default_value: '' },
        { name: 'category', dtype: 'varchar', is_primary: false, max_length: 32, nullable: false, default_value: '' },
        { name: 'price', dtype: 'float', is_primary: false, max_length: null, nullable: false, default_value: '' },
      ];
    } else if (which === 'id+year') {
      scalarRows = [
        { name: 'id', dtype: 'varchar', is_primary: true, max_length: 64, nullable: false, default_value: '' },
        { name: 'year', dtype: 'int32', is_primary: false, max_length: null, nullable: false, default_value: '' },
      ];
    }
    renderScalars();
  });
});
$$('[data-preset-index]').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var which = btn.getAttribute('data-preset-index');
    if (which === 'hnsw-cosine') {
      indexRows = [{ field_name: 'vector', metric_type: 'cosine', index_type: 'HNSW', params: { M: 16, efConstruction: 200 } }];
    } else if (which === 'ivf-l2') {
      indexRows = [{ field_name: 'vector', metric_type: 'l2', index_type: 'IVF_FLAT', params: { nlist: 64 } }];
    } else if (which === 'diskann-ip') {
      indexRows = [{ field_name: 'vector', metric_type: 'ip', index_type: 'DISKANN', params: {} }];
    } else if (which === 'default') {
      indexRows = [];
    }
    renderIndices();
  });
});

function collectScalarPayload() {
  var primary = scalarRows.filter(function (r) { return r.is_primary; });
  if (primary.length !== 1) {
    alert('必须且只能有一个主键字段（勾选 is_primary）。当前：' + primary.length);
    return null;
  }
  if (primary[0].dtype !== 'varchar') {
    alert('主键必须是 varchar 类型。当前：' + primary[0].dtype);
    return null;
  }
  if (!primary[0].name) {
    alert('主键字段名不能为空。');
    return null;
  }
  for (var i = 0; i < scalarRows.length; i++) {
    var r = scalarRows[i];
    if (r.dtype === 'varchar' && (!r.max_length || r.max_length < 1)) {
      alert('varchar 字段 ' + (r.name || '未命名') + ' 必须填写 max_length。');
      return null;
    }
    if (!r.name) { alert('每个字段都必须填写名称。'); return null; }
  }
  return scalarRows.map(function (r) {
    var out = { name: r.name, dtype: r.dtype, is_primary: !!r.is_primary };
    if (r.dtype === 'varchar') out.max_length = r.max_length;
    if (r.nullable) out.nullable = true;
    if (r.default_value !== '' && r.default_value !== null && r.default_value !== undefined) {
      out.default_value = r.default_value;
    }
    return out;
  });
}
function collectIndexPayload() {
  if (!indexRows.length) return [];
  return indexRows.map(function (r) {
    return {
      field_name: r.field_name,
      metric_type: r.metric_type,
      index_type: r.index_type,
      params: typeof r.params === 'string' ? (safeParse(r.params) || {}) : (r.params || {}),
    };
  });
}
$('#btn-create-coll').addEventListener('click', async function () {
  var db = activeDb(); if (!db) return;
  var name = $('#new-coll-name').value.trim();
  if (!name) { alert('请填写集合名。'); return; }
  var primary = $('#new-coll-primary').value.trim();
  if (!primary) { alert('请填写主键字段名。'); return; }
  var scalars = collectScalarPayload(); if (!scalars) return;
  if (scalars.find(function (s) { return s.is_primary; }).name !== primary) {
    alert('主键字段名 (' + primary + ') 与勾选了 is_primary 的字段名不一致。');
    return;
  }
  var vecName = $('#vec-field-name').value.trim() || 'vector';
  var vecDim  = Number($('#vec-field-dim').value);
  var vecMet  = $('#vec-field-metric').value;
  if (!Number.isFinite(vecDim) || vecDim < 1) { alert('向量维度必须 ≥ 1。'); return; }
  var body = {
    name: name,
    primary_field: primary,
    scalar_fields: scalars,
    vector_field: { name: vecName, dim: vecDim, metric_type: vecMet },
    index_params: collectIndexPayload(),
  };
  try {
    await api('POST', '/v1/databases/' + enc(db) + '/collections', body);
    $('#new-coll-name').value = '';
    refreshCollectionsInActive();
  } catch (_e) {}
});

/* ============ Vectors (upsert/fetch/delete) ============ */
$('#vec-mode').addEventListener('change', function () {
  var mode = $('#vec-mode').value;
  $('#vec-texts-row').classList.toggle('hidden', mode !== 'texts');
  $('#vec-embs-row').classList.toggle('hidden', mode !== 'vectors');
});
function buildVecBody() {
  var ids = safeParse($('#vec-ids').value);
  if (!Array.isArray(ids)) { alert('ids 必须是 JSON 数组。'); return null; }
  var body = {
    primary_field: $('#vec-primary').value.trim(),
    vector_field: $('#vec-vecfield').value.trim(),
    ids: ids,
  };
  if ($('#vec-mode').value === 'texts') {
    var texts = safeParse($('#vec-texts').value);
    if (!Array.isArray(texts)) { alert('texts 必须是 JSON 数组。'); return null; }
    body.texts = texts;
  } else {
    var embeddings = safeParse($('#vec-embs').value);
    if (!Array.isArray(embeddings)) { alert('vectors 必须是 JSON 数组。'); return null; }
    body.vectors = embeddings;
  }
  var fieldsRaw = $('#vec-fields').value.trim();
  if (fieldsRaw && fieldsRaw !== '[]') {
    var fields = safeParse(fieldsRaw);
    if (!Array.isArray(fields)) { alert('fields 必须是 JSON 数组。'); return null; }
    if (fields.length) body.fields = fields;
  }
  return body;
}
$('#btn-upsert').addEventListener('click', async function () {
  var db = $('#vec-db').value; if (!db) { alert('请选择数据库。'); return; }
  var name = $('#vec-coll').value; if (!name) { alert('请选择集合。'); return; }
  var body = buildVecBody(); if (!body) return;
  try { await api('PUT', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/vectors', body); } catch (_e) {}
});
$('#btn-fetch').addEventListener('click', async function () {
  var db = $('#vec-db').value; if (!db) { alert('请选择数据库。'); return; }
  var name = $('#vec-coll').value; if (!name) { alert('请选择集合。'); return; }
  var ids = safeParse($('#vec-ids').value);
  if (!Array.isArray(ids)) { alert('ids 必须是 JSON 数组。'); return; }
  try {
    await api('POST', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/vectors/get', {
      primary_field: $('#vec-primary').value.trim(), ids: ids,
    });
  } catch (_e) {}
});
$('#btn-delete').addEventListener('click', async function () {
  var db = $('#vec-db').value; if (!db) { alert('请选择数据库。'); return; }
  var name = $('#vec-coll').value; if (!name) { alert('请选择集合。'); return; }
  var ids = safeParse($('#vec-ids').value);
  if (!Array.isArray(ids)) { alert('ids 必须是 JSON 数组。'); return; }
  if (!confirm('确认删除 ' + ids.length + ' 条记录？')) return;
  try {
    await api('POST', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/vectors/delete', {
      primary_field: $('#vec-primary').value.trim(), ids: ids,
    });
  } catch (_e) {}
});

/* ============ Search ============ */
$('#srch-mode').addEventListener('change', function () {
  var mode = $('#srch-mode').value;
  $('#srch-text-row').classList.toggle('hidden', mode !== 'text');
  $('#srch-emb-row').classList.toggle('hidden', mode !== 'emb');
});
$('#srch-coll').addEventListener('change', async function () {
  var db = $('#srch-db').value;
  var coll = $('#srch-coll').value;
  if (!db || !coll) return;
  var k = db + '::' + coll;
  if (searchFieldTokens.schemaCache[k]) return;
  try {
    var r = await api('GET', '/v1/databases/' + enc(db) + '/collections/' + enc(coll));
    var fields = (r.payload && r.payload.fields) || [];
    searchFieldTokens.schemaCache[k] = fields.map(function (f) { return f.name; });
  } catch (_e) {
    searchFieldTokens.schemaCache[k] = [];
  }
  renderOutputTokens();
  renderFilterFieldTokens(collectCurrentSchemaFields());
});
$('#btn-search').addEventListener('click', async function () {
  var db = $('#srch-db').value; if (!db) { alert('请选择数据库。'); return; }
  var name = $('#srch-coll').value; if (!name) { alert('请选择集合。'); return; }
  var topk = parseInt($('#srch-topk').value, 10);
  var body = {
    primary_field: $('#srch-primary').value.trim(),
    vector_field: $('#srch-vecfield').value.trim(),
    top_k: isNaN(topk) ? 10 : topk,
  };
  if ($('#srch-mode').value === 'text') {
    body.query_text = $('#srch-text').value;
  } else {
    var emb = safeParse($('#srch-emb').value);
    if (!Array.isArray(emb)) { alert('query_vector 必须是 JSON 数组。'); return; }
    body.query_vector = emb;
  }
  var filterExpr = $('#srch-filter').value.trim();
  if (filterExpr) body.filter_expr = filterExpr;
  if (searchFieldTokens.outputs.size) {
    body.output_fields = Array.from(searchFieldTokens.outputs);
  }
  try { await api('POST', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/search', body); } catch (_e) {}
});

/* ============ Rerank ============ */
async function refreshRerankModels() {
  var data = [];
  try {
    var r = await api('GET', '/v1/models');
    var all = (r && r.payload && r.payload.data) || [];
    data = all.filter(function (m) { return m.type === 'reranker'; });
  } catch (_e) { data = []; }
  var list = $('#rerank-models-list');
  if (list) {
    list.innerHTML = '';
    if (!data.length) {
      list.innerHTML = '<div class="empty">暂无已注册的 reranker。</div>';
    } else {
      data.forEach(function (m) {
        var item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML =
          '<span class="name">' + escapeHtml(m.id) + '</span>' +
          '<span class="meta">' + (m.loaded ? '已加载 · reranker' : 'reranker') + '</span>';
        item.addEventListener('click', async function () {
          try { await api('GET', '/v1/models/' + enc(m.id)); } catch (_e) {}
        });
        list.appendChild(item);
      });
    }
  }
  var sel = $('#rerank-model');
  if (sel) {
    var prev = sel.value;
    sel.innerHTML = '';
    var o0 = document.createElement('option');
    o0.value = ''; o0.textContent = '（使用服务端默认）';
    sel.appendChild(o0);
    data.forEach(function (m) {
      var opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id;
      sel.appendChild(opt);
    });
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
  }
}

function buildDocumentsPayload() {
  var mode = $('#rerank-mode').value;
  var raw = $('#rerank-docs').value;
  if (mode === 'json') {
    var parsed = safeParse(raw);
    if (!Array.isArray(parsed)) { alert('JSON 模式下，文档必须是 JSON 数组。'); return null; }
    if (!parsed.every(function (x) { return typeof x === 'string'; })) { alert('文档数组的每个元素都必须是字符串。'); return null; }
    return parsed;
  }
  return raw.split(/[\r\n]+/).map(function (s) { return s.trim(); }).filter(function (s) { return s.length > 0; });
}

function renderRerankResults(payload, docs) {
  var root = $('#rerank-results');
  if (!root) return;
  if (!payload || !Array.isArray(payload.results)) {
    root.innerHTML = '<div class="empty">响应中未包含 results 数组。</div>';
    return;
  }
  var head =
    '<dl class="traffic-section" style="margin-bottom:12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>request_id</dt><dd>' + escapeHtml(payload.request_id || '') + '</dd>' +
      '<dt>命中数</dt><dd>' + payload.results.length + '</dd>' +
    '</dl>';
  if (!payload.results.length) {
    root.innerHTML = head + '<div class="empty">无结果。</div>';
    return;
  }
  var rows = payload.results.map(function (r, i) {
    var doc = (typeof r.index === 'number' && docs[r.index]) || '(原文未提供)';
    var truncated = doc.length > 120 ? doc.slice(0, 120) + '…' : doc;
    var score = (typeof r.score === 'number') ? r.score.toFixed(4) : String(r.score);
    return '<div class="result-row">' +
      '<span class="result-rank">#' + (i + 1) + '</span>' +
      '<span class="result-idx">idx=' + r.index + '</span>' +
      '<span class="result-score">' + score + '</span>' +
      '<span class="result-doc" title="' + escapeHtml(doc) + '">' + escapeHtml(truncated) + '</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

$('#rerank-mode').addEventListener('change', function () {
  var mode = $('#rerank-mode').value;
  $('#rerank-docs').placeholder = mode === 'json'
    ? '["无线鼠标", "机械键盘", "蓝牙耳机", "游戏手柄"]'
    : '无线鼠标\n机械键盘\n蓝牙耳机\n游戏手柄';
});
$('#btn-rerank-refresh-models').addEventListener('click', refreshRerankModels);
$('#btn-rerank').addEventListener('click', async function () {
  var query = $('#rerank-query').value.trim();
  if (!query) { alert('请填写查询文本。'); return; }
  var documents = buildDocumentsPayload();
  if (!documents || !documents.length) { alert('请至少提供一条文档。'); return; }
  var body = { query: query, documents: documents };
  var model = $('#rerank-model').value;
  if (model) body.model = model;
  var topN = $('#rerank-topn').value.trim();
  if (topN) {
    var n = parseInt(topN, 10);
    if (!Number.isFinite(n) || n < 1) { alert('top_n 必须是正整数。'); return; }
    body.top_n = n;
  }
  $('#rerank-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    var r = await api('POST', '/v1/rerank', body);
    renderRerankResults(r.payload, documents);
  } catch (e) {
    $('#rerank-results').innerHTML =
      '<div class="empty error">请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* ============ Image embeddings ============ */
async function refreshImageModels() {
  var data = [];
  try {
    var r = await api('GET', '/v1/models');
    var all = (r && r.payload && r.payload.data) || [];
    data = all.filter(function (m) { return m.type === 'image_embedder'; });
  } catch (_e) { data = []; }
  var list = $('#image-emb-models-list');
  if (list) {
    list.innerHTML = '';
    if (!data.length) {
      list.innerHTML = '<div class="empty">暂无已注册的图像嵌入模型。</div>';
    } else {
      data.forEach(function (m) {
        var item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML =
          '<span class="name">' + escapeHtml(m.id) + '</span>' +
          '<span class="meta">' + (m.loaded ? (m.dimensions + ' 维') : '未加载') + '</span>';
        item.addEventListener('click', async function () {
          try { await api('GET', '/v1/models/' + enc(m.id)); } catch (_e) {}
        });
        list.appendChild(item);
      });
    }
  }
  var sel = $('#image-emb-model');
  if (sel) {
    var prev = sel.value;
    sel.innerHTML = '';
    if (!data.length) {
      var o = document.createElement('option');
      o.value = ''; o.textContent = '（暂无模型）';
      sel.appendChild(o);
    } else {
      data.forEach(function (m) {
        var opt = document.createElement('option');
        opt.value = m.id; opt.textContent = m.id;
        sel.appendChild(opt);
      });
    }
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
  }
}

function readFileAsBase64(file) {
  return new Promise(function (resolve, reject) {
    var reader = new FileReader();
    reader.onerror = function () { reject(reader.error || new Error('FileReader failed')); };
    reader.onload = function () {
      var result = String(reader.result || '');
      var comma = result.indexOf(',');
      if (comma < 0) { reject(new Error('unexpected FileReader result')); return; }
      var meta = result.slice(5, comma);
      var semi = meta.indexOf(';');
      var guessedMime = semi > 0 ? meta.slice(0, semi) : (file.type || '');
      var data = result.slice(comma + 1);
      resolve({ data: data, mime: guessedMime });
    };
    reader.readAsDataURL(file);
  });
}

$('#image-emb-file').addEventListener('change', function () {
  var f = $('#image-emb-file').files[0];
  var info = $('#image-emb-file-info');
  if (!f) { info.textContent = '尚未选择文件。'; return; }
  info.textContent = f.name + ' · ' + f.type + ' · ' + (f.size / 1024).toFixed(1) + ' KB';
});
$('#image-emb-mode').addEventListener('change', function () {
  var mode = $('#image-emb-mode').value;
  $('#image-emb-single-row').classList.toggle('hidden', mode !== 'single');
  $('#image-emb-list-row').classList.toggle('hidden', mode !== 'list');
  $('#image-emb-mime-row').classList.toggle('hidden', mode !== 'single');
});

function renderImageEmbResults(payload) {
  var root = $('#image-emb-results');
  if (!root) return;
  if (!payload || !Array.isArray(payload.data)) {
    root.innerHTML = '<div class="empty">响应中未包含 data 数组。</div>';
    return;
  }
  var head =
    '<dl class="traffic-section" style="margin-bottom:12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>request_id</dt><dd>' + escapeHtml(payload.request_id || '') + '</dd>' +
      '<dt>数量</dt><dd>' + payload.data.length + '</dd>' +
    '</dl>';
  var rows = payload.data.map(function (item) {
    var v = Array.isArray(item.embedding) ? item.embedding : [];
    var preview = v.slice(0, 8).map(function (n) { return typeof n === 'number' ? n.toFixed(4) : String(n); });
    var more = v.length > 8 ? ' … (+' + (v.length - 8) + ')' : '';
    return '<div class="result-row">' +
      '<span class="result-rank">#' + (item.index + 1) + '</span>' +
      '<span class="result-idx">' + v.length + ' 维</span>' +
      '<span class="result-score">' + escapeHtml(preview.join(', ')) + more + '</span>' +
      '<span class="result-doc">[' + escapeHtml(preview.join(',')) + more + ']</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

$('#btn-image-emb-refresh-models').addEventListener('click', refreshImageModels);
$('#btn-image-emb').addEventListener('click', async function () {
  var model = $('#image-emb-model').value;
  if (!model) { alert('请先选择图像嵌入模型（点击上方"刷新"加载）。'); return; }
  var mode = $('#image-emb-mode').value;
  var input;
  if (mode === 'list') {
    var parsed = safeParse($('#image-emb-list').value);
    if (!Array.isArray(parsed) || !parsed.length) {
      alert('列表模式下，输入必须是包含至少一项 {"data","mime"} 的 JSON 数组。');
      return;
    }
    input = parsed;
  } else {
    var f = $('#image-emb-file').files[0];
    if (!f) { alert('请选择一张图片。'); return; }
    var mimeOverride = $('#image-emb-mime').value;
    var data, mime;
    try {
      var r = await readFileAsBase64(f);
      data = r.data; mime = r.mime;
    } catch (e) {
      alert('读取文件失败：' + (e && e.message || String(e)));
      return;
    }
    if (mimeOverride) mime = mimeOverride;
    if (!mime) { alert('无法识别图片 MIME 类型，请在右侧下拉框手动选择。'); return; }
    input = { data: data, mime: mime };
  }
  $('#image-emb-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    var rr = await api('POST', '/v1/image_embeddings', { model: model, input: input });
    renderImageEmbResults(rr.payload);
  } catch (e) {
    $('#image-emb-results').innerHTML =
      '<div class="empty error">请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* ============ Multimodal embeddings ============ */
async function refreshMultimodalModels() {
  var data = [];
  try {
    var r = await api('GET', '/v1/models');
    var all = (r && r.payload && r.payload.data) || [];
    data = all.filter(function (m) { return m.type === 'multimodal_embedder'; });
  } catch (_e) { data = []; }
  var list = $('#mm-emb-models-list');
  if (list) {
    list.innerHTML = '';
    if (!data.length) {
      list.innerHTML = '<div class="empty">暂无已注册的图文嵌入模型。</div>';
    } else {
      data.forEach(function (m) {
        var item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML =
          '<span class="name">' + escapeHtml(m.id) + '</span>' +
          '<span class="meta">' + (m.loaded ? (m.dimensions + ' 维') : '未加载') + '</span>';
        item.addEventListener('click', async function () {
          try { await api('GET', '/v1/models/' + enc(m.id)); } catch (_e) {}
        });
        list.appendChild(item);
      });
    }
  }
  var sel = $('#mm-emb-model');
  if (sel) {
    var prev = sel.value;
    sel.innerHTML = '';
    if (!data.length) {
      var o = document.createElement('option');
      o.value = ''; o.textContent = '（暂无模型）';
      sel.appendChild(o);
    } else {
      data.forEach(function (m) {
        var opt = document.createElement('option');
        opt.value = m.id; opt.textContent = m.id;
        sel.appendChild(opt);
      });
    }
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
  }
}

var mmEmbItems = [];

function rerenderMmEmbItems() {
  var root = $('#mm-emb-items');
  root.innerHTML = '';
  mmEmbItems.forEach(function (it, idx) {
    var card = document.createElement('div');
    card.className = 'field-card';
    var isText = it.kind === 'text';
    card.innerHTML =
      '<div class="field-card-header">' +
        '<span class="index-badge">项 #' + (idx + 1) + '</span>' +
        '<span class="title">' + (isText ? '文本' : '图片') + '</span>' +
        '<button class="remove" type="button" title="删除">×</button>' +
      '</div>' +
      (isText
        ? '<div class="grid"><div class="field"><label>中文文本</label>' +
            '<textarea data-role="text" rows="2" placeholder="例如：一只猫"></textarea></div></div>'
        : '<div class="grid cols-3">' +
            '<div class="field" style="grid-column: 1 / 3;"><label>选择图片 <span class="hint">MIME 从文件类型推断</span></label>' +
              '<input type="file" data-role="file" accept="image/png,image/jpeg,image/webp" /></div>' +
            '<div class="field"><label>MIME</label>' +
              '<select data-role="mime">' +
                '<option value="">（自动）</option>' +
                '<option value="image/png">image/png</option>' +
                '<option value="image/jpeg">image/jpeg</option>' +
                '<option value="image/webp">image/webp</option>' +
              '</select></div>' +
            '<div class="field" style="grid-column: 1 / 4;"><label>&nbsp;</label>' +
              '<div data-role="info" style="font-family:var(--mono);font-size:11px;color:var(--text-dim);">尚未选择文件。</div></div>' +
          '</div>');
    if (isText) {
      var ta = card.querySelector('[data-role="text"]');
      ta.value = it.text || '';
      ta.addEventListener('input', function () { it.text = ta.value; });
    } else {
      var fi = card.querySelector('[data-role="file"]');
      var mi = card.querySelector('[data-role="mime"]');
      var info = card.querySelector('[data-role="info"]');
      fi.addEventListener('change', function () {
        var f = fi.files[0];
        it.file = f || null;
        if (f) info.textContent = f.name + ' · ' + (f.type || '?') + ' · ' + (f.size / 1024).toFixed(1) + ' KB';
        else    info.textContent = '尚未选择文件。';
      });
      if (it.mimeOverride) mi.value = it.mimeOverride;
      mi.addEventListener('change', function () { it.mimeOverride = mi.value; });
    }
    card.querySelector('.remove').addEventListener('click', function () {
      mmEmbItems.splice(idx, 1);
      rerenderMmEmbItems();
    });
    root.appendChild(card);
  });
  if (!mmEmbItems.length) {
    root.innerHTML = '<div class="empty">还没有任何项。点击下方按钮添加文本或图片。</div>';
  }
}

$('#btn-mm-emb-add-text').addEventListener('click', function () {
  mmEmbItems.push({ kind: 'text', text: '' });
  rerenderMmEmbItems();
});
$('#btn-mm-emb-add-image').addEventListener('click', function () {
  mmEmbItems.push({ kind: 'image', file: null, mimeOverride: '' });
  rerenderMmEmbItems();
});
$('#btn-mm-emb-refresh-models').addEventListener('click', refreshMultimodalModels);

function renderMmEmbResults(payload) {
  var root = $('#mm-emb-results');
  if (!root) return;
  if (!payload || !Array.isArray(payload.data)) {
    root.innerHTML = '<div class="empty">响应中未包含 data 数组。</div>';
    return;
  }
  var head =
    '<dl class="traffic-section" style="margin-bottom:12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>数量</dt><dd>' + payload.data.length + '</dd>' +
    '</dl>';
  var rows = payload.data.map(function (item) {
    var v = Array.isArray(item.embedding) ? item.embedding : [];
    var preview = v.slice(0, 8).map(function (n) { return typeof n === 'number' ? n.toFixed(4) : String(n); });
    var more = v.length > 8 ? ' … (+' + (v.length - 8) + ')' : '';
    return '<div class="result-row">' +
      '<span class="result-rank">#' + (item.index + 1) + '</span>' +
      '<span class="result-idx">' + v.length + ' 维</span>' +
      '<span class="result-score">' + escapeHtml(preview.join(', ')) + more + '</span>' +
      '<span class="result-doc">[' + escapeHtml(preview.join(',')) + more + ']</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

$('#btn-mm-emb').addEventListener('click', async function () {
  var model = $('#mm-emb-model').value;
  if (!model) { alert('请先选择图文嵌入模型（点击上方"刷新"加载）。'); return; }
  if (!mmEmbItems.length) { alert('请至少添加一项输入。'); return; }

  var input = [];
  for (var i = 0; i < mmEmbItems.length; i++) {
    var it = mmEmbItems[i];
    if (it.kind === 'text') {
      var t = (it.text || '').trim();
      if (!t) { alert('第 ' + (i + 1) + ' 项文本为空。'); return; }
      input.push({ text: t });
    } else {
      if (!it.file) { alert('第 ' + (i + 1) + ' 项图片未选择。'); return; }
      var data, mime;
      try {
        var r = await readFileAsBase64(it.file);
        data = r.data; mime = r.mime;
      } catch (e) {
        alert('第 ' + (i + 1) + ' 项读取文件失败：' + (e && e.message || String(e)));
        return;
      }
      if (it.mimeOverride) mime = it.mimeOverride;
      if (!mime) { alert('第 ' + (i + 1) + ' 项无法识别 MIME，请在右侧下拉框手动选择。'); return; }
      input.push({ image: { data: data, mime: mime } });
    }
  }

  $('#mm-emb-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    var rr = await api('POST', '/v1/multimodal_embeddings', { model: model, input: input });
    renderMmEmbResults(rr.payload);
  } catch (e) {
    $('#mm-emb-results').innerHTML =
      '<div class="empty error">请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* ============ Similarity panels ============
   Three debug panels (text / image / multimodal) that share the same
   result-row layout. The endpoints always return results sorted
   most-similar-first (cosine/ip desc, l2 asc), so rank == position+1. */

function renderSimResults(rootId, payload, labels) {
  var root = $('#' + rootId);
  if (!root) return;
  if (!payload || !Array.isArray(payload.results)) {
    root.innerHTML = '<div class="empty">响应中未包含 results 数组。</div>';
    return;
  }
  var head =
    '<dl class="traffic-section" style="margin-bottom:12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>度量</dt><dd>' + escapeHtml(payload.metric || '') + '</dd>' +
      '<dt>命中数</dt><dd>' + payload.results.length + '</dd>' +
    '</dl>';
  if (!payload.results.length) {
    root.innerHTML = head + '<div class="empty">无结果。</div>';
    return;
  }
  var rows = payload.results.map(function (r, i) {
    var label = (labels && labels[r.index]) || ('#' + r.index);
    var truncated = label.length > 120 ? label.slice(0, 120) + '…' : label;
    var score = (typeof r.score === 'number') ? r.score.toFixed(4) : String(r.score);
    return '<div class="result-row">' +
      '<span class="result-rank">#' + (i + 1) + '</span>' +
      '<span class="result-idx">idx=' + r.index + '</span>' +
      '<span class="result-score">' + score + '</span>' +
      '<span class="result-doc" title="' + escapeHtml(label) + '">' + escapeHtml(truncated) + '</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

/* ---- text similarity ---- */
async function refreshTextSimModels() {
  var sel = $('#text-sim-model');
  var list = $('#text-sim-models-list');
  if (!sel || !list) return;
  var prev = sel.value;
  list.innerHTML = '<div class="empty">加载中…</div>';
  try {
    var rr = await api('GET', '/v1/models', null);
    var data = (rr.payload && rr.payload.data) || [];
    var embedders = data.filter(function (m) { return m.type === 'embedder'; });
    sel.innerHTML = '';
    if (!embedders.length) {
      var o = document.createElement('option');
      o.value = ''; o.textContent = '（暂无 embedder）';
      sel.appendChild(o);
      list.innerHTML = '<div class="empty">未注册任何 embedder。</div>';
      return;
    }
    embedders.forEach(function (m) {
      var opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id + (m.loaded ? ' · 已加载' : '');
      sel.appendChild(opt);
    });
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
    list.innerHTML = embedders.map(function (m) {
      return '<div class="list-item">' +
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.loaded ? '已加载' : '未加载') +
        (m.dimensions ? ' · ' + m.dimensions + ' 维' : '') + '</span>' +
      '</div>';
    }).join('');
  } catch (e) {
    list.innerHTML = '<div class="empty error">加载失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
}

$('#btn-text-sim-refresh-models').addEventListener('click', refreshTextSimModels);

$('#btn-text-sim').addEventListener('click', async function () {
  var model = $('#text-sim-model').value;
  if (!model) { alert('请先选择文本嵌入模型（点击上方"刷新"加载）。'); return; }
  var query = $('#text-sim-query').value.trim();
  if (!query) { alert('查询文本不能为空。'); return; }
  var docsRaw = $('#text-sim-docs').value;
  var documents = docsRaw.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(function (s) { return s.length > 0; });
  if (!documents.length) { alert('候选文档不能为空（每行一条）。'); return; }
  var metric = $('#text-sim-metric').value;

  $('#text-sim-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    var rr = await api('POST', '/v1/text_similarity', {
      model: model, query: query, documents: documents, metric: metric,
    });
    renderSimResults('text-sim-results', rr.payload, documents);
  } catch (e) {
    $('#text-sim-results').innerHTML =
      '<div class="empty error">请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* ---- image similarity ---- */
async function refreshImageSimModels() {
  var sel = $('#image-sim-model');
  var list = $('#image-sim-models-list');
  if (!sel || !list) return;
  var prev = sel.value;
  list.innerHTML = '<div class="empty">加载中…</div>';
  try {
    var rr = await api('GET', '/v1/models', null);
    var data = (rr.payload && rr.payload.data) || [];
    var embedders = data.filter(function (m) { return m.type === 'image_embedder'; });
    sel.innerHTML = '';
    if (!embedders.length) {
      var o = document.createElement('option');
      o.value = ''; o.textContent = '（暂无 image_embedder）';
      sel.appendChild(o);
      list.innerHTML = '<div class="empty">未注册任何 image_embedder。</div>';
      return;
    }
    embedders.forEach(function (m) {
      var opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id + (m.loaded ? ' · 已加载' : '');
      sel.appendChild(opt);
    });
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
    list.innerHTML = embedders.map(function (m) {
      return '<div class="list-item">' +
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.loaded ? '已加载' : '未加载') +
        (m.dimensions ? ' · ' + m.dimensions + ' 维' : '') + '</span>' +
      '</div>';
    }).join('');
  } catch (e) {
    list.innerHTML = '<div class="empty error">加载失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
}

$('#btn-image-sim-refresh-models').addEventListener('click', refreshImageSimModels);
$('#image-sim-file').addEventListener('change', function () {
  var f = $('#image-sim-file').files[0];
  var info = $('#image-sim-file-info');
  if (!f) { info.textContent = '尚未选择文件。'; return; }
  info.textContent = f.name + ' · ' + f.type + ' · ' + (f.size / 1024).toFixed(1) + ' KB';
});

$('#btn-image-sim').addEventListener('click', async function () {
  var model = $('#image-sim-model').value;
  if (!model) { alert('请先选择图像嵌入模型（点击上方"刷新"加载）。'); return; }
  var f = $('#image-sim-file').files[0];
  if (!f) { alert('请选择一张查询图片。'); return; }
  var data, mime;
  try {
    var r = await readFileAsBase64(f);
    data = r.data; mime = r.mime;
  } catch (e) {
    alert('读取文件失败：' + (e && e.message || String(e))); return;
  }
  var mimeOverride = $('#image-sim-mime').value;
  if (mimeOverride) mime = mimeOverride;
  if (!mime) { alert('无法识别图片 MIME 类型，请在右侧下拉框手动选择。'); return; }

  var docsRaw = $('#image-sim-docs').value.trim() || '[]';
  var documents;
  try {
    documents = safeParse(docsRaw);
    if (!Array.isArray(documents) || !documents.length) {
      alert('候选图片必须是包含至少一项 {"data","mime"} 的 JSON 数组。'); return;
    }
  } catch (_e) {
    alert('候选图片 JSON 解析失败。'); return;
  }
  var metric = $('#image-sim-metric').value;

  $('#image-sim-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    var rr = await api('POST', '/v1/image_similarity', {
      model: model,
      query: { data: data, mime: mime },
      documents: documents,
      metric: metric,
    });
    renderSimResults('image-sim-results', rr.payload,
      documents.map(function (_d, i) { return '#' + i; }));
  } catch (e) {
    $('#image-sim-results').innerHTML =
      '<div class="empty error">请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* ---- multimodal similarity ---- */
async function refreshMmSimModels() {
  var sel = $('#mm-sim-model');
  var list = $('#mm-sim-models-list');
  if (!sel || !list) return;
  var prev = sel.value;
  list.innerHTML = '<div class="empty">加载中…</div>';
  try {
    var rr = await api('GET', '/v1/models', null);
    var data = (rr.payload && rr.payload.data) || [];
    var embedders = data.filter(function (m) { return m.type === 'multimodal_embedder'; });
    sel.innerHTML = '';
    if (!embedders.length) {
      var o = document.createElement('option');
      o.value = ''; o.textContent = '（暂无 multimodal_embedder）';
      sel.appendChild(o);
      list.innerHTML = '<div class="empty">未注册任何 multimodal_embedder。</div>';
      return;
    }
    embedders.forEach(function (m) {
      var opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id + (m.loaded ? ' · 已加载' : '');
      sel.appendChild(opt);
    });
    if (prev && Array.from(sel.options).some(function (o) { return o.value === prev; })) sel.value = prev;
    list.innerHTML = embedders.map(function (m) {
      return '<div class="list-item">' +
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.loaded ? '已加载' : '未加载') +
        (m.dimensions ? ' · ' + m.dimensions + ' 维' : '') + '</span>' +
      '</div>';
    }).join('');
  } catch (e) {
    list.innerHTML = '<div class="empty error">加载失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
}

$('#btn-mm-sim-refresh-models').addEventListener('click', refreshMmSimModels);

$('#btn-mm-sim').addEventListener('click', async function () {
  var model = $('#mm-sim-model').value;
  if (!model) { alert('请先选择图文嵌入模型（点击上方"刷新"加载）。'); return; }
  var queryRaw = $('#mm-sim-query').value.trim();
  var query;
  try { query = safeParse(queryRaw); }
  catch (_e) { alert('查询项 JSON 解析失败。'); return; }
  if (!query || (query.text == null && query.image == null)) {
    alert('查询项必须是包含 text 或 image 之一的 JSON 对象。'); return;
  }

  var docsRaw = $('#mm-sim-docs').value.trim() || '[]';
  var documents;
  try {
    documents = safeParse(docsRaw);
    if (!Array.isArray(documents) || !documents.length) {
      alert('候选项必须是非空 JSON 数组。'); return;
    }
  } catch (_e) {
    alert('候选项 JSON 解析失败。'); return;
  }

  var metric = $('#mm-sim-metric').value;

  $('#mm-sim-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    var rr = await api('POST', '/v1/multimodal_similarity', {
      model: model,
      query: query,
      documents: documents,
      metric: metric,
    });
    renderSimResults('mm-sim-results', rr.payload,
      documents.map(function (d, i) { return d.text ? ('#' + i + ' · text · ' + d.text) : '#' + i + ' · image'; }));
  } catch (e) {
    $('#mm-sim-results').innerHTML =
      '<div class="empty error">请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* ============ Overview panel ============ */
function fmtBytes(n) {
  if (n == null) return '—';
  if (n < 1024) return n + ' B';
  var units = ['KB', 'MB', 'GB', 'TB'];
  var v = n / 1024; var i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(v >= 100 ? 0 : (v >= 10 ? 1 : 2)) + ' ' + units[i];
}
function fmtUptime(seconds) {
  if (seconds == null || !isFinite(seconds)) return '—';
  var s = Math.max(0, Math.floor(seconds));
  var d = Math.floor(s / 86400); s -= d * 86400;
  var h = Math.floor(s / 3600);  s -= h * 3600;
  var m = Math.floor(s / 60);    s -= m * 60;
  if (d > 0) return d + '天 ' + h + 'h';
  if (h > 0) return h + 'h ' + m + 'm';
  if (m > 0) return m + 'm ' + s + 's';
  return s + 's';
}
function fmtPercent(v) {
  if (v == null) return '—';
  return (Math.round(v * 10) / 10) + '<span class="unit">%</span>';
}
function barClass(pct) {
  if (pct == null) return '';
  if (pct >= 90) return 'bar--err';
  if (pct >= 75) return 'bar--warn';
  return '';
}
function setBar(id, pct) {
  var bar = $('#' + id);
  if (!bar) return;
  var span = bar.querySelector('span');
  if (pct == null) { bar.className = 'bar'; span.style.width = '0%'; return; }
  bar.className = 'bar ' + barClass(pct);
  span.style.width = Math.max(0, Math.min(100, pct)) + '%';
}
function setKpi(key, state) {
  var el = $('#' + key);
  if (el) el.className = 'kpi ' + (state ? 'kpi--' + state : '');
}
function famRow(name, info) {
  var id = info && info.loaded_id;
  var dim = info && info.dimensions;
  var idEl = $('#' + name + '-id');
  if (idEl) {
    if (id) { idEl.textContent = id; idEl.classList.remove('empty'); }
    else    { idEl.textContent = '未加载'; idEl.classList.add('empty'); }
  }
  var dimEl = $('#' + name + '-dim');
  if (dimEl) dimEl.textContent = (dim != null) ? (dim + ' 维') : '';
}

function renderOverview(payload) {
  overviewState.data = payload;
  if (!payload) return;
  var svc = payload.service || {};
  var mdls = payload.models || {};
  var store = payload.store || {};
  var sys = payload.system || {};

  $('#kpi-version-val').textContent = (svc.version || '—');
  $('#kpi-version-sub').textContent = (svc.embedding_backend || '?') + ' · ' + (svc.vector_store_backend || '?');
  $('#kpi-uptime-val').textContent = fmtUptime(svc.uptime_seconds);
  var reg = 0, ld = 0;
  Object.keys(mdls).forEach(function (k) {
    var m = mdls[k] || {};
    reg += (m.registered || []).length;
    if (m.loaded_id) ld += 1;
  });
  $('#kpi-models-val').innerHTML = ld + '<span class="unit">/ ' + reg + '</span>';
  $('#kpi-models-sub').textContent = (Object.keys(mdls).length) + ' 个模型族';

  setKpi('kpi-store', store.status === 'ok' ? 'ok' : (store.status === 'down' ? 'err' : ''));
  $('#kpi-store-val').textContent = (store.database_count != null) ? store.database_count : '—';
  $('#kpi-store-sub').textContent = store.backend || '—';

  famRow('fam-embedder', mdls.embedder);
  famRow('fam-image', mdls.image_embedder);
  famRow('fam-mm', mdls.multimodal_embedder);
  famRow('fam-reranker', mdls.reranker);
  var regNames = [];
  Object.keys(mdls).forEach(function (k) {
    (mdls[k].registered || []).forEach(function (n) { regNames.push(k + ':' + n); });
  });
  $('#models-summary-sub').textContent = '已注册: ' + (regNames.join(', ') || '—');

  $('#store-backend').textContent = (store.backend || '—') +
    (store.configured_backend && store.configured_backend !== store.backend
      ? ' (configured: ' + store.configured_backend + ')' : '');
  $('#store-uri').textContent = store.uri || '—';
  var stEl = $('#store-status');
  if (store.status === 'ok') {
    stEl.innerHTML = '<span class="pill success">已连接</span>';
  } else if (store.status === 'down') {
    stEl.innerHTML = '<span class="pill danger">断开</span> <span style="color:var(--danger);font-size:11px;">' +
      escapeHtml(store.error || '') + '</span>';
  } else {
    stEl.innerHTML = '<span class="pill warn">未挂载</span>';
  }
  $('#store-db-count').textContent = (store.database_count != null) ? store.database_count : '—';
  var chipRoot = $('#store-db-chips');
  if (chipRoot) {
    chipRoot.innerHTML = (store.databases || []).map(function (n) {
      return '<span class="db-chip">' + escapeHtml(n) + '</span>';
    }).join('');
  }
  $('#store-summary-sub').textContent = store.status === 'ok'
    ? '已连接 · ' + store.database_count + ' 个数据库'
    : (store.status === 'down' ? '连接断开' : '未挂载');

  var os = sys.os || {};
  $('#os-system').textContent = (os.system || '—') + ' ' + (os.release || '');
  $('#os-release').textContent = os.version || '—';
  $('#os-machine').textContent = os.machine || '—';
  $('#os-host').textContent = os.hostname || '—';
  $('#os-python').textContent = os.python || '—';
  $('#os-summary-sub').textContent = os.hostname ? (os.hostname + ' · ' + (os.system || '?')) : '—';

  var cpu = sys.cpu || {};
  var cpuPct = cpu.percent;
  setBar('kpi-cpu-bar', cpuPct);
  $('#kpi-cpu-val').innerHTML = fmtPercent(cpuPct);
  $('#kpi-cpu-sub').textContent = (cpu.logical_cores || '?') + ' 逻辑核' +
    (cpu.physical_cores ? ' / ' + cpu.physical_cores + ' 物理核' : '');
  setKpi('kpi-cpu', cpuPct == null ? '' : (cpuPct >= 90 ? 'err' : (cpuPct >= 75 ? 'warn' : 'ok')));

  var mem = sys.memory || {};
  var memPct = mem.percent;
  setBar('kpi-mem-bar', memPct);
  $('#kpi-mem-val').innerHTML = fmtPercent(memPct);
  $('#kpi-mem-sub').textContent = mem.used_bytes != null
    ? (fmtBytes(mem.used_bytes) + ' / ' + fmtBytes(mem.total_bytes))
    : (sys.psutil_available ? 'psutil 不可用' : '未安装 psutil');
  setKpi('kpi-mem', memPct == null ? '' : (memPct >= 90 ? 'err' : (memPct >= 75 ? 'warn' : 'ok')));

  var disk = sys.disk || {};
  var dPct = disk.percent;
  setBar('kpi-disk-bar', dPct);
  $('#kpi-disk-val').innerHTML = fmtPercent(dPct);
  $('#kpi-disk-sub').textContent = disk.total_bytes != null
    ? (fmtBytes(disk.used_bytes) + ' / ' + fmtBytes(disk.total_bytes) +
       (disk.path ? ' · ' + disk.path : ''))
    : '—';
  setKpi('kpi-disk', dPct == null ? '' : (dPct >= 90 ? 'err' : (dPct >= 75 ? 'warn' : 'ok')));

  var proc = sys.process || {};
  $('#kpi-proc-val').textContent = fmtBytes(proc.rss_bytes);
  $('#kpi-proc-sub').textContent = proc.cpu_percent != null
    ? ('CPU ' + (Math.round(proc.cpu_percent * 10) / 10) + '%')
    : '—';

  var gpus = sys.gpus || [];
  var listRoot = $('#gpu-list');
  var visible = gpus.filter(function (g) { return g && (g.available !== false && g.error == null); });
  $('#gpu-count').textContent = visible.length;
  if (!visible.length) {
    var hint = (gpus[0] && gpus[0].hint) ? gpus[0].hint : '';
    var msg = (gpus[0] && gpus[0].error)
      ? ('pynvml 错误：' + gpus[0].error)
      : '未检测到可用 GPU。';
    var hintHtml = hint
      ? '<div style="color:var(--text-muted);font-size:12px;margin-top:8px;">' + escapeHtml(hint) + '</div>'
      : '';
    listRoot.innerHTML = '<div class="empty">' + escapeHtml(msg) + hintHtml + '</div>';
    $('#gpu-sub').textContent = '—';
  } else {
    $('#gpu-sub').textContent = visible.length + ' 个设备 · ' +
      ((visible[0] && visible[0].vendor) || 'unknown');
    listRoot.innerHTML = visible.map(function (g) {
      var memPctG = g.memory_percent;
      var memUsed = g.memory_used_bytes != null ? fmtBytes(g.memory_used_bytes) : '—';
      var memTotal = g.memory_total_bytes != null ? fmtBytes(g.memory_total_bytes) : '—';
      var util = g.utilization_percent != null ? (Math.round(g.utilization_percent) + '%') : '—';
      return '<div class="gpu-card">' +
        '<div class="gpu-card-head">' +
          '<span class="name">GPU ' + g.index + ' · ' + escapeHtml(g.name || '?') + '</span>' +
          '<span class="meta">' + escapeHtml(g.vendor || '') + '</span>' +
        '</div>' +
        '<div class="row">' +
          '<label>显存</label>' +
          '<div class="bar ' + barClass(memPctG) + '"><span style="width:' +
            (memPctG != null ? Math.min(100, memPctG) : 0) + '%"></span></div>' +
          '<span class="num">' + (memPctG != null ? Math.round(memPctG) + '%' : '—') + ' · ' + memUsed + ' / ' + memTotal + '</span>' +
        '</div>' +
        '<div class="row">' +
          '<label>利用率</label>' +
          '<div class="bar"><span style="width:' +
            (g.utilization_percent != null ? Math.min(100, g.utilization_percent) : 0) + '%"></span></div>' +
          '<span class="num">' + util + '</span>' +
        '</div>' +
      '</div>';
    }).join('');
  }
}

async function refreshOverview(silent) {
  try {
    var r = await fetch('/v1/system/status', { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    var p = await r.json();
    renderOverview(p);
  } catch (e) {
    if (!silent) console.warn('overview poll failed', e);
    $('#kpi-version-sub').textContent = '加载失败：' + (e.message || e);
  }
}

function startOverviewPolling() {
  if (overviewState.timer) return;
  refreshOverview(true);
  overviewState.timer = setInterval(function () { refreshOverview(true); }, 5000);
}
function stopOverviewPolling() {
  if (overviewState.timer) { clearInterval(overviewState.timer); overviewState.timer = null; }
}

/* ============ Boot ============ */
refreshDatabases();
refreshModels();
startOverviewPolling();

/* Keyboard shortcut: Ctrl+\ toggles the traffic panel */
document.addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === '\\') {
    e.preventDefault();
    $('#btn-traffic-toggle').click();
  }
});

})();
""" + _DASHBOARD_JS_END


DASHBOARD_HTML = (
    DASHBOARD_HTML_HEAD
    + DASHBOARD_HTML_BODY
    + DASHBOARD_JS
    + "</script>\n</body>\n</html>\n"
)


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Interactive debug dashboard",
    description=(
        "Self-contained light-themed HTML page that exercises every public "
        "endpoint of the service. Useful for local development; not part of "
        "the machine-to-machine API surface."
    ),
    include_in_schema=False,
)
def dashboard_page() -> HTMLResponse:
    """Serve the interactive dashboard page."""
    return HTMLResponse(content=DASHBOARD_HTML)

