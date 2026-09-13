"""Interactive playground page for debugging the service.

Mounts a single self-contained HTML page at ``GET /playground``. The page
talks to the existing public endpoints only (healthz, /v1/models,
/v1/embeddings, /v1/databases family) — it does not call any debug-only
or hidden routes.

UI is Chinese-localised. Schema is caller-defined: the Collections
"Create" panel lets you build ``scalar_fields`` / ``vector_field`` /
``index_params`` via structured form rows instead of raw JSON, and the
Vectors / Search panels ask for ``primary_field`` + ``vector_field``
names so requests stay in sync with whatever schema you created.
``/search`` uses Milvus-native ``filter_expr`` plus an optional
``output_fields`` list.

A persistent **报文查看器** at the bottom of the page shows the last
HTTP request / response in full (method, URL, headers, body, status,
timing) and stays visible across tab switches.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["playground"])

PLAYGROUND_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>vector-service · 调试台</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23FFFFFF'/%3E%3Ccircle cx='16' cy='16' r='5' fill='none' stroke='%232563EB' stroke-width='2'/%3E%3Ccircle cx='16' cy='16' r='2' fill='%232563EB'/%3E%3C/svg%3E" />
<style>
  :root {
    --bg: #FFFFFF;
    --surface: #F7F8FA;
    --surface-2: #EEF1F5;
    --surface-3: #FAFBFC;
    --border: #E2E6EC;
    --border-strong: #C9CFD8;
    --text: #1A1F2B;
    --text-dim: #5B6473;
    --text-muted: #8A93A3;
    --accent: #2563EB;
    --accent-dim: #1E40AF;
    --accent-soft: rgba(37,99,235,0.10);
    --danger: #C53030;
    --success: #2F855A;
    --warn: #B7791F;
    --code-bg: #F2F4F7;
    --shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 4px 12px rgba(15, 23, 42, 0.08);
    --font: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
            "Microsoft YaHei", "Segoe UI", Roboto, ui-sans-serif, system-ui, sans-serif;
    --mono: ui-monospace, "JetBrains Mono", "Fira Code", "Cascadia Code", Consolas,
            "Source Han Mono SC", monospace;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.5;
    background: var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }
  a { color: var(--accent); }

  /* ============= Top bar ============= */
  .topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 9px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    height: 44px;
  }
  .brand { font-family: var(--mono); font-size: 13px; letter-spacing: 0.01em; color: var(--text); }
  .brand .sep { color: var(--text-muted); margin: 0 6px; }
  .brand .tag { color: var(--accent); font-weight: 600; }
  .brand .hint {
    margin-left: 14px; font-size: 11px; color: var(--text-muted);
    letter-spacing: 0.04em;
  }
  .indicators { display: flex; gap: 14px; margin-left: auto; }
  .led {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: var(--mono); font-size: 11px;
    color: var(--text-dim); letter-spacing: 0.04em;
  }
  .led::before {
    content: ''; width: 7px; height: 7px; border-radius: 50%;
    background: var(--text-muted);
  }
  .led.ok::before    { background: var(--success); box-shadow: 0 0 6px rgba(47,133,90,0.45); }
  .led.warn::before  { background: var(--warn);    box-shadow: 0 0 6px rgba(183,121,31,0.45); }
  .led.err::before   { background: var(--danger);  box-shadow: 0 0 6px rgba(197,48,48,0.45); }
  .led.checking::before {
    background: var(--accent); animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
  @media (prefers-reduced-motion: reduce) { .led.checking::before { animation: none; opacity: 1; } }

  /* ============= Layout ============= */
  .layout {
    display: grid;
    grid-template-columns: 1fr 360px;
    grid-template-rows: 1fr auto;
    grid-template-areas:
      "main side"
      "traffic side";
    height: calc(100vh - 44px);
  }
  .main { grid-area: main; display: flex; flex-direction: column; overflow: hidden; }
  .side {
    grid-area: side;
    border-left: 1px solid var(--border); background: var(--surface);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .traffic {
    grid-area: traffic;
    border-top: 1px solid var(--border);
    background: var(--surface);
    max-height: 280px;
    display: flex; flex-direction: column; overflow: hidden;
  }

  /* ============= Tabs ============= */
  .tabs {
    display: flex; border-bottom: 1px solid var(--border); background: var(--bg);
    padding: 0 8px; overflow-x: auto;
    flex-shrink: 0;
  }
  .tab-groups {
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    flex-shrink: 0;
  }
  .tab-groups .tabs { border-bottom: none; }
  .category-tabs { background: var(--bg); }
  .sub-tabs {
    background: var(--surface-3);
    border-top: 1px solid var(--border);
    padding-top: 0; padding-bottom: 0;
  }
  .sub-tabs .tab { padding: 9px 14px; font-size: 11px; }
  .tab {
    padding: 11px 14px; font-size: 12px;
    color: var(--text-dim);
    background: none; border: none; cursor: pointer;
    border-bottom: 2px solid transparent; margin-bottom: -1px; white-space: nowrap;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }

  .panels { flex: 1; overflow: auto; padding: 20px; }
  .panel { display: none; }
  .panel.active { display: block; }

  /* ============= Form controls ============= */
  .row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
  .row.split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .row.split > .row { margin-bottom: 0; }
  .row.three { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 12px; }
  .row.three > .row { margin-bottom: 0; }
  label {
    font-size: 11px;
    color: var(--text-dim); letter-spacing: 0.04em; font-weight: 600;
  }
  label .hint {
    color: var(--text-muted); font-size: 11px; font-weight: 400;
    margin-left: 8px;
  }

  input[type="text"], input[type="number"], select, textarea {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 8px 10px; color: var(--text);
    font-family: var(--mono); font-size: 12px; outline: none;
    box-shadow: var(--shadow);
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.18);
  }
  textarea { resize: vertical; min-height: 64px; line-height: 1.45; }
  input[type="checkbox"] {
    accent-color: var(--accent);
    width: 14px; height: 14px;
  }
  select { cursor: pointer; }
  .check-row { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-dim); }

  /* ============= Buttons ============= */
  button {
    font-size: 11px;
    background: var(--surface-2); color: var(--text);
    border: 1px solid var(--border-strong); border-radius: 4px;
    padding: 8px 14px; cursor: pointer;
    transition: border-color 0.12s, color 0.12s, background 0.12s;
  }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button.primary {
    background: var(--accent); border-color: var(--accent);
    color: #FFFFFF; font-weight: 600;
  }
  button.primary:hover { background: var(--accent-dim); border-color: var(--accent-dim); color: #FFFFFF; }
  button.ghost {
    background: transparent; border-style: dashed;
  }
  button.icon {
    padding: 4px 8px; font-size: 10px;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  .actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 12px; }

  /* ============= Sections ============= */
  .section { margin-bottom: 28px; }
  .section h3 {
    font-size: 11px; letter-spacing: 0.06em; font-weight: 700;
    color: var(--text-dim);
    margin: 0 0 12px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
    text-transform: uppercase;
  }

  /* ============= Lists / items ============= */
  .list { display: flex; flex-direction: column; gap: 4px; margin-bottom: 16px; }
  .list-item {
    display: flex; align-items: center; gap: 12px; padding: 8px 12px;
    border: 1px solid var(--border); border-radius: 4px;
    background: var(--bg); font-family: var(--mono); font-size: 12px;
    cursor: pointer; transition: border-color 0.12s;
  }
  .list-item:hover { border-color: var(--accent); }
  .list-item .name { color: var(--text); flex: 1; }
  .list-item .meta { color: var(--text-dim); font-size: 11px; }
  .list-item button { padding: 4px 10px; font-size: 10px; }
  .empty {
    color: var(--text-muted); font-style: italic; font-family: var(--mono);
    font-size: 12px; padding: 18px; border: 1px dashed var(--border);
    border-radius: 4px; text-align: center; background: var(--surface);
  }

  /* ============= Field cards (structured rows) ============= */
  .field-card {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    background: var(--surface-3);
    margin-bottom: 8px;
    position: relative;
  }
  .field-card-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 8px;
  }
  .field-card-header .index-badge {
    font-family: var(--mono); font-size: 10px; color: var(--text-muted);
    background: var(--surface-2);
    padding: 2px 6px; border-radius: 3px;
  }
  .field-card-header .title {
    font-family: var(--mono); font-size: 12px; color: var(--text); font-weight: 600;
  }
  .field-card-header .remove {
    margin-left: auto;
    background: transparent; border: none; color: var(--text-muted);
    font-size: 14px; padding: 0 6px; cursor: pointer;
  }
  .field-card-header .remove:hover { color: var(--danger); }
  .field-card .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }
  .field-card .grid.cols-3 {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .field-card .grid > .field { display: flex; flex-direction: column; gap: 4px; }
  .field-card .field label { font-size: 10px; color: var(--text-muted); }
  .field-card .field input, .field-card .field select {
    padding: 6px 8px; font-size: 12px;
  }
  .field-card.dim-hidden .dim-cell,
  .field-card.maxlen-hidden .maxlen-cell,
  .field-card.params-hidden .params-cell { display: none; }

  .add-row {
    display: flex; align-items: center; justify-content: center;
    padding: 8px;
    border: 1px dashed var(--border-strong);
    border-radius: 6px;
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    margin-top: 4px;
    font-size: 12px;
  }
  .add-row:hover { border-color: var(--accent); color: var(--accent); }

  .preset-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;
    margin-bottom: 12px;
  }
  .preset-grid button {
    text-align: left; padding: 8px 10px;
    font-family: var(--mono); font-size: 11px;
  }
  .preset-grid button .preset-title { color: var(--text); font-weight: 600; display: block; }
  .preset-grid button .preset-desc { color: var(--text-dim); font-size: 10px; display: block; margin-top: 2px; }

  details.collapsible {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface-3);
    margin-bottom: 12px;
  }
  details.collapsible > summary {
    cursor: pointer;
    padding: 10px 14px;
    font-size: 12px; color: var(--text-dim); font-weight: 600;
    list-style: none;
    user-select: none;
  }
  details.collapsible > summary::-webkit-details-marker { display: none; }
  details.collapsible > summary::before {
    content: '▸ '; color: var(--text-muted); margin-right: 4px;
    display: inline-block; transition: transform 0.15s;
  }
  details.collapsible[open] > summary::before { transform: rotate(90deg); }
  details.collapsible > .body { padding: 0 14px 14px; }

  /* ============= Response / code panes ============= */
  .response {
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--surface); overflow: hidden;
  }
  .response-head {
    display: flex; align-items: center; gap: 12px; padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  }
  .response-head .ts { margin-left: auto; color: var(--text-muted); }
  .status-chip {
    padding: 2px 8px; border-radius: 3px; background: var(--surface-2);
    color: var(--text); font-weight: 600; letter-spacing: 0.04em;
    border: 1px solid var(--border);
    font-family: var(--mono); font-size: 11px;
  }
  .status-chip.ok   { color: var(--success); border-color: var(--success); }
  .status-chip.warn { color: var(--warn);    border-color: var(--warn); }
  .status-chip.err  { color: var(--danger);  border-color: var(--danger); }
  pre.code-pane {
    margin: 0; padding: 12px 14px; font-family: var(--mono); font-size: 12px;
    line-height: 1.5; color: var(--text); overflow: auto; max-height: 380px;
    white-space: pre-wrap; word-break: break-word; background: var(--code-bg);
  }

  /* ============= Traffic panel (bottom) ============= */
  .traffic-head {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    display: flex; align-items: center; gap: 12px;
    flex-shrink: 0;
  }
  .traffic-head .title {
    font-size: 12px; font-weight: 600; color: var(--text);
  }
  .traffic-head .sub {
    font-size: 11px; color: var(--text-muted); font-family: var(--mono);
  }
  .traffic-head .right { margin-left: auto; display: flex; gap: 6px; }
  .traffic-tabs {
    display: flex; gap: 0;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    padding: 0 14px;
    flex-shrink: 0;
  }
  .traffic-tab {
    padding: 8px 12px;
    background: none; border: none; cursor: pointer;
    font-size: 12px; color: var(--text-dim);
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .traffic-tab.active {
    color: var(--accent); border-bottom-color: var(--accent); font-weight: 600;
  }
  .traffic-body { flex: 1; overflow: auto; padding: 12px 14px; }

  .traffic-section {
    display: grid; grid-template-columns: 80px 1fr; gap: 6px 14px;
    font-family: var(--mono); font-size: 12px;
    margin-bottom: 12px;
  }
  .traffic-section dt { color: var(--text-muted); }
  .traffic-section dd { margin: 0; color: var(--text); word-break: break-all; }

  /* ============= Side (request log) ============= */
  .side-head {
    padding: 12px 14px; border-bottom: 1px solid var(--border);
    font-size: 11px; color: var(--text-dim); font-weight: 600;
    letter-spacing: 0.06em;
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg);
  }
  .side-head button { padding: 4px 8px; font-size: 10px; }
  .log { flex: 1; overflow: auto; padding: 6px 8px; font-family: var(--mono); font-size: 11px; }
  .log-item {
    padding: 7px 8px; border-radius: 4px; cursor: pointer;
    border-left: 2px solid transparent; margin-bottom: 4px;
    background: var(--bg); border: 1px solid var(--border);
  }
  .log-item:hover { background: var(--surface-2); border-color: var(--accent); }
  .log-item.active { border-left-color: var(--accent); background: var(--accent-soft); border-color: var(--accent); }
  .log-line { display: flex; align-items: center; gap: 6px; min-width: 0; }
  .log-method { width: 38px; flex-shrink: 0; font-weight: 700; font-size: 10px; letter-spacing: 0.04em; }
  .log-method.POST    { color: var(--accent); }
  .log-method.PUT     { color: var(--warn); }
  .log-method.DELETE  { color: var(--danger); }
  .log-method.GET     { color: var(--success); }
  .log-path { color: var(--text); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
  .log-meta { color: var(--text-muted); font-size: 10px; padding-left: 44px; margin-top: 2px; }

  /* ============= Misc ============= */
  .hidden { display: none !important; }
  .footer-note {
    margin-top: 28px; padding-top: 14px;
    border-top: 1px solid var(--border);
    color: var(--text-muted); font-size: 11px;
  }

  .pill {
    display: inline-block; font-size: 10px;
    padding: 2px 6px; border-radius: 3px;
    background: var(--surface-2); color: var(--text-dim);
    border: 1px solid var(--border);
    font-family: var(--mono);
    margin-right: 4px;
  }
  .pill.accent { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }

  .tag-token {
    display: inline-block; font-size: 11px;
    padding: 3px 8px; border-radius: 99px;
    background: var(--surface-2); color: var(--text-dim);
    border: 1px solid var(--border);
    cursor: pointer; user-select: none;
    margin: 0 4px 4px 0;
    font-family: var(--mono);
  }
  .tag-token:hover { border-color: var(--accent); color: var(--accent); }
  .tag-token.active { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }

  /* ============= Rerank results ============= */
  .rerank-row {
    display: grid;
    grid-template-columns: 36px 64px 84px 1fr;
    gap: 10px;
    align-items: baseline;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg);
    margin-bottom: 4px;
    font-family: var(--mono);
    font-size: 12px;
  }
  .rerank-rank  { color: var(--accent); font-weight: 600; }
  .rerank-idx,
  .rerank-score { color: var(--text-dim); font-size: 11px; }
  .rerank-score { color: var(--text); }
  .rerank-doc {
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">
    vector-service<span class="sep">·</span><span class="tag">调试台</span>
    <span class="hint">仅调用公开接口 · 适合本地调试</span>
  </div>
  <div class="indicators">
    <div class="led" id="led-healthz"><span>healthz</span></div>
    <div class="led" id="led-readyz"><span>readyz</span></div>
  </div>
</div>

<div class="layout">
  <div class="main">
    <div class="tab-groups">
      <div class="tabs category-tabs" role="tablist">
        <button class="tab active" data-cat="models" role="tab">模型</button>
        <button class="tab" data-cat="store" role="tab">向量库</button>
      </div>
      <div class="tabs sub-tabs" data-cat="models" role="tablist">
        <button class="tab active" data-tab="models" role="tab">模型</button>
        <button class="tab" data-tab="embeddings" role="tab">嵌入</button>
        <button class="tab" data-tab="image-embeddings" role="tab">图像嵌入</button>
        <button class="tab" data-tab="multimodal-embeddings" role="tab">图文嵌入</button>
        <button class="tab" data-tab="rerank" role="tab">重排</button>
      </div>
      <div class="tabs sub-tabs hidden" data-cat="store" role="tablist">
        <button class="tab" data-tab="databases" role="tab">数据库</button>
        <button class="tab" data-tab="collections" role="tab">集合</button>
        <button class="tab" data-tab="vectors" role="tab">向量</button>
        <button class="tab" data-tab="search" role="tab">检索</button>
      </div>
    </div>
    <div class="panels">

      <!-- ===================== 模型 ===================== -->
      <div class="panel active" id="panel-models">
        <div class="section">
          <h3>模型列表 <span class="pill accent">GET /v1/models</span></h3>
          <div class="actions">
            <button class="primary" id="btn-refresh-models">刷新</button>
          </div>
          <div class="list" id="models-list"><div class="empty">点击"刷新"加载已注册的模型。</div></div>
        </div>
        <div class="footer-note">
          提示：点击列表中的模型可发起 <code>GET /v1/models/{id}</code> 并在底部报文面板查看响应。
        </div>
      </div>

      <!-- ===================== 嵌入 ===================== -->
      <div class="panel" id="panel-embeddings">
        <div class="section">
          <h3>文本嵌入 <span class="pill accent">POST /v1/embeddings</span></h3>
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
            <button class="primary" id="btn-embed">生成嵌入</button>
          </div>
        </div>
      </div>

      <!-- ===================== 图像嵌入 ===================== -->
      <div class="panel" id="panel-image-embeddings">
        <div class="section">
          <h3>已注册的图像嵌入后端 <span class="pill accent">GET /v1/models · type=image_embedder</span></h3>
          <div class="actions">
            <button class="primary" id="btn-image-emb-refresh-models">刷新</button>
          </div>
          <div class="list" id="image-emb-models-list"><div class="empty">点击"刷新"加载已注册的图像嵌入模型。</div></div>
        </div>

        <div class="section">
          <h3>生成图像嵌入 <span class="pill accent">POST /v1/image_embeddings</span></h3>
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
            <label>多张图片 <span class="hint">JSON 数组，每项 <code>{"data":"<base64>","mime":"image/png"}</code></span></label>
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
              <div id="image-emb-file-info" style="font-family: var(--mono); font-size: 11px; color: var(--text-dim);">尚未选择文件。</div>
            </div>
          </div>

          <div class="actions">
            <button class="primary" id="btn-image-emb">生成嵌入</button>
          </div>
        </div>

        <div class="section">
          <h3>结果</h3>
          <div id="image-emb-results"><div class="empty">尚无结果。点击"生成嵌入"查看响应。</div></div>
        </div>

        <div class="footer-note">
          提示：base64 字符串不带 <code>data:</code> URI 前缀；MIME 必须与服务端 <code>allowed_mime</code> 配置一致。
        </div>
      </div>

      <!-- ===================== 图文嵌入（跨模态） ===================== -->
      <div class="panel" id="panel-multimodal-embeddings">
        <div class="section">
          <h3>已注册的图文嵌入后端 <span class="pill accent">GET /v1/models · type=multimodal_embedder</span></h3>
          <div class="actions">
            <button class="primary" id="btn-mm-emb-refresh-models">刷新</button>
          </div>
          <div class="list" id="mm-emb-models-list"><div class="empty">点击"刷新"加载已注册的图文嵌入模型。</div></div>
        </div>

        <div class="section">
          <h3>生成图文嵌入 <span class="pill accent">POST /v1/multimodal_embeddings</span></h3>
          <div class="row">
            <label>模型</label>
            <select id="mm-emb-model"></select>
          </div>
          <div class="row">
            <label>输入项 <span class="hint">每项是文本或图片；输出顺序与请求顺序一一对应。文本与图片在同一向量空间（默认 512 维）。</span></label>
            <div id="mm-emb-items"></div>
            <button class="add-row" id="btn-mm-emb-add-text">＋ 添加文本项</button>
            <button class="add-row" id="btn-mm-emb-add-image" style="margin-top: 6px;">＋ 添加图片项</button>
          </div>
          <div class="actions">
            <button class="primary" id="btn-mm-emb">生成嵌入</button>
          </div>
        </div>

        <div class="section">
          <h3>结果</h3>
          <div id="mm-emb-results"><div class="empty">尚无结果。点击"生成嵌入"查看响应。</div></div>
        </div>

        <div class="footer-note">
          提示：文本项仅支持中文（Chinese-CLIP 训练集）。base64 字符串不带 <code>data:</code> URI 前缀；MIME 必须与服务端 <code>VS_MULTIMODAL_EMBEDDING__ALLOWED_MIME</code> 一致。
        </div>
      </div>

      <!-- ===================== 数据库 ===================== -->
      <div class="panel" id="panel-databases">
        <div class="section">
          <h3>数据库列表 <span class="pill accent">GET /v1/databases</span></h3>
          <div class="actions">
            <button class="primary" id="btn-refresh-dbs">刷新</button>
          </div>
          <div class="list" id="dbs-list"><div class="empty">点击"刷新"加载数据库。</div></div>
        </div>
        <div class="section">
          <h3>新建数据库 <span class="pill accent">POST /v1/databases</span></h3>
          <div class="row">
            <label>名称 <span class="hint">1-64 字符，字母 / 数字 / 下划线</span></label>
            <input type="text" id="new-db-name" placeholder="tenant-a" />
          </div>
          <div class="actions">
            <button class="primary" id="btn-create-db">创建</button>
          </div>
        </div>
        <div class="footer-note">
          提示：每个集合必须归属一个数据库；删除数据库会同时删除其下所有集合。
        </div>
      </div>

      <!-- ===================== 集合 ===================== -->
      <div class="panel" id="panel-collections">
        <div class="section">
          <h3>选择数据库</h3>
          <div class="row split">
            <div class="row">
              <label>当前数据库</label>
              <select id="colls-db"></select>
            </div>
            <div class="row">
              <label>&nbsp;</label>
              <button id="btn-colls-refresh-db" style="align-self:flex-start;">↻ 重新加载数据库列表</button>
            </div>
          </div>
        </div>
        <div class="section">
          <h3>集合列表 <span class="pill accent">GET /v1/databases/{db}/collections</span></h3>
          <div class="actions">
            <button class="primary" id="btn-refresh-colls">刷新</button>
          </div>
          <div class="list" id="colls-list"><div class="empty">选择数据库后点击"刷新"。</div></div>
        </div>

        <div class="section">
          <h3>新建集合 <span class="pill accent">POST /v1/databases/{db}/collections</span></h3>

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
              <div class="preset-grid" style="margin-top: 12px;">
                <button data-preset-scalar="id+category+price">
                  <span class="preset-title">预设：id + category + price</span>
                  <span class="preset-desc">varchar 主键 + varchar 分类 + float 价格</span>
                </button>
                <button data-preset-scalar="id+year">
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
              <div class="preset-grid" style="margin-top: 12px;">
                <button data-preset-index="hnsw-cosine">
                  <span class="preset-title">HNSW · 余弦</span>
                  <span class="preset-desc">M=16, efConstruction=200</span>
                </button>
                <button data-preset-index="ivf-l2">
                  <span class="preset-title">IVF_FLAT · 欧氏</span>
                  <span class="preset-desc">nlist=64</span>
                </button>
                <button data-preset-index="diskann-ip">
                  <span class="preset-title">DISKANN · 内积</span>
                  <span class="preset-desc">默认参数</span>
                </button>
                <button data-preset-index="default">
                  <span class="preset-title">使用默认</span>
                  <span class="preset-desc">由服务自动构建 HNSW + 向量字段的 metric</span>
                </button>
              </div>
            </div>
          </details>

          <div class="actions">
            <button class="primary" id="btn-create-coll">创建集合</button>
          </div>
        </div>
      </div>

      <!-- ===================== 向量 ===================== -->
      <div class="panel" id="panel-vectors">
        <div class="section">
          <h3>向量写入 <span class="pill accent">PUT /v1/databases/{db}/collections/{coll}/vectors</span></h3>
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
            <button class="primary" id="btn-upsert">写入（PUT）</button>
            <button id="btn-fetch">按主键获取（POST）</button>
            <button id="btn-delete">按主键删除（POST）</button>
          </div>
        </div>
      </div>

      <!-- ===================== 检索 ===================== -->
      <div class="panel" id="panel-search">
        <div class="section">
          <h3>向量检索 <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/search</span></h3>
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
                <label>filter_expr <span class="hint">例如 <code>category == 'mouse' and price < 100</code>；空 = 不过滤</span></label>
                <textarea id="srch-filter" rows="2" placeholder="category == 'mouse' and price < 100"></textarea>
              </div>
              <div id="srch-filter-tokens" style="margin-bottom: 8px;"></div>
              <div id="srch-filter-fields" class="tag-token-list"></div>
            </div>
          </details>

          <details class="collapsible">
            <summary>返回字段（output_fields）</summary>
            <div class="body">
              <div class="row">
                <label>输出字段 <span class="hint">留空 = 只返回主键。点击下方字段名可加入 / 移除</span></label>
                <div id="srch-output-tokens"></div>
                <div id="srch-output-fields" class="tag-token-list" style="margin-top: 6px;"></div>
              </div>
            </div>
          </details>

          <div class="actions">
            <button class="primary" id="btn-search">检索</button>
          </div>
        </div>
      </div>

      <!-- ===================== 重排 ===================== -->
      <div class="panel" id="panel-rerank">
        <div class="section">
          <h3>已注册的 Reranker 后端 <span class="pill accent">GET /v1/models · type=reranker</span></h3>
          <div class="actions">
            <button class="primary" id="btn-rerank-refresh-models">刷新</button>
          </div>
          <div class="list" id="rerank-models-list"><div class="empty">点击"刷新"加载已注册的 reranker。</div></div>
        </div>

        <div class="section">
          <h3>执行重排 <span class="pill accent">POST /v1/rerank</span></h3>
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
            <button class="primary" id="btn-rerank">执行重排</button>
          </div>
        </div>

        <div class="section">
          <h3>结果</h3>
          <div id="rerank-results"><div class="empty">尚无结果。点击"执行重排"查看响应。</div></div>
        </div>

        <div class="footer-note">
          提示：响应中的 <code>results[].index</code> 是请求文档数组里的下标；前端会用它反查你提交的原文以方便查看。
        </div>
      </div>

    </div>

    <!-- ========== Traffic / 报文 ========== -->
    <div class="traffic">
      <div class="traffic-head">
        <div class="title">报文</div>
        <div class="sub" id="traffic-summary">尚无调用</div>
        <div class="right">
          <button class="icon" id="btn-traffic-copy">复制报文</button>
          <button class="icon" id="btn-traffic-clear">清空</button>
        </div>
      </div>
      <div class="traffic-tabs" role="tablist">
        <button class="traffic-tab active" data-traffic="request">REQUEST</button>
        <button class="traffic-tab" data-traffic="response">RESPONSE</button>
      </div>
      <div class="traffic-body">
        <div id="traffic-request" class="traffic-pane">
          <div class="empty">选择上方任一标签页的操作即可发起调用，本面板会实时展示请求与响应报文。</div>
        </div>
        <div id="traffic-response" class="traffic-pane" hidden>
          <div class="empty">尚无响应。</div>
        </div>
      </div>
    </div>
  </div>

  <aside class="side" aria-label="请求历史">
    <div class="side-head">
      <span>请求历史</span>
      <button id="btn-clear-log">清空</button>
    </div>
    <div class="log" id="log">
      <div class="empty" style="margin: 8px;">还没有任何请求。</div>
    </div>
  </aside>
</div>

<script>
/* =========================================================================
 * 工具
 * ========================================================================= */
const $  = (s, p = document) => p.querySelector(s);
const $$ = (s, p = document) => Array.from(p.querySelectorAll(s));
const enc = s => encodeURIComponent(s);

function fmtJson(v) {
  try { return JSON.stringify(v, null, 2); }
  catch { return String(v); }
}
function safeParse(s) { try { return JSON.parse(s); } catch { return null; } }
function nowTs() { return new Date().toLocaleTimeString(); }

/* HTTP 状态码中文映射 */
function statusTextCN(code) {
  const map = {
    200: '成功', 201: '已创建', 204: '无内容',
    400: '请求错误', 401: '未授权', 403: '禁止访问', 404: '未找到',
    409: '冲突', 422: '参数错误',
    500: '服务器错误', 502: '网关不可达', 503: '服务不可用',
  };
  return map[code] || '';
}

/* =========================================================================
 * 报文查看器
 * ========================================================================= */
const trafficState = { entries: [], activeIdx: -1 };

function setTrafficSummary() {
  const sub = $('#traffic-summary');
  if (trafficState.activeIdx < 0) { sub.textContent = '尚无调用'; return; }
  const e = trafficState.entries[trafficState.activeIdx];
  if (!e) return;
  const dur = e.durationMs != null ? `${e.durationMs} ms` : '';
  const size = e.respSize != null ? `${e.respSize} B` : '';
  sub.textContent = `${e.method} ${e.url}  ·  ${e.status ?? '—'} ${statusTextCN(e.status ?? '')}  ·  ${dur}  ·  ${size}`;
}

function renderTrafficPanes() {
  const e = trafficState.activeIdx >= 0 ? trafficState.entries[trafficState.activeIdx] : null;

  // request pane
  const reqEl = $('#traffic-request');
  if (!e || !e.req) {
    reqEl.innerHTML = '<div class="empty">尚无请求。</div>';
  } else {
    reqEl.innerHTML =
      '<dl class="traffic-section">' +
        '<dt>方法</dt><dd>' + escapeHtml(e.method) + '</dd>' +
        '<dt>地址</dt><dd>' + escapeHtml(e.url) + '</dd>' +
        '<dt>请求头</dt><dd>' + renderHeaders(e.reqHeaders) + '</dd>' +
      '</dl>' +
      '<div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">请求体</div>' +
      '<pre class="code-pane">' + escapeHtml(e.req) + '</pre>';
  }

  // response pane
  const resEl = $('#traffic-response');
  if (!e || e.status == null) {
    resEl.innerHTML = '<div class="empty">尚无响应。</div>';
  } else {
    const errBlock = e.error
      ? '<div class="empty" style="color: var(--danger); border-color: var(--danger);">' +
          '请求失败：' + escapeHtml(e.error) + '</div>'
      : '';
    resEl.innerHTML =
      errBlock +
      '<dl class="traffic-section">' +
        '<dt>状态</dt><dd>' + (e.status || '—') + ' ' + escapeHtml(statusTextCN(e.status || 0)) + '</dd>' +
        '<dt>耗时</dt><dd>' + (e.durationMs != null ? e.durationMs + ' ms' : '—') + '</dd>' +
        '<dt>字节</dt><dd>' + (e.respSize != null ? e.respSize + ' B' : '—') + '</dd>' +
        '<dt>响应头</dt><dd>' + renderHeaders(e.respHeaders) + '</dd>' +
      '</dl>' +
      '<div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">响应体</div>' +
      '<pre class="code-pane">' + escapeHtml(e.resp || '') + '</pre>';
  }
}

function renderHeaders(headers) {
  if (!headers) return '—';
  const lines = Object.entries(headers).map(([k, v]) => escapeHtml(k) + ': ' + escapeHtml(String(v)));
  return lines.length ? lines.join('<br/>') : '—';
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&').replace(/</g, '<').replace(/>/g, '>')
    .replace(/"/g, '"').replace(/'/g, '&#39;');
}

$$('.traffic-tab').forEach(t => t.addEventListener('click', () => {
  $$('.traffic-tab').forEach(x => x.classList.toggle('active', x === t));
  const which = t.dataset.traffic;
  $('#traffic-request').hidden  = which !== 'request';
  $('#traffic-response').hidden = which !== 'response';
}));

$('#btn-traffic-copy').addEventListener('click', async () => {
  const e = trafficState.activeIdx >= 0 ? trafficState.entries[trafficState.activeIdx] : null;
  if (!e) { return; }
  const text = [
    '=== REQUEST ===',
    `${e.method} ${e.url}`,
    'Headers: ' + JSON.stringify(e.reqHeaders || {}, null, 2),
    '',
    e.req || '(no body)',
    '',
    '=== RESPONSE ===',
    `Status: ${e.status} ${statusTextCN(e.status || 0)}  (${e.durationMs ?? '—'} ms, ${e.respSize ?? '—'} B)`,
    'Headers: ' + JSON.stringify(e.respHeaders || {}, null, 2),
    '',
    e.resp || e.error || '(no body)',
  ].join('\\n');
  try {
    await navigator.clipboard.writeText(text);
    const btn = $('#btn-traffic-copy');
    const old = btn.textContent; btn.textContent = '已复制 ✓';
    setTimeout(() => { btn.textContent = old; }, 1200);
  } catch {}
});

$('#btn-traffic-clear').addEventListener('click', () => {
  trafficState.entries = []; trafficState.activeIdx = -1;
  setTrafficSummary();
  renderTrafficPanes();
  $$('.traffic-tab').forEach(x => x.classList.toggle('active', x.dataset.traffic === 'request'));
  $('#traffic-request').hidden = false; $('#traffic-response').hidden = true;
});

/* =========================================================================
 * HTTP 调用包装
 * ========================================================================= */
async function api(method, path, body) {
  const init = { method, headers: {} };
  let reqBodyText = '';
  if (body !== undefined && body !== null) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
    reqBodyText = JSON.stringify(body, null, 2);
  }

  const t0 = performance.now();
  let resp, text, error;
  try {
    resp = await fetch(path, init);
    text = await resp.text();
  } catch (e) {
    error = (e && e.message) || String(e);
  }
  const durationMs = Math.round(performance.now() - t0);
  let payload = text;
  let prettyText = text;
  if (text) {
    try {
      const parsed = JSON.parse(text);
      payload = parsed;
      prettyText = JSON.stringify(parsed, null, 2);
    } catch {}
  }

  // respHeaders 只展示关键字段，避免太长
  const respHeaders = {};
  if (resp && resp.headers) {
    resp.headers.forEach((v, k) => {
      if (['content-type', 'content-length', 'x-request-id', 'date', 'server'].includes(k.toLowerCase())) {
        respHeaders[k] = v;
      }
    });
  }

  // 把请求 url 拆成 path + query string
  const fullUrl = path;
  const entry = {
    method, url: fullUrl,
    req: reqBodyText,
    reqHeaders: { 'Content-Type': body !== undefined ? 'application/json' : '(none)' },
    status: resp ? resp.status : 0,
    durationMs,
    respSize: text ? text.length : 0,
    respHeaders,
    resp: prettyText,
    error,
    payload,  // 原始 parsed object，给列表渲染用
  };
  appendLog(entry);
  appendTraffic(entry);

  if (error) throw new Error(error);
  return { status: resp.status, ok: resp.ok && resp.status < 400, payload };
}

/* =========================================================================
 * 右侧请求历史
 * ========================================================================= */
const logEl = $('#log');
function appendLog(entry) {
  const empty = logEl.querySelector('.empty');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = 'log-item';
  const cls = !entry.status || entry.status >= 500 || entry.status === 0 ? 'err'
            : entry.status >= 400 ? 'warn'
            : entry.status >= 200 && entry.status < 300 ? 'ok' : '';
  const cn = statusTextCN(entry.status || 0);
  item.innerHTML =
    '<div class="log-line">' +
      '<span class="log-method ' + entry.method + '">' + entry.method + '</span>' +
      '<span class="log-path" title="' + escapeHtml(entry.url) + '">' + escapeHtml(entry.url) + '</span>' +
      '<span class="status-chip ' + cls + '">' + (entry.status || '×') + '</span>' +
    '</div>' +
    '<div class="log-meta">' + (entry.durationMs != null ? entry.durationMs + ' ms' : '—') +
      (cn ? ' · ' + cn : '') +
      (entry.error ? ' · ' + escapeHtml(entry.error) : '') +
    '</div>';
  item.addEventListener('click', () => {
    trafficState.activeIdx = trafficState.entries.length - 1 - trafficState.entries.indexOf(entry);
    // activeIdx 直接设为该 entry 的下标：
    trafficState.activeIdx = trafficState.entries.indexOf(entry);
    setTrafficSummary();
    renderTrafficPanes();
    $$('.log-item').forEach(x => x.classList.toggle('active', x === item));
  });
  logEl.insertBefore(item, logEl.firstChild);
  while (logEl.children.length > 80) logEl.removeChild(logEl.lastChild);
  $$('.log-item').forEach(x => x.classList.remove('active'));
  item.classList.add('active');
}

function appendTraffic(entry) {
  trafficState.entries.push(entry);
  if (trafficState.entries.length > 50) trafficState.entries.shift();
  trafficState.activeIdx = trafficState.entries.length - 1;
  setTrafficSummary();
  renderTrafficPanes();
}

$('#btn-clear-log').addEventListener('click', () => {
  logEl.innerHTML = '<div class="empty" style="margin: 8px;">还没有任何请求。</div>';
});

/* =========================================================================
 * 健康检查
 * ========================================================================= */
let checking = false;
async function pollHealth() {
  if (checking) return;
  checking = true;
  const setLed = (id, state) => { $('#' + id).className = 'led ' + (state === 'ok' ? 'ok' : state === 'warn' ? 'warn' : 'err'); };
  $('#led-healthz').className = 'led checking';
  $('#led-readyz').className  = 'led checking';
  try {
    const h = await fetch('/healthz').then(r => r.json()).catch(() => null);
    setLed('led-healthz', h && h.status === 'ok' ? 'ok' : 'err');
    const r = await fetch('/readyz').then(res => ({ ok: res.ok, status: res.status }))
                                   .catch(() => ({ ok: false }));
    setLed('led-readyz', r.ok ? 'ok' : 'warn');
  } finally {
    checking = false;
  }
}
pollHealth();
setInterval(pollHealth, 5000);

/* =========================================================================
 * Tab 切换（两级：分类 + 子 tab）
 * ========================================================================= */
const CATEGORY_CONFIG = {
  models: { defaultSub: 'models',    subTabs: ['models', 'embeddings', 'image-embeddings', 'multimodal-embeddings', 'rerank'] },
  store:  { defaultSub: 'databases', subTabs: ['databases', 'collections', 'vectors', 'search'] },
};
const navState = { category: 'models', subTab: 'models' };

function activateSubTab(name) {
  // 只切换有 data-tab 的按钮状态
  $$('[data-tab]').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
  if (name === 'vectors' || name === 'search' || name === 'collections') {
    refreshDatabases().then(refreshCollectionsInActive);
  } else if (name === 'databases') {
    refreshDatabases();
  } else if (name === 'embeddings') {
    refreshModels();
  } else if (name === 'image-embeddings') {
    refreshImageModels();
  } else if (name === 'multimodal-embeddings') {
    refreshMultimodalModels();
  } else if (name === 'rerank') {
    refreshRerankModels();
  }
}

$$('[data-cat]').forEach(tab => {
  tab.addEventListener('click', () => {
    const cat = tab.dataset.cat;
    if (cat === navState.category) return;
    navState.category = cat;
    // 大类 active 状态
    $$('[data-cat]').forEach(t => t.classList.toggle('active', t === tab));
    // 切换子 tab 行
    $$('.sub-tabs').forEach(row => row.classList.toggle('hidden', row.dataset.cat !== cat));
    // 子 tab 选择：新分类包含当前子 tab 则保留，否则取默认
    const cfg = CATEGORY_CONFIG[cat];
    const next = cfg.subTabs.includes(navState.subTab) ? navState.subTab : cfg.defaultSub;
    navState.subTab = next;
    activateSubTab(next);
  });
});

$$('[data-tab]').forEach(tab => {
  tab.addEventListener('click', () => {
    navState.subTab = tab.dataset.tab;
    activateSubTab(tab.dataset.tab);
  });
});

/* =========================================================================
 * 数据库
 * ========================================================================= */
async function refreshDatabases() {
  let dbs = [];
  try {
    const r = await api('GET', '/v1/databases');
    dbs = (r.payload && r.payload.databases) || [];
  } catch (e) {}
  const list = $('#dbs-list');
  list.innerHTML = '';
  if (!dbs.length) {
    list.innerHTML = '<div class="empty">暂无数据库。在下方表单创建第一个。</div>';
  } else {
    dbs.forEach(name => {
      const item = document.createElement('div');
      item.className = 'list-item';
      item.innerHTML =
        '<span class="name">' + escapeHtml(name) + '</span>' +
        '<button class="drop">删除</button>';
      item.querySelector('.drop').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm('确认删除数据库 "' + name + '" 及其下所有集合？')) return;
        try { await api('DELETE', '/v1/databases/' + enc(name)); } catch {}
        refreshDatabases();
      });
      list.appendChild(item);
    });
  }
  fillDbSelects(dbs);
  return dbs;
}
function fillDbSelects(dbs) {
  ['#colls-db', '#vec-db', '#srch-db'].forEach(sel => {
    const el = $(sel); if (!el) return;
    const prev = el.value;
    el.innerHTML = '';
    if (!dbs.length) {
      const o = document.createElement('option'); o.value = ''; o.textContent = '（暂无数据库）'; el.appendChild(o);
      return;
    }
    dbs.forEach(d => {
      const o = document.createElement('option'); o.value = d; o.textContent = d; el.appendChild(o);
    });
    if (prev && dbs.includes(prev)) el.value = prev;
  });
}
$('#btn-refresh-dbs').addEventListener('click', refreshDatabases);
$('#btn-colls-refresh-db').addEventListener('click', refreshDatabases);
$('#btn-create-db').addEventListener('click', async () => {
  const name = $('#new-db-name').value.trim();
  if (!name) { alert('请填写数据库名。'); return; }
  try {
    await api('POST', '/v1/databases', { name });
    $('#new-db-name').value = '';
    refreshDatabases();
  } catch (e) {}
});

/* =========================================================================
 * 模型
 * ========================================================================= */
async function refreshModels() {
  let data = null;
  try {
    const r = await api('GET', '/v1/models');
    data = (r.payload && r.payload.data) || [];
  } catch (e) {}
  const list = $('#models-list');
  list.innerHTML = '';
  if (!data || !data.length) {
    list.innerHTML = '<div class="empty">没有已注册的模型。</div>';
  } else {
    data.forEach(m => {
      const item = document.createElement('div');
      item.className = 'list-item';
      item.innerHTML =
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.dimensions ? m.dimensions + ' 维' : '维度未知') + '</span>';
      item.addEventListener('click', async () => {
        try { await api('GET', '/v1/models/' + enc(m.id)); } catch {}
      });
      list.appendChild(item);
    });
  }
  const sel = $('#emb-model'); sel.innerHTML = '';
  // 文本嵌入下拉只保留 embedder，排除 reranker / image_embedder
  (data || []).filter(m => m.type === 'embedder').forEach(m => {
    const o = document.createElement('option'); o.value = m.id; o.textContent = m.id; sel.appendChild(o);
  });
  return data || [];
}
$('#btn-refresh-models').addEventListener('click', refreshModels);
$('#btn-embed').addEventListener('click', async () => {
  const model = $('#emb-model').value;
  const mode  = $('#emb-mode').value;
  const inputRaw = $('#emb-input').value;
  let input;
  if (mode === 'list') {
    input = safeParse(inputRaw);
    if (!Array.isArray(input)) { alert('列表模式下，输入必须是 JSON 数组。'); return; }
  } else {
    input = inputRaw;
  }
  try { await api('POST', '/v1/embeddings', { model, input }); } catch {}
});

/* =========================================================================
 * 集合列表 + 创建
 * ========================================================================= */
function activeDb() {
  const v = $('#colls-db').value;
  if (!v) { alert('请先选择数据库。'); return null; }
  return v;
}
async function refreshCollectionsInActive() {
  const db = $('#colls-db').value;
  if (!db) {
    $('#colls-list').innerHTML = '<div class="empty">选择数据库后点击"刷新"。</div>';
    return [];
  }
  let colls = [];
  try {
    const r = await api('GET', '/v1/databases/' + enc(db) + '/collections');
    colls = (r.payload && r.payload.collections) || [];
  } catch (e) {}
  const list = $('#colls-list');
  list.innerHTML = '';
  if (!colls.length) {
    list.innerHTML = '<div class="empty">数据库 <code>' + escapeHtml(db) + '</code> 下暂无集合。</div>';
  } else {
    colls.forEach(name => {
      const item = document.createElement('div');
      item.className = 'list-item';
      item.innerHTML =
        '<span class="name">' + escapeHtml(db) + ' / ' + escapeHtml(name) + '</span>' +
        '<button class="drop">删除</button>';
      item.querySelector('.drop').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm('确认删除集合 "' + db + '/' + name + '"？')) return;
        try { await api('DELETE', '/v1/databases/' + enc(db) + '/collections/' + enc(name)); } catch {}
        refreshCollectionsInActive();
      });
      list.appendChild(item);
    });
  }
  const map = {}; map[$('#vec-db').value || db] = colls;
  fillCollSelects(map);
  rebuildSearchFieldTokens(db, colls);
  return colls;
}
function fillCollSelects(collsByDb) {
  const db = $('#vec-db').value;
  const colls = (collsByDb && collsByDb[db]) || [];
  ['#vec-coll', '#srch-coll'].forEach(sel => {
    const el = $(sel); if (!el) return;
    const prev = el.value;
    el.innerHTML = '';
    if (!colls.length) {
      const o = document.createElement('option'); o.value = ''; o.textContent = '（暂无集合）'; el.appendChild(o);
      return;
    }
    colls.forEach(c => {
      const o = document.createElement('option'); o.value = c; o.textContent = c; el.appendChild(o);
    });
    if (prev && colls.includes(prev)) el.value = prev;
  });
}
$('#colls-db').addEventListener('change', refreshCollectionsInActive);
$('#vec-db').addEventListener('change', refreshCollectionsInActive);
$('#btn-refresh-colls').addEventListener('click', refreshCollectionsInActive);

/* 字段 token 列表（检索标签） */
const searchFieldTokens = { outputs: new Set(), schemaCache: {} };

function rebuildSearchFieldTokens(dbName, colls) {
  searchFieldTokens.schemaCache = {};
  // 不主动拉 schema，留空；用户点进检索标签时按需加载
  renderOutputTokens();
  renderFilterFieldTokens([]);
}

function renderOutputTokens() {
  const root = $('#srch-output-tokens');
  const fields = collectCurrentSchemaFields();
  if (!fields.length) {
    root.innerHTML = '<div style="color: var(--text-muted); font-size: 11px;">尚无字段。点下方"加载字段"按钮可拉取当前集合的 schema。</div>';
    return;
  }
  root.innerHTML = fields.map(f => {
    const active = searchFieldTokens.outputs.has(f) ? 'active' : '';
    return '<span class="tag-token ' + active + '" data-field="' + escapeHtml(f) + '">' + escapeHtml(f) + '</span>';
  }).join('');
  root.querySelectorAll('.tag-token').forEach(el => {
    el.addEventListener('click', () => {
      const f = el.dataset.field;
      if (searchFieldTokens.outputs.has(f)) searchFieldTokens.outputs.delete(f);
      else searchFieldTokens.outputs.add(f);
      renderOutputTokens();
    });
  });
}

function renderFilterFieldTokens(fields) {
  const root = $('#srch-filter-fields');
  if (!fields.length) { root.innerHTML = ''; return; }
  root.innerHTML = '<div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">点击字段名可插入到过滤表达式：</div>' +
    fields.map(f => '<span class="tag-token" data-insert="' + escapeHtml(f) + '">' + escapeHtml(f) + '</span>').join('');
  root.querySelectorAll('.tag-token').forEach(el => {
    el.addEventListener('click', () => {
      const f = el.dataset.insert;
      const ta = $('#srch-filter');
      ta.value = (ta.value.trim() ? ta.value.trim() + ' and ' : '') + f + ' == ';
      ta.focus();
    });
  });
}

function collectCurrentSchemaFields() {
  // 从 schema cache 取当前 collection 的所有非向量字段名
  const coll = $('#srch-coll').value;
  const db   = $('#srch-db').value;
  if (!db || !coll) return [];
  const k = db + '::' + coll;
  const s = searchFieldTokens.schemaCache[k];
  return s ? s : [];
}

/* =========================================================================
 * 集合创建 — 动态表单
 * ========================================================================= */
const SCALAR_DTYPES = ['bool','int8','int16','int32','int64','float','double','varchar','json'];
const METRICS = ['cosine','ip','l2'];
const INDEX_TYPES = ['HNSW','IVF_FLAT','IVF_SQ8','IVF_PQ','DISKANN','FLAT','ANNOY','AUTOINDEX'];

function defaultScalarRow() {
  return { name: '', dtype: 'varchar', is_primary: false, max_length: 64, nullable: false, default_value: '' };
}
function defaultIndexRow() {
  return { field_name: 'vector', metric_type: 'cosine', index_type: 'HNSW', params: { M: 16, efConstruction: 200 } };
}

function renderScalarCard(row, idx) {
  const card = document.createElement('div');
  card.className = 'field-card';
  const isVarchar = row.dtype === 'varchar';
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
        '<select data-k="dtype">' + SCALAR_DTYPES.map(t => '<option value="' + t + '"' + (t === row.dtype ? ' selected' : '') + '>' + t + '</option>').join('') + '</select>' +
      '</div>' +
      '<div class="field maxlen-cell"><label>max_length（varchar 必填）</label><input data-k="max_length" type="number" min="1" max="65535" value="' + (row.max_length || 64) + '" /></div>' +
      '<div class="field"><label class="check-row"><input data-k="is_primary" type="checkbox"' + (row.is_primary ? ' checked' : '') + ' /> 作为主键</label></div>' +
      '<div class="field"><label class="check-row"><input data-k="nullable" type="checkbox"' + (row.nullable ? ' checked' : '') + ' /> 允许空值</label></div>' +
    '</div>';

  bindFieldCard(card, row);
  card.querySelector('.remove').addEventListener('click', () => card.remove());
  return card;
}

function renderIndexCard(row, idx) {
  const card = document.createElement('div');
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
        '<select data-k="metric_type">' + METRICS.map(t => '<option value="' + t + '"' + (t === row.metric_type ? ' selected' : '') + '>' + t + '</option>').join('') + '</select>' +
      '</div>' +
      '<div class="field"><label>索引类型</label>' +
        '<select data-k="index_type">' + INDEX_TYPES.map(t => '<option value="' + t + '"' + (t === row.index_type ? ' selected' : '') + '>' + t + '</option>').join('') + '</select>' +
      '</div>' +
    '</div>' +
    '<div class="grid" style="margin-top: 8px;">' +
      '<div class="field params-cell"><label>参数（JSON 对象）</label>' +
        '<textarea data-k="params" rows="2">' + escapeHtml(JSON.stringify(row.params || {})) + '</textarea>' +
      '</div>' +
    '</div>';

  bindFieldCard(card, row);
  card.querySelector('.remove').addEventListener('click', () => card.remove());
  return card;
}

function bindFieldCard(card, row) {
  card.querySelectorAll('[data-k]').forEach(el => {
    el.addEventListener('input', () => {
      const k = el.dataset.k;
      if (el.type === 'checkbox') row[k] = el.checked;
      else if (el.type === 'number') row[k] = el.value === '' ? null : Number(el.value);
      else row[k] = el.value;
      // varchar 联动 max_length 显隐
      if (k === 'dtype') {
        card.classList.toggle('maxlen-hidden', el.value !== 'varchar');
      }
      // 同步卡片标题
      const title = card.querySelector('.title');
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

let scalarRows = [defaultScalarRow()];
scalarRows[0].name = 'id';
scalarRows[0].is_primary = true;
let indexRows = [defaultIndexRow()];

function renderScalars() {
  const root = $('#scalars-list');
  root.innerHTML = '';
  scalarRows.forEach((r, i) => root.appendChild(renderScalarCard(r, i)));
}
function renderIndices() {
  const root = $('#index-list');
  root.innerHTML = '';
  indexRows.forEach((r, i) => root.appendChild(renderIndexCard(r, i)));
}
function renderVector() {
  const root = $('#vector-card');
  const row = { name: 'vector', dim: 1024, metric_type: 'cosine' };
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
          METRICS.map(t => '<option value="' + t + '"' + (t === 'cosine' ? ' selected' : '') + '>' + t + '</option>').join('') +
        '</select></div>' +
      '</div>' +
    '</div>';
}

renderScalars();
renderIndices();
renderVector();

$('#btn-add-scalar').addEventListener('click', () => {
  scalarRows.push(defaultScalarRow());
  renderScalars();
});
$('#btn-add-index').addEventListener('click', () => {
  indexRows.push(defaultIndexRow());
  renderIndices();
});
document.querySelectorAll('[data-preset-scalar]').forEach(btn => {
  btn.addEventListener('click', () => {
    const which = btn.getAttribute('data-preset-scalar');
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
document.querySelectorAll('[data-preset-index]').forEach(btn => {
  btn.addEventListener('click', () => {
    const which = btn.getAttribute('data-preset-index');
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
  // 校验主键
  const primary = scalarRows.filter(r => r.is_primary);
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
  for (const r of scalarRows) {
    if (r.dtype === 'varchar' && (!r.max_length || r.max_length < 1)) {
      alert('varchar 字段 ' + (r.name || '未命名') + ' 必须填写 max_length。');
      return null;
    }
    if (!r.name) { alert('每个字段都必须填写名称。'); return null; }
  }
  return scalarRows.map(r => {
    const out = { name: r.name, dtype: r.dtype, is_primary: !!r.is_primary };
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
  return indexRows.map(r => ({
    field_name: r.field_name,
    metric_type: r.metric_type,
    index_type: r.index_type,
    params: typeof r.params === 'string' ? (safeParse(r.params) || {}) : (r.params || {}),
  }));
}

$('#btn-create-coll').addEventListener('click', async () => {
  const db = activeDb(); if (!db) return;
  const name = $('#new-coll-name').value.trim();
  if (!name) { alert('请填写集合名。'); return; }
  const primary = $('#new-coll-primary').value.trim();
  if (!primary) { alert('请填写主键字段名。'); return; }

  const scalars = collectScalarPayload(); if (!scalars) return;
  // 确保主键名一致
  if (scalars.find(s => s.is_primary).name !== primary) {
    alert('主键字段名 (' + primary + ') 与勾选了 is_primary 的字段名不一致。');
    return;
  }

  const vecName = $('#vec-field-name').value.trim() || 'vector';
  const vecDim  = Number($('#vec-field-dim').value);
  const vecMet  = $('#vec-field-metric').value;
  if (!Number.isFinite(vecDim) || vecDim < 1) { alert('向量维度必须 ≥ 1。'); return; }

  const body = {
    name,
    primary_field: primary,
    scalar_fields: scalars,
    vector_field: { name: vecName, dim: vecDim, metric_type: vecMet },
    index_params: collectIndexPayload(),
  };
  try {
    await api('POST', '/v1/databases/' + enc(db) + '/collections', body);
    $('#new-coll-name').value = '';
    refreshCollectionsInActive();
  } catch (e) {}
});

/* =========================================================================
 * 向量写入
 * ========================================================================= */
$('#vec-mode').addEventListener('change', () => {
  const mode = $('#vec-mode').value;
  $('#vec-texts-row').classList.toggle('hidden', mode !== 'texts');
  $('#vec-embs-row').classList.toggle('hidden', mode !== 'vectors');
});
function buildVecBody() {
  const ids = safeParse($('#vec-ids').value);
  if (!Array.isArray(ids)) { alert('ids 必须是 JSON 数组。'); return null; }
  const body = {
    primary_field: $('#vec-primary').value.trim(),
    vector_field: $('#vec-vecfield').value.trim(),
    ids,
  };
  if ($('#vec-mode').value === 'texts') {
    const texts = safeParse($('#vec-texts').value);
    if (!Array.isArray(texts)) { alert('texts 必须是 JSON 数组。'); return null; }
    body.texts = texts;
  } else {
    const embeddings = safeParse($('#vec-embs').value);
    if (!Array.isArray(embeddings)) { alert('vectors 必须是 JSON 数组。'); return null; }
    body.vectors = embeddings;
  }
  const fieldsRaw = $('#vec-fields').value.trim();
  if (fieldsRaw && fieldsRaw !== '[]') {
    const fields = safeParse(fieldsRaw);
    if (!Array.isArray(fields)) { alert('fields 必须是 JSON 数组。'); return null; }
    if (fields.length) body.fields = fields;
  }
  return body;
}
$('#btn-upsert').addEventListener('click', async () => {
  const db = $('#vec-db').value; if (!db) { alert('请选择数据库。'); return; }
  const name = $('#vec-coll').value; if (!name) { alert('请选择集合。'); return; }
  const body = buildVecBody(); if (!body) return;
  try { await api('PUT', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/vectors', body); } catch {}
});
$('#btn-fetch').addEventListener('click', async () => {
  const db = $('#vec-db').value; if (!db) { alert('请选择数据库。'); return; }
  const name = $('#vec-coll').value; if (!name) { alert('请选择集合。'); return; }
  const ids = safeParse($('#vec-ids').value);
  if (!Array.isArray(ids)) { alert('ids 必须是 JSON 数组。'); return; }
  try {
    await api('POST', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/vectors/get', {
      primary_field: $('#vec-primary').value.trim(), ids,
    });
  } catch {}
});
$('#btn-delete').addEventListener('click', async () => {
  const db = $('#vec-db').value; if (!db) { alert('请选择数据库。'); return; }
  const name = $('#vec-coll').value; if (!name) { alert('请选择集合。'); return; }
  const ids = safeParse($('#vec-ids').value);
  if (!Array.isArray(ids)) { alert('ids 必须是 JSON 数组。'); return; }
  if (!confirm('确认删除 ' + ids.length + ' 条记录？')) return;
  try {
    await api('POST', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/vectors/delete', {
      primary_field: $('#vec-primary').value.trim(), ids,
    });
  } catch {}
});

/* =========================================================================
 * 检索
 * ========================================================================= */
$('#srch-mode').addEventListener('change', () => {
  const mode = $('#srch-mode').value;
  $('#srch-text-row').classList.toggle('hidden', mode !== 'text');
  $('#srch-emb-row').classList.toggle('hidden', mode !== 'emb');
});
$('#srch-coll').addEventListener('change', async () => {
  // 切换 collection 时按需拉 schema
  const db = $('#srch-db').value;
  const coll = $('#srch-coll').value;
  if (!db || !coll) return;
  const k = db + '::' + coll;
  if (searchFieldTokens.schemaCache[k]) return;
  try {
    const r = await api('GET', '/v1/databases/' + enc(db) + '/collections/' + enc(coll));
    const fields = (r.payload && r.payload.fields) || [];
    searchFieldTokens.schemaCache[k] = fields.map(f => f.name);
  } catch (e) {
    searchFieldTokens.schemaCache[k] = [];
  }
  renderOutputTokens();
  renderFilterFieldTokens(collectCurrentSchemaFields());
});
$('#btn-search').addEventListener('click', async () => {
  const db = $('#srch-db').value; if (!db) { alert('请选择数据库。'); return; }
  const name = $('#srch-coll').value; if (!name) { alert('请选择集合。'); return; }
  const topk = parseInt($('#srch-topk').value, 10);
  const body = {
    primary_field: $('#srch-primary').value.trim(),
    vector_field: $('#srch-vecfield').value.trim(),
    top_k: isNaN(topk) ? 10 : topk,
  };
  if ($('#srch-mode').value === 'text') {
    body.query_text = $('#srch-text').value;
  } else {
    const emb = safeParse($('#srch-emb').value);
    if (!Array.isArray(emb)) { alert('query_vector 必须是 JSON 数组。'); return; }
    body.query_vector = emb;
  }
  const filterExpr = $('#srch-filter').value.trim();
  if (filterExpr) body.filter_expr = filterExpr;
  if (searchFieldTokens.outputs.size) {
    body.output_fields = Array.from(searchFieldTokens.outputs);
  }
  try { await api('POST', '/v1/databases/' + enc(db) + '/collections/' + enc(name) + '/search', body); } catch {}
});

/* =========================================================================
 * 重排
 * ========================================================================= */
async function refreshRerankModels() {
  let data = [];
  try {
    const r = await api('GET', '/v1/models');
    const all = (r.payload && r.payload.data) || [];
    data = all.filter(m => m.type === 'reranker');
  } catch (e) {}
  const list = $('#rerank-models-list');
  list.innerHTML = '';
  if (!data.length) {
    list.innerHTML = '<div class="empty">暂无已注册的 reranker。</div>';
  } else {
    data.forEach(m => {
      const item = document.createElement('div');
      item.className = 'list-item';
      item.innerHTML =
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.dimensions != null ? m.dimensions + ' 维' : 'reranker') + '</span>';
      item.addEventListener('click', async () => {
        try { await api('GET', '/v1/models/' + enc(m.id)); } catch {}
      });
      list.appendChild(item);
    });
  }
  const sel = $('#rerank-model');
  const prev = sel.value;
  sel.innerHTML = '';
  const o0 = document.createElement('option');
  o0.value = ''; o0.textContent = '（使用服务端默认）';
  sel.appendChild(o0);
  data.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id; opt.textContent = m.id;
    sel.appendChild(opt);
  });
  if (prev && Array.from(sel.options).some(o => o.value === prev)) sel.value = prev;
}

/* =========================================================================
 * 图像嵌入
 * ========================================================================= */
async function refreshImageModels() {
  let data = [];
  try {
    const r = await api('GET', '/v1/models');
    const all = (r.payload && r.payload.data) || [];
    data = all.filter(m => m.type === 'image_embedder');
  } catch (e) {}
  const list = $('#image-emb-models-list');
  list.innerHTML = '';
  if (!data.length) {
    list.innerHTML = '<div class="empty">暂无已注册的图像嵌入模型。</div>';
  } else {
    data.forEach(m => {
      const item = document.createElement('div');
      item.className = 'list-item';
      item.innerHTML =
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.dimensions != null ? m.dimensions + ' 维' : '未加载') + '</span>';
      item.addEventListener('click', async () => {
        try { await api('GET', '/v1/models/' + enc(m.id)); } catch {}
      });
      list.appendChild(item);
    });
  }
  const sel = $('#image-emb-model');
  const prev = sel.value;
  sel.innerHTML = '';
  if (!data.length) {
    const o = document.createElement('option');
    o.value = ''; o.textContent = '（暂无模型）';
    sel.appendChild(o);
  } else {
    data.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id;
      sel.appendChild(opt);
    });
  }
  if (prev && Array.from(sel.options).some(o => o.value === prev)) sel.value = prev;
}

/* 文件 → base64（不带 data: 前缀）。MIME 默认取 file.type，可在下拉框强制覆盖。 */
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error('FileReader failed'));
    reader.onload = () => {
      // result 形如 "data:image/png;base64,iVBORw0..."
      const result = String(reader.result || '');
      const comma = result.indexOf(',');
      if (comma < 0) { reject(new Error('unexpected FileReader result')); return; }
      const meta = result.slice(5, comma); // "image/png;base64"
      const semi = meta.indexOf(';');
      const guessedMime = semi > 0 ? meta.slice(0, semi) : (file.type || '');
      const data = result.slice(comma + 1);
      resolve({ data, mime: guessedMime });
    };
    reader.readAsDataURL(file);
  });
}

$('#image-emb-file').addEventListener('change', () => {
  const f = $('#image-emb-file').files[0];
  const info = $('#image-emb-file-info');
  if (!f) { info.textContent = '尚未选择文件。'; return; }
  info.textContent = f.name + ' · ' + f.type + ' · ' + (f.size / 1024).toFixed(1) + ' KB';
});

$('#image-emb-mode').addEventListener('change', () => {
  const mode = $('#image-emb-mode').value;
  $('#image-emb-single-row').classList.toggle('hidden', mode !== 'single');
  $('#image-emb-list-row').classList.toggle('hidden', mode !== 'list');
  $('#image-emb-mime-row').classList.toggle('hidden', mode !== 'single');
});

function renderImageEmbResults(payload) {
  const root = $('#image-emb-results');
  if (!payload || !Array.isArray(payload.data)) {
    root.innerHTML = '<div class="empty">响应中未包含 data 数组。</div>';
    return;
  }
  const head =
    '<dl class="traffic-section" style="margin-bottom: 12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>request_id</dt><dd>' + escapeHtml(payload.request_id || '') + '</dd>' +
      '<dt>数量</dt><dd>' + payload.data.length + '</dd>' +
    '</dl>';
  const rows = payload.data.map(item => {
    const v = Array.isArray(item.embedding) ? item.embedding : [];
    const preview = v.slice(0, 8).map(n => (typeof n === 'number' ? n.toFixed(4) : String(n)));
    const more = v.length > 8 ? ' … (+' + (v.length - 8) + ')' : '';
    return '<div class="rerank-row">' +
      '<span class="rerank-rank">#' + (item.index + 1) + '</span>' +
      '<span class="rerank-idx">' + v.length + ' 维</span>' +
      '<span class="rerank-score">' + escapeHtml(preview.join(', ')) + more + '</span>' +
      '<span class="rerank-doc">[' + escapeHtml(preview.join(',')) + more + ']</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

$('#btn-image-emb-refresh-models').addEventListener('click', refreshImageModels);

$('#btn-image-emb').addEventListener('click', async () => {
  const model = $('#image-emb-model').value;
  if (!model) { alert('请先选择图像嵌入模型（点击上方"刷新"加载）。'); return; }
  const mode = $('#image-emb-mode').value;
  let input;
  if (mode === 'list') {
    const parsed = safeParse($('#image-emb-list').value);
    if (!Array.isArray(parsed) || !parsed.length) {
      alert('列表模式下，输入必须是包含至少一项 {"data","mime"} 的 JSON 数组。');
      return;
    }
    input = parsed;
  } else {
    const f = $('#image-emb-file').files[0];
    if (!f) { alert('请选择一张图片。'); return; }
    let mimeOverride = $('#image-emb-mime').value;
    let data, mime;
    try {
      ({ data, mime } = await readFileAsBase64(f));
    } catch (e) {
      alert('读取文件失败：' + (e && e.message || String(e)));
      return;
    }
    if (mimeOverride) mime = mimeOverride;
    if (!mime) { alert('无法识别图片 MIME 类型，请在右侧下拉框手动选择。'); return; }
    input = { data, mime };
  }
  $('#image-emb-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    const r = await api('POST', '/v1/image_embeddings', { model, input });
    renderImageEmbResults(r.payload);
  } catch (e) {
    $('#image-emb-results').innerHTML =
      '<div class="empty" style="color: var(--danger); border-color: var(--danger);">' +
      '请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

function buildDocumentsPayload() {
  const mode = $('#rerank-mode').value;
  const raw = $('#rerank-docs').value;
  if (mode === 'json') {
    const parsed = safeParse(raw);
    if (!Array.isArray(parsed)) { alert('JSON 模式下，文档必须是 JSON 数组。'); return null; }
    if (!parsed.every(x => typeof x === 'string')) { alert('文档数组的每个元素都必须是字符串。'); return null; }
    return parsed;
  }
  return raw.split(new RegExp("[\\r\\n]+")).map(s => s.trim()).filter(s => s.length > 0);
}

function renderRerankResults(payload, docs) {
  const root = $('#rerank-results');
  if (!payload || !Array.isArray(payload.results)) {
    root.innerHTML = '<div class="empty">响应中未包含 results 数组。</div>';
    return;
  }
  const head =
    '<dl class="traffic-section" style="margin-bottom: 12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>request_id</dt><dd>' + escapeHtml(payload.request_id || '') + '</dd>' +
      '<dt>命中数</dt><dd>' + payload.results.length + '</dd>' +
    '</dl>';
  if (!payload.results.length) {
    root.innerHTML = head + '<div class="empty">无结果。</div>';
    return;
  }
  const rows = payload.results.map((r, i) => {
    const doc = (typeof r.index === 'number' && docs[r.index]) || '(原文未提供)';
    const truncated = doc.length > 120 ? doc.slice(0, 120) + '…' : doc;
    const score = (typeof r.score === 'number') ? r.score.toFixed(4) : String(r.score);
    return '<div class="rerank-row">' +
      '<span class="rerank-rank">#' + (i + 1) + '</span>' +
      '<span class="rerank-idx">idx=' + r.index + '</span>' +
      '<span class="rerank-score">' + score + '</span>' +
      '<span class="rerank-doc" title="' + escapeHtml(doc) + '">' + escapeHtml(truncated) + '</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

$('#rerank-mode').addEventListener('change', () => {
  const mode = $('#rerank-mode').value;
  $('#rerank-docs').placeholder = mode === 'json'
    ? '["无线鼠标", "机械键盘", "蓝牙耳机", "游戏手柄"]'
    : '无线鼠标\\n机械键盘\\n蓝牙耳机\\n游戏手柄';
});

$('#btn-rerank-refresh-models').addEventListener('click', refreshRerankModels);

$('#btn-rerank').addEventListener('click', async () => {
  const query = $('#rerank-query').value.trim();
  if (!query) { alert('请填写查询文本。'); return; }
  const documents = buildDocumentsPayload();
  if (!documents || !documents.length) { alert('请至少提供一条文档。'); return; }
  const body = { query, documents };
  const model = $('#rerank-model').value;
  if (model) body.model = model;
  const topN = $('#rerank-topn').value.trim();
  if (topN) {
    const n = parseInt(topN, 10);
    if (!Number.isFinite(n) || n < 1) { alert('top_n 必须是正整数。'); return; }
    body.top_n = n;
  }
  $('#rerank-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    const r = await api('POST', '/v1/rerank', body);
    renderRerankResults(r.payload, documents);
  } catch (e) {
    $('#rerank-results').innerHTML =
      '<div class="empty" style="color: var(--danger); border-color: var(--danger);">' +
      '请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* =========================================================================
 * 图文嵌入（跨模态）
 * ========================================================================= */
async function refreshMultimodalModels() {
  let data = [];
  try {
    const r = await api('GET', '/v1/models');
    const all = (r.payload && r.payload.data) || [];
    data = all.filter(m => m.type === 'multimodal_embedder');
  } catch (e) {}
  const list = $('#mm-emb-models-list');
  list.innerHTML = '';
  if (!data.length) {
    list.innerHTML = '<div class="empty">暂无已注册的图文嵌入模型。</div>';
  } else {
    data.forEach(m => {
      const item = document.createElement('div');
      item.className = 'list-item';
      item.innerHTML =
        '<span class="name">' + escapeHtml(m.id) + '</span>' +
        '<span class="meta">' + (m.dimensions != null ? m.dimensions + ' 维' : '未加载') + '</span>';
      item.addEventListener('click', async () => {
        try { await api('GET', '/v1/models/' + enc(m.id)); } catch {}
      });
      list.appendChild(item);
    });
  }
  const sel = $('#mm-emb-model');
  const prev = sel.value;
  sel.innerHTML = '';
  if (!data.length) {
    const o = document.createElement('option');
    o.value = ''; o.textContent = '（暂无模型）';
    sel.appendChild(o);
  } else {
    data.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.id;
      sel.appendChild(opt);
    });
  }
  if (prev && Array.from(sel.options).some(o => o.value === prev)) sel.value = prev;
}

/* 图文输入项 — 每个项是 {kind: 'text'|'image', ...}，渲染成一张卡片行 */
const mmEmbItems = [];

function rerenderMmEmbItems() {
  const root = $('#mm-emb-items');
  root.innerHTML = '';
  mmEmbItems.forEach((it, idx) => {
    const card = document.createElement('div');
    card.className = 'field-card';
    const isText = it.kind === 'text';
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
              '<div data-role="info" style="font-family: var(--mono); font-size: 11px; color: var(--text-dim);">尚未选择文件。</div></div>' +
          '</div>');
    // 双向绑定
    if (isText) {
      const ta = card.querySelector('[data-role="text"]');
      ta.value = it.text || '';
      ta.addEventListener('input', () => { it.text = ta.value; });
    } else {
      const fi = card.querySelector('[data-role="file"]');
      const mi = card.querySelector('[data-role="mime"]');
      const info = card.querySelector('[data-role="info"]');
      fi.addEventListener('change', () => {
        const f = fi.files[0];
        it.file = f || null;
        if (f) info.textContent = f.name + ' · ' + (f.type || '?') + ' · ' + (f.size / 1024).toFixed(1) + ' KB';
        else    info.textContent = '尚未选择文件。';
      });
      if (it.mimeOverride) mi.value = it.mimeOverride;
      mi.addEventListener('change', () => { it.mimeOverride = mi.value; });
    }
    card.querySelector('.remove').addEventListener('click', () => {
      mmEmbItems.splice(idx, 1);
      rerenderMmEmbItems();
    });
    root.appendChild(card);
  });
  if (!mmEmbItems.length) {
    root.innerHTML = '<div class="empty">还没有任何项。点击下方按钮添加文本或图片。</div>';
  }
}

$('#btn-mm-emb-add-text').addEventListener('click', () => {
  mmEmbItems.push({ kind: 'text', text: '' });
  rerenderMmEmbItems();
});
$('#btn-mm-emb-add-image').addEventListener('click', () => {
  mmEmbItems.push({ kind: 'image', file: null, mimeOverride: '' });
  rerenderMmEmbItems();
});
$('#btn-mm-emb-refresh-models').addEventListener('click', refreshMultimodalModels);

function renderMmEmbResults(payload) {
  const root = $('#mm-emb-results');
  if (!payload || !Array.isArray(payload.data)) {
    root.innerHTML = '<div class="empty">响应中未包含 data 数组。</div>';
    return;
  }
  const head =
    '<dl class="traffic-section" style="margin-bottom: 12px;">' +
      '<dt>模型</dt><dd>' + escapeHtml(payload.model || '') + '</dd>' +
      '<dt>数量</dt><dd>' + payload.data.length + '</dd>' +
    '</dl>';
  const rows = payload.data.map(item => {
    const v = Array.isArray(item.embedding) ? item.embedding : [];
    const preview = v.slice(0, 8).map(n => (typeof n === 'number' ? n.toFixed(4) : String(n)));
    const more = v.length > 8 ? ' … (+' + (v.length - 8) + ')' : '';
    return '<div class="rerank-row">' +
      '<span class="rerank-rank">#' + (item.index + 1) + '</span>' +
      '<span class="rerank-idx">' + v.length + ' 维</span>' +
      '<span class="rerank-score">' + escapeHtml(preview.join(', ')) + more + '</span>' +
      '<span class="rerank-doc">[' + escapeHtml(preview.join(',')) + more + ']</span>' +
    '</div>';
  }).join('');
  root.innerHTML = head + rows;
}

$('#btn-mm-emb').addEventListener('click', async () => {
  const model = $('#mm-emb-model').value;
  if (!model) { alert('请先选择图文嵌入模型（点击上方"刷新"加载）。'); return; }
  if (!mmEmbItems.length) { alert('请至少添加一项输入。'); return; }

  const input = [];
  for (let i = 0; i < mmEmbItems.length; i++) {
    const it = mmEmbItems[i];
    if (it.kind === 'text') {
      const t = (it.text || '').trim();
      if (!t) { alert('第 ' + (i + 1) + ' 项文本为空。'); return; }
      input.push({ text: t });
    } else {
      if (!it.file) { alert('第 ' + (i + 1) + ' 项图片未选择。'); return; }
      let data, mime;
      try {
        ({ data, mime } = await readFileAsBase64(it.file));
      } catch (e) {
        alert('第 ' + (i + 1) + ' 项读取文件失败：' + (e && e.message || String(e)));
        return;
      }
      if (it.mimeOverride) mime = it.mimeOverride;
      if (!mime) { alert('第 ' + (i + 1) + ' 项无法识别 MIME，请在右侧下拉框手动选择。'); return; }
      input.push({ image: { data, mime } });
    }
  }

  $('#mm-emb-results').innerHTML = '<div class="empty">请求中…</div>';
  try {
    const r = await api('POST', '/v1/multimodal_embeddings', { model, input });
    renderMmEmbResults(r.payload);
  } catch (e) {
    $('#mm-emb-results').innerHTML =
      '<div class="empty" style="color: var(--danger); border-color: var(--danger);">' +
      '请求失败：' + escapeHtml(e.message || String(e)) + '</div>';
  }
});

/* =========================================================================
 * 启动
 * ========================================================================= */
refreshModels();
refreshDatabases();
</script>
</body>
</html>
"""


@router.get(
    "/playground",
    response_class=HTMLResponse,
    summary="Interactive debug UI",
    description=(
        "Self-contained Chinese-localised HTML page that exercises every "
        "public endpoint of the service. Useful for local development; "
        "not part of the machine-to-machine API surface."
    ),
    include_in_schema=False,
)
def playground_page() -> HTMLResponse:
    """Serve the interactive playground page."""
    return HTMLResponse(content=PLAYGROUND_HTML)
