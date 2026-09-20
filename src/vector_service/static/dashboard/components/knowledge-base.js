// Knowledge base panels: parse / chunk / ingest.
//
// The ingest view talks to POST /v1/ingest/stream and parse to
// POST /v1/parse/stream: an XHR carries the multipart upload (real
// byte-level upload % via xhr.upload.onprogress) while its response is
// consumed incrementally as newline-delimited JSON events —
// stage / progress then a terminal result/error event.
import { defineComponent, ref, computed, onMounted, watch } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError, t } from './app.js';
import { renderMarkdown } from './markdown.js';

// The ingest routes default to the fixed "ingest" collection that
// auto-creates on first use; the dashboard only picks the database.
const INGEST_COLLECTION = 'ingest';
// Mirrors _resolve_mime() in api/ingest.py; the server still validates
// MIME itself — accept="" only filters the file picker.
const INGEST_ACCEPT = '.pdf,.docx,.pptx,.html,.htm,.xhtml,.md,.markdown,.txt';

// Stage order is the public event contract of /v1/ingest/stream; the
// 'uploading' pseudo-stage precedes the first server stage event.
const STAGES = [
  { key: 'uploading', labelKey: 'ingest.stage.upload' },
  { key: 'parse', labelKey: 'ingest.stage.parse' },
  { key: 'chunk', labelKey: 'ingest.stage.chunk' },
  { key: 'embed', labelKey: 'ingest.stage.embed' },
  { key: 'upsert', labelKey: 'ingest.stage.upsert' },
];

function formatBytes(n) {
  if (n == null || Number.isNaN(n)) return '—';
  if (n < 1024) return n + ' B';
  const units = ['KB', 'MB', 'GB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return v.toFixed(v >= 100 ? 0 : 1) + ' ' + units[i];
}
function formatDuration(ms) {
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + ' s';
  return Math.floor(s / 60) + ' min ' + Math.round(s % 60) + ' s';
}
function formatCount(n) {
  return n == null ? '—' : Number(n).toLocaleString();
}

// Clipboard API is unavailable in non-secure contexts (plain http);
// fall back to a hidden textarea + execCommand. Returns success.
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).then(() => true).catch(() => {
      legacyCopy(text);
      return true;
    });
  }
  legacyCopy(text);
  return Promise.resolve(true);
}
function legacyCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); } catch (_) {}
  document.body.removeChild(ta);
}

export default defineComponent({
  name: 'KnowledgeBasePanel',
  props: { view: { type: String, default: 'parse' } },
  setup(props) {
    const status = ref('');
    const parseFile = ref(null);
    const parseResult = ref(null);
    const parseCopied = ref(false);
    // Result-pane toggle over the parsed markdown:
    // 'source' (raw markdown in a <pre>) | 'preview' (rendered HTML).
    const parseView = ref('source');

    // Parse run state. parseStage is 'uploading' (bytes going up) or
    // 'parse' (server-side conversion); parseProgress carries the last
    // {page,total} tick — Docling fires one per completed page.
    const parseBusy = ref(false);
    const parseStage = ref('uploading');
    const parseUploadPct = ref(0);
    const parseElapsed = ref(0);
    const parseProgress = ref(null);
    const parseError = ref(null);            // { code, message }
    let parseTimer = null;
    const chunkSize = ref(800);
    const chunkOverlap = ref(80);
    const chunkMd = ref('# title\n\nThis is sample text. Second paragraph.');
    const chunkResults = ref([]);
    const dbs = ref([]);
    const db = ref('');
    const models = ref([]);
    const model = ref('');
    const ingestFile = ref(null);
    const chunkParams = ref({ size: 800, overlap: 80 });
    const metadata = ref('{}');
    const ingestResult = ref(null);

    // Ingest run state. ingestStage is one of STAGES keys; it drives
    // the stepper; ingestFailedStage pins the red node after an error.
    const ingestBusy = ref(false);
    const ingestStage = ref('uploading');
    const ingestFailedStage = ref(null);
    // Last {page,total} tick while the ingest run is in the parse
    // stage; cleared on every stage transition past parse.
    const ingestParsePages = ref(null);
    const uploadPct = ref(0);
    const ingestElapsed = ref(0);
    const ingestError = ref(null);          // { code, message }
    const ingestMeta = ref(null);           // { database, collection, model, filename, sizeBytes, durationMs }
    const docIdCopied = ref(false);
    let ingestTimer = null;

    async function refreshDbs() {
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
        // If the selected db disappeared, fall back to the first one.
        if (db.value && !dbs.value.includes(db.value) && dbs.value.length) {
          db.value = dbs.value[0];
        }
      } catch (_e) {}
    }
    async function refreshModels() {
      try {
        const { payload } = await api('GET', '/v1/models');
        models.value = ((payload && payload.data) || []).filter(m => m.type === 'embedder');
        if (!model.value && models.value.length) {
          const loaded = models.value.find(m => m.loaded);
          model.value = (loaded || models.value[0]).id;
        }
      } catch (_e) {}
    }
    onMounted(() => { refreshDbs(); refreshModels(); });
    // The panel stays mounted under v-show, so onMounted only fires
    // once. Re-pull dbs/models whenever the user opens the tab — this
    // replaces the old manual "refresh dbs" button and also picks up
    // models loaded in another tab.
    watch(() => props.view, (v) => {
      if (v === 'ingest' && !ingestBusy.value) { refreshDbs(); refreshModels(); }
    });

    function onParseFile(ev) {
      parseFile.value = ev.target.files[0] || null;
      // A new file invalidates the previous parse result/error.
      parseResult.value = null;
      parseError.value = null;
      parseProgress.value = null;
      parseView.value = 'source';
    }

    async function doParse() {
      if (parseBusy.value) return;
      if (!parseFile.value) { alert(t('parse.err.no_file')); return; }

      const form = new FormData();
      form.append('file', parseFile.value);

      parseResult.value = null;
      parseError.value = null;
      parseProgress.value = null;
      parseView.value = 'source';
      parseBusy.value = true;
      parseStage.value = 'uploading';
      parseUploadPct.value = 0;
      parseElapsed.value = 0;
      const startedAt = performance.now();
      parseTimer = setInterval(() => {
        parseElapsed.value = (performance.now() - startedAt) / 1000;
      }, 200);

      try {
        const res = await streamPost(
          '/v1/parse/stream', form,
          (pct) => { parseUploadPct.value = pct; },
          (ev) => {
            if (ev.type === 'stage') parseStage.value = ev.stage;
            else if (ev.type === 'progress' && ev.stage === 'parse') {
              parseProgress.value = { page: ev.page, total: ev.total };
            }
          },
        );
        if (!res.ok) {
          const err = (res.payload && res.payload.error) || {};
          parseError.value = {
            code: err.code || ('HTTP ' + res.status),
            message: err.message || ('HTTP ' + res.status),
          };
        } else {
          parseResult.value = res.payload;
          status.value = 'ok';
        }
      } catch (e) {
        parseError.value = {
          code: 'network_error',
          message: extractApiError(e, t('parse.err.network')),
        };
      } finally {
        if (parseTimer) { clearInterval(parseTimer); parseTimer = null; }
        parseElapsed.value = (performance.now() - startedAt) / 1000;
        parseBusy.value = false;
      }
    }
    async function doChunk() {
      try {
        // Field names mirror ChunkRequest in api/chunk.py:
        // markdown / chunk_size / chunk_overlap.
        const { payload } = await api('POST', '/v1/chunk', {
          markdown: chunkMd.value,
          chunk_size: chunkSize.value,
          chunk_overlap: chunkOverlap.value,
        });
        chunkResults.value = (payload && payload.chunks) || [];
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }

    function onIngestFile(ev) {
      ingestFile.value = ev.target.files[0] || null;
      // A new file invalidates whatever the previous run produced.
      ingestResult.value = null;
      ingestError.value = null;
    }

    // Multipart POST to an NDJSON stream endpoint (/v1/ingest/stream
    // or /v1/parse/stream) with real upload-byte progress (fetch
    // cannot report upload progress) AND incremental newline-delimited
    // stage/progress events over the same response. Resolves to
    // { ok, status, payload } so HTTP error envelopes are handled by
    // the caller exactly like the shared api() helper. A pre-flight
    // JSON envelope (stream never started) lands via the onload
    // fallback below.
    function streamPost(url, form, onPct, onEvent) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url);
        let pos = 0;
        let buf = '';
        let settled = false;

        function finish(result) {
          if (settled) return;
          settled = true;
          resolve(result);
        }

        function handleEvent(ev) {
          if (ev.type === 'stage' || ev.type === 'progress') {
            onEvent(ev);
          } else if (ev.type === 'result') {
            finish({ ok: true, status: xhr.status, payload: ev });
          } else if (ev.type === 'error') {
            finish({
              ok: false,
              status: ev.status || xhr.status || 503,
              payload: { error: ev.error || { message: 'unknown error' } },
            });
          }
        }

        function drain(flush) {
          const full = xhr.responseText || '';
          buf += full.slice(pos);
          pos = full.length;
          let nl;
          while ((nl = buf.indexOf('\n')) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (line) handleEvent(JSON.parse(line));
          }
          if (flush && buf.trim()) handleEvent(JSON.parse(buf.trim()));
        }

        if (xhr.upload) {
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
              // Hold at 99% until upload.onload — the last progress
              // event can precede the request being fully sent.
              onPct(Math.min(99, Math.round((e.loaded / e.total) * 100)));
            }
          };
          xhr.upload.onload = () => onPct(100);
        }
        xhr.onprogress = () => {
          try { drain(false); }
          catch (e) {
            finish({ ok: false, status: 0,
              payload: { error: { code: 'bad_stream', message: String(e.message || e) } } });
          }
        };
        xhr.onload = () => {
          try { drain(true); }
          catch (e) {
            finish({ ok: false, status: 0,
              payload: { error: { code: 'bad_stream', message: String(e.message || e) } } });
            return;
          }
          if (settled) return;
          // No terminal event arrived: a pre-flight JSON envelope (the
          // stream never started) or an unexpected response.
          let payload = null;
          try { payload = JSON.parse(xhr.responseText); } catch (_e) { payload = null; }
          finish({
            ok: xhr.status >= 200 && xhr.status < 300,
            status: xhr.status,
            payload,
          });
        };
        xhr.onerror = () => {
          if (!settled) { settled = true; reject(new Error(t('ingest.err.network'))); }
        };
        xhr.send(form);
      });
    }

    // Text shown in the parse pane / copied / downloaded: the markdown
    // when the parser produced one, otherwise the raw JSON payload.
    function parseOutput() {
      if (!parseResult.value) return '';
      return parseResult.value.markdown
        || JSON.stringify(parseResult.value, null, 2);
    }

    // Rendered HTML for the "preview" tab; renderMarkdown escapes all
    // input, so the result is safe to bind via v-html.
    const renderedMarkdown = computed(() => renderMarkdown(parseOutput()));

    async function copyParseMarkdown() {
      if (!parseResult.value) return;
      try {
        await copyText(parseOutput());
      } catch (_e) { return; }
      parseCopied.value = true;
      setTimeout(() => { parseCopied.value = false; }, 1600);
    }

    function downloadParseMarkdown() {
      if (!parseResult.value) return;
      // report.pdf -> report.md; fall back when no file name is known.
      let name = (parseFile.value && parseFile.value.name) || 'parsed.md';
      name = name.replace(/\.[^.]*$/, '') + '.md';
      const blob = new Blob([parseOutput()], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    async function copyDocId() {
      if (!ingestResult.value || !ingestResult.value.doc_id) return;
      try {
        await copyText(ingestResult.value.doc_id);
      } catch (_e) { return; }
      docIdCopied.value = true;
      setTimeout(() => { docIdCopied.value = false; }, 1600);
    }

    function stageState(key) {
      const cur = STAGES.findIndex(s => s.key === ingestStage.value);
      const idx = STAGES.findIndex(s => s.key === key);
      if (ingestFailedStage.value) {
        if (key === ingestFailedStage.value) return 'failed';
        if (idx < cur) return 'done';
        return 'pending';
      }
      if (idx < cur) return 'done';
      if (idx === cur) return ingestBusy.value ? 'active' : 'pending';
      return 'pending';
    }
    function stageLabelKey(key) {
      const s = STAGES.find(x => x.key === key);
      return s ? s.labelKey : '';
    }
    function stageHintKey() {
      if (ingestStage.value === 'uploading') return '';
      return 'ingest.stage_hint.' + ingestStage.value;
    }

    async function doIngest() {
      if (ingestBusy.value) return;
      if (!db.value) { alert(t('ingest.err.no_db')); return; }
      if (!ingestFile.value) { alert(t('ingest.err.no_file')); return; }

      const size = Number(chunkParams.value.size);
      const overlap = Number(chunkParams.value.overlap);
      if (!Number.isFinite(size) || size < 1 || size > 8192) {
        ingestError.value = { code: 'invalid_chunk_size', message: t('ingest.err.bad_size') };
        ingestResult.value = null;
        return;
      }
      if (!Number.isFinite(overlap) || overlap < 0 || overlap >= size) {
        ingestError.value = { code: 'invalid_chunk_overlap', message: t('ingest.err.bad_overlap') };
        ingestResult.value = null;
        return;
      }

      let meta = {};
      const metaText = metadata.value.trim();
      if (metaText) {
        try {
          meta = JSON.parse(metaText);
        } catch (_e) {
          ingestError.value = { code: 'invalid_metadata', message: t('ingest.err.bad_meta') };
          ingestResult.value = null;
          return;
        }
        if (typeof meta !== 'object' || meta === null || Array.isArray(meta)) {
          ingestError.value = { code: 'invalid_metadata', message: t('ingest.err.bad_meta') };
          ingestResult.value = null;
          return;
        }
      }

      // Field names mirror the multipart contract in api/ingest.py /
      // docs/ingest-pipeline.md: embed_model, chunk_size, chunk_overlap,
      // metadata as a JSON string.
      const form = new FormData();
      form.append('file', ingestFile.value);
      form.append('database', db.value);
      form.append('collection', INGEST_COLLECTION);
      form.append('embed_model', model.value || '');
      form.append('chunk_size', String(size));
      form.append('chunk_overlap', String(overlap));
      form.append('metadata', JSON.stringify(meta));

      ingestResult.value = null;
      ingestError.value = null;
      ingestFailedStage.value = null;
      ingestParsePages.value = null;
      ingestBusy.value = true;
      ingestStage.value = 'uploading';
      uploadPct.value = 0;
      ingestElapsed.value = 0;
      docIdCopied.value = false;
      const startedAt = performance.now();
      ingestTimer = setInterval(() => {
        ingestElapsed.value = (performance.now() - startedAt) / 1000;
      }, 200);

      try {
        const res = await streamPost(
          '/v1/ingest/stream', form,
          (pct) => { uploadPct.value = pct; },
          (ev) => {
            if (ev.type === 'stage') {
              ingestStage.value = ev.stage;
              // Page ticks only belong to the parse stage.
              if (ev.stage !== 'parse') ingestParsePages.value = null;
            } else if (ev.type === 'progress' && ev.stage === 'parse') {
              ingestParsePages.value = { page: ev.page, total: ev.total };
            }
          },
        );
        const durationMs = performance.now() - startedAt;
        if (!res.ok) {
          const err = (res.payload && res.payload.error) || {};
          ingestFailedStage.value = ingestStage.value;
          ingestError.value = {
            code: err.code || ('HTTP ' + res.status),
            message: err.message || ('HTTP ' + res.status),
          };
        } else {
          ingestResult.value = res.payload || {};
          ingestMeta.value = {
            database: db.value,
            collection: INGEST_COLLECTION,
            model: model.value,
            filename: ingestFile.value.name,
            sizeBytes: ingestFile.value.size,
            durationMs,
          };
        }
      } catch (e) {
        ingestFailedStage.value = ingestStage.value;
        ingestError.value = {
          code: 'network_error',
          message: extractApiError(e, t('ingest.err.network')),
        };
      } finally {
        if (ingestTimer) { clearInterval(ingestTimer); ingestTimer = null; }
        ingestElapsed.value = (performance.now() - startedAt) / 1000;
        ingestBusy.value = false;
      }
    }

    return { status, parseFile, parseResult, parseCopied, chunkSize, chunkOverlap, chunkMd, chunkResults,
             dbs, db, models, model, ingestFile, chunkParams, metadata, ingestResult,
             ingestBusy, ingestStage, ingestFailedStage, ingestParsePages, uploadPct, ingestElapsed,
             ingestError, ingestMeta, docIdCopied,
             parseView, renderedMarkdown,
             parseBusy, parseStage, parseUploadPct, parseElapsed, parseProgress, parseError,
             doParse, doChunk, doIngest, onIngestFile, copyDocId,
             onParseFile,
             copyParseMarkdown, downloadParseMarkdown,
             stageState, stageLabelKey, stageHintKey,
             formatBytes, formatDuration, formatCount,
             STAGES, INGEST_ACCEPT };
  },
  template: `
    <div>
      <div class="section" v-show="view === 'parse'">
        <div class="section-head"><h3 class="section-title">parse <span class="pill accent">POST /v1/parse/stream</span></h3></div>
        <div class="row"><label>file</label>
          <input type="file" id="parse-file" :accept="INGEST_ACCEPT" :disabled="parseBusy" @change="onParseFile" />
          <div v-if="parseFile" class="ingest-file-meta">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
              <path d="M3 1.5h6L13 5v9.5H3z"/><path d="M9 1.5V5h4"/>
            </svg>
            <span class="ingest-file-name">{{ parseFile.name }}</span>
            <span>· {{ formatBytes(parseFile.size) }}</span>
          </div>
        </div>
        <div class="actions">
          <button class="btn primary" id="btn-parse" :disabled="parseBusy" @click="doParse">
            <span v-if="parseBusy" class="btn-spinner"></span>
            {{ parseBusy
               ? (parseStage === 'uploading' ? $t('parse.uploading') : $t('parse.parsing'))
               : $t('parse.upload') }}
          </button>
        </div>

        <!-- In-flight: real upload byte % then per-page parse progress. -->
        <div v-if="parseBusy" class="response ingest-progress" id="parse-progress">
          <div class="ingest-progress-head">
            <span class="spinner"></span>
            <span class="ingest-phase-label">{{ parseStage === 'uploading' ? $t('parse.uploading') : $t('parse.parsing') }}</span>
            <span class="ingest-elapsed">{{ parseElapsed.toFixed(1) }} s</span>
          </div>
          <div class="bar">
            <span v-if="parseStage === 'uploading'" :style="{ width: parseUploadPct + '%' }"></span>
            <span v-else-if="parseProgress && parseProgress.total"
                  :style="{ width: Math.round((parseProgress.page / parseProgress.total) * 100) + '%' }"></span>
            <span v-else class="indeterminate"></span>
          </div>
          <div class="ingest-progress-sub">
            <template v-if="parseStage === 'uploading'">
              <span class="ingest-file-name">{{ parseFile ? parseFile.name : '' }}</span>
              · {{ parseUploadPct }}%
            </template>
            <template v-else-if="parseProgress && parseProgress.total">
              {{ $t('parse.pages_prefix') }}{{ parseProgress.page }}/{{ parseProgress.total }}{{ $t('parse.pages_suffix') }}
            </template>
            <template v-else>{{ $t('parse.hint.parse') }}</template>
          </div>
        </div>

        <!-- Terminal failure: code pill + server message. -->
        <div v-else-if="parseError" class="response ingest-result-err" id="parse-error">
          <div class="result-banner err">
            <span class="result-glyph err">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 4l8 8M12 4l-8 8"/>
              </svg>
            </span>
            <div class="result-banner-text">
              <div class="result-title">{{ $t('parse.failed') }}</div>
              <div class="result-sub">{{ parseError.message }}</div>
            </div>
            <span class="pill danger">{{ parseError.code }}</span>
          </div>
        </div>

        <div v-if="parseResult" class="response" id="parse-result">
          <div class="response-head">
            <div class="seg-toggle" role="tablist">
              <button type="button" class="btn sm" id="btn-parse-source"
                      :class="{ primary: parseView === 'source' }"
                      @click="parseView = 'source'">source</button>
              <button type="button" class="btn sm" id="btn-parse-preview"
                      :class="{ primary: parseView === 'preview' }"
                      @click="parseView = 'preview'">preview</button>
            </div>
            <span class="head-actions">
              <button class="btn sm" id="btn-parse-copy" @click="copyParseMarkdown">
                {{ parseCopied ? 'copied' : 'copy' }}
              </button>
              <button class="btn sm" id="btn-parse-download" @click="downloadParseMarkdown">
                download .md
              </button>
            </span>
          </div>
          <pre v-if="parseView === 'source'" class="code-pane" id="parse-markdown">{{ parseResult.markdown || JSON.stringify(parseResult, null, 2) }}</pre>
          <!-- renderedMarkdown is HTML-escaped by markdown.js before v-html -->
          <div v-else class="md-preview" id="parse-markdown-preview" v-html="renderedMarkdown"></div>
        </div>
      </div>

      <div class="section" v-show="view === 'chunk'">
        <div class="section-head"><h3 class="section-title">chunk <span class="pill accent">POST /v1/chunk</span></h3></div>
        <div class="row split">
          <div class="row"><label>chunk size</label><input type="number" id="chunk-size" v-model.number="chunkSize" /></div>
          <div class="row"><label>overlap</label><input type="number" id="chunk-overlap" v-model.number="chunkOverlap" /></div>
        </div>
        <div class="row"><label>markdown text</label><textarea id="chunk-md" rows="6" v-model="chunkMd"></textarea></div>
        <div class="actions"><button class="btn primary" id="btn-chunk" @click="doChunk">chunk</button></div>
        <div v-if="chunkResults.length" class="response" id="chunk-results">
          <div class="response-head"><span>chunks: {{ chunkResults.length }}</span></div>
          <div v-for="(c, i) in chunkResults" :key="i" class="field-card">
            <div class="field-card-header">
              <span class="index-badge">#{{ i + 1 }}</span>
              <span class="title">{{ c.text.length }} chars · {{ c.token_count }} tokens</span>
              <span v-if="c.section_header" class="hint">{{ c.section_header }}</span>
              <span v-if="c.page_number != null" class="hint">p.{{ c.page_number }}</span>
            </div>
            <pre class="code-pane">{{ c.text }}</pre>
          </div>
        </div>
      </div>

      <div class="section" v-show="view === 'ingest'">
        <div class="section-head"><h3 class="section-title">ingest <span class="pill accent">POST /v1/ingest/stream</span></h3></div>
        <div class="row split">
          <div class="row"><label>database</label>
            <select id="ingest-db" v-model="db" :disabled="ingestBusy">
              <option v-if="!dbs.length" value="" disabled>{{ $t('ingest.no_dbs') }}</option>
              <option v-for="d in dbs" :key="d" :value="d">{{ d }}</option>
            </select>
          </div>
          <div class="row"><label>embed model</label>
            <select id="ingest-model" v-model="model" :disabled="ingestBusy">
              <option v-for="mm in models" :key="mm.id" :value="mm.id">
                {{ mm.id }}{{ mm.loaded ? ' · ' + $t('common.loaded') : '' }}
              </option>
            </select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>chunk size</label><input type="number" id="ingest-size" v-model.number="chunkParams.size" :disabled="ingestBusy" /></div>
          <div class="row"><label>overlap</label><input type="number" id="ingest-overlap" v-model.number="chunkParams.overlap" :disabled="ingestBusy" /></div>
        </div>
        <div class="row"><label>file</label>
          <input type="file" id="ingest-file" :accept="INGEST_ACCEPT" :disabled="ingestBusy" @change="onIngestFile" />
          <div v-if="ingestFile" class="ingest-file-meta">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
              <path d="M3 1.5h6L13 5v9.5H3z"/><path d="M9 1.5V5h4"/>
            </svg>
            <span class="ingest-file-name">{{ ingestFile.name }}</span>
            <span>· {{ formatBytes(ingestFile.size) }}</span>
          </div>
        </div>
        <div class="row"><label>metadata (JSON) <span class="hint">{{ $t('ingest.metadata_hint') }}</span></label>
          <textarea id="ingest-metadata" rows="2" v-model="metadata" :disabled="ingestBusy">{}</textarea>
        </div>
        <div class="actions">
          <button class="btn primary" id="btn-ingest-upload" :disabled="ingestBusy" @click="doIngest">
            <span v-if="ingestBusy" class="btn-spinner"></span>
            {{ ingestBusy
               ? (ingestStage === 'uploading' ? $t('ingest.uploading') : $t('ingest.processing'))
               : $t('ingest.upload') }}
          </button>
        </div>

        <!-- In-flight: 5-stage stepper. Upload has a real byte %, the
             parse stage has a real per-page % (progress events); the
             later stages move as stage events arrive with an
             indeterminate bar. -->
        <div v-if="ingestBusy" class="response ingest-progress" id="ingest-progress">
          <div class="ingest-progress-head">
            <span class="spinner"></span>
            <span class="ingest-phase-label">{{ $t(stageLabelKey(ingestStage)) }}</span>
            <span class="ingest-elapsed">{{ ingestElapsed.toFixed(1) }} s</span>
          </div>
          <ol class="ingest-stages">
            <li v-for="(s, i) in STAGES" :key="s.key"
                :class="'is-' + stageState(s.key)" :data-ingest-stage="s.key">
              <span class="stage-dot">
                <svg v-if="stageState(s.key) === 'done'" class="stage-ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2">
                  <path d="M3 8.5l3.2 3.2L13 5"/>
                </svg>
                <span v-else-if="stageState(s.key) === 'active'" class="stage-spin"></span>
                <span v-else-if="stageState(s.key) === 'failed'" class="stage-x">×</span>
                <span v-else class="stage-num">{{ i + 1 }}</span>
              </span>
              <span class="stage-label">{{ $t(s.labelKey) }}</span>
              <span v-if="s.key === 'uploading' && ingestStage === 'uploading'" class="stage-pct">{{ uploadPct }}%</span>
              <span v-else-if="s.key === 'parse' && ingestStage === 'parse' && ingestParsePages && ingestParsePages.total"
                    class="stage-pct" data-parse-pages="true">
                {{ ingestParsePages.page }}/{{ ingestParsePages.total }}
              </span>
            </li>
          </ol>
          <div class="bar">
            <span v-if="ingestStage === 'uploading'" :style="{ width: uploadPct + '%' }"></span>
            <span v-else-if="ingestStage === 'parse' && ingestParsePages && ingestParsePages.total"
                  :style="{ width: Math.round((ingestParsePages.page / ingestParsePages.total) * 100) + '%' }"></span>
            <span v-else class="indeterminate"></span>
          </div>
          <div class="ingest-progress-sub">
            <template v-if="ingestStage === 'uploading'">
              <span class="ingest-file-name">{{ ingestFile ? ingestFile.name : '' }}</span>
              · {{ uploadPct }}%
            </template>
            <template v-else-if="ingestStage === 'parse' && ingestParsePages && ingestParsePages.total">
              {{ $t('ingest.parse_pages_prefix') }}{{ ingestParsePages.page }}/{{ ingestParsePages.total }}{{ $t('ingest.parse_pages_suffix') }}
            </template>
            <template v-else>{{ $t(stageHintKey()) }}</template>
          </div>
        </div>

        <!-- Terminal failure: code pill + server message + stage. -->
        <div v-else-if="ingestError" class="response ingest-result-err" id="ingest-error">
          <div class="result-banner err">
            <span class="result-glyph err">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 4l8 8M12 4l-8 8"/>
              </svg>
            </span>
            <div class="result-banner-text">
              <div class="result-title">{{ $t('ingest.failed') }}</div>
              <div class="result-sub">{{ ingestError.message }}</div>
              <div v-if="ingestFailedStage" class="result-stage-at">
                {{ $t('ingest.failed_at') }}{{ $t(stageLabelKey(ingestFailedStage)) }}
              </div>
            </div>
            <span class="pill danger">{{ ingestError.code }}</span>
          </div>
        </div>

        <!-- Terminal success: banner + stat grid + copyable doc_id;
             the raw result event stays behind a collapsible. -->
        <div v-else-if="ingestResult" class="response ingest-result-ok" id="ingest-result">
          <div class="result-banner" :class="ingestResult.chunk_count ? 'ok' : 'warn'">
            <span class="result-glyph" :class="ingestResult.chunk_count ? 'ok' : 'warn'">
              <svg v-if="ingestResult.chunk_count" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 8.5l3.2 3.2L13 5"/>
              </svg>
              <span v-else>!</span>
            </span>
            <div class="result-banner-text">
              <div class="result-title">
                {{ ingestResult.chunk_count ? $t('ingest.success') : $t('ingest.success_empty') }}
              </div>
              <div class="result-sub">
                <span class="ingest-file-name">{{ ingestMeta.filename }}</span>
                → {{ ingestMeta.database }}/{{ ingestMeta.collection }}
                · {{ formatDuration(ingestMeta.durationMs) }}
              </div>
            </div>
            <span class="pill" :class="ingestResult.chunk_count ? 'success' : 'warn'">
              {{ ingestResult.chunk_count ? $t('ingest.banner_ok') : $t('ingest.banner_empty') }}
            </span>
          </div>

          <div class="kb-result">
            <div class="stat"><span class="key">{{ $t('ingest.stat.chunks') }}</span><span class="val accent">{{ formatCount(ingestResult.chunk_count) }}</span></div>
            <div class="stat"><span class="key">{{ $t('ingest.stat.pages') }}</span><span class="val">{{ formatCount(ingestResult.page_count) }}</span></div>
            <div class="stat"><span class="key">{{ $t('ingest.stat.tokens') }}</span><span class="val">{{ formatCount(ingestResult.tokens_used) }}</span></div>
            <div class="stat"><span class="key">{{ $t('ingest.stat.duration') }}</span><span class="val ingest-stat-sm">{{ formatDuration(ingestMeta.durationMs) }}</span></div>
            <div class="stat"><span class="key">{{ $t('ingest.stat.size') }}</span><span class="val ingest-stat-sm">{{ formatBytes(ingestMeta.sizeBytes) }}</span></div>
            <div class="stat"><span class="key">{{ $t('ingest.stat.model') }}</span><span class="val ingest-stat-xs">{{ ingestMeta.model || '—' }}</span></div>
          </div>

          <div class="ingest-docid">
            <span class="ingest-docid-key">{{ $t('ingest.doc_id') }}</span>
            <code class="ingest-docid-val" :title="ingestResult.doc_id">{{ ingestResult.doc_id }}</code>
            <button class="btn sm" id="btn-ingest-copy-docid" @click="copyDocId">
              {{ docIdCopied ? $t('ingest.copied') : $t('ingest.copy') }}
            </button>
          </div>

          <details class="collapsible">
            <summary>{{ $t('ingest.raw') }}</summary>
            <div class="body"><pre class="code-pane">{{ JSON.stringify(ingestResult, null, 2) }}</pre></div>
          </details>
        </div>
      </div>
      <div v-if="status" class="empty hint">status: {{ status }}</div>
    </div>
  `,
});
