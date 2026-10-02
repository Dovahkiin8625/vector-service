// Knowledge base panels: parse / chunk / ingest.
//
// Parse talks to POST /v1/parse/stream: an XHR carries the multipart
// upload (real byte-level upload % via xhr.upload.onprogress) while
// its response is consumed incrementally as newline-delimited JSON
// events — stage / progress then a terminal result/error event.
// Ingest goes through the async job queue instead: the same kind of
// XHR submits POST /v1/jobs/ingest (202 + job_id), the run is then
// followed over GET /v1/jobs/{id}/events (SSE `job` frames carrying
// full JobStatus snapshots) and cancelled via POST /v1/jobs/{id}/cancel.
import { defineComponent, ref, computed, onMounted, onUnmounted, watch } from '../vue.esm-browser.prod.js';
import { api, enc, extractApiError, t, modelsByType, refreshModels as loadModels } from './app.js';
import { intError, copyText } from './util.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';
import { renderMarkdown } from './markdown.js';
import ParserProfileCards from './parser-cards.js';
import UiPager from './pager.js';

// The ingest routes default to the fixed "ingest" collection that
// auto-creates on first use; the dashboard only picks the database.
const INGEST_COLLECTION = 'ingest';

// Scalar fields projected by the ingested-chunks browser. Mirrors the
// schema fields created in api/ingest.py (_ingest_scalar_fields); the
// vector field stays out by design.
const CHUNK_FIELDS = [
  'doc_id', 'chunk_index', 'text', 'section_header', 'page_number',
  'title', 'author', 'page_count', 'filename', 'token_count',
];
// Mirrors _resolve_mime() in api/ingest.py; the server still validates
// MIME itself — accept="" only filters the file picker.
const INGEST_ACCEPT = '.pdf,.docx,.pptx,.html,.htm,.xhtml,.jpg,.jpeg,.png,.tif,.tiff,.webp,.bmp,.md,.markdown,.txt';

// A parse/ingest stream that produces no bytes and no events for this
// long is treated as dead and aborted (see the watchdog in streamPost).
// Generous: a big document can sit in one server-side stage for a while.
const STREAM_IDLE_MS = 120000;

// Values match the profile form field validated by api/parse.py
// (PARSE_PROFILES); labels come from i18n profile.* keys.
const PROFILES = [
  { value: 'auto', labelKey: 'profile.auto' },
  { value: 'standard', labelKey: 'profile.standard' },
  { value: 'native', labelKey: 'profile.native' },
  { value: 'vlm', labelKey: 'profile.vlm' },
];

// Values match the strategy names registered by the chunking package
// (vector_service/chunking/); labels come from i18n chunk.strategy.*
// keys. Order here is the dropdown order; 'recursive' stays the
// default for backwards compatibility.
const STRATEGIES = [
  { value: 'fixed', labelKey: 'chunk.strategy.fixed' },
  { value: 'paragraph', labelKey: 'chunk.strategy.paragraph' },
  { value: 'recursive', labelKey: 'chunk.strategy.recursive' },
  { value: 'semantic', labelKey: 'chunk.strategy.semantic' },
  { value: 'llm', labelKey: 'chunk.strategy.llm' },
];

// Stage order is the public stepper contract. The job queue reports
// JobStatus.status words, mapped onto these keys by JOB_STAGE below;
// 'uploading' is the client-side pseudo-stage covering the multipart
// submit before the job row exists.
const STAGES = [
  { key: 'uploading', labelKey: 'ingest.stage.upload' },
  { key: 'parse', labelKey: 'ingest.stage.parse' },
  { key: 'chunk', labelKey: 'ingest.stage.chunk' },
  { key: 'embed', labelKey: 'ingest.stage.embed' },
  { key: 'upsert', labelKey: 'ingest.stage.upsert' },
];

// JobStatus.status → STAGES key. 'queued' is deliberately absent: the
// job is accepted but the worker has not started, so the stepper stays
// on the uploading node at 100%.
const JOB_STAGE = {
  parsing: 'parse',
  chunking: 'chunk',
  embedding: 'embed',
  upserting: 'upsert',
};

// Job statuses that will never transition again.
const JOB_DONE = new Set(['done', 'failed', 'cancelled']);

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
  const sec = t('common.seconds');
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + ' ' + sec;
  return Math.floor(s / 60) + ' ' + t('common.minutes') + ' '
    + Math.round(s % 60) + ' ' + sec;
}
function formatCount(n) {
  return n == null ? '—' : Number(n).toLocaleString();
}

export default defineComponent({
  name: 'KnowledgeBasePanel',
  components: { StatusBanner, BusyButton, EmptyState, ParserProfileCards, UiPager },
  props: { view: { type: String, default: 'parse' } },
  setup(props) {
    // Run state of the last parse / chunk action. `status` is an internal
    // value ('ok' / 'error'); only the localized rendering leaves the panel.
    const status = ref('');
    const statusMsg = ref('');
    // Form-level complaint (no file picked, no database selected). These
    // used to be native alert() calls — a blocking dialog for something
    // the panel can say inline next to the button that needs fixing.
    const formErr = ref('');
    // Selection load failures, previously swallowed: the db/model
    // dropdowns just came up empty with no explanation.
    const dbsErr = ref('');
    const modelsErr = ref('');
    // Same for the Docling engine status: a failed /v1/system/status used
    // to leave the warm/evict cards silently showing stale (or no) data.
    const parserErr = ref('');
    // The chunk submit had no in-flight state at all, so a slow chunk
    // run looked like a dead button and could be double-submitted.
    const chunkBusy = ref(false);
    // Replaces the old "last op: OK" footer line.
    const statusBannerKind = computed(() => (status.value === 'error' ? 'error' : 'success'));
    const statusBannerText = computed(() => {
      if (status.value === 'error') return statusMsg.value;
      if (status.value === 'ok') return statusMsg.value || t('common.status.ok');
      return '';
    });
    // Docling per-profile status (warm/evict cards); pulled from the
    // same aggregate payload as the overview.
    const parserStatus = ref(null);
    async function refreshParserStatus() {
      parserErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/system/status');
        if (payload) parserStatus.value = payload.parser;
      } catch (e) {
        parserErr.value = t('kb.parser_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    const parseFile = ref(null);
    const parseResult = ref(null);
    const parseCopy = ref('');      // '' | 'ok' | 'err' — flashes on the button
    // Result-pane toggle over the parsed markdown:
    // 'source' (raw markdown in a <pre>) | 'preview' (rendered HTML).
    const parseView = ref('source');
    const parseProfile = ref('auto');

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
    // Live socket of the in-flight run, so "cancel" can abort it. The
    // two views can't run at once, but each keeps its own handle.
    let parseXhr = null;
    // Ingest is job-queue backed: ingestXhr only covers the multipart
    // submit; once the job exists, cancel goes through the jobs API and
    // ingestEs is the SSE subscription delivering JobStatus snapshots.
    let ingestXhr = null;
    let ingestJobId = '';
    let ingestEs = null;
    const chunkSize = ref(800);
    const chunkOverlap = ref(80);
    const chunkStrategy = ref('recursive');
    // Semantic: percentile of adjacent-distance distribution at which
    // a cut is made (95 = only the 5% biggest distances).
    const chunkPercentile = ref(95);
    const chunkAddContext = ref(false);
    const chunkMd = ref('# title\n\nThis is sample text. Second paragraph.');
    const chunkResults = ref([]);
    const dbs = ref([]);
    const db = ref('');
    // S6: shared store cache filtered to embedders — no private GET copy.
    const models = computed(() => modelsByType('embedder'));
    const model = ref('');
    const ingestFile = ref(null);
    const chunkParams = ref({
      size: 800, overlap: 80, strategy: 'recursive',
      percentile: 95, addContext: false,
    });
    const metadata = ref('{}');
    const ingestProfile = ref('auto');
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
    const docIdCopy = ref('');      // '' | 'ok' | 'err' — flashes on the button
    let ingestTimer = null;

    // Ingested-chunks browser state (view === 'ingested'). The rows
    // endpoint is the same one the "Browse" tab uses, but the
    // collection is fixed to INGEST_COLLECTION and output fields are
    // pinned to the chunk schema; cards are re-sorted client-side by
    // doc_id then chunk_index because Milvus query order is unspecified.
    // The list is leaf-first: retrieval only ever searches level=chunk
    // rows, so that is what the browser pages over by default. The
    // hierarchy parents (section / document) stay reachable one level
    // away via the level selector — never mixed into the leaf count.
    const viewChunks = ref([]);
    const viewTotal = ref(0);
    const viewOffset = ref(0);
    const viewPageSize = ref(10);
    const viewDocId = ref('');        // exact doc_id == match
    const viewNameKw = ref('');       // filename like %keyword%
    const viewLevel = ref('chunk');   // 'chunk' | 'section' | 'document'
    const viewBusy = ref(false);
    const viewError = ref('');
    // Set when the server reports the ingest collection missing: the
    // empty state then says "ingest a document first" instead of
    // showing a raw 404.
    const viewNoColl = ref(false);
    // Leaf-detail dialog: opened by clicking a card, filled from
    // GET .../chunks/{id} (content + parent chain + index registry).
    const detailChunk = ref(null);    // ChunkDetailResponse payload
    const detailBusy = ref(false);
    const detailError = ref('');
    // Which group header's copy button last flashed, and whether the
    // clipboard actually took (plain http has no Clipboard API).
    const chunkDocCopy = ref({ id: '', ok: false });
    // Per-card "show full text" toggles for the 分片测试 results list,
    // which has no detail dialog (keyed 'chunk:N'). 入库浏览 cards
    // don't clamp-or-expand: they render a fixed excerpt and the full
    // body lives in the leaf detail dialog (chunkTextPreview below).
    const textExpanded = ref({});
    // Body view for the ingested cards, mirroring the parse page's
    // source/preview switch: raw markdown vs rendered output.
    const chunkView = ref('source');

    // Per-group fold state for the by-doc grouping below, keyed by
    // doc_id. Open by default: the page's job is showing the chunks;
    // folding is for when several docs share a page and the operator
    // wants to collapse the ones they are done with.
    const docGroupsOpen = ref({});
    function toggleDocGroup(id) {
      // Negate the effective state, not the raw map slot: an untouched
      // group reads as open (undefined), so `!undefined` would land on
      // "open" again and swallow the first click.
      docGroupsOpen.value = { ...docGroupsOpen.value, [id]: !docGroupOpen(id) };
    }
    // viewChunks is already sorted doc_id → chunk_index, so grouping is
    // one pass keeping order. The group header is the only place that
    // repeats the doc_id (was a row on every card).
    const chunkGroups = computed(() => {
      const groups = [];
      const index = new Map();
      for (const it of viewChunks.value) {
        const f = (it && it.fields) || {};
        const id = f.doc_id || '';
        let g = index.get(id);
        if (!g) {
          g = { docId: id, filename: f.filename || '', items: [] };
          index.set(id, g);
          groups.push(g);
        }
        if (!g.filename && f.filename) g.filename = f.filename;
        g.items.push(it);
      }
      return groups;
    });
    function docGroupOpen(id) { return docGroupsOpen.value[id] !== false; }
    // Rendered HTML per chunk body for the preview tab; renderMarkdown
    // escapes all input, so the values are safe to bind via v-html.
    // The list renders the same excerpt as the source tab — full text
    // is the detail dialog's job.
    const chunkMdHtml = computed(() => {
      const out = new Map();
      for (const it of viewChunks.value) {
        out.set(it.id, renderMarkdown(chunkTextPreview((it.fields && it.fields.text) || '')));
      }
      return out;
    });

    function escapeEq(s) {
      return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    }
    function escapeLike(s) {
      // Backslash first; % and _ are LIKE wildcards in Milvus.
      return s.replace(/\\/g, '\\\\')
        .replace(/%/g, '\\%').replace(/_/g, '\\_')
        .replace(/"/g, '\\"');
    }
    function buildChunkFilter() {
      const parts = [];
      const d = viewDocId.value.trim();
      const n = viewNameKw.value.trim();
      if (d) parts.push('doc_id == "' + escapeEq(d) + '"');
      if (n) parts.push('filename like "%' + escapeLike(n) + '%"');
      return parts.join(' and ');
    }

    async function loadChunks() {
      if (!db.value) { viewChunks.value = []; viewTotal.value = 0; return; }
      viewBusy.value = true;
      viewError.value = '';
      viewNoColl.value = false;
      const body = {
        primary_field: 'id',
        limit: viewPageSize.value,
        offset: viewOffset.value,
        output_fields: CHUNK_FIELDS,
        level: viewLevel.value,
      };
      const filterExpr = buildChunkFilter();
      if (filterExpr) body.filter_expr = filterExpr;
      try {
        const { payload } = await api(
          'POST',
          '/v1/databases/' + enc(db.value)
            + '/collections/' + INGEST_COLLECTION + '/rows',
          body,
        );
        const items = (payload && payload.items) || [];
        items.sort((a, b) => {
          const da = (a.fields && a.fields.doc_id) || '';
          const dbn = (b.fields && b.fields.doc_id) || '';
          if (da < dbn) return -1;
          if (da > dbn) return 1;
          return (Number(a.fields && a.fields.chunk_index) || 0)
            - (Number(b.fields && b.fields.chunk_index) || 0);
        });
        viewChunks.value = items;
        viewTotal.value = (payload && payload.total) || 0;
      } catch (e) {
        const msg = extractApiError(e, 'unknown');
        viewChunks.value = [];
        viewTotal.value = 0;
        if (/collection/i.test(msg)) viewNoColl.value = true;
        else viewError.value = msg;
      } finally {
        viewBusy.value = false;
      }
    }

    function chunksQuery() { viewOffset.value = 0; loadChunks(); }
    function chunksReset() {
      viewDocId.value = '';
      viewNameKw.value = '';
      viewLevel.value = 'chunk';
      viewOffset.value = 0;
      loadChunks();
    }
    const viewPage = computed(() =>
      Math.floor(viewOffset.value / viewPageSize.value) + 1);
    const viewPages = computed(() =>
      Math.max(1, Math.ceil(viewTotal.value / viewPageSize.value)));
    function chunksFirst() { viewOffset.value = 0; loadChunks(); }
    function chunksPrev() {
      viewOffset.value = Math.max(0, viewOffset.value - viewPageSize.value);
      loadChunks();
    }
    function chunksNext() {
      viewOffset.value = Math.min(
        viewOffset.value + viewPageSize.value,
        (viewPages.value - 1) * viewPageSize.value,
      );
      loadChunks();
    }
    function chunksLast() {
      viewOffset.value = (viewPages.value - 1) * viewPageSize.value;
      loadChunks();
    }
    async function copyChunkDoc(id) {
      const ok = await copyText(id);
      chunkDocCopy.value = { id, ok };
      setTimeout(() => {
        if (chunkDocCopy.value.id === id) chunkDocCopy.value = { id: '', ok: false };
      }, 1600);
    }

    // Long enough to clamp initially: ~20 wrapped lines at the pane's
    // 320px clamp height, or an early-fade guard by raw length.
    function chunkTextLong(s) {
      s = s || '';
      return s.length > 900 || s.split('\n').length > 22;
    }
    function chunkTextClamped(key, s) {
      return chunkTextLong(s) && !textExpanded.value[key];
    }
    function toggleChunkText(key) {
      textExpanded.value = { ...textExpanded.value, [key]: !textExpanded.value[key] };
    }
    // 入库浏览 list cards show only a fixed-size excerpt of each body —
    // enough to recognise the chunk, not enough to scroll the list.
    // The full body is one click away in the leaf detail dialog.
    const CHUNK_EXCERPT_CHARS = 220;
    function chunkTextPreview(s) {
      s = s || '';
      if (s.length <= CHUNK_EXCERPT_CHARS) return s;
      // Prefer cutting at the last newline so the excerpt keeps whole
      // lines; fall back to a hard slice for single-line bodies.
      const head = s.slice(0, CHUNK_EXCERPT_CHARS);
      const nl = head.lastIndexOf('\n');
      return (nl > 60 ? head.slice(0, nl) : head).replace(/\s+$/, '') + '…';
    }
    function chunkTextExcerpted(s) {
      return (s || '').length > CHUNK_EXCERPT_CHARS;
    }

    async function refreshDbs() {
      dbsErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
        // If the selected db disappeared, fall back to the first one.
        if (db.value && !dbs.value.includes(db.value) && dbs.value.length) {
          db.value = dbs.value[0];
        }
      } catch (e) {
        dbs.value = [];
        dbsErr.value = t('kb.dbs_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    async function refreshModels() {
      modelsErr.value = '';
      try {
        await loadModels();
      } catch (e) {
        modelsErr.value = t('kb.models_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    watch(models, (list) => {
      if (!model.value && list.length) {
        const loaded = list.find(m => m.loaded);
        model.value = (loaded || list[0]).id;
      }
    }, { immediate: true });

    // S6 numeric rules: min/max/whole-number checks with an inline hint,
    // surfaced in BOTH the chunk form and the ingest form. The overlap
    // rule ("must stay under the size") is the same one the ingest
    // endpoint enforces; it now shows before the request, not after.
    function overlapErr(size, overlap) {
      const e = intError(overlap, 0, 8192);
      if (e) return e;
      if (Number(overlap) >= Number(size)) return t('ingest.err.bad_overlap');
      return '';
    }
    const chunkSizeErr = computed(() => intError(chunkSize.value, 1, 8192));
    const chunkOverlapErr = computed(() => (
      chunkSizeErr.value ? intError(chunkOverlap.value, 0, 8192)
        : overlapErr(chunkSize.value, chunkOverlap.value)
    ));
    const chunkPercentileErr = computed(() => intError(chunkPercentile.value, 50, 100));
    const canChunk = computed(() => (
      !chunkSizeErr.value && !chunkOverlapErr.value && !chunkPercentileErr.value
    ));
    const ingestSizeErr = computed(() => intError(chunkParams.value.size, 1, 8192));
    const ingestOverlapErr = computed(() => (
      ingestSizeErr.value ? intError(chunkParams.value.overlap, 0, 8192)
        : overlapErr(chunkParams.value.size, chunkParams.value.overlap)
    ));
    const ingestPercentileErr = computed(() => intError(chunkParams.value.percentile, 50, 100));
    const canIngest = computed(() => (
      !!db.value && !!ingestFile.value
      && !ingestSizeErr.value && !ingestOverlapErr.value && !ingestPercentileErr.value
    ));

    onMounted(() => { refreshDbs(); refreshModels(); refreshParserStatus(); });
    // The panel stays mounted under v-show, so onMounted only fires
    // once. Re-pull dbs/models whenever the user opens the tab — this
    // replaces the old manual "refresh dbs" button and also picks up
    // models loaded in another tab.
    watch(() => props.view, (v) => {
      if (v === 'ingest' && !ingestBusy.value) { refreshDbs(); refreshModels(); }
      if (v === 'parse') refreshParserStatus();
      if (v === 'ingested') { refreshDbs().finally(loadChunks); }
    });
    // Re-query on db / page-size / level switches while the browser is open.
    watch(db, () => { if (props.view === 'ingested') { viewOffset.value = 0; loadChunks(); } });
    watch(viewPageSize, () => { if (props.view === 'ingested') { viewOffset.value = 0; loadChunks(); } });
    watch(viewLevel, () => { if (props.view === 'ingested') { viewOffset.value = 0; loadChunks(); } });

    // Leaf detail dialog. Only leaves are listed, so every card can open
    // one; the payload pulls the parent chain (section → document) and
    // the derived-index registry so the row's place in the hierarchy is
    // visible without paging the parents into the list.
    async function openChunkDetail(item) {
      detailChunk.value = {
        chunk_id: item.id,
        fields: (item && item.fields) || {},
        ancestors: [],
        indexes: [],
      };
      detailBusy.value = true;
      detailError.value = '';
      try {
        const { payload } = await api(
          'GET',
          '/v1/databases/' + enc(db.value)
            + '/collections/' + INGEST_COLLECTION
            + '/chunks/' + enc(item.id),
        );
        detailChunk.value = payload;
      } catch (e) {
        detailError.value = extractApiError(e, 'unknown');
      } finally {
        detailBusy.value = false;
      }
    }
    function closeChunkDetail() {
      detailChunk.value = null;
      detailError.value = '';
    }
    // Card-body click target. A drag-select inside the card ends with a
    // click too — skip opening so copying text out of a card doesn't
    // yank the dialog open over the selection.
    function openChunkDetailFromCard(item) {
      const sel = window.getSelection && window.getSelection();
      if (sel && String(sel).length) return;
      openChunkDetail(item);
    }
    // status-banner retry: re-fetch the chunk the dialog is showing.
    function retryDetail() {
      const cur = detailChunk.value;
      if (cur) openChunkDetail({ id: cur.chunk_id, fields: cur.fields });
    }
    function onDetailKeydown(ev) {
      if (!detailChunk.value) return;
      if (ev.key === 'Escape') { ev.preventDefault(); closeChunkDetail(); }
    }
    onMounted(() => window.addEventListener('keydown', onDetailKeydown));
    onUnmounted(() => window.removeEventListener('keydown', onDetailKeydown));

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
      formErr.value = '';
      if (!parseFile.value) { formErr.value = t('parse.err.no_file'); return; }

      const form = new FormData();
      form.append('file', parseFile.value);
      form.append('profile', parseProfile.value);

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
          (x) => { parseXhr = x; },
        );
        if (res.aborted) {
          // Cancelled or timed out: not a parse failure, so it reports
          // through the panel banner instead of the error pane.
          parseError.value = null;
          status.value = 'ok';
          statusMsg.value = res.payload.error.message;
        } else if (!res.ok) {
          const err = (res.payload && res.payload.error) || {};
          parseError.value = {
            code: err.code || ('HTTP ' + res.status),
            message: err.message || ('HTTP ' + res.status),
          };
        } else {
          parseResult.value = res.payload;
          status.value = 'ok';
          statusMsg.value = '';
        }
      } catch (e) {
        parseError.value = {
          code: 'network_error',
          message: extractApiError(e, t('parse.err.network')),
        };
      } finally {
        parseXhr = null;
        if (parseTimer) { clearInterval(parseTimer); parseTimer = null; }
        parseElapsed.value = (performance.now() - startedAt) / 1000;
        parseBusy.value = false;
      }
    }

    // Abort button next to the in-flight progress bar. The server sees a
    // dropped connection and stops; the panel unwinds through onabort.
    function cancelParse() {
      if (parseXhr) parseXhr.abort();
    }
    function cancelIngest() {
      // Two phases: while the multipart submit is still on the wire the
      // job row does not exist yet, so dropping the request is the only
      // cancel. Afterwards the worker owns the run and cancellation is
      // cooperative — POST /v1/jobs/{id}/cancel sets a flag the worker
      // honors at the next stage boundary; the SSE stream then delivers
      // the 'cancelled' terminal snapshot and the panel unwinds there.
      if (ingestXhr) { ingestXhr.abort(); return; }
      if (ingestJobId && ingestEs) {
        api('POST', '/v1/jobs/' + enc(ingestJobId) + '/cancel').catch(() => {});
      }
    }
    // Strategy-specific option bag; keys mirror what
    // build_chunker() consumes in chunking/factory.py.
    function strategyOptions(strategy, percentile) {
      const options = {};
      if (strategy === 'semantic') {
        options.breakpoint_percentile = Number(percentile);
      }
      return options;
    }

    async function doChunk() {
      if (chunkBusy.value) return;
      if (!canChunk.value) return;
      chunkBusy.value = true;
      status.value = '';
      statusMsg.value = '';
      try {
        // Field names mirror ChunkRequest in api/chunk.py.
        const { payload } = await api('POST', '/v1/chunk', {
          markdown: chunkMd.value,
          strategy: chunkStrategy.value,
          options: strategyOptions(chunkStrategy.value, chunkPercentile.value),
          add_context: chunkAddContext.value,
          chunk_size: chunkSize.value,
          chunk_overlap: chunkOverlap.value,
        });
        chunkResults.value = (payload && payload.chunks) || [];
        status.value = 'ok';
        // The count is the result the operator is looking for, so it
        // goes in the outcome banner rather than a bare "OK".
        statusMsg.value = t('kb.chunks_n', { n: chunkResults.value.length });
      } catch (e) {
        chunkResults.value = [];
        status.value = 'error';
        statusMsg.value = extractApiError(e, t('common.unknown'));
      } finally {
        chunkBusy.value = false;
      }
    }

    function onIngestFile(ev) {
      ingestFile.value = ev.target.files[0] || null;
      // A new file invalidates whatever the previous run produced.
      ingestResult.value = null;
      ingestError.value = null;
    }

    // Multipart POST that resolves a single JSON envelope, with real
    // upload-byte progress (fetch cannot report it). Used by the ingest
    // submit: POST /v1/jobs/ingest answers 202 { job_id, status } once
    // the upload is spooled — the pipeline itself runs on the job
    // worker. A cancel abort() resolves { aborted: true } so the
    // caller's await unwinds through the same path as streamPost.
    function postFormJson(url, form, onPct, onStart) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url);
        if (onStart) onStart(xhr);
        if (xhr.upload) {
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
              // Hold at 99% until the 202 lands — the last progress
              // event can precede the request being fully sent.
              onPct(Math.min(99, Math.round((e.loaded / e.total) * 100)));
            }
          };
          xhr.upload.onload = () => onPct(100);
        }
        xhr.onload = () => {
          let payload = null;
          try { payload = JSON.parse(xhr.responseText); } catch (_e) { payload = null; }
          resolve({
            ok: xhr.status >= 200 && xhr.status < 300,
            status: xhr.status,
            payload,
          });
        };
        xhr.onabort = () => resolve({ ok: false, status: 0, aborted: true, payload: null });
        xhr.onerror = () => reject(new Error(t('ingest.err.network')));
        xhr.send(form);
      });
    }

    // Follow GET /v1/jobs/{id}/events (SSE). Every `job` frame is a
    // full JobStatus snapshot — the initial one arrives immediately,
    // possibly already terminal. Resolves with the terminal snapshot;
    // a permanently closed EventSource (server gone) rejects instead of
    // spinning, while transient drops reconnect and re-send the state.
    function watchJob(jobId, onFrame) {
      return new Promise((resolve, reject) => {
        const es = new EventSource('/v1/jobs/' + enc(jobId) + '/events');
        ingestEs = es;
        es.addEventListener('job', (ev) => {
          let data = null;
          try { data = JSON.parse(ev.data); } catch (_e) { return; }
          onFrame(data);
          if (JOB_DONE.has(data.status)) {
            es.close();
            ingestEs = null;
            resolve(data);
          }
        });
        es.onerror = () => {
          if (es.readyState === EventSource.CLOSED) {
            es.close();
            ingestEs = null;
            reject(new Error(t('ingest.err.network')));
          }
        };
      });
    }

    // Map a JobStatus snapshot onto the stepper. 'queued' (and the
    // terminal words) leave the stage alone: queued means the upload is
    // done but the worker has not started, terminal is unwound by the
    // caller. Parse pages only exist during parsing — the worker's
    // progress counters are (page, total) there and untouched elsewhere.
    function applyJobFrame(job) {
      const ui = JOB_STAGE[job.status];
      if (!ui) return;
      ingestStage.value = ui;
      if (ui === 'parse' && job.progress && job.progress.total) {
        ingestParsePages.value = {
          page: job.progress.current,
          total: job.progress.total,
        };
      } else {
        ingestParsePages.value = null;
      }
    }

    // Multipart POST to the parse NDJSON stream endpoint
    // (/v1/parse/stream) with real upload-byte progress (fetch cannot
    // report upload progress) AND incremental newline-delimited
    // stage/progress events over the same response. Resolves to
    // { ok, status, payload } so HTTP error envelopes are handled by
    // the caller exactly like the shared api() helper. A pre-flight
    // JSON envelope (stream never started) lands via the onload
    // fallback below. (Ingest used to share this helper; it now goes
    // through the job queue — see doIngest.)
    function streamPost(url, form, onPct, onEvent, onStart) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url);
        let pos = 0;
        let buf = '';
        let settled = false;
        // Watchdog: a stream that stops emitting events (server wedged,
        // proxy holding the socket open) left the panel spinning
        // forever with no way out (B12). Re-armed on every sign of life
        // — upload bytes, response bytes, a parsed event.
        let idleTimer = null;
        let timedOut = false;

        function disarmIdle() {
          if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
        }
        function armIdle() {
          disarmIdle();
          idleTimer = setTimeout(() => {
            timedOut = true;
            xhr.abort();
          }, STREAM_IDLE_MS);
        }

        function finish(result) {
          if (settled) return;
          settled = true;
          disarmIdle();
          resolve(result);
        }

        // Hand the socket back so the caller can abort it from a
        // "cancel" button; the promise then settles through onabort.
        if (onStart) onStart(xhr);

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
          if (full.length) armIdle();
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
            armIdle();
            if (e.lengthComputable) {
              // Hold at 99% until upload.onload — the last progress
              // event can precede the request being fully sent.
              onPct(Math.min(99, Math.round((e.loaded / e.total) * 100)));
            }
          };
          xhr.upload.onload = () => { armIdle(); onPct(100); };
        }
        xhr.onprogress = () => {
          try { drain(false); }
          catch (e) {
            finish({ ok: false, status: 0,
              payload: { error: { code: 'bad_stream', message: String(e.message || e) } } });
          }
        };
        xhr.onload = () => {
          disarmIdle();
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
        // abort() lands here, whether it came from the cancel button or
        // from the watchdog. Resolving (rather than leaving the promise
        // pending) is what lets the caller's `await` unwind and clear
        // its busy flag.
        xhr.onabort = () => {
          finish({
            ok: false,
            status: 0,
            aborted: true,
            payload: { error: {
              code: timedOut ? 'stream_timeout' : 'cancelled',
              message: timedOut
                ? t('kb.err.timeout', { seconds: Math.round(STREAM_IDLE_MS / 1000) })
                : t('kb.cancelled'),
            } },
          });
        };
        xhr.onerror = () => {
          disarmIdle();
          if (!settled) { settled = true; reject(new Error(t('ingest.err.network'))); }
        };
        armIdle();
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

    // Stat row for the terminal parse result: char count from the
    // markdown, the rest from the nested metadata block (images falls
    // back to the top-level images list when the parser omitted it).
    const parseStats = computed(() => {
      const r = parseResult.value;
      if (!r) return { chars: 0, pages: null, images: null, tables: null, ocr: null };
      const md = r.metadata || {};
      return {
        chars: (r.markdown || '').length,
        pages: md.page_count,
        images: md.images_count != null ? md.images_count : (r.images || []).length,
        tables: md.tables_count,
        ocr: md.ocr_pages,
      };
    });

    async function copyParseMarkdown() {
      if (!parseResult.value) return;
      parseCopy.value = (await copyText(parseOutput())) ? 'ok' : 'err';
      setTimeout(() => { parseCopy.value = ''; }, 1600);
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
      docIdCopy.value = (await copyText(ingestResult.value.doc_id)) ? 'ok' : 'err';
      setTimeout(() => { docIdCopy.value = ''; }, 1600);
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
      if (!canIngest.value) return;
      formErr.value = '';
      if (!db.value) { formErr.value = t('ingest.err.no_db'); return; }
      if (!ingestFile.value) { formErr.value = t('ingest.err.no_file'); return; }

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

      // Field names mirror the multipart contract of POST /v1/jobs/ingest
      // (api/jobs.py → _prepare_ingest): embed_model, chunk_size,
      // chunk_overlap, metadata/chunk_options as JSON strings.
      const form = new FormData();
      form.append('file', ingestFile.value);
      form.append('database', db.value);
      form.append('collection', INGEST_COLLECTION);
      form.append('embed_model', model.value || '');
      form.append('chunk_size', String(size));
      form.append('chunk_overlap', String(overlap));
      form.append('metadata', JSON.stringify(meta));
      form.append('profile', ingestProfile.value);
      form.append('strategy', chunkParams.value.strategy);
      form.append('chunk_options', JSON.stringify(
        strategyOptions(chunkParams.value.strategy, chunkParams.value.percentile)
      ));
      form.append('add_context', String(!!chunkParams.value.addContext));

      ingestResult.value = null;
      ingestError.value = null;
      ingestFailedStage.value = null;
      ingestParsePages.value = null;
      ingestBusy.value = true;
      ingestStage.value = 'uploading';
      uploadPct.value = 0;
      ingestElapsed.value = 0;
      docIdCopy.value = '';
      const startedAt = performance.now();
      ingestTimer = setInterval(() => {
        ingestElapsed.value = (performance.now() - startedAt) / 1000;
      }, 200);

      try {
        // 1) Submit. The multipart body uploads with real byte %, then
        //    the API spools it and answers 202 + job_id — from here on
        //    the run is owned by the job worker.
        const res = await postFormJson(
          '/v1/jobs/ingest', form,
          (pct) => { uploadPct.value = pct; },
          (x) => { ingestXhr = x; },
        );
        if (res.aborted) {
          ingestFailedStage.value = null;
          ingestError.value = null;
          status.value = 'ok';
          statusMsg.value = t('kb.cancelled');
          return;
        }
        if (!res.ok) {
          const err = (res.payload && res.payload.error) || {};
          ingestFailedStage.value = ingestStage.value;
          ingestError.value = {
            code: err.code || ('HTTP ' + res.status),
            message: err.message || ('HTTP ' + res.status),
          };
          return;
        }
        ingestJobId = (res.payload && res.payload.job_id) || '';

        // 2) Follow the run over SSE. Every frame is a full JobStatus;
        //    the stream closes after the terminal one, which resolves.
        const job = await watchJob(ingestJobId, applyJobFrame);
        const durationMs = performance.now() - startedAt;
        if (job.status === 'cancelled') {
          // Cooperative cancel — the requested outcome, not an error.
          ingestFailedStage.value = null;
          ingestError.value = null;
          status.value = 'ok';
          statusMsg.value = t('kb.cancelled');
        } else if (job.status === 'failed') {
          const err = job.error || {};
          ingestFailedStage.value = ingestStage.value;
          ingestError.value = {
            code: err.code || 'job_failed',
            message: err.message || t('ingest.err.network'),
          };
        } else {
          // done: JobStatus carries exactly the terminal stats the
          // result block renders (doc_id / chunk_count / page_count /
          // tokens_used).
          ingestResult.value = job;
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
        ingestXhr = null;
        if (ingestEs) { ingestEs.close(); ingestEs = null; }
        ingestJobId = '';
        if (ingestTimer) { clearInterval(ingestTimer); ingestTimer = null; }
        ingestElapsed.value = (performance.now() - startedAt) / 1000;
        ingestBusy.value = false;
      }
    }

    return { status, statusBannerKind, statusBannerText, parserStatus, refreshParserStatus,
             formErr, dbsErr, modelsErr, parserErr, chunkBusy,
             parseFile, parseResult, parseCopy, chunkSize, chunkOverlap, chunkMd, chunkResults,
             chunkStrategy, chunkPercentile, chunkAddContext,
             chunkSizeErr, chunkOverlapErr, chunkPercentileErr, canChunk,
             ingestSizeErr, ingestOverlapErr, ingestPercentileErr, canIngest,
             dbs, db, models, model, ingestFile, chunkParams, metadata, ingestResult,
             ingestBusy, ingestStage, ingestFailedStage, ingestParsePages, uploadPct, ingestElapsed,
             ingestError, ingestMeta, docIdCopy,
             parseView, renderedMarkdown, parseProfile, parseStats,
             parseBusy, parseStage, parseUploadPct, parseElapsed, parseProgress, parseError,
             ingestProfile,
             doParse, doChunk, doIngest, onIngestFile, copyDocId,
             cancelParse, cancelIngest,
             onParseFile,
             copyParseMarkdown, downloadParseMarkdown,
             stageState, stageLabelKey, stageHintKey,
             formatBytes, formatDuration, formatCount,
             STAGES, INGEST_ACCEPT, PROFILES, STRATEGIES,
             viewChunks, viewTotal, viewOffset, viewPageSize,
             viewDocId, viewNameKw, viewLevel, viewBusy, viewError, viewNoColl,
             viewPage, viewPages, chunkDocCopy,
             detailChunk, detailBusy, detailError,
             openChunkDetail, openChunkDetailFromCard, closeChunkDetail, retryDetail,
             chunkView, chunkGroups, docGroupOpen, toggleDocGroup, chunkMdHtml,
             textExpanded, chunkTextLong, chunkTextClamped, toggleChunkText,
             chunkTextPreview, chunkTextExcerpted,
             refreshDbs, refreshModels,
             loadChunks, chunksQuery, chunksReset,
             chunksFirst, chunksPrev, chunksNext, chunksLast, copyChunkDoc };
  },
  template: `
    <div>
      <div class="section" v-show="view === 'parse'">
        <div class="section-head"><h3 class="section-title">{{ $t('kb.parse_title') }} <span class="pill accent">POST /v1/parse/stream</span></h3></div>

        <!-- Docling engine: per-profile warm / cache evict, plus the
             model components and resources each warm profile holds. -->
        <h4 class="cap-group-title">{{ $t('parse.engine_status') }}</h4>
        <status-banner kind="error" :text="parserErr"
                       :retry="parserErr ? refreshParserStatus : null" />
        <parser-profile-cards :parser="parserStatus" @changed="refreshParserStatus" />

        <div class="row"><label>{{ $t('kb.file') }}</label>
          <input type="file" id="parse-file" :accept="INGEST_ACCEPT" :disabled="parseBusy" @change="onParseFile" />
          <div v-if="parseFile" class="ingest-file-meta">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
              <path d="M3 1.5h6L13 5v9.5H3z"/><path d="M9 1.5V5h4"/>
            </svg>
            <span class="ingest-file-name">{{ parseFile.name }}</span>
            <span>· {{ formatBytes(parseFile.size) }}</span>
          </div>
        </div>
        <div class="row"><label>{{ $t('profile.label') }}</label>
          <select id="parse-profile" v-model="parseProfile" :disabled="parseBusy">
            <option v-for="p in PROFILES" :key="p.value" :value="p.value">
              {{ $t(p.labelKey) }}
            </option>
          </select>
        </div>
        <!-- What the selected profile does; updates live on change. -->
        <div class="profile-desc" id="parse-profile-desc">
          {{ $t('profile.desc.' + parseProfile) }}
        </div>
        <div class="actions">
          <button class="btn primary" id="btn-parse" :disabled="parseBusy" @click="doParse">
            <span v-if="parseBusy" class="btn-spinner"></span>
            {{ parseBusy
               ? (parseStage === 'uploading' ? $t('parse.uploading') : $t('parse.parsing'))
               : $t('parse.upload') }}
          </button>
        </div>
        <status-banner kind="error" :text="formErr" />

        <!-- In-flight: real upload byte % then per-page parse progress. -->
        <div v-if="parseBusy" class="response ingest-progress" id="parse-progress">
          <div class="ingest-progress-head">
            <span class="spinner"></span>
            <span class="ingest-phase-label">{{ parseStage === 'uploading' ? $t('parse.uploading') : $t('parse.parsing') }}</span>
            <span class="ingest-elapsed">{{ parseElapsed.toFixed(1) }} {{ $t('common.seconds') }}</span>
            <button class="btn sm" id="btn-parse-cancel" @click="cancelParse">{{ $t('common.cancel') }}</button>
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
                      role="tab" :aria-selected="parseView === 'source'"
                      :class="{ on: parseView === 'source' }"
                      @click="parseView = 'source'">{{ $t('kb.source') }}</button>
              <button type="button" class="btn sm" id="btn-parse-preview"
                      role="tab" :aria-selected="parseView === 'preview'"
                      :class="{ on: parseView === 'preview' }"
                      @click="parseView = 'preview'">{{ $t('kb.preview') }}</button>
            </div>
            <span class="head-actions">
              <button class="btn sm" id="btn-parse-copy" @click="copyParseMarkdown">
                {{ parseCopy === 'ok' ? $t('kb.copied') : (parseCopy === 'err' ? $t('common.copy_failed') : $t('kb.copy')) }}
              </button>
              <button class="btn sm" id="btn-parse-download" @click="downloadParseMarkdown">
                {{ $t('kb.download_md') }}
              </button>
            </span>
          </div>

          <!-- Parse stats: chars / pages / images / tables / OCR / time. -->
          <div class="kb-result" id="parse-stats">
            <div class="stat"><span class="key">{{ $t('parse.stat.chars') }}</span><span class="val accent">{{ formatCount(parseStats.chars) }}</span></div>
            <div class="stat"><span class="key">{{ $t('parse.stat.pages') }}</span><span class="val">{{ formatCount(parseStats.pages) }}</span></div>
            <div class="stat"><span class="key">{{ $t('parse.stat.images') }}</span><span class="val">{{ formatCount(parseStats.images) }}</span></div>
            <div class="stat"><span class="key">{{ $t('parse.stat.tables') }}</span><span class="val">{{ formatCount(parseStats.tables) }}</span></div>
            <div class="stat"><span class="key">{{ $t('parse.stat.ocr') }}</span><span class="val">{{ formatCount(parseStats.ocr) }}</span></div>
            <div class="stat"><span class="key">{{ $t('parse.stat.duration') }}</span><span class="val ingest-stat-sm">{{ formatDuration(parseElapsed * 1000) }}</span></div>
          </div>

          <pre v-if="parseView === 'source'" class="code-pane" id="parse-markdown">{{ parseResult.markdown || JSON.stringify(parseResult, null, 2) }}</pre>
          <!-- renderedMarkdown is HTML-escaped by markdown.js before v-html -->
          <div v-else class="md-preview" id="parse-markdown-preview" v-html="renderedMarkdown"></div>
        </div>
      </div>

      <div class="section" v-show="view === 'chunk'">
        <div class="section-head"><h3 class="section-title">{{ $t('kb.chunk_title') }} <span class="pill accent">POST /v1/chunk</span></h3></div>
        <div class="row split">
          <div class="row"><label>{{ $t('kb.strategy') }}</label>
            <select id="chunk-strategy" v-model="chunkStrategy">
              <option v-for="s in STRATEGIES" :key="s.value" :value="s.value">
                {{ $t(s.labelKey) }}
              </option>
            </select>
          </div>
          <div class="row" v-if="chunkStrategy === 'semantic'">
            <label>{{ $t('kb.breakpoint') }}</label>
            <input type="number" id="chunk-percentile" min="50" max="100"
                   v-model.number="chunkPercentile" />
            <span class="hint" v-if="chunkPercentileErr">{{ chunkPercentileErr }}</span>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('kb.chunk_size') }}</label><input type="number" id="chunk-size" min="1" max="8192" v-model.number="chunkSize" />
            <span class="hint" v-if="chunkSizeErr">{{ chunkSizeErr }}</span></div>
          <div class="row"><label>{{ $t('kb.overlap') }}</label><input type="number" id="chunk-overlap" min="0" max="8192" v-model.number="chunkOverlap" />
            <span class="hint" v-if="chunkOverlapErr">{{ chunkOverlapErr }}</span></div>
        </div>
        <div class="row checkbox-row">
          <label class="checkbox-label">
            <input type="checkbox" id="chunk-add-context" v-model="chunkAddContext" />
            {{ $t('chunk.add_context') }}
          </label>
        </div>
        <div class="row"><label>{{ $t('kb.markdown') }}</label><textarea id="chunk-md" rows="6" v-model="chunkMd"></textarea></div>
        <div class="actions">
          <busy-button id="btn-chunk" variant="primary" :busy="chunkBusy" :disabled="!canChunk" :label="$t('kb.chunk_btn')"
                       :busy-label="$t('kb.chunking')" @click="doChunk" />
        </div>
        <div v-if="chunkResults.length" class="response" id="chunk-results">
          <div class="response-head"><span>{{ $t('kb.chunks_n', { n: chunkResults.length }) }}</span></div>
          <div v-for="(c, i) in chunkResults" :key="i" class="field-card">
            <div class="field-card-header">
              <span class="index-badge">#{{ i + 1 }}</span>
              <span class="title">{{ $t('kb.chars_tokens', { chars: c.text.length, tokens: c.token_count }) }}</span>
              <span v-if="c.section_header" class="hint">{{ c.section_header }}</span>
              <span v-if="c.page_number != null" class="hint">{{ $t('kb.page', { n: c.page_number }) }}</span>
            </div>
            <!-- No id here: this sits inside a v-for, so a fixed id
                 repeated once per chunk and made the document invalid
                 (B12). The .chunk-context class is what styles it. -->
            <div v-if="c.context" class="chunk-context">
              <span class="hint">{{ $t('chunk.context_prefix') }}</span>{{ c.context }}
            </div>
            <pre class="code-pane chunk-text"
                 :class="{ clamped: chunkTextClamped('chunk:' + i, c.text) }">{{ c.text }}</pre>
            <button v-if="chunkTextLong(c.text)" type="button" class="btn sm chunk-expand"
                    :aria-expanded="!chunkTextClamped('chunk:' + i, c.text)"
                    @click="toggleChunkText('chunk:' + i)">
              {{ chunkTextClamped('chunk:' + i, c.text) ? $t('kb.expand_full') : $t('kb.collapse') }}
            </button>
          </div>
        </div>
      </div>

      <div class="section" v-show="view === 'ingest'">
        <div class="section-head"><h3 class="section-title">{{ $t('kb.ingest_title') }} <span class="pill accent">POST /v1/jobs/ingest</span></h3></div>
        <!-- Load failures for the two dropdowns below, previously
             swallowed: an empty database list looked like "no databases
             exist" rather than "the request failed". -->
        <status-banner kind="error" :text="dbsErr" :retry="dbsErr ? refreshDbs : null" />
        <status-banner kind="error" :text="modelsErr" :retry="modelsErr ? refreshModels : null" />
        <!-- 1 · source file -->
        <div class="form-group">
          <div class="form-group-title"><span class="form-group-step">1</span>{{ $t('ingest.group.file') }}</div>
          <div class="row"><label>{{ $t('kb.file') }}</label>
            <input type="file" id="ingest-file" :accept="INGEST_ACCEPT" :disabled="ingestBusy" @change="onIngestFile" />
            <div v-if="ingestFile" class="ingest-file-meta">
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
                <path d="M3 1.5h6L13 5v9.5H3z"/><path d="M9 1.5V5h4"/>
              </svg>
              <span class="ingest-file-name">{{ ingestFile.name }}</span>
              <span>· {{ formatBytes(ingestFile.size) }}</span>
            </div>
          </div>
        </div>

        <!-- 2 · parsing: profile -->
        <div class="form-group">
          <div class="form-group-title"><span class="form-group-step">2</span>{{ $t('ingest.group.parse') }}</div>
          <div class="row"><label>{{ $t('profile.label') }}</label>
            <select id="ingest-profile" v-model="ingestProfile" :disabled="ingestBusy">
              <option v-for="p in PROFILES" :key="p.value" :value="p.value">
                {{ $t(p.labelKey) }}
              </option>
            </select>
          </div>
          <!-- What the selected profile does; updates live on change. -->
          <div class="profile-desc" id="ingest-profile-desc">
            {{ $t('profile.desc.' + ingestProfile) }}
          </div>
        </div>

        <!-- 3 · chunking: strategy, size/overlap, contextual enrichment -->
        <div class="form-group">
          <div class="form-group-title"><span class="form-group-step">3</span>{{ $t('ingest.group.chunk') }}</div>
          <div class="row split">
            <div class="row"><label>{{ $t('kb.strategy') }}</label>
              <select id="ingest-strategy" v-model="chunkParams.strategy" :disabled="ingestBusy">
                <option v-for="s in STRATEGIES" :key="s.value" :value="s.value">
                  {{ $t(s.labelKey) }}
                </option>
              </select>
            </div>
            <div class="row" v-if="chunkParams.strategy === 'semantic'">
              <label>{{ $t('kb.breakpoint') }}</label>
              <input type="number" id="ingest-percentile" min="50" max="100"
                     v-model.number="chunkParams.percentile" :disabled="ingestBusy" />
              <span class="hint" v-if="ingestPercentileErr">{{ ingestPercentileErr }}</span>
            </div>
          </div>
          <div class="row split">
            <div class="row"><label>{{ $t('kb.chunk_size') }}</label><input type="number" id="ingest-size" min="1" max="8192" v-model.number="chunkParams.size" :disabled="ingestBusy" />
              <span class="hint" v-if="ingestSizeErr">{{ ingestSizeErr }}</span></div>
            <div class="row"><label>{{ $t('kb.overlap') }}</label><input type="number" id="ingest-overlap" min="0" max="8192" v-model.number="chunkParams.overlap" :disabled="ingestBusy" />
              <span class="hint" v-if="ingestOverlapErr">{{ ingestOverlapErr }}</span></div>
          </div>
          <div class="row checkbox-row">
            <label class="checkbox-label">
              <input type="checkbox" id="ingest-add-context"
                     v-model="chunkParams.addContext" :disabled="ingestBusy" />
              {{ $t('chunk.add_context') }}
            </label>
          </div>
        </div>

        <!-- 4 · embedding + storage destination -->
        <div class="form-group">
          <div class="form-group-title"><span class="form-group-step">4</span>{{ $t('ingest.group.dest') }}</div>
          <div class="row split">
            <div class="row"><label>{{ $t('common.database') }}</label>
              <select id="ingest-db" v-model="db" :disabled="ingestBusy">
                <option v-if="!dbs.length" value="" disabled>{{ $t('ingest.no_dbs') }}</option>
                <option v-for="d in dbs" :key="d" :value="d">{{ d }}</option>
              </select>
            </div>
            <div class="row"><label>{{ $t('kb.embed_model') }}</label>
              <select id="ingest-model" v-model="model" :disabled="ingestBusy">
                <option v-for="mm in models" :key="mm.id" :value="mm.id">
                  {{ mm.id }}{{ mm.loaded ? ' · ' + $t('common.loaded') : '' }}
                </option>
              </select>
            </div>
          </div>
          <div class="row"><label>{{ $t('kb.metadata') }} <span class="hint">{{ $t('ingest.metadata_hint') }}</span></label>
            <textarea id="ingest-metadata" class="code-input" rows="2" v-model="metadata" :disabled="ingestBusy">{}</textarea>
          </div>
        </div>
        <div class="actions">
          <button class="btn primary" id="btn-ingest-upload" :disabled="ingestBusy || !canIngest" @click="doIngest">
            <span v-if="ingestBusy" class="btn-spinner"></span>
            {{ ingestBusy
               ? (ingestStage === 'uploading' ? $t('ingest.uploading') : $t('ingest.processing'))
               : $t('ingest.upload') }}
          </button>
        </div>
        <status-banner kind="error" :text="formErr" />

        <!-- In-flight: 5-stage stepper. Upload has a real byte %, the
             parse stage has a real per-page % (JobStatus.progress);
             later stages move as SSE job frames arrive with an
             indeterminate bar. Cancel is cooperative once the job
             exists — the worker stops at the next stage boundary. -->
        <div v-if="ingestBusy" class="response ingest-progress" id="ingest-progress">
          <div class="ingest-progress-head">
            <span class="spinner"></span>
            <span class="ingest-phase-label">{{ $t(stageLabelKey(ingestStage)) }}</span>
            <span class="ingest-elapsed">{{ ingestElapsed.toFixed(1) }} {{ $t('common.seconds') }}</span>
            <button class="btn sm" id="btn-ingest-cancel" @click="cancelIngest">{{ $t('common.cancel') }}</button>
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
              {{ docIdCopy === 'ok' ? $t('ingest.copied') : (docIdCopy === 'err' ? $t('common.copy_failed') : $t('ingest.copy')) }}
            </button>
          </div>

          <details class="collapsible">
            <summary>{{ $t('ingest.raw') }}</summary>
            <div class="body"><pre class="code-pane">{{ JSON.stringify(ingestResult, null, 2) }}</pre></div>
          </details>
        </div>
      </div>

      <div class="section" v-show="view === 'ingested'">
        <div class="section-head"><h3 class="section-title">{{ $t('kb.chunks_title') }} <span class="pill accent">POST /v1/databases/&#123;db&#125;/collections/ingest/rows</span></h3></div>

        <div class="row split">
          <div class="row"><label>{{ $t('common.database') }}</label>
            <select id="chunks-db" v-model="db" :disabled="viewBusy">
              <option v-if="!dbs.length" value="" disabled>{{ $t('ingest.no_dbs') }}</option>
              <option v-for="d in dbs" :key="d" :value="d">{{ d }}</option>
            </select>
          </div>
          <div class="row"><label>{{ $t('chunks.page_size') }}</label>
            <select id="chunks-pagesize" v-model.number="viewPageSize" :disabled="viewBusy">
              <option :value="10">10</option><option :value="20">20</option><option :value="50">50</option>
            </select>
          </div>
          <div class="row"><label>{{ $t('chunks.level') }}</label>
            <select id="chunks-level" v-model="viewLevel" :disabled="viewBusy">
              <option value="chunk">{{ $t('chunks.level_chunk') }}</option>
              <option value="section">{{ $t('chunks.level_section') }}</option>
              <option value="document">{{ $t('chunks.level_document') }}</option>
            </select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('chunks.doc_filter') }}</label>
            <input type="text" id="chunks-docid" v-model="viewDocId"
                   :disabled="viewBusy" placeholder="uuid" />
          </div>
          <div class="row"><label>{{ $t('chunks.name_filter') }}</label>
            <input type="text" id="chunks-name" v-model="viewNameKw"
                   :disabled="viewBusy" placeholder="report.pdf" />
          </div>
        </div>
        <div class="actions">
          <busy-button id="btn-chunks-query" variant="primary" :busy="viewBusy" :label="$t('chunks.query')"
                       :busy-label="$t('chunks.running')" @click="chunksQuery" />
          <button class="btn" id="btn-chunks-reset" :disabled="viewBusy" @click="chunksReset">
            {{ $t('chunks.reset') }}
          </button>
        </div>

        <div class="response" id="chunks-result">
          <div class="response-head">
            <span>{{ $t('chunks.count') }}: {{ formatCount(viewTotal) }}</span>
            <span class="head-actions">
              <!-- Same dual view as the parse result: raw markdown vs
                   rendered output, applied to every card body. -->
              <div class="seg-toggle" role="tablist">
                <button type="button" class="btn sm" id="btn-chunk-source"
                        role="tab" :aria-selected="chunkView === 'source'"
                        :class="{ on: chunkView === 'source' }"
                        @click="chunkView = 'source'">{{ $t('kb.source') }}</button>
                <button type="button" class="btn sm" id="btn-chunk-preview"
                        role="tab" :aria-selected="chunkView === 'preview'"
                        :class="{ on: chunkView === 'preview' }"
                        @click="chunkView = 'preview'">{{ $t('kb.preview') }}</button>
              </div>
              <span v-if="viewBusy" class="spinner"></span>
            </span>
          </div>

          <!-- Four states: the missing-collection case and the failed
               case used to share the plain "no match" styling, so a
               broken query looked like an empty collection. -->
          <empty-state v-if="viewNoColl" id="chunks-no-coll" state="empty"
                       :text="$t('chunks.no_collection')" />
          <empty-state v-else-if="viewError" id="chunks-error" state="error" :text="viewError"
                       :retry="loadChunks" />
          <empty-state v-else-if="viewBusy && !viewChunks.length" state="loading" />
          <empty-state v-else-if="!viewChunks.length" id="chunks-empty" state="empty"
                       :text="$t('chunks.no_match')" />

          <template v-else>
            <div class="kb-result" id="chunks-stats">
              <div class="stat"><span class="key">{{ $t('chunks.stat_total') }}</span><span class="val accent">{{ formatCount(viewTotal) }}</span></div>
              <div class="stat"><span class="key">{{ $t('chunks.stat_page') }}</span><span class="val">{{ viewPage }} / {{ viewPages }}</span></div>
              <div class="stat"><span class="key">{{ $t('chunks.stat_returned') }}</span><span class="val">{{ viewChunks.length }}</span></div>
            </div>

            <div v-for="g in chunkGroups" :key="g.docId" class="doc-group">
              <!-- One doc_id row per document (was one per card): fold
                   caret, file name, chunk count, copyable id. -->
              <div class="doc-group-head">
                <button type="button" class="doc-fold" :aria-expanded="docGroupOpen(g.docId)"
                        :aria-label="docGroupOpen(g.docId) ? $t('kb.collapse') : $t('kb.expand_doc')"
                        @click="toggleDocGroup(g.docId)">
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 3l5 5-5 5"/>
                  </svg>
                </button>
                <span class="title" :title="g.filename || g.docId">{{ g.filename || g.docId }}</span>
                <span class="hint">{{ $t('kb.doc_chunks', { n: g.items.length }) }}</span>
                <span class="chunks-docid">
                  <span class="hint">doc_id</span>
                  <code :title="g.docId">{{ g.docId }}</code>
                  <button class="btn sm" @click="copyChunkDoc(g.docId)">
                    {{ chunkDocCopy.id === g.docId
                       ? (chunkDocCopy.ok ? $t('ingest.copied') : $t('common.copy_failed'))
                       : $t('ingest.copy') }}
                  </button>
                </span>
              </div>

              <template v-if="docGroupOpen(g.docId)">
                <div v-for="it in g.items" :key="it.id" class="field-card chunk-card"
                     role="button" tabindex="0"
                     :aria-label="$t('chunks.detail')"
                     @click="openChunkDetailFromCard(it)"
                     @keyup.enter="openChunkDetail(it)">
                  <!-- Row 1: file name › section path breadcrumb.
                       Row 2: chunk #, chars/tokens, page — meta, muted.
                       Click anywhere on the card opens the leaf detail
                       dialog (parent chain + index registry); the inner
                       buttons stop the click from bubbling up. -->
                  <div class="field-card-header">
                    <span class="title crumb" :title="(it.fields && it.fields.filename) || ''">{{ (it.fields && it.fields.filename) || '—' }}</span>
                    <span v-if="it.fields && it.fields.section_header" class="hint crumb-path" :title="it.fields.section_header">› {{ it.fields.section_header }}</span>
                    <button class="btn sm chunk-detail-btn"
                            :id="'btn-chunk-detail-' + it.id"
                            @click.stop="openChunkDetail(it)">{{ $t('chunks.detail') }}</button>
                  </div>
                  <div class="field-card-meta">
                    <span class="index-badge">#{{ it.fields && it.fields.chunk_index }}</span>
                    <span class="hint">{{ $t('kb.chars_tokens', { chars: (it.fields && it.fields.text ? it.fields.text.length : 0), tokens: it.fields && it.fields.token_count }) }}</span>
                    <span v-if="it.fields && it.fields.page_number != null" class="hint">{{ $t('kb.page', { n: it.fields.page_number }) }}</span>
                  </div>
                  <!-- Body is an excerpt of the chunk; the detail dialog
                       (above the fold on every card) carries the rest. -->
                  <pre v-if="chunkView === 'source'" class="code-pane chunk-text">{{ chunkTextPreview((it.fields && it.fields.text) || '') }}</pre>
                  <!-- chunkMdHtml is HTML-escaped by markdown.js before v-html -->
                  <div v-else class="md-preview chunk-md"
                       v-html="chunkMdHtml.get(it.id)"></div>
                  <div v-if="chunkTextExcerpted((it.fields && it.fields.text) || '')"
                       class="hint chunk-excerpt-hint">
                    {{ $t('chunks.excerpt_more') }}
                  </div>
                </div>
              </template>
            </div>

            <ui-pager root-id="chunks-pager"
                      first-id="btn-chunks-first" prev-id="btn-chunks-prev"
                      next-id="btn-chunks-next" last-id="btn-chunks-last"
                      :page="viewPage" :pages="viewPages" :busy="viewBusy"
                      :info-text="$t('kb.pager_info', { from: viewOffset + 1, to: viewOffset + viewChunks.length, total: viewTotal, page: viewPage, pages: viewPages })"
                      @first="chunksFirst" @prev="chunksPrev"
                      @next="chunksNext" @last="chunksLast" />
          </template>
        </div>

        <!-- Leaf detail dialog: this chunk's content fields, its parent
             chain (section → document) and the derived indexes it is
             registered in. The list itself stays leaf-only. -->
        <div v-if="detailChunk" class="modal-overlay" id="modal-chunk-detail"
             @click.self="closeChunkDetail">
          <div class="modal modal--wide" role="dialog" aria-modal="true"
               aria-labelledby="modal-chunk-detail-title">
            <header class="modal-head">
              <h3 id="modal-chunk-detail-title">{{ $t('chunks.detail_title') }}
                <span class="pill accent">{{ detailChunk.chunk_id }}</span>
              </h3>
              <button class="modal-close" @click="closeChunkDetail"
                      :aria-label="$t('common.close')">x</button>
            </header>
            <div class="modal-body">
              <empty-state v-if="detailBusy" state="loading" />
              <status-banner v-else-if="detailError" kind="error" :text="detailError"
                             :retry="retryDetail" />
              <template v-else>
                <div class="kb-result" id="chunk-detail-stats">
                  <div class="stat"><span class="key">{{ $t('chunks.detail_index') }}</span><span class="val accent">#{{ detailChunk.fields.chunk_index }}</span></div>
                  <div class="stat"><span class="key">{{ $t('chunks.detail_level') }}</span><span class="val">{{ detailChunk.fields.level }}</span></div>
                  <div class="stat"><span class="key">{{ $t('chunks.detail_tokens') }}</span><span class="val">{{ detailChunk.fields.token_count }}</span></div>
                  <div class="stat" v-if="detailChunk.fields.page_number != null"><span class="key">{{ $t('chunks.detail_page') }}</span><span class="val">{{ detailChunk.fields.page_number }}</span></div>
                  <div class="stat"><span class="key">{{ $t('chunks.detail_span') }}</span><span class="val">{{ detailChunk.fields.char_start }}–{{ detailChunk.fields.char_end }}</span></div>
                </div>

                <h4 class="cap-group-title">{{ $t('chunks.detail_text') }}</h4>
                <pre class="code-pane chunk-text" id="chunk-detail-text">{{ detailChunk.fields.text || '' }}</pre>

                <h4 class="cap-group-title">{{ $t('chunks.detail_ancestors') }}</h4>
                <div v-if="!detailChunk.ancestors.length" class="hint" id="chunk-detail-no-ancestors">{{ $t('chunks.detail_no_ancestors') }}</div>
                <div v-for="a in detailChunk.ancestors" :key="a.chunk_id" class="field-card ancestor-card">
                  <div class="field-card-header">
                    <span class="index-badge">{{ a.level }}</span>
                    <span class="title crumb" :title="a.section_header || a.chunk_id">{{ a.section_header || a.chunk_id }}</span>
                    <span class="hint">{{ $t('kb.chars_tokens', { chars: (a.text || '').length, tokens: a.token_count }) }}</span>
                  </div>
                  <pre class="code-pane chunk-text clamped">{{ a.text || '' }}</pre>
                </div>

                <h4 class="cap-group-title">{{ $t('chunks.detail_indexes') }}</h4>
                <div v-if="!detailChunk.indexes.length" class="hint" id="chunk-detail-no-indexes">{{ $t('chunks.detail_no_indexes') }}</div>
                <table v-else class="info-table" id="chunk-detail-indexes">
                  <tr v-for="ix in detailChunk.indexes" :key="ix.index_kind">
                    <th>{{ ix.index_kind }}</th>
                    <td><code>{{ ix.model }}</code> · <code>{{ ix.index_ref }}</code></td>
                  </tr>
                </table>
              </template>
            </div>
            <footer class="modal-foot">
              <button class="btn" id="btn-chunk-detail-close" @click="closeChunkDetail">{{ $t('common.close') }}</button>
            </footer>
          </div>
        </div>
      </div>

      <!-- Replaces the "last op: OK" line, which reported the word OK
           and nothing else — including for a failed chunk run. -->
      <status-banner :kind="statusBannerKind" :text="statusBannerText" />
    </div>
  `,
});
