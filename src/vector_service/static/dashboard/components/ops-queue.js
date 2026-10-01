// =====================================================================
// ops-queue.js -- Operations > Job queue.
//   GET  /v1/jobs                list (status filter, paging)
//   GET  /v1/jobs/{id}           detail snapshot
//   GET  /v1/jobs/{id}/events    SSE live updates (named "job" events)
//   POST /v1/jobs/{id}/cancel    cooperative cancel (409 if terminal)
// =====================================================================
import { defineComponent, ref, watch } from '../vue.esm-browser.prod.js';
import { store, api, t } from './app.js';
import {
  enc, formatTs, pillClass, progressText, statusLabel, TERMINAL,
} from './ops-common.js';

const PAGE = 20;
const STATUS_FILTERS = [
  '', 'queued', 'parsing', 'chunking', 'embedding', 'upserting',
  'extracting', 'graphing', 'evaluating', 'done', 'failed', 'cancelled',
];

export default defineComponent({
  name: 'OpsQueue',
  setup() {
    const items = ref([]);
    const total = ref(0);
    const offset = ref(0);
    const statusFilter = ref('');
    const loading = ref(false);
    const listErr = ref('');
    const detail = ref(null);
    const detailErr = ref('');
    let es = null;
    let timer = null;
    let started = false;

    async function load() {
      loading.value = true;
      listErr.value = '';
      try {
        const q = `/v1/jobs?limit=${PAGE}&offset=${offset.value}`
          + (statusFilter.value ? `&status=${enc(statusFilter.value)}` : '');
        const { payload } = await api('GET', q);
        if (payload) {
          items.value = payload.items || [];
          total.value = payload.total || 0;
        }
      } catch (e) { listErr.value = e.message; }
      finally { loading.value = false; }
    }

    function setFilter(s) {
      statusFilter.value = s;
      offset.value = 0;
      load();
    }
    function prevPage() {
      if (offset.value === 0) return;
      offset.value = Math.max(0, offset.value - PAGE);
      load();
    }
    function nextPage() {
      if (offset.value + items.value.length >= total.value) return;
      offset.value += PAGE;
      load();
    }

    function closeStream() {
      if (es) { es.close(); es = null; }
    }

    async function openDetail(jobId) {
      closeStream();
      detail.value = null;
      detailErr.value = '';
      try {
        const { payload } = await api('GET', `/v1/jobs/${enc(jobId)}`);
        detail.value = payload;
      } catch (e) { detailErr.value = e.message; return; }
      // Live updates: server sends a frame immediately, then one frame
      // per state change, and closes the stream after a terminal frame.
      // Listen for the NAMED "job" event (not onmessage) and close at
      // terminal so EventSource never auto-reconnects a finished stream.
      es = new EventSource(`/v1/jobs/${enc(jobId)}/events`);
      es.addEventListener('job', (ev) => {
        try {
          const data = JSON.parse(ev.data);
          detail.value = data;
          if (TERMINAL.has(data.status)) closeStream();
        } catch (_e) { /* ignore malformed frame; keep streaming */ }
      });
    }

    function closeDetail() {
      closeStream();
      detail.value = null;
    }

    async function cancelJob() {
      const job = detail.value;
      if (!job || TERMINAL.has(job.status)) return;
      if (!window.confirm(t('ops.queue.cancel_confirm'))) return;
      detailErr.value = '';
      try {
        const { payload } = await api('POST', `/v1/jobs/${enc(job.job_id)}/cancel`);
        detail.value = payload;
      } catch (e) { detailErr.value = e.message; }
    }

    function start() {
      if (started) return;
      started = true;
      load();
      timer = setInterval(load, 4000);
    }
    function stop() {
      if (!started) return;
      started = false;
      if (timer) { clearInterval(timer); timer = null; }
      closeStream();
      detail.value = null;
    }

    watch(() => store.view, (v) => (v === 'queue' ? start() : stop()));
    if (store.view === 'queue') start();

    return {
      STATUS_FILTERS, PAGE, items, total, offset, statusFilter, loading, listErr,
      detail, detailErr,
      load, setFilter, prevPage, nextPage, openDetail, closeDetail, cancelJob,
      formatTs, pillClass, progressText, statusLabel, TERMINAL,
    };
  },
  template: `
    <div class="ops-panel">
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('nav.queue') }}
            <span class="pill accent">GET /v1/jobs</span>
          </h3>
          <span class="section-sub">
            <button class="btn sm ghost" @click="load">{{ $t('common.refresh') }}</button>
          </span>
        </div>

        <div class="ops-filters">
          <button v-for="s in STATUS_FILTERS" :key="s || '__all'"
                  :class="['btn', 'sm', statusFilter === s ? 'primary' : 'ghost']"
                  @click="setFilter(s)">
            {{ s === '' ? $t('ops.queue.all') : statusLabel(s) }}
          </button>
        </div>

        <div v-if="listErr" class="empty error">{{ listErr }}</div>
        <div v-if="loading" class="spinner"></div>

        <div class="data-table-wrap">
          <table class="data-table ops-jobs-table">
            <thead>
              <tr>
                <th>{{ $t('ops.queue.created') }}</th>
                <th>{{ $t('ops.queue.job_id') }}</th>
                <th>{{ $t('ops.queue.status') }}</th>
                <th>{{ $t('ops.queue.stage') }}</th>
                <th>{{ $t('ops.queue.progress') }}</th>
                <th>{{ $t('ops.queue.attempts') }}</th>
                <th>{{ $t('ops.queue.filename') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="j in items" :key="j.job_id"
                  class="ops-job-row"
                  :class="{ 'ops-row-selected': detail && detail.job_id === j.job_id }"
                  @click="openDetail(j.job_id)">
                <td class="ops-ts">{{ formatTs(j.created_ts) }}</td>
                <td class="ops-mono">{{ j.job_id }}</td>
                <td><span :class="['pill', pillClass(j.status)]">{{ statusLabel(j.status) }}</span></td>
                <td>{{ j.stage ? statusLabel(j.stage) : '—' }}</td>
                <td>{{ progressText(j.progress) }}</td>
                <td>{{ j.attempts }}/{{ j.max_attempts }}</td>
                <td>{{ j.filename || '—' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="pager">
          <span class="pager-info">{{ $t('common.total') }} {{ total }} · {{ $t('common.offset') }} {{ offset }}</span>
          <button class="btn sm ghost" :disabled="offset === 0" @click="prevPage">← {{ $t('common.prev') }}</button>
          <button class="btn sm ghost" :disabled="offset + items.length >= total" @click="nextPage">{{ $t('common.next') }} →</button>
        </div>
      </div>

      <div v-if="detail" class="section ops-detail">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('ops.queue.detail') }}
            <span class="pill accent">SSE /v1/jobs/{id}/events</span>
          </h3>
          <span class="section-sub">
            <button v-if="!TERMINAL.has(detail.status)" class="btn sm danger" @click="cancelJob">
              {{ $t('ops.queue.cancel') }}
            </button>
            <button class="btn sm ghost" @click="closeDetail">{{ $t('common.close') }}</button>
          </span>
        </div>

        <div v-if="detailErr" class="empty error">{{ detailErr }}</div>

        <div class="ops-detail-grid">
          <table class="info-table">
            <tr><th>{{ $t('common.job_id') }}</th><td class="ops-mono">{{ detail.job_id }}</td></tr>
            <tr><th>{{ $t('common.doc_id') }}</th><td class="ops-mono">{{ detail.doc_id || '—' }}</td></tr>
            <tr><th>{{ $t('ops.queue.status') }}</th><td><span :class="['pill', pillClass(detail.status)]">{{ statusLabel(detail.status) }}</span></td></tr>
            <tr><th>{{ $t('ops.queue.stage') }}</th><td>{{ detail.stage ? statusLabel(detail.stage) : '—' }}</td></tr>
            <tr><th>{{ $t('common.scope') }}</th><td class="ops-mono">{{ detail.database }}/{{ detail.collection }}</td></tr>
            <tr><th>{{ $t('ops.queue.filename') }}</th><td>{{ detail.filename || '—' }} <span class="muted">{{ detail.mime || '' }}</span></td></tr>
            <tr><th>{{ $t('ops.queue.progress') }}</th><td>{{ progressText(detail.progress) }}</td></tr>
            <tr><th>{{ $t('ops.queue.attempts') }}</th><td>{{ detail.attempts }}/{{ detail.max_attempts }}</td></tr>
            <tr><th>{{ $t('ops.queue.cancel_requested') }}</th><td>{{ detail.cancel_requested ? '✓' : '—' }}</td></tr>
            <tr><th>{{ $t('ops.queue.chunk_count') }}</th><td>{{ detail.chunk_count }}</td></tr>
            <tr><th>{{ $t('ops.queue.page_count') }}</th><td>{{ detail.page_count }}</td></tr>
            <tr><th>{{ $t('ops.queue.tokens_used') }}</th><td>{{ detail.tokens_used }}</td></tr>
          </table>
          <div class="ops-detail-side">
            <div v-if="detail.error" class="empty error">
              <div class="ops-err-code ops-mono">{{ detail.error.code }}</div>
              <div>{{ detail.error.message }}</div>
            </div>
            <table class="info-table">
              <tr><th>{{ $t('ops.queue.created') }}</th><td>{{ formatTs(detail.created_ts) }}</td></tr>
              <tr><th>{{ $t('ops.queue.updated') }}</th><td>{{ formatTs(detail.updated_ts) }}</td></tr>
              <tr><th>{{ $t('ops.queue.finished') }}</th><td>{{ formatTs(detail.finished_ts) }}</td></tr>
            </table>
          </div>
        </div>
      </div>
    </div>
  `,
});
