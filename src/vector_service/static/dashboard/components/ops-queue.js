// =====================================================================
// ops-queue.js -- Operations > Job queue.
//   GET  /v1/jobs                list (status filter, paging)
//   GET  /v1/jobs/{id}           detail snapshot
//   GET  /v1/jobs/{id}/events    SSE live updates (named "job" events)
//   POST /v1/jobs/{id}/cancel    cooperative cancel (409 if terminal)
// =====================================================================
import { defineComponent, ref, computed, watch } from '../vue.esm-browser.prod.js';
import { store, api, t } from './app.js';
import { StatusBanner, BusyButton, EmptyState, askConfirm } from './feedback.js';
import UiPager from './pager.js';
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
    // Page-number paging fronting the API's offset (shared UiPager).
    const page = ref(1);
    const statusFilter = ref('');
    const loading = ref(false);
    const listErr = ref('');
    const detail = ref(null);
    const detailErr = ref('');
    // The cancel POST had no in-flight state: the button stayed live
    // for the whole round trip and could be fired repeatedly.
    const cancelBusy = ref(false);
    const manualBusy = ref(false);
    let es = null;
    let timer = null;
    let started = false;

    const pages = computed(() => Math.max(1, Math.ceil(total.value / PAGE)));
    const pageOffset = computed(() => (page.value - 1) * PAGE);

    async function load() {
      loading.value = true;
      listErr.value = '';
      try {
        const q = `/v1/jobs?limit=${PAGE}&offset=${pageOffset.value}`
          + (statusFilter.value ? `&status=${enc(statusFilter.value)}` : '');
        const { payload } = await api('GET', q);
        if (payload) {
          items.value = payload.items || [];
          total.value = payload.total || 0;
        }
      } catch (e) { listErr.value = e.message; }
      finally { loading.value = false; }
    }

    // The list also auto-polls every 4s, so `loading` is true on its own
    // schedule; only an operator-initiated refresh drives the button's
    // busy state (otherwise the spinner would flash on every poll).
    async function reload() {
      manualBusy.value = true;
      try { await load(); } finally { manualBusy.value = false; }
    }

    function setFilter(s) {
      statusFilter.value = s;
      page.value = 1;
      load();
    }
    // UiPager disables the edges and clamps the jump box; this clamps
    // again for the event handlers and skips a no-op landing.
    function goToPage(n) {
      const target = Math.min(Math.max(1, Number(n) || 1), pages.value);
      if (target === page.value) return;
      page.value = target;
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
      if (!job || TERMINAL.has(job.status) || cancelBusy.value) return;
      // The modal names the job being cancelled and the collection it is
      // writing into, so the operator confirms a specific job rather
      // than answering a yes/no question about an unnamed one.
      const confirmed = await askConfirm({
        title: t('ops.queue.cancel_title'),
        message: t('ops.queue.cancel_confirm'),
        details: [
          { label: t('common.job_id'), value: job.job_id },
          { label: t('common.scope'), value: job.database + '/' + job.collection },
          { label: t('ops.queue.filename'), value: job.filename || '—' },
          { label: t('ops.queue.status'), value: statusLabel(job.status) },
        ],
        confirmLabel: t('ops.queue.cancel'),
      });
      if (!confirmed) return;
      cancelBusy.value = true;
      detailErr.value = '';
      try {
        const { payload } = await api('POST', `/v1/jobs/${enc(job.job_id)}/cancel`);
        detail.value = payload;
      } catch (e) { detailErr.value = e.message; }
      finally { cancelBusy.value = false; }
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
      STATUS_FILTERS, items, total, page, pages, pageOffset,
      statusFilter, loading, listErr,
      detail, detailErr, cancelBusy, manualBusy,
      load, reload, setFilter, goToPage, openDetail, closeDetail, cancelJob,
      formatTs, pillClass, progressText, statusLabel, TERMINAL,
    };
  },
  components: { StatusBanner, BusyButton, EmptyState, UiPager },
  template: `
    <div class="ops-panel">
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('nav.queue') }}
            <span class="pill accent">GET /v1/jobs</span>
          </h3>
          <span class="section-sub">
            <busy-button class="sm" variant="ghost" :busy="manualBusy" :label="$t('common.refresh')"
                         @click="reload" />
          </span>
        </div>

        <div class="ops-filters">
          <button v-for="s in STATUS_FILTERS" :key="s || '__all'"
                  :class="['btn', 'sm', statusFilter === s ? 'primary' : 'ghost']"
                  @click="setFilter(s)">
            {{ s === '' ? $t('ops.queue.all') : statusLabel(s) }}
          </button>
        </div>

        <status-banner kind="error" :text="listErr" :retry="listErr ? reload : null" />
        <!-- Three list states that used to share one blank table: a
             first load, an honestly empty queue, and the rows themselves.
             Stale rows stay on screen during a failed refresh (the
             banner reports it), matching the browse convention. -->
        <div v-if="loading && !items.length" class="spinner"></div>
        <empty-state v-else-if="!loading && !listErr && !items.length"
                     state="empty" :text="$t('common.state.empty')" />

        <div v-else-if="items.length" class="data-table-wrap">
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
                  role="button" tabindex="0"
                  :aria-expanded="detail && detail.job_id === j.job_id ? 'true' : 'false'"
                  @click="openDetail(j.job_id)"
                  @keydown.enter.prevent="openDetail(j.job_id)"
                  @keydown.space.prevent="openDetail(j.job_id)">
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

        <!-- Shared pager (S6): page numbers, jump and totals — this list
             used to show only a bare count/offset line with prev/next. -->
        <ui-pager v-if="total" root-id="queue-pager" info-id="queue-pager-info"
                  first-id="btn-queue-first" prev-id="btn-queue-prev"
                  next-id="btn-queue-next" last-id="btn-queue-last"
                  jump-id="queue-jump" jump-btn-id="btn-queue-jump"
                  :page="page" :pages="pages"
                  :busy="loading" :show-jump="true"
                  :info-text="$t('ops.pager_info', { from: pageOffset + 1, to: pageOffset + items.length, total: total, page: page, pages: pages })"
                  @first="goToPage(1)" @prev="goToPage(page - 1)"
                  @next="goToPage(page + 1)" @last="goToPage(pages)"
                  @go="goToPage" />
      </div>

      <div v-if="detail" class="section ops-detail">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('ops.queue.detail') }}
            <span class="pill accent">SSE /v1/jobs/{id}/events</span>
          </h3>
          <span class="section-sub">
            <busy-button v-if="!TERMINAL.has(detail.status)" class="sm" variant="danger"
                         :busy="cancelBusy" :label="$t('ops.queue.cancel')"
                         :busy-label="$t('common.deleting')" @click="cancelJob" />
            <button class="btn sm ghost" @click="closeDetail">{{ $t('common.close') }}</button>
          </span>
        </div>

        <status-banner kind="error" :text="detailErr" />

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
