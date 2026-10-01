// =====================================================================
// ops-reindex.js -- Operations > Index rebuild.
//   POST /v1/databases/{db}/collections/{coll}/reindex
//        body {embed_model, canary_percent, batch_size} -> {job_id, index_ref}
//   GET  /v1/jobs/{job_id}                          poll rebuild progress
//   GET  /v1/evaluation/gates?... + /gates/{id}/checks  gate status
//   POST .../reindex/promote                         409 gate_blocked /
//                                                    409 reindex_no_canary
// =====================================================================
import { defineComponent, ref, computed, watch } from '../vue.esm-browser.prod.js';
import { store, api, t } from './app.js';
import {
  enc, formatTs, pillClass, progressText, statusLabel, TERMINAL,
  DEFAULT_SCOPE,
} from './ops-common.js';

export default defineComponent({
  name: 'OpsReindex',
  setup() {
    const db = ref(DEFAULT_SCOPE.db);
    const coll = ref(DEFAULT_SCOPE.coll);
    const embedModel = ref('');
    const canaryPercent = ref(0);
    const batchSize = ref(64);

    const submitting = ref(false);
    const formErr = ref('');
    const job = ref(null);
    const indexRef = ref(null);
    const promoted = ref(null);

    const gates = ref([]);
    const checks = ref([]);
    const gateErr = ref('');

    let timer = null;
    let started = false;

    const scopePath = () =>
      `/v1/databases/${enc(db.value)}/collections/${enc(coll.value)}`;
    const latestCheck = computed(() => checks.value[0] || null);

    async function loadJob(jobId) {
      try {
        const { payload } = await api('GET', `/v1/jobs/${enc(jobId)}`);
        job.value = payload;
      } catch (e) { formErr.value = e.message; }
    }

    function stopTimer() {
      if (timer) { clearInterval(timer); timer = null; }
    }

    function pollJob(jobId) {
      stopTimer();
      timer = setInterval(async () => {
        await loadJob(jobId);
        if (job.value && TERMINAL.has(job.value.status)) {
          stopTimer();
          if (job.value.status === 'done') await loadGates();
        }
      }, 3000);
    }

    async function submit() {
      if (!window.confirm(t('ops.reindex.confirm'))) return;
      submitting.value = true;
      formErr.value = '';
      promoted.value = null;
      try {
        const body = {
          embed_model: embedModel.value || null,
          canary_percent: Number(canaryPercent.value) || 0,
          batch_size: Number(batchSize.value) || 64,
        };
        const { payload } = await api('POST', `${scopePath()}/reindex`, body);
        indexRef.value = payload.index_ref;
        await loadJob(payload.job_id);
        pollJob(payload.job_id);
      } catch (e) { formErr.value = e.message; }
      finally { submitting.value = false; }
    }

    async function loadGates() {
      gateErr.value = '';
      try {
        const q = `/v1/evaluation/gates?database=${enc(db.value)}&collection=${enc(coll.value)}`;
        const { payload } = await api('GET', q);
        gates.value = (payload && payload.items) || [];
        if (gates.value.length) {
          const r = await api(
            'GET',
            `/v1/evaluation/gates/${enc(gates.value[0].gate_id)}/checks`,
          );
          checks.value = (r.payload && r.payload.items) || [];
        } else {
          checks.value = [];
        }
      } catch (e) { gateErr.value = e.message; }
    }

    async function promote() {
      if (!window.confirm(t('ops.reindex.promote_confirm'))) return;
      gateErr.value = '';
      try {
        const { payload } = await api('POST', `${scopePath()}/reindex/promote`);
        promoted.value = payload;
        stopTimer();
        await loadGates();
      } catch (e) { gateErr.value = e.message; }
    }

    function start() {
      if (started) return;
      started = true;
      loadGates();
    }
    function stop() {
      if (!started) return;
      started = false;
      stopTimer();
    }
    watch(() => store.view, (v) => (v === 'reindex' ? start() : stop()));
    if (store.view === 'reindex') start();

    return {
      db, coll, embedModel, canaryPercent, batchSize,
      submitting, formErr, job, indexRef, promoted,
      gates, checks, gateErr, latestCheck,
      submit, loadGates, promote,
      formatTs, pillClass, progressText, statusLabel, TERMINAL,
    };
  },
  template: `
    <div class="ops-panel">
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('nav.reindex') }}
            <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/reindex</span>
          </h3>
        </div>

        <div class="ops-scope">
          <label>{{ $t('common.database') }}
            <input v-model="db" type="text" spellcheck="false" />
          </label>
          <label>{{ $t('common.collection') }}
            <input v-model="coll" type="text" spellcheck="false" />
          </label>
        </div>

        <div class="ops-form-grid">
          <div class="form-group">
            <label class="form-label">{{ $t('ops.reindex.embed_model') }}</label>
            <input v-model="embedModel" type="text" spellcheck="false"
                   :placeholder="$t('ops.reindex.embed_keep')" />
            <span class="form-hint">{{ $t('ops.reindex.embed_hint') }}</span>
          </div>
          <div class="form-group">
            <label class="form-label">{{ $t('ops.reindex.canary_percent') }}</label>
            <input v-model.number="canaryPercent" type="number" min="0" max="100" step="1" />
            <span class="form-hint">{{ $t('ops.reindex.canary_hint') }}</span>
          </div>
          <div class="form-group">
            <label class="form-label">{{ $t('ops.reindex.batch_size') }}</label>
            <input v-model.number="batchSize" type="number" min="1" step="1" />
          </div>
        </div>

        <div v-if="formErr" class="empty error">{{ formErr }}</div>

        <div class="actions">
          <button class="btn primary" :disabled="submitting" @click="submit">
            <span v-if="submitting" class="btn-spinner"></span>
            {{ $t('ops.reindex.submit') }}
          </button>
        </div>
      </div>

      <div v-if="job" class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('ops.reindex.job') }}</h3>
          <span class="section-sub ops-mono">{{ indexRef }}</span>
        </div>
        <table class="info-table">
          <tr><th>{{ $t('common.job_id') }}</th><td class="ops-mono">{{ job.job_id }}</td></tr>
          <tr><th>{{ $t('ops.queue.status') }}</th><td><span :class="['pill', pillClass(job.status)]">{{ statusLabel(job.status) }}</span></td></tr>
          <tr><th>{{ $t('ops.queue.stage') }}</th><td>{{ job.stage ? statusLabel(job.stage) : '—' }}</td></tr>
          <tr><th>{{ $t('ops.queue.progress') }}</th><td>{{ progressText(job.progress) }}</td></tr>
          <tr><th>{{ $t('ops.queue.attempts') }}</th><td>{{ job.attempts }}/{{ job.max_attempts }}</td></tr>
          <tr v-if="job.error"><th>{{ $t('common.error') }}</th><td><span class="ops-mono">{{ job.error.code }}</span> — {{ job.error.message }}</td></tr>
          <tr><th>{{ $t('ops.queue.finished') }}</th><td>{{ formatTs(job.finished_ts) }}</td></tr>
        </table>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('ops.reindex.gate') }}</h3>
          <span class="section-sub">
            <button class="btn sm ghost" @click="loadGates">{{ $t('common.refresh') }}</button>
          </span>
        </div>

        <div v-if="gateErr" class="empty error">{{ gateErr }}</div>

        <div v-if="!gates.length" class="empty">
          {{ $t('ops.reindex.no_gate') }}
          <span class="hint">{{ $t('ops.reindex.no_gate_hint') }}</span>
        </div>

        <template v-else>
          <table class="info-table ops-gate-meta">
            <tr><th>{{ $t('common.gate_id') }}</th><td class="ops-mono">{{ gates[0].gate_id }}</td></tr>
            <tr><th>{{ $t('common.set_id') }}</th><td class="ops-mono">{{ gates[0].set_id }}</td></tr>
            <tr><th>{{ $t('common.baseline_run_id') }}</th><td class="ops-mono">{{ gates[0].baseline_run_id || '—' }}</td></tr>
          </table>

          <div v-if="!latestCheck" class="empty">{{ $t('ops.reindex.no_check') }}</div>
          <div v-else class="ops-check-card">
            <div class="ops-card-title">
              <span class="ops-mono">{{ latestCheck.check_id }}</span>
              <span :class="['pill', pillClass(latestCheck.status)]">{{ statusLabel(latestCheck.status) }}</span>
              <span class="muted">{{ formatTs(latestCheck.created_ts) }}</span>
            </div>
            <table class="info-table">
              <tr><th>{{ $t('common.candidate_ref') }}</th><td class="ops-mono">{{ latestCheck.candidate_ref }}</td></tr>
              <tr><th>{{ $t('common.run_id') }}</th><td class="ops-mono">{{ latestCheck.run_id || '—' }}</td></tr>
            </table>
          </div>

          <div class="actions">
            <button class="btn primary" @click="promote">{{ $t('ops.reindex.promote') }}</button>
          </div>
          <p class="ops-note">{{ $t('ops.reindex.promote_hint') }}</p>
        </template>

        <div v-if="promoted" class="ops-promoted">
          <span class="pill success">{{ $t('common.promoted') }}</span>
          <span class="ops-mono">{{ promoted.active_ref }}</span>
          <span class="muted">← {{ $t('common.retired') }}</span>
          <span class="ops-mono">{{ promoted.retired_ref }}</span>
        </div>
      </div>
    </div>
  `,
});
