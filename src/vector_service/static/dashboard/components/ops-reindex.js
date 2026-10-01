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
import { intError } from './util.js';
import { StatusBanner, BusyButton, EmptyState, askConfirm } from './feedback.js';
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
    // Promote and the manual gate refresh had no in-flight state: both
    // could be fired repeatedly while the previous call was still out.
    const promoting = ref(false);
    const refreshing = ref(false);

    const gates = ref([]);
    const checks = ref([]);
    // Gate-list load failure. Promotion failures get their own channel
    // below: a 409 from promote used to land here, where the banner's
    // retry button re-runs the gate LIST, which is not the failed call.
    const gateErr = ref('');
    const promoteErr = ref('');

    let timer = null;
    let started = false;

    const scopePath = () =>
      `/v1/databases/${enc(db.value)}/collections/${enc(coll.value)}`;
    // Both of these mirror what the promote endpoint itself resolves —
    // they are not a client-side "first item wins" guess.
    //   * regression_gates is UNIQUE(database, collection), so a scope
    //     has at most one gate; the list is already filtered by scope
    //     and the server reads that same row via get_gate_for_collection.
    //   * list_gate_checks orders newest first, and the server reads
    //     get_latest_gate_check — the newest row.
    // There is therefore nothing to choose between, and a selector here
    // would imply the client could promote against a gate the server
    // would not consult.
    const scopeGate = computed(() => gates.value[0] || null);
    const latestCheck = computed(() => checks.value[0] || null);
    // S6 numeric rules: cleared number boxes used to fall through the
    // `Number(x) || default` coercion silently. Both fields show an
    // inline hint and hold the submit button until they are whole
    // numbers in range (canary 0..100 — 0 is a legitimate "no canary").
    const canaryErr = computed(() => intError(canaryPercent.value, 0, 100));
    const batchErr = computed(() => intError(batchSize.value, 1, null));
    // The endpoint's own gate rule: with a gate registered, promotion is
    // allowed only when the latest check passed. With no gate the canary
    // can be promoted directly (a 409 still comes back if none is parked).
    const canPromote = computed(() => {
      if (!gates.value.length) return true;
      return !!latestCheck.value && latestCheck.value.status === 'passed';
    });
    const promoteBlockReason = computed(() => {
      if (canPromote.value) return '';
      if (!latestCheck.value) return t('ops.reindex.block_no_check');
      return t('ops.reindex.block_check_failed', {
        status: statusLabel(latestCheck.value.status),
      });
    });

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
      if (submitting.value) return;
      if (canaryErr.value || batchErr.value) return;
      const body = {
        embed_model: embedModel.value || null,
        canary_percent: Number(canaryPercent.value) || 0,
        batch_size: Number(batchSize.value) || 64,
      };
      // The modal restates the request body that is about to be sent, so
      // the operator confirms the actual scope/canary/model rather than
      // answering an unqualified "rebuild this index?".
      const confirmed = await askConfirm({
        title: t('ops.reindex.confirm'),
        message: t('ops.reindex.confirm'),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: coll.value },
          { label: t('ops.reindex.canary_percent'), value: String(body.canary_percent) },
          { label: t('ops.reindex.embed_model'), value: body.embed_model || t('ops.reindex.embed_keep') },
        ],
        confirmLabel: t('ops.reindex.submit'),
      });
      if (!confirmed) return;
      submitting.value = true;
      formErr.value = '';
      promoteErr.value = '';
      promoted.value = null;
      try {
        const { payload } = await api('POST', `${scopePath()}/reindex`, body);
        indexRef.value = payload.index_ref;
        await loadJob(payload.job_id);
        pollJob(payload.job_id);
      } catch (e) { formErr.value = e.message; }
      finally { submitting.value = false; }
    }

    async function loadGates() {
      refreshing.value = true;
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
      finally { refreshing.value = false; }
    }

    async function promote() {
      if (promoting.value || !canPromote.value) return;
      const gate = scopeGate.value;
      // Promotion retires the currently active index and makes the
      // candidate live, so the modal names both the gate being cleared
      // and the run that produced the candidate.
      const confirmed = await askConfirm({
        title: t('ops.reindex.promote_title'),
        message: t('ops.reindex.promote_confirm'),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: coll.value },
          { label: t('common.gate_id'), value: gate ? gate.gate_id : '—' },
          { label: t('common.candidate_ref'), value: latestCheck.value ? latestCheck.value.candidate_ref : '—' },
          { label: t('common.run_id'), value: (latestCheck.value && latestCheck.value.run_id) || '—' },
        ],
        confirmLabel: t('ops.reindex.promote'),
      });
      if (!confirmed) return;
      promoting.value = true;
      // Not gateErr: that banner belongs to the gate LIST load, and its
      // retry button re-runs loadGates() — the wrong call for a failed
      // promote. A 409 gate_blocked / reindex_no_canary is reported next
      // to the button that produced it.
      promoteErr.value = '';
      try {
        const { payload } = await api('POST', `${scopePath()}/reindex/promote`);
        promoted.value = payload;
        stopTimer();
        await loadGates();
      } catch (e) { promoteErr.value = e.message; }
      finally { promoting.value = false; }
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
      canaryErr, batchErr,
      submitting, formErr, job, indexRef, promoted,
      promoting, refreshing,
      gates, checks, gateErr, promoteErr, scopeGate, latestCheck,
      canPromote, promoteBlockReason,
      submit, loadGates, promote,
      formatTs, pillClass, progressText, statusLabel, TERMINAL,
    };
  },
  components: { StatusBanner, BusyButton, EmptyState },
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
            <label class="form-label" for="reindex-embed-model">{{ $t('ops.reindex.embed_model') }}</label>
            <input id="reindex-embed-model" v-model="embedModel" type="text" spellcheck="false"
                   :placeholder="$t('ops.reindex.embed_keep')" />
            <span class="form-hint">{{ $t('ops.reindex.embed_hint') }}</span>
          </div>
          <div class="form-group">
            <label class="form-label" for="reindex-canary">{{ $t('ops.reindex.canary_percent') }}</label>
            <input id="reindex-canary" v-model.number="canaryPercent" type="number" min="0" max="100" step="1" />
            <span class="form-hint">{{ $t('ops.reindex.canary_hint') }}</span>
            <span class="form-hint" v-if="canaryErr">{{ canaryErr }}</span>
          </div>
          <div class="form-group">
            <label class="form-label" for="reindex-batch">{{ $t('ops.reindex.batch_size') }}</label>
            <input id="reindex-batch" v-model.number="batchSize" type="number" min="1" step="1" />
            <span class="form-hint" v-if="batchErr">{{ batchErr }}</span>
          </div>
        </div>

        <status-banner kind="error" :text="formErr" />

        <div class="actions">
          <busy-button :busy="submitting" :disabled="!!canaryErr || !!batchErr" :label="$t('ops.reindex.submit')"
                       :busy-label="$t('ops.reindex.submitting')" @click="submit" />
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
            <busy-button class="sm" variant="ghost" :busy="refreshing"
                         :label="$t('common.refresh')"
                         :busy-label="$t('ops.reindex.refreshing')" @click="loadGates" />
          </span>
        </div>

        <status-banner kind="error" :text="gateErr"
                       :retry="gateErr ? loadGates : null" />

        <empty-state v-if="!gateErr && !gates.length" state="empty"
                     :text="$t('ops.reindex.no_gate')"
                     :hint="$t('ops.reindex.no_gate_hint')" />

        <template v-else-if="gates.length">
          <table class="info-table ops-gate-meta">
            <tr><th>{{ $t('common.gate_id') }}</th><td class="ops-mono">{{ scopeGate.gate_id }}</td></tr>
            <tr><th>{{ $t('common.set_id') }}</th><td class="ops-mono">{{ scopeGate.set_id }}</td></tr>
            <tr><th>{{ $t('common.baseline_run_id') }}</th><td class="ops-mono">{{ scopeGate.baseline_run_id || '—' }}</td></tr>
          </table>

          <empty-state v-if="!latestCheck" state="empty" :text="$t('ops.reindex.no_check')" />
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
          <p class="ops-note">{{ $t('ops.reindex.gate_scope_note') }}</p>
        </template>

        <!-- Promotion sits outside both branches above. It used to live
             inside the gates branch, so a scope with no gate — the very
             case whose empty state says the canary can be promoted
             directly — had no button at all (B8). -->
        <status-banner kind="error" :text="promoteErr" />
        <div class="actions">
          <busy-button :busy="promoting" :disabled="!canPromote"
                       :label="$t('ops.reindex.promote')"
                       :busy-label="$t('ops.reindex.promoting')" @click="promote" />
        </div>
        <p class="ops-note">{{ promoteBlockReason || $t('ops.reindex.promote_hint') }}</p>

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
