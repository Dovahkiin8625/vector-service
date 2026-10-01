// =====================================================================
// ops-consistency.js -- Operations > SQLite <-> Milvus consistency.
//   POST /v1/databases/{db}/collections/{coll}/consistency
//        body {repair: false | true}
//        -> {corpus_leaves, indexes: [{ref, canary, milvus_rows,
//               missing_in_milvus[], orphans_in_milvus[], ok}], ok, repaired}
//   Repair needs an embedder loaded -> 503 embedder_unavailable.
// =====================================================================
import { defineComponent, ref, watch } from '../vue.esm-browser.prod.js';
import { store, api, t } from './app.js';
import { enc, DEFAULT_SCOPE } from './ops-common.js';

export default defineComponent({
  name: 'OpsConsistency',
  setup() {
    const db = ref(DEFAULT_SCOPE.db);
    const coll = ref(DEFAULT_SCOPE.coll);
    const scanning = ref(false);
    const actionErr = ref('');
    const report = ref(null);

    const scopePath = () =>
      `/v1/databases/${enc(db.value)}/collections/${enc(coll.value)}`;

    async function scan(repair) {
      if (repair && !window.confirm(t('ops.consistency.repair_confirm'))) return;
      scanning.value = true;
      actionErr.value = '';
      try {
        const { payload } = await api('POST', `${scopePath()}/consistency`, { repair });
        report.value = payload;
      } catch (e) { actionErr.value = e.message; }
      finally { scanning.value = false; }
    }

    function start() { /* on-demand panel: no polling */ }
    watch(() => store.view, (v) => { if (v === 'consistency') start(); });

    return {
      db, coll, scanning, actionErr, report,
      scan: (repair) => scan(repair),
    };
  },
  template: `
    <div class="ops-panel">
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('nav.consistency') }}
            <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/consistency</span>
          </h3>
        </div>

        <div class="ops-scope">
          <label>database
            <input v-model="db" type="text" spellcheck="false" />
          </label>
          <label>collection
            <input v-model="coll" type="text" spellcheck="false" />
          </label>
        </div>

        <div class="actions">
          <button class="btn primary" :disabled="scanning" @click="scan(false)">
            <span v-if="scanning" class="btn-spinner"></span>
            {{ $t('ops.consistency.scan') }}
          </button>
          <button class="btn danger" :disabled="scanning" @click="scan(true)">
            {{ $t('ops.consistency.repair') }}
          </button>
        </div>
        <p class="ops-note">{{ $t('ops.consistency.repair_hint') }}</p>

        <div v-if="actionErr" class="empty error">{{ actionErr }}</div>
      </div>

      <div v-if="report" class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('ops.consistency.report') }}
            <span :class="['pill', report.ok ? 'success' : 'danger']">
              {{ report.ok ? 'OK' : 'DRIFT' }}
            </span>
            <span v-if="report.repaired" class="pill warn">repaired</span>
          </h3>
          <span class="section-sub ops-mono">{{ report.database }}/{{ report.collection }}</span>
        </div>

        <div class="ops-grid">
          <div class="kpi" :class="report.ok ? 'kpi--ok' : 'kpi--err'">
            <span class="label">overall</span>
            <span class="value">{{ report.ok ? 'OK' : 'DRIFT' }}</span>
            <span class="sub">{{ report.indexes.length }} physical index(es)</span>
          </div>
          <div class="kpi kpi--accent">
            <span class="label">SQLite leaves</span>
            <span class="value">{{ report.corpus_leaves }}</span>
            <span class="sub">source of truth</span>
          </div>
        </div>

        <div v-for="ix in report.indexes" :key="ix.ref" class="ops-index-card">
          <div class="ops-card-title">
            <span class="ops-mono ops-ref">{{ ix.ref }}</span>
            <span v-if="ix.canary" class="pill warn">canary</span>
            <span v-else class="pill accent">active</span>
            <span :class="['pill', ix.ok ? 'success' : 'danger']">
              {{ ix.ok ? 'OK' : 'DRIFT' }}
            </span>
          </div>
          <table class="info-table">
            <tr>
              <th>Milvus rows</th>
              <td>{{ ix.milvus_rows }}</td>
            </tr>
            <tr>
              <th>{{ $t('ops.consistency.missing') }}</th>
              <td>
                <span :class="['ops-count', ix.missing_in_milvus.length ? 'bad' : 'good']">
                  {{ ix.missing_in_milvus.length }}
                </span>
                <details v-if="ix.missing_in_milvus.length" class="ops-id-details">
                  <summary>{{ $t('ops.consistency.show_ids') }}</summary>
                  <div class="ops-id-list ops-mono">
                    <div v-for="m in ix.missing_in_milvus" :key="m">{{ m }}</div>
                  </div>
                </details>
              </td>
            </tr>
            <tr>
              <th>{{ $t('ops.consistency.orphans') }}</th>
              <td>
                <span :class="['ops-count', ix.orphans_in_milvus.length ? 'bad' : 'good']">
                  {{ ix.orphans_in_milvus.length }}
                </span>
                <details v-if="ix.orphans_in_milvus.length" class="ops-id-details">
                  <summary>{{ $t('ops.consistency.show_ids') }}</summary>
                  <div class="ops-id-list ops-mono">
                    <div v-for="o in ix.orphans_in_milvus" :key="o">{{ o }}</div>
                  </div>
                </details>
              </td>
            </tr>
          </table>
        </div>
      </div>
    </div>
  `,
});
