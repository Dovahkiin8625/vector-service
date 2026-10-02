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
import { StatusBanner, BusyButton, askConfirm } from './feedback.js';
import { enc, DEFAULT_SCOPE } from './ops-common.js';

export default defineComponent({
  name: 'OpsConsistency',
  setup() {
    const db = ref(DEFAULT_SCOPE.db);
    const coll = ref(DEFAULT_SCOPE.coll);
    const scanning = ref(false);
    // Which button is running ('scan' | 'repair') so only the one that
    // was clicked shows the busy state.
    const scanningKind = ref('');
    const actionErr = ref('');
    const report = ref(null);

    const scopePath = () =>
      `/v1/databases/${enc(db.value)}/collections/${enc(coll.value)}`;

    async function scan(repair) {
      if (scanning.value) return;
      if (repair) {
        // Repair mutates both stores, so the modal spells out the scope
        // it will be applied to — confirm() could only ask "sure?".
        const confirmed = await askConfirm({
          title: t('ops.consistency.repair_title'),
          message: t('ops.consistency.repair_confirm'),
          details: [
            { label: t('common.database'), value: db.value },
            { label: t('common.collection'), value: coll.value },
          ],
          confirmLabel: t('ops.consistency.repair'),
        });
        if (!confirmed) return;
      }
      scanning.value = true;
      scanningKind.value = repair ? 'repair' : 'scan';
      actionErr.value = '';
      try {
        const { payload } = await api('POST', `${scopePath()}/consistency`, { repair });
        report.value = payload;
      } catch (e) { actionErr.value = e.message; }
      finally { scanning.value = false; scanningKind.value = ''; }
    }

    function start() { /* on-demand panel: no polling */ }
    watch(() => store.view, (v) => { if (v === 'consistency') start(); });

    return {
      db, coll, scanning, scanningKind, actionErr, report,
      scan: (repair) => scan(repair),
    };
  },
  components: { StatusBanner, BusyButton },
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
          <label>{{ $t('common.database') }}
            <input v-model="db" type="text" spellcheck="false" />
          </label>
          <label>{{ $t('common.collection') }}
            <input v-model="coll" type="text" spellcheck="false" />
          </label>
        </div>

        <div class="actions">
          <busy-button variant="primary" :busy="scanningKind === 'scan'" :label="$t('ops.consistency.scan')"
                       :busy-label="$t('ops.consistency.scanning')" :disabled="scanning"
                       @click="scan(false)" />
          <busy-button variant="danger" :busy="scanningKind === 'repair'"
                       :label="$t('ops.consistency.repair')"
                       :busy-label="$t('ops.consistency.repairing')" :disabled="scanning"
                       @click="scan(true)" />
        </div>
        <p class="ops-note">{{ $t('ops.consistency.repair_hint') }}</p>

        <status-banner kind="error" :text="actionErr"
                       :retry="actionErr ? () => scan(false) : null" />
      </div>

      <div v-if="report" class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('ops.consistency.report') }}
            <span :class="['pill', report.ok ? 'success' : 'danger']">
              {{ report.ok ? 'OK' : $t('ops.consistency.drift') }}
            </span>
            <span v-if="report.repaired" class="pill warn">{{ $t('ops.consistency.repaired') }}</span>
          </h3>
          <span class="section-sub ops-mono">{{ report.database }}/{{ report.collection }}</span>
        </div>

        <div class="ops-grid">
          <div class="kpi" :class="report.ok ? 'kpi--ok' : 'kpi--err'">
            <span class="label">{{ $t('ops.consistency.overall') }}</span>
            <span class="value">{{ report.ok ? 'OK' : $t('ops.consistency.drift') }}</span>
            <span class="sub">{{ $t('ops.consistency.physical_indexes') }}: {{ report.indexes.length }}</span>
          </div>
          <div class="kpi kpi--accent">
            <span class="label">{{ $t('ops.consistency.sqlite_leaves') }}</span>
            <span class="value">{{ report.corpus_leaves }}</span>
            <span class="sub">{{ $t('ops.consistency.source_truth') }}</span>
          </div>
        </div>

        <div v-for="ix in report.indexes" :key="ix.ref" class="ops-index-card">
          <div class="ops-card-title">
            <span class="ops-mono ops-ref">{{ ix.ref }}</span>
            <span v-if="ix.canary" class="pill warn">{{ $t('common.canary') }}</span>
            <span v-else class="pill accent">{{ $t('common.active') }}</span>
            <span :class="['pill', ix.ok ? 'success' : 'danger']">
              {{ ix.ok ? 'OK' : $t('ops.consistency.drift') }}
            </span>
          </div>
          <table class="info-table">
            <tr>
              <th>{{ $t('ops.consistency.milvus_rows') }}</th>
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
