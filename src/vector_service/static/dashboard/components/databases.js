// Databases panel: list + expandable detail (metadata + collection list).
import { defineComponent, ref, computed, onMounted, onUnmounted } from '../vue.esm-browser.prod.js';
import { store, api, enc, extractApiError, t } from './app.js';
import { StatusBanner, BusyButton, EmptyState, askConfirm } from './feedback.js';

export default defineComponent({
  name: 'DatabasesPanel',
  setup() {
    const list = ref([]);
    const details = ref(Object.create(null));
    const loading = ref(Object.create(null));
    // Detail-load failures, keyed by database name: without these the
    // expansion silently rendered "no metadata, no collections" — a
    // failed load looked exactly like an empty database.
    const detailErr = ref(Object.create(null));
    const expanded = ref(new Set());
    // List-load failure and the outcome of the last destructive call.
    const loadErr = ref('');
    const opStatus = ref('idle');
    const opMsg = ref('');
    const busyKey = ref('');

    function opFailed(prefix, e) {
      opStatus.value = 'error';
      opMsg.value = t(prefix) + extractApiError(e, t('common.unknown'));
    }

    async function refresh() {
      loadErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/databases');
        const newList = (payload && payload.databases) || [];
        store.databases.list = newList;
        list.value = newList;
        const live = new Set(newList);
        Object.keys(details.value).forEach(k => { if (!live.has(k)) delete details.value[k]; });
        Object.keys(detailErr.value).forEach(k => { if (!live.has(k)) delete detailErr.value[k]; });
        expanded.value.forEach(k => { if (!live.has(k)) expanded.value.delete(k); });
      } catch (e) {
        list.value = [];
        loadErr.value = t('databases.list_failed') + extractApiError(e, t('common.unknown'));
      }
    }

    async function loadDetail(name) {
      if (details.value[name]) return details.value[name];
      loading.value[name] = true;
      delete detailErr.value[name];
      try {
        // Both halves are optional server-side, so each failure is
        // captured rather than thrown — but it is recorded, not
        // swallowed, so the panel can offer a retry.
        const [r1, r2] = await Promise.all([
          api('GET', '/v1/databases/' + enc(name)).catch(e => ({ payload: null, error: e })),
          api('GET', '/v1/databases/' + enc(name) + '/collections').catch(e => ({ payload: null, error: e })),
        ]);
        const failed = r1.error || r2.error;
        if (failed) {
          detailErr.value[name] = t('databases.detail_failed') + extractApiError(failed, t('common.unknown'));
          return null;
        }
        const info = (r1 && r1.payload) || { name, metadata: {} };
        const colls = (r2 && r2.payload && r2.payload.collections) || [];
        const data = { name, metadata: info.metadata || {}, collections: colls, total: colls.length };
        details.value[name] = data;
        return data;
      } finally {
        loading.value[name] = false;
      }
    }

    async function toggleDetail(name) {
      if (expanded.value.has(name)) { expanded.value.delete(name); return; }
      expanded.value.add(name);
      await loadDetail(name);
    }

    async function retryDetail(name) {
      delete detailErr.value[name];
      await loadDetail(name);
    }

    async function reload() {
      busyKey.value = 'list';
      try { await refresh(); } finally { busyKey.value = ''; }
    }

    async function dropDb(name) {
      const confirmed = await askConfirm({
        title: t('databases.confirm_drop_title'),
        message: t('databases.confirm_drop', { name }),
        details: [{ label: t('common.database'), value: name }],
        confirmLabel: t('common.delete'),
      });
      if (!confirmed) return;
      busyKey.value = 'drop:' + name;
      try {
        await api('DELETE', '/v1/databases/' + enc(name));
      } catch (e) {
        opFailed('common.delete_failed', e);
        return;
      } finally {
        busyKey.value = '';
      }
      delete details.value[name];
      delete detailErr.value[name];
      expanded.value.delete(name);
      opStatus.value = 'ok';
      opMsg.value = t('databases.dropped', { name });
      await refresh();
    }

    onMounted(() => {
      refresh();
      window.addEventListener('refresh-dbs', refresh);
    });
    // Without this the handler outlived the component: every remount
    // added another one, so one dispatched event fired N refreshes (B14).
    onUnmounted(() => window.removeEventListener('refresh-dbs', refresh));

    const opKind = computed(() => (opStatus.value === 'error' ? 'error' : 'success'));

    return {
      store,
      list, details, loading, detailErr, expanded, loadErr, opKind, opMsg, busyKey,
      refresh, reload, toggleDetail, retryDetail, dropDb };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('nav.databases') }} <span class="pill accent">GET /v1/databases</span></h3>
        </div>
        <status-banner kind="error" :text="loadErr" :retry="loadErr ? refresh : null" />
        <status-banner :kind="opKind" :text="opMsg" />
        <div class="actions">
          <busy-button id="btn-refresh-dbs" :busy="busyKey === 'list'" :label="$t('common.refresh')"
                       @click="reload" />
          <button id="btn-open-new-db" class="btn" @click="store.modals.newDb = true">{{ $t('common.new_db') }}</button>
        </div>
        <div class="list" id="dbs-list">
          <empty-state v-if="!list.length" state="idle" :text="$t('common.click_refresh')" />
          <template v-for="name in list" :key="name">
            <div class="list-item" :data-db-name="name" @click="toggleDetail(name)">
              <span class="name">{{ name }}</span>
              <!-- Two different numbers: collections and metadata
                   entries. This used to render the collection count on
                   both sides of the slash, so the row read
                   "3 collection count / 3" with no second source (B7). -->
              <span class="meta">
                <template v-if="details[name]">{{ details[name].total }} {{ $t('databases.coll_count') }} / {{ Object.keys(details[name].metadata || {}).length }} {{ $t('databases.meta_count') }}</template>
                <template v-else>— / —</template>
              </span>
              <span style="color:var(--text-muted);font-size:12px;">{{ expanded.has(name) ? '▾' : '▸' }}</span>
              <busy-button class="sm" variant="danger" :busy="busyKey === 'drop:' + name"
                           :label="$t('common.delete')" :busy-label="$t('common.deleting')"
                           @click.stop="dropDb(name)" />
            </div>
            <div v-if="expanded.has(name)" class="field-card" :data-detail-for-db="name"
                 style="margin-left:0;margin-top:4px;padding:14px 16px;">
              <empty-state v-if="detailErr[name]" state="error" :text="detailErr[name]"
                           :retry="() => retryDetail(name)" />
              <empty-state v-else-if="!details[name]" state="loading" :text="$t('databases.loading')" />
              <template v-else>
                <div class="kb-result">
                  <div class="stat"><span class="key">{{ $t('databases.coll_count') }}</span><span class="val accent">{{ details[name].total }}</span></div>
                  <div class="stat"><span class="key">{{ $t('databases.meta_count') }}</span><span class="val">{{ Object.keys(details[name].metadata || {}).length }}</span></div>
                </div>
                <details class="collapsible" open>
                  <summary>{{ $t('common.basic_info') }}</summary>
                  <div class="body">
                    <table class="info-table">
                      <tr><th>{{ $t('common.name') }}</th><td>{{ name }}</td></tr>
                      <tr><th>{{ $t('databases.coll_count') }}</th><td>{{ details[name].total }}</td></tr>
                    </table>
                  </div>
                </details>
                <details class="collapsible" open v-if="Object.keys(details[name].metadata || {}).length">
                  <summary>{{ $t('databases.backend_meta') }}</summary>
                  <div class="body">
                    <table class="info-table">
                      <tr v-for="(v, k) in details[name].metadata" :key="k">
                        <th>{{ k }}</th><td>{{ JSON.stringify(v) }}</td>
                      </tr>
                    </table>
                  </div>
                </details>
                <details class="collapsible" open v-if="details[name].collections.length">
                  <summary>{{ $t('databases.coll_list') }} ({{ details[name].collections.length }})</summary>
                  <div class="body">
                    <div class="list">
                      <div v-for="c in details[name].collections" :key="c" class="list-item" style="cursor:default;">
                        <span class="name">{{ c }}</span>
                      </div>
                    </div>
                  </div>
                </details>
                <div v-if="!details[name].collections.length" class="empty">{{ $t('databases.no_collections') }}</div>
              </template>
            </div>
          </template>
        </div>
      </div>
      <div class="empty hint">{{ $t('databases.hint') }}</div>
    </div>
  `,
});
