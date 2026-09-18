// Databases panel: list + expandable detail (metadata + collection list).
import { defineComponent, ref, onMounted } from '../vue.esm-browser.prod.js';
import { store, api, enc, extractApiError } from './app.js';

export default defineComponent({
  name: 'DatabasesPanel',
  setup() {
    const list = ref([]);
    const details = ref(Object.create(null));
    const loading = ref(Object.create(null));
    const expanded = ref(new Set());

    async function refresh() {
      try {
        const { payload } = await api('GET', '/v1/databases');
        const newList = (payload && payload.databases) || [];
        store.databases.list = newList;
        list.value = newList;
        const live = new Set(newList);
        Object.keys(details.value).forEach(k => { if (!live.has(k)) delete details.value[k]; });
        expanded.value.forEach(k => { if (!live.has(k)) expanded.value.delete(k); });
      } catch (_e) {}
    }

    async function loadDetail(name) {
      if (details.value[name]) return details.value[name];
      loading.value[name] = true;
      try {
        const [r1, r2] = await Promise.all([
          api('GET', '/v1/databases/' + enc(name)).catch(() => ({ payload: null })),
          api('GET', '/v1/databases/' + enc(name) + '/collections').catch(() => ({ payload: null })),
        ]);
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

    async function dropDb(name) {
      if (!confirm('确认删除数据库 "' + name + '" 及其下所有集合?')) return;
      try { await api('DELETE', '/v1/databases/' + enc(name)); }
      catch (e) { alert('删除失败: ' + extractApiError(e, 'unknown')); return; }
      delete details.value[name];
      expanded.value.delete(name);
      await refresh();
    }

    onMounted(() => {
      refresh();
      window.addEventListener('refresh-dbs', refresh);
    });

    return {
      store,
      list, details, loading, expanded, refresh, toggleDetail, dropDb };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('nav.databases') }} <span class="pill accent">GET /v1/databases</span></h3>
        </div>
        <div class="actions">
          <button id="btn-refresh-dbs" class="btn primary" @click="refresh">{{ $t('common.refresh') }}</button>
          <button id="btn-open-new-db" class="btn" @click="store.modals.newDb = true">{{ $t('common.new_db') }}</button>
        </div>
        <div class="list" id="dbs-list">
          <div v-if="!list.length" class="empty">{{ $t('common.click_refresh') }}</div>
          <template v-for="name in list" :key="name">
            <div class="list-item" :data-db-name="name" @click="toggleDetail(name)">
              <span class="name">{{ name }}</span>
              <span class="meta">
                <template v-if="details[name]">{{ details[name].total }} {{ $t('databases.coll_count') }} / {{ details[name].total }}</template>
                <template v-else>— / —</template>
              </span>
              <span style="color:var(--text-muted);font-size:12px;">{{ expanded.has(name) ? '▾' : '▸' }}</span>
              <button class="btn sm danger" @click.stop="dropDb(name)">{{ $t('common.delete') }}</button>
            </div>
            <div v-if="expanded.has(name)" class="field-card" :data-detail-for-db="name"
                 style="margin-left:0;margin-top:4px;padding:14px 16px;">
              <div v-if="!details[name]" class="empty"><span class="spinner"></span> {{ $t('databases.loading') }}</div>
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
