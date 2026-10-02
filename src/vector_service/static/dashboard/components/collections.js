// Collections panel: list + create (modal) + expandable detail (schema + indexes + index management).
import { defineComponent, ref, computed, watch, onMounted, onUnmounted } from '../vue.esm-browser.prod.js';
import { t, store, api, enc, extractApiError } from './app.js';
import { StatusBanner, BusyButton, EmptyState, askConfirm } from './feedback.js';

// Quick-fill params for the new-index form: the two shapes operators
// actually tune, offered as one-click templates instead of a blank
// JSON box everyone has to look up.
const INDEX_PARAM_TEMPLATES = {
  hnsw: { M: 16, efConstruction: 200 },
  ivf: { nlist: 128 },
};

export default defineComponent({
  name: 'CollectionsPanel',
  setup() {
    const db = ref('');
    const dbs = ref([]);
    const colls = ref([]);
    const detailCache = ref(Object.create(null));
    // Detail-load failures, keyed like detailCache. Without this the
    // expansion kept rendering "loading…" forever: the failure was
    // swallowed and nothing was ever written to the cache.
    const detailErr = ref(Object.create(null));
    // Per-collection state for the "new index" form, keyed like
    // detailCache and seeded in loadDetail from the collection's own
    // metric. The form used to hard-code `selected` on cosine/HNSW, so
    // on a collection built with l2 or ip the pre-selected metric
    // disagreed with the collection and the index it created did not
    // match the data already in it (B11).
    const newIndex = ref(Object.create(null));
    const expanded = ref(new Set());
    // List-level load failure (databases or collections).
    const loadErr = ref('');
    // Outcome of the last mutating call (drop collection / index).
    const opStatus = ref('idle');
    const opMsg = ref('');
    // Which control is in flight: 'drop:<name>' | 'index:<coll>:<field>'
    // | 'newindex:<coll>'. Keeps one row's spinner off its siblings.
    const busyKey = ref('');

    function opOk(msg) { opStatus.value = 'ok'; opMsg.value = msg; }
    function opFailed(prefix, e) {
      opStatus.value = 'error';
      opMsg.value = t(prefix) + extractApiError(e, t('common.unknown'));
    }

    async function refreshDbs() {
      loadErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        store.databases.list = dbs.value;
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
      } catch (e) {
        dbs.value = [];
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }

    async function refreshColls() {
      if (!db.value) { colls.value = []; return; }
      loadErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections');
        colls.value = (payload && payload.collections) || [];
        store.collections.list = colls.value;
        const live = new Set(colls.value);
        // detailCache/detailErr/newIndex are keyed "db::coll" while
        // `live` holds bare collection names, so the prune has to compare
        // the coll half. Comparing the whole key never matched and threw
        // away every cached detail on each refresh.
        const collOf = (k) => k.slice(k.indexOf('::') + 2);
        Object.keys(detailCache.value).forEach(k => {
          if (!live.has(collOf(k))) delete detailCache.value[k];
        });
        Object.keys(detailErr.value).forEach(k => {
          if (!live.has(collOf(k))) delete detailErr.value[k];
        });
        Object.keys(newIndex.value).forEach(k => {
          if (!live.has(collOf(k))) delete newIndex.value[k];
        });
        expanded.value.forEach(k => { if (!live.has(collOf(k))) expanded.value.delete(k); });
      } catch (e) {
        colls.value = [];
        loadErr.value = t('collections.list_failed') + extractApiError(e, t('common.unknown'));
      }
    }

    async function loadDetail(name) {
      const key = db.value + '::' + name;
      if (detailCache.value[key]) return detailCache.value[key];
      delete detailErr.value[key];
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections/' + enc(name));
        detailCache.value[key] = payload;
        // Seed the new-index form from this collection rather than from
        // constants: metric follows the collection, index type keeps its
        // own sensible default.
        newIndex.value[key] = {
          field: (payload && payload.vector_field) || '',
          metric: (payload && payload.metric) || 'cosine',
          type: 'HNSW',
          params: '{}',
        };
        return payload;
      } catch (e) {
        // Recorded, not swallowed: the expansion below renders the
        // reason plus a retry instead of spinning forever.
        detailErr.value[key] = t('collections.detail_failed') + extractApiError(e, t('common.unknown'));
        return null;
      }
    }
    // Retry entry point for the detail error state: drop the remembered
    // failure, then load again.
    async function reloadDetail(name) {
      const key = db.value + '::' + name;
      delete detailErr.value[key];
      await loadDetail(name);
    }

    async function toggleDetail(name) {
      const key = db.value + '::' + name;
      if (expanded.value.has(key)) { expanded.value.delete(key); return; }
      expanded.value.add(key);
      await loadDetail(name);
    }

    async function dropColl(name) {
      const confirmed = await askConfirm({
        title: t('collections.confirm_drop_title'),
        message: t('collections.confirm_drop', { name: db.value + '/' + name }),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: name },
        ],
        confirmLabel: t('common.delete'),
      });
      if (!confirmed) return;
      busyKey.value = 'drop:' + name;
      try {
        await api('DELETE', '/v1/databases/' + enc(db.value) + '/collections/' + enc(name));
      } catch (e) {
        opFailed('common.delete_failed', e);
        return;
      } finally {
        busyKey.value = '';
      }
      const key = db.value + '::' + name;
      delete detailCache.value[key];
      delete detailErr.value[key];
      delete newIndex.value[key];
      expanded.value.delete(key);
      opOk(t('collections.dropped', { name }));
      await refreshColls();
    }

    async function dropIndex(coll, fieldName) {
      const confirmed = await askConfirm({
        title: t('collections.confirm_drop_index_title'),
        message: t('collections.confirm_drop_index', { field: fieldName }),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: coll },
          { label: t('collections.index_field'), value: fieldName },
        ],
        confirmLabel: t('common.delete'),
      });
      if (!confirmed) return;
      busyKey.value = 'index:' + coll + ':' + fieldName;
      try {
        await api('DELETE',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll) + '/index?field_name=' + enc(fieldName));
        const key = db.value + '::' + coll;
        delete detailCache.value[key];
        await loadDetail(coll);
        opOk(t('collections.index_dropped', { field: fieldName }));
      } catch (e) { opFailed('collections.drop_index_failed', e); }
      finally { busyKey.value = ''; }
    }

    async function createIndex(coll, body) {
      busyKey.value = 'newindex:' + coll;
      try {
        await api('POST',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll) + '/index', body);
        const key = db.value + '::' + coll;
        delete detailCache.value[key];
        await loadDetail(coll);
        opOk(t('collections.index_created', { field: body.field_name }));
      } catch (e) { opFailed('collections.create_index_failed', e); }
      finally { busyKey.value = ''; }
    }

    function opErr(key) {
      opStatus.value = 'error';
      opMsg.value = t(key);
    }

    function fillIndexParams(name, which) {
      const form = newIndex.value[db.value + '::' + name];
      if (!form) return;
      form.params = which === 'clear'
        ? '{}'
        : JSON.stringify(INDEX_PARAM_TEMPLATES[which], null, 2);
    }

    function submitNewIndex(coll) {
      const form = newIndex.value[db.value + '::' + coll];
      if (!form) return;
      const raw = form.params.trim();
      let params = {};
      if (raw) {
        let parsed;
        try { parsed = JSON.parse(raw); } catch (_e) { parsed = undefined; }
        // JSON.parse('[]') / ('5') / ('null') all succeed but are not the
        // mapping the endpoint expects, so "it parses" is not enough.
        if (parsed === undefined || parsed === null
            || typeof parsed !== 'object' || Array.isArray(parsed)) {
          opErr('collections.err.bad_params');
          return;
        }
        params = parsed;
      }
      createIndex(coll, {
        field_name: form.field, metric_type: form.metric,
        index_type: form.type, params: params,
      });
    }

    // Busy wrappers for the two reload buttons, which had no in-flight
    // state at all (a slow refresh looked like a dead button).
    async function reloadDbs() {
      busyKey.value = 'dbs';
      try { await refreshDbs(); } finally { busyKey.value = ''; }
    }
    async function reloadColls() {
      busyKey.value = 'colls';
      try { await refreshColls(); } finally { busyKey.value = ''; }
    }

    const opKind = computed(() => (opStatus.value === 'error' ? 'error' : 'success'));

    watch(db, () => { refreshColls(); });
    onMounted(() => {
      refreshDbs();
      window.addEventListener('refresh-colls', refreshColls);
      window.addEventListener('refresh-dbs', refreshDbs);
    });
    // Listeners registered here used to survive unmount, so a remount
    // stacked a second pair and one event triggered two refreshes (B14).
    onUnmounted(() => {
      window.removeEventListener('refresh-colls', refreshColls);
      window.removeEventListener('refresh-dbs', refreshDbs);
    });

    return {
      store,
      db, dbs, colls, detailCache, detailErr, newIndex, expanded, loadErr,
      opKind, opMsg, busyKey, refreshDbs, reloadDbs, refreshColls, reloadColls,
             toggleDetail, reloadDetail, dropColl, dropIndex, createIndex, submitNewIndex,
      fillIndexParams };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <!-- Panel-level outcome strip: the mutating calls here (drop
           collection, drop/create index) had no visible result at all
           beyond the list happening to change. -->
      <status-banner kind="error" :text="loadErr" :retry="loadErr ? refreshDbs : null" />
      <status-banner :kind="opKind" :text="opMsg" />
      <div class="section">
        <div class="section-head"><h3 class="section-title">{{ $t('collections.select_db') }}</h3></div>
        <div class="row split">
          <div class="row">
            <label>{{ $t('collections.current_db') }}</label>
            <select id="colls-db" v-model="db">
              <option v-for="d in dbs" :key="d" :value="d">{{ d }}</option>
              <option v-if="!dbs.length" value="">{{ $t('collections.no_dbs') }}</option>
            </select>
          </div>
          <div class="row">
            <label>&nbsp;</label>
            <busy-button id="btn-colls-refresh-db" variant="ghost" :busy="busyKey === 'dbs'"
                         :label="'↻ ' + $t('collections.reload_dbs')" @click="reloadDbs" />
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('collections.list_title') }} <span class="pill accent">GET /v1/databases/{db}/collections</span></h3>
        </div>
        <div class="actions">
          <busy-button id="btn-refresh-colls" variant="ghost" :busy="busyKey === 'colls'" :label="$t('common.refresh')"
                       :disabled="!db" @click="reloadColls" />
          <button id="btn-open-new-coll" class="btn" :disabled="!db" @click="store.modals.newColl = true">{{ $t('collections.new') }}</button>
        </div>
        <div class="list" id="colls-list">
          <empty-state v-if="!db" state="idle" :text="$t('collections.pick_db_first')" />
          <empty-state v-else-if="!colls.length" state="empty" :text="$t('collections.empty')" />
          <template v-for="name in colls" :key="db + '::' + name">
            <div class="list-item" :data-coll-key="db + '::' + name" role="button" tabindex="0"
                 :aria-expanded="expanded.has(db + '::' + name) ? 'true' : 'false'"
                 @click="toggleDetail(name)"
                 @keydown.enter.prevent="toggleDetail(name)"
                 @keydown.space.prevent="toggleDetail(name)">
              <span class="name">{{ db }} / {{ name }}</span>
              <span class="meta">
                <!-- Same S5 rule as the db list: no "— · — · —" stack
                     before the detail lands, caret detached from meta. -->
                <template v-if="detailCache[db + '::' + name]">
                  {{ detailCache[db + '::' + name].metric || '—' }} · {{ $t('collections.dim_value', { dim: detailCache[db + '::' + name].dim || '—' }) }} · {{ formatCount(detailCache[db + '::' + name].count) }}
                </template>
              </span>
              <span class="expand-caret" aria-hidden="true">{{ expanded.has(db + '::' + name) ? '▾' : '▸' }}</span>
              <!-- S6 danger containment: same expand-first rule as the
                   database rows — the destructive action appears only
                   once the row's detail panel is open. -->
              <busy-button v-if="expanded.has(db + '::' + name)" class="sm" variant="danger"
                           :busy="busyKey === 'drop:' + name"
                           :label="$t('common.delete')" :busy-label="$t('common.deleting')"
                           @click.stop="dropColl(name)" />
            </div>
            <div v-if="expanded.has(db + '::' + name)" class="field-card" :data-detail-for="db + '::' + name"
                 style="margin-left:0;margin-top:4px;padding:14px 16px;">
              <!-- A failed detail load used to leave this spinning
                   forever: the error was swallowed and nothing was ever
                   written to detailCache. -->
              <empty-state v-if="detailErr[db + '::' + name]" state="error"
                           :text="detailErr[db + '::' + name]"
                           :retry="() => reloadDetail(name)" />
              <empty-state v-else-if="!detailCache[db + '::' + name]" state="loading"
                           :text="$t('collections.loading_detail')" />
              <template v-else>
                <div class="kb-result">
                  <div class="stat"><span class="key">{{ $t('collections.stat.dim') }}</span><span class="val accent">{{ detailCache[db + '::' + name].dim }}</span></div>
                  <div class="stat"><span class="key">{{ $t('collections.stat.count') }}</span><span class="val">{{ formatCount(detailCache[db + '::' + name].count) }}</span></div>
                  <div class="stat"><span class="key">{{ $t('collections.stat.metric') }}</span><span class="val">{{ detailCache[db + '::' + name].metric }}</span></div>
                  <div class="stat"><span class="key">{{ $t('collections.stat.fields') }}</span><span class="val">{{ detailCache[db + '::' + name].primary_field }} · {{ detailCache[db + '::' + name].vector_field }}</span></div>
                </div>
                <details class="collapsible" open>
                  <summary>{{ $t('common.basic_info') }}</summary>
                  <div class="body">
                    <table class="info-table">
                      <tr><th>{{ $t('collections.stat.dim') }}</th><td>{{ detailCache[db + '::' + name].dim }}</td></tr>
                      <tr><th>{{ $t('collections.stat.count') }}</th><td>{{ formatCount(detailCache[db + '::' + name].count) }}</td></tr>
                      <tr><th>{{ $t('collections.stat.metric') }}</th><td>{{ detailCache[db + '::' + name].metric }}</td></tr>
                      <tr><th>{{ $t('common.primary_field') }}</th><td>{{ detailCache[db + '::' + name].primary_field }}</td></tr>
                      <tr><th>{{ $t('common.vector_field') }}</th><td>{{ detailCache[db + '::' + name].vector_field }}</td></tr>
                    </table>
                  </div>
                </details>
                <details class="collapsible" open v-if="detailCache[db + '::' + name].fields && detailCache[db + '::' + name].fields.length">
                  <summary>{{ $t('collections.fields_n', { n: detailCache[db + '::' + name].fields.length }) }}</summary>
                  <div class="body">
                    <div v-for="f in detailCache[db + '::' + name].fields" :key="f.name" class="field-card">
                      <div class="field-card-header">
                        <span class="index-badge">{{ f.dtype }}</span>
                        <span class="title">{{ f.name }}</span>
                        <span v-if="f.is_primary" class="pill success">{{ $t('collections.primary') }}</span>
                        <span v-if="f.dim" class="pill accent">{{ $t('collections.dim_value', { dim: f.dim }) }}</span>
                      </div>
                      <div v-if="f.max_length || f.nullable || f.default_value" class="grid">
                        <div v-if="f.max_length" class="field"><label>max_length</label><div style="font-family:var(--mono);font-size:13px;">{{ f.max_length }}</div></div>
                        <div v-if="f.nullable" class="field"><label>nullable</label><div style="font-family:var(--mono);font-size:13px;">true</div></div>
                        <div v-if="f.default_value" class="field"><label>default_value</label><div style="font-family:var(--mono);font-size:13px;">{{ JSON.stringify(f.default_value) }}</div></div>
                      </div>
                    </div>
                  </div>
                </details>
                <div v-if="detailCache[db + '::' + name].indexes && detailCache[db + '::' + name].indexes.length" style="margin-top:14px;">
                  <div class="section-head"><h4 class="section-title">{{ $t('collections.indexes_n', { n: detailCache[db + '::' + name].indexes.length }) }}</h4></div>
                  <div v-for="ix in detailCache[db + '::' + name].indexes" :key="ix.field_name" class="field-card" :data-index-field="ix.field_name">
                    <div class="field-card-header">
                      <span class="index-badge">{{ ix.index_type }}</span>
                      <span class="title">{{ ix.metric_type }} → {{ ix.field_name }}</span>
                      <busy-button class="sm" variant="danger"
                                   :data-index-field="ix.field_name" :data-index-action="'drop-' + ix.field_name"
                                   :busy="busyKey === 'index:' + name + ':' + ix.field_name"
                                   :label="$t('common.delete')" :busy-label="$t('common.deleting')"
                                   @click="dropIndex(name, ix.field_name)" />
                    </div>
                    <div v-if="Object.keys(ix.params || {}).length" class="grid">
                      <div v-for="(v, k) in ix.params" :key="k" class="field">
                        <label>{{ k }}</label>
                        <div style="font-family:var(--mono);font-size:13px;">{{ JSON.stringify(v) }}</div>
                      </div>
                    </div>
                  </div>
                  <details class="collapsible" style="margin-top:10px;"
                           v-if="newIndex[db + '::' + name]">
                    <summary>{{ $t('collections.new_index') }}</summary>
                    <div class="body">
                      <div class="row">
                        <label>{{ $t('collections.target_field') }}</label>
                        <!-- The target is always the collection's vector
                             field: the old single-option select implied a
                             choice that does not exist (stage 3). -->
                        <span class="ops-mono" data-new-index-field>{{ newIndex[db + '::' + name].field }}</span>
                        <span class="hint">{{ $t('collections.vector_field_suffix') }}</span>
                      </div>
                      <div class="row split">
                        <div class="row"><label>{{ $t('common.metric') }}</label>
                          <select data-new-index-metric v-model="newIndex[db + '::' + name].metric">
                            <option value="cosine">cosine</option>
                            <option value="ip">ip</option>
                            <option value="l2">l2</option>
                          </select>
                        </div>
                        <div class="row"><label>{{ $t('collections.index_type') }}</label>
                          <select data-new-index-type v-model="newIndex[db + '::' + name].type">
                            <option value="HNSW">HNSW</option>
                            <option value="IVF_FLAT">IVF_FLAT</option>
                            <option value="IVF_SQ8">IVF_SQ8</option>
                            <option value="DISKANN">DISKANN</option>
                            <option value="FLAT">FLAT</option>
                          </select>
                        </div>
                      </div>
                      <div class="row">
                        <label>params <span class="hint">{{ $t('collections.params_hint') }}</span></label>
                        <div class="actions" style="margin:0 0 6px;">
                          <button type="button" class="btn sm" data-tpl-hnsw
                                  @click="fillIndexParams(name, 'hnsw')">{{ $t('collections.tpl_hnsw') }}</button>
                          <button type="button" class="btn sm" data-tpl-ivf
                                  @click="fillIndexParams(name, 'ivf')">{{ $t('collections.tpl_ivf') }}</button>
                          <button type="button" class="btn sm" data-tpl-clear
                                  @click="fillIndexParams(name, 'clear')">{{ $t('collections.tpl_clear') }}</button>
                        </div>
                        <textarea data-new-index-params rows="2" class="code-input"
                                  v-model="newIndex[db + '::' + name].params"></textarea>
                      </div>
                      <div class="actions">
                        <busy-button data-new-index-submit variant="primary" :busy="busyKey === 'newindex:' + name"
                                     :label="$t('collections.create_rebuild')"
                                     :busy-label="$t('collections.creating')"
                                     @click="submitNewIndex(name)" />
                      </div>
                    </div>
                  </details>
                </div>
              </template>
            </div>
          </template>
        </div>
      </div>
    </div>
  `,
  methods: {
    formatCount(n) {
      if (n == null || isNaN(n)) return '—';
      const v = Number(n);
      if (v < 1000) return String(v);
      if (v < 1e6) return (v / 1e3).toFixed(v < 1e4 ? 1 : 0) + 'k';
      return (v / 1e6).toFixed(v < 1e7 ? 2 : 1) + 'M';
    },
  },
});
