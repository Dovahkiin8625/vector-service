// Collections panel: list + create (modal) + expandable detail (schema + indexes + index management).
import { defineComponent, ref, watch, onMounted } from '../vue.esm-browser.prod.js';
import { store, api, enc, extractApiError } from './app.js';

export default defineComponent({
  name: 'CollectionsPanel',
  setup() {
    const db = ref('');
    const dbs = ref([]);
    const colls = ref([]);
    const detailCache = ref(Object.create(null));
    const expanded = ref(new Set());

    async function refreshDbs() {
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        store.databases.list = dbs.value;
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
      } catch (_e) {}
    }

    async function refreshColls() {
      if (!db.value) { colls.value = []; return; }
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections');
        colls.value = (payload && payload.collections) || [];
        store.collections.list = colls.value;
        const live = new Set(colls.value);
        Object.keys(detailCache.value).forEach(k => {
          if (!live.has(k)) delete detailCache.value[k];
        });
        expanded.value.forEach(k => { if (!live.has(k)) expanded.value.delete(k); });
      } catch (_e) { colls.value = []; }
    }

    async function loadDetail(name) {
      const key = db.value + '::' + name;
      if (detailCache.value[key]) return detailCache.value[key];
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections/' + enc(name));
        detailCache.value[key] = payload;
        return payload;
      } catch (_e) { return null; }
    }

    async function toggleDetail(name) {
      const key = db.value + '::' + name;
      if (expanded.value.has(key)) { expanded.value.delete(key); return; }
      expanded.value.add(key);
      await loadDetail(name);
    }

    async function dropColl(name) {
      if (!confirm('确认删除集合 "' + db.value + '/' + name + '"?')) return;
      try { await api('DELETE', '/v1/databases/' + enc(db.value) + '/collections/' + enc(name)); }
      catch (e) { alert('删除失败: ' + extractApiError(e, 'unknown')); return; }
      const key = db.value + '::' + name;
      delete detailCache.value[key];
      expanded.value.delete(key);
      await refreshColls();
    }

    async function dropIndex(coll, fieldName) {
      if (!confirm('确认删除字段 "' + fieldName + '" 上的索引?')) return;
      try {
        await api('DELETE',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll) + '/index?field_name=' + enc(fieldName));
        const key = db.value + '::' + coll;
        delete detailCache.value[key];
        await loadDetail(coll);
      } catch (e) { alert('删除索引失败: ' + extractApiError(e, 'unknown')); }
    }

    async function createIndex(coll, body) {
      try {
        await api('POST',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll) + '/index', body);
        const key = db.value + '::' + coll;
        delete detailCache.value[key];
        await loadDetail(coll);
      } catch (e) { alert('新建索引失败: ' + extractApiError(e, 'unknown')); }
    }

    function submitNewIndex(coll, ev) {
      const card = ev.target.closest('.field-card');
      const field = card.querySelector('[data-new-index-field]').value;
      const metric = card.querySelector('[data-new-index-metric]').value;
      const type = card.querySelector('[data-new-index-type]').value;
      const raw = card.querySelector('[data-new-index-params]').value.trim();
      let params = {};
      if (raw) {
        try { params = JSON.parse(raw); } catch (_e) { alert('params 必须是合法 JSON 对象.'); return; }
      }
      createIndex(coll, { field_name: field, metric_type: metric, index_type: type, params: params });
    }

    watch(db, () => { refreshColls(); });
    onMounted(() => {
      refreshDbs();
      window.addEventListener('refresh-colls', refreshColls);
      window.addEventListener('refresh-dbs', refreshDbs);
    });

    return {
      store,
      db, dbs, colls, detailCache, expanded, refreshDbs, refreshColls,
             toggleDetail, dropColl, dropIndex, createIndex, submitNewIndex };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head"><h3 class="section-title">选择数据库</h3></div>
        <div class="row split">
          <div class="row">
            <label>当前数据库</label>
            <select id="colls-db" v-model="db">
              <option v-for="d in dbs" :key="d" :value="d">{{ d }}</option>
              <option v-if="!dbs.length" value="">（暂无数据库）</option>
            </select>
          </div>
          <div class="row">
            <label>&nbsp;</label>
            <button id="btn-colls-refresh-db" class="btn" @click="refreshDbs">↻ 重新加载数据库列表</button>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">集合列表 <span class="pill accent">GET /v1/databases/{db}/collections</span></h3>
        </div>
        <div class="actions">
          <button id="btn-refresh-colls" class="btn primary" :disabled="!db" @click="refreshColls">刷新</button>
          <button id="btn-open-new-coll" class="btn" :disabled="!db" @click="store.modals.newColl = true">+ 新建集合</button>
        </div>
        <div class="list" id="colls-list">
          <div v-if="!db" class="empty">请先选择数据库。</div>
          <div v-else-if="!colls.length" class="empty">该数据库下暂无集合。</div>
          <template v-for="name in colls" :key="db + '::' + name">
            <div class="list-item" :data-coll-key="db + '::' + name" @click="toggleDetail(name)">
              <span class="name">{{ db }} / {{ name }}</span>
              <span class="meta">
                <template v-if="detailCache[db + '::' + name]">
                  {{ detailCache[db + '::' + name].metric || '—' }} · {{ detailCache[db + '::' + name].dim || '—' }} 维 · {{ formatCount(detailCache[db + '::' + name].count) }}
                </template>
                <template v-else>— · — · —</template>
              </span>
              <span style="color:var(--text-muted);font-size:12px;">{{ expanded.has(db + '::' + name) ? '▾' : '▸' }}</span>
              <button class="btn sm danger" @click.stop="dropColl(name)">删除</button>
            </div>
            <div v-if="expanded.has(db + '::' + name)" class="field-card" :data-detail-for="db + '::' + name"
                 style="margin-left:0;margin-top:4px;padding:14px 16px;">
              <div v-if="!detailCache[db + '::' + name]" class="empty"><span class="spinner"></span> 正在加载集合详情...</div>
              <template v-else>
                <div class="kb-result">
                  <div class="stat"><span class="key">维度</span><span class="val accent">{{ detailCache[db + '::' + name].dim }}</span></div>
                  <div class="stat"><span class="key">数据量</span><span class="val">{{ formatCount(detailCache[db + '::' + name].count) }}</span></div>
                  <div class="stat"><span class="key">度量</span><span class="val">{{ detailCache[db + '::' + name].metric }}</span></div>
                  <div class="stat"><span class="key">主键 / 向量字段</span><span class="val">{{ detailCache[db + '::' + name].primary_field }} · {{ detailCache[db + '::' + name].vector_field }}</span></div>
                </div>
                <details class="collapsible" open>
                  <summary>基础信息</summary>
                  <div class="body">
                    <table class="info-table">
                      <tr><th>维度 dim</th><td>{{ detailCache[db + '::' + name].dim }}</td></tr>
                      <tr><th>数据量 count</th><td>{{ formatCount(detailCache[db + '::' + name].count) }}</td></tr>
                      <tr><th>距离度量 metric</th><td>{{ detailCache[db + '::' + name].metric }}</td></tr>
                      <tr><th>主键字段</th><td>{{ detailCache[db + '::' + name].primary_field }}</td></tr>
                      <tr><th>向量字段</th><td>{{ detailCache[db + '::' + name].vector_field }}</td></tr>
                    </table>
                  </div>
                </details>
                <details class="collapsible" open v-if="detailCache[db + '::' + name].fields && detailCache[db + '::' + name].fields.length">
                  <summary>字段 ({{ detailCache[db + '::' + name].fields.length }})</summary>
                  <div class="body">
                    <div v-for="f in detailCache[db + '::' + name].fields" :key="f.name" class="field-card">
                      <div class="field-card-header">
                        <span class="index-badge">{{ f.dtype }}</span>
                        <span class="title">{{ f.name }}</span>
                        <span v-if="f.is_primary" class="pill success">主键</span>
                        <span v-if="f.dim" class="pill accent">{{ f.dim }} 维</span>
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
                  <div class="section-head"><h4 class="section-title">索引 ({{ detailCache[db + '::' + name].indexes.length }})</h4></div>
                  <div v-for="ix in detailCache[db + '::' + name].indexes" :key="ix.field_name" class="field-card" :data-index-field="ix.field_name">
                    <div class="field-card-header">
                      <span class="index-badge">{{ ix.index_type }}</span>
                      <span class="title">{{ ix.metric_type }} → {{ ix.field_name }}</span>
                      <button class="btn sm danger" :data-index-field="ix.field_name" :data-index-action="'drop-' + ix.field_name"
                              @click="dropIndex(name, ix.field_name)">删除</button>
                    </div>
                    <div v-if="Object.keys(ix.params || {}).length" class="grid">
                      <div v-for="(v, k) in ix.params" :key="k" class="field">
                        <label>{{ k }}</label>
                        <div style="font-family:var(--mono);font-size:13px;">{{ JSON.stringify(v) }}</div>
                      </div>
                    </div>
                  </div>
                  <details class="collapsible" style="margin-top:10px;">
                    <summary>＋ 新建索引</summary>
                    <div class="body">
                      <div class="row">
                        <label>目标字段</label>
                        <select :data-new-index-field="detailCache[db + '::' + name].vector_field">
                          <option :value="detailCache[db + '::' + name].vector_field">{{ detailCache[db + '::' + name].vector_field }} (向量字段)</option>
                        </select>
                      </div>
                      <div class="row split">
                        <div class="row"><label>度量</label>
                          <select data-new-index-metric>
                            <option value="cosine" selected>cosine</option>
                            <option value="ip">ip</option>
                            <option value="l2">l2</option>
                          </select>
                        </div>
                        <div class="row"><label>索引类型</label>
                          <select data-new-index-type>
                            <option value="HNSW" selected>HNSW</option>
                            <option value="IVF_FLAT">IVF_FLAT</option>
                            <option value="IVF_SQ8">IVF_SQ8</option>
                            <option value="DISKANN">DISKANN</option>
                            <option value="FLAT">FLAT</option>
                          </select>
                        </div>
                      </div>
                      <div class="row">
                        <label>params <span class="hint">JSON,例如 {"M":16,"efConstruction":200}</span></label>
                        <textarea data-new-index-params rows="2">{}</textarea>
                      </div>
                      <div class="actions">
                        <button class="btn primary" data-new-index-submit
                                @click="submitNewIndex(name, $event)">创建 / 重建</button>
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
