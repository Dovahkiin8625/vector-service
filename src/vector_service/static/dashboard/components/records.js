// Records panel: upsert / fetch by id / delete by id or filter.
import { defineComponent, ref, watch, onMounted } from '../vue.esm-browser.prod.js';
import { store, api, enc, extractApiError } from './app.js';

export default defineComponent({
  name: 'RecordsPanel',
  setup() {
    const dbs = ref([]);
    const colls = ref([]);
    const db = ref('');
    const coll = ref('');
    const primary = ref('id');
    const vecfield = ref('vector');
    const mode = ref('texts');
    const ids = ref('["sku-1", "sku-2"]');
    const texts = ref('["无线鼠标", "机械键盘"]');
    const embs = ref('[[0.1, 0.2, 0.3, 0.4]]');
    const fields = ref('[{"category":"mouse","price":29.9},{"category":"keyboard","price":119.0}]');
    const delMode = ref('ids');
    const delIds = ref('sku-1\nsku-2');
    const delFilter = ref('');
    const status = ref('');

    async function refreshDbs() {
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
      } catch (_e) {}
    }
    async function refreshColls() {
      if (!db.value) { colls.value = []; return; }
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections');
        colls.value = (payload && payload.collections) || [];
        if (!colls.value.includes(coll.value)) coll.value = colls.value[0] || '';
      } catch (_e) { colls.value = []; }
    }
    watch(db, refreshColls);
    onMounted(refreshDbs);

    function safeParse(s) { try { return JSON.parse(s); } catch (_e) { return null; } }

    function buildUpsertBody() {
      const idsArr = safeParse(ids.value);
      if (!Array.isArray(idsArr)) { alert('ids 必须是 JSON 数组.'); return null; }
      const body = { primary_field: primary.value.trim(), vector_field: vecfield.value.trim(), ids: idsArr };
      if (mode.value === 'texts') {
        const t = safeParse(texts.value);
        if (!Array.isArray(t)) { alert('texts 必须是 JSON 数组.'); return null; }
        body.texts = t;
      } else {
        const e = safeParse(embs.value);
        if (!Array.isArray(e)) { alert('vectors 必须是 JSON 数组.'); return null; }
        body.vectors = e;
      }
      const fRaw = fields.value.trim();
      if (fRaw && fRaw !== '[]') {
        const f = safeParse(fRaw);
        if (!Array.isArray(f)) { alert('fields 必须是 JSON 数组.'); return null; }
        if (f.length) body.fields = f;
      }
      return body;
    }

    async function doUpsert() {
      if (!db.value || !coll.value) { alert('请选择数据库和集合.'); return; }
      const body = buildUpsertBody(); if (!body) return;
      try { await api('PUT', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors', body); status.value = 'OK'; }
      catch (e) { status.value = 'FAIL: ' + extractApiError(e, 'unknown'); }
    }
    async function doFetch() {
      if (!db.value || !coll.value) { alert('请选择数据库和集合.'); return; }
      const idsArr = safeParse(ids.value);
      if (!Array.isArray(idsArr)) { alert('ids 必须是 JSON 数组.'); return; }
      try { await api('POST', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors/get', { primary_field: primary.value.trim(), ids: idsArr }); status.value = 'OK'; }
      catch (e) { status.value = 'FAIL: ' + extractApiError(e, 'unknown'); }
    }
    async function doDelete() {
      if (!db.value || !coll.value) { alert('请选择数据库和集合.'); return; }
      const body = { primary_field: primary.value.trim() };
      if (delMode.value === 'ids') {
        let arr = safeParse(delIds.value);
        if (!Array.isArray(arr)) {
          arr = delIds.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
        }
        if (!arr.length) { alert('请填写至少 1 个主键.'); return; }
        body.ids = arr;
      } else {
        if (!delFilter.value.trim()) { alert('请填写过滤表达式.'); return; }
        body.filter_expr = delFilter.value.trim();
      }
      if (!confirm('确认删除. 此操作不可撤销.')) return;
      try { await api('POST', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors/delete', body); status.value = 'OK'; }
      catch (e) { status.value = 'FAIL: ' + extractApiError(e, 'unknown'); }
    }

    return { dbs, colls, db, coll, primary, vecfield, mode, ids, texts, embs, fields,
             delMode, delIds, delFilter, status, refreshDbs, doUpsert, doFetch, doDelete };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">记录写入 <span class="pill accent">PUT /v1/databases/{db}/collections/{coll}/vectors</span></h3>
        </div>
        <div class="row split">
          <div class="row"><label>数据库</label>
            <select id="vec-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>集合</label>
            <select id="vec-coll" v-model="coll"><option v-for="c in colls" :key="c" :value="c">{{ c }}</option></select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>主键字段名</label><input type="text" id="vec-primary" v-model="primary" /></div>
          <div class="row"><label>向量字段名</label><input type="text" id="vec-vecfield" v-model="vecfield" /></div>
        </div>
        <div class="row"><label>输入方式</label>
          <select id="vec-mode" v-model="mode">
            <option value="texts">由服务端嵌入 (texts)</option>
            <option value="vectors">直接提供向量</option>
          </select>
        </div>
        <div class="row"><label>主键值数组 (ids)</label><textarea id="vec-ids" rows="2" v-model="ids"></textarea></div>
        <div class="row" v-show="mode === 'texts'"><label>待嵌入文本 (texts)</label><textarea id="vec-texts" rows="3" v-model="texts"></textarea></div>
        <div class="row" v-show="mode === 'vectors'"><label>已嵌入向量 (vectors)</label><textarea id="vec-embs" rows="3" v-model="embs"></textarea></div>
        <details class="collapsible">
          <summary>附加字段 (fields, 按行对齐 ids)</summary>
          <div class="body">
            <div class="row"><label>字段集</label><textarea id="vec-fields" rows="3" v-model="fields"></textarea></div>
          </div>
        </details>

        <details class="collapsible">
          <summary>删除参数 <span class="hint">用于下方的"按条件删除"按钮</span></summary>
          <div class="body">
            <div class="row"><label>删除方式</label>
              <select id="vec-del-mode" v-model="delMode">
                <option value="ids">按主键列表</option>
                <option value="filter">按过滤表达式</option>
              </select>
            </div>
            <div class="row" id="vec-del-ids-row" v-show="delMode === 'ids'"><label>主键值</label><textarea id="vec-del-ids" rows="2" v-model="delIds"></textarea></div>
            <div class="row" id="vec-del-filter-row" v-show="delMode === 'filter'"><label>过滤表达式</label><textarea id="vec-del-filter" rows="2" v-model="delFilter" placeholder="category == 'mouse'"></textarea></div>
          </div>
        </details>

        <div class="actions">
          <button class="btn primary" id="btn-upsert" @click="doUpsert">写入 (PUT)</button>
          <button class="btn" id="btn-fetch" @click="doFetch">按主键获取 (POST)</button>
          <button class="btn danger" id="btn-delete" @click="doDelete">按条件删除 (POST)</button>
        </div>
        <div v-if="status" class="empty hint">最近操作: {{ status }}</div>
      </div>
    </div>
  `,
});
