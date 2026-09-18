// Search panel: k-NN search with text / vector / image queries.
import { defineComponent, ref, watch, onMounted } from '../vue.esm-browser.prod.js';
import { store, api, enc, extractApiError } from './app.js';

export default defineComponent({
  name: 'SearchPanel',
  setup() {
    const dbs = ref([]);
    const colls = ref([]);
    const db = ref('');
    const coll = ref('');
    const primary = ref('id');
    const vecfield = ref('vector');
    const mode = ref('text');
    const topk = ref(5);
    const text = ref('computer peripherals');
    const emb = ref('[0.1, 0.2, 0.3, 0.4]');
    const filter = ref('');
    const outputFields = ref(new Set());
    const schemaFields = ref([]);
    const result = ref(null);

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
    async function refreshSchema() {
      schemaFields.value = [];
      if (!db.value || !coll.value) return;
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value));
        schemaFields.value = (payload && payload.fields || []).map(f => f.name);
      } catch (_e) {}
    }
    watch(db, refreshColls);
    watch(coll, refreshSchema);
    onMounted(refreshDbs);

    function safeParse(s) { try { return JSON.parse(s); } catch (_e) { return null; } }

    function buildBody() {
      const body = { primary_field: primary.value.trim(), vector_field: vecfield.value.trim(), top_k: topk.value };
      if (mode.value === 'text') {
        body.query_text = text.value;
      } else {
        const v = safeParse(emb.value);
        if (!Array.isArray(v)) { alert('query_vector must be a JSON array.'); return null; }
        body.query_vector = v;
      }
      if (filter.value.trim()) body.filter_expr = filter.value.trim();
      if (outputFields.value.size) body.output_fields = Array.from(outputFields.value);
      return body;
    }

    async function doSearch() {
      if (!db.value || !coll.value) { alert('select database and collection.'); return; }
      const body = buildBody(); if (!body) return;
      try {
        const { payload } = await api('POST', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/search', body);
        result.value = payload;
      } catch (e) { alert('search failed: ' + extractApiError(e, 'unknown')); result.value = null; }
    }

    function toggleOutput(name) {
      if (outputFields.value.has(name)) outputFields.value.delete(name);
      else outputFields.value.add(name);
      outputFields.value = new Set(outputFields.value);
    }

    return { dbs, colls, db, coll, primary, vecfield, mode, topk, text, emb, filter,
             schemaFields, outputFields, result, doSearch, toggleOutput };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">vector search <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/search</span></h3>
        </div>
        <div class="row split">
          <div class="row"><label>database</label>
            <select id="srch-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>collection</label>
            <select id="srch-coll" v-model="coll"><option v-for="c in colls" :key="c" :value="c">{{ c }}</option></select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>primary key field</label><input type="text" id="srch-primary" v-model="primary" /></div>
          <div class="row"><label>vector field</label><input type="text" id="srch-vecfield" v-model="vecfield" /></div>
        </div>
        <div class="row split">
          <div class="row"><label>query mode</label>
            <select id="srch-mode" v-model="mode">
              <option value="text">server-side embed (query_text)</option>
              <option value="emb">direct vector (query_vector)</option>
            </select>
          </div>
          <div class="row"><label>top_k</label><input type="number" id="srch-topk" v-model.number="topk" min="1" max="1000" /></div>
        </div>
        <div class="row" v-show="mode === 'text'"><label>query text</label><input type="text" id="srch-text" v-model="text" /></div>
        <div class="row" v-show="mode === 'emb'"><label>query vector</label><textarea id="srch-emb" rows="2" v-model="emb"></textarea></div>
        <details class="collapsible">
          <summary>filter expression (Milvus native)</summary>
          <div class="body">
            <div class="row"><label>filter_expr</label><textarea id="srch-filter" rows="2" v-model="filter" placeholder="category == 'mouse'"></textarea></div>
          </div>
        </details>
        <details class="collapsible" v-if="schemaFields.length">
          <summary>output fields</summary>
          <div class="body">
            <div class="row">
              <label>output fields <span class="hint">click to toggle</span></label>
              <div id="srch-output-tokens">
                <span v-for="f in schemaFields" :key="f"
                      :class="['tag-token', outputFields.has(f) ? 'active' : '']"
                      @click="toggleOutput(f)">{{ f }}</span>
              </div>
            </div>
          </div>
        </details>
        <div class="actions">
          <button class="btn primary" id="btn-search" @click="doSearch">search</button>
        </div>
      </div>
      <div class="section" v-if="result">
        <div class="section-head"><h3 class="section-title">results</h3></div>
        <div v-for="(hit, i) in result.hits" :key="i" class="result-row">
          <span class="result-rank">#{{ i + 1 }}</span>
          <span class="result-score">{{ (hit.score * 100).toFixed(2) }}</span>
          <span class="result-doc">{{ hit.id }}</span>
          <pre class="code-pane" v-if="hit.fields && Object.keys(hit.fields).length" style="margin:0;">{{ JSON.stringify(hit.fields, null, 2) }}</pre>
        </div>
      </div>
    </div>
  `,
});
