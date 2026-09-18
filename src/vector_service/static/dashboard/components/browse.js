// Browse panel: paginated list view of collection rows.
import { defineComponent, ref, watch, onMounted, computed } from '../vue.esm-browser.prod.js';
import { store, api, enc, extractApiError } from './app.js';

export default defineComponent({
  name: 'BrowsePanel',
  setup() {
    const dbs = ref([]);
    const colls = ref([]);
    const db = ref('');
    const coll = ref('');
    const primary = ref('id');
    const pageSize = ref(20);
    const filter = ref('');
    const offset = ref(0);
    const total = ref(0);
    const items = ref([]);
    const schema = ref(null);
    const status = ref('idle');

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
      schema.value = null;
      if (!db.value || !coll.value) return;
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value));
        schema.value = payload;
      } catch (_e) { schema.value = null; }
    }
    watch(db, () => { offset.value = 0; refreshColls(); });
    watch(coll, () => { offset.value = 0; refreshSchema(); runQuery(); });
    onMounted(refreshDbs);

    const columns = computed(() => {
      const set = new Set();
      items.value.forEach(it => Object.keys(it.fields || {}).forEach(k => set.add(k)));
      return [primary.value, ...Array.from(set)].filter((c, i, a) => a.indexOf(c) === i);
    });
    const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)));
    const currentPage = computed(() => Math.floor(offset.value / pageSize.value) + 1);

    async function runQuery() {
      if (!db.value || !coll.value) return;
      status.value = 'loading';
      const body = { primary_field: primary.value, limit: pageSize.value, offset: offset.value };
      if (filter.value.trim()) body.filter_expr = filter.value.trim();
      try {
        const { payload } = await api('POST',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/rows', body);
        items.value = (payload && payload.items) || [];
        total.value = (payload && payload.total) || 0;
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }
    function jump(delta) {
      offset.value = Math.max(0, Math.min(offset.value + delta, Math.max(0, (totalPages.value - 1) * pageSize.value)));
      runQuery();
    }
    function jumpTo(page) {
      if (isNaN(page) || page < 1) return;
      offset.value = Math.min((page - 1) * pageSize.value, Math.max(0, (totalPages.value - 1) * pageSize.value));
      runQuery();
    }

    return { dbs, colls, db, coll, primary, pageSize, filter, offset, total, items, schema,
             columns, totalPages, currentPage, status, refreshDbs, runQuery, jump, jumpTo };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">Select database / collection</h3>
          <span class="section-sub">paginated browse of collection rows</span>
        </div>
        <div class="row split">
          <div class="row"><label>database</label>
            <select id="brw-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>collection</label>
            <select id="brw-coll" v-model="coll"><option v-for="c in colls" :key="c" :value="c">{{ c }}</option></select>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">query params <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/rows</span></h3>
          <span class="section-sub">sorted by primary key asc; vector field always hidden</span>
        </div>
        <div class="row split">
          <div class="row"><label>primary key field</label><input type="text" id="brw-primary" v-model="primary" /></div>
          <div class="row"><label>page size</label>
            <select id="brw-page-size" v-model.number="pageSize">
              <option :value="20">20</option><option :value="50">50</option><option :value="100">100</option>
            </select>
          </div>
        </div>
        <div class="row"><label>filter expression</label>
          <textarea id="brw-filter" rows="2" v-model="filter" placeholder="category == 'mouse'"></textarea>
        </div>
        <div class="row" v-if="schema && schema.fields"><label>output fields <span class="hint">read-only</span></label>
          <div id="brw-output-tokens">
            <span v-for="f in schema.fields.filter(x => x.dtype !== 'float_vector')" :key="f.name"
                  :class="['tag-token', 'active']" :data-brw-field="f.name">
              <span v-if="f.name === primary" class="pill success" style="padding:0 5px;font-size:9px;margin-right:4px;">PK</span>
              {{ f.name }} <span style="color:var(--text-muted);font-size:10px;">{{ f.dtype }}</span>
            </span>
          </div>
        </div>
        <div class="actions">
          <button class="btn primary" id="btn-brw-query" @click="runQuery">query</button>
          <button class="btn" id="btn-brw-reset" @click="filter = ''; offset = 0; runQuery()">reset</button>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">results</h3>
          <span class="section-sub" id="brw-summary">
            {{ total ? (offset + 1) + ' - ' + (offset + items.length) + ' / ' + total + ' rows' : 'empty' }}
          </span>
        </div>
        <div v-if="total" class="kb-result" id="brw-result-stats">
          <div class="stat"><span class="key">total rows</span><span class="val accent" id="brw-stat-total">{{ formatCount(total) }}</span></div>
          <div class="stat"><span class="key">page</span><span class="val" id="brw-stat-page">{{ currentPage }} / {{ totalPages }}</span></div>
          <div class="stat"><span class="key">returned</span><span class="val" id="brw-stat-returned">{{ items.length }}</span></div>
        </div>
        <div id="brw-table-wrap">
          <div v-if="!items.length" class="empty">collection is empty or filter matches nothing.</div>
          <div v-else class="data-table-wrap">
            <table class="data-table">
              <thead><tr><th v-for="c in columns" :key="c">{{ c }}<span v-if="c === primary" class="pk">PK</span></th></tr></thead>
              <tbody>
                <tr v-for="row in items" :key="row.id">
                  <td v-for="c in columns" :key="c"
                      :class="[formatCell(row, c).cls, c === primary ? 'pk' : '']"
                      :title="formatCell(row, c).title">{{ formatCell(row, c).html }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div v-if="total" class="pager" id="brw-pager">
          <button class="btn sm" id="btn-brw-first" :disabled="currentPage <= 1" @click="jumpTo(1)">« first</button>
          <button class="btn sm" id="btn-brw-prev"  :disabled="currentPage <= 1" @click="jump(-pageSize)">‹ prev</button>
          <span class="info" id="brw-pager-info">rows {{ offset + 1 }}-{{ offset + items.length }} of {{ total }} / page {{ currentPage }} of {{ totalPages }}</span>
          <button class="btn sm" id="btn-brw-next" :disabled="currentPage >= totalPages" @click="jump(pageSize)">next ›</button>
          <button class="btn sm" id="btn-brw-last" :disabled="currentPage >= totalPages" @click="jumpTo(totalPages)">last »</button>
          <span class="info">jump to</span>
          <input type="number" id="brw-jump" :min="1" :max="totalPages" v-model.number="currentPage" style="width:72px;" />
          <button class="btn sm" id="btn-brw-jump" @click="jumpTo(currentPage)">go</button>
        </div>
      </div>
      <div class="empty hint">hover a cell to see the full JSON value; pager hides when the collection is empty.</div>
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
    formatCell(row, col) {
      const val = col === this.primary ? row.id : (row.fields ? row.fields[col] : undefined);
      if (val === null || val === undefined) return { html: 'null', cls: 'null', title: 'null' };
      if (typeof val === 'object') return { html: JSON.stringify(val), cls: 'json', title: JSON.stringify(val) };
      const s = String(val);
      return { html: s, cls: '', title: s };
    },
  },
});
