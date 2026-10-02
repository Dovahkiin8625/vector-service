// Search panel: k-NN search over one collection, either by text
// (server-side embed) or by a pre-computed query vector.
import { defineComponent, ref, computed, watch, onMounted } from '../vue.esm-browser.prod.js';
import { t, api, enc, extractApiError } from './app.js';
import { intError } from './util.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';

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
    // S6 numeric rule: whole number in 1..1000, with an inline hint and
    // a disabled run button while the box is invalid/empty.
    const topkErr = computed(() => intError(topk.value, 1, 1000));
    const text = ref('computer peripherals');
    const emb = ref('[0.1, 0.2, 0.3, 0.4]');
    const filter = ref('');
    const outputFields = ref(new Set());
    const schemaFields = ref([]);
    // The collection's distance metric, read from its detail payload. It
    // decides how a hit score may be read at all: cosine/ip are
    // similarities (bigger is closer), l2 is a distance (smaller is).
    const metric = ref('');
    const result = ref(null);
    // Outcome of the last search. Drives the results area, so "no hits",
    // "still running" and "the request failed" stop looking identical.
    const status = ref('idle');
    const errMsg = ref('');
    // Form-level complaints (nothing selected, malformed vector) are a
    // different thing from a failed run and get their own banner.
    const formErr = ref('');
    // Selection/schema loading failures: previously swallowed, which left
    // an empty dropdown with no explanation.
    const loadErr = ref('');

    async function refreshDbs() {
      loadErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
      } catch (e) {
        dbs.value = [];
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    async function refreshColls() {
      if (!db.value) { colls.value = []; return; }
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections');
        colls.value = (payload && payload.collections) || [];
        if (!colls.value.includes(coll.value)) coll.value = colls.value[0] || '';
      } catch (e) {
        colls.value = [];
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    async function refreshSchema() {
      schemaFields.value = [];
      metric.value = '';
      if (!db.value || !coll.value) return;
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value));
        schemaFields.value = (payload && payload.fields || []).map(f => f.name);
        metric.value = (payload && payload.metric) || '';
      } catch (e) {
        schemaFields.value = [];
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    watch(db, refreshColls);
    // Output fields are field names of the collection that was selected
    // when they were ticked; carrying them across a switch would submit
    // the previous collection's field names to the new one (B5).
    watch(coll, () => { outputFields.value = new Set(); refreshSchema(); });
    onMounted(refreshDbs);

    function safeParse(s) { try { return JSON.parse(s); } catch (_e) { return null; } }

    function buildBody() {
      const body = { primary_field: primary.value.trim(), vector_field: vecfield.value.trim(), top_k: topk.value };
      if (mode.value === 'text') {
        body.query_text = text.value;
      } else {
        const v = safeParse(emb.value);
        if (!Array.isArray(v)) { formErr.value = t('search.err.bad_vector'); return null; }
        body.query_vector = v;
      }
      if (filter.value.trim()) body.filter_expr = filter.value.trim();
      if (outputFields.value.size) body.output_fields = Array.from(outputFields.value);
      return body;
    }

    async function doSearch() {
      formErr.value = '';
      if (!db.value || !coll.value) { formErr.value = t('search.err.no_selection'); return; }
      // A cleared top_k box (v-model.number -> ''/NaN) used to ride into
      // the body unchecked; the shared integer rule now gates the run.
      if (topkErr.value) return;
      const body = buildBody(); if (!body) return;
      status.value = 'loading';
      errMsg.value = '';
      try {
        const { payload } = await api('POST', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/search', body);
        result.value = payload;
        status.value = 'ok';
      } catch (e) {
        result.value = null;
        status.value = 'error';
        errMsg.value = extractApiError(e, t('common.unknown'));
      }
    }

    // Four-state results area: loading / failed / genuinely no hits /
    // not run yet. `search.no_hits` was already in the dictionary but
    // had nothing rendering it.
    const hitCount = computed(() => (result.value && result.value.hits ? result.value.hits.length : 0));
    const resultState = computed(() => {
      if (status.value === 'loading') return 'loading';
      if (status.value === 'error') return 'error';
      return status.value === 'ok' ? 'empty' : 'idle';
    });
    const resultText = computed(() => {
      if (status.value === 'error') return t('common.status.error') + ': ' + errMsg.value;
      if (status.value === 'ok') return t('search.no_hits');
      if (status.value === 'idle') return t('search.idle_hint');
      return '';
    });
    const showResultList = computed(() => status.value === 'ok' && hitCount.value > 0);

    // Hit scores are metric-dependent and unbounded: an l2 distance can
    // exceed 1 and a negative inner product is normal. Rendering
    // `score * 100` as a percentage was therefore meaningless outside
    // cosine — and produced "NaN" for a missing score. Show the raw
    // value, and say which direction means "closer" for this collection.
    function formatScore(v) {
      const n = Number(v);
      return (v === null || v === undefined || !Number.isFinite(n)) ? '—' : n.toFixed(4);
    }
    const metricNote = computed(() => {
      if (!metric.value) return '';
      return metric.value === 'l2' ? t('search.metric_lower') : t('search.metric_higher');
    });

    function toggleOutput(name) {
      if (outputFields.value.has(name)) outputFields.value.delete(name);
      else outputFields.value.add(name);
      outputFields.value = new Set(outputFields.value);
    }

    return { dbs, colls, db, coll, primary, vecfield, mode, topk, topkErr, text, emb, filter,
             schemaFields, outputFields, result, doSearch, toggleOutput,
             status, formErr, loadErr, refreshDbs, resultState, resultText, showResultList,
             metric, metricNote, formatScore };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('search.title') }} <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/search</span></h3>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.database') }}</label>
            <select id="srch-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>{{ $t('common.collection') }}</label>
            <select id="srch-coll" v-model="coll"><option v-for="c in colls" :key="c" :value="c">{{ c }}</option></select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.primary_field') }}</label><input type="text" id="srch-primary" v-model="primary" /></div>
          <div class="row"><label>{{ $t('common.vector_field') }}</label><input type="text" id="srch-vecfield" v-model="vecfield" /></div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('search.query_mode') }}</label>
            <select id="srch-mode" v-model="mode">
              <option value="text">{{ $t('search.mode.text') }}</option>
              <option value="emb">{{ $t('search.mode.vector') }}</option>
            </select>
          </div>
          <div class="row"><label>top_k</label><input type="number" id="srch-topk" v-model.number="topk" min="1" max="1000" />
            <span class="hint" v-if="topkErr">{{ topkErr }}</span></div>
        </div>
        <div class="row" v-show="mode === 'text'"><label>{{ $t('search.query_text') }}</label><input type="text" id="srch-text" v-model="text" /></div>
        <div class="row" v-show="mode === 'emb'"><label>{{ $t('search.query_vector') }}</label><textarea id="srch-emb" class="code-input" rows="2" v-model="emb"></textarea></div>
        <details class="collapsible">
          <summary>{{ $t('search.filter_title') }}</summary>
          <div class="body">
            <div class="row"><label>filter_expr</label><textarea id="srch-filter" class="code-input" rows="2" v-model="filter" placeholder="category == 'mouse'"></textarea></div>
          </div>
        </details>
        <details class="collapsible" v-if="schemaFields.length">
          <summary>{{ $t('search.output_fields') }}</summary>
          <div class="body">
            <div class="row">
              <label>{{ $t('search.output_fields') }} <span class="hint">{{ $t('search.hint.click_toggle') }}</span></label>
              <div id="srch-output-tokens">
                <span v-for="f in schemaFields" :key="f"
                      :class="['tag-token', outputFields.has(f) ? 'active' : '']"
                      @click="toggleOutput(f)">{{ f }}</span>
              </div>
            </div>
          </div>
        </details>
        <div class="actions">
          <busy-button id="btn-search" variant="primary" :busy="status === 'loading'" :disabled="!!topkErr" :label="$t('search.run')"
                       :busy-label="$t('search.running')" @click="doSearch" />
        </div>
        <status-banner kind="error" :text="formErr" />
      </div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('common.results') }}</h3>
          <span class="section-sub" v-if="metricNote">{{ metric }} · {{ metricNote }}</span>
        </div>
        <status-banner kind="error" :text="loadErr" :retry="loadErr ? refreshDbs : null" />
        <empty-state v-if="!showResultList" :state="resultState" :text="resultText"
                     :retry="resultState === 'error' ? doSearch : null" />
        <template v-else>
          <div v-for="(hit, i) in result.hits" :key="i" class="result-row">
            <span class="result-rank">#{{ i + 1 }}</span>
            <span class="result-score" :title="$t('search.score')">{{ formatScore(hit.score) }}</span>
            <span class="result-doc">{{ hit.id }}</span>
            <pre class="code-pane" v-if="hit.fields && Object.keys(hit.fields).length" style="margin:0;">{{ JSON.stringify(hit.fields, null, 2) }}</pre>
          </div>
        </template>
      </div>
    </div>
  `,
});
