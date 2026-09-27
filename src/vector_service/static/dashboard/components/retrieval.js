// Retrieval panel: dense/BM25 recall, fusion, rewrite, MMR, rerank.
// Streams NDJSON stage events from /v1/retrieval/stream and renders the
// full retrieval trace (plan, per-channel raw tops, stage timings).
import {
  defineComponent, reactive, onMounted,
} from '../vue.esm-browser.prod.js';

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => null);
  return { status: resp.status, body };
}

function defaultState() {
  return {
    databases: [],
    database: 'default',
    caps: { llm_configured: false, schema_version: 2, migration_available: false },
    query: '',
    topK: 10,
    mode: 'hybrid',
    dense: true,
    bm25: true,
    fusionMethod: 'rrf',
    rrfK: 60,
    wDense: 0.5,
    wBm25: 0.5,
    rewrite: false,
    methods: { hyde: true, multi_query: true, step_back: true, decompose: true },
    hydeAlpha: 0.7,
    nVariants: 3,
    mmr: false,
    mmrLambda: 0.7,
    rerank: true,
    candidatePool: 25,
    docId: '',
    filename: '',
    busy: false,
    stages: [],
    error: '',
    result: null,
    showTrace: false,
    migrating: false,
  };
}

export default defineComponent({
  name: 'RetrievalPanel',
  setup() {
    const s = reactive(defaultState());

    async function loadDatabases() {
      const { body } = await getJson('/v1/databases');
      const names = (body && (body.databases || body.payload && body.payload.databases)) || [];
      s.databases = names;
      if (!names.includes(s.database) && names.length) s.database = names[0];
    }

    async function loadCaps() {
      const { body } = await getJson(
        '/v1/retrieval/capabilities?database=' + encodeURIComponent(s.database),
      );
      if (body && body.payload) Object.assign(s.caps, body.payload);
      else if (body) Object.assign(s.caps, body);
    }

    function setMode(mode) {
      s.mode = mode;
      if (mode === 'basic') {
        s.dense = true; s.bm25 = false;
        s.fusionMethod = 'rrf';
        s.rewrite = false; s.mmr = false; s.rerank = false;
      } else if (mode === 'hybrid') {
        s.dense = true; s.bm25 = true;
        s.fusionMethod = 'rrf';
        s.rewrite = false; s.mmr = false; s.rerank = true;
      } else if (mode === 'advanced') {
        s.dense = true; s.bm25 = true;
        s.fusionMethod = 'rrf';
        s.rewrite = true;
        s.methods = { hyde: true, multi_query: true, step_back: true, decompose: true };
        s.mmr = false; s.rerank = true;
      }
    }

    function buildBody() {
      const methods = Object.keys(s.methods).filter((k) => s.methods[k]);
      return {
        database: s.database,
        collection: 'ingest',
        query: s.query,
        top_k: Number(s.topK),
        filter: { doc_id: s.docId || '', filename: s.filename || '' },
        channels: { dense: s.dense, bm25: s.bm25 },
        fusion: {
          method: s.fusionMethod,
          rrf_k: Number(s.rrfK),
          weights: { dense: Number(s.wDense), bm25: Number(s.wBm25) },
        },
        rewrite: {
          enabled: s.rewrite,
          methods,
          hyde_alpha: Number(s.hydeAlpha),
          n_variants: Number(s.nVariants),
        },
        mmr: { enabled: s.mmr, lambda_mult: Number(s.mmrLambda) },
        rerank: { enabled: s.rerank, candidate_pool: Number(s.candidatePool) },
      };
    }

    async function run() {
      s.busy = true; s.error = ''; s.result = null; s.stages = [];
      try {
        const resp = await fetch('/v1/retrieval/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildBody()),
        });
        if (!resp.ok || !resp.headers.get('content-type').includes('ndjson')) {
          const err = await resp.json().catch(() => null);
          const info = err && (err.error || (err.payload && err.payload.error));
          throw new Error(info ? info.message || info.code : `HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === 'stage') s.stages.push(event.stage);
            else if (event.type === 'result') s.result = event;
            else if (event.type === 'error') {
              throw new Error(event.error.message || event.error.code);
            }
          }
        }
      } catch (e) {
        s.error = String(e.message || e);
      } finally {
        s.busy = false;
      }
    }

    async function migrate() {
      s.migrating = true; s.error = '';
      try {
        const url = '/v1/databases/' + encodeURIComponent(s.database)
          + '/collections/ingest/migrate';
        const resp = await fetch(url, { method: 'POST' });
        const body = await resp.json().catch(() => null);
        if (!resp.ok) {
          const info = body && body.error;
          throw new Error(info ? info.message || info.code : `HTTP ${resp.status}`);
        }
        await loadCaps();
      } catch (e) {
        s.error = String(e.message || e);
      } finally {
        s.migrating = false;
      }
    }

    onMounted(async () => {
      await loadDatabases();
      await loadCaps();
    });

    return { s, setMode, run, migrate };
  },
  template: `
  <div class="retrieval-panel">
    <div class="section">
      <div class="section-head">
        <h3 class="section-title">{{ s.database }} / ingest</h3>
        <span class="pill accent">POST /v1/retrieval/stream</span>
      </div>

      <div v-if="s.caps.schema_version === 1" class="retrieval-migrate-banner">
        <span>⚠ schema v1：BM25 全文检索不可用</span>
        <button class="btn sm" :disabled="s.migrating" @click="migrate">
          {{ s.migrating ? 'migrating…' : '一键迁移到 v2' }}
        </button>
      </div>

      <div class="retrieval-modes">
        <button v-for="m in ['basic','hybrid','advanced','custom']" :key="m"
          :class="['btn','sm', s.mode === m ? 'primary' : '']"
          @click="setMode(m)">{{ m }}</button>
      </div>

      <div class="row">
        <label>database</label>
        <select v-model="s.database" @change="loadCaps">
          <option v-for="d in s.databases" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>

      <div class="row">
        <label>query</label>
        <textarea v-model="s.query" rows="2" placeholder="输入检索问题"></textarea>
      </div>

      <div class="row">
        <label>top_k</label>
        <input type="number" min="1" max="100" v-model.number="s.topK">
      </div>

      <div class="cap-group-title">channels &amp; fusion</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.dense"> dense</label>
        <label><input type="checkbox" v-model="s.bm25"
          :disabled="s.caps.schema_version === 1"> bm25</label>
      </div>
      <div class="row">
        <label>fusion</label>
        <select v-model="s.fusionMethod">
          <option value="rrf">rrf</option>
          <option value="weighted">weighted</option>
        </select>
        <template v-if="s.fusionMethod === 'rrf'">
          <label>rrf_k</label>
          <input type="number" min="1" max="200" v-model.number="s.rrfK">
        </template>
        <template v-else>
          <label>w dense</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wDense">
          <label>w bm25</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wBm25">
        </template>
      </div>

      <div class="cap-group-title">query rewrite
        <span v-if="!s.caps.llm_configured" class="pill danger">LLM 未配置</span>
      </div>
      <fieldset class="retrieval-fieldset" :disabled="!s.caps.llm_configured">
        <div class="row">
          <label><input type="checkbox" v-model="s.rewrite"> enabled</label>
        </div>
        <div class="row">
          <label v-for="k in ['hyde','multi_query','step_back','decompose']" :key="k">
            <input type="checkbox" v-model="s.methods[k]"> {{ k }}
          </label>
        </div>
        <div class="row">
          <label>hyde α</label>
          <input type="number" min="0" max="1" step="0.05" v-model.number="s.hydeAlpha">
          <label>n variants</label>
          <input type="number" min="1" max="5" v-model.number="s.nVariants">
        </div>
      </fieldset>

      <div class="cap-group-title">diversity &amp; rerank</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.mmr"> mmr</label>
        <label>λ</label>
        <input type="number" min="0" max="1" step="0.05" v-model.number="s.mmrLambda">
      </div>
      <div class="row">
        <label><input type="checkbox" v-model="s.rerank"> cross-encoder rerank</label>
        <label>candidate pool</label>
        <input type="number" min="1" max="64" v-model.number="s.candidatePool">
      </div>

      <div class="cap-group-title">metadata filter</div>
      <div class="row">
        <label>doc_id</label>
        <input v-model="s.docId" placeholder="精确匹配">
        <label>filename</label>
        <input v-model="s.filename" placeholder="模糊匹配">
      </div>

      <div class="actions">
        <button class="btn primary" :disabled="s.busy || !s.query.trim()" @click="run">
          <span v-if="s.busy" class="btn-spinner"></span>
          {{ s.busy ? 'retrieving…' : '检索' }}
        </button>
      </div>

      <div v-if="s.stages.length" class="response retrieval-progress">
        <span v-for="(st, i) in s.stages" :key="i" class="retrieval-stage">
          <span class="spinner" v-if="i === s.stages.length - 1 && s.busy"></span>
          {{ st }}
        </span>
      </div>

      <div v-if="s.error" class="response ingest-result-err">
        <div class="result-banner err">
          <span class="result-glyph err">!</span>
          <span class="result-banner-text">{{ s.error }}</span>
        </div>
      </div>

      <div v-if="s.result" class="retrieval-results">
        <div v-for="(c, i) in s.result.chunks" :key="c.chunk_id" class="retrieval-chunk">
          <div class="retrieval-chunk-head">
            <strong>#{{ i + 1 }}</strong>
            <span v-for="ch in c.matched_channels" :key="ch" class="pill accent">{{ ch }}</span>
            <span class="retrieval-scores">
              fusion {{ c.fusion_score.toFixed(4) }}
              <template v-if="c.rerank_score !== null">
                · rerank {{ c.rerank_score.toFixed(4) }}
              </template>
            </span>
          </div>
          <div class="retrieval-chunk-text">{{ c.fields.text }}</div>
          <div class="retrieval-chunk-meta">
            doc_id {{ c.fields.doc_id }} · chunk #{{ c.fields.chunk_index }}
            <template v-if="c.fields.section_header"> · § {{ c.fields.section_header }}</template>
            <template v-if="c.fields.page_number"> · p.{{ c.fields.page_number }}</template>
            <template v-if="c.fields.filename"> · {{ c.fields.filename }}</template>
          </div>
        </div>

        <div class="retrieval-trace-toggle" @click="s.showTrace = !s.showTrace">
          {{ s.showTrace ? '▾' : '▸' }} 检索追踪
        </div>
        <div v-if="s.showTrace" class="retrieval-trace">
          <div class="stat" v-for="(tr, i) in s.result.traces" :key="i">
            <span class="key">{{ tr.stage }}</span>
            <span class="val">{{ tr.duration_ms }} ms · {{ JSON.stringify(tr.detail) }}</span>
          </div>
          <div class="stat" v-for="(run, i) in s.result.channel_runs" :key="'r'+i">
            <span class="key">{{ run.channel }}: {{ run.query }}</span>
            <span class="val">
              top:
              <template v-for="h in run.hits.slice(0, 3)" :key="h.chunk_id">
                {{ h.chunk_id }}({{ h.score.toFixed(2) }})
              </template>
            </span>
          </div>
          <div class="stat" v-if="s.result.plan.sub_queries.length">
            <span class="key">sub-queries</span>
            <span class="val">{{ s.result.plan.sub_queries.join(' | ') }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  `,
});
