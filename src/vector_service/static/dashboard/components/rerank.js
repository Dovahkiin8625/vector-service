// Rerank panel: cross-encoder rerank.
import { defineComponent, ref, onMounted } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError } from './app.js';

export default defineComponent({
  name: 'RerankPanel',
  setup() {
    const models = ref([]);
    const model = ref('');
    const mode = ref('lines');
    const docs = ref('doc one\ndoc two\ndoc three');
    const query = ref('search query');
    const topN = ref(3);
    const result = ref(null);
    const status = ref('idle');

    async function refreshModels() {
      try {
        const { payload } = await api('GET', '/v1/models');
        models.value = (payload && payload.data || []).filter(m => m.type === 'reranker');
        if (!model.value && models.value.length) {
          const loaded = models.value.find(m => m.loaded);
          model.value = (loaded || models.value[0]).id;
        }
      } catch (_e) {}
    }
    onMounted(refreshModels);

    function buildDocs() {
      if (mode.value === 'json') {
        try { const p = JSON.parse(docs.value); if (Array.isArray(p) && p.every(x => typeof x === 'string')) return p; } catch (_e) {}
        alert('JSON mode requires an array of strings.'); return null;
      }
      return docs.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
    }

    async function doRerank() {
      const list = buildDocs(); if (!list) return;
      const body = { model: model.value, query: query.value, documents: list, top_n: topN.value };
      try {
        status.value = 'loading';
        const { payload } = await api('POST', '/v1/rerank', body);
        result.value = payload;
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }

    return { models, model, mode, docs, query, topN, result, status, refreshModels, doRerank };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">registered rerankers <span class="pill accent">GET /v1/models - type=reranker</span></h3>
        </div>
        <div class="actions"><button class="btn primary" id="btn-rerank-refresh-models" @click="refreshModels">refresh</button></div>
        <div class="list" id="rerank-models-list">
          <div v-if="!models.length" class="empty">no reranker models.</div>
          <div v-for="m in models" :key="m.id" class="list-item" style="cursor:default;">
            <span class="name">{{ m.id }}</span><span class="meta">{{ m.loaded ? 'loaded' : 'unloaded' }} - reranker</span>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head"><h3 class="section-title">cross-encoder rerank <span class="pill accent">POST /v1/rerank</span></h3></div>
        <div class="row split">
          <div class="row"><label>model</label>
            <select id="rerank-model" v-model="model">
              <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
            </select>
          </div>
          <div class="row"><label>top_n</label><input type="number" v-model.number="topN" min="1" max="100" /></div>
        </div>
        <div class="row split">
          <div class="row"><label>query</label><input type="text" v-model="query" /></div>
          <div class="row"><label>input mode</label>
            <select id="rerank-mode" v-model="mode"><option value="lines">one per line</option><option value="json">JSON array</option></select>
          </div>
        </div>
        <div class="row"><label>documents</label><textarea id="rerank-docs" rows="4" v-model="docs"></textarea></div>
        <div class="actions"><button class="btn primary" id="btn-rerank" @click="doRerank">rerank</button></div>
        <div v-if="result" class="response">
          <div class="response-head"><span>results - hits {{ result.results.length }}</span></div>
          <pre class="code-pane">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>
    </div>
  `,
});
