// Rerank panel: cross-encoder rerank.
import { defineComponent, ref, computed, onMounted } from '../vue.esm-browser.prod.js';
import { t, api, extractApiError } from './app.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';

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
    const errMsg = ref('');

    const busy = computed(() => status.value === 'loading');
    // Only a failed request gets a retry button; a malformed JSON body
    // would fail the same way twice.
    const canRetry = ref(false);

    // Model-list load state, kept apart from the run state so a failed
    // GET /v1/models is never shown as "no reranker models registered".
    const modelsStatus = ref('loading');
    const modelsErr = ref('');
    const modelsText = computed(() => (
      modelsStatus.value === 'error'
        ? t('common.load_failed') + modelsErr.value
        : t('rerank.no_models')
    ));

    async function refreshModels() {
      modelsStatus.value = 'loading';
      modelsErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/models');
        models.value = (payload && payload.data || []).filter(m => m.type === 'reranker');
        if (!model.value && models.value.length) {
          const loaded = models.value.find(m => m.loaded);
          model.value = (loaded || models.value[0]).id;
        }
        modelsStatus.value = models.value.length ? 'ok' : 'empty';
      } catch (e) {
        models.value = [];
        modelsStatus.value = 'error';
        modelsErr.value = extractApiError(e, t('common.unknown'));
      }
    }
    onMounted(refreshModels);

    function fail(key) {
      status.value = 'error';
      errMsg.value = t(key);
      canRetry.value = false;
    }

    function buildDocs() {
      if (mode.value === 'json') {
        // A parse failure is the expected outcome of a bad paste, so it
        // is reported through the banner rather than the console.
        try { const p = JSON.parse(docs.value); if (Array.isArray(p) && p.every(x => typeof x === 'string')) return p; } catch (_e) {}
        fail('rerank.err.json'); return null;
      }
      return docs.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
    }

    async function doRerank() {
      const list = buildDocs(); if (!list) return;
      const body = { model: model.value, query: query.value, documents: list, top_n: topN.value };
      try {
        status.value = 'loading';
        errMsg.value = '';
        canRetry.value = false;
        const { payload } = await api('POST', '/v1/rerank', body);
        result.value = payload;
        status.value = 'ok';
      } catch (e) {
        status.value = 'error';
        errMsg.value = extractApiError(e, t('common.unknown'));
        canRetry.value = true;
      }
    }

    return {
      models, model, mode, docs, query, topN, result, status, errMsg,
      busy, canRetry, refreshModels, doRerank, modelsStatus, modelsText,
    };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('rerank.registered') }} <span class="pill accent">GET /v1/models - type=reranker</span></h3>
        </div>
        <div class="actions">
          <busy-button id="btn-rerank-refresh-models" :busy="modelsStatus === 'loading'"
                       :label="$t('common.refresh')" @click="refreshModels" />
        </div>
        <div class="list" id="rerank-models-list">
          <div v-for="m in models" :key="m.id" class="list-item" style="cursor:default;">
            <span class="name">{{ m.id }}</span><span class="meta">{{ m.loaded ? $t('common.loaded') : $t('common.unloaded') }} - {{ $t('rerank.suffix') }}</span>
          </div>
        </div>
        <empty-state v-if="modelsStatus !== 'ok'" :state="modelsStatus" :text="modelsText"
                     :retry="modelsStatus === 'error' ? refreshModels : null" />
      </div>

      <div class="section">
        <div class="section-head"><h3 class="section-title">{{ $t('rerank.title') }} <span class="pill accent">POST /v1/rerank</span></h3></div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.model') }}</label>
            <select id="rerank-model" v-model="model">
              <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
            </select>
          </div>
          <div class="row"><label>top_n</label><input type="number" v-model.number="topN" min="1" max="100" /></div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.query') }}</label><input type="text" v-model="query" /></div>
          <div class="row"><label>{{ $t('rerank.input_mode') }}</label>
            <select id="rerank-mode" v-model="mode">
              <option value="lines">{{ $t('rerank.mode.lines') }}</option>
              <option value="json">{{ $t('rerank.mode.json') }}</option>
            </select>
          </div>
        </div>
        <div class="row"><label>{{ $t('rerank.documents') }}</label><textarea id="rerank-docs" rows="4" v-model="docs"></textarea></div>
        <div class="actions">
          <busy-button id="btn-rerank" :busy="busy" :label="$t('rerank.run')"
                       :busy-label="$t('rerank.running')" @click="doRerank" />
        </div>
        <!-- Was previously rendered only inside v-if="result" (and only
             as a status pill), so a first-run failure showed nothing. -->
        <status-banner kind="error" :text="status === 'error' ? errMsg : ''"
                       :retry="canRetry ? doRerank : null" />
        <div v-if="result" class="response">
          <div class="response-head">
            <span class="pill accent">{{ $t('rerank.results', { n: result.results.length }) }}</span>
            <span v-if="status === 'ok'" class="pill success">{{ $t('common.status.ok') }}</span>
          </div>
          <pre class="code-pane">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>
    </div>
  `,
});
