// Rerank panel: cross-encoder rerank.
import { defineComponent, ref, computed, onMounted, watch } from '../vue.esm-browser.prod.js';
import { t, api, extractApiError, modelsByType, refreshModels as loadModels } from './app.js';
import { intError, modelTypeOf } from './util.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';

export default defineComponent({
  name: 'RerankPanel',
  setup() {
    // S6: shared store cache filtered to rerankers — no private GET copy.
    const models = computed(() => modelsByType(modelTypeOf('rerank')));
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
        await loadModels();
        modelsStatus.value = models.value.length ? 'ok' : 'empty';
      } catch (e) {
        modelsStatus.value = 'error';
        modelsErr.value = extractApiError(e, t('common.unknown'));
      }
    }
    watch(models, (list) => {
      if (!model.value && list.length) {
        const loaded = list.find(m => m.loaded);
        model.value = (loaded || list[0]).id;
      }
    }, { immediate: true });
    onMounted(refreshModels);

    // S6 form rules: top_n is an integer in 1..100, and the required
    // fields must be present before the submit button goes live —
    // cleared number boxes used to submit NaN/'' unchecked.
    const topNErr = computed(() => intError(topN.value, 1, 100));
    const canRun = computed(() => (
      !!model.value && !!query.value.trim() && !!docs.value.trim() && !topNErr.value
    ));

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
      if (!canRun.value) return;
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
      topNErr, canRun,
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
          <busy-button id="btn-rerank-refresh-models" variant="ghost" :busy="modelsStatus === 'loading'"
                       :label="$t('common.refresh')" @click="refreshModels" />
        </div>
        <!-- The non-operable model list that used to sit here duplicated
             the select below without offering any action (S6). -->
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
          <div class="row"><label>top_n</label><input type="number" v-model.number="topN" min="1" max="100" />
            <span class="hint" v-if="topNErr">{{ topNErr }}</span></div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.query') }}</label><textarea rows="2" v-model="query"></textarea></div>
          <div class="row"><label>{{ $t('rerank.input_mode') }}</label>
            <select id="rerank-mode" v-model="mode">
              <option value="lines">{{ $t('rerank.mode.lines') }}</option>
              <option value="json">{{ $t('rerank.mode.json') }}</option>
            </select>
          </div>
        </div>
        <div class="row"><label>{{ $t('rerank.documents') }}</label><textarea id="rerank-docs" rows="4" v-model="docs"></textarea></div>
        <div class="actions">
          <busy-button id="btn-rerank" variant="primary" :busy="busy" :disabled="!canRun" :label="$t('rerank.run')"
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
