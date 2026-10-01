// Similarity panel: text / image / multimodal merged into one panel.
// The internal mode switch picks the endpoint; request bodies match
// schemas/similarity.py (each family: `query` + `documents`).
import { defineComponent, ref, computed, onMounted, watch } from '../vue.esm-browser.prod.js';
import { t, api, extractApiError, modelsByType, refreshModels as loadModels } from './app.js';
import { fileToB64, DEFAULT_MIME, IMAGE_MIME_OPTIONS, IMAGE_ACCEPT } from './util.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';

// `key` is a mode identifier (also the i18n key suffix), `endpoint` and
// `modelType` are protocol literals.
const MODES = [
  { key: 'text', endpoint: '/v1/text_similarity', modelType: 'embedder' },
  { key: 'image', endpoint: '/v1/image_similarity', modelType: 'image_embedder' },
  { key: 'multimodal', endpoint: '/v1/multimodal_similarity', modelType: 'multimodal_embedder' },
];

export default defineComponent({
  name: 'SimilarityPanel',
  setup() {
    const modes = MODES;
    const mode = ref('text');
    // S6: shared store cache; switching mode re-filters this computed
    // instead of firing another GET /v1/models.
    const models = computed(() => {
      const wantType = MODES.find(m => m.key === mode.value).modelType;
      return modelsByType(wantType);
    });
    const model = ref('');
    const metric = ref('cosine');

    // text mode
    const query = ref('wireless mouse');
    const docs = ref('bluetooth mouse\nmechanical keyboard\nbluetooth headphone\ngame controller');

    // image mode
    const queryFiles = ref([]);
    const docFiles = ref([]);

    // multimodal mode
    const queryModality = ref('text');
    const queryText = ref('一只猫');
    const docText = ref('一只狗\n一只猫\n一辆汽车');

    const mime = ref('');
    const result = ref(null);
    const status = ref('idle');
    const errMsg = ref('');

    const busy = computed(() => status.value === 'loading');
    // Validation complaints are not retryable; a failed request is.
    const canRetry = ref(false);

    // Model-list load state, kept apart from the run state so a failed
    // GET /v1/models is never shown as "no models for this modality".
    const modelsStatus = ref('loading');
    const modelsErr = ref('');
    const modelsText = computed(() => (
      modelsStatus.value === 'error'
        ? t('common.load_failed') + modelsErr.value
        : t('embeddings.no_models', { kind: t('similarity.mode.' + mode.value) })
    ));

    function autoSelect() {
      if (model.value) return;
      const list = models.value;
      if (!list.length) return;
      const loaded = list.find(m => m.loaded);
      model.value = (loaded || list[0]).id;
    }

    async function refreshModels() {
      modelsStatus.value = 'loading';
      modelsErr.value = '';
      try {
        await loadModels();
        autoSelect();
        modelsStatus.value = models.value.length ? 'ok' : 'empty';
      } catch (e) {
        modelsStatus.value = 'error';
        modelsErr.value = extractApiError(e, t('common.unknown'));
      }
    }
    // Switching mode re-filters the model list and drops the previous
    // run's result (the response belongs to a different endpoint/family).
    // autoSelect runs at the end of this watcher so the cleared selection
    // is refilled for the new family regardless of watcher order.
    watch(mode, () => {
      model.value = '';
      result.value = null;
      status.value = 'idle';
      errMsg.value = '';
      autoSelect();
      if (modelsStatus.value === 'error') refreshModels();
    });
    watch(models, (list) => {
      if (!model.value && list.length) {
        const loaded = list.find(m => m.loaded);
        model.value = (loaded || list[0]).id;
      }
    }, { immediate: true });
    onMounted(refreshModels);

    function mimeOf(f) { return mime.value || f.type || DEFAULT_MIME; }

    // Reported through the panel's own banner instead of alert(): the
    // message stays next to the form that produced it.
    function failValidation(key) {
      status.value = 'error';
      errMsg.value = t(key);
      canRetry.value = false;
      return false;
    }

    async function run() {
      if (!model.value) return;
      try {
        status.value = 'loading';
        errMsg.value = '';
        canRetry.value = false;
        const body = { model: model.value, metric: metric.value };

        if (mode.value === 'text') {
          const documents = docs.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
          if (!query.value.trim()) return failValidation('similarity.err.query_empty');
          if (!documents.length) return failValidation('similarity.err.no_docs');
          body.query = query.value;
          body.documents = documents;
        } else if (mode.value === 'image') {
          if (!queryFiles.value.length) return failValidation('similarity.err.no_query_image');
          if (!docFiles.value.length) return failValidation('similarity.err.no_doc_images');
          body.query = { data: await fileToB64(queryFiles.value[0]), mime: mimeOf(queryFiles.value[0]) };
          body.documents = await Promise.all(Array.from(docFiles.value).map(async f => ({
            data: await fileToB64(f), mime: mimeOf(f),
          })));
        } else {  // multimodal
          if (queryModality.value === 'text') {
            if (!queryText.value.trim()) return failValidation('similarity.err.query_empty');
            body.query = { text: queryText.value };
          } else {
            if (!queryFiles.value.length) return failValidation('similarity.err.no_query_image');
            body.query = { image: { data: await fileToB64(queryFiles.value[0]), mime: mimeOf(queryFiles.value[0]) } };
          }
          const documents = docText.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean).map(s => ({ text: s }));
          for (const f of Array.from(docFiles.value)) {
            documents.push({ image: { data: await fileToB64(f), mime: mimeOf(f) } });
          }
          if (!documents.length) return failValidation('similarity.err.no_candidates');
          body.documents = documents;
        }

        const endpoint = MODES.find(m => m.key === mode.value).endpoint;
        const { payload } = await api('POST', endpoint, body);
        result.value = payload; status.value = 'ok';
      } catch (e) {
        status.value = 'error';
        errMsg.value = extractApiError(e, t('common.unknown'));
        canRetry.value = true;
      }
    }

    return {
      modes, mode, models, model, metric, query, docs,
      queryFiles, docFiles, queryModality, queryText, docText,
      mime, result, status, errMsg, busy, canRetry, refreshModels, run,
      modelsStatus, modelsText, IMAGE_MIME_OPTIONS, IMAGE_ACCEPT,
    };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head"><h3 class="section-title">{{ $t('similarity.title.' + mode) }} <span class="pill accent">POST {{ endpoint(mode) }}</span></h3></div>
        <div class="actions">
          <span class="seg-toggle">
            <button v-for="m in modes" :key="m.key" :class="['btn', mode === m.key ? 'primary' : '']" @click="mode = m.key">{{ $t('similarity.mode.' + m.key) }}</button>
          </span>
          <busy-button :busy="modelsStatus === 'loading'" :label="$t('common.refresh')"
                       @click="refreshModels" />
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.model') }}</label>
            <select v-model="model"><option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option></select>
          </div>
          <div class="row"><label>{{ $t('common.metric') }}</label>
            <select v-model="metric"><option value="cosine">cosine</option><option value="ip">ip</option><option value="l2">l2</option></select>
          </div>
        </div>
        <empty-state v-if="modelsStatus !== 'ok'" :state="modelsStatus" :text="modelsText"
                     :retry="modelsStatus === 'error' ? refreshModels : null" />

        <div v-if="mode === 'text'">
          <div class="row"><label>{{ $t('similarity.query_text') }}</label><textarea rows="2" v-model="query"></textarea></div>
          <div class="row"><label>{{ $t('similarity.candidate_docs') }} <span class="hint">{{ $t('similarity.hint.one_per_line') }}</span></label><textarea rows="5" v-model="docs"></textarea></div>
        </div>

        <div v-else-if="mode === 'image'">
          <div class="row"><label>{{ $t('similarity.query_image') }}</label><input type="file" :accept="IMAGE_ACCEPT" @change="queryFiles = $event.target.files" /></div>
          <div class="row"><label>{{ $t('similarity.candidate_images') }} <span class="hint">{{ $t('similarity.hint.pick_several') }}</span></label><input type="file" multiple :accept="IMAGE_ACCEPT" @change="docFiles = $event.target.files" /></div>
          <div class="row"><label>{{ $t('common.mime') }}</label>
            <select v-model="mime">
              <option value="">{{ $t('common.auto') }}</option>
              <option v-for="m in IMAGE_MIME_OPTIONS" :key="m" :value="m">{{ m }}</option>
            </select>
          </div>
        </div>

        <div v-else>
          <div class="row"><label>{{ $t('similarity.query_modality') }}</label>
            <select v-model="queryModality">
              <option value="text">{{ $t('similarity.modality.text') }}</option>
              <option value="image">{{ $t('similarity.modality.image') }}</option>
            </select>
          </div>
          <div class="row" v-if="queryModality === 'text'"><label>{{ $t('similarity.query_text') }}</label><textarea rows="2" v-model="queryText"></textarea></div>
          <div class="row" v-else><label>{{ $t('similarity.query_image') }}</label><input type="file" :accept="IMAGE_ACCEPT" @change="queryFiles = $event.target.files" /></div>
          <div class="row"><label>{{ $t('similarity.candidate_texts') }} <span class="hint">{{ $t('similarity.hint.one_per_line') }}</span></label><textarea rows="4" v-model="docText"></textarea></div>
          <div class="row"><label>{{ $t('similarity.candidate_images') }}</label><input type="file" multiple :accept="IMAGE_ACCEPT" @change="docFiles = $event.target.files" /></div>
          <div class="row"><label>{{ $t('common.mime') }}</label>
            <select v-model="mime">
              <option value="">{{ $t('common.auto') }}</option>
              <option v-for="m in IMAGE_MIME_OPTIONS" :key="m" :value="m">{{ m }}</option>
            </select>
          </div>
        </div>

        <div class="actions">
          <!-- An empty model list left model: '' submittable; the button
               now carries that state (S6). -->
          <busy-button :busy="busy" :disabled="!model" :label="$t('common.run')"
                       :busy-label="$t('similarity.running')" @click="run" />
        </div>
        <!-- Outside v-if="result": validation complaints and a failed
             first request both have no result to render inside. -->
        <status-banner kind="error" :text="status === 'error' ? errMsg : ''"
                       :retry="canRetry ? run : null" />
        <div v-if="result" class="response">
          <div class="response-head">
            <span v-if="status === 'ok'" class="pill success">{{ $t('common.status.ok') }}</span>
          </div>
          <pre class="code-pane">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>
    </div>
  `,
  methods: {
    endpoint(k) { const m = MODES.find(x => x.key === k); return m ? m.endpoint : ''; },
  },
});
