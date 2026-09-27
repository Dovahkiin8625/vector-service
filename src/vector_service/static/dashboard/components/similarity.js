// Similarity panel: text / image / multimodal merged into one panel.
// The internal mode switch picks the endpoint; request bodies match
// schemas/similarity.py (each family: `query` + `documents`).
import { defineComponent, ref, onMounted, watch } from '../vue.esm-browser.prod.js';
import { api, extractApiError } from './app.js';

const MODES = [
  { key: 'text', title: 'text similarity', endpoint: '/v1/text_similarity', modelType: 'embedder' },
  { key: 'image', title: 'image similarity', endpoint: '/v1/image_similarity', modelType: 'image_embedder' },
  { key: 'multimodal', title: 'multimodal similarity', endpoint: '/v1/multimodal_similarity', modelType: 'multimodal_embedder' },
];

export default defineComponent({
  name: 'SimilarityPanel',
  setup() {
    const modes = MODES;
    const mode = ref('text');
    const models = ref([]);
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

    async function refreshModels() {
      try {
        const { payload } = await api('GET', '/v1/models');
        const wantType = MODES.find(m => m.key === mode.value).modelType;
        models.value = (payload && payload.data || []).filter(m => m.type === wantType);
        if (!model.value && models.value.length) {
          const loaded = models.value.find(m => m.loaded);
          model.value = (loaded || models.value[0]).id;
        }
      } catch (_e) {}
    }
    onMounted(refreshModels);

    // Switching mode re-filters the model list and drops the previous
    // run's result (the response belongs to a different endpoint/family).
    watch(mode, () => {
      model.value = '';
      result.value = null;
      status.value = 'idle';
      refreshModels();
    });

    async function fileToB64(f) {
      return new Promise((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(String(r.result).split(',')[1] || '');
        r.onerror = rej;
        r.readAsDataURL(f);
      });
    }
    function mimeOf(f) { return mime.value || f.type || 'image/png'; }

    function failValidation(msg) { alert(msg); status.value = 'idle'; return false; }

    async function run() {
      try {
        status.value = 'loading';
        const body = { model: model.value, metric: metric.value };

        if (mode.value === 'text') {
          const documents = docs.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
          if (!query.value.trim()) return failValidation('query text must not be empty.');
          if (!documents.length) return failValidation('add at least one candidate document.');
          body.query = query.value;
          body.documents = documents;
        } else if (mode.value === 'image') {
          if (!queryFiles.value.length) return failValidation('select a query image.');
          if (!docFiles.value.length) return failValidation('select at least one candidate image.');
          body.query = { data: await fileToB64(queryFiles.value[0]), mime: mimeOf(queryFiles.value[0]) };
          body.documents = await Promise.all(Array.from(docFiles.value).map(async f => ({
            data: await fileToB64(f), mime: mimeOf(f),
          })));
        } else {  // multimodal
          if (queryModality.value === 'text') {
            if (!queryText.value.trim()) return failValidation('query text must not be empty.');
            body.query = { text: queryText.value };
          } else {
            if (!queryFiles.value.length) return failValidation('select a query image.');
            body.query = { image: { data: await fileToB64(queryFiles.value[0]), mime: mimeOf(queryFiles.value[0]) } };
          }
          const documents = docText.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean).map(s => ({ text: s }));
          for (const f of Array.from(docFiles.value)) {
            documents.push({ image: { data: await fileToB64(f), mime: mimeOf(f) } });
          }
          if (!documents.length) return failValidation('add at least one text or image candidate.');
          body.documents = documents;
        }

        const endpoint = MODES.find(m => m.key === mode.value).endpoint;
        const { payload } = await api('POST', endpoint, body);
        result.value = payload; status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }

    return {
      modes, mode, models, model, metric, query, docs,
      queryFiles, docFiles, queryModality, queryText, docText,
      mime, result, status, refreshModels, run,
    };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head"><h3 class="section-title">{{ title(mode) }} <span class="pill accent">POST {{ endpoint(mode) }}</span></h3></div>
        <div class="actions">
          <span class="seg-toggle">
            <button v-for="m in modes" :key="m.key" :class="['btn', mode === m.key ? 'primary' : '']" @click="mode = m.key">{{ m.key }}</button>
          </span>
          <button class="btn primary" @click="refreshModels">refresh</button>
        </div>
        <div class="row split">
          <div class="row"><label>model</label>
            <select v-model="model"><option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option></select>
          </div>
          <div class="row"><label>metric</label>
            <select v-model="metric"><option value="cosine">cosine</option><option value="ip">ip</option><option value="l2">l2</option></select>
          </div>
        </div>

        <div v-if="mode === 'text'">
          <div class="row"><label>query text</label><textarea rows="2" v-model="query"></textarea></div>
          <div class="row"><label>candidate documents <span class="hint">one per line</span></label><textarea rows="5" v-model="docs"></textarea></div>
        </div>

        <div v-else-if="mode === 'image'">
          <div class="row"><label>query image</label><input type="file" accept="image/*" @change="queryFiles = $event.target.files" /></div>
          <div class="row"><label>candidate images <span class="hint">pick several at once</span></label><input type="file" multiple accept="image/png,image/jpeg,image/webp" @change="docFiles = $event.target.files" /></div>
          <div class="row"><label>MIME</label>
            <select v-model="mime">
              <option value="">auto</option><option value="image/png">image/png</option>
              <option value="image/jpeg">image/jpeg</option><option value="image/webp">image/webp</option>
            </select>
          </div>
        </div>

        <div v-else>
          <div class="row"><label>query modality</label>
            <select v-model="queryModality"><option value="text">text</option><option value="image">image</option></select>
          </div>
          <div class="row" v-if="queryModality === 'text'"><label>query text</label><textarea rows="2" v-model="queryText"></textarea></div>
          <div class="row" v-else><label>query image</label><input type="file" accept="image/*" @change="queryFiles = $event.target.files" /></div>
          <div class="row"><label>candidate texts <span class="hint">one per line</span></label><textarea rows="4" v-model="docText"></textarea></div>
          <div class="row"><label>candidate images</label><input type="file" multiple accept="image/png,image/jpeg,image/webp" @change="docFiles = $event.target.files" /></div>
          <div class="row"><label>MIME</label>
            <select v-model="mime">
              <option value="">auto</option><option value="image/png">image/png</option>
              <option value="image/jpeg">image/jpeg</option><option value="image/webp">image/webp</option>
            </select>
          </div>
        </div>

        <div class="actions"><button class="btn primary" @click="run">run</button></div>
        <div v-if="result" class="response">
          <div class="response-head"><span>status: {{ status }}</span></div>
          <pre class="code-pane">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>
    </div>
  `,
  methods: {
    title(k) { const m = MODES.find(x => x.key === k); return m ? m.title : k; },
    endpoint(k) { const m = MODES.find(x => x.key === k); return m ? m.endpoint : ''; },
  },
});
