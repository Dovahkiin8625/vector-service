// Similarity panel: text / image / multimodal - kind prop picks endpoint.
import { defineComponent, ref, onMounted, watch } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError } from './app.js';

const TITLES = { text: 'text similarity', image: 'image similarity', multimodal: 'multimodal similarity' };
const ENDPOINTS = { text: '/v1/text_similarity', image: '/v1/image_similarity', multimodal: '/v1/multimodal_similarity' };

export default defineComponent({
  name: 'SimilarityPanel',
  props: { kind: { type: String, default: 'text' } },
  setup(props) {
    const models = ref([]);
    const model = ref('');
    const metric = ref('cosine');
    const query = ref('wireless mouse');
    const docs = ref('bluetooth mouse\nmechanical keyboard\nbluetooth headphone\ngame controller');
    const fileList = ref([]);
    const mime = ref('');
    const result = ref(null);
    const status = ref('idle');

    async function refreshModels() {
      try {
        const { payload } = await api('GET', '/v1/models');
        const wantType = props.kind === 'text' ? 'embedder' : (props.kind === 'image' ? 'image_embedder' : 'multimodal_embedder');
        models.value = (payload && payload.data || []).filter(m => m.type === wantType);
        if (!model.value && models.value.length) {
          const loaded = models.value.find(m => m.loaded);
          model.value = (loaded || models.value[0]).id;
        }
      } catch (_e) {}
    }
    onMounted(refreshModels);
    watch(() => props.kind, refreshModels);

    async function fileToB64(f) {
      return new Promise((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(String(r.result).split(',')[1] || '');
        r.onerror = rej;
        r.readAsDataURL(f);
      });
    }

    async function run() {
      try {
        status.value = 'loading';
        const body = { model: model.value, metric: metric.value };
        if (props.kind === 'text') {
          body.query = query.value;
          body.documents = docs.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
        } else if (props.kind === 'image') {
          if (!fileList.value.length) { alert('select a query image.'); status.value = 'idle'; return; }
          body.query_image = await fileToB64(fileList.value[0]);
          body.query_image_mime = mime.value || fileList.value[0].type || 'image/png';
          body.images = [];
          body.image_mimes = [];
          body.documents = docs.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
        } else {  // multimodal
          const items = [];
          for (const f of fileList.value) {
            items.push({ kind: 'image', data: await fileToB64(f), mime: mime.value || f.type || 'image/png' });
          }
          body.input = items;
        }
        const { payload } = await api('POST', ENDPOINTS[props.kind], body);
        result.value = payload; status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }

    return { models, model, metric, query, docs, fileList, mime, result, status, refreshModels, run };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head"><h3 class="section-title">{{ title(kind) }} <span class="pill accent">POST {{ endpoint(kind) }}</span></h3></div>
        <div class="actions"><button class="btn primary" @click="refreshModels">refresh</button></div>
        <div class="row split">
          <div class="row"><label>model</label>
            <select v-model="model"><option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option></select>
          </div>
          <div class="row"><label>metric</label>
            <select v-model="metric"><option value="cosine">cosine</option><option value="ip">ip</option><option value="l2">l2</option></select>
          </div>
        </div>
        <div v-if="kind === 'text'">
          <div class="row"><label>query text</label><textarea rows="2" v-model="query"></textarea></div>
          <div class="row"><label>candidate documents <span class="hint">one per line</span></label><textarea rows="5" v-model="docs"></textarea></div>
        </div>
        <div v-else>
          <div class="row"><label>query file</label><input type="file" accept="image/*" @change="fileList = $event.target.files" /></div>
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
    title(k) { return TITLES[k] || k; },
    endpoint(k) { return ENDPOINTS[k] || ''; },
  },
});
