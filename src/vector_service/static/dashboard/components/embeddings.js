// Embeddings panel: text / image / multimodal - kind prop picks mode.
import { defineComponent, ref, onMounted, watch } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError } from './app.js';

const TITLES = { text: 'text embeddings', image: 'image embeddings', multimodal: 'multimodal embeddings' };
const PILLS = { text: 'POST /v1/embeddings', image: 'POST /v1/image_embeddings', multimodal: 'POST /v1/multimodal_embeddings' };

export default defineComponent({
  name: 'EmbeddingsPanel',
  props: { kind: { type: String, default: 'text' } },
  setup(props) {
    const models = ref([]);
    const model = ref('');
    const textInput = ref('hello world');
    const listInput = ref('["a", "b", "c"]');
    const fileList = ref([]);
    const listJson = ref('[]');
    const mime = ref('');
    const modeText = ref('single');
    const status = ref('idle');
    const result = ref(null);

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

    function safeParse(s) { try { return JSON.parse(s); } catch (_e) { return null; } }

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
        if (props.kind === 'text') {
          const input = safeParse(listInput.value);
          const arr = Array.isArray(input) ? input : [textInput.value];
          const { payload } = await api('POST', '/v1/embeddings', { model: model.value, input: arr });
          result.value = payload;
        } else if (props.kind === 'image') {
          const items = [];
          const parsed = safeParse(listJson.value);
          if (Array.isArray(parsed) && parsed.length) {
            for (const it of parsed) items.push(it);
          } else if (fileList.value.length) {
            for (const f of fileList.value) {
              items.push({ data: await fileToB64(f), mime: mime.value || f.type || 'image/png' });
            }
          } else { alert('select files or paste a JSON array.'); status.value = 'idle'; return; }
          const body = { model: model.value, images: items.map(i => i.data), image_mimes: items.map(i => i.mime || mime.value || 'image/png') };
          const { payload } = await api('POST', '/v1/image_embeddings', body);
          result.value = payload;
        } else {  // multimodal
          const items = [];
          for (const f of fileList.value) {
            items.push({ kind: 'image', data: await fileToB64(f), mime: mime.value || f.type || 'image/png' });
          }
          const parsed = safeParse(listJson.value);
          if (Array.isArray(parsed)) {
            for (const t of parsed) if (typeof t === 'string') items.push({ kind: 'text', text: t });
          }
          if (!items.length) { alert('add at least one text or image item.'); status.value = 'idle'; return; }
          const { payload } = await api('POST', '/v1/multimodal_embeddings', { model: model.value, input: items });
          result.value = payload;
        }
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }

    return { models, model, textInput, listInput, fileList, listJson, mime, modeText, status, result, refreshModels, run };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ title(kind) }} <span class="pill accent">{{ pill(kind) }}</span></h3>
        </div>
        <div class="actions">
          <button class="btn primary" @click="refreshModels">refresh</button>
        </div>
        <div class="list" v-if="kind === 'image' || kind === 'multimodal'">
          <div v-if="!models.length" class="empty">no {{ kind }} models.</div>
          <div v-for="m in models" :key="m.id" class="list-item" style="cursor:default;">
            <span class="name">{{ m.id }}</span><span class="meta">{{ m.loaded ? 'loaded' : 'unloaded' }}</span>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="row" v-if="kind === 'text'">
          <label>model</label>
          <select v-model="model">
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </div>
        <div class="row split" v-if="kind === 'text'">
          <div class="row">
            <label>input mode</label>
            <select v-model="modeText"><option value="single">single</option><option value="list">list</option></select>
          </div>
        </div>
        <div class="row" v-if="kind === 'text' && (modeText === 'single' || !modeText)">
          <label>input text</label><textarea rows="3" v-model="textInput"></textarea>
        </div>
        <div class="row" v-if="kind === 'text' && modeText === 'list'">
          <label>list (JSON array)</label><textarea rows="3" v-model="listInput"></textarea>
        </div>

        <div class="row" v-if="kind !== 'text'">
          <label>model</label>
          <select v-model="model">
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>select images <span class="hint">local files -> auto base64</span></label>
          <input type="file" multiple accept="image/png,image/jpeg,image/webp" @change="fileList = $event.target.files" />
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>JSON list <span class="hint">each item: {data, mime}</span></label>
          <textarea rows="4" v-model="listJson" placeholder='[{"data":"<base64>","mime":"image/png"}]'>[]</textarea>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>MIME</label>
          <select v-model="mime">
            <option value="">auto</option>
            <option value="image/png">image/png</option>
            <option value="image/jpeg">image/jpeg</option>
            <option value="image/webp">image/webp</option>
          </select>
        </div>

        <div class="actions">
          <button class="btn primary" @click="run">run</button>
        </div>
        <div v-if="result" class="response">
          <div class="response-head"><span>status: {{ status }}</span></div>
          <pre class="code-pane">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>
    </div>
  `,
  methods: {
    title(k) { return TITLES[k] || k; },
    pill(k) { return PILLS[k] || ''; },
  },
});
