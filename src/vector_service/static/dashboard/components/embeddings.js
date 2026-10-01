// Embeddings panel: text / image / multimodal - kind prop picks mode.
import { defineComponent, ref, computed, onMounted, watch } from '../vue.esm-browser.prod.js';
import { t, api, extractApiError } from './app.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';

// Endpoint pills are protocol literals and stay untranslated.
const PILLS = { text: 'POST /v1/embeddings', image: 'POST /v1/image_embeddings', multimodal: 'POST /v1/multimodal_embeddings' };
const DEFAULT_MIME = 'image/png';

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
    const errMsg = ref('');
    const result = ref(null);

    const busy = computed(() => status.value === 'loading');
    // A failed request is worth offering a retry for; a form-validation
    // complaint is not — re-running it would fail the same way.
    const canRetry = ref(false);

    // Model-list load state. Kept apart from the run state above so a
    // failed GET /v1/models cannot be mistaken for "no models exist".
    const modelsStatus = ref('loading');
    const modelsErr = ref('');
    const modelsText = computed(() => {
      if (modelsStatus.value === 'error') return t('common.load_failed') + modelsErr.value;
      if (modelsStatus.value === 'empty') {
        return t('embeddings.no_models', { kind: t('embeddings.kind.' + props.kind) });
      }
      return '';
    });

    async function refreshModels() {
      modelsStatus.value = 'loading';
      modelsErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/models');
        const wantType = props.kind === 'text' ? 'embedder' : (props.kind === 'image' ? 'image_embedder' : 'multimodal_embedder');
        models.value = (payload && payload.data || []).filter(m => m.type === wantType);
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

    function fail(key) {
      status.value = 'error';
      errMsg.value = t(key);
      canRetry.value = false;
    }

    async function run() {
      try {
        status.value = 'loading';
        errMsg.value = '';
        canRetry.value = false;
        if (props.kind === 'text') {
          let arr;
          if (modeText.value === 'list') {
            // In list mode the single-text box is hidden, but a parse
            // failure used to fall back to it anyway — so a typo'd array
            // silently embedded the stale 'hello world' default instead
            // of the text the operator could see (B10).
            const input = safeParse(listInput.value);
            if (!Array.isArray(input)) { fail('embeddings.err.bad_list'); return; }
            arr = input;
          } else {
            arr = [textInput.value];
          }
          const { payload } = await api('POST', '/v1/embeddings', { model: model.value, input: arr });
          result.value = payload;
        } else if (props.kind === 'image') {
          const items = [];
          const parsed = safeParse(listJson.value);
          if (Array.isArray(parsed) && parsed.length) {
            for (const it of parsed) items.push(it);
          } else if (fileList.value.length) {
            for (const f of fileList.value) {
              items.push({ data: await fileToB64(f), mime: mime.value || f.type || DEFAULT_MIME });
            }
          } else { fail('embeddings.err.no_input'); return; }
          const body = { model: model.value, images: items.map(i => i.data), image_mimes: items.map(i => i.mime || mime.value || DEFAULT_MIME) };
          const { payload } = await api('POST', '/v1/image_embeddings', body);
          result.value = payload;
        } else {  // multimodal
          const items = [];
          for (const f of fileList.value) {
            items.push({ kind: 'image', data: await fileToB64(f), mime: mime.value || f.type || DEFAULT_MIME });
          }
          const parsed = safeParse(listJson.value);
          if (Array.isArray(parsed)) {
            for (const t of parsed) if (typeof t === 'string') items.push({ kind: 'text', text: t });
          }
          if (!items.length) { fail('embeddings.err.no_items'); return; }
          const { payload } = await api('POST', '/v1/multimodal_embeddings', { model: model.value, input: items });
          result.value = payload;
        }
        status.value = 'ok';
      } catch (e) {
        status.value = 'error';
        errMsg.value = extractApiError(e, t('common.unknown'));
        canRetry.value = true;
      }
    }

    return {
      models, model, textInput, listInput, fileList, listJson, mime, modeText,
      status, errMsg, result, busy, canRetry, refreshModels, run,
      modelsStatus, modelsText,
    };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('embeddings.title.' + kind) }} <span class="pill accent">{{ pill(kind) }}</span></h3>
        </div>
        <div class="actions">
          <busy-button :busy="modelsStatus === 'loading'" :label="$t('common.refresh')"
                       @click="refreshModels" />
        </div>
        <div class="list" v-if="kind === 'image' || kind === 'multimodal'">
          <div v-for="m in models" :key="m.id" class="list-item" style="cursor:default;">
            <span class="name">{{ m.id }}</span><span class="meta">{{ m.loaded ? $t('common.loaded') : $t('common.unloaded') }}</span>
          </div>
        </div>
        <!-- Loading / failed / genuinely-empty are three different
             situations and used to share one line of copy. -->
        <empty-state v-if="modelsStatus !== 'ok'" :state="modelsStatus" :text="modelsText"
                     :retry="modelsStatus === 'error' ? refreshModels : null" />
      </div>

      <div class="section">
        <div class="row" v-if="kind === 'text'">
          <label>{{ $t('common.model') }}</label>
          <select v-model="model">
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </div>
        <div class="row split" v-if="kind === 'text'">
          <div class="row">
            <label>{{ $t('embeddings.input_mode') }}</label>
            <select v-model="modeText">
              <option value="single">{{ $t('embeddings.mode.single') }}</option>
              <option value="list">{{ $t('embeddings.mode.list') }}</option>
            </select>
          </div>
        </div>
        <div class="row" v-if="kind === 'text' && (modeText === 'single' || !modeText)">
          <label>{{ $t('embeddings.input_text') }}</label><textarea rows="3" v-model="textInput"></textarea>
        </div>
        <div class="row" v-if="kind === 'text' && modeText === 'list'">
          <label>{{ $t('embeddings.list_json_array') }}</label><textarea rows="3" v-model="listInput"></textarea>
        </div>

        <div class="row" v-if="kind !== 'text'">
          <label>{{ $t('common.model') }}</label>
          <select v-model="model">
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>{{ $t('embeddings.select_images') }} <span class="hint">{{ $t('embeddings.hint.local_files') }}</span></label>
          <input type="file" multiple accept="image/png,image/jpeg,image/webp" @change="fileList = $event.target.files" />
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>{{ $t('embeddings.json_list') }} <span class="hint">{{ $t('embeddings.hint.each_item') }}</span></label>
          <textarea rows="4" v-model="listJson" placeholder='[{"data":"<base64>","mime":"image/png"}]'>[]</textarea>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>{{ $t('common.mime') }}</label>
          <select v-model="mime">
            <option value="">{{ $t('common.auto') }}</option>
            <option value="image/png">image/png</option>
            <option value="image/jpeg">image/jpeg</option>
            <option value="image/webp">image/webp</option>
          </select>
        </div>

        <div class="actions">
          <busy-button :busy="busy" :label="$t('common.run')" :busy-label="$t('embeddings.running')"
                       @click="run" />
        </div>
        <!-- Outside v-if="result": the first request's error has no
             result to hang off, which is exactly when it matters. -->
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
    pill(k) { return PILLS[k] || ''; },
  },
});
