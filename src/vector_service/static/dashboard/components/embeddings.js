// Embeddings panel: text / image / multimodal - kind prop picks mode.
import { defineComponent, ref, computed, onMounted, watch } from '../vue.esm-browser.prod.js';
import { t, api, extractApiError, modelsByType, refreshModels as loadModels } from './app.js';
import { fileToB64, copyText, DEFAULT_MIME, IMAGE_MIME_OPTIONS, IMAGE_ACCEPT, modelTypeOf } from './util.js';
import { StatusBanner, BusyButton, EmptyState } from './feedback.js';

// Endpoint pills are protocol literals and stay untranslated.
const PILLS = { text: 'POST /v1/embeddings', image: 'POST /v1/image_embeddings', multimodal: 'POST /v1/multimodal_embeddings' };

export default defineComponent({
  name: 'EmbeddingsPanel',
  props: { kind: { type: String, default: 'text' } },
  setup(props) {
    // S6: the list is a filter over the shared store cache, not a private
    // GET /v1/models copy (each kind mounted one — three at startup).
    const models = computed(() => modelsByType(modelTypeOf(props.kind)));
    const model = ref('');
    const textInput = ref('hello world');
    const listInput = ref('["a", "b", "c"]');
    // The <input type=file> FileList is immutable and re-picked files
    // replace the whole set; an owned array is what the removable chips
    // below need (and what run() reads).
    const files = ref([]);
    const fileInput = ref(null);
    const listJson = ref('[]');
    const mime = ref('');
    const modeText = ref('single');
    const status = ref('idle');
    const errMsg = ref('');
    const result = ref(null);
    const elapsedMs = ref(0);
    const copyState = ref('');  // '' | 'ok' | 'err' — flashes on the button
    let copyTimer = 0;

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
        await loadModels();
        modelsStatus.value = models.value.length ? 'ok' : 'empty';
      } catch (e) {
        modelsStatus.value = 'error';
        modelsErr.value = extractApiError(e, t('common.unknown'));
      }
    }
    // Prefer an already-loaded instance when picking a default.
    watch(models, (list) => {
      if (!model.value && list.length) {
        const loaded = list.find(m => m.loaded);
        model.value = (loaded || list[0]).id;
      }
    }, { immediate: true });
    onMounted(refreshModels);
    watch(() => props.kind, refreshModels);

    function safeParse(s) { try { return JSON.parse(s); } catch (_e) { return null; } }

    // Headline numbers for the response bar; the JSON dump stays folded.
    const resultCount = computed(() => {
      const data = result.value && result.value.data;
      return Array.isArray(data) ? data.length : 0;
    });
    const resultDim = computed(() => {
      const data = result.value && result.value.data;
      const first = Array.isArray(data) ? data[0] : null;
      return first && Array.isArray(first.embedding) ? first.embedding.length : 0;
    });

    function onFiles(e) {
      const picked = Array.from(e.target.files || []);
      for (const f of picked) {
        if (!files.value.some(x => x.name === f.name && x.size === f.size)) files.value.push(f);
      }
      // Clearing the input means re-picking a removed name fires change.
      e.target.value = '';
    }
    function removeFile(i) { files.value.splice(i, 1); }
    function clearFiles() {
      files.value = [];
      if (fileInput.value) fileInput.value = '';
    }

    async function copyResult() {
      const ok = await copyText(JSON.stringify(result.value, null, 2));
      copyState.value = ok ? 'ok' : 'err';
      clearTimeout(copyTimer);
      copyTimer = setTimeout(() => { copyState.value = ''; }, 2000);
    }

    function fail(key) {
      status.value = 'error';
      errMsg.value = t(key);
      canRetry.value = false;
    }

    async function run() {
      if (!model.value) return;
      const t0 = performance.now();
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
          } else if (files.value.length) {
            for (const f of files.value) {
              items.push({ data: await fileToB64(f), mime: mime.value || f.type || DEFAULT_MIME });
            }
          } else { fail('embeddings.err.no_input'); return; }
          const body = { model: model.value, images: items.map(i => i.data), image_mimes: items.map(i => i.mime || mime.value || DEFAULT_MIME) };
          const { payload } = await api('POST', '/v1/image_embeddings', body);
          result.value = payload;
        } else {  // multimodal
          const items = [];
          for (const f of files.value) {
            items.push({ kind: 'image', data: await fileToB64(f), mime: mime.value || f.type || DEFAULT_MIME });
          }
          const parsed = safeParse(listJson.value);
          if (Array.isArray(parsed)) {
            for (const s of parsed) if (typeof s === 'string') items.push({ kind: 'text', text: s });
          }
          if (!items.length) { fail('embeddings.err.no_items'); return; }
          const { payload } = await api('POST', '/v1/multimodal_embeddings', { model: model.value, input: items });
          result.value = payload;
        }
        status.value = 'ok';
        // Includes base64 conversion — the wall time the operator waited.
        // Not in `finally`: a validation fail makes no request and must
        // not restamp the summary still showing the previous result.
        elapsedMs.value = Math.round(performance.now() - t0);
      } catch (e) {
        status.value = 'error';
        errMsg.value = extractApiError(e, t('common.unknown'));
        canRetry.value = true;
        elapsedMs.value = Math.round(performance.now() - t0);
      }
    }

    return {
      models, model, textInput, listInput, files, fileInput, listJson, mime, modeText,
      status, errMsg, result, busy, canRetry, refreshModels, run,
      modelsStatus, modelsText, IMAGE_MIME_OPTIONS, IMAGE_ACCEPT,
      resultCount, resultDim, elapsedMs, copyState, copyResult,
      onFiles, removeFile, clearFiles,
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
        <!-- The non-operable model list that used to sit here duplicated
             the select below without offering any action (S6); the select
             is the single place to pick a model. -->
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
          <label>{{ $t('embeddings.list_json_array') }}</label><textarea rows="3" class="code-input" v-model="listInput"></textarea>
        </div>

        <div class="row" v-if="kind !== 'text'">
          <label>{{ $t('common.model') }}</label>
          <select v-model="model">
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>{{ $t('embeddings.select_images') }} <span class="hint">{{ $t('embeddings.hint.local_files') }}</span></label>
          <input ref="fileInput" type="file" multiple :accept="IMAGE_ACCEPT" @change="onFiles" />
          <!-- The picker's own "N files" text vanishes on blur; chips keep
               the selection visible and let one wrong pick be dropped. -->
          <div class="picked-files" v-if="files.length">
            <span class="hint">{{ $t('embeddings.files_n', { n: files.length }) }}</span>
            <span class="picked-file" v-for="(f, i) in files" :key="f.name + ':' + f.size">
              <span class="picked-name">{{ f.name }}</span>
              <button type="button" class="btn sm"
                      :aria-label="$t('embeddings.file_remove', { name: f.name })"
                      @click="removeFile(i)">&#215;</button>
            </span>
            <button type="button" class="btn sm" @click="clearFiles">{{ $t('common.clear') }}</button>
          </div>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>{{ $t('embeddings.json_list') }} <span class="hint">{{ $t('embeddings.hint.each_item') }}</span></label>
          <textarea rows="4" class="code-input" v-model="listJson" placeholder='[{"data":"<base64>","mime":"image/png"}]'>[]</textarea>
        </div>
        <div class="row" v-if="kind === 'image' || kind === 'multimodal'">
          <label>{{ $t('common.mime') }}</label>
          <select v-model="mime">
            <option value="">{{ $t('common.auto') }}</option>
            <option v-for="m in IMAGE_MIME_OPTIONS" :key="m" :value="m">{{ m }}</option>
          </select>
        </div>

        <div class="actions">
          <!-- An empty model list left model: '' submittable; the button
               now carries that state (S6). -->
          <busy-button :busy="busy" :disabled="!model" :label="$t('common.run')"
                       :busy-label="$t('embeddings.running')" @click="run" />
        </div>
        <!-- Outside v-if="result": the first request's error has no
             result to hang off, which is exactly when it matters. -->
        <status-banner kind="error" :text="status === 'error' ? errMsg : ''"
                       :retry="canRetry ? run : null" />
        <div v-if="result" class="response">
          <div class="response-head">
            <span v-if="status === 'ok'" class="pill success">{{ $t('common.status.ok') }}</span>
            <!-- count/dim/ms is what a run is checked for; the raw JSON
                 is evidence, folded by default (S7). -->
            <span class="emb-summary">{{ $t('embeddings.summary', { count: resultCount, dim: resultDim, ms: elapsedMs }) }}</span>
            <span class="head-actions">
              <button type="button" class="btn sm" id="btn-emb-copy"
                      @click="copyResult">{{ copyState === 'ok' ? $t('common.copied') : (copyState === 'err' ? $t('common.copy_failed') : $t('common.copy')) }}</button>
            </span>
          </div>
          <details class="collapsible">
            <summary>{{ $t('common.raw_json') }}</summary>
            <div class="body"><pre class="code-pane">{{ JSON.stringify(result, null, 2) }}</pre></div>
          </details>
        </div>
      </div>
    </div>
  `,
  methods: {
    pill(k) { return PILLS[k] || ''; },
  },
});
