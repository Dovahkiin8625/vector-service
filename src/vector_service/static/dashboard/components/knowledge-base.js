// Knowledge base panels: parse / chunk / ingest.
import { defineComponent, ref, onMounted } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError } from './app.js';

export default defineComponent({
  name: 'KnowledgeBasePanel',
  props: { view: { type: String, default: 'parse' } },
  setup(props) {
    const status = ref('');
    const parseFile = ref(null);
    const parseResult = ref(null);
    const chunkSize = ref(800);
    const chunkOverlap = ref(80);
    const chunkMd = ref('# title\n\nThis is sample text. Second paragraph.');
    const chunkResults = ref([]);
    const dbs = ref([]);
    const db = ref('');
    const models = ref([]);
    const model = ref('');
    const ingestFile = ref(null);
    const chunkParams = ref({ size: 800, overlap: 80 });
    const metadata = ref('{}');
    const ingestResult = ref(null);

    async function refreshDbs() {
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
      } catch (_e) {}
    }
    async function refreshModels() {
      try {
        const { payload } = await api('GET', '/v1/models');
        models.value = (payload && payload.data || []).filter(m => m.type === 'embedder');
        if (!model.value && models.value.length) {
          const loaded = models.value.find(m => m.loaded);
          model.value = (loaded || models.value[0]).id;
        }
      } catch (_e) {}
    }
    onMounted(() => { refreshDbs(); refreshModels(); });

    async function doParse() {
      if (!parseFile.value) { alert('select a file.'); return; }
      try {
        const form = new FormData();
        form.append('file', parseFile.value);
        const r = await fetch('/v1/parse', { method: 'POST', body: form });
        const payload = await r.json();
        if (!r.ok) throw new Error((payload.error && payload.error.message) || ('HTTP ' + r.status));
        parseResult.value = payload;
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }
    async function doChunk() {
      try {
        const { payload } = await api('POST', '/v1/chunk', {
          text: chunkMd.value, size: chunkSize.value, overlap: chunkOverlap.value,
        });
        chunkResults.value = (payload && payload.chunks) || [];
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }
    async function doIngest() {
      if (!db.value) { alert('select a database.'); return; }
      if (!ingestFile.value) { alert('select a file.'); return; }
      try {
        const form = new FormData();
        form.append('file', ingestFile.value);
        form.append('database', db.value);
        form.append('model', model.value);
        form.append('size', String(chunkParams.value.size));
        form.append('overlap', String(chunkParams.value.overlap));
        try { const m = JSON.parse(metadata.value); for (const k of Object.keys(m)) form.append('metadata.' + k, m[k]); } catch (_e) {}
        const r = await fetch('/v1/ingest', { method: 'POST', body: form });
        const payload = await r.json();
        if (!r.ok) throw new Error((payload.error && payload.error.message) || ('HTTP ' + r.status));
        ingestResult.value = payload;
        status.value = 'ok';
      } catch (e) { status.value = 'error: ' + extractApiError(e, 'unknown'); }
    }

    return { status, parseFile, parseResult, chunkSize, chunkOverlap, chunkMd, chunkResults,
             dbs, db, models, model, ingestFile, chunkParams, metadata, ingestResult,
             refreshDbs, refreshModels, doParse, doChunk, doIngest };
  },
  template: `
    <div>
      <div class="section" v-show="view === 'parse'">
        <div class="section-head"><h3 class="section-title">parse <span class="pill accent">POST /v1/parse</span></h3></div>
        <div class="row"><label>file</label><input type="file" id="parse-file" @change="parseFile = $event.target.files[0]" /></div>
        <div class="actions"><button class="btn primary" id="btn-parse" @click="doParse">parse</button></div>
        <div v-if="parseResult" class="response" id="parse-result">
          <div class="response-head"><span>parsed</span></div>
          <pre class="code-pane" id="parse-markdown">{{ parseResult.markdown || JSON.stringify(parseResult, null, 2) }}</pre>
        </div>
      </div>

      <div class="section" v-show="view === 'chunk'">
        <div class="section-head"><h3 class="section-title">chunk <span class="pill accent">POST /v1/chunk</span></h3></div>
        <div class="row split">
          <div class="row"><label>chunk size</label><input type="number" id="chunk-size" v-model.number="chunkSize" /></div>
          <div class="row"><label>overlap</label><input type="number" id="chunk-overlap" v-model.number="chunkOverlap" /></div>
        </div>
        <div class="row"><label>markdown text</label><textarea id="chunk-md" rows="6" v-model="chunkMd"></textarea></div>
        <div class="actions"><button class="btn primary" id="btn-chunk" @click="doChunk">chunk</button></div>
        <div v-if="chunkResults.length" class="response" id="chunk-results">
          <div class="response-head"><span>chunks: {{ chunkResults.length }}</span></div>
          <div v-for="(c, i) in chunkResults" :key="i" class="field-card">
            <div class="field-card-header">
              <span class="index-badge">#{{ i + 1 }}</span>
              <span class="title">{{ c.length }} chars</span>
            </div>
            <pre class="code-pane">{{ c }}</pre>
          </div>
        </div>
      </div>

      <div class="section" v-show="view === 'ingest'">
        <div class="section-head"><h3 class="section-title">ingest <span class="pill accent">POST /v1/ingest</span></h3></div>
        <div class="row split">
          <div class="row"><label>database</label>
            <select id="ingest-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>embed model</label>
            <select id="ingest-model" v-model="model"><option v-for="m in models" :key="m.id" :value="m.id">{{ m.id }}</option></select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>chunk size</label><input type="number" id="ingest-size" v-model.number="chunkParams.size" /></div>
          <div class="row"><label>overlap</label><input type="number" id="ingest-overlap" v-model.number="chunkParams.overlap" /></div>
        </div>
        <div class="row"><label>file</label><input type="file" id="ingest-file" @change="ingestFile = $event.target.files[0]" /></div>
        <div class="row"><label>metadata (JSON)</label><textarea id="ingest-metadata" rows="2" v-model="metadata">{}</textarea></div>
        <div class="actions">
          <button class="btn primary" id="btn-ingest-upload" @click="doIngest">upload</button>
          <button class="btn" id="btn-ingest-refresh" @click="refreshDbs">refresh dbs</button>
        </div>
        <div v-if="ingestResult" class="response" id="ingest-result">
          <div class="response-head"><span>ingest result</span></div>
          <pre class="code-pane">{{ JSON.stringify(ingestResult, null, 2) }}</pre>
        </div>
      </div>
      <div v-if="status" class="empty hint">status: {{ status }}</div>
    </div>
  `,
});
