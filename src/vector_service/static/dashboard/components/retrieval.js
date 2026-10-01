// Retrieval panel: dense/BM25 recall, fusion, rewrite, MMR, rerank.
// Streams NDJSON stage events from /v1/retrieval/stream and renders the
// full retrieval trace (plan, per-channel raw tops, stage timings).
import {
  defineComponent, reactive, computed, onMounted,
} from '../vue.esm-browser.prod.js';
import { t } from './app.js';
import { intError } from './util.js';

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => null);
  return { status: resp.status, body };
}

function defaultState() {
  return {
    databases: [],
    database: 'default',
    caps: { llm_configured: false },
    query: '',
    topK: 10,
    mode: 'hybrid',
    dense: true,
    bm25: true,
    fusionMethod: 'rrf',
    rrfK: 60,
    wDense: 0.5,
    wBm25: 0.5,
    wSummary: 0.5,
    wGraph: 0.5,
    routing: false,
    routeLlm: false,
    rewrite: false,
    methods: { hyde: true, multi_query: true, step_back: true, decompose: true },
    hydeAlpha: 0.7,
    nVariants: 3,
    mmr: false,
    mmrLambda: 0.7,
    rerank: true,
    candidatePool: 25,
    maxTokens: '',
    docId: '',
    filename: '',
    busy: false,
    stages: [],
    error: '',
    result: null,
    showTrace: false,
    // Raw request body for the 'custom' preset. Filled from the current
    // controls when the mode is entered, then editable verbatim.
    customJson: '',
  };
}

export default defineComponent({
  name: 'RetrievalPanel',
  setup() {
    const s = reactive(defaultState());

    async function loadDatabases() {
      const { body } = await getJson('/v1/databases');
      const names = (body && (body.databases || body.payload && body.payload.databases)) || [];
      s.databases = names;
      if (!names.includes(s.database) && names.length) s.database = names[0];
    }

    async function loadCaps() {
      const { body } = await getJson(
        '/v1/retrieval/capabilities?database=' + encodeURIComponent(s.database),
      );
      if (body && body.payload) Object.assign(s.caps, body.payload);
      else if (body) Object.assign(s.caps, body);
    }

    function setMode(mode) {
      const previous = s.mode;
      s.mode = mode;
      // 'custom' had no branch here, so clicking it only changed the
      // highlighted button: the request kept being built from the
      // controls, i.e. the mode was dead (B9). It now seeds a raw-JSON
      // body from whatever the controls currently say.
      if (mode === 'custom') {
        if (previous !== 'custom' || !s.customJson.trim()) {
          s.customJson = JSON.stringify(buildBody(), null, 2);
        }
        return;
      }
      if (mode === 'basic') {
        s.dense = true; s.bm25 = false;
        s.fusionMethod = 'rrf';
        s.rewrite = false; s.mmr = false; s.rerank = false;
      } else if (mode === 'hybrid') {
        s.dense = true; s.bm25 = true;
        s.fusionMethod = 'rrf';
        s.rewrite = false; s.mmr = false; s.rerank = true;
      } else if (mode === 'advanced') {
        s.dense = true; s.bm25 = true;
        s.fusionMethod = 'rrf';
        s.rewrite = true;
        s.methods = { hyde: true, multi_query: true, step_back: true, decompose: true };
        s.mmr = false; s.rerank = true;
      }
    }

    function buildBody() {
      const methods = Object.keys(s.methods).filter((k) => s.methods[k]);
      return {
        database: s.database,
        collection: 'ingest',
        query: s.query,
        top_k: Number(s.topK),
        filter: { doc_id: s.docId || '', filename: s.filename || '' },
        channels: { dense: s.dense, bm25: s.bm25 },
        fusion: {
          method: s.fusionMethod,
          rrf_k: Number(s.rrfK),
          weights: {
            dense: Number(s.wDense),
            bm25: Number(s.wBm25),
            summary: Number(s.wSummary),
            graph: Number(s.wGraph),
          },
        },
        rewrite: {
          enabled: s.rewrite,
          methods,
          hyde_alpha: Number(s.hydeAlpha),
          n_variants: Number(s.nVariants),
        },
        mmr: { enabled: s.mmr, lambda_mult: Number(s.mmrLambda) },
        rerank: { enabled: s.rerank, candidate_pool: Number(s.candidatePool) },
        routing: { enabled: s.routing, use_llm: s.routeLlm },
        context: {
          max_tokens: s.maxTokens === '' ? null : Number(s.maxTokens),
        },
      };
    }

    // The request body: the raw editor in 'custom' mode, otherwise the
    // one assembled from the controls. Returns null after writing the
    // parse failure to s.error.
    function requestBody() {
      if (s.mode !== 'custom') return buildBody();
      try {
        const parsed = JSON.parse(s.customJson);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          s.error = t('retrieval.err.bad_json');
          return null;
        }
        return parsed;
      } catch (e) {
        s.error = t('retrieval.err.bad_json') + String(e.message || e);
        return null;
      }
    }

    async function run() {
      s.busy = true; s.error = ''; s.result = null; s.stages = [];
      try {
        const body = requestBody();
        if (!body) return;
        const resp = await fetch('/v1/retrieval/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        // A response with no content-type at all (an error page, a proxy
        // reset) made this `.includes` throw a TypeError that was then
        // reported as the request failure (B6).
        const contentType = resp.headers.get('content-type') || '';
        if (!resp.ok || !contentType.includes('ndjson')) {
          const err = await resp.json().catch(() => null);
          const info = err && (err.error || (err.payload && err.payload.error));
          throw new Error(info ? info.message || info.code : `HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            // Stages arrive once per pass, but a retried/duplicated frame
            // would render the same chip twice (B6).
            if (event.type === 'stage') {
              if (!s.stages.includes(event.stage)) s.stages.push(event.stage);
            }
            else if (event.type === 'result') s.result = event;
            else if (event.type === 'error') {
              throw new Error(event.error.message || event.error.code);
            }
          }
        }
      } catch (e) {
        s.error = String(e.message || e);
      } finally {
        s.busy = false;
      }
    }

    onMounted(async () => {
      await loadDatabases();
      await loadCaps();
    });

    // S6 numeric rules: cleared/half-typed number boxes used to ride
    // into the body via Number() unchecked. '' is legitimate ONLY for
    // maxTokens (sends null = no limit); the integer fields must each be
    // a whole number in range before the run button goes live.
    const numErrs = computed(() => ({
      topK: intError(s.topK, 1, 100),
      rrfK: intError(s.rrfK, 1, 200),
      nVariants: intError(s.nVariants, 1, 5),
      candidatePool: intError(s.candidatePool, 1, 64),
      maxTokens: s.maxTokens === '' ? '' : intError(s.maxTokens, 1, null),
    }));
    const canRun = computed(() => s.mode === 'custom' || !Object.values(numErrs.value).some(Boolean));

    return { s, setMode, run, numErrs, canRun, regenCustom: () => { s.customJson = JSON.stringify(buildBody(), null, 2); } };
  },
  template: `
  <div class="retrieval-panel">
    <div class="section">
      <div class="section-head">
        <h3 class="section-title">{{ s.database }} / ingest</h3>
        <span class="pill accent">POST /v1/retrieval/stream</span>
      </div>

      <div class="retrieval-modes">
        <button v-for="m in ['basic','hybrid','advanced','custom']" :key="m"
          :class="['btn','sm', s.mode === m ? 'primary' : '']"
          @click="setMode(m)">{{ $t('retrieval.modes.' + m) }}</button>
      </div>

      <div class="row">
        <label>{{ $t('common.database') }}</label>
        <select v-model="s.database" @change="loadCaps">
          <option v-for="d in s.databases" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>

      <!-- 'custom' sends the textarea verbatim, so the control-driven
           body below would only contradict it. -->
      <template v-if="s.mode === 'custom'">
        <div class="row">
          <label>{{ $t('retrieval.custom_json') }}
            <span class="hint">{{ $t('retrieval.custom_hint') }}</span>
          </label>
          <textarea v-model="s.customJson" rows="14" spellcheck="false" class="code-input"
                    :aria-label="$t('retrieval.custom_json')"></textarea>
        </div>
        <div class="actions">
          <button class="btn sm" @click="regenCustom">{{ $t('retrieval.custom_regen') }}</button>
        </div>
      </template>

      <template v-else>
      <div class="row">
        <label>{{ $t('retrieval.query') }}</label>
        <textarea v-model="s.query" rows="2" :placeholder="$t('retrieval.query_placeholder')"></textarea>
      </div>

      <div class="row">
        <label>top_k</label>
        <input type="number" min="1" max="100" v-model.number="s.topK">
        <span class="hint" v-if="numErrs.topK">{{ numErrs.topK }}</span>
      </div>

      <div class="cap-group-title">{{ $t('retrieval.group.channels') }}</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.dense"> dense</label>
        <label><input type="checkbox" v-model="s.bm25"> bm25</label>
      </div>
      <div class="row">
        <label>{{ $t('retrieval.fusion') }}</label>
        <select v-model="s.fusionMethod">
          <option value="rrf">rrf</option>
          <option value="weighted">weighted</option>
        </select>
        <template v-if="s.fusionMethod === 'rrf'">
          <label>rrf_k</label>
          <input type="number" min="1" max="200" v-model.number="s.rrfK">
          <span class="hint" v-if="numErrs.rrfK">{{ numErrs.rrfK }}</span>
        </template>
        <template v-else>
          <label>w dense</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wDense">
          <label>w bm25</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wBm25">
          <label>w summary</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wSummary">
          <label>w graph</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wGraph">
        </template>
      </div>

      <div class="cap-group-title">{{ $t('retrieval.group.routing') }}</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.routing"> {{ $t('retrieval.auto_route') }}</label>
        <label><input type="checkbox" v-model="s.routeLlm"
          :disabled="!s.caps.llm_configured"> {{ $t('retrieval.use_llm') }}</label>
      </div>

      <div class="cap-group-title">{{ $t('retrieval.group.rewrite') }}
        <span v-if="!s.caps.llm_configured" class="pill danger">{{ $t('retrieval.llm_hint') }}</span>
      </div>
      <fieldset class="retrieval-fieldset" :disabled="!s.caps.llm_configured">
        <div class="row">
          <label><input type="checkbox" v-model="s.rewrite"> {{ $t('retrieval.rewrite_enabled') }}</label>
        </div>
        <div class="row">
          <label v-for="k in ['hyde','multi_query','step_back','decompose']" :key="k">
            <input type="checkbox" v-model="s.methods[k]"> {{ $t('retrieval.method.' + k) }}
          </label>
        </div>
        <div class="row">
          <label>hyde α</label>
          <input type="number" min="0" max="1" step="0.05" v-model.number="s.hydeAlpha">
          <label>{{ $t('retrieval.n_variants') }}</label>
          <input type="number" min="1" max="5" v-model.number="s.nVariants">
          <span class="hint" v-if="numErrs.nVariants">{{ numErrs.nVariants }}</span>
        </div>
      </fieldset>

      <div class="cap-group-title">{{ $t('retrieval.group.diversity') }}</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.mmr"> mmr</label>
        <label>λ</label>
        <input type="number" min="0" max="1" step="0.05" v-model.number="s.mmrLambda">
      </div>
      <div class="row">
        <label><input type="checkbox" v-model="s.rerank"> {{ $t('retrieval.cross_encoder') }}</label>
        <label>{{ $t('retrieval.candidate_pool') }}</label>
        <input type="number" min="1" max="64" v-model.number="s.candidatePool">
        <span class="hint" v-if="numErrs.candidatePool">{{ numErrs.candidatePool }}</span>
      </div>

      <div class="cap-group-title">{{ $t('retrieval.group.context') }}</div>
      <div class="row">
        <label>{{ $t('retrieval.max_tokens') }}</label>
        <input type="number" min="1" v-model.number="s.maxTokens"
          :placeholder="$t('retrieval.max_tokens_ph')">
        <span class="hint" v-if="numErrs.maxTokens">{{ numErrs.maxTokens }}</span>
      </div>
      <div class="row">
        <label>doc_id</label>
        <input v-model="s.docId" :placeholder="$t('retrieval.doc_id_ph')">
        <label>filename</label>
        <input v-model="s.filename" :placeholder="$t('retrieval.filename_ph')">
      </div>
      </template>

      <div class="actions">
        <!-- In custom mode the query lives in the JSON body, so the
             empty-query guard only applies to the control-driven modes.
             Same for the numeric-field rules (S6). -->
        <button class="btn primary" :disabled="s.busy || !canRun || (s.mode !== 'custom' && !s.query.trim())" @click="run">
          <span v-if="s.busy" class="btn-spinner"></span>
          {{ s.busy ? $t('retrieval.running') : $t('retrieval.run') }}
        </button>
      </div>

      <div v-if="s.stages.length" class="response retrieval-progress">
        <span v-for="(st, i) in s.stages" :key="i" class="retrieval-stage">
          <span class="spinner" v-if="i === s.stages.length - 1 && s.busy"></span>
          {{ stageLabel(st) }}
        </span>
      </div>

      <div v-if="s.error" class="response ingest-result-err">
        <div class="result-banner err">
          <span class="result-glyph err">!</span>
          <span class="result-banner-text">{{ s.error }}</span>
        </div>
      </div>

      <div v-if="s.result" class="retrieval-results">
        <div v-for="(c, i) in s.result.chunks" :key="c.chunk_id" class="retrieval-chunk">
          <div class="retrieval-chunk-head">
            <strong>#{{ i + 1 }}</strong>
            <span v-for="ch in c.matched_channels" :key="ch" class="pill accent">{{ ch }}</span>
            <span class="retrieval-scores">
              {{ $t('retrieval.stage.fuse') }} {{ fmtScore(c.fusion_score) }}
              <!-- Loose null check: an absent rerank_score arrives as
                   undefined, which a strict not-equals-null test let
                   through into .toFixed(). -->
              <template v-if="c.rerank_score != null">
                · {{ $t('retrieval.stage.rerank') }} {{ fmtScore(c.rerank_score) }}
              </template>
            </span>
          </div>
          <div class="retrieval-chunk-text">{{ c.fields.text }}</div>
          <div class="retrieval-chunk-meta">
            doc_id {{ c.fields.doc_id }} · {{ $t('retrieval.chunk_label') }} #{{ c.fields.chunk_index }}
            <template v-if="c.fields.section_header"> · § {{ c.fields.section_header }}</template>
            <template v-if="c.fields.page_number"> · p.{{ c.fields.page_number }}</template>
            <template v-if="c.fields.char_start !== null">
              · {{ $t('retrieval.chars') }} {{ c.fields.char_start }}-{{ c.fields.char_end }}
            </template>
            <template v-if="c.fields.filename"> · {{ c.fields.filename }}</template>
          </div>
        </div>

        <div class="retrieval-trace-toggle" @click="s.showTrace = !s.showTrace">
          {{ s.showTrace ? '▾' : '▸' }} {{ $t('retrieval.trace') }}
        </div>
        <div v-if="s.showTrace" class="retrieval-trace">
          <div class="stat" v-for="(tr, i) in s.result.traces" :key="i">
            <span class="key">{{ stageLabel(tr.stage) }}</span>
            <span class="val">{{ tr.duration_ms }} ms · {{ JSON.stringify(tr.detail) }}</span>
          </div>
          <div class="stat" v-for="(run, i) in s.result.channel_runs" :key="'r'+i">
            <span class="key">{{ run.channel }}: {{ run.query }}</span>
            <span class="val">
              {{ $t('retrieval.top') }}:
              <template v-for="h in run.hits.slice(0, 3)" :key="h.chunk_id">
                {{ h.chunk_id }}({{ fmtScore(h.score, 2) }})
              </template>
            </span>
          </div>
          <div class="stat" v-if="s.result.plan.sub_queries.length">
            <span class="key">{{ $t('retrieval.sub_queries') }}</span>
            <span class="val">{{ s.result.plan.sub_queries.join(' | ') }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  `,
  methods: {
    // Pipeline stage names come from the server (retrieval/pipeline.py);
    // an unknown stage falls back to its raw name instead of a key.
    stageLabel(stage) {
      const key = 'retrieval.stage.' + stage;
      const label = this.$t(key);
      return label === key ? stage : label;
    },
    // Fusion and rerank scores are optional on a chunk (a dense-only run
    // has no rerank score) and the trace's raw hits carry whatever the
    // channel produced. Calling .toFixed() on a missing one crashed the
    // whole result render (B6).
    fmtScore(v, digits = 4) {
      const n = Number(v);
      return (v === null || v === undefined || !Number.isFinite(n))
        ? '—' : n.toFixed(digits);
    },
  },
});
