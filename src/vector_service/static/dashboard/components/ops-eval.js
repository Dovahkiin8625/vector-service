// =====================================================================
// ops-eval.js -- Operations > Evaluation results.
//   GET  /v1/evaluation/sets?...                      set list
//   GET  /v1/evaluation/sets/{id}/questions|versions|runs
//   GET  /v1/evaluation/runs/{id}                     run detail
//   GET  /v1/evaluation/gates?... | /gates/{id} | /gates/{id}/checks
// =====================================================================
import { defineComponent, ref, computed, watch } from '../vue.esm-browser.prod.js';
import { store, api } from './app.js';
import { EmptyState } from './feedback.js';
import UiPager from './pager.js';
import {
  enc, formatTs, fixed, pct, pillClass, statusLabel, DEFAULT_SCOPE,
} from './ops-common.js';

const PAGE = 20;

export default defineComponent({
  name: 'OpsEval',
  setup() {
    const tab = ref('sets');               // 'sets' | 'gates'
    const db = ref(DEFAULT_SCOPE.db);
    const coll = ref(DEFAULT_SCOPE.coll);
    const errMsg = ref('');

    // ---- sets list / set detail ----
    const sets = ref([]);
    const setsTotal = ref(0);
    const setsPage = ref(1);
    const setsLoading = ref(false);
    const setDetail = ref(null);
    const questions = ref([]);
    const versions = ref([]);
    const runs = ref([]);
    // The three set-detail sub-lists load together; one flag keeps their
    // empty states from claiming "no data" while the fetch is in flight.
    const setLoading = ref(false);

    // ---- run detail ----
    const runDetail = ref(null);
    const runBack = ref(null);              // where to return: 'set' | 'gate'
    const runLoading = ref(false);

    // ---- gates ----
    const gates = ref([]);
    const gatesPage = ref(1);
    const gatesLoading = ref(false);
    const gateDetail = ref(null);
    const gateChecks = ref([]);
    const checksLoading = ref(false);

    const setsPages = computed(() => Math.max(1, Math.ceil(setsTotal.value / PAGE)));
    const setsOffset = computed(() => (setsPage.value - 1) * PAGE);
    // Gates come back unpaginated (scope-unique per database+collection,
    // so the list is usually one row); slice here so a long list still
    // pages exactly like the sets list above.
    const gatesPages = computed(() => Math.max(1, Math.ceil(gates.value.length / PAGE)));
    const gatesOffset = computed(() => (gatesPage.value - 1) * PAGE);
    const pagedGates = computed(() =>
      gates.value.slice(gatesOffset.value, gatesOffset.value + PAGE));

    function resetErr() { errMsg.value = ''; }
    async function loadSets() {
      resetErr();
      setsLoading.value = true;
      try {
        const q = `/v1/evaluation/sets?database=${enc(db.value)}`
          + `&collection=${enc(coll.value)}&limit=${PAGE}`
          + `&offset=${setsOffset.value}`;
        const { payload } = await api('GET', q);
        sets.value = (payload && payload.items) || [];
        setsTotal.value = (payload && payload.total) || 0;
      } catch (e) { errMsg.value = e.message; }
      finally { setsLoading.value = false; }
    }
    function setsGo(n) {
      const target = Math.min(Math.max(1, Number(n) || 1), setsPages.value);
      if (target === setsPage.value) return;
      setsPage.value = target;
      loadSets();
    }

    async function openSet(s) {
      resetErr();
      setDetail.value = s;
      runDetail.value = null;
      setLoading.value = true;
      try {
        const id = enc(s.set_id);
        const [q, v, r] = await Promise.all([
          api('GET', `/v1/evaluation/sets/${id}/questions`),
          api('GET', `/v1/evaluation/sets/${id}/versions`),
          api('GET', `/v1/evaluation/sets/${id}/runs`),
        ]);
        questions.value = (q.payload && q.payload.items) || q.payload || [];
        versions.value = (v.payload && v.payload.items) || v.payload || [];
        runs.value = (r.payload && r.payload.items) || r.payload || [];
      } catch (e) { errMsg.value = e.message; }
      finally { setLoading.value = false; }
    }
    function backToList() {
      setDetail.value = null;
      runDetail.value = null;
    }

    async function openRun(runId, backTo) {
      resetErr();
      runBack.value = backTo || 'set';
      runLoading.value = true;
      try {
        const { payload } = await api('GET', `/v1/evaluation/runs/${enc(runId)}`);
        runDetail.value = payload;
      } catch (e) { errMsg.value = e.message; }
      finally { runLoading.value = false; }
    }
    // Returns to the view the run was opened from. runBack was written
    // on every openRun but never read, so the back button just dropped
    // the detail and revealed whatever tab happened to be active.
    function closeRun() {
      runDetail.value = null;
      tab.value = runBack.value === 'gate' ? 'gates' : 'sets';
    }

    async function loadGates() {
      resetErr();
      gatesLoading.value = true;
      try {
        const q = `/v1/evaluation/gates?database=${enc(db.value)}&collection=${enc(coll.value)}`;
        const { payload } = await api('GET', q);
        gates.value = (payload && payload.items) || [];
        gatesPage.value = 1;
      } catch (e) { errMsg.value = e.message; }
      finally { gatesLoading.value = false; }
    }
    function gatesGo(n) {
      const target = Math.min(Math.max(1, Number(n) || 1), gatesPages.value);
      if (target === gatesPage.value) return;
      gatesPage.value = target;
    }

    async function openGate(g) {
      resetErr();
      gateDetail.value = g;
      runDetail.value = null;
      checksLoading.value = true;
      try {
        const { payload } = await api(
          'GET',
          `/v1/evaluation/gates/${enc(g.gate_id)}/checks`,
        );
        gateChecks.value = (payload && payload.items) || [];
      } catch (e) { errMsg.value = e.message; }
      finally { checksLoading.value = false; }
    }
    function backToGates() {
      gateDetail.value = null;
      runDetail.value = null;
    }

    function switchTab(name) {
      tab.value = name;
      // The run detail renders from a v-if that outranks the tab
      // branches, so switching tabs with one open left the run on screen
      // and the click looked like it did nothing (B13).
      runDetail.value = null;
      resetErr();
      if (name === 'sets') loadSets();
      else loadGates();
    }

    watch(() => store.view, (v) => {
      if (v === 'eval') {
        if (tab.value === 'sets') loadSets();
        else loadGates();
      }
    });
    if (store.view === 'eval') loadSets();

    // KPI grid and the two breakdown tables read runDetail.summary.*; a
    // run payload without (or with a partial) summary block threw during
    // render and took the whole panel down. fixed()/pct() already render
    // a missing value as '—'.
    const summary = computed(() => (runDetail.value && runDetail.value.summary) || {});

    return {
      tab, db, coll, errMsg,
      sets, setsTotal, setsPage, setsPages, setsOffset, setsLoading, setsGo,
      setDetail, questions, versions, runs, setLoading,
      loadSets, openSet, backToList,
      runDetail, runBack, runLoading, openRun, closeRun, summary,
      gates, gatesPage, gatesPages, gatesOffset, pagedGates, gatesLoading, gatesGo,
      gateDetail, gateChecks, checksLoading,
      loadGates, openGate, backToGates,
      switchTab,
      formatTs, fixed, pct, pillClass, statusLabel,
    };
  },
  components: { EmptyState, UiPager },
  template: `
    <div class="ops-panel">
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('nav.eval') }}
            <span class="pill accent">/v1/evaluation</span>
          </h3>
        </div>

        <div class="ops-scope">
          <label>{{ $t('common.database') }}
            <input v-model="db" type="text" spellcheck="false" />
          </label>
          <label>{{ $t('common.collection') }}
            <input v-model="coll" type="text" spellcheck="false" />
          </label>
          <button class="btn sm ghost" @click="tab === 'sets' ? loadSets() : loadGates()">
            {{ $t('common.apply') }}
          </button>
        </div>

        <div class="seg-toggle ops-tabs" role="tablist" :aria-label="$t('nav.eval')">
          <button type="button" class="btn sm" role="tab"
                  :aria-selected="tab === 'sets' ? 'true' : 'false'"
                  :class="{ on: tab === 'sets' }"
                  @click="switchTab('sets')">
            {{ $t('ops.eval.tab_sets') }}
          </button>
          <button type="button" class="btn sm" role="tab"
                  :aria-selected="tab === 'gates' ? 'true' : 'false'"
                  :class="{ on: tab === 'gates' }"
                  @click="switchTab('gates')">
            {{ $t('ops.eval.tab_gates') }}
          </button>
        </div>

        <div v-if="errMsg" class="empty error">{{ errMsg }}</div>
      </div>

      <!-- ============ run detail (shared) ============ -->
      <!-- The fetch used to show nothing between click and payload, so
           the row looked dead; now it announces itself. -->
      <div v-if="runLoading" class="section">
        <empty-state state="loading" :text="$t('common.state.loading')" />
      </div>
      <div v-else-if="runDetail" class="section">
        <div class="section-head">
          <h3 class="section-title">
            {{ $t('ops.eval.run') }}
            <span class="pill accent">GET /v1/evaluation/runs/{id}</span>
          </h3>
          <span class="section-sub">
            <button class="btn sm ghost" @click="closeRun">
              ← {{ $t('common.back') }}
            </button>
          </span>
        </div>

        <table class="info-table ops-run-meta">
          <tr><th>{{ $t('common.run_id') }}</th><td class="ops-mono">{{ runDetail.run_id }}</td></tr>
          <tr><th>{{ $t('common.set_id') }}</th><td class="ops-mono">{{ runDetail.set_id }}</td></tr>
          <tr><th>{{ $t('common.version_id') }}</th><td class="ops-mono">{{ runDetail.version_id || '—' }}</td></tr>
          <tr><th>{{ $t('common.template') }}</th><td>{{ runDetail.template }}</td></tr>
          <tr><th>{{ $t('common.include_answer') }}</th><td>{{ runDetail.include_answer ? '✓' : '—' }}</td></tr>
          <tr><th>{{ $t('ops.queue.created') }}</th><td>{{ formatTs(runDetail.created_ts) }}</td></tr>
        </table>

        <div class="ops-grid ops-kpi-grid">
          <div class="kpi kpi--accent">
            <span class="label">{{ $t('ops.eval.questions_kpi') }}</span>
            <span class="value">{{ summary.questions || '—' }}</span>
          </div>
          <div class="kpi">
            <span class="label">{{ $t('ops.eval.mean_recall') }}</span>
            <span class="value">{{ fixed(summary.mean_recall) }}</span>
          </div>
          <div class="kpi">
            <span class="label">{{ $t('ops.eval.mean_mrr') }}</span>
            <span class="value">{{ fixed(summary.mean_mrr) }}</span>
          </div>
          <div class="kpi">
            <span class="label">{{ $t('ops.eval.mean_ndcg') }}</span>
            <span class="value">{{ fixed(summary.mean_ndcg) }}</span>
          </div>
          <div class="kpi">
            <span class="label">{{ $t('ops.eval.doc_hit_rate') }}</span>
            <span class="value">{{ pct(summary.doc_hit_rate) }}</span>
          </div>
        </div>

        <div v-if="summary.rerank && summary.rerank.questions" class="ops-rerank-card">
          <h4 class="ops-sub-title">{{ $t('ops.eval.rerank_compare') }} · {{ summary.rerank.questions }} q</h4>
          <div class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr><th></th><th>{{ $t('ops.eval.pre') }}</th><th>{{ $t('ops.eval.post') }}</th></tr>
              </thead>
              <tbody>
                <tr><th>recall</th><td>{{ fixed(summary.rerank.mean_recall_pre) }}</td><td>{{ fixed(summary.rerank.mean_recall_post) }}</td></tr>
                <tr><th>MRR</th><td>{{ fixed(summary.rerank.mean_mrr_pre) }}</td><td>{{ fixed(summary.rerank.mean_mrr_post) }}</td></tr>
                <tr><th>nDCG</th><td>{{ fixed(summary.rerank.mean_ndcg_pre) }}</td><td>{{ fixed(summary.rerank.mean_ndcg_post) }}</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div v-if="summary.channel_attribution && summary.channel_attribution.questions" class="section">
          <h4 class="ops-sub-title">{{ $t('ops.eval.channel_attribution') }} · {{ summary.channel_attribution.questions }} q</h4>
          <div class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr><th>{{ $t('ops.eval.channel') }}</th><th>{{ $t('ops.eval.hit_share') }}</th><th>{{ $t('ops.eval.raw_recall') }}</th></tr>
              </thead>
              <tbody>
                <tr v-for="(c, name) in summary.channel_attribution.channels" :key="name">
                  <td class="ops-mono">{{ name }}</td>
                  <td>{{ pct(c.hit_share) }}</td>
                  <td>{{ fixed(c.raw_recall) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <h4 class="ops-sub-title">{{ $t('ops.eval.per_question') }}</h4>
        <empty-state v-if="!(runDetail.results || []).length" state="empty"
                     :text="$t('common.state.empty')" />
        <div v-else class="data-table-wrap">
          <table class="data-table ops-results-table">
            <thead>
              <tr>
                <th>#</th>
                <th>{{ $t('ops.eval.question') }}</th>
                <th>recall</th>
                <th>MRR</th>
                <th>nDCG</th>
                <th>{{ $t('ops.eval.doc_label') }}</th>
                <th>{{ $t('ops.eval.chunks') }}</th>
                <th v-if="runDetail.include_answer">{{ $t('ops.eval.answer') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, i) in runDetail.results" :key="r.question_id">
                <td>{{ i + 1 }}</td>
                <td class="ops-q-cell">{{ r.question }}</td>
                <td>{{ fixed(r.metrics && r.metrics.recall) }}</td>
                <td>{{ fixed(r.metrics && r.metrics.mrr) }}</td>
                <td>{{ fixed(r.metrics && r.metrics.ndcg) }}</td>
                <td>{{ fixed(r.metrics && r.metrics.doc_hit, 2) }}</td>
                <td>
                  <span v-if="r.chunk_ids && r.chunk_ids.length">{{ r.chunk_ids.length }}</span>
                  <details v-if="r.chunk_ids && r.chunk_ids.length" class="ops-id-details">
                    <summary>{{ $t('ops.consistency.show_ids') }}</summary>
                    <div class="ops-id-list ops-mono">
                      <div v-for="c in r.chunk_ids" :key="c">{{ c }}</div>
                    </div>
                  </details>
                  <span v-else>0</span>
                </td>
                <td v-if="runDetail.include_answer">
                  <details v-if="r.answer" class="ops-answer-details">
                    <summary>{{ $t('ops.eval.show_answer') }}</summary>
                    <div class="ops-answer">{{ r.answer }}</div>
                  </details>
                  <span v-else-if="r.answer_error" class="ops-err-code ops-mono">{{ r.answer_error }}</span>
                  <span v-else>—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ============ SETS ============ -->
      <template v-else-if="tab === 'sets'">
        <!-- set detail -->
        <div v-if="setDetail" class="section">
          <div class="section-head">
            <h3 class="section-title">
              {{ setDetail.name }}
              <span class="pill accent ops-mono">{{ setDetail.set_id }}</span>
            </h3>
            <span class="section-sub">
              <button class="btn sm ghost" @click="backToList">← {{ $t('ops.eval.all_sets') }}</button>
            </span>
          </div>
          <p class="ops-set-desc">{{ setDetail.description || '—' }}</p>

          <!-- The three sub-lists load in one batch; one loading block
               covers them so none can claim "empty" mid-flight. -->
          <empty-state v-if="setLoading" state="loading" :text="$t('common.state.loading')" />
          <template v-else>
          <h4 class="ops-sub-title">{{ $t('ops.eval.questions') }} · {{ questions.length }}</h4>
          <empty-state v-if="!questions.length" state="empty" :text="$t('common.state.empty')" />
          <div v-else class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr><th>{{ $t('common.question_id') }}</th><th>{{ $t('ops.eval.question') }}</th><th>{{ $t('ops.eval.expected') }}</th></tr>
              </thead>
              <tbody>
                <tr v-for="q in questions" :key="q.question_id">
                  <td class="ops-mono">{{ q.question_id }}</td>
                  <td class="ops-q-cell">{{ q.question }}</td>
                  <td>
                    <span v-if="q.expected_chunk_ids && q.expected_chunk_ids.length">
                      {{ q.expected_chunk_ids.length }} {{ $t('common.chunks_unit') }}
                    </span>
                    <span v-if="q.expected_doc_ids && q.expected_doc_ids.length">
                      · {{ q.expected_doc_ids.length }} {{ $t('common.docs_unit') }}
                    </span>
                    <span v-if="q.expected_answer"> · {{ $t('ops.eval.has_answer') }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <h4 class="ops-sub-title">{{ $t('ops.eval.versions') }} · {{ versions.length }}</h4>
          <empty-state v-if="!versions.length" state="empty" :text="$t('common.state.empty')" />
          <div v-else class="data-table-wrap">
            <table class="data-table">
              <thead><tr><th>{{ $t('common.version_id') }}</th><th>{{ $t('common.tag') }}</th><th>{{ $t('common.question_count') }}</th><th>{{ $t('ops.queue.created') }}</th></tr></thead>
              <tbody>
                <tr v-for="v in versions" :key="v.version_id">
                  <td class="ops-mono">{{ v.version_id }}</td>
                  <td>{{ v.tag || '—' }}</td>
                  <td>{{ v.question_count }}</td>
                  <td>{{ formatTs(v.created_ts) }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h4 class="ops-sub-title">{{ $t('ops.eval.runs') }} · {{ runs.length }}</h4>
          <empty-state v-if="!runs.length" state="empty" :text="$t('common.state.empty')" />
          <div v-else class="data-table-wrap">
            <table class="data-table">
              <thead><tr><th>{{ $t('common.run_id') }}</th><th>{{ $t('common.params') }}</th><th>{{ $t('ops.queue.created') }}</th></tr></thead>
              <tbody>
                <tr v-for="r in runs" :key="r.run_id" class="ops-job-row" role="button" tabindex="0"
                    @click="openRun(r.run_id, 'set')"
                    @keydown.enter.prevent="openRun(r.run_id, 'set')"
                    @keydown.space.prevent="openRun(r.run_id, 'set')">
                  <td class="ops-mono">{{ r.run_id }}</td>
                  <td class="ops-mono ops-params-cell">{{ r.params_json }}</td>
                  <td>{{ formatTs(r.created_ts) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          </template>
        </div>

        <!-- set list -->
        <div v-else class="section">
          <div v-if="setsLoading && !sets.length" class="spinner"></div>
          <empty-state v-else-if="!setsLoading && !errMsg && !sets.length"
                       state="empty" :text="$t('common.state.empty')" />
          <div v-else-if="sets.length" class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>{{ $t('ops.eval.set_name') }}</th>
                  <th>{{ $t('common.set_id') }}</th>
                  <th>{{ $t('common.scope') }}</th>
                  <th>{{ $t('common.question_count') }}</th>
                  <th>{{ $t('ops.queue.created') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="s in sets" :key="s.set_id" class="ops-job-row" role="button" tabindex="0"
                    @click="openSet(s)"
                    @keydown.enter.prevent="openSet(s)"
                    @keydown.space.prevent="openSet(s)">
                  <td>{{ s.name }}</td>
                  <td class="ops-mono">{{ s.set_id }}</td>
                  <td class="ops-mono">{{ s.database }}/{{ s.collection }}</td>
                  <td>{{ s.question_count }}</td>
                  <td>{{ formatTs(s.created_ts) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <ui-pager v-if="setsTotal" root-id="eval-sets-pager" info-id="eval-sets-pager-info"
                    first-id="btn-eval-sets-first" prev-id="btn-eval-sets-prev"
                    next-id="btn-eval-sets-next" last-id="btn-eval-sets-last"
                    jump-id="eval-sets-jump" jump-btn-id="btn-eval-sets-jump"
                    :page="setsPage" :pages="setsPages"
                    :busy="setsLoading" :show-jump="true"
                    :info-text="$t('ops.pager_info', { from: setsOffset + 1, to: setsOffset + sets.length, total: setsTotal, page: setsPage, pages: setsPages })"
                    @first="setsGo(1)" @prev="setsGo(setsPage - 1)"
                    @next="setsGo(setsPage + 1)" @last="setsGo(setsPages)"
                    @go="setsGo" />
        </div>
      </template>

      <!-- ============ GATES ============ -->
      <template v-else>
        <!-- gate detail -->
        <div v-if="gateDetail" class="section">
          <div class="section-head">
            <h3 class="section-title">
              <span class="ops-mono">{{ gateDetail.gate_id }}</span>
              <span class="pill accent">{{ $t('ops.eval.gate') }}</span>
            </h3>
            <span class="section-sub">
              <button class="btn sm ghost" @click="backToGates">← {{ $t('ops.eval.all_gates') }}</button>
            </span>
          </div>

          <table class="info-table">
            <tr><th>{{ $t('common.scope') }}</th><td class="ops-mono">{{ gateDetail.database }}/{{ gateDetail.collection }}</td></tr>
            <tr><th>{{ $t('common.set_id') }}</th><td class="ops-mono">{{ gateDetail.set_id }}</td></tr>
            <tr><th>{{ $t('common.version_id') }}</th><td class="ops-mono">{{ gateDetail.version_id || '—' }}</td></tr>
            <tr><th>{{ $t('common.template') }}</th><td>{{ gateDetail.template }}</td></tr>
            <tr><th>{{ $t('common.baseline_run_id') }}</th><td class="ops-mono">{{ gateDetail.baseline_run_id || '—' }}</td></tr>
          </table>

          <h4 class="ops-sub-title">{{ $t('ops.eval.thresholds') }}</h4>
          <div class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr><th>{{ $t('common.metric') }}</th><th>{{ $t('ops.eval.min') }}</th><th>{{ $t('ops.eval.max_drop') }}</th></tr>
              </thead>
              <tbody>
                <tr><th>recall</th><td>{{ fixed(gateDetail.min_recall) }}</td><td>{{ pct(gateDetail.max_recall_drop) }}</td></tr>
                <tr><th>MRR</th><td>{{ fixed(gateDetail.min_mrr) }}</td><td>{{ pct(gateDetail.max_mrr_drop) }}</td></tr>
                <tr><th>nDCG</th><td>{{ fixed(gateDetail.min_ndcg) }}</td><td>{{ pct(gateDetail.max_ndcg_drop) }}</td></tr>
              </tbody>
            </table>
          </div>

          <h4 class="ops-sub-title">{{ $t('ops.eval.checks') }} · {{ gateChecks.length }}</h4>
          <empty-state v-if="checksLoading" state="loading" :text="$t('common.state.loading')" />
          <empty-state v-else-if="!gateChecks.length" state="empty" :text="$t('common.state.empty')" />
          <div v-else class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr><th>{{ $t('common.check_id') }}</th><th>{{ $t('ops.queue.status') }}</th><th>{{ $t('common.candidate_ref') }}</th><th>{{ $t('common.run_id') }}</th><th>{{ $t('ops.queue.created') }}</th></tr>
              </thead>
              <tbody>
                <!-- A check with no run_id opens nothing — it gets a
                     plain row instead of a button affordance (§2 item
                     21). -->
                <tr v-for="c in gateChecks" :key="c.check_id"
                    :class="c.run_id ? 'ops-job-row' : 'ops-row-static'"
                    :role="c.run_id ? 'button' : null"
                    :tabindex="c.run_id ? 0 : null"
                    @click="c.run_id ? openRun(c.run_id, 'gate') : null"
                    @keydown.enter.prevent="c.run_id ? openRun(c.run_id, 'gate') : null"
                    @keydown.space.prevent="c.run_id ? openRun(c.run_id, 'gate') : null">
                  <td class="ops-mono">{{ c.check_id }}</td>
                  <td><span :class="['pill', pillClass(c.status)]">{{ statusLabel(c.status) }}</span></td>
                  <td class="ops-mono">{{ c.candidate_ref }}</td>
                  <td class="ops-mono">{{ c.run_id || '—' }}</td>
                  <td>{{ formatTs(c.created_ts) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- gate list -->
        <div v-else class="section">
          <div v-if="gatesLoading && !gates.length" class="spinner"></div>
          <empty-state v-else-if="!gatesLoading && !errMsg && !gates.length"
                       state="empty" :text="$t('common.state.empty')" />
          <div v-else-if="gates.length" class="data-table-wrap">
            <table class="data-table">
              <thead>
                <tr><th>{{ $t('common.gate_id') }}</th><th>{{ $t('common.scope') }}</th><th>{{ $t('common.set_id') }}</th><th>{{ $t('common.baseline') }}</th><th>{{ $t('ops.queue.created') }}</th></tr>
              </thead>
              <tbody>
                <tr v-for="g in pagedGates" :key="g.gate_id" class="ops-job-row" role="button" tabindex="0"
                    @click="openGate(g)"
                    @keydown.enter.prevent="openGate(g)"
                    @keydown.space.prevent="openGate(g)">
                  <td class="ops-mono">{{ g.gate_id }}</td>
                  <td class="ops-mono">{{ g.database }}/{{ g.collection }}</td>
                  <td class="ops-mono">{{ g.set_id }}</td>
                  <td class="ops-mono">{{ g.baseline_run_id || '—' }}</td>
                  <td>{{ formatTs(g.created_ts) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <!-- The list comes back whole from the API; page it here so a
               long list uses the same pager shape as the sets tab. -->
          <ui-pager v-if="gates.length" root-id="eval-gates-pager" info-id="eval-gates-pager-info"
                    first-id="btn-eval-gates-first" prev-id="btn-eval-gates-prev"
                    next-id="btn-eval-gates-next" last-id="btn-eval-gates-last"
                    jump-id="eval-gates-jump" jump-btn-id="btn-eval-gates-jump"
                    :page="gatesPage" :pages="gatesPages"
                    :busy="gatesLoading" :show-jump="true"
                    :info-text="$t('ops.pager_info', { from: gatesOffset + 1, to: gatesOffset + pagedGates.length, total: gates.length, page: gatesPage, pages: gatesPages })"
                    @first="gatesGo(1)" @prev="gatesGo(gatesPage - 1)"
                    @next="gatesGo(gatesPage + 1)" @last="gatesGo(gatesPages)"
                    @go="gatesGo" />
        </div>
      </template>
    </div>
  `,
});
