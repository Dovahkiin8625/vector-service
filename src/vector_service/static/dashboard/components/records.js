// Records panel: upsert / fetch by id / delete by id or filter.
// Stage 3 IA: the three verbs used to share one form (and one button
// row), so "write", "query" and "delete" read as one action. They are
// now tabs over a shared scope (database/collection/field names); each
// pane owns its own inputs and its own submit button.
import { defineComponent, ref, computed, watch, onMounted } from '../vue.esm-browser.prod.js';
import { t, api, enc, extractApiError } from './app.js';
import { StatusBanner, BusyButton, EmptyState, askConfirm } from './feedback.js';

export default defineComponent({
  name: 'RecordsPanel',
  setup() {
    const dbs = ref([]);
    const colls = ref([]);
    const db = ref('');
    const coll = ref('');
    const primary = ref('id');
    const vecfield = ref('vector');
    // Field-name suggestions from the collection schema: the primary/
    // vector field boxes stay free text (a schema-less backend or a
    // collection still loading must not block typing) but datalist
    // dropdowns now offer the real names instead of a guess (stage 3).
    const schemaFields = ref([]);
    const tab = ref('write');
    const mode = ref('texts');
    // No pre-filled example values. They used to ship as valid-looking
    // JSON (``["sku-1", "sku-2"]`` plus a row-aligned fields array), so
    // the write button pushed sample rows into whatever collection was
    // selected and the delete button removed sku-1/sku-2 from it — one
    // click, nothing typed. Format hints live in the placeholders now,
    // which are never submitted.
    const ids = ref('');
    const texts = ref('');
    const embs = ref('');
    const fields = ref('');
    const delMode = ref('ids');
    const delIds = ref('');
    const delFilter = ref('');
    // Which of the three operations is in flight ('' | upsert | fetch |
    // delete). One shared flag would light up all three buttons at once.
    const busy = ref('');
    // Outcome of the last operation. `status` drives the banner below the
    // buttons; it used to be a bare string rendered through runStateText,
    // which reported nothing but the word "OK".
    const status = ref('idle');
    const errMsg = ref('');
    const okMsg = ref('');
    // Validation complaints and selection/schema load failures each get
    // their own channel, so a malformed JSON body is never shown as
    // "the request failed" (and vice versa).
    const formErr = ref('');
    const loadErr = ref('');
    // Result of POST .../vectors/get. This response was previously
    // discarded outright — the panel had no ref to hold it, so a
    // successful fetch printed "OK" and nothing else (B2).
    const fetched = ref(null);        // GetVectorItem[] | null (never run)
    const fetchRequested = ref(0);
    const fetchMissing = ref([]);
    // Table vs raw-JSON rendering of the fetch result (stage 3).
    const fetchView = ref('table');

    function ok(msg) { status.value = 'ok'; errMsg.value = ''; okMsg.value = msg; }
    function failed(e) {
      status.value = 'error';
      okMsg.value = '';
      errMsg.value = extractApiError(e, t('common.unknown'));
    }

    async function refreshDbs() {
      loadErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/databases');
        dbs.value = (payload && payload.databases) || [];
        if (!db.value && dbs.value.length) db.value = dbs.value[0];
      } catch (e) {
        dbs.value = [];
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    async function refreshColls() {
      if (!db.value) { colls.value = []; return; }
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections');
        colls.value = (payload && payload.collections) || [];
        if (!colls.value.includes(coll.value)) coll.value = colls.value[0] || '';
      } catch (e) {
        colls.value = [];
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    // Schema names for the field-name datalists. Best-effort: a failed
    // lookup only costs the suggestions, it never blocks the form.
    async function refreshSchema() {
      schemaFields.value = [];
      if (!db.value || !coll.value) return;
      try {
        const { payload } = await api('GET',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value));
        schemaFields.value = ((payload && payload.fields) || []).map(f => f.name);
      } catch (_e) { /* hints are optional */ }
    }
    watch(db, refreshColls);
    watch(coll, refreshSchema);
    onMounted(refreshDbs);

    function safeParse(s) { try { return JSON.parse(s); } catch (_e) { return null; } }

    // Returns the body, or null after writing the reason to formErr.
    function buildUpsertBody() {
      const idsArr = safeParse(ids.value);
      if (!Array.isArray(idsArr)) { formErr.value = t('records.err.ids_json'); return null; }
      const body = { primary_field: primary.value.trim(), vector_field: vecfield.value.trim(), ids: idsArr };
      // texts/vectors/fields are row-aligned with ids; a ragged body
      // used to reach the server and fail (or worse, be accepted with
      // rows silently paired in order). Checked here so the complaint
      // names both counts.
      const dataArr = mode.value === 'texts' ? safeParse(texts.value) : safeParse(embs.value);
      if (!Array.isArray(dataArr)) {
        formErr.value = t(mode.value === 'texts' ? 'records.err.texts_json' : 'records.err.vectors_json');
        return null;
      }
      if (dataArr.length !== idsArr.length) {
        formErr.value = t('records.err.len_mismatch', { ids: idsArr.length, n: dataArr.length });
        return null;
      }
      if (mode.value === 'texts') body.texts = dataArr;
      else body.vectors = dataArr;
      const fRaw = fields.value.trim();
      if (fRaw && fRaw !== '[]') {
        const f = safeParse(fRaw);
        if (!Array.isArray(f)) { formErr.value = t('records.err.fields_json'); return null; }
        if (f.length) {
          if (f.length !== idsArr.length) {
            formErr.value = t('records.err.len_mismatch_fields', { ids: idsArr.length, n: f.length });
            return null;
          }
          body.fields = f;
        }
      }
      return body;
    }

    // Every operation starts here: clears the previous outcome, checks
    // the selection, and guards against a second submit while one is
    // already in flight.
    function begin(op) {
      if (busy.value) return false;
      formErr.value = '';
      if (!db.value || !coll.value) { formErr.value = t('records.err.no_selection'); return false; }
      busy.value = op;
      status.value = 'loading';
      errMsg.value = '';
      okMsg.value = '';
      return true;
    }

    async function doUpsert() {
      const body = buildUpsertBody();
      if (!body) return;
      if (!begin('upsert')) return;
      try {
        const { payload } = await api('PUT',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors', body);
        ok(t('records.upsert_ok', { n: (payload && payload.upserted) || 0 }));
      } catch (e) { failed(e); } finally { busy.value = ''; }
    }

    async function doFetch() {
      const idsArr = safeParse(ids.value);
      if (!Array.isArray(idsArr)) { formErr.value = t('records.err.ids_json'); return; }
      if (!begin('fetch')) return;
      try {
        const { payload } = await api('POST',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors/get',
          { primary_field: primary.value.trim(), ids: idsArr });
        const items = (payload && payload.items) || [];
        const found = new Set(items.map(it => it.id));
        fetched.value = items;
        fetchRequested.value = idsArr.length;
        fetchMissing.value = idsArr.filter(id => !found.has(id));
        ok(t('records.fetched_n', { n: items.length }));
      } catch (e) {
        fetched.value = null;
        failed(e);
      } finally { busy.value = ''; }
    }

    async function doDelete() {
      // Build and validate the body BEFORE the confirmation modal: the
      // blast radius shown in it is exactly what will be sent.
      const body = { primary_field: primary.value.trim() };
      let scope;
      if (delMode.value === 'ids') {
        let arr = safeParse(delIds.value);
        if (!Array.isArray(arr)) {
          arr = delIds.value.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
        }
        if (!arr.length) { formErr.value = t('records.err.no_ids'); return; }
        body.ids = arr;
        scope = { label: t('records.del_ids'), value: String(arr.length) };
      } else {
        if (!delFilter.value.trim()) { formErr.value = t('records.err.no_filter'); return; }
        body.filter_expr = delFilter.value.trim();
        scope = { label: t('records.del_filter'), value: body.filter_expr };
      }
      if (!db.value || !coll.value) { formErr.value = t('records.err.no_selection'); return; }
      const confirmed = await askConfirm({
        title: t('records.confirm_delete_title'),
        message: t('records.confirm_delete'),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: coll.value },
          scope,
        ],
        confirmLabel: t('records.delete'),
      });
      if (!confirmed) return;
      if (!begin('delete')) return;
      try {
        const { payload } = await api('POST',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors/delete', body);
        ok(t('records.delete_ok', { n: (payload && payload.deleted) || 0 }));
      } catch (e) { failed(e); } finally { busy.value = ''; }
    }

    // The delete button is the only control on this panel that destroys
    // data, so it stays disabled until the request it would send is
    // actually well-formed. It used to be enabled from the first render
    // with sku-1/sku-2 already in the box, which made "open the panel,
    // click the red button" a working delete of two real rows.
    // `deleteBlockReason` is the params half of that condition and is
    // rendered next to the params it refers to, so a greyed-out button
    // says what it is waiting for instead of just looking broken.
    const deleteBlockReason = computed(() => {
      if (delMode.value === 'ids') {
        return delIds.value.trim() ? '' : t('records.err.no_ids');
      }
      return delFilter.value.trim() ? '' : t('records.err.no_filter');
    });
    // The db/collection selection is a panel-wide precondition — all
    // three buttons need it — so it is folded in without a reason string
    // of its own: the empty selects above are the explanation.
    const canDelete = computed(
      () => !deleteBlockReason.value && !!db.value && !!coll.value);

    // The three buttons below are their own retry affordance, so the
    // banner carries no retry button.
    const outcomeKind = computed(() => (status.value === 'error' ? 'error' : 'success'));
    const outcomeText = computed(() => {
      if (status.value === 'error') return errMsg.value;
      return status.value === 'ok' ? okMsg.value : '';
    });

    return { dbs, colls, db, coll, primary, vecfield, schemaFields, tab, mode,
             ids, texts, embs, fields,
             delMode, delIds, delFilter, busy, status, formErr, loadErr,
             fetched, fetchRequested, fetchMissing, fetchView,
             outcomeKind, outcomeText,
             canDelete, deleteBlockReason,
             refreshDbs, doUpsert, doFetch, doDelete };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('records.title') }} <span class="pill accent">PUT /v1/databases/{db}/collections/{coll}/vectors</span></h3>
        </div>
        <!-- Shared scope: every verb below works on this db/collection
             and these field names. -->
        <div class="row split">
          <div class="row"><label>{{ $t('common.database') }}</label>
            <select id="vec-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>{{ $t('common.collection') }}</label>
            <select id="vec-coll" v-model="coll"><option v-for="c in colls" :key="c" :value="c">{{ c }}</option></select>
          </div>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.primary_field') }}</label>
            <input type="text" id="vec-primary" v-model="primary" list="vec-primary-opts" />
            <datalist id="vec-primary-opts">
              <option v-for="f in schemaFields" :key="f" :value="f"></option>
            </datalist>
          </div>
          <div class="row"><label>{{ $t('common.vector_field') }}</label>
            <input type="text" id="vec-vecfield" v-model="vecfield" list="vec-vecfield-opts" />
            <datalist id="vec-vecfield-opts">
              <option v-for="f in schemaFields" :key="f" :value="f"></option>
            </datalist>
          </div>
        </div>

        <!-- Three verbs, three panes. The ids box is shared by write and
             query (both target the same primary keys) and is rendered in
             both panes against one model. -->
        <div class="records-tabs" role="tablist">
          <button type="button" role="tab" id="tab-vec-write"
                  :aria-selected="tab === 'write' ? 'true' : 'false'"
                  :class="['btn', 'sm', tab === 'write' ? 'primary' : '']"
                  @click="tab = 'write'">{{ $t('records.tab.write') }}</button>
          <button type="button" role="tab" id="tab-vec-query"
                  :aria-selected="tab === 'query' ? 'true' : 'false'"
                  :class="['btn', 'sm', tab === 'query' ? 'primary' : '']"
                  @click="tab = 'query'">{{ $t('records.tab.query') }}</button>
          <button type="button" role="tab" id="tab-vec-delete"
                  :aria-selected="tab === 'delete' ? 'true' : 'false'"
                  :class="['btn', 'sm', tab === 'delete' ? 'primary' : '']"
                  @click="tab = 'delete'">{{ $t('records.tab.delete') }}</button>
        </div>

        <div v-show="tab === 'write'" role="tabpanel">
          <div class="row"><label>{{ $t('records.input_mode') }}</label>
            <select id="vec-mode" v-model="mode">
              <option value="texts">{{ $t('records.mode.texts') }}</option>
              <option value="vectors">{{ $t('records.mode.vectors') }}</option>
            </select>
          </div>
          <div class="row"><label>{{ $t('records.ids') }}</label><textarea id="vec-ids" class="code-input" rows="2" v-model="ids" :placeholder="$t('records.ph.ids')"></textarea></div>
          <div class="row" v-show="mode === 'texts'"><label>{{ $t('records.texts') }}</label><textarea id="vec-texts" class="code-input" rows="3" v-model="texts" :placeholder="$t('records.ph.texts')"></textarea></div>
          <div class="row" v-show="mode === 'vectors'"><label>{{ $t('records.vectors') }}</label><textarea id="vec-embs" class="code-input" rows="3" v-model="embs" :placeholder="$t('records.ph.vectors')"></textarea></div>
          <details class="collapsible">
            <summary>{{ $t('records.fields') }}</summary>
            <div class="body">
              <div class="row"><label>{{ $t('records.field_set') }}</label><textarea id="vec-fields" class="code-input" rows="3" v-model="fields" :placeholder="$t('records.ph.fields')"></textarea></div>
            </div>
          </details>
          <div class="actions">
            <busy-button id="btn-upsert" :busy="busy === 'upsert'" :label="$t('records.upsert')"
                         :busy-label="$t('records.upserting')" @click="doUpsert" />
          </div>
        </div>

        <div v-show="tab === 'query'" role="tabpanel">
          <div class="row"><label>{{ $t('records.ids') }}</label><textarea id="vec-fetch-ids" class="code-input" rows="2" v-model="ids" :placeholder="$t('records.ph.ids')"></textarea></div>
          <div class="actions">
            <busy-button id="btn-fetch" variant="" :busy="busy === 'fetch'" :label="$t('records.fetch')"
                         :busy-label="$t('records.fetching')" @click="doFetch" />
          </div>
        </div>

        <div v-show="tab === 'delete'" role="tabpanel">
          <div class="row"><label>{{ $t('records.delete_mode') }}</label>
            <select id="vec-del-mode" v-model="delMode">
              <option value="ids">{{ $t('records.del_mode.ids') }}</option>
              <option value="filter">{{ $t('records.del_mode.filter') }}</option>
            </select>
          </div>
          <div class="row" id="vec-del-ids-row" v-show="delMode === 'ids'"><label>{{ $t('records.del_ids') }}</label><textarea id="vec-del-ids" class="code-input" rows="2" v-model="delIds" :placeholder="$t('records.ph.del_ids')"></textarea></div>
          <div class="row" id="vec-del-filter-row" v-show="delMode === 'filter'"><label>{{ $t('records.del_filter') }}</label><textarea id="vec-del-filter" class="code-input" rows="2" v-model="delFilter" :placeholder="$t('records.ph.del_filter')"></textarea></div>
          <!-- Why the red button below is disabled. Lives here, beside
               the params that would unblock it. -->
          <div class="row" v-if="deleteBlockReason"><span class="hint">{{ deleteBlockReason }}</span></div>
          <div class="actions">
            <!-- Disabled until the delete request is well-formed. The title
                 attribute carries the reason for the case where the hint
                 above is not on screen.
                 (No backticks in here: this comment lives inside the
                 component's template literal.) -->
            <busy-button id="btn-delete" variant="danger" :busy="busy === 'delete'" :disabled="!canDelete"
                         :title="deleteBlockReason" :label="$t('records.delete')"
                         :busy-label="$t('records.deleting')" @click="doDelete" />
          </div>
        </div>

        <status-banner kind="error" :text="formErr" />
        <status-banner kind="error" :text="loadErr" :retry="loadErr ? refreshDbs : null" />
        <!-- Replaces the old "last op: OK" line, which reported the word
             OK and nothing else — not even for a failed request. -->
        <status-banner :kind="outcomeKind" :text="outcomeText" />
      </div>

      <!-- Result of the fetch-by-id call. The response used to be thrown
           away entirely, so a successful fetch showed no records. -->
      <div class="section" v-if="fetched">
        <div class="section-head">
          <h3 class="section-title">{{ $t('common.results') }} <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/vectors/get</span></h3>
          <span class="section-sub">{{ $t('records.fetched_n', { n: fetched.length }) }}</span>
          <span class="section-sub">
            <button type="button" class="btn sm" id="btn-fetch-view-table"
                    :class="fetchView === 'table' ? 'primary' : ''"
                    @click="fetchView = 'table'">{{ $t('records.view.table') }}</button>
            <button type="button" class="btn sm" id="btn-fetch-view-json"
                    :class="fetchView === 'json' ? 'primary' : ''"
                    @click="fetchView = 'json'">JSON</button>
          </span>
        </div>
        <empty-state v-if="!fetched.length" state="empty" :text="$t('records.fetched_n', { n: 0 })" />
        <pre v-else-if="fetchView === 'json'" class="code-pane">{{ JSON.stringify(fetched, null, 2) }}</pre>
        <div v-else class="data-table-wrap">
          <table class="data-table">
            <thead><tr>
              <th>{{ primary }}</th><th>vector</th><th>{{ $t('records.field_set') }}</th>
            </tr></thead>
            <tbody>
              <tr v-for="it in fetched" :key="it.id">
                <td class="pk">{{ it.id }}</td>
                <td>{{ it.vector ? $t('records.vector_dims', { n: it.vector.length }) : $t('records.no_vector') }}</td>
                <td><pre class="code-pane" style="margin:0;">{{ JSON.stringify(it.fields, null, 2) }}</pre></td>
              </tr>
            </tbody>
          </table>
        </div>
        <status-banner v-if="fetchMissing.length" kind="warn"
                       :text="$t('records.fetched_missing', { n: fetchMissing.length }) + ': ' + fetchMissing.join(', ')" />
      </div>
    </div>
  `,
});
