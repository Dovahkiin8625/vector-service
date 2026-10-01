// Browse panel: paginated list view of collection rows + row deletion
// (checkbox-selected PKs, or every row matching the filter expression).
import { defineComponent, ref, watch, onMounted, computed } from '../vue.esm-browser.prod.js';
import { t, api, enc, extractApiError } from './app.js';
import { StatusBanner, BusyButton, EmptyState, askConfirm } from './feedback.js';

export default defineComponent({
  name: 'BrowsePanel',
  setup() {
    const dbs = ref([]);
    const colls = ref([]);
    const db = ref('');
    const coll = ref('');
    const primary = ref('id');
    const pageSize = ref(20);
    const filter = ref('');
    const offset = ref(0);
    const total = ref(0);
    const items = ref([]);
    const schema = ref(null);
    // Query outcome. Previously a bare string that the template never
    // read, so a running query had no spinner and a failed one was
    // invisible; it now drives the four-state results area.
    const status = ref('idle');
    const queryErr = ref('');
    // Selection/schema loading failures (previously swallowed).
    const loadErr = ref('');
    // Form-level complaint (no filter expression typed).
    const formErr = ref('');
    // Row-level delete state. ``selected`` holds the primary keys the
    // operator ticked across pages (reassigned, never mutated in place,
    // so Vue picks the change up); the rest drives the delete toolbar.
    const selected = ref(new Set());
    const delBusy = ref(false);
    const delStatus = ref('');
    const delStatusKind = ref('');
    // The jump-to-page box needs a writable model of its own. It used to
    // be bound to `currentPage`, which is a read-only computed — typing a
    // page number assigned to a computed with no setter, so the jump
    // silently did nothing (B3).
    const jumpPage = ref(1);
    // The filter the rows on screen were fetched with. Tick-marks are
    // primary keys of rows the operator could see; once the filter
    // changes those rows may not match any more, so the selection is
    // dropped rather than carried into the bulk-delete action (B4).
    const appliedFilter = ref('');

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
    async function refreshSchema() {
      schema.value = null;
      if (!db.value || !coll.value) return;
      try {
        const { payload } = await api('GET', '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value));
        schema.value = payload;
      } catch (e) {
        schema.value = null;
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    // Every one of these switches invalidates the tick-set: the rows the
    // operator selected either belong to another collection (cross-
    // database delete) or are keyed by a field that no longer means the
    // same thing. Dropping the selection first is what keeps "delete
    // selected" from submitting primary keys the operator can no longer
    // see (B4).
    function resetSelectionState() {
      selected.value = new Set();
      delStatus.value = '';
      delStatusKind.value = '';
    }
    watch(db, () => { offset.value = 0; resetSelectionState(); refreshColls(); });
    watch(coll, () => {
      offset.value = 0;
      resetSelectionState();
      refreshSchema();
      runQuery();
    });
    // A different page size re-slices the result set, so offset has to go
    // back to the top; leaving it put would land past the end of the new
    // pagination.
    watch(pageSize, () => { offset.value = 0; runQuery(); });
    // The ticked ids are values of the old primary field.
    watch(primary, () => { resetSelectionState(); });
    onMounted(refreshDbs);

    const columns = computed(() => {
      const set = new Set();
      items.value.forEach(it => Object.keys(it.fields || {}).forEach(k => set.add(k)));
      return [primary.value, ...Array.from(set)].filter((c, i, a) => a.indexOf(c) === i);
    });
    const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)));
    const currentPage = computed(() => Math.floor(offset.value / pageSize.value) + 1);

    // ---- selection + deletion ----
    const pageIds = computed(() => items.value.map(r => r.id));
    const allPageSelected = computed(
      () => pageIds.value.length > 0 && pageIds.value.every(id => selected.value.has(id))
    );
    // Half-ticked header checkbox: some (not all) rows of this page are
    // selected — usually because the rest were ticked on another page.
    const somePageSelected = computed(
      () => !allPageSelected.value && pageIds.value.some(id => selected.value.has(id))
    );
    const selectedCount = computed(() => selected.value.size);
    // The count rides on the button label itself: it is the only place
    // the operator can see how many rows the pending modal will destroy.
    const deleteSelectedLabel = computed(
      () => t('browse.delete_selected') + (selectedCount.value ? ' (' + selectedCount.value + ')' : '')
    );

    // Reassign the Set instead of mutating it — cheap and guarantees
    // reactivity regardless of how the ref is unwrapped in the template.
    function mutateSelection(fn) {
      const next = new Set(selected.value);
      fn(next);
      selected.value = next;
    }
    function toggleOne(id) {
      mutateSelection(s => { if (s.has(id)) s.delete(id); else s.add(id); });
    }
    function togglePage(on) {
      mutateSelection(s => {
        pageIds.value.forEach(id => { if (on) s.add(id); else s.delete(id); });
      });
    }
    function clearSelection() { selected.value = new Set(); }

    function deleteUrl() {
      return '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/vectors/delete';
    }
    async function postDelete(body) {
      delBusy.value = true;
      delStatus.value = '';
      delStatusKind.value = '';
      try {
        const { payload } = await api('POST', deleteUrl(), body);
        const n = (payload && payload.deleted) || 0;
        delStatusKind.value = 'ok';
        delStatus.value = t(n === 1 ? 'browse.deleted_one' : 'browse.deleted_n', { n });
        return true;
      } catch (e) {
        delStatusKind.value = 'error';
        delStatus.value = t('browse.delete_failed') + extractApiError(e, t('common.unknown'));
        return false;
      } finally {
        delBusy.value = false;
      }
    }
    // Re-query after a successful delete; when rows on the current page
    // were removed and it renders empty (last page case), fall back to
    // page 1 rather than stranding the operator on a blank table.
    async function reloadAfterDelete(resetPage) {
      if (resetPage) offset.value = 0;
      // keepDeleteStatus: this refresh is the feedback for the delete
      // itself — don't let runQuery wipe "deleted N rows." (it must
      // survive even when the new total is 0 and the table goes empty).
      await runQuery({ keepDeleteStatus: true });
      if (!resetPage && !items.value.length && offset.value > 0) {
        offset.value = 0;
        await runQuery({ keepDeleteStatus: true });
      }
    }
    async function deleteSelected() {
      if (!db.value || !coll.value || !selected.value.size || delBusy.value) return;
      const ids = Array.from(selected.value);
      // The modal spells out the blast radius: which collection, how many
      // rows. confirm() could only ask a yes/no question.
      const confirmed = await askConfirm({
        title: t('browse.confirm_delete_title'),
        message: t(ids.length === 1 ? 'browse.confirm_delete_one' : 'browse.confirm_delete_n',
                   { n: ids.length }),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: coll.value },
          { label: t('common.rows'), value: String(ids.length) },
        ],
        confirmLabel: t('browse.delete_selected'),
      });
      if (!confirmed) return;
      const ok = await postDelete({ primary_field: primary.value, ids });
      if (ok) {
        selected.value = new Set();
        await reloadAfterDelete(false);
      }
    }
    async function deleteByFilter() {
      if (!db.value || !coll.value || delBusy.value) return;
      formErr.value = '';
      const expr = filter.value.trim();
      if (!expr) { formErr.value = t('browse.err.no_filter'); return; }
      // The filter runs over the WHOLE collection, not just the rows
      // visible on this page — the modal says so before the call.
      const confirmed = await askConfirm({
        title: t('browse.confirm_delete_title'),
        message: t('browse.confirm_delete_filter', { expr }),
        details: [
          { label: t('common.database'), value: db.value },
          { label: t('common.collection'), value: coll.value },
          { label: t('browse.filter_expr'), value: expr },
        ],
        confirmLabel: t('browse.delete_by_filter'),
      });
      if (!confirmed) return;
      const ok = await postDelete({ primary_field: primary.value, filter_expr: expr });
      if (ok) {
        selected.value = new Set();
        await reloadAfterDelete(true);
      }
    }

    async function runQuery(opts = {}) {
      if (!db.value || !coll.value) return;
      // Manual queries (query/reset/pager/coll switch) dismiss a stale
      // delete result; the post-delete reload opts out so its feedback
      // stays on screen — including when the collection is now empty.
      if (!opts.keepDeleteStatus) {
        delStatus.value = '';
        delStatusKind.value = '';
      }
      // A new filter re-scopes the whole result set; the previous ticks
      // were made against the old one and must not survive into it.
      const expr = filter.value.trim();
      if (expr !== appliedFilter.value) {
        appliedFilter.value = expr;
        selected.value = new Set();
      }
      status.value = 'loading';
      queryErr.value = '';
      const body = { primary_field: primary.value, limit: pageSize.value, offset: offset.value };
      if (expr) body.filter_expr = expr;
      try {
        const { payload } = await api('POST',
          '/v1/databases/' + enc(db.value) + '/collections/' + enc(coll.value) + '/rows', body);
        items.value = (payload && payload.items) || [];
        total.value = (payload && payload.total) || 0;
        status.value = 'ok';
      } catch (e) {
        // The previous page stays on screen (a failed page-turn should
        // not blank the table); the banner above it carries the reason.
        status.value = 'error';
        queryErr.value = extractApiError(e, t('common.unknown'));
      }
    }
    function retryQuery() { runQuery(); }

    // Four-state results area. Before this, "loading", "failed", "the
    // collection is empty" and "no filter match" all rendered the same
    // sentence, and a running query showed nothing at all.
    const tableState = computed(() => {
      if (status.value === 'loading') return 'loading';
      if (status.value === 'error') return 'error';
      return status.value === 'idle' ? 'idle' : 'empty';
    });
    const tableText = computed(() => {
      if (status.value === 'error') return t('common.status.error') + ': ' + queryErr.value;
      if (status.value === 'idle') return t('browse.idle_hint');
      return t('browse.empty');
    });
    // A failed page-turn keeps the previous page on screen, so there is
    // no empty block to report it in — the banner above the table is the
    // only place that failure can surface.
    const pageErrText = computed(
      () => (status.value === 'error' && items.value.length ? tableText.value : '')
    );
    function jump(delta) {
      offset.value = Math.max(0, Math.min(offset.value + delta, Math.max(0, (totalPages.value - 1) * pageSize.value)));
      runQuery();
    }
    function jumpTo(page) {
      const n = Number(page);
      if (!Number.isFinite(n) || n < 1) return;
      const target = Math.min(Math.floor(n), totalPages.value);
      offset.value = (target - 1) * pageSize.value;
      // Echo back where the jump actually landed, so an out-of-range
      // entry (page 999) visibly corrects itself to the last page.
      jumpPage.value = target;
      runQuery();
    }
    // Keep the box in step with page turns made by the other buttons; a
    // half-typed number never changes currentPage, so the two do not
    // fight over the field.
    watch(currentPage, (v) => { jumpPage.value = v; });

    return { dbs, colls, db, coll, primary, pageSize, filter, offset, total, items, schema,
             columns, totalPages, currentPage, jumpPage, status, queryErr, loadErr, formErr,
             tableState, tableText, pageErrText, retryQuery, refreshDbs, runQuery, jump, jumpTo,
             selected, allPageSelected, somePageSelected, selectedCount, deleteSelectedLabel,
             delBusy, delStatus, delStatusKind,
             toggleOne, togglePage, clearSelection, deleteSelected, deleteByFilter };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('browse.title') }}</h3>
          <span class="section-sub">{{ $t('browse.sub') }}</span>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.database') }}</label>
            <select id="brw-db" v-model="db"><option v-for="d in dbs" :key="d" :value="d">{{ d }}</option></select>
          </div>
          <div class="row"><label>{{ $t('common.collection') }}</label>
            <select id="brw-coll" v-model="coll"><option v-for="c in colls" :key="c" :value="c">{{ c }}</option></select>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('browse.query_params') }} <span class="pill accent">POST /v1/databases/{db}/collections/{coll}/rows</span></h3>
          <span class="section-sub">{{ $t('browse.query_params_sub') }}</span>
        </div>
        <div class="row split">
          <div class="row"><label>{{ $t('common.primary_field') }}</label><input type="text" id="brw-primary" v-model="primary" /></div>
          <div class="row"><label>{{ $t('browse.page_size') }}</label>
            <select id="brw-page-size" v-model.number="pageSize">
              <option :value="20">20</option><option :value="50">50</option><option :value="100">100</option>
            </select>
          </div>
        </div>
        <div class="row"><label>{{ $t('browse.filter_expr') }}</label>
          <textarea id="brw-filter" class="code-input" rows="2" v-model="filter" placeholder="category == 'mouse'"></textarea>
        </div>
        <div class="row" v-if="schema && schema.fields"><label>{{ $t('browse.output_fields') }} <span class="hint">{{ $t('browse.hint.readonly') }}</span></label>
          <div id="brw-output-tokens">
            <span v-for="f in schema.fields.filter(x => x.dtype !== 'float_vector')" :key="f.name"
                  :class="['tag-token', 'active']" :data-brw-field="f.name">
              <span v-if="f.name === primary" class="pill success" style="padding:0 5px;font-size:9px;margin-right:4px;">PK</span>
              {{ f.name }} <span style="color:var(--text-muted);font-size:10px;">{{ f.dtype }}</span>
            </span>
          </div>
        </div>
        <div class="actions">
          <busy-button id="btn-brw-query" :busy="status === 'loading'" :label="$t('common.query')"
                       :busy-label="$t('browse.running')" @click="runQuery()" />
          <button class="btn" id="btn-brw-reset" :disabled="status === 'loading'"
                  @click="filter = ''; offset = 0; clearSelection(); runQuery()">{{ $t('common.reset') }}</button>
        </div>
        <status-banner kind="error" :text="formErr" />
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('common.results') }}</h3>
          <span class="section-sub" id="brw-summary">
            {{ total ? $t('browse.rows_range', { from: offset + 1, to: offset + items.length, total: total }) : $t('browse.empty_summary') }}
          </span>
        </div>
        <!-- Selection/schema load failures were swallowed before: the
             dropdown just came up empty with no explanation. -->
        <status-banner kind="error" :text="loadErr" :retry="loadErr ? refreshDbs : null" />
        <!-- Failure of a page-turn or refresh, when rows are still on
             screen and the empty block below therefore never renders. -->
        <status-banner kind="error" :text="pageErrText" :retry="pageErrText ? retryQuery : null" />
        <div v-if="total" class="kb-result" id="brw-result-stats">
          <div class="stat"><span class="key">{{ $t('browse.stat_total') }}</span><span class="val accent" id="brw-stat-total">{{ formatCount(total) }}</span></div>
          <div class="stat"><span class="key">{{ $t('browse.stat_page') }}</span><span class="val" id="brw-stat-page">{{ currentPage }} / {{ totalPages }}</span></div>
          <div class="stat"><span class="key">{{ $t('browse.stat_returned') }}</span><span class="val" id="brw-stat-returned">{{ items.length }}</span></div>
        </div>
        <!-- Gated on the collection selection, not total: after a delete
             empties the collection (total -> 0) the success/failure line
             must stay on screen instead of vanishing with the toolbar. -->
        <div v-if="coll" class="actions brw-delete-bar" id="brw-delete-bar">
          <busy-button class="sm" variant="danger" id="btn-brw-delete-selected"
                       :busy="delBusy" :label="deleteSelectedLabel"
                       :busy-label="$t('common.deleting')" :disabled="!selectedCount"
                       @click="deleteSelected" />
          <busy-button class="sm" variant="danger" id="btn-brw-delete-filter"
                       :busy="delBusy" :label="$t('browse.delete_by_filter')"
                       :busy-label="$t('common.deleting')" :disabled="!filter.trim()"
                       @click="deleteByFilter" />
          <button class="btn sm" id="btn-brw-clear-sel"
                  :disabled="!selectedCount" @click="clearSelection">{{ $t('browse.clear_selection') }}</button>
          <span class="hint" id="brw-delete-hint">{{ $t('browse.delete_hint') }}</span>
          <status-banner id="brw-delete-status"
                         :kind="delStatusKind === 'ok' ? 'success' : 'error'" :text="delStatus" />
        </div>
        <div id="brw-table-wrap">
          <!-- Four states instead of one sentence: a running query now
               shows a spinner, a failed one shows the reason plus retry,
               and "not run yet" is told apart from "matched nothing". -->
          <empty-state v-if="!items.length" :state="tableState" :text="tableText"
                       :retry="tableState === 'error' ? retryQuery : null" />
          <div v-else class="data-table-wrap">
            <table class="data-table">
              <thead><tr>
                <th class="row-select"><input type="checkbox" id="brw-select-all"
                  :checked="allPageSelected" :indeterminate="somePageSelected" :disabled="delBusy"
                  @change="togglePage($event.target.checked)"
                  :title="$t('browse.select_all')" /></th>
                <th v-for="c in columns" :key="c">{{ c }}<span v-if="c === primary" class="pk">PK</span></th>
              </tr></thead>
              <tbody>
                <tr v-for="row in items" :key="row.id" :class="{ selected: selected.has(row.id) }">
                  <td class="row-select"><input type="checkbox" class="brw-row-check"
                    :value="row.id" :checked="selected.has(row.id)" :disabled="delBusy"
                    @change="toggleOne(row.id)" :title="$t('browse.select_row', { id: row.id })" /></td>
                  <td v-for="c in columns" :key="c"
                      :class="[formatCell(row, c).cls, c === primary ? 'pk' : '']"
                      :title="formatCell(row, c).title">{{ formatCell(row, c).html }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <!-- Pager buttons are disabled while a query is in flight rather
             than showing five spinners; the table block carries the
             loading state. -->
        <div v-if="total" class="pager" id="brw-pager">
          <button class="btn sm" id="btn-brw-first" :disabled="currentPage <= 1 || status === 'loading'" @click="jumpTo(1)">« {{ $t('common.first') }}</button>
          <button class="btn sm" id="btn-brw-prev"  :disabled="currentPage <= 1 || status === 'loading'" @click="jump(-pageSize)">‹ {{ $t('common.prev') }}</button>
          <span class="info" id="brw-pager-info">{{ $t('browse.pager_info', { from: offset + 1, to: offset + items.length, total: total, page: currentPage, pages: totalPages }) }}</span>
          <button class="btn sm" id="btn-brw-next" :disabled="currentPage >= totalPages || status === 'loading'" @click="jump(pageSize)">{{ $t('common.next') }} ›</button>
          <button class="btn sm" id="btn-brw-last" :disabled="currentPage >= totalPages || status === 'loading'" @click="jumpTo(totalPages)">{{ $t('common.last') }} »</button>
          <span class="info">{{ $t('browse.jump_to') }}</span>
          <input type="number" id="brw-jump" :min="1" :max="totalPages" v-model.number="jumpPage"
                 :aria-label="$t('browse.stat_page')" style="width:72px;" />
          <button class="btn sm" id="btn-brw-jump" :disabled="status === 'loading'" @click="jumpTo(jumpPage)">{{ $t('browse.go') }}</button>
        </div>
      </div>
      <div class="empty hint">{{ $t('browse.footer_hint') }}</div>
    </div>
  `,
  methods: {
    formatCount(n) {
      if (n == null || isNaN(n)) return '—';
      const v = Number(n);
      if (v < 1000) return String(v);
      if (v < 1e6) return (v / 1e3).toFixed(v < 1e4 ? 1 : 0) + 'k';
      return (v / 1e6).toFixed(v < 1e7 ? 2 : 1) + 'M';
    },
    formatCell(row, col) {
      const val = col === this.primary ? row.id : (row.fields ? row.fields[col] : undefined);
      if (val === null || val === undefined) return { html: 'null', cls: 'null', title: 'null' };
      if (typeof val === 'object') return { html: JSON.stringify(val), cls: 'json', title: JSON.stringify(val) };
      const s = String(val);
      return { html: s, cls: '', title: s };
    },
  },
});
