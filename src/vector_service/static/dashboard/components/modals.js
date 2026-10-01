// Modals: NewDbModal + NewCollModal.
import { defineComponent, ref, reactive } from '../vue.esm-browser.prod.js';
import { t, store, api, enc, extractApiError } from './app.js';

export const NewDbModal = defineComponent({
  name: 'NewDbModal',
  setup() {
    const name = ref('');
    function close() { store.modals.newDb = false; store.modalErr.newDb = ''; }
    async function submit() {
      const n = name.value.trim();
      if (!n) { store.modalErr.newDb = t('modals.err.db_name_required'); return; }
      store.modalErr.newDb = '';
      try {
        await api('POST', '/v1/databases', { name: n });
        close();
        window.dispatchEvent(new CustomEvent('refresh-dbs'));
      } catch (e) {
        store.modalErr.newDb = extractApiError(e, t('modals.err.create_failed'));
      }
    }
    // `store` must be returned: the template reads store.modalErr.newDb
    // to render the inline error, and a missing binding makes the render
    // throw before the modal can paint (NewCollModal returns it below).
    return { store, name, close, submit };
  },
  template: `
    <div class="modal-overlay" id="modal-new-db" @click.self="close">
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-new-db-title">
        <header class="modal-head">
          <h3 id="modal-new-db-title">{{ $t('modals.new_db') }} <span class="pill accent">POST /v1/databases</span></h3>
          <button class="modal-close" @click="close" aria-label="close">x</button>
        </header>
        <div class="modal-body">
          <div class="row">
            <label>{{ $t('common.name') }} <span class="hint">{{ $t('modals.name_hint') }}</span></label>
            <input type="text" id="new-db-name" v-model="name" placeholder="tenant-a" @keyup.enter="submit" />
          </div>
        </div>
        <footer class="modal-foot">
          <span class="modal-err" id="modal-new-db-err" v-show="store.modalErr.newDb">{{ store.modalErr.newDb }}</span>
          <button class="btn" @click="close" data-close-modal="modal-new-db">{{ $t('common.cancel') }}</button>
          <button class="btn primary" id="btn-create-db" @click="submit">{{ $t('common.create') }}</button>
        </footer>
      </div>
    </div>
  `,
});

const SCALAR_DTYPES = ['bool','int8','int16','int32','int64','float','double','varchar','json'];
const METRICS = ['cosine','ip','l2'];
const INDEX_TYPES = ['HNSW','IVF_FLAT','IVF_SQ8','DISKANN','FLAT'];

function defaultScalar() {
  return { name: '', dtype: 'varchar', is_primary: false, max_length: 64, nullable: false, default_value: '' };
}
function defaultIndex() {
  return { field_name: 'vector', metric_type: 'cosine', index_type: 'HNSW', params: { M: 16, efConstruction: 200 } };
}

export const NewCollModal = defineComponent({
  name: 'NewCollModal',
  setup() {
    const name = ref('');
    const primary = ref('id');
    const vectorName = ref('vector');
    const vectorDim = ref(1024);
    const vectorMetric = ref('cosine');
    const scalars = reactive([{ name: 'id', dtype: 'varchar', is_primary: true, max_length: 64, nullable: false, default_value: '' }]);
    const indices = reactive([defaultIndex()]);

    function close() { store.modals.newColl = false; store.modalErr.newColl = ''; }

    function addScalar() { scalars.push(defaultScalar()); }
    function removeScalar(i) { scalars.splice(i, 1); }
    function applyScalarPreset(preset) {
      scalars.splice(0);
      if (preset === 'id+category+price') {
        scalars.push(
          { name: 'id', dtype: 'varchar', is_primary: true, max_length: 64, nullable: false, default_value: '' },
          { name: 'category', dtype: 'varchar', is_primary: false, max_length: 32, nullable: false, default_value: '' },
          { name: 'price', dtype: 'float', is_primary: false, max_length: null, nullable: false, default_value: '' },
        );
      } else if (preset === 'id+year') {
        scalars.push(
          { name: 'id', dtype: 'varchar', is_primary: true, max_length: 64, nullable: false, default_value: '' },
          { name: 'year', dtype: 'int32', is_primary: false, max_length: null, nullable: false, default_value: '' },
        );
      }
    }
    function applyIndexPreset(preset) {
      indices.splice(0);
      if (preset === 'hnsw-cosine') indices.push({ field_name: 'vector', metric_type: 'cosine', index_type: 'HNSW', params: { M: 16, efConstruction: 200 } });
      else if (preset === 'ivf-l2') indices.push({ field_name: 'vector', metric_type: 'l2', index_type: 'IVF_FLAT', params: { nlist: 64 } });
      else if (preset === 'diskann-ip') indices.push({ field_name: 'vector', metric_type: 'ip', index_type: 'DISKANN', params: {} });
      else if (preset === 'default') indices.push(defaultIndex());
    }

    function buildBody() {
      const primaryCount = scalars.filter(s => s.is_primary).length;
      if (primaryCount !== 1) { store.modalErr.newColl = t('modals.err.one_primary', { n: primaryCount }); return null; }
      const pk = scalars.find(s => s.is_primary);
      if (pk.dtype !== 'varchar') { store.modalErr.newColl = t('modals.err.primary_varchar', { dtype: pk.dtype }); return null; }
      if (!pk.name) { store.modalErr.newColl = t('modals.err.primary_name_required'); return null; }
      for (const r of scalars) {
        if (r.dtype === 'varchar' && (!r.max_length || r.max_length < 1)) {
          store.modalErr.newColl = t('modals.err.varchar_max_length', { name: r.name || t('modals.unnamed') });
          return null;
        }
        if (!r.name) { store.modalErr.newColl = t('modals.err.field_name_required'); return null; }
      }
      if (pk.name !== primary.value.trim()) { store.modalErr.newColl = t('modals.err.primary_mismatch', { name: primary.value }); return null; }
      if (!Number.isFinite(vectorDim.value) || vectorDim.value < 1) { store.modalErr.newColl = t('modals.err.dim'); return null; }
      const scalarFields = scalars.map(r => {
        const out = { name: r.name, dtype: r.dtype, is_primary: !!r.is_primary };
        if (r.dtype === 'varchar') out.max_length = r.max_length;
        if (r.nullable) out.nullable = true;
        if (r.default_value !== '' && r.default_value !== null && r.default_value !== undefined) out.default_value = r.default_value;
        return out;
      });
      return {
        name: name.value.trim(),
        primary_field: primary.value.trim(),
        scalar_fields: scalarFields,
        vector_field: { name: vectorName.value.trim() || 'vector', dim: vectorDim.value, metric_type: vectorMetric.value },
        index_params: indices.map(r => ({ field_name: r.field_name, metric_type: r.metric_type, index_type: r.index_type, params: typeof r.params === 'string' ? (JSON.parse(r.params) || {}) : (r.params || {}) })),
      };
    }

    async function submit() {
      if (!store.databases.list.length) { store.modalErr.newColl = t('modals.err.no_db'); return; }
      if (!name.value.trim()) { store.modalErr.newColl = t('modals.err.coll_name_required'); return; }
      const body = buildBody(); if (!body) return;
      store.modalErr.newColl = '';
      try {
        const dbName = store.databases.list[0];
        await api('POST', '/v1/databases/' + enc(dbName) + '/collections', body);
        close();
        window.dispatchEvent(new CustomEvent('refresh-colls'));
      } catch (e) { store.modalErr.newColl = extractApiError(e, t('modals.err.create_failed')); }
    }

    return {
      store,
      name, primary, vectorName, vectorDim, vectorMetric,
             scalars, indices, addScalar, removeScalar, applyScalarPreset, applyIndexPreset,
             close, submit, SCALAR_DTYPES, METRICS, INDEX_TYPES };
  },
  template: `
    <div class="modal-overlay" id="modal-new-coll" @click.self="close">
      <div class="modal modal--wide" role="dialog" aria-modal="true" aria-labelledby="modal-new-coll-title">
        <header class="modal-head">
          <h3 id="modal-new-coll-title">{{ $t('modals.new_collection') }} <span class="pill accent">POST /v1/databases/{db}/collections</span></h3>
          <button class="modal-close" @click="close" aria-label="close">x</button>
        </header>
        <div class="modal-body">
          <div class="row split">
            <div class="row"><label>{{ $t('modals.coll_name') }} <span class="hint">{{ $t('modals.coll_name_hint') }}</span></label>
              <input type="text" id="new-coll-name" v-model="name" placeholder="products" /></div>
            <div class="row"><label>{{ $t('modals.pk_field') }} <span class="hint">{{ $t('modals.pk_field_hint') }}</span></label>
              <input type="text" id="new-coll-primary" v-model="primary" /></div>
          </div>

          <details class="collapsible" open>
            <summary>{{ $t('modals.scalar_fields') }}</summary>
            <div class="body">
              <div id="scalars-list">
                <div v-for="(s, i) in scalars" :key="i" class="field-card">
                  <div class="field-card-header">
                    <span class="index-badge">{{ s.dtype }}</span>
                    <span class="title">{{ s.name || $t('modals.unnamed') }}<span v-if="s.is_primary"> - {{ $t('modals.primary_suffix') }}</span></span>
                    <button class="remove" @click="removeScalar(i)">x</button>
                  </div>
                  <div class="grid">
                    <div class="field"><label>{{ $t('common.name') }}</label><input type="text" v-model="s.name" /></div>
                    <div class="field"><label>{{ $t('modals.dtype') }}</label>
                      <select v-model="s.dtype" @change="if (s.dtype !== 'varchar') s.max_length = null;">
                        <option v-for="d in SCALAR_DTYPES" :key="d" :value="d">{{ d }}</option>
                      </select>
                    </div>
                    <div class="field" v-if="s.dtype === 'varchar'"><label>{{ $t('modals.max_length') }}</label><input type="number" v-model.number="s.max_length" min="1" /></div>
                    <div class="field"><label><input type="checkbox" v-model="s.is_primary" /> {{ $t('modals.is_primary') }}</label></div>
                    <div class="field"><label><input type="checkbox" v-model="s.nullable" /> {{ $t('modals.nullable') }}</label></div>
                    <div class="field"><label>{{ $t('modals.default_value') }}</label><input type="text" v-model="s.default_value" /></div>
                  </div>
                </div>
              </div>
              <button class="add-row" id="btn-add-scalar" @click="addScalar">{{ $t('modals.add_scalar') }}</button>
              <div class="preset-grid" style="margin-top:12px;">
                <button class="btn" data-preset-scalar="id+category+price" @click="applyScalarPreset('id+category+price')">
                  <span class="preset-title">{{ $t('modals.preset_id_cat_price') }}</span>
                  <span class="preset-desc">{{ $t('modals.preset_id_cat_price_desc') }}</span>
                </button>
                <button class="btn" data-preset-scalar="id+year" @click="applyScalarPreset('id+year')">
                  <span class="preset-title">{{ $t('modals.preset_id_year') }}</span>
                  <span class="preset-desc">{{ $t('modals.preset_id_year_desc') }}</span>
                </button>
              </div>
            </div>
          </details>

          <details class="collapsible" open>
            <summary>{{ $t('modals.vector_field') }}</summary>
            <div class="body">
              <div id="vector-card" class="field-card">
                <div class="field-card-header">
                  <span class="index-badge">vector</span>
                  <span class="title">{{ $t('modals.unique') }}</span>
                </div>
                <div class="grid cols-3">
                  <div class="field"><label>{{ $t('modals.field_name') }}</label><input type="text" v-model="vectorName" /></div>
                  <div class="field"><label>{{ $t('modals.dim') }}</label><input type="number" v-model.number="vectorDim" min="1" max="32768" /></div>
                  <div class="field"><label>{{ $t('common.metric') }}</label>
                    <select v-model="vectorMetric">
                      <option v-for="m in METRICS" :key="m" :value="m">{{ m }}</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </details>

          <details class="collapsible">
            <summary>{{ $t('modals.index_params') }}</summary>
            <div class="body">
              <div id="index-list">
                <div v-for="(ix, i) in indices" :key="i" class="field-card">
                  <div class="field-card-header">
                    <span class="index-badge">{{ ix.index_type }}</span>
                    <span class="title">{{ ix.metric_type }} - {{ ix.index_type }} -> {{ ix.field_name }}</span>
                    <button class="remove" @click="indices.splice(i, 1)">x</button>
                  </div>
                  <div class="grid">
                    <div class="field"><label>{{ $t('modals.field_name') }}</label><input type="text" v-model="ix.field_name" /></div>
                    <div class="field"><label>{{ $t('modals.metric_type') }}</label>
                      <select v-model="ix.metric_type"><option v-for="m in METRICS" :key="m" :value="m">{{ m }}</option></select>
                    </div>
                    <div class="field"><label>{{ $t('modals.index_type') }}</label>
                      <select v-model="ix.index_type"><option v-for="t in INDEX_TYPES" :key="t" :value="t">{{ t }}</option></select>
                    </div>
                    <div class="field"><label>{{ $t('modals.params_json') }}</label>
                      <input type="text" :value="JSON.stringify(ix.params)" @input="ix.params = $event.target.value" />
                    </div>
                  </div>
                </div>
              </div>
              <button class="add-row" id="btn-add-index" @click="indices.push(defaultIndex())">{{ $t('modals.add_index') }}</button>
              <div class="preset-grid" style="margin-top:12px;">
                <button class="btn" data-preset-index="hnsw-cosine" @click="applyIndexPreset('hnsw-cosine')"><span class="preset-title">{{ $t('modals.preset_hnsw_cosine') }}</span><span class="preset-desc">{{ $t('modals.preset_hnsw_cosine_desc') }}</span></button>
                <button class="btn" data-preset-index="ivf-l2" @click="applyIndexPreset('ivf-l2')"><span class="preset-title">{{ $t('modals.preset_ivf_l2') }}</span><span class="preset-desc">{{ $t('modals.preset_ivf_l2_desc') }}</span></button>
                <button class="btn" data-preset-index="diskann-ip" @click="applyIndexPreset('diskann-ip')"><span class="preset-title">{{ $t('modals.preset_diskann_ip') }}</span><span class="preset-desc">{{ $t('modals.preset_diskann_ip_desc') }}</span></button>
                <button class="btn" data-preset-index="default" @click="applyIndexPreset('default')"><span class="preset-title">{{ $t('modals.preset_use_default') }}</span><span class="preset-desc">{{ $t('modals.preset_use_default_desc') }}</span></button>
              </div>
            </div>
          </details>
        </div>
        <footer class="modal-foot">
          <span class="modal-err" id="modal-new-coll-err" v-show="store.modalErr.newColl">{{ store.modalErr.newColl }}</span>
          <button class="btn" @click="close" data-close-modal="modal-new-coll">{{ $t('common.cancel') }}</button>
          <button class="btn primary" id="btn-create-coll" @click="submit">{{ $t('modals.create_collection') }}</button>
        </footer>
      </div>
    </div>
  `,
});
