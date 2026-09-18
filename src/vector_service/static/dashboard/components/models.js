// Models registry panel: lists every registered model with load/unload.
import { defineComponent, onMounted, ref } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError } from './app.js';

const FAMILY_LABELS = {
  embedder: 'embedder', image_embedder: 'image_embedder',
  multimodal_embedder: 'multimodal_embedder', reranker: 'reranker',
};

export default defineComponent({
  name: 'ModelsPanel',
  setup() {
    const refreshing = ref(false);
    async function refresh() {
      refreshing.value = true;
      try {
        const { payload } = await api('GET', '/v1/models');
        store.models.data = (payload && payload.data) || [];
      } catch (_e) { /* logged */ }
      refreshing.value = false;
    }
    async function lifecycle(verb, id) {
      store.models.busy[id] = true;
      try {
        await api('POST', '/v1/models/' + id + '/' + verb);
        await refresh();
      } catch (e) {
        const label = verb === 'load' ? '加载失败' : '卸载失败';
        alert(label + ': ' + extractApiError(e, 'unknown'));
      }
      store.models.busy[id] = false;
    }
    onMounted(refresh);
    return { store, refresh, lifecycle, refreshing };
  },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('nav.models') }} <span class="pill accent">GET /v1/models</span></h3>
          <span class="section-sub">{{ $t('common.model_id_ops') }}</span>
        </div>
        <div class="actions">
          <button class="btn primary" :disabled="refreshing" @click="refresh">{{ $t('common.refresh') }}</button>
          <button class="btn sm" @click="store.models.autoRefresh = !store.models.autoRefresh">
            {{ $t('models.auto_refresh_label') }} {{ store.models.autoRefresh ? '开' : '关' }}
          </button>
        </div>
        <div class="model-grid" id="models-grid">
          <div v-if="!store.models.data.length" class="empty">{{ $t('common.no_models') }}</div>
          <div v-for="m in store.models.data" :key="m.id" :data-id="m.id"
               :class="['model-card', m.loaded ? 'loaded' : '', store.models.busy[m.id] ? 'busy' : '']">
            <div class="model-card-head">
              <div class="model-title">
                <span class="id" :title="m.id">{{ m.id }}</span>
                <span class="family-tag">{{ familyLabel(m.type) }}</span>
              </div>
              <span :class="['status-pill', m.loaded ? 'loaded' : '']">
                <span class="dot"></span>{{ m.loaded ? $t('common.loaded') : $t('common.unloaded') }}
              </span>
            </div>
            <div class="model-current">
              <span v-if="m.loaded" class="dim">{{ m.dimensions }} {{ $t('topbar.dim_unit') }}</span>
              <span v-else class="model-empty">{{ $t('common.unloaded') }} - no instance</span>
            </div>
            <div class="model-dots">
              <span v-for="i in Math.min(m.dimensions || 16, 32)" :key="i"
                    :class="['mini-dot', m.loaded && i <= 8 ? 'accent' : '']"></span>
            </div>
            <div class="model-actions">
              <button v-if="!m.loaded" class="btn sm primary lc-card-load" :data-id="m.id"
                      :disabled="store.models.busy[m.id]" @click="lifecycle('load', m.id)">{{ $t('models.load') }}</button>
              <button v-else class="btn sm danger lc-card-unload" :data-id="m.id"
                      :disabled="store.models.busy[m.id]" @click="lifecycle('unload', m.id)">{{ $t('models.unload') }}</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  methods: {
    familyLabel(type) { return FAMILY_LABELS[type] || type || 'unknown'; },
  },
});
