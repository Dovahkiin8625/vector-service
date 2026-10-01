// Models registry panel: lists every registered model with load/unload.
import { defineComponent, onMounted, ref } from '../vue.esm-browser.prod.js';
import { store, api, extractApiError, applyModelsData, t } from './app.js';
import { notify, StatusBanner, BusyButton, EmptyState } from './feedback.js';

const FAMILY_LABELS = {
  embedder: 'embedder', image_embedder: 'image_embedder',
  multimodal_embedder: 'multimodal_embedder', reranker: 'reranker',
};

// ---- formatters for the loaded-instance meta block -----------------------
// ``model_info`` arrives from GET /v1/models once an instance is held:
// {device, dtype, param_count, memory_bytes, load_duration_seconds}.
// Every value is best-effort (null -> em dash) so the card degrades
// gracefully for backends with no discoverable torch module.
function formatParams(n) {
  if (n == null) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function formatBytes(b) {
  if (b == null) return '—';
  const GB = 1024 ** 3, MB = 1024 ** 2, KB = 1024;
  if (b >= GB) return (b / GB).toFixed(2) + ' GB';
  if (b >= MB) return (b / MB).toFixed(1) + ' MB';
  if (b >= KB) return (b / KB).toFixed(1) + ' KB';
  return b + ' B';
}

function formatDuration(s) {
  if (s == null) return '—';
  if (s >= 60) return Math.floor(s / 60) + 'm ' + Math.round(s % 60) + 's';
  if (s >= 1) return s.toFixed(1) + ' s';
  return Math.round(s * 1000) + ' ms';
}

const DTYPE_LABELS = {
  float16: 'FP16', bfloat16: 'BF16', float32: 'FP32',
  int8: 'INT8', int4: 'INT4',
};

export default defineComponent({
  name: 'ModelsPanel',
  setup() {
    const refreshing = ref(false);
    // The list refresh used to fail silently: an empty grid is also what
    // "no models registered" looks like, so the panel must say which one
    // it is.
    const loadErr = ref('');
    async function refresh() {
      refreshing.value = true;
      loadErr.value = '';
      try {
        const { payload } = await api('GET', '/v1/models');
        // Reconcile (busy flags, failure alerts, dim) in ONE place shared
        // with the background poller in app.js.
        applyModelsData((payload && payload.data) || []);
      } catch (e) {
        loadErr.value = t('models.load_failed') + ': ' + extractApiError(e, t('common.unknown'));
      }
      refreshing.value = false;
    }
    async function lifecycle(verb, id) {
      // Optimistic: remember WHICH verb is in flight and flip the card
      // synchronously, before the POST round-trips. The button area
      // swaps to a disabled "加载中…/卸载中…" on the same click so it
      // never looks like the click was ignored. applyModelsData clears
      // the flag once the polled server state settles.
      store.models.busy[id] = verb;
      try {
        await api('POST', '/v1/models/' + id + '/' + verb);
        // /load returns 202 immediately; the card keeps its spinner until
        // polling observes load_status loaded/failed (applyModelsData
        // clears the busy flag). /unload is synchronous and the refresh
        // below already shows a settled row.
        await refresh();
      } catch (e) {
        // Request itself failed (404/409/5xx) — no background state will
        // ever settle this card, so release the spinner here. The card
        // has no error slot of its own (its `load_error` line belongs to
        // a server-side load failure, not to a rejected request), so the
        // failure goes to the app-level notice bar — which is also where
        // a background poll failure for this panel would land.
        store.models.busy[id] = false;
        const label = verb === 'load' ? t('models.load_failed') : t('models.unload_failed');
        notify('error', id + ' — ' + label + ': ' + extractApiError(e, t('common.unknown')));
      }
    }
    onMounted(refresh);
    return { store, refresh, lifecycle, refreshing, loadErr };
  },
  components: { StatusBanner, BusyButton, EmptyState },
  template: `
    <div>
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('nav.models') }} <span class="pill accent">GET /v1/models</span></h3>
          <span class="section-sub">{{ $t('common.model_id_ops') }}</span>
        </div>
        <div class="actions">
          <busy-button :busy="refreshing" :label="$t('common.refresh')"
                       :busy-label="$t('models.refreshing')" @click="refresh" />
          <button class="btn sm" @click="store.models.autoRefresh = !store.models.autoRefresh">
            {{ $t('models.auto_refresh_label') }} {{ $t(store.models.autoRefresh ? 'common.on' : 'common.off') }}
          </button>
        </div>
        <status-banner kind="error" :text="loadErr" :retry="loadErr ? refresh : null" />
        <div class="model-grid" id="models-grid">
          <empty-state v-if="!loadErr && !store.models.data.length" state="empty"
                       :text="$t('common.no_models')" />
          <div v-for="m in store.models.data" :key="m.id" :data-id="m.id"
               :class="['model-card', m.loaded ? 'loaded' : '',
                        (isLoading(m) || isUnloading(m)) ? 'busy' : '',
                        (!isLoading(m) && m.load_status === 'failed') ? 'load-failed' : '']">
            <div class="model-card-head">
              <div class="model-title">
                <span class="id" :title="m.id">{{ m.id }}</span>
                <span class="family-tag">{{ familyLabel(m.type) }}</span>
              </div>
              <span :class="['status-pill', m.loaded ? 'loaded' : '',
                             isLoading(m) ? 'loading' : '',
                             (!isLoading(m) && m.load_status === 'failed') ? 'failed' : '']">
                <span class="dot"></span>{{ pillText(m) }}
              </span>
            </div>
            <div class="model-current">
              <span v-if="m.loaded" class="dim">{{ m.dimensions }} {{ $t('topbar.dim_unit') }}</span>
              <span v-else-if="!isLoading(m) && m.load_status === 'failed'" class="model-load-error" :title="m.load_error">{{ m.load_error }}</span>
              <span v-else-if="!isLoading(m)" class="model-empty">{{ $t('common.no_instance') }}</span>
              <span v-if="m.loaded && m.model_info && deviceKind(m)"
                    :class="['device-chip', isGpu(m) ? 'gpu' : 'cpu']"
                    :title="deviceTitle(m)">
                <svg class="device-ic" width="11" height="11" viewBox="0 0 16 16" fill="none"
                     stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true">
                  <rect x="4" y="4" width="8" height="8" rx="1.5"/>
                  <path d="M6 1.5V4M10 1.5V4M6 12v2.5M10 12v2.5M1.5 6H4M1.5 10H4M12 6h2.5M12 10h2.5"/>
                </svg>{{ deviceKind(m) }}<span v-if="dtypeText(m)" class="device-dtype">· {{ dtypeText(m) }}</span>
              </span>
            </div>
            <div v-if="m.loaded && m.model_info" class="model-meta">
              <div class="meta-item">
                <span class="meta-k">{{ $t('models.meta_params') }}</span>
                <span class="meta-v" :title="m.model_info.param_count != null ? Number(m.model_info.param_count).toLocaleString() : ''">{{ formatParams(m.model_info.param_count) }}</span>
              </div>
              <div class="meta-item">
                <span class="meta-k">{{ $t(isGpu(m) ? 'models.meta_vram' : 'models.meta_ram') }}</span>
                <span class="meta-v">{{ formatBytes(m.model_info.memory_bytes) }}</span>
              </div>
              <div class="meta-item wide">
                <span class="meta-k">{{ $t('models.meta_load_time') }}</span>
                <span class="meta-v">{{ formatDuration(m.model_info.load_duration_seconds) }}</span>
              </div>
            </div>
            <div class="model-dots">
              <span v-for="i in Math.min(m.dimensions || 16, 32)" :key="i"
                    :class="['mini-dot', m.loaded && i <= 8 ? 'accent' : '']"></span>
            </div>
            <div class="model-actions">
              <button v-if="isLoading(m)" class="btn sm is-pending lc-card-load" :data-id="m.id" disabled>
                <span class="btn-spinner" aria-hidden="true"></span>{{ $t('models.loading') }}
              </button>
              <button v-else-if="isUnloading(m)" class="btn sm is-pending lc-card-unload" :data-id="m.id" disabled>
                <span class="btn-spinner" aria-hidden="true"></span>{{ $t('models.unloading') }}
              </button>
              <button v-else-if="!m.loaded" class="btn sm primary lc-card-load" :data-id="m.id"
                      @click="lifecycle('load', m.id)">{{ $t('models.load') }}</button>
              <button v-else class="btn sm danger lc-card-unload" :data-id="m.id"
                      @click="lifecycle('unload', m.id)">{{ $t('models.unload') }}</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  methods: {
    familyLabel(type) { return FAMILY_LABELS[type] || type || this.$t('common.unknown'); },
    formatParams,
    formatBytes,
    formatDuration,
    isGpu(m) {
      const d = m.model_info && m.model_info.device;
      return typeof d === 'string' && d.toLowerCase().startsWith('cuda');
    },
    // Explicit GPU/CPU label — ``cuda*`` is a GPU, everything else with
    // a known device (cpu / mps / …) is rendered as CPU. Returns '' only
    // when the device is genuinely unknown (chip hidden).
    deviceKind(m) {
      const info = m.model_info;
      if (!info || !info.device) return '';
      return this.isGpu(m) ? 'GPU' : 'CPU';
    },
    dtypeText(m) {
      const d = m.model_info && m.model_info.dtype;
      return d ? (DTYPE_LABELS[d] || String(d).toUpperCase()) : '';
    },
    deviceTitle(m) {
      const info = m.model_info;
      if (!info) return '';
      const kind = this.deviceKind(m);
      return [
        info.device
          ? (kind ? kind + ' · ' : '') + this.$t('models.device_label') + ': ' + info.device
          : '',
        info.dtype ? this.$t('models.dtype_label') + ': ' + info.dtype : '',
      ].filter(Boolean).join('\n');
    },
    // A load is visually in flight when the SERVER says 'loading' OR when
    // OUR click is still being acknowledged (optimistic window between
    // the click and the first poll that observes server-side 'loading').
    isLoading(m) {
      return m.load_status === 'loading' || store.models.busy[m.id] === 'load';
    },
    isUnloading(m) {
      return store.models.busy[m.id] === 'unload';
    },
    pillText(m) {
      if (this.isLoading(m)) return this.$t('models.loading');
      if (m.loaded) return this.$t('common.loaded');
      if (m.load_status === 'failed') return this.$t('models.load_failed');
      return this.$t('common.unloaded');
    },
  },
});
