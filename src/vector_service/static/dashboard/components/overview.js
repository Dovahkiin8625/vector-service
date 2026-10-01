// Overview panel: service health + unified loaded-capability view.
import { defineComponent, onMounted, onUnmounted, ref } from '../vue.esm-browser.prod.js';
import { t, store, api, extractApiError } from './app.js';
import { formatParams, formatBytes, dtypeText } from './util.js';
import { StatusBanner } from './feedback.js';
import ParserProfileCards from './parser-cards.js';

function formatUptime(seconds) {
  if (seconds == null) return '—';
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return d + 'd ' + h + 'h';
  if (h) return h + 'h ' + m + 'm';
  return m + 'm';
}

export default defineComponent({
  name: 'OverviewPanel',
  components: { StatusBanner, ParserProfileCards },
  setup() {
    const summary = ref(null);
    const timer = ref(null);
    // The 5s poll failing used to be invisible: the KPI tiles silently
    // kept showing the last good values (or em-dashes) forever.
    const loadErr = ref('');
    async function refresh() {
      try {
        const { payload } = await api('GET', '/v1/system/status');
        if (payload) summary.value = payload;
        loadErr.value = '';
        const m = (store.models.data || []).find(x => x.type === 'embedder' && x.loaded);
        store.embedderDim = m && m.dimensions ? m.dimensions : 0;
      } catch (e) {
        loadErr.value = t('common.load_failed') + extractApiError(e, t('common.unknown'));
      }
    }
    onMounted(() => { refresh(); timer.value = setInterval(refresh, 5000); });
    onUnmounted(() => { if (timer.value) clearInterval(timer.value); });
    return { store, summary, loadErr, formatUptime, formatParams, formatBytes, dtypeText, refresh };
  },
  template: `
    <div>
      <status-banner kind="error" :text="loadErr" :retry="loadErr ? refresh : null" />
      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('nav.overview') }} <span class="pill accent">GET /v1/system/status</span></h3>
          <span class="section-sub">{{ $t('overview.auto_refresh') }}</span>
        </div>
        <div class="overview-grid">
          <div class="kpi kpi--accent"><span class="label">{{ $t('overview.version') }}</span><span class="value">{{ (summary && summary.service && summary.service.version) || '—' }}</span><span class="sub">vector-service</span></div>
          <div class="kpi"><span class="label">{{ $t('overview.uptime') }}</span><span class="value">{{ formatUptime(summary && summary.service && summary.service.uptime_seconds) }}</span><span class="sub">{{ $t('overview.uptime_sub') }}</span></div>
          <div class="kpi"><span class="label">{{ $t('overview.loaded_total') }}</span><span class="value">{{ loadedCount }} / {{ store.models.data.length }}</span><span class="sub">{{ $t('overview.loaded_total_sub') }}</span></div>
          <div class="kpi"><span class="label">{{ $t('overview.vector_store') }}</span><span class="value">{{ (summary && summary.store && summary.store.backend) || '—' }}</span><span class="sub">{{ (summary && summary.store && summary.store.status) || '—' }}</span></div>
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <h3 class="section-title">{{ $t('overview.capabilities') }}</h3>
          <span class="section-sub">{{ $t('overview.capabilities_sub') }}</span>
        </div>

        <h4 class="cap-group-title">{{ $t('overview.cap_inference') }} · {{ loadedModels.length }}</h4>
        <div class="cap-grid">
          <div v-for="m in loadedModels" :key="m.id" class="model-card loaded">
            <div class="model-card-head">
              <div class="model-title">
                <span class="id" :title="m.id">{{ m.id }}</span>
                <span class="family-tag">{{ m.type }}</span>
              </div>
              <span class="status-pill loaded"><span class="dot"></span>{{ $t('common.loaded') }}</span>
            </div>
            <div class="model-current">
              <span class="dim">{{ m.dimensions }} {{ $t('topbar.dim_unit') }}</span>
              <span v-if="m.model_info && m.model_info.device"
                    :class="['device-chip', isGpu(m.model_info.device) ? 'gpu' : 'cpu']">
                {{ isGpu(m.model_info.device) ? 'GPU' : 'CPU' }}<span v-if="m.model_info.dtype" class="device-dtype">· {{ dtypeText(m.model_info.dtype) }}</span>
              </span>
            </div>
            <div class="model-meta">
              <div class="meta-item">
                <span class="meta-k">{{ $t('models.meta_params') }}</span>
                <span class="meta-v">{{ formatParams(m.model_info && m.model_info.param_count) }}</span>
              </div>
              <div class="meta-item">
                <span class="meta-k">{{ $t(isGpu(m.model_info && m.model_info.device) ? 'models.meta_vram' : 'models.meta_ram') }}</span>
                <span class="meta-v">{{ formatBytes(m.model_info && m.model_info.memory_bytes) }}</span>
              </div>
            </div>
          </div>
          <div v-if="!loadedModels.length" class="empty cap-empty-inline">{{ $t('overview.cap_inference_empty') }}</div>
        </div>

        <h4 class="cap-group-title">{{ $t('overview.cap_parser') }} · {{ warmParserCount }} / {{ parserCount }}</h4>
        <parser-profile-cards :parser="summary && summary.parser" :readonly="true" />
        <p v-if="hasInvisibleOcr" class="cap-note">{{ $t('cap.onnx_note') }}</p>
      </div>

      <div class="section">
        <div class="section-head"><h3 class="section-title">{{ $t('overview.connection_status') }}</h3></div>
        <div style="background:var(--surface-1);border:1px solid var(--border);border-radius:var(--r-md);padding:14px 16px;">
          <table class="info-table">
            <tr><th>{{ $t('overview.backend') }}</th><td>{{ (summary && summary.store && summary.store.backend) || '—' }}</td></tr>
            <tr><th>URI</th><td class="muted">{{ (summary && summary.store && summary.store.uri) || '—' }}</td></tr>
            <tr><th>{{ $t('overview.status_label') }}</th><td>{{ (summary && summary.store && summary.store.status) || '—' }}</td></tr>
            <tr><th>{{ $t('overview.db_count') }}</th><td>{{ (summary && summary.store && summary.store.databases || []).length }}</td></tr>
          </table>
        </div>
      </div>
    </div>
  `,
  computed: {
    loadedCount() { return (store.models.data || []).filter(m => m.loaded).length; },
    loadedModels() {
      return (store.models.data || []).filter(m => m.loaded);
    },
    warmParserCount() {
      const profiles = this.summary && this.summary.parser && this.summary.parser.profiles;
      if (!profiles) return 0;
      return Object.values(profiles).filter(p => p.warm).length;
    },
    // Denominator comes from the profiles the server actually reports —
    // the header used to hardcode "/ 3".
    parserCount() {
      const profiles = this.summary && this.summary.parser && this.summary.parser.profiles;
      return profiles ? Object.keys(profiles).length : 0;
    },
    // RapidOCR ONNX sessions are invisible to the torch walk — surface
    // one global note whenever a warm standard/vlm profile carries one.
    hasInvisibleOcr() {
      const profiles = this.summary && this.summary.parser && this.summary.parser.profiles;
      if (!profiles) return false;
      return Object.values(profiles).some(
        p => p.warm && (p.components || []).some(
          c => c.kind === 'ocr' && c.resource_visible === false
        )
      );
    },
  },
  methods: {
    isGpu(device) {
      return typeof device === 'string' && device.toLowerCase().startsWith('cuda');
    },
  },
});
