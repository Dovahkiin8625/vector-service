// Overview panel: service health + model load state.
import { defineComponent, onMounted, onUnmounted, ref } from '../vue.esm-browser.prod.js';
import { store, api } from './app.js';

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
  setup() {
    const summary = ref(null);
    const timer = ref(null);
    async function refresh() {
      try {
        const { payload } = await api('GET', '/v1/system/status');
        if (payload) summary.value = payload;
        const m = (store.models.data || []).find(x => x.type === 'embedder' && x.loaded);
        store.embedderDim = m && m.dimensions ? m.dimensions : 0;
      } catch (_e) { /* logged via api() */ }
    }
    onMounted(() => { refresh(); timer.value = setInterval(refresh, 5000); });
    onUnmounted(() => { if (timer.value) clearInterval(timer.value); });
    return { store, summary, formatUptime, refresh };
  },
  template: `
    <div>
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
  },
});
