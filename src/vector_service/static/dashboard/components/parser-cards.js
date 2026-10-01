// Shared Docling per-profile status cards.
//
// Read-only mode (overview): shows warm profiles, their model
// components (DocLayNet / TableFormer / RapidOCR / Granite VLM) and
// torch weight resources.
// Interactive mode (parse pane): adds per-profile warm / evict
// actions, worded as cache operations — evict does NOT stop parsing,
// the converter rebuilds on the next request.
import { defineComponent, ref, watch } from '../vue.esm-browser.prod.js';
import { api, extractApiError, t } from './app.js';
import { formatParams, formatBytes } from './util.js';
import { StatusBanner } from './feedback.js';

const PROFILES = ['standard', 'native', 'vlm'];

export default defineComponent({
  name: 'ParserProfileCards',
  props: {
    parser: { type: Object, default: null },
    readonly: { type: Boolean, default: false },
  },
  emits: ['changed'],
  setup(props, { emit }) {
    // Per-profile in-flight action: 'warm' | 'evict' | false.
    const busy = ref(Object.create(null));
    // Per-profile failure of the last warm/evict request, shown on the
    // card that was clicked rather than in a native alert() — the action
    // belongs to one profile, so the complaint should too.
    const err = ref(Object.create(null));

    async function act(action, profile) {
      busy.value[profile] = action;
      err.value[profile] = '';
      try {
        await api('POST', '/v1/parser/' + action, { profile });
        // Let the parent re-pull status so the card settles from the
        // same payload the overview polls.
        emit('changed', { profile, action });
      } catch (e) {
        const prefix = action === 'warm'
          ? t('cap.warm_failed')
          : t('cap.evict_failed');
        err.value[profile] = prefix + ': ' + extractApiError(e, t('common.unknown'));
        busy.value[profile] = false;
      }
    }
    // Settle a card once the polled status reflects OUR action, mirroring
    // applyModelsData's busy reconciliation for the model cards.
    watch(
      () => props.parser,
      (status) => {
        if (!status || !status.profiles) return;
        for (const profile of PROFILES) {
          const warm = Boolean(status.profiles[profile] && status.profiles[profile].warm);
          if (busy.value[profile] === 'warm' && warm) busy.value[profile] = false;
          else if (busy.value[profile] === 'evict' && !warm) busy.value[profile] = false;
        }
      },
      { deep: true },
    );

    return { busy, err, act, PROFILES, formatParams, formatBytes };
  },
  components: { StatusBanner },
  template: `
    <div class="cap-grid">
      <div v-for="p in PROFILES" :key="p"
           :class="['model-card', profileData(p) && profileData(p).warm ? 'loaded' : '',
                    busy[p] ? 'busy' : '']">
        <div class="model-card-head">
          <div class="model-title">
            <span class="id">{{ $t('profile.' + p) }}</span>
            <span class="family-tag">docling</span>
          </div>
          <span :class="['status-pill', profileData(p) && profileData(p).warm ? 'loaded' : '']">
            <span class="dot"></span>{{ profileData(p) && profileData(p).warm ? $t('cap.warm') : $t('cap.cold') }}
          </span>
        </div>

        <!-- Warm: component inventory + resources -->
        <div v-if="profileData(p) && profileData(p).warm" class="cap-body">
          <div class="cap-comps">
            <div v-for="(c, i) in profileData(p).components" :key="i" class="cap-comp">
              <span class="cap-comp-name">{{ componentLabel(c) }}<span v-if="c.ref" class="cap-comp-ref"> · {{ c.ref }}</span></span>
              <span v-if="c.torch" class="cap-comp-stats">
                <span :class="['device-chip', isGpu(c.device) ? 'gpu' : 'cpu']">
                  {{ isGpu(c.device) ? 'GPU' : 'CPU' }}
                </span>
                <span class="cap-comp-num">{{ formatParams(c.param_count) }}</span>
                <span class="cap-comp-bytes">{{ formatBytes(c.memory_bytes) }}</span>
              </span>
              <span v-else class="cap-comp-hidden">ONNX · {{ $t('cap.resource_hidden') }}</span>
            </div>
            <div v-if="profileData(p).model_free" class="cap-comp">
              <span class="cap-comp-name">{{ $t('cap.model_free') }}</span>
            </div>
            <!-- Converter built but pipelines not (lazy version without
                 pre-init): not model-free — weights come on first parse. -->
            <div v-else-if="!profileData(p).pipeline_count" class="cap-comp">
              <span class="model-empty">{{ $t('cap.warm_idle') }}</span>
            </div>
          </div>
          <div class="cap-totals">
            <span class="meta-k">{{ $t('cap.total_weights') }}</span>
            <span class="meta-v">{{ formatParams(profileData(p).torch_param_count) }} · {{ formatBytes(profileData(p).torch_memory_bytes) }}</span>
          </div>
        </div>

        <!-- Cold: explain what warm means instead of pretending stopped -->
        <div v-else class="cap-body">
          <span class="model-empty">{{ $t('cap.cold_hint') }}</span>
        </div>

        <status-banner v-if="err[p]" kind="error" :text="err[p]" />

        <div v-if="!readonly" class="model-actions">
          <button v-if="busy[p] === 'warm'" class="btn sm is-pending" disabled>
            <span class="btn-spinner" aria-hidden="true"></span>{{ $t('cap.warming') }}
          </button>
          <button v-else-if="busy[p] === 'evict'" class="btn sm is-pending" disabled>
            <span class="btn-spinner" aria-hidden="true"></span>{{ $t('cap.evicting') }}
          </button>
          <button v-else-if="!(profileData(p) && profileData(p).warm)" class="btn sm primary"
                  @click="act('warm', p)">{{ $t('cap.warm_btn') }}</button>
          <button v-else class="btn sm danger"
                  @click="act('evict', p)">{{ $t('cap.evict_btn') }}</button>
        </div>
        <p v-if="!readonly" class="cap-note">{{ $t('cap.rebuild_note') }}</p>
      </div>
    </div>
  `,
  methods: {
    profileData(profile) {
      return this.parser && this.parser.profiles
        ? this.parser.profiles[profile]
        : null;
    },
    componentLabel(c) {
      return this.$t('cap.kind.' + c.kind);
    },
    isGpu(device) {
      return typeof device === 'string' && device.toLowerCase().startsWith('cuda');
    },
  },
});
