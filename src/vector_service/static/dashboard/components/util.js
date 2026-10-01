// =====================================================================
// util.js -- shared panel utilities (S6).
//   * fileToB64        base64 reads for image inputs (was 2 copies)
//   * image MIME set   the accept/MIME vocabulary (was 3 dropdown copies)
//   * model type map   kind/modality -> GET /v1/models type filter
//   * dtypeText        model_info.dtype display labels (was 2 copies)
//   * formatParams/formatBytes  model_info formatters (was 3 copies)
//   * intError         min/max/whole-number validation for number inputs
// Every value is best-effort (null -> em dash) so cards degrade
// gracefully for backends with no discoverable model_info.
// =====================================================================
import { t } from './app.js';

// Base64 without FileReader: read the file as bytes, then btoa over a
// byte string. Async so callers can `await` uniformly with text fields.
export function fileToB64(f) {
  return f.arrayBuffer().then((buf) => {
    const bytes = new Uint8Array(buf);
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(s);
  });
}

// The image vocabulary: the file picker accept list, the MIME dropdown
// options, and the fallback when a dropped file carries no type must all
// agree, so they come from one place.
export const DEFAULT_MIME = 'image/png';
export const IMAGE_MIME_OPTIONS = ['image/png', 'image/jpeg', 'image/webp'];
export const IMAGE_ACCEPT = IMAGE_MIME_OPTIONS.join(',');

// Panel filter kind ("text" / "image" / "multimodal" / "rerank") ->
// the `type` field of a GET /v1/models row.
export function modelTypeOf(kind) {
  if (kind === 'image') return 'image_embedder';
  if (kind === 'multimodal') return 'multimodal_embedder';
  if (kind === 'rerank') return 'reranker';
  return 'embedder';
}

export const DTYPE_LABELS = {
  float16: 'FP16', bfloat16: 'BF16', float32: 'FP32',
  int8: 'INT8', int4: 'INT4',
};

export function dtypeText(dtype) {
  // '' for missing so callers can v-if the badge away (models cards);
  // unknown backends fall back to the raw dtype, uppercased.
  if (!dtype) return '';
  return DTYPE_LABELS[dtype] || String(dtype).toUpperCase();
}

export function formatParams(n) {
  if (n == null) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function formatBytes(b) {
  if (b == null) return '—';
  const GB = 1024 ** 3, MB = 1024 ** 2, KB = 1024;
  if (b >= GB) return (b / GB).toFixed(2) + ' GB';
  if (b >= MB) return (b / MB).toFixed(1) + ' MB';
  if (b >= KB) return (b / KB).toFixed(1) + ' KB';
  return b + ' B';
}

// Numeric inputs with `v-model.number` keep '' when the box is cleared
// and can hold half-typed values; both used to ride into the request
// body unchecked. Returns '' when fine, otherwise the complaint to show
// under the field. Range messages carry min/max in both locales.
export function intError(value, min, max) {
  const n = Number(value);
  if (value === '' || value === null || value === undefined
      || !Number.isFinite(n) || !Number.isInteger(n)) {
    return t('common.err_int');
  }
  if ((min != null && n < min) || (max != null && n > max)) {
    // One-sided ranges get their own phrasing: interpolating a missing
    // bound into the two-sided message printed "... 1-null".
    if (min != null && max != null) return t('common.err_int_range', { min, max });
    if (min != null) return t('common.err_int_min', { min });
    return t('common.err_int_max', { max });
  }
  return '';
}
