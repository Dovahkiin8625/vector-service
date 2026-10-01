// =====================================================================
// ops-common.js -- shared helpers for the Operations panels
// (ops-reindex / ops-queue / ops-consistency / ops-eval).
// =====================================================================

import { enc } from './app.js';

export { enc };

// Job/run/check statuses that will never transition again.
export const TERMINAL = new Set(['done', 'failed', 'cancelled', 'passed']);

export function formatTs(ts) {
  if (ts === null || ts === undefined) return '—';
  return new Date(ts * 1000).toLocaleString();
}

export function pct(v, digits = 1) {
  if (v === null || v === undefined) return '—';
  return (v * 100).toFixed(digits) + '%';
}

export function fixed(v, digits = 3) {
  if (v === null || v === undefined) return '—';
  return Number(v).toFixed(digits);
}

// Map an arbitrary status string onto a pill colour variant.
export function pillClass(status) {
  if (status === 'done' || status === 'passed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'cancelled' || status === 'running') return 'warn';
  return 'accent';
}

export function progressText(p) {
  if (!p || p.current === null || p.total === null) return '—';
  return p.current + ' / ' + p.total;
}

// Standard scope picker markup is repeated per panel; expose the
// default scope so every panel starts in the same place.
export const DEFAULT_SCOPE = { db: 'default', coll: 'ingest' };
