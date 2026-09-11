/** Shared formatting. Numbers are the product here, so they are formatted once. */

const dateTime = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
});

const dateOnly = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' });

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return 'Unknown time';
  return dateTime.format(new Date(value));
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return 'Unknown date';
  return dateOnly.format(new Date(value));
}

export function formatClock(value: string | null | undefined): string {
  if (!value) return '—';
  return new Date(value).toLocaleTimeString();
}

/**
 * m:ss.s, the way an operator reads a position in a video.
 *
 * The default for a missing value reads as a sentence, which suits a summary
 * figure; a dense table cell should pass a short marker instead.
 */
export function formatSeconds(
  value: number | null | undefined,
  missing = 'Not detected',
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return missing;
  const total = Math.max(0, value);
  const minutes = Math.floor(total / 60);
  const seconds = total - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`;
}

export function formatDuration(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value < 60) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value - minutes * 60);
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat().format(value);
}

/** Model names and pipelines arrive as paths; only the leaf is meaningful. */
export function basename(value: string | null | undefined): string {
  if (!value) return '—';
  return value.replace(/\\/g, '/').split('/').pop() || value;
}

export function shortId(value: string): string {
  return value.slice(0, 8);
}

// Nouns that are their own plural. "Fish" is the one this product is about, so
// it is handled here rather than at every call site that counts them.
const INVARIANT_NOUNS = new Set(['fish', 'footage', 'species']);

function pluralise(noun: string): string {
  const words = noun.split(' ');
  // "fish" stays "fish"; "fish track" still becomes "fish tracks".
  return INVARIANT_NOUNS.has(words[words.length - 1]) ? noun : `${noun}s`;
}

/** "3 videos" / "1 video", so no sentence has to say "(s)". */
export function plural(count: number, singular: string, pluralForm?: string): string {
  return `${formatCount(count)} ${count === 1 ? singular : (pluralForm ?? pluralise(singular))}`;
}
