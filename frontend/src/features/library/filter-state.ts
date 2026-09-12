/**
 * The library filter state, and its round trip through the URL.
 *
 * The URL is the state: everything an operator sets here is serialisable, so a
 * shared link reproduces the screen and the back button steps through filter
 * changes. Values equal to the default are left out, so a clean library has a
 * clean URL.
 */

import type { ProcessingStatus, ReviewCategory } from '@/api/types';
import type { QueryValue } from '@/api/client';

export const PAGE_SIZE = 50;

export const NO_CAMERA = '__none__';

export const STATUS_VALUES: ProcessingStatus[] = [
  'uploaded',
  'queued',
  'processing',
  'completed',
  'failed',
];

export const STATUS_LABELS: Record<string, string> = {
  uploaded: 'Uploaded',
  queued: 'Queued',
  processing: 'Processing',
  completed: 'Completed',
  failed: 'Failed',
};

export const REVIEW_VALUES: ReviewCategory[] = [
  'flagged',
  'unreviewed',
  'borderline',
  'disputed',
  'done',
];

export const REVIEW_LABELS: Record<ReviewCategory, string> = {
  flagged: 'Needs review',
  unreviewed: 'Unreviewed',
  borderline: 'Borderline',
  disputed: 'Disputed',
  done: 'Fully reviewed',
};

export const REVIEW_STATUS_LABELS: Record<string, string> = {
  'n/a': 'Not processed',
  awaiting: 'Awaiting review',
  'in-progress': 'Review in progress',
  complete: 'Review complete',
};

export const SORT_VALUES = [
  'created_at',
  'filename',
  'camera_id',
  'status',
  'duration',
  'size',
  'accepted_fish',
  'detections',
  'unreviewed',
  'flagged',
  'review_status',
  'annotated_at',
] as const;

export type SortField = (typeof SORT_VALUES)[number];

export const SORT_LABELS: Record<SortField, string> = {
  created_at: 'Uploaded',
  filename: 'Filename',
  camera_id: 'Camera',
  status: 'Status',
  duration: 'Duration',
  size: 'File size',
  accepted_fish: 'Accepted fish',
  detections: 'Detections',
  unreviewed: 'Unreviewed tracks',
  flagged: 'Flagged tracks',
  review_status: 'Review status',
  annotated_at: 'Annotated',
};

export const DATE_PRESETS = ['any', 'today', '7d', '30d', 'custom'] as const;
export type DatePreset = (typeof DATE_PRESETS)[number];

export const DATE_LABELS: Record<DatePreset, string> = {
  any: 'Any date',
  today: 'Today',
  '7d': 'Last 7 days',
  '30d': 'Last 30 days',
  custom: 'Custom range',
};

export const HAS_VALUES = ['annotated', 'clips', 'detections'] as const;
export type HasFacet = (typeof HAS_VALUES)[number];

export const HAS_LABELS: Record<HasFacet, string> = {
  annotated: 'Annotated video',
  clips: 'Fish clips',
  detections: 'Detections > 0',
};

export interface LibraryFilters {
  q: string;
  status: ProcessingStatus[];
  review: ReviewCategory[];
  camera: string[];
  species: string[];
  date: DatePreset;
  from: string | null;
  to: string | null;
  has: HasFacet[];
  minFish: number | null;
  maxFish: number | null;
  sort: SortField;
  order: 'asc' | 'desc';
  /** 1-based, so the URL reads the way an operator would say it. */
  page: number;
}

export const DEFAULT_FILTERS: LibraryFilters = {
  q: '',
  status: [],
  review: [],
  camera: [],
  species: [],
  date: 'any',
  from: null,
  to: null,
  has: [],
  minFish: null,
  maxFish: null,
  sort: 'created_at',
  order: 'desc',
  page: 1,
};

/* --- Reducer -------------------------------------------------------------- */

export type FilterAction =
  | { type: 'patch'; patch: Partial<LibraryFilters> }
  | { type: 'toggle'; key: 'status' | 'review' | 'camera' | 'species' | 'has'; value: string }
  | { type: 'remove'; key: keyof LibraryFilters; value?: string }
  | { type: 'clear' };

/** Changing what is being filtered always returns to the first page. */
const PAGE_KEEPING_KEYS = new Set<keyof LibraryFilters>(['page']);

export function filterReducer(state: LibraryFilters, action: FilterAction): LibraryFilters {
  switch (action.type) {
    case 'patch': {
      const keys = Object.keys(action.patch) as (keyof LibraryFilters)[];
      const onlyPaging = keys.every((key) => PAGE_KEEPING_KEYS.has(key));
      const next = { ...state, ...action.patch };
      return onlyPaging ? next : { ...next, page: 1 };
    }
    case 'toggle': {
      const current = state[action.key] as string[];
      const next = current.includes(action.value)
        ? current.filter((entry) => entry !== action.value)
        : [...current, action.value];
      return { ...state, [action.key]: next, page: 1 } as LibraryFilters;
    }
    case 'remove': {
      const current = state[action.key];
      if (Array.isArray(current) && action.value !== undefined) {
        return {
          ...state,
          [action.key]: current.filter((entry) => entry !== action.value),
          page: 1,
        } as LibraryFilters;
      }
      return {
        ...state,
        [action.key]: DEFAULT_FILTERS[action.key],
        // Clearing a custom range must also drop its endpoints.
        ...(action.key === 'date' ? { from: null, to: null } : {}),
        page: 1,
      } as LibraryFilters;
    }
    case 'clear':
      // Sort survives a clear: it is how the operator reads the list, not what
      // they are looking for.
      return { ...DEFAULT_FILTERS, sort: state.sort, order: state.order };
  }
}

/* --- URL round trip ------------------------------------------------------- */

function readList(params: URLSearchParams, key: string, allowed?: readonly string[]): string[] {
  const values = params
    .getAll(key)
    .flatMap((value) => value.split(','))
    .map((value) => value.trim())
    .filter(Boolean);
  const unique = [...new Set(values)];
  return allowed ? unique.filter((value) => allowed.includes(value)) : unique;
}

function readNumber(params: URLSearchParams, key: string): number | null {
  const raw = params.get(key);
  if (raw === null || raw.trim() === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : null;
}

function readEnum<T extends string>(
  params: URLSearchParams,
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const raw = params.get(key);
  return allowed.includes(raw as T) ? (raw as T) : fallback;
}

export function parseFilters(params: URLSearchParams): LibraryFilters {
  const page = readNumber(params, 'page');
  return {
    q: params.get('q') ?? '',
    status: readList(params, 'status', STATUS_VALUES) as ProcessingStatus[],
    review: readList(params, 'review', REVIEW_VALUES) as ReviewCategory[],
    camera: readList(params, 'camera'),
    species: readList(params, 'species'),
    date: readEnum(params, 'date', DATE_PRESETS, 'any'),
    from: params.get('from') || null,
    to: params.get('to') || null,
    has: readList(params, 'has', HAS_VALUES) as HasFacet[],
    minFish: readNumber(params, 'minFish'),
    maxFish: readNumber(params, 'maxFish'),
    sort: readEnum(params, 'sort', SORT_VALUES, 'created_at'),
    order: readEnum(params, 'order', ['asc', 'desc'] as const, 'desc'),
    page: page && page >= 1 ? page : 1,
  };
}

/**
 * Serialise into `base`, leaving keys we do not own (the open tab, the open
 * inspector) alone so a filter change does not close a dialog.
 */
export function toSearchParams(filters: LibraryFilters, base?: URLSearchParams): URLSearchParams {
  const params = new URLSearchParams(base ? base.toString() : undefined);
  const OWNED = [
    'q', 'status', 'review', 'camera', 'species', 'date', 'from', 'to', 'has',
    'minFish', 'maxFish', 'sort', 'order', 'page',
  ];
  for (const key of OWNED) params.delete(key);

  if (filters.q.trim()) params.set('q', filters.q.trim());
  for (const value of filters.status) params.append('status', value);
  for (const value of filters.review) params.append('review', value);
  for (const value of filters.camera) params.append('camera', value);
  for (const value of filters.species) params.append('species', value);
  for (const value of filters.has) params.append('has', value);
  if (filters.date !== 'any') params.set('date', filters.date);
  if (filters.date === 'custom') {
    if (filters.from) params.set('from', filters.from);
    if (filters.to) params.set('to', filters.to);
  }
  if (filters.minFish !== null) params.set('minFish', String(filters.minFish));
  if (filters.maxFish !== null) params.set('maxFish', String(filters.maxFish));
  if (filters.sort !== DEFAULT_FILTERS.sort) params.set('sort', filters.sort);
  if (filters.order !== DEFAULT_FILTERS.order) params.set('order', filters.order);
  if (filters.page > 1) params.set('page', String(filters.page));
  return params;
}

/* --- Query building ------------------------------------------------------- */

function presetStart(preset: DatePreset, now: Date): string | undefined {
  if (preset === 'today') {
    const start = new Date(now);
    start.setHours(0, 0, 0, 0);
    return start.toISOString();
  }
  if (preset === '7d' || preset === '30d') {
    const days = preset === '7d' ? 7 : 30;
    return new Date(now.getTime() - days * 86_400_000).toISOString();
  }
  return undefined;
}

/** The exact query the API is asked for, including the band the server uses. */
export function toQueryParams(
  filters: LibraryFilters,
  now: Date = new Date(),
): Record<string, QueryValue | QueryValue[]> {
  const has = new Set(filters.has);
  // "Detections > 0" and an explicit floor are both floors; the higher wins.
  const minFish = Math.max(filters.minFish ?? 0, has.has('detections') ? 1 : 0) || undefined;
  const createdAfter =
    filters.date === 'custom'
      ? filters.from
        ? new Date(`${filters.from}T00:00:00`).toISOString()
        : undefined
      : presetStart(filters.date, now);
  const createdBefore =
    filters.date === 'custom' && filters.to
      ? new Date(`${filters.to}T23:59:59.999`).toISOString()
      : undefined;

  return {
    q: filters.q.trim() || undefined,
    status: filters.status,
    review: filters.review,
    camera_id: filters.camera,
    species: filters.species,
    annotated: has.has('annotated') ? true : undefined,
    has_clips: has.has('clips') ? true : undefined,
    min_fish: minFish,
    max_fish: filters.maxFish ?? undefined,
    created_after: createdAfter,
    created_before: createdBefore,
    sort: filters.sort,
    order: filters.order,
    limit: PAGE_SIZE,
    offset: (filters.page - 1) * PAGE_SIZE,
  };
}

/* --- Applied chips -------------------------------------------------------- */

export interface FilterChip {
  key: keyof LibraryFilters;
  value?: string;
  label: string;
}

export function describeChips(filters: LibraryFilters): FilterChip[] {
  const chips: FilterChip[] = [];
  if (filters.q.trim()) chips.push({ key: 'q', label: `Search: ${filters.q.trim()}` });
  for (const value of filters.status) {
    chips.push({ key: 'status', value, label: `Status: ${STATUS_LABELS[value] ?? value}` });
  }
  for (const value of filters.review) {
    chips.push({ key: 'review', value, label: `Review: ${REVIEW_LABELS[value]}` });
  }
  for (const value of filters.camera) {
    chips.push({
      key: 'camera',
      value,
      label: value === NO_CAMERA ? 'Camera: none' : `Camera: ${value}`,
    });
  }
  for (const value of filters.species) {
    chips.push({ key: 'species', value, label: `Species: ${value}` });
  }
  for (const value of filters.has) {
    chips.push({ key: 'has', value, label: `Has: ${HAS_LABELS[value]}` });
  }
  if (filters.date !== 'any') {
    const label =
      filters.date === 'custom'
        ? `Date: ${filters.from ?? 'any'} to ${filters.to ?? 'any'}`
        : `Date: ${DATE_LABELS[filters.date]}`;
    chips.push({ key: 'date', label });
  }
  if (filters.minFish !== null) {
    chips.push({ key: 'minFish', label: `Fish ≥ ${filters.minFish}` });
  }
  if (filters.maxFish !== null) {
    chips.push({ key: 'maxFish', label: `Fish ≤ ${filters.maxFish}` });
  }
  return chips;
}

export function isFiltered(filters: LibraryFilters): boolean {
  return describeChips(filters).length > 0;
}
