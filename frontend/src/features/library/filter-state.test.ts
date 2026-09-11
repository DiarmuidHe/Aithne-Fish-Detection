import { describe, expect, it } from 'vitest';

import {
  DEFAULT_FILTERS,
  PAGE_SIZE,
  describeChips,
  filterReducer,
  isFiltered,
  parseFilters,
  toQueryParams,
  toSearchParams,
  type LibraryFilters,
} from './filter-state';

function filters(patch: Partial<LibraryFilters> = {}): LibraryFilters {
  return { ...DEFAULT_FILTERS, ...patch };
}

describe('the filter reducer', () => {
  it('returns to the first page whenever what is being filtered changes', () => {
    const state = filters({ page: 4 });

    expect(filterReducer(state, { type: 'patch', patch: { q: 'reef' } }).page).toBe(1);
    expect(filterReducer(state, { type: 'toggle', key: 'status', value: 'failed' }).page).toBe(1);
    expect(filterReducer(state, { type: 'remove', key: 'q' }).page).toBe(1);
  });

  it('keeps the page when only the page changes', () => {
    const next = filterReducer(filters({ q: 'reef', page: 2 }), {
      type: 'patch',
      patch: { page: 3 },
    });

    expect(next.page).toBe(3);
    expect(next.q).toBe('reef');
  });

  it('toggles a value in and out of a multi-select', () => {
    const on = filterReducer(filters(), { type: 'toggle', key: 'review', value: 'flagged' });
    const off = filterReducer(on, { type: 'toggle', key: 'review', value: 'flagged' });

    expect(on.review).toEqual(['flagged']);
    expect(off.review).toEqual([]);
  });

  it('removes one value from a list without disturbing the others', () => {
    const state = filters({ status: ['failed', 'completed'] });

    const next = filterReducer(state, { type: 'remove', key: 'status', value: 'failed' });

    expect(next.status).toEqual(['completed']);
  });

  it('drops both endpoints when a custom date range is removed', () => {
    const state = filters({ date: 'custom', from: '2026-01-01', to: '2026-02-01' });

    const next = filterReducer(state, { type: 'remove', key: 'date' });

    expect(next).toMatchObject({ date: 'any', from: null, to: null });
  });

  it('clears every filter but keeps how the list is sorted', () => {
    const state = filters({
      q: 'reef',
      review: ['flagged'],
      camera: ['cam-1'],
      sort: 'accepted_fish',
      order: 'asc',
    });

    const next = filterReducer(state, { type: 'clear' });

    expect(isFiltered(next)).toBe(false);
    expect(next.sort).toBe('accepted_fish');
    expect(next.order).toBe('asc');
  });
});

describe('the URL round trip', () => {
  it('restores every field it serialised', () => {
    const original = filters({
      q: 'river cam',
      status: ['failed', 'completed'],
      review: ['flagged', 'borderline'],
      camera: ['cam-1', '__none__'],
      species: ['Gadus morhua'],
      date: 'custom',
      from: '2026-01-01',
      to: '2026-02-01',
      has: ['annotated', 'clips'],
      minFish: 2,
      maxFish: 40,
      sort: 'accepted_fish',
      order: 'asc',
      page: 3,
    });

    const restored = parseFilters(toSearchParams(original));

    expect(restored).toEqual(original);
  });

  it('leaves defaults out of the URL', () => {
    expect(toSearchParams(DEFAULT_FILTERS).toString()).toBe('');
  });

  it('keeps query keys it does not own, so a filter change does not close a dialog', () => {
    const base = new URLSearchParams('tab=tracks&track=abc123&q=stale');

    const params = toSearchParams(filters({ q: 'fresh' }), base);

    expect(params.get('tab')).toBe('tracks');
    expect(params.get('track')).toBe('abc123');
    expect(params.getAll('q')).toEqual(['fresh']);
  });

  it('ignores values outside the vocabulary rather than filtering on nonsense', () => {
    const restored = parseFilters(
      new URLSearchParams('status=archived&status=failed&review=maybe&sort=colour&order=sideways'),
    );

    expect(restored.status).toEqual(['failed']);
    expect(restored.review).toEqual([]);
    expect(restored.sort).toBe('created_at');
    expect(restored.order).toBe('desc');
  });

  it('treats a nonsense page as the first page', () => {
    expect(parseFilters(new URLSearchParams('page=0')).page).toBe(1);
    expect(parseFilters(new URLSearchParams('page=-4')).page).toBe(1);
    expect(parseFilters(new URLSearchParams('page=zero')).page).toBe(1);
  });

  it('accepts a comma-separated list, which is how people hand-edit a URL', () => {
    expect(parseFilters(new URLSearchParams('review=flagged,disputed')).review).toEqual([
      'flagged',
      'disputed',
    ]);
  });
});

describe('the query sent to the API', () => {
  it('turns the page number into a limit and an offset', () => {
    expect(toQueryParams(filters({ page: 3 }))).toMatchObject({
      limit: PAGE_SIZE,
      offset: 2 * PAGE_SIZE,
    });
  });

  it('sends "detections > 0" as a floor of one accepted fish', () => {
    expect(toQueryParams(filters({ has: ['detections'] })).min_fish).toBe(1);
  });

  it('keeps the higher floor when an explicit minimum is also set', () => {
    expect(toQueryParams(filters({ has: ['detections'], minFish: 5 })).min_fish).toBe(5);
    expect(toQueryParams(filters({ has: ['detections'], minFish: 0 })).min_fish).toBe(1);
  });

  it('turns the "has" toggles into the flags the API understands', () => {
    const params = toQueryParams(filters({ has: ['annotated', 'clips'] }));

    expect(params.annotated).toBe(true);
    expect(params.has_clips).toBe(true);
  });

  it('leaves untouched filters out entirely rather than sending empty values', () => {
    const params = toQueryParams(DEFAULT_FILTERS);

    expect(params.q).toBeUndefined();
    expect(params.annotated).toBeUndefined();
    expect(params.created_after).toBeUndefined();
    expect(params.min_fish).toBeUndefined();
  });

  it('resolves a date preset against the given moment', () => {
    const now = new Date('2026-03-15T12:00:00Z');

    const week = toQueryParams(filters({ date: '7d' }), now).created_after as string;

    expect(new Date(week).toISOString()).toBe('2026-03-08T12:00:00.000Z');
  });

  it('sends a custom range as a whole first day through a whole last day', () => {
    const params = toQueryParams(filters({ date: 'custom', from: '2026-01-02', to: '2026-01-03' }));

    expect(new Date(params.created_after as string).getHours()).toBe(0);
    expect(new Date(params.created_before as string).getHours()).toBe(23);
  });
});

describe('the applied filter chips', () => {
  it('names every set filter, one removable chip per value', () => {
    const chips = describeChips(
      filters({ q: 'reef', status: ['failed'], camera: ['__none__', 'cam-2'] }),
    );

    expect(chips.map((chip) => chip.label)).toEqual([
      'Search: reef',
      'Status: Failed',
      'Camera: none',
      'Camera: cam-2',
    ]);
    expect(chips[2]).toMatchObject({ key: 'camera', value: '__none__' });
  });

  it('shows nothing when nothing is filtered', () => {
    expect(describeChips(DEFAULT_FILTERS)).toEqual([]);
    expect(isFiltered(DEFAULT_FILTERS)).toBe(false);
  });

  it('does not count sort as a filter', () => {
    expect(isFiltered(filters({ sort: 'flagged', order: 'asc' }))).toBe(false);
  });
});
