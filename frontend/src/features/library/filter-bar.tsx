import * as React from 'react';

import type { VideoFacets } from '@/api/types';
import { Button } from '@/components/base/button';
import { NumberInput, SearchInput, Segmented } from '@/components/base/controls';
import { MultiCombobox, MultiSelect, SingleSelect } from '@/components/base/select';
import { PopoverPanel, popupStyles } from '@/components/base/popups';
import { CloseIcon, SortAscIcon, SortDescIcon } from '@/icons';
import { useDebouncedValue } from '@/hooks/use-debounced';

import {
  DATE_LABELS,
  DATE_PRESETS,
  HAS_LABELS,
  HAS_VALUES,
  NO_CAMERA,
  REVIEW_LABELS,
  REVIEW_VALUES,
  SORT_LABELS,
  SORT_VALUES,
  STATUS_LABELS,
  STATUS_VALUES,
  describeChips,
  type DatePreset,
  type FilterAction,
  type HasFacet,
  type LibraryFilters,
  type SortField,
} from './filter-state';

import styles from './library.module.css';
import toggles from '@/components/base/toggle.module.css';

export function FilterBar({
  filters,
  dispatch,
  facets,
}: {
  filters: LibraryFilters;
  dispatch: (action: FilterAction) => void;
  facets?: VideoFacets;
}) {
  const commitSearch = React.useCallback(
    (value: string) => dispatch({ type: 'patch', patch: { q: value } }),
    [dispatch],
  );
  const [draft, setDraft] = useDebouncedValue(filters.q, commitSearch);

  const statusOptions = STATUS_VALUES.map((value) => ({
    value,
    label: STATUS_LABELS[value],
    count: facets?.status[value],
  }));

  const reviewOptions = REVIEW_VALUES.map((value) => ({
    value,
    label: REVIEW_LABELS[value],
  }));

  const cameraOptions = [
    { value: NO_CAMERA, label: 'No camera' },
    ...(facets?.cameras ?? []).map((value) => ({ value, label: value })),
  ];

  const speciesOptions = (facets?.species ?? []).map((value) => ({ value, label: value }));

  return (
    <div className={styles.filterBar}>
      <SearchInput
        label="Search filenames and cameras"
        placeholder="Filename or camera"
        value={draft}
        onChange={setDraft}
      />

      <MultiSelect
        label="Status"
        options={statusOptions}
        value={filters.status}
        onChange={(value) =>
          dispatch({ type: 'patch', patch: { status: value as LibraryFilters['status'] } })
        }
      />

      <MultiSelect
        label="Review"
        options={reviewOptions}
        value={filters.review}
        onChange={(value) =>
          dispatch({ type: 'patch', patch: { review: value as LibraryFilters['review'] } })
        }
      />

      <MultiCombobox
        label="Camera"
        options={cameraOptions}
        value={filters.camera}
        onChange={(value) => dispatch({ type: 'patch', patch: { camera: value } })}
        emptyMessage="No camera matches that."
      />

      <MultiCombobox
        label="Species"
        options={speciesOptions}
        value={filters.species}
        onChange={(value) => dispatch({ type: 'patch', patch: { species: value } })}
        emptyMessage="No species has been recorded on an accepted track yet."
      />

      <DateFilter filters={filters} dispatch={dispatch} />

      <Segmented
        label="Has"
        multiple
        value={filters.has}
        onChange={(value) =>
          dispatch({ type: 'patch', patch: { has: value as HasFacet[] } })
        }
        options={HAS_VALUES.map((value) => ({ value, label: HAS_LABELS[value] }))}
      />

      <FishCountFilter filters={filters} dispatch={dispatch} />

      <div className={styles.sortGroup}>
        <SingleSelect
          label="Sort"
          options={SORT_VALUES.map((value) => ({ value, label: SORT_LABELS[value] }))}
          value={filters.sort}
          onChange={(value) =>
            dispatch({ type: 'patch', patch: { sort: value as SortField } })
          }
        />
        <Button
          iconOnly
          aria-label={filters.order === 'asc' ? 'Sort descending' : 'Sort ascending'}
          title={filters.order === 'asc' ? 'Ascending' : 'Descending'}
          onClick={() =>
            dispatch({
              type: 'patch',
              patch: { order: filters.order === 'asc' ? 'desc' : 'asc' },
            })
          }
        >
          {filters.order === 'asc' ? <SortAscIcon /> : <SortDescIcon />}
        </Button>
      </div>
    </div>
  );
}

function DateFilter({
  filters,
  dispatch,
}: {
  filters: LibraryFilters;
  dispatch: (action: FilterAction) => void;
}) {
  const presets = DATE_PRESETS.filter((preset) => preset !== 'custom');
  return (
    <div className={styles.filterBar}>
      <Segmented
        label="Upload date"
        value={filters.date === 'custom' ? [] : [filters.date]}
        onChange={(value) =>
          dispatch({
            type: 'patch',
            patch: { date: (value[0] as DatePreset) ?? 'any', from: null, to: null },
          })
        }
        options={presets.map((preset) => ({ value: preset, label: DATE_LABELS[preset] }))}
      />
      <PopoverPanel
        title="Custom upload range"
        trigger={
          <Button aria-label="Custom date range">
            {filters.date === 'custom'
              ? `${filters.from ?? 'any'} → ${filters.to ?? 'any'}`
              : 'Custom'}
          </Button>
        }
      >
        <div style={{ display: 'grid', gap: 'var(--space-3)', minWidth: 240 }}>
          <label style={{ display: 'grid', gap: 'var(--space-1)' }}>
            <span className="note">From</span>
            <input
              type="date"
              value={filters.from ?? ''}
              onChange={(event) =>
                dispatch({
                  type: 'patch',
                  patch: { date: 'custom', from: event.target.value || null },
                })
              }
            />
          </label>
          <label style={{ display: 'grid', gap: 'var(--space-1)' }}>
            <span className="note">To</span>
            <input
              type="date"
              value={filters.to ?? ''}
              onChange={(event) =>
                dispatch({
                  type: 'patch',
                  patch: { date: 'custom', to: event.target.value || null },
                })
              }
            />
          </label>
        </div>
      </PopoverPanel>
    </div>
  );
}

function FishCountFilter({
  filters,
  dispatch,
}: {
  filters: LibraryFilters;
  dispatch: (action: FilterAction) => void;
}) {
  const set = filters.minFish !== null || filters.maxFish !== null;
  return (
    <PopoverPanel
      title="Accepted fish per video"
      trigger={
        <Button>
          Fish count
          {set ? ` · ${filters.minFish ?? 0}–${filters.maxFish ?? '∞'}` : ''}
        </Button>
      }
    >
      <div style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'flex-end' }}>
        <NumberInput
          label="At least"
          min={0}
          value={filters.minFish}
          onChange={(value) => dispatch({ type: 'patch', patch: { minFish: value } })}
        />
        <NumberInput
          label="At most"
          min={0}
          value={filters.maxFish}
          onChange={(value) => dispatch({ type: 'patch', patch: { maxFish: value } })}
        />
      </div>
      <p className="note" style={{ marginTop: 'var(--space-2)', maxWidth: '34ch' }}>
        Counts accepted tracks, which include review decisions. Tracks are
        observations, not unique biological fish.
      </p>
    </PopoverPanel>
  );
}

/** Applied filters, each removable, with a clear-all once more than one is set. */
export function FilterChips({
  filters,
  dispatch,
}: {
  filters: LibraryFilters;
  dispatch: (action: FilterAction) => void;
}) {
  const chips = describeChips(filters);
  if (chips.length === 0) return null;
  return (
    <div className={toggles.chipRow}>
      {chips.map((chip) => (
        <button
          key={`${chip.key}:${chip.value ?? ''}`}
          type="button"
          className={toggles.chip}
          onClick={() => dispatch({ type: 'remove', key: chip.key, value: chip.value })}
        >
          {chip.label}
          <span className={toggles.chipRemove} aria-hidden="true">
            <CloseIcon width={12} height={12} />
          </span>
          <span className="visually-hidden">Remove this filter</span>
        </button>
      ))}
      {chips.length > 1 ? (
        <Button variant="link" size="small" onClick={() => dispatch({ type: 'clear' })}>
          Clear all
        </Button>
      ) : null}
    </div>
  );
}

export { popupStyles };
