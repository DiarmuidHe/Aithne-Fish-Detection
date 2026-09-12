import * as React from 'react';
import { useSearchParams } from 'react-router';
import { useQuery } from '@tanstack/react-query';

import type { TrackSummary } from '@/api/types';
import { fetchTrackSummaries, fetchTrackThumbnails } from '@/api/videos';
import { Button } from '@/components/base/button';
import { Check, NumberInput, RangeSlider } from '@/components/base/controls';
import {
  ConfidenceMeter,
  EmptyState,
  PanelError,
  Skeleton,
  StatusPill,
} from '@/components/base/feedback';
import { CheckedMenu, PopoverPanel } from '@/components/base/popups';
import { MultiSelect } from '@/components/base/select';
import {
  DataTable,
  Scroller,
  SortHeader,
  SpacerRow,
  useVirtualRows,
} from '@/components/base/table';
import { MoreIcon } from '@/icons';
import { formatCount, formatSeconds, plural } from '@/lib/format';
import { REVIEW_LABELS, REVIEW_VALUES } from '@/features/library/filter-state';

import { IdentificationBadge, IdentifyFishButton } from '@/features/species/identify-fish';
import { SpeciesLabel } from '@/features/species/species-reference';
import { AssignSpeciesButton } from '@/features/species/species-selector';

import { ReviewToggle, useReviewApplier } from './review-controls';
import { TrackInspector } from './track-inspector';
import styles from './detail.module.css';

type TrackSortField =
  | 'first_frame'
  | 'duration'
  | 'detection_count'
  | 'mean_confidence'
  | 'max_confidence'
  | 'species'
  | 'review_state';

const COLUMNS: {
  key: string;
  label: string;
  sort?: TrackSortField;
  numeric?: boolean;
  optional?: boolean;
}[] = [
  // The fish itself, ahead of everything said about it. A row of figures cannot
  // tell an operator what they are looking at; forty pixels of the animal can.
  { key: 'preview', label: 'Fish', optional: true },
  { key: 'track', label: 'Track', sort: 'first_frame' },
  { key: 'result', label: 'VIAME / Result' },
  { key: 'review', label: 'Review decision', sort: 'review_state' },
  { key: 'detections', label: 'Detections', sort: 'detection_count', numeric: true },
  { key: 'first', label: 'First seen', numeric: true },
  { key: 'last', label: 'Last seen', numeric: true, optional: true },
  { key: 'confidence', label: 'Max confidence', sort: 'max_confidence', numeric: true },
  { key: 'mean', label: 'Mean confidence', sort: 'mean_confidence', numeric: true, optional: true },
  { key: 'species', label: 'Species', sort: 'species' },
  // The classifier's answer sits beside the detector's, never in place of it.
  { key: 'fishial', label: 'Fishial ID', optional: true },
];

export function TracksTab({ videoId, fps }: { videoId: string; fps: number | null }) {
  const [params, setParams] = useSearchParams();
  const [hidden, setHidden] = React.useState<string[]>(['last', 'mean']);

  const showExcluded = params.get('tall') === '1';
  const review = params.getAll('tr');
  const species = params.getAll('tsp');
  const minDetections = Number(params.get('tmd')) || null;
  const timeFrom = params.has('tfrom') ? Number(params.get('tfrom')) : null;
  const timeTo = params.has('tto') ? Number(params.get('tto')) : null;
  const confidence: [number, number] = [
    Number(params.get('tcmin') ?? 0),
    Number(params.get('tcmax') ?? 1),
  ];
  const sort = (params.get('tsort') ?? 'first_frame') as TrackSortField;
  const order = params.get('torder') === 'desc' ? 'desc' : 'asc';
  const openTrackId = params.get('track');

  const setParam = React.useCallback(
    (mutate: (next: URLSearchParams) => void) => {
      setParams((current) => {
        const next = new URLSearchParams(current);
        mutate(next);
        return next;
      });
    },
    [setParams],
  );

  const queryParams = React.useMemo(
    () => ({
      // The review vocabulary and `accepted_only` both apply, so any review
      // filter has to switch the default off to reach excluded tracks.
      accepted_only: !(showExcluded || review.length > 0),
      review,
      species,
      min_detections: minDetections ?? undefined,
      time_from: timeFrom ?? undefined,
      time_to: timeTo ?? undefined,
      min_confidence: confidence[0] > 0 ? confidence[0] : undefined,
      max_confidence: confidence[1] < 1 ? confidence[1] : undefined,
      sort,
      order,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showExcluded, review.join(','), species.join(','), minDetections, timeFrom, timeTo,
     confidence[0], confidence[1], sort, order],
  );

  const tracks = useQuery({
    queryKey: ['track-summaries', videoId, queryParams],
    queryFn: () => fetchTrackSummaries(videoId, queryParams),
    placeholderData: (previous) => previous,
    // A requested identification finishes in a background task, so the table has
    // to look again while any of its rows is waiting on one.
    refetchInterval: (query) =>
      query.state.data?.some((row) => row.fishial_state === 'submitted') ? 3_000 : false,
  });

  // One request per video, not per row: the server reads the source in a single
  // pass, so asking for all of them costs barely more than asking for one. A
  // failure here leaves blank cells and never blocks the table.
  const thumbnails = useQuery({
    queryKey: ['track-thumbnails', videoId],
    queryFn: () => fetchTrackThumbnails(videoId),
    staleTime: Infinity,
    retry: false,
  });
  const thumbnailUrls = React.useMemo(() => {
    const map = new Map<string, string>();
    for (const row of thumbnails.data ?? []) map.set(row.track_id, row.url);
    return map;
  }, [thumbnails.data]);

  const rows = tracks.data ?? [];
  const applier = useReviewApplier([
    ['track-summaries', videoId],
    ['videos'],
    ['video', videoId],
    ['summary', videoId],
    ['facets'],
  ]);

  const visibleColumns = COLUMNS.filter((column) => !hidden.includes(column.key));
  const virtual = useVirtualRows({ rows });

  const speciesOptions = React.useMemo(() => {
    const found = new Set<string>();
    for (const row of rows) if (row.species) found.add(row.species);
    for (const value of species) found.add(value);
    return [...found].sort().map((value) => ({ value, label: value }));
  }, [rows, species]);

  function onSort(field: string) {
    setParam((next) => {
      if (sort === field) next.set('torder', order === 'asc' ? 'desc' : 'asc');
      else {
        next.set('tsort', field);
        next.set('torder', 'asc');
      }
    });
  }

  return (
    <div className={styles.panelFlush}>
      <div className={styles.trackToolbar}>
        <Check
          checked={showExcluded}
          onCheckedChange={(checked) =>
            setParam((next) => (checked ? next.set('tall', '1') : next.delete('tall')))
          }
          label="Show low-confidence tracks and other excluded tracks"
        />

        <MultiSelect
          label="Review"
          options={REVIEW_VALUES.map((value) => ({ value, label: REVIEW_LABELS[value] }))}
          value={review}
          onChange={(value) =>
            setParam((next) => {
              next.delete('tr');
              for (const entry of value) next.append('tr', entry);
            })
          }
        />

        <MultiSelect
          label="Species"
          options={speciesOptions}
          value={species}
          onChange={(value) =>
            setParam((next) => {
              next.delete('tsp');
              for (const entry of value) next.append('tsp', entry);
            })
          }
          emptyMessage="No species has been recorded on these tracks."
        />

        <PopoverPanel
          title="Narrow the track table"
          trigger={<Button size="small">Confidence, detections, time</Button>}
        >
          <div style={{ display: 'grid', gap: 'var(--space-4)', minWidth: 260 }}>
            <RangeSlider
              label="Max confidence"
              value={confidence}
              onChange={([low, high]) =>
                setParam((next) => {
                  next.set('tcmin', low.toFixed(2));
                  next.set('tcmax', high.toFixed(2));
                })
              }
              format={(value) => `${Math.round(value * 100)}%`}
            />
            <NumberInput
              label="Minimum detections"
              min={0}
              value={minDetections}
              onChange={(value) =>
                setParam((next) =>
                  value === null ? next.delete('tmd') : next.set('tmd', String(value)),
                )
              }
            />
            <div style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'flex-end' }}>
              <NumberInput
                label="From (seconds)"
                min={0}
                value={timeFrom}
                onChange={(value) =>
                  setParam((next) =>
                    value === null ? next.delete('tfrom') : next.set('tfrom', String(value)),
                  )
                }
              />
              <NumberInput
                label="To (seconds)"
                min={0}
                value={timeTo}
                onChange={(value) =>
                  setParam((next) =>
                    value === null ? next.delete('tto') : next.set('tto', String(value)),
                  )
                }
              />
            </div>
            <p className="note" style={{ maxWidth: '32ch' }}>
              A track matches when the span between its first and last observation overlaps the
              window. Tracks without timestamps are excluded once a window is set.
            </p>
          </div>
        </PopoverPanel>

        <span style={{ marginLeft: 'auto', display: 'flex', gap: 'var(--space-2)' }}>
          <BulkReviewMenu rows={rows} applier={applier} />
          <CheckedMenu
            label="Columns"
            trigger={
              <Button variant="quiet" size="small" iconOnly aria-label="Choose columns">
                <MoreIcon />
              </Button>
            }
            items={COLUMNS.filter((column) => column.optional).map((column) => ({
              value: column.key,
              label: column.label,
            }))}
            checked={COLUMNS.filter(
              (column) => column.optional && !hidden.includes(column.key),
            ).map((column) => column.key)}
            onCheckedChange={(value, next) =>
              setHidden((current) =>
                next ? current.filter((key) => key !== value) : [...current, value],
              )
            }
          />
        </span>
      </div>

      <p className={styles.trackNote}>
        Accept or reject to override VIAME. Needs-review excludes a track; reviewed keeps the
        threshold decision. Inspect a track to view source footage and recorded observations.
      </p>

      {tracks.isError ? (
        <div className={styles.panel}>
          <PanelError
            message={tracks.error instanceof Error ? tracks.error.message : 'Tracks failed'}
            onRetry={() => void tracks.refetch()}
          />
        </div>
      ) : tracks.isLoading ? (
        <div className={styles.panel}>
          {Array.from({ length: 8 }, (_, index) => (
            <Skeleton key={index} height={20} />
          ))}
        </div>
      ) : rows.length === 0 ? (
        <EmptyState
          title="No tracks match this view"
          body={
            showExcluded || review.length > 0
              ? 'Clear the track filters to see every track this run produced.'
              : 'This run accepted no tracks. Turn on "Show low-confidence tracks" to see what VIAME found and excluded.'
          }
          action={
            <Button
              size="small"
              onClick={() =>
                setParam((next) => {
                  for (const key of ['tall', 'tr', 'tsp', 'tmd', 'tcmin', 'tcmax', 'tfrom',
                                     'tto']) {
                    next.delete(key);
                  }
                })
              }
            >
              Clear track filters
            </Button>
          }
        />
      ) : (
        <Scroller className={styles.trackTableWrap}>
          <div ref={virtual.containerRef}>
            <DataTable>
              <caption className="visually-hidden">
                {plural(rows.length, 'fish track')} for this video
              </caption>
              <thead>
                <tr>
                  {visibleColumns.map((column) =>
                    column.sort ? (
                      <SortHeader
                        key={column.key}
                        label={column.label}
                        field={column.sort}
                        active={sort}
                        order={order}
                        onSort={onSort}
                        numeric={column.numeric}
                      />
                    ) : (
                      <th key={column.key} scope="col">
                        {column.label}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                <SpacerRow height={virtual.padTop} columns={visibleColumns.length} />
                {virtual.visibleRows.map((track) => (
                  <TrackRow
                    key={track.id}
                    track={track}
                    columns={visibleColumns}
                    fps={fps}
                    thumbnail={thumbnailUrls.get(track.id)}
                    pendingThumbnail={thumbnails.isLoading}
                    onInspect={() => setParam((next) => next.set('track', track.id))}
                    onReview={(next) => applier.applyOne(track, next)}
                  />
                ))}
                <SpacerRow height={virtual.padBottom} columns={visibleColumns.length} />
              </tbody>
            </DataTable>
          </div>
        </Scroller>
      )}

      <TrackInspector
        trackId={openTrackId}
        fps={fps}
        onClose={() => setParam((next) => next.delete('track'))}
      />
    </div>
  );
}

function TrackRow({
  track,
  columns,
  fps,
  thumbnail,
  pendingThumbnail,
  onInspect,
  onReview,
}: {
  track: TrackSummary;
  columns: typeof COLUMNS;
  fps: number | null;
  thumbnail?: string;
  pendingThumbnail?: boolean;
  onInspect: () => void;
  onReview: (next: TrackSummary['review_state']) => void;
}) {
  const name = track.fishial_species ?? track.species ?? 'this fish';
  const cells: Record<string, React.ReactNode> = {
    preview: thumbnail ? (
      // Opens the same inspector as the track link: the picture is the natural
      // thing to click when you want a closer look.
      <button
        type="button"
        className={styles.previewButton}
        onClick={onInspect}
        aria-label={`Inspect ${name}, track ${track.viame_track_id}`}
      >
        <img src={thumbnail} alt={`Cropped frame of ${name}`} loading="lazy" decoding="async" />
      </button>
    ) : (
      <span className={styles.previewEmpty} data-pending={pendingThumbnail || undefined} />
    ),
    track: (
      <button type="button" className={styles.trackLink} onClick={onInspect}>
        #{track.viame_track_id}
      </button>
    ),
    result: (
      <span>
        <StatusPill status={track.accepted ? 'completed' : 'uploaded'}>
          {track.accepted ? 'Accepted' : 'Excluded'}
        </StatusPill>{' '}
        <span className={styles.trackCategories}>
          {track.review_categories
            .filter((category) => category === 'borderline' || category === 'disputed')
            .map((category) => (
              <span key={category} className={styles.categoryTag} data-category={category}>
                {category}
              </span>
            ))}
        </span>
      </span>
    ),
    review: (
      <ReviewToggle
        value={track.review_state}
        onChange={onReview}
        trackLabel={`track ${track.viame_track_id}`}
      />
    ),
    detections: formatCount(track.detection_count),
    first: formatSeconds(track.first_timestamp_seconds ?? frameSeconds(track.first_frame, fps), '—'),
    last: formatSeconds(track.last_timestamp_seconds ?? frameSeconds(track.last_frame, fps), '—'),
    confidence: (
      <ConfidenceMeter
        value={track.max_confidence}
        accepted={track.machine_accepted}
        label={`Max confidence for track ${track.viame_track_id}`}
      />
    ),
    mean: `${(track.mean_confidence * 100).toFixed(1)}%`,
    species: <SpeciesLabel value={track.species} />,
    fishial: (
      <span className={styles.fishialCell}>
        <IdentificationBadge
          identification={{
            state: track.fishial_state,
            species: track.fishial_species,
            confidence: track.fishial_species_confidence,
          }}
          manualSpecies={track.manual_species}
        />
        <IdentifyFishButton
          kind="track"
          trackId={track.id}
          state={track.fishial_state}
          variant="quiet"
          label="Identify"
        />
        <AssignSpeciesButton
          kind="track"
          trackId={track.id}
          current={track.manual_species}
          identification={{
            state: track.fishial_state,
            species: track.fishial_species,
            confidence: track.fishial_species_confidence,
          }}
          variant="quiet"
          label={track.manual_species ? 'Change name' : 'Name it'}
        />
      </span>
    ),
  };

  return (
    <tr>
      {columns.map((column) => (
        <td
          key={column.key}
          className={column.numeric && column.key !== 'confidence' ? 'num' : undefined}
          style={column.numeric && column.key !== 'confidence' ? { textAlign: 'right' } : undefined}
        >
          {cells[column.key]}
        </td>
      ))}
    </tr>
  );
}

function BulkReviewMenu({
  rows,
  applier,
}: {
  rows: TrackSummary[];
  applier: ReturnType<typeof useReviewApplier>;
}) {
  const [open, setOpen] = React.useState(false);
  return (
    <PopoverPanel
      title={`Apply to all ${rows.length} visible tracks`}
      align="end"
      trigger={
        <Button size="small" disabled={rows.length === 0} onClick={() => setOpen(!open)}>
          Review all visible
        </Button>
      }
    >
      <p className="note" style={{ maxWidth: '32ch', marginBottom: 'var(--space-3)' }}>
        Applies one decision to every track the filters currently show. This can be undone for
        eight seconds afterwards.
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
        <Button size="small" onClick={() => applier.applyMany(rows, 'accepted')}>
          Accept all
        </Button>
        <Button size="small" onClick={() => applier.applyMany(rows, 'reviewed')}>
          Mark reviewed
        </Button>
        <Button size="small" onClick={() => applier.applyMany(rows, 'needs-review')}>
          Flag all
        </Button>
      </div>
    </PopoverPanel>
  );
}

function frameSeconds(frame: number, fps: number | null): number | null {
  return fps ? frame / fps : null;
}
