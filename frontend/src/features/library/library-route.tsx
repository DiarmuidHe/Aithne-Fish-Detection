import * as React from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import type { VideoListRow } from '@/api/types';
import {
  annotateVideo,
  fetchFacets,
  fetchVideos,
  startProcessing,
  videoExportUrl,
} from '@/api/videos';
import { Button } from '@/components/base/button';
import { Check } from '@/components/base/controls';
import { PanelError } from '@/components/base/feedback';
import { useToast } from '@/components/base/toast';
import { AppBarSlot } from '@/components/layout/shell';
import { UploadIcon } from '@/icons';
import { useFilters } from '@/hooks/use-filters';
import { useSelection } from '@/hooks/use-selection';
import { useConditionalPoll } from '@/hooks/use-poll';
import { formatCount } from '@/lib/format';
import { VideoDetail } from '@/features/video-detail/video-detail';

import { BulkBar } from './bulk-bar';
import { FilterBar, FilterChips } from './filter-bar';
import { PAGE_SIZE, isFiltered, toQueryParams } from './filter-state';
import { SavedViewsBar } from './saved-views-bar';
import { UploadDialog } from './upload-dialog';
import { LibraryEmptyState, VideoList } from './video-list';
import styles from './library.module.css';

const MIN_RAIL = 320;
const MAX_RAIL = 900;

export function LibraryRoute() {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const queryClient = useQueryClient();
  const toast = useToast();
  const { filters, dispatch, replace, query } = useFilters();
  const [uploadOpen, setUploadOpen] = React.useState(false);
  const [railWidth, setRailWidth] = React.useState(() => readRailWidth());

  const queryParams = React.useMemo(() => toQueryParams(filters), [filters]);

  const videos = useQuery({
    queryKey: ['videos', queryParams],
    queryFn: () => fetchVideos(queryParams),
    placeholderData: (previous) => previous,
  });

  const facets = useQuery({ queryKey: ['facets'], queryFn: fetchFacets, staleTime: 30_000 });

  const rows = videos.data?.rows ?? [];
  const anyActive = rows.some((row) =>
    ['queued', 'processing'].includes(row.processing_status),
  );
  const interval = useConditionalPoll(anyActive);

  React.useEffect(() => {
    if (interval === false) return;
    const timer = window.setInterval(() => {
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
    }, interval);
    return () => window.clearInterval(timer);
  }, [interval, queryClient]);

  const selection = useSelection(rows.map((row) => row.id));

  const process = useMutation({
    mutationFn: (video: VideoListRow) => startProcessing(video.id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
      toast.add({ title: 'Processing job queued', type: 'success' });
    },
    onError: (error) =>
      toast.add({ title: 'Could not queue processing', description: message(error), type: 'error' }),
  });

  const annotate = useMutation({
    mutationFn: (video: VideoListRow) => annotateVideo(video.id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
      toast.add({ title: 'Annotated video is ready', type: 'success' });
    },
    onError: (error) =>
      toast.add({ title: 'Annotation failed', description: message(error), type: 'error' }),
  });

  const filtered = videos.data?.filtered ?? 0;
  const total = videos.data?.total ?? 0;
  const firstRow = filtered === 0 ? 0 : (filters.page - 1) * PAGE_SIZE + 1;
  const lastRow = Math.min(filtered, filters.page * PAGE_SIZE);
  const pageCount = Math.max(1, Math.ceil(filtered / PAGE_SIZE));

  return (
    <div
      className={styles.layout}
      style={{ ['--rail' as string]: `${railWidth}px` }}
      data-detail-open={videoId ? '' : undefined}
    >
      {/* The bar carries the page identity; the pane below carries the work. */}
      <AppBarSlot area="context">
        <span className="num">{formatCount(filtered)}</span>
        <span>of</span>
        <span className="num">{formatCount(total)}</span>
        <span>recordings</span>
      </AppBarSlot>
      <AppBarSlot area="actions">
        <Button variant="primary" size="small" onClick={() => setUploadOpen(true)}>
          <UploadIcon />
          Upload
        </Button>
      </AppBarSlot>

      <div className={styles.listPane}>
        <div className={styles.listHeader}>
          <SavedViewsBar query={query} onApply={replace} />
          <FilterBar filters={filters} dispatch={dispatch} facets={facets.data} />
          <FilterChips filters={filters} dispatch={dispatch} />

          <div className={styles.resultLine}>
            <span className={styles.resultText}>
              <span className={styles.resultCount}>{formatCount(filtered)}</span>
              {filtered === 1 ? ' recording' : ' recordings'}
              {videos.isFetching ? (
                <span className={styles.updating}>updating</span>
              ) : null}
            </span>
            {rows.length > 0 ? (
              <Check
                checked={selection.allVisibleSelected}
                indeterminate={selection.someVisibleSelected}
                onCheckedChange={(checked) =>
                  checked
                    ? selection.selectAllFiltered(
                        rows.map((row) => row.id),
                        filtered,
                      )
                    : selection.clear()
                }
                label={
                  filtered > rows.length
                    ? `Select these ${rows.length}`
                    : `Select all ${formatCount(filtered)}`
                }
              />
            ) : null}
          </div>
        </div>

        {videos.isError ? (
          <div style={{ padding: 'var(--space-4)' }}>
            <PanelError
              title="The library did not load"
              message={message(videos.error)}
              onRetry={() => void videos.refetch()}
            />
          </div>
        ) : (
          <VideoList
            rows={rows}
            loading={videos.isLoading}
            selection={selection}
            currentVideoId={videoId}
            linkFor={(video) => `/library/${video.id}?${params.toString()}`}
            actions={{
              onProcess: (video) => process.mutate(video),
              onAnnotate: (video) => annotate.mutate(video),
              onExport: (video) =>
                window.location.assign(videoExportUrl(video.id, 'summary')),
            }}
            empty={
              <LibraryEmptyState
                filtered={filtered}
                total={total}
                reviewFilterSet={filters.review.length > 0}
                onClear={() => dispatch({ type: 'clear' })}
              />
            }
          />
        )}

        {pageCount > 1 ? (
          <div className={styles.pager}>
            <span className={styles.pagerRange}>
              {formatCount(firstRow)}–{formatCount(lastRow)} of {formatCount(filtered)}
            </span>
            <span className={styles.pagerControls}>
              <Button
                size="small"
                disabled={filters.page <= 1}
                onClick={() => dispatch({ type: 'patch', patch: { page: filters.page - 1 } })}
              >
                Previous
              </Button>
              <span className={styles.pagerRange}>
                Page {filters.page} of {pageCount}
              </span>
              <Button
                size="small"
                disabled={filters.page >= pageCount}
                onClick={() => dispatch({ type: 'patch', patch: { page: filters.page + 1 } })}
              >
                Next
              </Button>
            </span>
          </div>
        ) : null}

        <BulkBar
          selection={selection}
          onDone={() => {
            void queryClient.invalidateQueries({ queryKey: ['videos'] });
            void queryClient.invalidateQueries({ queryKey: ['facets'] });
          }}
        />
      </div>

      <div
        className={styles.resizer}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the video list"
        // A focusable separator is a window splitter, and must report its value.
        aria-valuenow={Math.round(railWidth)}
        aria-valuemin={MIN_RAIL}
        aria-valuemax={MAX_RAIL}
        aria-valuetext={`${Math.round(railWidth)} pixels`}
        tabIndex={0}
        onPointerDown={(event) => {
          const start = event.clientX;
          const startWidth = railWidth;
          const element = event.currentTarget;
          element.setPointerCapture(event.pointerId);
          element.dataset.dragging = '';
          const move = (moveEvent: PointerEvent) => {
            const next = Math.min(MAX_RAIL, Math.max(MIN_RAIL, startWidth + moveEvent.clientX - start));
            setRailWidth(next);
          };
          const up = () => {
            delete element.dataset.dragging;
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
            storeRailWidth();
          };
          window.addEventListener('pointermove', move);
          window.addEventListener('pointerup', up);
        }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft') setRailWidth((width) => Math.max(MIN_RAIL, width - 16));
          if (event.key === 'ArrowRight') setRailWidth((width) => Math.min(MAX_RAIL, width + 16));
        }}
      />

      <div className={styles.detailPane}>
        <VideoDetail
          videoId={videoId}
          onClose={() => navigate(`/library?${params.toString()}`)}
          filtersApplied={isFiltered(filters)}
        />
      </div>

      <UploadDialog
        open={uploadOpen}
        onOpenChange={setUploadOpen}
        onUploaded={(id) => navigate(`/library/${id}?${params.toString()}`)}
      />
    </div>
  );

  function storeRailWidth() {
    try {
      window.localStorage.setItem('library-rail-width', String(railWidth));
    } catch {
      // The width is a convenience; losing it costs nothing.
    }
  }
}

function readRailWidth(): number {
  try {
    const raw = Number(window.localStorage.getItem('library-rail-width'));
    if (Number.isFinite(raw) && raw >= MIN_RAIL) return raw;
  } catch {
    // Fall through to the default.
  }
  return 440;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Request failed';
}
