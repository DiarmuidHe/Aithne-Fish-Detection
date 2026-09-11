import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';

import { fetchSystemStatus } from '@/api/system';
import { fetchFacets, fetchLibraryAnalytics, fetchVideos } from '@/api/videos';
import { LinkButton } from '@/components/base/button';
import { BarList, ChartSkeleton } from '@/components/charts/bar-chart';
import {
  EmptyState,
  PanelError,
  Skeleton,
  SkeletonRows,
  StatusPill,
} from '@/components/base/feedback';
import {
  AppBarSlot,
  Band,
  MetricCard,
  MetricPlaceholder,
  MetricRow,
  Page,
} from '@/components/layout/shell';
import { useConditionalPoll } from '@/hooks/use-poll';
import { formatCount, formatDateTime, plural } from '@/lib/format';

import styles from './overview.module.css';

/*
 * The Overview answers one question: what is happening across the monitoring
 * operation right now.
 *
 * It is composed as four surfaces of four different materials, not as a grid of
 * cards. A dark console strip carries the figures and butts against the
 * application bar, so the marine spine wraps the top of the workspace. Anything
 * needing a person sits directly on the page under a hairline. The working half
 * — what arrived, and how far review has got — is one bordered workspace split
 * by a rule. The long view of the library sits in a sunken band at the bottom.
 * Nothing decorative gets a card, and nothing repeats the material above it.
 */

export function OverviewRoute() {
  const system = useQuery({ queryKey: ['system-status'], queryFn: fetchSystemStatus });
  const facets = useQuery({ queryKey: ['facets'], queryFn: fetchFacets });
  const analytics = useQuery({ queryKey: ['library-analytics'], queryFn: fetchLibraryAnalytics });

  const queueBusy = (system.data?.queue.queued ?? 0) + (system.data?.queue.processing ?? 0) > 0;
  const interval = useConditionalPoll(queueBusy);

  const recent = useQuery({
    queryKey: ['videos', 'recent'],
    queryFn: () => fetchVideos({ sort: 'created_at', order: 'desc', limit: 8 }),
    refetchInterval: interval,
  });

  const awaiting = facets.data?.review_status.awaiting ?? 0;
  const inProgress = facets.data?.review_status['in-progress'] ?? 0;
  const reviewed = facets.data?.review_status.complete ?? 0;
  const failed = facets.data?.status.failed ?? 0;
  const total = facets.data?.total ?? 0;
  const inFlight = (system.data?.queue.queued ?? 0) + (system.data?.queue.processing ?? 0);

  const acceptedFish = (analytics.data ?? []).reduce(
    (sum, row) => sum + row.accepted_fish_count,
    0,
  );

  return (
    <Page>
      <AppBarSlot area="actions">
        <LinkButton size="small" variant="primary" to="/review">
          Open review queue
        </LinkButton>
      </AppBarSlot>

      {/* --- The console strip. One surface, five figures, a station line. --- */}
      {facets.isLoading || system.isLoading ? (
        <MetricRow tone="console">
          {Array.from({ length: 5 }, (_, index) => (
            <MetricPlaceholder key={index} />
          ))}
        </MetricRow>
      ) : (
        <MetricRow
          tone="console"
          aside={
            <span className={styles.station}>
              <span className={styles.stationName}>All monitoring sites</span>
              <span className={styles.stationMeta}>
                {system.data?.processing_mode === 'gpu' ? 'VIAME GPU' : 'Mock detections'} ·{' '}
                {plural(system.data?.worker.active_workers ?? 0, 'worker')}
              </span>
            </span>
          }
        >
          <MetricCard
            label="Recordings"
            value={formatCount(total)}
            meta={`${formatCount(reviewed)} fully reviewed`}
            to="/library"
          />
          <MetricCard
            label="Confirmed fish"
            value={formatCount(acceptedFish)}
            meta="Accepted across the library"
          />
          <MetricCard
            label="Awaiting review"
            value={formatCount(awaiting)}
            meta="Not yet touched"
            tone={awaiting > 0 ? 'flagged' : undefined}
            to="/library?review=unreviewed&sort=unreviewed&order=desc"
          />
          <MetricCard
            label="Part reviewed"
            value={formatCount(inProgress)}
            meta="A decision is still outstanding"
            to="/library?sort=review_status&order=desc"
          />
          <MetricCard
            label="In flight"
            value={formatCount(inFlight)}
            meta={inFlight > 0 ? 'Queued or processing' : 'Nothing queued'}
            tone={inFlight > 0 ? 'accent' : undefined}
            to="/settings"
          />
        </MetricRow>
      )}

      {/*
        * --- Attention. Rules on the page itself, not another box. Only
        * rendered when there is something a person has to act on.
        */}
      {failed > 0 || awaiting > 0 ? (
        <div className={styles.attention}>
          <span className={styles.attentionLabel}>Needs attention</span>
          {failed > 0 ? (
            <Link className={styles.attentionRow} to="/library?status=failed" data-tone="failed">
              <StatusPill status="failed" />
              <span className={styles.attentionText}>
                {plural(failed, 'run')} failed and produced no tracks.
              </span>
              <span className={styles.attentionGo}>Review runs</span>
            </Link>
          ) : null}
          {awaiting > 0 ? (
            <Link className={styles.attentionRow} to="/review" data-tone="flagged">
              <StatusPill status="awaiting" />
              <span className={styles.attentionText}>
                {plural(awaiting, 'recording')} {awaiting === 1 ? 'has' : 'have'} tracks nobody
                has decided on.
              </span>
              <span className={styles.attentionGo}>Open queue</span>
            </Link>
          ) : null}
        </div>
      ) : null}

      {/*
        * --- The working half. What arrived and how far review has got belong
        * together, so they are one surface with a rule down the middle rather
        * than two panels with a gap between them.
        */}
      <div className={styles.workspace}>
        <div className={styles.workspaceMain}>
          <header className={styles.regionHead}>
            <div className={styles.regionHeading}>
              <h2 className={styles.regionTitle}>Recent recordings</h2>
              <p className={styles.regionNote}>Newest first</p>
            </div>
            <LinkButton size="small" variant="quiet" to="/library">
              All recordings
            </LinkButton>
          </header>

          {recent.isLoading ? (
            <SkeletonRows rows={6} height={40} lines={2} />
          ) : recent.isError ? (
            <div className={styles.regionPad}>
              <PanelError
                message={recent.error instanceof Error ? recent.error.message : 'Request failed'}
                onRetry={() => void recent.refetch()}
              />
            </div>
          ) : recent.data && recent.data.rows.length === 0 ? (
            <EmptyState
              title="Nothing uploaded yet"
              body="Upload an underwater recording to start. Uploading stores the file; you choose when processing runs."
              action={
                <LinkButton size="small" variant="primary" to="/library">
                  Go to recordings
                </LinkButton>
              }
            />
          ) : (
            <ul className={styles.recent}>
              {(recent.data?.rows ?? []).map((video, index) => (
                <li key={video.id} style={{ ['--i' as string]: index }}>
                  <Link className={styles.recentRow} to={`/library/${video.id}`}>
                    <span className={styles.recentName}>{video.original_filename}</span>
                    <span className={styles.recentWhen}>{formatDateTime(video.created_at)}</span>
                    <span className={styles.recentCount}>
                      {video.processing_status === 'completed'
                        ? plural(video.accepted_track_count, 'fish')
                        : '—'}
                    </span>
                    <StatusPill status={video.processing_status} />
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>

        <aside className={styles.workspaceSide}>
          <header className={styles.regionHead}>
            <div className={styles.regionHeading}>
              <h2 className={styles.regionTitle}>Review completion</h2>
              <p className={styles.regionNote}>Recordings with every track decided</p>
            </div>
          </header>
          <div className={styles.regionPad}>
            {facets.isLoading ? (
              <Skeleton height={120} />
            ) : (
              <ReviewProgress
                reviewed={reviewed}
                inProgress={inProgress}
                awaiting={awaiting}
                total={total}
              />
            )}
          </div>
        </aside>
      </div>

      {/* --- The long view. A sunken band, two lists, one rule between them. --- */}
      <Band
        tone="sunken"
        label="Across the library"
        note="Tracks are observations, not unique biological fish."
      >
        {analytics.isLoading ? (
          <div className={styles.charts}>
            <div className={styles.chart}>
              <ChartSkeleton title="Accepted fish per recording" />
            </div>
            <div className={styles.chart}>
              <ChartSkeleton title="Accepted detections per recording" />
            </div>
          </div>
        ) : analytics.isError || !analytics.data ? (
          <PanelError
            message={analytics.error instanceof Error ? analytics.error.message : 'Request failed'}
            onRetry={() => void analytics.refetch()}
          />
        ) : (
          <div className={styles.charts}>
            <div className={styles.chart}>
              <BarList
                title="Accepted fish per recording"
                note="Includes review decisions"
                valueLabel="fish"
                emptyMessage="No completed runs yet."
                rows={[...analytics.data]
                  .sort((left, right) => right.accepted_fish_count - left.accepted_fish_count)
                  .map((row) => ({
                    label: row.filename,
                    value: row.accepted_fish_count,
                    to: `/library/${row.video_id}`,
                  }))}
              />
            </div>
            <div className={styles.chart}>
              <BarList
                title="Accepted detections per recording"
                note="Observations, not unique fish"
                valueLabel="detections"
                emptyMessage="No completed runs yet."
                rows={[...analytics.data]
                  .sort((left, right) => right.accepted_detections - left.accepted_detections)
                  .map((row) => ({
                    label: row.filename,
                    value: row.accepted_detections,
                    to: `/library/${row.video_id}`,
                  }))}
              />
            </div>
          </div>
        )}
      </Band>
    </Page>
  );
}

/**
 * The backlog as one bar in three parts, plus the numbers underneath. A ring
 * would say the same thing and be harder to read off.
 */
function ReviewProgress({
  reviewed,
  inProgress,
  awaiting,
  total,
}: {
  reviewed: number;
  inProgress: number;
  awaiting: number;
  total: number;
}) {
  const denominator = Math.max(1, total);
  const parts = [
    { key: 'complete', label: 'Reviewed', value: reviewed, to: '/library?review=done' },
    {
      key: 'in-progress',
      label: 'Part reviewed',
      value: inProgress,
      to: '/library?sort=review_status&order=desc',
    },
    { key: 'awaiting', label: 'Awaiting', value: awaiting, to: '/library?review=unreviewed' },
  ];

  return (
    <div className={styles.completion}>
      <p className={styles.completionHead}>
        <span className={styles.completionValue}>
          {total === 0 ? '—' : `${Math.round((reviewed / denominator) * 100)}%`}
        </span>
        <span className="note">of {formatCount(total)} recordings fully reviewed</span>
      </p>

      <div
        className={styles.completionBar}
        role="img"
        aria-label={`${reviewed} reviewed, ${inProgress} part reviewed, ${awaiting} awaiting, of ${total} recordings`}
      >
        {parts.map((part) => (
          <span
            key={part.key}
            className={styles.completionSegment}
            data-part={part.key}
            style={{ width: `${(part.value / denominator) * 100}%` }}
          />
        ))}
      </div>

      <dl className={styles.completionKey}>
        {parts.map((part) => (
          <div className={styles.completionKeyItem} key={part.key}>
            <dt>
              <Link to={part.to} className={styles.completionKeyLink}>
                <span className={styles.completionSwatch} data-part={part.key} aria-hidden="true" />
                {part.label}
              </Link>
            </dt>
            <dd className={styles.completionKeyValue}>{formatCount(part.value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
