import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';

import { fetchSystemStatus } from '@/api/system';
import { fetchFacets, fetchLibraryAnalytics, fetchVideos } from '@/api/videos';
import { LinkButton } from '@/components/base/button';
import { BarList, ChartSkeleton } from '@/components/charts/bar-chart';
import { EmptyState, PanelError, Skeleton, StatusPill } from '@/components/base/feedback';
import { Page, Section } from '@/components/layout/shell';
import { useConditionalPoll } from '@/hooks/use-poll';
import { formatCount, formatDateTime, plural } from '@/lib/format';

import styles from './overview.module.css';

export function OverviewRoute() {
  const system = useQuery({ queryKey: ['system-status'], queryFn: fetchSystemStatus });
  const facets = useQuery({ queryKey: ['facets'], queryFn: fetchFacets });
  const analytics = useQuery({ queryKey: ['library-analytics'], queryFn: fetchLibraryAnalytics });

  const queueBusy =
    (system.data?.queue.queued ?? 0) + (system.data?.queue.processing ?? 0) > 0;
  const interval = useConditionalPoll(queueBusy);

  const recent = useQuery({
    queryKey: ['videos', 'recent'],
    queryFn: () => fetchVideos({ sort: 'created_at', order: 'desc', limit: 8 }),
    refetchInterval: interval,
  });

  const awaiting = facets.data?.review_status.awaiting ?? 0;
  const inProgress = facets.data?.review_status['in-progress'] ?? 0;
  const failed = facets.data?.status.failed ?? 0;

  return (
    <Page
      title="Overview"
      subtitle="Worker health, what is in the queue, and what is still waiting for a human decision."
    >
      <Section title="System">
        {system.isLoading ? (
          <Skeleton height={72} />
        ) : system.isError || !system.data ? (
          <PanelError
            title="System status is unavailable"
            message={system.error instanceof Error ? system.error.message : 'Request failed'}
            onRetry={() => void system.refetch()}
          />
        ) : (
          <div className={styles.system} aria-live="polite">
            <div className={styles.systemMain}>
              <div className={styles.systemLine}>
                <StatusPill status={system.data.ready ? 'completed' : 'failed'}>
                  {system.data.ready ? 'Worker ready' : 'No worker'}
                </StatusPill>
                <span className={styles.systemMode}>
                  {system.data.processing_mode === 'gpu'
                    ? 'Real VIAME GPU processing'
                    : 'Mock processing'}
                </span>
              </div>
              <p className="note">
                {system.data.worker.message}. {system.data.database.message}.
              </p>
              {system.data.ready ? null : (
                <p className="note">
                  Start or restart the stack with:{' '}
                  <code>{system.data.compose_command}</code>
                </p>
              )}
            </div>

            <dl className={styles.queue}>
              <QueueFigure label="Queued" value={system.data.queue.queued} />
              <QueueFigure label="Processing" value={system.data.queue.processing} />
              <QueueFigure
                label="Failed jobs"
                value={system.data.queue.failed}
                tone={system.data.queue.failed > 0 ? 'failed' : undefined}
              />
              <QueueFigure label="Workers" value={system.data.worker.active_workers} />
            </dl>
          </div>
        )}
      </Section>

      <Section
        title="Review debt"
        action={
          <LinkButton size="small" to="/review">
            Open the review queue
          </LinkButton>
        }
      >
        {facets.isLoading ? (
          <Skeleton height={60} />
        ) : (
          <div className={styles.debt}>
            <DebtFigure
              value={awaiting}
              label="videos not yet touched"
              to="/library?review_status=awaiting&sort=review_status&order=desc"
            />
            <DebtFigure
              value={inProgress}
              label="videos part-reviewed"
              to="/library?review_status=in-progress"
            />
            <DebtFigure value={failed} label="failed runs" to="/library?status=failed" />
            <DebtFigure
              value={facets.data?.total ?? 0}
              label="videos in the library"
              to="/library"
            />
          </div>
        )}
        <p className="note">
          A video counts as reviewed once every track carries a decision and none is flagged.
          Tracks are observations, not unique biological fish.
        </p>
      </Section>

      <Section
        title="Recent uploads"
        action={
          <LinkButton size="small" variant="link" to="/library">
            All videos
          </LinkButton>
        }
      >
        {recent.isLoading ? (
          <Skeleton height={120} />
        ) : recent.data && recent.data.rows.length === 0 ? (
          <EmptyState
            title="Nothing uploaded yet"
            body="Upload an underwater video from the library to start."
            action={
              <LinkButton size="small" variant="primary" to="/library">
                Go to the library
              </LinkButton>
            }
          />
        ) : (
          <ul className={styles.recent}>
            {(recent.data?.rows ?? []).map((video) => (
              <li key={video.id} className={styles.recentRow}>
                <Link className={styles.recentLink} to={`/library/${video.id}`}>
                  <span className="truncate">{video.original_filename}</span>
                </Link>
                <span className="note">{formatDateTime(video.created_at)}</span>
                <span className="note">
                  {video.processing_status === 'completed'
                    ? plural(video.accepted_track_count, 'fish')
                    : '—'}
                </span>
                <StatusPill status={video.processing_status} />
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Across the library">
        {analytics.isLoading ? (
          <ChartSkeleton title="Accepted fish per video" />
        ) : analytics.isError || !analytics.data ? (
          <PanelError
            message={analytics.error instanceof Error ? analytics.error.message : 'Request failed'}
            onRetry={() => void analytics.refetch()}
          />
        ) : (
          <div className={styles.charts}>
            <BarList
              title="Accepted fish per video"
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
            <BarList
              title="Accepted detections per video"
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
        )}
      </Section>
    </Page>
  );
}

function QueueFigure({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: 'failed';
}) {
  return (
    <div className={styles.queueFigure}>
      <dt className={styles.queueLabel}>{label}</dt>
      <dd className={styles.queueValue} data-tone={tone}>
        {formatCount(value)}
      </dd>
    </div>
  );
}

function DebtFigure({ value, label, to }: { value: number; label: string; to: string }) {
  return (
    <Link className={styles.debtFigure} to={to}>
      <span className={styles.debtValue}>{formatCount(value)}</span>
      <span className={styles.debtLabel}>{label}</span>
    </Link>
  );
}
