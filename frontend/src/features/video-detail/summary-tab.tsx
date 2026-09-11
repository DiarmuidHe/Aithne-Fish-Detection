import { useQuery } from '@tanstack/react-query';

import type { Video } from '@/api/types';
import { EXPORT_KINDS, fetchSummary, videoExportUrl } from '@/api/videos';
import { AnchorButton } from '@/components/base/button';
import { PanelError, Skeleton } from '@/components/base/feedback';
import { Hint } from '@/components/base/popups';
import { basename, formatCount, formatPercent, formatSeconds } from '@/lib/format';

import styles from './detail.module.css';

export function SummaryTab({ video }: { video: Video }) {
  const summary = useQuery({
    queryKey: ['summary', video.id],
    queryFn: () => fetchSummary(video.id),
    enabled: video.processing_status === 'completed',
  });

  if (video.processing_status !== 'completed') {
    return (
      <div className={styles.panel}>
        <p className="note">
          Fish results appear here once processing completes. Nothing has been counted for this
          video yet.
        </p>
      </div>
    );
  }

  if (summary.isLoading) {
    return (
      <div className={styles.panel}>
        <div className={styles.figures}>
          {Array.from({ length: 5 }, (_, index) => (
            <div className={styles.figure} key={index}>
              <Skeleton width={70} height={28} />
              <Skeleton width={100} height={11} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (summary.isError || !summary.data) {
    return (
      <div className={styles.panel}>
        <PanelError
          message={summary.error instanceof Error ? summary.error.message : 'Summary failed'}
          onRetry={() => void summary.refetch()}
        />
      </div>
    );
  }

  const data = summary.data;
  const job = video.latest_job;

  return (
    <div className={styles.panel}>
      <div className={styles.figures}>
        <Figure value={formatCount(data.fish_tracks)} label="Accepted fish tracks" />
        <Figure value={formatCount(data.total_detections)} label="Accepted detections" />
        <Figure value={formatSeconds(data.first_fish_timestamp_seconds)} label="First fish" />
        <Figure value={formatSeconds(data.last_fish_timestamp_seconds)} label="Last fish" />
        <Figure
          value={formatPercent(data.mean_track_confidence, 1)}
          label="Mean track confidence"
        />
        <Figure
          value={data.confidence_threshold.toFixed(2)}
          label="Confidence threshold"
        />
      </div>

      <p className="note">
        Tracks are observations, not unique biological fish. Accepted totals include review
        decisions.
      </p>

      <dl className={styles.provenance}>
        <Provenance term="Processing pipeline" value={basename(data.pipeline_name)} full={data.pipeline_name} />
        <Provenance
          term="Model"
          value={[data.model_name, data.model_version].filter(Boolean).join(' · ') || 'Unknown'}
        />
        <Provenance term="VIAME version" value={data.viame_version ?? 'Not reported'} />
        <Provenance term="Processing mode" value={job?.worker_mode ?? 'Unknown'} />
        <Provenance
          term="Run command SHA-256"
          value={job?.configuration.run_command_sha256?.slice(0, 12) ?? 'Not recorded'}
          full={job?.configuration.run_command_sha256 ?? undefined}
        />
        <Provenance
          term="Content SHA-256"
          value={video.content_sha256?.slice(0, 12) ?? 'Not recorded'}
          full={video.content_sha256 ?? undefined}
        />
      </dl>

      <div>
        <p className="note" style={{ marginBottom: 'var(--space-2)' }}>
          Exports carry the full provenance of the run that produced them.
        </p>
        <div className={styles.exportRow}>
          {EXPORT_KINDS.map(({ kind, label }) => (
            <AnchorButton key={kind} size="small" href={videoExportUrl(video.id, kind)}>
              {label} CSV
            </AnchorButton>
          ))}
        </div>
      </div>
    </div>
  );
}

function Figure({ value, label }: { value: string; label: string }) {
  // "Not detected" is a fact, not a figure, so it does not take display size.
  const isNumber = /\d/.test(value);
  return (
    <div className={styles.figure}>
      <span className={styles.figureValue} data-absent={isNumber ? undefined : ''}>
        {value}
      </span>
      <span className={styles.figureLabel}>{label}</span>
    </div>
  );
}

function Provenance({ term, value, full }: { term: string; value: string; full?: string }) {
  const body = <dd className={styles.provenanceValue}>{value}</dd>;
  return (
    <div>
      <dt className={styles.provenanceTerm}>{term}</dt>
      {full ? <Hint content={full}>{body}</Hint> : body}
    </div>
  );
}
