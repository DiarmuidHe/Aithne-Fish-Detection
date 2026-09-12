import * as React from 'react';
import { Meter } from '@base-ui/react/meter';
import { Progress } from '@base-ui/react/progress';
import { Separator as BaseSeparator } from '@base-ui/react/separator';
import { Toolbar } from '@base-ui/react/toolbar';

import { AlertIcon } from '@/icons';
import { Button } from './button';

import styles from './feedback.module.css';

export { Toolbar };

const STATUS_WORDS: Record<string, string> = {
  uploaded: 'Uploaded',
  queued: 'Queued',
  processing: 'Processing',
  completed: 'Completed',
  failed: 'Failed',
  awaiting: 'Awaiting review',
  'in-progress': 'In review',
  complete: 'Reviewed',
  'n/a': 'Not processed',
  flagged: 'Needs review',
};

/**
 * A status is a word first. The colour repeats it, so an operator who cannot
 * distinguish the hues loses nothing. Work in progress gets a travelling
 * hairline under the word rather than a spinner beside it.
 */
export function StatusPill({ status, children }: { status: string; children?: React.ReactNode }) {
  const active = status === 'processing' || status === 'queued';
  return (
    <span className={styles.status} data-status={status} data-active={active || undefined}>
      <span className={styles.statusDot} aria-hidden="true" />
      {children ?? STATUS_WORDS[status] ?? status}
    </span>
  );
}

export function ProgressBar({
  label,
  value,
  max = 100,
  valueLabel,
  size = 'default',
}: {
  label: string;
  value: number | null;
  max?: number;
  valueLabel?: string;
  size?: 'default' | 'small';
}) {
  return (
    <Progress.Root className={styles.progressRoot} data-size={size} value={value} max={max}>
      <div className={styles.progressHeader}>
        <Progress.Label className={styles.progressLabel}>{label}</Progress.Label>
        <span className={styles.progressValue}>{valueLabel ?? `${value ?? 0}/${max}`}</span>
      </div>
      <Progress.Track className={styles.progressTrack}>
        <Progress.Indicator className={styles.progressIndicator} />
      </Progress.Track>
    </Progress.Root>
  );
}

/**
 * Work is happening and the server reports no percentage. A travelling band,
 * not a spinner: it sits on the edge of the region it belongs to, so it reads
 * as "this panel is busy" rather than "the application is busy".
 */
export function LoadingBar({ label = 'Loading' }: { label?: string }) {
  return (
    <div className={styles.loadingBar} role="progressbar" aria-label={label} aria-busy="true">
      <span className={styles.loadingBarTrack}>
        <span className={styles.loadingBarIndicator} />
      </span>
    </div>
  );
}

/** Confidence in a table cell: a bar plus the number, never the bar alone. */
export function ConfidenceMeter({
  value,
  accepted,
  label,
}: {
  value: number;
  accepted: boolean;
  label: string;
}) {
  return (
    <Meter.Root
      className={`${styles.meterRoot} ${accepted ? styles.meterAccepted : ''}`}
      value={value * 100}
      max={100}
      aria-label={label}
    >
      <Meter.Track className={styles.meterTrack}>
        <Meter.Indicator className={styles.meterIndicator} />
      </Meter.Track>
      <span className={styles.meterValue}>{(value * 100).toFixed(1)}%</span>
    </Meter.Root>
  );
}

export function Separator({ orientation = 'horizontal' }: { orientation?: 'horizontal' | 'vertical' }) {
  return <BaseSeparator className={styles.separator} orientation={orientation} />;
}

export function Skeleton({
  width,
  height = 14,
  radius,
}: {
  width?: number | string;
  height?: number | string;
  radius?: number;
}) {
  return (
    <span
      className={styles.skeleton}
      style={{ display: 'block', width: width ?? '100%', height, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}

/**
 * A skeleton shaped like the rows it stands in for, so the page does not jump
 * when the data lands. Widths vary a little down the list; identical bars read
 * as a broken render rather than as loading.
 */
export function SkeletonRows({
  rows = 6,
  height = 36,
  lines = 1,
}: {
  rows?: number;
  height?: number;
  lines?: 1 | 2;
}) {
  const widths = ['72%', '54%', '65%', '48%', '78%', '58%'];
  return (
    <div className={styles.skeletonRows} aria-hidden="true">
      {Array.from({ length: rows }, (_, index) => (
        <div
          className={styles.skeletonRow}
          key={index}
          style={{ minHeight: height, animationDelay: `${index * 45}ms` }}
        >
          <Skeleton width={widths[index % widths.length]} height={11} />
          {lines === 2 ? <Skeleton width={`${30 + ((index * 7) % 25)}%`} height={9} /> : null}
        </div>
      ))}
    </div>
  );
}

/** Empty states name the next action; they never say "No data available". */
export function EmptyState({
  title,
  body,
  action,
}: {
  title: string;
  body?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className={styles.empty}>
      <p className={styles.emptyTitle}>{title}</p>
      {body ? <p className={styles.emptyBody}>{body}</p> : null}
      {action}
    </div>
  );
}

/** A failure inside one panel stays inside that panel, with a way back. */
export function PanelError({
  title = 'This did not load',
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className={styles.panelError} role="alert">
      <p className={styles.panelErrorTitle}>
        <AlertIcon />
        {title}
      </p>
      <p className="note">{message}</p>
      {onRetry ? (
        <Button size="small" onClick={onRetry}>
          Try again
        </Button>
      ) : null}
    </div>
  );
}
