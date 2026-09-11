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
 * distinguish the hues loses nothing.
 */
export function StatusPill({ status, children }: { status: string; children?: React.ReactNode }) {
  return (
    <span className={styles.status} data-status={status}>
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
}: {
  label: string;
  value: number | null;
  max?: number;
  valueLabel?: string;
}) {
  return (
    <Progress.Root className={styles.progressRoot} value={value} max={max}>
      <div className={styles.progressHeader}>
        <Progress.Label>{label}</Progress.Label>
        <span className={styles.progressValue}>{valueLabel ?? `${value ?? 0}/${max}`}</span>
      </div>
      <Progress.Track className={styles.progressTrack}>
        <Progress.Indicator className={styles.progressIndicator} />
      </Progress.Track>
    </Progress.Root>
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
