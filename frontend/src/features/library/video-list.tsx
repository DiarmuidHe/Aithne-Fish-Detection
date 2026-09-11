import { Link } from 'react-router';

import type { VideoListRow } from '@/api/types';
import { Check } from '@/components/base/controls';
import { EmptyState, Skeleton, StatusPill } from '@/components/base/feedback';
import { ActionMenu, ContextMenu, Hint, popupStyles } from '@/components/base/popups';
import { Scroller } from '@/components/base/table';
import { Button } from '@/components/base/button';
import { MoreIcon } from '@/icons';
import { formatBytes, formatDateTime, formatDuration, plural } from '@/lib/format';
import type { Selection } from '@/hooks/use-selection';

import { REVIEW_STATUS_LABELS } from './filter-state';
import styles from './library.module.css';

export interface RowActions {
  onProcess: (video: VideoListRow) => void;
  onAnnotate: (video: VideoListRow) => void;
  onExport: (video: VideoListRow) => void;
}

export function VideoList({
  rows,
  loading,
  selection,
  currentVideoId,
  linkFor,
  actions,
  empty,
}: {
  rows: VideoListRow[];
  loading: boolean;
  selection: Selection;
  currentVideoId?: string;
  linkFor: (video: VideoListRow) => string;
  actions: RowActions;
  empty: React.ReactNode;
}) {
  if (loading) {
    return (
      <div className={styles.list}>
        {Array.from({ length: 8 }, (_, index) => (
          <div className={styles.skeletonRow} key={index}>
            <Skeleton width="60%" height={13} />
            <Skeleton width="40%" height={11} />
          </div>
        ))}
      </div>
    );
  }

  if (rows.length === 0) return <div className={styles.list}>{empty}</div>;

  return (
    <Scroller className={styles.list}>
      <ul>
        {rows.map((video) => (
          <li key={video.id}>
            <VideoRow
              video={video}
              selected={selection.has(video.id)}
              current={video.id === currentVideoId}
              onToggle={() => selection.toggle(video.id)}
              to={linkFor(video)}
              actions={actions}
            />
          </li>
        ))}
      </ul>
    </Scroller>
  );
}

function VideoRow({
  video,
  selected,
  current,
  onToggle,
  to,
  actions,
}: {
  video: VideoListRow;
  selected: boolean;
  current: boolean;
  onToggle: () => void;
  to: string;
  actions: RowActions;
}) {
  const menuItems = [
    {
      label: video.processing_status === 'completed' ? 'Process again' : 'Start processing',
      onSelect: () => actions.onProcess(video),
      disabled: ['queued', 'processing'].includes(video.processing_status),
    },
    {
      label: video.has_annotation ? 'Regenerate annotated video' : 'Generate annotated video',
      onSelect: () => actions.onAnnotate(video),
      disabled: video.processing_status !== 'completed',
    },
    { label: 'Export summary CSV', onSelect: () => actions.onExport(video) },
  ];

  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger
        className={styles.rowWrap}
        data-selected={selected || undefined}
        data-current={current || undefined}
      >
        <span className={styles.rowCheck}>
          <Check
            checked={selected}
            onCheckedChange={onToggle}
            hideLabel
            label={`Select ${video.original_filename}`}
          />
        </span>

        <Link className={styles.rowLink} to={to}>
          <span className={styles.rowTop}>
            <Hint content={video.original_filename}>
              <span className={styles.rowName}>{video.original_filename}</span>
            </Hint>
            <StatusPill status={video.processing_status} />
          </span>
          <span className={styles.rowMeta}>
            <span>{formatDateTime(video.created_at)}</span>
            <span>·</span>
            <span>{video.camera_id ?? 'No camera'}</span>
            <span>·</span>
            <span className={styles.rowStat}>{formatBytes(video.size_bytes)}</span>
            {video.duration_seconds ? (
              <>
                <span>·</span>
                <span className={styles.rowStat}>{formatDuration(video.duration_seconds)}</span>
              </>
            ) : null}
          </span>
          {video.processing_status === 'completed' ? (
            <span className={styles.rowMeta}>
              <span className={styles.rowStat}>
                {plural(video.accepted_track_count, 'fish')}
              </span>
              <span>·</span>
              <span className={styles.rowStat}>
                {plural(video.detection_count, 'detection')}
              </span>
              {video.flagged_count > 0 ? (
                <>
                  <span>·</span>
                  <span className={`${styles.rowStat} ${styles.rowFlag}`}>
                    {video.flagged_count} flagged
                  </span>
                </>
              ) : null}
              {video.unreviewed_count > 0 ? (
                <>
                  <span>·</span>
                  <span className={styles.rowStat}>{video.unreviewed_count} unreviewed</span>
                </>
              ) : null}
              <span>·</span>
              <span>{REVIEW_STATUS_LABELS[video.review_status]}</span>
            </span>
          ) : null}
          {video.processing_status === 'failed' && video.latest_job?.error_message ? (
            <span className={`${styles.rowMeta} ${styles.rowFlag}`}>
              {video.latest_job.error_message}
            </span>
          ) : null}
        </Link>

        <span className={styles.rowActions}>
          <ActionMenu
            items={menuItems}
            trigger={
              <Button
                variant="quiet"
                size="small"
                iconOnly
                aria-label={`Actions for ${video.original_filename}`}
              >
                <MoreIcon />
              </Button>
            }
          />
        </span>
      </ContextMenu.Trigger>

      <ContextMenu.Portal>
        <ContextMenu.Positioner className={popupStyles.positioner}>
          <ContextMenu.Popup className={popupStyles.popup}>
            {menuItems.map((item) => (
              <ContextMenu.Item
                key={item.label}
                className={popupStyles.item}
                disabled={item.disabled}
                onClick={item.onSelect}
              >
                {item.label}
              </ContextMenu.Item>
            ))}
          </ContextMenu.Popup>
        </ContextMenu.Positioner>
      </ContextMenu.Portal>
    </ContextMenu.Root>
  );
}

/** Empty states name the next action, and say what to undo to see more. */
export function LibraryEmptyState({
  filtered,
  total,
  reviewFilterSet,
  onClear,
}: {
  filtered: number;
  total: number;
  reviewFilterSet: boolean;
  onClear: () => void;
}) {
  if (total === 0) {
    return (
      <EmptyState
        title="Nothing in the library yet"
        body="Upload an underwater video to start. Uploading stores the file; you choose when processing starts."
      />
    );
  }
  return (
    <EmptyState
      title="No videos match these filters"
      body={
        reviewFilterSet
          ? `Clear the review filter to see the other ${total - filtered} videos.`
          : `${total - filtered} videos are hidden by the current filters.`
      }
      action={
        <Button size="small" onClick={onClear}>
          Clear all filters
        </Button>
      }
    />
  );
}
