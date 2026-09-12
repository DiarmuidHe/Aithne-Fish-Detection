import type { Video } from '@/api/types';
import { annotatedVideoUrl } from '@/api/videos';
import { AnchorButton } from '@/components/base/button';
import { EmptyState } from '@/components/base/feedback';
import { DownloadIcon } from '@/icons';
import { formatDateTime } from '@/lib/format';

import styles from './detail.module.css';

export function AnnotatedTab({ video }: { video: Video }) {
  if (!video.annotated_at) {
    return (
      <EmptyState
        title="No annotated copy yet"
        body='Select "Generate annotated video" above to draw boxes on the frames that contain accepted detections. Generating does not change any stored result.'
      />
    );
  }

  return (
    <div className={styles.mediaPanel}>
      <video
        className={styles.player}
        controls
        preload="metadata"
        src={annotatedVideoUrl(video.id, video.annotated_at)}
      />
      <div className={styles.actions}>
        <AnchorButton
          size="small"
          href={annotatedVideoUrl(video.id, video.annotated_at, true)}
          download
        >
          <DownloadIcon />
          Download MP4
        </AnchorButton>
        <span className="note">Generated {formatDateTime(video.annotated_at)}</span>
      </div>
      <p className={styles.mediaNote}>
        Boxes are drawn only on frames containing accepted database detections. A later review
        decision clears this copy, so regenerate it after reviewing.
      </p>
    </div>
  );
}
