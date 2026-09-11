import { useQuery } from '@tanstack/react-query';

import { fetchFishClips } from '@/api/videos';
import { trackClipUrl } from '@/api/tracks';
import { AnchorButton } from '@/components/base/button';
import { EmptyState, PanelError, Skeleton } from '@/components/base/feedback';
import { formatPercent, formatSeconds, plural } from '@/lib/format';

import styles from './detail.module.css';

export function ClipsTab({ videoId }: { videoId: string }) {
  const clips = useQuery({
    queryKey: ['fish-clips', videoId],
    queryFn: () => fetchFishClips(videoId),
  });

  if (clips.isLoading) {
    return (
      <div className={styles.panel}>
        <div className={styles.clipGrid}>
          {Array.from({ length: 6 }, (_, index) => (
            <Skeleton key={index} height={160} />
          ))}
        </div>
      </div>
    );
  }

  if (clips.isError || !clips.data) {
    return (
      <div className={styles.panel}>
        <PanelError
          message={clips.error instanceof Error ? clips.error.message : 'Clips failed to load'}
          onRetry={() => void clips.refetch()}
        />
      </div>
    );
  }

  if (clips.data.length === 0) {
    return (
      <EmptyState
        title="No fish clips yet"
        body='Select "Generate fish clips" above to cut one short cropped clip per accepted track. Generating clips does not change any stored result.'
      />
    );
  }

  return (
    <div className={styles.panel}>
      <p className="note">
        {plural(clips.data.length, 'clip')}. Each clip follows one accepted fish track, with
        lead-in and lead-out around its observations.
      </p>
      <div className={styles.clipGrid}>
        {clips.data.map((clip) => (
          <figure className={styles.clipCard} key={clip.track_id}>
            <video
              src={trackClipUrl(clip.track_id, clip.generated_at)}
              controls
              loop
              muted
              playsInline
              preload="none"
            />
            <figcaption className={styles.clipCaption}>
              <span className={styles.clipTitle}>
                Fish #{clip.viame_track_id}
                {clip.species ? ` · ${clip.species}` : ''}
              </span>
              <span className={styles.clipStats}>
                {formatSeconds(clip.start_seconds)}–{formatSeconds(clip.end_seconds)} ·{' '}
                {clip.duration_seconds.toFixed(1)} s · {clip.detection_count} observations ·{' '}
                {formatPercent(clip.max_confidence)}
              </span>
              <AnchorButton
                variant="link"
                size="small"
                href={trackClipUrl(clip.track_id, clip.generated_at, true)}
                download
              >
                Download clip
              </AnchorButton>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
