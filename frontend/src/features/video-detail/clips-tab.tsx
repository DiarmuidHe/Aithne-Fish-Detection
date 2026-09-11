import { useQuery } from '@tanstack/react-query';

import { fetchFishClips } from '@/api/videos';
import { trackClipUrl } from '@/api/tracks';
import { AnchorButton } from '@/components/base/button';
import { EmptyState, PanelError, Skeleton } from '@/components/base/feedback';
import { MediaGrid, MediaTile } from '@/components/base/media-grid';
import { formatPercent, formatSeconds, plural } from '@/lib/format';

import styles from './detail.module.css';

/*
 * One clip per accepted fish track. This is an evidence board, so the footage
 * is the content and the caption is two lines underneath it — no card, no
 * button printed on every tile.
 */

export function ClipsTab({ videoId }: { videoId: string }) {
  const clips = useQuery({
    queryKey: ['fish-clips', videoId],
    queryFn: () => fetchFishClips(videoId),
  });

  if (clips.isLoading) {
    return (
      <div className={styles.mediaPanel}>
        <MediaGrid>
          {Array.from({ length: 6 }, (_, index) => (
            <Skeleton key={index} height={160} radius={5} />
          ))}
        </MediaGrid>
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
    <div className={styles.mediaPanel}>
      <p className={styles.mediaNote}>
        {plural(clips.data.length, 'clip')}. Each clip follows one accepted fish track, with
        lead-in and lead-out around its observations.
      </p>
      <MediaGrid>
        {clips.data.map((clip) => (
          <MediaTile
            key={clip.track_id}
            ratio="16 / 9"
            media={
              <video
                src={trackClipUrl(clip.track_id, clip.generated_at)}
                controls
                loop
                muted
                playsInline
                preload="none"
                aria-label={`Clip following fish ${clip.viame_track_id}`}
              />
            }
            title={clip.species ?? `Fish ${clip.viame_track_id}`}
            subtitle={`#${clip.viame_track_id}`}
            meta={`${formatSeconds(clip.start_seconds)}–${formatSeconds(clip.end_seconds)} · ${clip.duration_seconds.toFixed(1)} s · ${clip.detection_count} observations · ${formatPercent(clip.max_confidence)}`}
            actions={
              <AnchorButton
                variant="quiet"
                size="small"
                href={trackClipUrl(clip.track_id, clip.generated_at, true)}
                download
              >
                Save
              </AnchorButton>
            }
          />
        ))}
      </MediaGrid>
    </div>
  );
}
