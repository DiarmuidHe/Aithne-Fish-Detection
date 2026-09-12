import { useQuery } from '@tanstack/react-query';

import { fetchFishClips } from '@/api/videos';
import { trackClipUrl } from '@/api/tracks';
import { AnchorButton } from '@/components/base/button';
import { EmptyState, PanelError, Skeleton } from '@/components/base/feedback';
import { MediaGrid, MediaTile } from '@/components/base/media-grid';
import { formatPercent, formatSeconds, plural } from '@/lib/format';
import { IdentificationBadge, IdentifyFishButton } from '@/features/species/identify-fish';
import { AssignSpeciesButton } from '@/features/species/species-selector';

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
    // A requested identification lands in a background task, so keep looking
    // while any fish on this board is waiting on one.
    refetchInterval: (query) =>
      query.state.data?.some((clip) => clip.fishial_state === 'submitted') ? 3_000 : false,
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
            title={
              clip.manual_species ??
              clip.fishial_species ??
              clip.species ??
              `Fish ${clip.viame_track_id}`
            }
            subtitle={`#${clip.viame_track_id}`}
            flagTone={
              clip.manual_species || clip.fishial_state === 'identified' ? 'named' : 'neutral'
            }
            flag={
              clip.fishial_state === 'none' && !clip.manual_species ? undefined : (
                <IdentificationBadge
                  identification={{
                    state: clip.fishial_state,
                    species: clip.fishial_species,
                    confidence: clip.fishial_species_confidence,
                  }}
                  manualSpecies={clip.manual_species}
                />
              )
            }
            meta={`${formatSeconds(clip.start_seconds)}–${formatSeconds(clip.end_seconds)} · ${clip.duration_seconds.toFixed(1)} s · ${clip.detection_count} observations · ${formatPercent(clip.max_confidence)}`}
            actions={
              <>
                <IdentifyFishButton
                  kind="track"
                  trackId={clip.track_id}
                  state={clip.fishial_state}
                  variant="quiet"
                  label="Identify"
                />
                <AssignSpeciesButton
                  kind="track"
                  trackId={clip.track_id}
                  current={clip.manual_species}
                  identification={{
                    state: clip.fishial_state,
                    species: clip.fishial_species,
                    confidence: clip.fishial_species_confidence,
                  }}
                  variant="quiet"
                  label={clip.manual_species ? 'Change name' : 'Name it'}
                />
                <AnchorButton
                  variant="quiet"
                  size="small"
                  href={trackClipUrl(clip.track_id, clip.generated_at, true)}
                  download
                >
                  Save
                </AnchorButton>
              </>
            }
          />
        ))}
      </MediaGrid>
    </div>
  );
}
