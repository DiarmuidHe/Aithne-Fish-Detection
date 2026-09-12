import * as React from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';

import { fetchTrack, fetchTrackIdentification, generateTrackClip, trackClipUrl } from '@/api/tracks';
import { fetchVideo, sourceVideoUrl } from '@/api/videos';
import { Button } from '@/components/base/button';
import { PanelError, Skeleton } from '@/components/base/feedback';
import { Modal } from '@/components/base/popups';
import { DataTable, Scroller } from '@/components/base/table';
import { formatPercent, formatSeconds } from '@/lib/format';
import { IdentificationPanel } from '@/features/species/identify-fish';

import styles from './detail.module.css';

export function TrackInspector({
  trackId,
  fps,
  onClose,
}: {
  trackId: string | null;
  fps: number | null;
  onClose: () => void;
}) {
  const player = React.useRef<HTMLVideoElement | null>(null);

  const track = useQuery({
    queryKey: ['track', trackId],
    queryFn: () => fetchTrack(trackId as string),
    enabled: Boolean(trackId),
  });

  const video = useQuery({
    queryKey: ['video', track.data?.video_id],
    queryFn: () => fetchVideo(track.data?.video_id as string),
    enabled: Boolean(track.data?.video_id),
  });

  // Polled while a request is in flight: the answer lands in a background task,
  // so nothing else on this screen would tell us it arrived.
  const identification = useQuery({
    queryKey: ['track-identification', trackId],
    queryFn: () => fetchTrackIdentification(trackId as string),
    enabled: Boolean(trackId),
    refetchInterval: (query) => (query.state.data?.state === 'submitted' ? 2_000 : false),
  });

  const clip = useMutation({ mutationFn: () => generateTrackClip(trackId as string) });

  // A different track means the previous clip is no longer the one on screen.
  React.useEffect(() => {
    clip.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackId]);

  const effectiveFps = video.data?.fps ?? fps;

  return (
    <Modal
      open={Boolean(trackId)}
      onOpenChange={(open) => !open && onClose()}
      title={track.data ? `Track #${track.data.viame_track_id}` : 'Track observations'}
      description={
        track.data
          ? `${track.data.review_state} · ${track.data.detection_count} observations · VIAME max confidence ${formatPercent(track.data.max_confidence)}`
          : undefined
      }
    >
      {track.isLoading ? (
        <Skeleton height={200} />
      ) : track.isError || !track.data ? (
        <PanelError
          message={track.error instanceof Error ? track.error.message : 'Track failed to load'}
          onRetry={() => void track.refetch()}
        />
      ) : (
        <div className={styles.inspector}>
          <div className={styles.inspectorMedia}>
            <video
              ref={player}
              className={styles.player}
              controls
              preload="metadata"
              src={sourceVideoUrl(track.data.video_id)}
            />
            <p className="note">
              Original footage. Select an observation to seek to its stored frame; the
              coordinates listed are the original database boxes.
            </p>

            {clip.data ? (
              <video
                className={styles.player}
                controls
                loop
                muted
                playsInline
                preload="metadata"
                src={trackClipUrl(track.data.id, clip.data.generated_at)}
              />
            ) : (
              <Button
                onClick={() => clip.mutate()}
                disabled={clip.isPending}
              >
                {clip.isPending ? 'Cutting clip…' : 'Show cropped clip of this fish'}
              </Button>
            )}
            {clip.isError ? (
              <PanelError
                title="Clip was not created"
                message={clip.error instanceof Error ? clip.error.message : 'Clip failed'}
                onRetry={() => clip.mutate()}
              />
            ) : null}

            {/* Naming the fish belongs beside the footage of it, not in a menu. */}
            {identification.data ? (
              <IdentificationPanel
                kind="track"
                trackId={track.data.id}
                identification={identification.data}
                manualSpecies={track.data.manual_species}
                manualSpeciesAt={track.data.manual_species_at}
              />
            ) : null}
          </div>

          <Scroller className={styles.observationScroll}>
            <DataTable stickyFirstColumn={false}>
              <caption className="visually-hidden">
                Recorded observations for track {track.data.viame_track_id}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Frame</th>
                  <th scope="col">Time</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Box (x1, y1, x2, y2)</th>
                </tr>
              </thead>
              <tbody>
                {track.data.detections.map((detection) => {
                  const seconds = effectiveFps
                    ? detection.frame_number / effectiveFps
                    : detection.timestamp_seconds;
                  return (
                    <tr key={detection.id}>
                      <td>
                        <button
                          type="button"
                          className={styles.trackLink}
                          disabled={seconds === null}
                          onClick={() => {
                            if (seconds === null || !player.current) return;
                            player.current.pause();
                            player.current.currentTime = seconds;
                          }}
                        >
                          {detection.frame_number}
                        </button>
                      </td>
                      <td className="num">{formatSeconds(detection.timestamp_seconds, '—')}</td>
                      <td className="num">{formatPercent(detection.confidence)}</td>
                      <td className="num">
                        {[detection.x1, detection.y1, detection.x2, detection.y2]
                          .map((value) => value.toFixed(1))
                          .join(', ')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </DataTable>
          </Scroller>
        </div>
      )}
    </Modal>
  );
}
