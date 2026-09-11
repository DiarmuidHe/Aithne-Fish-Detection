import * as React from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { useMutation, useQuery } from '@tanstack/react-query';

import type { ReviewState, TrackSummary, VideoListRow } from '@/api/types';
import { generateTrackClip, trackClipUrl } from '@/api/tracks';
import { fetchTrackSummaries, fetchVideos, sourceVideoUrl } from '@/api/videos';
import { Button, LinkButton } from '@/components/base/button';
import { EmptyState, PanelError, Skeleton, StatusPill } from '@/components/base/feedback';
import { Modal } from '@/components/base/popups';
import { KeyboardIcon } from '@/icons';
import { useReviewApplier } from '@/features/video-detail/review-controls';
import { formatDateTime, formatPercent, formatSeconds, plural } from '@/lib/format';

import { KEY_BINDINGS, resolveCommand, type QueueCommand } from './keyboard-map';
import styles from './review-queue.module.css';

/** The three categories where a human decision actually changes the count. */
const OUTSTANDING = ['flagged', 'borderline', 'disputed'];

const DECISIONS: { command: QueueCommand; state: ReviewState; label: string; key: string }[] = [
  { command: 'accept', state: 'accepted', label: 'Accept', key: 'a' },
  { command: 'reject', state: 'rejected', label: 'Reject', key: 'r' },
  { command: 'flag', state: 'needs-review', label: 'Flag', key: 'f' },
  { command: 'unreview', state: 'unreviewed', label: 'Unreview', key: 'u' },
];

export function ReviewQueueRoute() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [helpOpen, setHelpOpen] = React.useState(false);
  const player = React.useRef<HTMLVideoElement | null>(null);

  const videos = useQuery({
    queryKey: ['review-queue-videos'],
    queryFn: () =>
      fetchVideos({
        review: OUTSTANDING,
        status: 'completed',
        sort: 'flagged',
        order: 'desc',
        limit: 200,
      }),
  });

  const queueVideos = videos.data?.rows ?? [];
  const videoId = params.get('v') ?? queueVideos[0]?.id;
  const videoIndex = Math.max(
    0,
    queueVideos.findIndex((video) => video.id === videoId),
  );
  const currentVideo: VideoListRow | undefined = queueVideos[videoIndex];

  const tracks = useQuery({
    queryKey: ['review-queue-tracks', currentVideo?.id],
    queryFn: () =>
      fetchTrackSummaries(currentVideo?.id as string, {
        accepted_only: false,
        review: OUTSTANDING,
        sort: 'first_frame',
        order: 'asc',
      }),
    enabled: Boolean(currentVideo?.id),
  });

  const queueTracks = tracks.data ?? [];
  const trackId = params.get('t') ?? queueTracks[0]?.id;
  const trackIndex = Math.max(
    0,
    queueTracks.findIndex((track) => track.id === trackId),
  );
  const currentTrack: TrackSummary | undefined = queueTracks[trackIndex];

  const applier = useReviewApplier([
    ['review-queue-videos'],
    ['review-queue-tracks'],
    ['videos'],
    ['facets'],
  ]);

  const clip = useMutation({ mutationFn: (id: string) => generateTrackClip(id) });

  // The queue's whole point is seeing the fish, so cut the clip on arrival.
  // Clips are cached on disk, so revisiting a track costs nothing.
  React.useEffect(() => {
    clip.reset();
    if (currentTrack) clip.mutate(currentTrack.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTrack?.id]);

  const go = React.useCallback(
    (direction: 1 | -1) => {
      const nextTrackIndex = trackIndex + direction;
      if (nextTrackIndex >= 0 && nextTrackIndex < queueTracks.length) {
        setParams((current) => {
          const next = new URLSearchParams(current);
          next.set('t', queueTracks[nextTrackIndex].id);
          return next;
        });
        return;
      }
      const nextVideoIndex = videoIndex + direction;
      if (nextVideoIndex < 0 || nextVideoIndex >= queueVideos.length) return;
      setParams((current) => {
        const next = new URLSearchParams(current);
        next.set('v', queueVideos[nextVideoIndex].id);
        next.delete('t');
        return next;
      });
    },
    [trackIndex, queueTracks, videoIndex, queueVideos, setParams],
  );

  const decide = React.useCallback(
    (state: ReviewState) => {
      if (!currentTrack) return;
      applier.applyOne(currentTrack, state);
      // A decided track leaves the queue, so move on rather than sit on it.
      go(1);
    },
    [currentTrack, applier, go],
  );

  const run = React.useCallback(
    (command: QueueCommand) => {
      switch (command) {
        case 'next':
          go(1);
          break;
        case 'previous':
          go(-1);
          break;
        case 'accept':
          decide('accepted');
          break;
        case 'reject':
          decide('rejected');
          break;
        case 'flag':
          decide('needs-review');
          break;
        case 'unreview':
          decide('unreviewed');
          break;
        case 'toggle-play':
          if (player.current) {
            if (player.current.paused) void player.current.play();
            else player.current.pause();
          }
          break;
        case 'open-source':
          if (currentVideo) navigate(`/library/${currentVideo.id}?tab=tracks`);
          break;
        case 'back':
          if (helpOpen) setHelpOpen(false);
          else navigate('/library');
          break;
        case 'help':
          setHelpOpen(true);
          break;
      }
    },
    [go, decide, currentVideo, navigate, helpOpen],
  );

  React.useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const command = resolveCommand(event);
      if (!command) return;
      event.preventDefault();
      run(command);
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [run]);

  if (videos.isLoading) {
    return (
      <div className={styles.body}>
        <Skeleton height={320} />
        <Skeleton height={200} />
      </div>
    );
  }

  if (videos.isError) {
    return (
      <div className={styles.body}>
        <PanelError
          title="The review queue did not load"
          message={videos.error instanceof Error ? videos.error.message : 'Request failed'}
          onRetry={() => void videos.refetch()}
        />
      </div>
    );
  }

  if (queueVideos.length === 0 || (tracks.isFetched && queueTracks.length === 0 && !currentTrack)) {
    return (
      <EmptyState
        title="Nothing is waiting for a decision"
        body="No track across the library is flagged, borderline or disputed. Videos still holding unreviewed tracks that sit well away from the threshold are counted by the detector alone."
        action={
          <LinkButton size="small" to="/library?review=unreviewed">
            See unreviewed videos
          </LinkButton>
        }
      />
    );
  }

  return (
    <div className={styles.queue}>
      <header className={styles.header}>
        <div className={styles.context}>
          <h1 className={styles.contextTitle}>
            {currentVideo?.original_filename ?? 'Loading'}
          </h1>
          <p className={styles.contextMeta}>
            <span>{currentVideo ? formatDateTime(currentVideo.created_at) : ''}</span>
            <span>·</span>
            <span>{currentVideo?.camera_id ?? 'No camera'}</span>
            <span>·</span>
            <span>{plural(currentVideo?.flagged_count ?? 0, 'flagged track')}</span>
          </p>
        </div>

        <div className={styles.position}>
          <span className={styles.positionValue} aria-live="polite">
            Track {queueTracks.length === 0 ? 0 : trackIndex + 1} of {queueTracks.length} · Video{' '}
            {videoIndex + 1} of {queueVideos.length}
          </span>
          <Button size="small" onClick={() => run('previous')} disabled={queueTracks.length === 0}>
            Previous
          </Button>
          <Button size="small" onClick={() => run('next')} disabled={queueTracks.length === 0}>
            Next
          </Button>
          <Button
            size="small"
            iconOnly
            aria-label="Keyboard shortcuts"
            onClick={() => setHelpOpen(true)}
          >
            <KeyboardIcon />
          </Button>
        </div>
      </header>

      <div className={styles.body}>
        <div className={styles.media}>
          {tracks.isLoading ? (
            <Skeleton height={320} />
          ) : clip.data ? (
            <video
              ref={player}
              className={styles.player}
              controls
              loop
              muted
              playsInline
              autoPlay
              preload="auto"
              src={trackClipUrl(clip.data.track_id, clip.data.generated_at)}
            />
          ) : clip.isError ? (
            <>
              <PanelError
                title="A cropped clip could not be cut"
                message={clip.error instanceof Error ? clip.error.message : 'Clip failed'}
                onRetry={() => currentTrack && clip.mutate(currentTrack.id)}
              />
              {currentVideo ? (
                <video
                  ref={player}
                  className={styles.player}
                  controls
                  preload="metadata"
                  src={sourceVideoUrl(currentVideo.id)}
                />
              ) : null}
            </>
          ) : (
            <Skeleton height={320} />
          )}
          <p className="note">
            The clip follows this one track through the source video. Clips are cached, so coming
            back to a track costs nothing.
          </p>
        </div>

        <div className={styles.decision}>
          {currentTrack ? (
            <>
              <p className={styles.reason}>{reasonFor(currentTrack)}</p>

              <dl className={styles.facts}>
                <dt className={styles.factTerm}>Track</dt>
                <dd className={styles.factValue}>#{currentTrack.viame_track_id}</dd>
                <dt className={styles.factTerm}>Max confidence</dt>
                <dd className={styles.factValue}>
                  {formatPercent(currentTrack.max_confidence)} against a{' '}
                  {formatPercent(currentTrack.run_threshold, 0)} run threshold
                </dd>
                <dt className={styles.factTerm}>Mean confidence</dt>
                <dd className={styles.factValue}>
                  {formatPercent(currentTrack.mean_confidence)}
                </dd>
                <dt className={styles.factTerm}>Observations</dt>
                <dd className={styles.factValue}>{currentTrack.detection_count}</dd>
                <dt className={styles.factTerm}>In video</dt>
                <dd className={styles.factValue}>{spanOf(currentTrack)}</dd>
                <dt className={styles.factTerm}>Species</dt>
                <dd className={styles.factValue}>{currentTrack.species ?? 'Unclassified'}</dd>
                <dt className={styles.factTerm}>Current decision</dt>
                <dd className={styles.factValue}>
                  <StatusPill
                    status={currentTrack.accepted ? 'completed' : 'uploaded'}
                  >
                    {currentTrack.review_state}
                  </StatusPill>
                </dd>
              </dl>

              <div className={styles.decisionButtons}>
                <div className={styles.decisionRow}>
                  {DECISIONS.map((decision) => (
                    <Button
                      key={decision.state}
                      variant={decision.state === 'accepted' ? 'primary' : 'default'}
                      onClick={() => run(decision.command)}
                    >
                      {decision.label}
                      <span className={styles.key} aria-hidden="true">
                        {decision.key}
                      </span>
                    </Button>
                  ))}
                </div>
                <div className={styles.decisionRow}>
                  <LinkButton
                    size="small"
                    variant="quiet"
                    to={`/library/${currentTrack.video_id}?tab=tracks&track=${currentTrack.id}`}
                  >
                    Open in the library
                    <span className={styles.key} aria-hidden="true">
                      Enter
                    </span>
                  </LinkButton>
                </div>
              </div>
            </>
          ) : (
            <Skeleton height={220} />
          )}
        </div>
      </div>

      <Modal
        open={helpOpen}
        onOpenChange={setHelpOpen}
        narrow
        title="Keyboard shortcuts"
        description="Every one of these has a button on screen as well."
      >
        <dl className={styles.keyTable}>
          {KEY_BINDINGS.map((binding) => (
            <React.Fragment key={binding.key}>
              <dt>
                <span className={styles.key}>{binding.display}</span>
              </dt>
              <dd>{binding.description}</dd>
            </React.Fragment>
          ))}
        </dl>
      </Modal>
    </div>
  );
}

/** Where in the video this track was seen, or that the run recorded no times. */
function spanOf(track: TrackSummary): string {
  const { first_timestamp_seconds: first, last_timestamp_seconds: last } = track;
  if (first === null && last === null) {
    return `frames ${track.first_frame}–${track.last_frame}, not timed`;
  }
  return `${formatSeconds(first, '—')}–${formatSeconds(last, '—')}`;
}

function reasonFor(track: TrackSummary): string {
  if (track.review_categories.includes('flagged')) {
    return 'Flagged by an operator, so it is excluded from the count until someone decides.';
  }
  if (track.review_categories.includes('disputed')) {
    return track.review_state === 'accepted'
      ? 'Accepted by a human even though the detector put it below the run threshold.'
      : 'Rejected by a human even though the detector put it above the run threshold.';
  }
  return 'The detector landed close to its own cut-off here, so a decision changes the count.';
}
