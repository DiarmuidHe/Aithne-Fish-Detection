import * as React from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { useMutation, useQuery } from '@tanstack/react-query';

import type { ReviewState, TrackSummary, VideoListRow } from '@/api/types';
import { generateTrackClip, trackClipUrl } from '@/api/tracks';
import { fetchTrackSummaries, fetchVideos, sourceVideoUrl } from '@/api/videos';
import { Button, LinkButton } from '@/components/base/button';
import {
  EmptyState,
  LoadingBar,
  PanelError,
  Skeleton,
  SkeletonRows,
  StatusPill,
} from '@/components/base/feedback';
import { Modal } from '@/components/base/popups';
import { AppBarSlot } from '@/components/layout/shell';
import { KeyboardIcon } from '@/icons';
import { SpeciesLabel } from '@/features/species/species-reference';
import { useReviewApplier } from '@/features/video-detail/review-controls';
import { formatDateTime, formatPercent, formatSeconds, plural } from '@/lib/format';

import { KEY_BINDINGS, resolveCommand, type QueueCommand } from './keyboard-map';
import styles from './review-queue.module.css';

/*
 * The operator workspace: queue on the left, evidence in the middle, the record
 * and its actions on the right. The flow the layout has to make obvious is
 * select → inspect → check → decide → next, so the three regions read in that
 * order and the decision buttons sit at the end of the last one.
 */

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

  const selectTrack = React.useCallback(
    (id: string) => {
      setParams((current) => {
        const next = new URLSearchParams(current);
        next.set('t', id);
        return next;
      });
    },
    [setParams],
  );

  const selectVideo = React.useCallback(
    (id: string) => {
      setParams((current) => {
        const next = new URLSearchParams(current);
        next.set('v', id);
        next.delete('t');
        return next;
      });
    },
    [setParams],
  );

  const go = React.useCallback(
    (direction: 1 | -1) => {
      const nextTrackIndex = trackIndex + direction;
      if (nextTrackIndex >= 0 && nextTrackIndex < queueTracks.length) {
        selectTrack(queueTracks[nextTrackIndex].id);
        return;
      }
      const nextVideoIndex = videoIndex + direction;
      if (nextVideoIndex < 0 || nextVideoIndex >= queueVideos.length) return;
      selectVideo(queueVideos[nextVideoIndex].id);
    },
    [trackIndex, queueTracks, videoIndex, queueVideos, selectTrack, selectVideo],
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
      <div className={styles.workspace}>
        <aside className={styles.queuePane}>
          <div className={styles.paneHeader}>
            <span className="label">Queue</span>
          </div>
          <SkeletonRows rows={8} height={56} lines={2} />
        </aside>
        <div className={styles.evidencePane}>
          <Skeleton height={360} />
        </div>
        <aside className={styles.recordPane}>
          <div className={styles.paneHeader}>
            <span className="label">Record</span>
          </div>
          <div className={styles.recordBody}>
            <Skeleton height={220} />
          </div>
        </aside>
      </div>
    );
  }

  if (videos.isError) {
    return (
      <div style={{ padding: 'var(--space-6)' }}>
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
            See unreviewed recordings
          </LinkButton>
        }
      />
    );
  }

  const position = `Track ${queueTracks.length === 0 ? 0 : trackIndex + 1} of ${
    queueTracks.length
  } · Recording ${videoIndex + 1} of ${queueVideos.length}`;

  return (
    <div className={styles.workspace}>
      <AppBarSlot area="context">
        <span className="truncate">{currentVideo?.original_filename ?? ''}</span>
      </AppBarSlot>
      <AppBarSlot area="actions">
        <span className={styles.position} aria-live="polite">
          {position}
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
      </AppBarSlot>

      {/* --- 1. The queue ---------------------------------------------------- */}
      <aside className={styles.queuePane} aria-label="Review queue">
        <div className={styles.paneHeader}>
          <span className="label">Queue</span>
          <span className={styles.paneCount}>
            {plural(queueTracks.length, 'track')} here · {queueVideos.length} recordings
          </span>
        </div>

        {tracks.isLoading ? (
          <SkeletonRows rows={8} height={56} lines={2} />
        ) : (
          <div className={styles.queueScroll}>
            <ul className={styles.queueList}>
              {queueTracks.map((track, index) => (
                <li key={track.id} style={{ ['--i' as string]: Math.min(index, 14) }}>
                  <button
                    type="button"
                    className={styles.queueItem}
                    data-current={track.id === currentTrack?.id || undefined}
                    onClick={() => selectTrack(track.id)}
                    aria-current={track.id === currentTrack?.id ? 'true' : undefined}
                  >
                    <span className={styles.queueItemTop}>
                      <span className={styles.queueTrackId}>#{track.viame_track_id}</span>
                      <span className={styles.queueConfidence}>
                        {formatPercent(track.max_confidence, 0)}
                      </span>
                    </span>
                    <span className={styles.queueItemMeta}>
                      <span className={styles.queueSpecies}>
                        <SpeciesLabel value={track.species} />
                      </span>
                      <span className={styles.queueSpan}>{spanOf(track)}</span>
                    </span>
                    <span className={styles.queueReason} data-reason={reasonKey(track)}>
                      {reasonLabel(track)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>

            {/* Other recordings still in the queue, so the operator can jump. */}
            {queueVideos.length > 1 ? (
              <div className={styles.queueVideos}>
                <span className="label">Other recordings</span>
                {queueVideos.map((video, index) =>
                  video.id === currentVideo?.id ? null : (
                    <button
                      type="button"
                      key={video.id}
                      className={styles.queueVideo}
                      onClick={() => selectVideo(video.id)}
                    >
                      <span className="truncate">{video.original_filename}</span>
                      <span className={styles.queueVideoCount}>
                        {video.flagged_count + video.unreviewed_count}
                      </span>
                      <span className="visually-hidden">
                        outstanding tracks, recording {index + 1}
                      </span>
                    </button>
                  ),
                )}
              </div>
            ) : null}
          </div>
        )}
      </aside>

      {/* --- 2. The evidence -------------------------------------------------- */}
      <div className={styles.evidencePane}>
        <div className={styles.stage}>
          {tracks.isLoading || (!clip.data && !clip.isError) ? (
            <div className={styles.stageLoading}>
              <div className={styles.stagePlaceholder} />
              <div className={styles.stageLoadingBar}>
                <LoadingBar label="Cutting the clip for this track" />
                <p className={styles.stageLoadingText}>Cutting a clip around this track…</p>
              </div>
            </div>
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
          ) : (
            <div className={styles.stageFallback}>
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
            </div>
          )}
        </div>

        <div className={styles.evidenceFoot}>
          <p className="note">
            The clip follows this one track through the source recording. Clips are cached, so
            coming back to a track costs nothing.
          </p>
          {currentVideo ? (
            <p className={styles.evidenceMeta}>
              <span>{formatDateTime(currentVideo.created_at)}</span>
              <span className={styles.metaDot} aria-hidden="true" />
              <span>{currentVideo.camera_id ?? 'No camera'}</span>
            </p>
          ) : null}
        </div>
      </div>

      {/* --- 3. The record and the decision ----------------------------------- */}
      <aside className={styles.recordPane} aria-label="Detection record">
        <div className={styles.paneHeader}>
          <span className="label">Record</span>
          {currentTrack ? (
            <span className={styles.paneCount}>#{currentTrack.viame_track_id}</span>
          ) : null}
        </div>

        {currentTrack ? (
          <>
            <div className={styles.recordBody}>
              <p className={styles.reason} data-reason={reasonKey(currentTrack)}>
                {reasonFor(currentTrack)}
              </p>

              <dl className={styles.facts}>
                <Fact term="Species" value={<SpeciesLabel value={currentTrack.species} />} />
                <Fact term="Confidence" value={formatPercent(currentTrack.max_confidence)} />
                <Fact
                  term="Mean confidence"
                  value={formatPercent(currentTrack.mean_confidence)}
                />
                <Fact
                  term="Run threshold"
                  value={formatPercent(currentTrack.run_threshold, 0)}
                />
                <Fact term="Observations" value={String(currentTrack.detection_count)} />
                <Fact term="In recording" value={spanOf(currentTrack)} />
                <Fact
                  term="Frames"
                  value={`${currentTrack.first_frame}–${currentTrack.last_frame}`}
                />
                <Fact
                  term="Decision"
                  value={
                    <StatusPill status={currentTrack.accepted ? 'completed' : 'uploaded'}>
                      {currentTrack.review_state}
                    </StatusPill>
                  }
                />
                {currentTrack.reviewed_at ? (
                  <Fact term="Reviewed" value={formatDateTime(currentTrack.reviewed_at)} />
                ) : null}
              </dl>

              <LinkButton
                size="small"
                variant="link"
                to={`/library/${currentTrack.video_id}?tab=tracks&track=${currentTrack.id}`}
              >
                Open this track in the recording
              </LinkButton>
            </div>

            {/* The decision bar is pinned: it is the end of every pass. */}
            <div className={styles.decisionBar}>
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
          </>
        ) : (
          <div className={styles.recordBody}>
            <Skeleton height={220} />
          </div>
        )}
      </aside>

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

function Fact({ term, value }: { term: string; value: React.ReactNode }) {
  return (
    <div className={styles.fact}>
      <dt className={styles.factTerm}>{term}</dt>
      <dd className={styles.factValue}>{value}</dd>
    </div>
  );
}

/** Where in the recording this track was seen, or that the run recorded no times. */
function spanOf(track: TrackSummary): string {
  const { first_timestamp_seconds: first, last_timestamp_seconds: last } = track;
  if (first === null && last === null) {
    return `frames ${track.first_frame}–${track.last_frame}`;
  }
  return `${formatSeconds(first, '—')}–${formatSeconds(last, '—')}`;
}

function reasonKey(track: TrackSummary): 'flagged' | 'disputed' | 'borderline' {
  if (track.review_categories.includes('flagged')) return 'flagged';
  if (track.review_categories.includes('disputed')) return 'disputed';
  return 'borderline';
}

function reasonLabel(track: TrackSummary): string {
  const key = reasonKey(track);
  return key === 'flagged' ? 'Flagged' : key === 'disputed' ? 'Disputed' : 'Borderline';
}

function reasonFor(track: TrackSummary): string {
  const key = reasonKey(track);
  if (key === 'flagged') {
    return 'Flagged by an operator, so it is excluded from the count until someone decides.';
  }
  if (key === 'disputed') {
    return track.review_state === 'accepted'
      ? 'Accepted by a human even though the detector put it below the run threshold.'
      : 'Rejected by a human even though the detector put it above the run threshold.';
  }
  return 'The detector landed close to its own cut-off here, so a decision changes the count.';
}
