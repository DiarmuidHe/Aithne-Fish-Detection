import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import type { LiveTrack } from '@/api/types';
import {
  fetchLiveActivity,
  fetchLiveClips,
  fetchLiveLatest,
  fetchLiveSources,
  fetchLiveSpecies,
  liveSnapshotUrl,
  startLive,
  stopLive,
} from '@/api/live';
import { AnchorButton, Button } from '@/components/base/button';
import { NumberInput, Segmented } from '@/components/base/controls';
import {
  EmptyState,
  LoadingBar,
  PanelError,
  Skeleton,
  StatusPill,
} from '@/components/base/feedback';
import { MediaGrid, MediaTile } from '@/components/base/media-grid';
import { SingleSelect } from '@/components/base/select';
import { ColumnChart } from '@/components/charts/bar-chart';
import { AppBarSlot, Band, Page } from '@/components/layout/shell';
import { useToast } from '@/components/base/toast';
import { PlayIcon, StopIcon } from '@/icons';
import { useDocumentVisible } from '@/hooks/use-poll';
import { formatClock, formatCount, formatPercent, plural } from '@/lib/format';
import { IdentifyFishButton } from '@/features/species/identify-fish';
import { SpeciesNameplate, SpeciesThumb } from '@/features/species/species-reference';
import { AssignSpeciesButton } from '@/features/species/species-selector';

import styles from './live.module.css';

const OPEN_STATUSES = new Set(['queued', 'starting', 'running', 'reconnecting', 'stopping']);

const SOURCE_KEY_STORAGE = 'live-source-key';
const SPECIES_TARGET_STORAGE = 'live-species-fish-target';

type GallerySort = 'recent' | 'duration' | 'confidence';

/*
 * Live monitoring is an observation station, not a page of cards.
 *
 * The camera toolbar, the footage, the counts and the lag line are one dark
 * console slab running the full width of the workspace, divided internally by
 * hairlines. It takes the marine spine's own ground, so the instrument reads as
 * continuous with the rail rather than as a widget sitting on paper. Below it,
 * on light ground, sits the record of what swam past.
 */

export function LiveRoute() {
  const toast = useToast();
  const queryClient = useQueryClient();
  const visible = useDocumentVisible();

  const [sourceKey, setSourceKey] = React.useState<string | null>(() => stored(SOURCE_KEY_STORAGE));
  const [speciesTarget, setSpeciesTarget] = React.useState<number | null>(() =>
    Number(stored(SPECIES_TARGET_STORAGE)) || 0,
  );
  const [framesPerFish, setFramesPerFish] = React.useState<number | null>(null);
  const [gallerySpecies, setGallerySpecies] = React.useState<string[]>([]);
  const [identifiedOnly, setIdentifiedOnly] = React.useState<string[]>([]);
  const [gallerySort, setGallerySort] = React.useState<GallerySort>('recent');

  const sources = useQuery({ queryKey: ['live-sources'], queryFn: fetchLiveSources });

  // Only one session runs at a time, so a reload or a second tab must show the
  // camera that is actually being analyzed rather than a remembered one.
  const running = sources.data?.sources.find((source) => source.active_session_id);
  const effectiveKey =
    running?.key ??
    (sources.data?.sources.some((source) => source.key === sourceKey)
      ? (sourceKey as string)
      : sources.data?.default_key) ??
    null;

  React.useEffect(() => {
    if (framesPerFish === null && sources.data) {
      setFramesPerFish(sources.data.fishial.default_frames_per_fish);
    }
  }, [framesPerFish, sources.data]);

  const latest = useQuery({
    queryKey: ['live-latest', effectiveKey],
    queryFn: () => fetchLiveLatest(effectiveKey as string),
    enabled: Boolean(effectiveKey),
  });

  const session = latest.data?.session ?? null;
  const isRunning = session !== null && OPEN_STATUSES.has(session.status);
  // The worker finishes the segment it is mid-way through before it winds down,
  // so "stopping" can last a while and has to look like progress, not a dead control.
  const isStopping = session?.status === 'stopping';
  // Live is the one screen that genuinely needs a short interval, and only
  // while a session is open and the operator is looking at it.
  const interval = visible ? (isRunning ? 2_000 : 10_000) : false;

  React.useEffect(() => {
    if (interval === false) return;
    const timer = window.setInterval(() => {
      void queryClient.invalidateQueries({ queryKey: ['live-latest'] });
      void queryClient.invalidateQueries({ queryKey: ['live-sources'] });
      if (session) {
        void queryClient.invalidateQueries({ queryKey: ['live-activity', session.id] });
        void queryClient.invalidateQueries({ queryKey: ['live-clips', session.id] });
        void queryClient.invalidateQueries({ queryKey: ['live-species', session.id] });
      }
    }, interval);
    return () => window.clearInterval(timer);
  }, [interval, queryClient, session]);

  const activity = useQuery({
    queryKey: ['live-activity', session?.id],
    queryFn: () => fetchLiveActivity(session?.id as string),
    enabled: Boolean(session?.id),
  });

  const clips = useQuery({
    queryKey: ['live-clips', session?.id],
    queryFn: () => fetchLiveClips(session?.id as string),
    enabled: Boolean(session?.id),
  });

  const species = useQuery({
    queryKey: ['live-species', session?.id],
    queryFn: () => fetchLiveSpecies(session?.id as string),
    enabled: Boolean(session?.id),
  });

  const start = useMutation({
    mutationFn: () =>
      startLive(effectiveKey as string, speciesTarget ?? 0, framesPerFish ?? 5),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['live-latest'] });
      void queryClient.invalidateQueries({ queryKey: ['live-sources'] });
    },
    onError: (error) =>
      toast.add({ title: 'Could not start', description: text(error), type: 'error', timeout: 0 }),
  });

  const stop = useMutation({
    mutationFn: () => stopLive(session?.id as string),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['live-latest'] });
      void queryClient.invalidateQueries({ queryKey: ['live-sources'] });
    },
    onError: (error) =>
      toast.add({ title: 'Could not stop', description: text(error), type: 'error', timeout: 0 }),
  });

  const enabled = latest.data?.enabled ?? sources.data?.enabled ?? false;
  const available = latest.data?.available ?? sources.data?.available ?? false;
  const fishial = sources.data?.fishial;

  const galleryTracks = React.useMemo(
    () => filterAndSort(clips.data ?? [], gallerySpecies, identifiedOnly, gallerySort),
    [clips.data, gallerySpecies, identifiedOnly, gallerySort],
  );

  const gallerySpeciesOptions = React.useMemo(() => {
    const found = new Set<string>();
    for (const track of clips.data ?? []) {
      const name = track.fishial_state === 'identified' ? track.fishial_species : track.species;
      if (name) found.add(name);
    }
    return [...found].sort().map((value) => ({ value, label: value }));
  }, [clips.data]);

  if (sources.isError) {
    return (
      <Page>
        <PanelError
          message={sources.error instanceof Error ? sources.error.message : 'Request failed'}
          onRetry={() => void sources.refetch()}
        />
      </Page>
    );
  }

  const activeLabel = sources.data?.sources.find((source) => source.key === effectiveKey)?.label;

  return (
    <Page width="full">
      {/* Start and stop are the page's primary actions, so they live in the bar. */}
      <AppBarSlot area="context">
        <span>{activeLabel ?? 'No camera selected'}</span>
        {session ? (
          <StatusPill status={isRunning ? 'processing' : 'completed'}>{session.status}</StatusPill>
        ) : null}
      </AppBarSlot>
      <AppBarSlot area="actions">
        <Button
          variant="primary"
          size="small"
          disabled={!enabled || !available || isRunning || start.isPending}
          onClick={() => start.mutate()}
        >
          <PlayIcon />
          Start monitoring
        </Button>
        <Button
          size="small"
          disabled={!isRunning || stop.isPending || (isStopping && !session?.worker_stale)}
          onClick={() => stop.mutate()}
        >
          <StopIcon />
          {isStopping ? 'Stopping…' : 'Stop'}
        </Button>
      </AppBarSlot>

      {/*
        * --- The console. Toolbar, footage, counts and lag on one dark slab.
        * Every semantic token is remapped to the chrome family inside it, so
        * the selects, number fields and the activity chart come along without
        * a second set of styles.
        */}
      <div className={styles.console}>
        <div className={styles.toolbar}>
          <SingleSelect
            label="Camera"
            value={effectiveKey ?? ''}
            onChange={(value) => {
              setSourceKey(value);
              store(SOURCE_KEY_STORAGE, value);
            }}
            options={(sources.data?.sources ?? []).map((source) => ({
              value: source.key,
              label: `${source.label}${source.location ? ` · ${source.location}` : ''}${
                source.active_session_id ? ' · monitoring' : ''
              }`,
            }))}
          />

          {fishial?.enabled ? (
            <>
              <span className={styles.toolbarRule} aria-hidden="true" />
              <NumberInput
                label="Identify species for the first"
                min={0}
                max={fishial.max_fish_per_session}
                value={speciesTarget}
                disabled={isRunning}
                onChange={(value) => {
                  setSpeciesTarget(value);
                  store(SPECIES_TARGET_STORAGE, String(value ?? 0));
                }}
              />
              <NumberInput
                label="Clear frames per fish"
                min={1}
                max={20}
                value={framesPerFish}
                disabled={isRunning}
                onChange={setFramesPerFish}
              />
              <p className={styles.toolbarNote}>
                0 = off. Sends fish crops to the Fishial identification service; each frame
                uses one API call.
              </p>
            </>
          ) : null}

          <p className={styles.toolbarGuard}>
            {isRunning && fishial?.enabled
              ? 'Stop monitoring to change the camera or species identification.'
              : isRunning
                ? 'Stop monitoring to switch to another camera.'
                : 'One camera at a time.'}
          </p>
        </div>

        <div className={styles.body}>
          <div className={styles.stage}>
            {session?.annotated_stream_url ? (
              <img
                className={styles.player}
                alt="Live camera with boxes and labels for detected fish"
                // A fresh immutable snapshot each poll avoids stuck MJPEG sockets
                // after the worker reconnects.
                src={liveSnapshotUrl(session.annotated_stream_url, session.last_frame_at)}
              />
            ) : (
              <div className={styles.placeholder}>
                <p>
                  {isRunning
                    ? 'Waiting for an annotated frame…'
                    : !enabled
                      ? 'Live monitoring is disabled for this deployment.'
                      : !available
                        ? 'Live monitoring needs the VIAME worker in GPU mode.'
                        : 'Start monitoring to view analyzed camera frames.'}
                </p>
                {isRunning ? (
                  <span className={styles.placeholderBar}>
                    <LoadingBar label="Waiting for an annotated frame" />
                  </span>
                ) : null}
              </div>
            )}

            {/* The camera label sits on the footage, the way an overlay does. */}
            {activeLabel ? (
              <span className={styles.stageLabel}>
                {activeLabel}
                {isRunning ? (
                  <>
                    <span className={styles.liveDot} aria-hidden="true" />
                    <span className={styles.liveWord}>Live</span>
                  </>
                ) : null}
              </span>
            ) : null}
          </div>

          {/*
            * The counts belong to the footage beside them, so they share its
            * slab and are separated from it by a rule rather than by a gap.
            */}
          <aside className={styles.side} aria-label="Session activity">
            {activity.data ? (
              <>
                <dl className={styles.stats}>
                  <div
                    className={styles.stat}
                    data-live={activity.data.active_tracks > 0 || undefined}
                  >
                    <dt className={styles.statLabel}>Active tracks</dt>
                    <dd className={styles.statValue}>{formatCount(activity.data.active_tracks)}</dd>
                  </div>
                  <div className={styles.stat}>
                    <dt className={styles.statLabel}>
                      In window
                      <span className={styles.statNote}>
                        {formatCount(activity.data.window_detections)} detections
                      </span>
                    </dt>
                    <dd className={styles.statValue}>{formatCount(activity.data.window_tracks)}</dd>
                  </div>
                  <div className={styles.stat}>
                    <dt className={styles.statLabel}>Finalised</dt>
                    <dd className={styles.statValue}>
                      {formatCount(activity.data.finalized_tracks)}
                    </dd>
                  </div>
                </dl>

                <div className={styles.activity}>
                  <p className={styles.activityHead}>
                    <span className="label">Activity</span>
                    <span className={styles.activityRange}>
                      Last {activity.data.window_seconds}s analyzed
                    </span>
                  </p>
                  <ColumnChart
                    title=""
                    valueLabel="distinct tracks"
                    columns={activity.data.series.map((point) => ({
                      label: formatClock(point.at),
                      value: point.tracks,
                    }))}
                  />
                </div>
              </>
            ) : session ? (
              <div className={styles.sidePad}>
                <Skeleton height={180} radius={6} />
              </div>
            ) : (
              <div className={styles.sideIdle}>
                <p className="label">Not monitoring</p>
                <p className={styles.sideIdleBody}>
                  Counts and the activity trace appear here once a session is open.
                </p>
              </div>
            )}
          </aside>
        </div>

        {/* The lag line: the bottom rail of the instrument. */}
        <p
          className={styles.statusLine}
          role="status"
          aria-live="polite"
          data-running={isRunning || undefined}
        >
          {!enabled ? (
            'Live monitoring is disabled. Enable LIVE_MONITOR_ENABLED on the API and live worker.'
          ) : !available ? (
            'Live monitoring needs the VIAME worker in GPU mode. This deployment uses mock detections.'
          ) : !session ? (
            `Ready to monitor ${activeLabel ?? 'the camera'}.`
          ) : (
            <>
              <span className={styles.lag}>
                {session.lag_seconds === null
                  ? 'Waiting for frames'
                  : `${session.lag_seconds.toFixed(1)}s since the latest analyzed frame`}
              </span>
              <span className={styles.lagDot} aria-hidden="true" />
              <span className={styles.lag}>{session.reconnect_count} reconnects</span>
              <span className={styles.lagDot} aria-hidden="true" />
              <span className={styles.lag}>{session.dropped_segments} skipped segments</span>
              {session.status === 'queued' ? (
                <span className={styles.lagNote}>Waiting for the live worker</span>
              ) : null}
              {isStopping ? (
                <span className={styles.lagNote}>
                  Finishing the segment it is already analyzing, then stopping
                </span>
              ) : null}
            </>
          )}
        </p>
      </div>

      {session?.worker_stale ? (
        <PanelError
          title="The live worker heartbeat is overdue"
          message="Check the live worker service."
        />
      ) : session?.error_message ? (
        <PanelError title="The session reported an error" message={session.error_message} />
      ) : null}

      {session ? (
        <Band
          tone="sunken"
          label="Finalised fish histories"
          note={`A history finishes after about ${session.lost_track_seconds}s without a detection, or when monitoring stops. Latest 50 shown.`}
          action={
            <span className={styles.shownCount}>{formatCount(galleryTracks.length)} shown</span>
          }
        >
          <div className={styles.galleryFilters}>
            <SingleSelect
              label="Species"
              value={gallerySpecies[0] ?? ''}
              onChange={(value) => setGallerySpecies(value ? [value] : [])}
              options={[{ value: '', label: 'Any' }, ...gallerySpeciesOptions]}
            />
            <Segmented
              label="Identification"
              value={identifiedOnly}
              multiple
              onChange={setIdentifiedOnly}
              options={[
                { value: 'identified', label: 'Named' },
                { value: 'unnamed', label: 'Not named' },
              ]}
            />
            <Segmented
              label="Sort finalised histories"
              value={[gallerySort]}
              onChange={(value) => setGallerySort((value[0] as GallerySort) ?? 'recent')}
              options={[
                { value: 'recent', label: 'Most recent' },
                { value: 'duration', label: 'Longest' },
                { value: 'confidence', label: 'Highest confidence' },
              ]}
            />
          </div>

          {galleryTracks.length === 0 ? (
            <EmptyState
              title="No finalised histories match"
              body={
                (clips.data?.length ?? 0) > 0
                  ? 'Clear the species or identification filter to see the other histories.'
                  : 'Finalised fish clips and crops appear here once a fish leaves the frame.'
              }
            />
          ) : (
            <MediaGrid size="large">
              {galleryTracks.map((track) => (
                <LiveTile key={track.id} track={track} />
              ))}
            </MediaGrid>
          )}
        </Band>
      ) : null}

      {session?.species_id.enabled && species.data ? (
        <Band label="Species identification" note="Names supplied by the Fishial service">
          <dl className={styles.speciesList}>
            {species.data.species.map((entry) => (
              <div className={styles.speciesRow} key={entry.species}>
                <dt className={styles.speciesName}>
                  <SpeciesThumb name={entry.species} size="row" />
                  <span>{entry.species}</span>
                </dt>
                <dd className={styles.speciesCount}>{plural(entry.count, 'fish')}</dd>
                <dd className={styles.speciesCount}>
                  {formatPercent(entry.mean_confidence, 0)} mean
                </dd>
              </div>
            ))}
          </dl>
          <p className={`note ${styles.speciesNote}`}>
            {session.species_id.fish_enrolled}/{session.species_id.fish_target} enrolled ·{' '}
            {species.data.review_required} need review ·{' '}
            {species.data.declined > 0 ? `${species.data.declined} not named by the model · ` : ''}
            {species.data.api_calls} Fishial image calls ({species.data.calls_saved} saved)
            {session.species_id.manual_api_calls > 0
              ? ` · ${session.species_id.manual_api_calls} requested by hand`
              : ''}
            {Object.entries(species.data.review_reasons)
              .map(([reason, count]) => ` · ${count} ${reason}`)
              .join('')}
          </p>
        </Band>
      ) : null}
    </Page>
  );
}

/**
 * One finalised fish history. The clip is the record; the identifier, the
 * species and the figures are the label written under it.
 */
function LiveTile({ track }: { track: LiveTrack }) {
  // A name a person put on outranks the classifier's, which outranks the
  // detector's class; the tile has room for one, and that is the order.
  const name =
    track.manual_species ??
    (track.fishial_state === 'identified' ? track.fishial_species : (track.species ?? 'Fish'));
  const state = track.fishial_state;
  const flagTone =
    track.manual_species || state === 'identified' ? 'named'
      : state === 'review_required' || state === 'error' ? 'flagged'
      : 'neutral';

  return (
    <MediaTile
      flagTone={flagTone}
      media={
        track.clip_url ? (
          <video
            src={track.clip_url}
            poster={track.crop_url ?? undefined}
            controls
            loop
            muted
            playsInline
            preload="none"
            aria-label={`Clip following ${name}`}
          />
        ) : track.crop_url ? (
          <img src={track.crop_url} alt={`Annotated crop of ${name}`} loading="lazy" />
        ) : null
      }
      title={
        (track.manual_species || track.fishial_state === 'identified') && name ? (
          <SpeciesNameplate name={name} size="tile" />
        ) : (
          (name ?? 'Fish')
        )
      }
      subtitle={track.id.slice(0, 8)}
      flag={badgeFor(track) || undefined}
      meta={`${track.detection_count} detections · ${formatPercent(track.max_confidence, 0)} · ${formatClock(track.last_seen_at)}`}
      footnote={track.media_error ?? undefined}
      actions={
        <>
          {/* Naming a fish is the action an operator reaches for while watching,
              so it sits on the fish itself rather than behind the session. */}
          <IdentifyFishButton
            kind="live"
            trackId={track.id}
            state={track.fishial_state}
            variant="quiet"
            label="Identify"
          />
          <AssignSpeciesButton
            kind="live"
            trackId={track.id}
            current={track.manual_species}
            identification={track.identification}
            variant="quiet"
            label={track.manual_species ? 'Change name' : 'Name it'}
          />
          {track.crop_url ? (
            <AnchorButton variant="quiet" size="small" href={track.crop_url}>
              Crop
            </AnchorButton>
          ) : null}
        </>
      }
    />
  );
}

function badgeFor(track: LiveTrack): string {
  if (track.manual_species) return 'Named by hand';
  const state = track.fishial_state;
  if (state === 'identified') {
    return `Named · ${formatPercent(track.fishial_species_confidence ?? 0, 0)}`;
  }
  if (state === 'review_required' || state === 'error') {
    const diagnostics = track.fishial_diagnostics ?? {};
    const reason = diagnostics.reason ?? 'Consensus not reached';
    const scope = diagnostics.budget_scope ? ` (${diagnostics.budget_scope})` : '';
    const candidates = (diagnostics.candidates ?? []).map((candidate) => {
      const label = candidate.common_name
        ? `${candidate.common_name} (${candidate.species})`
        : candidate.species;
      const rejected = candidate.rejected_for_region ? '; rejected for region' : '';
      return `${label} (${formatPercent(candidate.max_score, 0)} best score${rejected})`;
    });
    return candidates.length
      ? `Review required · Tentative: ${candidates.join('; ')} · ${reason}${scope}`
      : `Review required: ${reason}${scope}`;
  }
  if (state === 'candidate') return 'Staged for identification';
  if (state === 'disabled') return '';
  return 'Identifying species…';
}

function filterAndSort(
  tracks: LiveTrack[],
  species: string[],
  identification: string[],
  sort: GallerySort,
): LiveTrack[] {
  const filtered = tracks.filter((track) => {
    const name =
      track.fishial_state === 'identified' ? track.fishial_species : track.species;
    if (species.length > 0 && (!name || !species.includes(name))) return false;
    if (identification.length === 1) {
      const identified = track.fishial_state === 'identified';
      if (identification[0] === 'identified' && !identified) return false;
      if (identification[0] === 'unnamed' && identified) return false;
    }
    return true;
  });

  const duration = (track: LiveTrack) =>
    new Date(track.last_seen_at).getTime() - new Date(track.first_seen_at).getTime();

  return [...filtered].sort((left, right) => {
    if (sort === 'duration') return duration(right) - duration(left);
    if (sort === 'confidence') return right.max_confidence - left.max_confidence;
    return new Date(right.last_seen_at).getTime() - new Date(left.last_seen_at).getTime();
  });
}

function stored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    // A browser can refuse storage entirely; the default camera is still usable.
    return null;
  }
}

function store(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Remembering the choice is a convenience, not a requirement.
  }
}

function text(error: unknown): string {
  return error instanceof Error ? error.message : 'Request failed';
}
