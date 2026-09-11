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
import { EmptyState, PanelError, Skeleton, StatusPill } from '@/components/base/feedback';
import { SingleSelect } from '@/components/base/select';
import { ColumnChart } from '@/components/charts/bar-chart';
import { Page, Section } from '@/components/layout/shell';
import { useToast } from '@/components/base/toast';
import { PlayIcon, StopIcon } from '@/icons';
import { useDocumentVisible } from '@/hooks/use-poll';
import { formatClock, formatCount, formatPercent, plural } from '@/lib/format';

import styles from './live.module.css';

const OPEN_STATUSES = new Set(['queued', 'starting', 'running', 'reconnecting', 'stopping']);

const SOURCE_KEY_STORAGE = 'live-source-key';
const SPECIES_TARGET_STORAGE = 'live-species-fish-target';

type GallerySort = 'recent' | 'duration' | 'confidence';

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
      <Page title="Live monitoring">
        <PanelError
          message={sources.error instanceof Error ? sources.error.message : 'Request failed'}
          onRetry={() => void sources.refetch()}
        />
      </Page>
    );
  }

  return (
    <Page
      title="Live monitoring"
      subtitle="Analyzed camera frames, one camera at a time. Tracks are observations, not unique biological fish."
    >
      <Section title="Camera">
        <div className={styles.controls}>
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
            <div className={styles.speciesControls}>
              <NumberInput
                label="Identify species for the first N fish"
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
              <p className="note" style={{ maxWidth: '34ch' }}>
                0 = off. Sends fish crops to Fishial AI; each frame uses an API call.
              </p>
            </div>
          ) : null}

          <Button
            variant="primary"
            disabled={!enabled || !available || isRunning || start.isPending}
            onClick={() => start.mutate()}
          >
            <PlayIcon />
            Start monitoring
          </Button>
          <Button
            disabled={!isRunning || session?.status === 'stopping' || stop.isPending}
            onClick={() => stop.mutate()}
          >
            <StopIcon />
            Stop
          </Button>

          {isRunning ? (
            <>
              <p className={styles.guard}>Stop monitoring to switch to another camera.</p>
              {fishial?.enabled ? (
                <p className={styles.guard}>Stop monitoring to change species identification.</p>
              ) : null}
            </>
          ) : null}
        </div>

        <p className={styles.statusLine} role="status" aria-live="polite">
          {!enabled ? (
            'Live monitoring is disabled. Enable LIVE_MONITOR_ENABLED on the API and live worker.'
          ) : !available ? (
            'Live monitoring needs the VIAME worker in GPU mode. This deployment uses mock detections.'
          ) : !session ? (
            `Ready to monitor ${sources.data?.sources.find((source) => source.key === effectiveKey)?.label ?? 'the camera'}.`
          ) : (
            <>
              <StatusPill status={isRunning ? 'processing' : 'completed'}>
                {session.status}
              </StatusPill>
              <span className={styles.lag}>
                {session.lag_seconds === null
                  ? 'Waiting for frames'
                  : `${session.lag_seconds.toFixed(1)}s since the latest analyzed frame`}
              </span>
              <span>·</span>
              <span className={styles.lag}>{session.reconnect_count} reconnects</span>
              <span>·</span>
              <span className={styles.lag}>{session.dropped_segments} skipped segments</span>
              {session.status === 'queued' ? <span>· Waiting for the live worker</span> : null}
            </>
          )}
        </p>

        {session?.worker_stale ? (
          <PanelError
            title="The live worker heartbeat is overdue"
            message="Check the live worker service."
          />
        ) : session?.error_message ? (
          <PanelError title="The session reported an error" message={session.error_message} />
        ) : null}
      </Section>

      <div className={styles.monitor}>
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
            <p className={styles.placeholder}>
              {isRunning
                ? 'Waiting for an annotated frame…'
                : 'Start monitoring to view analyzed camera frames.'}
            </p>
          )}
          <p className="note">
            The annotated view updates after each analyzed segment. Processing lag is shown above.
          </p>
        </div>

        <div className={styles.side}>
          {activity.data ? (
            <>
              <dl className={styles.metrics}>
                <Metric value={activity.data.active_tracks} label="Active tracks" />
                <Metric value={activity.data.window_tracks} label="Tracks in window" />
                <Metric value={activity.data.window_detections} label="Detections in window" />
                <Metric value={activity.data.finalized_tracks} label="Finalized" />
              </dl>
              <ColumnChart
                title="Recent fish activity"
                note={`Last ${activity.data.window_seconds}s of analyzed footage`}
                valueLabel="distinct tracks"
                columns={activity.data.series.map((point) => ({
                  label: formatClock(point.at),
                  value: point.tracks,
                }))}
              />
            </>
          ) : session ? (
            <Skeleton height={180} />
          ) : null}
        </div>
      </div>

      {session ? (
        <Section
          title="Finalized fish histories"
          action={<span className="note">{formatCount(galleryTracks.length)} shown</span>}
        >
          <p className="note">
            Fish histories finish after about {session.lost_track_seconds} seconds without a
            detection, or when monitoring stops. Showing the latest 50 histories.
          </p>

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
                { value: 'identified', label: 'Named by Fishial' },
                { value: 'unnamed', label: 'Not named' },
              ]}
            />
            <Segmented
              label="Sort finalized histories"
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
              title="No finalized histories match"
              body={
                (clips.data?.length ?? 0) > 0
                  ? 'Clear the species or identification filter to see the other histories.'
                  : 'Finalized fish clips and crops appear here once a fish leaves the frame.'
              }
            />
          ) : (
            <div className={styles.gallery}>
              {galleryTracks.map((track) => (
                <LiveCard key={track.id} track={track} />
              ))}
            </div>
          )}
        </Section>
      ) : null}

      {session?.species_id.enabled && species.data ? (
        <Section title="Session species breakdown · Fishial AI">
          <div className={styles.speciesList}>
            {species.data.species.map((entry) => (
              <div className={styles.speciesRow} key={entry.species}>
                <span>{entry.species}</span>
                <span className={styles.speciesCount}>{plural(entry.count, 'fish')}</span>
                <span className={styles.speciesCount}>
                  {formatPercent(entry.mean_confidence, 0)} mean
                </span>
              </div>
            ))}
          </div>
          <p className="note">
            {session.species_id.fish_enrolled}/{session.species_id.fish_target} enrolled ·{' '}
            {species.data.review_required} need review ·{' '}
            {species.data.declined > 0 ? `${species.data.declined} not named by the model · ` : ''}
            {species.data.api_calls} Fishial image calls ({species.data.calls_saved} saved)
            {Object.entries(species.data.review_reasons)
              .map(([reason, count]) => ` · ${count} ${reason}`)
              .join('')}
          </p>
        </Section>
      ) : null}
    </Page>
  );
}

function Metric({ value, label }: { value: number; label: string }) {
  return (
    <div className={styles.metric}>
      <dd className={styles.metricValue}>{formatCount(value)}</dd>
      <dt className={styles.metricLabel}>{label}</dt>
    </div>
  );
}

function LiveCard({ track }: { track: LiveTrack }) {
  const name =
    track.fishial_state === 'identified' ? track.fishial_species : (track.species ?? 'Fish');
  return (
    <figure className={styles.card}>
      {track.clip_url ? (
        <video
          src={track.clip_url}
          poster={track.crop_url ?? undefined}
          controls
          loop
          muted
          playsInline
          preload="none"
        />
      ) : track.crop_url ? (
        <img src={track.crop_url} alt={`Annotated crop of ${name}`} loading="lazy" />
      ) : null}
      <figcaption>
        <p className={styles.cardTitle}>
          {name} · <span className="mono">{track.id.slice(0, 8)}</span>
        </p>
        <p className={styles.cardBadge} data-state={track.fishial_state}>
          {badgeFor(track)}
        </p>
        <p className={styles.cardMeta}>
          {track.detection_count} detections · {formatPercent(track.max_confidence, 0)} ·{' '}
          {track.finalization_reason} · {formatClock(track.last_seen_at)}
        </p>
        {track.crop_url ? (
          <AnchorButton variant="link" size="small" href={track.crop_url}>
            Open annotated crop
          </AnchorButton>
        ) : null}
        {track.media_error ? <p className="note">{track.media_error}</p> : null}
      </figcaption>
    </figure>
  );
}

function badgeFor(track: LiveTrack): string {
  const state = track.fishial_state;
  if (state === 'identified') {
    return `${formatPercent(track.fishial_species_confidence ?? 0, 0)} · Fishial AI`;
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
      return `${label} (${formatPercent(candidate.max_score, 0)} best Fishial score${rejected})`;
    });
    return candidates.length
      ? `Review required · Tentative species: ${candidates.join('; ')} · ${reason}${scope}`
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
