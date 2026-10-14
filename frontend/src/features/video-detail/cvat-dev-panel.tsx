import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import {
  assignCvatTrackSpecies, completeReviewRegion, fetchCvatDevStatus, fetchCvatTracks,
  fetchCvatVideoStatus, fetchReviewQueue, retryCvatSync, undoReviewRegion,
} from '@/api/cvat';
import type { ProposedRegion, ReviewRegion } from '@/api/cvat';
import { fetchSpeciesSearch } from '@/api/species';
import { Button } from '@/components/base/button';

function SpeciesRow({ videoId, trackId, current, regionKey }: {
  videoId: string; trackId: number; current: string | null; regionKey: string;
}) {
  const client = useQueryClient();
  const [search, setSearch] = React.useState('');
  const found = useQuery({
    queryKey: ['cvat-species-search', search],
    queryFn: () => fetchSpeciesSearch(search),
    enabled: search.trim().length >= 3,
  });
  const save = useMutation({
    mutationFn: (species: string) => assignCvatTrackSpecies(videoId, trackId, species),
    onSuccess: () => void client.invalidateQueries({ queryKey: ['cvat-region-tracks', regionKey] }),
  });
  return <div style={{ display: 'grid', gap: 5, padding: 8, border: '1px solid var(--hairline)' }}>
    <span>CVAT track {trackId} · {current ?? 'species not reviewed'}</span>
    <div style={{ display: 'flex', gap: 6 }}>
      <input aria-label={`Search species for CVAT track ${trackId}`} value={search}
        onChange={(event) => setSearch(event.target.value)} placeholder="Search Galway Bay species" />
      <Button size="small" disabled={save.isPending} onClick={() => save.mutate('unknown')}>
        Mark unknown
      </Button>
    </div>
    {found.data?.slice(0, 8).map((entry) => <Button key={entry.species} size="small"
      disabled={save.isPending} onClick={() => save.mutate(entry.species)}>
      {entry.common_name ?? entry.species} · {entry.species}
    </Button>)}
    {save.error ? <span role="alert">{String(save.error)}</span> : null}
  </div>;
}

function ReviewItem({ videoId, proposed, completed }: {
  videoId: string; proposed: ProposedRegion; completed?: ReviewRegion;
}) {
  const client = useQueryClient();
  const [showTracks, setShowTracks] = React.useState(false);
  const [ids, setIds] = React.useState(proposed.kind === 'track_window');
  const [species, setSpecies] = React.useState(false);
  const [minutes, setMinutes] = React.useState(2);
  const regionKey = `${videoId}:${proposed.start_frame}:${proposed.end_frame}`;
  const tracks = useQuery({
    queryKey: ['cvat-region-tracks', regionKey],
    queryFn: () => fetchCvatTracks(videoId, proposed.start_frame, proposed.end_frame),
    enabled: showTracks,
  });
  const complete = useMutation({
    mutationFn: () => completeReviewRegion(videoId, {
      start_frame: proposed.start_frame, end_frame: proposed.end_frame,
      ids_complete: ids, species_complete: species, review_minutes: minutes,
    }),
    onSuccess: () => void client.invalidateQueries({ queryKey: ['cvat-review', videoId] }),
  });
  const undo = useMutation({
    mutationFn: () => undoReviewRegion(videoId, completed?.id ?? ''),
    onSuccess: () => void client.invalidateQueries({ queryKey: ['cvat-review', videoId] }),
  });
  return <div style={{ borderTop: '1px solid var(--hairline)', padding: 'var(--space-2) 0',
    display: 'grid', gap: 6 }}>
    <strong>{proposed.kind === 'frame' ? 'Frame' : 'Track window'} ·
      {' '}{proposed.start_seconds.toFixed(1)}–{proposed.end_seconds.toFixed(1)} s</strong>
    <span className="note">Source frames {proposed.start_frame}–{proposed.end_frame}
      {completed ? ' · certified' : ' · awaiting review'}</span>
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
      <Button size="small" onClick={() => setShowTracks((value) => !value)}>
        {showTracks ? 'Hide tracks' : 'Review species'}
      </Button>
      {completed ? <Button size="small" disabled={undo.isPending}
        onClick={() => undo.mutate()}>Undo certification</Button> : <>
        <label><input type="checkbox" checked={ids}
          onChange={(event) => { setIds(event.target.checked); if (!event.target.checked) setSpecies(false); }} />
          {' '}IDs checked</label>
        <label><input type="checkbox" checked={species} disabled={!ids}
          onChange={(event) => setSpecies(event.target.checked)} /> Species checked</label>
        <label>Minutes <input type="number" min="0.1" max="30" step="0.1"
          value={minutes} onChange={(event) => setMinutes(Number(event.target.value))}
          style={{ width: 65 }} /></label>
        <Button size="small" disabled={complete.isPending || minutes <= 0 ||
          (proposed.kind === 'track_window' && (!ids || !species))}
          onClick={() => complete.mutate()}>Mark boxes complete</Button>
      </>}
    </div>
    {showTracks ? <div style={{ display: 'grid', gap: 5 }}>
      {tracks.isLoading ? <span className="note">Reading CVAT tracks…</span> : null}
      {tracks.data?.tracks.length === 0 ? <span className="note">No fish tracks in this region.</span> : null}
      {tracks.data?.tracks.map((track) => <SpeciesRow key={track.track_id} videoId={videoId}
        trackId={track.track_id} current={track.species} regionKey={regionKey} />)}
      {tracks.error ? <span role="alert">{String(tracks.error)}</span> : null}
    </div> : null}
    {complete.error || undo.error ? <span role="alert">{String(complete.error ?? undo.error)}</span> : null}
    {completed?.ids_complete ? <span className="note">Track identities checked.</span> : null}
    {completed?.species_complete ? <span className="note">Species decisions checked.</span> : null}
  </div>;
}

export function CvatDevPanel({ videoId }: { videoId: string }) {
  const client = useQueryClient();
  const dev = useQuery({ queryKey: ['cvat-dev-status'], queryFn: fetchCvatDevStatus });
  const sync = useQuery({
    queryKey: ['cvat-video', videoId],
    queryFn: () => fetchCvatVideoStatus(videoId),
    enabled: dev.data?.enabled === true,
    refetchInterval: (query) =>
      ['queued', 'uploading', 'importing', 'ready'].includes(query.state.data?.status ?? '')
        ? 5000 : false,
  });
  const review = useQuery({
    queryKey: ['cvat-review', videoId],
    queryFn: () => fetchReviewQueue(videoId),
    enabled: dev.data?.configured === true && !!sync.data?.task_id,
  });
  const retry = useMutation({
    mutationFn: () => retryCvatSync(videoId),
    onSuccess: () => void client.invalidateQueries({ queryKey: ['cvat-video', videoId] }),
  });
  if (!dev.data?.enabled) return null;
  const state = sync.data;
  return <div style={{ display: 'grid', gap: 'var(--space-2)', padding: 'var(--space-3)',
    border: '1px solid var(--hairline)', borderRadius: 'var(--radius-panel)' }}>
    <strong>Development · CVAT review</strong>
    <span className="note">
      {state?.status === 'review' ? 'VIAME suggestions are ready for human correction.'
        : state?.status === 'ready' ? 'Video uploaded. Start VIAME processing to add suggestions.'
        : state?.status === 'needs_review' ? 'CVAT already has annotations; automatic import stopped.'
        : state?.status === 'failed' ? (state.error ?? 'CVAT sync failed.')
        : state?.status === 'uploading' ? 'Uploading video to CVAT…'
        : state?.status === 'importing' ? 'Adding VIAME tracks to CVAT…'
        : state?.status === 'queued' ? 'Waiting to sync with CVAT…'
        : 'This video has not been sent to CVAT.'}
    </span>
    <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
      {state?.task_url ? <a href={state.task_url} target="_blank"
        rel="noopener noreferrer">Open CVAT task</a> : null}
      {dev.data.configured && (!state || ['not_requested', 'failed'].includes(state.status)) ?
        <Button size="small" disabled={retry.isPending} onClick={() => retry.mutate()}>
          {retry.isPending ? 'Syncing…' : 'Send to CVAT'}
        </Button> : null}
    </div>
    {review.data ? <div style={{ display: 'grid', gap: 6 }}>
      <strong>30 minute review queue · {review.data.used_minutes.toFixed(1)} minutes recorded</strong>
      <span className="note">Correct every fish box in CVAT first, including empty frames.
        Check IDs and species across each full track window. Only completed regions become
        training truth; other VIAME suggestions remain unreviewed.</span>
      {review.data.hotspots?.hotspots.length ? <span className="note">
        Inspect possible missed detections, long track gaps, or overlapping IDs near {' '}
        {review.data.hotspots.hotspots.slice(0, 5)
          .map((item) => `${item.seconds.toFixed(1)}s`).join(', ')}.
        These are review candidates, not verified errors.
      </span> : null}
      {review.data.suggested.map((region) => <ReviewItem
        key={`${region.start_frame}:${region.end_frame}`} videoId={videoId} proposed={region}
        completed={review.data.completed.find((item) =>
          item.start_frame === region.start_frame && item.end_frame === region.end_frame)} />)}
    </div> : null}
    {review.error ? <span role="alert">{String(review.error)}</span> : null}
  </div>;
}
