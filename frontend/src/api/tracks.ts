import { patchJson, postJson, request } from './client';
import type { ReviewState, Track, TrackClip, TrackIdentification, TrackSummary } from './types';

export function fetchTrack(trackId: string): Promise<Track> {
  return request<Track>(`/tracks/${encodeURIComponent(trackId)}`);
}

export function reviewTrack(trackId: string, reviewState: ReviewState): Promise<Track> {
  return patchJson<Track>(`/tracks/${encodeURIComponent(trackId)}/review`, {
    review_state: reviewState,
  });
}

/** The queue's "accept all visible" and its undo are both one request. */
export function reviewTracks(
  trackIds: string[],
  reviewState: ReviewState,
): Promise<TrackSummary[]> {
  return postJson<TrackSummary[]>('/tracks/review', {
    track_ids: trackIds,
    review_state: reviewState,
  });
}

/**
 * Record the name a person decided this fish carries, or `null` to take it back.
 *
 * Its own field: the detector's class and Fishial's answer are both left alone, so
 * clearing this puts the machine's name back without anything having been lost.
 */
export function assignTrackSpecies(trackId: string, species: string | null): Promise<Track> {
  return patchJson<Track>(`/tracks/${encodeURIComponent(trackId)}/species`, { species });
}

export const BULK_REVIEW_LIMIT = 200;

export function generateTrackClip(trackId: string): Promise<TrackClip> {
  return request<TrackClip>(`/tracks/${encodeURIComponent(trackId)}/clip`, { method: 'POST' });
}

export function trackClipUrl(trackId: string, version?: string, download = false): string {
  const params = new URLSearchParams();
  if (version) params.set('v', version);
  if (download) params.set('download', 'true');
  const query = params.toString();
  return `/tracks/${encodeURIComponent(trackId)}/clip${query ? `?${query}` : ''}`;
}

/**
 * Ask Fishial to name one chosen fish. Returns as soon as the fish is claimed;
 * the frames are chosen, sent and voted on afterwards, so poll the identification
 * until it leaves `submitted`.
 */
export function identifyTrack(trackId: string, frames: number): Promise<TrackIdentification> {
  return postJson<TrackIdentification>(`/tracks/${encodeURIComponent(trackId)}/identify`, {
    frames,
  });
}

export function fetchTrackIdentification(trackId: string): Promise<TrackIdentification> {
  return request<TrackIdentification>(`/tracks/${encodeURIComponent(trackId)}/identification`);
}
