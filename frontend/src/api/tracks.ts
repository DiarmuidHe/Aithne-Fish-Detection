import { patchJson, postJson, request } from './client';
import type { ReviewState, Track, TrackClip, TrackSummary } from './types';

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
