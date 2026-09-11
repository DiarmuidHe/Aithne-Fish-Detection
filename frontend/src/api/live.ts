import { postJson, request, toQuery } from './client';
import type {
  LiveActivity,
  LiveLatest,
  LiveSession,
  LiveSources,
  LiveSpecies,
  LiveTrack,
} from './types';

export function fetchLiveSources(): Promise<LiveSources> {
  return request<LiveSources>('/live/sources');
}

export function fetchLiveLatest(sourceKey: string): Promise<LiveLatest> {
  return request<LiveLatest>(`/live/latest${toQuery({ source: sourceKey })}`);
}

/**
 * Cameras are chosen by key. The client never sends a stream URL: the resolver
 * would become an SSRF sink if it accepted one.
 */
export function startLive(
  sourceKey: string,
  speciesTarget: number,
  framesPerFish: number,
): Promise<LiveSession> {
  return postJson<LiveSession>('/live/start', {
    source: sourceKey,
    species_id_fish_target: speciesTarget,
    species_id_frames_per_fish: framesPerFish,
  });
}

export function stopLive(sessionId: string): Promise<LiveSession> {
  return request<LiveSession>(`/live/${encodeURIComponent(sessionId)}/stop`, { method: 'POST' });
}

export function fetchLiveStatus(sessionId: string): Promise<LiveSession> {
  return request<LiveSession>(`/live/${encodeURIComponent(sessionId)}/status`);
}

export function fetchLiveActivity(sessionId: string): Promise<LiveActivity> {
  return request<LiveActivity>(`/live/${encodeURIComponent(sessionId)}/activity`);
}

export function fetchLiveClips(sessionId: string): Promise<LiveTrack[]> {
  return request<LiveTrack[]>(`/live/${encodeURIComponent(sessionId)}/clips`);
}

export function fetchLiveSpecies(sessionId: string): Promise<LiveSpecies> {
  return request<LiveSpecies>(`/live/${encodeURIComponent(sessionId)}/species`);
}

export function liveSnapshotUrl(streamUrl: string, version: string | null): string {
  return `${streamUrl}?snapshot=true&t=${encodeURIComponent(version ?? '')}`;
}
