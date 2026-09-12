import { postJson, request, requestWithResponse, toQuery, type QueryValue } from './client';
import type {
  BatchResponse,
  LibraryAnalyticsRow,
  LiveSession,
  TrackClip,
  TrackSummary,
  TrackThumbnail,
  Video,
  VideoAnalytics,
  VideoAnnotation,
  VideoFacets,
  VideoListRow,
  VideoPage,
  VideoSummary,
} from './types';

export type ExportKind = 'summary' | 'accepted-tracks' | 'all-tracks' | 'detections';

export const EXPORT_KINDS: { kind: ExportKind; label: string }[] = [
  { kind: 'summary', label: 'Summaries' },
  { kind: 'accepted-tracks', label: 'Accepted tracks' },
  { kind: 'all-tracks', label: 'All tracks' },
  { kind: 'detections', label: 'Observations' },
];

/** The list stays a bare array; the totals arrive as headers. */
export async function fetchVideos(
  params: Record<string, QueryValue | QueryValue[]>,
): Promise<VideoPage> {
  const { data, response } = await requestWithResponse<VideoListRow[]>(
    `/videos${toQuery(params)}`,
  );
  return {
    rows: data,
    total: Number(response.headers.get('X-Total-Count') ?? data.length),
    filtered: Number(response.headers.get('X-Filtered-Count') ?? data.length),
  };
}

export function fetchFacets(): Promise<VideoFacets> {
  return request<VideoFacets>('/videos/facets');
}

export function fetchVideo(videoId: string): Promise<Video> {
  return request<Video>(`/videos/${encodeURIComponent(videoId)}`);
}

/** The live session a recording came from; 404 for an ordinary upload. */
export function fetchSourceSession(videoId: string): Promise<LiveSession> {
  return request<LiveSession>(`/videos/${encodeURIComponent(videoId)}/source-session`);
}

export function fetchSummary(videoId: string): Promise<VideoSummary> {
  return request<VideoSummary>(`/videos/${encodeURIComponent(videoId)}/summary`);
}

export function fetchTrackSummaries(
  videoId: string,
  params: Record<string, QueryValue | QueryValue[]> = {},
): Promise<TrackSummary[]> {
  return request<TrackSummary[]>(
    `/videos/${encodeURIComponent(videoId)}/track-summaries${toQuery(params)}`,
  );
}

export function fetchVideoAnalytics(videoId: string): Promise<VideoAnalytics> {
  return request<VideoAnalytics>(`/analytics/videos/${encodeURIComponent(videoId)}`);
}

export function fetchLibraryAnalytics(): Promise<LibraryAnalyticsRow[]> {
  return request<LibraryAnalyticsRow[]>('/analytics/videos');
}

export function startProcessing(videoId: string) {
  return request(`/videos/${encodeURIComponent(videoId)}/process`, { method: 'POST' });
}

export function annotateVideo(videoId: string): Promise<VideoAnnotation> {
  return request<VideoAnnotation>(`/videos/${encodeURIComponent(videoId)}/annotate`, {
    method: 'POST',
  });
}

export function fetchFishClips(videoId: string): Promise<TrackClip[]> {
  return request<TrackClip[]>(`/videos/${encodeURIComponent(videoId)}/fish-clips`);
}

export function generateFishClips(videoId: string): Promise<TrackClip[]> {
  return request<TrackClip[]>(`/videos/${encodeURIComponent(videoId)}/fish-clips`, {
    method: 'POST',
  });
}

/**
 * Every fish preview for a video, generating any that are missing. Asked once per
 * video rather than once per row: the server reads the source in a single pass, so
 * one request costs barely more than one thumbnail would.
 */
export function fetchTrackThumbnails(videoId: string): Promise<TrackThumbnail[]> {
  return request<TrackThumbnail[]>(`/videos/${encodeURIComponent(videoId)}/thumbnails`);
}

export function uploadVideo(file: File, cameraId: string): Promise<Video> {
  const body = new FormData();
  body.append('file', file);
  if (cameraId.trim()) body.append('camera_id', cameraId.trim());
  return request<Video>('/videos', { method: 'POST', body });
}

export function runBatch(action: 'process' | 'annotate', videoIds: string[]) {
  return postJson<BatchResponse>(`/batch/${action}`, { video_ids: videoIds });
}

export function videoExportUrl(videoId: string, kind: ExportKind): string {
  return `/videos/${encodeURIComponent(videoId)}/exports/${kind}.csv`;
}

export function batchExportUrl(kind: ExportKind, videoIds?: string[]): string {
  return `/exports/batch.csv${toQuery({ kind, video_ids: videoIds })}`;
}

export function annotatedVideoUrl(videoId: string, annotatedAt: string, download = false): string {
  return `/videos/${encodeURIComponent(videoId)}/annotated-video${toQuery({
    v: annotatedAt,
    download: download || undefined,
  })}`;
}

export function sourceVideoUrl(videoId: string): string {
  return `/videos/${encodeURIComponent(videoId)}/source-video`;
}
