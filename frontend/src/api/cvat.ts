import { request } from './client';

export interface CvatDevStatus {
  enabled: boolean;
  configured: boolean;
  public_url: string | null;
}

export interface CvatVideoStatus {
  status: 'not_requested' | 'queued' | 'uploading' | 'ready' | 'importing' | 'review' | 'needs_review' | 'failed';
  task_id: number | null;
  task_url: string | null;
  error: string | null;
}

export const fetchCvatDevStatus = () => request<CvatDevStatus>('/cvat/status');
export const fetchCvatVideoStatus = (videoId: string) =>
  request<CvatVideoStatus>(`/cvat/videos/${encodeURIComponent(videoId)}`);
export const retryCvatSync = (videoId: string) =>
  request(`/cvat/videos/${encodeURIComponent(videoId)}/sync`, { method: 'POST' });

export interface ReviewRegion {
  id: string;
  start_frame: number;
  end_frame: number;
  boxes_complete: boolean;
  ids_complete: boolean;
  species_complete: boolean;
  review_minutes: number;
  completed_at: string;
}

export interface ProposedRegion {
  start_frame: number;
  end_frame: number;
  start_seconds: number;
  end_seconds: number;
  kind: 'frame' | 'track_window';
}

export interface ReviewQueue {
  task_id: number;
  fps: number;
  frame_count: number;
  budget_minutes: number;
  used_minutes: number;
  suggested: ProposedRegion[];
  completed: ReviewRegion[];
  hotspots: {
    overlap_frames: number;
    unmatched_candidate_frames: number;
    hotspots: { frame: number; seconds: number; overlapping_ids: number;
      unmatched_detector_boxes: number; long_track_gaps: number }[];
  } | null;
}

const videoPath = (videoId: string) => `/cvat/videos/${encodeURIComponent(videoId)}`;
export const fetchReviewQueue = (videoId: string) =>
  request<ReviewQueue>(`${videoPath(videoId)}/review`);
export const fetchCvatTracks = (videoId: string, start: number, end: number) =>
  request<{ tracks: { track_id: number; species: string | null }[] }>(
    `${videoPath(videoId)}/tracks?start_frame=${start}&end_frame=${end}`);
export const completeReviewRegion = (
  videoId: string, region: { start_frame: number; end_frame: number;
    ids_complete: boolean; species_complete: boolean; review_minutes: number },
) => request<ReviewRegion>(`${videoPath(videoId)}/review-regions`,
  { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(region) });
export const undoReviewRegion = (videoId: string, regionId: string) =>
  request(`${videoPath(videoId)}/review-regions/${regionId}`, { method: 'DELETE' });
export const assignCvatTrackSpecies = (videoId: string, trackId: number, species: string) =>
  request<{ track_id: number; species: string }>(
    `${videoPath(videoId)}/tracks/${trackId}/species`,
    { method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ species }) });
