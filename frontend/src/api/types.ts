/** Shapes returned by the FastAPI service. Kept in step with app/schemas. */

export type ProcessingStatus =
  | 'uploaded'
  | 'queued'
  | 'processing'
  | 'completed'
  | 'failed';

export type ReviewState =
  | 'unreviewed'
  | 'reviewed'
  | 'accepted'
  | 'rejected'
  | 'needs-review';

export type ReviewCategory =
  | 'flagged'
  | 'unreviewed'
  | 'borderline'
  | 'disputed'
  | 'done';

export type ReviewStatus = 'n/a' | 'awaiting' | 'in-progress' | 'complete';

export interface ProcessingConfiguration {
  worker_mode: string | null;
  pipeline: string | null;
  confidence_threshold: number | null;
  model_name: string | null;
  model_version: string | null;
  viame_version: string | null;
  downsample_fps: number | null;
  frame_number_offset: number | null;
  run_command_sha256: string | null;
}

export interface Job {
  id: string;
  video_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  worker_mode: string;
  heartbeat_at: string | null;
  configuration: ProcessingConfiguration;
}

export interface Video {
  id: string;
  original_filename: string;
  storage_path: string;
  camera_id: string | null;
  created_at: string;
  processing_status: ProcessingStatus;
  fps: number | null;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
  codec: string | null;
  size_bytes: number | null;
  content_sha256: string | null;
  viame_version: string | null;
  model_name: string;
  model_version: string | null;
  pipeline_name: string;
  confidence_threshold: number;
  annotated_at: string | null;
  latest_job: Job | null;
}

export interface VideoListRow extends Video {
  track_count: number;
  accepted_track_count: number;
  detection_count: number;
  unreviewed_count: number;
  flagged_count: number;
  disputed_count: number;
  review_status: ReviewStatus;
  has_annotation: boolean;
  clip_count: number;
  species: string[];
}

export interface VideoPage {
  rows: VideoListRow[];
  /** Every video in the library, before any filter. */
  total: number;
  /** Videos matching the current filters, before paging. */
  filtered: number;
}

export interface VideoFacets {
  cameras: string[];
  species: string[];
  status: Record<string, number>;
  review_status: Record<string, number>;
  total: number;
}

export interface VideoSummary {
  video_id: string;
  status: string;
  fish_tracks: number;
  total_detections: number;
  mean_track_confidence: number | null;
  first_fish_timestamp_seconds: number | null;
  last_fish_timestamp_seconds: number | null;
  pipeline_name: string;
  confidence_threshold: number;
  model_name: string;
  model_version: string | null;
  viame_version: string | null;
}

export interface TrackSummary {
  id: string;
  video_id: string;
  processing_job_id: string | null;
  review_state: ReviewState;
  reviewed_at: string | null;
  machine_accepted: boolean;
  viame_track_id: string;
  first_frame: number;
  last_frame: number;
  first_timestamp_seconds: number | null;
  last_timestamp_seconds: number | null;
  detection_count: number;
  mean_confidence: number;
  max_confidence: number;
  species: string | null;
  species_confidence: number | null;
  accepted: boolean;
  run_threshold: number;
  review_categories: ReviewCategory[];
}

export interface Detection {
  id: string;
  frame_number: number;
  timestamp_seconds: number | null;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_name: string | null;
  class_confidence: number | null;
}

export interface Track {
  id: string;
  video_id: string;
  processing_job_id: string | null;
  review_state: ReviewState;
  reviewed_at: string | null;
  viame_track_id: string;
  first_frame: number;
  last_frame: number;
  first_timestamp_seconds: number | null;
  last_timestamp_seconds: number | null;
  detection_count: number;
  mean_confidence: number;
  max_confidence: number;
  species: string | null;
  species_confidence: number | null;
  detections: Detection[];
}

export interface TrackClip {
  track_id: string;
  video_id: string;
  viame_track_id: string;
  species: string | null;
  filename: string;
  media_type: string;
  size_bytes: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
  detection_count: number;
  max_confidence: number;
  generated_at: string;
  cached: boolean;
  url: string;
}

export interface VideoAnnotation {
  video_id: string;
  annotated_at: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  url: string;
}

export interface SystemStatus {
  ready: boolean;
  processing_mode: string;
  compose_command: string;
  database: { available: boolean; message: string };
  worker: {
    available: boolean;
    expected_mode: string;
    active_workers: number;
    other_mode_workers: number;
    current_jobs: number;
    last_seen_at: string | null;
    seconds_since_last_seen: number | null;
    message: string;
  };
  queue: { queued: number; processing: number; failed: number };
}

export interface HistogramBin {
  start: number;
  end: number;
  count: number;
}

export interface TimeBin {
  start: number;
  end: number;
  all_detections: number;
  accepted_detections: number;
}

export interface VideoAnalytics {
  video_id: string;
  status: string;
  time_bins: TimeBin[];
  confidence_distribution: HistogramBin[];
  track_duration_distribution: HistogramBin[];
  unknown_timestamp_detections: number;
  unknown_duration_tracks: number;
  all_track_count: number;
  accepted_fish_count: number;
}

export interface LibraryAnalyticsRow {
  video_id: string;
  filename: string;
  status: string;
  accepted_fish_count: number;
  accepted_detections: number;
}

export interface BatchResult {
  video_id: string;
  ok: boolean;
  status: string;
  job_id?: string;
  url?: string;
  error?: string;
}

export interface BatchResponse {
  results: BatchResult[];
  total: number;
  succeeded: number;
  failed: number;
}

/* --- Live monitoring ------------------------------------------------------ */

export interface LiveSource {
  key: string;
  label: string;
  location: string | null;
  url: string;
  active_session_id: string | null;
}

export interface LiveSources {
  default_key: string;
  enabled: boolean;
  available: boolean;
  fishial: {
    enabled: boolean;
    max_fish_per_session: number;
    default_frames_per_fish: number;
  };
  sources: LiveSource[];
}

export interface LiveSpeciesProgress {
  enabled: boolean;
  fish_target: number;
  frames_per_fish: number;
  fish_enrolled: number;
  fish_identified: number;
  fish_review_required: number;
  api_calls: number;
  candidates: number;
  calls_saved: number;
  region: string | null;
}

export interface LiveSession {
  id: string;
  status: string;
  created_at: string;
  started_at: string | null;
  stopped_at: string | null;
  last_frame_at: string | null;
  heartbeat_at: string | null;
  worker_stale: boolean;
  error_message: string | null;
  frames_processed: number;
  dropped_segments: number;
  reconnect_count: number;
  lag_seconds: number | null;
  lost_track_seconds: number;
  annotated_stream_url: string | null;
  source_key: string;
  source_label: string;
  species_id: LiveSpeciesProgress;
}

export interface LiveLatest {
  enabled: boolean;
  available: boolean;
  source_key: string;
  session: LiveSession | null;
}

export interface LiveTrack {
  id: string;
  session_id: string;
  status: string;
  first_seen_at: string;
  last_seen_at: string;
  finalized_at: string | null;
  finalization_reason: string | null;
  detection_count: number;
  max_confidence: number;
  mean_confidence: number;
  species: string | null;
  bbox: [number, number, number, number];
  clip_url: string | null;
  crop_url: string | null;
  media_error: string | null;
  fishial_state: string;
  fishial_species: string | null;
  fishial_species_confidence: number | null;
  fishial_frames_used: number;
  fishial_votes: Record<string, unknown>;
  fishial_quality_score: number | null;
  fishial_diagnostics: {
    reason?: string;
    stop_reason?: string;
    budget_scope?: string;
    candidates?: {
      species: string;
      common_name?: string;
      max_score: number;
      rejected_for_region?: boolean;
    }[];
    frame_reasons?: Record<string, number>;
  } | null;
}

export interface LiveActivity {
  session_id: string;
  as_of: string;
  window_seconds: number;
  active_tracks: number;
  finalized_tracks: number;
  window_tracks: number;
  window_detections: number;
  series: { at: string; tracks: number }[];
}

export interface LiveSpecies {
  session_id: string;
  as_of: string;
  api_calls: number;
  review_required: number;
  calls_saved: number;
  declined: number;
  review_reasons: Record<string, number>;
  species: { species: string; count: number; mean_confidence: number }[];
}
