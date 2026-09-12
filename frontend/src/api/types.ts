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
  /** A completed live session, recorded and registered as an ordinary video. */
  is_live_recording: boolean;
  source_session_id: string | null;
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

/** What a library track's identification can be. */
export type FishialState =
  | 'none'
  | 'submitted'
  | 'identified'
  | 'review_required'
  | 'error';

/**
 * A live track adds the automatic pass's own states: `candidate` is staged but
 * unpaid, while `pending` and `ready` have been selected and will be paid for.
 */
export type LiveFishialState =
  | FishialState
  | 'disabled'
  | 'candidate'
  | 'pending'
  | 'ready';

/** Diagnostics explaining an abstention, shared by live and recorded fish. */
export interface FishialDiagnostics {
  reason?: string;
  stop_reason?: string;
  budget_scope?: string | null;
  candidates?: {
    species: string;
    common_name?: string | null;
    max_score: number;
    frames?: number;
    rejected_for_region?: boolean;
  }[];
  frame_reasons?: Record<string, number>;
}

/** One fish's Fishial identification: unasked, running, or finished. */
export interface TrackIdentification {
  track_id: string;
  state: LiveFishialState;
  species: string | null;
  confidence: number | null;
  /**
   * `frames_requested` is what the operator asked to pay for; `frames_selected`
   * is what the fish was actually seen doing in enough separate moments to be
   * worth sending, which can be fewer.
   */
  frames_requested: number | null;
  frames_selected: number;
  frames_submitted: number;
  /**
   * Distinct moments the sent frames span. Fewer than `frames_submitted` means some
   * of them share a moment, which is weaker evidence and is shown as such.
   */
  windows: number;
  calls_saved: number;
  quality_score: number | null;
  requested_at: string | null;
  completed_at: string | null;
  stopped_early: string | null;
  /** Set when the reported name is not on this camera's regional species list. */
  implausible_for_region: string | null;
  tally: Record<string, number>;
  diagnostics: FishialDiagnostics | null;
}

/**
 * A stock photo of what a named species looks like - never a detected fish.
 *
 * `unknown` means the server has not looked the name up yet, and a later request
 * may well come back with a photo; `missing` means it looked and there is none.
 */
export interface SpeciesReference {
  species: string;
  slug: string;
  state: 'found' | 'missing' | 'unknown';
  common_name: string | null;
  image_url: string | null;
  attribution: string | null;
  licence: string | null;
  source: string | null;
  source_url: string | null;
}

/**
 * A species an operator may choose, with whatever portrait we already hold.
 *
 * `origin` says where the name came from: `catalogue` is the curated list for
 * these waters, `observed` is a name this deployment has already recorded and can
 * therefore be chosen again even though no curated list holds it.
 */
export interface SpeciesSearchResult extends SpeciesReference {
  origin: 'catalogue' | 'observed';
}

/** One photo in a species' gallery, with the credit it must be shown under. */
export interface SpeciesPhoto {
  url: string;
  media_type: string;
  attribution: string | null;
  licence: string | null;
  source_url: string | null;
}

/** Every reference photo held for one species, the portrait first - never the fish. */
export interface SpeciesGallery {
  species: string;
  slug: string;
  common_name: string | null;
  photos: SpeciesPhoto[];
}

/** Fishial fields carried on every track row, beside the detector's own name. */
export interface FishialFields {
  fishial_state: LiveFishialState;
  fishial_species: string | null;
  fishial_species_confidence: number | null;
}

/**
 * The name a person put on a fish, beside whatever the two machines said.
 *
 * Its own field, never a replacement: a track can carry a VIAME class, a Fishial
 * species and an operator's decision at once, and a reviewer needs to be able to
 * tell which they are reading. Absent means nobody has decided, which is not the
 * same as somebody having agreed.
 */
export interface ManualSpecies {
  manual_species: string | null;
  manual_species_at: string | null;
}

export interface TrackSummary extends FishialFields, ManualSpecies {
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

export interface Track extends FishialFields, ManualSpecies {
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

export interface TrackClip extends FishialFields, ManualSpecies {
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

/** Where to find one fish's preview crop. */
export interface TrackThumbnail {
  track_id: string;
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
  /** Whether an operator may ask Fishial to name one chosen fish, and the limits. */
  species_identification: {
    available: boolean;
    max_frames: number;
    default_frames: number;
  };
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
    request_max_frames: number;
    request_default_frames: number;
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
  /** Calls an operator asked for, counted apart from the session's own budget. */
  manual_api_calls: number;
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
  fishial_state: LiveFishialState;
  fishial_species: string | null;
  fishial_species_confidence: number | null;
  fishial_frames_used: number;
  fishial_votes: Record<string, unknown>;
  fishial_quality_score: number | null;
  fishial_diagnostics: FishialDiagnostics | null;
  manual_species: string | null;
  manual_species_at: string | null;
  identification: TrackIdentification;
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
