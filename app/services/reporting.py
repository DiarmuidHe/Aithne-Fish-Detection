"""Shared reporting semantics. Never serialize ORM objects or job logs into exports."""

from app.db.models import FishTrack, ProcessingJob, Video
from app.services.fish_counter import is_accepted_track


def basename(value):
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def result_job(video: Video) -> ProcessingJob | None:
    completed = [job for job in video.jobs if job.status == "completed"]
    return max(completed, key=lambda j: (j.finished_at or j.created_at, str(j.id)), default=None)


def threshold(video: Video, job: ProcessingJob | None) -> float:
    return job.configuration.get("confidence_threshold", video.confidence_threshold) if job else video.confidence_threshold


def track_accepted(track: FishTrack, video: Video, jobs: dict) -> bool:
    return is_accepted_track(track, threshold(video, jobs.get(track.processing_job_id)))


def provenance(video: Video, job: ProcessingJob | None) -> dict:
    config = job.configuration if job else {}
    return {
        "video_id": video.id, "filename": basename(video.original_filename),
        "video_status": video.processing_status, "content_sha256": video.content_sha256,
        "video_created_at": video.created_at, "video_fps": video.fps,
        "job_id": job.id if job else None,
        "job_created_at": job.created_at if job else None,
        "processing_started_at": job.started_at if job else None,
        "processing_finished_at": job.finished_at if job else None,
        "pipeline": basename(config.get("pipeline", video.pipeline_name)),
        "confidence_threshold": threshold(video, job),
        "model_name": basename(config.get("model_name", video.model_name)),
        "model_version": basename(config.get("model_version", video.model_version)),
        "viame_version": basename(video.viame_version),
        "frame_offset": config.get("frame_number_offset"),
        "downsample_fps": config.get("downsample_fps"),
        "processing_mode": job.worker_mode if job else None,
        "run_command_sha256": config.get("run_command_sha256"),
    }


def track_row(track: FishTrack, video: Video, jobs: dict) -> dict:
    job = jobs.get(track.processing_job_id)
    return {
        **provenance(video, job), "track_id": track.id, "viame_track_id": track.viame_track_id,
        "machine_accepted": track.max_confidence >= threshold(video, job),
        "accepted": track_accepted(track, video, jobs), "review_state": track.review_state,
        "reviewed_at": track.reviewed_at, "first_frame": track.first_frame,
        "last_frame": track.last_frame, "first_timestamp_seconds": track.first_timestamp_seconds,
        "last_timestamp_seconds": track.last_timestamp_seconds,
        "detection_count": track.detection_count, "mean_confidence": track.mean_confidence,
        "max_confidence": track.max_confidence, "species": track.species,
        "species_confidence": track.species_confidence,
    }


# --- Review vocabulary -------------------------------------------------------
# Defined once here so exports, the API and the dashboard cannot drift apart.

DEFAULT_BORDERLINE_BAND = 0.10

# A track can hold more than one of these at once: an unreviewed track sitting
# next to the cut-off is both "unreviewed" and "borderline".
REVIEW_CATEGORIES = ("flagged", "unreviewed", "borderline", "disputed", "done")

DONE_REVIEW_STATES = frozenset({"reviewed", "accepted", "rejected"})


def review_categories(track, run_threshold: float, band: float = DEFAULT_BORDERLINE_BAND) -> set:
    """Which review categories a track belongs to, against the threshold its run used."""

    state = getattr(track, "review_state", "unreviewed")
    confidence = track.max_confidence
    found = set()
    if state == "needs-review":
        found.add("flagged")
    if state == "unreviewed":
        found.add("unreviewed")
        # Near its own cut-off, so a human decision would change the count.
        if abs(confidence - run_threshold) <= band:
            found.add("borderline")
    if state in DONE_REVIEW_STATES:
        found.add("done")
    if (state == "accepted" and confidence < run_threshold) or (
        state == "rejected" and confidence >= run_threshold
    ):
        found.add("disputed")
    return found


def track_review_categories(track, video: Video, jobs: dict,
                            band: float = DEFAULT_BORDERLINE_BAND) -> set:
    """Categories for a stored track, using its own run's threshold snapshot."""

    return review_categories(track, threshold(video, jobs.get(track.processing_job_id)), band)


def track_summary_row(track: FishTrack, video: Video, jobs: dict,
                      band: float = DEFAULT_BORDERLINE_BAND) -> dict:
    """One track table row, judged against the threshold its own run used."""

    run_threshold = threshold(video, jobs.get(track.processing_job_id))
    return {
        "id": track.id,
        "video_id": track.video_id,
        "processing_job_id": track.processing_job_id,
        "review_state": track.review_state,
        "reviewed_at": track.reviewed_at,
        "machine_accepted": track.max_confidence >= run_threshold,
        "viame_track_id": track.viame_track_id,
        "first_frame": track.first_frame,
        "last_frame": track.last_frame,
        "first_timestamp_seconds": track.first_timestamp_seconds,
        "last_timestamp_seconds": track.last_timestamp_seconds,
        "detection_count": track.detection_count,
        "mean_confidence": track.mean_confidence,
        "max_confidence": track.max_confidence,
        "species": track.species,
        "species_confidence": track.species_confidence,
        "accepted": is_accepted_track(track, run_threshold),
        "run_threshold": run_threshold,
        "review_categories": sorted(review_categories(track, run_threshold, band)),
    }
