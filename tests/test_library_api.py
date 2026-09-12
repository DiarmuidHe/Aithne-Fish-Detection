"""Coverage for the library list: aggregates, review vocabulary, filters, facets, bulk review."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import select

from app.db.models import FishTrack, ProcessingJob, Video
from app.services.track_clip import clip_paths
from app.workers.processing_worker import process_job


def upload(client, name="sample.mp4", camera_id=None):
    data = {"camera_id": camera_id} if camera_id else None
    response = client.post("/videos", files={"file": (name, b"video", "video/mp4")}, data=data)
    assert response.status_code == 201
    return response.json()["id"]


def complete(client, db_session_factory, test_settings, name="sample.mp4", camera_id=None):
    """Upload a video and run it through the mock worker.

    The fixture yields three tracks with max confidences 0.94, 0.55 and 0.89
    against a run threshold of 0.60, so one track sits inside the default
    borderline band and two are accepted.
    """

    video_id = upload(client, name, camera_id)
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        video.fps = 10
        db.commit()
    job = client.post(f"/videos/{video_id}/process").json()
    process_job(UUID(job["id"]), settings=test_settings, session_factory=db_session_factory)
    return video_id


def tracks_of(db_session_factory, video_id, order_by_confidence=True):
    with db_session_factory() as db:
        rows = list(db.scalars(select(FishTrack).where(FishTrack.video_id == UUID(video_id))))
    if order_by_confidence:
        rows.sort(key=lambda track: track.max_confidence)
    return rows


def listing(client, **params):
    response = client.get("/videos", params=params)
    assert response.status_code == 200, response.text
    return response


def ids(response):
    return [row["id"] for row in response.json()]


# --- Aggregates --------------------------------------------------------------


def test_list_reports_track_aggregates_for_a_completed_video(client, db_session_factory,
                                                             test_settings):
    video_id = complete(client, db_session_factory, test_settings)

    row = listing(client).json()[0]

    assert row["id"] == video_id
    assert row["track_count"] == 3
    assert row["accepted_track_count"] == 2
    assert row["detection_count"] == 5
    assert row["unreviewed_count"] == 3
    assert row["flagged_count"] == 0
    assert row["disputed_count"] == 0
    assert row["review_status"] == "awaiting"
    assert row["has_annotation"] is False
    assert row["clip_count"] == 0
    assert row["species"] == ["fish"]
    # The pre-existing shape is still intact for callers that predate this change.
    assert "/" not in row["storage_path"] and "\\" not in row["storage_path"]
    assert row["latest_job"]["status"] == "completed"


def test_an_unprocessed_video_reports_no_review_work(client):
    upload(client)

    row = listing(client).json()[0]

    assert row["review_status"] == "n/a"
    assert row["track_count"] == 0
    assert row["accepted_track_count"] == 0
    assert row["species"] == []


def test_list_stays_a_bare_array_and_puts_the_totals_in_headers(client, db_session_factory,
                                                                test_settings):
    complete(client, db_session_factory, test_settings, name="kept.mp4")
    upload(client, "other.mp4")

    response = listing(client, q="kept")

    assert isinstance(response.json(), list)
    assert response.headers["X-Total-Count"] == "2"
    assert response.headers["X-Filtered-Count"] == "1"


def test_aggregates_do_not_grow_a_query_per_listed_video(client, db_session_factory,
                                                         test_settings, monkeypatch):
    for index in range(3):
        complete(client, db_session_factory, test_settings, name=f"clip-{index}.mp4")
    from app.services import library

    calls = []
    original = library.aggregate_videos

    def counting(db, videos, settings, band=0.10, include_clips=True):
        calls.append(len(videos))
        return original(db, videos, settings, band, include_clips)

    monkeypatch.setattr(library, "aggregate_videos", counting)
    monkeypatch.setattr("app.api.videos.aggregate_videos", counting)

    assert len(listing(client).json()) == 3
    # One aggregation pass covering every candidate, not one per row.
    assert calls == [3]


# --- Review vocabulary -------------------------------------------------------


def test_borderline_uses_the_band_around_the_threshold(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)

    # 0.55 is 0.05 below the 0.60 run threshold, so it is inside the default band.
    assert ids(listing(client, review="borderline")) == [video_id]
    assert ids(listing(client, review="borderline", band=0.01)) == []
    assert ids(listing(client, review="borderline", band=0.4)) == [video_id]


def test_borderline_is_measured_against_the_run_threshold_not_the_current_one(
    client, db_session_factory, test_settings
):
    video_id = complete(client, db_session_factory, test_settings)
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        # A later edit to the video must not reinterpret a finished run.
        video.confidence_threshold = 0.20
        db.commit()

    assert ids(listing(client, review="borderline")) == [video_id]

    with db_session_factory() as db:
        job = db.scalars(select(ProcessingJob).where(ProcessingJob.video_id == UUID(video_id))).one()
        configuration = json.loads(job.configuration_json)
        configuration["confidence_threshold"] = 0.20
        job.configuration_json = json.dumps(configuration)
        db.commit()

    # Now the run itself says 0.20, and 0.55 is no longer near the cut-off.
    assert ids(listing(client, review="borderline")) == []


def test_disputed_covers_both_directions_of_disagreement(client, db_session_factory,
                                                         test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    low, _, high = tracks_of(db_session_factory, video_id)

    assert ids(listing(client, review="disputed")) == []

    # Accepting a track the detector put below its threshold.
    client.patch(f"/tracks/{low.id}/review", json={"review_state": "accepted"})
    assert ids(listing(client, review="disputed")) == [video_id]
    assert listing(client).json()[0]["disputed_count"] == 1

    client.patch(f"/tracks/{low.id}/review", json={"review_state": "unreviewed"})
    assert ids(listing(client, review="disputed")) == []

    # Rejecting one it put above.
    client.patch(f"/tracks/{high.id}/review", json={"review_state": "rejected"})
    assert ids(listing(client, review="disputed")) == [video_id]


def test_flagged_unreviewed_and_done_partition_the_review_words(client, db_session_factory,
                                                                test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    low, middle, high = tracks_of(db_session_factory, video_id)

    assert ids(listing(client, review="unreviewed")) == [video_id]
    assert ids(listing(client, review="flagged")) == []
    assert ids(listing(client, review="done")) == []

    client.patch(f"/tracks/{middle.id}/review", json={"review_state": "needs-review"})
    client.patch(f"/tracks/{high.id}/review", json={"review_state": "reviewed"})

    row = listing(client).json()[0]
    assert row["flagged_count"] == 1
    assert row["unreviewed_count"] == 1
    assert ids(listing(client, review="flagged")) == [video_id]
    assert ids(listing(client, review="done")) == [video_id]


def test_review_filter_values_are_a_union(client, db_session_factory, test_settings):
    flagged_video = complete(client, db_session_factory, test_settings, name="flagged.mp4")
    other = complete(client, db_session_factory, test_settings, name="other.mp4")
    track = tracks_of(db_session_factory, flagged_video)[0]
    client.patch(f"/tracks/{track.id}/review", json={"review_state": "needs-review"})

    matched = set(ids(listing(client, review=["flagged", "unreviewed"])))

    assert matched == {flagged_video, other}


def test_review_status_moves_from_awaiting_through_to_complete(client, db_session_factory,
                                                               test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    low, middle, high = tracks_of(db_session_factory, video_id)
    assert listing(client).json()[0]["review_status"] == "awaiting"

    client.patch(f"/tracks/{low.id}/review", json={"review_state": "accepted"})
    assert listing(client).json()[0]["review_status"] == "in-progress"

    client.patch(f"/tracks/{middle.id}/review", json={"review_state": "needs-review"})
    client.patch(f"/tracks/{high.id}/review", json={"review_state": "reviewed"})
    # A flagged track is outstanding work, so this is not complete yet.
    assert listing(client).json()[0]["review_status"] == "in-progress"

    client.patch(f"/tracks/{middle.id}/review", json={"review_state": "rejected"})
    assert listing(client).json()[0]["review_status"] == "complete"


# --- Filters -----------------------------------------------------------------


def test_search_matches_filename_and_camera_case_insensitively(client):
    named = upload(client, "Reef-Survey.mp4")
    camera = upload(client, "other.mp4", camera_id="River camera 2")

    assert ids(listing(client, q="reef")) == [named]
    assert ids(listing(client, q="RIVER")) == [camera]
    assert ids(listing(client, q="nothing")) == []


def test_status_filter_accepts_the_stored_word_and_its_dashboard_alias(client):
    video_id = upload(client)

    assert ids(listing(client, status="uploaded")) == [video_id]
    assert ids(listing(client, status="pending")) == [video_id]
    assert ids(listing(client, status="completed")) == []


def test_unknown_enum_values_are_rejected(client):
    assert client.get("/videos", params={"status": "archived"}).status_code == 422
    assert client.get("/videos", params={"review": "maybe"}).status_code == 422
    assert client.get("/videos", params={"sort": "colour"}).status_code == 422
    assert client.get("/videos", params={"order": "sideways"}).status_code == 422
    assert client.get("/videos", params={"band": 1.5}).status_code == 422
    assert client.get("/videos", params={"limit": 500}).status_code == 422


def test_camera_filter_selects_named_cameras_and_the_absence_of_one(client):
    with_camera = upload(client, "a.mp4", camera_id="cam-1")
    other_camera = upload(client, "b.mp4", camera_id="cam-2")
    without = upload(client, "c.mp4")

    assert ids(listing(client, camera_id="cam-1")) == [with_camera]
    assert ids(listing(client, camera_id="__none__")) == [without]
    assert set(ids(listing(client, camera_id=["cam-2", "__none__"]))) == {other_camera, without}


def test_annotated_and_clip_filters(client, db_session_factory, test_settings):
    plain = complete(client, db_session_factory, test_settings, name="plain.mp4")
    annotated = complete(client, db_session_factory, test_settings, name="annotated.mp4")
    with db_session_factory() as db:
        db.get(Video, UUID(annotated)).annotated_at = datetime.now(timezone.utc)
        db.commit()

    assert ids(listing(client, annotated=True)) == [annotated]
    assert ids(listing(client, annotated=False)) == [plain]
    assert listing(client, annotated=True).json()[0]["has_annotation"] is True

    track = tracks_of(db_session_factory, plain)[0]
    clip_path, metadata_path = clip_paths(UUID(plain), track.id, test_settings)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"clip")
    metadata_path.write_text("{}", encoding="utf-8")

    assert ids(listing(client, has_clips=True)) == [plain]
    assert ids(listing(client, has_clips=False)) == [annotated]
    assert listing(client, has_clips=True).json()[0]["clip_count"] == 1


def test_species_filter_matches_accepted_tracks_only(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    low = tracks_of(db_session_factory, video_id)[0]
    with db_session_factory() as db:
        db.get(FishTrack, low.id).species = "Gadus morhua"
        db.commit()

    # The cod track sits below the threshold, so it is not an accepted species yet.
    assert ids(listing(client, species="Gadus morhua")) == []
    assert listing(client).json()[0]["species"] == ["fish"]

    client.patch(f"/tracks/{low.id}/review", json={"review_state": "accepted"})
    assert ids(listing(client, species="Gadus morhua")) == [video_id]
    assert listing(client).json()[0]["species"] == ["Gadus morhua", "fish"]


def test_fish_count_duration_and_date_ranges(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    created = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        video.duration_seconds = 42.0
        video.created_at = created
        db.commit()

    assert ids(listing(client, min_fish=2)) == [video_id]
    assert ids(listing(client, min_fish=3)) == []
    assert ids(listing(client, max_fish=1)) == []
    assert ids(listing(client, min_duration=40, max_duration=50)) == [video_id]
    assert ids(listing(client, min_duration=50)) == []
    assert ids(listing(client, created_after=(created - timedelta(days=1)).isoformat())) == [video_id]
    assert ids(listing(client, created_before=(created - timedelta(days=1)).isoformat())) == []


# --- Sorting and paging ------------------------------------------------------


def test_sorting_by_filename_in_both_directions(client):
    upload(client, "b.mp4")
    upload(client, "a.mp4")
    upload(client, "c.mp4")

    names = [row["original_filename"] for row in listing(client, sort="filename", order="asc").json()]
    assert names == ["a.mp4", "b.mp4", "c.mp4"]
    names = [row["original_filename"] for row in listing(client, sort="filename", order="desc").json()]
    assert names == ["c.mp4", "b.mp4", "a.mp4"]


@pytest.mark.parametrize(
    "sort",
    ["created_at", "filename", "camera_id", "status", "duration", "size", "accepted_fish",
     "detections", "unreviewed", "flagged", "review_status", "annotated_at"],
)
def test_every_sort_field_is_accepted(client, db_session_factory, test_settings, sort):
    complete(client, db_session_factory, test_settings, name="one.mp4", camera_id="cam-1")
    upload(client, "two.mp4")

    for order in ("asc", "desc"):
        assert len(listing(client, sort=sort, order=order).json()) == 2


def test_sorting_by_accepted_fish_puts_the_busiest_video_first(client, db_session_factory,
                                                               test_settings):
    busy = complete(client, db_session_factory, test_settings, name="busy.mp4")
    quiet = complete(client, db_session_factory, test_settings, name="quiet.mp4")
    for track in tracks_of(db_session_factory, quiet):
        client.patch(f"/tracks/{track.id}/review", json={"review_state": "rejected"})

    assert ids(listing(client, sort="accepted_fish", order="desc")) == [busy, quiet]
    assert ids(listing(client, sort="accepted_fish", order="asc")) == [quiet, busy]


def test_paging_walks_the_filtered_set_without_repeating_a_row(client):
    for index in range(5):
        upload(client, f"video-{index}.mp4")

    first = ids(listing(client, sort="filename", order="asc", limit=2, offset=0))
    second = ids(listing(client, sort="filename", order="asc", limit=2, offset=2))
    third = ids(listing(client, sort="filename", order="asc", limit=2, offset=4))

    assert len(first) == len(second) == 2 and len(third) == 1
    assert len(set(first + second + third)) == 5
    assert listing(client, limit=2).headers["X-Filtered-Count"] == "5"


# --- Facets ------------------------------------------------------------------


def test_facets_report_menu_contents_and_counts(client, db_session_factory, test_settings):
    completed_id = complete(client, db_session_factory, test_settings, name="done.mp4",
                            camera_id="cam-1")
    upload(client, "waiting.mp4", camera_id="cam-2")
    upload(client, "no-camera.mp4")

    facets = client.get("/videos/facets")
    assert facets.status_code == 200
    body = facets.json()

    assert body["cameras"] == ["cam-1", "cam-2"]
    assert body["species"] == ["fish"]
    assert body["status"] == {"completed": 1, "uploaded": 2}
    assert body["review_status"] == {"awaiting": 1, "n/a": 2}
    assert body["total"] == 3
    assert completed_id in ids(listing(client, status="completed"))


def test_facets_are_not_confused_for_a_video_id(client):
    # The literal route has to win over /videos/{video_id}.
    assert client.get("/videos/facets").status_code == 200


# --- Bulk review -------------------------------------------------------------


def test_bulk_review_applies_one_decision_to_many_tracks(client, db_session_factory,
                                                         test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    track_ids = [str(track.id) for track in tracks_of(db_session_factory, video_id)]
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        video.annotated_at = datetime.now(timezone.utc)
        video.annotated_video_path = "/private/old.mp4"
        db.commit()

    response = client.post("/tracks/review",
                           json={"track_ids": track_ids, "review_state": "accepted"})

    assert response.status_code == 200
    rows = response.json()
    assert [row["id"] for row in rows] == track_ids
    assert all(row["review_state"] == "accepted" for row in rows)
    assert all(row["accepted"] for row in rows)
    assert all(row["reviewed_at"] is not None for row in rows)
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        # The stored annotation no longer matches the decisions, so it is dropped.
        assert video.annotated_at is None
        assert video.annotated_video_path is None
    assert listing(client).json()[0]["accepted_track_count"] == 3


def test_bulk_review_returns_the_review_words_for_each_track(client, db_session_factory,
                                                             test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    low = tracks_of(db_session_factory, video_id)[0]

    rows = client.post("/tracks/review",
                       json={"track_ids": [str(low.id)], "review_state": "accepted"}).json()

    assert rows[0]["review_categories"] == ["disputed", "done"]
    assert rows[0]["run_threshold"] == pytest.approx(0.60)


def test_bulk_review_clears_reviewed_at_when_returning_to_unreviewed(client, db_session_factory,
                                                                     test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    track_ids = [str(track.id) for track in tracks_of(db_session_factory, video_id)]
    client.post("/tracks/review", json={"track_ids": track_ids, "review_state": "reviewed"})

    rows = client.post("/tracks/review",
                       json={"track_ids": track_ids, "review_state": "unreviewed"}).json()

    assert all(row["reviewed_at"] is None for row in rows)
    assert listing(client).json()[0]["review_status"] == "awaiting"


def test_bulk_review_spans_videos(client, db_session_factory, test_settings):
    first = complete(client, db_session_factory, test_settings, name="first.mp4")
    second = complete(client, db_session_factory, test_settings, name="second.mp4")
    track_ids = [str(tracks_of(db_session_factory, first)[0].id),
                 str(tracks_of(db_session_factory, second)[0].id)]

    response = client.post("/tracks/review",
                           json={"track_ids": track_ids, "review_state": "needs-review"})

    assert response.status_code == 200
    assert {row["video_id"] for row in response.json()} == {first, second}


def test_bulk_review_is_capped_at_two_hundred_ids(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    track_id = str(tracks_of(db_session_factory, video_id)[0].id)

    over = client.post("/tracks/review",
                       json={"track_ids": [track_id] * 201, "review_state": "reviewed"})
    empty = client.post("/tracks/review", json={"track_ids": [], "review_state": "reviewed"})

    assert over.status_code == 422
    assert empty.status_code == 422


def test_bulk_review_rejects_unknown_tracks_and_unfinished_videos(client, db_session_factory,
                                                                  test_settings):
    from uuid import uuid4

    video_id = complete(client, db_session_factory, test_settings)
    track_id = str(tracks_of(db_session_factory, video_id)[0].id)

    missing = client.post("/tracks/review",
                          json={"track_ids": [track_id, str(uuid4())],
                                "review_state": "reviewed"})
    assert missing.status_code == 404

    with db_session_factory() as db:
        db.get(Video, UUID(video_id)).processing_status = "processing"
        db.commit()
    unfinished = client.post("/tracks/review",
                             json={"track_ids": [track_id], "review_state": "reviewed"})
    assert unfinished.status_code == 409


def test_bulk_review_rejects_an_unknown_decision(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    track_id = str(tracks_of(db_session_factory, video_id)[0].id)

    response = client.post("/tracks/review",
                           json={"track_ids": [track_id], "review_state": "maybe"})

    assert response.status_code == 422


# --- Track table filters -----------------------------------------------------


def summaries(client, video_id, **params):
    response = client.get(f"/videos/{video_id}/track-summaries", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_track_summaries_keep_their_current_default(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)

    assert len(summaries(client, video_id)) == 2
    assert len(summaries(client, video_id, accepted_only=False)) == 3


def test_track_summaries_filter_by_review_word_and_state(client, db_session_factory,
                                                         test_settings):
    video_id = complete(client, db_session_factory, test_settings)
    low, middle, _ = tracks_of(db_session_factory, video_id)
    client.patch(f"/tracks/{middle.id}/review", json={"review_state": "needs-review"})

    borderline = summaries(client, video_id, accepted_only=False, review="borderline")
    flagged = summaries(client, video_id, accepted_only=False, review="flagged")
    states = summaries(client, video_id, accepted_only=False, review_state="needs-review")

    assert [row["id"] for row in borderline] == [str(low.id)]
    assert [row["id"] for row in flagged] == [str(middle.id)]
    assert [row["id"] for row in states] == [str(middle.id)]
    assert borderline[0]["review_categories"] == ["borderline", "unreviewed"]


def test_track_summaries_filter_by_confidence_detections_species_and_time(
    client, db_session_factory, test_settings
):
    video_id = complete(client, db_session_factory, test_settings)
    low = tracks_of(db_session_factory, video_id)[0]
    with db_session_factory() as db:
        db.get(FishTrack, low.id).species = "Gadus morhua"
        db.commit()

    assert len(summaries(client, video_id, accepted_only=False, min_confidence=0.9)) == 1
    assert len(summaries(client, video_id, accepted_only=False, max_confidence=0.6)) == 1
    assert len(summaries(client, video_id, accepted_only=False, min_detections=3)) == 1
    assert len(summaries(client, video_id, accepted_only=False, species="Gadus morhua")) == 1
    # Track 1 spans frames 0-2 at 10 fps, so only it starts before 0.25s.
    assert len(summaries(client, video_id, accepted_only=False, time_to=0.25)) == 1
    assert len(summaries(client, video_id, accepted_only=False, time_from=3.0)) == 1


@pytest.mark.parametrize(
    "sort",
    ["first_frame", "duration", "detection_count", "mean_confidence", "max_confidence",
     "species", "review_state"],
)
def test_every_track_sort_field_is_accepted(client, db_session_factory, test_settings, sort):
    video_id = complete(client, db_session_factory, test_settings)

    for order in ("asc", "desc"):
        assert len(summaries(client, video_id, accepted_only=False, sort=sort, order=order)) == 3


def test_track_sorting_by_confidence(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)

    ascending = summaries(client, video_id, accepted_only=False, sort="max_confidence",
                          order="asc")
    descending = summaries(client, video_id, accepted_only=False, sort="max_confidence",
                           order="desc")

    assert [row["max_confidence"] for row in ascending] == sorted(
        row["max_confidence"] for row in ascending
    )
    assert descending == list(reversed(ascending))


def test_track_summaries_reject_unknown_filter_values(client, db_session_factory, test_settings):
    video_id = complete(client, db_session_factory, test_settings)

    assert client.get(f"/videos/{video_id}/track-summaries",
                      params={"review": "maybe"}).status_code == 422
    assert client.get(f"/videos/{video_id}/track-summaries",
                      params={"review_state": "maybe"}).status_code == 422
    assert client.get(f"/videos/{video_id}/track-summaries",
                      params={"sort": "colour"}).status_code == 422


def test_review_status_can_be_filtered_as_well_as_counted(client, db_session_factory,
                                                          test_settings):
    reviewed = complete(client, db_session_factory, test_settings, name="reviewed.mp4")
    untouched = complete(client, db_session_factory, test_settings, name="untouched.mp4")
    upload(client, "waiting.mp4")
    for track in tracks_of(db_session_factory, reviewed):
        client.patch(f"/tracks/{track.id}/review", json={"review_state": "reviewed"})

    assert ids(listing(client, review_status="complete")) == [reviewed]
    assert ids(listing(client, review_status="awaiting")) == [untouched]
    assert len(listing(client, review_status="n/a").json()) == 1
    assert client.get("/videos", params={"review_status": "partly"}).status_code == 422
