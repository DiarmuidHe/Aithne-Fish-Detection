from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

from pydantic import SecretStr

from app.db.models import CvatSync, ReviewRegion, Video
from app.services.review_dataset import (
    annotation_sha256,
    candidate_audit,
    corrected_rows,
    proposal_regions,
)


def _shape(frame, box, outside=False):
    return SimpleNamespace(type="rectangle", frame=frame, points=box, outside=outside)


def test_corrected_rows_interpolate_and_keep_verified_empty_frames():
    annotations = SimpleNamespace(tracks=[
        SimpleNamespace(id=7, label_id=1, shapes=[
            _shape(1, [10, 10, 20, 20]),
            _shape(3, [20, 10, 30, 20]),
            _shape(4, [20, 10, 30, 20], outside=True),
        ]),
    ], shapes=[])
    rows = corrected_rows(annotations, 0, 5, 100, 100, {1})
    assert rows[0] == rows[4] == rows[5] == []
    assert rows[2] == [{"track_id": 7, "box": [15, 10, 25, 20]}]
    assert len(annotation_sha256(rows)) == 64


def test_open_track_continues_through_review_bounds_and_species_hash_changes():
    annotations = SimpleNamespace(tracks=[SimpleNamespace(
        id=7, label_id=1, shapes=[_shape(1, [10, 10, 20, 20])])], shapes=[])
    rows = corrected_rows(annotations, 3, 5, 100, 100, {1})
    assert all(rows[frame] == [{"track_id": 7, "box": [10, 10, 20, 20]}]
               for frame in range(3, 6))
    assert annotation_sha256(rows, {7: "unknown"}) != annotation_sha256(rows, {7: "cod"})


def test_latest_pilot_has_distributed_frames_and_two_windows():
    result = proposal_regions(uuid.UUID("85ea19d4-db7b-4ed7-90a0-a8acccba2b0c"), 12.5, 1440)
    assert len(result) == 14
    assert [(entry["start_frame"], entry["end_frame"]) for entry in result[-2:]] == [
        (600, 637), (1100, 1137),
    ]


def test_review_hotspots_include_long_track_gaps(tmp_path):
    tracks = tmp_path / "viame_tracks.csv"
    tracks.write_text("".join(
        f"1,video,{frame},10,20,30,40,0.8,-1,fish,0.8\n"
        for frame in (0, 3, 6, 15, 18)), encoding="utf-8")
    audit = candidate_audit(tracks, None, 30, frame_offset=0)
    assert audit["long_gap_frames"] == 1
    assert audit["hotspots"][0]["frame"] == 9
    assert audit["hotspots"][0]["long_track_gaps"] == 1


def test_pilot_window_cannot_be_marked_complete_without_ids_and_species(
    client, db_session_factory, test_settings,
):
    test_settings.cvat_dev_mode = True
    test_settings.cvat_access_token = SecretStr("test-token")
    video_id = uuid.UUID("85ea19d4-db7b-4ed7-90a0-a8acccba2b0c")
    with db_session_factory() as db:
        db.add(Video(id=video_id, original_filename="pilot.mp4", storage_path="/tmp/pilot.mp4",
                     fps=12.5, width=1920, height=1080, duration_seconds=115.2,
                     model_name="test", pipeline_name="test", confidence_threshold=.6))
        db.add(CvatSync(video_id=video_id, task_id=3, status="review"))
        db.commit()
    response = client.put(f"/cvat/videos/{video_id}/review-regions", json={
        "start_frame": 600, "end_frame": 637, "review_minutes": 2,
        "ids_complete": True, "species_complete": False,
    })
    assert response.status_code == 422


def test_review_completion_requires_explicit_species_and_budget(
    client, db_session_factory, test_settings, monkeypatch,
):
    test_settings.cvat_dev_mode = True
    test_settings.cvat_access_token = SecretStr("test-token")
    video_id = uuid.uuid4()
    with db_session_factory() as db:
        db.add(Video(id=video_id, original_filename="fish.mp4", storage_path="/tmp/fish.mp4",
                     fps=10, width=100, height=100, duration_seconds=10,
                     model_name="test", pipeline_name="test", confidence_threshold=.6))
        db.add(CvatSync(video_id=video_id, task_id=42, status="review"))
        db.commit()
    annotations = SimpleNamespace(tracks=[
        SimpleNamespace(id=5, label_id=1, shapes=[
            _shape(0, [10, 10, 20, 20]), _shape(2, [12, 10, 22, 20]),
            _shape(3, [12, 10, 22, 20], outside=True),
        ]),
    ], shapes=[])
    task = SimpleNamespace(get_labels=lambda: [SimpleNamespace(id=1, name="fish")],
                           get_annotations=lambda: annotations)
    @contextmanager
    def gateway(_settings):
        yield SimpleNamespace(client=SimpleNamespace(
            tasks=SimpleNamespace(retrieve=lambda _id: task)))
    monkeypatch.setattr("app.api.cvat.gateway_for", gateway)

    path = f"/cvat/videos/{video_id}"
    reviewed = {"start_frame": 0, "end_frame": 2, "ids_complete": True,
                "species_complete": True, "review_minutes": 12}
    assert client.put(f"{path}/review-regions", json=reviewed).status_code == 422
    assert client.put(f"{path}/tracks/5/species", json={"species": "unknown"}).status_code == 200
    completed = client.put(f"{path}/review-regions", json=reviewed)
    assert completed.status_code == 200
    with db_session_factory() as db:
        assert db.query(ReviewRegion).count() == 1
    assert client.put(f"{path}/review-regions", json={
        "start_frame": 4, "end_frame": 4, "review_minutes": 20,
    }).status_code == 422
    assert client.delete(f"{path}/review-regions/{completed.json()['id']}").status_code == 204
