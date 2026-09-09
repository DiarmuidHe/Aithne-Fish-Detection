from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from threading import Barrier
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import FishDetection, FishTrack, ProcessingJob, Video
from app.services.video_annotator import _load_annotations_by_frame
from app.workers.processing_worker import process_job


def upload(client, name="sample.mp4"):
    response = client.post("/videos", files={"file": (name, b"video", "video/mp4")})
    assert response.status_code == 201
    return response.json()["id"]


@pytest.fixture
def completed(client, db_session_factory, test_settings):
    video_id = upload(client)
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        video.fps = 10
        db.commit()
    job = client.post(f"/videos/{video_id}/process").json()
    process_job(UUID(job["id"]), settings=test_settings, session_factory=db_session_factory)
    return video_id, job["id"]


def rows(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    return list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))


def test_review_overrides_preserve_evidence_and_annotation_frames(client, completed, db_session_factory):
    video_id, job_id = completed
    tracks = client.get(f"/videos/{video_id}/track-summaries?accepted_only=false").json()
    high = next(t for t in tracks if t["machine_accepted"])
    low = next(t for t in tracks if not t["machine_accepted"])
    original = client.get(f"/tracks/{high['id']}").json()
    assert original["processing_job_id"] == job_id
    with db_session_factory() as db:
        count = db.scalar(select(func.count()).select_from(FishDetection))
        video = db.get(Video, UUID(video_id))
        video.annotated_video_path = "/private/old.mp4"
        db.commit()
    for decision, accepted in [("rejected", False), ("needs-review", False),
                               ("accepted", True), ("reviewed", True), ("unreviewed", True)]:
        response = client.patch(f"/tracks/{high['id']}/review", json={"review_state": decision})
        assert response.status_code == 200
        body = response.json()
        assert body["detections"] == original["detections"]
        assert body["max_confidence"] == original["max_confidence"]
        assert bool(body["reviewed_at"]) == (decision != "unreviewed")
        filtered = client.get(f"/videos/{video_id}/tracks").json()
        assert (high["id"] in {t["id"] for t in filtered}) == accepted
    client.patch(f"/tracks/{high['id']}/review", json={"review_state": "rejected"})
    client.patch(f"/tracks/{low['id']}/review", json={"review_state": "accepted"})
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        assert video.annotated_video_path is None
        assert db.scalar(select(func.count()).select_from(FishDetection)) == count
        annotations = _load_annotations_by_frame(db, video)
        accepted_ids = {t["viame_track_id"] for t in tracks if t["id"] != high["id"]}
        expected_frames = set(db.scalars(select(FishDetection.frame_number).join(FishTrack)
            .where(FishTrack.video_id == video.id, FishTrack.viame_track_id.in_(accepted_ids))))
        assert set(annotations) == expected_frames
        assert all(a.track_label != high["viame_track_id"] for items in annotations.values() for a in items)


def test_review_validation(client, completed):
    video_id, _ = completed
    track = client.get(f"/videos/{video_id}/tracks").json()[0]
    endpoint = f"/tracks/{track['id']}/review"
    assert client.patch(endpoint, json={"review_state": "deleted"}).status_code == 422
    assert client.patch(endpoint, json={"review_state": "accepted", "max_confidence": 1}).status_code == 422
    assert client.patch(f"/tracks/{uuid4()}/review", json={"review_state": "accepted"}).status_code == 404
    client.post(f"/videos/{video_id}/process")
    assert client.patch(endpoint, json={"review_state": "rejected"}).status_code == 409


def test_exports_include_all_evidence_and_safe_provenance(client, completed, db_session_factory):
    video_id, job_id = completed
    with db_session_factory() as db:
        video = db.get(Video, UUID(video_id))
        video.original_filename = "=DANGEROUS.mp4"
        job = db.get(ProcessingJob, UUID(job_id))
        config = job.configuration
        config["pipeline"] = "C:\\private\\pipeline.pipe"
        config["raw_logs"] = "/private/secret.log"
        job.configuration_json = json.dumps(config)
        db.commit()
    accepted = rows(client.get(f"/videos/{video_id}/exports/accepted-tracks.csv"))
    all_tracks = rows(client.get(f"/videos/{video_id}/exports/all-tracks.csv"))
    detections_response = client.get(f"/videos/{video_id}/exports/detections.csv")
    detections = rows(detections_response)
    summary = rows(client.get(f"/videos/{video_id}/exports/summary.csv"))[0]
    assert len(accepted) == 2 and len(all_tracks) == 3 and len(detections) == 7
    assert summary["accepted_fish_count"] == "2"
    assert summary["accepted_detections"] == "5"
    assert all(row["job_id"] == job_id for row in all_tracks + detections)
    assert all(row["pipeline"] == "pipeline.pipe" for row in all_tracks)
    assert summary["filename"].startswith("'=")
    assert summary["processing_mode"] == "mock" and summary["frame_offset"] == "0"
    assert summary["downsample_fps"] == "10.0"
    assert summary["processing_finished_at"] and summary["exported_at"]
    assert "private" not in detections_response.text and "raw_logs" not in detections_response.text
    track_id = accepted[0]["track_id"]
    client.patch(f"/tracks/{track_id}/review", json={"review_state": "rejected"})
    assert len(rows(client.get(f"/videos/{video_id}/exports/accepted-tracks.csv"))) == 1
    assert len(rows(client.get(f"/videos/{video_id}/exports/detections.csv"))) == 7


def test_exports_keep_previous_result_job_during_retry(client, completed):
    video_id, original_job_id = completed
    new_job = client.post(f"/videos/{video_id}/process").json()
    assert new_job["id"] != original_job_id
    summary = rows(client.get(f"/videos/{video_id}/exports/summary.csv"))[0]
    assert summary["job_id"] == original_job_id
    assert summary["video_status"] == "queued"
    assert all(r["job_id"] == original_job_id for r in rows(client.get(f"/videos/{video_id}/exports/all-tracks.csv")))


def test_batch_exports_and_missing_videos(client, completed):
    video_id, _ = completed
    other_id = upload(client, "other.mp4")
    assert len(rows(client.get("/exports/batch.csv"))) == 2
    assert len(rows(client.get("/exports/batch.csv", params={"video_ids": video_id}))) == 1
    assert len(rows(client.get("/exports/batch.csv", params={"kind": "all-tracks"}))) == 3
    assert client.get("/exports/batch.csv", params={"video_ids": str(uuid4())}).status_code == 404
    assert client.get(f"/videos/{other_id}/exports/wrong.csv").status_code == 422
    assert rows(client.get(f"/videos/{other_id}/exports/detections.csv")) == []


def test_batch_processing_partial_success_idempotent_and_validation(client, db_session_factory):
    ids = [upload(client, f"video-{i}.mp4") for i in range(2)]
    missing = str(uuid4())
    first = client.post("/batch/process", json={"video_ids": ids + ids + [missing]}).json()
    assert first["total"] == 3 and first["succeeded"] == 2 and first["failed"] == 1
    second = client.post("/batch/process", json={"video_ids": ids}).json()
    assert [r["job_id"] for r in first["results"][:2]] == [r["job_id"] for r in second["results"]]
    assert client.post("/batch/process", json={"video_ids": []}).status_code == 422
    assert client.post("/batch/process", json={"video_ids": ids * 51}).status_code == 422
    assert client.post("/batch/process", json={"video_ids": ["invalid"]}).status_code == 422


def test_batch_annotation_partial_success(client, completed, monkeypatch):
    from datetime import datetime, timezone
    from app.services.video_annotator import AnnotatedVideoResult
    from pathlib import Path

    video_id, _ = completed
    other_id = upload(client)
    def fake_render(**kwargs):
        return AnnotatedVideoResult(video_id=kwargs["video"].id, output_path=Path("safe.mp4"),
            annotated_at=datetime.now(timezone.utc), media_type="video/mp4", size_bytes=100,
            fps=10, width=640, height=360, frame_count=35)
    monkeypatch.setattr("app.api.videos.generate_annotated_video", fake_render)
    result = client.post("/batch/annotate", json={"video_ids": [video_id, other_id, str(uuid4())]}).json()
    assert result["total"] == 3 and result["succeeded"] == 1
    assert result["results"][0]["url"] == f"/videos/{video_id}/annotated-video"
    assert "not completed" in result["results"][1]["error"]
    def fail(**kwargs):
        raise RuntimeError("/private/path raw log secret")
    monkeypatch.setattr("app.api.videos.generate_annotated_video", fail)
    response = client.post("/batch/annotate", json={"video_ids": [video_id]})
    assert "private" not in response.text and response.json()["failed"] == 1


def test_analytics_observations_histograms_and_review(client, completed, db_session_factory):
    video_id, _ = completed
    data = client.get(f"/analytics/videos/{video_id}?bins=4").json()
    assert len(data["time_bins"]) == 4
    assert sum(row["all_detections"] for row in data["time_bins"]) == 7
    assert sum(row["accepted_detections"] for row in data["time_bins"]) == 5
    assert sum(row["count"] for row in data["confidence_distribution"]) == 7
    assert sum(row["count"] for row in data["track_duration_distribution"]) == 3
    assert data["unknown_timestamp_detections"] == 0
    track = client.get(f"/videos/{video_id}/tracks").json()[0]
    client.patch(f"/tracks/{track['id']}/review", json={"review_state": "rejected"})
    overview = client.get("/analytics/videos").json()[0]
    assert overview["accepted_fish_count"] == 1
    assert overview["accepted_detections"] == 5 - track["detection_count"]
    assert client.get(f"/analytics/videos/{video_id}?bins=0").status_code == 422
    assert client.get(f"/analytics/videos/{uuid4()}").status_code == 404
    with db_session_factory() as db:
        detection = db.scalars(select(FishDetection)).first()
        detection.timestamp_seconds = None
        db.commit()
    assert client.get(f"/analytics/videos/{video_id}").json()["unknown_timestamp_detections"] == 1


def test_concurrent_enqueue_reuses_one_active_job(tmp_path, test_settings, client):
    from app.main import app
    from app.db.database import get_db
    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent.db'}", connect_args={"timeout": 20})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db:
        video = Video(original_filename="race.mp4", storage_path="race.mp4", model_name="mock",
                      pipeline_name="mock.pipe", confidence_threshold=.6)
        db.add(video)
        db.commit()
        video_id = video.id
    barrier = Barrier(4)
    def override_db():
        with factory() as db:
            yield db
    previous_override = app.dependency_overrides[get_db]
    app.dependency_overrides[get_db] = override_db
    def enqueue(index):
        barrier.wait(timeout=10)
        if index % 2:
            response = client.post("/batch/process", json={"video_ids": [str(video_id)]})
            assert response.status_code == 200
            return response.json()["results"][0]["job_id"]
        response = client.post(f"/videos/{video_id}/process")
        assert response.status_code == 202
        return response.json()["id"]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = list(pool.map(enqueue, range(4)))
    finally:
        app.dependency_overrides[get_db] = previous_override
    assert len(set(jobs)) == 1
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(ProcessingJob)) == 1
    engine.dispose()


def test_migration_backfills_only_completed_job_without_losing_observations(tmp_path):
    import os
    import subprocess
    import sys
    from sqlalchemy import MetaData

    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    env = {**os.environ, "DATABASE_URL": database_url}
    def migrate(revision):
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", revision],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    migrate("0003_job_reliability")
    engine = create_engine(database_url)
    metadata = MetaData()
    metadata.reflect(engine)
    from datetime import datetime
    video_id, job_id, track_id = uuid4().hex, uuid4().hex, uuid4().hex
    now = datetime.now()
    with engine.begin() as conn:
        conn.execute(metadata.tables["videos"].insert().values(id=video_id, original_filename="legacy.mp4",
            storage_path="legacy.mp4", created_at=now, processing_status="failed", model_name="mock",
            pipeline_name="sample.pipe", confidence_threshold=.6))
        conn.execute(metadata.tables["processing_jobs"].insert(), [
            {"id": job_id, "video_id": video_id, "status": "completed", "created_at": now, "finished_at": now},
            {"id": uuid4().hex, "video_id": video_id, "status": "failed", "created_at": now, "finished_at": now}])
        conn.execute(metadata.tables["fish_tracks"].insert().values(id=track_id, video_id=video_id,
            viame_track_id="1", first_frame=0, last_frame=0, detection_count=1,
            mean_confidence=.8, max_confidence=.8))
        conn.execute(metadata.tables["fish_detections"].insert().values(id=uuid4().hex,
            fish_track_id=track_id, frame_number=0, x1=1, y1=2, x2=3, y2=4, confidence=.8))
    migrate("head")
    with sessionmaker(engine)() as db:
        track = db.get(FishTrack, UUID(track_id))
        assert track.processing_job_id == UUID(job_id)
        assert track.review_state == "unreviewed" and track.reviewed_at is None
        assert len(track.detections) == 1 and track.detections[0].frame_number == 0
    engine.dispose()


def test_source_video_restricts_paths(client, completed, db_session_factory):
    video_id, _ = completed
    assert client.get(f"/videos/{video_id}/source-video").content == b"video"
    with db_session_factory() as db:
        db.get(Video, UUID(video_id)).storage_path = str(__file__)
        db.commit()
    response = client.get(f"/videos/{video_id}/source-video")
    assert response.status_code == 404 and __file__ not in response.text
