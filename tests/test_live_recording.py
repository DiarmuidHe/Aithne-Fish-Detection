"""A finished live session becomes a reviewable, annotatable video."""

from __future__ import annotations

import shutil
import threading
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select

from app.db.models import (
    FishDetection,
    FishTrack,
    LiveFishDetection,
    LiveFishTrack,
    LiveMonitorSession,
    Video,
    utc_now,
)
from app.services import video_media
from app.services.live_recording import (
    AssembledRecording,
    RecordingPart,
    RecordingWriter,
    convert_live_detections_to_video_detections,
    finalize_live_recording,
    register_live_recording_as_video,
)
from app.services.live_source import Segment
from app.services.video_annotator import generate_annotated_video
from app.workers.live_worker import claim_session, recover_stale_sessions, run_session

# Live monitoring already refuses to start without FFmpeg, so this only skips
# where the developer environment lacks it.
requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg is required to assemble a recording"
)


@pytest.fixture
def assembler(monkeypatch, test_settings):
    """Assemble with FFmpeg where it exists, and stand in for it where it does not.

    Assembly is a stream copy, so what the rest of the pipeline depends on is the
    frame sequence the recording holds, not the container FFmpeg would have
    written. Concatenating the same chunks with OpenCV produces the same sequence,
    which keeps the session-to-library chain testable on a machine without FFmpeg.
    """

    if shutil.which("ffmpeg"):
        return
    import app.services.live_recording as recording

    def concat(arguments, timeout, failure):
        assert "concat" in arguments, "only assembly has a stand-in"
        listing = Path(arguments[arguments.index("-i") + 1])
        sources = [Path(line.split("'")[1]) for line in listing.read_text().splitlines() if line]
        cv2, writer = video_media.load_cv2(), None
        try:
            for source in sources:
                reader = cv2.VideoCapture(str(source))
                try:
                    while True:
                        ok, frame = reader.read()
                        if not ok:
                            break
                        if writer is None:
                            writer = video_media.open_video_writer(
                                cv2, Path(arguments[-1]), test_settings.live_fps,
                                frame.shape[1], frame.shape[0])
                        writer.write(frame)
                finally:
                    reader.release()
        finally:
            if writer is not None:
                writer.release()

    monkeypatch.setattr(recording, "_run_ffmpeg", concat)


def new_session(db, **kwargs):
    session = LiveMonitorSession(source_url="https://camera.example/", **kwargs)
    db.add(session)
    db.commit()
    return session


def write_chunk(path: Path, settings, frames: int = 3, width: int = 240, height: int = 160) -> Path:
    cv2 = video_media.load_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = video_media.open_video_writer(cv2, path, settings.live_fps, width, height)
    assert writer is not None
    try:
        for index in range(frames):
            frame = np.full((height, width, 3), (40 + index, 90, 150), dtype=np.uint8)
            frame[40:70, 50:85] = (20, 210, 230)
            writer.write(frame)
    finally:
        writer.release()
    return path


def live_track(db, session, frames, *, confidence=0.9, species=None, **kwargs):
    track = LiveFishTrack(session_id=session.id, status="finalized", first_seen_at=utc_now(),
                          last_seen_at=utc_now(), detection_count=len(frames),
                          max_confidence=confidence, mean_confidence=confidence, species=species,
                          x1=10, y1=20, x2=60, y2=70, **kwargs)
    db.add(track)
    db.flush()
    for frame in frames:
        db.add(LiveFishDetection(track_id=track.id, observed_at=utc_now(), frame_number=frame,
                                 confidence=confidence, species=species,
                                 x1=10, y1=20, x2=60, y2=70))
    db.commit()
    return track


def stub_recording(settings, frames: int, name: str = "live-recording.mp4",
                   width=240, height=160) -> AssembledRecording:
    """What ``RecordingWriter.assemble`` produces, without needing FFmpeg.

    Written under ``upload_root`` because that is where an assembled recording
    lands: it is the session's source footage, served like any other upload.
    """

    path = write_chunk(settings.upload_root / name, settings, frames, width, height)
    return AssembledRecording(path=path, frames=frames, width=width, height=height,
                              fps=settings.live_fps,
                              parts=(RecordingPart(0, frames, width, height),), truncated=False)


def test_writer_retains_only_committed_chunks_in_order(test_settings, tmp_path):
    writer = RecordingWriter(uuid.uuid4(), test_settings)
    for index in range(3):
        assert writer.append(write_chunk(tmp_path / f"{index}.mp4", test_settings), 3, 240, 160)
    assert writer.frames == 9
    assert [part.start_frame for part in writer.parts] == [0, 3, 6]
    # The chunk is moved, not copied: the capture directory stays bounded.
    assert not (tmp_path / "0.mp4").exists()
    assert [path.name for path in sorted(writer.directory.glob("[0-9]*.mp4"))] == [
        "00000000.mp4", "00000001.mp4", "00000002.mp4"]


def test_byte_cap_stops_recording_without_stopping_the_session(test_settings, tmp_path):
    test_settings.live_recording_max_bytes = 1
    writer = RecordingWriter(uuid.uuid4(), test_settings)
    assert writer.append(write_chunk(tmp_path / "chunk.mp4", test_settings), 3, 240, 160) is False
    assert writer.frames == 0 and writer.truncated
    # Refusing the chunk must not leave it behind in the recording directory.
    assert not list(writer.directory.glob("[0-9]*.mp4"))


def test_a_reshaped_chunk_moves_its_boxes_with_it(test_settings):
    """A reconnect can resolve a differently shaped rendition.

    Such a chunk is letterboxed into the recording, so the boxes stored against
    its frames have to move by the same scale and offset or every annotation on
    those frames lands somewhere else.
    """

    recording = AssembledRecording(
        path=Path("unused.mp4"), frames=6, width=240, height=160, fps=test_settings.live_fps,
        parts=(RecordingPart(0, 3, 240, 160),
               RecordingPart(3, 3, 240, 160, scale=0.5, offset_x=40, offset_y=0)),
        truncated=False)
    assert recording.part_for(1).place(10, 20, 60, 70) == (10, 20, 60, 70)
    assert recording.part_for(4).place(10, 20, 60, 70) == (45, 10, 70, 35)
    assert recording.part_for(6) is None


def test_conversion_copies_tracks_and_ignores_unrecorded_frames(db_session_factory,
                                                                test_settings):
    with db_session_factory() as db:
        session = new_session(db, source_key="smartbay-cam1", started_at=utc_now())
        live_track(db, session, [0, 1, 2], species="fish")
        # Fishial's answer supersedes the detector's own class name.
        live_track(db, session, [1, 3], confidence=0.7, species="fish",
                   fishial_state="identified", fishial_species="Pollachius pollachius",
                   fishial_species_confidence=0.82)
        # Footage the recording never reached: the byte cap stopped it at frame 4.
        live_track(db, session, [7, 8])
        recording = stub_recording(test_settings, 4)
        video = register_live_recording_as_video(db, session, recording, test_settings)
        assert video.is_live_recording and video.source_session_id == session.id
        assert video.processing_status == "completed" and video.camera_id == "smartbay-cam1"
        assert video.fps == test_settings.live_fps

        assert convert_live_detections_to_video_detections(db, session, video, recording) == 5
        tracks = list(db.scalars(select(FishTrack).where(FishTrack.video_id == video.id)
                                .order_by(FishTrack.first_frame)))
        assert len(tracks) == 2
        assert all(track.review_state == "unreviewed" for track in tracks)
        assert tracks[0].first_frame == 0 and tracks[0].last_frame == 2
        assert tracks[0].first_timestamp_seconds == 0
        assert tracks[0].last_timestamp_seconds == 2 / test_settings.live_fps
        assert tracks[1].species == "Pollachius pollachius"
        assert tracks[1].species_confidence == 0.82
        # Counts describe the recording, not the session, so the library's fish
        # count matches what the annotated video actually shows.
        assert [track.detection_count for track in tracks] == [3, 2]
        assert db.scalar(select(func.count()).select_from(FishDetection)) == 5


def test_a_live_recording_annotates_through_the_existing_pipeline(db_session_factory,
                                                                  test_settings):
    """The success criterion: boxes over the recorded live footage, no new pipeline."""

    cv2 = video_media.load_cv2()
    with db_session_factory() as db:
        session = new_session(db, started_at=utc_now())
        live_track(db, session, [0, 1, 2, 3])
        recording = stub_recording(test_settings, 5)
        path = recording.path
        video = register_live_recording_as_video(db, session, recording, test_settings)
        convert_live_detections_to_video_detections(db, session, video, recording)

        result = generate_annotated_video(db=db, video=video, settings=test_settings)
        assert result.frame_count == 5
        assert Path(video.annotated_video_path).is_file() and video.annotated_at is not None
        reader = cv2.VideoCapture(str(result.output_path))
        try:
            ok, first = reader.read()
            assert ok
        finally:
            reader.release()
        source = cv2.VideoCapture(str(path))
        try:
            _, original = source.read()
        finally:
            source.release()
        # The annotated copy differs from the source exactly where a box was drawn.
        assert np.any(first != original)


def test_recordings_reach_the_library_and_their_session(client, db_session_factory,
                                                        test_settings):
    with db_session_factory() as db:
        session = new_session(db, source_key="smartbay-cam2", started_at=utc_now())
        live_track(db, session, [0, 1])
        recording = stub_recording(test_settings, 3)
        video = register_live_recording_as_video(db, session, recording, test_settings)
        convert_live_detections_to_video_detections(db, session, video, recording)
        video_id, session_id = str(video.id), str(session.id)

    rows = client.get("/videos").json()
    row = next(item for item in rows if item["id"] == video_id)
    assert row["is_live_recording"] and row["source_session_id"] == session_id
    assert row["camera_id"] == "smartbay-cam2" and row["accepted_track_count"] == 1

    linked = client.get(f"/videos/{video_id}/source-session")
    assert linked.status_code == 200
    assert linked.json()["id"] == session_id
    assert linked.json()["source_key"] == "smartbay-cam2"
    # The recording is a source video like any other upload, so it plays back.
    assert client.get(f"/videos/{video_id}/source-video").status_code == 200

    uploaded = client.post("/videos", files={"file": ("plain.mp4", b"bytes", "video/mp4")}).json()
    assert uploaded["is_live_recording"] is False and uploaded["source_session_id"] is None
    assert client.get(f"/videos/{uploaded['id']}/source-session").status_code == 404


def test_a_session_without_footage_registers_nothing(db_session_factory, test_settings):
    with db_session_factory() as db:
        session = new_session(db)
        assert finalize_live_recording(db, session, test_settings) is None
        assert db.scalar(select(func.count()).select_from(Video)) == 0


def test_finalizing_twice_reuses_the_first_recording(db_session_factory, test_settings, tmp_path):
    """Recovery must never publish a session the worker already published."""

    with db_session_factory() as db:
        session = new_session(db, started_at=utc_now())
        live_track(db, session, [0, 1])
        recording = stub_recording(test_settings, 3)
        first = register_live_recording_as_video(db, session, recording, test_settings)
        writer = RecordingWriter(session.id, test_settings)
        writer.append(write_chunk(tmp_path / "late.mp4", test_settings), 3, 240, 160)
        assert finalize_live_recording(db, session, test_settings, writer).id == first.id
        assert db.scalar(select(func.count()).select_from(Video)) == 1
        assert not writer.directory.exists()


class StubCapture:
    """Serves a fixed number of chunks, then stops the session."""

    dropped = 0
    frames = 3

    def __init__(self, url, directory, settings):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.settings, self.served = directory, settings, 0

    def next_segment(self):
        if self.served >= self.chunks:
            self.shutdown.set()
            return None
        self.served += 1
        path = write_chunk(self.directory / f"chunk{self.served}.mp4", self.settings, self.frames)
        return Segment(path, utc_now() + timedelta(seconds=self.served))

    def close(self):
        pass


def test_a_finished_session_is_published_as_a_video(db_session_factory, test_settings, assembler):
    from app.services.viame_parser import VIAMEDetection

    shutdown = threading.Event()
    capture = type("Capture", (StubCapture,), {"chunks": 2, "shutdown": shutdown})
    with db_session_factory() as db:
        session_id = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(session_id, test_settings, db_session_factory, shutdown,
                resolver=lambda *_: "stub", capture_factory=capture,
                detector=lambda *_: [VIAMEDetection("1", "synthetic", index, 50, 40, 85, 70,
                                                    0.9, None, "fish", 0.9, None)
                                     for index in range(3)])
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, session_id)
        assert session.status == "stopped" and session.frames_processed == 6
        video = db.scalar(select(Video).where(Video.source_session_id == session_id))
        assert video is not None and video.is_live_recording
        assert video.processing_status == "completed"
        assert Path(video.storage_path).is_file()
        assert video.storage_path.startswith(str(test_settings.upload_root.resolve()))
        # Every analyzed frame is in the recording, so every observation is placed.
        assert video.duration_seconds == pytest.approx(6 / test_settings.live_fps, abs=0.2)
        assert db.scalar(select(func.count()).select_from(FishDetection)) == 6
        track = db.scalar(select(FishTrack).where(FishTrack.video_id == video.id))
        assert (track.first_frame, track.last_frame) == (0, 5)
        assert generate_annotated_video(db=db, video=video, settings=test_settings).frame_count == 6
        # The retained chunks are released once the recording is assembled.
        assert not (test_settings.output_root / "live" / str(session_id) / "recording").exists()


def test_crossing_identities_survive_worker_segments_and_recording(
        db_session_factory, test_settings, assembler):
    from app.services.viame_parser import VIAMEDetection

    test_settings.live_fps = 2
    shutdown, now = threading.Event(), utc_now()

    class Capture(StubCapture):
        def next_segment(self):
            if self.served == 2:
                shutdown.set()
                return None
            path = write_chunk(self.directory / f"{self.served}.mp4", self.settings, frames=2)
            segment = Segment(path, now + timedelta(seconds=self.served))
            self.served += 1
            return segment

    def detector(segment, *_):
        observations = []
        start = int(segment.path.stem) * 2
        for local in range(2):
            frame = start + local
            # Two fish cross. In the final frame VIAME gives each the other's ID.
            for key, x in (("2" if frame == 3 else "1", 20 + 30 * frame),
                           ("1" if frame == 3 else "2", 160 - 30 * frame)):
                observations.append(VIAMEDetection(key, "synthetic", local, x, 40, x + 35, 70,
                                                   .9, None, "fish", .9, None))
        return observations

    with db_session_factory() as db:
        session_id = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(session_id, test_settings, db_session_factory, shutdown,
                resolver=lambda *_: "stub", capture_factory=Capture, detector=detector)
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, session_id)
        assert session.status == "stopped" and session.frames_processed == 4
        live_tracks = db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == session_id)).all()
        assert len(live_tracks) == 2
        assert all(t.detection_count == 4 for t in live_tracks)
        video = db.scalar(select(Video).where(Video.source_session_id == session_id))
        assert video is not None and Path(video.storage_path).is_file()
        tracks = db.scalars(select(FishTrack).where(FishTrack.video_id == video.id)).all()
        paths = []
        for track in tracks:
            rows = db.scalars(select(FishDetection).where(FishDetection.fish_track_id == track.id)
                              .order_by(FishDetection.frame_number)).all()
            paths.append([row.x1 for row in rows])
        assert sorted(paths) == [[20, 50, 80, 110], [160, 130, 100, 70]]


def test_a_lost_worker_still_publishes_the_footage_it_analyzed(db_session_factory, test_settings,
                                                               tmp_path, assembler):
    """A crashed worker leaves its chunks and manifest behind; recovery uses them."""

    with db_session_factory() as db:
        session = new_session(db, status="running", worker_id="lost",
                              heartbeat_at=utc_now() - timedelta(seconds=121))
        live_track(db, session, [0, 1, 2])
        session_id = session.id
        writer = RecordingWriter(session_id, test_settings)
        assert writer.append(write_chunk(tmp_path / "chunk.mp4", test_settings), 3, 240, 160)

    recover_stale_sessions(db_session_factory, test_settings)
    with db_session_factory() as db:
        assert db.get(LiveMonitorSession, session_id).status == "failed"
        video = db.scalar(select(Video).where(Video.source_session_id == session_id))
        assert video is not None and Path(video.storage_path).is_file()
        assert db.scalar(select(func.count()).select_from(FishDetection)) == 3


@requires_ffmpeg
def test_ffmpeg_assembles_the_retained_chunks_in_order(test_settings, tmp_path):
    """The real command: a stream copy of the chunks, in the order they were analyzed."""

    cv2 = video_media.load_cv2()
    session_id = uuid.uuid4()
    writer = RecordingWriter(session_id, test_settings)
    for index in range(3):
        assert writer.append(write_chunk(tmp_path / f"{index}.mp4", test_settings), 3, 240, 160)
    recording = writer.assemble()
    assert recording.path == test_settings.upload_root.resolve() / f"live-{session_id}.mp4"
    assert (recording.frames, recording.width, recording.height) == (9, 240, 160)
    reader = cv2.VideoCapture(str(recording.path))
    try:
        decoded = 0
        while reader.read()[0]:
            decoded += 1
    finally:
        reader.release()
    # Every analyzed frame is present, so a detection's frame number still indexes it.
    assert decoded == recording.frames


def test_disabling_recording_leaves_no_footage(db_session_factory, test_settings):
    from app.services.viame_parser import VIAMEDetection

    test_settings.live_recording_enabled = False
    shutdown = threading.Event()
    capture = type("Capture", (StubCapture,), {"chunks": 1, "shutdown": shutdown})
    with db_session_factory() as db:
        session_id = new_session(db).id
    claim_session(db_session_factory, "test")
    run_session(session_id, test_settings, db_session_factory, shutdown,
                resolver=lambda *_: "stub", capture_factory=capture,
                detector=lambda *_: [VIAMEDetection("1", "synthetic", 0, 50, 40, 85, 70,
                                                    0.9, None, "fish", 0.9, None)])
    with db_session_factory() as db:
        assert db.get(LiveMonitorSession, session_id).status == "stopped"
        assert db.scalar(select(func.count()).select_from(Video)) == 0
    assert not (test_settings.output_root / "live" / str(session_id) / "recording").exists()


def test_recording_failure_does_not_fail_the_session(db_session_factory, test_settings, tmp_path,
                                                     monkeypatch):
    import app.services.live_recording as recording

    def explode(*_, **__):
        raise recording.LiveRecordingError("The live recording could not be assembled")

    monkeypatch.setattr(RecordingWriter, "assemble", explode)
    from app.workers.live_worker import finish_recording

    with db_session_factory() as db:
        session = new_session(db, status="stopped", started_at=utc_now())
        writer = RecordingWriter(session.id, test_settings)
        writer.append(write_chunk(tmp_path / "chunk.mp4", test_settings), 3, 240, 160)
        assert finish_recording(db, session, test_settings, writer) is None
        db.refresh(session)
        # The session still completed; only the review copy is missing, and it says so.
        assert session.status == "stopped" and "could not be saved" in session.error_message
        assert db.scalar(select(func.count()).select_from(Video)) == 0
