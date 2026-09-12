r"""Drive a whole live session through to an annotated recording, without a camera.

Runs the real live worker loop, the real recording writer and the real FFmpeg
assembly against synthetic H.264 chunks and a stubbed detector, then registers,
converts and annotates the result through the HTTP API. What this does NOT cover
is camera resolution (yt-dlp/HLS) and VIAME itself; everything after a chunk has
been analyzed is exercised exactly as it runs in production.

Run from the repository:
  .venv\Scripts\python.exe scripts/validate_live_recording.py
  .venv\Scripts\python.exe scripts/validate_live_recording.py --serve

Each run writes only to a new directory below data/live-recording-validation.
"""

import argparse
import itertools
import os
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
validation = root / "data" / "live-recording-validation" / str(uuid4())
validation.mkdir(parents=True)
CHUNKS, FRAMES_PER_CHUNK, FPS = 3, 10, 5.0
os.environ.update({
    "DATABASE_URL": f"sqlite:///{(validation / 'validation.db').as_posix()}",
    "UPLOAD_ROOT": str(validation / "uploads"), "OUTPUT_ROOT": str(validation / "outputs"),
    "JOB_ROOT": str(validation / "jobs"), "LIVE_SCRATCH_ROOT": str(validation / "scratch"),
    "VIAME_MOCK": "true", "DEPLOYMENT_MODE": "mock", "AUTO_CREATE_TABLES": "false",
    "LIVE_MONITOR_ENABLED": "true", "LIVE_RECORDING_ENABLED": "true",
    "LIVE_FPS": str(FPS), "LIVE_SEGMENT_SECONDS": str(FRAMES_PER_CHUNK / FPS),
    # This run never calls Fishial, and the repository .env may select a
    # preprocessing mode whose model is not installed here.
    "FISHIAL_ENABLED": "false", "FISHIAL_PREPROCESS": "none",
})

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import FishDetection, FishTrack, LiveMonitorSession, Video, utc_now
from app.main import app
from app.services.live_source import Segment
from app.services.viame_parser import VIAMEDetection
from app.workers.live_worker import claim_session, run_session


def synthetic_capture(shutdown, chunks):
    """Stand in for SegmentCapture: real H.264 chunks at LIVE_FPS, but no camera."""

    class SyntheticCapture:
        dropped = 0

        def __init__(self, url, directory, settings):
            directory.mkdir(parents=True, exist_ok=True)
            self.directory, self.settings, self.served = directory, settings, 0

        def next_segment(self):
            if self.served >= chunks:
                shutdown.set()
                return None
            self.served += 1
            path = self.directory / f"{self.served:08d}.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", f"testsrc2=size=1280x720:rate={FPS}",
                 "-t", str(FRAMES_PER_CHUNK / FPS), "-an", "-c:v", "libx264",
                 "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
                check=True, timeout=60, capture_output=True)
            print(f"  chunk {self.served}/{chunks} captured", flush=True)
            return Segment(path, utc_now())

        def close(self):
            pass

    return SyntheticCapture


def synthetic_detector():
    """One fish crossing the frame, in place of a VIAME run over each chunk.

    It keeps moving across chunk boundaries, because VIAME's own track IDs
    restart with every chunk and it is spatial association that carries one fish
    between them. A fish that teleported back to the left edge each chunk would
    be a different fish, correctly, and would say nothing about association.
    """

    chunks = itertools.count()

    def detector(segment, settings, session_id):
        offset = next(chunks) * FRAMES_PER_CHUNK
        return [VIAMEDetection("1", "synthetic", index, 200 + 30 * (offset + index), 300,
                               400 + 30 * (offset + index), 460, 0.92, None, "fish", 0.9, None)
                for index in range(FRAMES_PER_CHUNK)]

    return detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true",
                        help="Serve the dashboard afterwards so the result can be reviewed")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ffmpeg-bin", type=Path, help="Optional portable FFmpeg bin directory")
    args = parser.parse_args()
    if args.ffmpeg_bin:
        os.environ["PATH"] = str(args.ffmpeg_bin.resolve()) + os.pathsep + os.environ.get("PATH", "")
    for binary in ("ffmpeg", "ffprobe"):
        if subprocess.run(["where" if os.name == "nt" else "which", binary],
                          capture_output=True, check=False).returncode != 0:
            raise SystemExit(f"{binary} is not on PATH. Install FFmpeg, or pass --ffmpeg-bin "
                             "pointing at a portable build's bin directory.")
    command.upgrade(Config(str(root / "alembic.ini")), "head")
    settings = get_settings()

    expected = CHUNKS * FRAMES_PER_CHUNK
    shutdown = threading.Event()
    with SessionLocal() as db:
        session = LiveMonitorSession(source_url="https://camera.example/", source_key="coral-city")
        db.add(session)
        db.commit()
        session_id = session.id
    print(f"Monitoring a synthetic camera: {CHUNKS} chunks of {FRAMES_PER_CHUNK} frames", flush=True)
    claim_session(SessionLocal, "live-recording-validation")
    run_session(session_id, settings, SessionLocal, shutdown, resolver=lambda *_: "stub",
                capture_factory=synthetic_capture(shutdown, CHUNKS),
                detector=synthetic_detector())

    with SessionLocal() as db:
        session = db.get(LiveMonitorSession, session_id)
        assert session.status == "stopped", f"Session ended {session.status}: {session.error_message}"
        assert session.frames_processed == expected, session.frames_processed
        video = db.scalar(select(Video).where(Video.source_session_id == session_id))
        assert video is not None, "The finished session did not reach the library"
        assert video.is_live_recording and video.processing_status == "completed"
        recording = Path(video.storage_path)
        assert recording.is_file(), recording
        tracks = db.scalar(select(func.count()).select_from(FishTrack))
        # One fish crossed the frame, so association carried it between chunks.
        assert tracks == 1, f"One fish crossed the frame, but the library shows {tracks} tracks"
        detections = db.scalar(select(func.count()).select_from(FishDetection))
        # Every analyzed frame is in the recording, so every observation is placed.
        assert detections == expected, f"{detections} observations for {expected} frames"
        video_id = str(video.id)
        assert not (settings.output_root / "live" / str(session_id) / "recording").exists(), \
            "The retained chunks were not released after assembly"

    # Frame-for-frame: the recording holds exactly the frames the session analyzed.
    decoded = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,width,height", "-of", "csv=p=0",
         str(recording)], capture_output=True, text=True, check=True).stdout.strip()
    width, height, packets = decoded.split(",")
    assert int(packets) == expected, f"The recording holds {packets} of {expected} frames"

    with TestClient(app) as client:
        row = next(item for item in client.get("/videos").json() if item["id"] == video_id)
        assert row["is_live_recording"] and row["source_session_id"] == str(session_id)
        assert row["camera_id"] == "coral-city" and row["accepted_track_count"] == tracks
        linked = client.get(f"/videos/{video_id}/source-session")
        assert linked.status_code == 200 and linked.json()["id"] == str(session_id)
        assert client.get(f"/videos/{video_id}/source-video").status_code == 200
        annotated = client.post(f"/videos/{video_id}/annotate")
        assert annotated.status_code == 200, annotated.text
        assert annotated.json()["frame_count"] == expected, annotated.json()
        assert client.get(f"/videos/{video_id}/annotated-video").status_code == 200
        assert client.get(f"/videos/{video_id}/exports/detections.csv").status_code == 200

    print(f"\nLive recording validated: {expected} analyzed frames recorded at {width}x{height}, "
          f"{tracks} fish {'track' if tracks == 1 else 'tracks'} and {detections} observations "
          f"copied to the library, annotated video rendered over the recording."
          f"\nData: {validation}", flush=True)
    if args.serve:
        import uvicorn
        print(f"\nOpen http://127.0.0.1:{args.port}/library and look for the video marked LIVE.",
              flush=True)
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
