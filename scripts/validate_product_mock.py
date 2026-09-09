r"""Isolated GPU-free smoke run; optionally serve its dashboard for visual QA.

Run from the repository: .venv\Scripts\python.exe scripts/validate_product_mock.py --serve
Each run writes only to a new directory below data/product-validation.
"""

import argparse
import os
from pathlib import Path
import sys
from uuid import uuid4

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
validation = root / "data" / "product-validation" / str(uuid4())
validation.mkdir(parents=True)
os.environ.update({
    "DATABASE_URL": f"sqlite:///{(validation / 'validation.db').as_posix()}",
    "UPLOAD_ROOT": str(validation / "uploads"), "OUTPUT_ROOT": str(validation / "outputs"),
    "JOB_ROOT": str(validation / "jobs"), "VIAME_MOCK": "true", "DEPLOYMENT_MODE": "mock",
    "VIAME_SAMPLE_CSV": str(root / "app/fixtures/sample_viame_output.csv"),
    "AUTO_CREATE_TABLES": "false",
})

from alembic import command
from alembic.config import Config
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.workers.processing_worker import claim_next_job, process_job, record_worker_heartbeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ffmpeg-bin", type=Path, help="Optional portable FFmpeg bin directory")
    args = parser.parse_args()
    if args.ffmpeg_bin:
        os.environ["PATH"] = str(args.ffmpeg_bin.resolve()) + os.pathsep + os.environ.get("PATH", "")
    command.upgrade(Config(str(root / "alembic.ini")), "head")
    source = validation / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    assert writer.isOpened()
    for index in range(50):
        frame = np.full((360, 640, 3), (70, 100, 50), dtype=np.uint8)
        cv2.putText(frame, f"Mock review validation / frame {index}", (25, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 255, 240), 1)
        writer.write(frame)
    writer.release()
    # Use the same browser-compatible encoding path as annotation output.
    from app.services.video_annotator import _make_browser_compatible
    _make_browser_compatible(source, 60)
    with TestClient(app) as client:
        ids = []
        for name in ["River camera - morning.mp4", "River camera - afternoon.mp4"]:
            with source.open("rb") as file:
                response = client.post("/videos", files={"file": (name, file, "video/mp4")})
            assert response.status_code == 201
            ids.append(response.json()["id"])
        assert client.post("/batch/process", json={"video_ids": ids}).json()["succeeded"] == 2
        while job_id := claim_next_job(worker_id="validation", worker_mode="mock"):
            process_job(job_id, worker_id="validation")
        annotated = client.post("/batch/annotate", json={"video_ids": ids})
        assert annotated.json()["succeeded"] == 2, annotated.text
        assert client.get(f"/videos/{ids[0]}/annotated-video").status_code == 200
        assert client.get("/exports/batch.csv").status_code == 200
        assert len(client.get("/analytics/videos").json()) == 2
        record_worker_heartbeat("validation", "mock", None)
    print(f"Mock batch processing, annotations, exports, analytics and migrations passed. Data: {validation}", flush=True)
    if args.serve:
        import uvicorn
        import threading
        from app.workers.processing_worker import run_worker_forever
        threading.Thread(target=run_worker_forever, daemon=True).start()
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
