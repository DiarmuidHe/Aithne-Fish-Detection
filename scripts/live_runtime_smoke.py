"""Offline MP4 decode + FUnIE smoke using the live worker's exact runtime.

Run through live_worker_viame.sh with FISHIAL_PREPROCESS=funie_gan. No camera,
database, Fishial client or HTTP request is used; temporary video is synthetic.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def smoke(model):
    from app.services.video_media import check_live_video_runtime, load_cv2
    from scripts.funie_smoke import smoke as model_smoke

    check_live_video_runtime()
    cv2 = load_cv2()
    with tempfile.TemporaryDirectory(prefix="live-runtime-") as temporary:
        video = Path(temporary) / "synthetic.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=5", "-t", "2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(video)], check=True, timeout=30)
        reader = cv2.VideoCapture(str(video))
        frames = 0
        try:
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                assert frame.shape == (180, 320, 3)
                frames += 1
        finally:
            reader.release()
        assert frames == 10, f"Expected 10 synthetic video frames, decoded {frames}"
    model_smoke(model)
    print(f"Live runtime passed: OpenCV {cv2.__version__}, {frames} H.264 frames, FUnIE inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    smoke(parser.parse_args().model)
