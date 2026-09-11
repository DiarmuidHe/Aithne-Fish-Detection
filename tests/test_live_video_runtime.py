"""A model-compatible OpenCV is not necessarily a video-compatible OpenCV."""
from types import SimpleNamespace

import pytest

from app.services import video_media
from app.workers import live_worker


@pytest.mark.parametrize("available", [True, False])
def test_video_runtime_requires_ffmpeg_backend(monkeypatch, available):
    def has_backend(backend):
        assert backend == 1900
        return available
    monkeypatch.setattr(video_media, "load_cv2", lambda: SimpleNamespace(
        CAP_FFMPEG=1900, videoio_registry=SimpleNamespace(hasBackend=has_backend)))
    if available:
        video_media.check_live_video_runtime()
    else:
        with pytest.raises(video_media.MediaDependencyError, match="FFmpeg video decoding"):
            video_media.check_live_video_runtime()


def test_worker_checks_video_runtime_before_claiming_sessions(test_settings, monkeypatch):
    test_settings.live_monitor_enabled = True
    test_settings.viame_mock = False
    monkeypatch.setattr(live_worker, "get_settings", lambda: test_settings)
    def fail():
        raise video_media.MediaDependencyError("Live monitoring requires OpenCV with FFmpeg video decoding")
    monkeypatch.setattr(live_worker, "check_live_video_runtime", fail)
    monkeypatch.setattr(live_worker, "initialize_enhancement", lambda *args: pytest.fail("Initialized model"))
    monkeypatch.setattr(live_worker, "claim_session", lambda *args: pytest.fail("Claimed session"))
    with pytest.raises(video_media.MediaDependencyError, match="FFmpeg video decoding"):
        live_worker.run_worker_forever()
