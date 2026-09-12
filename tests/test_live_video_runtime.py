"""A model-compatible OpenCV is not necessarily a video-compatible OpenCV."""
from types import SimpleNamespace

import pytest

from app.services import video_media
from app.workers import live_worker


@pytest.mark.parametrize("fps", [2, 4, 5])
def test_capture_and_viame_use_the_same_live_sampling_rate(test_settings, tmp_path, monkeypatch, fps):
    import uuid
    from app.services import live_source

    test_settings.live_fps = fps
    test_settings.viame_downsample_fps = 1  # Batch settings must not thin live frames again.
    commands = []
    monkeypatch.setattr(live_source.subprocess, "Popen",
                        lambda command, **kwargs: commands.append(command))
    live_source.SegmentCapture("https://camera.example/video", tmp_path, test_settings)
    filters = commands[0][commands[0].index("-vf") + 1]
    assert filters.startswith(f"fps={fps},scale=")
    assert "scale=w='min(iw,1920)':h='min(ih,1080)'" in filters
    assert "force_original_aspect_ratio=decrease:force_divisible_by=2" in filters
    csv = tmp_path / "tracks.csv"
    csv.write_text("", encoding="utf-8")

    def runner(settings):
        assert settings.viame_downsample_fps == fps
        return SimpleNamespace(run=lambda *args: SimpleNamespace(output_csv_path=csv))

    def parse(path, **kwargs):
        assert kwargs == {"fps": fps, "frame_number_offset": test_settings.viame_frame_number_offset}
        return SimpleNamespace(skipped_rows=0, detections=[])

    monkeypatch.setattr(live_worker, "build_viame_runner", runner)
    monkeypatch.setattr(live_worker, "parse_viame_csv", parse)
    assert live_worker.detect_segment(SimpleNamespace(path=tmp_path / "chunk.mp4"),
                                      test_settings, uuid.uuid4()) == []


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
