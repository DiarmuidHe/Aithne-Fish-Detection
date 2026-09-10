from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.services import viame_runner
from app.services.viame_runner import VIAMERunner, VIAMERunnerError


@pytest.fixture
def runner_settings(tmp_path: Path) -> Settings:
    setup = tmp_path / "setup_viame.sh"
    setup.write_text("", encoding="utf-8")
    pipeline = tmp_path / "tracker.pipe"
    pipeline.write_text("", encoding="utf-8")
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        job_root=tmp_path / "jobs",
        viame_mock=False,
        viame_setup_script=setup,
        viame_tracker_pipeline=pipeline,
        viame_timeout_seconds=1,
    )


def test_timeout_logs_partial_output_captured_as_bytes(runner_settings, tmp_path, monkeypatch):
    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"stub")

    def timeout(*args, **kwargs):
        # subprocess.run re-raises with the raw output captured before the kill,
        # which stays bytes even though the call asked for text mode.
        raise subprocess.TimeoutExpired(
            cmd="kwiver", timeout=1, output=None, stderr=b"model load\xff interrupted")

    monkeypatch.setattr(viame_runner.subprocess, "run", timeout)
    job_id = uuid.uuid4()
    with pytest.raises(VIAMERunnerError, match="timed out"):
        VIAMERunner(runner_settings).run(video, job_id)

    stderr_log = next((runner_settings.job_root / str(job_id)).glob("*/viame_stderr.log"))
    assert "model load" in stderr_log.read_text(encoding="utf-8")
