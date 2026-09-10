from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


def _captured_text(value: str | bytes | None) -> str:
    """TimeoutExpired carries raw bytes even when the run asked for text mode."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


class VIAMERunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class VIAMERunResult:
    job_dir: Path
    run_dir: Path
    output_csv_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    return_code: int
    duration_seconds: float
    pipeline_path: Path
    viame_version: str | None = None


class VIAMERunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, video_path: Path, job_id: uuid.UUID, pipeline_path: Path | None = None) -> VIAMERunResult:
        pipeline = pipeline_path or self.settings.viame_tracker_pipeline
        run_dir = self._create_run_dir(job_id)
        input_list_path = run_dir / "input_list.txt"
        output_csv_path = run_dir / "viame_tracks.csv"
        stdout_log_path = run_dir / "viame_stdout.log"
        stderr_log_path = run_dir / "viame_stderr.log"

        if not video_path.exists():
            raise VIAMERunnerError(f"Input video does not exist: {video_path}")
        if not self.settings.viame_setup_script.exists():
            raise VIAMERunnerError(f"VIAME setup script does not exist: {self.settings.viame_setup_script}")
        if not pipeline.exists():
            raise VIAMERunnerError(f"VIAME pipeline does not exist: {pipeline}")

        input_list_path.write_text(f"{video_path}\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "VIAME_ROOT": str(self.settings.viame_root),
                "VIAME_SETUP_SCRIPT": str(self.settings.viame_setup_script),
                "VIAME_PIPELINE": str(pipeline),
                "VIAME_INPUT_VIDEO": str(video_path),
                "VIAME_INPUT_LIST": str(input_list_path),
                "VIAME_OUTPUT_CSV": str(output_csv_path),
                "VIAME_DOWNSAMPLE_FPS": str(self.settings.viame_downsample_fps),
                "VIAME_DETECTOR_SCORE_THRESHOLD": str(
                    self.settings.viame_detector_score_threshold
                ),
                "VIAME_TRACKER_HIGH_THRESHOLD": str(
                    self.settings.viame_tracker_high_threshold
                ),
                "VIAME_TRACKER_LOW_THRESHOLD": str(
                    self.settings.viame_tracker_low_threshold
                ),
                "VIAME_TRACKER_NEW_TRACK_THRESHOLD": str(
                    self.settings.viame_tracker_new_track_threshold
                ),
                "VIAME_TRACKER_BUFFER_FRAMES": str(
                    self.settings.viame_tracker_buffer_frames
                ),
            }
        )

        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                ["bash", "-lc", self.settings.viame_run_command],
                cwd=run_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.settings.viame_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_log_path.write_text(_captured_text(exc.stdout), encoding="utf-8")
            stderr_log_path.write_text(_captured_text(exc.stderr), encoding="utf-8")
            raise VIAMERunnerError(
                f"VIAME timed out after {self.settings.viame_timeout_seconds} seconds"
            ) from exc

        duration_seconds = time.monotonic() - started_at
        stdout_log_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_log_path.write_text(completed.stderr or "", encoding="utf-8")

        if completed.returncode != 0:
            raise VIAMERunnerError(
                f"VIAME exited with code {completed.returncode}. See worker logs."
            )

        detected_csv = self._find_output_csv(run_dir, output_csv_path)
        return VIAMERunResult(
            job_dir=run_dir.parent,
            run_dir=run_dir,
            output_csv_path=detected_csv,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            return_code=completed.returncode,
            duration_seconds=duration_seconds,
            pipeline_path=pipeline,
            viame_version=self.settings.viame_version,
        )

    def _create_run_dir(self, job_id: uuid.UUID) -> Path:
        job_dir = self.settings.job_root / str(job_id)
        run_dir = job_dir / str(uuid.uuid4())
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _find_output_csv(run_dir: Path, expected_csv_path: Path) -> Path:
        if expected_csv_path.exists():
            return expected_csv_path
        csv_candidates = sorted(run_dir.rglob("*.csv"))
        if not csv_candidates:
            raise VIAMERunnerError(f"VIAME completed but no CSV output was found in {run_dir}")
        preferred = [
            path
            for path in csv_candidates
            if "track" in path.name.lower() or "detect" in path.name.lower()
        ]
        return preferred[0] if preferred else csv_candidates[0]


class MockVIAMERunner(VIAMERunner):
    def run(self, video_path: Path, job_id: uuid.UUID, pipeline_path: Path | None = None) -> VIAMERunResult:
        pipeline = pipeline_path or self.settings.viame_tracker_pipeline
        run_dir = self._create_run_dir(job_id)
        sample_csv = self.settings.viame_sample_csv
        if not sample_csv.is_absolute():
            sample_csv = Path.cwd() / sample_csv
        if not sample_csv.exists():
            raise VIAMERunnerError(f"Mock VIAME sample CSV does not exist: {sample_csv}")

        output_csv_path = run_dir / "viame_tracks.csv"
        stdout_log_path = run_dir / "viame_stdout.log"
        stderr_log_path = run_dir / "viame_stderr.log"
        input_list_path = run_dir / "input_list.txt"
        input_list_path.write_text(f"{video_path}\n", encoding="utf-8")

        started_at = time.monotonic()
        shutil.copyfile(sample_csv, output_csv_path)
        stdout_log_path.write_text("VIAME_MOCK=true: copied sample VIAME CSV fixture.\n", encoding="utf-8")
        stderr_log_path.write_text("", encoding="utf-8")

        return VIAMERunResult(
            job_dir=run_dir.parent,
            run_dir=run_dir,
            output_csv_path=output_csv_path,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            return_code=0,
            duration_seconds=time.monotonic() - started_at,
            pipeline_path=pipeline,
            viame_version=self.settings.viame_version or "mock",
        )


def build_viame_runner(settings: Settings) -> VIAMERunner:
    if settings.viame_mock:
        return MockVIAMERunner(settings)
    return VIAMERunner(settings)
