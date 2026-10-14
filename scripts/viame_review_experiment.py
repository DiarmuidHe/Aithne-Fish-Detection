"""Offline VIAME candidate runs and matched raw/CLAHE video preparation.

Run `run` inside the VIAME worker image after reviewed truth exists. It never
changes application settings, jobs or CVAT annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

VIAME_ROOT = Path(os.environ.get("VIAME_ROOT", "/opt/noaa/viame"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enhance_frame(frame, clahe):
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def paired_clahe(video: Path, output: Path) -> dict:
    import cv2

    if output.exists():
        raise FileExistsError(output)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if fps <= 0 or min(size) <= 0:
        raise ValueError("Video fps and dimensions are required")
    output.mkdir(parents=True)
    paths = [output / "raw-reencoded.mp4", output / "clahe.mp4"]
    writers = [cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
               for path in paths]
    if not all(writer.isOpened() for writer in writers):
        raise RuntimeError("Cannot create paired video outputs")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writers[0].write(frame)
            writers[1].write(enhance_frame(frame, clahe))
            frames += 1
    finally:
        cap.release()
        for writer in writers:
            writer.release()
    report = {"source": str(video), "source_sha256": sha256(video), "fps": fps,
              "width": size[0], "height": size[1], "frames": frames,
              "transform": "LAB-L CLAHE clipLimit=2.0 tileGridSize=8x8",
              "paired_outputs": [{"path": str(path), "sha256": sha256(path)} for path in paths]}
    (output / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def pipeline_processes(pipeline: Path, search: tuple[Path, ...] = ()) -> set[str]:
    """Process names declared by a .pipe file and its includes, resolved like kwiver -I."""
    names, pending, seen = set(), [pipeline], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if match := re.match(r"\s*process\s+(\w+)", line):
                names.add(match.group(1))
            elif match := re.match(r"\s*include\s+(\S+)", line):
                found = next((d / match.group(1) for d in (path.parent, *search)
                              if (d / match.group(1)).is_file()), None)
                if found is None:
                    raise FileNotFoundError(f"{path}: include {match.group(1)} not found")
                pending.append(found)
    return names


def kwiver_settings(args, processes: set[str]) -> dict[str, str]:
    """The run_viame.sh overrides, limited to processes the pipeline declares."""
    settings = {"input:video_filename": str(args.video.resolve()),
                "input:video_reader:type": "image_list" if args.image_list else "vidl_ffmpeg"}
    if args.image_list:
        settings["input:frame_time"] = repr(args.frame_time)
    if "downsampler" in processes:
        settings |= {"downsampler:target_frame_rate": str(args.fps),
                     "downsampler:renumber_frames": "false"}
    if "detector" in processes:
        settings["detector:detector:onnx:score_thresh"] = str(args.score)
    if "detection_reader" in processes:
        if not args.detections:
            raise ValueError("This pipeline replays saved detections; pass --detections")
        settings["detection_reader:file_name"] = str(args.detections.resolve())
    if "tracker" in processes:
        prefix = "tracker:track_objects:bytetrack:"
        tracker = {"high_thresh": args.high, "low_thresh": args.low,
                   "new_track_thresh": args.new_track, "track_buffer": args.buffer,
                   "match_thresh": args.match, "second_match_thresh": args.second_match,
                   "unconfirmed_match_thresh": args.unconfirmed_match}
        settings |= {prefix + key: str(value) for key, value in tracker.items() if value is not None}
    if "track_writer" in processes:
        settings["track_writer:file_name"] = "viame_tracks.csv"
    return settings


def run_candidate(args) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.detections and not args.detections.is_file():
        raise FileNotFoundError(args.detections)
    if args.pipeline and not args.pipeline.is_file():
        raise FileNotFoundError(args.pipeline)
    # Plain video runs keep the production command in run_viame.sh unchanged.
    direct = args.image_list or args.detections or any(
        value is not None for value in (args.match, args.second_match, args.unconfirmed_match))
    if direct and not args.pipeline:
        raise ValueError("Image-list, replay and match-threshold runs need an explicit --pipeline")
    settings = None
    if direct:
        processes = pipeline_processes(args.pipeline, (VIAME_ROOT / "configs/pipelines",))
        settings = kwiver_settings(args, processes)
    args.output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({
        "VIAME_DOWNSAMPLE_FPS": str(args.fps),
        "VIAME_DETECTOR_SCORE_THRESHOLD": str(args.score),
        "VIAME_TRACKER_HIGH_THRESHOLD": str(args.high),
        "VIAME_TRACKER_LOW_THRESHOLD": str(args.low),
        "VIAME_TRACKER_NEW_TRACK_THRESHOLD": str(args.new_track),
        "VIAME_TRACKER_BUFFER_FRAMES": str(args.buffer),
    })
    if args.pipeline:
        environment["VIAME_TRACKER_PIPELINE"] = str(args.pipeline)
    if settings is None:
        command = ["bash", "/app/scripts/run_viame.sh", str(args.video.resolve()), "viame_tracks.csv"]
    else:
        command = ["bash", "-c", 'source "$1" && shift && exec kwiver runner "$@"', "viame",
                   str(VIAME_ROOT / "setup_viame.sh"), "-I", str(VIAME_ROOT / "configs/pipelines"),
                   str(args.pipeline.resolve())]
        for key, value in settings.items():
            command += ["-s", f"{key}={value}"]
    started = time.monotonic()
    with (args.output / "stdout.log").open("w", encoding="utf-8") as stdout, (
            args.output / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=args.output, env=environment, stdout=stdout,
                                   stderr=stderr, timeout=args.timeout, check=False)
    expected = ("viame_tracks.csv" if settings is None or "track_writer:file_name" in settings
                else "computed_detections.csv")
    report = {"video": str(args.video), "video_sha256": sha256(args.video),
              "pipeline": environment.get("VIAME_TRACKER_PIPELINE"),
              "pipeline_sha256": sha256(args.pipeline) if args.pipeline else None,
              "fps": args.fps, "score": args.score, "high": args.high,
              "low": args.low, "new_track": args.new_track, "buffer": args.buffer,
              "match": args.match, "second_match": args.second_match,
              "unconfirmed_match": args.unconfirmed_match,
              "image_list": args.image_list, "frame_time": args.frame_time if args.image_list else None,
              "detections": str(args.detections) if args.detections else None,
              "detections_sha256": sha256(args.detections) if args.detections else None,
              "kwiver_settings": settings,
              "processing_seconds": time.monotonic() - started,
              "return_code": completed.returncode, "review_status": "unreviewed prediction"}
    (args.output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if completed.returncode or not (args.output / expected).is_file():
        raise RuntimeError(f"VIAME candidate failed; see {args.output / 'stderr.log'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paired = commands.add_parser("paired-clahe")
    paired.add_argument("video", type=Path)
    paired.add_argument("output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("video", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--pipeline", type=Path)
    run.add_argument("--fps", type=float, default=10)
    run.add_argument("--score", type=float, default=.10)
    run.add_argument("--high", type=float, default=.45)
    run.add_argument("--low", type=float, default=.10)
    run.add_argument("--new-track", type=float, default=.50)
    run.add_argument("--buffer", type=int, default=30)
    for flag in ("--match", "--second-match", "--unconfirmed-match"):
        # ByteTrack IoU *distance* bounds (1 - IoU): a higher value is looser.
        run.add_argument(flag, type=float)
    run.add_argument("--image-list", action="store_true",
                     help="VIDEO is an image list file read by VIAME's image_list reader")
    run.add_argument("--frame-time", type=float, default=1.0,
                     help="Seconds between image-list frames; sets the downsampler's input rate")
    run.add_argument("--detections", type=Path,
                     help="Saved viame_csv detections for a replay pipeline")
    run.add_argument("--timeout", type=int, default=4 * 3600)
    args = parser.parse_args()
    if args.command == "paired-clahe":
        print(json.dumps(paired_clahe(args.video, args.output), indent=2))
    else:
        print(json.dumps(run_candidate(args), indent=2))


if __name__ == "__main__":
    main()
