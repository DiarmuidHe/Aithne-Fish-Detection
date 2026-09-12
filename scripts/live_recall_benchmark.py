"""Offline comparison of live sampling and input detail; no labels or paid APIs.

Run inside the existing worker with this repository available on PYTHONPATH.
Reuses the fixed clips from viame_zero_label_benchmark. Completed passes are cached.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import viame_zero_label_benchmark as benchmark


def summarize(detections, tracks, fps):
    accepted = [row for row in tracks if row["confidence"] >= .6]
    counts = Counter(row["id"] for row in accepted)
    return {"observations_at_60": len(accepted),
            "tracks_with_3_observations_at_60": sum(n >= 3 for n in counts.values()),
            "accepted_singletons": sum(n == 1 for n in counts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=["base2", "base4", "enhanced4", "detail4"])
    parser.add_argument("--clips", nargs="+", default=["1"])
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path("/data/outputs/viame-zero-label-benchmark")
    baseline = root / "baseline.pipe"
    if not baseline.exists():
        baseline.write_text(Path("/app/app/viame_pipelines/tracker_fish_no_motion.pipe").read_text())
    enhanced = root / "enhanced.pipe"
    if not enhanced.exists():
        enhanced.write_text(baseline.read_text().replace(
            "connect from downsampler.output_1\n        to   detector_input.image",
            """process enhancement
  :: image_filter
  :filter:type ocv_enhancer
  :filter:ocv_enhancer:apply_smoothing false
  :filter:ocv_enhancer:apply_denoising false
  :filter:ocv_enhancer:auto_balance false
  :filter:ocv_enhancer:force_8bit true
  :filter:ocv_enhancer:apply_clahe true
  :filter:ocv_enhancer:clip_limit 2
  :filter:ocv_enhancer:saturation 1.0
connect from downsampler.output_1
        to   enhancement.image
connect from enhancement.image
        to   detector_input.image"""))
    manifest = {"baseline_sha256": benchmark.digest(baseline),
                "enhanced_sha256": benchmark.digest(enhanced),
                "sources": {str(p): benchmark.digest(p) for p in sorted(source_root.glob("clip-*.mp4"))}}
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and benchmark.read(manifest_path) != manifest:
        raise RuntimeError("Input provenance changed")
    benchmark.save(manifest_path, manifest)
    journal = benchmark.Journal(root / "journal.json", benchmark.digest(manifest_path))
    for clip in args.clips:
        source = (Path(benchmark.read(source_root / "manifest.json")["clips"][0]["source"])
                  if clip == "full" else source_root / f"clip-{clip}.mp4")
        for profile in args.profiles:
            if profile not in {"base2", "base4", "enhanced4", "detail4"}:
                raise ValueError(profile)
            fps = 2 if profile == "base2" else 4
            height = 1080 if profile == "detail4" else 720
            path = root / f"clip-{clip}-{fps}-{height}.mp4"
            if not path.exists():
                benchmark.command(["ffmpeg", "-v", "error", "-y", "-i", source,
                    "-an", "-vf", f"fps={fps},scale=-2:{height}", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", path], timeout=60)
            config = {"path": str(path), "sampled_fps": fps,
                      "pipeline": str(enhanced if profile == "enhanced4" else baseline),
                      "score": .1, "high": .45, "low": .1, "new": .5, "buffer": 30,
                      "input_sha256": benchmark.digest(path), "acceptance": .6,
                      "fishial_enabled": False}
            benchmark.execute_run(f"{profile}-{clip}", config, root, journal)
            print(f"Completed {profile}-{clip}", flush=True)
    report = {}
    for name, run in journal.data["runs"].items():
        if run["status"] != "complete":
            report[name] = {"status": run["status"]}
            continue
        config = run["config"]
        timestamps = benchmark.pts(Path(config["path"]))
        directory = root / name
        detections, di = benchmark.observations(directory / "detections.csv", timestamps)
        tracks, ti = benchmark.observations(directory / "tracks.csv", timestamps)
        if any(value for key, value in {**di, **ti}.items() if key != "duplicate_rows"):
            raise RuntimeError(f"Invalid outputs: {di}, {ti}")
        meta = benchmark.probe(Path(config["path"]))
        summary = benchmark.metrics(detections, tracks, meta["width"], meta["height"], config["sampled_fps"])
        summary.update(summarize(detections, tracks, config["sampled_fps"]))
        summary["runtime_seconds"] = run["duration_seconds"]
        summary["duration_seconds"] = meta["duration"]
        report[name] = summary
    benchmark.save(root / "report.json", report)
    print({name: {k: row.get(k) for k in ("observations_at_60", "tracks_with_3_observations_at_60",
           "accepted_singletons", "runtime_seconds")} for name, row in report.items()}, flush=True)


if __name__ == "__main__":
    main()
