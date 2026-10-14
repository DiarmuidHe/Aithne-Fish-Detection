"""SAM 3 stages of ``scripts/sam3_botsort.py``: the Part 0 GPU probe, inference per prompt
variant, reliable labels, and ``run`` (the whole pipeline on one upload).

SAM 3 GPU time is charged to the journal's ``sam3_gpu`` budget (2 h for this task); a step
that would exceed it is refused before it starts and stopped if it runs over.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from scripts.sam3_autolabel import (
    SAM3_CHECKPOINT,
    SAM3_LICENCE,
    SAM3_SHA256,
    build_predictor,
    detect_video,
    infer_video,
    load_frames,
    observation_rows,
    reliable_labels,
    save_frames,
    write_candidates,
)
from scripts.sam3_botsort import (
    LABEL_RULES,
    PROMPTS,
    Journal,
    by_frame,
    compare_stage,
    file_id,
    pair_frames,
    read,
    read_rows,
    render_stage,
    save,
    settings_for,
    static_stage,
    track_rows,
    track_stage,
    viame_stage,
    video_dir,
    video_record,
    write_rows,
)

PROBE_FRAMES = 150
# SAM 3's detector on every frame, then BoT-SORT for identities. The video predictor
# (detector plus SAM 3's own mask tracker) ran at 0.29 FPS here, so a pass of this
# 115 s video would take ~83 of the 120 budgeted GPU minutes; see the probe entry.
DEFAULT_SETTINGS = {"mode": "image", "imgsz": 1008, "fp16": True, "conf": .25}
PROBE_PASSES = {"image-fp16-1008": (DEFAULT_SETTINGS, PROBE_FRAMES),
                "image-fp32-1008": (DEFAULT_SETTINGS | {"fp16": False}, 30)}


def _memory() -> dict:
    import psutil
    import torch

    return {"gpu_allocated_mb": torch.cuda.memory_allocated() / 2**20,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_reserved_mb": torch.cuda.memory_reserved() / 2**20,
            "host_rss_mb": psutil.Process().memory_info().rss / 2**20,
            "host_available_mb": psutil.virtual_memory().available / 2**20}


def _run(video: Path, prompt: str, settings: dict, *, max_frames: int | None, exemplar=None,
         heartbeat=None, every: int = 25) -> tuple[list[dict], dict]:
    """One SAM 3 pass: frames plus load time, speed and the memory curve."""
    import gc

    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    predictor, provenance = build_predictor(SAM3_CHECKPOINT, mode=settings["mode"], imgsz=settings["imgsz"],
                                            fp16=settings["fp16"], conf=settings["conf"])
    loaded = time.monotonic() - started
    curve = []

    def on_frame(row):
        if row["frame"] % every == 0:
            curve.append({"frame": row["frame"], "seconds": row["seconds"]} | _memory())
            if heartbeat:
                heartbeat(frame=row["frame"])

    if settings["mode"] == "image":
        frames = list(detect_video(predictor, video, prompt, max_frames=max_frames, on_frame=on_frame))
    else:
        frames = list(infer_video(predictor, video, prompt, exemplar=exemplar, max_frames=max_frames,
                                  on_frame=on_frame))
    inference = frames[-1]["seconds"] if frames else 0.0
    stats = {"checkpoint": provenance, "settings": settings, "prompt": prompt, "exemplar": exemplar,
             "frames": len(frames), "wall_seconds": time.monotonic() - started, "load_seconds": loaded,
             "inference_seconds": inference, "fps": len(frames) / inference if inference else None,
             "memory_curve": curve, "memory_end": _memory(), "boxes": sum(len(f["boxes"]) for f in frames),
             "scores": score_summary([s for f in frames for s in f["scores"]])}
    del predictor
    gc.collect()  # SAM 3 is built on the CPU first; give that memory back.
    torch.cuda.empty_cache()
    return frames, stats


def score_summary(scores: list[float]) -> dict:
    scores = sorted(scores)
    at = lambda q: scores[int(q * (len(scores) - 1))] if scores else None
    return {"n": len(scores), "p10": at(.1), "median": at(.5), "p90": at(.9),
            "share_at_least_0.5": sum(s >= .5 for s in scores) / len(scores) if scores else None}


def sam3_probe_stage(args) -> dict:
    """Part 0 on this GPU: does SAM 3 fit at its native 1008 px in FP16 and FP32, how fast
    is it, does memory grow over the frames, and do the boxes land on the right pixels and
    frames (overlay, and a frame-shift count against VIAME when present). Each pass is
    saved as it finishes, so a resumed probe does not repeat it."""
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts) / "sam3" / "probe"
    video = Path(facts["file"]["path"])

    def step(heartbeat):
        import torch

        result = {"gpu": torch.cuda.get_device_name(0),
                  "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory / 2**20, "runs": {}}
        for label, (settings, frames_wanted) in PROBE_PASSES.items():
            saved = directory / label / "stats.json"
            if not saved.exists():
                try:
                    frames, stats = _run(video, PROMPTS["fish"], settings, max_frames=frames_wanted,
                                         heartbeat=heartbeat, every=10)
                    save_frames(frames, directory / label / "frames.jsonl")
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    stats = {"settings": settings, "out_of_memory": str(error)[:500]}
                save(saved, stats)
            result["runs"][label] = read(saved)
        result["video_predictor"] = journal.data["entries"][f"{facts['key']}/sam3/probe"]["attempts"][0]
        frames = load_frames(directory / "image-fp16-1008" / "frames.jsonl")
        result["alignment"] = probe_alignment(root, facts, frames, directory / "overlay.jpg")
        return result

    return journal.run(f"{facts['key']}/sam3/probe", "sam3_gpu",
                       {"video": facts["file"]["sha256"], "checkpoint": SAM3_SHA256},
                       {"passes": {k: [s, n] for k, (s, n) in PROBE_PASSES.items()}}, step, estimate=600)


def probe_alignment(root: Path, facts: dict, frames: list[dict], overlay: Path) -> dict:
    """Pixel and frame alignment: SAM 3 boxes inside the 1920x1080 frame, an overlay of
    three frames with SAM 3 (cyan) and VIAME (magenta) boxes, and matched boxes against
    VIAME at frame shifts -2..+2 (a peak at 0 means the frame numbers agree)."""
    import cv2
    import numpy as np

    width, height = facts["file"]["width"], facts["file"]["height"]
    boxes = [b for f in frames for b in f["boxes"]]
    inside = sum(-1 <= b[0] < b[2] <= width + 1 and -1 <= b[1] < b[3] <= height + 1 for b in boxes)
    sam3 = by_frame(observation_rows(frames, facts["file"]["fps"]))
    viame_path = video_dir(root, facts) / "viame" / "tracks.jsonl"
    viame = by_frame(read_rows(viame_path)) if viame_path.exists() else {}
    shift = None
    if viame:
        analysed = set(read(video_dir(root, facts) / "viame" / "analysed.json")["frames"])
        shared = analysed & set(sam3)
        shift = {str(s): pair_frames(shared, viame, {f + s: v for f, v in sam3.items() if f + s in shared})["matched"]
                 for s in (-2, -1, 0, 1, 2)}
    capture, tiles = cv2.VideoCapture(facts["file"]["path"]), []
    for frame in (0, len(frames) // 2, len(frames) - 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            continue
        for colour, rows in (((255, 0, 255), viame.get(frame, [])), ((255, 255, 0), sam3.get(frame, []))):
            for row in rows:
                x1, y1, x2, y2 = map(int, row["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(image, f"frame {frame}: SAM 3 cyan, VIAME magenta ({width}x{height})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(cv2.resize(image, (960, 540)))
    capture.release()
    overlay.parent.mkdir(parents=True, exist_ok=True)
    if tiles:
        cv2.imwrite(str(overlay), np.vstack(tiles))
    return {"boxes": len(boxes), "boxes_inside_frame": inside, "frame_size": [width, height],
            "matched_with_viame_by_frame_shift": shift, "overlay": overlay.as_posix()}


def variants(args, journal: Journal) -> dict[str, dict]:
    """Prompt variants to run: --prompts, else the recorded choice, else fish and small fish.

    No exemplar-box variant: in Ultralytics' SAM 3 a box prompt describes the frame it is
    drawn on (its features are pooled there), so it cannot stand for "fish like this" on
    other frames; the video predictor applies it to its first frame only."""
    if args.prompts:
        names = args.prompts.split(",")
    elif "sam3_prompt_choice" in journal.data["decisions"]:
        names = [journal.data["decisions"]["sam3_prompt_choice"]["variant"]]
    else:
        names = ["fish", "small-fish"]
    return {name: {"prompt": PROMPTS[name]} for name in names}


def sam3_stage(args) -> dict:
    """SAM 3's detector on every frame per prompt variant, then BoT-SORT (the settings
    chosen on BrackishMOT, track lifetime converted to this video's frame rate) for
    identities. Raw detections go to frames.jsonl, tracked boxes to tracked.jsonl."""
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts) / "sam3"
    fps = facts["file"]["fps"]
    probe = journal.data["entries"].get(f"{facts['key']}/sam3/probe", {}).get("result") or {}
    speed = probe.get("runs", {}).get("image-fp16-1008", {}).get("fps") or 1.0
    tracker = settings_for(journal.data["decisions"].get("brackishmot_settings"), "botsort", fps)
    results = {}
    for name, variant in variants(args, journal).items():
        def step(heartbeat, variant=variant, name=name):
            frames, stats = _run(Path(facts["file"]["path"]), variant["prompt"], DEFAULT_SETTINGS,
                                 max_frames=None, heartbeat=heartbeat)
            if len(frames) != facts["file"]["decoded_frames"]:
                raise RuntimeError(f"SAM 3 returned {len(frames)} frames, the video has "
                                   f"{facts['file']['decoded_frames']}")
            save_frames(frames, directory / name / "frames.jsonl")
            started = time.perf_counter()
            tracked = track_rows(observation_rows(frames, fps), [f["frame"] for f in frames], tracker, fps)
            for row in tracked:
                row["id"] = f"sam3-{row['id']}"
            write_rows(tracked, directory / name / "tracked.jsonl")
            stats |= {"tracker": tracker.__dict__, "tracker_seconds": time.perf_counter() - started,
                      "tracks": len({r["id"] for r in tracked}), "tracked_boxes": len(tracked),
                      "frames_file": file_id(directory / name / "frames.jsonl"),
                      "tracked_file": file_id(directory / name / "tracked.jsonl")}
            save(directory / name / "report.json", stats | {"licence": SAM3_LICENCE,
                                                            "review_status": "unreviewed suggestions"})
            return stats

        results[name] = journal.run(f"{facts['key']}/sam3/{name}", "sam3_gpu",
                                    {"video": facts["file"]["sha256"], "checkpoint": SAM3_SHA256},
                                    DEFAULT_SETTINGS | variant | {"tracker": tracker.tag()}, step,
                                    estimate=1.1 * facts["file"]["decoded_frames"] / speed + 60)
    return {k: {x: v[x] for x in ("frames", "fps", "boxes", "tracks", "tracked_boxes", "wall_seconds")}
            for k, v in results.items()}


def variants_stage(args) -> dict:
    """Evidence for the prompt-variant choice (decision ``sam3_choice_rule``), before any
    comparison with VIAME: per variant, after static suppression, the reliable-label
    tracks it would give, and a contact sheet of boxes no other variant has."""
    import cv2
    import numpy as np

    root, facts = video_record(args)
    directory = video_dir(root, facts)
    names = sorted(p.parent.name for p in (directory / "sam3").glob("*/frames.jsonl") if p.parent.name != "probe")
    kept = {name: read_rows(directory / "static" / f"sam3-{name}" / "kept.jsonl") for name in names}
    result = {}
    for name in names:
        labels, report = reliable_labels(kept[name], **LABEL_RULES)
        others = [by_frame(kept[o]) for o in names if o != name]
        mine = by_frame(kept[name])
        unique = defaultdict(list)
        for frame, rows in mine.items():
            for row in rows:
                if all(not pair_frames([frame], {frame: [row]}, other)["matched"] for other in others):
                    unique[frame].append(row)
        static = read(directory / "static" / f"sam3-{name}" / "report.json")["counts"]
        result[name] = {"reliable_label_tracks": report["tracks"]["kept"], "reliable_label_boxes": len(labels),
                        "boxes_after_static": len(kept[name]), "static": static,
                        "boxes_no_other_variant_has": sum(map(len, unique.values()))}
        chosen = sorted(unique, key=lambda f: (-len(unique[f]), f))
        frames = []
        for frame in chosen:
            if all(abs(frame - c) > 100 for c in frames):
                frames.append(frame)
            if len(frames) == 4:
                break
        capture, tiles = cv2.VideoCapture(facts["file"]["path"]), []
        for frame in sorted(frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                continue
            for row in mine.get(frame, []):
                x1, y1, x2, y2 = map(int, row["box"])
                colour = (0, 165, 255) if row in unique[frame] else (80, 220, 80)
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(image, f"{name} frame {frame}: only this prompt orange, shared green", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            tiles.append(cv2.resize(image, (960, 540)))
        capture.release()
        if tiles:
            if len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(directory / "sam3" / f"{name}-unique.jpg"),
                        np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    save(directory / "sam3" / "variants.json", result)
    return result


def labels_stage(args) -> dict:
    """Reliable SAM 3 labels (after static suppression) as an unreviewed candidate CSV
    that ``scripts/external_fish.py cvat-suggestions`` loads into a separate CVAT task."""
    root, facts = video_record(args)
    journal = Journal(root)
    directory = video_dir(root, facts)
    choice = journal.data["decisions"].get("sam3_prompt_choice")
    name = args.prompts or (choice or {}).get("variant")
    if not name:
        raise SystemExit("Choose a prompt variant first (--prompts, or the recorded sam3_prompt_choice)")
    kept_path = directory / "static" / f"sam3-{name}" / "kept.jsonl"

    def step(heartbeat):
        rows, report = reliable_labels(read_rows(kept_path), **LABEL_RULES)
        target = directory / "labels" / name
        write_rows(rows, target / "rows.jsonl")
        count = write_candidates(rows, Path(facts["file"]["path"]).name, target / "candidates.csv")
        report |= {"rules": LABEL_RULES, "candidate_rows": count, "candidates": file_id(target / "candidates.csv"),
                   "frame_numbering": "zero-based source frames (import with frame offset 0)",
                   "review_status": "unreviewed suggestions; never ground truth", "licence": SAM3_LICENCE}
        save(target / "report.json", report)
        return report

    return journal.run(f"{facts['key']}/labels/{name}", "cpu", {"kept": file_id(kept_path)}, LABEL_RULES, step)


def run_stage(args) -> dict:
    """The whole pipeline on one upload, by video ID: VIAME's stored result (unless
    --no-viame), SAM 3, static suppression, labels, comparison, tracking and the video.
    Completed steps are skipped; nothing is trained here."""
    summary = {}
    if not args.no_viame:
        summary["viame"] = viame_stage(args)
    summary["sam3"] = sam3_stage(args)
    summary["static"] = static_stage(args)
    root, _ = video_record(args)
    journal = Journal(root)
    if "sam3_prompt_choice" in journal.data["decisions"] or args.prompts:
        summary["labels"] = labels_stage(args)
    if not args.no_viame:
        summary["compare"] = compare_stage(args)
    summary["track"] = {k: v.get("proxies") for k, v in track_stage(args).items()}
    summary["render"] = render_stage(args)
    return summary


STAGES = {"sam3-probe": sam3_probe_stage, "sam3": sam3_stage, "variants": variants_stage,
          "labels": labels_stage, "run": run_stage}
