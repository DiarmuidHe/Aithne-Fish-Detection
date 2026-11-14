"""Bigger or more gently trained students against round 1's, on round 1's own data, so the only
difference is the model. The runbook is ``docs/student-variants.md``. On the instance, from the
repository root:

    /opt/pytorch/bin/python -m scripts.student_variants STAGE [--arm ARM ...]

Stages, in order:

- ``weights``: the COCO checkpoints the chosen arms need, from RF-DETR's own bucket, each checked
  against rfdetr 1.11.2's MD5 before anything loads it. The only stage besides ``restore`` that
  uses the network.
- ``restore``: round 1's cut windows from the public archive (``hard_negatives.restore``).
- ``data``: round 1's dataset, unchanged, built once and shared by every arm. Refused if its
  counts differ from what round 1 trained on.
- ``train``: each arm in its own folder with its own journal, for round 1's epoch count. After the
  first epoch the run is stopped if the rest would not fit the arm's budget, so an arm never ends
  early on the clock and becomes unfair to compare.
- ``eval``: ``round_eval`` on round 1's validation and test days; each arm's threshold is chosen
  on validation days only, as round 1's was.
- ``compare``: round 1 and every evaluated arm side by side in ``comparison.json``.

On the PC, after the results are copied back:

- ``local``: round 1 and every arm on SAM 3-labelled clips (``--video-dir``), such as the upload
  from a camera round 1 never saw, with each student's speed on this GPU.

**What is compared.** Agreement with SAM 3's unreviewed labels (never accuracy), boxes per frame on
empty and murky water, and the boxes most likely to be background: student boxes with **no SAM 3
box near** them at all, as ``hard_negatives`` counts them. Most boxes SAM 3's labels don't match
are the same fish drawn differently, so those aren't counted as wrong.

**The arms.** Nano, Small and Medium share one backbone (DINOv2 ViT-S): a bigger one of those has a
bigger decoder, not more general knowledge. ``large-vitb`` is the one size with a bigger backbone.
A lower backbone learning rate keeps more of what the backbone learned before it saw SmartBay.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from scripts.cloud_labelling.run_windows import download
from scripts.hard_negatives import (
    all_windows,
    mine_window,
    restore,
    restored_windows,
    sam3_rows_in,
    zone_totals,
)
from scripts.sam3_autolabel import SAM3_LICENCE, UnverifiedCheckpoint
from scripts.sam3_botsort import OFFLINE, BudgetExceeded, Journal, file_id, read, read_rows, save
from scripts.student_detector import (
    LABELS,
    LOCAL_DIR,
    NAME,
    ROUND_STRIDE,
    SPLIT_DIRS,
    STRATA,
    TRAIN,
    VARIANTS,
    _train,
    coco_counts,
    extract_frames,
    local_eval,
    round_coco,
    round_eval,
    split_of,
    variant_checkpoint,
    variant_licence,
    verify_rfdetr,
)

ROUND = Path("data/outputs/smartbay-rounds/round-1")
OUT = Path("data/outputs/smartbay-rounds/round-1-variants")
GENTLE = 1.5e-5  # A tenth of RF-DETR's default backbone rate, TRAIN's lr_encoder of 1.5e-4
# Fixed before any arm ran. Budgets are training hours: Nano took about 9 min 15 s an epoch on
# the A10G, so 15 epochs about 2.3 h; Medium adds two decoder layers to the same backbone; ViT-B
# has about four times the backbone's work. They are caps, checked after the first epoch.
ARMS = {
    "nano-gentle": {"variant": "nano", "train": {"lr_encoder": GENTLE}, "model": {}, "budget_hours": 3.0,
                    "question": "Does keeping more of the pretrained backbone help?"},
    "medium": {"variant": "medium", "train": {}, "model": {}, "budget_hours": 3.5,
               "question": "Does a bigger RF-DETR (bigger decoder, same backbone) help?"},
    "vitb-gentle": {"variant": "large-vitb", "train": {"lr_encoder": GENTLE}, "model": {}, "budget_hours": 9.0,
                    "question": "Does a bigger backbone help, trained gently? Compare with nano-gentle."},
    "nano-frozen": {"variant": "nano", "train": {}, "model": {"freeze_encoder": True}, "budget_hours": 2.5,
                    "question": "Does keeping all of the backbone help?"},
}
DEFAULT_ARMS = ("nano-gentle", "medium")
SETUP_MINUTES = 75  # weights, restore (43 min last time) and the dataset build
EVAL_MINUTES = {"nano": 20, "small": 20, "medium": 25, "large-vitb": 45}  # round 1's eval took 10 min
LIVE_FPS = 15.0  # The live stream's frame rate: a student slower than this falls behind


# ---------------------------------------------------------------- weights --

def weights(arms: list[str], out: Path = OUT, fetch=download) -> dict:
    """Each size's COCO checkpoint, fetched once and kept only if rfdetr's MD5 matches."""
    fetched = {}
    for variant in sorted({ARMS[arm]["variant"] for arm in arms}):
        path = variant_checkpoint(variant)
        try:
            fetched[variant] = verify_rfdetr(variant=variant) | {"fetched": False}
            continue
        except UnverifiedCheckpoint:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        got = fetch(VARIANTS[variant]["url"], path)
        try:
            fetched[variant] = verify_rfdetr(variant=variant) | {"fetched": True, "seconds": got.get("seconds")}
        except UnverifiedCheckpoint:
            path.unlink(missing_ok=True)
            raise
    out.mkdir(parents=True, exist_ok=True)
    save(out / "weights.json", fetched)
    return fetched


# ------------------------------------------------------------------- data --

def build(root: Path = ROUND, out: Path = OUT) -> dict:
    """Round 1's dataset, rebuilt from the restored windows and checked against round 1's counts."""
    journal = Journal(out)
    validation = set(read(root / "journal.json")["decisions"]["validation_days"]["days"])
    restored = restored_windows(root, out)
    windows = [w | {"video_sha256": restored.get(w["window_id"], w.get("video_sha256"))} for w in all_windows(root)]
    base = read(root / "student" / "round_dataset.json")
    target = out / "dataset"

    def step(heartbeat):
        kept = [w for w in windows if not w["excluded"] and split_of(w["role"], w["day"], validation)]
        labels = {w["window_id"]: read_rows(root / "videos" / w["window_id"] / "labels" / LABELS / "rows.jsonl")
                  for w in kept}
        coco = round_coco(windows, labels, validation)
        if coco_counts(coco) != base["counts"]:
            raise RuntimeError(f"The dataset differs from {root}'s: {coco_counts(coco)} against {base['counts']}")
        by_key = {w["window_id"]: w for w in kept}
        for name, data in coco.items():
            folder = target / SPLIT_DIRS[name]
            folder.mkdir(parents=True, exist_ok=True)
            frames = defaultdict(dict)
            for image in data["images"]:
                frames[image["window_id"]][image["frame"]] = image["file_name"]
            for n, (key, wanted) in enumerate(frames.items(), start=1):
                extract_frames(by_key[key], wanted, folder)
                heartbeat(split=name, window=n, of=len(frames))
            save(folder / "_annotations.coco.json", data)
        result = {"dataset": target.as_posix(), "counts": coco_counts(coco), "stride": ROUND_STRIDE,
                  "validation_days": sorted(validation),
                  "windows": [{k: w.get(k) for k in ("window_id", "planned_window_id", "day", "role", "evidence",
                                                     "round", "excluded", "video", "video_sha256", "fps", "frames")}
                              | {"split": None if w["excluded"] else split_of(w["role"], w["day"], validation)}
                              for w in windows],
                  "review_status": "unreviewed SAM 3 suggestions; never ground truth", "licence": SAM3_LICENCE}
        save(out / "student" / "round_dataset.json", result)
        return result

    (out / "student").mkdir(parents=True, exist_ok=True)
    return journal.run("round/student/data", "cpu",
                       {"round_dataset": file_id(root / "student" / "round_dataset.json")["sha256"],
                        "restored": restored}, {"stride": ROUND_STRIDE, "jpeg_quality": 92}, step)


# ------------------------------------------------------------ train, eval --

def arm_name(arm: str) -> str:
    return f"rfdetr-{ARMS[arm]['variant']}"


def train(arm: str, root: Path = ROUND, out: Path = OUT) -> dict:
    """One arm from its size's COCO checkpoint, for round 1's epoch count, on the shared dataset."""
    spec = ARMS[arm]
    folder = out / arm
    journal = Journal(folder, {"training": spec["budget_hours"] * 3600})
    data = read(out / "student" / "round_dataset.json")
    epochs = read(root / "journal.json")["entries"]["round/student/train"]["result"]["epochs_run"]
    checkpoint = verify_rfdetr(variant=spec["variant"])
    models = folder / "models" / f"{arm_name(arm)}-{arm}"
    settings = TRAIN | spec["train"] | {"model": spec["model"], "variant": spec["variant"], "epochs": epochs,
                                        "resolution": VARIANTS[spec["variant"]]["resolution"]}
    (folder / "student").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "student" / "round_dataset.json", folder / "student" / "round_dataset.json")

    def step(heartbeat):
        started, budget = time.monotonic(), journal.remaining("training") - 120

        def beat(**progress):
            heartbeat(**progress)
            spent = time.monotonic() - started
            if progress.get("epoch") == 1 and spent * epochs > budget:
                raise BudgetExceeded(f"{arm}: the first epoch took {spent:.0f} s, so {epochs} epochs need about "
                                     f"{spent * epochs / 3600:.1f} h, more than the {budget / 3600:.1f} h budget")

        result = _train(Path(data["dataset"]), models / "train", epochs, budget, beat, variant=spec["variant"],
                        train_settings=spec["train"], model_settings=spec["model"])
        if result["best"] is None:
            raise RuntimeError(f"Training finished without {models / 'train' / 'checkpoint_best_total.pth'}")
        return result | {"licences": [variant_licence(spec["variant"]), SAM3_LICENCE]}

    result = journal.run("round/student/train", "training", {"dataset": data["counts"], "checkpoint": checkpoint["sha256"]},
                         settings, step)
    journal.decide("student_model", {
        "checkpoint": result["best"] | {"variant": spec["variant"]}, "trained_on": f"round {root.name}", "arm": arm,
        "name": arm_name(arm), "variant": spec["variant"], "backbone": VARIANTS[spec["variant"]]["backbone"],
        "resolution": settings["resolution"], "epochs_run": result["epochs_run"],
        "stopped_by_max_time": result["stopped_by_max_time"], "settings": settings, "coco_checkpoint": checkpoint,
        "dataset": data["counts"], "licences": [variant_licence(spec["variant"]), SAM3_LICENCE]})
    return result


def evaluate(arm: str, root: Path = ROUND, out: Path = OUT) -> dict:
    """``round_eval`` on round 1's validation and test windows, the threshold again chosen on
    validation only; SAM 3's labels come from round 1's folder."""
    return round_eval(out / arm, labels_root=root)


def students(root: Path, out: Path, *, local: bool = False) -> dict[str, dict]:
    """Round 1's student and every arm that has a trained, thresholded student under ``out``."""
    found = {}
    for name, folder in [("round-1", root), *((arm, out / arm) for arm in ARMS)]:
        path = folder / "journal.json"
        if not path.exists():
            continue
        decisions = read(path)["decisions"]
        if "student_model" in decisions and "student_threshold" in decisions and (
                local or (folder / "student" / "round_eval.json").exists()):
            found[name] = {"folder": folder, "model": decisions["student_model"],
                           "threshold": decisions["student_threshold"]["threshold"]}
    return found


def described(name: str, model: dict) -> dict:
    variant = model.get("variant", "nano")
    settings = model.get("settings", {})
    return {"variant": variant, "backbone": VARIANTS[variant]["backbone"],
            "decoder_layers": VARIANTS[variant]["decoder_layers"],
            "lr_encoder": settings.get("lr_encoder", TRAIN["lr_encoder"]),
            "freeze_encoder": settings.get("model", {}).get("freeze_encoder", False),
            "epochs_run": model.get("epochs_run"), "stopped_by_max_time": model.get("stopped_by_max_time", False),
            "question": ARMS[name]["question"] if name in ARMS else "the baseline"}


def compare(root: Path = ROUND, out: Path = OUT) -> dict:
    """Round 1 and every evaluated arm on the same test days: agreement per stratum, boxes on
    empty and murky water, boxes with no SAM 3 box near, static zones and speed."""
    windows = [w for w in read(out / "student" / "round_dataset.json")["windows"] if w["split"] == "test"]
    rows = {}
    for name, student in students(root, out).items():
        evaluation = read(student["folder"] / "student" / "round_eval.json")
        folder = student["folder"] / "student" / student["model"].get("name", NAME)
        totals = zone_totals(root, windows, lambda key, f=folder: f / key / "frames.jsonl", student["threshold"])
        agreement = evaluation["agreement"]
        rows[name] = described(name, student["model"]) | {
            "threshold": student["threshold"], "agreement": agreement["all"]["agreement"],
            "agreement_by_stratum": {s: agreement[s]["agreement"] for s in STRATA if s in agreement},
            "share_of_sam3_found": agreement["all"]["share_of_sam3_matched"],
            "student_boxes_per_frame_on_empty_water": {s: v["student_boxes_per_frame"]
                                                       for s, v in evaluation["empty_water_proxy"].items()},
            "student_boxes": totals.get("student_boxes", 0), "no_sam3_box_near": totals.get("no_sam3_box_near", 0),
            "no_sam3_box_near_share": round(totals.get("no_sam3_box_near", 0) / max(1, totals.get("student_boxes", 0)), 4),
            "hard_negative_zones": totals.get("hard_negative_zones", 0),
            "fps_median_1080p": evaluation["speed_1080p"]["fps_median"], "gpu": evaluation["speed_1080p"]["gpu"]}
    result = {"note": "Agreement with SAM 3's unreviewed labels on test days no student trained on, never accuracy. "
                      "Every student trained on the same data for the same epochs; only the model differs.",
              "test_windows": len(windows), "students": rows}
    save(out / "comparison.json", result)
    return result


# ------------------------------------------------------------------ local --

def local(video_dirs: list[Path], root: Path = ROUND / "cloud", out: Path = OUT, render: str | None = None) -> dict:
    """Each student on the PC's SAM 3-labelled clips, scored as ``compare`` scores test days, plus
    its speed on this GPU against the live stream's frame rate."""
    table = {}
    for directory in video_dirs:
        facts = read(directory / "video.json")
        # Every student here trained on round 1's windows, which are all SmartBay Cam 1
        table[facts["key"]] = {"video": facts["file"]["path"], "camera_in_training": facts["key"].startswith("SmartBayCam1_"),
                               "students": {}}
    for name, student in students(root, out, local=True).items():
        summary = local_eval(student["folder"], video_dirs, render)
        for directory in video_dirs:
            facts = read(directory / "video.json")
            key, video = facts["key"], Path(facts["file"]["path"])
            result = summary["videos"][key]
            labels, raw, removed = sam3_rows_in(directory)
            window = {"window_id": key, "fps": facts["file"]["fps"], "frames": facts["file"]["decoded_frames"]}
            mined = mine_window(window, read_rows(student["folder"] / LOCAL_DIR / key / "frames.jsonl"), labels, raw,
                                removed, student["threshold"], video if video.is_file() else None)
            fps = result["speed"]["fps"]
            table[key]["students"][name] = described(name, student["model"]) | {
                "threshold": student["threshold"], "agreement": result["agreement"]["all"]["agreement"],
                "student_boxes_per_frame": result["agreement"]["all"]["student_boxes_per_frame"],
                "sam3_boxes_per_frame": result["agreement"]["all"]["sam3_boxes_per_frame"],
                "student_boxes": mined["student_boxes"], "no_sam3_box_near": mined["unsupported_boxes"],
                "no_sam3_box_near_share": round(mined["unsupported_boxes"] / max(1, mined["student_boxes"]), 4),
                "hard_negative_zones": sum(z["verdict"] == "hard_negative" for z in mined["zones"]),
                "fps_this_pc": None if fps is None else round(fps, 1), "gpu": result["speed"].get("gpu"),
                "keeps_up_with_live": fps is not None and result["speed"]["total_ms"]["p95"] <= 1000 / LIVE_FPS}
    result = {"note": "Agreement with SAM 3's unreviewed labels, never accuracy. 'no SAM 3 box near' boxes are the "
                      "likeliest background boxes, but SAM 3 also misses fish, so look at the renders too.",
              "live_fps": LIVE_FPS, "videos": table}
    out.mkdir(parents=True, exist_ok=True)
    save(out / "local_comparison.json", result)
    return result


def guard_minutes(arms: list[str]) -> int:
    """How long the instance may run before its guard stops it: setup, then each arm's training
    budget and evaluation, plus 15 minutes' slack."""
    return SETUP_MINUTES + 15 + sum(round(ARMS[a]["budget_hours"] * 60) + EVAL_MINUTES[ARMS[a]["variant"]]
                                    for a in arms)


# -------------------------------------------------------------------- CLI --

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["weights", "restore", "data", "train", "eval", "compare", "local",
                                          "guard-minutes"])
    parser.add_argument("--arm", action="append", choices=list(ARMS),
                        help=f"train, eval, weights, guard-minutes (repeatable; default {' '.join(DEFAULT_ARMS)})")
    parser.add_argument("--root", type=Path, help="Round 1's folder (default: the instance's, or round-1/cloud for local)")
    parser.add_argument("--out", type=Path, default=OUT, help="Folder for everything this writes")
    parser.add_argument("--video-dir", type=Path, action="append", default=[],
                        help="local: a SAM 3 output folder holding video.json and labels/ (repeatable)")
    parser.add_argument("--render", nargs="?", const="compare", choices=["compare", "alone"],
                        help="local: also write an MP4 per student and clip")
    args = parser.parse_args()
    arms = args.arm or list(DEFAULT_ARMS)
    if args.stage == "guard-minutes":
        print(guard_minutes(arms))
        return
    if args.stage not in ("weights", "restore"):
        os.environ.update(OFFLINE)  # Only fetching checkpoints and windows needs the network
    root = args.root or (ROUND / "cloud" if args.stage == "local" else ROUND)
    if args.stage == "weights":
        result = weights(arms, args.out)
    elif args.stage == "restore":
        result = restore(root, args.out)
    elif args.stage == "data":
        result = build(root, args.out)
    elif args.stage == "train":
        result = {arm: train(arm, root, args.out) for arm in arms}
    elif args.stage == "eval":
        result = {arm: evaluate(arm, root, args.out) for arm in arms}
    elif args.stage == "compare":
        result = compare(root, args.out)
    else:
        if not args.video_dir:
            parser.error("local needs at least one --video-dir")
        result = local(args.video_dir, root, args.out, args.render)
    print(json.dumps(result, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
