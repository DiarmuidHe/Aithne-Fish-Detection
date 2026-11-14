"""Part 1 of the SmartBay labelling pipeline: choose varied, day-disjoint 30 s
windows from the SmartBay archive scan, for SAM 3 "fish" labelling on one EC2 g5.xlarge.

PowerShell, from the repository root (no network, no GPU, no cost):

    .venv\\Scripts\\python.exe -m scripts.smartbay_label_selection

It reads only ``data/archive_scan/`` and writes ``selection/`` under
``data/outputs/sam3-cloud-labels-<UTC date>/``:

- ``day_roles.json``: every completed SmartBayCam1 day as a training, test, expert or unused
  day, assigned (seeded, spread over the weeks) before any window is chosen;
- ``manifest.csv``: one row per window (and its reserve clip), with its strata and the
  CC BY 4.0 attribution;
- ``summary.json``: each quota's target against what was achieved, and the cost estimate;
- ``contact_sheets/<evidence>.jpg``: the scan's own PNG thumbnails, one per window.

Deterministic under ``--seed``. The scan's ranks are unreviewed screening: they sort clips
here and are never labels. Expert days are reserved for a human-reviewed check set, so no
window is chosen from them; test days are labelled but never trained on.

Later rounds (Part 1 of the SmartBay rolling-training pipeline) keep an earlier
selection and add only the days completed since:

    .venv\\Scripts\\python.exe -m scripts.smartbay_label_selection `
      --keep data/outputs/sam3-cloud-labels-20261021/selection

That writes ``data/outputs/smartbay-rounds/round-<n>/selection/``. Every kept day keeps its
role, every kept window keeps its row (with a ``round`` column), and the light and clarity
tertile cuts stay the kept round's. New days get roles by the same rules at
``ROLE_PERCENT``; new windows come only from new training and test days, with the quotas
over this round's windows. ``to_label.csv`` lists the windows not yet labelled
(``--labelled`` names those that are), ``summary.json`` has this round's quotas and the
cumulative totals, and the contact sheets show this round's windows.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from scripts.smartbay_archive_scan.core import nearby_images

SCAN = Path("data/archive_scan")
OUTPUTS = Path("data/outputs")
CAMERA = "SmartBayCam1"
SEED = 17
# Confirmed on 2026-10-21: days the scan could not screen come out of training (planned
# 45 / 9 / 6 of 60) and are never used; every other screened day trains.
ROLE_DAYS = {"expert": 6, "test": 9}
# Later rounds (confirmed 2026-10-30): round 0's shares of the new completed days, rounded
# half up, and at least one of each from MIN_DAYS_FOR_EACH new days.
ROLE_PERCENT = {"expert": 10, "test": 15}
MIN_DAYS_FOR_EACH = 7
WINDOWS_PER_DAY = {"train": 3, "test": 2}
WINDOW_SECONDS = 30
# The scan's clips: 1,680 frames at 15000/1001 FPS (112 s). Most are; a few are short, so
# every window has a reserve, and the instance checks the real length before cutting.
CLIP_FRAMES, CLIP_FPS = 1680, 15000 / 1001
WINDOW_FRAMES = round(WINDOW_SECONDS * CLIP_FPS)
# Windows on one day at least 30 minutes apart: clip starts at least 30 min + one clip
# apart guarantee it wherever the windows sit inside their clips.
MIN_CLIP_GAP = timedelta(minutes=30) + timedelta(seconds=math.ceil(CLIP_FRAMES / CLIP_FPS))
EVIDENCE_SHARE = {"strong": .55, "weak": .15, "none": .15, "murky": .15}
LIGHT_MIN, HOUR_MIN = .20, .15
HOUR_BINS = ("00-06", "06-12", "12-18", "18-24")
TERTILES = ("low", "mid", "high")
BAD = {"black", "camera_off", "frozen"}
# Days a model has already been developed or checked on: never expert days.
TOUCHED_DAYS = {"2025-07-17": "Cam 2's video of this date trained the RF-DETR student (2026-10-21)",
                "2025-08-01": "the RF-DETR student's Cam 1 evaluation and the Part 0 download (2026-10-21)"}
# Other cameras (round 2, 2026-11-14). A day of another camera is keyed "<date>@<camera>", so
# its role is its own: Cam 2 gets its own training, test and expert days. Cam 2's clips are
# 1,440 frames at 12.5 FPS (every clip the scan sampled), Cam 1's 1,680 at 15000/1001.
CAMERAS = {CAMERA: {"clip_frames": 1680, "fps": 15000 / 1001},
           "SmartBayCam2": {"clip_frames": 1440, "fps": 12.5}}
# Cam 2's 2025-07-17 is the PC's one SAM 3-labelled clip from a camera round 1 never saw
# (upload 85ea19d4): held out, so it stays an unseen day once Cam 2 trains.
FORCED_ROLES = {"2025-07-17@SmartBayCam2": ("test", ("the PC's SAM 3-labelled Cam 2 clip (upload 85ea19d4) is from "
                                                     "this day; it trained the smoke-test student, never round 2's"))}
# Estimate inputs; Part 3 replaces the speed with one measured on the A10G.
ASSUMED_FPS = 2.5           # SAM 3 detector, FP16, 1008 px, on this laptop
LOAD_SECONDS = 17           # SAM 3 load per window (16-18 s on this laptop)
CPU_SECONDS = 20            # static suppression, labels, download and cut per window
# g5.xlarge on-demand, eu-west-1, from third-party copies of the price list (DevZero,
# aws-pricing.com, 2026-10-22); the 'default' profile may not read AWS's own (pricing:GetProducts).
ON_DEMAND_USD_PER_HOUR = 1.123
SETUP_HOURS = 1.0           # bootstrap, smoke run and the 20-minute probe
# Round estimates: the laptop's measured sam3 stage (450 frames at 2.12 FPS, load timed
# apart) and a guess for the A10G, which Part 3 replaces; training shares the instance.
ROUND_FPS = (2.12, 4.0)
TRAINING_HOURS = 3.0
ATTRIBUTION =("Video: Marine Institute SmartBay Observatory (https://smartbay.marine.ie), "
               "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)")


def utc_today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def day_key(camera: str, day: str) -> str:
    """A day's key in the roles: the date for Cam 1, as every earlier round wrote it, and
    "<date>@<camera>" for another camera."""
    return day if camera == CAMERA else f"{day}@{camera}"


def split_key(key: str) -> tuple[str, str]:
    """A day key's camera and date."""
    day, _, camera = key.partition("@")
    return camera or CAMERA, day


def date_of(key: str) -> date:
    return date.fromisoformat(split_key(key)[1])


def window_frames(camera: str = CAMERA) -> int:
    return round(WINDOW_SECONDS * CAMERAS[camera]["fps"])


# ---------------------------------------------------------------- the scan --

def completed_days(scan: Path = SCAN) -> list[str]:
    with (scan / f"{CAMERA}_2025" / "year_progress.csv").open(encoding="utf-8") as stream:
        return sorted(row["day"] for row in csv.DictReader(stream) if row["status"] == "completed")


def ledger_days(scan: Path = SCAN) -> list[str]:
    """Days both ledger files call completed, so a pull caught between writing the two can't
    add the day in progress."""
    state = json.loads((scan / f"{CAMERA}_2025" / "year_state.json").read_text(encoding="utf-8"))
    return sorted(set(completed_days(scan)) & set(state["completed_days"]))


def scanned_days(camera: str, scan: Path = SCAN) -> list[str]:
    """Another camera's days, from each day's own report rather than a year ledger: Cam 2's
    were scanned as a sample and then one day at a time. A day counts when every PNG and MP4
    it selected was processed."""
    days = []
    for path in sorted(scan.glob(f"{camera}_2025-*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (summary.get("processed_pngs") == summary.get("selected_pngs")
                and summary.get("processed_mp4s") == summary.get("selected_mp4s")):
            days.append(summary["day"])
    return days


def pick_scan_days(camera: str, count: int, scan: Path = SCAN, seed: int = SEED) -> list[str]:
    """``count`` more of a camera's 2025 archive days to scan: one seeded pick from each
    consecutive slice of the days its year ledger discovered and no scan has touched, so they
    spread over the season as the first sample did."""
    state = json.loads((scan / f"{camera}_2025" / "year_state.json").read_text(encoding="utf-8"))
    touched = {p.name.removeprefix(f"{camera}_") for p in scan.glob(f"{camera}_2025-*") if p.is_dir()}
    left = [d for d in sorted(state["discovered_days"]) if d not in touched]
    if count >= len(left):
        return left
    rng = random.Random(seed)
    return [rng.choice(left[i * len(left) // count:(i + 1) * len(left) // count]) for i in range(count)]


def load_day(day: str, scan: Path = SCAN, camera: str = CAMERA) -> tuple[dict, list[dict]]:
    directory = scan / f"{camera}_{day}"
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    with (directory / "review_manifest.csv").open(encoding="utf-8") as stream:
        return state, list(csv.DictReader(stream))


def evidence_of(rank: str, reason: str, nearby: list[dict], scan_error: str = "",
                frames: int | None = None, window: int = WINDOW_FRAMES) -> tuple[str | None, str]:
    """A clip's evidence stratum, or None and why it is excluded. Murky clips (``unusable``
    only because their PNGs are ``obscured``) are kept: murky water is what the student must
    handle, though the scan's quality rule drops it. ``frames`` is the clip's length where
    the scan sampled it."""
    qualities = {p.get("quality") for p in nearby}
    if not nearby:
        return None, "no PNG coverage"
    if scan_error or any(p.get("error") for p in nearby):
        return None, "scan error"
    if frames is not None and frames < window:
        return None, "clip shorter than the window (scan sampled it)"
    if qualities <= BAD:
        return None, "PNGs all black, camera-off or frozen"
    if rank == "likely_fish":
        return "strong", ""
    if rank == "uncertain" and "weaker SAM 3 evidence" in reason:
        return "weak", ""
    if rank == "candidate_clear_and_empty":
        return "none", ""
    if rank == "unusable" and qualities == {"obscured"}:
        return "murky", ""
    return None, f"in no stratum ({rank}; PNGs {'+'.join(sorted(q or '?' for q in qualities))})"


def describe_day(day: str, state: dict, rows: list[dict], camera: str = CAMERA) -> list[dict]:
    """Every clip of a day with its evidence, light, clarity, hour and density, from the
    PNGs ``rank_clip`` pairs it with (1 minute before to 3 minutes after its start). Each
    clip's ``day`` is its day key (``day_key``)."""
    thumbs = [u for u in state.get("selected_png_urls", []) if "thumbnail" in state["images"].get(u, {})]
    thumb_index = {u: i for i, u in enumerate(thumbs)}
    images = [{**e, **state["images"][e["url"]], "thumb_index": thumb_index.get(e["url"])}
              for e in state["entries"] if e["url"] in state["images"]]
    entries = {e["url"]: e for e in state["entries"] if e["kind"] == "mp4"}
    clips = []
    for row in rows:
        entry = entries[row["source_mp4_url"]]
        nearby = nearby_images(entry, images)
        evidence, excluded = evidence_of(row["screening_rank"], row["rank_reason"], nearby, row.get("scan_error", ""),
                                         state["clips"].get(entry["url"], {}).get("frame_count"), window_frames(camera))
        seen = [p for p in nearby if p.get("quality") not in BAD and "metrics" in p]
        usable = [p for p in nearby if p.get("quality") == "usable"]
        detections = max((p.get("detection_count", 0) for p in usable), default=0)
        clips.append({
            "day": day_key(camera, day), "camera": camera, "url": entry["url"], "stem": Path(entry["name"]).stem,
            "clip_time": entry["timestamp"],
            "rank": row["screening_rank"], "rank_reason": row["rank_reason"], "evidence": evidence,
            "excluded": excluded, "nearby": nearby,
            "light_mean": statistics.median(p["metrics"]["mean"] for p in seen) if seen else None,
            "contrast": statistics.median(p["metrics"]["contrast"] for p in seen) if seen else None,
            "hour_bin": HOUR_BINS[int(entry["timestamp"][11:13]) // 6],
            "density": density_bin(detections) if evidence == "strong" else "",
            "png_detections": detections,
            "png_max_score": max((p.get("max_score", 0) for p in usable), default=0)})
    return clips


def describe_days(days: list[str], scan: Path = SCAN, camera: str = CAMERA) -> tuple[list[dict], list[dict]]:
    """Every clip of one camera's days, and every PNG the scan recorded on them."""
    clips, images = [], []
    for day in days:
        state, rows = load_day(day, scan, camera)
        clips += describe_day(day, state, rows, camera)
        images += list(state["images"].values())
    return clips, images


def density_bin(detections: int) -> str:
    if detections <= 0:
        return "0 (MP4 frames only)"
    return "1-5" if detections <= 5 else "6-15" if detections <= 15 else "16+"


def tertile_cuts(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return values[len(values) // 3], values[2 * len(values) // 3]


def tertile(value: float, cuts: tuple[float, float]) -> str:
    return TERTILES[0] if value < cuts[0] else TERTILES[1] if value < cuts[1] else TERTILES[2]


def add_tertiles(clips: list[dict], images: list[dict], cuts: dict | None = None) -> dict:
    """Light and clarity tertiles, cut over every eligible PNG (not black, camera-off or
    frozen) of the screened days, or at an earlier round's ``cuts``."""
    seen = [p for p in images if p.get("quality") not in BAD and "metrics" in p]
    cuts = cuts or {"light": tertile_cuts([p["metrics"]["mean"] for p in seen]),
                    "clarity": tertile_cuts([p["metrics"]["contrast"] for p in seen])}
    for clip in clips:
        if clip["light_mean"] is not None:
            clip["light"] = tertile(clip["light_mean"], cuts["light"])
            clip["clarity"] = tertile(clip["contrast"], cuts["clarity"])
    return {k: [round(v, 2) for v in cuts[k]] for k in ("light", "clarity")} | {"eligible_pngs": len(seen)}


# --------------------------------------------------------------- day roles --

def capacity(clips: list[dict]) -> int:
    """How many windows a day's eligible clips can hold at ``MIN_CLIP_GAP`` (earliest-first
    is optimal for points on a line)."""
    count, last = 0, None
    for start in sorted(datetime.fromisoformat(c["clip_time"]) for c in clips if c["evidence"]):
        if last is None or start - last >= MIN_CLIP_GAP:
            count, last = count + 1, start
    return count


def assign_roles(days: list[str], room: dict[str, int], seed: int = SEED, counts: dict = ROLE_DAYS,
                 touched: dict = TOUCHED_DAYS, first: str | None = None, weekly: bool = True) -> dict[str, str]:
    """Day roles before any window: expert days first, then test days, each one seeded pick
    from its own consecutive stretch of the screened days, so both spread over the season;
    a pick that would leave a week without a training day is skipped. Every other screened
    day trains. ``room`` is each day's capacity: a day that can't hold a test day's windows
    is unscreened and gets no role ("unused"); one that can't hold a training day's is
    taken as its stretch's test day. Touched days are never expert days. Weeks count from
    ``first`` (default the first day). ``weekly`` False drops the week rule, for a camera
    scanned as a sparse sample, where most weeks hold one day."""
    first = date_of(first or days[0])
    week = lambda d: (date_of(d) - first).days // 7  # noqa: E731
    pool = [d for d in days if room.get(d, 0) >= WINDOWS_PER_DAY["test"]]
    roles = {d: "unused" for d in days if d not in pool}
    rng = random.Random(seed)

    def keeps_training(day: str) -> bool:
        return not weekly or any(d != day and d not in roles and week(d) == week(day)
                                 and room[d] >= WINDOWS_PER_DAY["train"] for d in pool)

    for role, allowed in (("expert", lambda d: d not in touched), ("test", lambda d: True)):
        n = counts[role]
        for i in range(n):
            stretch = pool[round(i * len(pool) / n):round((i + 1) * len(pool) / n)]
            candidates = [d for d in stretch if d not in roles and allowed(d) and keeps_training(d)]
            if role == "test":
                candidates = [d for d in candidates if room[d] < WINDOWS_PER_DAY["train"]] or candidates
            if not candidates:
                raise RuntimeError(f"no {role} day possible in {stretch[0]}..{stretch[-1]}")
            roles[rng.choice(candidates)] = role
    for day in pool:
        if day not in roles and room[day] < WINDOWS_PER_DAY["train"]:
            raise RuntimeError(f"{day} holds only {room[day]} windows, too few for a training day")
        roles.setdefault(day, "train")
    return dict(sorted(roles.items()))


def round_role_counts(new_days: int) -> dict[str, int]:
    """A later round's expert and test days: ``ROLE_PERCENT`` of its new completed days,
    rounded half up, and at least one of each from ``MIN_DAYS_FOR_EACH`` new days."""
    least = 1 if new_days >= MIN_DAYS_FOR_EACH else 0
    return {role: max(least, (percent * new_days + 50) // 100) for role, percent in ROLE_PERCENT.items()}


# ----------------------------------------------------------------- windows --

def place(rng: random.Random, clip_frames: int = CLIP_FRAMES, count: int = WINDOW_FRAMES) -> int:
    """A seeded start frame that leaves the full window inside the clip."""
    if clip_frames < count:
        raise ValueError(f"a {clip_frames}-frame clip can't hold a {count}-frame window")
    return rng.randint(0, clip_frames - count)


def quota_targets(total: int) -> dict[str, int]:
    """Evidence shares as whole windows (largest remainder; ties go to the order above)."""
    exact = {k: total * v for k, v in EVIDENCE_SHARE.items()}
    targets = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - targets[k]))[:total - sum(targets.values())]:
        targets[k] += 1
    return targets


def choose_windows(clips: list[dict], roles: dict[str, str], seed: int = SEED) -> tuple[list[dict], dict]:
    """Fill the evidence quotas, scarcest stratum first, one window at a time: the day with
    the fewest windows of that stratum so far, then the role (training or test) furthest
    below its share of the stratum, then the clip that most reduces the light and hour
    shortfalls and spreads clarity and density. No clip is used twice, and clips on one day
    are at least ``MIN_CLIP_GAP`` apart. Each window then gets a reserve clip from its own
    day and stratum, in case its clip turns out short or fails to download. A slot still
    empty after the final strong fill is reported as ``unfilled_slots``, never padded."""
    rng = random.Random(seed)
    slots = {d: WINDOWS_PER_DAY[r] for d, r in roles.items() if r in WINDOWS_PER_DAY}
    total = sum(slots.values())
    role_share = {r: sum(n for d, n in slots.items() if roles[d] == r) / total for r in WINDOWS_PER_DAY}
    pool = [c for c in clips if c["evidence"] and c["day"] in slots]
    targets = quota_targets(total)
    light_min, hour_min = math.ceil(LIGHT_MIN * total), math.ceil(HOUR_MIN * total)
    hours_with_footage = {c["hour_bin"] for c in pool}
    counts = {k: Counter() for k in ("light", "hour", "clarity", "density")}
    per_day = defaultdict(list)
    used: set[str] = set()
    windows = []
    by_stratum = defaultdict(list)
    for clip in pool:
        by_stratum[clip["evidence"]].append(clip)
    order = sorted(targets, key=lambda e: (len({c["day"] for c in by_stratum[e]}), list(targets).index(e)))
    shortfalls, unfilled = {}, 0
    starts ={c["url"]: datetime.fromisoformat(c["clip_time"]) for c in pool}

    def spaced(clip: dict, others: list[dict]) -> bool:
        return all(abs(starts[clip["url"]] - starts[o["url"]]) >= MIN_CLIP_GAP for o in others)

    def score(clip: dict) -> tuple:
        light = max(0, light_min - counts["light"][clip["light"]]) / light_min
        hour = max(0, hour_min - counts["hour"][clip["hour_bin"]]) / hour_min
        spread = -(counts["clarity"][clip["clarity"]] + counts["density"][clip["density"]]) / total
        return light + hour, spread

    def geometry(clip: dict) -> tuple[int, int]:
        camera = clip.get("camera", CAMERA)
        return CAMERAS[camera]["clip_frames"], window_frames(camera)

    def take(clip: dict, reason: str) -> None:
        start = place(rng, *geometry(clip))
        window = clip | {"start_frame": start, "frame_count": geometry(clip)[1], "role": roles[clip["day"]],
                         "window_id": f"{clip['stem']}_f{start:04d}", "reason": reason}
        windows.append(window)
        per_day[clip["day"]].append(window)
        used.add(clip["url"])
        for key, value in (("light", clip["light"]), ("hour", clip["hour_bin"]), ("clarity", clip["clarity"]),
                           ("density", clip["density"])):
            counts[key][value] += 1

    # Strong comes again at the end: it fills any slot a scarce stratum could not.
    for evidence, need in [(e, targets[e]) for e in order] + [("strong", None)]:
        for _ in range(total - len(windows) if need is None else need):
            best, key = None, None
            in_role = Counter(w["role"] for w in windows if w["evidence"] == evidence)
            taken = sum(in_role.values())
            for clip in by_stratum[evidence]:
                day = clip["day"]
                if clip["url"] in used or len(per_day[day]) >= slots[day] or not spaced(clip, per_day[day]):
                    continue
                role = roles[day]
                candidate = (-sum(w["evidence"] == evidence for w in per_day[day]),
                             round((taken + 1) * role_share[role] - in_role[role], 6), *score(clip), rng.random())
                if key is None or candidate > key:
                    best, key = clip, candidate
            if best is None:
                if need is None:  # the strong quota already counted this slot's miss
                    unfilled += 1
                else:
                    shortfalls[evidence] = shortfalls.get(evidence, 0) + 1
                continue
            take(best, f"{evidence} quota; " + ", ".join(
                f"{name} {value}" for name, value in (("light", best["light"]), ("hour", best["hour_bin"]))))
    for window in windows:
        others = [o for w in per_day[window["day"]] if w is not window for o in (w, w.get("reserve")) if o]
        options = [c for c in by_stratum[window["evidence"]] if c["day"] == window["day"]
                   and c["url"] not in used and spaced(c, others)]
        options.sort(key=lambda c: (c["light"] != window["light"], c["hour_bin"] != window["hour_bin"], c["clip_time"]))
        if options:
            reserve = options[0]
            used.add(reserve["url"])
            start = place(rng, *geometry(reserve))
            window["reserve"] = {"url": reserve["url"], "clip_time": reserve["clip_time"], "start_frame": start,
                                 "window_id": f"{reserve['stem']}_f{start:04d}"}
    windows.sort(key=lambda w: (w["day"], w["clip_time"]))
    achieved = {
        "evidence": {e: {"target": targets[e], "achieved": sum(w["evidence"] == e for w in windows)} for e in targets},
        "light": {t: {"target_at_least": light_min, "achieved": counts["light"][t]} for t in TERTILES},
        "hour": {h: {"target_at_least": hour_min if h in hours_with_footage else 0,
                     "usable_footage": h in hours_with_footage, "achieved": counts["hour"][h]} for h in HOUR_BINS},
        "clarity": dict(sorted(counts["clarity"].items())),
        "density_strong_only": dict(sorted((k, v) for k, v in counts["density"].items() if k)),
        "evidence_shortfalls": shortfalls,
        "light_shortfalls": {t: light_min - counts["light"][t] for t in TERTILES if counts["light"][t] < light_min},
        "hour_shortfalls": {h: hour_min - counts["hour"][h] for h in hours_with_footage
                            if counts["hour"][h] < hour_min},
    }
    if unfilled:  # a key only when some slot is empty, as the 2026-10-21 selection had none
        achieved["unfilled_slots"] = unfilled
    return windows, achieved


def plan_round(days: list[str], clips: list[dict], room: dict[str, int], kept_roles: dict[str, str],
               seed: int = SEED, touched: dict = TOUCHED_DAYS) -> tuple[dict[str, str], list[dict], dict]:
    """A later round's new day roles and windows. New days are the completed days without a
    role; ``assign_roles`` gives theirs at ``round_role_counts``, with weeks counted from the
    first day of every round. Windows come only from new training and test days, with the
    quotas over this round's windows alone, so no kept day gains or loses a window."""
    new_days = [d for d in days if d not in kept_roles]
    if not new_days:
        raise RuntimeError("every completed day already has a role: nothing new since the kept round")
    roles = assign_roles(new_days, room, seed, round_role_counts(len(new_days)), touched,
                         min([*kept_roles, *new_days]))
    if not any(r in WINDOWS_PER_DAY for r in roles.values()):
        raise RuntimeError(f"no new day can hold windows ({len(new_days)} new days, all {set(roles.values())})")
    windows, achieved = choose_windows([c for c in clips if c["day"] in roles], roles, seed)
    return roles, windows, achieved


# ------------------------------------------------------------------ output --

@functools.lru_cache(maxsize=8)
def _sheet(path: str):
    import cv2

    return cv2.imread(path)


def thumbnail(scan: Path, day: str, index: int | None):
    """The scan's 400x252 thumbnail of a PNG, cut from the day's PNG contact sheet (40 to a
    sheet, 4 across); the separate thumbnail files were cleaned up after each day. ``day`` is
    a day key."""
    if index is None:
        return None
    camera, date_text = split_key(day)
    name = "png_pilot.jpg" if index < 40 else f"png_page_{index // 40 + 1:03d}.jpg"
    sheet = _sheet(str(scan / f"{camera}_{date_text}" / "contact_sheets" / name))
    if sheet is None:
        return None
    row, col = divmod(index % 40, 4)
    return sheet[row * 252:(row + 1) * 252, col * 400:(col + 1) * 400].copy()


def contact_sheet(windows: list[dict], scan: Path, path: Path) -> None:
    """One tile per window: its nearest PNG's scan thumbnail, labelled with the window."""
    import cv2
    import numpy as np

    tiles = []
    for window in windows:
        fps = CAMERAS[window.get("camera", CAMERA)]["fps"]
        start = datetime.fromisoformat(window["clip_time"]) + timedelta(seconds=window["start_frame"] / fps)
        nearest = min(window["nearby"], key=lambda p: abs(datetime.fromisoformat(p["timestamp"]) - start))
        tile = thumbnail(scan, window["day"], nearest.get("thumb_index"))
        if tile is None:
            tile = np.full((252, 400, 3), 60, np.uint8)
        cv2.rectangle(tile, (0, 0), (400, 40), (30, 30, 30), -1)
        cv2.putText(tile, window["window_id"].removeprefix(f"{CAMERA}_"), (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"{window['role']} | light {window['light']} | clarity {window['clarity']} | "
                          f"{window['hour_bin']}", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 220, 255), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def estimate(windows: list[dict]) -> dict:
    frames = sum(w["frame_count"] for w in windows)
    per_window = WINDOW_FRAMES / ASSUMED_FPS + LOAD_SECONDS + CPU_SECONDS
    hours = len(windows) * per_window / 3600
    return {"windows": len(windows), "frames": frames, "assumed_sam3_fps": ASSUMED_FPS,
            "assumed_load_seconds_per_window": LOAD_SECONDS, "assumed_cpu_seconds_per_window": CPU_SECONDS,
            "sam3_gpu_hours": round(frames / ASSUMED_FPS / 3600, 2),
            "instance_hours_one_process": round(hours, 2), "setup_hours": SETUP_HOURS,
            "on_demand_usd_per_hour": ON_DEMAND_USD_PER_HOUR,
            "instance_usd_one_process": round((hours + SETUP_HOURS) * ON_DEMAND_USD_PER_HOUR, 2),
            "note": "laptop speed (RTX 5060, thermally throttled); Part 3 measures the A10G and one against two "
                    "processes, and re-estimates. EBS, S3 and transfer are estimated separately."}


MANIFEST_FIELDS = ["window_id", "day", "role", "source_mp4_url", "clip_time", "start_frame", "frame_count",
                   "assumed_clip_frames", "fps", "evidence", "light", "light_mean", "clarity", "contrast", "hour_bin",
                   "density", "screening_rank", "rank_reason", "png_max_score", "png_detections", "nearby_pngs",
                   "reason", "reserve_window_id", "reserve_url", "reserve_start_frame", "attribution"]


def manifest_row(window: dict) -> dict:
    reserve = window.get("reserve", {})
    camera = CAMERAS[window.get("camera", CAMERA)]
    return {"window_id": window["window_id"], "day": window["day"], "role": window["role"],
            "source_mp4_url": window["url"], "clip_time": window["clip_time"], "start_frame": window["start_frame"],
            "frame_count": window["frame_count"], "assumed_clip_frames": camera["clip_frames"],
            "fps": f"{camera['fps']:.6f}",
            "evidence": window["evidence"], "light": window["light"], "light_mean": window["light_mean"],
            "clarity": window["clarity"], "contrast": window["contrast"], "hour_bin": window["hour_bin"],
            "density": window["density"], "screening_rank": window["rank"], "rank_reason": window["rank_reason"],
            "png_max_score": window["png_max_score"], "png_detections": window["png_detections"],
            "nearby_pngs": len(window["nearby"]), "reason": window["reason"],
            "reserve_window_id": reserve.get("window_id", ""), "reserve_url": reserve.get("url", ""),
            "reserve_start_frame": reserve.get("start_frame", ""), "attribution": ATTRIBUTION}


ROLE_RULE = ("expert days, then test days: one seeded pick from each consecutive stretch of the screened days, "
             "never leaving a week without a training day; a day holding fewer than 3 windows is its stretch's "
             "test day; the rest train")
UNUSED_RULE = "completed days the scan could not screen: no PNGs, or room for fewer than 2 windows"
ROLE_MEANING = {"train": "labelled, trained on", "test": "labelled, never trained on",
                "expert": "reserved for a human-reviewed check set; nothing downloaded or labelled here",
                "unused": "no role"}


def by_week(roles: dict[str, str]) -> dict:
    """Each week's days by role, weeks counted from the first day."""
    first = min(date_of(d) for d in roles)
    weeks = defaultdict(lambda: defaultdict(list))
    for day, role in roles.items():
        weeks[f"week {(date_of(day) - first).days // 7 + 1}"][role].append(day)
    return {w: dict(r) for w, r in weeks.items()}


def write_csv(path: Path, rows, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheets(windows: list[dict], scan: Path, out: Path) -> None:
    for evidence in EVIDENCE_SHARE:
        contact_sheet([w for w in windows if w["evidence"] == evidence], scan, out / "contact_sheets" / f"{evidence}.jpg")


def select(scan: Path, out: Path, seed: int = SEED) -> dict:
    days = completed_days(scan)
    clips, images = describe_days(days, scan)
    room = {day: capacity([c for c in clips if c["day"] == day]) for day in days}
    cuts = add_tertiles(clips, images)
    roles = assign_roles(days, room, seed)
    screened = {d for d, r in roles.items() if r != "unused"}
    windows, achieved = choose_windows(clips, roles, seed)

    out.mkdir(parents=True, exist_ok=True)
    (out / "day_roles.json").write_text(json.dumps({
        "seed": seed, "counts": dict(Counter(roles.values())), "planned": ROLE_DAYS | {"train": "the rest"},
        "rule": ROLE_RULE, "never_expert": TOUCHED_DAYS, "unused": UNUSED_RULE,
        "capacity": {d: room[d] for d in days}, "meaning": ROLE_MEANING,
        "roles": roles, "by_week": by_week(roles)}, indent=2), encoding="utf-8")
    write_csv(out / "manifest.csv", (manifest_row(w) for w in windows), MANIFEST_FIELDS)
    contact_sheets(windows, scan, out)
    eligible = [c for c in clips if c["evidence"] and roles[c["day"]] in WINDOWS_PER_DAY]
    summary = {
        "seed": seed, "scan": scan.as_posix(), "completed_days": len(days), "screened_days": len(screened),
        "window_seconds": WINDOW_SECONDS, "window_frames": WINDOW_FRAMES, "min_clip_gap_minutes":
            MIN_CLIP_GAP.total_seconds() / 60,
        "clips": len(clips), "excluded": dict(Counter(c["excluded"] for c in clips if not c["evidence"])),
        "eligible_by_evidence_all_days": dict(Counter(c["evidence"] for c in clips if c["evidence"])),
        "eligible_by_evidence_train_test_days": dict(Counter(c["evidence"] for c in eligible)),
        "tertile_cuts": cuts, "roles": dict(Counter(roles.values())),
        "windows_by_role": dict(Counter(w["role"] for w in windows)),
        "windows_with_reserve": sum("reserve" in w for w in windows), "quotas": achieved,
        "evidence_by_role": {r: dict(Counter(w["evidence"] for w in windows if w["role"] == r))
                             for r in WINDOWS_PER_DAY},
        "estimate": estimate(windows), "attribution": ATTRIBUTION,
        "interpretation": "Unreviewed screening sorts the clips; it is never a label."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ------------------------------------------------------------ later rounds --

def kept_round(keep: Path) -> int:
    """The round an earlier selection made; the first, made without ``--keep``, is round 0."""
    return json.loads((keep / "day_roles.json").read_text(encoding="utf-8")).get("round", 0)


def read_kept(keep: Path) -> dict:
    """An earlier selection: its day roles and the round that gave each, its windows exactly
    as written (with their round), its tertile cuts and its files' SHA256s. A window whose
    role is not its day's training or test role is refused."""
    day_roles = json.loads((keep / "day_roles.json").read_text(encoding="utf-8"))
    cuts = json.loads((keep / "summary.json").read_text(encoding="utf-8"))["tertile_cuts"]
    n = day_roles.get("round", 0)
    with (keep / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row | {"round": row.get("round") or str(n)} for row in csv.DictReader(stream)]
    roles = day_roles["roles"]
    wrong = [r["window_id"] for r in rows if r["role"] not in WINDOWS_PER_DAY or roles.get(r["day"]) != r["role"]]
    if wrong:
        raise ValueError(f"{keep}: windows whose role is not their day's training or test role: {wrong[:5]}")
    return {"round": n, "roles": roles, "day_round": day_roles.get("day_round") or dict.fromkeys(roles, n),
            "capacity": day_roles["capacity"], "rows": rows, "cuts": {k: cuts[k] for k in ("light", "clarity")},
            "sha256": {name: hashlib.sha256((keep / name).read_bytes()).hexdigest()
                       for name in ("day_roles.json", "manifest.csv", "summary.json")}}


TALLY_ORDER = {"round": None, "role": tuple(WINDOWS_PER_DAY), "evidence": tuple(EVIDENCE_SHARE), "light": TERTILES,
               "hour_bin": HOUR_BINS, "clarity": TERTILES}


def tally(rows: list[dict]) -> dict:
    """Manifest rows counted by round, role, evidence, light, hour and clarity, and evidence
    by role."""
    counts = {column: Counter(str(r[column]) for r in rows) for column in TALLY_ORDER}
    return {"windows": len(rows),
            **{column: {k: counts[column][k] for k in order or sorted(counts[column])}
               for column, order in TALLY_ORDER.items()},
            "evidence_by_role": {role: {e: sum(r["role"] == role and r["evidence"] == e for r in rows)
                                        for e in EVIDENCE_SHARE} for role in WINDOWS_PER_DAY},
            "with_reserve": sum(bool(r["reserve_window_id"]) for r in rows)}


def round_estimate(rows: list[dict]) -> dict:
    """Instance time and cost to label ``rows`` with one process and then train, at each of
    ``ROUND_FPS``."""
    frames = sum(int(r["frame_count"]) for r in rows)
    at = {}
    for fps in ROUND_FPS:
        labelling = (frames / fps + len(rows) * (LOAD_SECONDS + CPU_SECONDS)) / 3600
        hours = labelling + SETUP_HOURS + TRAINING_HOURS
        at[f"{fps:g}"] = {"sam3_gpu_hours": round(frames / fps / 3600, 2),
                          "labelling_hours_one_process": round(labelling, 2), "instance_hours": round(hours, 2),
                          "instance_usd": round(hours * ON_DEMAND_USD_PER_HOUR, 2)}
    return {"windows": len(rows), "frames": frames, "assumed_load_seconds_per_window": LOAD_SECONDS,
            "assumed_cpu_seconds_per_window": CPU_SECONDS, "setup_hours": SETUP_HOURS,
            "training_hours": TRAINING_HOURS, "on_demand_usd_per_hour": ON_DEMAND_USD_PER_HOUR, "at_sam3_fps": at,
            "note": "2.12 FPS is the laptop's sam3 stage on one window; 4 FPS is a guess for the A10G, which Part 3 "
                    "measures with one process against two. EBS and transfer are estimated separately."}


def plan_camera(camera: str, scan: Path, kept: dict, seed: int = SEED) -> dict | None:
    """Another camera's new days, roles and windows for a round: its days are keyed
    ``<date>@<camera>`` and get roles of their own at ``round_role_counts``, with
    ``FORCED_ROLES`` applied after; its windows fill the quotas within the camera. The
    tertile cuts stay the kept round's, so "light" means the same on every camera. None
    when the camera has no new day."""
    days = scanned_days(camera, scan)
    keys = {day_key(camera, d): d for d in days}
    if lost := sorted(k for k in kept["roles"] if split_key(k)[0] == camera and k not in keys):
        raise RuntimeError(f"kept {camera} days the scan doesn't have: {lost[:5]}; is --scan right?")
    new_days = [d for d in days if day_key(camera, d) not in kept["roles"]]
    if not new_days:
        return None
    clips, images = describe_days(new_days, scan, camera)
    room = {day_key(camera, d): capacity([c for c in clips if c["day"] == day_key(camera, d)]) for d in new_days}
    add_tertiles(clips, images, kept["cuts"])
    touched = {k: reason for k, (_, reason) in FORCED_ROLES.items()} | TOUCHED_DAYS
    counts = round_role_counts(len(new_days))
    # A sampled camera has about one day a week, so the week rule would block every pick; the
    # stretches still spread expert and test days over the season.
    roles = assign_roles(list(room), room, seed, counts, touched, min([*kept["roles"], *room]), weekly=False)
    forced = {}
    for key, (role, reason) in FORCED_ROLES.items():
        if roles.get(key, "unused") != "unused" and roles[key] != role:
            forced[key] = {"assigned": roles[key], "forced": role, "reason": reason}
            roles[key] = role
    windows, achieved = choose_windows(clips, roles, seed)
    return {"camera": camera, "days": new_days, "clips": clips, "room": room, "roles": roles, "windows": windows,
            "achieved": achieved, "planned": counts | {"train": "the rest"}, "forced": forced}


def select_round(scan: Path, out: Path, keep: Path, seed: int = SEED, labelled: frozenset[str] = frozenset(),
                 cameras: tuple[str, ...] = (), only_new: bool = False) -> dict:
    """A later round over the selection in ``keep`` (see the module docstring). ``cameras``
    adds other cameras' new days (``plan_camera``); ``only_new`` lists only this round's new
    windows in ``to_label.csv``, so earlier rounds' labels are reused rather than redone."""
    if out.resolve() == keep.resolve():
        raise ValueError(f"{out} is the kept selection; write the new round elsewhere")
    kept = read_kept(keep)
    n = kept["round"] + 1
    days = ledger_days(scan)
    if lost := sorted(d for d in kept["roles"] if split_key(d)[0] == CAMERA and d not in days):
        raise RuntimeError(f"kept days the ledger doesn't call completed: {lost[:5]}; is --scan right?")
    new_days = [d for d in days if d not in kept["roles"]]
    clips, images = describe_days(new_days, scan)
    room = {day: capacity([c for c in clips if c["day"] == day]) for day in new_days}
    cuts = add_tertiles(clips, images, kept["cuts"])
    new_roles, windows, achieved = plan_round(days, clips, room, kept["roles"], seed)
    cam1_roles = dict(new_roles)
    others = [p for p in (plan_camera(c, scan, kept, seed) for c in cameras) if p]
    for plan in others:
        new_roles |= plan["roles"]
        windows += plan["windows"]
        clips += plan["clips"]
        room |= plan["room"]
    roles = dict(sorted((kept["roles"] | new_roles).items()))
    day_round = dict(sorted((kept["day_round"] | dict.fromkeys(new_roles, n)).items()))
    new_rows = [manifest_row(w) | {"round": str(n)} for w in windows]
    rows = kept["rows"] + new_rows
    if len({r["window_id"] for r in rows}) != len(rows):
        raise RuntimeError("a new window repeats a kept window's ID")
    if unknown := sorted(labelled - {r["window_id"] for r in rows}):
        raise ValueError(f"--labelled names windows in no round: {unknown[:5]}")
    to_label = [r for r in (new_rows if only_new else rows) if r["window_id"] not in labelled]
    fields = ["window_id", "round", *MANIFEST_FIELDS[1:]]
    planned = round_role_counts(len(new_days)) | {"train": "the rest"}
    kept_files = {"dir": keep.as_posix(), "round": kept["round"], "sha256": kept["sha256"]}

    out.mkdir(parents=True, exist_ok=True)
    (out / "day_roles.json").write_text(json.dumps({
        "round": n, "seed": seed, "counts": dict(Counter(roles.values())),
        "counts_by_round": {str(k): dict(Counter(roles[d] for d, r in day_round.items() if r == k))
                            for k in sorted(set(day_round.values()))},
        "planned_this_round": planned, "shares_percent": ROLE_PERCENT,
        "rule": f"a day keeps the role its round gave it; each round's new days: {ROLE_RULE} (weeks counted from "
                f"{min(roles)}); new windows come only from new training and test days",
        "kept": kept_files, "never_expert": TOUCHED_DAYS, "unused": UNUSED_RULE,
        "capacity": dict(sorted((kept["capacity"] | room).items())), "meaning": ROLE_MEANING,
        "day_round": day_round, "roles": roles, "by_week": by_week(roles)}, indent=2), encoding="utf-8")
    write_csv(out / "manifest.csv", rows, fields)
    write_csv(out / "to_label.csv", to_label, fields)
    contact_sheets(windows, scan, out)
    eligible = [c for c in clips if c["evidence"] and new_roles[c["day"]] in WINDOWS_PER_DAY]
    per_day = Counter(w["day"] for w in windows)
    short_days = {d: {"role": r, "windows": per_day[d], "slots": WINDOWS_PER_DAY[r],
                      "eligible_clips": dict(Counter(c["evidence"] for c in eligible if c["day"] == d))}
                  for d, r in new_roles.items() if r in WINDOWS_PER_DAY and per_day[d] < WINDOWS_PER_DAY[r]}
    summary = {
        "round": n,"seed": seed, "scan": scan.as_posix(),
        "ledger": {"completed_days": len(days), "first": days[0], "last": days[-1]},
        "kept": kept_files | {"days": len(kept["roles"]), "windows": len(kept["rows"])},
        "new_days": {"count": len(new_days), "first": new_days[0], "last": new_days[-1], "planned": planned,
                     "roles": dict(Counter(cam1_roles.values()))},
        "window_seconds": WINDOW_SECONDS, "window_frames": WINDOW_FRAMES,
        "min_clip_gap_minutes": MIN_CLIP_GAP.total_seconds() / 60,
        "tertile_cuts": {"light": cuts["light"], "clarity": cuts["clarity"],
                         "kept_from": (keep / "summary.json").as_posix(), "eligible_pngs_new_days": cuts["eligible_pngs"]},
        "this_round": {
            "clips": len(clips), "excluded": dict(Counter(c["excluded"] for c in clips if not c["evidence"])),
            "eligible_by_evidence_new_days": dict(Counter(c["evidence"] for c in clips if c["evidence"])),
            "eligible_by_evidence_train_test_days": dict(Counter(c["evidence"] for c in eligible)),
            "slots": sum(WINDOWS_PER_DAY[r] for r in new_roles.values() if r in WINDOWS_PER_DAY),
            "short_days": short_days, "quotas": achieved, **tally(new_rows)},
        "cumulative": {"days_by_role": dict(Counter(roles.values())), **tally(rows)},
        "other_cameras": {p["camera"]: {
            "new_days": {"count": len(p["days"]), "first": p["days"][0], "last": p["days"][-1],
                         "planned": p["planned"], "roles": dict(Counter(p["roles"].values()))},
            "forced_roles": p["forced"], "quotas": p["achieved"],
            "role_rule": "as Cam 1's, without the week rule: a sampled camera has about one day a week",
            **tally([r for r in new_rows if split_key(r["day"])[0] == p["camera"]])} for p in others},
        "to_label": {"windows": len(to_label), "already_labelled": len(rows) - len(to_label), "only_new": only_new,
                     "by_round": dict(sorted(Counter(r["round"] for r in to_label).items())),
                     "by_camera": dict(sorted(Counter(split_key(r["day"])[0] for r in to_label).items()))},
        "estimate": round_estimate(to_label), "attribution": ATTRIBUTION,
        "interpretation": "Unreviewed screening sorts the clips; it is never a label."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scan", type=Path, default=SCAN)
    parser.add_argument("--root", type=Path, help="Run folder (default data/outputs/sam3-cloud-labels-<UTC date>, "
                                                 "or data/outputs/smartbay-rounds/round-<n> with --keep)")
    parser.add_argument("--keep", type=Path, help="An earlier selection folder: keep its day roles and windows, and "
                                                 "choose windows from new days only")
    parser.add_argument("--labelled", type=Path, help="With --keep: a file of window IDs already labelled, one per "
                                                     "line; they are left out of to_label.csv")
    others = [c for c in CAMERAS if c != CAMERA]
    parser.add_argument("--add-camera", action="append", default=[], choices=others,
                        help="With --keep: also choose windows from this camera's new scanned days (repeatable)")
    parser.add_argument("--only-new", action="store_true",
                        help="With --keep: list only this round's new windows in to_label.csv")
    parser.add_argument("--pick-scan-days", type=int, metavar="N",
                        help="Instead of selecting: write N more days of --camera to scan, to --days-file")
    parser.add_argument("--camera", choices=others, default=others[0], help="With --pick-scan-days")
    parser.add_argument("--days-file", type=Path, help="With --pick-scan-days (default <round 2>/scan/<camera>-days.txt)")
    args = parser.parse_args()
    if args.pick_scan_days:
        days = pick_scan_days(args.camera, args.pick_scan_days, args.scan, args.seed)
        path = args.days_file or OUTPUTS / "smartbay-rounds" / "round-2" / "scan" / f"{args.camera}-days.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{d}\n" for d in days), encoding="utf-8", newline="\n")
        print(json.dumps({"camera": args.camera, "days": days, "count": len(days), "file": path.as_posix()}, indent=2))
        return
    if (args.labelled or args.add_camera or args.only_new) and not args.keep:
        parser.error("--labelled, --add-camera and --only-new need --keep")
    if args.keep:
        root = args.root or OUTPUTS / "smartbay-rounds" / f"round-{kept_round(args.keep) + 1}"
        labelled = frozenset(args.labelled.read_text(encoding="utf-8").split()) if args.labelled else frozenset()
        summary = select_round(args.scan, root / "selection", args.keep, args.seed, labelled,
                               tuple(args.add_camera), args.only_new)
    else:
        summary = select(args.scan, (args.root or OUTPUTS / f"sam3-cloud-labels-{utc_today()}") / "selection", args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
