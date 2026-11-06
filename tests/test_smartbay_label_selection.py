import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from scripts import smartbay_label_selection as sel

DAYS = [(date(2025, 6, 19) + timedelta(days=i)).isoformat() for i in range(60)]
STRATA = ("strong", "weak", "none", "murky")
SCAN = Path("data/archive_scan")  # never read: stub_scan stands in for it
# What the 2026-10-21 code (before --keep) wrote for stub_scan(DAYS[:30], empty={DAYS[12]}).
GOLDEN = {"day_roles.json": "03d4aadd0a735b6e1ab18649bc4ddd13ba699f305ad48f7a116215baea98e37e",
          "manifest.csv": "be4ab8482a72b3026c143af059bb5fca408e9896ce684460f0c102f9ecb97299",
          "summary.json": "031be48a2a0201769f3d1717d8d450601137377aa9607a2b52fc6491d996fcd3"}


def png(stamp, quality="usable", score=0.0, detections=0, mean=120.0, contrast=30.0):
    return {"url": f"https://example.test/{stamp}.png", "kind": "png", "timestamp": stamp, "quality": quality,
            "max_score": score, "detection_count": detections,
            "metrics": {"mean": mean, "contrast": contrast, "dark_fraction": 0.0, "sharpness": 10.0}}


def clip(day, minute, evidence, light="mid", clarity="mid"):
    stamp = (datetime.fromisoformat(f"{day}T00:00:00") + timedelta(minutes=minute)).isoformat()
    stem = f"SmartBayCam1_{day}_{stamp[11:13]}{stamp[14:16]}"
    return {"day": day, "url": f"https://example.test/{stem}.mp4", "stem": stem, "clip_time": stamp,
            "evidence": evidence, "excluded": "", "light": light, "clarity": clarity,
            "hour_bin": sel.HOUR_BINS[minute // 360], "density": "6-15" if evidence == "strong" else "",
            "nearby": [], "rank": "", "rank_reason": "", "light_mean": 0, "contrast": 0, "png_max_score": 0,
            "png_detections": 0}


def full_day(day):
    """A clip every 2 minutes all day; strata, light and clarity cycle through them."""
    return [clip(day, m, STRATA[(m // 2) % 4], sel.TERTILES[(m // 8) % 3], sel.TERTILES[(m // 24) % 3])
            for m in range(0, 1440, 2)]


def roles_for(days, seed=sel.SEED):
    return sel.assign_roles(days, {d: 40 for d in days}, seed, counts={"expert": 2, "test": 3})


def stub_scan(monkeypatch, days, empty=()):
    """``days`` as a completed scan without files: a clip every 6 minutes (strata, light and
    clarity cycling), and PNGs whose tertile cuts give the clips back their light and
    clarity; ``empty`` days have PNGs but no clips, so they can't be screened."""
    light, clarity = dict(zip(sel.TERTILES, (100, 120, 140))), dict(zip(sel.TERTILES, (20, 30, 40)))
    pngs = {f"p{m}-{k}": png(f"p{m}-{k}", mean=m, contrast=k) for m in light.values() for k in clarity.values()}

    def describe(day, state, rows):
        return [c | {"light_mean": light[c["light"]], "contrast": clarity[c["clarity"]]}
                for c in ([] if day in empty else full_day(day)[::3])]

    monkeypatch.setattr(sel, "completed_days", lambda scan=sel.SCAN: list(days))
    monkeypatch.setattr(sel, "ledger_days", lambda scan=sel.SCAN: list(days), raising=False)
    monkeypatch.setattr(sel, "load_day", lambda day, scan=sel.SCAN: ({"images": pngs}, []))
    monkeypatch.setattr(sel, "describe_day", describe)
    monkeypatch.setattr(sel, "contact_sheet", lambda windows, scan, path: None)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digests(folder):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("day_roles.json", "manifest.csv", "summary.json")}


@pytest.fixture
def round_0(monkeypatch, tmp_path):
    stub_scan(monkeypatch, DAYS[:30], empty={DAYS[12]})
    sel.select(SCAN, tmp_path / "r0")
    return tmp_path / "r0"


def plan_inputs(days=DAYS[:36], kept_days=24, empty=30):
    """Round 0's roles over the first ``kept_days`` days, and clips on every day but ``empty``."""
    clips = [c for d in days if d != days[empty] for c in full_day(d)[::3]]
    room = {d: sel.capacity([c for c in clips if c["day"] == d]) for d in days}
    return days, clips, room, roles_for(days[:kept_days])


def test_murky_is_kept_while_black_frozen_and_camera_off_clips_are_excluded():
    stamp = "2025-07-01T12:00:00"
    murky = sel.evidence_of("unusable", "all nearby PNGs fail quality screening",
                            [png(stamp, "obscured"), png(stamp, "obscured")])
    assert murky == ("murky", "")
    for bad in ("black", "frozen", "camera_off"):
        evidence, why = sel.evidence_of("unusable", "all nearby PNGs fail quality screening", [png(stamp, bad)])
        assert evidence is None and "black, camera-off or frozen" in why
    mixed, _ = sel.evidence_of("unusable", "all nearby PNGs fail quality screening",
                               [png(stamp, "black"), png(stamp, "obscured")])
    assert mixed is None  # not unusable *only* because of murk
    assert sel.evidence_of("uncertain", "no reliable nearby PNG coverage", []) == (None, "no PNG coverage")
    short, why = sel.evidence_of("likely_fish", "x", [png(stamp, score=.9)], frames=49)
    assert short is None and "shorter" in why
    assert sel.evidence_of("likely_fish", "x", [png(stamp, score=.9)])[0] == "strong"
    assert sel.evidence_of("uncertain", "nearby PNG has weaker SAM 3 evidence", [png(stamp, score=.3)])[0] == "weak"
    assert sel.evidence_of("candidate_clear_and_empty", "x", [png(stamp), png(stamp)])[0] == "none"


def test_describe_day_takes_light_clarity_and_density_from_the_paired_pngs():
    day = "2025-07-01"
    images = [png(f"{day}T12:00:00", score=.8, detections=7, mean=100, contrast=20),
              png(f"{day}T12:02:00", score=.6, detections=3, mean=140, contrast=40),
              png(f"{day}T15:00:00", "black", mean=5, contrast=1)]
    entries = [{**{k: p[k] for k in ("url", "kind", "timestamp")}} for p in images]
    entries.append({"url": "https://example.test/c.mp4", "kind": "mp4", "name": f"SmartBayCam1_{day}_1200.mp4",
                    "timestamp": f"{day}T12:00:00"})
    state = {"entries": entries, "images": {p["url"]: p for p in images}, "clips": {},
             "selected_png_urls": [p["url"] for p in images]}
    for p in state["images"].values():
        p["thumbnail"] = "thumbs/x.jpg"
    rows = [{"source_mp4_url": "https://example.test/c.mp4", "screening_rank": "likely_fish",
             "rank_reason": "nearby PNG has SAM 3 score >= 0.50", "scan_error": ""}]
    [described] = sel.describe_day(day, state, rows)
    assert described["evidence"] == "strong"
    assert described["light_mean"] == 120 and described["contrast"] == 30  # medians of the two paired PNGs
    assert described["density"] == "6-15" and described["hour_bin"] == "12-18"
    assert [p["thumb_index"] for p in described["nearby"]] == [0, 1]


def test_day_roles_never_mix_spread_over_weeks_and_respect_capacity():
    room = {d: 40 for d in DAYS}
    for d in DAYS[22:27] + [DAYS[31]]:
        room[d] = 0  # no PNGs
    room[DAYS[30]] = 2  # holds only a test day's windows
    roles = sel.assign_roles(DAYS, room, touched={DAYS[45]: "used"})
    counts = Counter(roles.values())
    assert counts == {"unused": 6, "expert": 6, "test": 9, "train": 39}
    assert set(roles) == set(DAYS)  # one role per day, so no day is in two roles
    assert roles[DAYS[45]] != "expert"
    assert roles[DAYS[30]] in {"test", "expert"}
    weeks = {}
    for day, role in roles.items():
        weeks.setdefault((date.fromisoformat(day) - date.fromisoformat(DAYS[0])).days // 7, []).append(role)
    assert all("train" in r for r in weeks.values() if set(r) != {"unused"})
    tests = sorted(date.fromisoformat(d) for d, r in roles.items() if r == "test")
    assert max((b - a).days for a, b in zip(tests, tests[1:])) <= 14  # spread, not bunched


def test_quotas_are_filled_and_windows_stay_on_their_days():
    days = DAYS[:12]
    roles = roles_for(days)
    clips = [c for d in days for c in full_day(d)]
    windows, achieved = sel.choose_windows(clips, roles)
    total = sum(sel.WINDOWS_PER_DAY[r] for r in roles.values() if r in sel.WINDOWS_PER_DAY)
    assert len(windows) == total
    assert {e: v["achieved"] for e, v in achieved["evidence"].items()} == sel.quota_targets(total)
    assert all(v["achieved"] >= math.ceil(sel.LIGHT_MIN * total) for v in achieved["light"].values())
    assert all(v["achieved"] >= math.ceil(sel.HOUR_MIN * total) for v in achieved["hour"].values())
    assert not achieved["evidence_shortfalls"] and not achieved["hour_shortfalls"]
    assert {w["role"] for w in windows} <= {"train", "test"}
    assert all(roles[w["day"]] == w["role"] for w in windows)
    per_day = Counter(w["day"] for w in windows)
    assert all(per_day[d] == sel.WINDOWS_PER_DAY[r] for d, r in roles.items() if r in sel.WINDOWS_PER_DAY)
    urls = [w["url"] for w in windows] + [w["reserve"]["url"] for w in windows if "reserve" in w]
    assert len(urls) == len(set(urls))  # no clip twice, reserves included
    for day in per_day:
        starts = sorted(datetime.fromisoformat(w["clip_time"]) for w in windows if w["day"] == day)
        assert all(b - a >= sel.MIN_CLIP_GAP for a, b in zip(starts, starts[1:]))


def test_an_hour_bin_short_of_footage_is_reported_not_padded():
    days = DAYS[:12]
    roles = roles_for(days)
    clips = [c for d in days for c in full_day(d) if c["hour_bin"] != "18-24"]
    clips.append(clip(days[0], 19 * 60, "strong"))  # one evening clip in all
    windows, achieved = sel.choose_windows(clips, roles)
    assert sum(w["hour_bin"] == "18-24" for w in windows) == 1
    assert achieved["hour_shortfalls"] == {"18-24": math.ceil(sel.HOUR_MIN * len(windows)) - 1}


def test_slots_no_clip_can_fill_are_reported_once_not_padded():
    days = DAYS[:12]
    roles = roles_for(days)
    murky_day = next(d for d, r in roles.items() if r == "train")
    clips = [c for d in days if d != murky_day for c in full_day(d)]
    clips.append(next(c for c in full_day(murky_day) if c["evidence"] == "murky"))  # room for one window
    windows, achieved = sel.choose_windows(clips, roles)
    total = sum(sel.WINDOWS_PER_DAY[r] for r in roles.values() if r in sel.WINDOWS_PER_DAY)
    assert sum(w["day"] == murky_day for w in windows) <= 1
    assert achieved["unfilled_slots"] == total - len(windows) >= 2
    assert achieved["evidence_shortfalls"] == {e: v["target"] - v["achieved"] for e, v in achieved["evidence"].items()
                                               if v["achieved"] < v["target"]}
    full, achieved = sel.choose_windows([c for d in days for c in full_day(d)], roles)
    assert "unfilled_slots" not in achieved


def test_selection_is_deterministic_under_the_seed():
    days = DAYS[:12]
    clips = [c for d in days for c in full_day(d)]
    first = sel.choose_windows(clips, roles_for(days, 3), 3)[0]
    again = sel.choose_windows(clips, roles_for(days, 3), 3)[0]
    other = sel.choose_windows(clips, roles_for(days, 4), 4)[0]
    ids = lambda ws: [(w["window_id"], w.get("reserve", {}).get("window_id")) for w in ws]  # noqa: E731
    assert ids(first) == ids(again)
    assert ids(first) != ids(other)
    assert roles_for(days, 3) == roles_for(days, 3)


def test_windows_are_placed_inside_their_clip():
    assert sel.WINDOW_FRAMES == 450  # 30 s at 15000/1001 FPS
    rng = random.Random(0)
    starts = [sel.place(rng) for _ in range(2000)]
    assert min(starts) >= 0 and max(starts) + sel.WINDOW_FRAMES <= sel.CLIP_FRAMES
    assert max(starts) > sel.CLIP_FRAMES - sel.WINDOW_FRAMES - 50  # the whole range is used
    assert sel.place(random.Random(1), clip_frames=450) == 0
    with pytest.raises(ValueError):
        sel.place(rng, clip_frames=449)
    windows, _ = sel.choose_windows([c for d in DAYS[:12] for c in full_day(d)], roles_for(DAYS[:12]))
    assert all(w["window_id"].endswith(f"_f{w['start_frame']:04d}") for w in windows)
    assert all(0 <= w["start_frame"] <= sel.CLIP_FRAMES - w["frame_count"] for w in windows)


def test_thumbnails_are_cut_from_the_scan_contact_sheets(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    directory = tmp_path / "SmartBayCam1_2025-07-01" / "contact_sheets"
    directory.mkdir(parents=True)
    for page, name in enumerate(("png_pilot.jpg", "png_page_002.jpg")):
        sheet = np.zeros((2520, 1600, 3), np.uint8)
        for index in range(40):
            row, col = divmod(index, 4)
            sheet[row * 252:(row + 1) * 252, col * 400:(col + 1) * 400] = (page * 40 + index) * 3
        cv2.imwrite(str(directory / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 100])
    sel._sheet.cache_clear()
    tile = sel.thumbnail(tmp_path, "2025-07-01", 45)
    assert tile.shape == (252, 400, 3)
    assert abs(float(tile[20:-20, 20:-20].mean()) - 45 * 3) < 2
    assert sel.thumbnail(tmp_path, "2025-07-01", None) is None


def test_without_keep_the_output_matches_the_2026_10_21_code(round_0):
    assert digests(round_0) == GOLDEN


def test_round_role_counts_are_round_0s_shares_rounded_half_up():
    assert sel.round_role_counts(60) == sel.ROLE_DAYS  # round 0's 6 and 9 of 60
    assert sel.round_role_counts(41) == {"expert": 4, "test": 6}
    assert sel.round_role_counts(45) == {"expert": 5, "test": 7}  # 4.5 and 6.75
    assert sel.round_role_counts(7) == {"expert": 1, "test": 1}  # at least one of each from 7 days
    assert sel.round_role_counts(4) == {"expert": 0, "test": 1}


def test_later_rounds_keep_every_day_role_and_window(monkeypatch, tmp_path, round_0):
    stub_scan(monkeypatch, DAYS[:45], empty={DAYS[12], DAYS[33]})
    first = sel.select_round(SCAN, tmp_path / "r1", round_0)
    stub_scan(monkeypatch, DAYS, empty={DAYS[12], DAYS[33]})
    second = sel.select_round(SCAN, tmp_path / "r2", tmp_path / "r1")
    roles = {r: json.loads((tmp_path / r / "day_roles.json").read_text(encoding="utf-8")) for r in ("r0", "r1", "r2")}
    assert roles["r1"]["roles"].items() >= roles["r0"]["roles"].items()
    assert roles["r2"]["roles"].items() >= roles["r1"]["roles"].items()
    assert list(roles["r2"]["roles"]) == DAYS
    assert roles["r2"]["day_round"] == {d: 0 if d in DAYS[:30] else 1 if d in DAYS[:45] else 2 for d in DAYS}
    new = {d: r for d, r in roles["r1"]["roles"].items() if d not in roles["r0"]["roles"]}
    assert new[DAYS[33]] == "unused"
    assert {r: Counter(new.values())[r] for r in sel.ROLE_PERCENT} == sel.round_role_counts(15)

    r0, r1, r2 = (read_csv(tmp_path / r / "manifest.csv") for r in ("r0", "r1", "r2"))
    assert r1[:len(r0)] == [row | {"round": "0"} for row in r0]  # every kept value, as written
    assert r2[:len(r1)] == r1
    assert {r["round"] for r in r1[len(r0):]} == {"1"} and {r["round"] for r in r2[len(r1):]} == {"2"}
    assert list(r2[0])[:2] == ["window_id", "round"]
    assert read_csv(tmp_path / "r2" / "to_label.csv") == r2  # nothing labelled yet
    assert second["this_round"]["windows"] == len(r2) - len(r1) and second["cumulative"]["windows"] == len(r2)
    assert first["kept"]["sha256"] == digests(round_0)
    r0_summary = json.loads((round_0 / "summary.json").read_text(encoding="utf-8"))
    assert second["tertile_cuts"]["light"] == r0_summary["tertile_cuts"]["light"]


def test_new_windows_never_land_on_kept_expert_or_unused_days():
    days, clips, room, kept = plan_inputs()
    roles, windows, achieved = sel.plan_round(days, clips, room, kept, touched={days[26]: "used"})
    assert list(roles) == days[24:]
    assert roles[days[30]] == "unused" and roles[days[26]] != "expert"
    assert {r: Counter(roles.values())[r] for r in sel.ROLE_PERCENT} == sel.round_role_counts(12)
    assert {w["day"] for w in windows} <= {d for d, r in roles.items() if r in sel.WINDOWS_PER_DAY}
    assert all(w["role"] == roles[w["day"]] for w in windows)
    per_day = Counter(w["day"] for w in windows)
    assert all(per_day[d] == sel.WINDOWS_PER_DAY[r] for d, r in roles.items() if r in sel.WINDOWS_PER_DAY)
    assert {e: v["achieved"] for e, v in achieved["evidence"].items()} == sel.quota_targets(len(windows))
    reserves = {w["reserve"]["url"] for w in windows if "reserve" in w}
    assert all(c["day"] in roles for c in clips if c["url"] in reserves)


def test_round_selection_is_deterministic_under_the_seed():
    days, clips, room, kept = plan_inputs()

    def ids(seed):
        roles, windows, _ = sel.plan_round(days, clips, room, kept, seed)
        return roles, [(w["window_id"], w.get("reserve", {}).get("window_id")) for w in windows]

    assert ids(3) == ids(3)
    assert ids(3) != ids(4)


def test_labelled_windows_leave_to_label_and_the_kept_round_is_checked(monkeypatch, tmp_path, round_0):
    stub_scan(monkeypatch, DAYS[:45], empty={DAYS[12]})
    done = frozenset(r["window_id"] for r in read_csv(round_0 / "manifest.csv")[:3])
    summary = sel.select_round(SCAN, tmp_path / "r1", round_0, labelled=done)
    to_label = read_csv(tmp_path / "r1" / "to_label.csv")
    assert not done & {r["window_id"] for r in to_label}
    assert len(to_label) == len(read_csv(tmp_path / "r1" / "manifest.csv")) - 3 == summary["to_label"]["windows"]
    with pytest.raises(ValueError, match="in no round"):
        sel.select_round(SCAN, tmp_path / "r1b", round_0, labelled=frozenset({"SmartBayCam1_2099-01-01_0000_f0000"}))
    with pytest.raises(ValueError, match="kept selection"):
        sel.select_round(SCAN, round_0, round_0)
    stub_scan(monkeypatch, DAYS[1:45])  # the ledger lost a kept day: wrong --scan
    with pytest.raises(RuntimeError, match="ledger"):
        sel.select_round(SCAN, tmp_path / "r1c", round_0)
