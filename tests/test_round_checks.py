"""Part 5 automatic checks: no network, GPU or AWS."""
import csv
import json

from scripts.cloud_labelling import round_checks as rc

COMPLETE = {name: "complete" for name in rc.STAGE_ENTRIES}


def row(window_id, evidence="strong", role="train", reserve="", day="2025-07-01"):
    return {"window_id": window_id, "round": "1", "day": day, "role": role, "evidence": evidence, "light": "mid",
            "hour_bin": "06-12", "source_mp4_url": f"https://example.test/{window_id}.mp4", "start_frame": "100",
            "frame_count": "450", "reserve_window_id": reserve, "reserve_url": f"https://example.test/{reserve}.mp4",
            "reserve_start_frame": "200"}


def test_each_exclusion_rule_and_the_keep_but_flag_rule():
    ok = {"stages": COMPLETE, "label_boxes": 30, "max_boxes_in_a_frame": 3, "static_removed_share": .1}
    assert rc.judge(row("a"), ok) == ({}, [])
    reasons, _ = rc.judge(row("a"), ok | {"stages": COMPLETE | {"labels/fish": "failed"}})
    assert list(reasons) == ["stage_failed"] and "labels/fish failed" in reasons["stage_failed"]
    assert list(rc.judge(row("a"), ok | {"max_boxes_in_a_frame": 151})[0]) == ["max_label_boxes_in_a_frame"]
    assert rc.judge(row("a"), ok | {"max_boxes_in_a_frame": 150})[0] == {}
    assert list(rc.judge(row("a"), ok | {"static_removed_share": .51})[0]) == ["max_static_removed_share"]
    assert list(rc.judge(row("a"), ok | {"label_boxes": 0})[0]) == ["strong_needs_labels"]
    missing = {k: v for k, v in ok.items() if k != "label_boxes"}
    assert "output file is missing" in rc.judge(row("a"), missing)[0]["stage_failed"]
    assert rc.judge(row("a", "weak"), ok | {"label_boxes": 0}) == ({}, [])
    reasons, flags = rc.judge(row("a", "murky"), ok)
    assert reasons == {} and "murky window with 30 label boxes" in flags[0]


def test_a_labelled_round_is_measured_with_reserves_and_the_rules_recorded_first(tmp_path):
    root = tmp_path / "round-1"
    rows = [row("w1"), row("w2", reserve="w2r"), row("w3", "none", role="test"), row("w4")]
    entries = {}
    for key in ("w1", "w2r", "w3"):
        entries |= {f"{key}/{name}": {"status": "complete"} for name in rc.STAGE_ENTRIES}
        labels = root / "videos" / key / "labels" / "fish"
        labels.mkdir(parents=True)
        boxes = [] if key == "w3" else [{"id": f"t{i % 2}", "frame": i // 2, "box": [0, 0, 20, 20 + i],
                                         "confidence": .5 + i / 100} for i in range(10)]
        (labels / "rows.jsonl").write_text("".join(json.dumps(b) + "\n" for b in boxes))
        static = root / "videos" / key / "static" / "sam3-fish"
        static.mkdir(parents=True)
        (static / "report.json").write_text(json.dumps({"counts": {"boxes": 20, "removed_boxes": 15 if key == "w2r" else 2}}))
    (root / "windows").mkdir()
    (root / "windows" / "w2r.json").write_text("{}")
    (root / "journal.json").write_text(json.dumps({"schema_version": 1, "created_utc": "x", "budgets_seconds": {},
                                                   "entries": entries, "decisions": {}}))
    (root / "selection").mkdir()
    with (root / "selection" / "to_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = rc.check(root, root / "selection" / "to_label.csv", sheets=False)
    by_id = {w["window_id"]: w for w in report["per_window"]}
    assert by_id["w1"]["excluded"] is False and by_id["w1"]["label_boxes"] == 10 and by_id["w1"]["tracks"] == 2
    assert by_id["w1"]["max_boxes_in_a_frame"] == 2
    assert by_id["w1"]["median_box_side_px"] == 22.1  # median of sqrt(20 x (20..29)): (21.91 + 22.36) / 2
    assert by_id["w2"]["labelled_as"] == "w2r" and list(by_id["w2"]["exclusion_reasons"]) == ["max_static_removed_share"]
    assert by_id["w3"]["excluded"] is False and by_id["w3"]["flags"] == []  # a none window with no labels is fine
    assert by_id["w4"]["labelled_as"] is None and list(by_id["w4"]["exclusion_reasons"]) == ["stage_failed"]
    assert report["labelled"] == 3 and report["labelled_share"] == .75 and report["kept"] == 2
    journal = json.loads((root / "journal.json").read_text())
    assert journal["decisions"]["auto_exclusions"]["rules"] == rc.RULES
    assert (root / "checks" / "quality_report.json").exists() and (root / "checks" / "windows.csv").exists()
