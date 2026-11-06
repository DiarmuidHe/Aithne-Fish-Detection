"""Parts 6 and 7 for a labelling round: no network, GPU, checkpoint or rfdetr import."""
from datetime import date, timedelta

from scripts import student_detector as sd

DAYS = [(date(2025, 6, 19) + timedelta(days=i)).isoformat() for i in range(105)]


def window(n, day, role, evidence="strong", excluded=False, start=100, frames=450, round_="0"):
    return {"window_id": f"SmartBayCam1_{day}_{n:04d}_f{start:04d}", "day": day, "role": role, "evidence": evidence,
            "round": round_, "excluded": excluded, "start_frame": start, "frames": frames, "width": 1920,
            "height": 1080}


def label(frame, box, track="sam3-1"):
    return {"id": track, "frame": frame, "box": box, "confidence": .9}


def test_validation_is_a_seeded_tenth_of_training_days_spread_and_frozen_for_later_rounds():
    first = sd.choose_validation(DAYS[:62])
    assert len(first["days"]) == 6 and set(first["days"]) <= set(DAYS[:62])
    picks = sorted(date.fromisoformat(d) for d in first["days"])
    assert max((b - a).days for a, b in zip(picks, picks[1:])) <= 21  # at least one per three weeks
    assert sd.choose_validation(DAYS[:62]) == first  # seeded
    later = sd.choose_validation(DAYS[:92], frozen=first)
    assert set(first["days"]) < set(later["days"])  # earlier validation days stay validation
    assert len(later["days"]) == 6 + 3 and all(d in DAYS[62:92] for d in set(later["days"]) - set(first["days"]))
    assert later["considered"] == sorted(DAYS[:92])
    assert sd.choose_validation(DAYS[:3])["days"] == []  # too few new days for a validation pick


def test_day_roles_never_mix_across_splits_and_excluded_or_reserved_windows_never_appear():
    windows = [window(1, DAYS[0], "train"), window(2, DAYS[1], "train"), window(3, DAYS[2], "test"),
               window(4, DAYS[3], "train", excluded=True), window(5, DAYS[4], "expert"), window(6, DAYS[5], "unused")]
    coco = sd.round_coco(windows, {}, validation={DAYS[1]})
    days = {name: {i["day"] for i in data["images"]} for name, data in coco.items()}
    assert days == {"train": {DAYS[0]}, "validation": {DAYS[1]}, "test": {DAYS[2]}}
    seen = {i["window_id"] for data in coco.values() for i in data["images"]}
    assert not seen & {windows[3]["window_id"], windows[4]["window_id"], windows[5]["window_id"]}


def test_every_fifth_frame_is_an_image_mapped_to_its_source_frame_and_negatives_stay():
    w = window(1, DAYS[0], "train", start=946)
    labels = {w["window_id"]: [label(0, [10, 10, 50, 40]), label(5, [-5, 1070, 30, 1090]), label(5, [100, 100, 101, 140]),
                               label(7, [10, 10, 50, 40])]}
    coco = sd.round_coco([w], labels, validation=set())["train"]
    assert [i["frame"] for i in coco["images"]] == list(range(0, 450, 5))
    first, second = coco["images"][:2]
    assert (first["source_frame"], second["source_frame"]) == (946, 951)
    assert second["file_name"] == f"{w['window_id']}_005.jpg" and first["evidence"] == "strong"
    boxes = {a["image_id"]: a["bbox"] for a in coco["annotations"]}
    assert len(coco["annotations"]) == 2  # frame 7 is not sampled; the 1 px sliver is dropped
    assert boxes[second["id"]] == [0.0, 1070, 30.0, 10.0]  # clipped to the frame
    counts = sd.coco_counts({"train": coco})["train"]
    assert counts["images"] == 90 and counts["negative_images"] == 88 and counts["by_stratum"]["strong"]["boxes"] == 2


def test_a_later_rounds_dataset_holds_the_earlier_one_with_the_same_splits():
    round_1 = [window(n, DAYS[n], "test" if n % 7 == 0 else "train") for n in range(20)]
    round_2 = round_1 + [window(n, DAYS[n], "train", round_="2") for n in range(20, 30)]
    validation_1 = sd.choose_validation([w["day"] for w in round_1 if w["role"] == "train"])
    validation_2 = sd.choose_validation([w["day"] for w in round_2 if w["role"] == "train"], frozen=validation_1)
    before = sd.round_coco(round_1, {}, set(validation_1["days"]))
    after = sd.round_coco(round_2, {}, set(validation_2["days"]))
    split_of = lambda coco: {i["file_name"]: name for name, data in coco.items() for i in data["images"]}  # noqa: E731
    old, new = split_of(before), split_of(after)
    assert all(new[name] == split for name, split in old.items())
    assert len(new) == len(old) + 10 * 90


def test_agreement_is_pooled_over_windows():
    pooled = sd.pool([{"left_boxes": 10, "right_boxes": 8, "matched": 6}, {"left_boxes": 0, "right_boxes": 2, "matched": 0}])
    assert pooled == {"windows": 2, "student_boxes": 10, "sam3_label_boxes": 10, "matched": 6, "agreement": 0.4286,
                      "share_of_student_matched": 0.6, "share_of_sam3_matched": 0.6}
    assert sd.pool([])["agreement"] == 0


def test_round_epochs_allow_a_long_epoch_but_refuse_one_that_cannot_repeat():
    assert sd.size_epochs(1500, 10800 - 1500, bounds=sd.ROUND_EPOCHS) == 5
    assert sd.size_epochs(4000, 10800 - 4000, bounds=sd.ROUND_EPOCHS) == 1
    try:
        sd.size_epochs(6000, 10800 - 6000, bounds=sd.ROUND_EPOCHS)
    except sd.BudgetExceeded:
        pass
    else:
        raise AssertionError("an epoch that cannot run again within the budget must be refused")
