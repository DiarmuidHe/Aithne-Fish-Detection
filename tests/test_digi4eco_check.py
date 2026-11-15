import stat
import zipfile

import pytest

from scripts import digi4eco_check as check

CVAT = """<?xml version='1.0' encoding='utf-8'?>
<annotations>
  <version>1.1</version>
  <meta>
    <job><size>3</size><start_frame>10</start_frame><stop_frame>14</stop_frame>
      <frame_filter>step=2</frame_filter></job>
    <original_size><width>100</width><height>50</height></original_size>
    <sequence_info><video_filename>SmartBayCam1_2024-07-16_0054.mp4</video_filename></sequence_info>
  </meta>
  <track id="0" label="Fish">
    <box frame="10" xtl="1" ytl="2" xbr="11" ybr="12" outside="0" occluded="0" keyframe="1"/>
    <box frame="12" xtl="2" ytl="2" xbr="12" ybr="12" outside="0" occluded="1" keyframe="0"/>
    <box frame="14" xtl="3" ytl="2" xbr="13" ybr="12" outside="1" occluded="0" keyframe="1"/>
  </track>
  <track id="1" label="Fish">
    <box frame="14" xtl="50" ytl="20" xbr="60" ybr="30" outside="0" occluded="0" keyframe="1"/>
  </track>
</annotations>"""


def test_parse_cvat_keeps_visible_boxes_with_flags_and_maps_frames(tmp_path):
    path = tmp_path / "annotations.xml"
    path.write_text(CVAT, encoding="utf-8")
    meta, rows = check.parse_cvat(path)
    assert meta["tracks"] == 2 and meta["start_frame"] == 10 and meta["frame_step"] == 2
    assert meta["original_size"] == [100, 50]
    assert meta["sequence_info"]["video_filename"] == "SmartBayCam1_2024-07-16_0054.mp4"
    assert [(r["frame"], r["track"], r["occluded"], r["keyframe"]) for r in rows] == [
        (0, 0, False, True), (1, 0, True, False), (2, 1, False, True)]  # The outside box is dropped
    assert rows[0]["box"] == [1.0, 2.0, 11.0, 12.0]


def test_unsafe_members_refuses_absolute_parent_and_link_entries(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ok/video.mp4", b"x")
        zipped.writestr("../escape.txt", b"x")
        zipped.writestr("/absolute.txt", b"x")
        link = zipfile.ZipInfo("ok/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zipped.writestr(link, "target")
    assert sorted(check.unsafe_members(archive)) == ["../escape.txt", "/absolute.txt", "ok/link"]


def person(frame, box, track=0, occluded=False, keyframe=True):
    return {"frame": frame, "track": track, "box": box, "occluded": occluded, "keyframe": keyframe}


def model(frame, box, confidence=.9):
    return {"id": None, "frame": frame, "box": box, "confidence": confidence}


def test_score_model_precision_recall_and_breakdowns():
    people = [person(0, [100, 100, 140, 120], track=1), person(0, [300, 300, 310, 320], track=2, occluded=True),
              person(1, [100, 100, 140, 120], track=1)]
    predicted = [model(0, [101, 100, 141, 120]), model(0, [500, 500, 540, 520]), model(1, [100, 101, 140, 121])]
    scored, _ = check.score_model(people, predicted, frames=2, width=1920, height=1080)
    strict = scored["iou_0.5"]
    assert (strict["matched"], strict["predicted"], strict["people"]) == (2, 3, 3)
    assert strict["precision"] == pytest.approx(2 / 3) and strict["recall"] == pytest.approx(2 / 3)
    assert scored["recall_by"]["occluded"]["occluded"]["recall"] == 0
    assert scored["recall_by"]["size"]["0-16"]["people"] == 1  # Short side 10 px
    assert scored["tracks"] == {"people_tracks": 2, "found_at_least_once": 1, "found_in_half_their_frames": 1}
    assert scored["frame_shift_check"]["0"] == 2


def test_average_precision_takes_detections_in_score_order():
    people = [person(0, [0, 0, 10, 10]), person(0, [20, 0, 30, 10])]
    perfect = [model(0, [0, 0, 10, 10], .9), model(0, [20, 0, 30, 10], .8)]
    assert check.average_precision(perfect, people)["ap50"] == pytest.approx(1.0)
    # A confident false box first: precision 0, 1/2, 2/3 at recall 0, .5, 1. The interpolated
    # envelope takes the best precision at or beyond each recall, so both steps count 2/3
    worse = [model(0, [50, 50, 60, 60], .95)] + perfect
    assert check.average_precision(worse, people)["ap50"] == pytest.approx(2 / 3)


def test_spread_sample_is_seeded_and_spread():
    rows = [model(f, [0, 0, 20 + (f % 3) * 30, 20 + (f % 3) * 30]) for f in range(300)]
    first = check.spread_sample(rows, 12, frames=300)
    assert first == check.spread_sample(rows, 12, frames=300) and len(first) == 12
    assert {min(2, 3 * r["frame"] // 300) for r in first} == {0, 1, 2}


def test_wilson_interval():
    low, high = check.wilson(50, 60)
    assert low < 50 / 60 < high and 0 <= low and high <= 1
    assert check.wilson(0, 0) is None
