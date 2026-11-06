from datetime import datetime, timedelta

from scripts import smartbay_finalize_camera_selection as finalize


def clip(url: str, day: str, hour: int, evidence: str = "strong") -> dict:
    stamp = datetime.fromisoformat(f"{day}T{hour:02d}:00:00")
    return {
        "camera": "SmartBayCam1",
        "day": day,
        "url": url,
        "stem": url.rsplit("/", 1)[-1].removesuffix(".mp4"),
        "clip_time": stamp.isoformat(),
        "rank": "likely_fish",
        "rank_reason": "fixture",
        "evidence": evidence,
        "excluded": "",
        "nearby": [
            {
                "timestamp": (stamp + timedelta(seconds=50)).isoformat(),
                "max_score": 0.9,
                "detection_count": 3,
            }
        ],
        "light_mean": 100,
        "contrast": 30,
        "light": "mid",
        "clarity": "mid",
        "hour_bin": "00-06" if hour < 6 else "06-12",
        "density": "1-5",
        "png_detections": 3,
        "png_max_score": 0.9,
    }


def manifest_row(window_id: str, url: str, day: str, split: str = "train") -> dict[str, str]:
    return {
        "window_id": window_id,
        "camera": "SmartBayCam1",
        "day": day,
        "split": split,
        "labelling_route": "sam3_then_human_review",
        "source_mp4_url": url,
        "clip_time": f"{day}T00:00:00",
        "start_frame": "0",
        "frame_count": "450",
        "fps": "14.985015",
        "evidence": "strong",
        "light": "mid",
        "light_mean": "100",
        "clarity": "mid",
        "contrast": "30",
        "hour_bin": "00-06",
        "density": "1-5",
        "screening_rank": "likely_fish",
        "rank_reason": "fixture",
        "png_max_score": "0.9",
        "png_detections": "3",
        "nearby_pngs": "1",
        "attribution": "fixture",
    }


def test_choose_replacements_preserves_split_and_evidence():
    day = "2025-01-01"
    original_url = "https://example.test/original.mp4"
    approved_url = "https://example.test/approved.mp4"
    replacement_url = "https://example.test/replacement.mp4"
    clips = [
        clip(original_url, day, 0),
        clip(approved_url, day, 2),
        clip(replacement_url, day, 4),
    ]
    manifest = [
        manifest_row("rejected", original_url, day),
        manifest_row("approved", approved_url, day),
    ]
    manifest[1]["clip_time"] = f"{day}T02:00:00"
    review = [
        {
            "window_id": "rejected",
            "decision": "reject",
            "replacement_reason": "bad",
            "review_notes": "bad",
        },
        {
            "window_id": "approved",
            "decision": "approve",
            "replacement_reason": "",
            "review_notes": "good",
        },
    ]
    roles = {"SmartBayCam1": {day: "train"}}

    windows, replacements = finalize.choose_replacements(clips, manifest, review, roles, 17)

    assert len(windows) == 2
    assert {window["url"] for window in windows} == {approved_url, replacement_url}
    assert replacements[0]["rejected_window_id"] == "rejected"
    assert replacements[0]["replacement_evidence"] == "strong"
    assert replacements[0]["evidence_preserved"]
    assert all(window["split"] == "train" for window in windows)


def test_verify_rejects_day_split_leak():
    rows = [
        manifest_row("one", "https://example.test/one.mp4", "2025-01-01", "train"),
        manifest_row("two", "https://example.test/two.mp4", "2025-01-01", "test"),
    ]
    result = finalize.verify(rows, rows, set())
    assert not result["valid"]
    assert result["day_split_leaks"] == ["SmartBayCam1:2025-01-01"]
