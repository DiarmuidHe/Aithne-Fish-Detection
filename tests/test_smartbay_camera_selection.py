from datetime import date, datetime, timedelta

from scripts import smartbay_camera_selection as selection


def test_day_roles_are_disjoint_and_have_capacity_for_each_split_budget():
    days = [(date(2025, 1, 1) + timedelta(days=index)).isoformat() for index in range(40)]
    budget = selection.largest_remainder(120, selection.ROLE_SHARES)
    roles = selection.assign_day_roles(days, budget, 17, never_expert={days[3]})
    assert set(roles) == set(days)
    assert roles[days[3]] != "expert"
    for role, count in budget.items():
        assert sum(value == role for value in roles.values()) * selection.MAX_WINDOWS_PER_DAY >= count


def test_window_selection_is_fixed_budget_day_disjoint_and_deterministic():
    days = [(date(2025, 1, 1) + timedelta(days=index)).isoformat() for index in range(16)]
    role_budget = selection.largest_remainder(40, selection.ROLE_SHARES)
    roles = selection.assign_day_roles(days, role_budget, 17)
    clips = []
    evidence = tuple(selection.EVIDENCE_SHARES)
    for day_index, day in enumerate(days):
        for clip_index in range(16):
            stamp = datetime.fromisoformat(f"{day}T00:00:00") + timedelta(minutes=120 * clip_index)
            kind = evidence[clip_index % len(evidence)]
            clips.append({
                "camera": "SmartBayCam2", "day": day,
                "url": f"https://example.test/{day}/{clip_index}.mp4",
                "stem": f"SmartBayCam2_{day}_{stamp:%H%M}", "clip_time": stamp.isoformat(),
                "rank": "likely_fish", "rank_reason": "fixture", "evidence": kind,
                "excluded": "", "nearby": [{"timestamp": stamp.isoformat(), "thumb_index": None}],
                "light_mean": 100, "contrast": 30, "light": selection.TERTILES[clip_index % 3],
                "clarity": selection.TERTILES[(clip_index // 3) % 3],
                "hour_bin": selection.HOUR_BINS[stamp.hour // 6], "density": "1-5",
                "png_detections": 1, "png_max_score": .8,
            })
    first, report = selection.choose_windows(clips, roles, "SmartBayCam2", 40, 17)
    again, _ = selection.choose_windows(clips, roles, "SmartBayCam2", 40, 17)
    assert report["achieved"] == 40
    assert [(item["window_id"], item["split"]) for item in first] == [
        (item["window_id"], item["split"]) for item in again
    ]
    assert len({item["url"] for item in first}) == 40
    assert all(roles[item["day"]] == item["split"] for item in first)
    assert all(len({item["split"] for item in first if item["day"] == day}) <= 1 for day in days)
