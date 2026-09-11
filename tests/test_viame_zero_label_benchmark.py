from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.viame_zero_label_benchmark import (
    Journal, align, candidate_evidence, canonical, clean_env, compare_quality, iou, metrics, observations,
    run_config, save, stats,
)


def row(track="1", frame=0, t=0., box=(10, 10, 20, 20), confidence=.6):
    return {"id": track, "frame": frame, "t": t, "box": list(box), "confidence": confidence}


def test_metric_arithmetic_and_equal_track_weight():
    rows = [row(frame=i, t=i / 2, box=(10 + i, 10, 20 + i, 20)) for i in range(5)]
    rows += [row("2", t=5, confidence=.5)]
    m = metrics(rows, rows, 100, 100, 2)
    assert m["tracks"] == 2
    assert m["accepted_tracks"] == {"0.45": 2, "0.50": 2, "0.55": 1, "0.60": 1, "0.65": 0}
    assert m["length_fractions"] == {"1": .5, "2": 0, "3-4": 0, "5+": .5}
    assert m["temporal_coverage"]["5"] == pytest.approx(5 / 6)
    assert m["gap_seconds"]["median"] == .5
    assert m["center_velocity"]["median"] == pytest.approx(.02)
    assert m["center_acceleration"]["median"] == pytest.approx(0)
    assert m["absolute_log_area_change_per_second"]["median"] == 0
    assert stats([1, 2, 3])["p90"] == pytest.approx(2.8)


def test_timestamp_alignment_does_not_assume_same_frame_numbers():
    a = [row(frame=30, t=1), row(frame=60, t=2)]
    b = [row(frame=2, t=1.001), row(frame=4, t=2.001)]
    assert align(a, b)["matched"] == 2
    assert align(a, [row(frame=30, t=5)])["matched"] == 0
    assert len(align(a, [row(frame=30, t=5)])["left_unaligned"]) == 2
    assert align([row(), row("2")], [row()])["matched"] == 1
    assert align([], [row()])["agreement"] == 0
    assert align([], [])["agreement"] is None


def test_application_parser_with_real_pts_and_bad_csv(tmp_path):
    path = tmp_path / "tracks.csv"
    path.write_text("# header\n1,source,2,10,10,20,20,0.6,-1,fish,0.6\n"
                    "1,source,2,10,10,20,20,0.6,-1,fish,0.6\n"
                    "2,source,1,nan,10,20,20,0.6,-1\n"
                    "3,source,1,20,10,10,20,0.6,-1\n"
                    "4,source,99,10,10,20,20,0.6,-1\n"
                    "broken\n", encoding="utf-8")
    rows, issues = observations(path, [0, .043, .090])
    assert rows[0]["frame"] == 1
    assert rows[0]["t"] == .043
    assert len(rows) == 1
    assert issues == {"malformed": 1, "invalid": 2, "duplicate_rows": 1,
                      "conflicting_rows": 0, "missing_timestamp": 1}
    path.write_text("1,s,1,10,10,20,20,.6,-1\n1,s,1,11,10,20,20,.6,-1\n")
    assert observations(path, [0])[1]["conflicting_rows"] == 1


def test_retention_uses_effective_fps_without_changing_baseline(tmp_path):
    manifest = {"clips": [{"id": "live", "path": "live.mp4"}, {"id": "0", "path": "batch.mp4"}],
                "safe_configuration": {"VIAME_DOWNSAMPLE_FPS": "10", "VIAME_TRACKER_BUFFER_FRAMES": "30",
                "VIAME_DETECTOR_SCORE_THRESHOLD": ".10", "VIAME_TRACKER_HIGH_THRESHOLD": ".45",
                "VIAME_TRACKER_LOW_THRESHOLD": ".10", "VIAME_TRACKER_NEW_TRACK_THRESHOLD": ".50",
                "VIAME_FRAME_NUMBER_OFFSET": "-1"}}
    baseline = run_config("baseline-0", tmp_path, manifest)
    live = run_config("live-baseline", tmp_path, manifest)
    candidate = run_config("live-retention", tmp_path, manifest)
    assert baseline["buffer"] == 30 and baseline["retention_seconds"] == 3
    assert live["buffer"] == 30 and live["retention_seconds"] == 15
    assert candidate["buffer"] == 6 and candidate["retention_seconds"] == 3
    assert baseline["fishial_enabled"] is False


def test_detection_writer_uses_timestamp_not_sampled_ordinal(tmp_path):
    path = tmp_path / "detections.csv"
    path.write_text("0,00:00:00.000000,0,10,10,20,20,.6,-1\n"
                    "1,00:00:00.100000,1,10,10,20,20,.6,-1\n")
    rows, issues = observations(path, [0, .0333, .0667, .1])
    assert [r["frame"] for r in rows] == [0, 3]
    assert [r["t"] for r in rows] == [0, .1]
    assert not any(issues.values())


def test_nonzero_clip_pts_origin_aligns_detection_and_track_writers(tmp_path):
    detections = tmp_path / "d.csv"
    tracks = tmp_path / "t.csv"
    detections.write_text("0,00:00:00.032990,0,10,10,20,20,.6,-1\n")
    tracks.write_text("1,input_list.txt,1,10,10,20,20,.6,-1\n")
    times = [0, .033426, .066852]
    d, _ = observations(detections, times, timestamp_origin=.032990)
    t, _ = observations(tracks, times, timestamp_origin=.032990)
    assert d[0]["frame"] == t[0]["frame"] == 0
    assert align(d, t)["agreement"] == 1


def test_journal_resume_no_repeat_and_provenance_guard(tmp_path):
    path = tmp_path / "journal.json"
    j = Journal(path, "manifest")
    assert j.reserve("baseline", {"fps": 10})
    j.finish("baseline", status="complete", duration_seconds=12)
    resumed = Journal(path, "manifest")
    assert not resumed.reserve("baseline", {"fps": 10})
    assert resumed.used() == (1, 12)
    with pytest.raises(RuntimeError, match="configuration"):
        resumed.reserve("baseline", {"fps": 5})
    with pytest.raises(RuntimeError, match="Manifest"):
        Journal(path, "changed")


def test_interrupted_reservation_is_charged_and_not_retried(tmp_path):
    path = tmp_path / "journal.json"
    Journal(path, "m").reserve("run", {}, timeout=500)
    resumed = Journal(path, "m")
    assert resumed.used() == (1, 500)
    assert not resumed.reserve("run", {})


def test_pass_and_time_budgets(tmp_path):
    j = Journal(tmp_path / "passes.json", "m")
    for i in range(10):
        assert j.reserve(str(i), {}, timeout=1)
        j.finish(str(i), duration_seconds=.1)
    with pytest.raises(RuntimeError, match="budget"):
        j.reserve("eleventh", {})
    j = Journal(tmp_path / "time.json", "m")
    j.reserve("long", {}, timeout=5399)
    j.reserve("last", {}, timeout=10)
    assert j.data["runs"]["last"]["reserved_seconds"] == 1
    with pytest.raises(RuntimeError, match="budget"):
        j.reserve("overflow", {})


def test_atomic_save_rejects_non_finite_json_without_losing_old_file(tmp_path):
    path = tmp_path / "report.json"
    save(path, {"ok": 1})
    with pytest.raises(ValueError):
        save(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"ok": 1}


def test_duplicate_overlap_requires_sustained_frames_and_fragment_motion():
    rows = [row(str(j), i, i / 2) for i in range(3) for j in (1, 2)]
    m = metrics(rows, rows, 100, 100, 2)
    assert len(m["duplicate_pairs"]) == 1
    assert metrics(rows[:2], rows[:2], 100, 100, 2)["duplicate_pairs"] == []
    rows = [row("1", 0, 0, (0, 10, 10, 20)), row("1", 1, .5, (5, 10, 15, 20)),
            row("2", 2, 1, (10, 10, 20, 20))]
    assert len(metrics(rows, rows, 100, 100, 2)["fragmentation_pairs"]) == 1
    rows[-1]["box"] = [70, 10, 80, 20]
    assert not metrics(rows, rows, 100, 100, 2)["fragmentation_pairs"]


def test_empty_metrics_and_geometry():
    m = metrics([], [], 100, 100, 10)
    assert m["temporal_coverage"]["3"] is None
    assert m["tracks"] == 0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert compare_quality(m, m)["coverage_3"]["relative_change"] is None


def test_determinism_detects_identity_partition_change():
    a = [row("1", 0, 0), row("1", 1, .5)]
    b = [dict(r, id="100") for r in a]
    assert canonical(a) == canonical(b)
    b[1]["id"] = "101"
    assert canonical(a) != canonical(b)


def test_child_environment_excludes_credentials(monkeypatch):
    monkeypatch.setenv("FISHIAL_CLIENT_SECRET", "not-a-real-secret")
    monkeypatch.setenv("DATABASE_URL", "not-a-real-db")
    monkeypatch.setenv("FISHIAL_ENABLED", "true")
    assert "FISHIAL_CLIENT_SECRET" not in clean_env()
    assert "DATABASE_URL" not in clean_env()
    assert clean_env()["FISHIAL_ENABLED"] == "false"


def test_promotion_requires_every_clip_and_independent_proxy_families():
    evidence = candidate_evidence({}, {})
    assert evidence["motion_fusion"]["1_determinism"]["status"] == "unproven"
    assert evidence["motion_fusion"]["2_independent_temporal_improvements"]["status"] == "not_met"
    assert evidence["live_retention_3s"]["6_runtime"]["status"] == "unproven_dropped_segments"
    base = metrics([row()], [row()], 100, 100, 2)
    candidate = metrics([row(frame=i, t=i / 2) for i in range(5)],
                        [row(frame=i, t=i / 2) for i in range(5)], 100, 100, 2)
    runs = {"live-baseline": {"metrics": base, "duration_seconds": 10, "runtime_over_source_duration": .5},
            "live-retention": {"metrics": candidate, "duration_seconds": 15, "runtime_over_source_duration": .75}}
    noise = {k: {"baseline": 0, "candidate": 0} for k in ("coverage_3", "coverage_5")}
    evidence = candidate_evidence(runs, {"baseline-1:repeat-1": {"temporal_quality": noise}})
    criterion = evidence["live_retention_3s"]["2_independent_temporal_improvements"]
    assert criterion["status"] == "not_met"
    assert criterion["clip_values"][0]["independent_improved_families"] == ["persistence"]
