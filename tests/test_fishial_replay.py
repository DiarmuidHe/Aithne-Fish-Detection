"""The offline tuning harness. --dry-run must never reach the network."""
from __future__ import annotations

import json
import uuid

import pytest

from app.db.models import LiveFishTrack, LiveMonitorSession, utc_now
from app.services.fishial import FishialClient, trim_raw
from scripts import fishial_replay


def response(species=(("winner", .9),), definitions=None, objects=1, bbox=(10, 8, 190, 92)):
    payload = {"ok": True, "objects": [], "definitions": dict(
        definitions or {"winner": {"scientificName": "Gadus morhua"}})}
    for index in range(objects):
        payload["objects"].append({
            "bbox": list(bbox) if index == 0 else [160, 5, 199, 45],
            "shape": [[index, index]] * 400,
            "species": [{"id": name, "certainty": score} for name, score in species]
            if index == 0 else [],
        })
    return payload


def seed(db, settings, frames, region=None, staged_boxes=True):
    session = LiveMonitorSession(source_url="https://camera.example/", species_id_enabled=True,
        species_id_fish_target=1, species_id_frames_per_fish=5, species_id_region=region,
        species_id_api_calls=len(frames))
    db.add(session)
    db.flush()
    staged, records = [], []
    for number, raw in enumerate(frames):
        entry = {"frame_number": number, "timestamp": number, "window": number, "score": 1.0,
                 "quality": {"short_side": 120, "sharpness": 200, "confidence": .85,
                             "luminance": 120, "contrast": 60, "colorfulness": 40}}
        if staged_boxes:
            entry |= {"expected_box": [.05, .1, .95, .9], "crop_size": [200, 100]}
        staged.append(entry)
        records.append({"frame_number": number, "attempts": [{"retry": False}], "raw": raw,
                        "species": None, "score": None, "voted": False, "succeeded": True})
    track = LiveFishTrack(id=uuid.uuid4(), session_id=session.id, status="finalized",
        first_seen_at=utc_now(), last_seen_at=utc_now(), detection_count=len(frames),
        max_confidence=.8, mean_confidence=.7, x1=1, y1=1, x2=100, y2=100,
        fishial_state="review_required",
        fishial_votes_json=json.dumps({"staged": staged, "frames": records,
                                       "reason": "consensus not reached"}))
    db.add(track)
    db.commit()
    return session, track


@pytest.fixture
def replay_settings(test_settings, db_session_factory, monkeypatch):
    """Point the harness at the in-memory test database and settings."""

    from pydantic import SecretStr
    test_settings.fishial_enabled = True
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    monkeypatch.setattr(fishial_replay, "get_settings", lambda: test_settings)
    monkeypatch.setattr(fishial_replay, "create_engine", lambda *a, **k: _Engine())
    monkeypatch.setattr(fishial_replay, "sessionmaker",
                        lambda *a, **k: db_session_factory)
    return test_settings


class _Engine:
    def dispose(self):
        pass


def test_dry_run_reproduces_the_recorded_verdict_without_calling_the_api(
        replay_settings, db_session_factory, capsys):
    calls = []
    with db_session_factory() as db:
        session, _track = seed(db, replay_settings, [response()] * 5)
        sid = str(session.id)

    def parse(raw, target):
        calls.append(raw)
        return FishialClient._parse(raw, target, replay_settings)

    assert fishial_replay.main(["--dry-run", "--session", sid], parse=parse) == 0
    # Every response was re-parsed locally; nothing reached a client.
    assert len(calls) >= 3
    out = capsys.readouterr().out
    assert "identified" in out
    assert "Gadus morhua" in out


def test_dry_run_never_constructs_a_client(replay_settings, db_session_factory, monkeypatch, capsys):
    monkeypatch.setattr(fishial_replay, "FishialClient",
                        type("Forbidden", (), {"_parse": staticmethod(
                            lambda *a, **k: pytest.fail("dry run parsed through a live client"))}))
    with db_session_factory() as db:
        session, _ = seed(db, replay_settings, [response()] * 3)
        sid = str(session.id)
    stub = lambda raw, target: FishialClient._parse(raw, target, replay_settings)
    assert fishial_replay.main(["--dry-run", "--session", sid], parse=stub) == 0
    assert "calls actually spent: 3" in capsys.readouterr().out


def test_dry_run_recovers_multi_object_answers_and_applies_the_region(
        replay_settings, db_session_factory, capsys):
    caribbean = {"reef": {"scientificName": "Sparisoma aurofrenatum"}}
    with db_session_factory() as db:
        session, _ = seed(db, replay_settings,
                          [response((("reef", .9),), caribbean, objects=2)] * 5,
                          region="north_east_atlantic")
        sid = str(session.id)
    stub = lambda raw, target: FishialClient._parse(raw, target, replay_settings)
    assert fishial_replay.main(["--dry-run", "--session", sid], parse=stub) == 0
    out = capsys.readouterr().out
    # The matcher recovers the answer from the two-object response, and the regional
    # filter then rejects it - which is exactly what happened on SmartBay 3.
    assert "implausible_for_region: Sparisoma aurofrenatum" in out
    assert "review_required" in out


def test_dry_run_reports_what_the_stop_rules_would_have_saved(
        replay_settings, db_session_factory, capsys):
    with db_session_factory() as db:
        session, _ = seed(db, replay_settings, [response(species=())] * 5)
        sid = str(session.id)
    stub = lambda raw, target: FishialClient._parse(raw, target, replay_settings)
    fishial_replay.main(["--dry-run", "--session", sid, "--set", "fishial_max_empty_responses=2"],
                        parse=stub)
    out = capsys.readouterr().out
    assert "calls actually spent: 5   calls the stop rules would spend: 2 (saved 3)" in out


def test_dry_run_still_decides_from_a_trimmed_audit(replay_settings, db_session_factory, capsys):
    """Trimming must not remove anything the harness needs."""

    with db_session_factory() as db:
        session, track = seed(db, replay_settings, [trim_raw(response()) for _ in range(5)])
        sid = str(session.id)
        assert all("shape" not in obj for frame in track.fishial_votes["frames"]
                   for obj in frame["raw"]["objects"])
    stub = lambda raw, target: FishialClient._parse(raw, target, replay_settings)
    assert fishial_replay.main(["--dry-run", "--session", sid], parse=stub) == 0
    assert "Gadus morhua" in capsys.readouterr().out


def test_dry_run_reconstructs_a_target_box_for_a_legacy_session(
        replay_settings, db_session_factory, capsys):
    """Sessions staged before expected_box existed can still be re-decided."""

    from app.db.models import LiveFishDetection
    with db_session_factory() as db:
        session, track = seed(db, replay_settings, [response(objects=2)] * 5, staged_boxes=False)
        for number in range(5):
            db.add(LiveFishDetection(track_id=track.id, observed_at=utc_now(),
                frame_number=number, confidence=.7, x1=157, y1=366, x2=692, y2=602))
        db.commit()
        sid = str(session.id)
    stub = lambda raw, target: FishialClient._parse(raw, target, replay_settings)
    assert fishial_replay.main(["--dry-run", "--session", sid], parse=stub) == 0
    assert "calls actually spent: 5" in capsys.readouterr().out


def test_replay_refuses_to_spend_without_an_explicit_ceiling(replay_settings, tmp_path):
    with pytest.raises(SystemExit) as exc:
        fishial_replay.main(["--replay", str(tmp_path)])
    assert exc.value.code != 0


def test_overrides_are_validated(replay_settings):
    tuned = fishial_replay.apply_overrides(replay_settings, ["fishial_quality_floor=0.4"])
    assert tuned.fishial_quality_floor == .4
    with pytest.raises(SystemExit):
        fishial_replay.apply_overrides(replay_settings, ["not_a_setting=1"])
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        fishial_replay.apply_overrides(replay_settings, ['fishial_preprocess="sharpen"'])


def test_a_missing_session_is_a_clear_error(replay_settings):
    stub = lambda raw, target: None
    with pytest.raises(SystemExit):
        fishial_replay.main(["--dry-run", "--session", str(uuid.uuid4())], parse=stub)


@pytest.mark.parametrize("staged_count,raws,reason", [
    (5, [response()] + [response(species=())] * 2, "classifier returned no candidates"),
    (4, [response(species=())] * 2, "classifier returned no candidates"),
    (3, [response((("gar", .46),), {"gar": {"scientificName": "Lepisosteus oculatus"}})],
     "implausible_for_region"),
])
def test_smartbay_early_stopped_audit_keeps_unsubmitted_capacity(
        replay_settings, db_session_factory, staged_count, raws, reason):
    with db_session_factory() as db:
        _, track = seed(db, replay_settings, [response()] * staged_count)
        audit = track.fishial_votes
        audit["frames"] = audit["frames"][:len(raws)]
        for frame, raw in zip(audit["frames"], raws):
            frame["raw"] = raw
        track.fishial_votes_json = json.dumps(audit)
        result = fishial_replay.rescore_track(track, replay_settings, "north_east_atlantic",
            lambda raw, target: FishialClient._parse(raw, target, replay_settings), frames_per_fish=5)
    assert result["calls_recorded"] == result["calls_replayed"] == len(raws)
    assert result["new"] == "review_required"
    assert result["new_reason"] == reason
    assert result["incomplete"] is False


def test_replay_counts_reserved_retries_but_not_unreserved_frames(replay_settings, db_session_factory):
    with db_session_factory() as db:
        _, track = seed(db, replay_settings, [response(species=())] * 5)
        audit = track.fishial_votes
        audit["frames"][0]["attempts"].append({"retry": True})
        audit["frames"][4] = {"frame_number": 4, "attempts": [], "reason": "budget exhausted"}
        track.fishial_votes_json = json.dumps(audit)
        result = fishial_replay.rescore_track(track, replay_settings, None,
            lambda raw, target: FishialClient._parse(raw, target, replay_settings))
    assert result["calls_recorded"] == 5
    assert result["calls_replayed"] == 3  # Two frames, one retry.


def test_replay_marks_missing_answers_instead_of_inventing_savings(replay_settings, db_session_factory):
    replay_settings.fishial_max_empty_responses = 0
    with db_session_factory() as db:
        _, track = seed(db, replay_settings, [response(species=())] * 5)
        audit = track.fishial_votes
        audit["frames"] = audit["frames"][:2]
        track.fishial_votes_json = json.dumps(audit)
        result = fishial_replay.rescore_track(track, replay_settings, None,
            lambda raw, target: FishialClient._parse(raw, target, replay_settings))
    assert result["incomplete"] is True
    assert result["calls_replayed"] == result["calls_recorded"] == 2
