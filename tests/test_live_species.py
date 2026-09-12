import json

import httpx
import pytest
from sqlalchemy import select

from app.db.models import LiveFishTrack, LiveMonitorSession, utc_now
from app.services.fishial import FishialClient, FishialError, FishialPrediction
from app.services.live_monitor import scratch_path
from app.services.live_species import SpeciesIdentifier


def crop_bytes():
    import cv2
    import numpy as np
    return cv2.imencode(".jpg", np.full((120, 200, 3), [30, 70, 90], np.uint8))[1].tobytes()


@pytest.fixture
def enabled_settings(test_settings):
    from pydantic import SecretStr
    test_settings.fishial_enabled = True
    test_settings.fishial_client_id = "id"
    test_settings.fishial_client_secret = SecretStr("secret")
    return test_settings


def session_row(db, target=1, **kwargs):
    row = LiveMonitorSession(source_url="https://camera.example/", species_id_enabled=True,
        species_id_fish_target=target, species_id_frames_per_fish=5,
        species_id_candidate_pool_size=20, **kwargs)
    db.add(row)
    db.commit()
    return row


def quality(short_side=120.0, sharpness=200.0, confidence=.85, contrast=60.0, colorfulness=40.0):
    return {"short_side": short_side, "sharpness": sharpness, "confidence": confidence,
            "luminance": 120.0, "contrast": contrast, "colorfulness": colorfulness}


def staged_track(db, session, settings, count=5, scores=None, measures=None,
                 state="candidate", status="finalized", **kwargs):
    """A finalized candidate holding ``count`` staged crops, ready for selection."""

    track = LiveFishTrack(session_id=session.id, first_seen_at=utc_now(), last_seen_at=utc_now(),
        species="fish", x1=10, y1=10, x2=110, y2=110, fishial_frames_used=count,
        fishial_state=state, status=status, **kwargs)
    db.add(track)
    db.flush()
    directory = scratch_path(settings, str(session.id), str(track.id), "fishial")
    directory.mkdir(parents=True)
    staged = []
    for number in range(count):
        (directory / f"{number:012d}.jpg").write_bytes(crop_bytes())
        staged.append({"frame_number": number, "timestamp": number, "window": number,
                       "score": (scores or {}).get(number, 1.0 - number / 100),
                       "expected_box": [.1, .1, .9, .9], "crop_size": [200, 120],
                       "quality": (measures or {}).get(number, quality())})
    track.fishial_votes_json = json.dumps({"staged": staged, "window_origin": 0})
    db.commit()
    return track


class StubClient:
    """One image call per identify(), like the source adapters the worker injects."""

    def __init__(self, results):
        self.results, self.calls, self.boxes = iter(results), 0, []

    def identify(self, image, expected_box=None):
        assert image == crop_bytes()
        self.calls += 1
        self.boxes.append(expected_box)
        value = next(self.results)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, list):
            return FishialPrediction(value, {}, 0, 1.0, 1)
        name, score = value
        return FishialPrediction([(name, score)], {"stored": True}, 0, 1.0, 1)


@pytest.mark.parametrize("failure", [401, 429, 500])
def test_enhance_once_before_reservation_and_retry_reuses_bytes(
        db_session_factory, enabled_settings, failure):
    import hashlib

    from app.services.fish_enhancement import PROVENANCE, EnhancedCrop, enhance_crop
    events, sent, boxes = [], [], []
    enabled_settings.fishial_preprocess = "funie_gan"
    enabled_settings.fishial_funie_model_sha256 = "0" * 64
    def fake_enhance(image, settings):
        events.append("enhance")
        adjusted = enhance_crop(image, settings.model_copy(update={"fishial_preprocess": "both"}))
        return EnhancedCrop(adjusted.image, {**adjusted.metadata, "mode": "funie_gan",
            "model_id": PROVENANCE["model_id"], "model_sha256": "0" * 64})
    def handle(request):
        if request.url.path.endswith("auth"):
            return httpx.Response(200, json={"access_token": "secret-token"})
        events.append("send")
        sent.append(request.content)
        assert events[-2] == "reserve"
        return httpx.Response(failure) if len(sent) == 1 else httpx.Response(200, json={
            "ok": True, "objects": [{"species": [{"id": "fish", "certainty": .9}]}],
            "definitions": {"fish": {"scientificName": "Pollachius virens"}}})
    with db_session_factory() as db, httpx.Client(transport=httpx.MockTransport(handle)) as http:
        session = session_row(db)
        track = staged_track(db, session, enabled_settings, count=3)
        audit = track.fishial_votes
        expected = [0, .08, .45, .92]  # Off-center edge-clamped target.
        for staged in audit["staged"]:
            staged["expected_box"] = expected
        track.fishial_votes_json = json.dumps(audit)
        db.commit()
        client = FishialClient(enabled_settings, http)
        real_identify = client.identify
        def identify(image, box):
            boxes.append(box)
            return real_identify(image, box)
        client.identify = identify
        identifier = SpeciesIdentifier(db, session, enabled_settings, client, fake_enhance)
        real_reserve = identifier._reserve
        def reserve(*args):
            assert "preprocessing" in track.fishial_votes["frames"][-1]
            events.append("reserve")
            return real_reserve(*args)
        identifier._reserve = reserve
        identifier.run(final=True)
        assert events[:5] == ["enhance", "reserve", "send", "reserve", "send"]
        assert events.count("enhance") == 3 and len(sent) == 4
        assert sent[0] == sent[1] and sent[0] != crop_bytes()
        assert boxes == [expected] * 3
        meta = track.fishial_votes["frames"][0]["preprocessing"]
        assert meta["submitted_sha256"] == hashlib.sha256(sent[0]).hexdigest()
        assert meta["model_id"] == PROVENANCE["model_id"] and "inference_ms" in meta
        assert all(secret not in json.dumps(meta) for secret in ("secret-token", "https://", str(enabled_settings.output_root)))
        assert not scratch_path(enabled_settings, str(session.id), str(track.id), "fishial").exists()


def test_preprocessing_failure_spends_nothing_and_stops_when_consensus_unreachable(
        db_session_factory, enabled_settings):
    calls = []
    def fail(image, settings):
        calls.append(image)
        raise RuntimeError("secret-token private-image private-path")
    with db_session_factory() as db:
        session = session_row(db)
        track = staged_track(db, session, enabled_settings, count=3)
        client = StubClient([])
        identifier = SpeciesIdentifier(db, session, enabled_settings, client, fail)
        identifier._reserve = lambda *args: pytest.fail("Reserved a failed enhancement")
        identifier.run(final=True)
        assert len(calls) == 1 and client.calls == 0 and session.species_id_api_calls == 0
        assert track.fishial_state == "review_required"
        assert track.fishial_votes["frames"][0]["reason"] == "preprocessing failed"
        assert "secret" not in track.fishial_votes_json


def test_unselected_candidates_are_never_enhanced(db_session_factory, enabled_settings):
    with db_session_factory() as db:
        session = session_row(db)
        staged_track(db, session, enabled_settings, status="active")
        identifier = SpeciesIdentifier(db, session, enabled_settings, StubClient([]),
            lambda *args: pytest.fail("Enhanced an unpaid candidate"))
        identifier.run()


@pytest.mark.parametrize("results,state,tally,calls,confidence,reason", [
    # Decisive: three agreeing votes cannot be unseated by the two frames left, so
    # the last two calls are never bought.
    ([("Lutjanus griseus", .8), ("Lutjanus griseus", .9), ("Lutjanus griseus", .7),
      ("Lutjanus griseus", 1), ("Other", .9)], "identified", {"Lutjanus griseus": 3}, 3, .8, None),
    ([("A", .9)] * 2 + [("B", .9)] * 2 + [("C", .9)], "review_required",
     {"A": 2, "B": 2, "C": 1}, 5, None, "consensus not reached"),
    ([("A", .4)] * 5, "review_required", {"A": 5}, 5, None, "consensus not reached"),
    ([("A", .9)] * 4 + [FishialError()], "identified", {"A": 3}, 3, .9, None),
    # Hopeless: after three failures even a clean sweep of the rest cannot reach
    # min_votes within the vote ratio, so the last two calls are never bought.
    ([FishialError()] * 5, "error", {}, 3, None, "all frames failed"),
])
def test_consensus_and_early_stopping(db_session_factory, enabled_settings, results, state,
                                      tally, calls, confidence, reason):
    with db_session_factory() as db:
        session = session_row(db)
        track = staged_track(db, session, enabled_settings)
        client = StubClient(results)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert track.fishial_state == state
        assert track.fishial_votes["tally"] == tally
        assert len(track.fishial_votes["frames"]) == calls
        assert client.calls == session.species_id_api_calls == calls
        assert session.species_id_calls_saved == 5 - calls
        assert session.species_id_fish_enrolled == 1  # Counted at selection.
        assert track.species == "fish"
        assert track.fishial_votes.get("reason") == reason
        # Every call carries the stored expected box, so a multi-object response can
        # be resolved against the fish we centred the crop on.
        assert client.boxes == [[.1, .1, .9, .9]] * calls
        if state == "identified":
            assert track.fishial_species in tally
            assert track.fishial_species_confidence == pytest.approx(confidence)
        else:
            assert track.fishial_species is None
        assert track.fishial_completed_at
        assert not scratch_path(enabled_settings, str(session.id), str(track.id), "fishial").exists()


def test_ambiguous_and_failed_frames_stay_in_denominator(db_session_factory, enabled_settings):
    enabled_settings.fishial_min_frame_margin = .2
    enabled_settings.fishial_vote_ratio = .8
    with db_session_factory() as db:
        session = session_row(db)
        track = staged_track(db, session, enabled_settings)
        client = StubClient([[("A", .9), ("B", .85)]] + [("A", .9)] * 3 + [FishialError()])
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert track.fishial_state == "review_required"
        assert track.fishial_votes["tally"] == {"A": 3}
        assert track.fishial_votes["submitted_frames"] == 5
        assert track.fishial_votes["frames"][0]["reason"] == "ambiguous frame"


def test_insufficient_frames_and_off_never_construct_client(db_session_factory, enabled_settings, monkeypatch):
    import app.services.live_species as module
    def forbidden(*args):
        pytest.fail("Client constructed without eligible crops")
    monkeypatch.setattr(module, "FishialClient", forbidden)
    with db_session_factory() as db:
        session = session_row(db)
        track = staged_track(db, session, enabled_settings, count=2, status="finalized")
        SpeciesIdentifier(db, session, enabled_settings).run()
        assert track.fishial_state == "review_required"
        assert track.fishial_votes["reason"] == "insufficient clear frames"
        assert session.species_id_api_calls == 0
        other = staged_track(db, session, enabled_settings)
        enabled_settings.fishial_enabled = False
        SpeciesIdentifier(db, session, enabled_settings).run()
        assert other.fishial_state == "candidate"


def test_budget_ceiling_and_persisted_restart(db_session_factory, enabled_settings):
    """The hard ceiling still binds; early stopping only stretches it over more fish."""

    with db_session_factory() as db:
        # Three tracks already selected by an earlier pass, as after a worker restart.
        session = session_row(db, species_id_fish_enrolled=3)
        tracks = [staged_track(db, session, enabled_settings, state="ready") for _ in range(3)]
        client = StubClient([("A", .9)] * 30)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        # target(1) * frames_per_fish(5) + max_api_retries(2) = 7, unchanged.
        assert client.calls == session.species_id_api_calls == 7
        assert [t.fishial_state for t in tracks] == ["identified", "identified", "review_required"]
        assert tracks[2].fishial_votes["reason"] == "budget exhausted"
        sid = session.id
    with db_session_factory() as db:
        session = db.get(LiveMonitorSession, sid)
        other = staged_track(db, session, enabled_settings, state="ready")
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert client.calls == 7
        assert other.fishial_votes["reason"] == "budget exhausted"


def test_crash_after_reservation_never_replays_frame(db_session_factory, enabled_settings):
    class Crash(BaseException):
        pass
    class CrashingClient:
        def identify(self, image, expected_box=None):
            raise Crash()
    with db_session_factory() as db:
        session = session_row(db)
        track = staged_track(db, session, enabled_settings)
        sid, tid = session.id, track.id
        with pytest.raises(Crash):
            SpeciesIdentifier(db, session, enabled_settings, CrashingClient()).run()
    with db_session_factory() as db:
        session, track = db.get(LiveMonitorSession, sid), db.get(LiveFishTrack, tid)
        assert session.species_id_api_calls == 1
        assert track.fishial_state == "submitted"
        client = StubClient([("A", .9)] * 4)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        # The uncertain send is never replayed, and three clean votes then decide it.
        assert client.calls == 3
        assert session.species_id_api_calls == 4
        assert track.fishial_state == "identified"
        assert track.fishial_votes["frames"][0]["reason"] == "interrupted"


def test_actual_http_retries_share_durable_allowance(db_session_factory, enabled_settings):
    calls = []
    def handle(request):
        if request.url.path.endswith("auth"):
            return httpx.Response(200, json={"access_token": "token"})
        calls.append(1)
        return httpx.Response(503)
    # Pacing is exercised separately; hold nothing back so both tracks are selected.
    enabled_settings.fishial_late_reserve_fish = 0
    with db_session_factory() as db, httpx.Client(transport=httpx.MockTransport(handle)) as http:
        session = session_row(db, target=2)
        track = staged_track(db, session, enabled_settings)
        SpeciesIdentifier(db, session, enabled_settings, FishialClient(enabled_settings, http)).run()
        # Frame 0 spends its own reservation plus both session retries; frames 1 and 2
        # spend one each, after which no distribution of the two remaining frames can
        # reach min_votes, so the track stops instead of buying them.
        assert len(calls) == session.species_id_api_calls == 5
        assert track.fishial_state == "error"
        assert track.fishial_votes["stopped_early"] == "consensus unreachable"
        other = staged_track(db, session, enabled_settings)
        SpeciesIdentifier(db, session, enabled_settings, FishialClient(enabled_settings, http)).run()
        # The retry allowance is session-wide and already spent, so the second track
        # gets one reservation per frame and stops on the same rule.
        assert len(calls) == session.species_id_api_calls == 8
        assert other.fishial_state == "error"
        assert session.species_id_calls_saved == 4
        assert sum(a["retry"] for t in (track, other) for f in t.fishial_votes["frames"]
                   for a in f["attempts"]) == 2


def test_species_api_validation_and_aggregation(client, db_session_factory, test_settings):
    test_settings.live_monitor_enabled = True
    test_settings.viame_mock = False
    assert client.get("/live/sources").json()["fishial"] == {
        "enabled": False, "max_fish_per_session": 20, "default_frames_per_fish": 5,
        "request_max_frames": 12, "request_default_frames": 5}
    assert client.post("/live/start", json={"species_id_fish_target": 1}).status_code == 409
    test_settings.fishial_enabled = True
    for body in ({"species_id_fish_target": -1}, {"species_id_fish_target": 21},
                 {"species_id_frames_per_fish": 0}, {"species_id_frames_per_fish": 21}):
        assert client.post("/live/start", json=body).status_code == 422
    response = client.post("/live/start", json={"species_id_fish_target": 4, "species_id_frames_per_fish": 6})
    assert response.status_code == 202
    data = response.json()
    assert data["species_id"]["fish_target"] == 4
    assert data["species_id"]["frames_per_fish"] == 6
    with db_session_factory() as db:
        session = db.scalar(select(LiveMonitorSession))
        session.species_id_fish_enrolled, session.species_id_api_calls = 4, 18
        for state, score in (("identified", .8), ("identified", 1), ("review_required", None), ("error", None)):
            db.add(LiveFishTrack(session_id=session.id, first_seen_at=utc_now(), last_seen_at=utc_now(),
                x1=1, y1=1, x2=100, y2=100, fishial_state=state,
                fishial_species="A" if score else None, fishial_species_confidence=score,
                fishial_votes_json='{"tally": {"A": 4}}'))
        db.commit()
    summary = client.get(f"/live/{data['id']}/species").json()
    assert summary["api_calls"] == 18 and summary["review_required"] == 2
    assert summary["species"] == [{"species": "A", "count": 2, "mean_confidence": .9}]
    status = client.get(f"/live/{data['id']}/status").json()["species_id"]
    assert status["fish_identified"] == status["fish_review_required"] == 2
    tracks = client.get(f"/live/{data['id']}/tracks").json()
    assert tracks[0]["fishial_votes"] == {"tally": {"A": 4}}
    assert "fishial_frames_used" in tracks[0]
    client.post(f"/live/{data['id']}/stop")
    assert client.post("/live/start").json()["species_id"]["enabled"] is False


def test_selection_pays_for_the_best_fish_not_the_first(db_session_factory, enabled_settings):
    """Arrival order is uncorrelated with identifiability, so rank before spending."""

    enabled_settings.fishial_late_reserve_fish = 0
    with db_session_factory() as db:
        session = session_row(db, target=2, started_at=utc_now())
        # Staged first, but small and soft: exactly the fish first-come enrollment
        # used to spend the whole budget on.
        early = staged_track(db, session, enabled_settings,
                             measures={n: quality(short_side=20, sharpness=15) for n in range(5)})
        best = staged_track(db, session, enabled_settings,
                            measures={n: quality(short_side=140, sharpness=400) for n in range(5)})
        middle = staged_track(db, session, enabled_settings,
                              measures={n: quality(short_side=70, sharpness=120) for n in range(5)})
        client = StubClient([("A", .9)] * 30)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert best.fishial_state == "identified"
        assert middle.fishial_state == "identified"
        # Never selected, so never paid for; it only resolves at finish().
        assert early.fishial_state == "candidate"
        assert session.species_id_fish_enrolled == 2
        assert best.fishial_quality_score > middle.fishial_quality_score > early.fishial_quality_score


def test_sub_floor_candidates_end_review_required_with_zero_calls(db_session_factory, enabled_settings):
    enabled_settings.fishial_quality_floor = .8
    enabled_settings.fishial_late_reserve_fish = 0
    with db_session_factory() as db:
        session = session_row(db, target=2, started_at=utc_now())
        poor = staged_track(db, session, enabled_settings,
                            measures={n: quality(short_side=10, sharpness=5, contrast=3,
                                                 colorfulness=1, confidence=.5) for n in range(5)})
        good = staged_track(db, session, enabled_settings)
        client = StubClient([("A", .9)] * 30)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert poor.fishial_state == "review_required"
        assert poor.fishial_votes["reason"] == "below quality floor"
        assert poor.fishial_votes["frames"] == []
        # Its staged frames are freed immediately: nothing will ever be bought for it.
        assert not scratch_path(enabled_settings, str(session.id), str(poor.id), "fishial").exists()
        assert good.fishial_state == "identified"
        assert session.species_id_fish_enrolled == 1
        assert client.calls == session.species_id_api_calls


def test_an_unfinalized_track_is_never_paid_for(db_session_factory, enabled_settings):
    """A fish still in view can still stage a better frame, so never buy it mid-life."""

    enabled_settings.fishial_late_reserve_fish = 0
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now())
        alive = staged_track(db, session, enabled_settings, status="active")
        client = StubClient([("A", .9)] * 30)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert alive.fishial_state == "candidate"
        assert client.calls == session.species_id_api_calls == 0
        assert session.species_id_fish_enrolled == 0
        alive.status = "finalized"
        db.commit()
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert alive.fishial_state == "identified"
        assert session.species_id_fish_enrolled == 1


def test_pacing_holds_slots_back_until_the_reserve_is_released(db_session_factory, enabled_settings):
    """The session cannot know its best fish until it ends, so do not spend early."""

    enabled_settings.fishial_late_reserve_fish = 2
    enabled_settings.fishial_reserve_after_seconds = 3600
    with db_session_factory() as db:
        session = session_row(db, target=4, started_at=utc_now())
        tracks = [staged_track(db, session, enabled_settings,
                               measures={n: quality(short_side=40 + 20 * index) for n in range(5)})
                  for index in range(4)]
        client = StubClient([("A", .9)] * 60)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        # fish_target(4) - late_reserve_fish(2) = 2 may be identified so far.
        assert session.species_id_fish_enrolled == 2
        assert sum(t.fishial_state == "identified" for t in tracks) == 2
        assert sum(t.fishial_state == "candidate" for t in tracks) == 2
        # finish() releases the reserve and spends what is left on the best remaining.
        SpeciesIdentifier(db, session, enabled_settings, client).finish()
        assert session.species_id_fish_enrolled == 4
        assert all(t.fishial_state == "identified" for t in tracks)


def test_the_reserve_is_clamped_below_the_target(db_session_factory, enabled_settings):
    enabled_settings.fishial_late_reserve_fish = 5
    enabled_settings.fishial_reserve_after_seconds = 3600
    with db_session_factory() as db:
        session = session_row(db, target=1, started_at=utc_now())
        track = staged_track(db, session, enabled_settings)
        client = StubClient([("A", .9)] * 30)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        # A reserve of target-1 = 0 still leaves one slot spendable during the session.
        assert track.fishial_state == "identified"


def test_finish_distinguishes_unselected_candidates_from_interrupted_sends(db_session_factory, enabled_settings):
    enabled_settings.fishial_quality_floor = 0
    with db_session_factory() as db:
        session = session_row(db, target=1, started_at=utc_now())
        selected = staged_track(db, session, enabled_settings)
        loser = staged_track(db, session, enabled_settings,
                             measures={n: quality(short_side=20) for n in range(5)})
        thin = staged_track(db, session, enabled_settings, count=2)
        client = StubClient([("A", .9)] * 30)
        SpeciesIdentifier(db, session, enabled_settings, client).finish()
        assert selected.fishial_state == "identified"
        # Never selected: the budget ran out before its rank came up. That is a
        # different operational problem from a track whose frames failed.
        assert loser.fishial_state == "review_required"
        assert loser.fishial_votes["reason"] == "budget exhausted"
        assert thin.fishial_votes["reason"] == "insufficient clear frames"
        assert not scratch_path(enabled_settings, str(session.id), str(loser.id), "fishial").exists()


def test_repeated_empty_responses_stop_the_track(db_session_factory, enabled_settings):
    """Once a track starts coming back empty the rest are near-certain to be empty."""

    enabled_settings.fishial_max_empty_responses = 2
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now())
        track = staged_track(db, session, enabled_settings)
        client = StubClient([[]] * 5)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert client.calls == session.species_id_api_calls == 2
        assert session.species_id_calls_saved == 3
        assert track.fishial_state == "review_required"
        assert track.fishial_votes["reason"] == "classifier returned no candidates"
        assert track.fishial_votes["stopped_early"] == "classifier returned no candidates"


def test_empty_response_stop_can_be_disabled(db_session_factory, enabled_settings):
    enabled_settings.fishial_max_empty_responses = 0
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now())
        track = staged_track(db, session, enabled_settings)
        client = StubClient([[]] * 5)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        # Without the rule only the "hopeless" counting rule stops it: after three
        # non-votes, min_votes can no longer be reached.
        assert client.calls == 3
        assert track.fishial_votes["reason"] == "consensus unreachable"


def test_a_non_consecutive_empty_run_does_not_stop_the_track(db_session_factory, enabled_settings):
    enabled_settings.fishial_max_empty_responses = 2
    enabled_settings.fishial_min_votes = 2
    enabled_settings.fishial_vote_ratio = 0
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now())
        track = staged_track(db, session, enabled_settings)
        client = StubClient([[], ("A", .9), [], [], ("A", .9)])
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        # The run of empties is broken by the vote, so the counter restarts; the
        # track stops on the second consecutive empty, not the second overall.
        assert client.calls == 4
        assert track.fishial_votes["reason"] == "classifier returned no candidates"


def test_implausible_species_are_dropped_before_voting(db_session_factory, enabled_settings):
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now(), species_id_region="north_east_atlantic")
        track = staged_track(db, session, enabled_settings)
        client = StubClient([[("Sparisoma aurofrenatum", .9)]] * 5)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert track.fishial_state == "review_required"
        assert track.fishial_votes["tally"] == {}
        frame = track.fishial_votes["frames"][0]
        assert frame["dropped_species"] == ["Sparisoma aurofrenatum"]
        assert frame["dropped_reason"] == "implausible_for_region"
        # Filtering to empty abstains rather than falling through to the name.
        assert frame["reason"] == "implausible_for_region"
        assert frame["voted"] is False


def test_the_region_filter_keeps_the_next_plausible_candidate(db_session_factory, enabled_settings):
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now(), species_id_region="north_east_atlantic")
        track = staged_track(db, session, enabled_settings)
        client = StubClient([[("Sparisoma aurofrenatum", .95), ("Gadus morhua", .8)]] * 5)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert track.fishial_species == "Gadus morhua"
        assert track.fishial_state == "identified"


def test_the_region_filter_can_be_switched_off(db_session_factory, enabled_settings):
    enabled_settings.fishial_region_filter_enabled = False
    with db_session_factory() as db:
        session = session_row(db, started_at=utc_now(), species_id_region="north_east_atlantic")
        track = staged_track(db, session, enabled_settings)
        client = StubClient([("Sparisoma aurofrenatum", .9)] * 5)
        SpeciesIdentifier(db, session, enabled_settings, client).run()
        assert track.fishial_species == "Sparisoma aurofrenatum"


def test_budget_ceiling_holds_when_selection_and_stopping_all_fire(db_session_factory, enabled_settings):
    """The invariant across every new path: never more than the formula allows."""

    enabled_settings.fishial_late_reserve_fish = 1
    enabled_settings.fishial_reserve_after_seconds = 0
    enabled_settings.fishial_quality_floor = .2
    enabled_settings.fishial_max_empty_responses = 2
    with db_session_factory() as db:
        session = session_row(db, target=3, started_at=utc_now())
        good = [staged_track(db, session, enabled_settings) for _ in range(3)]
        empty = staged_track(db, session, enabled_settings,
                             measures={n: quality(short_side=100) for n in range(5)})
        floored = staged_track(db, session, enabled_settings, count=3,
                               measures={n: quality(short_side=1, sharpness=1, contrast=1,
                                                    colorfulness=1, confidence=.5)
                                         for n in range(3)})
        thin = staged_track(db, session, enabled_settings, count=1)
        client = StubClient([("A", .9)] * 9 + [[]] * 20)
        identifier = SpeciesIdentifier(db, session, enabled_settings, client)
        identifier.run()
        identifier.finish()
        ceiling = (session.species_id_fish_target * session.species_id_frames_per_fish
                   + enabled_settings.fishial_max_api_retries)
        assert session.species_id_api_calls == client.calls <= ceiling
        assert session.species_id_fish_enrolled <= session.species_id_fish_target
        assert sum(t.fishial_state == "identified" for t in good) == 3
        assert floored.fishial_votes["reason"] == "below quality floor"
        assert thin.fishial_votes["reason"] == "insufficient clear frames"
        assert empty.fishial_state in {"review_required", "candidate"}
        # Selection, a quality floor rejection, early stopping and the ceiling all
        # fired, and no track is left in an in-progress state.
        assert not db.scalars(select(LiveFishTrack).where(
            LiveFishTrack.session_id == session.id,
            LiveFishTrack.fishial_state.in_(("candidate", "pending", "ready", "submitted")))).all()


def test_feature_off_stages_nothing_and_constructs_no_client(db_session_factory, enabled_settings, monkeypatch):
    import app.services.live_species as module
    monkeypatch.setattr(module, "FishialClient",
                        lambda *a, **k: pytest.fail("Client constructed while disabled"))
    with db_session_factory() as db:
        for key, (target, enabled) in zip(("coral-city", "smartbay-cam1"),
                                          ((0, True), (1, False))):
            enabled_settings.fishial_enabled = enabled
            session = session_row(db, target=target, started_at=utc_now(), source_key=key)
            track = staged_track(db, session, enabled_settings)
            identifier = SpeciesIdentifier(db, session, enabled_settings)
            assert identifier.should_run() in (False, None, 0)
            identifier.run()
            assert track.fishial_state == "candidate"
            assert session.species_id_api_calls == 0


@pytest.mark.parametrize("value", [None, "bad json", "[]", "null", "42"])
def test_votes_property_defensive(value):
    assert LiveFishTrack(fishial_votes_json=value).fishial_votes == {}


@pytest.mark.parametrize("frame,expected", [
    ({"empty": True, "reason": "no unambiguous fish prediction"}, "classifier returned no candidates"),
    ({"reason": "implausible_for_region"}, "implausible_for_region"),
    ({"reason": "invalid response"}, "invalid response"),
])
def test_api_explains_unreachable_smartbay_tracks_without_rewriting_audit(
        client, db_session_factory, test_settings, frame, expected):
    with db_session_factory() as db:
        session = session_row(db, source_key="smartbay-cam3")
        track = staged_track(db, session, test_settings, state="review_required")
        audit = {"reason": "consensus unreachable", "frames": [frame]}
        track.fishial_votes_json = json.dumps(audit)
        db.commit()
        sid = str(session.id)
        from app.api.live import track_data
        payload = track_data(track)
        assert payload["fishial_diagnostics"]["reason"] == expected
        assert payload["fishial_diagnostics"]["stop_reason"] == "consensus unreachable"
        assert payload["fishial_votes"] == audit
    summary = client.get(f"/live/{sid}/species").json()
    assert summary["review_reasons"] == {expected: 1}
    assert summary["declined"] == int(expected == "classifier returned no candidates")


def test_diagnostics_preserve_mixed_votes_and_explain_selection_limit():
    from app.services.species_quality import review_diagnostics
    diagnostic = review_diagnostics({"reason": "consensus unreachable", "frames": [
        {"voted": True, "species": "Pollachius virens", "score": .51}, {"empty": True}]})
    assert diagnostic["reason"] == "consensus unreachable"
    assert diagnostic["frame_reasons"] == {"classifier returned no candidates": 1}
    assert diagnostic["candidates"] == [{"species": "Pollachius virens", "common_name": None,
        "max_score": .51, "frames": 1, "rejected_for_region": False}]
    assert review_diagnostics({"reason": "budget exhausted"})["budget_scope"] == "fish selection limit"
    assert review_diagnostics({"reason": "budget exhausted", "frames": [
        {"attempts": [{}]}]})["budget_scope"] is None


def test_api_exposes_tentative_matched_species_without_confirming_track(
        client, db_session_factory, test_settings):
    with db_session_factory() as db:
        session = session_row(db, source_key="smartbay-cam3")
        track = staged_track(db, session, test_settings, state="review_required")
        track.fishial_votes_json = json.dumps({"reason": "consensus unreachable", "frames": [
            {"object_index": 0, "dropped_species": ["Lepisosteus oculatus"], "raw": {
                "objects": [{"species": [{"id": "gar", "certainty": .46}]},
                            {"species": [{"id": "other", "certainty": .99}]}],
                "definitions": {"gar": {"scientificName": "Lepisosteus oculatus",
                                        "commonName": "Spotted gar"},
                                "other": {"scientificName": "Other fish"}}}}]})
        db.commit()
        sid = str(session.id)
    payload = client.get(f"/live/{sid}/tracks").json()[0]
    assert payload["fishial_state"] == "review_required"
    assert payload["fishial_species"] is None
    assert payload["fishial_diagnostics"]["candidates"] == [{
        "species": "Lepisosteus oculatus", "common_name": "Spotted gar",
        "max_score": .46, "frames": 1, "rejected_for_region": True}]
    assert client.get(f"/live/{sid}/species").json()["species"] == []


@pytest.mark.parametrize("index", [None, -1, 2, True])
def test_review_never_uses_names_from_unmatched_objects(index):
    from app.services.species_quality import review_diagnostics
    assert review_diagnostics({"frames": [{"object_index": index, "raw": {
        "objects": [{"species": [{"id": "fish", "certainty": .9}]}],
        "definitions": {"fish": {"scientificName": "Other fish"}}}}]})["candidates"] == []


def test_review_empty_target_ignores_named_neighbour_and_combines_repeat_scores():
    from app.services.species_quality import review_diagnostics
    frames = [{"object_index": 0, "empty": True, "raw": {
        "objects": [{"species": []}, {"species": [{"id": "fish", "certainty": .9}]}],
        "definitions": {"fish": {"scientificName": "Other fish"}}}}]
    assert review_diagnostics({"frames": frames})["candidates"] == []
    frames += [{"species": "Pollachius virens", "score": score} for score in [.51, .45]]
    assert review_diagnostics({"frames": frames})["candidates"] == [{
        "species": "Pollachius virens", "common_name": None,
        "max_score": .51, "frames": 2, "rejected_for_region": False}]
