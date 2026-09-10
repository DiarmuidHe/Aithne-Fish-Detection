import json

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services.fishial import FishialClient, FishialError


@pytest.fixture
def fishial_settings(test_settings):
    return Settings(**{**test_settings.model_dump(), "fishial_enabled": True,
                       "fishial_client_id": "client-id", "fishial_client_secret": "private-secret"},
                    _env_file=None)


def response_payload():
    return {"ok": True, "objects": [{"species": [
        {"id": "other", "certainty": .1}, {"id": "winner", "certainty": .9}]}],
        "definitions": {"winner": {"scientificName": "Lutjanus griseus"},
                        "other": {"scientificName": "Other fish"}}}


def test_v2_contract_token_cache_expiry_and_timeout(fishial_settings, monkeypatch):
    import app.services.fishial as module
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = []

    def handle(request):
        calls.append(request.url.path)
        assert request.extensions["timeout"]["read"] == 30
        if request.url.path.endswith("/auth"):
            assert json.loads(request.content) == {"client_id": "client-id", "client_secret": "private-secret"}
            return httpx.Response(200, json={"access_token": "private-token"})
        assert request.url.path == "/v2/recognize"
        assert request.headers["Authorization"] == "Bearer private-token"
        assert request.headers["Content-Type"] == "image/jpeg"
        assert request.content == b"private-image"
        return httpx.Response(200, json=response_payload())

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FishialClient(fishial_settings, http)
        prediction = client.identify(b"private-image")
        assert prediction.species == [("Lutjanus griseus", .9), ("Other fish", .1)]
        assert prediction.raw == response_payload()
        client.identify(b"private-image")
        assert calls.count("/v2/auth") == 1
        clock[0] += 600
        client.identify(b"private-image")
        assert calls.count("/v2/auth") == 2
        for secret in ("private-secret", "private-token", "private-image"):
            assert secret not in repr(client)
            assert secret not in repr(fishial_settings)


def test_401_refresh_once_and_every_image_attempt_is_reserved(fishial_settings):
    calls, reservations = [], []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/auth"):
            return httpx.Response(200, json={"access_token": "token"})
        return httpx.Response(401, text="private-secret private-image")

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FishialClient(fishial_settings, http)
        client.before_image_call = reservations.append
        with pytest.raises(FishialError) as exc:
            client.identify(b"private-image")
        assert reservations == [False, True]
        assert calls.count("/v2/auth") == 2
        assert "private" not in str(exc.value)
        assert exc.value.__cause__ is None


@pytest.mark.parametrize("failure", ["http", "json", "schema", "score", "timeout", "auth"])
def test_failures_are_redacted(fishial_settings, failure):
    def handle(request):
        if request.url.path.endswith("/auth"):
            return httpx.Response(401 if failure == "auth" else 200,
                                  json={"access_token": "token", "message": "private-secret"})
        if failure == "timeout":
            raise httpx.ReadTimeout("private-secret private-image", request=request)
        if failure == "http":
            return httpx.Response(500, text="private-secret private-image")
        if failure == "json":
            return httpx.Response(200, text="private-secret private-image")
        if failure == "schema":
            return httpx.Response(200, json={"objects": "private-secret"})
        payload = response_payload()
        payload["objects"][0]["species"][0]["certainty"] = 1.5
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(FishialError) as exc:
            FishialClient(fishial_settings, http).identify(b"private-image")
        assert "private" not in str(exc.value)


def test_401_can_recover(fishial_settings):
    images = []
    def handle(request):
        if request.url.path.endswith("/auth"):
            return httpx.Response(200, json={"access_token": "token"})
        images.append(request.content)
        return httpx.Response(401) if len(images) == 1 else httpx.Response(200, json=response_payload())
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        assert FishialClient(fishial_settings, http).identify(b"crop").species[0][0] == "Lutjanus griseus"
        assert images == [b"crop", b"crop"]


@pytest.mark.parametrize("count", [0, 2])
def test_no_or_multiple_fish_abstains_without_a_target_box(fishial_settings, count):
    payload = response_payload()
    payload["objects"] *= count
    def handle(request):
        return httpx.Response(200, json={"access_token": "token"} if request.url.path.endswith("auth") else payload)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        assert FishialClient(fishial_settings, http).identify(b"crop").species == []


def jpeg(width=200, height=100):
    import cv2
    import numpy as np
    ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def two_object_payload(target_bbox, neighbour_bbox):
    """The target carries the species; the neighbour is segmented but unnamed.

    This is the shape both recovered responses from the SmartBay 3 reference session
    actually had.
    """

    payload = response_payload()
    payload["objects"] = [
        {"bbox": list(target_bbox), "shape": [[1, 2]] * 500,
         "species": [{"id": "winner", "certainty": .9}]},
        {"bbox": list(neighbour_bbox), "shape": [[3, 4]] * 500, "species": []},
    ]
    return payload


@pytest.mark.parametrize("neighbour,expected", [
    ((160, 5, 199, 45), "Lutjanus griseus"),   # IoU ~0.8 vs ~0.03: unambiguous.
    ((14, 12, 186, 88), None),                 # Both overlap the target: abstain.
])
def test_multi_object_resolves_against_the_stored_target_box(fishial_settings, neighbour, expected):
    payload = two_object_payload((10, 8, 190, 92), neighbour)
    def handle(request):
        return httpx.Response(200, json={"access_token": "token"}
                              if request.url.path.endswith("auth") else payload)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FishialClient(fishial_settings, http)
        # Normalised box within a 200x100 crop, matching the target object.
        prediction = client.identify(jpeg(), (.05, .1, .95, .9))
        assert prediction.object_count == 2
        if expected:
            assert prediction.species[0][0] == expected
            assert prediction.object_index == 0
            assert prediction.object_iou > fishial_settings.fishial_object_match_min_iou
        else:
            # Ambiguous: never attribute a neighbour's identity to this track.
            assert prediction.species == []
            assert prediction.object_index is None


def test_single_object_and_missing_box_are_unchanged(fishial_settings):
    def handle(request):
        return httpx.Response(200, json={"access_token": "token"}
                              if request.url.path.endswith("auth") else response_payload())
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FishialClient(fishial_settings, http)
        for box in (None, (.05, .1, .95, .9)):
            prediction = client.identify(jpeg(), box)
            assert prediction.species == [("Lutjanus griseus", .9), ("Other fish", .1)]
            assert prediction.object_index == 0 and prediction.object_count == 1


def test_a_clamped_edge_crop_uses_the_stored_box_not_a_centred_one(fishial_settings):
    """At a frame edge the fish is no longer centred in its crop.

    A recomputed centred box would match the neighbour instead, so the box written at
    staging time is the one that must travel.
    """

    # Crop 200x100; the fish sits hard against the left edge, the neighbour right.
    payload = two_object_payload((0, 8, 90, 92), (110, 8, 199, 92))
    def handle(request):
        return httpx.Response(200, json={"access_token": "token"}
                              if request.url.path.endswith("auth") else payload)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FishialClient(fishial_settings, http)
        assert client.identify(jpeg(), (.0, .08, .45, .92)).species[0][0] == "Lutjanus griseus"
        # A centred box, which is what recomputing from the detection row would give,
        # matches neither object well enough and abstains.
        assert client.identify(jpeg(), (.28, .08, .73, .92)).species == []


def test_unreadable_crop_falls_back_to_abstaining(fishial_settings):
    payload = two_object_payload((10, 8, 190, 92), (160, 5, 199, 45))
    def handle(request):
        return httpx.Response(200, json={"access_token": "token"}
                              if request.url.path.endswith("auth") else payload)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        # Not a JPEG, so the crop size is unknown and the match cannot be trusted.
        assert FishialClient(fishial_settings, http).identify(b"not-a-jpeg", (.05, .1, .95, .9)).species == []


def test_stored_response_is_trimmed_of_shape_polygons(fishial_settings):
    payload = two_object_payload((10, 8, 190, 92), (160, 5, 199, 45))
    def handle(request):
        return httpx.Response(200, json={"access_token": "token"}
                              if request.url.path.endswith("auth") else payload)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        prediction = FishialClient(fishial_settings, http).identify(jpeg(), (.05, .1, .95, .9))
    stored = prediction.raw
    assert all("shape" not in entry for entry in stored["objects"])
    assert [entry["bbox"] for entry in stored["objects"]] == [[10, 8, 190, 92], [160, 5, 199, 45]]
    # Only the definitions a surviving species id references are kept.
    assert set(stored["definitions"]) == {"winner"}
    assert len(json.dumps(stored)) < len(json.dumps(payload)) / 4


def test_jpeg_size_reads_real_headers_and_rejects_junk():
    from app.services.fishial import jpeg_size
    assert jpeg_size(jpeg(320, 240)) == (320, 240)
    assert jpeg_size(b"") is None
    assert jpeg_size(b"\xff\xd8truncated") is None
    assert jpeg_size("not bytes") is None


@pytest.mark.parametrize("override", [
    {"fishial_enabled": True},
    {"fishial_preprocess": "sharpen"},
    {"fishial_late_reserve_fish": 20},
    {"fishial_max_staged_frames_per_candidate": 2},
    {"fishial_quality_weights": {"short_side": -1, "sharpness": 1, "confidence": 1,
                                 "contrast": 1, "colorfulness": 1, "frames": 1}},
    {"fishial_quality_weights": {"short_side": 0, "sharpness": 0, "confidence": 0,
                                 "contrast": 0, "colorfulness": 0, "frames": 0}},
    {"fishial_quality_weights": {"short_side": 1}},
    {"fishial_enabled": True, "fishial_client_id": "id", "fishial_client_secret": " "},
    {"fishial_min_votes": 6}, {"fishial_min_frames_to_vote": 6},
    {"fishial_default_frames_per_fish": 21}, {"fishial_request_timeout_seconds": 0},
])
def test_invalid_config(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{"fishial_client_id": None, "fishial_client_secret": None, **override})
