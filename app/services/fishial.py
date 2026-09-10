"""Source-independent Fishial v2 adapter.

Contract: https://docs.fishial.ai/api/api_reference (verified 2026-09-10).
V2 accepts a JPEG body directly; the deprecated v1 signed-upload flow is unnecessary.
"""
from __future__ import annotations

import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from app.config import Settings


class FishialError(Exception):
    """Only fixed local reason codes may escape the adapter, never server text."""

    def __init__(self, reason="request failed"):
        self.reason = reason if reason in {
            "request failed", "invalid response", "budget exhausted", "retry limit reached",
        } else "request failed"
        super().__init__(f"Fishial: {self.reason}")


@dataclass
class FishialPrediction:
    species: list[tuple[str, float]]
    raw: dict = field(default_factory=dict, repr=False)
    # Which returned object we attributed to this track, and how well it matched.
    object_index: int | None = None
    object_iou: float | None = None
    object_count: int = 0


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap = (max(0.0, min(ax2, bx2) - max(ax1, bx1))
               * max(0.0, min(ay2, by2) - max(ay1, by1)))
    union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
             + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - overlap)
    return overlap / union if union > 0 else 0.0


def jpeg_size(image: bytes) -> tuple[int, int] | None:
    """Width and height from a JPEG's SOF marker, or ``None`` if unreadable.

    Returned object boxes are in crop pixels while the caller knows its target only
    as a normalised box, so the two must be reconciled somewhere. Reading the size
    from the bytes we are already sending keeps this adapter free of an image
    dependency and free of any caller-supplied dimension it would have to trust.
    """

    if not isinstance(image, (bytes, bytearray)) or bytes(image[:2]) != b"\xff\xd8":
        return None
    offset, end = 2, len(image)
    while offset + 4 <= end:
        if image[offset] != 0xFF:
            return None
        marker = image[offset + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        (length,) = struct.unpack_from(">H", image, offset + 2)
        if length < 2:
            return None
        # SOF0..SOF15, minus the DHT/JPG/DAC markers interleaved in that range.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if offset + 9 > end:
                return None
            height, width = struct.unpack_from(">HH", image, offset + 5)
            return (width, height) if width and height else None
        offset += 2 + length
    return None


def trim_raw(raw: dict) -> dict:
    """Drop the ``shape`` polygons we never read, keeping everything else.

    ``shape`` outlines are ~90% of a stored response (measured over the reference
    session: 20.1 KB -> 2.2 KB per called track) and nothing downstream reads them -
    not voting, not consensus, not the multi-object matcher, not the replay harness.
    ``definitions`` is narrowed to the entries a surviving ``species`` id references.
    """

    if not isinstance(raw, dict):
        return {}
    objects = raw.get("objects")
    if not isinstance(objects, list):
        return dict(raw)
    used, trimmed = set(), []
    for entry in objects:
        if not isinstance(entry, dict):
            trimmed.append(entry)
            continue
        used.update(candidate.get("id") for candidate in entry.get("species") or []
                    if isinstance(candidate, dict))
        trimmed.append({key: value for key, value in entry.items() if key != "shape"})
    slim = {**raw, "objects": trimmed}
    definitions = raw.get("definitions")
    if isinstance(definitions, dict):
        slim["definitions"] = {key: value for key, value in definitions.items() if key in used}
    return slim


class FishialClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self._settings = settings
        self._client = client or httpx.Client(follow_redirects=False)
        self._owns_client = client is None
        self._token = None
        self._expires_at = 0.0
        # The live caller installs a durable reservation BEFORE every actual image
        # request, including the hidden 401 retry. Auth calls carry no images.
        self.before_image_call: Callable[[bool], None] | None = None

    def close(self):
        if self._owns_client:
            self._client.close()

    def _post(self, endpoint, **kwargs):
        return self._client.post(
            self._settings.fishial_api_base_url.rstrip("/") + endpoint,
            timeout=self._settings.fishial_request_timeout_seconds,
            follow_redirects=False, **kwargs,
        )

    def _authenticate(self):
        response = self._post("/auth", json={
            "client_id": self._settings.fishial_client_id,
            "client_secret": self._settings.fishial_client_secret.get_secret_value(),
        })
        response.raise_for_status()
        payload = response.json()
        token = payload["access_token"]
        if not isinstance(token, str) or not token.strip():
            raise FishialError("invalid response")
        self._token = token
        # Documented TTL is 600 seconds. Refresh slightly early; never decode JWTs.
        self._expires_at = time.monotonic() + 590

    def identify(self, image: bytes, expected_box=None) -> FishialPrediction:
        """``expected_box`` is the target's normalised ``(x1, y1, x2, y2)`` in the crop.

        Read the answer we already bought: when a response carries several objects,
        resolve it against the box we deliberately centred the crop on instead of
        discarding the whole response. An ambiguous match still abstains - this must
        never become a way to attribute a neighbour's identity to this track.
        """

        try:
            target = self._pixel_box(image, expected_box)
            if not self._token or time.monotonic() >= self._expires_at:
                self._authenticate()
            refreshed = False
            for attempt in range(self._settings.fishial_max_api_retries + 1):
                if self.before_image_call:
                    self.before_image_call(attempt > 0)
                try:
                    response = self._post("/recognize", content=image, headers={
                        "Authorization": f"Bearer {self._token}", "Content-Type": "image/jpeg",
                    })
                except httpx.TransportError:
                    if attempt < self._settings.fishial_max_api_retries:
                        continue
                    raise
                if response.status_code == 401 and not refreshed:
                    refreshed = True
                    self._authenticate()
                    continue
                if (response.status_code == 429 or response.status_code >= 500
                        ) and attempt < self._settings.fishial_max_api_retries:
                    continue
                response.raise_for_status()
                return self._parse(response.json(), target, self._settings)
            raise FishialError()
        except FishialError:
            raise
        except Exception:  # noqa: BLE001 - redact all transport/parser exception text
            # Do not propagate httpx exceptions: URLs, headers, and server messages
            # can contain secrets or echoed bytes. Also suppress exception chaining.
            raise FishialError() from None

    @staticmethod
    def _pixel_box(image, expected_box):
        """Normalised expected box -> crop pixel coordinates, or ``None`` to abstain."""

        if expected_box is None:
            return None
        size = jpeg_size(image)
        if size is None:
            # An unreadable crop cannot be matched; fall back to abstaining.
            return None
        width, height = size
        try:
            x1, y1, x2, y2 = (float(value) for value in expected_box)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None
        return (x1 * width, y1 * height, x2 * width, y2 * height)

    @staticmethod
    def _match(objects, target, settings):
        """``(index, iou)`` of the object that is our fish, or ``None`` to abstain."""

        if len(objects) == 1:
            return 0, 1.0
        # A clean crop should contain one fish. With no target box, or with a match
        # too weak or too close to its runner-up, abstain rather than pick.
        if not objects or target is None or settings is None:
            return None
        scored = []
        for index, entry in enumerate(objects):
            box = entry.get("bbox") if isinstance(entry, dict) else None
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise FishialError("invalid response")
            values = [float(value) for value in box]
            if not all(math.isfinite(value) for value in values):
                raise FishialError("invalid response")
            scored.append((_iou(target, values), index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        (best, index), (runner_up, _) = scored[0], scored[1]
        if (best >= settings.fishial_object_match_min_iou
                and best - runner_up >= settings.fishial_object_match_min_margin):
            return index, best
        return None

    @staticmethod
    def _parse(raw, target=None, settings=None):
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            raise FishialError("invalid response")
        objects, definitions = raw.get("objects"), raw.get("definitions")
        if not isinstance(objects, list) or not isinstance(definitions, dict):
            raise FishialError("invalid response")
        slim = trim_raw(raw)
        matched = FishialClient._match(objects, target, settings)
        if matched is None:
            return FishialPrediction([], slim, None, None, len(objects))
        index, iou = matched
        ranked = {}
        for candidate in objects[index]["species"]:
            name = definitions[candidate["id"]]["scientificName"]
            score = candidate["certainty"]
            if (not isinstance(name, str) or not name.strip() or len(name) > 256
                    or isinstance(score, bool) or not isinstance(score, (float, int))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise FishialError("invalid response")
            ranked[name.strip()] = max(score, ranked.get(name.strip(), 0))
        return FishialPrediction(sorted(ranked.items(), key=lambda item: item[1], reverse=True),
                                 slim, index, iou, len(objects))
