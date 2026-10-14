from types import SimpleNamespace

from app.services.cvat_sync import planned_cvat_shapes, typical_frame_step
from app.services.viame_parser import ParsedVIAMEResult, VIAMEDetection, group_detections_by_track


def detection(track_id: str, frame: int) -> VIAMEDetection:
    return VIAMEDetection(track_id, "video", frame, 10, 20, 30, 40,
                          .8, None, "fish", .8, None)


def test_import_ends_visibility_through_long_detection_gaps() -> None:
    detections = [detection("one", frame) for frame in (0, 3, 6, 15, 18)]
    parsed = ParsedVIAMEResult(detections, group_detections_by_track(detections), [])
    assert typical_frame_step(parsed) == 3
    planned = planned_cvat_shapes(parsed.tracks[0], 100, 100, 30, 3)
    assert [(frame, outside) for frame, _, outside, _ in planned] == [
        (0, False), (3, False), (6, False), (9, True),
        (15, False), (18, False), (19, True)]


def test_single_observation_ends_immediately() -> None:
    track = SimpleNamespace(observations=[SimpleNamespace(
        frame_number=4, x1=10, y1=20, x2=30, y2=40, confidence=.7)])
    assert [(frame, outside) for frame, _, outside, _ in
            planned_cvat_shapes(track, 100, 100, 10, 1)] == [(4, False), (5, True)]


def test_default_import_keeps_original_interpolation_behavior() -> None:
    detections = [detection("one", frame) for frame in (0, 3, 12)]
    track = group_detections_by_track(detections)[0]
    assert [(frame, outside) for frame, _, outside, _ in
            planned_cvat_shapes(track, 100, 100, 20, None)] == [
                (0, False), (3, False), (12, False), (13, True)]
