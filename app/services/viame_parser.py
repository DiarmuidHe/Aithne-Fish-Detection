from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class SkippedRow:
    line_number: int
    row: list[str]
    reason: str


@dataclass(frozen=True)
class ClassScore:
    name: str
    confidence: float


@dataclass(frozen=True)
class VIAMEDetection:
    track_id: str
    source: str
    frame_number: int
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    confidence: float
    target_length: float | None
    class_name: str | None
    class_confidence: float | None
    timestamp_seconds: float | None


@dataclass(frozen=True)
class BoundingBoxObservation:
    frame_number: int
    timestamp_seconds: float | None
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str | None
    class_confidence: float | None


@dataclass(frozen=True)
class ParsedFishTrack:
    viame_track_id: str
    first_frame: int
    last_frame: int
    detection_count: int
    max_confidence: float
    mean_confidence: float
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    species: str | None
    species_confidence: float | None
    observations: list[BoundingBoxObservation]


@dataclass(frozen=True)
class ParsedVIAMEResult:
    detections: list[VIAMEDetection]
    tracks: list[ParsedFishTrack]
    skipped_rows: list[SkippedRow]


class VIAMEParseError(ValueError):
    pass


def parse_viame_csv(
    csv_path: Path,
    fps: float | None = None,
    strict: bool = False,
    frame_number_offset: int = 0,
) -> ParsedVIAMEResult:
    detections: list[VIAMEDetection] = []
    skipped_rows: list[SkippedRow] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for line_number, row in enumerate(reader, start=1):
            cells = [cell.strip() for cell in row]
            if not cells or not any(cells) or cells[0].startswith("#"):
                continue
            if _looks_like_header(cells):
                continue
            try:
                detections.append(_parse_detection_row(cells, fps, frame_number_offset))
            except ValueError as exc:
                if strict:
                    raise VIAMEParseError(f"Malformed VIAME CSV row {line_number}: {exc}") from exc
                skipped_rows.append(SkippedRow(line_number=line_number, row=cells, reason=str(exc)))

    return ParsedVIAMEResult(
        detections=detections,
        tracks=group_detections_by_track(detections),
        skipped_rows=skipped_rows,
    )


def group_detections_by_track(detections: list[VIAMEDetection]) -> list[ParsedFishTrack]:
    grouped: dict[str, list[VIAMEDetection]] = {}
    for detection in detections:
        grouped.setdefault(detection.track_id, []).append(detection)

    tracks: list[ParsedFishTrack] = []
    for track_id, track_detections in grouped.items():
        sorted_detections = sorted(track_detections, key=lambda detection: detection.frame_number)
        confidences = [detection.confidence for detection in sorted_detections]
        class_scores = [
            ClassScore(detection.class_name, detection.class_confidence)
            for detection in sorted_detections
            if detection.class_name is not None and detection.class_confidence is not None
        ]
        best_class = max(class_scores, key=lambda score: score.confidence, default=None)

        observations = [
            BoundingBoxObservation(
                frame_number=detection.frame_number,
                timestamp_seconds=detection.timestamp_seconds,
                x1=detection.bbox_left,
                y1=detection.bbox_top,
                x2=detection.bbox_right,
                y2=detection.bbox_bottom,
                confidence=detection.confidence,
                class_name=detection.class_name,
                class_confidence=detection.class_confidence,
            )
            for detection in sorted_detections
        ]

        tracks.append(
            ParsedFishTrack(
                viame_track_id=track_id,
                first_frame=sorted_detections[0].frame_number,
                last_frame=sorted_detections[-1].frame_number,
                detection_count=len(sorted_detections),
                max_confidence=max(confidences),
                mean_confidence=mean(confidences),
                first_timestamp_seconds=sorted_detections[0].timestamp_seconds,
                last_timestamp_seconds=sorted_detections[-1].timestamp_seconds,
                species=best_class.name if best_class else None,
                species_confidence=best_class.confidence if best_class else None,
                observations=observations,
            )
        )

    return sorted(tracks, key=lambda track: (track.first_frame, track.viame_track_id))


def frame_to_timestamp_seconds(frame_number: int, fps: float | None) -> float | None:
    if fps is None or fps <= 0:
        return None
    return frame_number / fps


def _looks_like_header(cells: list[str]) -> bool:
    normalized = cells[0].strip().lower().replace("-", "_")
    return normalized in {"track_id", "target_id"}


def _parse_detection_row(
    cells: list[str], fps: float | None, frame_number_offset: int
) -> VIAMEDetection:
    if len(cells) < 9:
        raise ValueError("expected at least 9 columns")

    track_id = _required_text(cells[0], "track_id")
    source = _required_text(cells[1], "source")
    frame_number = _parse_int(cells[2], "frame_number") + frame_number_offset
    if frame_number < 0:
        raise ValueError("normalized frame_number must not be negative")
    bbox_left = _parse_float(cells[3], "bbox_left")
    bbox_top = _parse_float(cells[4], "bbox_top")
    bbox_right = _parse_float(cells[5], "bbox_right")
    bbox_bottom = _parse_float(cells[6], "bbox_bottom")
    confidence = _parse_float(cells[7], "confidence")
    target_length = _parse_optional_float(cells[8])
    best_class = _best_class_score(cells[9:])

    return VIAMEDetection(
        track_id=track_id,
        source=source,
        frame_number=frame_number,
        bbox_left=bbox_left,
        bbox_top=bbox_top,
        bbox_right=bbox_right,
        bbox_bottom=bbox_bottom,
        confidence=confidence,
        target_length=target_length,
        class_name=best_class.name if best_class else None,
        class_confidence=best_class.confidence if best_class else None,
        timestamp_seconds=frame_to_timestamp_seconds(frame_number, fps),
    )


def _best_class_score(cells: list[str]) -> ClassScore | None:
    scores: list[ClassScore] = []
    for index in range(0, len(cells) - 1, 2):
        name = cells[index].strip()
        confidence_text = cells[index + 1].strip()
        if not name or not confidence_text:
            continue
        try:
            scores.append(ClassScore(name=name, confidence=float(confidence_text)))
        except ValueError:
            continue
    return max(scores, key=lambda score: score.confidence, default=None)


def _required_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "-1", "-1.0", "null", "None"}:
        return None
    return float(value)
