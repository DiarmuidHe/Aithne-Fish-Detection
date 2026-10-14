"""Simple class-aware one-to-one detection counts at a fixed IoU threshold."""
from __future__ import annotations


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0.0


def count_matches(truth: list[tuple[int, tuple]], predicted: list[tuple[int, tuple, float]],
                  threshold: float = .5) -> dict:
    matched = set()
    true_positive = false_positive = 0
    for cls, box, confidence in sorted(predicted, key=lambda row: row[2], reverse=True):
        candidates = [(iou(box, target), index) for index, (target_cls, target) in enumerate(truth)
                      if target_cls == cls and index not in matched]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= threshold:
            matched.add(best[1])
            true_positive += 1
        else:
            false_positive += 1
    return {"tp": true_positive, "fp": false_positive, "fn": len(truth) - len(matched)}
