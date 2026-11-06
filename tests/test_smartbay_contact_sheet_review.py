import csv
import json

import cv2
import numpy as np

from scripts import smartbay_contact_sheet_review as review


def test_build_extracts_sheet_tiles_and_applies_proposals(tmp_path):
    sheet_dir = tmp_path / "contact_sheets"
    sheet_dir.mkdir()
    sheet = np.zeros((review.TILE_HEIGHT, review.TILE_WIDTH * 4, 3), np.uint8)
    for column, value in enumerate((20, 60, 100, 140)):
        sheet[:, column * review.TILE_WIDTH : (column + 1) * review.TILE_WIDTH] = value
    assert cv2.imwrite(str(sheet_dir / "SmartBayCam1-train.jpg"), sheet)

    queue = tmp_path / "contact_sheet_review.csv"
    rows = []
    for index, evidence in enumerate(("strong", "weak", "none", "murky")):
        rows.append(
            {
                "window_id": f"window-{index}",
                "camera": "SmartBayCam1",
                "day": "2025-01-01",
                "split": "train",
                "evidence": evidence,
                "contact_sheet": "contact_sheets/SmartBayCam1-train.jpg",
                "decision": "",
                "replacement_reason": "",
                "review_notes": "",
            }
        )
    with queue.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    proposals = tmp_path / "proposals.json"
    proposals.write_text(
        json.dumps(
            {
                "window-1": {
                    "decision": "reject",
                    "replacement_reason": "false_positive",
                    "review_notes": "Only hardware is visible.",
                }
            }
        ),
        encoding="utf-8",
    )

    counts = review.build(queue, tmp_path / "ui", proposals)

    assert counts == {"approve": 3, "reject": 1}
    with (tmp_path / "ui" / "contact_sheet_review_proposed.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reviewed = list(csv.DictReader(stream))
    assert reviewed[1]["replacement_reason"] == "false_positive"
    assert reviewed[2]["decision"] == "approve"
    assert "usable negative/control" in reviewed[2]["review_notes"]
    first = cv2.imread(str(tmp_path / "ui" / "tiles" / "window-0.jpg"))
    fourth = cv2.imread(str(tmp_path / "ui" / "tiles" / "window-3.jpg"))
    assert first.shape[:2] == (review.TILE_HEIGHT, review.TILE_WIDTH)
    assert int(first.mean()) < int(fourth.mean())
    page = (tmp_path / "ui" / "review.html").read_text(encoding="utf-8")
    assert "Export reviewed CSV" in page
    assert "window-3" in page


def test_unknown_proposal_window_is_rejected(tmp_path):
    rows = [
        {
            "window_id": "known",
            "camera": "SmartBayCam1",
            "day": "2025-01-01",
            "split": "train",
            "evidence": "none",
            "contact_sheet": "sheet.jpg",
            "decision": "",
            "replacement_reason": "",
            "review_notes": "",
        }
    ]
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps({"unknown": {"decision": "reject"}}), encoding="utf-8")
    try:
        review.apply_proposals(rows, proposals)
    except ValueError as error:
        assert "unknown window IDs" in str(error)
    else:
        raise AssertionError("unknown proposal should fail")
