from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace as NS

from PIL import Image

from scripts.galway_dataset.common import Store, allowed_license
from scripts.galway_dataset.fathom_source import run as fathom_run
from scripts.galway_dataset.inat_source import run as inat_run
from scripts.galway_dataset.training import create_manifest


def jpeg(size=(150, 120), color="blue") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG")
    return output.getvalue()


class FakeResponse:
    def __init__(self, *, data=None, content=b""):
        self.data, self.content = data, content

    def json(self):
        return self.data


def test_licence_allowlist():
    assert allowed_license("https://creativecommons.org/licenses/by/4.0/", False) == "cc-by"
    assert allowed_license("CC BY-NC 4.0", False) == ""
    assert allowed_license("CC BY-NC 4.0", True) == "cc-by-nc"
    assert allowed_license("CC BY-ND 4.0", True) == ""
    assert allowed_license("Images licensed as CC BY-NC-ND (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode), annotations licensed as CC BY-NC", True) == ""
    assert allowed_license(None, True) == ""


def test_fathom_exact_verified_box_and_crop(monkeypatch, tmp_path: Path):
    from fathomnet.api import images, imagesetuploads

    boxes = [NS(uuid="good", concept="Gadus morhua", reviewState=NS(value="VERIFIED"),
                rejected=False, groupOf=False, x=10, y=10, width=70, height=70),
             NS(uuid="wrong", concept="Gadus", reviewState=NS(value="VERIFIED"),
                rejected=False, groupOf=False, x=0, y=0, width=80, height=80),
             NS(uuid="unverified", concept="Gadus morhua", reviewState=NS(value="UNVERIFIED"),
                rejected=False, groupOf=False, x=0, y=0, width=80, height=80)]
    image = NS(uuid="frame", url="https://example.org/frame.jpg", boundingBoxes=boxes,
               valid=True, depthMeters=20, imagingType="ROV")
    monkeypatch.setattr(images, "find_by_concept", lambda name: [image])
    monkeypatch.setattr(imagesetuploads, "find_by_image_uuid", lambda uuid: [
        NS(darwinCore=NS(license="CC BY 4.0", rightsHolder="Example"))])
    http = NS(get=lambda url: FakeResponse(content=jpeg()))
    root = tmp_path / "dataset"
    info = fathom_run("Gadus morhua", Store(root), http, 10, False)
    assert info["images"] == 1
    with (root / "metadata" / "images.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["source_id"] for row in rows] == ["good"]
    assert (rows[0]["width"], rows[0]["height"], rows[0]["license"]) == ("70", "70", "cc-by")


def test_inat_exact_species_and_observation_group(tmp_path: Path):
    observation = {"id": 100, "taxon": {"id": 7, "name": "Gadus morhua", "rank": "species"},
                   "quality_grade": "research",
                   "photos": [{"id": 11, "url": "https://static.inaturalist.org/photos/11/medium.jpg",
                               "license_code": "cc0"},
                              {"id": 12, "url": "https://static.inaturalist.org/photos/12/medium.jpg",
                               "license_code": "cc0"}]}
    wrong = {**observation, "id": 101, "taxon": {"id": 8, "name": "Gadus", "rank": "genus"}}

    def get(url, params=None):
        if url.endswith("/taxa/autocomplete"):
            return FakeResponse(data={"results": [{"id": 7, "name": "Gadus morhua",
                                                   "rank": "species", "is_active": True}]})
        if url.endswith("/observations"):
            return FakeResponse(data={"total_results": 2, "results": [wrong, observation]})
        return FakeResponse(content=jpeg(color="green" if "/11/" in url else "red"))

    root = tmp_path / "dataset"
    info = inat_run("Gadus morhua", Store(root), NS(get=get), 5, False)
    assert info["observations"] == 2
    with (root / "metadata" / "images.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["source_id"] for r in rows} == {"11", "12"}
    assert {r["group_id"] for r in rows} == {"100"}
    assert create_manifest(root)["total"] == 0  # Unknown-context photos are excluded by default.
    with (root / "metadata" / "context_review.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "context", "underwater_status",
                                               "species_id_status", "include_training", "review_notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image_path": row["image_path"], "context": "natural_underwater",
                             "underwater_status": "yes", "species_id_status": "reviewed",
                             "include_training": "yes", "review_notes": ""})
    report = create_manifest(root)
    assert report["groups"] == 1
    assert len(report["train"]) == 1 and not report["val"]
