"""Download, validate, and record source images without losing provenance."""

from __future__ import annotations

import csv
import hashlib
import io
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FIELDS = [
    "image_path", "scientific_name", "source", "source_id", "observation_id",
    "source_url", "image_url", "license", "attribution", "width", "height",
    "cropped", "annotation_type", "annotation_id", "underwater_status",
    "context", "identification_evidence", "group_id", "sha256", "dhash",
    "near_duplicate_of", "status", "notes",
]
REJECT_FIELDS = ["scientific_name", "source", "source_id", "image_url", "reason"]
ALLOWED = {"cc0", "cc-by", "cc-by-sa"}
OPTIONAL_NC = {"cc-by-nc", "cc-by-nc-sa"}


def license_code(value: object) -> str:
    """Normalize only known Creative Commons strings; never guess unknown licences."""
    raw = str(value or "").lower().strip().replace("https://", "").replace("http://", "")
    if "images licensed as" in raw:
        raw = raw.split("images licensed as", 1)[1].split("(", 1)[0].split(",", 1)[0].strip()
    if "creativecommons.org/publicdomain/zero/" in raw or raw in {"cc0", "cc-0", "public domain"}:
        return "cc0"
    if "creativecommons.org/licenses/" in raw:
        raw = "cc-" + raw.split("creativecommons.org/licenses/", 1)[1].split("/", 1)[0]
    raw = raw.replace("_", "-").replace(" ", "-")
    if raw.endswith("-4.0"):
        raw = raw[:-4]
    if raw.endswith("-3.0"):
        raw = raw[:-4]
    return raw if raw in ALLOWED | OPTIONAL_NC | {"cc-by-nd", "cc-by-nc-nd", "cc-by-nc-sa-nd"} else ""


def allowed_license(value: object, include_nc: bool) -> str:
    code = license_code(value)
    return code if code in (ALLOWED | (OPTIONAL_NC if include_nc else set())) else ""


def dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8))
    px = gray.load()
    bits = sum((px[x, y] > px[x + 1, y]) << (y * 8 + x)
               for y in range(8) for x in range(8))
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


class Http:
    def __init__(self, delay: float = 1.0, timeout: float = 30.0):
        self.delay, self.timeout, self.last = delay, timeout, 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "galway-fish-dataset/1.0 (research; contact via repository)"
        retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                      respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get(self, url: str, *, params: dict | None = None) -> requests.Response:
        wait = self.delay - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response


class Store:
    def __init__(self, root: Path, min_side: int = 64, near_distance: int = 5):
        self.root, self.min_side, self.near_distance = root, min_side, near_distance
        self.meta = root / "metadata"
        self.meta.mkdir(parents=True, exist_ok=True)
        self.images_path = self.meta / "images.csv"
        self.rejected_path = self.meta / "rejected_images.csv"
        self.rows: list[dict[str, str]] = []
        if self.images_path.exists():
            with self.images_path.open(newline="", encoding="utf-8") as fh:
                self.rows = list(csv.DictReader(fh))
        stale = False
        for row in self.rows:
            if row["status"] == "accepted" and not (root / row["image_path"]).is_file():
                row["status"] = "missing_file"
                stale = True
        if stale:
            with self.images_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
        self.keys = {(r["source"], r["source_id"]) for r in self.rows
                     if r["status"] == "accepted" and (root / r["image_path"]).is_file()}
        self.hashes = {r["sha256"]: r["image_path"] for r in self.rows
                       if r["status"] == "accepted" and r.get("sha256")
                       and (root / r["image_path"]).is_file()}
        self.visuals = [(r["dhash"], r["image_path"]) for r in self.rows
                        if r["status"] == "accepted" and r.get("dhash")
                        and (root / r["image_path"]).is_file()]

    @staticmethod
    def append(path: Path, fields: list[str], row: dict) -> None:
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def reject(self, row: dict, reason: str) -> None:
        self.append(self.rejected_path, REJECT_FIELDS, {**row, "reason": reason})

    def save(self, row: dict, data: bytes, *, box: tuple[int, int, int, int] | None = None) -> bool:
        key = (str(row["source"]), str(row["source_id"]))
        if key in self.keys:
            return False
        try:
            with Image.open(io.BytesIO(data)) as opened:
                opened.verify()
            with Image.open(io.BytesIO(data)) as opened:
                img = ImageOps.exif_transpose(opened).convert("RGB")
            if box is not None:
                x, y, w, h = box
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img.width or y + h > img.height:
                    raise ValueError("invalid bounding box")
                img = img.crop((x, y, x + w, y + h))
            if min(img.size) < self.min_side:
                raise ValueError(f"image smaller than {self.min_side}px minimum side")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            self.reject(row, str(exc))
            return False
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        digest = hashlib.sha256(output.getvalue()).hexdigest()
        if digest in self.hashes:
            self.reject(row, f"exact duplicate of {self.hashes[digest]}")
            return False
        visual = dhash(img)
        near = next((path for old, path in self.visuals
                     if hamming(visual, old) <= self.near_distance), "")
        species_dir = str(row["scientific_name"]).replace(" ", "_")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(row["source_id"]))
        relative = Path(str(row["source"])) / species_dir / f"{safe_id}.jpg"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".part")
        temporary.write_bytes(output.getvalue())
        temporary.replace(destination)
        record = {**row, "image_path": relative.as_posix(), "width": img.width,
                  "height": img.height, "sha256": digest, "dhash": visual,
                  "near_duplicate_of": near, "status": "accepted"}
        self.append(self.images_path, FIELDS, record)
        self.rows.append(record)
        self.keys.add(key)
        self.hashes[digest] = relative.as_posix()
        self.visuals.append((visual, relative.as_posix()))
        return True

    def report(self, species: list[str], availability: dict[tuple[str, str], dict]) -> None:
        path = self.meta / "species_counts.csv"
        fields = ["scientific_name", "source", "available_observations", "available_images",
                  "count_status", "downloaded_usable", "near_duplicate_flags", "notes"]
        counts = Counter((r["scientific_name"], r["source"]) for r in self.rows
                         if r["status"] == "accepted" and (self.root / r["image_path"]).exists())
        near = Counter((r["scientific_name"], r["source"]) for r in self.rows
                       if r["status"] == "accepted" and r.get("near_duplicate_of"))
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for name in species:
                for source in ("fathomnet", "inaturalist"):
                    status = availability.get((name, source), {})
                    writer.writerow({"scientific_name": name, "source": source,
                                     "available_observations": status.get("observations", ""),
                                     "available_images": status.get("images", ""),
                                     "count_status": status.get("status", "unknown"),
                                     "downloaded_usable": counts[name, source],
                                     "near_duplicate_flags": near[name, source],
                                     "notes": status.get("notes", "")})
        licenses = Counter((r["source"], r["license"]) for r in self.rows if r["status"] == "accepted")
        with (self.meta / "licenses.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "license", "downloaded_usable"])
            for (source, license_name), count in sorted(licenses.items()):
                writer.writerow([source, license_name, count])
