"""Exact-species FathomNet localization downloader."""

from __future__ import annotations

from urllib.parse import urlparse

from .common import Http, Store, allowed_license


def _verified(box: object) -> bool:
    state = getattr(box, "reviewState", None)
    return bool(getattr(box, "verified", False) or getattr(state, "value", state) == "VERIFIED")


def _boxes(image: object, species: str, verified_only: bool) -> list:
    return [box for box in (getattr(image, "boundingBoxes", None) or [])
            if getattr(box, "concept", None) == species
            and not getattr(box, "rejected", False)
            and getattr(getattr(box, "reviewState", None), "value", None) != "REJECTED"
            and not getattr(box, "groupOf", False)
            and (not verified_only or _verified(box))]


def run(species: str, store: Store, http: Http, limit: int, include_nc: bool,
        *, dry_run: bool = False, verified_only: bool = True) -> dict:
    # These functions and DTO fields are documented by fathomnet-py 1.10.
    from fathomnet.api import images, imagesetuploads

    records = images.find_by_concept(species)
    candidates = [(im, box) for im in records for box in _boxes(im, species, verified_only)]
    info = {"images": len({getattr(im, "uuid", None) for im, _ in candidates}),
            "status": "verified", "notes": f"{len(candidates)} eligible exact-name boxes; licence not included in search count"}
    print(f"FathomNet {species}: {info['images']} source images / {len(candidates)} boxes")
    if dry_run:
        return info
    saved = 0
    for image, box in candidates:
        if saved >= limit:
            break
        image_uuid = getattr(image, "uuid", None)
        box_uuid = getattr(box, "uuid", None)
        url = getattr(image, "url", None)
        if not image_uuid or not box_uuid or not url:
            store.reject({"scientific_name": species, "source": "fathomnet",
                          "source_id": str(box_uuid or ""), "image_url": str(url or "")},
                         "missing image UUID, box UUID, or URL")
            continue
        row = {"scientific_name": species, "source": "fathomnet", "source_id": box_uuid,
               "observation_id": image_uuid,
               "group_id": urlparse(url).path.rsplit("/", 1)[0] or image_uuid,
               "source_url": url,
               "image_url": url, "cropped": "true", "annotation_type": "bounding_box",
               "annotation_id": box_uuid, "underwater_status": "verified" if getattr(image, "depthMeters", None) is not None else "probable",
               "context": str(getattr(image, "imagingType", "") or ""),
               "identification_evidence": "verified_bbox" if _verified(box) else "unverified_bbox",
               "notes": f"depth_m={getattr(image, 'depthMeters', '')}; bbox={getattr(box, 'x', '')},{getattr(box, 'y', '')},{getattr(box, 'width', '')},{getattr(box, 'height', '')}"}
        if ("fathomnet", str(box_uuid)) in store.keys:
            continue
        if getattr(image, "valid", True) is False:
            store.reject(row, "FathomNet image marked invalid")
            continue
        try:
            uploads = imagesetuploads.find_by_image_uuid(image_uuid)
        except Exception as exc:
            store.reject(row, f"licence lookup failed: {exc}")
            continue
        licences = {allowed_license(getattr(getattr(item, "darwinCore", None), "license", None), include_nc)
                    for item in uploads}
        licences.discard("")
        if len(licences) != 1 or any(
            not allowed_license(getattr(getattr(item, "darwinCore", None), "license", None), include_nc)
            for item in uploads
        ):
            store.reject(row, "licence absent, conflicting, or outside allowlist")
            continue
        row["license"] = licences.pop()
        row["attribution"] = "; ".join(sorted({str(getattr(getattr(item, "darwinCore", None), "rightsHolder", "") or "")
                                                 for item in uploads} - {""}))
        try:
            data = http.get(url).content
            box_tuple = tuple(int(getattr(box, field)) for field in ("x", "y", "width", "height"))
            if store.save(row, data, box=box_tuple):
                saved += 1
                print(f"  saved {saved}/{limit}: {box_uuid}")
        except Exception as exc:
            store.reject(row, f"download/crop error: {exc}")
    return info
