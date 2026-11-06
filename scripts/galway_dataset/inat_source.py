"""Targeted iNaturalist search with licensed Open Data S3 image delivery."""

from __future__ import annotations

from urllib.parse import urlparse

from .common import Http, Store, allowed_license

API = "https://api.inaturalist.org/v1"
S3 = "https://inaturalist-open-data.s3.amazonaws.com/photos"


def resolve_taxon(http: Http, species: str) -> int | None:
    result = http.get(f"{API}/taxa/autocomplete", params={"q": species, "per_page": 200}).json()
    matches = [taxon for taxon in result.get("results", [])
               if taxon.get("name") == species and taxon.get("rank") == "species"
               and taxon.get("is_active", taxon.get("active", True))]
    if len(matches) > 1:
        raise ValueError(f"multiple active exact species taxa for {species}")
    return int(matches[0]["id"]) if matches else None


def _photo_url(photo: dict, size: str) -> str | None:
    photo_id = photo.get("id")
    original_url = photo.get("url")
    if not isinstance(photo_id, int) or not isinstance(original_url, str):
        return None
    extension = urlparse(original_url).path.rsplit(".", 1)[-1].lower()
    if extension not in {"jpg", "jpeg", "png"}:
        return None
    return f"{S3}/{photo_id}/{size}.{extension}"


def run(species: str, store: Store, http: Http, limit: int, include_nc: bool,
        *, dry_run: bool = False, image_size: str = "original",
        max_observations: int = 10000) -> dict:
    taxon_id = resolve_taxon(http, species)
    if taxon_id is None:
        print(f"iNaturalist {species}: exact active species taxon absent")
        return {"observations": 0, "images": 0, "status": "verified",
                "notes": "no exact active species taxon found"}
    params = {"taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
              "per_page": 200, "order_by": "id", "order": "asc"}
    first = http.get(f"{API}/observations", params=params).json()
    total = first.get("total_results")
    info = {"observations": total if isinstance(total, int) else "",
            "images": 0 if total == 0 else "", "status": "verified" if isinstance(total, int) else "unknown",
            "notes": "observation count includes descendant taxa; exact-species licensed photo count unknown"}
    print(f"iNaturalist {species}: {total if total is not None else 'unknown'} matching observations; exact licensed photo count unknown")
    if dry_run or total == 0:
        return info
    seen_obs = set()
    page = first
    scanned = saved = 0
    while scanned < max_observations and saved < limit:
        observations = page.get("results", [])
        if not observations:
            break
        for observation in observations:
            obs_id = observation.get("id")
            if obs_id in seen_obs:
                continue
            seen_obs.add(obs_id)
            scanned += 1
            taxon = observation.get("taxon") or {}
            if taxon.get("id") != taxon_id or taxon.get("name") != species or taxon.get("rank") != "species":
                continue
            if observation.get("quality_grade") != "research":
                continue
            for photo in observation.get("photos") or []:
                if saved >= limit:
                    break
                photo_id = photo.get("id")
                url = _photo_url(photo, image_size)
                row = {"scientific_name": species, "source": "inaturalist",
                       "source_id": str(photo_id or ""), "observation_id": str(obs_id),
                       "group_id": str(obs_id),
                       "source_url": f"https://www.inaturalist.org/observations/{obs_id}",
                       "image_url": url or "", "cropped": "false",
                       "annotation_type": "observation_taxon", "annotation_id": str(taxon_id),
                       "underwater_status": "unknown", "context": "unknown",
                       "identification_evidence": "research_grade_exact_species",
                       "attribution": photo.get("attribution") or "",
                       "notes": "iNaturalist photos have no individual-fish box; inspect for multiple taxa/context"}
                if not photo_id or not url:
                    store.reject(row, "missing photo ID or supported extension")
                    continue
                if ("inaturalist", str(photo_id)) in store.keys:
                    continue
                licence = allowed_license(photo.get("license_code"), include_nc)
                if not licence:
                    store.reject(row, "photo licence absent or outside allowlist")
                    continue
                row["license"] = licence
                try:
                    if store.save(row, http.get(url).content):
                        saved += 1
                        print(f"  saved {saved}/{limit}: photo {photo_id}")
                except Exception as exc:
                    store.reject(row, f"S3 download error: {exc}")
            if scanned >= max_observations:
                break
        last_id = observations[-1].get("id")
        if not isinstance(last_id, int) or len(observations) < 200:
            break
        page = http.get(f"{API}/observations", params={**params, "id_above": last_id}).json()
    info["notes"] += f"; scanned {scanned} observations this run"
    return info
