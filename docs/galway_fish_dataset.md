# Galway fish species image downloader

This is a separate classification dataset workflow for the existing detector/tracker. It fetches original photographs only. It does not infer species from genus or family labels, and it does not treat an iNaturalist observation as proof that a fish is visible underwater.

## Verified source access (15 October 2026)

- **FathomNet:** `fathomnet.api.images.find_by_concept(scientific_name)` returns image records with `uuid`, `url`, and `boundingBoxes`. Each box has `concept`, coordinates, review state, and a box UUID. The downloader accepts only a box whose concept exactly equals the binomial. By default it also requires a verified box. It calls `fathomnet.api.imagesetuploads.find_by_image_uuid(image_uuid)` and reads `darwinCore.license` for each linked upload. Images lacking one consistent allowed licence are skipped. The search count is source image records containing eligible boxes, not a claim about downloadable, licensed crops. [Official example](https://www.fathomnet.org/post/how-to-download-images-and-bounding-boxes), [API reference](https://fathomnet-py.readthedocs.io/en/latest/api.html), [DTO reference](https://fathomnet-py.readthedocs.io/en/latest/dto.html).
- **iNaturalist:** The targeted `/v1/taxa/autocomplete` query must resolve to one active taxon whose name and rank exactly match the requested species. `/v1/observations` is queried with `taxon_id`, `quality_grade=research`, and `photos=true`; every result is checked again for the same exact taxon ID/name/rank. Each photo's own `license_code` is checked before downloading. The photo ID and extension produce the official Open Data URL `https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{extension}`. Its documented `original` rendition has a maximum dimension of 2048 pixels, so it is **not necessarily the camera's unscaled original**. The bucket metadata are monthly TSV snapshots at `observations.csv.gz`, `photos.csv.gz`, `taxa.csv.gz`, and `observers.csv.gz`; this downloader uses the live targeted API to avoid scanning hundreds of millions of unrelated records. The API's `total_results` counts matching observations, including descendant taxa, so exact-species licensed *photo* count is shown as unknown until scanned. [Open Data registry](https://registry.opendata.aws/inaturalist-open-data/), [metadata schema](https://github.com/inaturalist/inaturalist-open-data/blob/main/Metadata/README.md), [image layout](https://github.com/inaturalist/inaturalist-open-data/blob/main/README.md), [API practices](https://www.inaturalist.org/pages/api+recommended+practices).

FathomNet permits Database visual content for machine-learning development but the contributor's Creative Commons licence still applies. iNaturalist photographs retain their photographers' copyright. The default allowlist is CC0, CC BY, and CC BY-SA; `--include-nc` adds CC BY-NC and CC BY-NC-SA for noncommercial work. No-derivatives, missing, or unrecognized licences are rejected because crops/augmentation and redistribution may be involved. Keep attribution and original links when sharing data or results. Review the exact licence terms for your use. [FathomNet terms](https://www.fathomnet.org/terms), [iNaturalist licensing help](https://help.inaturalist.org/en/support/solutions/articles/151000173511).

## Install and run

From the repository root on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\galway_fish_dataset.py --species "Gadus morhua" --species "Labrus bergylta" --limit 2 --max-observations 200
```

Search without image downloads:

```powershell
.venv\Scripts\python.exe scripts\galway_fish_dataset.py --species "Gadus morhua" --dry-run
```

Larger resumable collection across the enabled list:

```powershell
.venv\Scripts\python.exe scripts\galway_fish_dataset.py --limit 500 --max-observations 20000 --delay 1.5
```

`--limit` is the **total accepted images per source and species**, including earlier runs. Repeat the command to resume; source IDs already present in `images.csv` are skipped. `--sources inaturalist` or `--sources fathomnet` narrows a run. Edit `configs/galway_species.yaml` to add exact scientific names or disable candidates. `--allow-unverified-fathomnet` relaxes box review status and should be used only for a manual review queue. `--image-size large` requests the 1024-pixel Open Data rendition when bandwidth matters. The program retries 429/5xx errors, obeys Retry-After, and throttles its HTTP requests. The FathomNet client controls its own API calls; use conservative runs and respect service limits.

Output goes under `data/galway_fish_dataset/`, with `fathomnet/<Genus_species>/` and `inaturalist/<Genus_species>/` image directories. `metadata/images.csv` records accepted species, source/photo/box IDs, URLs, licence, attribution, dimensions, crop type, observation/frame group, hashes, context and status. `metadata/rejected_images.csv` records failed candidates. `metadata/species_counts.csv` distinguishes `0` from blank/unknown counts; `metadata/licenses.csv` summarises accepted licences. Re-run the downloader to refresh these reports. The copied config is under `configs/species.yaml` in the dataset.

To print counts based on images actually present on disk, run `.venv\Scripts\python.exe scripts\galway_dataset_report.py`.

## Underwater and label review

FathomNet records with a depth are marked underwater verified; other FathomNet frames are only probable. iNaturalist context starts **unknown**, because research grade establishes a community identification, not image setting or whether the photographed fish is living. Do not mistake deck, market, aquarium, or museum images for SmartBay camera images. The dataset retains these as supplemental examples until reviewed.

```powershell
.venv\Scripts\python.exe scripts\galway_context_review.py
```

Edit `metadata/context_review.csv`: `context` can be `natural_underwater`, `aquarium`, `fishing_deck`, `museum`, `out_of_water`, or `unknown`; set `underwater_status` to `yes`, `no`, or `unknown`. New rows start with `include_training=no`. Set it to `yes` only after checking the image, including whether the fish is large enough and the species label is plausible; use `species_id_status=needs_review` when diagnostic features are unclear. The review script preserves existing entries when new images arrive. Never crop iNaturalist images automatically without a trusted fish localization; full photographs stay whole. FathomNet crops derive from its supplied boxes. Visually similar images are flagged in `near_duplicate_of`; inspect those before training. Exact image duplicates are rejected automatically. Use `scripts/galway_review_contact_sheet.py` to make a sample grid for triage.

## PyTorch training manifest

```powershell
.venv\Scripts\python.exe scripts\galway_training_manifest.py
```

This writes `metadata/training_manifest.csv`, `species_to_index.json`, and `training_stats.json`. By default it includes **only images explicitly reviewed with `include_training=yes` and `underwater_status=yes`**. `--all-contexts` is available for a separate supplemental dataset of reviewed out-of-water photos. The manifest keeps all images of one iNaturalist observation or one FathomNet source URL directory in one split, and joins groups that have flagged near duplicates. The FathomNet directory is a conservative recording proxy because the search DTO has no definitive recording ID. When there are too few independent groups, a species may have no validation examples; collect more groups before evaluating that class. The report gives per-class train/validation counts and near-duplicate flags.

Install a PyTorch/torchvision build appropriate for your machine, then:

```python
from pathlib import Path
from scripts.galway_dataset.training import FishSpeciesDataset, classification_transforms

root = Path("data/galway_fish_dataset")
train = FishSpeciesDataset(root, "train", classification_transforms(train=True))
val = FishSpeciesDataset(root, "val", classification_transforms(train=False))
image, class_index = train[0]
```

The train transforms use modest crop, horizontal flip, and colour jitter. They leave dark/turbid examples in the dataset. Use class weights or a balanced sampler if `training_stats.json` shows severe imbalance, and inspect each class's source mix.

## Regional scope and limitations

The species list is a **candidate list for Irish coastal/Northeast Atlantic training**, not a list of confirmed SmartBay sightings. The [Marine Institute's species resources](https://www.marine.ie/site-area/areas-activity/fisheries-ecosystems/fisheries-overview?language=en) cover Irish waters; the [SmartBay site](https://smartbay.marine.ie/) describes a camera platform at roughly 25 m depth. Rock cook is documented among common Irish wrasses by the [Marine Institute](https://www.marine.ie/sites/default/files/MIFiles/Docs/Aquaculture/IFB%20No.%2053%20%20National%20survey%20of%20Sea%20lice%20in%20Fish%20farms%202021.pdf), and [MarLIN](https://www.marlin.ac.uk/species/detail/1670) documents cuckoo wrasse around Britain and Ireland. Those regional sources do not verify presence at the SmartBay camera. Compare local footage and georeferenced occurrence records before assigning site-specific priors.

Run `.venv\Scripts\python.exe scripts\galway_occurrence_report.py` to create `metadata/regional_occurrence.csv` from research-grade iNaturalist observation counts inside broad Ireland and Galway Bay bounding boxes. These are community-reported location counts (including descendant taxa and out-of-water photos), not verified camera sightings or population estimates. A zero is a verified search result, not proof of absence. This query covers iNaturalist records only; reconcile important classes with Marine Institute surveys and the SmartBay footage.

Research grade can still be wrong, particularly for similar gobies, juvenile wrasses and flatfish. FathomNet is heavily drawn from other regions/depths. Neither source supplies a dependable universal underwater/deck label, and no automated visual filter should discard difficult but useful SmartBay-like footage. The near-duplicate hash is a review aid, not a guarantee against leakage. The API may return records that disappear from the S3 snapshot; failures are logged for later retry.

## Troubleshooting

- `count_status=unknown` with a search error: check DNS/network access, 429 responses, and the service status. It is never converted to zero.
- Confirmed zero FathomNet boxes: retry a dry run with `--allow-unverified-fathomnet` to distinguish absent exact concepts from merely unreviewed boxes. Keep those crops out of final training until reviewed.
- Missing iNaturalist species: check exact active taxon spelling; aliases and genus-level names are deliberately excluded.
- S3 404: Open Data snapshots can lag the API or contain removed photos. The failure appears in `rejected_images.csv`; rerun later or choose `--image-size large`.
- Too few usable images: inspect `rejected_images.csv` for licence, small-image and box errors; expand `--max-observations` gradually. Do not enable noncommercial images for commercial work.
- Training split lacks validation for a species: obtain more independent observations. Multiple photos from one observation are one group.
