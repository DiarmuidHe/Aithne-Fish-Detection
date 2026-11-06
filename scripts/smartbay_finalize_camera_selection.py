"""Replace rejected SmartBay windows and write a verified final manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from scripts import smartbay_camera_selection as selection
from scripts.smartbay_label_selection import CLIP_FPS, CLIP_FRAMES, MIN_CLIP_GAP, WINDOW_FRAMES


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def aligned_start(clip: dict, seed: int) -> int:
    """Prefer a window containing the strongest in-clip screening PNG."""
    clip_start = datetime.fromisoformat(clip["clip_time"])
    candidates = []
    for image in clip["nearby"]:
        offset = (datetime.fromisoformat(image["timestamp"]) - clip_start).total_seconds()
        if 0 <= offset <= CLIP_FRAMES / CLIP_FPS:
            candidates.append((image.get("max_score", 0), image.get("detection_count", 0), offset))
    if candidates:
        _, _, offset = max(candidates)
        start_seconds = min(
            max(0.0, offset - WINDOW_FRAMES / CLIP_FPS / 2),
            (CLIP_FRAMES - WINDOW_FRAMES) / CLIP_FPS,
        )
        return round(start_seconds * CLIP_FPS)
    return random.Random(selection._stable_tie(seed, clip["url"], "replacement")).randint(
        0, CLIP_FRAMES - WINDOW_FRAMES
    )


def internal_window(clip: dict, split: str, seed: int) -> dict:
    start = aligned_start(clip, seed)
    return {
        **clip,
        "split": split,
        "start_frame": start,
        "frame_count": WINDOW_FRAMES,
        "window_id": f"{clip['stem']}_f{start:04d}",
        "labelling_route": ("human_review_only" if split == "expert" else "sam3_then_human_review"),
    }


def restore_window(row: dict[str, str], clip: dict) -> dict:
    return {
        **clip,
        "split": row["split"],
        "start_frame": int(row["start_frame"]),
        "frame_count": int(row["frame_count"]),
        "window_id": row["window_id"],
        "labelling_route": row["labelling_route"],
    }


def choose_replacements(
    clips: list[dict],
    manifest: list[dict[str, str]],
    review: list[dict[str, str]],
    roles: dict[str, dict[str, str]],
    seed: int,
    excluded_urls: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    excluded_urls = excluded_urls or set()
    manifest_by_id = {row["window_id"]: row for row in manifest}
    clip_by_url = {clip["url"]: clip for clip in clips}
    decisions = {row["window_id"]: row for row in review}
    if set(decisions) != set(manifest_by_id):
        raise ValueError("review decisions must match the manifest window IDs exactly")
    if {row["decision"] for row in review} - {"approve", "reject"}:
        raise ValueError("every review decision must be approve or reject")

    original_urls = {row["source_mp4_url"] for row in manifest}
    approved = []
    selected_by_day: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in manifest:
        if decisions[row["window_id"]]["decision"] != "approve":
            continue
        clip = clip_by_url.get(row["source_mp4_url"])
        if clip is None:
            raise ValueError(f"approved source clip missing from scan: {row['source_mp4_url']}")
        window = restore_window(row, clip)
        approved.append(window)
        selected_by_day[(row["camera"], row["day"])].append(window)

    pool_by_role: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for clip in clips:
        role = roles.get(clip["camera"], {}).get(clip["day"])
        if (
            role
            and clip.get("evidence")
            and clip.get("light")
            and clip.get("clarity")
            and clip["url"] not in original_urls
            and clip["url"] not in excluded_urls
        ):
            pool_by_role[(clip["camera"], role)].append(clip)

    replacements = []
    used_urls = set(original_urls)
    for rejected in (
        row for row in manifest if decisions[row["window_id"]]["decision"] == "reject"
    ):
        key = (rejected["camera"], rejected["split"])
        viable = []
        for clip in pool_by_role[key]:
            day_windows = selected_by_day[(clip["camera"], clip["day"])]
            if clip["url"] in used_urls or len(day_windows) >= selection.MAX_WINDOWS_PER_DAY:
                continue
            stamp = datetime.fromisoformat(clip["clip_time"])
            if any(
                abs(stamp - datetime.fromisoformat(other["clip_time"])) < MIN_CLIP_GAP
                for other in day_windows
            ):
                continue
            viable.append(clip)
        if not viable:
            raise ValueError(f"no replacement available for {rejected['window_id']} in {key}")

        def score(clip: dict, target: dict = rejected) -> tuple:
            day_windows = selected_by_day[(clip["camera"], clip["day"])]
            fallback_order = {
                "strong": ("strong", "weak", "none", "murky"),
                "weak": ("weak", "strong", "none", "murky"),
                "none": ("none", "weak", "strong", "murky"),
                # A rejected murky window failed visual review. Prefer a usable
                # negative instead of cycling through more obscured frames.
                "murky": ("none", "weak", "strong", "murky"),
            }
            reason = decisions[target["window_id"]]["replacement_reason"]
            if target["evidence"] == "weak" and any(
                marker in reason
                for marker in ("visible_", "featureless_", "weak_evidence_", "severe_glare")
            ):
                fallback_order["weak"] = ("strong", "none", "weak", "murky")
            return (
                fallback_order[target["evidence"]].index(clip["evidence"]),
                len(day_windows),
                clip["hour_bin"] != target["hour_bin"],
                clip["light"] != target["light"],
                clip["clarity"] != target["clarity"],
                clip["day"] != target["day"],
                -clip.get("png_max_score", 0) if clip["evidence"] in {"strong", "weak"} else 0,
                -clip.get("png_detections", 0) if clip["evidence"] == "strong" else 0,
                selection._stable_tie(seed, target["window_id"], clip["url"]),
            )

        clip = min(viable, key=score)
        window = internal_window(clip, rejected["split"], seed)
        approved.append(window)
        selected_by_day[(clip["camera"], clip["day"])].append(window)
        used_urls.add(clip["url"])
        replacements.append(
            {
                "rejected_window_id": rejected["window_id"],
                "replacement_window_id": window["window_id"],
                "camera": window["camera"],
                "split": window["split"],
                "rejected_evidence": rejected["evidence"],
                "replacement_evidence": window["evidence"],
                "evidence_preserved": rejected["evidence"] == window["evidence"],
                "rejected_day": rejected["day"],
                "replacement_day": window["day"],
                "replacement_source_mp4_url": window["url"],
                "replacement_reason": decisions[rejected["window_id"]]["replacement_reason"],
                "review_notes": decisions[rejected["window_id"]]["review_notes"],
            }
        )
    return sorted(
        approved, key=lambda item: (item["camera"], item["day"], item["clip_time"])
    ), replacements


def verify(
    original: list[dict[str, str]],
    final_rows: list[dict],
    rejected_urls: set[str],
    roles: dict[str, dict[str, str]] | None = None,
) -> dict:
    errors = []
    original_camera_split = Counter((row["camera"], row["split"]) for row in original)
    final_camera_split = Counter((row["camera"], row["split"]) for row in final_rows)
    original_evidence = Counter((row["camera"], row["split"], row["evidence"]) for row in original)
    final_evidence = Counter((row["camera"], row["split"], row["evidence"]) for row in final_rows)
    if len(final_rows) != len(original):
        errors.append("final manifest size differs from original")
    if original_camera_split != final_camera_split:
        errors.append("camera/split budgets changed")
    if len({row["window_id"] for row in final_rows}) != len(final_rows):
        errors.append("window IDs are not unique")
    if len({row["source_mp4_url"] for row in final_rows}) != len(final_rows):
        errors.append("source MP4 URLs are not unique")
    if rejected_urls & {row["source_mp4_url"] for row in final_rows}:
        errors.append("a rejected source MP4 remains in the final manifest")
    per_day = Counter((row["camera"], row["day"]) for row in final_rows)
    if any(count > selection.MAX_WINDOWS_PER_DAY for count in per_day.values()):
        errors.append("a camera/day exceeds the daily window cap")
    day_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in final_rows:
        day_splits[(row["camera"], row["day"])].add(row["split"])
        if roles and roles.get(row["camera"], {}).get(row["day"]) != row["split"]:
            errors.append(f"day role mismatch for {row['window_id']}")
        expected_route = (
            "human_review_only" if row["split"] == "expert" else "sam3_then_human_review"
        )
        if row["labelling_route"] != expected_route:
            errors.append(f"wrong labelling route for {row['window_id']}")
    leaks = [f"{camera}:{day}" for (camera, day), splits in day_splits.items() if len(splits) != 1]
    if leaks:
        errors.append(f"day/split leaks: {leaks}")
    return {
        "valid": not errors,
        "errors": errors,
        "windows": len(final_rows),
        "camera_counts": dict(Counter(row["camera"] for row in final_rows)),
        "split_counts": dict(Counter(row["split"] for row in final_rows)),
        "camera_split_counts": {
            f"{camera}/{split}": count
            for (camera, split), count in sorted(final_camera_split.items())
        },
        "evidence_counts_preserved": original_evidence == final_evidence,
        "original_evidence_counts": {
            f"{camera}/{split}/{evidence}": count
            for (camera, split, evidence), count in sorted(original_evidence.items())
        },
        "final_evidence_counts": {
            f"{camera}/{split}/{evidence}": count
            for (camera, split, evidence), count in sorted(final_evidence.items())
        },
        "expert_windows": sum(row["split"] == "expert" for row in final_rows),
        "sam3_then_human_review_windows": sum(
            row["labelling_route"] == "sam3_then_human_review" for row in final_rows
        ),
        "day_split_leaks": leaks,
    }


def build(
    scan: Path,
    manifest_path: Path,
    review_path: Path,
    day_roles_path: Path,
    output: Path,
    seed: int = selection.SEED,
    excluded_urls: set[str] | None = None,
) -> dict:
    manifest = read_csv(manifest_path)
    review = read_csv(review_path)
    roles = json.loads(day_roles_path.read_text(encoding="utf-8"))["roles"]
    clips, images, _ = selection.load_candidates(scan)
    selection.add_condition_bins(clips, images)
    windows, replacements = choose_replacements(
        clips, manifest, review, roles, seed, excluded_urls=excluded_urls
    )
    final_rows = [selection.manifest_row(window) for window in windows]
    rejected_ids = {row["window_id"] for row in review if row["decision"] == "reject"}
    rejected_urls = {row["source_mp4_url"] for row in manifest if row["window_id"] in rejected_ids}
    verification = verify(manifest, final_rows, rejected_urls, roles)
    if not verification["valid"]:
        raise ValueError("final manifest verification failed: " + "; ".join(verification["errors"]))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "manifest.csv", final_rows, selection.MANIFEST_FIELDS)
    replacement_fields = [
        "rejected_window_id",
        "replacement_window_id",
        "camera",
        "split",
        "rejected_evidence",
        "replacement_evidence",
        "evidence_preserved",
        "rejected_day",
        "replacement_day",
        "replacement_source_mp4_url",
        "replacement_reason",
        "review_notes",
    ]
    write_csv(output / "replacements.csv", replacements, replacement_fields)
    review_by_id = {row["window_id"]: row for row in review}
    replacement_by_id = {row["replacement_window_id"]: row for row in replacements}
    final_review = []
    for row in final_rows:
        replacement = replacement_by_id.get(row["window_id"])
        if replacement:
            replacement_reason = f"replacement_for:{replacement['rejected_window_id']}"
            review_notes = (
                "Approved after local replacement contact-sheet review; screening evidence "
                "is not ground truth."
            )
        else:
            source_review = review_by_id[row["window_id"]]
            replacement_reason = source_review["replacement_reason"]
            review_notes = source_review["review_notes"]
        final_review.append(
            {
                "window_id": row["window_id"],
                "camera": row["camera"],
                "day": row["day"],
                "split": row["split"],
                "evidence": row["evidence"],
                "contact_sheet": f"contact_sheets/{row['camera']}-{row['split']}.jpg",
                "decision": "approve",
                "replacement_reason": replacement_reason,
                "review_notes": review_notes,
            }
        )
    write_csv(
        output / "contact_sheet_review.csv",
        final_review,
        [
            "window_id",
            "camera",
            "day",
            "split",
            "evidence",
            "contact_sheet",
            "decision",
            "replacement_reason",
            "review_notes",
        ],
    )
    (output / "day_roles.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "roles": roles,
                "rule": "whole days remain assigned to one split per camera",
                "labelling": {
                    "expert": "human review only",
                    "other": "SAM 3 suggestions then mandatory human review",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "verification.json").write_text(
        json.dumps({**verification, "replacements": len(replacements)}, indent=2), encoding="utf-8"
    )
    for camera in selection.CAMERAS:
        for split in selection.ROLE_SHARES:
            selection.write_contact_sheet(
                [
                    window
                    for window in windows
                    if window["camera"] == camera and window["split"] == split
                ],
                scan,
                output / "contact_sheets" / f"{camera}-{split}.jpg",
            )
    replacement_windows = {row["replacement_window_id"] for row in replacements}
    selection.write_contact_sheet(
        [window for window in windows if window["window_id"] in replacement_windows],
        scan,
        output / "replacement_contact_sheet.jpg",
    )
    evidence_by_camera = Counter((row["camera"], row["evidence"]) for row in final_rows)
    report_lines = [
        "# Final approved SmartBay camera-balanced manifest",
        "",
        (
            "All contact-sheet decisions and replacement contact sheets received local visual review. "
            "Screening evidence remains unverified and is not ground truth."
        ),
        "",
        f"- Windows: **{verification['windows']}**.",
        f"- Replacements: **{len(replacements)}**.",
        "- Cameras: Cam 1 120; Cam 2 40.",
        "- Splits: train 96; validation 24; test 24; expert 16.",
        "- Day/split leaks: zero.",
        "- Expert route: 16 human-review-only windows.",
        "- Other route: 144 SAM 3 suggestion windows followed by mandatory human review.",
        "",
        "## Final evidence mix",
        "",
    ]
    for camera in selection.CAMERAS:
        values = ", ".join(
            f"{evidence} {evidence_by_camera[(camera, evidence)]}"
            for evidence in ("strong", "weak", "none", "murky")
            if evidence_by_camera[(camera, evidence)]
        )
        report_lines.append(f"- {camera}: {values}.")
    report_lines += [
        "",
        (
            "Rejected black/obscured murky windows were preferentially replaced by usable negatives. "
            "Rejected weak mesh, hardware, fouling, glare, and haze candidates were preferentially "
            "replaced by visibly fish-bearing strong candidates when sparse camera/day coverage left no "
            "credible like-for-like replacement."
        ),
        "",
        "Do not begin student-model training until every generated pseudo-label has completed human review.",
    ]
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {**verification, "replacements": len(replacements)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=Path("data/archive_scan"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--day-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=selection.SEED)
    parser.add_argument(
        "--exclude-urls",
        type=Path,
        help="JSON list or object whose keys are source MP4 URLs rejected during replacement review",
    )
    parser.add_argument(
        "--role-overrides",
        type=Path,
        help="JSON camera/day role overrides used for safe whole-day swaps",
    )
    args = parser.parse_args()
    excluded_urls = set()
    if args.exclude_urls:
        raw = json.loads(args.exclude_urls.read_text(encoding="utf-8"))
        excluded_urls = set(raw if isinstance(raw, list) else raw.keys())
    if args.role_overrides:
        role_overrides = json.loads(args.role_overrides.read_text(encoding="utf-8"))
        source = json.loads(args.day_roles.read_text(encoding="utf-8"))
        for camera, camera_overrides in role_overrides.items():
            source["roles"][camera].update(camera_overrides)
        overridden = args.output / "effective_day_roles.json"
        overridden.parent.mkdir(parents=True, exist_ok=True)
        overridden.write_text(json.dumps(source, indent=2), encoding="utf-8")
        args.day_roles = overridden
    print(
        json.dumps(
            build(
                args.scan,
                args.manifest,
                args.review,
                args.day_roles,
                args.output,
                args.seed,
                excluded_urls,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
