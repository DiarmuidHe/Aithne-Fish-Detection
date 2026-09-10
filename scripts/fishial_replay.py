"""Tune Fishial identification against stored data, at zero API cost.

Every raw response a session ever received is already persisted in
``live_fish_tracks.fishial_votes_json``, so parsing, the multi-object matcher, the
regional filter, voting, consensus and the early-stop rules can all be re-run offline
under a different configuration. That is what ``--dry-run`` does, and it makes
**zero** network calls.

``--replay`` is the only mode that spends money, and it refuses to run without an
explicit ``--max-calls`` ceiling.

Examples::

    python scripts/fishial_replay.py --dry-run --session ab69aaca-...
    python scripts/fishial_replay.py --dry-run --session ab69aaca-... \\
        --set fishial_max_empty_responses=1 --set fishial_min_species_score=0.4
    python scripts/fishial_replay.py --replay data/outputs/live/<id>/<track>/fishial \\
        --max-calls 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import Settings, get_settings
from app.db.models import LiveFishTrack, LiveMonitorSession
from app.services.fishial import FishialClient, FishialError, trim_raw
from app.services.species_quality import consensus_outlook, preprocess_crop, review_diagnostics
from app.services.species_region import filter_species


def apply_overrides(settings: Settings, overrides: list[str]) -> Settings:
    """``--set key=value`` for any setting, so thresholds sweep without editing .env."""

    values = {}
    for override in overrides:
        key, _, raw = override.partition("=")
        key = key.strip()
        field = Settings.model_fields.get(key)
        if field is None:
            raise SystemExit(f"Unknown setting {key!r}")
        try:
            values[key] = json.loads(raw)
        except ValueError:
            values[key] = raw
    return settings.model_copy(update=values).model_validate(
        {**settings.model_dump(), **values})


def _stop_reason(frames, remaining, settings):
    """Mirror of SpeciesIdentifier._stop_reason, over stored frame records."""

    limit = settings.fishial_max_empty_responses
    if limit:
        empty = 0
        for frame in reversed(frames):
            if not frame.get("attempts"):
                continue
            if not frame.get("empty"):
                break
            empty += 1
        if empty >= limit:
            return "classifier returned no candidates"
    outlook = consensus_outlook(frames, remaining, settings)
    if outlook == "decided":
        return "decided"
    if outlook == "unreachable":
        return "consensus unreachable"
    return None


def target_box(staged, geometry):
    """Pixel-space target box for a staged frame, or ``None`` to abstain.

    Sessions recorded before this feature have no ``expected_box``, so reconstruct it
    from the detection row and the crop margin - the same arithmetic
    ``_stage_species_crop`` used to write the crop. ``geometry`` supplies that
    fallback keyed by frame number.
    """

    if staged and staged.get("expected_box") and staged.get("crop_size"):
        width, height = staged["crop_size"]
        x1, y1, x2, y2 = staged["expected_box"]
        return (x1 * width, y1 * height, x2 * width, y2 * height)
    return geometry


def reconstruct(detection, frame_size, margin):
    """Crop geometry for a legacy staged frame: pixel target box within the crop."""

    import math
    x1, y1, x2, y2 = detection
    dx, dy = (x2 - x1) * (margin - 1) / 2, (y2 - y1) * (margin - 1) / 2
    # Only the crop's origin matters: the returned box is relative to it, exactly as
    # _stage_species_crop writes it. frame_size documents the clamp that produced it.
    del frame_size
    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    return (x1 - left, y1 - top, x2 - left, y2 - top)


def rescore_track(track, settings, region, client_parse, geometry=None, frames_per_fish=None) -> dict:
    """Re-run the whole decision path over one track's stored responses."""

    audit = track.fishial_votes
    stored = audit.get("frames") or []
    staged_by_number = {entry.get("frame_number"): entry
                        for entry in audit.get("staged") or []}
    geometry = geometry or {}
    # What the session actually paid, independent of where the replay stops.
    def attempts(frame):
        # Explicitly unreserved frames cost nothing. Legacy raw responses without
        # a journal prove one call, but never imply an extra retry.
        return frame.get("attempts", [{}] if frame.get("raw") else [])

    spent = sum(len(attempts(frame)) for frame in stored)
    capacity = min(len(staged_by_number), frames_per_fish or settings.fishial_default_frames_per_fish)
    capacity = max(capacity, len(stored))  # Legacy audits may lack a manifest.
    replayed, stopped = [], None
    for position, frame in enumerate(stored):
        raw = frame.get("raw")
        number = frame.get("frame_number")
        record = {"frame_number": number, "attempts": attempts(frame),
                  "voted": False, "species": None, "score": None, "empty": False}
        if not raw:
            record["reason"] = frame.get("reason")
            replayed.append(record)
            continue
        try:
            prediction = client_parse(
                raw, target_box(staged_by_number.get(number), geometry.get(number)))
        except FishialError as exc:
            record["reason"] = exc.reason
            replayed.append(record)
            continue
        record["succeeded"] = True
        record["object_count"] = prediction.object_count
        record["object_iou"] = prediction.object_iou
        record["empty"] = prediction.object_index is not None and not prediction.species
        ranked, dropped = filter_species(prediction.species, region)
        record["dropped_species"] = dropped
        record["reason"] = ("implausible_for_region" if dropped and not ranked else
                            "no unambiguous fish prediction")
        if ranked:
            name, score = ranked[0]
            margin = score - (ranked[1][1] if len(ranked) > 1 else 0)
            record.update(species=name, score=score,
                          voted=margin >= settings.fishial_min_frame_margin)
            record["reason"] = None if record["voted"] else "ambiguous frame"
        replayed.append(record)
        # An early-stopped session still had unsubmitted staged frames available.
        # Counting only stored responses falsely makes consensus unreachable early.
        remaining = capacity - position - 1
        stopped = _stop_reason(replayed, remaining, settings) if remaining else None
        if stopped:
            break

    votes = Counter(f["species"] for f in replayed if f.get("voted"))
    submitted = sum(bool(f.get("attempts")) for f in replayed)
    outcome, species = "review_required", None
    if votes:
        winner, count = votes.most_common(1)[0]
        mean = sum(f["score"] for f in replayed if f.get("voted") and f["species"] == winner) / count
        if (sum(value == count for value in votes.values()) == 1
                and count >= settings.fishial_min_votes
                and submitted >= settings.fishial_min_frames_to_vote
                and submitted and count / submitted >= settings.fishial_vote_ratio
                and mean >= settings.fishial_min_species_score):
            outcome, species = "identified", winner
    if not stored:
        outcome = track.fishial_state
    reason = stopped if stopped and stopped != "decided" else audit.get("reason")
    diagnostics = review_diagnostics({"reason": reason, "frames": replayed}) if stored else review_diagnostics(audit)
    return {"track": str(track.id), "old": track.fishial_state, "old_reason": audit.get("reason"),
            "new": outcome, "new_reason": None if outcome == "identified" else diagnostics["reason"],
            "stop_reason": reason, "frame_reasons": diagnostics["frame_reasons"],
            "incomplete": bool(stored and not stopped and len(replayed) < capacity),
            "species": species, "calls_recorded": spent,
            "calls_replayed": sum(len(f["attempts"]) for f in replayed),
            "dropped": [name for f in replayed for name in f.get("dropped_species") or []],
            "raw_bytes": sum(len(json.dumps(f.get("raw"))) for f in stored if f.get("raw")),
            "trimmed_bytes": sum(len(json.dumps(trim_raw(f["raw"]))) for f in stored if f.get("raw"))}


def legacy_geometry(db, track, settings) -> dict:
    """Reconstructed target boxes for a session staged before ``expected_box`` existed."""

    audit = track.fishial_votes
    numbers = [entry.get("frame_number") for entry in audit.get("staged") or []
               if not entry.get("expected_box")]
    if not numbers:
        return {}
    from app.db.models import LiveFishDetection
    rows = db.execute(select(LiveFishDetection.frame_number, LiveFishDetection.x1,
                             LiveFishDetection.y1, LiveFishDetection.x2, LiveFishDetection.y2)
                      .where(LiveFishDetection.track_id == track.id,
                             LiveFishDetection.frame_number.in_(numbers))).all()
    bounds = db.execute(select(LiveFishDetection.x2, LiveFishDetection.y2)
                        .where(LiveFishDetection.track_id == track.id)).all()
    if not bounds:
        return {}
    # Boxes are stored clamped to width-1/height-1, so the observed maxima recover
    # the frame size for a session that never recorded it.
    frame_size = (max(row[0] for row in bounds) + 1, max(row[1] for row in bounds) + 1)
    return {number: reconstruct((x1, y1, x2, y2), frame_size, settings.fishial_crop_margin)
            for number, x1, y1, x2, y2 in rows}


def dry_run(args, settings) -> int:
    """Re-decide a stored session offline. Makes zero ``identify`` calls."""

    engine = create_engine(settings.database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        with factory() as db:
            session = db.get(LiveMonitorSession, args.session)
            if session is None:
                raise SystemExit(f"No live session {args.session}")
            region = (session.species_id_region if settings.fishial_region_filter_enabled else None)
            tracks = db.scalars(select(LiveFishTrack).where(
                LiveFishTrack.session_id == session.id,
                LiveFishTrack.fishial_state != "disabled").order_by(
                    LiveFishTrack.first_seen_at)).all()
            rows = [rescore_track(track, settings, region, args.parse,
                                  legacy_geometry(db, track, settings),
                                  session.species_id_frames_per_fish) for track in tracks]
    finally:
        engine.dispose()

    print(f"session {args.session}  camera={session.source_key}  region={region}")
    print(f"  budget ceiling: {session.species_id_fish_target} x "
          f"{session.species_id_frames_per_fish} + {settings.fishial_max_api_retries} = "
          f"{session.species_id_fish_target * session.species_id_frames_per_fish + settings.fishial_max_api_retries}")
    for row in rows:
        change = "" if row["old"] == row["new"] else "  <-- CHANGED"
        print(f"  {row['track'][:8]}  {row['old']}({row['old_reason']}) -> "
              f"{row['new']}({row['new_reason']}) species={row['species']} "
              f"calls {row['calls_recorded']} -> {row['calls_replayed']}{change}")
        if row["dropped"]:
            print(f"            dropped as implausible_for_region: {', '.join(row['dropped'])}")
        if row["frame_reasons"]:
            print(f"            frame abstentions: {row['frame_reasons']} (stop: {row['stop_reason']})")
        if row["incomplete"]:
            print("            incomplete replay: further staged frames have no stored responses")
    counts = Counter(row["new"] for row in rows)
    recorded = sum(row["calls_recorded"] for row in rows)
    # Compare reserved attempts on the recorded prefix, including retries. Never
    # invent answers or savings for unsubmitted frames.
    replayed = sum(row["calls_replayed"] for row in rows)
    raw_bytes = sum(row["raw_bytes"] for row in rows)
    trimmed = sum(row["trimmed_bytes"] for row in rows)
    print(f"  outcomes: identified={counts.get('identified', 0)} "
          f"review_required={counts.get('review_required', 0)} error={counts.get('error', 0)}")
    print(f"  review reasons: {dict(Counter(row['new_reason'] for row in rows if row['new'] != 'identified'))}")
    print(f"  calls actually spent: {recorded}   calls the stop rules would spend: {replayed} "
          f"(saved {recorded - replayed})")
    if raw_bytes:
        print(f"  stored raw: {raw_bytes} B -> {trimmed} B trimmed "
              f"({100 * (1 - trimmed / raw_bytes):.1f}% saved)")
    return 0


def replay(args, settings) -> int:
    """Replay a frozen experiment with a durable ceiling on actual image attempts."""

    from scripts.fishial_replay_journal import ReplayJournal

    directory = Path(args.replay)
    if args.capture_source:
        from scripts.fishial_capture import collect
        collect(directory, args.capture_source, settings, args.capture_seconds)
    crops = sorted(directory.glob("*.jpg"))
    if not crops:
        raise SystemExit(f"No stored crops under {directory}")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {entry["file"]: entry for entry in manifest.get("crops", [])}
    modes = ["none", "clahe", "white_balance"] if args.compare_preprocess else [settings.fishial_preprocess]
    if args.compare_preprocess:
        if len(crops) != 2 or set(entries) != {p.name for p in crops} or manifest.get("preprocess") != "none":
            raise SystemExit("Comparison requires two clean original crops with a capture manifest")
        if not manifest.get("region") or any(not entry.get("expected_box") for entry in entries.values()):
            raise SystemExit("Comparison requires a region and target boxes")
    region = manifest.get("region")
    # Freeze bytes before hashing: a file changed during a run cannot change what
    # is sent under an existing journal. Stored variants also make the test reviewable.
    inputs = {path.name: path.read_bytes() for path in crops}
    spec = {"max_calls": args.max_calls, "max_retries": settings.fishial_max_api_retries,
            "modes": modes, "region": region, "entries": entries,
            "hashes": {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
            "settings": {name: getattr(settings, name) for name in (
                "fishial_clahe_clip", "fishial_upscale_short_side", "fishial_min_species_score",
                "fishial_min_frame_margin", "fishial_object_match_min_iou", "fishial_object_match_min_margin")}}
    journal = ReplayJournal(directory / "replay.sqlite", spec)
    client = None
    try:
        # Mode-first order gives every fish a baseline before testing corrections.
        for mode in modes:
            for crop in crops:
                key = f"{mode}/{crop.name}"
                if journal.spent >= args.max_calls or not journal.claim(key):
                    continue
                payload = inputs[crop.name]
                if mode != "none" or settings.fishial_upscale_short_side:
                    import cv2
                    import numpy as np
                    original = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if original is None:
                        journal.finish(key, {"reason": "unreadable crop"})
                        continue
                    tuned = settings.model_copy(update={"fishial_preprocess": mode})
                    ok, encoded = cv2.imencode(".jpg", preprocess_crop(original, tuned, cv2),
                                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        journal.finish(key, {"reason": "unreadable crop"})
                        continue
                    payload = encoded.tobytes()
                variant = directory / "variants" / mode / crop.name
                variant.parent.mkdir(parents=True, exist_ok=True)
                variant.write_bytes(payload)
                if client is None:
                    client = FishialClient(settings)
                client.before_image_call = partial(journal.reserve, key)
                print(f"  {key} (reserved {journal.spent}/{args.max_calls})", flush=True)
                try:
                    prediction = client.identify(payload, entries.get(crop.name, {}).get("expected_box"))
                    ranked, dropped = filter_species(prediction.species, region)
                    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0) if ranked else 0
                    usable = bool(ranked and ranked[0][1] >= settings.fishial_min_species_score
                                  and margin >= settings.fishial_min_frame_margin)
                    raw = trim_raw(prediction.raw)
                    raw.pop("queryToken", None)
                    result = {"species": ranked, "dropped": dropped, "usable": usable,
                              "object_index": prediction.object_index, "object_iou": prediction.object_iou,
                              "raw": raw}
                    print(f"      species={ranked}, usable={usable}", flush=True)
                except FishialError as exc:
                    result = {"reason": exc.reason}
                    print(f"      error: {exc.reason}", flush=True)
                journal.finish(key, result)
        results = journal.results()
        report = {"calls_reserved": journal.spent, "max_calls": args.max_calls,
                  "source": manifest.get("source"), "results": results,
                  "usable_by_mode": {mode: sum(bool(results.get(f"{mode}/{p.name}", {}).get("usable"))
                                                for p in crops) for mode in modes},
                  "recommendation": "Keep fishial_preprocess=none. This small screen cannot establish species accuracy or justify a production change."}
        temporary = directory / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(directory / "report.json")
        print(f"  calls reserved: {journal.spent} (ceiling {args.max_calls}); report: {directory / 'report.json'}")
    finally:
        if client is not None:
            client.close()
        journal.close()
    return 0


def main(argv=None, parse=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="Re-decide a stored session offline. Makes zero API calls.")
    mode.add_argument("--replay", metavar="DIRECTORY",
                      help="Send stored crops through the real Fishial client.")
    parser.add_argument("--session", help="Live session id, required with --dry-run.")
    parser.add_argument("--max-calls", type=int,
                        help="Hard ceiling on image calls. Required with --replay.")
    parser.add_argument("--capture-source", help="Collect two clean crops from this configured camera before replay.")
    parser.add_argument("--capture-seconds", type=int, default=120,
                        help="Maximum capture window (default 120s), plus one bounded detector pass.")
    parser.add_argument("--compare-preprocess", action="store_true",
                        help="Compare none, CLAHE and white balance on two manifest crops.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any setting, e.g. --set fishial_quality_floor=0.4")
    args = parser.parse_args(argv)
    if args.replay and (args.max_calls is None or args.max_calls <= 0):
        parser.error("--replay requires an explicit positive --max-calls ceiling")
    if (args.capture_source or args.compare_preprocess) and not args.replay:
        parser.error("Capture and comparison require --replay with --max-calls")
    if not 1 <= args.capture_seconds <= 600:
        parser.error("--capture-seconds must be between 1 and 600")
    if args.dry_run and not args.session:
        parser.error("--dry-run requires --session")
    settings = apply_overrides(get_settings(), args.set)
    if args.dry_run:
        # Injected so a test can assert the dry run never reaches a live client.
        # _parse is pure: it takes a stored response and a pixel target box, and
        # touches no socket, so a dry run cannot spend a call even by accident.
        args.parse = parse or (lambda raw, target: FishialClient._parse(raw, target, settings))
        return dry_run(args, settings)
    return replay(args, settings)


if __name__ == "__main__":  # pragma: no cover - thin CLI shell
    raise SystemExit(main())
