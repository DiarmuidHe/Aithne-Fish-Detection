"""Offline retained-crop benchmark. Never imports a Fishial client or opens a camera.

Explicit --output opts into retaining the paired images for human review.
Only original capture-manifest crops and retained live Fishial crops are discovered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.services.fish_enhancement import PROVENANCE, enhance_crop, initialize_enhancement


def discover(root):
    found = {}
    for manifest in sorted((root / "fishial-experiments").glob("*/manifest.json")):
        data = json.loads(manifest.read_text())
        if data.get("preprocess") != "none":
            continue
        for entry in data.get("crops", []):
            path = manifest.parent / entry["file"]
            if path.is_file() and path.parent == manifest.parent and path.suffix == ".jpg":
                found[path] = entry
    for path in sorted((root / "live").glob("*/*/fishial/*.jpg")):
        found.setdefault(path, {})
    return sorted(found.items())


def benchmark(args):
    import numpy as np
    settings = Settings(_env_file=None, fishial_enabled=False, fishial_preprocess="funie_gan",
                        fishial_funie_model_path=str(args.model),
                        fishial_funie_model_sha256=PROVENANCE["sha256"],
                        fishial_funie_device=args.device)
    entries = discover(args.root)
    entries = [(p, e) for p, e in entries if not p.is_relative_to(args.output)]
    if not entries:
        raise SystemExit("No clean retained crops available")
    initialize_enhancement(settings)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # One warm-up is excluded from all reported latency samples.
    enhance_crop(entries[0][0].read_bytes(), settings)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, failures, totals = [], 0, []
    for index, (path, entry) in enumerate(entries):
        payload = path.read_bytes()
        name = f"crop-{index + 1}.jpg"
        try:
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = enhance_crop(payload, settings)
                elapsed = (time.perf_counter() - start) * 1000
                totals.append(elapsed)
                samples.append({key: result.metadata[key] for key in
                                ("decode_ms", "inference_ms", "encode_ms")} | {"total_ms": elapsed})
            for mode, image in (("none", payload), ("funie_gan", result.image)):
                folder = args.output / "pairs" / args.device / mode
                folder.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(image)
            rows.append({"file": name, "source": path.relative_to(args.root).as_posix(),
                         "expected_box": entry.get("expected_box"), **result.metadata,
                         "samples": samples})
        except Exception:  # noqa: BLE001 - persist only a fixed local failure reason
            failures += 1
            rows.append({"file": name, "reason": "preprocessing failed",
                         "original_sha256": hashlib.sha256(payload).hexdigest()})
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        rss = None
    report = {"device_requested": args.device, "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "model": PROVENANCE,
              "crops": rows, "failures": failures, "warmups": 1, "repeats": args.repeats,
              "median_ms": statistics.median(totals) if totals else None,
              "p95_ms": float(np.percentile(totals, 95)) if totals else None,
              "peak_process_rss_bytes": rss,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
              "runtime_added_bytes": 0, "model_added_bytes": PROVENANCE["bytes"],
              "accuracy_evidence": "None: no labelled Fishial calls. Keep FISHIAL_PREPROCESS=none."}
    (args.output / f"offline-report-{args.device}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("median_ms", "p95_ms", "failures",
                      "peak_process_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")}))
    return int(failures > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    raise SystemExit(benchmark(args))
