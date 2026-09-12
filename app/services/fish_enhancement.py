"""Offline, lazy Fishial JPEG enhancement. No optional runtime import at module load.

FUnIE uses Pillow bicubic resizing to match upstream torchvision's PIL path.
Output is clamped to the generator's [-1, 1] tanh range, mapped to uint8 with
round-to-nearest, then JPEG encoded at configurable quality (95 by default).
No per-image contrast stretch, classical correction, or upscaling is added.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.services.fishial import jpeg_size

PROVENANCE = json.loads((Path(__file__).parents[1] / "vendor/funie_gan/provenance.json").read_text())
_models = {}
_load_lock = threading.Lock()


class EnhancementError(Exception):
    def __init__(self):
        super().__init__("preprocessing failed")


@dataclass(frozen=True)
class EnhancedCrop:
    image: bytes = field(repr=False)
    metadata: dict


class _Generator:
    def __init__(self, settings):
        # Hash the very bytes safely loaded, avoiding path replacement races.
        data = Path(settings.fishial_funie_model_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != settings.fishial_funie_model_sha256:
            raise EnhancementError()
        import torch

        from app.vendor.funie_gan.generator import GeneratorFunieGAN

        device = settings.fishial_funie_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise EnhancementError()
        self.torch, self.device, self.lock = torch, device, threading.Lock()
        self.model = GeneratorFunieGAN()
        self.model.load_state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True),
                                   strict=True)
        self.model.to(device).eval().requires_grad_(False)
        # Validate state, device, output and kernel compatibility before any session.
        with torch.inference_mode():
            self._check(self.model(torch.zeros((1, 3, 256, 256), device=device)))

    def _check(self, output):
        if tuple(output.shape) != (1, 3, 256, 256) or not self.torch.isfinite(output).all().item():
            raise EnhancementError()
        return output

    def infer(self, rgb):
        torch = self.torch
        with self.lock, torch.inference_mode():
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).sub(.5).div(.5)
            output = self._check(self.model(tensor.unsqueeze(0).to(self.device)))
            return (output.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
                    .squeeze(0).permute(1, 2, 0).cpu().numpy())


def _generator(settings):
    key = (str(settings.fishial_funie_model_path), settings.fishial_funie_model_sha256,
           settings.fishial_funie_device)
    with _load_lock:
        if key not in _models:
            _models[key] = _Generator(settings)
        return _models[key]


def initialize_enhancement(settings):
    """Live-worker startup validation; never import torch for classical/none modes."""
    if settings.fishial_preprocess == "funie_gan":
        try:
            _generator(settings)
        except Exception:  # noqa: BLE001 - fixed local error, never runtime paths or tensor data
            raise EnhancementError() from None


def enhance_crop(image: bytes, settings) -> EnhancedCrop:
    """One selected frame -> exact submitted bytes and safe, compact provenance."""
    try:
        return _enhance(image, settings)
    except Exception:  # noqa: BLE001 - redact decoder/runtime diagnostics at the boundary
        raise EnhancementError() from None


def _enhance(image, settings):
    mode = settings.fishial_preprocess
    size = jpeg_size(image)
    if size is None or not image.endswith(b"\xff\xd9"):
        raise EnhancementError()
    metadata = {"mode": mode, "model_id": PROVENANCE["model_id"] if mode == "funie_gan" else None,
                "model_sha256": settings.fishial_funie_model_sha256 if mode == "funie_gan" else None,
                "original_sha256": hashlib.sha256(image).hexdigest(),
                "input_dimensions": list(size), "decode_ms": 0.0, "inference_ms": 0.0,
                "encode_ms": 0.0, "succeeded": True}
    if mode == "none":
        payload = image  # Byte-identical; no image decoder, re-encoder or learned model.
    else:
        import cv2
        import numpy as np
        start = time.perf_counter()
        crop = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            raise EnhancementError()
        metadata["decode_ms"] = (time.perf_counter() - start) * 1000
        if mode == "funie_gan":
            from PIL import Image
            generator = _generator(settings)
            rgb = np.asarray(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
                (256, 256), Image.Resampling.BICUBIC))
            start = time.perf_counter()
            output = generator.infer(rgb)
            metadata["inference_ms"] = (time.perf_counter() - start) * 1000
            if output.shape != (256, 256, 3) or output.dtype != np.uint8:
                raise EnhancementError()
            crop = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        elif mode in ("white_balance", "clahe", "both"):
            from app.services.species_quality import preprocess_crop
            start = time.perf_counter()
            crop = preprocess_crop(crop, settings, cv2)
            metadata["inference_ms"] = (time.perf_counter() - start) * 1000
        else:
            raise EnhancementError()
        start = time.perf_counter()
        quality = settings.fishial_funie_jpeg_quality if mode == "funie_gan" else 95
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise EnhancementError()
        payload = encoded.tobytes()
        metadata["encode_ms"] = (time.perf_counter() - start) * 1000
    result = EnhancedCrop(payload, {**metadata, "submitted_sha256": hashlib.sha256(payload).hexdigest(),
                                   "output_dimensions": list(jpeg_size(payload) or ())})
    validate_enhancement(result, image, settings)
    return result


def validate_enhancement(result, original, settings):
    """Check boundary/test-double output before writing the pre-send audit."""
    meta = result.metadata
    size = jpeg_size(result.image)
    if (not size or not result.image.endswith(b"\xff\xd9") or meta.get("succeeded") is not True
            or meta.get("mode") != settings.fishial_preprocess
            or meta.get("original_sha256") != hashlib.sha256(original).hexdigest()
            or meta.get("submitted_sha256") != hashlib.sha256(result.image).hexdigest()
            or meta.get("output_dimensions") != list(size)
            or meta.get("input_dimensions") != list(jpeg_size(original) or ())
            or set(meta) != {"mode", "model_id", "model_sha256", "original_sha256",
                             "submitted_sha256", "input_dimensions", "output_dimensions",
                             "decode_ms", "inference_ms", "encode_ms", "succeeded"}
            or any(type(meta.get(key)) not in (int, float) or not math.isfinite(meta[key])
                   or meta[key] < 0 for key in ("decode_ms", "inference_ms", "encode_ms"))):
        raise EnhancementError()
    learned = settings.fishial_preprocess == "funie_gan"
    if (meta["model_id"] != (PROVENANCE["model_id"] if learned else None)
            or meta["model_sha256"] != (settings.fishial_funie_model_sha256 if learned else None)):
        raise EnhancementError()


def failure_metadata(image, settings):
    learned = settings.fishial_preprocess == "funie_gan"
    return {"mode": settings.fishial_preprocess,
            "model_id": PROVENANCE["model_id"] if learned else None,
            "model_sha256": settings.fishial_funie_model_sha256 if learned else None,
            "original_sha256": hashlib.sha256(image).hexdigest() if image else None,
            "submitted_sha256": None, "input_dimensions": list(jpeg_size(image) or ()),
            "output_dimensions": None, "inference_ms": None,
            "succeeded": False, "reason": "preprocessing failed"}
