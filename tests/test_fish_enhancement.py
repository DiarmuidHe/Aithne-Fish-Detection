"""Default suite is offline and torch-free; real weights are an explicit opt-in."""
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services import fish_enhancement as enhancement
from app.services.species_quality import preprocess_crop


def jpeg():
    image = np.full((37, 89, 3), [90, 30, 50], np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def test_none_byte_identical_without_decoder_or_model(test_settings, monkeypatch):
    original = jpeg()
    monkeypatch.setattr(enhancement, "_Generator", lambda *_: pytest.fail("Loaded learned runtime"))
    monkeypatch.setattr(cv2, "imdecode", lambda *_: pytest.fail("Decoded none input"))
    result = enhancement.enhance_crop(original, test_settings)
    assert result.image is original
    assert result.metadata["original_sha256"] == result.metadata["submitted_sha256"]
    assert result.metadata["inference_ms"] == 0
    # Also prove module imports do not pull torch into a fresh API process.
    subprocess.run([sys.executable, "-c", ("import sys; import app.services.fish_enhancement; "
                    "assert 'torch' not in sys.modules")], check=True)


@pytest.mark.parametrize("mode", ["white_balance", "clahe", "both"])
def test_classical_pixels_match_existing_transform(test_settings, mode):
    test_settings.fishial_preprocess = mode
    test_settings.fishial_upscale_short_side = 80
    original = jpeg()
    crop = cv2.imdecode(np.frombuffer(original, np.uint8), 1)
    expected = cv2.imencode(".jpg", preprocess_crop(crop, test_settings, cv2),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    assert enhancement.enhance_crop(original, test_settings).image == expected
    assert enhancement.enhance_crop(original, test_settings).image == expected


@pytest.mark.parametrize("values", [
    {"fishial_preprocess": "unknown"}, {"fishial_funie_device": "mps"},
    {"fishial_funie_model_sha256": "A" * 64}, {"fishial_funie_model_sha256": "0" * 63},
    {"fishial_funie_jpeg_quality": 0}, {"fishial_funie_jpeg_quality": 101},
    {"fishial_preprocess": "funie_gan"},
    {"fishial_preprocess": "funie_gan", "fishial_funie_model_path": "model.pth"},
])
def test_settings_reject_invalid_values(test_settings, values):
    with pytest.raises(ValidationError):
        Settings.model_validate({**test_settings.model_dump(), **values})


def test_settings_require_model_only_when_selected(test_settings):
    assert Settings.model_validate(test_settings.model_dump()).fishial_preprocess == "none"
    values = {**test_settings.model_dump(), "fishial_preprocess": "funie_gan",
              "fishial_funie_model_path": "private-model-path", "fishial_funie_model_sha256": "0" * 64}
    assert Settings.model_validate(values).fishial_preprocess == "funie_gan"
    assert "private-model-path" not in repr(Settings.model_validate(values))


def test_model_cache_initializes_once_and_serializes_concurrent_loads(test_settings, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    created = []
    monkeypatch.setattr(enhancement, "_models", {})
    def construct(settings):
        created.append(settings)
        return object()
    monkeypatch.setattr(enhancement, "_Generator", construct)
    test_settings.fishial_preprocess = "funie_gan"
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: enhancement._generator(test_settings), range(8)))
    assert len(created) == 1 and all(result is results[0] for result in results)


@pytest.mark.parametrize("artifact", ["missing", "hash-mismatch", "corrupt"])
def test_bad_model_fails_locally_and_redacts(test_settings, tmp_path, artifact, monkeypatch):
    path = tmp_path / "private-model"
    test_settings.fishial_preprocess = "funie_gan"
    test_settings.fishial_funie_model_path = str(path)
    test_settings.fishial_funie_model_sha256 = "0" * 64
    if artifact != "missing":
        path.write_bytes(b"invalid")
    if artifact == "corrupt":
        test_settings.fishial_funie_model_sha256 = hashlib.sha256(b"invalid").hexdigest()
        def unsafe(*args, **kwargs):
            raise RuntimeError("private-model secret")
        monkeypatch.setattr(enhancement, "_Generator", unsafe)
    with pytest.raises(enhancement.EnhancementError) as exc:
        enhancement.initialize_enhancement(test_settings)
    assert str(exc.value) == "preprocessing failed" and exc.value.__cause__ is None


def test_nonfinite_or_wrong_shape_output_rejected_without_torch():
    generator = object.__new__(enhancement._Generator)
    generator.torch = SimpleNamespace(isfinite=np.isfinite)
    for output in (np.full((1, 3, 256, 256), np.nan), np.zeros((1, 3, 20, 20))):
        with pytest.raises(enhancement.EnhancementError):
            generator._check(output)


@pytest.mark.parametrize("device,incompatible", [("cpu", False), ("cuda", False), ("cpu", True)])
def test_safe_loader_eval_and_device_validation_with_fake_runtime(
        test_settings, tmp_path, monkeypatch, device, incompatible):
    from contextlib import contextmanager
    events = []
    @contextmanager
    def inference_mode():
        events.append("inference")
        yield
    class Model:
        def load_state_dict(self, state, strict):
            assert state == {"weights": "safe"} and strict is True
            if incompatible:
                raise ValueError("private state shape mismatch")
        def to(self, selected):
            assert selected == "cpu"
            return self
        def eval(self):
            events.append("eval")
            return self
        def requires_grad_(self, enabled):
            assert enabled is False
            return self
        def __call__(self, tensor):
            assert events == ["eval", "inference"]
            return tensor
    def load(stream, map_location, weights_only):
        assert stream.read() == b"safe fixture"
        assert map_location == "cpu" and weights_only is True
        return {"weights": "safe"}
    torch = SimpleNamespace(load=load, cuda=SimpleNamespace(is_available=lambda: False),
                            inference_mode=inference_mode, isfinite=np.isfinite,
                            zeros=lambda shape, device: np.zeros(shape))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "app.vendor.funie_gan.generator",
                        SimpleNamespace(GeneratorFunieGAN=Model))
    path = tmp_path / "weights"
    path.write_bytes(b"safe fixture")
    test_settings.fishial_preprocess = "funie_gan"
    test_settings.fishial_funie_model_path = str(path)
    test_settings.fishial_funie_model_sha256 = hashlib.sha256(b"safe fixture").hexdigest()
    test_settings.fishial_funie_device = device
    if device == "cuda" or incompatible:
        with pytest.raises(enhancement.EnhancementError):
            enhancement.initialize_enhancement(test_settings)
    else:
        enhancement.initialize_enhancement(test_settings)
        assert events == ["eval", "inference"]


def test_live_worker_validates_before_claiming_sessions(test_settings, monkeypatch):
    from app.workers import live_worker
    test_settings.live_monitor_enabled = True
    test_settings.viame_mock = False
    test_settings.fishial_preprocess = "funie_gan"
    test_settings.fishial_funie_model_path = "missing-private-model"
    test_settings.fishial_funie_model_sha256 = "0" * 64
    monkeypatch.setattr(live_worker, "get_settings", lambda: test_settings)
    monkeypatch.setattr(live_worker, "claim_session", lambda *args: pytest.fail("Claimed session"))
    monkeypatch.setattr(live_worker, "recover_stale_sessions", lambda *args: pytest.fail("Recovered session"))
    with pytest.raises(enhancement.EnhancementError, match="preprocessing failed"):
        live_worker.run_worker_forever()


def test_invalid_encoded_output_and_metadata_rejected(test_settings):
    original = jpeg()
    result = enhancement.enhance_crop(original, test_settings)
    for meta in ({**result.metadata, "inference_ms": float("nan")},
                 {**result.metadata, "private_path": "secret"},
                 {**result.metadata, "submitted_sha256": "0" * 64}):
        with pytest.raises(enhancement.EnhancementError):
            enhancement.validate_enhancement(enhancement.EnhancedCrop(result.image, meta), original, test_settings)
    with pytest.raises(enhancement.EnhancementError):
        enhancement.enhance_crop(b"invalid", test_settings)


@pytest.mark.funie_smoke
def test_real_model_cpu_smoke():
    if os.getenv("FUNIE_SMOKE") != "1":
        pytest.skip("Set FUNIE_SMOKE=1 to run the explicit offline model smoke test")
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("PIL") is None:
        pytest.skip("Optional PyTorch/Pillow runtime absent")
    path = Path(os.getenv("FUNIE_MODEL", "data/models/funie/funie_generator.pth"))
    if not path.is_file():
        pytest.skip("Pinned local FUnIE model artifact absent")
    from scripts.funie_smoke import smoke
    smoke(path)
