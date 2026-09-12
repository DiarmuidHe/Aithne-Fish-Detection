"""Explicit offline real-model smoke test, also usable without pytest installed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def smoke(path):
    import cv2
    import numpy as np
    import torch

    from app.config import Settings
    from app.services.fish_enhancement import PROVENANCE, _generator, enhance_crop
    settings = Settings(_env_file=None, fishial_enabled=False, fishial_preprocess="funie_gan",
                        fishial_funie_model_path=str(path),
                        fishial_funie_model_sha256=PROVENANCE["sha256"], fishial_funie_device="cpu")
    image = cv2.imencode(".jpg", np.full((17, 31, 3), [10, 60, 80], np.uint8))[1].tobytes()
    first, second = enhance_crop(image, settings), enhance_crop(image, settings)
    assert first.image == second.image
    assert cv2.imdecode(np.frombuffer(first.image, np.uint8), 1).shape == (256, 256, 3)
    generator = _generator(settings)
    assert not generator.model.training
    assert all(not p.requires_grad for p in generator.model.parameters())
    seen = []
    handle = generator.model.register_forward_pre_hook(lambda *args: seen.append(torch.is_inference_mode_enabled()))
    enhance_crop(image, settings)
    handle.remove()
    assert seen == [True]
    print("Pinned model CPU smoke passed: safe load, eval/inference, deterministic valid JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    smoke(parser.parse_args().model)
