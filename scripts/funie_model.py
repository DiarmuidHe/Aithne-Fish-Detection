"""Explicit administrator-only HTTPS download of the pinned FUnIE artifact."""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.fish_enhancement import PROVENANCE


def download(destination):
    destination = Path(destination)
    expected = PROVENANCE["sha256"]
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise SystemExit("Existing model hash mismatch; choose a new destination")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(PROVENANCE["url"], timeout=60) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("HTTPS required")
                digest, size = hashlib.sha256(), 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size > PROVENANCE["bytes"]:
                        raise ValueError("Unexpected model size")
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected or size != PROVENANCE["bytes"]:
            raise ValueError("Model verification failed")
        os.replace(temporary, destination)
    except Exception:  # noqa: BLE001 - avoid exposing filesystem/transport diagnostics
        raise SystemExit("Model download or checksum verification failed") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    download(parser.parse_args().output)
    print("Pinned model verified")
