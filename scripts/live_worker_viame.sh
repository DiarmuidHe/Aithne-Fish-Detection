#!/usr/bin/env bash
set -e
# Preserve the application's Python and site packages. VIAME's setup replaces
# PYTHONPATH with its own packages, including an OpenCV without MP4 decoding.
app_python=$(command -v python3)
if [ "${FISHIAL_PREPROCESS:-none}" = "funie_gan" ]; then
    app_packages=$("$app_python" -c 'import site; print(":".join(site.getsitepackages()))')
    source "${VIAME_SETUP_SCRIPT:-/opt/noaa/viame/setup_viame.sh}"
    export PYTHONPATH="$app_packages${PYTHONPATH:+:$PYTHONPATH}"
fi
# Explicit Python arguments also let offline checks use the exact worker runtime.
if [ "$#" -eq 0 ]; then
    set -- -m app.workers.live_worker
fi
exec "$app_python" "$@"
