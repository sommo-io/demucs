#!/usr/bin/env bash
# Deploy the Modal app, then pre-build memory snapshots (see warmup.py).
set -euo pipefail
cd "$(dirname "$0")/.."
modal deploy modal_app/app.py
uv run --quiet --no-project --with modal python modal_app/warmup.py demucs "${WARMUP_CONTAINERS:-6}"
