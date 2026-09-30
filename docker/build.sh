#!/usr/bin/env bash
# Build the HaptoFlow image.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-haptoflow:uv}"
docker build -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE" "$@" "$REPO_ROOT"
