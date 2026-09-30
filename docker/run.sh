#!/usr/bin/env bash
# Run the HaptoFlow image with the repo bind-mounted at /workspace.
#
#   docker/run.sh                                   # interactive shell
#   docker/run.sh python scripts/train.py --model flow_matching
#
# Environment knobs:
#   IMAGE=haptoflow:uv   image to run
#   GPUS=all             value for --gpus; set GPUS= to run on CPU
#   SHM_SIZE=8g          /dev/shm size (DataLoader workers need it)
#   PORTS=               host ports to publish 1:1 on 127.0.0.1, e.g. "7860"
#                        (Gradio); off by default so concurrent
#                        containers do not collide
# HF_TOKEN / WANDB_API_KEY / WANDB_MODE are forwarded when set.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-haptoflow:uv}"
GPUS="${GPUS-all}"
SHM_SIZE="${SHM_SIZE:-8g}"
PORTS="${PORTS:-}"

args=(--rm -i --shm-size="$SHM_SIZE")
[ -t 0 ] && [ -t 1 ] && args+=(-t)
[ -n "$GPUS" ] && args+=(--gpus "$GPUS")
for p in $PORTS; do args+=(-p "127.0.0.1:$p:$p"); done

args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp/home -e USER="$(id -un)" -e LOGNAME="$(id -un)")
args+=(-v "$REPO_ROOT:/workspace" -w /workspace)

# Hugging Face cache (+ token, if set, for higher Hub rate limits).
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
args+=(-v "$HF_CACHE:/hf-cache" -e HF_HOME=/hf-cache)
# wandb reads its API key from ~/.netrc after `wandb login`.
[ -f "$HOME/.netrc" ] && args+=(-v "$HOME/.netrc:/tmp/home/.netrc:ro")
for v in HF_TOKEN WANDB_API_KEY WANDB_MODE; do
    [ -n "${!v:-}" ] && args+=(-e "$v")
done

# Gradio binds to 127.0.0.1 by default, which is unreachable through -p;
# the host side of -p above is still loopback-only.
args+=(-e GRADIO_SERVER_NAME=0.0.0.0)

# X11 forwarding for matplotlib windows, only when a display exists.
if [ -n "${DISPLAY:-}" ]; then
    command -v xhost >/dev/null && xhost +local:docker >/dev/null
    args+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
    [ -f "$HOME/.Xauthority" ] && args+=(-v "$HOME/.Xauthority:/tmp/home/.Xauthority:ro")
fi

exec docker run "${args[@]}" "$IMAGE" "$@"
