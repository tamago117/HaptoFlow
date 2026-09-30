# CLAUDE.md

Guidance for coding agents working in this repository. HaptoFlow generates
vibrotactile waveforms in real time with Flow Matching on EnCodec latents,
conditioned on a material label, stroking velocity and applied force. The user
guide is [README.md](README.md); details are in [docs/](docs/).

## Environment

- Python via [uv](https://docs.astral.sh/uv/): `uv sync`, then run everything with `uv run <script>`.
Dependencies are pinned in `uv.lock`; add or remove them with `uv add` / `uv remove`.
- Docker alternative: `docker/build.sh`, then `docker/run.sh <command>` (see [docs/setup.md](docs/setup.md)).
- Training and the app expect a CUDA GPU; CPU works but is slow.

## Common tasks

```bash
# Demo app (http://127.0.0.1:7860); pulls weights from hf://tamago117/HaptoFlow on first use
uv run scripts/serving/app_generate_waveform.py

# Data: download (~10 GB) -> convert to datasets/texture_dataset_accel_converted/
uv run scripts/dataset/download_dataset.py
uv run scripts/dataset/convert_dataset.py --config configs/dataset/convert_dataset.yaml

# Training (phase 1, then phase 2 from a phase-1 checkpoint)
uv run scripts/train.py --model flow_matching -r <run>
uv run scripts/train.py --model flow_matching --config configs/train/flow_matching_phase2.yaml \
    --init-from runs/<run>/epoch_0050 -r <run2>
```

- `-r <run>` sets the run name; checkpoints go to `runs/<run>/epoch_XXXX/`.
- `--resume <checkpoint dir>` continues an interrupted run (needs `training_state.pt`);
`--init-from` starts a new run from weights only and also accepts the published
weights, e.g. `hf://tamago117/HaptoFlow/haptoflow_accel`.
- Without `-r` on an interactive terminal, `train.py` asks to confirm the run name.
Always pass `-r` when running non-interactively.

## Configuration

- YAML configs support `_base: <relative path>` inheritance (deep merge). There are
no command-line overrides: to change settings, write a small YAML such as
`configs/train/my_run.yaml` starting with `_base: flow_matching.yaml` (path
relative to the new file) and pass it with `--config`.
- `configs/train/_base.yaml` holds the shared defaults (data, optimizer, W&B,
visualization); `flow_matching.yaml` / `flow_matching_phase2.yaml` are the two phases.
- `wandb.mode` defaults to `online` and takes precedence over the `WANDB_MODE`
environment variable. Set `wandb: {mode: disabled}` in a config for local runs
that should not log to W&B.
- `data.root` must point to a converted dataset (default
`datasets/texture_dataset_accel_converted`). For audio, set
`cluster_dataset.sensor_type: audio` in the converter config and `data.root` accordingly.

## Code layout


| Path                                       | Contents                                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `scripts/train.py`                         | Training entry point (`src/runtime/trainer.py`)                                            |
| `scripts/serving/app_generate_waveform.py` | Gradio demo app                                                                            |
| `scripts/dataset/`                         | Dataset download, conversion, label thumbnail export                                       |
| `src/models/`                              | UNet + Flow Matching model, EnCodec wrapper, embeddings                                    |
| `src/strategies/`                          | Per-model-family build / train / eval hooks (registry; `flow_matching` is the only family) |
| `src/utils/training_loops/flow_matching/`  | Training step, phase-2 chain generation                                                    |
| `src/inference/`                           | Inference engine used by the app and by validation-time generation                         |
| `src/dataset_converter/`                   | Raw dataset → training format                                                              |
| `src/datasets/`                            | PyTorch dataset reading the converted data                                                 |
| `assets/texture_label_images/`             | Material thumbnails shown in the app                                                       |


A checkpoint directory contains `meta.json` (full training config and dataset
metadata) and `model.safetensors`; `training_state.pt` is only needed for
`--resume`. The app lists `hf://` checkpoints and local ones under `checkpoints/`.

## Conventions

- Comments and docstrings are in English and kept minimal: explain non-obvious
reasons, not what the code does.
- Match the surrounding indentation: `src/dataset_converter/` and `scripts/dataset/`
mostly use tabs, everything else uses 4 spaces.
- `datasets/`, `runs/`, `wandb/`, `.output/` and `checkpoints/*` are git-ignored outputs.

