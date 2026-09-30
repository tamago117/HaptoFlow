# Cluster Haptic Texture Dataset

The dataset is hosted on the Hugging Face Hub at
[tamago117/cluster-haptic-texture-dataset](https://huggingface.co/datasets/tamago117/cluster-haptic-texture-dataset).

## Download

```bash
uv run scripts/dataset/download_dataset.py                      # everything (~10 GB) into datasets/texture_dataset/
uv run scripts/dataset/download_dataset.py --subset accel,force,position
uv run scripts/dataset/download_dataset.py --textures 0,38,49   # only some textures
uv run scripts/dataset/download_dataset.py --list-subsets
```

Other options: `--local-dir`, `--revision <tag|commit>`, `--dry-run`.

| subset | size |
|---|---|
| `audio` | 1.4 GB |
| `raw_audio` | 2.5 GB |
| `accel` | 3.5 GB |
| `force` | 2.4 GB |
| `position` | 80 MB |
| `images` | 137 MB |

Training on accelerometer data needs `accel`, `force` and `position`.

## File format

Sensor data are `.parquet` tables (timestamp in `time_ns`, int64 nanoseconds;
read them with `src.utils.data.sensor_io.read_sensor_frame()`), and audio
recordings are 16-bit `.flac`.

## Dataset location

`src.utils.data.hf_dataset.resolve_dataset_root()` uses `$HAPTOFLOW_DATASET_ROOT`
if set, otherwise `datasets/texture_dataset`.
