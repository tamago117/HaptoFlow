# Dataset converter

`scripts/dataset/convert_dataset.py` turns the raw dataset into the training
format. All options are in
[configs/dataset/convert_dataset.yaml](../configs/dataset/convert_dataset.yaml).

For each recording it trims the ends, low-pass filters, resamples
(`target_sample_rate`), normalizes by the dataset-wide peak and attaches the
force / velocity signals. For accelerometer data, the three axes are combined
into one with DFT321 (`accel_transform`).

Key options:

- `cluster_dataset.sensor_type`: `accel` or `audio`.
- `segmentation.duration`: segment length the model generates (0.1 s).
- `silence_padding`: pads zero-amplitude rest / contact / air states onto
  recording ends so the model also learns silence.

## Output

Written to `datasets/texture_dataset_<sensor_type>_converted/`:

- `raw_signal/`: one `.parquet` per recording (`timestamp`, `force`, `velocity_x`,
  `velocity_y`, `amplitude`, `synthetic`)
- `meta.jsonl`: one record per recording
- `stats.jsonl`: force / velocity statistics per segment window
- `dataset_info.json`: labels, preprocessing and segment length
