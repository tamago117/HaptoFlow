"""Reading the packed Cluster Haptic Texture Dataset.

The dataset is distributed as ``.parquet`` sensor tables and ``.flac``
recordings.  Raw ``.csv`` / ``.wav`` input is not supported.

Parquet tables store the timestamp as int64 nanoseconds in ``time_ns``.
:func:`read_sensor_frame` converts it back to a float ``time`` column in
seconds, so callers keep the schema the original CSV files had.
"""
import os
from typing import Optional

import pandas as pd

SENSOR_TABLE_EXT = ".parquet"
AUDIO_EXT = ".flac"

SENSOR_TABLE_PATTERN = f"**/*{SENSOR_TABLE_EXT}"
AUDIO_PATTERN = f"**/*{AUDIO_EXT}"

TIME_COLUMN = "time"
TIME_NS_COLUMN = "time_ns"


def read_sensor_frame(path: str) -> pd.DataFrame:
    """Read a packed sensor table, restoring ``time`` in seconds.

    The returned frame has the same columns as the original CSV recording:
    ``time`` plus the sensor value columns (``X``/``Y``/``Z`` for accel,
    ``force``, ``X``/``Y`` for position).
    """
    df = pd.read_parquet(path)
    if TIME_NS_COLUMN in df.columns:
        time_s = df[TIME_NS_COLUMN].to_numpy() / 1e9
        df = df.drop(columns=[TIME_NS_COLUMN])
        df.insert(0, TIME_COLUMN, time_s)
    return df


def sibling_sensor_path(
    sensor_data_dir: str,
    sensor: str,
    texture_id: str,
    name_no_ext: str,
) -> str:
    """Path of the companion table for one recording under ``sensor_data/``.

    Example: the force recording that accompanies
    ``sensor_data/accel/0/0_135_40_500_0.parquet`` is
    ``sensor_data/force/0/0_135_40_500_0.parquet``.
    """
    return os.path.join(sensor_data_dir, sensor, texture_id, f"{name_no_ext}{SENSOR_TABLE_EXT}")
