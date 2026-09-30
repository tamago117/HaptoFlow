"""
Dataset index used to pick the initial previous waveform at inference start
(``initial_prev_mode == "dataset"``), when prev_waveform_buffer is None.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np


def build_prev_waveform_index_from_stats(
    stats_path: str,
    label_to_int: Dict[str, int],
    control_stats: Dict[str, Dict[str, float]],
    subsample_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    """
    Build an index for previous waveform selection from stats.jsonl.

    Each line in stats.jsonl is expected to have:
      - "label": material label (string)
      - "columns.force.mean", "columns.velocity_x.mean", "columns.velocity_y.mean"
      - "source_file": relative path to the recording parquet
      - "start_idx": first sample of the window within that recording

    Synthetic (silence-padded) windows are skipped. The index is subsampled by
    subsample_ratio to avoid excessive memory usage.

    Returns:
        labels:  np.ndarray[int32]   shape (M,)
        controls: np.ndarray[float32] shape (M, 3)  (normalized [force, vx, vy])
        windows: list[(str, int)]    length M, (recording parquet path, window start)
    """
    stats_path = str(stats_path)
    if not os.path.exists(stats_path):
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float32), []

    subsample_ratio = float(subsample_ratio)
    if subsample_ratio <= 0.0 or subsample_ratio > 1.0:
        subsample_ratio = 1.0

    # control_stats: expect keys "force", "velocity_x", "velocity_y"
    def _get_norm_params(key: str) -> Tuple[float, float]:
        s = control_stats.get(key, {})
        mean = float(s.get("mean", 0.0))
        std = float(s.get("std", 1.0)) or 1.0
        return mean, std

    f_mean, f_std = _get_norm_params("force")
    vx_mean, vx_std = _get_norm_params("velocity_x")
    vy_mean, vy_std = _get_norm_params("velocity_y")

    labels: List[int] = []
    controls: List[List[float]] = []
    windows: List[Tuple[str, int]] = []

    t0 = time.time()
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if random.random() > subsample_ratio:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("synthetic"):
                continue

            label_str = rec.get("label")
            if label_str is None:
                continue
            if label_str not in label_to_int:
                continue
            label_id = int(label_to_int[label_str])

            cols = rec.get("columns", {})
            force_stats = cols.get("force", {})
            vx_stats = cols.get("velocity_x", {})
            vy_stats = cols.get("velocity_y", {})

            try:
                f_mean_raw = float(force_stats.get("mean"))
                vx_mean_raw = float(vx_stats.get("mean"))
                vy_mean_raw = float(vy_stats.get("mean"))
            except (TypeError, ValueError):
                continue

            f_n = (f_mean_raw - f_mean) / f_std
            vx_n = (vx_mean_raw - vx_mean) / vx_std
            vy_n = (vy_mean_raw - vy_mean) / vy_std

            src = rec.get("source_file")
            if not isinstance(src, str) or not src or "start_idx" not in rec:
                continue

            labels.append(label_id)
            controls.append([float(f_n), float(vx_n), float(vy_n)])
            windows.append((src, int(rec["start_idx"])))

    labels_arr = np.asarray(labels, dtype=np.int32)
    controls_arr = np.asarray(controls, dtype=np.float32).reshape(-1, 3) if controls else np.zeros((0, 3), dtype=np.float32)

    elapsed = time.time() - t0
    print(
        f"[prev_dataset_index] build time: {elapsed:.3f} s, "
        f"subsample_ratio={subsample_ratio}, entries={labels_arr.shape[0]}"
    )

    return labels_arr, controls_arr, windows
