"""I/O utilities for saving converter output."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def build_recording_frame(
	timestamps: np.ndarray,
	amplitudes: np.ndarray,
	metadata: Dict[str, Any],
	synthetic: np.ndarray,
) -> pd.DataFrame:
	"""Build the per-recording table: ``timestamp``, per-sample metadata arrays, ``amplitude``, ``synthetic``.

	Scalar metadata is not repeated per row; it lives in ``meta.jsonl``.
	``synthetic`` flags padded samples (see ``silence_padding``).
	"""
	if len(timestamps) != len(amplitudes):
		raise ValueError("timestamps and amplitudes must have the same length")

	data: Dict[str, Any] = {"timestamp": np.round(np.asarray(timestamps, dtype=np.float64), 6).astype(np.float32)}
	for k, v in metadata.items():
		if isinstance(v, (np.ndarray, list)) and len(v) == len(timestamps):
			arr = np.asarray(v, dtype=np.float32)
			if k in ("velocity_x", "velocity_y"):
				arr = np.round(arr, 2).astype(np.float32)
			data[k] = arr
	data["amplitude"] = np.asarray(amplitudes, dtype=np.float32)
	if len(synthetic) != len(timestamps):
		raise ValueError("synthetic must have the same length as timestamps")
	data["synthetic"] = np.asarray(synthetic, dtype=bool)
	return pd.DataFrame(data)


def save_recording_parquet(df: pd.DataFrame, output_path: str) -> None:
	"""Write a recording table built by ``build_recording_frame``."""
	df.to_parquet(output_path, index=False, compression="snappy")


def compute_window_statistics(
	df: pd.DataFrame,
	segment_len: int,
	source_file: str,
	label: Optional[str] = None,
) -> List[Dict[str, Any]]:
	"""Per-window statistics over non-overlapping ``segment_len`` windows of a recording.

	One row per window: ``source_file``, ``start_idx``, ``num_rows``,
	``synthetic`` (any padded sample) and ``columns.<col>.{min,max,mean}`` for
	every numeric column. A trailing partial window is dropped.
	"""
	num_windows = len(df) // segment_len
	if num_windows <= 0:
		return []

	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	values = df[numeric_cols].to_numpy(dtype=np.float64)[: num_windows * segment_len]
	values = values.reshape(num_windows, segment_len, len(numeric_cols))
	mins = np.round(values.min(axis=1), 6)
	maxs = np.round(values.max(axis=1), 6)
	means = np.round(values.mean(axis=1), 6)
	synthetic = df["synthetic"].to_numpy(dtype=bool)[: num_windows * segment_len].reshape(num_windows, segment_len).any(axis=1)

	rows = []
	for w in range(num_windows):
		row: Dict[str, Any] = {
			"source_file": source_file,
			"start_idx": w * segment_len,
			"num_rows": segment_len,
			"synthetic": bool(synthetic[w]),
			"columns": {
				col: {"min": float(mins[w, c]), "max": float(maxs[w, c]), "mean": float(means[w, c])}
				for c, col in enumerate(numeric_cols)
			},
		}
		if label is not None:
			row["label"] = label
		rows.append(row)
	return rows
