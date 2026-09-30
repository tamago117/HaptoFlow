"""Waveform processing utilities."""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import librosa


def trim_waveform(
	waveform: np.ndarray,
	timestamps: np.ndarray,
	sample_rate: int,
	trim_start: float,
	trim_end: float,
	metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
	"""Trim waveform and corresponding metadata arrays."""
	if len(waveform) != len(timestamps):
		raise ValueError("waveform and timestamps must have the same length")
	
	if trim_start < 0 or trim_end < 0:
		raise ValueError("trim_start and trim_end must be non-negative")
	
	trim_start_samples = int(trim_start * sample_rate)
	trim_end_samples = int(trim_end * sample_rate)
	
	total_trim_samples = trim_start_samples + trim_end_samples
	if total_trim_samples >= len(waveform):
		raise ValueError(
			f"Trimming {trim_start + trim_end} seconds would remove all samples. "
			f"Waveform duration: {len(waveform) / sample_rate:.3f} seconds"
		)
	
	end_idx = -trim_end_samples if trim_end_samples > 0 else None
	
	trimmed_waveform = waveform[trim_start_samples:end_idx].copy()
	trimmed_timestamps_original = timestamps[trim_start_samples:end_idx]
	
	# Restart timestamps at 0
	trimmed_timestamps = trimmed_timestamps_original - trimmed_timestamps_original[0]
	
	# Trim per-sample metadata arrays
	trimmed_metadata = {}
	if metadata:
		for k, v in metadata.items():
			if isinstance(v, (np.ndarray, list)) and len(v) == len(waveform):
				trimmed_metadata[k] = np.array(v)[trim_start_samples:end_idx]
			else:
				trimmed_metadata[k] = v
	
	return trimmed_waveform, trimmed_timestamps, trimmed_metadata


def estimate_sample_rate(timestamps: np.ndarray) -> float:
	"""
	Estimate sample rate from timestamps array.
	
	Args:
		timestamps: Array of timestamps in seconds
		
	Returns:
		Estimated sample rate in Hz
	"""
	if len(timestamps) < 2:
		raise ValueError("Need at least 2 timestamps to estimate sample rate")
	
	diffs = np.diff(timestamps)
	
	diffs = diffs[diffs > 0]
	
	if len(diffs) == 0:
		raise ValueError("No valid time differences found in timestamps")
	
	# Use median to be robust against outliers
	median_dt = np.median(diffs)
	
	sample_rate = 1.0 / median_dt
	
	return float(sample_rate)


def resample_waveform(
	waveform: np.ndarray,
	timestamps: np.ndarray,
	original_sample_rate: float,
	target_sample_rate: float,
	metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
	"""
	Resample waveform and timestamps to target sample rate.
	
	Args:
		waveform: Input waveform array
		timestamps: Corresponding timestamps array
		original_sample_rate: Original sample rate in Hz
		target_sample_rate: Target sample rate in Hz
		metadata: Optional metadata dictionary
		
	Returns:
		(resampled_waveform, resampled_timestamps, updated_metadata)
	"""
	if len(waveform) != len(timestamps):
		raise ValueError("waveform and timestamps must have the same length")
	
	if original_sample_rate <= 0 or target_sample_rate <= 0:
		raise ValueError("Sample rates must be positive")
	
	if abs(original_sample_rate - target_sample_rate) < 1e-6:
		return waveform.copy(), timestamps.copy(), (metadata.copy() if metadata else {})
	
	resampled_waveform = librosa.resample(
		waveform.astype(np.float32),
		orig_sr=original_sample_rate,
		target_sr=target_sample_rate
	).astype(np.float32)
	
	duration = len(waveform) / original_sample_rate
	new_length = len(resampled_waveform)
	
	resampled_timestamps = np.linspace(
		0.0,
		duration,
		new_length,
		dtype=np.float32
	)
	
	resampled_metadata = {}
	if metadata:
		for k, v in metadata.items():
			if isinstance(v, (np.ndarray, list)) and len(v) == len(waveform):
				# Interpolate per-sample metadata onto the new timestamps
				from scipy.interpolate import interp1d
				interp_func = interp1d(
					timestamps,
					np.array(v),
					kind='linear',
					bounds_error=False,
					fill_value=(v[0] if len(v) > 0 else 0, v[-1] if len(v) > 0 else 0)
				)
				resampled_metadata[k] = interp_func(resampled_timestamps).astype(np.float32)
			else:
				resampled_metadata[k] = v
	
	return resampled_waveform, resampled_timestamps, resampled_metadata

