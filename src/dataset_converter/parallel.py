"""Parallel processing utilities."""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from src.dataset_converter.converters.cluster import ClusterDatasetConverter
from src.dataset_converter.waveform import trim_waveform, estimate_sample_rate, resample_waveform


def _process_file_parallel(args: Tuple[str, Dict, str]) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
	"""
	Parallel worker: build a converter from ``cfg`` and process one file.
	
	Args:
		args: (file_path, cfg, converter_class_name)
		
	Returns:
		(records, window_stats, stats): see ``BaseConverter.process_single_file``
	"""
	file_path, cfg, converter_class_name = args
	
	if converter_class_name == "ClusterDatasetConverter":
		converter = ClusterDatasetConverter(cfg)
	else:
		raise ValueError(f"Unknown converter class: {converter_class_name}")
	
	return converter.process_single_file(file_path)


def _compute_file_peak_parallel(args: Tuple[str, Dict, str]) -> List[float]:
	"""
	Parallel worker for the global peak scan: return the peak of each stream in one file.
	
	Args:
		args: (file_path, cfg, converter_class_name)
		
	Returns:
		Per-stream peak values
	"""
	file_path, cfg, converter_class_name = args
	
	if converter_class_name == "ClusterDatasetConverter":
		converter = ClusterDatasetConverter(cfg)
	else:
		raise ValueError(f"Unknown converter class: {converter_class_name}")
	
	per_stream_peaks: List[float] = []
	
	try:
		streams = converter.read_input(file_path)
	except Exception as e:
		print(f"Warning: Peak scan skipped {file_path} - {e}")
		return per_stream_peaks
	
	if not streams:
		return per_stream_peaks
	
	for waveform, timestamps, metadata, suffix in streams:
		wav = waveform
		ts = timestamps
		meta = metadata
		
		current_sample_rate = converter.sample_rate
		original_sr = None
		try:
			original_sr = estimate_sample_rate(ts)
			current_sample_rate = original_sr
		except Exception:
			pass
		
		# Lowpass before resampling (use original sample rate)
		if converter.lowpass_enabled:
			try:
				wav = converter._apply_lowpass(wav, sample_rate=original_sr if original_sr is not None else converter.sample_rate)
			except Exception as e:
				print(f"Warning: Peak scan lowpass failed on {file_path}{suffix}: {e}")
				continue
		
		if converter.target_sample_rate is not None and original_sr is not None:
			try:
				if abs(original_sr - converter.target_sample_rate) > 1e-6:
					wav, ts, meta = resample_waveform(
						wav, ts, original_sr, converter.target_sample_rate, meta
					)
					current_sample_rate = converter.target_sample_rate
			except Exception as e:
				print(f"Warning: Peak scan resampling failed for {file_path}{suffix}: {e}")
				# Continue with original sample rate
		
		if converter.trim_start > 0 or converter.trim_end > 0:
			try:
				wav, ts, meta = trim_waveform(
					wav, ts, current_sample_rate,
					converter.trim_start, converter.trim_end, meta
				)
			except ValueError as e:
				print(f"Warning: Peak scan skip {file_path}{suffix} - {e}")
				continue
		
		if converter.lowpass_enabled:
			try:
				wav = converter._apply_lowpass(wav)
			except Exception as e:
				print(f"Warning: Peak scan lowpass failed on {file_path}{suffix}: {e}")
				continue
		
		if wav.size == 0:
			continue
		
		try:
			peak = float(np.max(np.abs(wav)))
		except Exception as e:
			print(f"Warning: Peak scan failed on {file_path}{suffix}: {e}")
			continue
		
		per_stream_peaks.append(peak)
	
	return per_stream_peaks
