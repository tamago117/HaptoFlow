"""Base converter class."""
import os
import json
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from src.dataset_converter.converters.dataset_info import write_dataset_info
from src.dataset_converter.converters.global_peak import compute_global_peak
from src.dataset_converter.converters.lowpass import LowpassFilter
from src.dataset_converter.file_utils import find_input_files, ensure_dir, confirm_delete
from src.dataset_converter.silence_padding import SilencePadding
from src.dataset_converter.waveform import trim_waveform, estimate_sample_rate, resample_waveform
from src.dataset_converter.io_utils import (
	build_recording_frame,
	compute_window_statistics,
	save_recording_parquet,
)


class BaseConverter(ABC):
	# Source dataset this converter reads; written to dataset_info.json as ``source``.
	source: ClassVar[str]

	def __init__(self, cfg: Dict):
		self.cfg = cfg
		
		input_cfg = cfg.get("input", {})
		self.input_dir = input_cfg.get("dir", "datasets/input")
		self.pattern = input_cfg.get("pattern") or "*"
		self.limit = input_cfg.get("limit", None)
		# Audio decode rate, and the fallback when a rate can't be estimated from timestamps
		self.sample_rate = int(input_cfg.get("sample_rate", 22000))
		
		output_cfg = cfg.get("output", {})
		output_dir = output_cfg.get("dir")
		if output_dir is None:
			output_dir = f"{self.input_dir}_converted"
		self.output_dir = output_dir
		self.group_by_label = bool(output_cfg.get("group_by_label", True))
		
		preprocessing_cfg = cfg.get("preprocessing", {})
		self.trim_start = float(preprocessing_cfg.get("trim_start", 0.0))
		self.trim_end = float(preprocessing_cfg.get("trim_end", 0.0))
		self.global_peak_normalize = bool(preprocessing_cfg.get("global_peak_normalize", False))
		self.global_peak_epsilon = float(preprocessing_cfg.get("global_peak_epsilon", 1e-8))
		self.global_peak_ignore_outliers = bool(preprocessing_cfg.get("global_peak_ignore_outliers", False))
		self.global_peak_outlier_percentile = float(preprocessing_cfg.get("global_peak_outlier_percentile", 99.9))
		self.global_peak_value = preprocessing_cfg.get("global_peak_value", None)
		if self.global_peak_value is not None:
			try:
				self.global_peak_value = float(self.global_peak_value)
			except (TypeError, ValueError):
				self.global_peak_value = None
		
		# Lowpass filter settings (kept as plain attrs for dataset_info + parallel pickling)
		self.lowpass_enabled = bool(preprocessing_cfg.get("lowpass_enabled", False))
		self.lowpass_cutoff_hz = preprocessing_cfg.get("lowpass_cutoff_hz", None)
		self.lowpass_order = preprocessing_cfg.get("lowpass_order", None)
		if self.lowpass_order is not None:
			try:
				self.lowpass_order = int(self.lowpass_order)
			except (TypeError, ValueError):
				self.lowpass_order = None
		self.lowpass_type = (preprocessing_cfg.get("lowpass_type", "butter") or "butter").lower()  # butter or fir
		
		self.target_sample_rate = preprocessing_cfg.get("target_sample_rate", None)
		if self.target_sample_rate is not None:
			try:
				self.target_sample_rate = float(self.target_sample_rate)
				if self.target_sample_rate <= 0:
					raise ValueError("target_sample_rate must be positive")
			except (TypeError, ValueError):
				self.target_sample_rate = None
		
		# Segmentation: the segment length the model generates. Recordings are stored
		# whole; stats.jsonl uses non-overlapping windows of it.
		segmentation_cfg = cfg.get("segmentation", {})
		if segmentation_cfg.get("duration") is None:
			raise ValueError("segmentation.duration is required")
		self.segment_duration = float(segmentation_cfg["duration"])

		# Zero-amplitude rest / contact / air states padded onto recordings
		self.silence_padding = SilencePadding(cfg.get("silence_padding"))

		self.recording_lengths: List[int] = []
		self.segment_len: Optional[int] = None
		# Unique labels collected during run() for dataset_info
		self.labels = set()

		self._lowpass = LowpassFilter(
			enabled=self.lowpass_enabled,
			cutoff_hz=self.lowpass_cutoff_hz,
			order=self.lowpass_order,
			filter_type=self.lowpass_type,
			default_sample_rate=float(self.sample_rate),
		)

	def find_inputs(self) -> List[str]:
		return find_input_files(self.input_dir, self.pattern, self.limit)

	@abstractmethod
	def read_input(self, file_path: str) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], str]]:
		"""
		Reads input file and returns a list of streams to process.
		Returns: List of (waveform, timestamps, metadata, filename_suffix)
		filename_suffix is used to distinguish streams from same file (e.g. _X, _Y, _Z).
		"""
		pass

	def get_dataset_info_extras(self) -> Dict[str, Any]:
		return {}
	
	def _ensure_lowpass_filter(self, sample_rate: Optional[float] = None) -> None:
		"""Design lowpass coefficients on first use (delegates to ``self._lowpass``)."""
		self._lowpass._design(sample_rate)

	def _apply_lowpass(self, waveform: np.ndarray, sample_rate: Optional[float] = None) -> np.ndarray:
		"""Apply lowpass filter (delegates to ``self._lowpass``)."""
		return self._lowpass.apply(waveform, sample_rate)
	
	def _compute_global_peak(self, inputs: List[str]) -> float:
		"""Delegate to ``global_peak.compute_global_peak``; same trim/lowpass pipeline as the main pass."""
		return compute_global_peak(self, inputs)

	def segment_samples(self, sample_rate: float) -> int:
		"""Samples per segment at ``sample_rate`` (round avoids float truncation)."""
		return int(np.round(self.segment_duration * sample_rate))

	def process_single_file(self, file_path: str) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
		"""
		Process one input file; each stream is written as one recording parquet.

		Designed to run in a worker process.

		Returns:
			(records, window_stats, stats): one meta record per recording, the
			``stats.jsonl`` rows of its non-overlapping segment windows, and run stats.
		"""
		records: List[Dict] = []
		window_stats: List[Dict] = []
		stats: Dict[str, Any] = {
			"recording_lengths": [],
			"segment_lens": set(),
		}

		try:
			streams = self.read_input(file_path)
		except Exception as e:
			print(f"Error reading {file_path}: {e}")
			return records, window_stats, stats

		if not streams:
			return records, window_stats, stats

		base_name_original = os.path.splitext(os.path.basename(file_path))[0]

		for waveform, timestamps, metadata, suffix in streams:
			base_name = base_name_original + suffix
			label = str(metadata.get("label", "unknown"))

			current_sample_rate = self.sample_rate
			original_sr = None
			try:
				original_sr = estimate_sample_rate(timestamps)
				# Round to avoid floating-point noise
				current_sample_rate = int(np.round(original_sr))
			except Exception:
				pass

			# Lowpass at the original sample rate, before resampling
			if self.lowpass_enabled:
				waveform = self._apply_lowpass(waveform, sample_rate=original_sr if original_sr is not None else self.sample_rate)

			if self.target_sample_rate is not None and original_sr is not None:
				try:
					if abs(original_sr - self.target_sample_rate) > 1e-6:
						waveform, timestamps, metadata = resample_waveform(
							waveform, timestamps, original_sr, self.target_sample_rate, metadata
						)
						current_sample_rate = self.target_sample_rate
				except Exception as e:
					print(f"Warning: Resampling failed for {base_name}: {e}")
					# Continue with original sample rate (estimated or configured)

			if self.trim_start > 0 or self.trim_end > 0:
				try:
					waveform, timestamps, metadata = trim_waveform(
						waveform, timestamps, current_sample_rate,
						self.trim_start, self.trim_end, metadata
					)
				except ValueError as e:
					print(f"Warning: Skipping {base_name} - {e}")
					continue

			if self.global_peak_normalize and self.global_peak_value and self.global_peak_value > self.global_peak_epsilon:
				waveform = waveform / self.global_peak_value
				# If a local peak still exceeds [-1, 1] after global normalization,
				# rescale the whole recording so it stays within range
				try:
					local_peak = float(np.max(np.abs(waveform)))
				except Exception as e:
					print(f"Warning: Local peak computation failed for {base_name}: {e}")
					local_peak = None

				if local_peak is not None and local_peak > 1.0 + self.global_peak_epsilon:
					waveform = waveform / local_peak
			elif self.global_peak_normalize and (self.global_peak_value is None or self.global_peak_value <= self.global_peak_epsilon):
				print("Warning: Global peak normalization enabled but global peak is not valid; skipping normalization.")

			segment_len = self.segment_samples(current_sample_rate)
			if len(waveform) < segment_len:
				print(f"Warning: Skipping {base_name} - shorter than one segment ({len(waveform)} < {segment_len} samples)")
				continue

			waveform, timestamps, metadata, synthetic = self.silence_padding.apply(
				waveform, timestamps, metadata, segment_len, current_sample_rate, key=f"{label}/{base_name}"
			)

			raw_signal_dir = os.path.join(self.output_dir, "raw_signal", label) if self.group_by_label else os.path.join(self.output_dir, "raw_signal")
			ensure_dir(raw_signal_dir)
			dst_file_path = os.path.join(raw_signal_dir, f"{base_name}.parquet")
			df = build_recording_frame(timestamps, waveform, metadata, synthetic)
			try:
				save_recording_parquet(df, dst_file_path)
			except Exception as e:
				print(f"Error saving parquet: {e}")
				continue
			rel_parquet = os.path.relpath(dst_file_path, start=self.output_dir)

			rows = compute_window_statistics(df, segment_len, rel_parquet, label)

			records.append({
				"label": label,
				**{k: v for k, v in metadata.items() if not isinstance(v, (np.ndarray, list))},  # Scalar meta
				"source_parquet": rel_parquet,
				"sample_rate": current_sample_rate,
				"num_samples": int(len(df)),
			})
			window_stats.extend(rows)
			stats["recording_lengths"].append(int(len(df)))
			stats["segment_lens"].add(segment_len)

		return records, window_stats, stats

	def run(self):
		inputs = self.find_inputs()
		if not inputs:
			raise FileNotFoundError(f"No files matched {os.path.join(self.input_dir, self.pattern)}")

		if not confirm_delete(self.output_dir):
			print("Operation cancelled.")
			return

		if self.lowpass_enabled:
			self._ensure_lowpass_filter()

		if self.global_peak_normalize:
			preproc_cfg = self.cfg.get("preprocessing", {})
			self.global_peak_value = self._compute_global_peak(inputs)
			preproc_cfg["global_peak_value"] = self.global_peak_value
			self.cfg["preprocessing"] = preproc_cfg

			if self.global_peak_value is None or self.global_peak_value <= self.global_peak_epsilon:
				print(f"Warning: Global peak ({self.global_peak_value}) is below epsilon ({self.global_peak_epsilon}); normalization will be skipped.")
			else:
				print(f"Computed global peak: {self.global_peak_value}")

		ensure_dir(self.output_dir)
		output_cfg = self.cfg.get("output", {})
		meta_path = output_cfg.get("meta_path") or os.path.join(self.output_dir, "meta.jsonl")
		dataset_info_path = output_cfg.get("dataset_info_path") or os.path.join(self.output_dir, "dataset_info.json")
		stats_path = os.path.join(self.output_dir, "stats.jsonl")

		os.makedirs(os.path.dirname(meta_path), exist_ok=True)

		parallel_cfg = self.cfg.get("parallel", {})
		num_workers = parallel_cfg.get("num_workers") or cpu_count()
		use_parallel = parallel_cfg.get("enabled", True) and len(inputs) > 1 and num_workers > 1

		desc = f"Converting dataset ({self.__class__.__name__})"
		segment_lens = set()
		with open(meta_path, "w", encoding="utf-8") as meta_f, open(stats_path, "w", encoding="utf-8") as stats_f:
			def _collect(records, window_stats, stats):
				for record in records:
					meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
					self.labels.add(str(record["label"]))
				for row in window_stats:
					stats_f.write(json.dumps(row, ensure_ascii=False) + "\n")
				self.recording_lengths.extend(stats["recording_lengths"])
				segment_lens.update(stats["segment_lens"])

			if use_parallel:
				# Imported here to avoid a circular import
				from src.dataset_converter.parallel import _process_file_parallel

				args_list = [(file_path, self.cfg, self.__class__.__name__) for file_path in inputs]
				with Pool(processes=num_workers) as pool:
					for result in tqdm(
						pool.imap(_process_file_parallel, args_list),
						total=len(inputs),
						desc=f"{desc} [Parallel: {num_workers} workers]",
						unit="file",
					):
						_collect(*result)
			else:
				for file_path in tqdm(inputs, desc=desc, unit="file"):
					_collect(*self.process_single_file(file_path))

		if len(segment_lens) > 1:
			raise ValueError(f"Recordings ended up at different sample rates (segment lengths {sorted(segment_lens)})")
		self.segment_len = next(iter(segment_lens)) if segment_lens else None
		print(f"Wrote meta JSONL to {meta_path}")
		print(f"Wrote statistics JSONL to {stats_path}")

		self.write_dataset_info(dataset_info_path)

	def write_dataset_info(self, path: str):
		"""Write ``dataset_info.json``; schema in ``dataset_info.build_dataset_info``."""
		write_dataset_info(self, path)
