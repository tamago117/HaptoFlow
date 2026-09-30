"""Synthetic zero-amplitude states padded onto recordings.

The source datasets never hold still or lift off, so a model has no example of
"no vibration". Padding a recording's head / tail with short states teaches
silence and the transitions into and out of it:

- ``rest``: no contact, not moving (force ~ 0, velocity 0)
- ``contact``: pressing without moving (force of the adjacent real data, velocity 0)
- ``air``: moving without contact (force ~ 0, velocity of the adjacent real data)

Padded samples have amplitude 0 and are flagged in the ``synthetic`` column.
"""
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Head states end next to the real data, tail states start next to it
_HEAD_PATTERNS = (("rest",), ("rest", "contact"), ("contact",), ("air",))
_TAIL_PATTERNS = (("rest",), ("contact", "rest"), ("contact",), ("air",))
_CONTROL_KEYS = ("force", "velocity_x", "velocity_y")


class SilencePadding:
	"""``silence_padding`` converter config; absent or ``enabled: false`` disables it."""

	def __init__(self, cfg: Optional[Dict[str, Any]]):
		cfg = cfg or {}
		self.enabled = bool(cfg.get("enabled", False))
		self.prob = float(cfg.get("prob", 0.3))
		lo, hi = cfg.get("state_segments", [1, 3])
		self.state_segments = (int(lo), int(hi))
		self.force_off = float(cfg.get("force_off", 0.1))
		self.fade_ms = float(cfg.get("fade_ms", 10.0))
		self.seed = int(cfg.get("seed", 0))
		if not 0.0 <= self.prob <= 1.0:
			raise ValueError(f"silence_padding.prob must be in [0, 1], got {self.prob}")
		if not 1 <= self.state_segments[0] <= self.state_segments[1]:
			raise ValueError(f"silence_padding.state_segments must be 1 <= lo <= hi, got {self.state_segments}")

	def info(self) -> Dict[str, Any]:
		"""Settings for ``dataset_info.json``."""
		return {
			"enabled": self.enabled,
			"prob": self.prob,
			"state_segments": list(self.state_segments),
			"force_off": self.force_off,
			"fade_ms": self.fade_ms,
			"seed": self.seed,
		}

	def apply(
		self,
		waveform: np.ndarray,
		timestamps: np.ndarray,
		metadata: Dict[str, Any],
		segment_len: int,
		sample_rate: float,
		key: str,
	) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
		"""Pad one recording; ``key`` seeds its draws so reruns are identical.

		Returns ``(waveform, timestamps, metadata, synthetic)``. ``metadata`` gains
		``pad_head`` / ``pad_tail`` (samples); ``synthetic`` is the per-sample flag.
		"""
		n = len(waveform)
		synthetic = np.zeros(n, dtype=bool)
		if not self.enabled or any(k not in metadata for k in _CONTROL_KEYS):
			return waveform, timestamps, {**metadata, "pad_head": 0, "pad_tail": 0}, synthetic

		rng = np.random.default_rng([self.seed, zlib.crc32(key.encode())])
		edge = min(segment_len, n)
		head_ctrl = {k: float(np.mean(metadata[k][:edge])) for k in _CONTROL_KEYS}
		tail_ctrl = {k: float(np.mean(metadata[k][-edge:])) for k in _CONTROL_KEYS}
		head = self._draw(rng, _HEAD_PATTERNS, head_ctrl, segment_len)
		tail = self._draw(rng, _TAIL_PATTERNS, tail_ctrl, segment_len)
		n_head = len(head["force"]) if head else 0
		n_tail = len(tail["force"]) if tail else 0
		if n_head == 0 and n_tail == 0:
			return waveform, timestamps, {**metadata, "pad_head": 0, "pad_tail": 0}, synthetic

		waveform = self._fade(np.asarray(waveform, dtype=np.float32).copy(), n_head > 0, n_tail > 0, sample_rate)
		out_meta = dict(metadata)
		for k in _CONTROL_KEYS:
			parts = [head[k]] if head else []
			parts.append(np.asarray(metadata[k], dtype=np.float32))
			if tail:
				parts.append(tail[k])
			out_meta[k] = np.concatenate(parts)
		out_meta["pad_head"] = n_head
		out_meta["pad_tail"] = n_tail

		waveform = np.concatenate([np.zeros(n_head, np.float32), waveform, np.zeros(n_tail, np.float32)])
		dt = 1.0 / float(sample_rate)
		t0 = float(timestamps[0])
		timestamps = np.concatenate([
			t0 + dt * np.arange(-n_head, 0),
			np.asarray(timestamps, dtype=np.float64),
			float(timestamps[-1]) + dt * np.arange(1, n_tail + 1),
		])
		timestamps = timestamps - timestamps[0]
		synthetic = np.concatenate([np.ones(n_head, bool), synthetic, np.ones(n_tail, bool)])
		return waveform, timestamps, out_meta, synthetic

	def _draw(
		self,
		rng: np.random.Generator,
		patterns: Tuple[Tuple[str, ...], ...],
		edge_ctrl: Dict[str, float],
		segment_len: int,
	) -> Optional[Dict[str, np.ndarray]]:
		"""Control arrays of one padded end, or None when this end is not padded."""
		if rng.random() >= self.prob:
			return None
		pattern = patterns[int(rng.integers(len(patterns)))]
		lo, hi = self.state_segments
		parts: Dict[str, list] = {k: [] for k in _CONTROL_KEYS}
		for state in pattern:
			length = int(rng.integers(lo, hi + 1)) * segment_len
			off_force = float(rng.uniform(0.0, self.force_off))
			if state == "rest":
				values = (off_force, 0.0, 0.0)
			elif state == "contact":
				values = (edge_ctrl["force"], 0.0, 0.0)
			else:  # air
				values = (off_force, edge_ctrl["velocity_x"], edge_ctrl["velocity_y"])
			for k, v in zip(_CONTROL_KEYS, values):
				parts[k].append(np.full(length, v, dtype=np.float32))
		return {k: np.concatenate(v) for k, v in parts.items()}

	def _fade(self, waveform: np.ndarray, fade_in: bool, fade_out: bool, sample_rate: float) -> np.ndarray:
		"""Raised-cosine fade of the real signal where it meets a padded end."""
		m = min(int(round(self.fade_ms * 1e-3 * sample_rate)), len(waveform) // 2)
		if m <= 0:
			return waveform
		ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)).astype(np.float32)
		if fade_in:
			waveform[:m] *= ramp
		if fade_out:
			waveform[-m:] *= ramp[::-1]
		return waveform
