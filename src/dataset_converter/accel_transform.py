"""Acceleration signal transformation utilities (e.g., DFT321)."""

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt


def dft321_reduce(
	ax: np.ndarray,
	ay: np.ndarray,
	az: np.ndarray,
	fs: float,
	frame_ms: float = 35.0,
	hop_ms: float = 17.5,
	apply_window: bool = True,
	smooth_bins: int = 0,
	highpass_cutoff_hz: float = 10.0,
	highpass_order: int = 4,
) -> np.ndarray:
	"""
	Reduce 3-axis acceleration (ax, ay, az) to a single-axis signal using the
	DFT321 algorithm (Landin et al. 2010).

	Args:
		ax, ay, az:
			1D arrays of the same length N (time-domain acceleration signals).
		fs:
			Sampling rate in Hz.
		frame_ms:
			Frame length in milliseconds.
		hop_ms:
			Hop length in milliseconds.
		apply_window:
			If True, apply Hann window and perform overlap-add.
		smooth_bins:
			If > 1, apply simple moving average smoothing to magnitude spectrum
			with given kernel width (in FFT bins).
		highpass_cutoff_hz:
			High-pass filter cutoff frequency in Hz. If <= 0, high-pass
			filtering is disabled.
		highpass_order:
			Order of the Butterworth high-pass filter.

	Returns:
		1D numpy array of shape (N,), single-axis reduced signal.
	"""
	assert ax.shape == ay.shape == az.shape
	N = ax.shape[0]
	if N == 0:
		return np.zeros(0, dtype=np.float32)

	L = int(round(fs * frame_ms / 1000.0))  # frame length in samples
	H = int(round(fs * hop_ms / 1000.0))  # hop length in samples
	L = max(L, 2)
	H = max(H, 1)

	# Optional high-pass filtering to remove low-frequency components (e.g., gravity)
	def _apply_highpass(x: np.ndarray) -> np.ndarray:
		if highpass_cutoff_hz is None or highpass_cutoff_hz <= 0:
			return x
		nyq = 0.5 * fs
		if highpass_cutoff_hz >= nyq:
			# Invalid cutoff; skip high-pass
			return x
		try:
			b, a = butter(highpass_order, highpass_cutoff_hz / nyq, btype="highpass")
			if x.size < max(len(b), len(a)) * 3:
				# Too short for filtfilt; skip high-pass
				return x
			return filtfilt(b, a, x).astype(np.float64)
		except Exception:
			# Fail-safe: return original signal on any error
			return x

	ax_hp = _apply_highpass(ax.astype(np.float64))
	ay_hp = _apply_highpass(ay.astype(np.float64))
	az_hp = _apply_highpass(az.astype(np.float64))

	out = np.zeros(N, dtype=np.float64)
	wsum = np.zeros(N, dtype=np.float64)

	if apply_window:
		win = np.hanning(L).astype(np.float64)
	else:
		win = np.ones(L, dtype=np.float64)

	kernel: Optional[np.ndarray] = None
	if smooth_bins and smooth_bins > 1:
		kernel = np.ones(smooth_bins, dtype=np.float64) / float(smooth_bins)

	def maybe_smooth_spectrum(A: np.ndarray) -> np.ndarray:
		"""Optionally smooth magnitude spectrum while preserving phase."""
		nonlocal kernel
		if kernel is None:
			return A
		mag = np.abs(A)
		phase = np.angle(A)
		mag_s = np.convolve(mag, kernel, mode="same")
		return mag_s * np.exp(1j * phase)

	for start in range(0, N - L + 1, H):
		sl = slice(start, start + L)

		x = ax_hp[sl] * win
		y = ay_hp[sl] * win
		z = az_hp[sl] * win

		Ax = np.fft.rfft(x)
		Ay = np.fft.rfft(y)
		Az = np.fft.rfft(z)

		Ax = maybe_smooth_spectrum(Ax)
		Ay = maybe_smooth_spectrum(Ay)
		Az = maybe_smooth_spectrum(Az)

		# Combined magnitude: root of the summed per-axis power
		mag = np.sqrt(np.abs(Ax) ** 2 + np.abs(Ay) ** 2 + np.abs(Az) ** 2)

		# Phase of the summed axis spectra
		s = Ax + Ay + Az
		theta = np.angle(s)

		As = mag * np.exp(1j * theta)

		s_frame = np.fft.irfft(As, n=L)

		# Overlap-add (use same window on synthesis side)
		out[sl] += s_frame * win
		wsum[sl] += win ** 2

	# Normalize by window overlap
	eps = 1e-12
	valid = wsum > eps
	out_final = np.zeros_like(out)
	out_final[valid] = out[valid] / wsum[valid]

	return out_final.astype(np.float32)



ACCEL_AXES = ("x", "y", "z", "norm")


def normalize_accel_transform(value):
	"""Normalize the ``accel_transform`` config value.

	Returns ``"dft321"``, a list of lower-case axis names, or ``None`` (all four
	streams). ``"none"`` is accepted with a deprecation warning.
	"""
	if isinstance(value, str):
		value = value.lower()
		if value == "none":
			print("Warning: accel_transform: 'none' is deprecated. Use ['X', 'Y', 'Z', 'Norm'] or omit the setting for default behavior.")
			return None
		if value != "dft321":
			raise ValueError(f"Invalid accel_transform '{value}'. Use 'dft321' or a list of X, Y, Z, Norm")
		return value
	if isinstance(value, (list, tuple)):
		axes = [str(axis).lower() for axis in value]
		for axis in axes:
			if axis not in ACCEL_AXES:
				raise ValueError(f"Invalid axis '{axis}' in accel_transform. Valid options: X, Y, Z, Norm")
		return axes
	return None


def select_accel_streams(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	fs: float,
	transform,
) -> list:
	"""Turn DC-removed 3-axis acceleration into ``[(waveform, suffix), ...]`` streams.

	``transform`` is the output of :func:`normalize_accel_transform`:
	``"dft321"`` → one ``_DFT321`` stream (may be shorter than the input; callers
	trim their timestamps to it), a non-empty axis list → those axes, and
	``None`` / ``[]`` → ``_X``, ``_Y``, ``_Z``, ``_Norm``.
	"""
	if transform == "dft321":
		return [(dft321_reduce(x, y, z, fs=fs).astype(np.float32), "_DFT321")]
	norm = np.sqrt(x**2 + y**2 + z**2)
	if not transform:
		return [(x, "_X"), (y, "_Y"), (z, "_Z"), (norm, "_Norm")]
	available = {"x": x, "y": y, "z": z, "norm": norm}
	return [(available[axis], f"_{axis.upper()}") for axis in transform]
