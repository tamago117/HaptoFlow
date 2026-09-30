"""Lowpass filter design + application, encapsulated in a small class.

Used by ``BaseConverter`` during peak scanning, resampling, and segmentation.
Caches the filter (b, a) coefficients per sample rate so repeated calls at the
same SR don't redesign the filter.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, firwin


class LowpassFilter:
    """Designed-on-first-use IIR/FIR lowpass with a one-slot cache.

    No-op when disabled; falls back to identity when the input is too short
    for filtfilt or scipy raises.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        cutoff_hz: Optional[float],
        order: Optional[int],
        filter_type: str = "butter",
        default_sample_rate: float = 16000.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.cutoff_hz = cutoff_hz
        self.order = order
        self.filter_type = (filter_type or "butter").lower()
        self.default_sample_rate = float(default_sample_rate)
        self._ba: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._sr: Optional[float] = None

    def _design(self, sample_rate: Optional[float] = None) -> None:
        """Design the (b, a) coefficients for ``sample_rate`` (or default), cached."""
        if not self.enabled:
            return

        sr = sample_rate if sample_rate is not None else self.default_sample_rate

        if self._ba is not None and self._sr is not None:
            if abs(self._sr - sr) < 1e-6:
                return  # already designed for this SR
            self._ba = None
            self._sr = None

        if self.cutoff_hz is None:
            raise ValueError(
                "preprocessing.lowpass_cutoff_hz must be set when lowpass_enabled is true."
            )

        try:
            cutoff_hz = float(self.cutoff_hz)
        except (TypeError, ValueError):
            raise ValueError("preprocessing.lowpass_cutoff_hz must be a number.") from None

        nyquist = 0.5 * sr
        if cutoff_hz <= 0 or cutoff_hz >= nyquist:
            raise ValueError(
                f"preprocessing.lowpass_cutoff_hz must be between 0 and Nyquist ({nyquist})."
            )

        if self.filter_type == "fir":
            order = self.order if self.order else 129
            if order % 2 == 0:
                order += 1  # firwin prefers odd taps for lowpass
            b = firwin(order, cutoff_hz / nyquist)
            a = np.array([1.0], dtype=np.float64)
        else:
            order = self.order if self.order else 4
            b, a = butter(order, cutoff_hz / nyquist, btype="lowpass")

        self._ba = (b.astype(np.float64), a.astype(np.float64))
        self._sr = sr

    def apply(self, waveform: np.ndarray, sample_rate: Optional[float] = None) -> np.ndarray:
        """Apply the lowpass filter. Returns the input unchanged when disabled or on failure."""
        if not self.enabled:
            return waveform

        sr = sample_rate if sample_rate is not None else self.default_sample_rate
        self._design(sr)
        if self._ba is None:
            return waveform

        b, a = self._ba
        if waveform.size < max(len(b), len(a)) * 3:
            print("Warning: Waveform too short for lowpass filtfilt; skipping lowpass.")
            return waveform

        try:
            return filtfilt(b, a, waveform).astype(np.float32)
        except Exception as e:
            print(f"Warning: Lowpass filtering failed: {e}")
            return waveform
