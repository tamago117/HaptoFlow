"""Shared value object for STFT/Mel parameters (metrics and visualization)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SpectralParams:
    """STFT/Mel parameters used by the spectral metrics and the spectrogram images."""
    stft_n_fft: int = 1024
    stft_hop_length: int = 256
    stft_win_length: int = 1024
    mel_n_mels: int = 80
    mel_f_min: float = 0.0
    mel_f_max: Optional[float] = None

    @classmethod
    def from_cfg(cls, cfg: Any) -> "SpectralParams":
        """Build SpectralParams from an OmegaConf cfg, falling back to defaults."""
        train_cfg = cfg.train
        stft_cfg = getattr(train_cfg, "stft", None)
        n_fft = int(getattr(stft_cfg, "n_fft", 1024)) if stft_cfg is not None else 1024
        win_length = int(getattr(stft_cfg, "win_length", n_fft)) if stft_cfg is not None else n_fft
        hop_length = int(getattr(stft_cfg, "hop_length", win_length // 4)) if stft_cfg is not None else win_length // 4

        mel_cfg = getattr(train_cfg, "mel", None)
        mel_f_max_raw = getattr(mel_cfg, "f_max", None) if mel_cfg is not None else None
        mel_f_max = float(mel_f_max_raw) if mel_f_max_raw is not None else None

        return cls(
            stft_n_fft=n_fft,
            stft_hop_length=hop_length,
            stft_win_length=win_length,
            mel_n_mels=int(getattr(mel_cfg, "n_mels", 80)) if mel_cfg is not None else 80,
            mel_f_min=float(getattr(mel_cfg, "f_min", 0.0)) if mel_cfg is not None else 0.0,
            mel_f_max=mel_f_max,
        )
