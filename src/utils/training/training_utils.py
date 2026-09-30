"""Training utility functions."""
from typing import Dict
import numpy as np
import torch

from src.models.shared.embeddings import ControlEmbedding


def _compute_freq_spectrum_from_waveform(
    wf_np: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """Compute frequency spectrum (DFT amplitude) from waveform using STFT time averaging.
    
    Args:
        wf_np: Waveform as numpy array (1D or 2D).
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    
    Returns:
        Frequency spectrum (DFT amplitude) as 1D numpy array [freq_bins].
    """
    wf_1d = wf_np.squeeze()
    if wf_1d.ndim == 0:
        wf_1d = wf_1d.reshape(1)
    L = wf_1d.shape[-1]
    if L <= 0:
        return np.zeros(n_fft // 2 + 1, dtype=np.float32)

    win_length = int(win_length)
    if win_length <= 0 or win_length > L:
        win_length = min(L, n_fft)
    hop = int(hop_length)
    if hop <= 0:
        hop = max(1, win_length // 4)

    if L <= win_length:
        frames = wf_1d[None, :]
        if frames.shape[-1] < win_length:
            pad = win_length - frames.shape[-1]
            frames = np.pad(frames, ((0, 0), (0, pad)))
    else:
        starts = np.arange(0, L - win_length + 1, hop, dtype=int)
        if len(starts) == 0:
            starts = np.array([0], dtype=int)
        frames = np.stack([wf_1d[s : s + win_length] for s in starts], axis=0)

    window = np.hanning(win_length).astype(frames.dtype)
    frames_win = frames * window[None, :]
    spec = np.fft.rfft(frames_win, n=n_fft, axis=-1)  # [F, freq]
    mag = np.abs(spec)  # [F, freq]
    mag_mean = mag.mean(axis=0)  # [freq] - time-averaged spectrum
    
    return mag_mean.astype(np.float32)


def compute_gfc(
    true_wf_np: np.ndarray,
    pred_wf_np: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> float:
    """Compute Generalized Frequency Correlation (GFC) between true and predicted waveforms.
    
    GFC = |Σᵢ A_d(fᵢ) A_m(fᵢ)| / (√Σⱼ [A_d(fⱼ)]² * √Σₖ [A_m(fₖ)]²)
    
    where:
    - A_d(fᵢ): DFT amplitude at frequency fᵢ for measured data (true waveform)
    - A_m(fᵢ): DFT amplitude at frequency fᵢ for model-estimated signal (predicted waveform)
    
    Args:
        true_wf_np: True waveform as numpy array.
        pred_wf_np: Predicted waveform as numpy array.
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    
    Returns:
        GFC value in range [0, 1] (1 indicates perfect match).
    """
    try:
        A_d = _compute_freq_spectrum_from_waveform(true_wf_np, sample_rate, n_fft, hop_length, win_length)
        A_m = _compute_freq_spectrum_from_waveform(pred_wf_np, sample_rate, n_fft, hop_length, win_length)
        
        min_len = min(len(A_d), len(A_m))
        A_d = A_d[:min_len]
        A_m = A_m[:min_len]
        
        numerator = np.abs(np.sum(A_d * A_m))
        denom_d = np.sqrt(np.sum(A_d ** 2))
        denom_m = np.sqrt(np.sum(A_m ** 2))
        
        if denom_d == 0.0 or denom_m == 0.0:
            return 0.0
        
        gfc = numerator / (denom_d * denom_m)
        return float(np.clip(gfc, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_freq_spectrum_rmse(
    true_wf_np: np.ndarray,
    pred_wf_np: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> float:
    """Compute RMSE between frequency spectra of true and predicted waveforms.
    
    Args:
        true_wf_np: True waveform as numpy array.
        pred_wf_np: Predicted waveform as numpy array.
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    
    Returns:
        RMSE value (non-negative).
    """
    try:
        A_d = _compute_freq_spectrum_from_waveform(true_wf_np, sample_rate, n_fft, hop_length, win_length)
        A_m = _compute_freq_spectrum_from_waveform(pred_wf_np, sample_rate, n_fft, hop_length, win_length)
        
        min_len = min(len(A_d), len(A_m))
        A_d = A_d[:min_len]
        A_m = A_m[:min_len]
        
        mse = np.mean((A_d - A_m) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)
    except Exception:
        return 0.0


def normalize_waveform_tensor(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize waveform tensor to [B, 1, T] shape.
    
    Args:
        waveform: Waveform tensor with various shapes.
    
    Returns:
        Normalized waveform tensor with shape [B, 1, T].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)
    elif waveform.dim() == 3 and waveform.shape[1] != 1:
        waveform = waveform[:, :1, :]
    return waveform


def control_input(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Stack the batch's normalized force / velocity means into ``[B, 3]``.

    Raises ``KeyError`` when ``force_mean`` / ``velocity_mean`` are missing:
    every model that calls this is conditioned on them, so silently dropping
    the condition would corrupt training or generation.
    """
    force_mean = batch["force_mean"].to(device)  # [B, 1] or [B]
    velocity_mean = batch["velocity_mean"].to(device)  # [B, 2]
    if force_mean.dim() == 1:
        force_mean = force_mean.unsqueeze(1)
    return torch.cat([force_mean, velocity_mean], dim=1)


def build_control_condition(
    batch: Dict[str, torch.Tensor],
    control_embedding: ControlEmbedding,
    device: torch.device,
) -> torch.Tensor:
    """Compute control condition (Force/Velocity mean) with control embedding.
    
    Args:
        batch: Batch dictionary containing 'force_mean' and 'velocity_mean'.
        control_embedding: Control embedding module.
        device: Target device.
    
    Returns:
        Control condition tensor [B, D].
    """
    return control_embedding(control_input(batch, device))
