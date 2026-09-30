"""Utility functions for EnCodec audio processing."""
from typing import Optional
import torch
import torch.nn.functional as F


def align_waveform_length(
    waveform: torch.Tensor,
    target_length: int,
    mode: str = "pad_or_trim"
) -> torch.Tensor:
    """Align waveform length to target length.
    
    Args:
        waveform: Waveform tensor of shape [B, C, T] or [C, T] or [T]
        target_length: Target length in samples
        mode: Alignment mode ("pad_or_trim" or "trim_only")
    
    Returns:
        Aligned waveform of same shape but with T=target_length
    """
    original_shape = waveform.shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [C, T] -> [1, C, T]
    
    T = waveform.shape[-1]
    
    if T < target_length:
        if mode == "pad_or_trim":
            pad = target_length - T
            waveform = F.pad(waveform, (0, pad))
        elif mode == "trim_only":
            pad = target_length - T
            waveform = F.pad(waveform, (0, pad))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    elif T > target_length:
        waveform = waveform[..., :target_length]
    
    if len(original_shape) == 1:
        waveform = waveform.squeeze(0).squeeze(0)
    elif len(original_shape) == 2:
        waveform = waveform.squeeze(0)
    
    return waveform


def batch_codes_to_frames(
    codes: torch.LongTensor,
    batch_size: Optional[int] = None
) -> list:
    """Convert batch of codes to EnCodec frames format.
    
    Args:
        codes: Codes tensor of shape [B, Q, T]
        batch_size: Optional batch size (if None, uses codes.shape[0])
    
    Returns:
        List of frames, one per batch item: List[(codes_i, scale_i), ...]
    """
    B, Q, T = codes.shape
    if batch_size is None:
        batch_size = B
    
    frames = []
    for i in range(min(B, batch_size)):
        codes_i = codes[i:i+1]  # [1, Q, T]
        scale_i = torch.ones(1, 1, 1, device=codes.device, dtype=torch.float32)
        frames.append((codes_i, scale_i))
    
    return frames


def compute_snr_db(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12
) -> float:
    """Compute Signal-to-Noise Ratio in dB.
    
    Args:
        x: Reference signal
        y: Reconstructed signal
        eps: Small epsilon to avoid division by zero
    
    Returns:
        SNR in dB
    """
    import math
    
    min_len = min(x.shape[-1], y.shape[-1])
    x = x[..., :min_len]
    y = y[..., :min_len]
    
    signal_power = (x.pow(2).mean()).clamp_min(eps)
    noise_power = ((x - y).pow(2).mean()).clamp_min(eps)
    snr_linear = signal_power / noise_power
    snr_db = 10.0 * math.log10(snr_linear.item())
    
    return float(snr_db)

