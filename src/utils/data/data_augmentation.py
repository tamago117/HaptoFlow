from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F


def shuffle_prev_within_batch(
    prev_waveform: Optional[torch.Tensor],
    has_prev: Optional[torch.Tensor],
    shuffle_cfg: Any,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Shuffle prev_waveform within batch to simulate mismatched history conditions.
    
    Args:
        prev_waveform: Previous waveform tensor [B, 1, T] or [B, T]
        has_prev: Optional mask tensor indicating which samples have prev [B]
        shuffle_cfg: Configuration for shuffle augmentation
        device: Target device
        
    Returns:
        Tuple of (shuffled_prev_waveform, shuffled_has_prev)
    """
    if prev_waveform is None:
        return None, has_prev
    
    if shuffle_cfg is None:
        return prev_waveform, has_prev
    
    prob = float(getattr(shuffle_cfg, "prob", 0.0))
    if prob <= 0.0:
        return prev_waveform, has_prev
    
    x = prev_waveform
    B = x.shape[0]
    
    perm = torch.randperm(B, device=device)
    
    shuffle_mask = torch.rand(B, device=device) < prob
    
    x_shuffled = x.clone()
    has_prev_shuffled = has_prev.clone() if has_prev is not None else None
    
    for b in range(B):
        if shuffle_mask[b]:
            src_idx = perm[b].item()
            x_shuffled[b] = x[src_idx]
            if has_prev_shuffled is not None:
                has_prev_shuffled[b] = has_prev[src_idx]
    
    return x_shuffled, has_prev_shuffled


def get_zero_prev_prob(data_augmentation_cfg: Any) -> float:
    """``zero_prev.prob``: per-sample probability of starting from a zero prev (0 when absent)."""
    zero_cfg = getattr(data_augmentation_cfg, "zero_prev", None) if data_augmentation_cfg is not None else None
    return float(getattr(zero_cfg, "prob", 0.0)) if zero_cfg is not None else 0.0


def zero_prev_within_batch(
    prev_waveform: Optional[torch.Tensor],
    data_augmentation_cfg: Any,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Replace prev_waveform with zeros per sample to simulate the zero-padded
    initial history used at inference start.

    Args:
        prev_waveform: Previous waveform tensor [B, 1, T] or [B, T]
        data_augmentation_cfg: Data augmentation config (reads ``zero_prev``)
        device: Target device

    Returns:
        prev_waveform with the selected samples zeroed (same shape as input)
    """
    if prev_waveform is None:
        return None
    if data_augmentation_cfg is None:
        return prev_waveform

    prob = get_zero_prev_prob(data_augmentation_cfg)
    if prob <= 0.0:
        return prev_waveform

    B = prev_waveform.shape[0]
    keep = torch.rand(B, device=device) >= prob
    keep = keep.view([B] + [1] * (prev_waveform.dim() - 1))
    return prev_waveform * keep.to(prev_waveform.dtype)


def apply_lowpass_blur(
    prev_waveform: torch.Tensor,
    lowpass_cfg: Any,
    device: torch.device,
) -> torch.Tensor:
    """Apply random low-pass filtering to prev_waveform to simulate degraded/blurred history.
    
    Args:
        prev_waveform: Input waveform tensor [B, 1, T] or [B, T]
        lowpass_cfg: Configuration for lowpass augmentation
        device: Target device
        
    Returns:
        Low-pass filtered waveform with same shape as input
    """
    if lowpass_cfg is None:
        return prev_waveform
    
    prob = float(getattr(lowpass_cfg, "prob", 0.0))
    if prob <= 0.0:
        return prev_waveform
    
    x = prev_waveform
    original_shape = x.shape
    
    if x.dim() == 2:
        x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
    elif x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]
    
    B, C, T = x.shape
    
    kernel_size_min = int(getattr(lowpass_cfg, "kernel_size_min", 5))
    kernel_size_max = int(getattr(lowpass_cfg, "kernel_size_max", 15))
    method = str(getattr(lowpass_cfg, "method", "avg")).lower()
    
    for b in range(B):
        if torch.rand((), device=device) < prob:
            k = torch.randint(kernel_size_min, kernel_size_max + 1, (1,), device=device).item()
            if k % 2 == 0:
                k += 1  # Odd kernel keeps the output centered
            
            if method == "gaussian":
                sigma = float(k) / 6.0
                kernel_1d = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2.0
                kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
                kernel_1d = kernel_1d / kernel_1d.sum()
            else:
                kernel_1d = torch.ones(k, dtype=torch.float32, device=device) / float(k)
            
            kernel = kernel_1d.view(1, 1, k)
            
            pad_size = k // 2
            x_b = x[b:b+1, :, :]  # [1, 1, T]
            x_b_padded = F.pad(x_b, (pad_size, pad_size), mode='reflect')
            x_b_filtered = F.conv1d(x_b_padded, kernel, padding=0)
            x[b:b+1, :, :] = x_b_filtered
    
    if len(original_shape) == 2:
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]
    elif len(original_shape) == 1:
        x = x.squeeze(0).squeeze(0)  # [1, 1, T] -> [T]
    
    return x


def apply_history_noise_and_mask(
    prev_waveform: Optional[torch.Tensor],
    data_augmentation_cfg: Any,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Apply noise and time masking augmentation to previous waveform.

    This operates on raw waveform tensors and is shared across all models.
    The tensor shape is assumed to be [B, 1, T] or broadcastable to it.
    """
    if prev_waveform is None:
        return None
    if data_augmentation_cfg is None:
        return prev_waveform

    x = prev_waveform

    # --- Low-pass filtering (blur) ---
    lowpass_cfg = getattr(data_augmentation_cfg, "lowpass", None)
    if lowpass_cfg is not None:
        x = apply_lowpass_blur(x, lowpass_cfg, device)

    # --- Noise augmentation (Gaussian noise + gain + DC offset) ---
    noise_cfg = getattr(data_augmentation_cfg, "noise", None)
    if noise_cfg is not None:
        prob = float(getattr(noise_cfg, "prob", 0.0))
        if prob > 0.0 and torch.rand((), device=device) < prob:
            sigma = float(getattr(noise_cfg, "sigma", 0.0))
            if sigma > 0.0:
                x = x + torch.randn_like(x) * sigma

            gain_min = float(getattr(noise_cfg, "gain_min", 1.0))
            gain_max = float(getattr(noise_cfg, "gain_max", 1.0))
            if gain_max > 0.0 and gain_max != 1.0 or gain_min != 1.0:
                # Per-sample gain
                if x.dim() >= 2:
                    gain_shape = [x.shape[0]] + [1] * (x.dim() - 1)
                else:
                    gain_shape = x.shape
                gains = torch.empty(gain_shape, device=device).uniform_(gain_min, gain_max)
                x = x * gains

            dc_sigma = float(getattr(noise_cfg, "dc_sigma", 0.0))
            if dc_sigma > 0.0:
                if x.dim() >= 2:
                    dc_shape = [x.shape[0]] + [1] * (x.dim() - 1)
                else:
                    dc_shape = x.shape
                dc = torch.randn(dc_shape, device=device) * dc_sigma
                x = x + dc

    # --- Time masking (span-wise dropout along T dimension) ---
    mask_cfg = getattr(data_augmentation_cfg, "mask", None)
    if mask_cfg is not None:
        prob = float(getattr(mask_cfg, "prob", 0.0))
        max_ratio = float(getattr(mask_cfg, "max_ratio", 0.0))
        if prob > 0.0 and max_ratio > 0.0 and x.dim() >= 2:
            B = x.shape[0]
            T = x.shape[-1]
            max_len = max(1, int(T * max_ratio))
            for b in range(B):
                if torch.rand((), device=device) < prob:
                    span_len = int(torch.randint(1, max_len + 1, (1,), device=device))
                    start = int(torch.randint(0, max(1, T - span_len + 1), (1,), device=device))
                    end = start + span_len
                    x[b, ..., start:end] = 0.0

    return x
