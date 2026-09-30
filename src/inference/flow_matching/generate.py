"""Batched Flow Matching generation, shared by inference and self-forcing training.

The caller owns the grad mode (inference_mode for serving, no_grad in training).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.utils.training.model_builder import build_prev_features
from src.utils.training.training_utils import build_control_condition


def fix_length(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad / truncate the last dim of ``wave`` to ``target_len``."""
    cur_len = int(wave.shape[-1])
    if cur_len < target_len:
        return F.pad(wave, (0, target_len - cur_len))
    return wave[..., :target_len]


def generate_segments(
    *,
    model,
    label_embed,
    control_embedding,
    encodec_wrapper,
    prev_segment_encoder,
    labels: torch.Tensor,
    force_mean: torch.Tensor,
    velocity_mean: torch.Tensor,
    prev_waveform: Optional[torch.Tensor],
    steps: int,
    latent_shape: torch.Size,
    sample_rate: float,
    target_len: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Generate one segment per row.

    Args:
        labels: [n] label ids.
        force_mean / velocity_mean: [n, 1] / [n, 2] normalized controls.
        prev_waveform: [n, L] previous segments at ``sample_rate``, or None to
            generate without previous-segment conditioning.
        latent_shape: Per-sample latent shape ``[*, D, T']`` (leading dim ignored).
        target_len: Output length in samples (None keeps the decoded length).

    Returns:
        [n, target_len] waveforms at ``sample_rate`` (on CPU).
    """
    n = int(labels.shape[0])
    cond = label_embed(labels.long())
    control_cond = build_control_condition(
        {"force_mean": force_mean, "velocity_mean": velocity_mean}, control_embedding, device
    )
    cond = torch.cat([cond, control_cond], dim=-1)

    encoder_feats = None
    if prev_segment_encoder is not None and prev_waveform is not None:
        # input_sample_rate matches training (encode_latent_batch uses the dataset rate)
        prev_latent = encodec_wrapper.encode_waveform(
            prev_waveform,
            return_latent=True,
            return_codes=False,
            input_sample_rate=float(sample_rate),
        ).get("latent")
        if prev_latent is not None:
            encoder_feats = build_prev_features(
                {"latent": prev_latent.to(device)},
                torch.ones(n, device=device, dtype=torch.bool),
                prev_segment_encoder,
                device,
            )

    shape = torch.Size([n] + list(latent_shape[1:]))
    x_cont = model.generate(steps=int(steps), cond=cond, shape=shape, encoder_feats=encoder_feats)
    codes, _ = encodec_wrapper.quantize_latent(x_cont.detach())
    waveform = encodec_wrapper.decode_from_codes(codes, target_sample_rate=float(sample_rate))
    # [n, 1, T'] (or [1, T'] when n == 1) → [n, T']
    waveform = waveform.reshape(n, -1)
    return waveform if target_len is None else fix_length(waveform, int(target_len))
