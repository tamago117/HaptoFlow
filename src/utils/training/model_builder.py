"""Cross-family model building helpers.

Prev-segment / control embeddings, the optimizer + scheduler factory, and the
helper that lifts prev-segment dict batches into encoder features. The
backbone builder lives in ``src/models/flow_matching/builder.py``.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig

from src.models.flow_matching.conditional_flow_matching import ConditionalFlowMatching
from src.models.shared.embeddings import (
    ControlEmbedding,
    SinusoidalTimeEmbedding,
)
from src.models.shared.prev_segment_encoder import PrevSegmentEncoder


def build_prev_segment_encoder(
    use_prev_segment: bool,
    in_ch: int,
    cfg: DictConfig,
    device: torch.device,
) -> Tuple[Optional[PrevSegmentEncoder], Optional[List[int]]]:
    """Build previous segment encoder if enabled.
    
    Args:
        use_prev_segment: Whether to use previous segment.
        in_ch: Input channel dimension.
        cfg: Configuration object.
        device: Target device.
    
    Returns:
        Tuple of (prev_segment_encoder or None, encoder_feature_channels or None).
    """
    if not use_prev_segment:
        return None, None
    
    base_ch = int(cfg.model.base_ch)
    ch_mults = list(cfg.model.ch_mults)
    feature_channels = [base_ch * m for m in ch_mults]
    
    prev_encoder_groups = int(getattr(cfg.model.prev_segment_encoder, "groups", cfg.model.groups))
    use_pos_embed = bool(getattr(cfg.model.prev_segment_encoder, "use_positional_embedding", True))
    pos_embed_dim = getattr(cfg.model.prev_segment_encoder, "pos_embed_dim", None)
    if pos_embed_dim is not None:
        pos_embed_dim = int(pos_embed_dim)
    pos_embed_max_distance = getattr(cfg.model.prev_segment_encoder, "pos_embed_max_distance", None)
    if pos_embed_max_distance is not None:
        pos_embed_max_distance = int(pos_embed_max_distance)
    
    prev_segment_encoder = PrevSegmentEncoder(
        in_ch=in_ch,
        feature_channels=feature_channels,
        groups=prev_encoder_groups,
        use_positional_embedding=use_pos_embed,
        pos_embed_dim=pos_embed_dim,
        pos_embed_max_distance=pos_embed_max_distance,
    ).to(device)
    
    return prev_segment_encoder, feature_channels


def build_control_embedding(
    cfg: DictConfig,
    device: torch.device,
) -> Tuple[ControlEmbedding, int]:
    """Build control embedding (Force/Velocity).
    
    Args:
        cfg: Configuration object.
        device: Target device.
    
    Returns:
        Tuple of (control_embedding, control_cond_dim).
    """
    control_cfg = getattr(cfg.model, "control_embedding", None)
    
    out_dim = 64
    if control_cfg is not None:
        out_dim = int(getattr(control_cfg, "out_dim", 64))
    
    use_sinusoidal = True
    if control_cfg is not None:
        use_sinusoidal = bool(getattr(control_cfg, "use_sinusoidal", True))
        
    control_embedding = ControlEmbedding(
        hidden_dim=out_dim,
        out_dim=out_dim,
        input_dim=3, # Force + Vx + Vy
        num_layers=2,
        use_sinusoidal=use_sinusoidal,
    ).to(device)
    
    return control_embedding, out_dim


def build_prev_features(
    prev_seg_batch: Optional[Dict[str, torch.Tensor]],
    prev_mask_tensor: Optional[torch.Tensor],
    prev_segment_encoder: Optional[PrevSegmentEncoder],
    device: torch.device,
) -> Optional[List[torch.Tensor]]:
    """Compute previous segment features with optional masking.
    
    Args:
        prev_seg_batch: Previous segment batch dictionary.
        prev_mask_tensor: Mask tensor indicating which samples have previous segments.
        prev_segment_encoder: Previous segment encoder module.
        device: Target device.
    
    Returns:
        List of feature tensors or None.
    """
    if prev_segment_encoder is None or prev_seg_batch is None:
        return None
    
    mask: Optional[torch.Tensor] = None
    if prev_mask_tensor is not None:
        mask = prev_mask_tensor.to(device)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not bool(mask.any().item()):
            return None
    
    prev_x0: Optional[torch.Tensor] = None
    if "latent" in prev_seg_batch:
        prev_x0 = prev_seg_batch["latent"].to(device)
        # PrevSegmentEncoder expects [B, C, T]; app may pass [C, T] after squeeze(0).
        if prev_x0.dim() == 2:
            prev_x0 = prev_x0.unsqueeze(0)
    else:
        return None

    prev_features = prev_segment_encoder(prev_x0)
    if mask is not None:
        mask_view = mask.view(mask.shape[0], 1, 1)
        prev_features = [feat * mask_view for feat in prev_features]
    return prev_features


def build_optimizer_and_scheduler(
    model: ConditionalFlowMatching,
    time_embed: Optional[SinusoidalTimeEmbedding],
    prev_segment_encoder: Optional[PrevSegmentEncoder],
    control_embedding: Optional[ControlEmbedding],
    cfg: DictConfig,
    steps_per_epoch: int = 1,
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """Build optimizer and learning rate scheduler.
    
    Args:
        model: Flow Matching model.
        time_embed: Optional time embedding module.
        prev_segment_encoder: Optional previous segment encoder.
        control_embedding: Optional control embedding.
        cfg: Configuration object.
    
    Returns:
        Tuple of (optimizer, scheduler or None).
    """
    params_to_optimize = list(model.parameters())
    if time_embed is not None:
        params_to_optimize += list(time_embed.parameters())
    if prev_segment_encoder is not None:
        params_to_optimize += list(prev_segment_encoder.parameters())
    if control_embedding is not None:
        params_to_optimize += list(control_embedding.parameters())
    
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas)
    )

    # Step-based: the trainer steps the scheduler once per batch.
    scheduler = None
    if hasattr(cfg, "scheduler") and str(cfg.scheduler.type) == "cosine":
        T_max_val = cfg.scheduler.cosine.T_max
        T_max_epochs = float(T_max_val) if T_max_val is not None else float(cfg.train.epochs)
        eta_min = float(cfg.scheduler.cosine.eta_min)
        warmup_ratio = float(getattr(cfg.scheduler.cosine, "warmup_ratio", 0.0) or 0.0)
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f"scheduler.cosine.warmup_ratio must be in [0, 1), got {warmup_ratio}")

        spe = max(1, int(steps_per_epoch))
        total_steps = max(1, int(round(T_max_epochs * spe)))
        warmup_steps = int(round(warmup_ratio * total_steps))
        initial_lr = float(cfg.optim.lr)
        eta_ratio = eta_min / initial_lr if initial_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            cosine_T = max(1, total_steps - warmup_steps)
            progress = float(step - warmup_steps) / float(cosine_T)
            progress = min(1.0, max(0.0, progress))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_ratio + (1.0 - eta_ratio) * cosine_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


