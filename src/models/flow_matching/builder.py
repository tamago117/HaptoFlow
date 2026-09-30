"""Flow Matching model graph builder.

Wires ``ConditionalUNet1D`` + ``ConditionalFlowMatching`` from a config.
"""
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig

from src.models.flow_matching.cond_flow_unet_1d import ConditionalUNet1D
from src.models.flow_matching.conditional_flow_matching import ConditionalFlowMatching
from src.models.shared.embeddings import SinusoidalTimeEmbedding


def build_models(
    in_ch: int,
    base_cond_dim: int,
    time_embed_dim: int,
    control_cond_dim: int,
    cfg: DictConfig,
    encoder_feature_channels: Optional[List[int]],
    time_embed: Optional[SinusoidalTimeEmbedding],
    device: torch.device,
) -> Tuple[ConditionalUNet1D, ConditionalFlowMatching]:
    """Build the 1D conditional UNet backbone and wrap it with ConditionalFlowMatching.

    cond_dim layout: base label cond + (optional) time embed + control embed.
    """
    unet_cond_dim = base_cond_dim + (time_embed_dim if time_embed_dim > 0 else 0) + control_cond_dim
    unet = ConditionalUNet1D(
        in_ch=in_ch,
        base_ch=int(cfg.model.base_ch),
        ch_mults=list(cfg.model.ch_mults),
        cond_dim=unet_cond_dim,
        groups=int(cfg.model.groups),
        encoder_feature_channels=encoder_feature_channels,
    ).to(device)

    sampling = getattr(cfg, "sampling", None)
    model = ConditionalFlowMatching(
        unet,
        is_1d=True,
        time_embed=time_embed,
        solver=str(getattr(sampling, "solver", "euler")),
        cond_dropout=float(getattr(cfg.train, "cond_dropout", 0.0)),
        guidance_scale=float(getattr(sampling, "guidance_scale", 1.0)),
    ).to(device)
    return unet, model
