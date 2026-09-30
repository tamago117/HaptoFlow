from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


def group_norm_1d(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """1D GroupNorm for 1D convolutions."""
    g = min(num_groups, num_channels)
    # Largest divisor of num_channels that is <= num_groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    g = max(1, g)
    return nn.GroupNorm(g, num_channels)


class FiLM1D(nn.Module):
    """FiLM (Feature-wise Linear Modulation) for 1D tensors."""
    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: [B, C, T], cond: [B, cond_dim]
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        while scale.dim() < h.dim():
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return h * (1.0 + scale) + shift


class ResBlock1D(nn.Module):
    """1D Residual block with conditional FiLM."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = group_norm_1d(in_ch, groups)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.cond1 = FiLM1D(cond_dim, out_ch)

        self.norm2 = group_norm_1d(out_ch, groups)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.cond2 = FiLM1D(cond_dim, out_ch)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, T], cond: [B, cond_dim]
        h = self.conv1(self.act(self.norm1(x)))
        h = self.cond1(h, cond)
        h = self.conv2(self.act(self.norm2(h)))
        h = self.cond2(h, cond)
        return h + self.skip(x)


class Down1D(nn.Module):
    """1D Downsampling block."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.block = ResBlock1D(in_ch, out_ch, cond_dim, groups)
        self.pool = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.block(x, cond)
        return self.pool(x)
    
    def forward_with_skip(self, x: torch.Tensor, cond: torch.Tensor):
        """Forward pass that also returns the feature before downsampling for skip connection."""
        skip_feat = self.block(x, cond)
        return self.pool(skip_feat), skip_feat


class Up1D(nn.Module):
    """1D Upsampling block with skip connection."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8, skip_ch: Optional[int] = None) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        if skip_ch is not None:
            self.block = ResBlock1D(in_ch + skip_ch, out_ch, cond_dim, groups)
        else:
            self.block = ResBlock1D(in_ch, out_ch, cond_dim, groups)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x, cond)


class ConditionalUNet1D(nn.Module):
    """1D Conditional UNet for Flow Matching on time-series features.
    
    Input shape: [B, C, T] where:
        - B: batch size
        - C: number of feature channels (e.g., from EnCodec encoder)
        - T: time steps
    
    Output shape: [B, C, T] (same as input)
    """
    
    def __init__(
        self,
        in_ch: int = 128,
        base_ch: int = 64,
        ch_mults: List[int] = [1, 2, 4],
        cond_dim: int = 128,
        groups: int = 8,
        encoder_feature_channels: Optional[List[int]] = None,
    ) -> None:
        """Initialize 1D Conditional UNet.
        
        Args:
            in_ch: Number of input channels (feature dimensions from encoder)
            base_ch: Base number of channels
            ch_mults: Channel multipliers for downsampling/upsampling
            cond_dim: Dimension of conditioning vector (label + time + control embeddings)
            groups: Number of groups for GroupNorm
            encoder_feature_channels: Optional channel dimensions of the encoder features
                                    fused after each downsampling stage
        """
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base_ch, 3, padding=1)

        downs = []
        chs = [base_ch]
        curr = base_ch
        for m in ch_mults:
            outc = base_ch * m
            downs.append(Down1D(curr, outc, cond_dim, groups))
            chs.append(outc)
            curr = outc
        self.downs = nn.ModuleList(downs)

        self.mid = ResBlock1D(curr, curr, cond_dim, groups)

        ups = []
        skip_chs = list(reversed(chs[1:]))
        for i, m in enumerate(reversed(ch_mults)):
            inc = curr
            outc = base_ch * m
            skip_ch = skip_chs[i] if i < len(skip_chs) else None
            ups.append(Up1D(inc, outc, cond_dim, groups, skip_ch=skip_ch))
            curr = outc
        self.ups = nn.ModuleList(ups)

        self.out_norm = group_norm_1d(curr, groups)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(curr, in_ch, 3, padding=1)

        # Optional encoder fusion (concat then 1x1)
        self.encoder_feature_channels = encoder_feature_channels or []
        if self.encoder_feature_channels:
            self.fuse = nn.ModuleList([
                nn.Conv1d(ch + ef, ch, 1) for ch, ef in zip(chs[1:], self.encoder_feature_channels)
            ])
        else:
            self.fuse = None

        # Output length alignment: "pad" (default) or "interpolate"
        self.length_fix_mode: str = "pad"

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        encoder_feats: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, T]
            cond: Conditioning vector of shape [B, cond_dim]
            encoder_feats: Optional list of encoder features for fusion
        
        Returns:
            Output tensor of shape [B, C, T]
        """
        orig_T = x.shape[-1]
        if self.length_fix_mode == "pad":
            total_stride = 1
            for _ in self.downs:
                total_stride *= 2
            pad_T = (total_stride - (orig_T % total_stride)) % total_stride
            if pad_T > 0:
                x = F.pad(x, (0, pad_T))

        h = self.in_conv(x)
        feats: List[torch.Tensor] = []
        for i, down in enumerate(self.downs):
            h, skip_feat = down.forward_with_skip(h, cond)
            if self.fuse is not None and encoder_feats is not None and i < len(self.fuse):
                h = torch.cat([h, encoder_feats[i]], dim=1)
                h = self.fuse[i](h)
            feats.append(skip_feat)

        h = self.mid(h, cond)

        for i, up in enumerate(self.ups):
            skip = feats[-(i+1)] if i < len(feats) else None
            h = up(h, cond, skip=skip)
        h = self.out_conv(self.out_act(self.out_norm(h)))
        if self.length_fix_mode == "interpolate":
            if h.shape[-1] != orig_T:
                h = F.interpolate(h, size=orig_T, mode="linear", align_corners=False)
            return h
        else:
            if h.shape[-1] != orig_T:
                h = h[..., :orig_T]
        return h

