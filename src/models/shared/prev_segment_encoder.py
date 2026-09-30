"""Encoder for previous segment to generate features for UNet fusion."""
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


class BoundaryDistanceEmbedding(nn.Module):
    """Positional embedding based on distance from boundary.
    
    Encodes the distance from the boundary (end of previous segment).
    Closer to boundary = higher embedding values.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_distance: Optional[int] = None,
    ) -> None:
        """Initialize boundary distance embedding.
        
        Args:
            embed_dim: Embedding dimension
            max_distance: Maximum distance to consider (None = use sequence length)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_distance = max_distance
        
        if max_distance is not None:
            self.embedding = nn.Embedding(max_distance + 1, embed_dim)
        else:
            # Sinusoidal embedding for variable length
            self.embedding = None
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate positional embeddings.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensors on
            
        Returns:
            Positional embeddings of shape [seq_len, embed_dim]
            where position 0 is farthest from boundary, position seq_len-1 is at boundary
        """
        if self.max_distance is not None:
            distances = torch.arange(seq_len, device=device)
            distances = torch.clamp(distances, 0, self.max_distance)
            return self.embedding(distances)  # [seq_len, embed_dim]
        else:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            # Normalized to [0, 1]; 1 is at the boundary
            positions = positions / max(1, seq_len - 1)
            
            dim_t = torch.arange(self.embed_dim, device=device, dtype=torch.float32)
            dim_t = 10000 ** (2 * (dim_t // 2) / self.embed_dim)
            
            pos_enc = positions.unsqueeze(-1) / dim_t.unsqueeze(0)
            pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
            pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])
            
            return pos_enc  # [seq_len, embed_dim]


class PrevSegmentEncoder(nn.Module):
    """Encoder for previous segment that generates multi-scale features for UNet fusion.
    
    This encoder processes the previous segment (codes or latent) and generates
    features at multiple scales that correspond to UNet's downsampling layers.
    These features are then fused into the UNet at each corresponding layer.
    
    Input: [B, C, T] where C is the channel dimension (from codes/latent embedding)
    Output: List of features [f0: [B, ch0, T/2], f1: [B, ch1, T/4], ...]
    """
    
    def __init__(
        self,
        in_ch: int,
        feature_channels: List[int],
        groups: int = 8,
        use_positional_embedding: bool = True,
        pos_embed_dim: Optional[int] = None,
        pos_embed_max_distance: Optional[int] = None,
    ) -> None:
        """Initialize previous segment encoder.
        
        Args:
            in_ch: Input channel dimension (from codes/latent embedding)
            feature_channels: List of output channel dimensions for each scale level.
                             Should match UNet's downsampling layer channels.
            groups: Number of groups for GroupNorm
            use_positional_embedding: Whether to use positional embedding
            pos_embed_dim: Dimension of positional embedding (default: in_ch // 4)
            pos_embed_max_distance: Maximum distance for positional embedding (None = sinusoidal)
        """
        super().__init__()
        self.in_ch = in_ch
        self.feature_channels = feature_channels
        self.groups = groups
        self.use_positional_embedding = use_positional_embedding
        
        if use_positional_embedding:
            if pos_embed_dim is None:
                pos_embed_dim = max(8, in_ch // 4)
            self.pos_embed_dim = pos_embed_dim
            self.pos_embed = BoundaryDistanceEmbedding(
                embed_dim=pos_embed_dim,
                max_distance=pos_embed_max_distance,
            )
            self.pos_proj = nn.Linear(pos_embed_dim, in_ch)
        else:
            self.pos_embed = None
            self.pos_proj = None
        
        # Stride-2 convs mirroring the UNet downsampling path
        self.encoder = nn.ModuleList()
        prev_ch = in_ch
        
        for i, out_ch in enumerate(feature_channels):
            layer = nn.Sequential(
                nn.Conv1d(
                    prev_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                group_norm_1d(out_ch, groups),
                nn.SiLU(),
            )
            self.encoder.append(layer)
            prev_ch = out_ch
    
    def forward(self, prev_x: torch.Tensor) -> List[torch.Tensor]:
        """Encode previous segment into multi-scale features.
        
        Args:
            prev_x: Previous segment tensor of shape [B, C, T]
            
        Returns:
            List of feature tensors at different scales:
            [f0: [B, ch0, T/2], f1: [B, ch1, T/4], ...]
        """
        if prev_x.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, C, T], got shape {prev_x.shape}")
        B, C, T = prev_x.shape
        # Pad to a multiple of the total stride (2^layers) so feature lengths
        # stay aligned with the UNet for fusion.
        total_stride = 1
        for _ in self.feature_channels:
            total_stride *= 2
        pad_T = (total_stride - (T % total_stride)) % total_stride
        if pad_T > 0:
            prev_x = F.pad(prev_x, (0, pad_T))
            T = prev_x.shape[-1]
        
        if self.use_positional_embedding and self.pos_embed is not None:
            pos_emb = self.pos_embed(T, prev_x.device)
            pos_emb_proj = self.pos_proj(pos_emb)  # [T, C]
            pos_emb_proj = pos_emb_proj.transpose(0, 1).unsqueeze(0)  # [1, C, T]
            prev_x = prev_x + pos_emb_proj
        
        features = []
        x = prev_x
        
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        
        return features

