import math
from typing import Optional

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1) or (B,)
        half = self.dim // 2
        device = t.device
        t = t.view(-1, 1)
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=t.dtype)
            * -(math.log(10000.0) / max(1, half - 1))
        )
        args = t * freqs  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # (B, dim)


class FrequencyEmbedding(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        out_dim: int, 
        num_layers: int = 2,
        use_sinusoidal: bool = True,
        freq_scales: Optional[list] = None,
    ) -> None:
        """Frequency embedding with optional sinusoidal components for smooth interpolation.
        
        Args:
            hidden_dim: Hidden dimension for MLP layers
            out_dim: Output dimension
            num_layers: Number of MLP layers
            use_sinusoidal: If True, add sinusoidal embeddings for smooth interpolation
            freq_scales: List of frequency scales for sinusoidal embedding. 
                        If None, uses [0.5, 1.0, 2.0, 4.0]
        """
        super().__init__()
        self.use_sinusoidal = use_sinusoidal
        
        if use_sinusoidal:
            if freq_scales is None:
                freq_scales = [0.5, 1.0, 2.0, 4.0]
            self.freq_scales = freq_scales
            sinusoidal_dim = len(self.freq_scales) * 2  # sin + cos per scale
            in_dim = 1 + sinusoidal_dim  # raw value + sinusoidal features
        else:
            self.freq_scales = []
            in_dim = 1
            
        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, freq_z: torch.Tensor) -> torch.Tensor:
        # freq_z: (B,) or (B,1)
        if freq_z.dim() == 1:
            freq_z = freq_z.unsqueeze(-1)
        
        if self.use_sinusoidal:
            sin_cos_features = []
            for scale in self.freq_scales:
                scaled_freq = freq_z * scale * (2.0 * math.pi)
                sin_cos_features.append(torch.sin(scaled_freq))
                sin_cos_features.append(torch.cos(scaled_freq))
            sinusoidal_emb = torch.cat(sin_cos_features, dim=-1)  # (B, len(freq_scales)*2)
            freq_input = torch.cat([freq_z, sinusoidal_emb], dim=-1)  # (B, 1 + len(freq_scales)*2)
        else:
            freq_input = freq_z
            
        return self.net(freq_input)


class ControlEmbedding(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        out_dim: int, 
        input_dim: int = 3,
        num_layers: int = 2,
        use_sinusoidal: bool = True,
        freq_scales: Optional[list] = None,
    ) -> None:
        """Control embedding (Force/Velocity) with optional sinusoidal components.
        
        Args:
            hidden_dim: Hidden dimension for MLP layers
            out_dim: Output dimension
            input_dim: Input dimension (default 3 for Force + Velocity X/Y)
            num_layers: Number of MLP layers
            use_sinusoidal: If True, add sinusoidal embeddings for smooth interpolation
            freq_scales: List of frequency scales for sinusoidal embedding. 
                        If None, uses [0.5, 1.0, 2.0, 4.0]
        """
        super().__init__()
        self.use_sinusoidal = use_sinusoidal
        self.input_dim = input_dim
        
        if use_sinusoidal:
            if freq_scales is None:
                freq_scales = [0.5, 1.0, 2.0, 4.0]
            self.freq_scales = freq_scales
            sinusoidal_dim = len(self.freq_scales) * 2 * input_dim
            in_dim = input_dim + sinusoidal_dim
        else:
            self.freq_scales = []
            in_dim = input_dim
            
        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        # control: (B, input_dim)
        if control.dim() == 1:
            control = control.unsqueeze(0)
        
        if self.use_sinusoidal:
            sin_cos_features = []
            for scale in self.freq_scales:
                scaled = control * scale * (2.0 * math.pi)
                sin_cos_features.append(torch.sin(scaled))
                sin_cos_features.append(torch.cos(scaled))
            
            sinusoidal_emb = torch.cat(sin_cos_features, dim=-1)
            control_input = torch.cat([control, sinusoidal_emb], dim=-1)
        else:
            control_input = control
            
        return self.net(control_input)


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.emb(labels)


class ConditionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(cond)
