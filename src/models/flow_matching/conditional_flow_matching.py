from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ConditionalFlowMatching(nn.Module):
    """Velocity field model wrapper for Conditional Flow Matching.

    Expects a UNet-like `backbone` with signature: backbone(x_t, cond, encoder_feats=None)->v_pred
    Supports both 2D (image) and 1D (time-series) inputs by auto-detecting dimensions.
    """

    def __init__(
        self, 
        backbone: nn.Module, 
        is_1d: Optional[bool] = None,
        time_embed: Optional[nn.Module] = None,
        solver: str = "euler",
        cond_dropout: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> None:
        """Initialize Conditional Flow Matching.
        
        Args:
            backbone: UNet-like backbone model
            is_1d: If True, treat input as 1D (time-series). If False, treat as 2D (image).
                   If None, auto-detect from first forward pass.
            time_embed: Optional time embedding module. If provided, time information will be
                       added to the condition vector.
            solver: ODE solver for ``generate``: "euler" (1 evaluation per step) or
                    "heun" (2 evaluations per step).
            cond_dropout: Training probability of zeroing ``cond`` (the null condition for CFG).
            guidance_scale: Classifier-free guidance weight in ``generate`` (1.0 = off).
                The prev-segment features stay in both branches.
        """
        super().__init__()
        self.backbone = backbone
        self._is_1d = is_1d  # None means auto-detect
        self.time_embed = time_embed
        if solver not in ("euler", "heun"):
            raise ValueError(f"sampling.solver must be 'euler' or 'heun', got {solver!r}")
        self.solver = solver
        self.cond_dropout = float(cond_dropout)
        self.guidance_scale = float(guidance_scale)

    @staticmethod
    def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, 1, device=device)

    def _detect_dimensions(self, x0: torch.Tensor) -> bool:
        """Detect if input is 1D (time-series) or 2D (image).
        
        Returns True if 1D, False if 2D.
        """
        # 1D: [B, C, T]; 2D: [B, C, H, W]
        return x0.dim() == 3

    def forward(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        encoder_feats: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        B = x0.shape[0]
        device = x0.device
        t = self.sample_time(B, device)  # (B,1)
        
        if self._is_1d is None:
            self._is_1d = self._detect_dimensions(x0)
        
        # Reshape for broadcasting: 1D -> (B,1,1), 2D -> (B,1,1,1)
        if self._is_1d:
            t_b = t.view(B, 1, 1)
        else:
            t_b = t.view(B, 1, 1, 1)
        
        if self.training and self.cond_dropout > 0.0:
            keep = (torch.rand(B, 1, device=device) >= self.cond_dropout).to(cond.dtype)
            cond = cond * keep

        x1 = torch.randn_like(x0)
        x_t = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0
        
        if self.time_embed is not None:
            t_emb = self.time_embed(t)  # (B, time_emb_dim)
            cond_with_time = torch.cat([cond, t_emb], dim=-1)
        else:
            cond_with_time = cond
        
        v_pred = self.backbone(x_t, cond_with_time, encoder_feats=encoder_feats)
        loss = F.mse_loss(v_pred, v_target)
        return {"loss": loss, "v_pred": v_pred, "t": t, "x_t": x_t}

    def _velocity(self, x: torch.Tensor, t_val: float, cond: torch.Tensor, encoder_feats: Optional[list]) -> torch.Tensor:
        w = self.guidance_scale
        if w != 1.0:
            # Conditional and null-condition branches in one batch
            x = torch.cat([x, x])
            cond = torch.cat([cond, torch.zeros_like(cond)])
            if encoder_feats is not None:
                encoder_feats = [torch.cat([f, f]) for f in encoder_feats]
        t_tensor = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=torch.float32)
        if self.time_embed is not None:
            cond = torch.cat([cond, self.time_embed(t_tensor)], dim=-1)
        v = self.backbone(x, cond, encoder_feats=encoder_feats)
        if w != 1.0:
            v_cond, v_null = v.chunk(2)
            v = v_null + w * (v_cond - v_null)
        return v

    @torch.no_grad()
    def generate(
        self,
        steps: int,
        cond: torch.Tensor,
        shape: torch.Size,
        encoder: Optional[nn.Module] = None,
        encoder_feats: Optional[list] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate from noise (t=1) to data (t=0) in ``steps`` steps with ``self.solver``.

        ``noise`` overrides the initial ``randn(shape)`` sample.
        """
        x = torch.randn(shape, device=cond.device) if noise is None else noise
        dt = 1.0 / steps
        for i in range(steps):
            t = 1.0 - i * dt
            if self.solver == "heun":
                v = self._velocity(x, t, cond, encoder_feats)
                x_pred = x - dt * v
                v_next = self._velocity(x_pred, t - dt, cond, encoder_feats)
                x = x - 0.5 * dt * (v + v_next)
            else:
                # Velocity at the interval midpoint time, taken from the start state
                x = x - dt * self._velocity(x, t - 0.5 * dt, cond, encoder_feats)
        return x
