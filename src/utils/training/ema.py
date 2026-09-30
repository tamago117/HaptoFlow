"""Exponential moving average of the trained weights (``train.ema_decay``).

The EMA covers every tensor that goes into ``model.safetensors`` (``bundle.model``
plus the ``nn.Module`` extras), keyed by the same ``<prefix>.<name>`` names, so
the shadow can be saved as the checkpoint weights directly. The raw weights go
to ``training_state.pt`` for ``--resume``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch

from src.strategies import TrainingBundle
from src.utils.training.checkpoint import _collect_module_tensors


class ModelEMA:
    """Shadow copy of the bundle weights, updated after each optimizer step."""

    def __init__(self, bundle: TrainingBundle, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"train.ema_decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        # state_dict() tensors alias the live parameters / buffers
        self.live: Dict[str, torch.Tensor] = _collect_module_tensors(bundle)
        self.shadow: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in self.live.items()}
        self._float_keys = [k for k, v in self.live.items() if v.is_floating_point()]
        self._other_keys = [k for k, v in self.live.items() if not v.is_floating_point()]

    def current_decay(self) -> float:
        # Warmup so early validations are not dominated by the initialization
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self) -> None:
        weight = 1.0 - self.current_decay()
        torch._foreach_lerp_(
            [self.shadow[k] for k in self._float_keys],
            [self.live[k] for k in self._float_keys],
            weight,
        )
        for k in self._other_keys:
            self.shadow[k].copy_(self.live[k])
        self.num_updates += 1

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Temporarily load the EMA weights into the bundle (validation)."""
        with torch.no_grad():
            backup = {k: v.detach().clone() for k, v in self.live.items()}
            self.copy_to_bundle()
        try:
            yield
        finally:
            with torch.no_grad():
                for k, v in self.live.items():
                    v.copy_(backup[k])

    @torch.no_grad()
    def copy_to_bundle(self) -> None:
        """Load the EMA weights into the bundle for good (end of training)."""
        for k, v in self.live.items():
            v.copy_(self.shadow[k])

    def training_state(self) -> Dict[str, Any]:
        """Raw weights + counter for ``training_state.pt`` (the EMA itself is the safetensors)."""
        return {
            "num_updates": self.num_updates,
            "raw_weights": {k: v.detach().cpu().clone() for k, v in self.live.items()},
        }

    @torch.no_grad()
    def load_training_state(self, state: Dict[str, Any]) -> None:
        """Resume: the bundle holds the EMA (from safetensors); restore the raw weights."""
        raw = state["raw_weights"]
        if set(raw) != set(self.live):
            raise ValueError("training_state.pt ema.raw_weights do not match the model")
        for k, v in self.live.items():
            self.shadow[k].copy_(v)
            v.copy_(raw[k].to(v.device))
        self.num_updates = int(state["num_updates"])


def build_ema(cfg: Any, bundle: TrainingBundle) -> Optional[ModelEMA]:
    """``ModelEMA`` for ``train.ema_decay``; None when it is unset or 0."""
    decay = getattr(cfg.train, "ema_decay", None)
    if not decay:
        return None
    return ModelEMA(bundle, float(decay))
