"""``train.self_forcing`` settings (phase-2 training on the model's own outputs)."""
from typing import Any, NamedTuple, Optional


class SelfForcingParams(NamedTuple):
    num_unroll: int = 1
    prob: float = 0.0
    loss_reduction: str = "mean"
    loss_pairs: Optional[int] = None  # Pairs per chain in the loss (None = all)


def get_self_forcing_params(cfg: Any) -> SelfForcingParams:
    """Settings of a ``self_forcing`` section; defaults (no unroll) when it is None."""
    if cfg is None:
        return SelfForcingParams()
    loss_pairs = getattr(cfg, "loss_pairs", None)
    return SelfForcingParams(
        num_unroll=max(1, int(getattr(cfg, "num_unroll", 1))),
        prob=float(getattr(cfg, "prob", 0.0)),
        loss_reduction=str(getattr(cfg, "loss_reduction", "mean")).lower(),
        loss_pairs=int(loss_pairs) if loss_pairs is not None else None,
    )
