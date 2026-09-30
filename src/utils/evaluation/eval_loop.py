"""Shared evaluation loop utilities (loss accumulation across val/test)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch
from tqdm.auto import tqdm

from src.strategies import ModelStrategy, StepContext, TrainingBundle


class PreparedBatches:
    """``dataloader`` with ``strategy.prepare_eval_batch`` applied to every batch.

    A fixed ``seed`` gives each pass the same phase-2 prev choices.
    """

    def __init__(self, strategy: ModelStrategy, dataloader: Any, bundle: TrainingBundle, ctx: StepContext, seed: int) -> None:
        self.strategy, self.dataloader, self.bundle, self.ctx, self.seed = strategy, dataloader, bundle, ctx, int(seed)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for batch in self.dataloader:
            yield self.strategy.prepare_eval_batch(batch, self.bundle, self.ctx, generator)

    def first(self) -> Any:
        return next(iter(self))


@dataclass
class EvalLossSummary:
    """Aggregated evaluation metrics over a dataloader."""
    loss_mean: float = 0.0
    infer_time_mean_ms: float = 0.0
    infer_time_list: List[float] = field(default_factory=list)
    count: int = 0


def accumulate_eval_losses(
    *,
    strategy: ModelStrategy,
    bundle: TrainingBundle,
    dataloader: Any,
    ctx: StepContext,
    desc: Optional[str] = None,
    on_batch: Optional[Callable[[], None]] = None,
) -> EvalLossSummary:
    """Iterate dataloader once and aggregate strategy.eval_losses results.

    If ``desc`` is provided, a tqdm progress bar with that description is shown.

    ``on_batch`` is called at each batch boundary before any work is done; it
    may raise to stop the loop early (so the memory guard and interrupts also
    cover the val/test loss passes).
    """
    s = EvalLossSummary()
    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    iterator = (
        tqdm(dataloader, total=total, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
        if desc is not None
        else dataloader
    )
    with torch.inference_mode():
        for batch in iterator:
            if on_batch is not None:
                on_batch()
            loss, infer_time_ms = strategy.eval_losses(batch, bundle, ctx)

            s.loss_mean += float(loss)
            s.infer_time_mean_ms += float(infer_time_ms)
            s.infer_time_list.append(float(infer_time_ms))
            s.count += 1

    n = max(1, s.count)
    s.loss_mean /= n
    s.infer_time_mean_ms /= n
    return s
