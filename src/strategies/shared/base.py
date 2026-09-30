"""ModelStrategy abstraction for per-model training/evaluation behavior."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class DatasetParams:
    """Dataset-prep flags decided by a strategy before datasets are built."""
    # Back-to-back segments the dataset returns before each target window
    context_segments: int = 1
    # Whether the strategy needs an EncodecWrapper (built on the training device)
    use_encodec: bool = False


@dataclass
class BuildContext:
    """Inputs passed to ModelStrategy.build()."""
    cfg: Any
    device: torch.device
    full_dataset: Any
    encodec_wrapper: Optional[Any] = None
    steps_per_epoch: int = 1


@dataclass
class TrainingBundle:
    """Opaque container for the model and supporting state a strategy needs."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepContext:
    """Cross-cutting per-step state shared across strategies."""
    cfg: Any
    device: torch.device
    full_dataset: Any
    encodec_wrapper: Optional[Any] = None
    scaler: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class ModelStrategy(ABC):
    """Per-model-family strategy. Subclasses are registered by `name`."""

    name: ClassVar[str]

    @abstractmethod
    def dataset_params(self, cfg: Any) -> DatasetParams: ...

    @abstractmethod
    def build(self, ctx: BuildContext) -> TrainingBundle: ...

    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Dict[str, float]: ...

    @abstractmethod
    def eval_losses(
        self,
        batch: Dict[str, torch.Tensor],
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Any: ...

    @abstractmethod
    def generate_samples(
        self,
        batch: Dict[str, torch.Tensor],
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Any: ...

    @abstractmethod
    def compute_metrics(
        self,
        dataloader: Any,
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Any: ...

    def build_inference(
        self,
        bundle: TrainingBundle,
        ctx: StepContext,
    ) -> Any:
        """Return an inference engine (subclass of BaseInference) wrapping ``bundle``.

        Optional hook. Strategies that override this enable training-time
        visualization to go through the same code path as standalone
        inference (``src.inference.*``). Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"Strategy {type(self).__name__} does not implement build_inference()."
        )

    def prepare_eval_batch(
        self, batch: Dict[str, Any], bundle: TrainingBundle, ctx: StepContext, generator: Any = None,
    ) -> Dict[str, Any]:
        """Batch as training would see it (e.g. phase-2 prevs) for val / test loss, metrics and images."""
        return batch

    def self_forcing_metrics(self, dataset: Any, indices: Any, bundle: TrainingBundle, ctx: StepContext) -> Dict[str, float]:
        """Chain-generation metrics per ``eval.self_forcing``; empty when unsupported or unset."""
        return {}

    def set_train_mode(self, bundle: TrainingBundle) -> None:
        """Set model and all nn.Module in extras to train mode."""
        bundle.model.train()
        for v in bundle.extras.values():
            if isinstance(v, nn.Module):
                v.train()

    def set_eval_mode(self, bundle: TrainingBundle) -> None:
        """Set model and all nn.Module in extras to eval mode."""
        bundle.model.eval()
        for v in bundle.extras.values():
            if isinstance(v, nn.Module):
                v.eval()
