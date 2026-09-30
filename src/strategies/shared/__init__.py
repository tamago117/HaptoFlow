"""Shared strategy infrastructure: ABCs, registry, value objects, common helpers."""
from src.strategies.shared.base import (
    BuildContext,
    DatasetParams,
    ModelStrategy,
    StepContext,
    TrainingBundle,
)
from src.strategies.shared.spectral_params import SpectralParams
from src.strategies.shared.registry import get, names, register

__all__ = [
    "BuildContext",
    "DatasetParams",
    "ModelStrategy",
    "SpectralParams",
    "StepContext",
    "TrainingBundle",
    "get",
    "names",
    "register",
]
