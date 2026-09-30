"""Per-model-family Strategy abstraction and registry."""
from src.strategies.shared.base import (
    BuildContext,
    DatasetParams,
    ModelStrategy,
    StepContext,
    TrainingBundle,
)
from src.strategies.shared.spectral_params import SpectralParams
from src.strategies.shared.registry import get, names, register

# Built-in strategies self-register on import; `get()` relies on this.
from src.strategies import flow_matching  # noqa: F401

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
