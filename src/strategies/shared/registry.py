"""Registry mapping strategy names to ModelStrategy subclasses."""
from __future__ import annotations

from typing import Dict, List, Type

from src.strategies.shared.base import ModelStrategy


_REGISTRY: Dict[str, Type[ModelStrategy]] = {}


def register(cls: Type[ModelStrategy]) -> Type[ModelStrategy]:
    """Decorator: register a ModelStrategy subclass by its `name`."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    if name in _REGISTRY:
        raise ValueError(
            f"Strategy {name!r} already registered by {_REGISTRY[name].__name__}"
        )
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> ModelStrategy:
    """Instantiate the strategy registered under `name`."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def names() -> List[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_REGISTRY)
