"""Runtime layer: shared setup + training orchestration.

This layer sits **above** ``src.strategies`` and **below** ``scripts/``: the
strategy abstraction handles "this model family's forward pass and weight
graph"; the runtime layer handles "argparse → cfg → strategy/bundle/
dataloaders → epoch loop → save". Scripts only parse CLI args and call
``Trainer.from_args(args).run()``.
"""
from src.runtime.setup import RuntimeEnv, build_runtime_environment
from src.runtime.trainer import Trainer

__all__ = ["RuntimeEnv", "Trainer", "build_runtime_environment"]
