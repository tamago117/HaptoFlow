"""Setup pipeline used by ``Trainer``.

Given a resolved ``cfg`` + ``model_type``, materializes everything downstream
needs to train:

- ``strategy``  (registered ModelStrategy for the family)
- ``bundle``    (model + optimizer + scheduler + extras from Strategy.build)
- ``train_dl`` / ``val_dl`` / ``test_dl``
- ``full_ds``   (used by viz + bundle build)
- ``encodec_wrapper`` (None unless the strategy sets ``use_encodec``)
- ``control_stats`` (force/velocity normalization stats)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from src.models.shared.encodec_wrapper import EncodecWrapper
from src.strategies import BuildContext, TrainingBundle
from src.strategies import get as get_strategy
from src.strategies.shared.base import ModelStrategy
from src.utils.data.data_utils import (
    create_dataloaders,
    load_column_stats_from_jsonl,
    prepare_datasets,
)


@dataclass
class RuntimeEnv:
    """Container for the resolved runtime state used by the Trainer."""
    cfg: Any
    device: torch.device
    model_type: str
    strategy: ModelStrategy
    bundle: TrainingBundle
    full_ds: Any
    train_dl: Any
    val_dl: Any
    test_dl: Any
    encodec_wrapper: Optional[EncodecWrapper]
    control_stats: Dict[str, Dict[str, float]]
    num_classes: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


def _resolve_control_stats(cfg: Any, project_root: str) -> Dict[str, Dict[str, float]]:
    """Load stats.jsonl from cfg.data.stats or next-to-meta default."""
    stats_path_cfg = getattr(cfg.data, "stats", None)
    default_stats_path = os.path.join(os.path.dirname(str(cfg.data.meta)), "stats.jsonl")
    stats_path = stats_path_cfg or default_stats_path
    if not os.path.isabs(str(stats_path)):
        stats_path = os.path.join(project_root, str(stats_path))
    control_stats = load_column_stats_from_jsonl(
        stats_path,
        target_columns=["force", "velocity_x", "velocity_y"],
    )
    if not control_stats:
        print(f"Warning: stats.jsonl not found or empty at {stats_path}; control stats will be skipped.")
    return control_stats


def build_runtime_environment(
    cfg: Any,
    model_type: str,
    *,
    project_root: str,
) -> RuntimeEnv:
    """Build a fully-wired ``RuntimeEnv`` from ``cfg`` + ``model_type``.

    Mutates ``cfg.data.root`` and ``cfg.data.meta`` to absolute paths so
    downstream code reads consistent state. Does NOT load checkpoint weights
    and does NOT initialize Weights & Biases (the Trainer owns both).
    """
    data_root = str(cfg.data.root)
    if not os.path.isabs(data_root):
        data_root = os.path.join(project_root, data_root)
    cfg.data.root = data_root
    cfg.data.meta = os.path.join(data_root, "meta.jsonl")

    torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    strategy = get_strategy(model_type)
    dparams = strategy.dataset_params(cfg)

    control_stats = _resolve_control_stats(cfg, project_root)

    train_ds, val_ds, test_ds, full_ds = prepare_datasets(
        cfg,
        context_segments=dparams.context_segments,
        control_stats=control_stats,
    )
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, cfg)

    encodec_wrapper: Optional[EncodecWrapper] = None
    if dparams.use_encodec:
        encodec_wrapper = EncodecWrapper(
            model_name=str(cfg.encodec.model_name),
            bandwidth=float(getattr(cfg.encodec, "bandwidth_kbps", 6.0)),
            device=device,
        )

    bundle = strategy.build(BuildContext(
        cfg=cfg,
        device=device,
        full_dataset=full_ds,
        encodec_wrapper=encodec_wrapper,
        steps_per_epoch=len(train_dl),
    ))

    return RuntimeEnv(
        cfg=cfg,
        device=device,
        model_type=model_type,
        strategy=strategy,
        bundle=bundle,
        full_ds=full_ds,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        encodec_wrapper=encodec_wrapper,
        control_stats=control_stats,
        num_classes=len(full_ds.classes),
    )
